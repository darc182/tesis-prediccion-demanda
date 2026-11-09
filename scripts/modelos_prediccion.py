"""Regresores independientes por serie, con horizonte recursivo de 15 días.

Configuraciones iniciales, sin optimización ni evaluación: 100 árboles/etapas
acotan el coste experimental; los mínimos de hoja moderan el ajuste local.
Boosting usa pasos de 0.05 y hasta 15 hojas por árbol. early_stopping=False
mantiene 100 iteraciones sin separación automática de validación. Semilla 42.
Se admiten configuraciones opcionales del contrato F2, conservando esos defaults.
Un hilo para los estimadores y bibliotecas numéricas limita el paralelismo.
Se conservan decimales; tras comprobar finitud se aplica max(0, predicción).
No se persisten modelos ni resultados. Instalar requirements-modelado.txt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import isfinite
from numbers import Real
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analizar_periodo_seleccion import FIN_SELECCION, HORIZONTE_DIAS, TRAIN
from scripts.linea_base_prediccion import cargar_historico_csv
from scripts.preparar_series_modelado import historico_hasta, leer_configuracion
from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS,
    construir_tabla_serie, construir_variables_futuras,
)
from backend.app.services import configuracion_experimentos as configuracion


TECNICAS = ("random_forest", "hist_gradient_boosting")


@dataclass(frozen=True)
class ModeloEntrenado:
    store_nbr: int
    family: str
    tecnica: str
    fecha_corte: date
    estimador: Any
    filas_validas: int
    filas_descartadas: int
    ultima_fecha_entrenamiento: date
    segundos_entrenamiento: float
    columnas_predictoras: tuple[str, ...] = field(default=COLUMNAS_PREDICTORAS, kw_only=True)

    def __post_init__(self):
        # Copiar secuencias del llamador evita conservar listas mutables.
        object.__setattr__(self, "columnas_predictoras", _columnas_canonicas(self.columnas_predictoras))


@dataclass(frozen=True)
class Prediccion:
    store_nbr: int
    family: str
    tecnica: str
    fecha_corte: date
    paso: int
    fecha_objetivo: date
    venta_estimada: float
    ajustada_a_cero: bool


@dataclass(frozen=True)
class Resultado:
    modelo: ModeloEntrenado
    predicciones: tuple[Prediccion, ...]
    segundos_pronostico: float

    @property
    def ajustes_a_cero(self) -> int:
        return sum(p.ajustada_a_cero for p in self.predicciones)


def normalizar_configuracion_entrenamiento(tecnica: str, configuracion_solicitada: dict | None = None) -> dict:
    """Valida F2 y las restricciones del motor F3, sin entrenar ni modificar entradas.

    F2 puede identificar permutaciones históricas v1; F3 solo ejecuta los grupos
    canónicos: seis columnas con contrato/reglas 1/1 o siete con 2/2.
    """
    if tecnica not in TECNICAS:
        raise ValueError(f"Técnica desconocida: {tecnica}. Opciones: {TECNICAS}")
    efectiva = configuracion.normalizar_configuracion(
        {"tecnica": tecnica} if configuracion_solicitada is None else configuracion_solicitada)
    if efectiva["tecnica"] != tecnica:
        raise ValueError("La técnica de la configuración no coincide con la solicitada.")
    grupos = {
        (1, 1): COLUMNAS_PREDICTORAS,
        (2, 2): COLUMNAS_PREDICTORAS_ALTERNATIVAS,
    }
    columnas = grupos.get((efectiva["version_contrato"], efectiva["version_reglas"]))
    if tuple(efectiva["variables"]) != columnas:
        raise ValueError("F3 requiere las variables canónicas en su orden original para cada contrato.")
    return efectiva


def _columnas_canonicas(columnas) -> tuple[str, ...]:
    if not isinstance(columnas, (tuple, list)):
        raise ValueError("Las columnas del modelo deben ser un grupo canónico ordenado.")
    seleccion = tuple(columnas)
    if seleccion not in (COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS):
        raise ValueError("Las columnas del modelo deben ser un grupo canónico ordenado.")
    return seleccion


def _comprobar_esquema(estimador, columnas: tuple[str, ...]) -> None:
    """Los regresores reales ajustados deben acreditar dimensión y nombres.

    Los dobles históricos pueden carecer de atributos sklearn. Si los declaran,
    también deben ser coherentes; nunca se infiere otro grupo desde el estimador.
    """
    real = isinstance(estimador, (RandomForestRegressor, HistGradientBoostingRegressor))
    if real or hasattr(estimador, "n_features_in_"):
        if getattr(estimador, "n_features_in_", None) != len(columnas):
            raise ValueError("n_features_in_ no coincide con las columnas del modelo.")
    if real or hasattr(estimador, "feature_names_in_"):
        nombres = getattr(estimador, "feature_names_in_", ())
        if tuple(nombres) != columnas:
            raise ValueError("feature_names_in_ no coincide con las columnas del modelo y su orden.")


def comprobar_parametros_efectivos(estimador, efectiva: dict) -> None:
    """Contrasta TODO get_params con lo declarado, sin perder tipos significativos.

    No completar defaults aquí: una clave omitida por el estimador es un error.
    En RF max_features=1 (número) no equivale a 1.0 (fracción de variables).
    """
    actual = json.dumps(estimador.get_params(deep=False), sort_keys=True, allow_nan=False)
    declarado = json.dumps(efectiva["parametros"], sort_keys=True, allow_nan=False)
    if actual != declarado:
        raise ValueError("Los parámetros reales no corresponden a la configuración solicitada.")


def crear_estimador(tecnica: str, *, configuracion_solicitada: dict | None = None):
    """Nueva instancia con parámetros F2 efectivos; sin argumento conserva HU004."""
    efectiva = normalizar_configuracion_entrenamiento(tecnica, configuracion_solicitada)
    clases = {"random_forest": RandomForestRegressor,
              "hist_gradient_boosting": HistGradientBoostingRegressor}
    estimador = clases[tecnica](**efectiva["parametros"])
    comprobar_parametros_efectivos(estimador, efectiva)
    return estimador


def entrenar_modelo(
    datos: pd.DataFrame, store_nbr: int, family: str, tecnica: str, fecha_corte: date,
    *, configuracion_solicitada: dict | None = None,
) -> ModeloEntrenado:
    """Ajusta con las columnas del contrato efectivo y las respuestas hasta corte.

    Se exige al menos una fila completa para permitir fit. Este mínimo es un
    requisito técnico, no una afirmación sobre suficiencia estadística.
    configuracion_solicitada acepta el objeto F2 (parcial o completo). Se valida
    antes de preparar X o ejecutar fit; no se modifica el DataFrame del llamador.
    """
    inicio = perf_counter()
    contexto = f"{store_nbr}/{family}, {tecnica}, corte {fecha_corte}"
    try:
        efectiva = normalizar_configuracion_entrenamiento(tecnica, configuracion_solicitada)
        columnas = tuple(efectiva["variables"])
        argumentos = {} if columnas == COLUMNAS_PREDICTORAS else {"columnas_predictoras": columnas}
        preparado = construir_tabla_serie(datos, store_nbr, family, fecha_corte, **argumentos)
        if preparado.columnas_predictoras != columnas:
            raise ValueError("Las columnas preparadas no coinciden con la configuración.")
        if preparado.tabla.empty:
            raise ValueError("No hay filas completas de entrenamiento.")
        orden = ("store_nbr", "family", "fecha_objetivo", *columnas, "sales")
        if tuple(preparado.tabla.columns) != orden:
            raise ValueError("Las columnas preparadas están incompletas o fuera de orden.")
        x = preparado.tabla.loc[:, list(columnas)]
        y = preparado.tabla["sales"]
        estimador = crear_estimador(tecnica) if configuracion_solicitada is None else crear_estimador(
            tecnica, configuracion_solicitada=efectiva)
        if callable(getattr(estimador, "get_params", None)):
            comprobar_parametros_efectivos(estimador, efectiva)
        # threadpoolctl es una dependencia de scikit-learn. HGB usa OpenMP
        # y no tiene n_jobs; también limitamos sus hilos durante fit/predict.
        with threadpool_limits(limits=1):
            estimador.fit(x, y)
        if callable(getattr(estimador, "get_params", None)):
            comprobar_parametros_efectivos(estimador, efectiva)
        _comprobar_esquema(estimador, columnas)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Error de entrenamiento en {contexto}: {error}") from error
    return ModeloEntrenado(
        store_nbr, family, tecnica, fecha_corte, estimador,
        len(preparado.tabla), len(preparado.descartes),
        preparado.tabla["fecha_objetivo"].max(), perf_counter() - inicio,
        columnas_predictoras=columnas,
    )


def pronosticar_recursivo(modelo: ModeloEntrenado, datos: pd.DataFrame) -> tuple[Prediccion, ...]:
    """Construye cada X con historia hasta corte y estimaciones de pasos previos.

    El diccionario temporal se crea de nuevo por ejecución. Los ceros que
    provengan del ajuste de negativos también se reutilizan en la recursión.
    """
    contexto = f"{modelo.store_nbr}/{modelo.family}, {modelo.tecnica}, corte {modelo.fecha_corte}"
    try:
        columnas = _columnas_canonicas(modelo.columnas_predictoras)
        _comprobar_esquema(modelo.estimador, columnas)
        historial = datos.loc[
            pd.to_numeric(datos["store_nbr"], errors="coerce").eq(modelo.store_nbr)
            & datos["family"].eq(modelo.family)
        ].copy()
        historial["date"] = pd.to_datetime(historial["date"], format="%Y-%m-%d", errors="raise")
        historial = historico_hasta(historial, modelo.fecha_corte)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Historial inválido en {contexto}: {error}") from error
    previas: dict[date, float] = {}
    resultados = []
    with threadpool_limits(limits=1):
        for paso in range(1, HORIZONTE_DIAS + 1):
            objetivo = modelo.fecha_corte + timedelta(days=paso)
            try:
                argumentos = {} if columnas == COLUMNAS_PREDICTORAS else {"columnas_predictoras": columnas}
                x = construir_variables_futuras(
                    historial, modelo.store_nbr, modelo.family,
                    modelo.fecha_corte, objetivo, previas,
                    **argumentos,
                )
                if tuple(x.columns) != columnas:
                    raise ValueError("Las columnas de predict no coinciden con las columnas de fit y su orden.")
                _comprobar_esquema(modelo.estimador, columnas)
                salida = modelo.estimador.predict(x)
                if len(salida) != 1 or isinstance(salida[0], bool) or not isinstance(salida[0], Real):
                    raise ValueError("El estimador debe devolver un único valor numérico real.")
                valor = float(salida[0])
                if not isfinite(valor):
                    raise ValueError("La predicción no es finita.")
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"Error de pronóstico en {contexto}, objetivo {objetivo}: {error}") from error
            ajustada = valor < 0
            valor = max(0.0, valor)
            previas[objetivo] = valor
            resultados.append(Prediccion(
                modelo.store_nbr, modelo.family, modelo.tecnica, modelo.fecha_corte,
                paso, objetivo, valor, ajustada,
            ))
    return tuple(resultados)


def ejecutar_modelos(
    datos: pd.DataFrame, fecha_corte: date,
    claves: tuple[tuple[int, str], ...] | None = None,
    tecnicas: tuple[str, ...] = TECNICAS,
) -> tuple[Resultado, ...]:
    """Por defecto genera diez series por dos técnicas, sin persistencia."""
    if claves is None:
        claves = leer_configuracion()
    if not claves or not tecnicas or any(t not in TECNICAS for t in tecnicas):
        raise ValueError("Indique series y técnicas válidas.")
    resultados = []
    for tienda, familia in claves:
        for tecnica in tecnicas:
            modelo = entrenar_modelo(datos, tienda, familia, tecnica, fecha_corte)
            inicio = perf_counter()
            predicciones = pronosticar_recursivo(modelo, datos)
            resultados.append(Resultado(modelo, predicciones, perf_counter() - inicio))
    return tuple(resultados)


def imprimir_informe(resultados: tuple[Resultado, ...], segundos_totales: float) -> None:
    print(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__}")
    print("Técnicas: " + ", ".join(dict.fromkeys(r.modelo.tecnica for r in resultados)))
    print("X: " + ", ".join(COLUMNAS_PREDICTORAS) + "; respuesta: sales.")
    print("Regla: comprobar finitud y después max(0, predicción); sin redondeo interno.")
    claves = tuple(dict.fromkeys((r.modelo.store_nbr, r.modelo.family) for r in resultados))
    referencia = (17, "PREPARED FOODS") if (17, "PREPARED FOODS") in claves else claves[0]
    for r in resultados:
        m = r.modelo
        print(f"\n{m.store_nbr} | {m.family} | {m.tecnica} | corte {m.fecha_corte}")
        print(f"Entrenamiento: {m.filas_validas} válidas, {m.filas_descartadas} descartadas; "
              f"última respuesta real: {m.ultima_fecha_entrenamiento}.")
        print(f"Pronóstico: {len(r.predicciones)} días consecutivos, "
              f"{r.predicciones[0].fecha_objetivo} a {r.predicciones[-1].fecha_objetivo}; "
              f"ajustes a cero: {r.ajustes_a_cero}.")
        print(f"Segundos: preparación/fit={m.segundos_entrenamiento:.3f}; "
              f"pronóstico={r.segundos_pronostico:.3f}.")
        if (m.store_nbr, m.family) == referencia:
            for p in r.predicciones:
                print(f"  {p.paso:2} {p.fecha_objetivo} estimada={p.venta_estimada!r}")
    print(f"\nTotal: {len(claves)} series, {len(resultados)} combinaciones, "
          f"{sum(len(r.predicciones) for r in resultados)} predicciones; "
          f"{sum(r.ajustes_a_cero for r in resultados)} ajustes a cero.")
    print(f"Tiempo total, incluida lectura: {segundos_totales:.3f} s.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha-corte", type=date.fromisoformat, default=FIN_SELECCION)
    parser.add_argument("--tecnica", choices=TECNICAS)
    parser.add_argument("--store-nbr", type=int)
    parser.add_argument("--family")
    args = parser.parse_args(argv)
    if (args.store_nbr is None) != (args.family is None):
        parser.error("--store-nbr y --family deben indicarse juntos.")
    inicio = perf_counter()
    try:
        claves = leer_configuracion()
        if args.store_nbr is not None:
            clave = (args.store_nbr, args.family)
            if clave not in claves:
                raise ValueError(f"Serie no registrada: {clave}.")
            claves = (clave,)
        tecnicas = (args.tecnica,) if args.tecnica else TECNICAS
        historico = cargar_historico_csv(TRAIN, claves, args.fecha_corte)
        resultados = ejecutar_modelos(historico, args.fecha_corte, claves, tecnicas)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudo completar el experimento: {error}")
        return 1
    imprimir_informe(resultados, perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
