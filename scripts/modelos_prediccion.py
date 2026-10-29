"""Regresores independientes por serie, con horizonte recursivo de 15 días.

Configuraciones iniciales, sin optimización ni evaluación: 100 árboles/etapas
acotan el coste experimental; los mínimos de hoja moderan el ajuste local.
Boosting usa pasos de 0.05 y hasta 15 hojas por árbol. early_stopping=False
mantiene 100 iteraciones sin separación automática de validación. Semilla 42.
Un hilo para los estimadores y bibliotecas numéricas limita el paralelismo.
Se conservan decimales; tras comprobar finitud se aplica max(0, predicción).
No se persisten modelos ni resultados. Instalar requirements-modelado.txt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    COLUMNAS_PREDICTORAS, construir_tabla_serie, construir_variables_futuras,
)


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


def crear_estimador(tecnica: str):
    """Siempre devuelve una instancia nueva con configuración fija."""
    if tecnica == "random_forest":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                     random_state=42, n_jobs=1)
    if tecnica == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, early_stopping=False, random_state=42,
        )
    raise ValueError(f"Técnica desconocida: {tecnica}. Opciones: {TECNICAS}")


def entrenar_modelo(
    datos: pd.DataFrame, store_nbr: int, family: str, tecnica: str, fecha_corte: date,
) -> ModeloEntrenado:
    """Ajusta solo con las seis columnas aprobadas y las respuestas hasta corte.

    Se exige al menos una fila completa para permitir fit. Este mínimo es un
    requisito técnico, no una afirmación sobre suficiencia estadística.
    """
    inicio = perf_counter()
    contexto = f"{store_nbr}/{family}, {tecnica}, corte {fecha_corte}"
    try:
        preparado = construir_tabla_serie(datos, store_nbr, family, fecha_corte)
        if preparado.tabla.empty:
            raise ValueError("No hay filas completas de entrenamiento.")
        x = preparado.tabla.loc[:, list(COLUMNAS_PREDICTORAS)]
        y = preparado.tabla["sales"]
        estimador = crear_estimador(tecnica)
        # threadpoolctl es una dependencia de scikit-learn. HGB usa OpenMP
        # y no tiene n_jobs; también limitamos sus hilos durante fit/predict.
        with threadpool_limits(limits=1):
            estimador.fit(x, y)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Error de entrenamiento en {contexto}: {error}") from error
    return ModeloEntrenado(
        store_nbr, family, tecnica, fecha_corte, estimador,
        len(preparado.tabla), len(preparado.descartes),
        preparado.tabla["fecha_objetivo"].max(), perf_counter() - inicio,
    )


def pronosticar_recursivo(modelo: ModeloEntrenado, datos: pd.DataFrame) -> tuple[Prediccion, ...]:
    """Construye cada X con historia hasta corte y estimaciones de pasos previos.

    El diccionario temporal se crea de nuevo por ejecución. Los ceros que
    provengan del ajuste de negativos también se reutilizan en la recursión.
    """
    contexto = f"{modelo.store_nbr}/{modelo.family}, {modelo.tecnica}, corte {modelo.fecha_corte}"
    try:
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
                x = construir_variables_futuras(
                    historial, modelo.store_nbr, modelo.family,
                    modelo.fecha_corte, objetivo, previas,
                ).loc[:, list(COLUMNAS_PREDICTORAS)]
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
