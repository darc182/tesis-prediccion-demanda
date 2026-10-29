"""Variables por días naturales exactos, sin imputación ni entrenamiento.

Ejecución: .venv/Scripts/python scripts/variables_predictoras.py
Orden para X: lag_1, lag_7, lag_14, media_7, dia_semana, mes.
dia_semana usa lunes=0 y domingo=6; mes usa enero=1 y diciembre=12.
Las tablas conservan sales como respuesta; onpromotion no integra X.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from math import fsum, isfinite
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analizar_periodo_seleccion import FIN_SELECCION, TRAIN
from scripts.linea_base_prediccion import cargar_historico_csv
from scripts.preparar_series_modelado import (
    COLUMNAS, historico_hasta, leer_configuracion, preparar_observaciones,
)


COLUMNAS_PREDICTORAS = ("lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes")
COLUMNAS_TABLA = ("store_nbr", "family", "fecha_objetivo", *COLUMNAS_PREDICTORAS, "sales")


class HistorialInsuficiente(ValueError):
    """Conserva las variables afectadas y las fechas exactas que faltan."""

    def __init__(self, objetivo: date, faltantes: dict[str, tuple[date, ...]]):
        self.objetivo = objetivo
        self.faltantes = faltantes
        detalle = "; ".join(f"{nombre}: {', '.join(map(str, fechas))}"
                            for nombre, fechas in faltantes.items())
        super().__init__(f"Historial insuficiente para {objetivo}. Faltan {detalle}.")


@dataclass(frozen=True)
class Descarte:
    fecha_objetivo: date
    faltantes: dict[str, tuple[date, ...]]


@dataclass(frozen=True)
class ResultadoSerie:
    store_nbr: int
    family: str
    fecha_corte: date
    observaciones_historicas: int
    tabla: pd.DataFrame
    descartes: tuple[Descarte, ...]

    @property
    def descartes_por_variable(self) -> dict[str, int]:
        """Una fila puede carecer de varias variables; estos conteos se solapan."""
        conteos = Counter(nombre for d in self.descartes for nombre in d.faltantes)
        return {nombre: conteos[nombre] for nombre in COLUMNAS_PREDICTORAS}


def _validar_fecha(valor: date, nombre: str) -> None:
    if type(valor) is not date:
        raise ValueError(f"{nombre} debe ser datetime.date.")


def _historico_serie(
    datos: pd.DataFrame, store_nbr: int, family: str, fecha_corte: date,
) -> pd.DataFrame:
    _validar_fecha(fecha_corte, "fecha_corte")
    if type(store_nbr) is not int or store_nbr <= 0 or not isinstance(family, str) or not family.strip():
        raise ValueError("La serie requiere store_nbr positivo y family no vacía.")
    faltantes = set(COLUMNAS) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de observaciones: {sorted(faltantes)}")
    serie = datos.loc[
        pd.to_numeric(datos["store_nbr"], errors="coerce").eq(store_nbr)
        & datos["family"].eq(family)
    ].copy()
    # Primero se recorta por fechas; las ventas futuras no se validan ni usan.
    serie["date"] = pd.to_datetime(serie["date"], format="%Y-%m-%d", errors="raise")
    return preparar_observaciones(historico_hasta(serie, fecha_corte))


def _variables_por_fecha(ventas: Mapping[date, float], objetivo: date) -> dict[str, float | int]:
    """Núcleo común: nunca consulta la venta del objetivo ni fechas posteriores."""
    requeridas = {
        "lag_1": (objetivo - timedelta(days=1),),
        "lag_7": (objetivo - timedelta(days=7),),
        "lag_14": (objetivo - timedelta(days=14),),
        "media_7": tuple(objetivo - timedelta(days=d) for d in range(1, 8)),
    }
    faltantes = {nombre: tuple(d for d in fechas if d not in ventas)
                 for nombre, fechas in requeridas.items()}
    faltantes = {nombre: fechas for nombre, fechas in faltantes.items() if fechas}
    if faltantes:
        raise HistorialInsuficiente(objetivo, faltantes)
    # Dividir antes de sumar evita desbordar al sumar siete valores grandes.
    return {
        "lag_1": ventas[requeridas["lag_1"][0]],
        "lag_7": ventas[requeridas["lag_7"][0]],
        "lag_14": ventas[requeridas["lag_14"][0]],
        "media_7": fsum(ventas[d] / 7 for d in requeridas["media_7"]),
        "dia_semana": objetivo.weekday(),
        "mes": objetivo.month,
    }


def construir_tabla_serie(
    datos: pd.DataFrame, store_nbr: int, family: str, fecha_corte: date,
) -> ResultadoSerie:
    """Construye X e y solo para objetivos reales observados hasta el corte.

    X = resultado.tabla.loc[:, list(COLUMNAS_PREDICTORAS)]
    y = resultado.tabla['sales']
    Las fechas sin respuesta real no se convierten en candidatos artificiales.
    """
    historico = _historico_serie(datos, store_nbr, family, fecha_corte)
    if historico.empty:
        raise ValueError(f"Sin historial hasta {fecha_corte}: {store_nbr}/{family}.")
    ventas = dict(zip(historico["date"].dt.date, historico["sales"]))
    registros = []
    descartes = []
    for objetivo, respuesta in ventas.items():
        try:
            variables = _variables_por_fecha(ventas, objetivo)
        except HistorialInsuficiente as error:
            descartes.append(Descarte(objetivo, error.faltantes))
            continue
        registros.append({"store_nbr": store_nbr, "family": family,
                          "fecha_objetivo": objetivo, **variables, "sales": respuesta})
    tabla = pd.DataFrame.from_records(registros, columns=COLUMNAS_TABLA)
    return ResultadoSerie(store_nbr, family, fecha_corte, len(historico), tabla, tuple(descartes))


def construir_tablas(
    datos: pd.DataFrame, fecha_corte: date,
    claves: tuple[tuple[int, str], ...] | None = None,
) -> tuple[ResultadoSerie, ...]:
    """Procesa por defecto las diez series aprobadas, manteniendo su orden."""
    if claves is None:
        claves = leer_configuracion()
    return tuple(construir_tabla_serie(datos, t, f, fecha_corte) for t, f in claves)


def construir_variables_futuras(
    datos: pd.DataFrame, store_nbr: int, family: str,
    fecha_corte: date, fecha_objetivo: date,
    predicciones_previas: Mapping[date, float] | None = None,
) -> pd.DataFrame:
    """Devuelve una fila X en el mismo orden que entrenamiento.

    datos aporta solo ventas reales hasta fecha_corte. predicciones_previas
    pertenece exclusivamente a la serie indicada y contiene valores que el
    llamador ya tiene disponibles, con corte < fecha < objetivo. No se toman
    ventas reales de ese intervalo aunque estén presentes en datos.
    """
    _validar_fecha(fecha_objetivo, "fecha_objetivo")
    _validar_fecha(fecha_corte, "fecha_corte")
    if fecha_objetivo <= fecha_corte:
        raise ValueError("fecha_objetivo debe ser posterior a fecha_corte.")
    historico = _historico_serie(datos, store_nbr, family, fecha_corte)
    ventas = dict(zip(historico["date"].dt.date, historico["sales"]))
    if predicciones_previas is not None:
        for fecha, valor in predicciones_previas.items():
            _validar_fecha(fecha, "fecha de predicción previa")
            if not fecha_corte < fecha < fecha_objetivo:
                raise ValueError("Las predicciones previas requieren corte < fecha < objetivo.")
            try:
                numero = float(valor)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("La predicción previa debe ser finita y no negativa.") from error
            if isinstance(valor, bool) or not isfinite(numero) or numero < 0:
                raise ValueError("La predicción previa debe ser finita y no negativa.")
            ventas[fecha] = numero
    variables = _variables_por_fecha(ventas, fecha_objetivo)
    return pd.DataFrame([variables], columns=COLUMNAS_PREDICTORAS)


def imprimir_informe(resultados: tuple[ResultadoSerie, ...]) -> None:
    print("Variables predictoras: " + ", ".join(COLUMNAS_PREDICTORAS))
    print("dia_semana: lunes=0, domingo=6; mes: 1 a 12. Respuesta: sales.")
    for r in resultados:
        print(f"\nTienda {r.store_nbr} | {r.family} | corte {r.fecha_corte}")
        print(f"Históricas: {r.observaciones_historicas}; válidas: {len(r.tabla)}; "
              f"descartadas: {len(r.descartes)}.")
        print("Filas con cada variable ausente (conteos que pueden solaparse): "
              + ", ".join(f"{k}={v}" for k, v in r.descartes_por_variable.items()))
        if r.tabla.empty:
            print("Sin filas válidas: historial insuficiente para entrenar.")
        else:
            print(r.tabla.head(2).to_string(index=False))
    print(f"\nTotal: {len(resultados)} series; "
          f"{sum(len(r.tabla) for r in resultados)} filas válidas; "
          f"{sum(len(r.descartes) for r in resultados)} descartadas.")
    print("Variables de ventas: solo fechas estrictamente anteriores al objetivo; "
          "calendario derivado de su fecha. Sin ventas posteriores al corte.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha-corte", type=date.fromisoformat, default=FIN_SELECCION)
    args = parser.parse_args(argv)
    try:
        claves = leer_configuracion()
        historico = cargar_historico_csv(TRAIN, claves, args.fecha_corte)
        resultados = construir_tablas(historico, args.fecha_corte, claves)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudieron construir las variables: {error}")
        return 1
    imprimir_informe(resultados)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
