"""Línea base semanal recursiva de 15 días, sin evaluación ni persistencia.

Ejecutar desde la raíz: .venv/Scripts/python scripts/linea_base_prediccion.py
El lector reutiliza el prefijo temporal existente para no materializar ventas
posteriores al corte. Las funciones de cálculo también aceptan DataFrames.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analizar_periodo_seleccion import (
    FIN_SELECCION, HORIZONTE_DIAS, TAMANO_BLOQUE, TRAIN,
    contar_prefijo_fechas, leer_bloques_hasta_corte, verificar_columnas,
)
from scripts.preparar_series_modelado import (
    COLUMNAS, historico_hasta, leer_configuracion, preparar_observaciones,
)


TECNICA = "estacional_semanal_recursiva"


@dataclass(frozen=True)
class Prediccion:
    store_nbr: int
    family: str
    tecnica: str
    fecha_corte: date
    paso: int
    fecha_objetivo: date
    venta_estimada: float
    regla: str
    fecha_referencia: date
    fecha_origen_real: date

    @property
    def uso_respaldo(self) -> bool:
        return self.regla.startswith("respaldo_")


def _validar_corte(fecha_corte: date) -> None:
    if type(fecha_corte) is not date:
        raise ValueError("fecha_corte debe ser una fecha datetime.date explícita.")


def cargar_historico_csv(
    ruta: Path, claves: tuple[tuple[int, str], ...], fecha_corte: date,
    tamano_bloque: int = TAMANO_BLOQUE,
) -> pd.DataFrame:
    """Reutiliza la lectura por bloques hasta corte y la preparación de HU004-A.

    No usa cargar_series_csv porque ese lector carga también ventas futuras
    para la auditoría completa. Aquí solo se leen sus fechas para verificar
    el orden global; sales/onpromotion se leen exclusivamente en el prefijo.
    """
    _validar_corte(fecha_corte)
    if tamano_bloque < 1:
        raise ValueError("tamano_bloque debe ser positivo.")
    verificar_columnas(ruta)
    prefijo, _ = contar_prefijo_fechas(ruta, fin=fecha_corte,
                                     tamano_bloque=tamano_bloque)
    bloques = []
    permitidas = pd.MultiIndex.from_tuples(claves, names=["store_nbr", "family"])
    for bloque in leer_bloques_hasta_corte(ruta, prefijo, tamano_bloque):
        indice = pd.MultiIndex.from_arrays([
            pd.to_numeric(bloque["store_nbr"], errors="coerce"), bloque["family"]
        ], names=["store_nbr", "family"])
        seleccion = bloque.loc[indice.isin(permitidas)]
        if not seleccion.empty:
            bloques.append(seleccion)
    datos = pd.concat(bloques, ignore_index=True) if bloques else pd.DataFrame(columns=COLUMNAS)
    historico = historico_hasta(preparar_observaciones(datos), fecha_corte)
    presentes = set(zip(historico["store_nbr"], historico["family"]))
    for clave in claves:
        if clave not in presentes:
            raise ValueError(f"Sin historial utilizable hasta {fecha_corte}: {clave}.")
    return historico


def pronosticar_serie(
    datos: pd.DataFrame, store_nbr: int, family: str, fecha_corte: date,
) -> tuple[Prediccion, ...]:
    """Copia ventas reales semanales y después predicciones ya generadas.

    Recorta por fecha antes de validar o utilizar ventas. Un respaldo busca
    solo observaciones reales: primero el mismo día semanal y luego la última
    disponible. fecha_origen_real conserva la trazabilidad incluso al recurrir.
    """
    _validar_corte(fecha_corte)
    if type(store_nbr) is not int or store_nbr <= 0 or not isinstance(family, str) or not family.strip():
        raise ValueError("Indique store_nbr positivo y family no vacía.")
    faltantes = set(COLUMNAS) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    seleccion = datos.loc[
        pd.to_numeric(datos["store_nbr"], errors="coerce").eq(store_nbr)
        & datos["family"].eq(family)
    ].copy()
    if seleccion.empty:
        raise ValueError(f"Serie no encontrada: {store_nbr}/{family}.")
    seleccion["date"] = pd.to_datetime(seleccion["date"], format="%Y-%m-%d", errors="raise")
    historial = preparar_observaciones(historico_hasta(seleccion, fecha_corte))
    if historial.empty:
        raise ValueError(f"Sin historial utilizable hasta {fecha_corte}: {store_nbr}/{family}.")

    reales = dict(zip(historial["date"].dt.date, historial["sales"]))
    ultimo_por_dia = {d.weekday(): d for d in reales}
    ultima_fecha = max(reales)
    generadas: dict[date, Prediccion] = {}
    for paso in range(1, HORIZONTE_DIAS + 1):
        objetivo = fecha_corte + timedelta(days=paso)
        referencia = objetivo - timedelta(days=7)
        if referencia in reales:
            origen = referencia
            valor = reales[origen]
            regla = "historico_semanal"
        elif referencia in generadas:
            anterior = generadas[referencia]
            origen = anterior.fecha_origen_real
            valor = anterior.venta_estimada
            regla = "recursiva"
        elif objetivo.weekday() in ultimo_por_dia:
            origen = ultimo_por_dia[objetivo.weekday()]
            valor = reales[origen]
            regla = "respaldo_mismo_dia"
        else:
            origen = ultima_fecha
            valor = reales[origen]
            regla = "respaldo_ultimo_valor"
        valor = float(valor)
        if not isfinite(valor) or valor < 0:
            raise ValueError(f"Predicción inválida para {store_nbr}/{family} en {objetivo}.")
        generadas[objetivo] = Prediccion(
            store_nbr, family, TECNICA, fecha_corte, paso, objetivo, valor,
            regla, referencia, origen,
        )
    return tuple(generadas.values())


def pronosticar_series(
    datos: pd.DataFrame, fecha_corte: date,
    claves: tuple[tuple[int, str], ...] | None = None,
) -> tuple[Prediccion, ...]:
    """Por defecto procesa las diez claves de la configuración aprobada."""
    if claves is None:
        claves = leer_configuracion()
    return tuple(p for tienda, familia in claves
                 for p in pronosticar_serie(datos, tienda, familia, fecha_corte))


def fechas_consecutivas(predicciones: tuple[Prediccion, ...]) -> bool:
    return len(predicciones) == HORIZONTE_DIAS and all(
        p.paso == i and p.fecha_objetivo == p.fecha_corte + timedelta(days=i)
        for i, p in enumerate(predicciones, start=1)
    )


def imprimir_informe(predicciones: tuple[Prediccion, ...]) -> None:
    print(f"Método: {TECNICA}; estimaciones de demanda, horizonte {HORIZONTE_DIAS} días.")
    print(f"Fecha de corte: {predicciones[0].fecha_corte}")
    claves = tuple(dict.fromkeys((p.store_nbr, p.family) for p in predicciones))
    referencia = (17, "PREPARED FOODS")
    if referencia not in claves:
        referencia = claves[0]
    for tienda, familia in claves:
        grupo = tuple(p for p in predicciones if (p.store_nbr, p.family) == (tienda, familia))
        consecutivas = fechas_consecutivas(grupo)
        print(f"\nTienda {tienda} | {familia}: {len(grupo)} predicciones; "
              f"15 fechas consecutivas: {'sí' if consecutivas else 'no'}; "
              f"respaldos directos: {sum(p.uso_respaldo for p in grupo)}.")
        if (tienda, familia) == referencia:
            print("Paso Fecha objetivo Venta estimada Regla Fecha origen real")
            for p in grupo:
                print(f"{p.paso:>4} {p.fecha_objetivo} {p.venta_estimada!r} "
                      f"{p.regla} {p.fecha_origen_real}")
        for p in grupo:
            if p.uso_respaldo:
                print(f"  Respaldo: objetivo {p.fecha_objetivo}; referencia ausente "
                      f"{p.fecha_referencia}; {p.regla}; origen real {p.fecha_origen_real}.")
    print(f"\nTotal: {len(claves)} series y {len(predicciones)} predicciones.")
    print(f"Última fecha real utilizada como origen: {max(p.fecha_origen_real for p in predicciones)}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha-corte", type=date.fromisoformat, default=FIN_SELECCION)
    parser.add_argument("--store-nbr", type=int)
    parser.add_argument("--family")
    args = parser.parse_args(argv)
    if (args.store_nbr is None) != (args.family is None):
        parser.error("--store-nbr y --family deben indicarse juntos.")
    try:
        claves = leer_configuracion()
        if args.store_nbr is not None:
            clave = (args.store_nbr, args.family)
            if clave not in claves:
                raise ValueError(f"Serie no registrada en la configuración: {clave}.")
            claves = (clave,)
        datos = cargar_historico_csv(TRAIN, claves, args.fecha_corte)
        print(f"Histórico cargado: {len(datos)} observaciones; "
              f"última fecha disponible: {datos['date'].max()}; corte {args.fecha_corte}.")
        predicciones = pronosticar_series(datos, args.fecha_corte, claves)
        for tienda, familia in claves:
            grupo = tuple(p for p in predicciones if (p.store_nbr, p.family) == (tienda, familia))
            if not fechas_consecutivas(grupo):
                raise ValueError(f"Horizonte inválido para {tienda}/{familia}.")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudo completar la línea base: {error}")
        return 1
    imprimir_informe(predicciones)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
