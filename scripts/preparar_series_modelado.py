"""Prepara observaciones de las series aprobadas y audita su calendario.

Ejecución: .venv/Scripts/python scripts/preparar_series_modelado.py
La auditoría lee train.csv por bloques. El entrenamiento posterior debe usar
historico_hasta con su fecha de corte; este módulo no entrena modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.carga_datos import _series_habilitadas
from scripts.analizar_periodo_seleccion import (
    COLUMNAS_REQUERIDAS, FIN_SELECCION, HORIZONTE_DIAS,
    INICIO_SELECCION, TAMANO_BLOQUE, TRAIN, VENTANAS_POSTERIORES,
    contar_prefijo_fechas, validar_ventanas, verificar_columnas,
)


RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "config" / "series_seleccionadas.json"
COLUMNAS = ["date", "store_nbr", "family", "sales", "onpromotion"]


@dataclass(frozen=True)
class CoberturaVentana:
    nombre: str
    disponibles: tuple[date, ...]
    ausentes: tuple[date, ...]


@dataclass(frozen=True)
class CoberturaSerie:
    store_nbr: int
    family: str
    primera_fecha: date
    ultima_fecha: date
    registros_entrenamiento: int
    dias_esperados: int
    dias_sin_registro: int
    ventas_cero: int
    ventas_fraccionarias: bool
    ventanas: tuple[CoberturaVentana, ...]

    @property
    def entrenamiento_completo(self) -> bool:
        return self.dias_sin_registro == 0


def leer_configuracion(ruta: Path = CONFIG) -> tuple[tuple[int, str], ...]:
    """Lee las diez claves registradas y comprueba sus metadatos temporales."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la configuración: {ruta}")
    try:
        config = json.loads(ruta.read_text(encoding="utf-8"))
        claves = tuple(_series_habilitadas(ruta))
        periodo = config["periodo_seleccion"]
        if periodo != {"inicio": INICIO_SELECCION.isoformat(),
                       "fin": FIN_SELECCION.isoformat()}:
            raise ValueError("El periodo de la configuración no coincide con el corte aprobado.")
        if config["horizonte_prediccion_dias"] != HORIZONTE_DIAS:
            raise ValueError("El horizonte de la configuración no coincide con el aprobado.")
        if config["regla_seleccion"]["granularidad"] != ["store_nbr", "family"]:
            raise ValueError("La granularidad de la configuración es incoherente.")
        if config["dataset"]["archivo"] != "data/raw/train.csv":
            raise ValueError("La configuración no apunta al train.csv original.")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Configuración incoherente: {error}") from error
    validar_ventanas()
    return claves


def preparar_observaciones(datos: pd.DataFrame) -> pd.DataFrame:
    """Valida y ordena filas existentes, sin completar fechas ni alterar ventas."""
    faltantes = COLUMNAS_REQUERIDAS - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")
    resultado = datos.loc[:, COLUMNAS].copy()
    if resultado.empty:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "store_nbr": pd.Series(dtype="int64"),
            "family": pd.Series(dtype="str"),
            "sales": pd.Series(dtype="float64"),
            "onpromotion": pd.Series(dtype="int64"),
        })

    fechas_texto = resultado["date"].astype("string")
    fechas = pd.to_datetime(fechas_texto, format="%Y-%m-%d", errors="coerce")
    if fechas.isna().any() or not fechas_texto.eq(fechas.dt.strftime("%Y-%m-%d")).all():
        raise ValueError("date debe contener fechas válidas YYYY-MM-DD.")
    resultado["date"] = fechas

    tiendas = resultado["store_nbr"].astype("string")
    tiendas_num = pd.to_numeric(tiendas, errors="coerce")
    if (not tiendas.str.fullmatch(r"[0-9]+").fillna(False).all()
            or tiendas_num.isna().any() or tiendas_num.le(0).any()
            or tiendas_num.gt(2**63 - 1).any()):
        raise ValueError("store_nbr debe ser un entero positivo.")
    resultado["store_nbr"] = tiendas_num.astype("int64")

    familias = resultado["family"].astype("string")
    if familias.isna().any() or familias.str.strip().eq("").any():
        raise ValueError("family no puede estar vacía.")
    resultado["family"] = familias

    ventas = pd.to_numeric(resultado["sales"], errors="coerce")
    if ventas.isna().any() or not np.isfinite(ventas.to_numpy(dtype="float64")).all() \
            or ventas.lt(0).any():
        raise ValueError("sales debe ser un número finito no negativo.")
    resultado["sales"] = ventas.astype("float64")

    promociones = resultado["onpromotion"].astype("string")
    promociones_num = pd.to_numeric(promociones, errors="coerce")
    if (not promociones.str.fullmatch(r"[0-9]+").fillna(False).all()
            or promociones_num.isna().any() or promociones_num.gt(2**63 - 1).any()):
        raise ValueError("onpromotion debe ser un recuento entero no negativo.")
    resultado["onpromotion"] = promociones_num.astype("int64")

    clave = ["store_nbr", "family", "date"]
    if resultado.duplicated(clave).any():
        duplicado = resultado.loc[resultado.duplicated(clave), clave].iloc[0]
        raise ValueError(f"Serie con fecha duplicada: tienda {duplicado.store_nbr}, "
                         f"familia {duplicado.family}, {duplicado.date.date()}.")
    return resultado.sort_values(["date", "store_nbr", "family"],
                                 kind="stable").reset_index(drop=True)


def cargar_series_csv(
    ruta_csv: Path, claves: tuple[tuple[int, str], ...],
    tamano_bloque: int = TAMANO_BLOQUE,
) -> pd.DataFrame:
    """Lee el CSV original por bloques y conserva solo las series aprobadas."""
    if tamano_bloque < 1:
        raise ValueError("tamano_bloque debe ser positivo.")
    verificar_columnas(ruta_csv)
    # Reutiliza el control de fechas y orden global, también entre bloques.
    contar_prefijo_fechas(ruta_csv, tamano_bloque=tamano_bloque)
    permitidas = set(claves)
    bloques = []
    for bloque in pd.read_csv(
        ruta_csv, usecols=COLUMNAS, chunksize=tamano_bloque,
        dtype="string", keep_default_na=False, na_values=[""],
    ):
        tiendas = pd.to_numeric(bloque["store_nbr"], errors="coerce")
        elegidas = pd.Series(
            [(int(t), f) in permitidas if pd.notna(t) and t > 0 and t <= 2**63 - 1
             else False for t, f in zip(tiendas, bloque["family"])],
            index=bloque.index,
        )
        if elegidas.any():
            bloques.append(preparar_observaciones(bloque.loc[elegidas]))
    datos = preparar_observaciones(pd.concat(bloques, ignore_index=True)) \
        if bloques else preparar_observaciones(pd.DataFrame(columns=COLUMNAS))
    encontradas = set(zip(datos["store_nbr"], datos["family"]))
    faltantes = permitidas - encontradas
    if faltantes:
        detalle = ", ".join(f"{tienda}/{familia}" for tienda, familia in sorted(faltantes))
        raise ValueError(f"Series seleccionadas no encontradas: {detalle}")
    return datos


def historico_hasta(datos: pd.DataFrame, corte: date) -> pd.DataFrame:
    """Entrega observaciones hasta corte inclusive; la fecha debe ser datetime."""
    if not pd.api.types.is_datetime64_any_dtype(datos["date"]):
        raise ValueError("Prepare primero las observaciones con preparar_observaciones.")
    return datos.loc[datos["date"].le(pd.Timestamp(corte))].copy()


def informar_cobertura(
    datos: pd.DataFrame, claves: tuple[tuple[int, str], ...],
    inicio: date = INICIO_SELECCION, fin: date = FIN_SELECCION,
) -> tuple[CoberturaSerie, ...]:
    """Calcula métricas iniciales y solo fechas presentes/ausentes del futuro."""
    validar_ventanas()
    if inicio > fin:
        raise ValueError("El inicio del entrenamiento supera su fin.")
    informes = []
    dias_esperados = (fin - inicio).days + 1
    for tienda, familia in claves:
        serie = datos.loc[datos["store_nbr"].eq(tienda) & datos["family"].eq(familia)]
        if serie.empty:
            raise ValueError(f"Serie no encontrada: {tienda}/{familia}")
        inicial = historico_hasta(serie, fin)
        inicial = inicial.loc[inicial["date"].ge(pd.Timestamp(inicio))]
        fechas_iniciales = set(inicial["date"].dt.date)
        ventanas = []
        for ventana in VENTANAS_POSTERIORES:
            calendario = pd.date_range(ventana.inicio, ventana.fin).date
            fechas = set(serie.loc[
                serie["date"].between(pd.Timestamp(ventana.inicio),
                                      pd.Timestamp(ventana.fin)), "date"
            ].dt.date)
            ventanas.append(CoberturaVentana(
                ventana.nombre,
                tuple(d for d in calendario if d in fechas),
                tuple(d for d in calendario if d not in fechas),
            ))
        informes.append(CoberturaSerie(
            tienda, familia, serie["date"].min().date(),
            serie["date"].max().date(), len(inicial), dias_esperados,
            dias_esperados - len(fechas_iniciales),
            int(inicial["sales"].eq(0).sum()),
            bool(inicial["sales"].mod(1).ne(0).any()), tuple(ventanas),
        ))
    return tuple(informes)


def imprimir_informe(informes: tuple[CoberturaSerie, ...]) -> None:
    print("\n=== Auditoría temporal de las diez series seleccionadas ===")
    print(f"Entrenamiento inicial: {INICIO_SELECCION} a {FIN_SELECCION}.")
    print("Las ventanas posteriores describen fechas; sus ventas no intervienen en el diagnóstico inicial.")
    for serie in informes:
        print(f"\nTienda {serie.store_nbr} | {serie.family}")
        print(f"  Observada: {serie.primera_fecha} a {serie.ultima_fecha}")
        print(f"  Inicial: {serie.registros_entrenamiento}/{serie.dias_esperados} registros/días; "
              f"sin registro: {serie.dias_sin_registro}; sales=0: {serie.ventas_cero}; "
              f"ventas fraccionarias: {'sí' if serie.ventas_fraccionarias else 'no'}")
        if not serie.entrenamiento_completo:
            print("  Cobertura inicial incompleta: revisar antes de entrenar.")
        for ventana in serie.ventanas:
            presentes = ", ".join(map(str, ventana.disponibles)) or "ninguna"
            ausentes = ", ".join(map(str, ventana.ausentes)) or "ninguna"
            print(f"  {ventana.nombre}: disponibles [{presentes}]; ausentes [{ausentes}]")


def main() -> int:
    try:
        claves = leer_configuracion()
        datos = cargar_series_csv(TRAIN, claves)
        informes = informar_cobertura(datos, claves)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudo preparar la auditoría: {error}")
        return 1
    imprimir_informe(informes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
