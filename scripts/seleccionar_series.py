"""Selección provisional de diez series tienda-familia sin usar datos posteriores al corte.

Ejecución: .venv/Scripts/python scripts/seleccionar_series.py
No escribe resultados. Reutiliza la lectura cronológica y la agregación por bloques
del análisis exploratorio del periodo de selección.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite, log1p
from pathlib import Path
import sys

import pandas as pd

# Permite ejecutar por ruta y también importar como módulo en las pruebas.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analizar_periodo_seleccion import (
    FIN_SELECCION,
    INICIO_SELECCION,
    TAMANO_BLOQUE,
    TRAIN,
    calcular_caracteristicas,
    contar_prefijo_fechas,
    leer_bloques_hasta_corte,
    verificar_columnas,
)


CUOTAS = {"A": 2, "B": 3, "C": 3, "D": 2}
PERCENTILES = {2: (0.25, 0.75), 3: (0.20, 0.50, 0.80)}


def grupo_por_ceros(porcentaje: float) -> str | None:
    """Los límites se aplican al cociente exacto, antes de redondear."""
    if porcentaje <= 5:
        return "A"
    if porcentaje <= 35:
        return "B"
    if porcentaje <= 70:
        return "C"
    if porcentaje < 100:
        return "D"
    return None


def preparar_candidatos(series: pd.DataFrame) -> pd.DataFrame:
    """Usa filas registradas como denominador; no crea fechas ni ventas cero."""
    requeridas = {
        "store_nbr", "family", "observaciones", "ventas_cero",
        "volumen_total", "media", "desviacion", "porcentaje_promocion",
    }
    faltantes = requeridas - set(series.columns)
    if faltantes:
        raise ValueError(f"Faltan métricas por serie: {sorted(faltantes)}")
    candidatos = series.loc[
        series["observaciones"].gt(0) & series["volumen_total"].gt(0)
    ].copy()
    candidatos["porcentaje_cero"] = (
        100 * candidatos["ventas_cero"] / candidatos["observaciones"]
    )
    candidatos["grupo"] = candidatos["porcentaje_cero"].map(grupo_por_ceros)
    candidatos = candidatos.loc[candidatos["grupo"].notna()].copy()
    candidatos["coeficiente_variacion"] = (
        candidatos["desviacion"] / candidatos["media"]
    )
    return candidatos


def seleccionar_series(candidatos: pd.DataFrame) -> pd.DataFrame:
    """Elige el más cercano a cada objetivo con desempate y topes globales.

    Los percentiles se fijan con todos los candidatos elegibles del grupo,
    antes de aplicar los topes causados por selecciones anteriores.
    """
    elegidos: list[int] = []
    tiendas: Counter[str] = Counter()
    familias: Counter[str] = Counter()

    for grupo, cuota in CUOTAS.items():
        miembros = candidatos.loc[candidatos["grupo"].eq(grupo)]
        if len(miembros) < cuota:
            raise ValueError(
                f"Grupo {grupo}: {len(miembros)} candidatos elegibles para "
                f"una cuota de {cuota}; faltan {cuota - len(miembros)}."
            )
        volumenes = miembros["volumen_total"].map(log1p)
        for percentil in PERCENTILES[cuota]:
            objetivo = float(volumenes.quantile(percentil))
            disponibles = miembros.loc[
                ~miembros.index.isin(elegidos)
                & miembros["store_nbr"].map(lambda x: tiendas[str(x)] < 2)
                & miembros["family"].map(lambda x: familias[str(x)] < 2)
            ]
            if disponibles.empty:
                sin_repetir = miembros.loc[~miembros.index.isin(elegidos)]
                limite_tienda = sum(
                    tiendas[str(t)] >= 2 for t in sin_repetir["store_nbr"]
                )
                limite_familia = sum(
                    familias[str(f)] >= 2 for f in sin_repetir["family"]
                )
                raise ValueError(
                    f"Grupo {grupo}, percentil {percentil:.0%}: no quedan "
                    "candidatos que cumplan los topes; "
                    f"{len(sin_repetir)} sin seleccionar, "
                    f"{limite_tienda} bloqueados por tienda y "
                    f"{limite_familia} por familia. "
                    f"Se obtuvieron {len(elegidos)} de 10 series."
                )
            elegido = min(
                disponibles.index,
                key=lambda i: (
                    abs(log1p(float(candidatos.at[i, "volumen_total"])) - objetivo),
                    int(candidatos.at[i, "store_nbr"]),
                    str(candidatos.at[i, "family"]),
                ),
            )
            elegidos.append(elegido)
            tiendas[str(candidatos.at[elegido, "store_nbr"])] += 1
            familias[str(candidatos.at[elegido, "family"])] += 1

    return candidatos.loc[elegidos].reset_index(drop=True)


def _numero(valor: float, decimales: int = 2) -> str:
    return f"{valor:,.{decimales}f}" if isfinite(valor) else "N/D"


def imprimir_informe(elegidos: pd.DataFrame, filas_prefijo: int,
                    filas_totales: int) -> None:
    print("=== SELECCIÓN PROVISIONAL de series tienda-familia ===")
    print(f"Periodo observado: {INICIO_SELECCION} a {FIN_SELECCION} (inclusive).")
    print("\nGrupo Tienda Familia                        Volumen       % cero"
          "       Media       Desv.         CV      % promo")
    for fila in elegidos.itertuples(index=False):
        print(
            f"{fila.grupo:>5} {str(fila.store_nbr):>6} "
            f"{fila.family:<29} {_numero(float(fila.volumen_total)):>13} "
            f"{_numero(float(fila.porcentaje_cero)):>12} "
            f"{_numero(float(fila.media)):>11} "
            f"{_numero(float(fila.desviacion)):>11} "
            f"{_numero(float(fila.coeficiente_variacion)):>10} "
            f"{_numero(float(fila.porcentaje_promocion)):>12}"
        )
    print(f"\nDiversidad: {elegidos['store_nbr'].nunique()} tiendas y "
          f"{elegidos['family'].nunique()} familias.")
    print(f"Control temporal: se revisaron {filas_totales} fechas; "
          f"se leyeron sales/onpromotion de solo {filas_prefijo} filas "
          f"hasta {FIN_SELECCION}. No se consultaron ventas ni promociones posteriores.")
    print("Limitaciones: selección provisional descriptiva y dependiente de los "
          "umbrales, percentiles y topes fijados; no garantiza representar "
          "todas las series ni anticipa desempeño predictivo. Los porcentajes "
          "usan solo filas existentes; fechas ausentes no son ventas cero. "
          "Promoción significa onpromotion > 0. La desviación estándar es "
          "muestral y CV = desviación/media.")


def main() -> int:
    try:
        verificar_columnas(TRAIN)
        filas_prefijo, filas_totales = contar_prefijo_fechas(TRAIN)
        if not filas_prefijo:
            raise ValueError("No hay observaciones hasta el corte.")
        resultado = calcular_caracteristicas(
            leer_bloques_hasta_corte(TRAIN, filas_prefijo, TAMANO_BLOQUE)
        )
        candidatos = preparar_candidatos(resultado.series)
        print("Candidatos elegibles por grupo:")
        for grupo, cuota in CUOTAS.items():
            print(f"  {grupo}: {int(candidatos['grupo'].eq(grupo).sum())} (cuota {cuota})")
        elegidos = seleccionar_series(candidatos)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"Selección detenida: {error}")
        return 1
    imprimir_informe(elegidos, filas_prefijo, filas_totales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
