"""Descripción reproducible de las series tienda-familia de train.csv.

Ejecutar con: .venv/Scripts/python scripts/analizar_series.py

Se lee train.csv en bloques de 100.000 filas. Para las medianas exactas se
retiene una matriz compacta de valores float por serie (unos 24 MB para los
3 millones de ventas actuales), sin conservar el CSV completo en memoria.
Los resultados se imprimen; no se escriben archivos ni se seleccionan series.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "raw"
TAMANO_BLOQUE = 100_000
FECHAS_AUSENTES = tuple(f"{anio}-12-25" for anio in range(2013, 2017))
COLUMNAS_TRAIN = {"date", "store_nbr", "family", "sales", "onpromotion"}
COLUMNAS_FERIADOS = {
    "date", "type", "locale", "locale_name", "description", "transferred"
}
COLUMNAS_TRANSACCIONES = {"date", "store_nbr", "transactions"}


def entero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def numero(valor: float, decimales: int = 2) -> str:
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def verificar_columnas(nombre: str, requeridas: set[str]) -> Path:
    """Falla con explicación si el archivo o alguna columna requerida falta."""
    ruta = DATOS / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    disponibles = set(pd.read_csv(ruta, nrows=0).columns)
    faltantes = requeridas - disponibles
    if faltantes:
        raise ValueError(f"{nombre}: faltan columnas {', '.join(sorted(faltantes))}")
    print(f"{nombre}: columnas disponibles: {', '.join(sorted(disponibles))}.")
    return ruta


@dataclass
class AcumuladoSerie:
    observaciones: int = 0
    ventas: array = field(default_factory=lambda: array("d"))
    promociones_validas: int = 0
    dias_con_promocion: int = 0


def resumir_train(ruta: Path) -> tuple[pd.DataFrame, set[str]]:
    """Calcula medidas por serie; porcentajes usan valores numéricos válidos.

    Una venta fraccionaria cumple sales % 1 != 0. Un día con promoción cumple
    onpromotion > 0. La desviación estándar es muestral (ddof=1). La mediana
    se calcula sobre todas las ventas numéricas de cada serie, no por bloques.
    """
    acumulados: dict[tuple[str, str], AcumuladoSerie] = {}
    fechas_observadas: set[str] = set()
    filas = 0
    claves_incompletas = 0
    ventas_vacias = 0
    promociones_vacias = 0
    ventas_no_numericas = 0
    promociones_no_numericas = 0

    bloques = pd.read_csv(
        ruta,
        usecols=list(COLUMNAS_TRAIN),
        chunksize=TAMANO_BLOQUE,
        dtype="string",
        keep_default_na=False,
        na_values=[""],
    )
    for bloque in bloques:
        filas += len(bloque)
        fechas_observadas.update(bloque["date"].dropna().unique().tolist())
        claves_validas = bloque[["store_nbr", "family"]].notna().all(axis=1)
        claves_incompletas += int((~claves_validas).sum())
        ventas_vacias += int(bloque["sales"].isna().sum())
        promociones_vacias += int(bloque["onpromotion"].isna().sum())

        bloque["sales_num"] = pd.to_numeric(bloque["sales"], errors="coerce")
        bloque["promocion_num"] = pd.to_numeric(
            bloque["onpromotion"], errors="coerce"
        )
        ventas_no_numericas += int(
            (bloque["sales"].notna() & bloque["sales_num"].isna()).sum()
        )
        promociones_no_numericas += int(
            (bloque["onpromotion"].notna() & bloque["promocion_num"].isna()).sum()
        )

        for clave, grupo in bloque.loc[claves_validas].groupby(
            ["store_nbr", "family"], sort=False
        ):
            acumulado = acumulados.setdefault(clave, AcumuladoSerie())
            acumulado.observaciones += len(grupo)
            acumulado.ventas.extend(
                grupo["sales_num"].to_numpy(dtype="float64", na_value=float("nan"))
            )
            promocion = grupo["promocion_num"]
            acumulado.promociones_validas += int(promocion.count())
            acumulado.dias_con_promocion += int(promocion.gt(0).sum())

    registros = []
    for (tienda, familia), acumulado in acumulados.items():
        ventas = pd.Series(acumulado.ventas, dtype="float64").dropna()
        n_ventas = len(ventas)
        if n_ventas:
            ventas_cero = int(ventas.eq(0).sum())
            ventas_fraccionarias = int(ventas.mod(1).ne(0).sum())
            minimo = float(ventas.min())
            maximo = float(ventas.max())
            total = float(ventas.sum())
            media = float(ventas.mean())
            mediana = float(ventas.median())
            desviacion = float(ventas.std(ddof=1)) if n_ventas > 1 else float("nan")
            porcentaje_cero = 100 * ventas_cero / n_ventas
            porcentaje_fraccion = 100 * ventas_fraccionarias / n_ventas
        else:
            ventas_cero = ventas_fraccionarias = 0
            minimo = maximo = total = media = mediana = desviacion = float("nan")
            porcentaje_cero = porcentaje_fraccion = float("nan")

        porcentaje_promocion = (
            100 * acumulado.dias_con_promocion / acumulado.promociones_validas
            if acumulado.promociones_validas else float("nan")
        )
        registros.append({
            "store_nbr": tienda,
            "family": familia,
            "observaciones": acumulado.observaciones,
            "ventas_validas": n_ventas,
            "volumen_total": total,
            "media": media,
            "mediana": mediana,
            "desviacion": desviacion,
            "minimo": minimo,
            "maximo": maximo,
            "ventas_cero": ventas_cero,
            "ventas_fraccionarias": ventas_fraccionarias,
            "porcentaje_cero": porcentaje_cero,
            "porcentaje_fraccion": porcentaje_fraccion,
            "dias_con_promocion": acumulado.dias_con_promocion,
            "porcentaje_promocion": porcentaje_promocion,
        })

    resumen = pd.DataFrame.from_records(registros)
    print("\n=== Cobertura del cálculo por serie ===")
    print(f"Filas de train.csv recorridas: {entero(filas)}.")
    print(f"Series tienda-familia: {entero(len(resumen))}.")
    print(f"Filas sin tienda o familia: {entero(claves_incompletas)}.")
    print(f"Ventas vacías: {entero(ventas_vacias)}.")
    print(f"Promociones vacías: {entero(promociones_vacias)}.")
    print(f"Ventas no numéricas no vacías: {entero(ventas_no_numericas)}.")
    print(f"Promociones no numéricas no vacías: {entero(promociones_no_numericas)}.")
    if (claves_incompletas or ventas_vacias or promociones_vacias
            or ventas_no_numericas or promociones_no_numericas):
        print("Los porcentajes usan solo valores numéricos; revisar exclusiones antes de comparar.")
    return resumen, fechas_observadas


def imprimir_distribuciones(resumen: pd.DataFrame) -> None:
    """Muestra cuantiles descriptivos, no criterios de selección."""
    print("\n=== Distribución entre series ===")
    print("Métrica                         mínimo        p25      mediana        p75        p90     máximo")
    metricas = (
        ("observaciones", "Observaciones"),
        ("volumen_total", "Volumen total sales"),
        ("media", "Media sales"),
        ("mediana", "Mediana sales"),
        ("desviacion", "Desv. estándar sales"),
        ("porcentaje_cero", "% días con sales=0"),
        ("porcentaje_fraccion", "% ventas fraccionarias"),
        ("porcentaje_promocion", "% días con promoción"),
    )
    for columna, etiqueta in metricas:
        cuantiles = resumen[columna].quantile([0, 0.25, 0.5, 0.75, 0.9, 1])
        valores = " ".join(f"{numero(float(v)):>10}" for v in cuantiles)
        print(f"{etiqueta:<29} {valores}")

    sin_variacion = resumen["minimo"].eq(resumen["maximo"]) & resumen["ventas_validas"].gt(0)
    todas_cero = resumen["ventas_cero"].eq(resumen["ventas_validas"]) & resumen["ventas_validas"].gt(0)
    p95_ceros = float(resumen["porcentaje_cero"].quantile(0.95))
    cola_alta = resumen["porcentaje_cero"].ge(p95_ceros)
    print("\n=== Casos descriptivos ===")
    print(f"Series sin variación de sales (mínimo = máximo): {entero(int(sin_variacion.sum()))}.")
    print(f"Series con todas las ventas cero: {entero(int(todas_cero.sum()))}.")
    print(f"Series constantes con ventas no cero: {entero(int((sin_variacion & ~todas_cero).sum()))}.")
    print(
        f"Percentil 95 del porcentaje de ceros: {numero(p95_ceros)}%; "
        f"series en o sobre ese valor: {entero(int(cola_alta.sum()))}."
    )
    print("El percentil 95 describe la cola observada; no establece un umbral de selección.")
    print(
        "Series con al menos un día con promoción: "
        f"{entero(int(resumen['dias_con_promocion'].gt(0).sum()))}; "
        "sin días con promoción: "
        f"{entero(int(resumen['dias_con_promocion'].eq(0).sum()))}."
    )
    print(
        "Totales de comprobación: ventas cero="
        f"{entero(int(resumen['ventas_cero'].sum()))}, ventas fraccionarias="
        f"{entero(int(resumen['ventas_fraccionarias'].sum()))}."
    )


def imprimir_grupos(resumen: pd.DataFrame, columna: str, etiqueta: str) -> None:
    """Compara grupos sin listar todas las tiendas, familias ni series."""
    grupos = resumen.groupby(columna).agg(
        series=("observaciones", "size"),
        volumen_total=("volumen_total", "sum"),
        porcentaje_cero_medio=("porcentaje_cero", "mean"),
        porcentaje_promocion_medio=("porcentaje_promocion", "mean"),
    )
    print(f"\n=== Diferencias por {etiqueta} ===")
    print(f"Grupos: {entero(len(grupos))}; series por grupo: "
          f"{entero(int(grupos['series'].min()))} a {entero(int(grupos['series'].max()))}.")
    for metrica, nombre, sufijo in (
        ("volumen_total", "Volumen total", ""),
        ("porcentaje_cero_medio", "Promedio del % de ceros de sus series", "%"),
        ("porcentaje_promocion_medio", "Promedio del % de días promocionados", "%"),
    ):
        valores = grupos[metrica]
        menor = valores.idxmin()
        mayor = valores.idxmax()
        print(
            f"{nombre}: mínimo {menor}={numero(float(valores.loc[menor]))}{sufijo}; "
            f"máximo {mayor}={numero(float(valores.loc[mayor]))}{sufijo}; "
            f"mediana entre grupos={numero(float(valores.median()))}{sufijo}."
        )
    print("Son agregados descriptivos del periodo completo, no una clasificación para seleccionar series.")


def examinar_fechas(
    fechas_observadas: set[str], ruta_feriados: Path, ruta_transacciones: Path
) -> None:
    """Contrasta ausencia de filas con feriados y transacciones cercanas."""
    feriados = pd.read_csv(ruta_feriados, dtype="string")
    transacciones = pd.read_csv(ruta_transacciones, dtype="string")
    transacciones["transactions_num"] = pd.to_numeric(
        transacciones["transactions"], errors="coerce"
    )
    transacciones_vacias = int(transacciones["transactions"].isna().sum())
    transacciones_no_numericas = int((
        transacciones["transactions"].notna()
        & transacciones["transactions_num"].isna()
    ).sum())

    print("\n=== Cuatro fechas sin filas de train.csv ===")
    print(
        "transactions.csv: valores vacíos="
        f"{entero(transacciones_vacias)}, no numéricos="
        f"{entero(transacciones_no_numericas)}."
    )
    for fecha in FECHAS_AUSENTES:
        print(f"\n{fecha}: filas en train.csv: "
              f"{'presentes' if fecha in fechas_observadas else 'ninguna'}.")
        eventos = feriados.loc[feriados["date"].eq(fecha)]
        print(f"  holidays_events.csv: {entero(len(eventos))} registro(s).")
        for evento in eventos.itertuples(index=False):
            print(
                f"    {evento.type}; {evento.locale}/{evento.locale_name}; "
                f"{evento.description}; transferred={evento.transferred}"
            )
        trans_dia = transacciones.loc[transacciones["date"].eq(fecha)]
        print(f"  transactions.csv: {entero(len(trans_dia))} fila(s) en la fecha.")
        dia = pd.Timestamp(fecha)
        for vecino in (dia - pd.Timedelta(days=1), dia + pd.Timedelta(days=1)):
            fecha_vecina = str(vecino.date())
            filas = transacciones.loc[transacciones["date"].eq(fecha_vecina)]
            print(
                f"    {fecha_vecina}: {entero(len(filas))} fila(s), "
                f"{entero(int(filas['transactions_num'].sum()))} transacciones registradas."
            )

    print(
        "Una fecha sin filas es ausencia de observación; sales=0 requiere una fila "
        "existente con valor cero. Los eventos y la ausencia de transacciones "
        "son evidencia contextual, pero no prueban por sí solos que todas las "
        "tiendas cerraron ni explican el proceso de generación del dataset."
    )


def main() -> int:
    try:
        train = verificar_columnas("train.csv", COLUMNAS_TRAIN)
        feriados = verificar_columnas("holidays_events.csv", COLUMNAS_FERIADOS)
        transacciones = verificar_columnas("transactions.csv", COLUMNAS_TRANSACCIONES)
        resumen, fechas = resumir_train(train)
        if resumen.empty:
            print("No hay series con clave tienda-familia válida.")
            return 1
        imprimir_distribuciones(resumen)
        imprimir_grupos(resumen, "store_nbr", "tienda")
        imprimir_grupos(resumen, "family", "familia")
        examinar_fechas(fechas, feriados, transacciones)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudo completar el análisis: {error}")
        return 1
    print("\nEste análisis usa todo train.csv solo para descripción; no selecciona series ni fija cortes temporales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
