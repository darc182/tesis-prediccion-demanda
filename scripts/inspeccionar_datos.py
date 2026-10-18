"""Inspección inicial de los CSV públicos, sin modificar datos ni crear salidas.

Ejecutar desde cualquier directorio con el Python del entorno virtual:
    .venv/Scripts/python scripts/inspeccionar_datos.py

train.csv se recorre por bloques. Las estadísticas numéricas se combinan entre
bloques; los días observados por tienda-familia se guardan como mapas de bits.
Esto permite revisar todas las claves válidas sin retener las filas completas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "raw"
TAMANO_BLOQUE = 100_000

ARCHIVOS = {
    "train.csv": "ventas históricas por fecha, tienda y familia",
    "test.csv": "fechas, tiendas, familias y promociones sin ventas reales",
    "stores.csv": "atributos de las tiendas",
    "transactions.csv": "transacciones por fecha y tienda",
    "holidays_events.csv": "descripciones de feriados y eventos",
    "oil.csv": "precio diario de petróleo",
    "sample_submission.csv": "formato de entrega; sales no es una venta real observada",
}
COLUMNAS_TRAIN = {"id", "date", "store_nbr", "family", "sales", "onpromotion"}


def entero(valor: int) -> str:
    """Muestra conteos con separador de miles legible en español."""
    return f"{valor:,}".replace(",", ".")


def numero(valor: float | None) -> str:
    """Muestra cuatro decimales o informa la ausencia de observaciones."""
    if valor is None:
        return "sin datos"
    return f"{valor:,.4f}".replace(",", "_").replace(".", ",").replace("_", ".")


@dataclass
class Estadisticas:
    """Media y desviación estándar muestral combinables entre bloques."""

    cantidad: int = 0
    media: float = 0.0
    suma_cuadrados: float = 0.0
    minimo: float | None = None
    maximo: float | None = None

    def agregar(self, valores: pd.Series) -> None:
        validos = valores.dropna()
        n = len(validos)
        if not n:
            return

        media_bloque = float(validos.mean())
        cuadrados_bloque = float(((validos - media_bloque) ** 2).sum())
        diferencia = media_bloque - self.media
        total = self.cantidad + n
        self.suma_cuadrados += (
            cuadrados_bloque
            + diferencia**2 * self.cantidad * n / total
        )
        self.media += diferencia * n / total
        self.cantidad = total

        minimo_bloque = float(validos.min())
        maximo_bloque = float(validos.max())
        self.minimo = (
            minimo_bloque if self.minimo is None else min(self.minimo, minimo_bloque)
        )
        self.maximo = (
            maximo_bloque if self.maximo is None else max(self.maximo, maximo_bloque)
        )

    @property
    def desviacion(self) -> float | None:
        if self.cantidad < 2:
            return None
        return (self.suma_cuadrados / (self.cantidad - 1)) ** 0.5


@dataclass
class Cobertura:
    """Un bit por día distinto observado para una tienda y familia."""

    dias: int = 0
    primero: int | None = None
    ultimo: int | None = None

    def agregar(self, dia: int, origen: int) -> bool:
        """Registra el día y devuelve si ya existía en esta combinación."""
        bit = 1 << (dia - origen)
        existia = bool(self.dias & bit)
        self.dias |= bit
        self.primero = dia if self.primero is None else min(self.primero, dia)
        self.ultimo = dia if self.ultimo is None else max(self.ultimo, dia)
        return existia


def inspeccionar_archivos() -> bool:
    """Comprueba presencia y lee solo la cabecera de cada CSV."""
    print("=== Archivos originales ===")
    completos = True
    for nombre, contenido in ARCHIVOS.items():
        ruta = DATOS / nombre
        if not ruta.is_file():
            print(f"{nombre}: NO ENCONTRADO")
            completos = False
            continue
        try:
            columnas = pd.read_csv(ruta, nrows=0).columns.tolist()
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            print(f"{nombre}: no se pudo leer la cabecera ({error})")
            completos = False
            continue
        print(f"{nombre}: {', '.join(columnas)}")
        print(f"  Contenido indicado por las columnas: {contenido}.")
        if nombre == "train.csv" and not COLUMNAS_TRAIN.issubset(columnas):
            faltantes = sorted(COLUMNAS_TRAIN - set(columnas))
            print(f"  Faltan columnas necesarias para el análisis: {faltantes}")
            completos = False
    print("Las columnas complementarias se describen, sin decidir su uso en modelos.")
    return completos


def imprimir_estadisticas(nombre: str, estadisticas: Estadisticas) -> None:
    print(
        f"{nombre}: n={entero(estadisticas.cantidad)}, "
        f"mín={numero(estadisticas.minimo)}, "
        f"media={numero(estadisticas.media) if estadisticas.cantidad else 'sin datos'}, "
        f"desv. estándar muestral={numero(estadisticas.desviacion)}, "
        f"máx={numero(estadisticas.maximo)}"
    )


def inspeccionar_train() -> None:
    """Recorre todas las filas y calcula métricas exactas sobre claves válidas.

    Los duplicados se cuentan por cada aparición posterior de una misma clave
    (date, store_nbr, family). Las fechas no ISO o claves incompletas se excluyen
    de esa revisión y de la cobertura, y se informa si ocurre.
    """
    filas = 0
    faltantes: dict[str, int] = {}
    tiendas: set[str] = set()
    familias: set[str] = set()
    combinaciones: set[tuple[str, str]] = set()
    coberturas: dict[tuple[str, str], Cobertura] = {}
    origen: int | None = None
    fecha_min: int | None = None
    fecha_max: int | None = None
    excluidas_claves = 0
    fechas_invalidas = 0
    fechas_no_iso = 0
    ventas_no_numericas = 0
    promociones_no_numericas = 0
    duplicados = 0
    ventas_cero = 0
    ventas_negativas = 0
    ventas_fraccionarias = 0
    ventas = Estadisticas()
    promociones = Estadisticas()

    bloques = pd.read_csv(
        DATOS / "train.csv",
        chunksize=TAMANO_BLOQUE,
        dtype="string",
        keep_default_na=False,
        na_values=[""],
    )
    for bloque in bloques:
        filas += len(bloque)
        for columna, cantidad in bloque.isna().sum().items():
            faltantes[columna] = faltantes.get(columna, 0) + int(cantidad)

        tiendas.update(bloque["store_nbr"].dropna().unique().tolist())
        familias.update(bloque["family"].dropna().unique().tolist())
        pares_validos = bloque["store_nbr"].notna() & bloque["family"].notna()
        combinaciones.update(
            bloque.loc[pares_validos, ["store_nbr", "family"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        sales = pd.to_numeric(bloque["sales"], errors="coerce")
        onpromotion = pd.to_numeric(bloque["onpromotion"], errors="coerce")
        ventas_no_numericas += int((bloque["sales"].notna() & sales.isna()).sum())
        promociones_no_numericas += int(
            (bloque["onpromotion"].notna() & onpromotion.isna()).sum()
        )
        ventas.agregar(sales)
        promociones.agregar(onpromotion)
        sales_validas = sales.dropna()
        ventas_cero += int(sales_validas.eq(0).sum())
        ventas_negativas += int(sales_validas.lt(0).sum())
        ventas_fraccionarias += int(sales_validas.mod(1).ne(0).sum())

        fechas = pd.to_datetime(bloque["date"], format="%Y-%m-%d", errors="coerce")
        fechas_invalidas += int((bloque["date"].notna() & fechas.isna()).sum())
        no_iso = (
            bloque["date"].notna()
            & fechas.notna()
            & bloque["date"].ne(fechas.dt.strftime("%Y-%m-%d"))
        ).fillna(False)
        fechas_no_iso += int(no_iso.sum())
        claves_validas = (
            bloque[["date", "store_nbr", "family"]].notna().all(axis=1)
            & fechas.notna()
            & ~no_iso
        )
        excluidas_claves += int((~claves_validas).sum())
        if not claves_validas.any():
            continue

        # La resta temporal evita depender de la resolución interna (ns/us)
        # que pandas asigna a datetime64 en cada versión.
        dias = (
            fechas.loc[claves_validas] - pd.Timestamp("1970-01-01")
        ).dt.days.astype("int64")
        minimo_bloque = int(dias.min())
        maximo_bloque = int(dias.max())
        fecha_min = minimo_bloque if fecha_min is None else min(fecha_min, minimo_bloque)
        fecha_max = maximo_bloque if fecha_max is None else max(fecha_max, maximo_bloque)

        # Si un bloque posterior contiene una fecha anterior, se desplaza cada
        # mapa de bits para mantener el mismo origen sin perder días ya vistos.
        if origen is None:
            origen = minimo_bloque
        elif minimo_bloque < origen:
            desplazamiento = origen - minimo_bloque
            for cobertura in coberturas.values():
                cobertura.dias <<= desplazamiento
            origen = minimo_bloque

        claves = bloque.loc[claves_validas, ["store_nbr", "family"]].copy()
        claves["dia"] = dias.to_numpy()
        for clave, grupo in claves.groupby(["store_nbr", "family"], sort=False):
            cobertura = coberturas.setdefault(clave, Cobertura())
            for dia, cantidad in grupo["dia"].value_counts(sort=False).items():
                existia = cobertura.agregar(int(dia), origen)
                duplicados += int(cantidad) if existia else int(cantidad) - 1

    print("\n=== train.csv: alcance y calidad ===")
    print(f"Filas leídas: {entero(filas)} (archivo completo, bloques de {entero(TAMANO_BLOQUE)}).")
    if fecha_min is not None and fecha_max is not None:
        inicio = pd.Timestamp("1970-01-01") + pd.Timedelta(days=fecha_min)
        fin = pd.Timestamp("1970-01-01") + pd.Timedelta(days=fecha_max)
        print(f"Rango de fechas válidas: {inicio.date()} a {fin.date()}.")
    else:
        print("Rango de fechas: no calculable; no se hallaron claves con fecha válida.")
    print(f"Tiendas distintas no nulas: {entero(len(tiendas))}.")
    print(f"Familias distintas no nulas: {entero(len(familias))}.")
    print(f"Combinaciones tienda-familia distintas no nulas: {entero(len(combinaciones))}.")

    print("\nValores faltantes por columna (celdas vacías del CSV):")
    for columna, cantidad in faltantes.items():
        print(f"  {columna}: {entero(cantidad)}")
    print(f"Valores sales no numéricos no vacíos: {entero(ventas_no_numericas)}.")
    print(f"Valores onpromotion no numéricos no vacíos: {entero(promociones_no_numericas)}.")
    print(f"Fechas no interpretables no vacías: {entero(fechas_invalidas)}.")
    print(f"Fechas interpretables pero no ISO AAAA-MM-DD: {entero(fechas_no_iso)}.")

    print("\nVentas y promociones (solo valores numéricos):")
    print(f"Ventas iguales a cero: {entero(ventas_cero)}.")
    print(f"Ventas negativas: {entero(ventas_negativas)}.")
    print(f"Ventas fraccionarias: {entero(ventas_fraccionarias)} (sales % 1 != 0).")
    imprimir_estadisticas("sales", ventas)
    imprimir_estadisticas("onpromotion", promociones)

    print("\nClaves (date, store_nbr, family):")
    print(f"Filas duplicadas posteriores a la primera: {entero(duplicados)}.")
    if excluidas_claves:
        print(
            f"Revisión PARCIAL: {entero(excluidas_claves)} filas con clave incompleta, "
            "fecha inválida o formato de fecha no canónico quedaron excluidas."
        )
    else:
        print("Revisión COMPLETA: todas las filas y todos los bloques fueron comparados.")

    print("\nCobertura temporal inicial por combinación tienda-familia:")
    if coberturas:
        dias_distintos = [c.dias.bit_count() for c in coberturas.values()]
        huecos = [
            c.ultimo - c.primero + 1 - c.dias.bit_count()
            for c in coberturas.values()
        ]
        comienzan_despues = sum(c.primero > fecha_min for c in coberturas.values())
        terminan_antes = sum(c.ultimo < fecha_max for c in coberturas.values())
        print(f"Combinaciones con al menos una fecha válida: {entero(len(coberturas))}.")
        print(
            "Días distintos observados por combinación: "
            f"mín={entero(min(dias_distintos))}, "
            f"mediana={numero(float(pd.Series(dias_distintos).median()))}, "
            f"máx={entero(max(dias_distintos))}."
        )
        print(
            "Combinaciones con días sin fila entre su primera y última fecha: "
            f"{entero(sum(h > 0 for h in huecos))}."
        )
        print(f"Total de días sin fila dentro de esos intervalos: {entero(sum(huecos))}.")
        union_dias = 0
        for cobertura in coberturas.values():
            union_dias |= cobertura.dias
        ausencia_global = [
            dia
            for dia in range(fecha_min, fecha_max + 1)
            if not union_dias & (1 << (dia - origen))
        ]
        print(
            "Días sin ninguna fila en todo train.csv dentro del rango global: "
            f"{entero(len(ausencia_global))}."
        )
        if len(ausencia_global) <= 10:
            fechas_ausentes = [
                str((pd.Timestamp("1970-01-01") + pd.Timedelta(days=dia)).date())
                for dia in ausencia_global
            ]
            print(f"Fechas sin filas: {', '.join(fechas_ausentes) or 'ninguna'}.")
        patrones = {(c.primero, c.ultimo, c.dias) for c in coberturas.values()}
        print(f"Patrones distintos de fechas observadas: {entero(len(patrones))}.")
        print(f"Combinaciones que comienzan después del mínimo global: {entero(comienzan_despues)}.")
        print(f"Combinaciones que terminan antes del máximo global: {entero(terminan_antes)}.")
        print("Los días sin fila describen cobertura; por sí solos no prueban errores.")
    else:
        print("No calculable: no se hallaron claves con fecha válida.")

    print("\n=== Observaciones de calidad ===")
    problemas = []
    if any(faltantes.values()):
        problemas.append("Hay celdas vacías en train.csv.")
    if ventas_no_numericas or promociones_no_numericas:
        problemas.append("Hay valores no numéricos en columnas numéricas.")
    if fechas_invalidas or fechas_no_iso:
        problemas.append("Hay fechas inválidas o con formato no canónico.")
    if duplicados:
        problemas.append("Hay claves fecha-tienda-familia repetidas.")
    if ventas_negativas:
        problemas.append("Hay ventas negativas; requieren interpretación antes de limpiarlas.")
    if excluidas_claves:
        problemas.append("La revisión de duplicados y cobertura es parcial.")
    if problemas:
        for problema in problemas:
            print(f"- {problema}")
    else:
        print("No se detectaron faltantes, claves duplicadas ni ventas negativas.")
    print("Ceros, fracciones y huecos temporales se informan sin tratarlos como errores automáticamente.")


def main() -> int:
    if not inspeccionar_archivos():
        print("\nAnálisis detenido: faltan archivos o columnas necesarios para una revisión completa.")
        return 1
    try:
        inspeccionar_train()
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"\nNo se pudo completar la lectura de train.csv: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
