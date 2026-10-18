"""Describe series usando exclusivamente el periodo previo a la validación.

Ejecución: .venv/Scripts/python scripts/analizar_periodo_seleccion.py

Primero se lee solamente la columna date de todo train.csv y se comprueba su
orden. Después, pandas lee sales y onpromotion solo en el prefijo que termina
el 2017-06-16. El análisis usa bloques y acumula medidas por tienda-familia;
no guarda resultados, no selecciona series y no fija umbrales de exclusión.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from typing import Iterable

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
TRAIN = RAIZ / "data" / "raw" / "train.csv"
TAMANO_BLOQUE = 100_000
INICIO_SELECCION = date(2013, 1, 1)
FIN_SELECCION = date(2017, 6, 16)
HORIZONTE_DIAS = 15
COLUMNAS_REQUERIDAS = {"date", "store_nbr", "family", "sales", "onpromotion"}


@dataclass(frozen=True)
class Ventana:
    nombre: str
    inicio: date
    fin: date

    @property
    def dias(self) -> int:
        return (self.fin - self.inicio).days + 1


VENTANAS_POSTERIORES = (
    Ventana("Validación 1", date(2017, 6, 17), date(2017, 7, 1)),
    Ventana("Validación 2", date(2017, 7, 2), date(2017, 7, 16)),
    Ventana("Validación 3", date(2017, 7, 17), date(2017, 7, 31)),
    Ventana("Prueba final", date(2017, 8, 1), date(2017, 8, 15)),
)


def validar_ventanas(
    ventanas: tuple[Ventana, ...] = VENTANAS_POSTERIORES,
    fin_seleccion: date = FIN_SELECCION,
) -> None:
    """Comprueba horizonte, orden, continuidad y ausencia de solapamiento."""
    if len(ventanas) != 4:
        raise ValueError("Deben definirse tres validaciones y una prueba final.")
    anterior = fin_seleccion
    for ventana in ventanas:
        if ventana.dias != HORIZONTE_DIAS:
            raise ValueError(f"{ventana.nombre} no tiene {HORIZONTE_DIAS} días.")
        if ventana.inicio != anterior + timedelta(days=1):
            raise ValueError(f"{ventana.nombre} se solapa o deja un hueco temporal.")
        anterior = ventana.fin


def verificar_columnas(ruta: Path) -> None:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    disponibles = set(pd.read_csv(ruta, nrows=0).columns)
    faltantes = COLUMNAS_REQUERIDAS - disponibles
    if faltantes:
        raise ValueError(f"Faltan columnas en train.csv: {', '.join(sorted(faltantes))}")
    print(f"Columnas disponibles en train.csv: {', '.join(sorted(disponibles))}.")


def contar_prefijo_fechas(
    ruta: Path, fin: date = FIN_SELECCION, tamano_bloque: int = TAMANO_BLOQUE
) -> tuple[int, int]:
    """Lee solo date; exige orden para poder limitar la lectura de ventas.

    Devuelve (filas hasta fin inclusive, filas totales). Si el CSV no está en
    orden cronológico, falla antes de abrir sales/onpromotion: un nrows no sería
    suficiente para impedir que entraran valores del periodo reservado.
    """
    filas_prefijo = 0
    filas_totales = 0
    ultima_fecha: pd.Timestamp | None = None
    for bloque in pd.read_csv(
        ruta, usecols=["date"], chunksize=tamano_bloque, dtype="string",
        keep_default_na=False, na_values=[""],
    ):
        if bloque["date"].isna().any():
            raise ValueError("Hay fechas vacías; no se puede garantizar el corte.")
        fechas = pd.to_datetime(bloque["date"], format="%Y-%m-%d", errors="raise")
        if not bloque["date"].eq(fechas.dt.strftime("%Y-%m-%d")).all():
            raise ValueError("Hay fechas con formato no canónico AAAA-MM-DD.")
        if not fechas.is_monotonic_increasing:
            raise ValueError("train.csv no está ordenado cronológicamente.")
        if ultima_fecha is not None and fechas.iloc[0] < ultima_fecha:
            raise ValueError("train.csv no está ordenado entre bloques.")
        ultima_fecha = fechas.iloc[-1]
        filas_prefijo += int(fechas.le(pd.Timestamp(fin)).sum())
        filas_totales += len(bloque)
    return filas_prefijo, filas_totales


def leer_bloques_hasta_corte(
    ruta: Path, filas_prefijo: int, tamano_bloque: int = TAMANO_BLOQUE
) -> Iterable[pd.DataFrame]:
    """Materializa sales/onpromotion solo en las filas anteriores al corte."""
    return pd.read_csv(
        ruta,
        usecols=sorted(COLUMNAS_REQUERIDAS),
        nrows=filas_prefijo,
        chunksize=tamano_bloque,
        dtype="string",
        keep_default_na=False,
        na_values=[""],
    )


@dataclass
class Acumulado:
    observaciones: int = 0
    ventas_validas: int = 0
    suma_ventas: float = 0.0
    media: float = 0.0
    suma_cuadrados: float = 0.0
    minimo: float | None = None
    maximo: float | None = None
    ventas_cero: int = 0
    ventas_fraccionarias: int = 0
    observaciones_promocion: int = 0
    dias_observados: int = 0
    hubo_venta_positiva: bool = False
    cero_despues_de_positiva: bool = False
    alternancia_cero_positiva: bool = False

    def agregar(self, grupo: pd.DataFrame) -> None:
        """Combina momentos para desviación muestral sin retener ventas."""
        self.observaciones += len(grupo)
        for dia in grupo["dia_periodo"].unique():
            self.dias_observados |= 1 << int(dia)

        ventas = grupo["sales_num"].dropna().astype("float64")
        n = len(ventas)
        if n:
            media_bloque = float(ventas.mean())
            cuadrados_bloque = float(((ventas - media_bloque) ** 2).sum())
            total_n = self.ventas_validas + n
            diferencia = media_bloque - self.media
            self.suma_cuadrados += (
                cuadrados_bloque
                + diferencia**2 * self.ventas_validas * n / total_n
            )
            self.media += diferencia * n / total_n
            self.ventas_validas = total_n
            self.suma_ventas += float(ventas.sum())
            minimo_bloque = float(ventas.min())
            maximo_bloque = float(ventas.max())
            self.minimo = (
                minimo_bloque if self.minimo is None
                else min(self.minimo, minimo_bloque)
            )
            self.maximo = (
                maximo_bloque if self.maximo is None
                else max(self.maximo, maximo_bloque)
            )
            self.ventas_cero += int(ventas.eq(0).sum())
            self.ventas_fraccionarias += int(ventas.mod(1).ne(0).sum())

        promociones = grupo["promocion_num"]
        self.observaciones_promocion += int(promociones.gt(0).sum())

        # Señal descriptiva: dos tramos con ventas positivas separados por al
        # menos una fila observada con sales=0. Una fecha ausente no es un cero.
        for valor in grupo["sales_num"]:
            if pd.isna(valor):
                continue
            if valor > 0:
                if self.cero_despues_de_positiva:
                    self.alternancia_cero_positiva = True
                self.hubo_venta_positiva = True
                self.cero_despues_de_positiva = False
            elif valor == 0 and self.hubo_venta_positiva:
                self.cero_despues_de_positiva = True

    def registro(self, tienda: str, familia: str) -> dict[str, object]:
        desviacion = (
            (self.suma_cuadrados / (self.ventas_validas - 1)) ** 0.5
            if self.ventas_validas > 1 else float("nan")
        )
        return {
            "store_nbr": tienda,
            "family": familia,
            "observaciones": self.observaciones,
            "dias_distintos": self.dias_observados.bit_count(),
            "ventas_validas": self.ventas_validas,
            "volumen_total": self.suma_ventas,
            "media": self.media if self.ventas_validas else float("nan"),
            "desviacion": desviacion,
            "minimo": self.minimo,
            "maximo": self.maximo,
            "ventas_cero": self.ventas_cero,
            "porcentaje_cero": (
                100 * self.ventas_cero / self.ventas_validas
                if self.ventas_validas else float("nan")
            ),
            "porcentaje_fraccionarias": (
                100 * self.ventas_fraccionarias / self.ventas_validas
                if self.ventas_validas else float("nan")
            ),
            "porcentaje_promocion": (
                100 * self.observaciones_promocion / self.observaciones
                if self.observaciones else float("nan")
            ),
            "alternancia_cero_positiva": self.alternancia_cero_positiva,
        }


@dataclass
class Resultado:
    series: pd.DataFrame
    fechas_observadas: set[date]
    filas_seleccion: int
    filas_fuera_intervalo: int
    ventas_vacias: int
    promociones_vacias: int
    ventas_no_numericas: int
    promociones_no_numericas: int


def calcular_caracteristicas(
    bloques: Iterable[pd.DataFrame],
    inicio: date = INICIO_SELECCION,
    fin: date = FIN_SELECCION,
) -> Resultado:
    """Filtra por fecha antes de convertir ventas o promociones en cada bloque."""
    acumulados: dict[tuple[str, str], Acumulado] = {}
    fechas_observadas: set[date] = set()
    filas_seleccion = filas_fuera_intervalo = 0
    ventas_vacias = promociones_vacias = 0
    ventas_no_numericas = promociones_no_numericas = 0
    for bloque in bloques:
        faltantes = COLUMNAS_REQUERIDAS - set(bloque.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas en un bloque: {sorted(faltantes)}")
        fechas = pd.to_datetime(bloque["date"], format="%Y-%m-%d", errors="raise")
        dentro = fechas.between(pd.Timestamp(inicio), pd.Timestamp(fin))
        filas_fuera_intervalo += int((~dentro).sum())
        if not dentro.any():
            continue
        seleccionado = bloque.loc[dentro].copy()
        fechas_seleccionadas = fechas.loc[dentro]
        seleccionado["dia_periodo"] = (
            fechas_seleccionadas - pd.Timestamp(inicio)
        ).dt.days.to_numpy()
        fechas_observadas.update(fechas_seleccionadas.dt.date.unique().tolist())
        filas_seleccion += len(seleccionado)

        if seleccionado[["store_nbr", "family"]].isna().any().any():
            raise ValueError("Hay filas del periodo sin tienda o familia.")
        ventas_vacias += int(seleccionado["sales"].isna().sum())
        promociones_vacias += int(seleccionado["onpromotion"].isna().sum())
        seleccionado["sales_num"] = pd.to_numeric(
            seleccionado["sales"], errors="coerce"
        )
        seleccionado["promocion_num"] = pd.to_numeric(
            seleccionado["onpromotion"], errors="coerce"
        )
        ventas_no_numericas += int((
            seleccionado["sales"].notna() & seleccionado["sales_num"].isna()
        ).sum())
        promociones_no_numericas += int((
            seleccionado["onpromotion"].notna()
            & seleccionado["promocion_num"].isna()
        ).sum())

        for clave, grupo in seleccionado.groupby(["store_nbr", "family"], sort=False):
            acumulados.setdefault(clave, Acumulado()).agregar(grupo)

    registros = [
        acumulado.registro(tienda, familia)
        for (tienda, familia), acumulado in acumulados.items()
    ]
    return Resultado(
        pd.DataFrame.from_records(registros), fechas_observadas,
        filas_seleccion, filas_fuera_intervalo, ventas_vacias,
        promociones_vacias, ventas_no_numericas, promociones_no_numericas,
    )


def dias_sin_registro(
    fechas_observadas: set[date], inicio: date = INICIO_SELECCION,
    fin: date = FIN_SELECCION,
) -> list[date]:
    """Días del calendario sin ninguna fila, independientemente de sales=0."""
    return [
        inicio + timedelta(days=desplazamiento)
        for desplazamiento in range((fin - inicio).days + 1)
        if inicio + timedelta(days=desplazamiento) not in fechas_observadas
    ]


def entero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def numero(valor: float) -> str:
    if not isfinite(valor):
        return "N/D"
    return f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def imprimir_distribuciones(series: pd.DataFrame) -> None:
    print("\n=== Distribución entre series del periodo inicial ===")
    print("Métrica                     mínimo        p25        p50        p75        p90        p95     máximo")
    for columna, etiqueta in (
        ("observaciones", "Observaciones"),
        ("volumen_total", "Volumen total sales"),
        ("media", "Media sales"),
        ("desviacion", "Desv. estándar sales"),
        ("porcentaje_cero", "% ventas cero"),
        ("porcentaje_fraccionarias", "% ventas fraccionarias"),
        ("porcentaje_promocion", "% observaciones promo>0"),
    ):
        cuantiles = series[columna].quantile([0, .25, .5, .75, .9, .95, 1])
        print(f"{etiqueta:<29}" + " ".join(
            f"{numero(float(valor)):>10}" for valor in cuantiles
        ))


def imprimir_diversidad(series: pd.DataFrame) -> None:
    print("\n=== Diversidad y casos observados ===")
    print(f"Combinaciones tienda-familia: {entero(len(series))}; "
          f"tiendas: {entero(series['store_nbr'].nunique())}; "
          f"familias: {entero(series['family'].nunique())}.")
    con_ventas = series["ventas_validas"].gt(0)
    todas_cero = con_ventas & series["ventas_cero"].eq(series["ventas_validas"])
    constantes = con_ventas & series["minimo"].eq(series["maximo"])
    alternantes = series["alternancia_cero_positiva"]
    print(f"Series con todas las ventas cero: {entero(int(todas_cero.sum()))}.")
    print(f"Series constantes (mínimo=máximo): {entero(int(constantes.sum()))}.")
    print(f"Series con patrón positivo-cero-positivo observado: "
          f"{entero(int(alternantes.sum()))}.")
    print("Alternancia significa dos tramos de ventas positivas separados por "
          "al menos un cero registrado; no es una regla de exclusión.")

    for columna, nombre in (("store_nbr", "tienda"), ("family", "familia")):
        grupos = series.groupby(columna).agg(
            n_series=("observaciones", "size"),
            ventas=("volumen_total", "sum"),
            cero_medio=("porcentaje_cero", "mean"),
        )
        menor_venta, mayor_venta = grupos["ventas"].idxmin(), grupos["ventas"].idxmax()
        menor_cero, mayor_cero = grupos["cero_medio"].idxmin(), grupos["cero_medio"].idxmax()
        print(f"Por {nombre}: {entero(len(grupos))} grupos; "
              f"{entero(int(grupos['n_series'].min()))}–"
              f"{entero(int(grupos['n_series'].max()))} series por grupo.")
        print(f"  Volumen: mínimo {menor_venta}="
              f"{numero(float(grupos.loc[menor_venta, 'ventas']))}, "
              f"máximo {mayor_venta}="
              f"{numero(float(grupos.loc[mayor_venta, 'ventas']))}.")
        print(f"  % cero medio: mínimo {menor_cero}="
              f"{numero(float(grupos.loc[menor_cero, 'cero_medio']))}%, "
              f"máximo {mayor_cero}="
              f"{numero(float(grupos.loc[mayor_cero, 'cero_medio']))}%.")


def imprimir_calendario(resultado: Resultado) -> None:
    print("\n=== Calendario del periodo inicial ===")
    dias_esperados = (FIN_SELECCION - INICIO_SELECCION).days + 1
    ausentes = dias_sin_registro(resultado.fechas_observadas)
    print(f"Días calendario esperados: {entero(dias_esperados)}; "
          f"con alguna fila: {entero(len(resultado.fechas_observadas))}; "
          f"sin registro global: {entero(len(ausentes))}.")
    if len(ausentes) <= 20:
        print("Fechas sin registro global: "
              + (", ".join(map(str, ausentes)) or "ninguna"))
    print(f"Filas observadas con sales=0: "
          f"{entero(int(resultado.series['ventas_cero'].sum()))}.")
    faltantes_por_serie = dias_esperados - resultado.series["dias_distintos"]
    print("Días sin fila por serie: "
          f"mínimo={entero(int(faltantes_por_serie.min()))}, "
          f"mediana={numero(float(faltantes_por_serie.median()))}, "
          f"máximo={entero(int(faltantes_por_serie.max()))}.")
    print("Un día sin fila no se contabiliza como sales=0; las ventas cero "
          "proceden solo de observaciones existentes.")


def main() -> int:
    try:
        validar_ventanas()
        verificar_columnas(TRAIN)
        filas_prefijo, filas_totales = contar_prefijo_fechas(TRAIN)
        if not filas_prefijo:
            raise ValueError("No hay filas antes del corte de selección.")
        resultado = calcular_caracteristicas(
            leer_bloques_hasta_corte(TRAIN, filas_prefijo)
        )
        if resultado.series.empty:
            raise ValueError("No hay series en el periodo de selección.")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"No se pudo completar el análisis: {error}")
        return 1

    print("\n=== Corte aprobado y control de lectura ===")
    print(f"Selección y entrenamiento inicial: {INICIO_SELECCION} a {FIN_SELECCION}.")
    print(f"Filas de fechas revisadas: {entero(filas_totales)}; "
          f"filas leídas con sales/onpromotion hasta el corte: "
          f"{entero(filas_prefijo)}; filas posteriores leídas solo por fecha: "
          f"{entero(filas_totales - filas_prefijo)}.")
    print(f"Filas dentro del intervalo inicial: {entero(resultado.filas_seleccion)}; "
          f"anteriores al inicio excluidas: {entero(resultado.filas_fuera_intervalo)}.")
    print(f"Sales vacías/no numéricas: {entero(resultado.ventas_vacias)}/"
          f"{entero(resultado.ventas_no_numericas)}; onpromotion vacías/no numéricas: "
          f"{entero(resultado.promociones_vacias)}/"
          f"{entero(resultado.promociones_no_numericas)}.")
    print("Ventanas posteriores verificadas (15 días cada una, cronológicas y sin solapamiento):")
    for ventana in VENTANAS_POSTERIORES:
        print(f"  {ventana.nombre}: {ventana.inicio} a {ventana.fin} "
              f"({ventana.dias} días).")

    imprimir_distribuciones(resultado.series)
    imprimir_diversidad(resultado.series)
    imprimir_calendario(resultado)
    print("\nLos percentiles son descriptivos. No se eligieron series ni se fijaron "
          "umbrales de selección.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
