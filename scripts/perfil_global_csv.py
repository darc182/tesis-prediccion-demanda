"""Panorama retrospectivo del CSV completo, sin SQL, selección ni entrenamiento.

Memoria: un bloque de B filas y mapas O(T + F + C + D) para tiendas, familias,
combinaciones y fechas distintas; no se guardan filas ni claves por observación.
Los límites siguientes acotan también el JSON y el calendario de ausencias.
El lector admite archivos desordenados. No verifica unicidad de date/tienda/familia:
esa validación independiente corresponde a validar_csv(), que ordena claves en
disco. Aquí se reutilizan sus reglas de columnas, fechas y enteros, sin ejecutarlo.

Antes del cálculo se comprueba la anchura de cada registro con csv.reader:
usecols puede ocultar campos sobrantes a pandas aunque on_bad_lines sea error.
Esta pasada adicional es secuencial, O(bytes del archivo), retiene un único
registro (incluidas columnas ignoradas) y no guarda filas ni claves. Después
se rebobina el mismo archivo abierto para el cálculo por bloques. La anchura
se compara con toda la cabecera, permitiendo columnas adicionales declaradas.

Ventas: float64 con suma compensada de Neumaier, sin redondeo de presentación.
La clasificación fraccionaria usa el Decimal del texto original. Se rechazan
valores no representables como float finito, subdesbordamientos a cero y sumas
no finitas. Las promociones se suman como enteros, interpretando el conteo literal
de la columna, sin afirmar causalidad ni cantidad de artículos únicos.

CLI: python -m scripts.perfil_global_csv ARCHIVO.csv [--tamano-bloque 100000]
JSON a stdout, resumen derivado del mismo reporte a stderr. No guarda archivos.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import json
from math import isfinite
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.validacion_csv import (
    COLUMNAS_OBLIGATORIAS, FECHA_ISO, _entero_no_negativo,
)


COLUMNAS = ("date", "store_nbr", "family", "sales", "onpromotion")
MAX_TAMANO_BLOQUE = 100_000
MAX_TIENDAS = 1_000
MAX_FAMILIAS = 1_000
MAX_COMBINACIONES = 10_000
MAX_FECHAS = 36_600
MAX_DIAS_CALENDARIO = 36_600
MAX_LONGITUD_FAMILIA = 512
MAX_LONGITUD_NUMERO = 128


class ErrorPerfilCSV(ValueError):
    """Perfil rechazado: no se devuelve un resultado parcial."""


@dataclass
class _Suma:
    valor: float = 0.0
    correccion: float = 0.0

    def agregar(self, numero):
        total = self.valor + numero
        if not isfinite(total):
            raise ErrorPerfilCSV("Acumulado de ventas no finito.")
        ajuste = ((self.valor - total) + numero if abs(self.valor) >= abs(numero)
                  else (numero - total) + self.valor)
        self.correccion += ajuste
        self.valor = total
        self.total()  # comprobar también la suma corregida en cada actualización

    def total(self):
        total = self.valor + self.correccion
        if not isfinite(total):
            raise ErrorPerfilCSV("Acumulado de ventas no finito.")
        return total


@dataclass
class _Grupo:
    filas: int = 0
    ventas: _Suma = field(default_factory=_Suma)

    def agregar(self, venta):
        self.filas += 1
        self.ventas.agregar(venta)


def _grupo(mapa, clave, limite, nombre):
    if clave not in mapa:
        if len(mapa) >= limite:
            raise ErrorPerfilCSV(f"Límite de {nombre} distintas excedido ({limite}).")
        mapa[clave] = _Grupo()
    return mapa[clave]


def _fila_valida(fila, numero):
    fecha_texto, tienda_texto, familia, venta_texto, promo_texto = fila
    error = f"Registro {numero}: "
    if not FECHA_ISO.fullmatch(fecha_texto):
        raise ErrorPerfilCSV(error + "date debe ser canónica YYYY-MM-DD.")
    try:
        fecha = date.fromisoformat(fecha_texto)
    except ValueError as causa:
        raise ErrorPerfilCSV(error + "date no es una fecha real.") from causa
    if len(tienda_texto) > MAX_LONGITUD_NUMERO or len(promo_texto) > MAX_LONGITUD_NUMERO:
        raise ErrorPerfilCSV(error + "Texto entero demasiado largo.")
    tienda = _entero_no_negativo(tienda_texto)
    promocion = _entero_no_negativo(promo_texto)
    if tienda is None or tienda == 0:
        raise ErrorPerfilCSV(error + "store_nbr debe ser entero positivo dentro del rango INTEGER de SQLite.")
    if promocion is None:
        raise ErrorPerfilCSV(error + "onpromotion debe ser entero no negativo dentro del rango INTEGER de SQLite.")
    if not familia.strip() or len(familia) > MAX_LONGITUD_FAMILIA:
        raise ErrorPerfilCSV(error + "family debe ser no vacía y tener como máximo 512 caracteres.")
    if len(venta_texto) > MAX_LONGITUD_NUMERO:
        raise ErrorPerfilCSV(error + "Texto de sales demasiado largo.")
    try:
        decimal = Decimal(venta_texto.strip())
        if not decimal.is_finite() or decimal < 0:
            raise ValueError
        venta = float(decimal)
        if not isfinite(venta) or (decimal > 0 and venta == 0):
            raise ValueError
    except (InvalidOperation, ValueError, OverflowError) as causa:
        raise ErrorPerfilCSV(error + "sales debe ser no negativo, finito y representable sin convertirse en cero.") from causa
    return fecha, tienda, familia, 0.0 if decimal == 0 else venta, promocion, decimal != decimal.to_integral_value()


def _comprobar_anchura(lector, columnas):
    for numero, registro in enumerate(lector, 1):
        if len(registro) > columnas:
            raise ErrorPerfilCSV(
                f"Registro {numero} (línea {lector.line_num}): más valores que cabeceras.")


def perfilar_csv_global(ruta_csv: str | Path, tamano_bloque: int = 100_000) -> dict:
    """Examina todo el archivo explícito; rechaza errores sin emitir perfiles parciales.

    Los porcentajes y promedios usan filas observadas, nunca días imputados.
    No verifica unicidad: cada registro leído contribuye una vez, incluso si
    su clave se repite. Los límites de enteros coinciden con el validador existente.
    No abre conexiones, escribe archivos ni aplica cortes de entrenamiento.
    """
    if type(tamano_bloque) is not int or not 1 <= tamano_bloque <= MAX_TAMANO_BLOQUE:
        raise ErrorPerfilCSV(f"tamano_bloque debe ser entero entre 1 y {MAX_TAMANO_BLOQUE}.")
    if not isinstance(ruta_csv, (str, Path)):
        raise ErrorPerfilCSV("Debe proporcionar una ruta CSV explícita.")
    ruta = Path(ruta_csv)
    tiendas, familias, combinaciones = {}, {}, {}
    fechas, anios = Counter(), Counter()
    ventas = _Suma()
    filas = ceros = fraccionarias = con_promocion = total_promociones = 0
    minimo = maximo = inicio = fin = None
    promo_min = promo_max = None
    try:
        with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
            lector = csv.reader(archivo, strict=True)
            cabeceras = next(lector, None)
            if not cabeceras:
                raise ErrorPerfilCSV("El CSV está vacío o no contiene cabeceras.")
            if len(cabeceras) != len(set(cabeceras)):
                raise ErrorPerfilCSV("El CSV contiene cabeceras duplicadas.")
            if COLUMNAS_OBLIGATORIAS - set(cabeceras):
                raise ErrorPerfilCSV("Faltan columnas obligatorias: " + ", ".join(sorted(COLUMNAS_OBLIGATORIAS - set(cabeceras))))
            _comprobar_anchura(lector, len(cabeceras))
            del lector
            archivo.seek(0)
            with pd.read_csv(archivo, usecols=list(COLUMNAS), chunksize=tamano_bloque,
                             dtype="string", keep_default_na=False, na_filter=False,
                             on_bad_lines="error") as bloques:
                for bloque in bloques:
                    for fila in bloque.loc[:, list(COLUMNAS)].itertuples(index=False, name=None):
                        fecha, tienda, familia, venta, promocion, fraccionaria = _fila_valida(fila, filas + 1)
                        filas += 1
                        inicio = fecha if inicio is None else min(inicio, fecha)
                        fin = fecha if fin is None else max(fin, fecha)
                        if (fin - inicio).days + 1 > MAX_DIAS_CALENDARIO:
                            raise ErrorPerfilCSV("Intervalo de calendario demasiado amplio.")
                        if fecha not in fechas and len(fechas) >= MAX_FECHAS:
                            raise ErrorPerfilCSV("Límite de fechas distintas excedido.")
                        fechas[fecha] += 1
                        anios[fecha.year] += 1
                        _grupo(tiendas, tienda, MAX_TIENDAS, "tiendas").agregar(venta)
                        _grupo(familias, familia, MAX_FAMILIAS, "familias").agregar(venta)
                        _grupo(combinaciones, (tienda, familia), MAX_COMBINACIONES, "combinaciones").agregar(venta)
                        ventas.agregar(venta)
                        minimo = venta if minimo is None else min(minimo, venta)
                        maximo = venta if maximo is None else max(maximo, venta)
                        ceros += venta == 0
                        fraccionarias += fraccionaria
                        con_promocion += promocion > 0
                        total_promociones += promocion
                        promo_min = promocion if promo_min is None else min(promo_min, promocion)
                        promo_max = promocion if promo_max is None else max(promo_max, promocion)
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError) as causa:
        raise ErrorPerfilCSV("No se pudo leer el archivo CSV; no se generó un perfil.") from causa
    if not filas:
        raise ErrorPerfilCSV("El CSV tiene cabeceras pero no contiene registros.")
    calendario = (fin - inicio).days + 1
    ausentes = [(inicio + timedelta(days=i)).isoformat() for i in range(calendario)
                if inicio + timedelta(days=i) not in fechas]
    mayores = sorted(combinaciones, key=lambda c: (-combinaciones[c].ventas.total(), c[0], c[1]))[:10]
    return {
        "fuente": {"nombre_archivo": ruta.name, "columnas_examinadas": list(COLUMNAS)},
        "alcance": "panorama_descriptivo_global",
        "interpretacion_temporal": {
            "retrospectivo": True, "incluye_todas_las_fechas_del_archivo": True,
            "autorizado_para_seleccion_de_modelos": False,
            "nota": "Puede incluir observaciones posteriores a los cortes de validación; no es evidencia disponible antes del entrenamiento.",
        },
        "volumen": {"filas_examinadas": filas, "filas_validas": filas, "filas_invalidas": 0,
                    "tiendas_distintas": len(tiendas), "familias_distintas": len(familias),
                    "combinaciones_observadas": len(combinaciones)},
        "temporalidad": {"primera_fecha": inicio.isoformat(), "ultima_fecha": fin.isoformat(),
                         "fechas_distintas": len(fechas), "dias_calendario_inclusivos": calendario,
                         "fechas_sin_registro_global": ausentes,
                         "filas_por_anio": [{"anio": a, "filas": anios[a]} for a in sorted(anios)],
                         "filas_por_fecha": [{"fecha": f.isoformat(), "filas": fechas[f]} for f in sorted(fechas)]},
        "ventas": {"suma": ventas.total(), "promedio_por_registro": ventas.total() / filas,
                   "minimo": minimo, "maximo": maximo, "registros_cero": ceros,
                   "porcentaje_cero": 100 * ceros / filas, "registros_fraccionarios": fraccionarias,
                   "porcentaje_fraccionarios": 100 * fraccionarias / filas, "registros_positivos": filas - ceros},
        "promociones": {"registros_cero": filas - con_promocion, "registros_positivos": con_promocion,
                        "porcentaje_con_promocion": 100 * con_promocion / filas,
                        "suma_conteos_onpromotion": total_promociones,
                        "minimo": promo_min, "maximo": promo_max,
                        "promedio_por_registro": total_promociones / filas},
        "distribuciones": {
            "por_tienda": [{"store_nbr": t, "filas": tiendas[t].filas} for t in sorted(tiendas)],
            "por_familia": [{"family": f, "filas": familias[f].filas, "ventas": familias[f].ventas.total()}
                            for f in sorted(familias)],
            "combinaciones_mayor_volumen": [{"store_nbr": t, "family": f,
                                             "filas": combinaciones[t, f].filas,
                                             "ventas": combinaciones[t, f].ventas.total()} for t, f in mayores],
            "criterio_mayores": "Hasta 10; ventas descendentes, tienda numérica ascendente, familia lexicográfica ascendente.",
        },
        "advertencias": [
            "No se verifica unicidad date+store_nbr+family; ejecutar validar_csv() independientemente si se requiere.",
            "Las ausencias son globales; no describen cobertura diaria individual ni se imputan como ceros.",
            "Las mayores combinaciones no son la selección oficial de diez series de la tesis.",
            "onpromotion se resume como conteo registrado; no mide artículos únicos ni causalidad sobre ventas.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", type=Path, help="CSV explícito; no hay archivo predeterminado")
    parser.add_argument("--tamano-bloque", type=int, default=100_000)
    args = parser.parse_args(argv)
    try:
        reporte = perfilar_csv_global(args.archivo, args.tamano_bloque)
    except ErrorPerfilCSV as causa:
        print(f"Perfil rechazado: {causa}", file=sys.stderr)
        return 1
    print(json.dumps(reporte, ensure_ascii=False, allow_nan=False, indent=2))
    print(f"Panorama descriptivo global: {reporte['volumen']['filas_validas']} registros; "
          f"ventas={reporte['ventas']['suma']}; "
          f"{len(reporte['temporalidad']['fechas_sin_registro_global'])} días sin registro global. "
          "No habilita selección de modelos.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
