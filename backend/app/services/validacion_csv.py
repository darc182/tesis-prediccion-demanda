"""Validación secuencial de CSV de ventas, sin modificar el archivo de entrada.

Las claves se ordenan en fragmentos temporales y luego se fusionan. Esto
detecta duplicados entre fragmentos sin retener todas las filas en memoria ni
usar SQLite. Los archivos temporales se eliminan al terminar la validación.
"""

from __future__ import annotations

import csv
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
from heapq import merge
import json
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterator


COLUMNAS_OBLIGATORIAS = frozenset(
    {"date", "store_nbr", "family", "sales", "onpromotion"}
)
FECHA_ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
ENTERO = re.compile(r"[0-9]+\Z")
TAMANO_BLOQUE = 100_000
MAX_ERRORES = 10
TAMANO_HASH = 1024 * 1024
MAX_ENTERO_SQLITE = 2**63 - 1


@dataclass(frozen=True)
class ResultadoValidacionCSV:
    valido: bool
    registros_examinados: int
    registros_invalidos: int
    errores: tuple[str, ...]
    fecha_minima: str | None
    fecha_maxima: str | None
    sha256: str | None


def _guardar_bloque(
    claves: list[tuple[str, int, str, int, bool]], directorio: Path,
    indice: int,
) -> Path:
    claves.sort()
    ruta = directorio / f"claves_{indice:06d}.jsonl"
    with ruta.open("w", encoding="utf-8", newline="\n") as archivo:
        for clave in claves:
            archivo.write(json.dumps(clave, ensure_ascii=True, separators=(",", ":")))
            archivo.write("\n")
    claves.clear()
    return ruta


def _entero_no_negativo(valor: str | None) -> int | None:
    if valor is None:
        return None
    texto = valor.strip()
    if not ENTERO.fullmatch(texto):
        return None
    try:
        numero = int(texto)
    except ValueError:
        return None
    return numero if numero <= MAX_ENTERO_SQLITE else None


def validar_csv(
    ruta: str | Path,
    tamano_bloque: int = TAMANO_BLOQUE,
    max_errores: int = MAX_ERRORES,
) -> ResultadoValidacionCSV:
    """Devuelve un diagnóstico completo sin cargar el CSV en memoria.

    Una fila con varios fallos se cuenta una sola vez como inválida. Las claves
    duplicadas se comparan por fecha ISO, tienda numérica y texto de familia
    tal como figura en el archivo. Las fechas ausentes no se rellenan.
    """
    if tamano_bloque < 1 or max_errores < 1:
        raise ValueError("tamano_bloque y max_errores deben ser positivos.")

    ruta = Path(ruta)
    errores: list[str] = []
    omitidos = 0
    examinados = invalidos = 0
    fecha_minima: date | None = None
    fecha_maxima: date | None = None
    sha256: str | None = None
    fallo_general = False

    def agregar_error(mensaje: str) -> None:
        nonlocal omitidos
        if len(errores) < max_errores:
            errores.append(mensaje)
        else:
            omitidos += 1

    try:
        huella = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(TAMANO_HASH), b""):
                huella.update(bloque)
        sha256 = huella.hexdigest()
        if ruta.stat().st_size == 0:
            agregar_error("El archivo CSV está vacío.")
            fallo_general = True
        else:
            with TemporaryDirectory(prefix="validacion_csv_") as temporal:
                directorio = Path(temporal)
                archivos_claves: list[Path] = []
                claves: list[tuple[str, int, str, int, bool]] = []
                with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
                    lector = csv.DictReader(archivo, strict=True)
                    cabeceras = lector.fieldnames
                    if cabeceras is None:
                        agregar_error("El archivo CSV no contiene cabeceras.")
                        fallo_general = True
                    else:
                        faltantes = COLUMNAS_OBLIGATORIAS - set(cabeceras)
                        if faltantes:
                            agregar_error(
                                "Faltan columnas obligatorias: "
                                + ", ".join(sorted(faltantes)) + "."
                            )
                            fallo_general = True
                        if len(cabeceras) != len(set(cabeceras)):
                            agregar_error("Hay cabeceras duplicadas.")
                            fallo_general = True

                    if not fallo_general:
                        for fila in lector:
                            examinados += 1
                            motivos: list[str] = []
                            if None in fila:
                                motivos.append("más valores que cabeceras")

                            fecha_texto = fila["date"]
                            fecha: date | None = None
                            if fecha_texto is not None and FECHA_ISO.fullmatch(fecha_texto):
                                try:
                                    fecha = date.fromisoformat(fecha_texto)
                                except ValueError:
                                    pass
                            if fecha is None:
                                motivos.append("date debe ser una fecha válida YYYY-MM-DD")
                            else:
                                fecha_minima = (
                                    fecha if fecha_minima is None else min(fecha_minima, fecha)
                                )
                                fecha_maxima = (
                                    fecha if fecha_maxima is None else max(fecha_maxima, fecha)
                                )

                            tienda = _entero_no_negativo(fila["store_nbr"])
                            if tienda is None or tienda == 0:
                                motivos.append(
                                    "store_nbr debe ser un entero positivo dentro del "
                                    "rango INTEGER de SQLite"
                                )

                            familia = fila["family"]
                            if familia is None or not familia.strip():
                                motivos.append("family no puede estar vacía")

                            venta_texto = fila["sales"]
                            try:
                                venta = Decimal(venta_texto.strip()) if venta_texto is not None else None
                            except InvalidOperation:
                                venta = None
                            if venta is None or not venta.is_finite() or venta < 0:
                                motivos.append("sales debe ser un número finito no negativo")
                            else:
                                try:
                                    venta_real = float(venta)
                                except (OverflowError, ValueError):
                                    venta_real = float("inf")
                                if not isfinite(venta_real) or (venta > 0 and venta_real == 0):
                                    motivos.append(
                                        "sales no se representa como REAL finito "
                                        "sin convertirse en cero"
                                    )

                            promocion = _entero_no_negativo(fila["onpromotion"])
                            if promocion is None:
                                motivos.append(
                                    "onpromotion debe ser un entero no negativo "
                                    "dentro del rango INTEGER de SQLite"
                                )

                            if motivos:
                                invalidos += 1
                                agregar_error(f"Registro {examinados} (línea {lector.line_num}): "
                                              + "; ".join(motivos) + ".")
                            if fecha is not None and tienda is not None and tienda > 0 \
                                    and familia is not None and familia.strip():
                                claves.append((fecha.isoformat(), tienda, familia,
                                               examinados, bool(motivos)))
                            if len(claves) >= tamano_bloque:
                                archivos_claves.append(
                                    _guardar_bloque(claves, directorio, len(archivos_claves))
                                )

                        if claves:
                            archivos_claves.append(
                                _guardar_bloque(claves, directorio, len(archivos_claves))
                            )
                        if examinados == 0:
                            agregar_error("El CSV tiene cabeceras, pero no contiene registros.")
                            fallo_general = True

                        with ExitStack() as pila:
                            iteradores = [pila.enter_context(_abrir_claves(r))
                                          for r in archivos_claves]
                            clave_anterior: tuple[str, int, str] | None = None
                            for fecha, tienda, familia, numero, ya_invalida in merge(*iteradores):
                                clave = (fecha, tienda, familia)
                                if clave == clave_anterior:
                                    if not ya_invalida:
                                        invalidos += 1
                                    agregar_error(
                                        f"Registro {numero}: clave duplicada "
                                        f"({fecha}, {tienda}, {familia})."
                                    )
                                clave_anterior = clave
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        agregar_error(f"No se pudo completar la validación del CSV: {error}")
        fallo_general = True

    if omitidos:
        errores[-1] += f" Se omitieron {omitidos} mensajes adicionales."
    return ResultadoValidacionCSV(
        valido=not fallo_general and invalidos == 0,
        registros_examinados=examinados,
        registros_invalidos=invalidos,
        errores=tuple(errores),
        fecha_minima=fecha_minima.isoformat() if fecha_minima else None,
        fecha_maxima=fecha_maxima.isoformat() if fecha_maxima else None,
        sha256=sha256,
    )


@contextmanager
def _abrir_claves(ruta: Path) -> Iterator[Iterator[tuple[str, int, str, int, bool]]]:
    with ruta.open("r", encoding="utf-8") as archivo:
        def iterar() -> Iterator[tuple[str, int, str, int, bool]]:
            for linea in archivo:
                fecha, tienda, familia, fila, invalida = json.loads(linea)
                yield fecha, tienda, familia, fila, invalida
        yield iterar()
