"""Carga transaccional de las series habilitadas de un CSV ya validado.

El CSV se copia antes de validarlo: la importación lee esa misma copia. Los
conjuntos anteriores permanecen en la base y el nuevo se activa solo al final
de una transacción que incluye ventas y perfil del CSV completo. G1 y su
preparación se ejecutan antes de adquirir el bloqueo de escritura.
Tras importar se relee la copia por bloques bajo ese bloqueo para verificar
su SHA-256 antes de guardar el perfil y activar el conjunto.
"""

from __future__ import annotations

import csv
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

from backend.app.services.validacion_csv import TAMANO_HASH, validar_csv
from backend.app.services.perfil_global_conjunto import (
    ErrorPerfilGlobal, preparar_perfil_global, guardar_perfil_global,
)
from scripts.perfil_global_csv import ErrorPerfilCSV, perfilar_csv_global


CONFIG_SERIES = Path(__file__).resolve().parents[3] / "config" / "series_seleccionadas.json"
TAMANO_LOTE = 1_000
CodigoFalloCarga: TypeAlias = Literal[
    "csv_invalido", "sin_series_utilizables", "base_no_disponible",
    "configuracion_invalida", "fallo_interno", "perfil_no_generable",
]
_ERROR_BASE = "La base SQLite no está disponible o no tiene el esquema necesario."
_ERROR_PERFIL = "No se pudo generar un perfil global válido para el archivo."
_ERROR_INTERNO = "No se pudo completar la carga de datos."


class _ErrorIdentidadCopia(RuntimeError):
    """La copia consumida ya no coincide con la identidad validada del CSV."""


def _sha256_copia(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(TAMANO_HASH), b""):
            huella.update(bloque)
    return huella.hexdigest()


@dataclass(frozen=True)
class ResultadoCargaDatos:
    exito: bool
    conjunto_id: int | None
    sha256: str | None
    series_importadas: int
    observaciones_importadas: int
    errores: tuple[str, ...]
    advertencias: tuple[str, ...]
    codigo_error: CodigoFalloCarga | None = None


def _series_habilitadas(ruta_config: Path) -> dict[tuple[int, str], str]:
    """Lee las diez claves registradas; no calcula una nueva selección."""
    config = json.loads(ruta_config.read_text(encoding="utf-8"))
    version = config["regla_seleccion"]["version"]
    series = config["series"]
    if version != "1.0" or not isinstance(series, list) or len(series) != 10:
        raise ValueError("La configuración debe contener diez series de la regla 1.0.")
    habilitadas: dict[tuple[int, str], str] = {}
    for serie in series:
        tienda, familia, grupo = (
            serie["store_nbr"], serie["family"], serie["grupo"]
        )
        if (type(tienda) is not int or tienda <= 0 or not isinstance(familia, str)
                or not familia.strip() or grupo not in {"A", "B", "C", "D"}):
            raise ValueError("La configuración contiene una serie inválida.")
        clave = (tienda, familia)
        if clave in habilitadas:
            raise ValueError(f"La configuración repite la serie {clave}.")
        habilitadas[clave] = grupo
    return habilitadas


def _fallo(
    mensaje: str, codigo_error: CodigoFalloCarga, sha256: str | None = None,
) -> ResultadoCargaDatos:
    return ResultadoCargaDatos(False, None, sha256, 0, 0, (mensaje,), (), codigo_error)


def _comprobar_almacenamiento(conexion: sqlite3.Connection) -> None:
    """Solo lectura; se repite bajo BEGIN IMMEDIATE, sin inicializar tablas."""
    columnas = {
        "conjunto_datos": "id,nombre_archivo,hash,fecha_carga,activo",
        "serie_temporal": "id,conjunto_id,store_nbr,family,criterio_seleccion",
        "venta_diaria": "id,serie_id,fecha,sales,onpromotion",
        "perfil_global_conjunto": "conjunto_id,version_perfil,sha256_csv,reporte_json,sha256_reporte,series_importadas,observaciones_importadas",
    }
    tablas = {f[0] for f in conexion.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not columnas.keys() <= tablas:
        raise sqlite3.DatabaseError("Esquema de carga incompleto")
    for tabla, campos in columnas.items():
        conexion.execute(f"SELECT {campos} FROM {tabla} LIMIT 0")


def _abrir_escritura(ruta: Path) -> sqlite3.Connection:
    # rw impide crear una base si desaparece después de la comprobación previa.
    conexion = sqlite3.connect(f"{ruta.resolve().as_uri()}?mode=rw", uri=True)
    try:
        conexion.execute("PRAGMA foreign_keys=ON")
        return conexion
    except Exception:
        conexion.close()
        raise


def cargar_csv(
    ruta_csv: str | Path,
    ruta_base: str | Path,
    ruta_config: str | Path = CONFIG_SERIES,
    tamano_lote: int = TAMANO_LOTE,
) -> ResultadoCargaDatos:
    """Valida una copia y carga solo las observaciones de series habilitadas.

    La base y la tabla de perfiles deben existir. Cualquier fallo al importar,
    guardar el perfil o activar revierte el conjunto nuevo y deja el anterior.
    """
    if tamano_lote < 1:
        raise ValueError("tamano_lote debe ser positivo.")
    ruta_csv = Path(ruta_csv)
    ruta_base = Path(ruta_base)
    ruta_config = Path(ruta_config)

    sha256 = None
    try:
        with TemporaryDirectory(prefix="carga_csv_") as temporal:
            copia = Path(temporal) / "datos.csv"
            shutil.copyfile(ruta_csv, copia)
            validacion = validar_csv(copia)
            sha256 = validacion.sha256
            if not validacion.valido:
                return ResultadoCargaDatos(
                    False, None, validacion.sha256, 0, 0,
                    validacion.errores, (), "csv_invalido",
                )

            try:
                habilitadas = _series_habilitadas(ruta_config)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError):
                return _fallo("La configuración de series no es válida.",
                              "configuracion_invalida", validacion.sha256)
            try:
                with closing(sqlite3.connect(f"{ruta_base.resolve().as_uri()}?mode=ro", uri=True)) as lectura:
                    _comprobar_almacenamiento(lectura)
            except (OSError, ValueError, sqlite3.Error):
                return _fallo(_ERROR_BASE, "base_no_disponible", sha256)

            try:
                reporte = perfilar_csv_global(copia)
                preparado = preparar_perfil_global(reporte, ruta_csv.name)
            except ErrorPerfilCSV:
                return _fallo(_ERROR_PERFIL, "perfil_no_generable", sha256)
            except ErrorPerfilGlobal as error:
                if error.codigo in {"perfil_invalido", "perfil_demasiado_grande"}:
                    return _fallo(_ERROR_PERFIL, "perfil_no_generable", sha256)
                return _fallo(_ERROR_INTERNO, "fallo_interno", sha256)
            if (reporte["volumen"]["filas_examinadas"] != validacion.registros_examinados
                    or reporte["temporalidad"]["primera_fecha"] != validacion.fecha_minima
                    or reporte["temporalidad"]["ultima_fecha"] != validacion.fecha_maxima):
                return _fallo(_ERROR_INTERNO, "fallo_interno", sha256)

            try:
                with closing(_abrir_escritura(ruta_base)) as conexion:
                    try:
                        conexion.execute("BEGIN IMMEDIATE")
                        _comprobar_almacenamiento(conexion)
                        cursor = conexion.execute(
                            "INSERT INTO conjunto_datos "
                            "(nombre_archivo, hash, fecha_carga, activo) "
                            "VALUES (?, ?, ?, 0)",
                            (ruta_csv.name, validacion.sha256,
                             datetime.now(timezone.utc).isoformat(timespec="seconds")),
                        )
                        conjunto_id = int(cursor.lastrowid)
                        ids_series: dict[tuple[int, str], int] = {}
                        lote: list[tuple[int, str, float, int]] = []
                        observaciones = 0

                        with copia.open("r", encoding="utf-8-sig", newline="") as archivo:
                            for fila in csv.DictReader(archivo, strict=True):
                                clave = (int(fila["store_nbr"]), fila["family"])
                                grupo = habilitadas.get(clave)
                                if grupo is None:
                                    continue
                                if clave not in ids_series:
                                    cursor = conexion.execute(
                                        "INSERT INTO serie_temporal "
                                        "(conjunto_id, store_nbr, family, criterio_seleccion) "
                                        "VALUES (?, ?, ?, ?)",
                                        (conjunto_id, clave[0], clave[1],
                                         f"Regla de selección 1.0; grupo {grupo}"),
                                    )
                                    ids_series[clave] = int(cursor.lastrowid)
                                lote.append((
                                    ids_series[clave], fila["date"],
                                    float(fila["sales"]), int(fila["onpromotion"]),
                                ))
                                if len(lote) >= tamano_lote:
                                    conexion.executemany(
                                        "INSERT INTO venta_diaria "
                                        "(serie_id, fecha, sales, onpromotion) "
                                        "VALUES (?, ?, ?, ?)", lote,
                                    )
                                    observaciones += len(lote)
                                    lote.clear()

                        if not ids_series:
                            conexion.rollback()
                            return _fallo(
                                "El CSV no contiene ninguna de las series habilitadas; "
                                "no hay series utilizables.",
                                "sin_series_utilizables", validacion.sha256,
                            )
                        if lote:
                            conexion.executemany(
                                "INSERT INTO venta_diaria "
                                "(serie_id, fecha, sales, onpromotion) "
                                "VALUES (?, ?, ?, ?)", lote,
                            )
                            observaciones += len(lote)

                        # Pasada final independiente, después de consumir/importar las
                        # observaciones. Una divergencia revierte también las ventas.
                        if _sha256_copia(copia) != validacion.sha256:
                            raise _ErrorIdentidadCopia()
                        guardar_perfil_global(conexion, conjunto_id, preparado, validacion.sha256,
                                              len(ids_series), observaciones)
                        conexion.execute("UPDATE conjunto_datos SET activo = 0 WHERE activo = 1")
                        conexion.execute(
                            "UPDATE conjunto_datos SET activo = 1 WHERE id = ?",
                            (conjunto_id,),
                        )
                        conexion.commit()
                    except Exception as error:
                        conexion.rollback()
                        codigo_error: CodigoFalloCarga = (
                            "base_no_disponible"
                            if (isinstance(error, sqlite3.DatabaseError) and not isinstance(error, sqlite3.IntegrityError))
                            or (isinstance(error, ErrorPerfilGlobal) and error.codigo == "almacenamiento_no_disponible")
                            else "fallo_interno"
                        )
                        return _fallo(_ERROR_BASE if codigo_error == "base_no_disponible" else _ERROR_INTERNO,
                                      codigo_error, None if isinstance(error, _ErrorIdentidadCopia) else validacion.sha256)
            except (OSError, sqlite3.Error):
                return _fallo(_ERROR_BASE,
                              "base_no_disponible", validacion.sha256)

            ausentes = [
                f"Falta la serie tienda {tienda}, familia {familia} (grupo {grupo})."
                for (tienda, familia), grupo in habilitadas.items()
                if (tienda, familia) not in ids_series
            ]
            return ResultadoCargaDatos(
                True, conjunto_id, validacion.sha256, len(ids_series),
                observaciones, (), tuple(ausentes),
            )
    except Exception:
        return _fallo(_ERROR_INTERNO, "fallo_interno", sha256)
