"""Carga transaccional de las series habilitadas de un CSV ya validado.

El CSV se copia antes de validarlo: la importación lee esa misma copia. Los
conjuntos anteriores permanecen en la base y el nuevo se activa solo al final
de una transacción que incluye todas las inserciones.
"""

from __future__ import annotations

import csv
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

from backend.app.services.validacion_csv import validar_csv
from backend.app.storage.db import conectar


CONFIG_SERIES = Path(__file__).resolve().parents[3] / "config" / "series_seleccionadas.json"
TAMANO_LOTE = 1_000
CodigoFalloCarga: TypeAlias = Literal[
    "csv_invalido", "sin_series_utilizables", "base_no_disponible",
    "configuracion_invalida", "fallo_interno",
]


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


def cargar_csv(
    ruta_csv: str | Path,
    ruta_base: str | Path,
    ruta_config: str | Path = CONFIG_SERIES,
    tamano_lote: int = TAMANO_LOTE,
) -> ResultadoCargaDatos:
    """Valida una copia y carga solo las observaciones de series habilitadas.

    La base debe existir y tener el esquema inicializado. Cualquier fallo en
    la importación o activación revierte el conjunto nuevo y deja el anterior.
    """
    if tamano_lote < 1:
        raise ValueError("tamano_lote debe ser positivo.")
    ruta_csv = Path(ruta_csv)
    ruta_base = Path(ruta_base)
    ruta_config = Path(ruta_config)

    try:
        with TemporaryDirectory(prefix="carga_csv_") as temporal:
            copia = Path(temporal) / "datos.csv"
            shutil.copyfile(ruta_csv, copia)
            validacion = validar_csv(copia)
            if not validacion.valido:
                return ResultadoCargaDatos(
                    False, None, validacion.sha256, 0, 0,
                    validacion.errores, (), "csv_invalido",
                )

            try:
                habilitadas = _series_habilitadas(ruta_config)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
                return _fallo(f"No se pudo leer la configuración de series: {error}",
                              "configuracion_invalida", validacion.sha256)
            if not ruta_base.is_file():
                return _fallo("La base SQLite no existe o no está inicializada.",
                              "base_no_disponible", validacion.sha256)

            try:
                with closing(conectar(ruta_base)) as conexion:
                    try:
                        conexion.execute("BEGIN IMMEDIATE")
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
                            if isinstance(error, sqlite3.DatabaseError)
                            and not isinstance(error, sqlite3.IntegrityError)
                            else "fallo_interno"
                        )
                        return _fallo(f"La carga se revirtió por un error: {error}",
                                      codigo_error, validacion.sha256)
            except (OSError, sqlite3.Error) as error:
                return _fallo(f"No se pudo abrir la base SQLite: {error}",
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
    except (OSError, UnicodeError, ValueError) as error:
        return _fallo(f"No se pudo preparar la carga: {error}", "fallo_interno")
