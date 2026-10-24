"""Endpoint HTTP para cargar CSV mediante el servicio transaccional existente."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.services.carga_datos import (
    CodigoFalloCarga, ResultadoCargaDatos, cargar_csv,
)


RAIZ = Path(__file__).resolve().parents[3]
TAMANO_COPIA = 1024 * 1024
NOMBRES_RESERVADOS = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefijo}{numero}" for prefijo in ("COM", "LPT") for numero in range(1, 10)
}

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def obtener_ruta_base() -> Path:
    """Resuelve TESIS_DB_PATH desde la raíz si se proporciona una ruta relativa."""
    valor = os.environ.get("TESIS_DB_PATH") or "data/local/tesis.sqlite3"
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def _nombre_seguro(nombre: str | None) -> str | None:
    if not nombre:
        return None
    base = nombre.replace("\\", "/").rsplit("/", 1)[-1]
    if (not base or len(base) > 255 or not base.lower().endswith(".csv")
            or not base[:-4].strip(" .")
            or base[:-4].strip(" .").upper() in NOMBRES_RESERVADOS
            or any(not (caracter.isalnum() or caracter in "._- ()") for caracter in base)):
        return None
    return base


def _contenido_publico(resultado: ResultadoCargaDatos) -> dict:
    """Conserva la respuesta HTTP sin exponer el código interno de fallo."""
    return {
        "exito": resultado.exito,
        "conjunto_id": resultado.conjunto_id,
        "sha256": resultado.sha256,
        "series_importadas": resultado.series_importadas,
        "observaciones_importadas": resultado.observaciones_importadas,
        "errores": resultado.errores,
        "advertencias": resultado.advertencias,
    }


def _respuesta_error(codigo_error: CodigoFalloCarga, mensaje: str) -> JSONResponse:
    resultado = ResultadoCargaDatos(
        False, None, None, 0, 0, (mensaje,), (), codigo_error,
    )
    return JSONResponse(
        status_code=_codigo_fallo(resultado), content=_contenido_publico(resultado),
    )


def _codigo_fallo(resultado: ResultadoCargaDatos) -> int:
    return {
        "csv_invalido": 422,
        "sin_series_utilizables": 422,
        "base_no_disponible": 503,
        "configuracion_invalida": 500,
        "fallo_interno": 500,
    }.get(resultado.codigo_error, 500)


@router.get("/active/series")
def consultar_series_activas(ruta_base: Path = Depends(obtener_ruta_base)) -> dict:
    """Consulta las series del conjunto activo sin crear ni modificar la base."""
    consulta = """
        SELECT c.id, s.id, s.store_nbr, s.family
        FROM (
            SELECT id FROM conjunto_datos WHERE activo = 1
            ORDER BY id DESC LIMIT 1
        ) AS c
        LEFT JOIN serie_temporal AS s ON s.conjunto_id = c.id
        ORDER BY s.store_nbr ASC, s.family ASC
    """
    try:
        uri = f"{ruta_base.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conexion:
            filas = conexion.execute(consulta).fetchall()
    except (sqlite3.Error, OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="La base de datos no está disponible o no tiene el esquema necesario.",
        ) from error

    if not filas:
        return {"conjunto_id": None, "series": []}
    return {
        "conjunto_id": filas[0][0],
        "series": [
            {"id": serie_id, "store_nbr": tienda, "family": familia}
            for _, serie_id, tienda, familia in filas if serie_id is not None
        ],
    }


@router.post("/upload")
def subir_csv(
    file: UploadFile | None = File(default=None),
    ruta_base: Path = Depends(obtener_ruta_base),
) -> JSONResponse:
    """Copia por bloques y delega la validación e importación a cargar_csv."""
    if file is None:
        return _respuesta_error("csv_invalido", "Debe enviar un archivo CSV en el campo file.")
    nombre = _nombre_seguro(file.filename)
    if nombre is None:
        return _respuesta_error("csv_invalido", "El archivo debe tener un nombre base seguro y extensión .csv.")

    try:
        with TemporaryDirectory(prefix="upload_csv_") as temporal:
            ruta_temporal = Path(temporal) / nombre
            with ruta_temporal.open("wb") as destino:
                file.file.seek(0)
                while bloque := file.file.read(TAMANO_COPIA):
                    destino.write(bloque)
            resultado = cargar_csv(ruta_temporal, ruta_base)
    except Exception:
        return _respuesta_error("fallo_interno", "Ocurrió un error interno al procesar el archivo.")

    codigo = 200 if resultado.exito else _codigo_fallo(resultado)
    return JSONResponse(status_code=codigo, content=_contenido_publico(resultado))
