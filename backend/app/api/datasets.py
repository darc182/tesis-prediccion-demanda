"""Endpoints HTTP de carga y consulta del conjunto de datos activo."""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from datetime import date
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


def _validar_fecha(nombre: str, valor: str | None) -> str | None:
    if valor is None:
        return None
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", valor) is None:
        raise HTTPException(status_code=422, detail=f"{nombre} debe tener el formato YYYY-MM-DD.")
    try:
        date.fromisoformat(valor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"{nombre} debe ser una fecha real.") from error
    return valor


@router.get("/active/series/{serie_id}/sales")
def consultar_ventas_historicas(
    serie_id: int,
    desde: str | None = None,
    hasta: str | None = None,
    ruta_base: Path = Depends(obtener_ruta_base),
) -> dict:
    """Devuelve solo las observaciones almacenadas de una serie del conjunto activo."""
    desde = _validar_fecha("desde", desde)
    hasta = _validar_fecha("hasta", hasta)
    if desde is not None and hasta is not None and desde > hasta:
        raise HTTPException(status_code=422, detail="desde no puede ser posterior a hasta.")

    consulta = """
        SELECT c.id, s.id, s.store_nbr, s.family, v.fecha, v.sales, v.onpromotion
        FROM (
            SELECT id FROM conjunto_datos WHERE activo = 1
            ORDER BY id DESC LIMIT 1
        ) AS c
        JOIN serie_temporal AS s ON s.conjunto_id = c.id AND s.id = ?
        LEFT JOIN venta_diaria AS v ON v.serie_id = s.id
    """
    parametros: list[int | str] = [serie_id]
    if desde is not None:
        consulta += " AND v.fecha >= ?"
        parametros.append(desde)
    if hasta is not None:
        consulta += " AND v.fecha <= ?"
        parametros.append(hasta)
    consulta += " ORDER BY v.fecha ASC"

    try:
        uri = f"{ruta_base.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conexion:
            filas = conexion.execute(consulta, parametros).fetchall()
    except (sqlite3.Error, OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="La base de datos no está disponible o no tiene el esquema necesario.",
        ) from error

    if not filas:
        raise HTTPException(status_code=404, detail="La serie no existe en el conjunto activo.")

    conjunto_id, id_serie, tienda, familia = filas[0][:4]
    return {
        "conjunto_id": conjunto_id,
        "serie": {"id": id_serie, "store_nbr": tienda, "family": familia},
        "observaciones": [
            {"fecha": fecha, "sales": sales, "onpromotion": onpromotion}
            for _, _, _, _, fecha, sales, onpromotion in filas if fecha is not None
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
