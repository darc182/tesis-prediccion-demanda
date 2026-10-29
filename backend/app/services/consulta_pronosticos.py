"""Consultas de pronósticos almacenados: conexiones ro y ninguna predicción."""

from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta
from math import isfinite
from numbers import Real
from pathlib import Path
import sqlite3

from scripts.analizar_periodo_seleccion import HORIZONTE_DIAS


class ErrorConsultaPronostico(ValueError):
    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


@contextmanager
def _conexion_lectura(ruta_base: str | Path):
    try:
        uri = f"{Path(ruta_base).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conexion:
            conexion.row_factory = sqlite3.Row
            # Conjunto, cabecera y detalles se leen en la misma instantánea.
            conexion.execute("BEGIN")
            yield conexion
    except (sqlite3.Error, OSError) as error:
        raise ErrorConsultaPronostico("base_no_disponible", "La base de datos no está disponible.") from error


def _activo(conexion):
    fila = conexion.execute(
        "SELECT id FROM conjunto_datos WHERE activo = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if fila is None:
        raise ErrorConsultaPronostico("sin_conjunto_activo", "No existe un conjunto activo.")
    return fila["id"]


def _fecha(texto):
    fecha = date.fromisoformat(texto)
    if fecha.isoformat() != texto:
        raise ValueError("Fecha no canónica")
    return fecha


def _corrupto():
    return ErrorConsultaPronostico(
        "pronostico_corrupto", "El pronóstico almacenado está incompleto o presenta datos inconsistentes.")


def _metadatos(fila):
    """Expone solo metadatos públicos; no incluye rutas ni configuración interna."""
    claves = ("id", "experimento_id", "tecnica", "fecha_corte", "tipo", "horizonte",
              "fecha_generacion", "inicio_entrenamiento", "fin_entrenamiento", "duracion_segundos")
    datos = {k: fila[k] for k in claves}
    try:
        corte = _fecha(datos["fecha_corte"])
        inicio = _fecha(datos["inicio_entrenamiento"])
        fin = _fecha(datos["fin_entrenamiento"])
        datetime.fromisoformat(datos["fecha_generacion"])
        duracion = datos["duracion_segundos"]
        if (type(datos["horizonte"]) is not int or datos["horizonte"] != HORIZONTE_DIAS
                or not inicio <= fin <= corte or datos["tipo"] not in {"validacion", "prueba", "futuro"}
                or (duracion is not None and (not isinstance(duracion, Real)
                                              or not isfinite(duracion) or duracion < 0))):
            raise ValueError("Metadatos inconsistentes")
    except (TypeError, ValueError, OverflowError) as error:
        raise _corrupto() from error
    return datos


_CABECERA = """
    SELECT p.id, p.experimento_id, e.tecnica, p.fecha_corte, p.tipo,
           p.horizonte, p.fecha_generacion, e.inicio_entrenamiento,
           e.fin_entrenamiento, p.duracion_segundos
    FROM pronostico p JOIN experimento e ON e.id = p.experimento_id
"""


def listar_pronosticos(ruta_base: str | Path, serie_id: int) -> dict:
    with _conexion_lectura(ruta_base) as conexion:
        conjunto_id = _activo(conexion)
        serie = conexion.execute(
            "SELECT id, store_nbr, family FROM serie_temporal WHERE id = ? AND conjunto_id = ?",
            (serie_id, conjunto_id),
        ).fetchone()
        if serie is None:
            raise ErrorConsultaPronostico("serie_inexistente", "La serie no existe en el conjunto activo.")
        filas = conexion.execute(
            _CABECERA + " WHERE e.serie_id = ? ORDER BY p.fecha_corte DESC, e.tecnica ASC, p.id ASC",
            (serie_id,),
        ).fetchall()
        return {"conjunto_id": conjunto_id, "serie": dict(serie),
                "pronosticos": [_metadatos(fila) for fila in filas]}


def consultar_pronostico(ruta_base: str | Path, pronostico_id: int) -> dict:
    with _conexion_lectura(ruta_base) as conexion:
        conjunto_id = _activo(conexion)
        fila = conexion.execute(
            "SELECT p.id, p.experimento_id, e.tecnica, p.fecha_corte, p.tipo, p.horizonte, "
            "p.fecha_generacion, e.inicio_entrenamiento, e.fin_entrenamiento, p.duracion_segundos, "
            "s.id AS serie_id, s.store_nbr, s.family FROM pronostico p "
            "JOIN experimento e ON e.id = p.experimento_id "
            "JOIN serie_temporal s ON s.id = e.serie_id "
            "WHERE p.id = ? AND s.conjunto_id = ?", (pronostico_id, conjunto_id),
        ).fetchone()
        if fila is None:
            raise ErrorConsultaPronostico("pronostico_inexistente", "El pronóstico no existe en el conjunto activo.")
        metadatos = _metadatos(fila)
        detalles = conexion.execute(
            "SELECT paso, fecha_objetivo, venta_estimada FROM detalle_pronostico "
            "WHERE pronostico_id = ? ORDER BY paso", (pronostico_id,),
        ).fetchall()
        if len(detalles) != HORIZONTE_DIAS:
            raise _corrupto()
        corte = _fecha(metadatos["fecha_corte"])
        try:
            for paso, detalle in enumerate(detalles, 1):
                valor = detalle["venta_estimada"]
                if (type(detalle["paso"]) is not int or detalle["paso"] != paso
                        or _fecha(detalle["fecha_objetivo"]) != corte + timedelta(days=paso)
                        or isinstance(valor, bool) or not isinstance(valor, Real)
                        or not isfinite(valor) or valor < 0):
                    raise ValueError("Detalle inconsistente")
        except (TypeError, ValueError, OverflowError) as error:
            raise _corrupto() from error
        return {"conjunto_id": conjunto_id,
                "serie": {"id": fila["serie_id"], "store_nbr": fila["store_nbr"], "family": fila["family"]},
                "pronostico": metadatos, "detalles": [dict(d) for d in detalles]}
