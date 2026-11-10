"""Disponibilidad de ejemplos de entrenamiento; solo lectura, sin estimadores."""

from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import sqlite3

import pandas as pd

from backend.app.services import configuracion_experimentos as contratos
from backend.app.services.registro_pronosticos import CORTES_VALIDACION, PRUEBA_FINAL
from scripts.analizar_periodo_seleccion import HORIZONTE_DIAS
from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS, construir_tabla_serie,
)


class ErrorConsultaFilas(ValueError):
    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


def _validar(serie_id, fecha_corte, grupo):
    # Validar todo antes de abrir la base; no inferir grupos ni técnicas.
    if type(serie_id) is not int or not 0 < serie_id <= 2**63 - 1:
        raise ErrorConsultaFilas("serie_invalida", "Identificador de serie inválido.")
    if type(grupo) is not str or grupo not in ("clasico", "ampliado"):
        raise ErrorConsultaFilas("grupo_invalido", "Grupo de variables inválido.")
    try:
        if type(fecha_corte) is str:
            corte = date.fromisoformat(fecha_corte)
            if corte.isoformat() != fecha_corte:
                raise ValueError("Fecha no canónica.")
        elif type(fecha_corte) is date:
            corte = fecha_corte
        else:
            raise ValueError("Fecha inválida.")
    except ValueError as error:
        raise ErrorConsultaFilas("corte_invalido", "Fecha de corte inválida.") from error
    # Incluye tanto cortes cuyo horizonte alcanza la reserva como fechas de
    # la propia prueba final (su límite superior es inclusivo).
    if PRUEBA_FINAL.inicio - timedelta(days=HORIZONTE_DIAS) <= corte <= PRUEBA_FINAL.fin:
        raise ErrorConsultaFilas("prueba_reservada", "La prueba final permanece reservada.")
    if corte not in CORTES_VALIDACION:
        raise ErrorConsultaFilas("corte_no_autorizado", "Corte de validación no autorizado.")
    return corte


def consultar_filas_entrenamiento(
    ruta_base: str | Path, serie_id: int, fecha_corte: date | str, grupo: str,
) -> dict:
    """Cuenta solo objetivos observados de la serie activa hasta el corte.

    Una observación puede tener varias variables ausentes. Los motivos no se
    suman para obtener los descartes. Cero filas aptas es un resultado válido
    si existe historial; no supone que un modelo pueda entrenarse.
    """
    corte = _validar(serie_id, fecha_corte, grupo)
    ampliado = grupo == "ampliado"
    columnas = COLUMNAS_PREDICTORAS_ALTERNATIVAS if ampliado else COLUMNAS_PREDICTORAS
    try:
        uri = f"{Path(ruta_base).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conexion:
            conexion.row_factory = sqlite3.Row
            conexion.execute("BEGIN")
            activo = conexion.execute(
                "SELECT id FROM conjunto_datos WHERE activo = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if activo is None:
                raise ErrorConsultaFilas("sin_conjunto_activo", "No existe un conjunto activo.")
            serie = conexion.execute(
                "SELECT id, store_nbr, family FROM serie_temporal WHERE id = ? AND conjunto_id = ?",
                (serie_id, activo["id"]),
            ).fetchone()
            if serie is None:
                raise ErrorConsultaFilas("serie_inexistente", "La serie no existe en el conjunto activo.")
            filas = conexion.execute(
                "SELECT fecha, sales, onpromotion FROM venta_diaria "
                "WHERE serie_id = ? AND fecha <= ? ORDER BY fecha",
                (serie_id, corte.isoformat()),
            ).fetchall()
            if not filas:
                raise ErrorConsultaFilas("sin_historial", "No hay observaciones hasta el corte.")
            datos = pd.DataFrame.from_records(
                [tuple(fila) for fila in filas], columns=["date", "sales", "onpromotion"])
            datos["store_nbr"] = serie["store_nbr"]
            datos["family"] = serie["family"]
            try:
                argumentos = {"columnas_predictoras": columnas} if ampliado else {}
                resultado = construir_tabla_serie(
                    datos, serie["store_nbr"], serie["family"], corte, **argumentos)
            except (ValueError, TypeError, OverflowError) as error:
                raise ErrorConsultaFilas(
                    "historial_invalido", "Las observaciones históricas presentan datos inválidos.") from error
            aptas, descartadas = len(resultado.tabla), len(resultado.descartes)
            if aptas + descartadas != resultado.observaciones_historicas:
                raise ErrorConsultaFilas("error_interno", "Conteos de preparación inconsistentes.")
            return {
                "conjunto_id": activo["id"], "serie": dict(serie),
                "fecha_corte": corte.isoformat(), "grupo": grupo,
                "version_contrato": (contratos.VERSION_CONTRATO_ALTERNATIVO if ampliado
                                     else contratos.VERSION_CONTRATO),
                "version_reglas": (contratos.VERSION_REGLAS_ALTERNATIVO if ampliado
                                   else contratos.VERSION_REGLAS),
                "variables": list(columnas),
                "observaciones_historicas": resultado.observaciones_historicas,
                "filas_aptas": aptas, "filas_descartadas": descartadas,
                "descartes_por_variable": resultado.descartes_por_variable,
            }
    except (sqlite3.Error, OSError) as error:
        raise ErrorConsultaFilas(
            "base_no_disponible", "La base de datos no está disponible.") from error
