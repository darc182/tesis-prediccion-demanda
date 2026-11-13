"""Registro explícito, atómico e idempotente de evaluaciones de validación.

La tabla no almacena cobertura, huella ni fecha de evaluación. La transacción
garantiza consistencia durante el cálculo y la inserción, no inmutabilidad
posterior. Métricas iguales no prueban identidad de los datos originales.
Toda consulta posterior de vigencia debe contrastar los datos actuales.
"""

from contextlib import closing
from math import isfinite
from pathlib import Path
import sqlite3

from backend.app.services import consulta_evaluaciones as consulta


class ErrorRegistroEvaluacion(ValueError):
    """Error público de validación, conflicto o persistencia, sin detalles privados."""

    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


def _coincide(fila, valores) -> bool:
    """Compara números finitos no negativos por igualdad exacta, sin tolerancias.

    SQLite puede representar un mismo número como int o float. Solo WAPE
    admite NULL; un RMSE antiguo NULL no constituye una evaluación reutilizable.
    """
    for campo, actual in zip(("mae", "wape", "rmse"), valores):
        guardado = fila[campo]
        if guardado is None:
            if campo != "wape" or actual is not None:
                return False
        elif (type(guardado) not in (int, float) or not isfinite(guardado)
              or guardado < 0 or actual is None or guardado != actual):
            return False
    return True


def registrar_evaluacion(ruta_base: str | Path, pronostico_id: int) -> dict:
    """Calcula desde SQLite vigente y guarda solo una evaluación autorizada.

    Abre exclusivamente mode=rw, activa FK y ejecuta BEGIN IMMEDIATE antes de
    cualquier SELECT. El helper de A2 valida y calcula en esa misma conexión;
    no se llama a la consulta pública ni se duplican las fórmulas de A1.

    Devuelve el resultado público completo de A2 más estado_registro (creado,
    reutilizado o no_guardado), hubo_escritura y motivo. Cobertura parcial se
    guarda como métricas sobre observaciones disponibles. Sin observaciones
    devuelve no_guardado/sin_observaciones, o conflicto si ya había una fila.
    Una fila con métricas diferentes o inválidas nunca se sobrescribe.

    Reutilizar significa coincidencia de las tres métricas con el cálculo
    actual; no acredita cobertura histórica ni identidad del histórico. La
    cobertura de la respuesta se obtiene siempre de los datos actuales.

    Conserva los códigos públicos de A2 y añade evaluacion_desactualizada
    y error_persistencia. No realiza operaciones por importar el módulo.
    """
    if type(pronostico_id) is not int or not 0 < pronostico_id <= 2**63 - 1:
        raise ErrorRegistroEvaluacion("pronostico_invalido", "El identificador debe ser un entero positivo válido.")
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base) or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        uri = Path(ruta_base).resolve().as_uri() + "?mode=rw"
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ErrorRegistroEvaluacion("base_no_disponible", "La base de datos no está disponible.") from error

    try:
        with closing(sqlite3.connect(uri, uri=True)) as conexion:
            conexion.row_factory = sqlite3.Row
            conexion.execute("PRAGMA foreign_keys = ON")
            try:
                conexion.execute("BEGIN IMMEDIATE")
                actual = consulta._evaluar_en_conexion(conexion, pronostico_id)
                previa = conexion.execute(
                    "SELECT mae, wape, rmse FROM evaluacion WHERE pronostico_id = ?",
                    (pronostico_id,),
                ).fetchone()
                evaluacion = actual["evaluacion"]
                valores = tuple(evaluacion[campo] for campo in ("mae", "wape", "rmse"))
                motivo = None
                if not evaluacion["observaciones_evaluadas"]:
                    if previa is not None:
                        raise ErrorRegistroEvaluacion(
                            "evaluacion_desactualizada", "La evaluación guardada no coincide con los datos actuales.")
                    estado, motivo = "no_guardado", "sin_observaciones"
                    conexion.rollback()
                elif previa is not None:
                    if not _coincide(previa, valores):
                        raise ErrorRegistroEvaluacion(
                            "evaluacion_desactualizada", "La evaluación guardada no coincide con los datos actuales.")
                    estado = "reutilizado"
                    conexion.rollback()
                else:
                    try:
                        cambios = conexion.total_changes
                        cursor = conexion.execute(
                            "INSERT INTO evaluacion (pronostico_id, mae, wape, rmse) VALUES (?, ?, ?, ?)",
                            (pronostico_id, *valores),
                        )
                        guardada = conexion.execute(
                            "SELECT mae, wape, rmse FROM evaluacion WHERE pronostico_id = ?",
                            (pronostico_id,),
                        ).fetchone()
                        # Detecta también INSERT ignorado o alterado por triggers.
                        if (cursor.rowcount != 1 or conexion.total_changes - cambios != 1
                                or guardada is None or not _coincide(guardada, valores)):
                            raise ErrorRegistroEvaluacion(
                                "error_persistencia", "No se pudo guardar una única evaluación consistente.")
                        conexion.commit()
                    except sqlite3.Error as error:
                        raise ErrorRegistroEvaluacion(
                            "error_persistencia", "No se pudo guardar la evaluación; la operación se revirtió.") from error
                    estado = "creado"
                return {**actual, "estado_registro": estado, "hubo_escritura": estado == "creado", "motivo": motivo}
            except Exception:
                conexion.rollback()
                raise
    except consulta.ErrorConsultaEvaluacion as error:
        raise ErrorRegistroEvaluacion(error.codigo, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise ErrorRegistroEvaluacion("base_no_disponible", "La base de datos no está disponible.") from error
