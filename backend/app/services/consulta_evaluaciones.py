"""Evaluación de pronósticos guardados en una única instantánea de lectura."""

from datetime import timedelta
from pathlib import Path

from backend.app.services import consulta_pronosticos as pronosticos
from backend.app.services import metricas_evaluacion as metricas


class ErrorConsultaEvaluacion(ValueError):
    """Rechazo controlado; código estable y mensaje sin SQL ni rutas privadas."""

    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


def _inconsistente():
    return ErrorConsultaEvaluacion(
        "datos_inconsistentes", "Los registros de la evaluación están incompletos o son inconsistentes.")


def _cabecera(conexion, pronostico_id, conjunto_id):
    # LEFT JOIN permite detectar referencias rotas, en vez de confundirlas con
    # un ID inexistente. No se necesita el JSON interno ni la ruta del modelo.
    fila = conexion.execute(
        "SELECT p.id, p.experimento_id, p.fecha_corte, p.tipo, p.horizonte, "
        "p.fecha_generacion, p.duracion_segundos, e.tecnica, e.inicio_entrenamiento, "
        "e.fin_entrenamiento, s.id AS serie_id, s.conjunto_id, s.store_nbr, s.family "
        "FROM pronostico p LEFT JOIN experimento e ON e.id = p.experimento_id "
        "LEFT JOIN serie_temporal s ON s.id = e.serie_id WHERE p.id = ?",
        (pronostico_id,),
    ).fetchone()
    if fila is None:
        raise ErrorConsultaEvaluacion("pronostico_inexistente", "El pronóstico no existe.")
    for campo in ("id", "experimento_id", "serie_id", "conjunto_id", "store_nbr"):
        if type(fila[campo]) is not int or fila[campo] <= 0:
            raise _inconsistente()
    for campo in ("family", "tecnica"):
        if not isinstance(fila[campo], str) or not fila[campo].strip():
            raise _inconsistente()
    if fila["conjunto_id"] != conjunto_id:
        raise ErrorConsultaEvaluacion(
            "pronostico_inactivo", "La serie del pronóstico no pertenece al conjunto activo.")
    return fila


def consultar_evaluacion(ruta_base: str | Path, pronostico_id: int) -> dict:
    """Evalúa una validación almacenada sin escribir ni reconstruir modelos.

    Ambos parámetros son obligatorios. Reutiliza _conexion_lectura (URI
    mode=ro, BEGIN explícito y cierre garantizado), _activo, _metadatos y
    _detalles_validos de consulta_pronosticos. No llama a sus servicios públicos
    ni abre otra conexión. La primera lectura del activo fija la instantánea;
    cabecera, detalles y ventas pertenecen a esa misma transacción.

    Solo tras validar tipo, corte y detalles se leen fecha y sales de la serie
    para corte < fecha <= corte+15. A1 realiza todo el cálculo y la alineación
    por fecha. evaluacion conserva íntegro su resultado, sin imputar ni redondear.
    estado_cobertura describe disponibilidad, nunca calidad predictiva. En
    parcial, alcance_metricas indica métricas sobre observaciones disponibles.

    Códigos: pronostico_invalido, base_no_disponible, sin_conjunto_activo,
    pronostico_inexistente, pronostico_inactivo, tipo_no_evaluable,
    corte_no_autorizado y datos_inconsistentes. No persiste evaluaciones.
    """
    if type(pronostico_id) is not int or not 0 < pronostico_id <= 2**63 - 1:
        raise ErrorConsultaEvaluacion("pronostico_invalido", "El identificador debe ser un entero positivo válido.")
    # Normalizar solo la ruta explícita, nunca recurrir a una base por defecto.
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base) or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        ruta = Path(ruta_base).resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ErrorConsultaEvaluacion("base_no_disponible", "La base de datos no está disponible.") from error

    try:
        with pronosticos._conexion_lectura(ruta) as conexion:
            return _evaluar_en_conexion(conexion, pronostico_id)
    except pronosticos.ErrorConsultaPronostico as error:
        if error.codigo in ("base_no_disponible", "sin_conjunto_activo"):
            raise ErrorConsultaEvaluacion(error.codigo, str(error)) from error
        raise _inconsistente() from error
    except metricas.ErrorMetricasEvaluacion as error:
        raise _inconsistente() from error


def _evaluar_en_conexion(conexion, pronostico_id: int) -> dict:
    """Calcula A2 en la conexión y transacción del llamador, con filas sqlite3.Row.

    El llamador valida el ID y administra apertura, BEGIN, commit/rollback y
    cierre. Este helper solo consulta y calcula; no cambia la transacción.
    """
    try:
        conjunto_id = pronosticos._activo(conexion)
        fila = _cabecera(conexion, pronostico_id, conjunto_id)
        metadatos = pronosticos._metadatos(fila)
        if metadatos["tipo"] != "validacion":
            raise ErrorConsultaEvaluacion("tipo_no_evaluable", "Solo se evalúan pronósticos de validación.")
        corte = pronosticos._fecha(metadatos["fecha_corte"])
        if corte not in metricas.CORTES_AUTORIZADOS:
            raise ErrorConsultaEvaluacion("corte_no_autorizado", "El corte no pertenece a las ventanas autorizadas.")
        detalles = pronosticos._detalles_validos(conexion, pronostico_id, metadatos)
        observaciones = conexion.execute(
            "SELECT fecha, sales FROM venta_diaria "
            "WHERE serie_id = ? AND fecha > ? AND fecha <= ? ORDER BY fecha",
            (fila["serie_id"], corte.isoformat(),
             (corte + timedelta(days=metricas.HORIZONTE_DIAS)).isoformat()),
        ).fetchall()
        evaluacion = metricas.calcular_metricas(
            metadatos["fecha_corte"], [dict(d) for d in detalles], [dict(o) for o in observaciones])
        cantidad = evaluacion["observaciones_evaluadas"]
        if cantidad == metricas.HORIZONTE_DIAS:
            estado, alcance = "completa", "Métricas sobre las quince observaciones del horizonte."
        elif cantidad:
            estado, alcance = "parcial", "Métricas sobre observaciones disponibles."
        else:
            estado, alcance = "sin_observaciones", "Sin observaciones reales; métricas no definidas."
        return {
            "conjunto_id": conjunto_id,
            "serie": {"id": fila["serie_id"], "store_nbr": fila["store_nbr"], "family": fila["family"]},
            "pronostico_id": metadatos["id"], "experimento_id": metadatos["experimento_id"],
            "tecnica": metadatos["tecnica"], "fecha_corte": metadatos["fecha_corte"],
            "horizonte": metadatos["horizonte"], "estado_cobertura": estado,
            "alcance_metricas": alcance, "evaluacion": evaluacion,
        }
    except pronosticos.ErrorConsultaPronostico as error:
        if error.codigo in ("base_no_disponible", "sin_conjunto_activo"):
            raise ErrorConsultaEvaluacion(error.codigo, str(error)) from error
        raise _inconsistente() from error
    except metricas.ErrorMetricasEvaluacion as error:
        raise _inconsistente() from error
