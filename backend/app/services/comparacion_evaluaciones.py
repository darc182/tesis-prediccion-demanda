"""Comparación de dos o tres validaciones actuales en una instantánea de lectura."""

from pathlib import Path

from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import consulta_pronosticos as pronosticos


_MENSAJES = {
    "entrada_invalida": "Seleccione una lista o tupla de dos o tres identificadores distintos.",
    "pronostico_invalido": "Cada identificador debe ser un entero Python positivo dentro del límite SQLite.",
    "pronostico_inexistente": "Uno de los pronósticos seleccionados no existe.",
    "pronostico_inactivo": "Uno de los pronósticos no pertenece al conjunto de datos activo.",
    "sin_conjunto_activo": "No existe un conjunto de datos activo para la comparación.",
    "tipo_no_evaluable": "Solo se comparan pronósticos de validación.",
    "corte_no_autorizado": "Uno de los cortes no pertenece a las ventanas de validación autorizadas.",
    "pronosticos_no_comparables": "Los pronósticos no comparten conjunto, serie, periodo y observaciones reales.",
    "datos_inconsistentes": "Los registros necesarios para la comparación son inconsistentes.",
    "base_no_disponible": "La base de datos no está disponible o no tiene el esquema necesario.",
}


class ErrorComparacionEvaluaciones(ValueError):
    """Código estable y mensaje fijo; nunca incorpora mensajes internos de A2."""

    def __init__(self, codigo: str):
        self.codigo = codigo if type(codigo) is str and codigo in _MENSAJES else "datos_inconsistentes"
        super().__init__(_MENSAJES[self.codigo])


def _condiciones(resultado):
    """Identifica el contexto común y las ventas por fecha, sin comparar estimaciones.

    A2/A1 ya validan el calendario consecutivo y el horizonte de quince días.
    Cero es una observación: únicamente None representa una fecha ausente.
    Los valores reales se comparan exactamente, sin tolerancias ni redondeo.
    """
    detalles = resultado["evaluacion"]["detalles"]
    fechas = tuple(d["fecha_objetivo"] for d in detalles)
    reales = {d["fecha_objetivo"]: d["venta_real"] for d in detalles if d["venta_real"] is not None}
    return (resultado["conjunto_id"], resultado["serie"], resultado["fecha_corte"],
            resultado["horizonte"], fechas, reales)


def comparar_evaluaciones(ruta_base: str | Path, pronostico_ids: list[int] | tuple[int, ...]) -> dict:
    """Compara cálculos actuales sin persistir, clasificar ni seleccionar ganadores.

    Contrato D1 para D2: devuelve conjunto_id, serie (id, store_nbr, family),
    fecha_corte, horizonte, estado_cobertura, alcance_metricas,
    observaciones_evaluadas, cobertura, fechas_ausentes y resultados. Cada
    resultado contiene pronostico_id, experimento_id, tecnica y evaluacion
    (resultado íntegro de A1), en el orden solicitado. No lee la tabla evaluacion.

    Valida todos los IDs antes de abrir SQLite. _conexion_lectura proporciona
    una sola conexión mode=ro con BEGIN; _evaluar_en_conexion de A2 se ejecuta
    una vez por ID en esa misma transacción. No se usa la consulta pública A2.

    Exige el mismo conjunto activo, serie, corte autorizado, horizonte y quince
    fechas objetivo, además de las mismas fechas y valores reales observados.
    Técnicas o configuraciones distintas se admiten sin leer JSON privado.
    Una incompatibilidad o error de cualquier selección rechaza la operación
    completa. Ningún resultado parcial se devuelve.

    Con cobertura parcial, las métricas solo describen las fechas disponibles
    compartidas; fechas_ausentes las identifica sin imputar ceros. Sin reales,
    estado_cobertura es sin_observaciones y las métricas son None: no se puede
    evaluar el desempeño predictivo. La suma real cero deja solo WAPE en None.

    Los códigos de rechazo son los de _MENSAJES. La excepción pública recibe
    únicamente un código; el mensaje siempre es fijo y no revela rutas ni SQL.
    """
    if not isinstance(pronostico_ids, (list, tuple)) or len(pronostico_ids) not in (2, 3):
        raise ErrorComparacionEvaluaciones("entrada_invalida")
    seleccion = tuple(pronostico_ids)
    if any(type(i) is not int or not 0 < i <= 2**63 - 1 for i in seleccion):
        raise ErrorComparacionEvaluaciones("pronostico_invalido")
    if len(set(seleccion)) != len(seleccion):
        raise ErrorComparacionEvaluaciones("entrada_invalida")
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base) or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        ruta = Path(ruta_base).resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ErrorComparacionEvaluaciones("base_no_disponible") from error

    try:
        with pronosticos._conexion_lectura(ruta) as conexion:
            resultados = []
            referencia = None
            for identificador in seleccion:
                actual = consulta._evaluar_en_conexion(conexion, identificador)
                condiciones = _condiciones(actual)
                if referencia is None:
                    referencia, condiciones_comunes = actual, condiciones
                elif condiciones != condiciones_comunes:
                    raise ErrorComparacionEvaluaciones("pronosticos_no_comparables")
                resultados.append({k: actual[k] for k in ("pronostico_id", "experimento_id", "tecnica", "evaluacion")})
            evaluacion = referencia["evaluacion"]
            return {
                "conjunto_id": referencia["conjunto_id"], "serie": dict(referencia["serie"]),
                "fecha_corte": referencia["fecha_corte"], "horizonte": referencia["horizonte"],
                "estado_cobertura": referencia["estado_cobertura"], "alcance_metricas": referencia["alcance_metricas"],
                "observaciones_evaluadas": evaluacion["observaciones_evaluadas"], "cobertura": evaluacion["cobertura"],
                "fechas_ausentes": list(evaluacion["fechas_ausentes"]), "resultados": resultados,
            }
    except (consulta.ErrorConsultaEvaluacion, pronosticos.ErrorConsultaPronostico) as error:
        raise ErrorComparacionEvaluaciones(error.codigo) from error
