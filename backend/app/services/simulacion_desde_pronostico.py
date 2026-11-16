"""S4-A2: simulación desde un pronóstico en una instantánea SQLite de lectura."""

from pathlib import Path

from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import simulacion_reposicion as motor


CORTES_AUTORIZADOS = ("2017-06-16", "2017-07-01", "2017-07-16")
_MENSAJES = {
    "pronostico_invalido": "El identificador debe ser un entero positivo dentro del límite SQLite.",
    "pronostico_inexistente": "El pronóstico solicitado no existe.",
    "pronostico_inactivo": "El pronóstico no pertenece al conjunto activo.",
    "sin_conjunto_activo": "No existe un conjunto de datos activo.",
    "tipo_no_evaluable": "Solo se simulan pronósticos de validación.",
    "corte_no_autorizado": "El corte no pertenece a las tres ventanas de validación autorizadas.",
    "pronostico_corrupto": "El pronóstico o su procedencia presentan registros incompletos o inconsistentes.",
    "parametros_invalidos": "El inventario, la reserva o el plazo hipotéticos son inválidos.",
    "precision_no_representable": "Las cantidades no pueden representarse con la precisión admitida por el motor.",
    "resultado_no_finito": "Las cantidades resultantes exceden el rango numérico admitido.",
    "base_no_disponible": "La base de datos no está disponible o no tiene el esquema necesario.",
}


class ErrorSimulacionDesdePronostico(ValueError):
    """Código estable y mensaje constante, sin incorporar texto de excepciones internas."""

    def __init__(self, codigo: str):
        self.codigo = codigo if type(codigo) is str and codigo in _MENSAJES else "pronostico_corrupto"
        super().__init__(_MENSAJES[self.codigo])


def _cabecera(conexion, identificador, conjunto):
    # No se usa _cabecera_activa: consulta JSON privado y confunde inactivo con
    # inexistente. LEFT JOIN distingue referencias rotas sin consultar modelos.
    fila = conexion.execute(
        "SELECT p.id, p.experimento_id, p.fecha_corte, p.tipo, p.horizonte, "
        "p.fecha_generacion, p.duracion_segundos, e.tecnica, e.inicio_entrenamiento, "
        "e.fin_entrenamiento, e.id AS referencia_experimento, s.id AS serie_id, "
        "s.conjunto_id, s.store_nbr, s.family FROM pronostico p "
        "LEFT JOIN experimento e ON e.id=p.experimento_id "
        "LEFT JOIN serie_temporal s ON s.id=e.serie_id WHERE p.id=?",
        (identificador,),
    ).fetchone()
    if fila is None:
        raise ErrorSimulacionDesdePronostico("pronostico_inexistente")
    if any(type(fila[k]) is not int or not 0 < fila[k] <= 2**63 - 1 for k in
           ("id", "experimento_id", "referencia_experimento", "serie_id", "conjunto_id", "store_nbr")):
        raise ErrorSimulacionDesdePronostico("pronostico_corrupto")
    if (fila["id"] != identificador or fila["experimento_id"] != fila["referencia_experimento"]
            or type(fila["family"]) is not str or not fila["family"].strip()
            or fila["tecnica"] not in motor.TECNICAS):
        raise ErrorSimulacionDesdePronostico("pronostico_corrupto")
    if fila["conjunto_id"] != conjunto:
        raise ErrorSimulacionDesdePronostico("pronostico_inactivo")
    return fila


def simular_pronostico(ruta_base: str | Path, pronostico_id: int,
                      inventario_disponible, tiempo_abastecimiento, reserva) -> dict:
    """Lee y simula sin guardar, entrenar ni consultar observaciones o métricas.

    Reutiliza los helpers privados _conexion_lectura, _activo, _metadatos y
    _detalles_validos de consulta_pronosticos: dependencia deliberada que debe
    revisarse si cambia ese módulo. Una conexión mode=ro con BEGIN contiene
    conjunto, cabecera, detalles y la única invocación del motor publicado.
    No se llaman servicios públicos que abran conexiones adicionales.

    Solo acepta tipo validacion y los tres cortes autorizados, comprobados
    antes de leer detalles. No se consulta venta_diaria, evaluacion, simulacion,
    configuracion_json ni ruta_modelo. La salida es el resultado íntegro del
    motor más conjunto_id y serie (id, store_nbr, family), sin duplicaciones.
    Los supuestos del núcleo describen su alcance; este servicio sí acredita
    existencia, procedencia activa y elegibilidad durante la instantánea.

    Los parámetros numéricos se validan exclusivamente en S4-A1. Sus errores
    de precisión y rango se conservan; valor/plazo inválidos se traducen a
    parametros_invalidos. No se publican mensajes internos. Los errores de
    programación inesperados no se suprimen ni producen resultados parciales.
    """
    if type(pronostico_id) is not int or not 0 < pronostico_id <= 2**63 - 1:
        raise ErrorSimulacionDesdePronostico("pronostico_invalido")
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base) or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        ruta = Path(ruta_base).resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ErrorSimulacionDesdePronostico("base_no_disponible") from error
    try:
        with consulta._conexion_lectura(ruta) as conexion:
            conjunto = consulta._activo(conexion)
            if type(conjunto) is not int or not 0 < conjunto <= 2**63 - 1:
                raise ErrorSimulacionDesdePronostico("pronostico_corrupto")
            fila = _cabecera(conexion, pronostico_id, conjunto)
            cabecera = consulta._metadatos(fila)
            if cabecera["tipo"] != "validacion":
                raise ErrorSimulacionDesdePronostico("tipo_no_evaluable")
            if cabecera["fecha_corte"] not in CORTES_AUTORIZADOS:
                raise ErrorSimulacionDesdePronostico("corte_no_autorizado")
            detalles = consulta._detalles_validos(conexion, pronostico_id, cabecera)
            resultado = motor.calcular_simulacion(
                cabecera, [dict(d) for d in detalles], inventario_disponible,
                tiempo_abastecimiento, reserva)
            return {**resultado, "conjunto_id": conjunto,
                    "serie": {"id": fila["serie_id"], "store_nbr": fila["store_nbr"], "family": fila["family"]}}
    except consulta.ErrorConsultaPronostico as error:
        codigo = error.codigo if error.codigo in ("sin_conjunto_activo", "base_no_disponible") else "pronostico_corrupto"
        raise ErrorSimulacionDesdePronostico(codigo) from error
    except motor.ErrorSimulacionReposicion as error:
        codigo = error.codigo
        if codigo in ("valor_invalido", "plazo_invalido"):
            codigo = "parametros_invalidos"
        elif codigo not in ("precision_no_representable", "resultado_no_finito"):
            codigo = "pronostico_corrupto"
        raise ErrorSimulacionDesdePronostico(codigo) from error
