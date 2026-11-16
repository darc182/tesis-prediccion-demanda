"""S4-A3-03: comparación A/B/C desde una única instantánea SQLite de lectura."""

from pathlib import Path

from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import simulacion_desde_pronostico as lectura
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import trayectoria_simulacion as trayectoria


_MENSAJES = {
    **lectura._MENSAJES,
    "escenarios_invalidos": "Indique dos escenarios A/B o tres A/B/C, completos y en ese orden, sin campos adicionales.",
    "resultados_incompatibles": "Los resultados no corresponden al mismo pronóstico, calendario y demanda compartidos.",
}


class ErrorComparacionDesdePronostico(ValueError):
    """Código público estable y mensaje fijo; nunca publica texto de la causa."""

    def __init__(self, codigo: str):
        self.codigo = codigo if type(codigo) is str and codigo in _MENSAJES else "pronostico_corrupto"
        super().__init__(_MENSAJES[self.codigo])


def comparar_escenarios_pronostico(ruta_base: str | Path, pronostico_id: int, escenarios) -> dict:
    """Recupera una sola cabecera y quince detalles para el comparador puro.

    Dependencias privadas S4-A2: _validar_identificador, _normalizar_ruta,
    _leer_pronostico_simulable_en_conexion y catálogo _MENSAJES. Revisar si
    cambia aquel módulo. Se conserva su validación y orden de lectura sin
    llamar a simular_pronostico ni abrir conexiones por escenario.

    _conexion_lectura proporciona una conexión mode=ro, BEGIN y cierre. La
    lectura y la única invocación a comparar_escenarios ocurren dentro de esa
    transacción. El comparador valida escenarios, parámetros y coherencia;
    aquí no se duplican reglas ni cálculos. Se devuelve íntegra su respuesta
    (contexto/resultados/advertencias_comparacion) más conjunto_id y serie.
    No hay ventas observadas, evaluaciones, JSON privado ni rutas de modelos.

    Errores de lectura conservan códigos S4-A2; escenarios/resultados inválidos
    conservan códigos S4-A3-02; valor/plazo inválido pasa a parametros_invalidos
    como en S4-A2. Precisión y rango de ambos motores conservan sus códigos.
    Todos los mensajes son constantes locales. Errores inesperados de
    programación se propagan; nunca se publica una comparación parcial.
    La procedencia solo queda acreditada durante la instantánea de esta llamada.
    """
    try:
        lectura._validar_identificador(pronostico_id)
        ruta = lectura._normalizar_ruta(ruta_base)
        with consulta._conexion_lectura(ruta) as conexion:
            conjunto, serie, pronostico, detalles = lectura._leer_pronostico_simulable_en_conexion(
                conexion, pronostico_id)
            resultado = comparacion.comparar_escenarios(pronostico, detalles, escenarios)
            return {**resultado, "conjunto_id": conjunto, "serie": serie}
    except lectura.ErrorSimulacionDesdePronostico as error:
        raise ErrorComparacionDesdePronostico(error.codigo) from error
    except consulta.ErrorConsultaPronostico as error:
        codigo = error.codigo if error.codigo in ("sin_conjunto_activo", "base_no_disponible") else "pronostico_corrupto"
        raise ErrorComparacionDesdePronostico(codigo) from error
    except comparacion.ErrorComparacionSimulaciones as error:
        raise ErrorComparacionDesdePronostico(error.codigo) from error
    except (motor.ErrorSimulacionReposicion, trayectoria.ErrorTrayectoriaSimulacion) as error:
        codigo = error.codigo
        if codigo in ("valor_invalido", "plazo_invalido"):
            codigo = "parametros_invalidos"
        elif codigo not in ("precision_no_representable", "resultado_no_finito"):
            codigo = "pronostico_corrupto"
        raise ErrorComparacionDesdePronostico(codigo) from error
