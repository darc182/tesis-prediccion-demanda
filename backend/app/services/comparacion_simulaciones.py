"""Comparación pura A/B/C sobre un único pronóstico, sin nuevas fórmulas."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from math import isfinite

from backend.app.services import trayectoria_simulacion as trayectoria


_CAMPOS = {"etiqueta", "inventario_disponible", "tiempo_abastecimiento", "reserva"}
_CONTEXTO = ("pronostico_id", "experimento_id", "tecnica", "fecha_corte", "horizonte")


class ErrorComparacionSimulaciones(ValueError):
    """Estructura de escenarios inválida o salida contradictoria, sin parciales."""

    def __init__(self, codigo: str):
        self.codigo = codigo
        mensajes = {
            "escenarios_invalidos": "Indique dos escenarios A/B o tres A/B/C, completos y en ese orden, sin campos adicionales.",
            "resultados_incompatibles": "Los resultados no corresponden al mismo pronóstico, calendario y demanda compartidos.",
            "resultado_no_finito": "La comparación contiene un resultado no representable como JSON finito.",
        }
        super().__init__(mensajes[codigo])


def _seleccion(escenarios):
    # Revisar TODOS los objetos antes de ejecutar el primer escenario. Los
    # valores de parámetros se validan exclusivamente en el motor publicado.
    if type(escenarios) not in (list, tuple) or len(escenarios) not in (2, 3):
        raise ErrorComparacionSimulaciones("escenarios_invalidos")
    for etiqueta, escenario in zip(("A", "B", "C"), escenarios):
        if (not isinstance(escenario, Mapping) or set(escenario) != _CAMPOS
                or type(escenario["etiqueta"]) is not str or escenario["etiqueta"] != etiqueta):
            raise ErrorComparacionSimulaciones("escenarios_invalidos")


def _json_finito(valor):
    """Verifica representación de la salida; no valida ni recalcula cantidades."""
    if valor is None or type(valor) in (str, bool, int):
        return True
    if type(valor) is float:
        return isfinite(valor)
    if type(valor) is list:
        return all(_json_finito(v) for v in valor)
    if type(valor) is dict:
        return all(type(k) is str and _json_finito(v) for k, v in valor.items())
    return False


def _tipados(valores):
    # bool/int y float/int no son intercambiables en los metadatos públicos.
    return tuple((type(v), v) for v in valores)


def _firma(resultado):
    try:
        if type(resultado) is not dict or type(resultado["trayectoria"]) is not list or len(resultado["trayectoria"]) != 15:
            raise ValueError("Salida incompleta")
        contexto = _tipados(resultado[k] for k in _CONTEXTO)
        dias = tuple(_tipados((f["paso"], f["fecha_objetivo"], f["demanda_pronosticada"])) for f in resultado["trayectoria"])
        if type(resultado["version_regla"]) is not str or not resultado["version_regla"]:
            raise ValueError("Regla ausente")
        return contexto, dias, resultado["version_regla"], _tipados((resultado["demanda_considerada"],))
    except (KeyError, TypeError, ValueError) as error:
        raise ErrorComparacionSimulaciones("resultados_incompatibles") from error


def comparar_escenarios(pronostico, detalles, escenarios) -> dict:
    """Compara dos o tres escenarios canónicos sobre las mismas entradas.

    escenarios: list/tuple de Mapping con exactamente etiqueta,
    inventario_disponible, tiempo_abastecimiento y reserva. Etiquetas A/B o
    A/B/C en orden; se permiten parámetros iguales. La estructura completa se
    valida primero. calcular_trayectoria recibe los mismos objetos pronostico
    y detalles una vez por escenario; conserva validación y política numérica.
    Sus errores de entrada, precisión y rango se propagan sin reinterpretarlos.

    Salida: contexto con pronostico_id, experimento_id, tecnica, fecha_corte,
    horizonte, version_regla y fechas_objetivo; resultados con etiqueta y
    simulacion (salida íntegra S4-A3-01); advertencias_comparacion. El contexto
    se comprueba contra la entrada validada por trayectoria, y entre resultados
    se comparan también las quince demandas diarias y demanda_considerada.
    Las copias de metadatos dentro de cada salida íntegra deben coincidir.

    No hay fórmulas, diferencias numéricas, hashes, ranking ni selección. No
    se duplican validadores de parámetros o saldos. Se verifica finitud JSON
    como garantía de representación pública, sin redondeos ni tolerancias.
    Cada resultado se copia por separado para evitar alias mutables, incluso
    si un proveedor devolviera el mismo objeto en llamadas sucesivas.
    Si falla cualquier escenario o comprobación, no se devuelve comparación.
    No hay E/S ni comprobación de pertenencia al conjunto activo.
    """
    _seleccion(escenarios)
    resultados = []
    referencia = None
    for escenario in escenarios:
        actual = trayectoria.calcular_trayectoria(
            pronostico, detalles, escenario["inventario_disponible"],
            escenario["tiempo_abastecimiento"], escenario["reserva"])
        firma = _firma(actual)
        try:
            parametros = actual["parametros"]
            esperados = {"inventario_disponible": float(escenario["inventario_disponible"]),
                         "tiempo_abastecimiento": escenario["tiempo_abastecimiento"],
                         "reserva": float(escenario["reserva"])}
            if (type(parametros) is not dict or set(parametros) != set(esperados)
                    or _tipados(parametros[k] for k in esperados) != _tipados(esperados.values())):
                raise ValueError("Parámetros contradictorios")
        except (KeyError, TypeError, ValueError) as error:
            raise ErrorComparacionSimulaciones("resultados_incompatibles") from error
        if referencia is None:
            # Solo después de la validación del motor se normaliza la forma
            # pública de fechas y números para comprobar identidad, sin sumar.
            corte = pronostico["fecha_corte"]
            corte = corte.isoformat() if type(corte) is date else corte
            esperado = _tipados((pronostico["id"], pronostico.get("experimento_id"),
                                pronostico["tecnica"], corte, pronostico["horizonte"]))
            dias = tuple(_tipados((d["paso"], d["fecha_objetivo"].isoformat() if type(d["fecha_objetivo"]) is date
                                  else d["fecha_objetivo"], float(d["venta_estimada"]))) for d in detalles)
            if firma[0] != esperado or firma[1] != dias:
                raise ErrorComparacionSimulaciones("resultados_incompatibles")
            referencia = firma
        elif firma != referencia:
            raise ErrorComparacionSimulaciones("resultados_incompatibles")
        if not _json_finito(actual):
            raise ErrorComparacionSimulaciones("resultado_no_finito")
        resultados.append({"etiqueta": escenario["etiqueta"], "simulacion": deepcopy(actual)})
    primera = resultados[0]["simulacion"]
    return {
        "contexto": {**{k: primera[k] for k in _CONTEXTO}, "version_regla": primera["version_regla"],
                     "fechas_objetivo": [d["fecha_objetivo"] for d in primera["trayectoria"]]},
        "resultados": resultados,
        "advertencias_comparacion": [
            "Las etiquetas identifican escenarios hipotéticos, sin prioridad operativa ni ganador automático.",
            "Menor reposición, mayor inventario o menor pendiente no acreditan por sí solos un escenario mejor ni nivel de servicio.",
            "El balance negativo no es inventario físico negativo; la demanda pendiente no equivale a ventas perdidas ni se suma entre días.",
        ],
    }
