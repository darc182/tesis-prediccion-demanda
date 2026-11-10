"""Plan puro de cinco configuraciones; sin comprobar datos ni ejecutar modelos.

La identidad predictiva pertenece a la configuración F2. El contexto de
conjunto/serie/corte se conserva aparte y deberá verificarse contra el conjunto
activo al ejecutar. Construir un plan no acredita pertenencia ni historial.
"""

from datetime import date, timedelta

from backend.app.services import configuracion_experimentos as contrato
from backend.app.services.validacion_csv import MAX_ENTERO_SQLITE
from scripts.analizar_periodo_seleccion import HORIZONTE_DIAS, VENTANAS_POSTERIORES


_CORTES = tuple(v.inicio - timedelta(days=1) for v in VENTANAS_POSTERIORES[:3])
_CASOS = ((contrato.BASE, False),
          ("random_forest", False), ("random_forest", True),
          ("hist_gradient_boosting", False), ("hist_gradient_boosting", True))


class ErrorPlanBateria(ValueError):
    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


def _validar_contexto(conjunto_id, serie_id, fecha_corte):
    for nombre, valor, codigo in (("conjunto_id", conjunto_id, "conjunto_invalido"),
                                  ("serie_id", serie_id, "serie_invalida")):
        if type(valor) is not int or not 1 <= valor <= MAX_ENTERO_SQLITE:
            raise ErrorPlanBateria(codigo, f"{nombre} debe ser un entero positivo válido.")
    try:
        if type(fecha_corte) is str:
            corte = date.fromisoformat(fecha_corte)
            if corte.isoformat() != fecha_corte:
                raise ValueError("Fecha no canónica.")
        elif type(fecha_corte) is date:
            corte = fecha_corte
        else:
            raise ValueError("Tipo de fecha inválido.")
    except ValueError as error:
        raise ErrorPlanBateria("corte_invalido", "Use una fecha válida YYYY-MM-DD.") from error
    reservada = VENTANAS_POSTERIORES[-1]
    if reservada.inicio - timedelta(days=HORIZONTE_DIAS) <= corte <= reservada.fin:
        raise ErrorPlanBateria("prueba_reservada", "La prueba final permanece reservada.")
    if corte not in _CORTES:
        raise ErrorPlanBateria("corte_no_autorizado", "El corte de validación no está autorizado.")
    return corte


def construir_plan_bateria(conjunto_id: int, serie_id: int, fecha_corte: date | str) -> dict:
    """Devuelve un plan nuevo, serializable y con contexto aún no verificado.

    Orden: base semanal, RF clásico/ampliado, HGB clásico/ampliado. Los parámetros
    se completan exclusivamente mediante normalizar_configuracion() de F2.
    No admite personalizar técnicas, grupos, versiones, horizonte ni parámetros.
    Los dict/list devueltos son independientes entre elementos y llamadas.
    """
    corte = _validar_contexto(conjunto_id, serie_id, fecha_corte)
    elementos = []
    for tecnica, ampliado in _CASOS:
        solicitud = {"tecnica": tecnica}
        if ampliado:
            solicitud.update({
                "version_contrato": contrato.VERSION_CONTRATO_ALTERNATIVO,
                "version_reglas": contrato.VERSION_REGLAS_ALTERNATIVO,
                "variables": list(contrato.VARIABLES_ALTERNATIVAS_V2),
            })
        efectiva = contrato.normalizar_configuracion(solicitud)
        elementos.append({
            "tecnica": efectiva["tecnica"], "configuracion": efectiva,
            "variables": list(efectiva["variables"]),
            "version_contrato": efectiva["version_contrato"],
            "version_reglas": efectiva["version_reglas"],
            "horizonte": efectiva["horizonte"],
            "identidad_predictiva": contrato.identidad_configuracion(efectiva),
        })
    if len(elementos) != 5 or len({e["identidad_predictiva"] for e in elementos}) != 5:
        raise ErrorPlanBateria("plan_invalido", "Se requieren cinco configuraciones efectivas distintas.")
    return {"conjunto_id": conjunto_id, "serie_id": serie_id,
            "fecha_corte": corte.isoformat(), "elementos": elementos}
