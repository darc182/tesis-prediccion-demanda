"""Núcleo puro S4-A1: cobertura de quince días, sin trayectoria ni E/S."""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from fractions import Fraction
from math import isfinite
from numbers import Real
from sys import float_info


VERSION_REGLA = "cobertura_horizonte_15_pendientes_v1"
HORIZONTE = 15
TECNICAS = ("estacional_semanal_recursiva", "random_forest", "hist_gradient_boosting")
_MAXIMO = Fraction.from_float(float_info.max)
_MENSAJES = {
    "estructura_invalida": "La cabecera y los detalles del pronóstico tienen una estructura inválida.",
    "identificador_invalido": "Los identificadores deben ser enteros positivos SQLite y coherentes entre sí.",
    "metadatos_invalidos": "La técnica, el horizonte o los metadatos temporales son inválidos.",
    "fecha_invalida": "Las fechas deben ser fechas naturales válidas y canónicas YYYY-MM-DD.",
    "alineacion_invalida": "Se requieren quince pasos y fechas consecutivos posteriores al corte, en orden.",
    "valor_invalido": "Las cantidades deben ser reales finitas, no negativas y representables como float.",
    "precision_no_representable": "Una cantidad entera o de entrada perdería precisión al representarse como float.",
    "plazo_invalido": "El tiempo de abastecimiento debe ser un entero entre cero y quince, no booleano.",
    "resultado_no_finito": "El calendario o las cantidades resultantes exceden el rango representable.",
}


class ErrorSimulacionReposicion(ValueError):
    """Error controlado con código estable y mensaje público constante."""

    def __init__(self, codigo: str):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _fecha(valor):
    if type(valor) is date:
        return valor
    if type(valor) is str:
        try:
            resultado = date.fromisoformat(valor)
            if resultado.isoformat() == valor:
                return resultado
        except ValueError:
            pass
    raise ErrorSimulacionReposicion("fecha_invalida")


def _identificador(valor):
    if type(valor) is not int or not 0 < valor <= 2**63 - 1:
        raise ErrorSimulacionReposicion("identificador_invalido")
    return valor


def _numero(valor):
    if isinstance(valor, bool) or not isinstance(valor, Real):
        raise ErrorSimulacionReposicion("valor_invalido")
    try:
        numero = float(valor)
        if valor < 0 or not isfinite(numero) or (valor != 0 and numero == 0):
            raise ValueError("Fuera de rango")
    except (ValueError, OverflowError) as error:
        raise ErrorSimulacionReposicion("valor_invalido") from error
    if valor != numero:
        raise ErrorSimulacionReposicion("precision_no_representable")
    return Fraction.from_float(numero)


def _representar(valor):
    """Salida binary64: redondeo natural de fracciones, nunca unidades enteras perdidas."""
    if valor > _MAXIMO:
        raise ErrorSimulacionReposicion("resultado_no_finito")
    numero = float(valor)
    if not isfinite(numero) or (valor > 0 and numero == 0):
        raise ErrorSimulacionReposicion("resultado_no_finito")
    if valor.denominator == 1 and Fraction.from_float(numero) != valor:
        raise ErrorSimulacionReposicion("precision_no_representable")
    return numero


def calcular_simulacion(pronostico, detalles, inventario_disponible,
                        tiempo_abastecimiento, reserva) -> dict:
    """Calcula Q=max(0,D15+R-I) sobre quince estimaciones, sin consultar datos.

    pronostico es un Mapping con id, tecnica, fecha_corte y horizonte (15),
    coherente con la cabecera pública del GET de pronósticos. experimento_id
    es opcional; si falta se devuelve None. Se reconocen las tres técnicas
    publicadas. Si constan tipo o límites de entrenamiento, se validan.
    Otros metadatos adicionales no intervienen en el cálculo. detalles es una
    secuencia de Mapping con paso, fecha_objetivo y venta_estimada. Un eventual
    pronostico_id en cabecera o detalles debe coincidir con id.

    Fechas: date exacto o ISO canónico, nunca datetime. Cantidades: numbers.Real
    salvo bool. Un float significa su valor binario recibido, no el decimal de
    str(float). Decimal y strings no se admiten. Enteros y otros Real solo se
    aceptan si su conversión binary64 es exacta; no se pierden unidades de entrada.
    El plazo es int Python exacto 0..15. No se mutan entradas.

    D15 y Q se calculan con Fraction desde cada entrada, sin redondear D15 antes
    de la cancelación. Las salidas siguen siendo float compatibles con JSON y
    SQLite REAL: fracciones no enteras se aproximan una sola vez con redondeo
    natural binary64. Un total entero que no cabe exactamente se rechaza con
    precision_no_representable, aunque Q aislada pudiera representarse. Se
    rechazan desbordamientos y positivos convertidos a cero. No hay tolerancias
    ni reinterpretación decimal: 3*float(0.1)-float(0.3) es positivo, no cero.

    La recepción ocurre al inicio del día L+1; L=15 queda fuera del horizonte.
    L no altera Q. No se calcula trayectoria, déficit ni cumplimiento de reserva.
    La elegibilidad de cortes y la existencia en SQLite pertenecen a integración.
    """
    if not isinstance(pronostico, Mapping) or any(
        k not in pronostico for k in ("id", "tecnica", "fecha_corte", "horizonte")
    ):
        raise ErrorSimulacionReposicion("estructura_invalida")
    identificador = _identificador(pronostico["id"])
    experimento = (_identificador(pronostico["experimento_id"])
                   if "experimento_id" in pronostico else None)
    if "pronostico_id" in pronostico and _identificador(pronostico["pronostico_id"]) != identificador:
        raise ErrorSimulacionReposicion("identificador_invalido")
    if (type(pronostico["tecnica"]) is not str or pronostico["tecnica"] not in TECNICAS
            or type(pronostico["horizonte"]) is not int or pronostico["horizonte"] != HORIZONTE
            or ("tipo" in pronostico and pronostico["tipo"] not in ("validacion", "prueba", "futuro"))):
        raise ErrorSimulacionReposicion("metadatos_invalidos")
    corte = _fecha(pronostico["fecha_corte"])
    if "inicio_entrenamiento" in pronostico or "fin_entrenamiento" in pronostico:
        if not all(k in pronostico for k in ("inicio_entrenamiento", "fin_entrenamiento")):
            raise ErrorSimulacionReposicion("metadatos_invalidos")
        if not _fecha(pronostico["inicio_entrenamiento"]) <= _fecha(pronostico["fin_entrenamiento"]) <= corte:
            raise ErrorSimulacionReposicion("metadatos_invalidos")
    if type(tiempo_abastecimiento) is not int or not 0 <= tiempo_abastecimiento <= HORIZONTE:
        raise ErrorSimulacionReposicion("plazo_invalido")
    inventario, reserva_normalizada = _numero(inventario_disponible), _numero(reserva)
    try:
        fechas = [corte + timedelta(days=p) for p in range(1, HORIZONTE + 1)]
        recepcion = corte + timedelta(days=tiempo_abastecimiento + 1)
    except OverflowError as error:
        raise ErrorSimulacionReposicion("resultado_no_finito") from error
    if not isinstance(detalles, Sequence) or isinstance(detalles, (str, bytes, bytearray)):
        raise ErrorSimulacionReposicion("estructura_invalida")
    if len(detalles) != HORIZONTE:
        raise ErrorSimulacionReposicion("alineacion_invalida")
    estimaciones = []
    for paso, (detalle, fecha) in enumerate(zip(detalles, fechas), 1):
        if not isinstance(detalle, Mapping) or any(
            k not in detalle for k in ("paso", "fecha_objetivo", "venta_estimada")
        ):
            raise ErrorSimulacionReposicion("estructura_invalida")
        if "pronostico_id" in detalle and _identificador(detalle["pronostico_id"]) != identificador:
            raise ErrorSimulacionReposicion("identificador_invalido")
        if type(detalle["paso"]) is not int or detalle["paso"] != paso or _fecha(detalle["fecha_objetivo"]) != fecha:
            raise ErrorSimulacionReposicion("alineacion_invalida")
        estimaciones.append(_numero(detalle["venta_estimada"]))
    demanda_exacta = sum(estimaciones, Fraction(0))
    neto = demanda_exacta + reserva_normalizada - inventario
    demanda = _representar(demanda_exacta)
    reposicion = _representar(max(Fraction(0), neto))
    advertencias = ["Cantidad hipotética; no constituye una orden real ni garantiza cobertura durante el plazo."]
    if Fraction.from_float(demanda) != demanda_exacta or Fraction.from_float(reposicion) != max(Fraction(0), neto):
        advertencias.append("La representación float de cantidades fraccionarias es aproximada; Q se calculó antes de esa conversión final.")
    if tiempo_abastecimiento == HORIZONTE:
        advertencias.append("La recepción queda fuera del horizonte; no se considera recibida durante las quince fechas.")
    return {
        "version_regla": VERSION_REGLA, "pronostico_id": identificador,
        "experimento_id": experimento, "tecnica": pronostico["tecnica"],
        "fecha_corte": corte.isoformat(), "horizonte": HORIZONTE,
        "parametros": {"inventario_disponible": float(inventario),
                       "tiempo_abastecimiento": tiempo_abastecimiento, "reserva": float(reserva_normalizada)},
        "periodo_cobertura": {"fecha_inicio": fechas[0].isoformat(), "fecha_fin": fechas[-1].isoformat(), "dias": HORIZONTE},
        "demanda_considerada": demanda, "reposicion": reposicion,
        "fecha_recepcion": recepcion.isoformat(), "recepcion_dentro_horizonte": tiempo_abastecimiento < HORIZONTE,
        "supuestos": ["Los float se interpretan en binario; las salidas fraccionarias tienen el redondeo natural binary64.",
                      "Inventario, reserva y estimaciones comparten la misma unidad hipotética.",
                      "Cobertura del horizonte completo, sin otras entradas ni pedidos pendientes iniciales.",
                      "Recepción al inicio del día L+1; el plazo no altera la cantidad calculada.",
                      "Reserva como objetivo hipotético, sin acreditar nivel de servicio.",
                      "Sin trayectoria diaria ni comprobación de existencia o elegibilidad en SQLite."],
        "advertencias": advertencias,
    }
