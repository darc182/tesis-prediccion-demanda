"""Trayectoria pura de quince días con demanda pendiente acumulada (S4-A3-01)."""

from copy import deepcopy
from datetime import date
from fractions import Fraction

from backend.app.services import simulacion_reposicion as motor


class ErrorTrayectoriaSimulacion(ValueError):
    """Salida diaria no representable; las entradas conservan los errores de S4-A1."""

    def __init__(self, codigo: str):
        self.codigo = codigo
        mensajes = {
            "precision_no_representable": "Un saldo entero de la trayectoria perdería unidades al representarse como float.",
            "resultado_no_finito": "Un saldo de la trayectoria excede el rango numérico representable.",
        }
        super().__init__(mensajes[codigo])


def _saldo(valor: Fraction) -> float:
    """Adapta la política privada S4-A1 _representar a magnitudes con signo.

    Dependencia deliberada: revisar si cambia el representador del motor.
    Rechaza tanto desbordamiento negativo como positivos o negativos no nulos
    que se convertirían a cero; no aproxima enteros perdiendo unidades.
    """
    try:
        magnitud = motor._representar(abs(valor))
    except motor.ErrorSimulacionReposicion as error:
        raise ErrorTrayectoriaSimulacion(error.codigo) from error
    return -magnitud if valor < 0 else magnitud


def calcular_trayectoria(pronostico, detalles, inventario_disponible,
                         tiempo_abastecimiento, reserva) -> dict:
    """Invoca calcular_simulacion una vez y extiende su respuesta sin mutarla.

    S4-A1 es autoridad de validación, horizonte, regla, Q y recepción. Sus
    ErrorSimulacionReposicion se propagan sin alterar sus códigos. La salida
    conserva identificación y campos de S4-A1, reemplaza el supuesto que decía
    no calcular trayectoria y agrega trayectoria (15 filas) y resumen.

    B0=I; Bt=B(t-1)+At-dt. At es exclusivamente la Q float publicada por S4-A1,
    recibida en su fecha, si cae dentro del horizonte. No se reconstruye Q
    desde demanda_considerada ni desde una nueva fórmula de reposición.
    Cada entrada ya validada y la Q publicada se convierten a Fraction binaria
    exacta. Los balances internos nunca se sustituyen por sus salidas float.
    Por tanto B15=I+recepción efectiva-suma exacta de demandas, incluso cuando
    demanda_considerada publicada o Q son aproximadas. No se promete saldo final
    igual a la reserva: una Q aproximada puede producir residuos no nulos.

    Disponible=max(0,Bt); pendiente_actual=max(0,-Bt). Los pendientes son saldos
    acumulados, no ventas perdidas, y no se suman entre días. Resumen identifica
    primera fecha pendiente (o None), máximo pendiente actual, número de días
    negativos y saldos finales. Se cuentan signos exactos, sin tolerancias.
    _saldo comparte la política de representación de S4-A1. Una aproximación
    fraccionaria final se advierte; un saldo fuera de rango rechaza toda la
    función. No hay E/S, ventas reales, selección de cortes ni verificación SQL.
    """
    base = motor.calcular_simulacion(
        pronostico, detalles, inventario_disponible, tiempo_abastecimiento, reserva)
    balance = Fraction.from_float(base["parametros"]["inventario_disponible"])
    pedido = Fraction.from_float(base["reposicion"])
    filas = []
    primera = None
    maximo = Fraction(0)
    dias_negativos = 0
    aproximada = False
    for detalle in detalles:
        fecha = detalle["fecha_objetivo"]
        fecha = fecha.isoformat() if type(fecha) is date else fecha
        demanda = Fraction.from_float(float(detalle["venta_estimada"]))
        recibida = pedido if base["recepcion_dentro_horizonte"] and fecha == base["fecha_recepcion"] else Fraction(0)
        balance = balance + recibida - demanda
        pendiente = max(Fraction(0), -balance)
        disponible = max(Fraction(0), balance)
        saldo, stock, deuda = _saldo(balance), _saldo(disponible), _saldo(pendiente)
        aproximada |= Fraction.from_float(saldo) != balance
        if balance < 0:
            dias_negativos += 1
            if primera is None:
                primera = fecha
        maximo = max(maximo, pendiente)
        filas.append({
            "paso": detalle["paso"], "fecha_objetivo": fecha,
            "demanda_pronosticada": float(demanda), "recepcion_simulada": float(recibida),
            "balance_neto": saldo, "disponible_hipotetico": stock,
            "demanda_pendiente_actual": deuda,
        })
    resultado = deepcopy(base)
    resultado["supuestos"] = [s for s in base["supuestos"]
                              if s != "Sin trayectoria diaria ni comprobación de existencia o elegibilidad en SQLite."]
    resultado["supuestos"].extend([
        "Modelo hipotético de demanda pendiente acumulada; no refleja necesariamente el comportamiento de consumidores.",
        "El balance negativo no es inventario físico negativo ni equivale a ventas perdidas.",
        "Los saldos pendientes diarios no se suman como demandas insatisfechas independientes.",
        "Se recibe una única Q publicada por S4-A1; no se reconstruye su valor previo a la conversión float.",
        "Sin otros pedidos, nivel de servicio acreditado ni comprobación de existencia o elegibilidad en SQLite.",
    ])
    if aproximada:
        resultado["advertencias"].append(
            "Algunos saldos fraccionarios publicados son aproximados; la evolución interna conserva sus valores binarios exactos.")
    resultado["advertencias"].append(
        "La Q publicada puede ser aproximada; no se fuerza el balance final a la reserva ni se eliminan residuos mediante tolerancias.")
    resultado["trayectoria"] = filas
    resultado["resumen"] = {
        "fecha_primera_demanda_pendiente": primera,
        "mayor_demanda_pendiente_actual": _saldo(maximo),
        "dias_con_balance_negativo": dias_negativos,
        "balance_neto_final": filas[-1]["balance_neto"],
        "disponible_hipotetico_final": filas[-1]["disponible_hipotetico"],
        "demanda_pendiente_final": filas[-1]["demanda_pendiente_actual"],
        "recepcion_dentro_horizonte": base["recepcion_dentro_horizonte"],
    }
    return resultado
