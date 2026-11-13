"""MAE, WAPE, RMSE y cobertura de validación, sin E/S ni ejecución de modelos.

Contrato local de HU005-A1: tres cortes autorizados y quince días naturales.
Las constantes coinciden con analizar_periodo_seleccion; no se importa ese
script para mantener este núcleo independiente de pandas y del acceso a datos.
"""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from math import fsum, isfinite, sqrt
from numbers import Real


HORIZONTE_DIAS = 15
CORTES_AUTORIZADOS = (date(2017, 6, 16), date(2017, 7, 1), date(2017, 7, 16))


class ErrorMetricasEvaluacion(ValueError):
    """Entrada inválida o cálculo que no puede representarse como float finito."""

    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


def _fecha(valor, campo: str) -> date:
    if type(valor) is date:
        return valor
    if isinstance(valor, str):
        try:
            fecha = date.fromisoformat(valor)
            if fecha.isoformat() == valor:
                return fecha
        except ValueError:
            pass
    raise ErrorMetricasEvaluacion("fecha_invalida", f"{campo} debe ser date o texto YYYY-MM-DD.")


def _numero(valor, campo: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, Real):
        raise ErrorMetricasEvaluacion("valor_invalido", f"{campo} debe ser un número real, no booleano.")
    try:
        numero = float(valor)
        if valor < 0 or not isfinite(numero) or (valor != 0 and numero == 0):
            raise ValueError("Fuera del rango admitido")
    except (ValueError, OverflowError) as error:
        raise ErrorMetricasEvaluacion(
            "valor_invalido", f"{campo} debe ser finito, no negativo y representable como float.") from error
    return numero


def _filas(filas, campos: tuple[str, ...], nombre: str) -> None:
    if not isinstance(filas, Sequence) or isinstance(filas, (str, bytes, bytearray)):
        raise ErrorMetricasEvaluacion("estructura_invalida", f"{nombre} debe ser una secuencia de diccionarios.")
    for fila in filas:
        if not isinstance(fila, Mapping) or any(campo not in fila for campo in campos):
            raise ErrorMetricasEvaluacion(
                "estructura_invalida", f"Cada fila de {nombre} debe contener {', '.join(campos)}.")


def calcular_metricas(
    fecha_corte: date | str,
    detalles: Sequence[Mapping],
    observaciones: Sequence[Mapping],
) -> dict:
    """Evalúa un único pronóstico sobre sus observaciones reales disponibles.

    detalles usa las claves del GET: paso, fecha_objetivo, venta_estimada.
    Debe contener exactamente los pasos 1..15 en orden y las fechas corte+paso.
    observaciones usa fecha y sales; admite cualquier orden, nunca duplicados
    ni fechas ajenas al horizonte. Las fechas aceptan date o ISO YYYY-MM-DD
    canónico; se rechazan datetime. Los valores aceptan numbers.Real salvo bool
    y se convierten a float. Se ignoran claves adicionales, sin mutar entradas.

    MAE = fsum(errores absolutos) / observaciones_evaluadas.
    WAPE = 100 * fsum(errores absolutos) / fsum(abs(sales)), en porcentaje;
    no representa exactitud y puede superar 100.
    RMSE = sqrt(media de errores al cuadrado), en las mismas fechas observadas.
    Se normalizan los errores por su máximo antes de elevarlos al cuadrado
    y se restituye esa escala después de la raíz, evitando desbordamientos.
    Si el máximo es cero, RMSE es cero. Sin reales las tres métricas son None;
    con reales de suma cero solo WAPE es None. No se imputan fechas ausentes.
    cobertura = observaciones_evaluadas / 15, una proporción entre 0 y 1.
    El detalle siempre incluye quince fechas, con venta_real y error_absoluto
    None cuando no existe observación. Las métricas parciales se acompañan
    de sus conteos y cobertura; no se comparan ni clasifican experimentos.

    Lanza ErrorMetricasEvaluacion (ValueError) ante entradas inválidas,
    sumas desbordadas o resultados no finitos. No redondea resultados.
    """
    corte = _fecha(fecha_corte, "fecha_corte")
    if corte not in CORTES_AUTORIZADOS:
        raise ErrorMetricasEvaluacion(
            "corte_no_autorizado", "Solo se admiten los tres cortes de validación; la prueba final está reservada.")
    _filas(detalles, ("paso", "fecha_objetivo", "venta_estimada"), "detalles")
    _filas(observaciones, ("fecha", "sales"), "observaciones")
    if len(detalles) != HORIZONTE_DIAS:
        raise ErrorMetricasEvaluacion("horizonte_invalido", "Se requieren exactamente quince estimaciones.")

    predicciones = []
    for paso, fila in enumerate(detalles, 1):
        fecha = corte + timedelta(days=paso)
        if (type(fila["paso"]) is not int or fila["paso"] != paso
                or _fecha(fila["fecha_objetivo"], "fecha_objetivo") != fecha):
            raise ErrorMetricasEvaluacion(
                "alineacion_invalida", "Se requieren pasos 1 a 15 y fechas corte+paso, en orden y sin omisiones.")
        predicciones.append((paso, fecha, _numero(fila["venta_estimada"], "venta_estimada")))

    reales = {}
    for fila in observaciones:
        fecha = _fecha(fila["fecha"], "fecha")
        if not corte < fecha <= corte + timedelta(days=HORIZONTE_DIAS):
            raise ErrorMetricasEvaluacion("observacion_fuera_horizonte", "La fecha real está fuera del horizonte.")
        if fecha in reales:
            raise ErrorMetricasEvaluacion("fecha_duplicada", "Existe más de una observación para la misma fecha.")
        reales[fecha] = _numero(fila["sales"], "sales")

    resultado_detalles = []
    ausentes = []
    errores = []
    for paso, fecha, estimada in predicciones:
        real = reales.get(fecha)
        error = None if real is None else abs(real - estimada)
        if error is None:
            ausentes.append(fecha.isoformat())
        else:
            errores.append(error)
        resultado_detalles.append({
            "paso": paso, "fecha_objetivo": fecha.isoformat(), "venta_estimada": estimada,
            "venta_real": real, "error_absoluto": error,
        })

    cantidad = len(reales)
    mae = wape = rmse = None
    if cantidad:
        try:
            suma_errores = fsum(errores)
            suma_reales = fsum(abs(valor) for valor in reales.values())
            mae = suma_errores / cantidad
            # Dividir primero evita desbordar al multiplicar la suma por 100.
            wape = (suma_errores / suma_reales) * 100 if suma_reales else None
            escala = max(errores)
            rmse = (escala * sqrt(fsum((error / escala) ** 2 for error in errores) / cantidad)
                    if escala else 0.0)
            if not all(isfinite(v) for v in (suma_errores, suma_reales, mae, rmse)
                       + (() if wape is None else (wape,))):
                raise ValueError("Resultado no finito")
        except (OverflowError, ValueError) as error:
            raise ErrorMetricasEvaluacion(
                "resultado_no_finito", "Las sumas o métricas exceden el rango numérico finito.") from error

    return {
        "fecha_corte": corte.isoformat(), "horizonte": HORIZONTE_DIAS,
        "observaciones_evaluadas": cantidad, "fechas_sin_registro": len(ausentes),
        "cobertura": cantidad / HORIZONTE_DIAS,
        "definicion_cobertura": "observaciones_evaluadas / horizonte (15 fechas naturales)",
        "mae": mae, "wape": wape, "rmse": rmse, "detalles": resultado_detalles,
        "fechas_ausentes": ausentes,
    }
