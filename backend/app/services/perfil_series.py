"""Perfil descriptivo de una serie activa hasta un corte autorizado, sin escritura.

Se reutiliza la conexión ro con BEGIN y la selección del activo de HU004-F.
Se retiene únicamente una serie (máximo 10.000 filas; consulta LIMIT N+1),
para percentiles exactos sobre los valores REAL almacenados. El calendario se
limita a 36.600 días. Memoria O(N + días del periodo), sin matrices predictoras.
No abre bases al importar, ni publica rutas, hashes o SQL en sus errores.
"""

from datetime import date, timedelta
from math import fsum, isfinite
from pathlib import Path
from statistics import stdev

from backend.app.services.consulta_pronosticos import (
    ErrorConsultaPronostico, _activo, _conexion_lectura,
)
from backend.app.services.validacion_csv import FECHA_ISO, MAX_ENTERO_SQLITE
from scripts.analizar_periodo_seleccion import VENTANAS_POSTERIORES


CORTES_AUTORIZADOS = tuple(v.inicio - timedelta(days=1) for v in VENTANAS_POSTERIORES[:3])
MAX_OBSERVACIONES = 10_000
MAX_DIAS_CALENDARIO = 36_600
_MENSAJES = {
    "serie_invalida": "El identificador de serie debe ser un entero positivo válido.",
    "corte_invalido": "La fecha de corte debe ser real y canónica YYYY-MM-DD.",
    "corte_no_autorizado": "El corte no está autorizado para perfilar la serie.",
    "prueba_reservada": "La prueba final permanece reservada.",
    "base_no_disponible": "La base de datos no está disponible o no tiene el esquema necesario.",
    "sin_conjunto_activo": "No existe un conjunto de datos activo.",
    "serie_inexistente": "La serie no existe en el conjunto activo.",
    "historial_insuficiente": "La serie no tiene observaciones hasta el corte solicitado.",
    "datos_inconsistentes": "La serie contiene observaciones o metadatos inconsistentes.",
    "limite_observaciones": "La serie excede el límite de observaciones permitido.",
    "periodo_excesivo": "El periodo de cobertura excede el límite permitido.",
    "resultado_no_finito": "Los datos no permiten obtener estadísticas finitas.",
}


class ErrorPerfilSerie(ValueError):
    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _fecha_canonica(texto):
    if type(texto) is not str or not FECHA_ISO.fullmatch(texto):
        raise ValueError("Fecha no canónica")
    return date.fromisoformat(texto)


def _validar_solicitud(serie_id, fecha_corte):
    if type(serie_id) is not int or not 1 <= serie_id <= MAX_ENTERO_SQLITE:
        raise ErrorPerfilSerie("serie_invalida")
    try:
        corte = _fecha_canonica(fecha_corte)
    except ValueError as causa:
        raise ErrorPerfilSerie("corte_invalido") from causa
    if VENTANAS_POSTERIORES[-1].inicio <= corte <= VENTANAS_POSTERIORES[-1].fin:
        raise ErrorPerfilSerie("prueba_reservada")
    if corte not in CORTES_AUTORIZADOS:
        raise ErrorPerfilSerie("corte_no_autorizado")
    return corte


def _validar_filas(filas, corte):
    observadas = []
    anterior = None
    try:
        for fila in filas:
            fecha = _fecha_canonica(fila["fecha"])
            venta, promocion = fila["sales"], fila["onpromotion"]
            if (fecha > corte or (anterior is not None and fecha <= anterior)
                    or type(venta) not in (int, float) or not isfinite(venta) or venta < 0
                    or type(promocion) is not int or not 0 <= promocion <= MAX_ENTERO_SQLITE):
                raise ValueError("Observación inconsistente")
            observadas.append((fecha, float(venta), promocion))
            anterior = fecha
    except (TypeError, ValueError, OverflowError) as causa:
        raise ErrorPerfilSerie("datos_inconsistentes") from causa
    return observadas


def _percentil(ordenadas, p):
    """Interpolación lineal: h=(n-1)*p; se interpola entre floor(h) y ceil(h)."""
    indice = (len(ordenadas) - 1) * p
    inferior = int(indice)
    superior = min(inferior + 1, len(ordenadas) - 1)
    return ordenadas[inferior] + (indice - inferior) * (ordenadas[superior] - ordenadas[inferior])


def _resumen(ventas):
    total = fsum(ventas)
    return {"observaciones": len(ventas), "total_ventas": total,
            "promedio_por_observacion": total / len(ventas) if ventas else None,
            "ventas_cero": sum(v == 0 for v in ventas)}


def _comprobar_finitud(valor):
    if isinstance(valor, dict):
        for v in valor.values():
            _comprobar_finitud(v)
    elif isinstance(valor, list):
        for v in valor:
            _comprobar_finitud(v)
    elif type(valor) is float and not isfinite(valor):
        raise ErrorPerfilSerie("resultado_no_finito")


def _reporte(conjunto, serie, corte, observadas):
    inicio, ultima = observadas[0][0], observadas[-1][0]
    dias = (corte - inicio).days + 1
    if dias > MAX_DIAS_CALENDARIO:
        raise ErrorPerfilSerie("periodo_excesivo")
    fechas = {f for f, _, _ in observadas}
    ausentes = [(inicio + timedelta(days=i)).isoformat() for i in range(dias)
                if inicio + timedelta(days=i) not in fechas]
    ventas = [v for _, v, _ in observadas]
    ordenadas = sorted(ventas)
    resumen = _resumen(ventas)
    q1, mediana, q3 = (_percentil(ordenadas, p) for p in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    extremos = [(f, v) for f, v, _ in observadas if v < inferior or v > superior]
    dias_semana, meses = {}, {}
    sin_promo, con_promo = [], []
    for fecha, venta, promo in observadas:
        dias_semana.setdefault(fecha.weekday(), []).append(venta)
        meses.setdefault(fecha.month, []).append(venta)
        (con_promo if promo > 0 else sin_promo).append(venta)
    reporte = {
        "alcance": "perfil_descriptivo_serie_hasta_corte",
        "conjunto_id": conjunto,
        "serie": {"id": serie["id"], "store_nbr": serie["store_nbr"], "family": serie["family"]},
        "fecha_corte": corte.isoformat(),
        "cobertura": {
            "periodo_utilizado": {"inicio": inicio.isoformat(), "fin": corte.isoformat(), "ambos_inclusive": True},
            "primera_fecha_observada": inicio.isoformat(), "ultima_fecha_observada": ultima.isoformat(),
            "observaciones": len(observadas), "dias_naturales": dias,
            "fechas_observadas": len(fechas), "fechas_sin_registro": len(ausentes),
            "porcentaje_cobertura_calendario": 100 * len(fechas) / dias,
            "fechas_ausentes": ausentes,
            "definicion": "Desde la primera observación hasta el corte inclusive; no mide ausencias anteriores a esa primera observación.",
        },
        "ventas": {
            "total": resumen["total_ventas"], "promedio_por_observacion": resumen["promedio_por_observacion"],
            "mediana": mediana, "minimo": ordenadas[0], "maximo": ordenadas[-1],
            "desviacion_estandar_muestral": stdev(ventas) if len(ventas) > 1 else None,
            "percentil_25": q1, "percentil_75": q3, "rango_intercuartilico": iqr,
            "metodo_percentiles": "Interpolación lineal con índice (n-1)*p.",
            "ventas_cero": resumen["ventas_cero"], "porcentaje_cero": 100 * resumen["ventas_cero"] / len(ventas),
            "ventas_positivas": len(ventas) - resumen["ventas_cero"],
            "ventas_fraccionarias": sum(not v.is_integer() for v in ventas),
            "porcentaje_fraccionarias": 100 * sum(not v.is_integer() for v in ventas) / len(ventas),
        },
        "estacionalidad": {
            "por_dia_semana": [{"dia_semana": d, **_resumen(dias_semana[d])} for d in sorted(dias_semana)],
            "por_mes": [{"mes": m, **_resumen(meses[m])} for m in sorted(meses)],
            "definicion": "Solo grupos observados; lunes=0, domingo=6; enero=1, diciembre=12.",
        },
        "promociones": {
            "observaciones_sin_promocion": len(sin_promo), "observaciones_con_promocion": len(con_promo),
            "porcentaje_promocionadas": 100 * len(con_promo) / len(ventas),
            "suma_conteos_onpromotion": sum(p for _, _, p in observadas),
            "ventas_sin_promocion": _resumen(sin_promo), "ventas_con_promocion": _resumen(con_promo),
        },
        "valores_extremos": {
            "metodo": "Fuera del intervalo cerrado [Q1-1.5*IQR, Q3+1.5*IQR], con percentiles lineales.",
            "limite_inferior": inferior, "limite_superior": superior, "cantidad": len(extremos),
            "ejemplos": [{"fecha": f.isoformat(), "venta": v} for f, v in extremos[:10]],
            "orden_ejemplos": "Fecha ascendente; máximo diez observaciones.",
            "interpretacion_iqr_cero": "Si IQR=0, ambos límites son Q1=Q3; cualquier valor distinto queda fuera.",
        },
        "advertencias": [
            "Se utilizan exclusivamente observaciones de esta serie hasta el corte inclusive.",
            "Una fecha sin registro no es una venta cero; no se imputan ni completan ventas.",
            "Las diferencias entre grupos temporales no prueban estacionalidad significativa.",
            "Las asociaciones con promociones no prueban causalidad.",
            "Los valores fuera de límites IQR son descriptivos; no se eliminan ni se etiquetan automáticamente como anomalías.",
        ],
    }
    _comprobar_finitud(reporte)
    return reporte


def perfilar_serie(ruta_base: str | Path, serie_id: int, fecha_corte: str) -> dict:
    """Consulta una serie activa hasta uno de los tres cortes (cadena YYYY-MM-DD).

    Los tres argumentos son obligatorios. Errores tipificados mediante
    ErrorPerfilSerie.codigo y mensajes fijos; nunca se devuelve un perfil parcial.
    """
    corte = _validar_solicitud(serie_id, fecha_corte)
    if not isinstance(ruta_base, (str, Path)) or "\0" in str(ruta_base):
        raise ErrorPerfilSerie("base_no_disponible")
    try:
        with _conexion_lectura(ruta_base) as conexion:
            conjunto = _activo(conexion)
            serie = conexion.execute(
                "SELECT id, store_nbr, family FROM serie_temporal WHERE id=? AND conjunto_id=?",
                (serie_id, conjunto),
            ).fetchone()
            if serie is None:
                raise ErrorPerfilSerie("serie_inexistente")
            if (type(serie["store_nbr"]) is not int or serie["store_nbr"] <= 0
                    or type(serie["family"]) is not str or not serie["family"].strip()):
                raise ErrorPerfilSerie("datos_inconsistentes")
            filas = conexion.execute(
                "SELECT fecha, sales, onpromotion FROM venta_diaria "
                "WHERE serie_id=? AND fecha<=? ORDER BY fecha, id LIMIT ?",
                (serie_id, corte.isoformat(), MAX_OBSERVACIONES + 1),
            ).fetchall()
    except ErrorConsultaPronostico as causa:
        raise ErrorPerfilSerie(causa.codigo) from causa
    if len(filas) > MAX_OBSERVACIONES:
        raise ErrorPerfilSerie("limite_observaciones")
    if not filas:
        raise ErrorPerfilSerie("historial_insuficiente")
    observadas = _validar_filas(filas, corte)
    try:
        return _reporte(conjunto, serie, corte, observadas)
    except (OverflowError, ValueError) as causa:
        if isinstance(causa, ErrorPerfilSerie):
            raise
        raise ErrorPerfilSerie("resultado_no_finito") from causa
