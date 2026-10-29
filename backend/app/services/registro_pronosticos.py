"""Generación y registro atómico de validaciones, sin efectos al importar.

La lectura usa una instantánea consistente y se cierra antes del cálculo.
BEGIN IMMEDIATE serializa la comprobación de duplicados y las tres escrituras.
Los límites de experimento describen el histórico real disponible, no un
calendario completado ni todas las filas utilizadas para ajustar un regresor.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import platform
import sqlite3
from time import perf_counter

import pandas as pd

from scripts import linea_base_prediccion as linea_base
from scripts.analizar_periodo_seleccion import HORIZONTE_DIAS, VENTANAS_POSTERIORES
from scripts.preparar_series_modelado import preparar_observaciones


TECNICAS = (linea_base.TECNICA, "random_forest", "hist_gradient_boosting")
CORTES_VALIDACION = tuple(v.inicio - timedelta(days=1) for v in VENTANAS_POSTERIORES[:3])
PRUEBA_FINAL = VENTANAS_POSTERIORES[-1]


class ErrorRegistroPronostico(ValueError):
    def __init__(self, codigo: str, mensaje: str):
        self.codigo = codigo
        super().__init__(mensaje)


@dataclass(frozen=True)
class ResultadoRegistro:
    conjunto_id: int
    serie_id: int
    experimento_id: int
    pronostico_id: int
    tecnica: str
    fecha_corte: str
    horizonte: int
    cantidad_detalles: int
    nuevo: bool

    @property
    def reutilizado(self) -> bool:
        return not self.nuevo


@dataclass(frozen=True)
class _Instantanea:
    conjunto_id: int
    hash_conjunto: str
    serie_id: int
    store_nbr: int
    family: str
    filas: tuple[tuple[str, float, int], ...]


def _validar_solicitud(serie_id: int, tecnica: str, fecha_corte: date | str) -> date:
    if type(serie_id) is not int or serie_id <= 0:
        raise ErrorRegistroPronostico("serie_invalida", "serie_id debe ser un entero positivo.")
    if tecnica not in TECNICAS:
        raise ErrorRegistroPronostico("tecnica_invalida", f"Técnica desconocida: {tecnica}.")
    if isinstance(fecha_corte, str):
        try:
            corte = date.fromisoformat(fecha_corte)
            if corte.isoformat() != fecha_corte:
                raise ValueError("Formato no canónico")
        except ValueError as error:
            raise ErrorRegistroPronostico("corte_invalido", "Fecha de corte inválida; use YYYY-MM-DD.") from error
    elif type(fecha_corte) is date:
        corte = fecha_corte
    else:
        raise ErrorRegistroPronostico("corte_invalido", "Fecha de corte inválida.")
    # Comprobar la reserva explícitamente, incluso para cortes no autorizados.
    if PRUEBA_FINAL.inicio - timedelta(days=HORIZONTE_DIAS) <= corte < PRUEBA_FINAL.fin:
        raise ErrorRegistroPronostico("prueba_reservada", "El horizonte intersecta la prueba final reservada.")
    if corte not in CORTES_VALIDACION:
        raise ErrorRegistroPronostico("corte_no_autorizado", f"Corte no autorizado: {corte}.")
    return corte


def _abrir_base(ruta: Path, escritura: bool = False) -> sqlite3.Connection:
    """mode=rw/ro evita crear archivos incluso si la ruta desaparece al abrir."""
    if not ruta.is_file():
        raise ErrorRegistroPronostico("base_no_disponible", f"La base no existe: {ruta}")
    modo = "rw" if escritura else "ro"
    conexion = sqlite3.connect(f"{ruta.resolve().as_uri()}?mode={modo}", uri=True)
    try:
        conexion.execute("PRAGMA foreign_keys = ON")
    except Exception:
        conexion.close()
        raise
    return conexion


def _leer_instantanea(conexion: sqlite3.Connection, serie_id: int, corte: date) -> _Instantanea:
    # Misma selección del activo que los endpoints existentes; ningún ID fijo.
    activo = conexion.execute(
        "SELECT id, hash FROM conjunto_datos WHERE activo = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if activo is None:
        raise ErrorRegistroPronostico("sin_conjunto_activo", "No existe un conjunto activo.")
    serie = conexion.execute(
        "SELECT conjunto_id, store_nbr, family FROM serie_temporal WHERE id = ?", (serie_id,)
    ).fetchone()
    if serie is None:
        raise ErrorRegistroPronostico("serie_inexistente", f"La serie {serie_id} no existe.")
    if serie[0] != activo[0]:
        raise ErrorRegistroPronostico("serie_inactiva", f"La serie {serie_id} no pertenece al conjunto activo.")
    filas = tuple(conexion.execute(
        "SELECT fecha, sales, onpromotion FROM venta_diaria "
        "WHERE serie_id = ? AND fecha <= ? ORDER BY fecha", (serie_id, corte.isoformat())
    ).fetchall())
    if not filas:
        raise ErrorRegistroPronostico("sin_historial", f"Serie {serie_id}: no hay historial hasta {corte}.")
    return _Instantanea(activo[0], activo[1], serie_id, serie[1], serie[2], filas)


def _calcular(instantanea: _Instantanea, tecnica: str, corte: date):
    """Adaptador a los scripts existentes: sin SQL ni lógica predictiva propia."""
    datos = pd.DataFrame(instantanea.filas, columns=["date", "sales", "onpromotion"])
    datos["store_nbr"] = instantanea.store_nbr
    datos["family"] = instantanea.family
    datos = preparar_observaciones(datos)
    reglas = {"horizonte_dias": HORIZONTE_DIAS, "ventas_reales_hasta_corte_inclusive": True,
              "ausencias_no_son_ceros": True, "redondeo_interno": False}
    versiones = {"python": platform.python_version(), "pandas": pd.__version__}
    if tecnica == linea_base.TECNICA:
        predicciones = linea_base.pronosticar_serie(
            datos, instantanea.store_nbr, instantanea.family, corte,
        )
        parametros = {"periodo_estacional_dias": 7}
        reglas.update({"referencia": "objetivo menos 7 días; real hasta corte o predicción previa",
                       "respaldos": ["última observación real del mismo día semanal",
                                     "última observación real disponible"],
                       "sin_historial": "error"})
        entrenamiento = {"observaciones_disponibles": len(datos)}
        auditoria = {"respaldos_directos": sum(p.uso_respaldo for p in predicciones)}
    else:
        # Importación diferida: la línea base no necesita cargar scikit-learn.
        from scripts import modelos_prediccion as modelos
        modelo = modelos.entrenar_modelo(datos, instantanea.store_nbr, instantanea.family, tecnica, corte)
        predicciones = modelos.pronosticar_recursivo(modelo, datos)
        parametros = modelo.estimador.get_params(deep=False)
        reglas.update({"predictoras": list(modelos.COLUMNAS_PREDICTORAS),
                       "onpromotion_predictor": False, "dias_naturales_exactos": True,
                       "media_7": "siete observaciones de los siete días anteriores completos",
                       "dia_semana": "lunes=0, domingo=6", "mes": "enero=1, diciembre=12",
                       "entrenamiento_incompleto": "excluir fila",
                       "inferencia_incompleta": "error sin imputación",
                       "recursion": "incorporar solo estimaciones previas al objetivo",
                       "negativos": "comprobar finitud y aplicar max(0, predicción)",
                       "hilos_numericos": 1})
        versiones["scikit_learn"] = modelos.sklearn.__version__
        entrenamiento = {"filas_validas": modelo.filas_validas,
                          "filas_descartadas": modelo.filas_descartadas,
                          "ultima_respuesta_real": modelo.ultima_fecha_entrenamiento.isoformat()}
        auditoria = {"ajustes_a_cero": sum(p.ajustada_a_cero for p in predicciones)}
    inicio = datos["date"].min().date().isoformat()
    fin = datos["date"].max().date().isoformat()
    huella = hashlib.sha256(json.dumps(instantanea.filas, ensure_ascii=False,
                                      allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    config = {"version_registro": 1, "tecnica": tecnica, "parametros": parametros,
              "reglas": reglas, "versiones": versiones, "entrenamiento": entrenamiento,
              "auditoria": auditoria,
              "historial": {"inicio_disponible": inicio, "fin_disponible": fin,
                            "cantidad": len(datos), "sha256_observaciones": huella,
                            "conjunto_id": instantanea.conjunto_id,
                            "hash_conjunto": instantanea.hash_conjunto}}
    return tuple(predicciones), config, inicio, fin


def _validar_predicciones(predicciones, instantanea: _Instantanea, tecnica: str, corte: date):
    if len(predicciones) != HORIZONTE_DIAS:
        raise ErrorRegistroPronostico("resultado_invalido", "Se requieren exactamente 15 predicciones.")
    detalles = []
    for paso, p in enumerate(predicciones, 1):
        if (type(p.store_nbr) is not int or type(p.fecha_corte) is not date
                or (p.store_nbr, p.family, p.tecnica, p.fecha_corte) != (
                    instantanea.store_nbr, instantanea.family, tecnica, corte)):
            raise ErrorRegistroPronostico("resultado_invalido", "Los metadatos no corresponden a la serie, técnica y corte solicitados.")
        if (type(p.paso) is not int or p.paso != paso or type(p.fecha_objetivo) is not date
                or p.fecha_objetivo != corte + timedelta(days=paso)):
            raise ErrorRegistroPronostico("resultado_invalido", "Pasos o fechas objetivo incorrectos.")
        if isinstance(p.venta_estimada, bool) or not isinstance(p.venta_estimada, Real):
            raise ErrorRegistroPronostico("resultado_invalido", "La venta estimada debe ser numérica.")
        valor = float(p.venta_estimada)
        if not isfinite(valor) or valor < 0:
            raise ErrorRegistroPronostico("resultado_invalido", "La venta estimada debe ser finita y no negativa.")
        detalles.append((paso, p.fecha_objetivo.isoformat(), valor))
    return detalles


def _leer_detalles(conexion, pronostico_id):
    return conexion.execute(
        "SELECT paso, fecha_objetivo, venta_estimada FROM detalle_pronostico "
        "WHERE pronostico_id = ? ORDER BY paso", (pronostico_id,),
    ).fetchall()


def _buscar_identico(conexion, serie_id, tecnica, corte, config, inicio, fin, detalles):
    existentes = conexion.execute(
        "SELECT e.id, p.id, e.configuracion_json, e.inicio_entrenamiento, "
        "e.fin_entrenamiento, e.ruta_modelo, p.horizonte FROM experimento e "
        "JOIN pronostico p ON p.experimento_id = e.id "
        "WHERE e.serie_id = ? AND e.tecnica = ? AND p.fecha_corte = ? AND p.tipo = ?",
        (serie_id, tecnica, corte.isoformat(), "validacion"),
    ).fetchall()
    if not existentes:
        return None
    if len(existentes) == 1:
        e, p, texto, inicio_guardado, fin_guardado, ruta_modelo, horizonte = existentes[0]
        try:
            misma_config = json.loads(texto) == config
        except (ValueError, TypeError):
            misma_config = False
        guardados = _leer_detalles(conexion, p)
        # La hora de generación y la duración medida difieren entre ejecuciones;
        # no son parámetros ni valores del resultado a comparar.
        if (misma_config and inicio_guardado == inicio and fin_guardado == fin
                and ruta_modelo is None and horizonte == HORIZONTE_DIAS and guardados == detalles):
            return e, p
    raise ErrorRegistroPronostico("conflicto", "Ya existe un pronóstico conflictivo o incompleto para esta serie, técnica, corte y tipo.")


def registrar_pronostico(
    ruta_base: str | Path, serie_id: int, tecnica: str, fecha_corte: date | str,
) -> ResultadoRegistro:
    """Genera antes del bloqueo; registra o reutiliza una validación completa.

    Lanza ErrorRegistroPronostico con código y mensaje ante rechazo o fallo.
    No se inicia ninguna operación por importar este módulo.
    """
    corte = _validar_solicitud(serie_id, tecnica, fecha_corte)
    ruta = Path(ruta_base)
    try:
        with closing(_abrir_base(ruta)) as conexion:
            conexion.execute("BEGIN")
            instantanea = _leer_instantanea(conexion, serie_id, corte)
    except sqlite3.Error as error:
        raise ErrorRegistroPronostico("base_no_disponible", f"Base no disponible o sin esquema: {error}") from error

    inicio_calculo = perf_counter()
    try:
        predicciones, config, inicio, fin = _calcular(instantanea, tecnica, corte)
        detalles = _validar_predicciones(predicciones, instantanea, tecnica, corte)
        texto_config = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except ErrorRegistroPronostico:
        raise
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        raise ErrorRegistroPronostico("error_calculo", f"Serie {serie_id}, corte {corte}: {error}") from error
    duracion = perf_counter() - inicio_calculo
    try:
        with closing(_abrir_base(ruta, escritura=True)) as conexion:
            try:
                conexion.execute("BEGIN IMMEDIATE")
                actual = _leer_instantanea(conexion, serie_id, corte)
                if actual != instantanea:
                    raise ErrorRegistroPronostico("historial_cambiado", "El conjunto, la serie o el histórico cambiaron durante el cálculo.")
                existente = _buscar_identico(conexion, serie_id, tecnica, corte, config, inicio, fin, detalles)
                if existente is not None:
                    conexion.rollback()  # libera el bloqueo sin ninguna escritura
                    e, p = existente
                    return ResultadoRegistro(actual.conjunto_id, serie_id, e, p, tecnica,
                                             corte.isoformat(), HORIZONTE_DIAS, len(detalles), False)
                cursor = conexion.execute(
                    "INSERT INTO experimento (serie_id, tecnica, configuracion_json, "
                    "inicio_entrenamiento, fin_entrenamiento, ruta_modelo) VALUES (?, ?, ?, ?, ?, NULL)",
                    (serie_id, tecnica, texto_config, inicio, fin),
                )
                experimento_id = int(cursor.lastrowid)
                cursor = conexion.execute(
                    "INSERT INTO pronostico (experimento_id, fecha_corte, tipo, horizonte, "
                    "fecha_generacion, duracion_segundos) VALUES (?, ?, ?, ?, ?, ?)",
                    (experimento_id, corte.isoformat(), "validacion", HORIZONTE_DIAS,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"), duracion),
                )
                pronostico_id = int(cursor.lastrowid)
                conexion.executemany(
                    "INSERT INTO detalle_pronostico (pronostico_id, paso, fecha_objetivo, venta_estimada) "
                    "VALUES (?, ?, ?, ?)", [(pronostico_id, *detalle) for detalle in detalles],
                )
                if _leer_detalles(conexion, pronostico_id) != detalles:
                    raise ErrorRegistroPronostico(
                        "error_persistencia", "Registro revertido: no se almacenaron los quince detalles esperados.")
                conexion.commit()
            except Exception:
                conexion.rollback()
                raise
    except sqlite3.Error as error:
        raise ErrorRegistroPronostico("error_persistencia", f"Registro revertido por un error de SQLite: {error}") from error
    return ResultadoRegistro(instantanea.conjunto_id, serie_id, experimento_id, pronostico_id,
                             tecnica, corte.isoformat(), HORIZONTE_DIAS, len(detalles), True)
