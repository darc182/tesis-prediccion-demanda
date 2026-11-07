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
from backend.app.services import configuracion_experimentos as configuracion


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


def _calcular(instantanea: _Instantanea, tecnica: str, corte: date,
              *, configuracion_solicitada: dict | None = None):
    """Adaptador a los scripts existentes: sin SQL ni lógica predictiva propia."""
    datos = pd.DataFrame(instantanea.filas, columns=["date", "sales", "onpromotion"])
    datos["store_nbr"] = instantanea.store_nbr
    datos["family"] = instantanea.family
    datos = preparar_observaciones(datos)
    reglas = configuracion.reglas_heredadas(tecnica)
    versiones = {"python": platform.python_version(), "pandas": pd.__version__}
    if tecnica == linea_base.TECNICA:
        predicciones = linea_base.pronosticar_serie(
            datos, instantanea.store_nbr, instantanea.family, corte,
        )
        parametros = {"periodo_estacional_dias": 7}
        entrenamiento = {"observaciones_disponibles": len(datos)}
        auditoria = {"respaldos_directos": sum(p.uso_respaldo for p in predicciones)}
    else:
        # Importación diferida: la línea base no necesita cargar scikit-learn.
        from scripts import modelos_prediccion as modelos
        efectiva = modelos.normalizar_configuracion_entrenamiento(tecnica, configuracion_solicitada)
        argumentos = {} if configuracion_solicitada is None else {"configuracion_solicitada": efectiva}
        modelo = modelos.entrenar_modelo(
            datos, instantanea.store_nbr, instantanea.family, tecnica, corte, **argumentos)
        modelos.comprobar_parametros_efectivos(modelo.estimador, efectiva)
        predicciones = modelos.pronosticar_recursivo(modelo, datos)
        parametros = modelo.estimador.get_params(deep=False)
        if tuple(modelos.COLUMNAS_PREDICTORAS) != configuracion.VARIABLES:
            raise ValueError("Las variables reales no corresponden al contrato predictivo.")
        versiones["scikit_learn"] = modelos.sklearn.__version__
        entrenamiento = {"filas_validas": modelo.filas_validas,
                          "filas_descartadas": modelo.filas_descartadas,
                          "ultima_respuesta_real": modelo.ultima_fecha_entrenamiento.isoformat()}
        auditoria = {"ajustes_a_cero": sum(p.ajustada_a_cero for p in predicciones)}
    inicio = datos["date"].min().date().isoformat()
    fin = datos["date"].max().date().isoformat()
    huella = hashlib.sha256(json.dumps(instantanea.filas, ensure_ascii=False,
                                      allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    predictiva = configuracion.normalizar_configuracion({"tecnica": tecnica, "parametros": parametros})
    if configuracion.json_canonico(predictiva) != configuracion.json_canonico(
            configuracion.normalizar_configuracion(
                {"tecnica": tecnica} if configuracion_solicitada is None else configuracion_solicitada)):
        raise ValueError("Los parámetros reales no corresponden a la configuración solicitada.")
    config = {"version_registro": 2, "configuracion_predictiva": predictiva,
              "tecnica": tecnica, "parametros": parametros,
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


def _contenido_config(config):
    """Identidad predictiva común a registros HU004/F1 y F2; excluye auditoría."""
    return configuracion.identidad_configuracion(configuracion.recuperar_configuracion(config))


def _leer_configuracion_estricta(texto):
    """Lee sin resolver claves ambiguas ni admitir infinitos en la auditoría."""
    def objeto_univoco(pares):
        objeto = {}
        for clave, valor in pares:
            if clave in objeto:
                raise ValueError("Clave JSON duplicada.")
            objeto[clave] = valor
        return objeto

    guardada = json.loads(texto, object_pairs_hook=objeto_univoco)
    # Incluye NaN/Infinity aceptados por el lector y desbordamientos como 1e309.
    # Solo valida en memoria: nunca sanea ni reescribe el texto almacenado.
    json.dumps(guardada, allow_nan=False)
    return guardada


def _buscar_identico(conexion, serie_id, tecnica, corte, config, inicio, fin, detalles):
    existentes = conexion.execute(
        "SELECT e.id, p.id, e.configuracion_json, e.inicio_entrenamiento, "
        "e.fin_entrenamiento, e.ruta_modelo, p.horizonte FROM experimento e "
        "JOIN pronostico p ON p.experimento_id = e.id "
        "WHERE e.serie_id = ? AND e.tecnica = ? AND p.fecha_corte = ? AND p.tipo = ?",
        (serie_id, tecnica, corte.isoformat(), "validacion"),
    ).fetchall()
    contenido = _contenido_config(config)
    coincidencias = []
    for fila in existentes:
        try:
            guardada = _leer_configuracion_estricta(fila[2])
            if not isinstance(guardada, dict) or guardada.get("tecnica") != tecnica:
                raise ValueError("Técnica almacenada inconsistente con el experimento.")
            misma_config = _contenido_config(guardada) == contenido
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise ErrorRegistroPronostico(
                "conflicto", "No se puede reconstruir con seguridad la identidad de un pronóstico almacenado.")
        if misma_config:
            coincidencias.append((fila, guardada))
    if not coincidencias:
        return None
    # No elegir arbitrariamente ni siquiera si uno de los duplicados coincide.
    if len(coincidencias) != 1:
        raise ErrorRegistroPronostico(
            "conflicto", "Existen varios pronósticos ambiguos con la misma configuración.")
    fila, guardada = coincidencias[0]
    e, p, _, inicio_guardado, fin_guardado, ruta_modelo, horizonte = fila
    # No forma parte de la identidad predictiva. Sigue siendo evidencia obligatoria
    # para reutilizar: también detecta cambios entre ejecuciones, aunque y_hat coincida.
    if (not isinstance(guardada.get("historial"), dict)
            or guardada["historial"] != config.get("historial")):
        raise ErrorRegistroPronostico(
            "conflicto", "El histórico asociado a la misma configuración cambió o no puede verificarse.")
    guardados = _leer_detalles(conexion, p)
    if len(guardados) != HORIZONTE_DIAS:
        raise ErrorRegistroPronostico(
            "conflicto", "Error de integridad: pronóstico incompleto para la misma configuración.")
    # La hora de generación y la duración medida difieren entre ejecuciones;
    # no son parámetros ni valores del resultado a comparar.
    if (inicio_guardado == inicio and fin_guardado == fin and ruta_modelo is None
            and horizonte == HORIZONTE_DIAS and guardados == detalles):
        return e, p
    raise ErrorRegistroPronostico(
        "conflicto", "Ya existe un pronóstico conflictivo para la misma configuración.")


def registrar_pronostico(
    ruta_base: str | Path, serie_id: int, tecnica: str, fecha_corte: date | str,
    *, configuracion_solicitada: dict | None = None,
) -> ResultadoRegistro:
    """Genera antes del bloqueo; registra o reutiliza una validación completa.

    Lanza ErrorRegistroPronostico con código y mensaje ante rechazo o fallo.
    configuracion_solicitada es un objeto F2 parcial o completo, solo por palabra
    clave. None conserva HU004. La API HTTP actual no expone este argumento.
    No se inicia ninguna operación por importar este módulo.
    """
    corte = _validar_solicitud(serie_id, tecnica, fecha_corte)
    try:
        efectiva = configuracion.normalizar_configuracion(
            {"tecnica": tecnica} if configuracion_solicitada is None else configuracion_solicitada)
        if efectiva["tecnica"] != tecnica:
            raise ValueError("La técnica de la configuración no coincide con la solicitada.")
        variables = [] if tecnica == linea_base.TECNICA else list(configuracion.VARIABLES)
        if efectiva["variables"] != variables:
            raise ValueError("F3 requiere las variables originales en su orden original.")
    except (ValueError, TypeError) as error:
        raise ErrorRegistroPronostico("error_calculo", f"Configuración inválida: {error}") from error
    ruta = Path(ruta_base)
    try:
        with closing(_abrir_base(ruta)) as conexion:
            conexion.execute("BEGIN")
            instantanea = _leer_instantanea(conexion, serie_id, corte)
    except sqlite3.Error as error:
        raise ErrorRegistroPronostico("base_no_disponible", f"Base no disponible o sin esquema: {error}") from error

    inicio_calculo = perf_counter()
    try:
        argumentos = {} if configuracion_solicitada is None else {"configuracion_solicitada": efectiva}
        predicciones, config, inicio, fin = _calcular(instantanea, tecnica, corte, **argumentos)
        detalles = _validar_predicciones(predicciones, instantanea, tecnica, corte)
        if not isinstance(config, dict) or config.get("tecnica") != tecnica:
            raise ErrorRegistroPronostico("resultado_invalido", "La configuración no corresponde a la técnica solicitada.")
        _contenido_config(config)  # validar antes del bloqueo, también al primer registro
        if (configuracion_solicitada is not None
                and _contenido_config(config) != configuracion.identidad_configuracion(efectiva)):
            raise ErrorRegistroPronostico("resultado_invalido", "La configuración calculada no corresponde a la solicitada.")
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
