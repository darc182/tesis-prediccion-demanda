"""S4-A4-04: historial hipotético desde snapshots, sin reconstruir el origen.

Dependencias privadas deliberadas de S4-A4-03: _verificar_esquema,
_recuperar_en_conexion y el normalizador de ruta de su módulo lectura.
La clave y solicitud necesarias para el lector se obtienen de la cabecera por
ID, bajo la misma conexión. Nunca se publican claves de idempotencia.
"""

from contextlib import contextmanager

from backend.app.services import consulta_pronosticos as conexiones
from backend.app.services import registro_simulaciones as registro


MAX_ENTERO_SQLITE = 2**63 - 1
_MENSAJES = {
    "ejecucion_invalida": "El identificador de ejecución debe ser un entero positivo válido.",
    "filtro_invalido": "Los filtros de historial no tienen los tipos o valores admitidos.",
    "paginacion_invalida": "El límite debe estar entre uno y cincuenta y el desplazamiento debe ser un entero no negativo válido.",
    "ejecucion_inexistente": "La ejecución histórica solicitada no existe.",
    "esquema_no_preparado": "La base necesita preparación explícita del esquema de simulaciones.",
    "esquema_incompatible": "El esquema de simulaciones está parcial o no corresponde al contrato.",
    "version_no_soportada": "La versión del esquema histórico no está soportada.",
    "snapshot_corrupto": "La ejecución histórica está incompleta o presenta documentos inconsistentes.",
    "snapshot_incompatible": "El formato o la implementación histórica no están soportados.",
    "base_no_disponible": "La base de datos existente no está disponible.",
}
_COMPARTIDOS = {"esquema_no_preparado", "esquema_incompatible", "version_no_soportada", "snapshot_corrupto", "snapshot_incompatible"}


class ErrorConsultaHistorialSimulaciones(ValueError):
    """Código estable y mensaje constante; sin SQL, rutas ni texto de causas."""

    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _exigir(condicion, codigo):
    if not condicion:
        raise ErrorConsultaHistorialSimulaciones(codigo)


def _id_valido(valor):
    return type(valor) is int and 0 < valor <= MAX_ENTERO_SQLITE


@contextmanager
def _lectura(ruta_base):
    try:
        ruta = registro.lectura._normalizar_ruta(ruta_base)
        with conexiones._conexion_lectura(ruta) as conexion:
            registro._verificar_esquema(conexion)
            yield conexion
    except registro.ErrorRegistroSimulacion as error:
        codigo = error.codigo if error.codigo in _COMPARTIDOS else "snapshot_corrupto"
        raise ErrorConsultaHistorialSimulaciones(codigo) from error
    except (registro.lectura.ErrorSimulacionDesdePronostico, conexiones.ErrorConsultaPronostico) as error:
        raise ErrorConsultaHistorialSimulaciones("base_no_disponible") from error


def _detalle_en_conexion(conexion, ejecucion_id):
    fila = conexion.execute(
        "SELECT id,clave_idempotencia,solicitud_json FROM ejecucion_simulacion WHERE id=?",
        (ejecucion_id,),
    ).fetchone()
    _exigir(fila is not None, "ejecucion_inexistente")
    clave = fila["clave_idempotencia"]
    _exigir(_id_valido(fila["id"]) and type(clave) is str and bool(clave.strip())
            and "\0" not in clave and type(fila["solicitud_json"]) is str, "snapshot_corrupto")
    try:
        _exigir(len(clave.encode("utf-8")) <= 128, "snapshot_corrupto")
    except UnicodeError as error:
        raise ErrorConsultaHistorialSimulaciones("snapshot_corrupto") from error
    historico = registro._recuperar_en_conexion(conexion, clave, fila["solicitud_json"])
    _exigir(historico is not None and historico["ejecucion_id"] == ejecucion_id, "snapshot_corrupto")
    # Selección explícita del contrato público, sin extenderlo con datos internos.
    return {k: historico[k] for k in ("ejecucion_id", "fecha_registro", "modalidad", "numero_escenarios", "historico")}


def consultar_ejecucion_simulacion(ruta_base, ejecucion_id) -> dict:
    """Recupera el histórico original por ID en una única lectura mode=ro/BEGIN.

    No exige conjunto activo ni consulta filas actuales de conjunto, serie,
    experimento, pronóstico o detalles. Delega documentos, hashes, versiones,
    cardinalidad e identidades a los lectores v1 publicados. No calcula motores.
    """
    _exigir(_id_valido(ejecucion_id), "ejecucion_invalida")
    with _lectura(ruta_base) as conexion:
        return _detalle_en_conexion(conexion, ejecucion_id)


def listar_ejecuciones_simulaciones(ruta_base, *, limite=20, desplazamiento=0,
                                  pronostico_id=None, conjunto_id=None, serie_id=None, modalidad=None) -> dict:
    """Página de resúmenes verificados, ordenados por ejecucion_id DESC.

    limite: int exacto 1..50; desplazamiento: int exacto 0..2**63-1. Filtros
    opcionales: IDs positivos SQLite y modalidad individual/comparacion.
    COUNT, selección de IDs y verificación completa de la página comparten una
    conexión y transacción. total cuenta cabeceras coincidentes; no se auditan
    documentos fuera de la página solicitada. Una página corrupta se rechaza
    íntegramente, sin omitir filas ni devolver resultados parciales.

    OFFSET es estable dentro de esta instantánea, no entre peticiones distintas
    cuando existen nuevas inserciones. No se impone activación del conjunto.
    """
    _exigir(type(limite) is int and 1 <= limite <= 50 and type(desplazamiento) is int
            and 0 <= desplazamiento <= MAX_ENTERO_SQLITE, "paginacion_invalida")
    filtros = (("pronostico_id", pronostico_id), ("conjunto_id", conjunto_id), ("serie_id", serie_id))
    _exigir(all(valor is None or _id_valido(valor) for _, valor in filtros), "filtro_invalido")
    _exigir(modalidad is None or (type(modalidad) is str and modalidad in ("individual", "comparacion")), "filtro_invalido")
    seleccionados = [(campo, valor) for campo, valor in filtros if valor is not None]
    if modalidad is not None:
        seleccionados.append(("modalidad", modalidad))
    # Solo se interpolan nombres constantes; todos los valores están parametrizados.
    where = " WHERE " + " AND ".join(campo + "=?" for campo, _ in seleccionados) if seleccionados else ""
    valores = tuple(valor for _, valor in seleccionados)
    with _lectura(ruta_base) as conexion:
        total = conexion.execute("SELECT COUNT(*) FROM ejecucion_simulacion" + where, valores).fetchone()[0]
        ids = conexion.execute("SELECT id FROM ejecucion_simulacion" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
                               (*valores, limite, desplazamiento)).fetchall()
        ejecuciones = []
        for fila in ids:
            _exigir(_id_valido(fila["id"]), "snapshot_corrupto")
            detalle = _detalle_en_conexion(conexion, fila["id"])
            origen = detalle["historico"]["origen"]
            ejecuciones.append({**{k: detalle[k] for k in ("ejecucion_id", "modalidad", "numero_escenarios", "fecha_registro")},
                                **{k: origen[k] for k in ("conjunto_id", "serie", "pronostico_id", "experimento_id", "tecnica", "fecha_corte")},
                                "etiquetas": [r["etiqueta"] for r in detalle["historico"]["resultados"]]})
        return {"total": total, "limite": limite, "desplazamiento": desplazamiento, "ejecuciones": ejecuciones}
