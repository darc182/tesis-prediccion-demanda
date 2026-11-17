"""Registro explícito S4-A4-03: snapshot original e idempotencia de solicitud.

No prepara esquemas ni ejecuta DDL. Dependencias privadas deliberadas: lectores
S4-A2, _seleccion del comparador y _numero del motor solo para solicitudes nuevas,
_parametros_v1/_parametros/_igual del snapshot y _DDL/_tokens del esquema.
Revisar al cambiar contratos; la identidad histórica permanece bajo v1.
"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
from math import isfinite
import sqlite3

from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import simulacion_desde_pronostico as lectura
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.services import trayectoria_simulacion as trayectoria
from backend.app.storage import migraciones_simulacion as esquema


_MENSAJES = {
    **lectura._MENSAJES,
    "solicitud_invalida": "La solicitud no tiene la estructura admitida.",
    "clave_idempotencia_invalida": "La clave debe ser texto no vacío de hasta 128 bytes UTF-8, sin nulos.",
    "conflicto_idempotencia": "La clave ya corresponde a una solicitud diferente.",
    "escenarios_invalidos": "Se requieren escenarios completos A/B o A/B/C en ese orden.",
    "resultados_incompatibles": "Los resultados no corresponden a una única simulación coherente.",
    "snapshot_incompatible": "El formato o la implementación histórica no están soportados.",
    "snapshot_corrupto": "La ejecución histórica está incompleta o presenta datos inconsistentes.",
    "esquema_no_preparado": "La base necesita preparación explícita del esquema de simulaciones.",
    "esquema_incompatible": "La estructura de simulaciones está parcial o no corresponde al contrato.",
    "version_no_soportada": "La versión del esquema no está soportada.",
    "origen_modificado": "El origen cambió durante el cálculo; no se guardó la ejecución.",
    "error_persistencia": "No se pudo guardar una ejecución completa; la operación se revirtió.",
}
_DOCUMENTOS = ("solicitud", "origen", "contexto", "advertencias")
_PAQUETE = ("version_documento", "version_regla", "identificador_implementacion", "version_politica_numerica",
            "modalidad", "numero_escenarios", "conjunto_id", "serie_id", "pronostico_id") + tuple(
                campo for n in _DOCUMENTOS for campo in (n + "_json", "sha256_" + n))
_CABECERA = ("id", "clave_idempotencia", "fecha_registro", "version_esquema") + _PAQUETE
_HIJOS = ("ejecucion_id", "etiqueta", "posicion", "resultado_json", "sha256_resultado")
_BASE = ("conjunto_datos", "serie_temporal", "experimento", "pronostico", "detalle_pronostico",
         "venta_diaria", "evaluacion", "simulacion", "perfil_global_conjunto")


class ErrorRegistroSimulacion(ValueError):
    """Código público y mensaje constante; no publica texto de las causas."""

    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _exigir(condicion, codigo):
    if not condicion:
        raise ErrorRegistroSimulacion(codigo)


def _clave(valor):
    _exigir(type(valor) is str, "clave_idempotencia_invalida")
    try:
        datos = valor.encode("utf-8")
    except UnicodeError as error:
        raise ErrorRegistroSimulacion("clave_idempotencia_invalida") from error
    _exigir(bool(valor.strip()) and "\0" not in valor and len(datos) <= 128, "clave_idempotencia_invalida")


def _solicitud(pronostico_id, modalidad, escenarios):
    """Reconoce exclusivamente solicitudes v1; no aplica políticas vigentes."""
    _exigir(type(pronostico_id) is int and 0 < pronostico_id <= 2**63-1, "pronostico_invalido")
    _exigir(type(modalidad) is str and modalidad in ("individual", "comparacion"), "solicitud_invalida")
    _exigir(type(escenarios) in (list, tuple)
            and len(escenarios) in ((1,) if modalidad == "individual" else (2, 3)), "escenarios_invalidos")
    campos = {"etiqueta", "inventario_disponible", "tiempo_abastecimiento", "reserva"}
    for etiqueta, escenario in zip("ABC", escenarios):
        _exigir(type(escenario) is dict and set(escenario) == campos
                and type(escenario["etiqueta"]) is str and escenario["etiqueta"] == etiqueta, "escenarios_invalidos")
        for nombre in ("inventario_disponible", "reserva"):
            valor = escenario[nombre]
            _exigir(type(valor) in (int, float), "parametros_invalidos")
            # Conserva el código público de precisión del registro anterior,
            # sin llamar motor._numero ni alterar el entero original.
            if type(valor) is int and valor >= 0:
                try:
                    numero = float(valor)
                except OverflowError as error:
                    raise ErrorRegistroSimulacion("parametros_invalidos") from error
                _exigir(isfinite(numero), "parametros_invalidos")
                _exigir(numero == valor, "precision_no_representable")
        try:
            snapshot._parametros_v1({k: escenario[k] for k in ("inventario_disponible", "tiempo_abastecimiento", "reserva")})
        except snapshot.ErrorSnapshotSimulacion as error:
            raise ErrorRegistroSimulacion("parametros_invalidos") from error
    texto = snapshot.serializar_documento({"pronostico_id": pronostico_id, "modalidad": modalidad, "escenarios": list(escenarios)})
    # Copia de primitivas, sin invocar protocolos de objetos externos ni mutar
    # las entradas. Conserva int/float y -0.0 en la identidad de la solicitud.
    return texto, snapshot.leer_documento(texto, snapshot.calcular_sha256(texto), version_documento=1)


def _validar_vigente(pronostico_id, solicitud):
    """Solo tras no encontrar la clave: conserva las comprobaciones actuales."""
    lectura._validar_identificador(pronostico_id)
    if solicitud["modalidad"] == "comparacion":
        comparacion._seleccion(solicitud["escenarios"])
    for escenario in solicitud["escenarios"]:
        for nombre in ("inventario_disponible", "reserva"):
            motor._numero(escenario[nombre])
        try:
            snapshot._parametros({k: escenario[k] for k in ("inventario_disponible", "tiempo_abastecimiento", "reserva")})
        except snapshot.ErrorSnapshotSimulacion as error:
            raise ErrorRegistroSimulacion("parametros_invalidos") from error


@contextmanager
def _conexion(ruta, escritura=False):
    uri = ruta.as_uri() + ("?mode=rw" if escritura else "?mode=ro")
    with closing(sqlite3.connect(uri, uri=True)) as c:
        c.row_factory = sqlite3.Row
        try:
            if escritura:
                c.execute("PRAGMA foreign_keys=ON")
                _exigir(c.execute("PRAGMA foreign_keys").fetchone()[0] == 1, "error_persistencia")
            c.execute("BEGIN IMMEDIATE" if escritura else "BEGIN")
            yield c
        finally:
            if c.in_transaction:
                c.rollback()


def _verificar_esquema(c):
    """Introspección sin DDL, sin leer filas de ventas o JSON privados.

    Usa el DDL reconocido como contrato textual y contrasta columnas, FK e
    índices reales. No llama al preparador ni construye una base de referencia.
    La validación pública del origen permanece en S4-A2.
    """
    catalogo = esquema._tablas(c)
    presentes = set(esquema.TABLAS_NUEVAS) & set(catalogo)
    _exigir(bool(presentes), "esquema_no_preparado")
    _exigir(presentes == set(esquema.TABLAS_NUEVAS), "esquema_incompatible")
    _exigir(all(n in catalogo and catalogo[n][0] == "table" for n in _BASE), "esquema_incompatible")
    necesarias = {
        "conjunto_datos": {"id", "activo"},
        "serie_temporal": {"id", "conjunto_id", "store_nbr", "family"},
        "experimento": {"id", "serie_id", "tecnica", "inicio_entrenamiento", "fin_entrenamiento"},
        "pronostico": {"id", "experimento_id", "fecha_corte", "tipo", "horizonte", "fecha_generacion", "duracion_segundos"},
        "detalle_pronostico": {"pronostico_id", "paso", "fecha_objetivo", "venta_estimada"},
    }
    for tabla, campos in necesarias.items():
        _exigir(campos <= {f[1] for f in c.execute(f'PRAGMA table_xinfo("{tabla}")')}, "esquema_incompatible")
    for nombre, ddl in zip(esquema.TABLAS_NUEVAS, esquema._DDL):
        _exigir(catalogo[nombre][0] == "table" and esquema._tokens(catalogo[nombre][1]) == esquema._tokens(ddl), "esquema_incompatible")
    marcador = [tuple(f) for f in c.execute("SELECT id,version FROM version_esquema_simulaciones")]
    _exigir(len(marcador) == 1 and marcador[0][0] == 1 and type(marcador[0][1]) is int and marcador[0][1] > 0, "esquema_incompatible")
    _exigir(marcador[0][1] == 1, "version_no_soportada")
    orden_cabecera = ("id", "clave_idempotencia", "modalidad", "numero_escenarios", "conjunto_id", "serie_id", "pronostico_id",
                      "fecha_registro", "version_esquema", "version_documento", "version_regla", "identificador_implementacion",
                      "version_politica_numerica") + _PAQUETE[9:]
    enteros = {"id", "version", "numero_escenarios", "conjunto_id", "serie_id", "pronostico_id", "version_esquema", "version_documento",
               "ejecucion_id", "posicion"}
    for tabla, columnas, pk in ((esquema.TABLAS_NUEVAS[0], ("id", "version"), {"id": 1}),
                               (esquema.TABLAS_NUEVAS[1], orden_cabecera, {"id": 1}),
                               (esquema.TABLAS_NUEVAS[2], _HIJOS, {"ejecucion_id": 1, "etiqueta": 2})):
        esperado = [(i, n, "INTEGER" if n in enteros else "TEXT", 0 if n == "id" else 1, None, pk.get(n, 0), 0)
                    for i, n in enumerate(columnas)]
        _exigir([tuple(f) for f in c.execute(f'PRAGMA table_xinfo("{tabla}")')] == esperado, "esquema_incompatible")
        fks = {tuple(f)[2:] for f in c.execute(f'PRAGMA foreign_key_list("{tabla}")')}
        padres = ({("ejecucion_simulacion", "ejecucion_id", "id", "RESTRICT", "RESTRICT", "NONE")} if tabla == "escenario_simulacion"
                  else {(p, k, "id", "RESTRICT", "RESTRICT", "NONE") for p, k in
                        (("conjunto_datos", "conjunto_id"), ("serie_temporal", "serie_id"), ("pronostico", "pronostico_id"))}
                  if tabla == "ejecucion_simulacion" else set())
        _exigir(fks == padres, "esquema_incompatible")
        indices = set()
        for fila in c.execute(f'PRAGMA index_list("{tabla}")'):
            citado = fila[1].replace('"', '""')
            campos = tuple((f[2], f[3], f[4]) for f in c.execute(f'PRAGMA index_xinfo("{citado}")') if f[5])
            indices.add((fila[2], fila[3], fila[4], campos))
        unicos = ({(1, "pk", 0, (("ejecucion_id", 0, "BINARY"), ("etiqueta", 0, "BINARY"))),
                   (1, "u", 0, (("ejecucion_id", 0, "BINARY"), ("posicion", 0, "BINARY")))} if tabla == "escenario_simulacion"
                  else {(1, "u", 0, (("clave_idempotencia", 0, "BINARY"),))} if tabla == "ejecucion_simulacion" else set())
        _exigir(indices == unicos, "esquema_incompatible")
    _exigir(c.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name COLLATE NOCASE IN (?,?,?)",
                       esquema.TABLAS_NUEVAS).fetchone() is None, "esquema_incompatible")


def _recuperar_en_conexion(c, clave, solicitud_json):
    """Lector mínimo reutilizable en A4-04; no consulta origen ni motores."""
    fila = c.execute(f"SELECT {','.join(_CABECERA)} FROM ejecucion_simulacion WHERE clave_idempotencia=?", (clave,)).fetchone()
    if fila is None:
        return None
    _exigir(type(fila["id"]) is int and fila["id"] > 0 and fila["version_esquema"] == 1, "snapshot_corrupto")
    fecha = fila["fecha_registro"]
    try:
        _exigir(type(fecha) is str and len(fecha) == 20 and fecha.endswith("Z"), "snapshot_corrupto")
        normalizada = datetime.fromisoformat(fecha[:-1] + "+00:00").isoformat(timespec="seconds").replace("+00:00", "Z")
        _exigir(normalizada == fecha, "snapshot_corrupto")
    except ValueError as error:
        if isinstance(error, ErrorRegistroSimulacion):
            raise
        raise ErrorRegistroSimulacion("snapshot_corrupto") from error
    paquete = {k: fila[k] for k in _PAQUETE}
    paquete["escenarios"] = [dict(f) for f in c.execute(
        "SELECT etiqueta,posicion,resultado_json,sha256_resultado FROM escenario_simulacion WHERE ejecucion_id=? ORDER BY posicion",
        (fila["id"],))]
    try:
        historico = snapshot.verificar_paquete(paquete)
    except snapshot.ErrorSnapshotSimulacion as error:
        raise ErrorRegistroSimulacion("snapshot_incompatible" if error.codigo == "version_no_soportada" else "snapshot_corrupto") from error
    _exigir(fila["solicitud_json"] == solicitud_json, "conflicto_idempotencia")
    return {"ejecucion_id": fila["id"], "modalidad": fila["modalidad"], "numero_escenarios": fila["numero_escenarios"],
            "fecha_registro": fecha, "historico": historico}


def _respuesta(recuperado, creado):
    return {"estado_registro": "creado" if creado else "reutilizado", "hubo_escritura": creado, **recuperado}


def _registrar(ruta_base, pronostico_id, escenarios, clave_idempotencia, modalidad):
    escribiendo = False
    try:
        _clave(clave_idempotencia)
        texto, solicitud = _solicitud(pronostico_id, modalidad, escenarios)
        ruta = lectura._normalizar_ruta(ruta_base)
        with _conexion(ruta) as c:
            _verificar_esquema(c)
            existente = _recuperar_en_conexion(c, clave_idempotencia, texto)
            if existente is not None:
                return _respuesta(existente, False)
            _validar_vigente(pronostico_id, solicitud)
            origen = lectura._leer_pronostico_simulable_en_conexion(c, pronostico_id)
        conjunto, serie, cabecera, detalles = origen
        seleccion = solicitud["escenarios"]
        if modalidad == "individual":
            e = seleccion[0]
            resultado = trayectoria.calcular_trayectoria(cabecera, detalles, e["inventario_disponible"], e["tiempo_abastecimiento"], e["reserva"])
        else:
            resultado = comparacion.comparar_escenarios(cabecera, detalles, seleccion)
        paquete = snapshot.preparar_snapshot(cabecera, detalles, {"conjunto_id": conjunto, "serie": serie}, seleccion, resultado, modalidad=modalidad)
        historico_preparado = snapshot.verificar_paquete(paquete)
        _exigir(paquete["solicitud_json"] == texto, "resultados_incompatibles")
        with _conexion(ruta, escritura=True) as c:
            _verificar_esquema(c)
            existente = _recuperar_en_conexion(c, clave_idempotencia, texto)
            if existente is not None:
                return _respuesta(existente, False)
            try:
                actual = lectura._leer_pronostico_simulable_en_conexion(c, pronostico_id)
            except (lectura.ErrorSimulacionDesdePronostico, consulta.ErrorConsultaPronostico) as error:
                if error.codigo == "base_no_disponible":
                    raise
                raise ErrorRegistroSimulacion("origen_modificado") from error
            _exigir(snapshot._igual(list(origen), list(actual)), "origen_modificado")
            escribiendo = True
            cambios = c.total_changes
            valores = {k: paquete[k] for k in _PAQUETE}
            valores.update(clave_idempotencia=clave_idempotencia, version_esquema=1,
                           fecha_registro=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
            cursor = c.execute(f"INSERT INTO ejecucion_simulacion ({','.join(valores)}) VALUES ({','.join('?' for _ in valores)})", tuple(valores.values()))
            identificador = cursor.lastrowid
            _exigir(cursor.rowcount == 1, "error_persistencia")
            for fila in paquete["escenarios"]:
                cursor = c.execute("INSERT INTO escenario_simulacion (ejecucion_id,etiqueta,posicion,resultado_json,sha256_resultado) VALUES (?,?,?,?,?)",
                                   (identificador, fila["etiqueta"], fila["posicion"], fila["resultado_json"], fila["sha256_resultado"]))
                _exigir(cursor.rowcount == 1, "error_persistencia")
            _exigir(c.total_changes - cambios == 1 + len(seleccion), "error_persistencia")
            try:
                guardado = _recuperar_en_conexion(c, clave_idempotencia, texto)
            except ErrorRegistroSimulacion as error:
                raise ErrorRegistroSimulacion("error_persistencia") from error
            _exigir(guardado is not None and guardado["ejecucion_id"] == identificador
                    and snapshot._igual(guardado["historico"], historico_preparado), "error_persistencia")
            respuesta = _respuesta(guardado, True)
            c.commit()
            return respuesta
    except (lectura.ErrorSimulacionDesdePronostico, consulta.ErrorConsultaPronostico) as error:
        codigo = error.codigo if error.codigo in _MENSAJES else "pronostico_corrupto"
        raise ErrorRegistroSimulacion(codigo) from error
    except (motor.ErrorSimulacionReposicion, trayectoria.ErrorTrayectoriaSimulacion) as error:
        codigo = "parametros_invalidos" if error.codigo in ("valor_invalido", "plazo_invalido") else error.codigo
        raise ErrorRegistroSimulacion(codigo if codigo in _MENSAJES else "pronostico_corrupto") from error
    except comparacion.ErrorComparacionSimulaciones as error:
        raise ErrorRegistroSimulacion(error.codigo) from error
    except snapshot.ErrorSnapshotSimulacion as error:
        raise ErrorRegistroSimulacion("snapshot_incompatible" if error.codigo == "version_no_soportada" else "snapshot_corrupto") from error
    except (sqlite3.Error, OSError) as error:
        raise ErrorRegistroSimulacion("error_persistencia" if escribiendo else "base_no_disponible") from error


def registrar_simulacion(ruta_base, pronostico_id, inventario_disponible, tiempo_abastecimiento, reserva, *, clave_idempotencia) -> dict:
    """Guarda una trayectoria individual A o recupera la misma solicitud histórica.

    Solo bases existentes y explícitamente preparadas. Dos conexiones sucesivas:
    RO/BEGIN para clave y origen, cálculo sin bloqueo, RW/BEGIN IMMEDIATE para
    clave, revalidación e INSERT atómico. Un reintento no lee el origen actual.
    """
    escenarios = [{"etiqueta": "A", "inventario_disponible": inventario_disponible,
                   "tiempo_abastecimiento": tiempo_abastecimiento, "reserva": reserva}]
    return _registrar(ruta_base, pronostico_id, escenarios, clave_idempotencia, "individual")


def registrar_comparacion_simulaciones(ruta_base, pronostico_id, escenarios, *, clave_idempotencia) -> dict:
    """Guarda A/B o A/B/C completos, sin rankings ni órdenes operativas reales.

    La clave distingue acciones deliberadas; misma clave y solicitud exacta
    reutiliza, otra solicitud entra en conflicto. No hay sobrescrituras, DDL,
    migraciones, reintentos automáticos ni cálculos dentro del bloqueo escritor.
    """
    return _registrar(ruta_base, pronostico_id, escenarios, clave_idempotencia, "comparacion")
