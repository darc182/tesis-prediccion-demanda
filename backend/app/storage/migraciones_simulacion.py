"""Preparación explícita S4-A4-01; no se ejecuta al importar ni guarda escenarios.

Versión propia, independiente de user_version. Se reconoce conservadoramente el
DDL publicado, sus columnas, FK e índices; no se reconstruyen tablas adulteradas.
Los JSON solo reservan el formato histórico: canonicidad, huellas, finitud y
coherencia entre documentos corresponden al futuro servicio de snapshots.
"""

from contextlib import closing
from pathlib import Path
import re
import sqlite3

from backend.app.storage.db import SCHEMA_PATH


VERSION_ESQUEMA = 1
MAX_BYTES_DOCUMENTO = 1048576
TABLAS_NUEVAS = ("version_esquema_simulaciones", "ejecucion_simulacion", "escenario_simulacion")
_MENSAJES = {
    "base_no_disponible": "La ruta o la base existente no están disponibles.",
    "esquema_original_incompatible": "El esquema original no corresponde al contrato reconocido.",
    "esquema_simulaciones_incompatible": "El esquema de simulaciones está parcial o es inconsistente.",
    "version_no_soportada": "La versión del esquema de simulaciones no está soportada.",
    "error_persistencia": "No se pudo preparar el esquema; la operación se revirtió.",
    "fallo_integridad": "La base presenta referencias o estructuras inconsistentes.",
}


class ErrorPreparacionSimulaciones(ValueError):
    """Código estable y mensaje constante, sin detalles del sistema o de SQLite."""

    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _texto(nombre, limite):
    return (f"{nombre} TEXT NOT NULL CHECK (typeof({nombre})='text' "
            f"AND length(trim({nombre}))>0 AND instr({nombre},char(0))=0 "
            f"AND length(CAST({nombre} AS BLOB))<={limite})")


def _documento(nombre, tipo="object"):
    return (f"{nombre} TEXT NOT NULL CHECK (typeof({nombre})='text' "
            f"AND length(CAST({nombre} AS BLOB))<={MAX_BYTES_DOCUMENTO} "
            f"AND CASE WHEN json_valid({nombre}) THEN json_type({nombre})='{tipo}' ELSE 0 END)")


def _huella(nombre):
    return (f"{nombre} TEXT NOT NULL CHECK (typeof({nombre})='text' "
            f"AND length(CAST({nombre} AS BLOB))=64 AND {nombre} NOT GLOB '*[^0-9a-f]*')")


def _positivo(nombre):
    return f"{nombre} INTEGER NOT NULL CHECK (typeof({nombre})='integer' AND {nombre}>0)"


_DDL = (
    """CREATE TABLE version_esquema_simulaciones (
        id INTEGER PRIMARY KEY CHECK (typeof(id)='integer' AND id=1),
        version INTEGER NOT NULL CHECK (typeof(version)='integer' AND version>0)
    )""",
    f"""CREATE TABLE ejecucion_simulacion (
        id INTEGER PRIMARY KEY CHECK (typeof(id)='integer' AND id>0),
        {_texto('clave_idempotencia', 128)} UNIQUE,
        modalidad TEXT NOT NULL CHECK (typeof(modalidad)='text' AND modalidad IN ('individual','comparacion')),
        numero_escenarios INTEGER NOT NULL CHECK (typeof(numero_escenarios)='integer'),
        {_positivo('conjunto_id')} REFERENCES conjunto_datos(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
        {_positivo('serie_id')} REFERENCES serie_temporal(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
        {_positivo('pronostico_id')} REFERENCES pronostico(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
        fecha_registro TEXT NOT NULL CHECK (typeof(fecha_registro)='text'
            AND length(CAST(fecha_registro AS BLOB))=20
            AND fecha_registro GLOB '????-??-??T??:??:??Z'
            AND strftime('%Y-%m-%dT%H:%M:%SZ',fecha_registro,'+0 seconds') IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%SZ',fecha_registro,'+0 seconds')=fecha_registro),
        {_positivo('version_esquema')},
        {_positivo('version_documento')},
        {_texto('version_regla', 128)},
        {_texto('identificador_implementacion', 128)},
        {_texto('version_politica_numerica', 128)},
        {_documento('solicitud_json')}, {_huella('sha256_solicitud')},
        {_documento('origen_json')}, {_huella('sha256_origen')},
        {_documento('contexto_json')}, {_huella('sha256_contexto')},
        {_documento('advertencias_json', 'array')}, {_huella('sha256_advertencias')},
        CHECK ((modalidad='individual' AND numero_escenarios=1)
            OR (modalidad='comparacion' AND numero_escenarios IN (2,3)))
    )""",
    f"""CREATE TABLE escenario_simulacion (
        {_positivo('ejecucion_id')} REFERENCES ejecucion_simulacion(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
        etiqueta TEXT NOT NULL CHECK (typeof(etiqueta)='text' AND etiqueta IN ('A','B','C')),
        posicion INTEGER NOT NULL CHECK (typeof(posicion)='integer' AND posicion BETWEEN 1 AND 3),
        {_documento('resultado_json')}, {_huella('sha256_resultado')},
        PRIMARY KEY (ejecucion_id,etiqueta),
        UNIQUE (ejecucion_id,posicion),
        CHECK ((etiqueta='A' AND posicion=1) OR (etiqueta='B' AND posicion=2)
            OR (etiqueta='C' AND posicion=3))
    ) WITHOUT ROWID""",
)


def _tokens(sql):
    # Conserva literalmente cadenas e identificadores entre comillas. Solo
    # ignora comentarios, espacios y mayúsculas de tokens no entrecomillados.
    patron = r"--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\[[^]]*\]|`[^`]*`|[\w]+|[^\s]"
    return tuple(t if t[0] in "'\"[`" else t.lower()
                 for t in re.findall(patron, sql)
                 if not t.startswith(("--", "/*")))


def _tablas(conexion):
    return {n: (tipo, sql) for n, tipo, sql in conexion.execute(
        "SELECT name,type,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}


def _firma(conexion, tabla):
    # Los nombres proceden exclusivamente del contrato, nunca de entradas SQL.
    columnas = tuple(conexion.execute(f'PRAGMA table_xinfo("{tabla}")'))
    claves = tuple(sorted(conexion.execute(f'PRAGMA foreign_key_list("{tabla}")')))
    indices = []
    for _, nombre, unico, origen, parcial in conexion.execute(f'PRAGMA index_list("{tabla}")'):
        citado = nombre.replace('"', '""')
        campos = tuple(conexion.execute(f'PRAGMA index_xinfo("{citado}")'))
        indices.append((unico, origen, parcial, campos))
    return columnas, claves, tuple(sorted(indices))


def _verificar(conexion, referencia, nombres, codigo):
    actual, esperado = _tablas(conexion), _tablas(referencia)
    for nombre in nombres:
        if (nombre not in actual or actual[nombre][0] != "table"
                or _tokens(actual[nombre][1]) != _tokens(esperado[nombre][1])
                or _firma(conexion, nombre) != _firma(referencia, nombre)):
            raise ErrorPreparacionSimulaciones(codigo)


def preparar_esquema_simulaciones(ruta_base) -> dict:
    """Prepara versión 1, o verifica sin DML una preparación ya reconocida.

    Solo mode=rw para la base explícita; BEGIN IMMEDIATE serializa inspección y
    DDL. La referencia se construye en memoria con el esquema publicado, sin
    tocar archivos de datos. No se usa executescript en la base de destino
    porque podría confirmar anticipadamente una transacción pendiente.

    Se exige el contrato de las nueve tablas originales, incluidos sus índices;
    DDL desconocido, aunque pudiera ser equivalente, requiere revisión explícita.
    Se toleran objetos ajenos que no alteren esas tablas. No se toleran triggers
    sobre las tres nuevas tablas. No cambia journal_mode ni user_version.

    Las tres FK de procedencia son independientes: no demuestran pertenencia
    conjunta. Tampoco SQL garantiza cardinalidad de hijos, canonicidad, huellas
    correctas o finitud dentro de JSON. Esas garantías serán del registro futuro.
    Las FK restrictivas impiden borrar físicamente el origen referenciado.
    """
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base).strip() or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        uri = Path(ruta_base).resolve().as_uri() + "?mode=rw"
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise ErrorPreparacionSimulaciones("base_no_disponible") from error
    try:
        with closing(sqlite3.connect(":memory:")) as referencia:
            referencia.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            originales = tuple(_tablas(referencia))
            for ddl in _DDL:
                referencia.execute(ddl)
            try:
                conexion = sqlite3.connect(uri, uri=True)
            except (sqlite3.Error, OSError) as error:
                raise ErrorPreparacionSimulaciones("base_no_disponible") from error
            with closing(conexion):
                escribiendo = False
                try:
                    conexion.execute("PRAGMA foreign_keys=ON")
                    if conexion.execute("PRAGMA foreign_keys").fetchone() != (1,):
                        raise ErrorPreparacionSimulaciones("fallo_integridad")
                    conexion.execute("BEGIN IMMEDIATE")
                    _verificar(conexion, referencia, originales, "esquema_original_incompatible")
                    presentes = set(TABLAS_NUEVAS).intersection(_tablas(conexion))
                    if presentes and presentes != set(TABLAS_NUEVAS):
                        raise ErrorPreparacionSimulaciones("esquema_simulaciones_incompatible")
                    if presentes:
                        _verificar(conexion, referencia, TABLAS_NUEVAS, "esquema_simulaciones_incompatible")
                        marcador = conexion.execute("SELECT id,version FROM version_esquema_simulaciones").fetchall()
                        if (len(marcador) != 1 or marcador[0][0] != 1
                                or type(marcador[0][1]) is not int or marcador[0][1] <= 0):
                            raise ErrorPreparacionSimulaciones("esquema_simulaciones_incompatible")
                        if marcador[0][1] != VERSION_ESQUEMA:
                            raise ErrorPreparacionSimulaciones("version_no_soportada")
                    # Los identificadores ASCII de SQLite no distinguen caja;
                    # sqlite_master conserva la escritura usada al crear el trigger.
                    if conexion.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name COLLATE NOCASE IN (?,?,?)",
                                        TABLAS_NUEVAS).fetchone():
                        raise ErrorPreparacionSimulaciones("esquema_simulaciones_incompatible")
                    if conexion.execute("PRAGMA foreign_key_check").fetchone():
                        raise ErrorPreparacionSimulaciones("fallo_integridad")
                    if presentes:
                        conexion.rollback()
                        return {"version_esquema": VERSION_ESQUEMA, "estado": "verificado", "hubo_escritura": False}
                    escribiendo = True
                    for ddl in _DDL:
                        conexion.execute(ddl)
                    conexion.execute("INSERT INTO version_esquema_simulaciones (id,version) VALUES (1,?)", (VERSION_ESQUEMA,))
                    _verificar(conexion, referencia, TABLAS_NUEVAS, "fallo_integridad")
                    if conexion.execute("SELECT id,version FROM version_esquema_simulaciones").fetchall() != [(1, VERSION_ESQUEMA)]:
                        raise ErrorPreparacionSimulaciones("fallo_integridad")
                    if conexion.execute("PRAGMA foreign_key_check").fetchone():
                        raise ErrorPreparacionSimulaciones("fallo_integridad")
                    conexion.commit()
                    return {"version_esquema": VERSION_ESQUEMA, "estado": "preparado", "hubo_escritura": True}
                except sqlite3.Error as error:
                    conexion.rollback()
                    codigo = "error_persistencia" if escribiendo else "base_no_disponible"
                    raise ErrorPreparacionSimulaciones(codigo) from error
                except Exception:
                    conexion.rollback()
                    raise
    except OSError as error:
        raise ErrorPreparacionSimulaciones("base_no_disponible") from error
