"""Contrato de persistencia G1 v1, sin abrir CSV, conexiones ni transacciones.

Preparar no modifica el reporte. Guardar y recuperar reciben una conexión con
transacción activa; el llamador decide BEGIN, commit y rollback. La inserción
requiere foreign_keys=ON. No se ejecuta G1 ni se selecciona un conjunto activo.
El checksum detecta corrupción accidental, no es una firma de autenticidad.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from math import fsum, isfinite
import re
import sqlite3


VERSION_PERFIL = 1
MAX_BYTES_REPORTE = 16 * 1024 * 1024
MAX_ENTERO_SEGURO = 2**53 - 1
# Límites del contrato G1 publicado, sin importar ni ejecutar el analizador.
_COLUMNAS = ["date", "store_nbr", "family", "sales", "onpromotion"]
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FECHA = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_MENSAJES = {
    "perfil_invalido": "El reporte global o sus metadatos no son válidos.",
    "perfil_demasiado_grande": "El reporte global excede el tamaño permitido.",
    "perfil_no_disponible": "El conjunto no dispone de un perfil global.",
    "perfil_corrupto": "El perfil global almacenado presenta inconsistencias.",
    "version_no_soportada": "La versión del perfil global no está soportada.",
    "almacenamiento_no_disponible": "El almacenamiento de perfiles globales no está disponible.",
}


class ErrorPerfilGlobal(ValueError):
    """Código estable y mensaje público fijo, sin contenido ni errores SQLite."""

    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


@dataclass(frozen=True)
class PerfilGlobalPreparado:
    version_perfil: int
    reporte_json: str
    sha256_reporte: str


def _exigir(condicion):
    if not condicion:
        raise ErrorPerfilGlobal("perfil_invalido")


def _objeto(valor, campos):
    _exigir(type(valor) is dict and set(valor) == set(campos.split()))


def _entero(valor, minimo=0, maximo=MAX_ENTERO_SEGURO):
    _exigir(type(valor) is int and minimo <= valor <= maximo)
    return valor


def _numero(valor):
    _exigir(type(valor) in (int, float))
    if type(valor) is int:
        _entero(valor)
    _exigir(isfinite(valor) and valor >= 0)
    return valor


def _texto(valor, limite=None):
    _exigir(type(valor) is str and bool(valor.strip()) and (limite is None or len(valor) <= limite))
    return valor


def _nombre(valor):
    _texto(valor, 255)
    # Un nombre base puede contener espacios iniciales o puntos internos, como en la carga.
    # No se extrae un basename de una ruta recibida: los separadores siguen prohibidos.
    _exigir(valor.lower().endswith(".csv") and bool(valor[:-4].strip(" ."))
            and all(c.isalnum() or c in "._- ()" for c in valor))
    base = valor.split(".", 1)[0].strip(" .").upper()
    _exigir(base not in {"CON", "PRN", "AUX", "NUL"}
            and base not in {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)})
    return valor


def _fecha(valor):
    _exigir(type(valor) is str and _FECHA.fullmatch(valor) is not None)
    return date.fromisoformat(valor)


def _lista(valor, limite, minimo=0):
    _exigir(type(valor) is list and minimo <= len(valor) <= limite)
    return valor


def _suma_coherente(valores, total):
    """Validación con atol=1e-15 y rtol=1e-12, escalada para no desbordar."""
    escala = max(abs(total), *(abs(v) for v in valores))
    if escala == 0:
        return True
    suma = fsum(v / escala for v in valores)
    esperado = total / escala
    exceso = abs(suma - esperado) - 1e-12 * max(abs(suma), abs(esperado))
    return exceso <= 0 or (exceso * escala <= 1e-15 if escala <= 1e-15 else exceso <= 1e-15 / escala)


def _porcentaje(valor, parte, total):
    _numero(valor)
    _exigir(valor <= 100 and _suma_coherente([parte / total], valor / 100))


def _estadisticas(reporte, n):
    v = reporte["ventas"]
    _objeto(v, "suma promedio_por_registro minimo maximo registros_cero porcentaje_cero "
            "registros_fraccionarios porcentaje_fraccionarios registros_positivos")
    for clave in ("suma", "promedio_por_registro", "minimo", "maximo"):
        _numero(v[clave])
    ceros = _entero(v["registros_cero"], maximo=n)
    positivos = _entero(v["registros_positivos"], maximo=n)
    _entero(v["registros_fraccionarios"], maximo=positivos)
    _exigir(ceros + positivos == n and v["minimo"] <= v["maximo"] <= v["suma"])
    _exigir((v["minimo"] == 0) == (ceros > 0) and (v["maximo"] > 0) == (positivos > 0))
    _exigir(_suma_coherente([v["suma"] / n], v["promedio_por_registro"]))
    media = v["promedio_por_registro"]
    _exigir(media >= v["minimo"] or _suma_coherente([v["minimo"]], media))
    _exigir(media <= v["maximo"] or _suma_coherente([v["maximo"]], media))
    _porcentaje(v["porcentaje_cero"], ceros, n)
    _porcentaje(v["porcentaje_fraccionarios"], v["registros_fraccionarios"], n)

    p = reporte["promociones"]
    _objeto(p, "registros_cero registros_positivos porcentaje_con_promocion suma_conteos_onpromotion "
            "minimo maximo promedio_por_registro")
    for clave in ("registros_cero", "registros_positivos", "suma_conteos_onpromotion", "minimo", "maximo"):
        _entero(p[clave])
    _exigir(p["registros_cero"] + p["registros_positivos"] == n
            and p["registros_positivos"] <= p["suma_conteos_onpromotion"]
            and p["minimo"] * n <= p["suma_conteos_onpromotion"] <= p["maximo"] * n
            and (p["minimo"] == 0) == (p["registros_cero"] > 0)
            and (p["maximo"] > 0) == (p["registros_positivos"] > 0))
    _numero(p["promedio_por_registro"])
    _exigir(_suma_coherente([p["suma_conteos_onpromotion"] / n], p["promedio_por_registro"]))
    _porcentaje(p["porcentaje_con_promocion"], p["registros_positivos"], n)


def _calendario(t, n):
    _objeto(t, "primera_fecha ultima_fecha fechas_distintas dias_calendario_inclusivos "
            "fechas_sin_registro_global filas_por_anio filas_por_fecha")
    primera, ultima = _fecha(t["primera_fecha"]), _fecha(t["ultima_fecha"])
    dias = _entero(t["dias_calendario_inclusivos"], 1, 36600)
    distintas = _entero(t["fechas_distintas"], 1, min(dias, n))
    _exigir((ultima - primera).days + 1 == dias)
    fechas, anios = [], {}
    for item in _lista(t["filas_por_fecha"], 36600, 1):
        _objeto(item, "fecha filas")
        fecha = _fecha(item["fecha"])
        filas = _entero(item["filas"], 1, n)
        _exigir(primera <= fecha <= ultima and (not fechas or fecha > fechas[-1]))
        fechas.append(fecha)
        anios[fecha.year] = anios.get(fecha.year, 0) + filas
    _exigir(len(fechas) == distintas and fechas[0] == primera and fechas[-1] == ultima
            and sum(anios.values()) == n)
    ausentes = [_fecha(f) for f in _lista(t["fechas_sin_registro_global"], 36600)]
    _exigir(all(primera <= f <= ultima and (i == 0 or f > ausentes[i - 1]) for i, f in enumerate(ausentes)))
    # G1 sí enumera todas las fechas observadas: partición exacta del calendario.
    _exigir(len(ausentes) + distintas == dias and not set(ausentes).intersection(fechas))
    declarados = {}
    anterior = 0
    max_anios = min(ultima.year - primera.year + 1, distintas)
    for item in _lista(t["filas_por_anio"], max_anios, 1):
        _objeto(item, "anio filas")
        anio = _entero(item["anio"], anterior + 1, 9999)
        declarados[anio] = _entero(item["filas"], 1, n)
        anterior = anio
    _exigir(declarados == anios)


def _distribuciones(d, volumen, ventas):
    _objeto(d, "por_tienda por_familia combinaciones_mayor_volumen criterio_mayores")
    _texto(d["criterio_mayores"], 4096)
    n, total = volumen["filas_validas"], ventas["suma"]
    tiendas, familias, montos = {}, {}, {}
    for item in _lista(d["por_tienda"], 1000, 1):
        _objeto(item, "store_nbr filas")
        tienda = _entero(item["store_nbr"], 1)
        _exigir(not tiendas or tienda > next(reversed(tiendas)))
        tiendas[tienda] = _entero(item["filas"], 1, n)
    for item in _lista(d["por_familia"], 1000, 1):
        _objeto(item, "family filas ventas")
        familia = _texto(item["family"], 512)
        _exigir(not familias or familia > next(reversed(familias)))
        familias[familia] = _entero(item["filas"], 1, n)
        montos[familia] = _numero(item["ventas"])
    _exigir(len(tiendas) == volumen["tiendas_distintas"] and len(familias) == volumen["familias_distintas"]
            and sum(tiendas.values()) == sum(familias.values()) == n
            and _suma_coherente(list(montos.values()), total))
    combinaciones = volumen["combinaciones_observadas"]
    _exigir(max(len(tiendas), len(familias)) <= combinaciones <= min(n, len(tiendas) * len(familias)))
    mayores = _lista(d["combinaciones_mayor_volumen"], 10, 1)
    _exigir(len(mayores) == min(10, combinaciones))
    claves, orden, filas_t, filas_f, ventas_f = set(), [], {}, {}, {}
    for item in mayores:
        _objeto(item, "store_nbr family filas ventas")
        tienda, familia = _entero(item["store_nbr"], 1), _texto(item["family"], 512)
        filas, venta = _entero(item["filas"], 1, n), _numero(item["ventas"])
        _exigir(tienda in tiendas and familia in familias and (tienda, familia) not in claves)
        claves.add((tienda, familia))
        orden.append((-venta, tienda, familia))
        filas_t[tienda] = filas_t.get(tienda, 0) + filas
        filas_f[familia] = filas_f.get(familia, 0) + filas
        ventas_f.setdefault(familia, []).append(venta)
    _exigir(orden == sorted(orden))
    _exigir(all(filas <= tiendas[t] for t, filas in filas_t.items())
            and all(filas <= familias[f] for f, filas in filas_f.items()))
    for familia, valores in ventas_f.items():
        # Para un top parcial solo es verificable la cota, no las combinaciones omitidas.
        escala = max(montos[familia], *valores)
        if escala:
            _exigir(fsum(v / escala for v in valores) <= montos[familia] / escala
                    or _suma_coherente(valores, montos[familia]))
    if combinaciones <= 10:
        _exigir(filas_t == tiendas and filas_f == familias)
        _exigir(all(_suma_coherente(ventas_f[f], montos[f]) for f in familias))


def validar_reporte_global(reporte: dict) -> None:
    """Comprueba contrato G1 y relaciones comprobables; no produce estadísticas."""
    try:
        _objeto(reporte, "fuente alcance interpretacion_temporal volumen temporalidad ventas promociones distribuciones advertencias")
        _objeto(reporte["fuente"], "nombre_archivo columnas_examinadas")
        _nombre(reporte["fuente"]["nombre_archivo"])
        _exigir(type(reporte["fuente"]["columnas_examinadas"]) is list
                and reporte["fuente"]["columnas_examinadas"] == _COLUMNAS)
        _exigir(_texto(reporte["alcance"]) == "panorama_descriptivo_global")
        temporal = reporte["interpretacion_temporal"]
        _objeto(temporal, "retrospectivo incluye_todas_las_fechas_del_archivo autorizado_para_seleccion_de_modelos nota")
        _exigir(temporal["retrospectivo"] is True and temporal["incluye_todas_las_fechas_del_archivo"] is True
                and temporal["autorizado_para_seleccion_de_modelos"] is False)
        _texto(temporal["nota"], 4096)
        v = reporte["volumen"]
        _objeto(v, "filas_examinadas filas_validas filas_invalidas tiendas_distintas familias_distintas combinaciones_observadas")
        n = _entero(v["filas_examinadas"], 1)
        _exigir(_entero(v["filas_validas"]) == n and _entero(v["filas_invalidas"]) == 0)
        _entero(v["tiendas_distintas"], 1, min(1000, n))
        _entero(v["familias_distintas"], 1, min(1000, n))
        _entero(v["combinaciones_observadas"], 1, min(10000, n))
        _estadisticas(reporte, n)
        _calendario(reporte["temporalidad"], n)
        _distribuciones(reporte["distribuciones"], v, reporte["ventas"])
        for aviso in _lista(reporte["advertencias"], 16, 1):
            _texto(aviso)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        if isinstance(error, ErrorPerfilGlobal):
            raise
        raise ErrorPerfilGlobal("perfil_invalido") from None


def _bytes(texto):
    _exigir(type(texto) is str)
    # Cada carácter requiere al menos un byte: rechazar antes de codificar textos enormes.
    if len(texto) > MAX_BYTES_REPORTE:
        raise ErrorPerfilGlobal("perfil_demasiado_grande")
    try:
        datos = texto.encode("utf-8")
    except UnicodeError:
        raise ErrorPerfilGlobal("perfil_invalido") from None
    if len(datos) > MAX_BYTES_REPORTE:
        raise ErrorPerfilGlobal("perfil_demasiado_grande")
    return datos


def _leer_json(texto):
    def pares(items):
        resultado = {}
        for clave, valor in items:
            _exigir(clave not in resultado)
            resultado[clave] = valor
        return resultado

    def real(valor):
        numero = float(valor)
        _exigir(isfinite(numero))
        return numero

    def constante(_):
        raise ErrorPerfilGlobal("perfil_invalido")

    return json.loads(texto, object_pairs_hook=pares, parse_float=real, parse_constant=constante)


def serializar_reporte_global(reporte: dict) -> str:
    """JSON canónico UTF-8 <=16 MiB. Conserva números y orden de las listas."""
    validar_reporte_global(reporte)
    try:
        texto = json.dumps(reporte, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        _bytes(texto)
        _exigir(_leer_json(texto) == reporte)
        return texto
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        if isinstance(error, ErrorPerfilGlobal):
            raise
        raise ErrorPerfilGlobal("perfil_invalido") from None


def calcular_sha256_reporte(reporte_json: str) -> str:
    """Huella de los bytes UTF-8 exactos; no cambia el texto recibido."""
    return hashlib.sha256(_bytes(reporte_json)).hexdigest()


def preparar_perfil_global(reporte: dict, nombre_archivo: str) -> PerfilGlobalPreparado:
    """Copia G1 y sustituye solo su nombre por el nombre público seguro."""
    validar_reporte_global(reporte)
    nombre = _nombre(nombre_archivo)
    copia = deepcopy(reporte)
    copia["fuente"]["nombre_archivo"] = nombre
    texto = serializar_reporte_global(copia)
    return PerfilGlobalPreparado(VERSION_PERFIL, texto, calcular_sha256_reporte(texto))


def _documento(version, texto, checksum):
    if type(version) is not int or version != VERSION_PERFIL:
        raise ErrorPerfilGlobal("version_no_soportada")
    _exigir(type(checksum) is str and _HASH.fullmatch(checksum) is not None)
    _exigir(calcular_sha256_reporte(texto) == checksum)
    try:
        reporte = _leer_json(texto)
        validar_reporte_global(reporte)
        _exigir(serializar_reporte_global(reporte) == texto)
        return reporte
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        if isinstance(error, ErrorPerfilGlobal):
            raise
        raise ErrorPerfilGlobal("perfil_invalido") from None


def _conexion(conexion, escritura=False):
    if not isinstance(conexion, sqlite3.Connection) or not conexion.in_transaction:
        raise ErrorPerfilGlobal("almacenamiento_no_disponible")
    if escritura and conexion.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ErrorPerfilGlobal("almacenamiento_no_disponible")


def _metadatos(conexion, conjunto_id, csv_hash, reporte, series, observaciones):
    _entero(conjunto_id, 1)
    _exigir(type(csv_hash) is str and _HASH.fullmatch(csv_hash) is not None)
    fila = conexion.execute("SELECT nombre_archivo,hash FROM conjunto_datos WHERE id=?", (conjunto_id,)).fetchone()
    _exigir(fila is not None and fila[0] == reporte["fuente"]["nombre_archivo"] and fila[1] == csv_hash)
    _entero(series, maximo=reporte["volumen"]["combinaciones_observadas"])
    _entero(observaciones, maximo=reporte["volumen"]["filas_validas"])
    reales = conexion.execute("SELECT COUNT(*) FROM serie_temporal WHERE conjunto_id=?", (conjunto_id,)).fetchone()[0]
    filas = conexion.execute("SELECT COUNT(*) FROM venta_diaria v JOIN serie_temporal s ON s.id=v.serie_id "
                             "WHERE s.conjunto_id=?", (conjunto_id,)).fetchone()[0]
    _exigir((series, observaciones) == (reales, filas))


def guardar_perfil_global(conexion: sqlite3.Connection, conjunto_id: int, preparado: PerfilGlobalPreparado,
                          sha256_csv: str, series_importadas: int, observaciones_importadas: int) -> None:
    """INSERT único en la transacción externa; nunca reemplaza ni confirma filas."""
    try:
        _conexion(conexion, escritura=True)
        _exigir(type(preparado) is PerfilGlobalPreparado)
        reporte = _documento(preparado.version_perfil, preparado.reporte_json, preparado.sha256_reporte)
        _metadatos(conexion, conjunto_id, sha256_csv, reporte, series_importadas, observaciones_importadas)
        conexion.execute("INSERT INTO perfil_global_conjunto "
                         "(conjunto_id,version_perfil,sha256_csv,reporte_json,sha256_reporte,series_importadas,observaciones_importadas) "
                         "VALUES (?,?,?,?,?,?,?)", (conjunto_id, preparado.version_perfil, sha256_csv,
                                                   preparado.reporte_json, preparado.sha256_reporte,
                                                   series_importadas, observaciones_importadas))
    except sqlite3.IntegrityError:
        raise ErrorPerfilGlobal("perfil_invalido") from None
    except sqlite3.Error:
        raise ErrorPerfilGlobal("almacenamiento_no_disponible") from None


def recuperar_perfil_global(conexion: sqlite3.Connection, conjunto_id: int) -> dict:
    """Lee un conjunto específico (también inactivo), sin BEGIN, DML ni migraciones.

    Requiere transacción externa. Retorna metadatos públicos y reporte G1 nuevo
    en cada llamada. Una base anterior sin la tabla se considera sin perfil.
    """
    try:
        _conexion(conexion)
        _entero(conjunto_id, 1)
        existe = conexion.execute("SELECT id FROM conjunto_datos WHERE id=?", (conjunto_id,)).fetchone()
        tabla = conexion.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='perfil_global_conjunto'").fetchone()
        if existe is None or tabla is None:
            raise ErrorPerfilGlobal("perfil_no_disponible")
        # El CASE evita recuperar el texto grande antes de comprobar su tamaño en SQLite.
        fila = conexion.execute("SELECT version_perfil,sha256_csv,sha256_reporte,series_importadas,observaciones_importadas,"
                                 "length(CAST(reporte_json AS BLOB)),typeof(reporte_json),"
                                 "CASE WHEN length(CAST(reporte_json AS BLOB))<=? THEN reporte_json END "
                                 "FROM perfil_global_conjunto WHERE conjunto_id=?",
                                 (MAX_BYTES_REPORTE, conjunto_id)).fetchone()
        if fila is None:
            raise ErrorPerfilGlobal("perfil_no_disponible")
        try:
            _exigir(fila[6] == "text" and type(fila[5]) is int)
            if fila[5] > MAX_BYTES_REPORTE:
                raise ErrorPerfilGlobal("perfil_demasiado_grande")
            reporte = _documento(fila[0], fila[7], fila[2])
            _metadatos(conexion, conjunto_id, fila[1], reporte, fila[3], fila[4])
        except ErrorPerfilGlobal as error:
            if error.codigo in ("version_no_soportada", "perfil_demasiado_grande"):
                raise
            raise ErrorPerfilGlobal("perfil_corrupto") from None
        return {"conjunto_id": conjunto_id, "version_perfil": fila[0], "sha256_csv": fila[1],
                "series_importadas": fila[3], "observaciones_importadas": fila[4], "reporte": reporte}
    except sqlite3.Error:
        raise ErrorPerfilGlobal("almacenamiento_no_disponible") from None
