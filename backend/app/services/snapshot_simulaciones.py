"""Snapshots v1 puros: documentos originales, sin E/S ni recálculo de motores.

La versión se suministra al lector desde la cabecera histórica; advertencias
conserva su raíz array exigida por SQL. La huella comprueba bytes, no autoría ni
corrección matemática. Se verifican redundancias y calendario, nunca Q ni la
recurrencia del balance. No se acredita existencia, activación o elegibilidad SQL.
"""

from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
from math import isfinite
import re

from backend.app.services import simulacion_reposicion as motor_actual


# Contrato histórico v1 cerrado. Nunca derivarlo del catálogo o las versiones
# vigentes del motor/escritor: esas decisiones pueden cambiar después del registro.
VERSION_DOCUMENTO_V1 = 1
VERSION_REGLA_V1 = "cobertura_horizonte_15_pendientes_v1"
TECNICAS_V1 = ("estacional_semanal_recursiva", "random_forest", "hist_gradient_boosting")
IDENTIFICADOR_IMPLEMENTACION_V1 = "s4-a1-r1_s4-a3-01_snapshot_v1"
VERSION_POLITICA_NUMERICA_V1 = "binary64_fraction_v1"

VERSION_DOCUMENTO = VERSION_DOCUMENTO_V1
IDENTIFICADOR_IMPLEMENTACION = IDENTIFICADOR_IMPLEMENTACION_V1
VERSION_POLITICA_NUMERICA = VERSION_POLITICA_NUMERICA_V1
MAX_BYTES_DOCUMENTO = 1048576
MAX_PROFUNDIDAD = 64
_MENSAJES = {
    "estructura_invalida": "El documento histórico no tiene la estructura admitida.",
    "version_no_soportada": "La versión histórica no está soportada.",
    "representacion_invalida": "Un valor no conserva la representación numérica o UTF-8 admitida.",
    "integridad_invalida": "Los documentos históricos no son canónicos o presentan contradicciones.",
    "documento_demasiado_grande": "El documento supera el límite de bytes permitido.",
}
_IDENTIDAD = ("pronostico_id", "experimento_id", "tecnica", "fecha_corte", "horizonte")
_PARAMETROS = {"inventario_disponible", "tiempo_abastecimiento", "reserva"}
_DOCUMENTOS = ("solicitud", "origen", "contexto", "advertencias")
_META = {"version_documento", "version_regla", "identificador_implementacion", "version_politica_numerica"}
_RESULTADO = set(_IDENTIDAD) | {"version_regla", "parametros", "periodo_cobertura", "demanda_considerada",
    "reposicion", "fecha_recepcion", "recepcion_dentro_horizonte", "supuestos", "advertencias", "trayectoria", "resumen"}
_FILA = {"paso", "fecha_objetivo", "demanda_pronosticada", "recepcion_simulada", "balance_neto",
         "disponible_hipotetico", "demanda_pendiente_actual"}
_RESUMEN = {"fecha_primera_demanda_pendiente", "mayor_demanda_pendiente_actual", "dias_con_balance_negativo",
            "balance_neto_final", "disponible_hipotetico_final", "demanda_pendiente_final", "recepcion_dentro_horizonte"}


class ErrorSnapshotSimulacion(ValueError):
    """Código estable y mensaje constante; nunca incluye la causa privada."""

    def __init__(self, codigo):
        self.codigo = codigo
        super().__init__(_MENSAJES[codigo])


def _exigir(condicion, codigo="estructura_invalida"):
    if not condicion:
        raise ErrorSnapshotSimulacion(codigo)


def _objeto(valor, campos):
    _exigir(type(valor) is dict and set(valor) == set(campos))


def _igual(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return a.hex() == b.hex()
    if type(a) is dict:
        return a.keys() == b.keys() and all(_igual(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(_igual(x, y) for x, y in zip(a, b))
    return a == b


def _coincide(a, b):
    _exigir(_igual(a, b), "integridad_invalida")


def _json(valor, profundidad=0):
    _exigir(profundidad <= MAX_PROFUNDIDAD)
    if valor is None or type(valor) in (str, bool, int):
        return
    if type(valor) is float:
        _exigir(isfinite(valor), "representacion_invalida")
    elif type(valor) is list:
        for elemento in valor:
            _json(elemento, profundidad + 1)
    elif type(valor) is dict:
        for clave, elemento in valor.items():
            _exigir(type(clave) is str)
            _json(elemento, profundidad + 1)
    else:
        raise ErrorSnapshotSimulacion("estructura_invalida")


def _bytes(texto):
    _exigir(type(texto) is str)
    _exigir(len(texto) <= MAX_BYTES_DOCUMENTO, "documento_demasiado_grande")
    try:
        datos = texto.encode("utf-8")
    except UnicodeError as error:
        raise ErrorSnapshotSimulacion("representacion_invalida") from error
    _exigir(len(datos) <= MAX_BYTES_DOCUMENTO, "documento_demasiado_grande")
    return datos


def _decodificar(texto):
    def pares(elementos):
        objeto = {}
        for clave, valor in elementos:
            _exigir(clave not in objeto, "integridad_invalida")
            objeto[clave] = valor
        return objeto

    def constante(_):
        raise ErrorSnapshotSimulacion("representacion_invalida")

    try:
        valor = json.loads(texto, object_pairs_hook=pares, parse_constant=constante)
    except (ValueError, RecursionError) as error:
        if isinstance(error, ErrorSnapshotSimulacion):
            raise
        raise ErrorSnapshotSimulacion("estructura_invalida") from error
    _exigir(type(valor) in (dict, list))
    _json(valor)
    return valor


def serializar_documento(documento) -> str:
    """JSON UTF-8 sin escapes Unicode artificiales, claves ordenadas y sin espacios.

    Solo dict/list y primitivas JSON exactas, profundidad máxima 64. No admite
    tuple, date, Decimal, Fraction ni clases numéricas convertidas implícitamente.
    Conserva int frente a float y el signo binario de -0.0. No normaliza Unicode.
    """
    _exigir(type(documento) in (dict, list))
    _json(documento)
    try:
        texto = json.dumps(documento, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, OverflowError, RecursionError) as error:
        raise ErrorSnapshotSimulacion("representacion_invalida") from error
    _bytes(texto)
    _exigir(_igual(documento, _decodificar(texto)), "representacion_invalida")
    return texto


def calcular_sha256(texto: str) -> str:
    """Huella de los bytes UTF-8 exactos; no sanea ni acredita el documento."""
    return hashlib.sha256(_bytes(texto)).hexdigest()


def leer_documento(texto, huella, *, version_documento):
    """Lee un objeto/array canónico y verifica huella y versión de cabecera."""
    _exigir(type(version_documento) is int and version_documento == VERSION_DOCUMENTO_V1, "version_no_soportada")
    _bytes(texto)
    _exigir(type(huella) is str and re.fullmatch("[0-9a-f]{64}", huella) is not None, "integridad_invalida")
    _exigir(calcular_sha256(texto) == huella, "integridad_invalida")
    documento = _decodificar(texto)
    _exigir(serializar_documento(documento) == texto, "integridad_invalida")
    return documento


def _id(valor, opcional=False):
    _exigir((opcional and valor is None) or (type(valor) is int and 0 < valor <= 2**63 - 1))


def _fecha(valor):
    _exigir(type(valor) is str)
    try:
        fecha = date.fromisoformat(valor)
        _exigir(fecha.isoformat() == valor)
        return fecha
    except ValueError as error:
        raise ErrorSnapshotSimulacion("estructura_invalida") from error


def _cantidad(valor, publicada=False, signo=False):
    _exigir(type(valor) is float if publicada else type(valor) in (int, float))
    try:
        numero = float(valor)
    except OverflowError as error:
        raise ErrorSnapshotSimulacion("representacion_invalida") from error
    _exigir(isfinite(numero) and (signo or numero >= 0), "representacion_invalida")
    _exigir(valor == numero, "representacion_invalida")
    # El motor normaliza el cero al pasar por Fraction; el documento de solicitud
    # y el de origen conservan sus tipos y signos originales, sin modificarlos.
    return 0.0 if numero == 0 else numero


def _textos(valor):
    _exigir(type(valor) is list and all(type(t) is str and bool(t.strip()) for t in valor))


def _parametros(parametros, publicada=False):
    _objeto(parametros, _PARAMETROS)
    for campo in ("inventario_disponible", "reserva"):
        _cantidad(parametros[campo], publicada)
    _exigir(type(parametros["tiempo_abastecimiento"]) is int and 0 <= parametros["tiempo_abastecimiento"] <= 15)


def _validar_documentos(solicitud, origen, contexto, advertencias, resultados):
    _objeto(solicitud, {"pronostico_id", "modalidad", "escenarios"})
    _objeto(origen, {"conjunto_id", "serie", *_IDENTIDAD, "detalles"})
    _objeto(origen["serie"], {"id", "store_nbr", "family"})
    for valor in (origen["conjunto_id"], origen["serie"]["id"], origen["serie"]["store_nbr"], origen["pronostico_id"]):
        _id(valor)
    _id(origen["experimento_id"], opcional=True)
    _exigir(type(origen["serie"]["family"]) is str and bool(origen["serie"]["family"].strip()))
    _exigir(type(origen["tecnica"]) is str and origen["tecnica"] in TECNICAS_V1)
    _exigir(type(origen["horizonte"]) is int and origen["horizonte"] == 15)
    corte = _fecha(origen["fecha_corte"])
    try:
        fechas = [(corte + timedelta(days=i)).isoformat() for i in range(1, 16)]
    except OverflowError as error:
        raise ErrorSnapshotSimulacion("estructura_invalida") from error
    _objeto(contexto, {*_IDENTIDAD, "version_regla", "fechas_objetivo"})
    _coincide({k: contexto[k] for k in _IDENTIDAD}, {k: origen[k] for k in _IDENTIDAD})
    _coincide(contexto["fechas_objetivo"], fechas)
    _coincide(contexto["version_regla"], VERSION_REGLA_V1)
    _coincide(solicitud["pronostico_id"], origen["pronostico_id"])
    _exigir(type(solicitud["modalidad"]) is str and solicitud["modalidad"] in ("individual", "comparacion"))
    escenarios = solicitud["escenarios"]
    _exigir(type(escenarios) is list and len(escenarios) in ((1,) if solicitud["modalidad"] == "individual" else (2, 3)))
    _exigir(type(resultados) is list and len(resultados) == len(escenarios))
    _textos(advertencias)
    detalles = origen["detalles"]
    _exigir(type(detalles) is list and len(detalles) == 15)
    demandas = []
    for paso, detalle in enumerate(detalles, 1):
        _objeto(detalle, {"paso", "fecha_objetivo", "venta_estimada"})
        _coincide(detalle["paso"], paso)
        _coincide(detalle["fecha_objetivo"], fechas[paso - 1])
        demandas.append(_cantidad(detalle["venta_estimada"]))
    demanda_comun = None
    for posicion, (escenario, entrada) in enumerate(zip(escenarios, resultados), 1):
        _objeto(escenario, {"etiqueta", *_PARAMETROS})
        _coincide(escenario["etiqueta"], "ABC"[posicion - 1])
        parametros = {k: escenario[k] for k in _PARAMETROS}
        _parametros(parametros)
        _objeto(entrada, {"etiqueta", "simulacion"})
        _coincide(entrada["etiqueta"], escenario["etiqueta"])
        r = entrada["simulacion"]
        _objeto(r, _RESULTADO)
        _coincide({k: r[k] for k in _IDENTIDAD}, {k: origen[k] for k in _IDENTIDAD})
        _coincide(r["version_regla"], contexto["version_regla"])
        _parametros(r["parametros"], publicada=True)
        esperado = {k: _cantidad(parametros[k]) if k != "tiempo_abastecimiento" else parametros[k] for k in parametros}
        _coincide(r["parametros"], esperado)
        for campo in ("demanda_considerada", "reposicion"):
            _cantidad(r[campo], publicada=True)
        if demanda_comun is None:
            demanda_comun = r["demanda_considerada"]
        _coincide(r["demanda_considerada"], demanda_comun)
        _coincide(r["periodo_cobertura"], {"fecha_inicio": fechas[0], "fecha_fin": fechas[-1], "dias": 15})
        plazo = parametros["tiempo_abastecimiento"]
        try:
            recepcion = (corte + timedelta(days=plazo + 1)).isoformat()
        except OverflowError as error:
            raise ErrorSnapshotSimulacion("estructura_invalida") from error
        _coincide(r["fecha_recepcion"], recepcion)
        _coincide(r["recepcion_dentro_horizonte"], plazo < 15)
        _textos(r["supuestos"])
        _textos(r["advertencias"])
        filas = r["trayectoria"]
        _exigir(type(filas) is list and len(filas) == 15)
        for paso, fila in enumerate(filas, 1):
            _objeto(fila, _FILA)
            _coincide(fila["paso"], paso)
            _coincide(fila["fecha_objetivo"], fechas[paso - 1])
            for campo in _FILA - {"paso", "fecha_objetivo"}:
                _cantidad(fila[campo], publicada=True, signo=campo == "balance_neto")
            _coincide(fila["demanda_pronosticada"], demandas[paso - 1])
            recibida = r["reposicion"] if plazo < 15 and paso == plazo + 1 else 0.0
            _coincide(fila["recepcion_simulada"], recibida)
            # Redundancias de una fila publicada, sin reconstruir Bt desde Bt-1.
            saldo = fila["balance_neto"]
            _coincide(fila["disponible_hipotetico"], saldo if saldo > 0 else 0.0)
            _coincide(fila["demanda_pendiente_actual"], -saldo if saldo < 0 else 0.0)
        resumen = r["resumen"]
        _objeto(resumen, _RESUMEN)
        for resumen_campo, diario in (("balance_neto_final", "balance_neto"), ("disponible_hipotetico_final", "disponible_hipotetico"),
                                     ("demanda_pendiente_final", "demanda_pendiente_actual")):
            _coincide(resumen[resumen_campo], filas[-1][diario])
        negativas = [f for f in filas if f["balance_neto"] < 0]
        _coincide(resumen["dias_con_balance_negativo"], len(negativas))
        _coincide(resumen["fecha_primera_demanda_pendiente"], negativas[0]["fecha_objetivo"] if negativas else None)
        _coincide(resumen["mayor_demanda_pendiente_actual"], max(f["demanda_pendiente_actual"] for f in filas))
        _coincide(resumen["recepcion_dentro_horizonte"], r["recepcion_dentro_horizonte"])
    if solicitud["modalidad"] == "individual":
        _coincide(advertencias, resultados[0]["simulacion"]["advertencias"])


def verificar_paquete(paquete) -> dict:
    """Verifica columnas serializadas y devuelve documentos independientes leídos.

    No usa datos actuales ni llama motores. Comprueba estructura, tipos,
    calendario y redundancias; no certifica Q, sumas ni recurrencias matemáticas.
    """
    campos = _META | {"modalidad", "numero_escenarios", "conjunto_id", "serie_id", "pronostico_id", "escenarios"}
    for nombre in _DOCUMENTOS:
        campos |= {nombre + "_json", "sha256_" + nombre}
    _objeto(paquete, campos)
    _coincide(paquete["version_regla"], VERSION_REGLA_V1)
    _exigir(type(paquete["identificador_implementacion"]) is str
            and type(paquete["version_politica_numerica"]) is str
            and paquete["identificador_implementacion"] == IDENTIFICADOR_IMPLEMENTACION_V1
            and paquete["version_politica_numerica"] == VERSION_POLITICA_NUMERICA_V1, "version_no_soportada")
    documentos = {n: leer_documento(paquete[n + "_json"], paquete["sha256_" + n],
                                   version_documento=paquete["version_documento"]) for n in _DOCUMENTOS}
    _exigir(type(paquete["escenarios"]) is list and len(paquete["escenarios"]) in (1, 2, 3))
    resultados = []
    for posicion, fila in enumerate(paquete["escenarios"], 1):
        _objeto(fila, {"etiqueta", "posicion", "resultado_json", "sha256_resultado"})
        _coincide(fila["posicion"], posicion)
        resultados.append({"etiqueta": fila["etiqueta"], "simulacion": leer_documento(
            fila["resultado_json"], fila["sha256_resultado"], version_documento=paquete["version_documento"])})
    _validar_documentos(**documentos, resultados=resultados)
    _coincide(paquete["modalidad"], documentos["solicitud"]["modalidad"])
    _coincide(paquete["numero_escenarios"], len(resultados))
    for campo, valor in (("conjunto_id", documentos["origen"]["conjunto_id"]),
                         ("serie_id", documentos["origen"]["serie"]["id"]),
                         ("pronostico_id", documentos["origen"]["pronostico_id"])):
        _coincide(paquete[campo], valor)
    return {**{k: paquete[k] for k in _META}, **documentos, "resultados": resultados}


def _admitir_escritura_v1(pronostico):
    # Solo el escritor depende del motor actual. Un motor con regla diferente
    # necesita compatibilidad expresa; no basta con recibir una etiqueta antigua.
    for actual, historico in ((VERSION_DOCUMENTO, VERSION_DOCUMENTO_V1),
                             (IDENTIFICADOR_IMPLEMENTACION, IDENTIFICADOR_IMPLEMENTACION_V1),
                             (VERSION_POLITICA_NUMERICA, VERSION_POLITICA_NUMERICA_V1)):
        _exigir(_igual(actual, historico), "version_no_soportada")
    regla = getattr(motor_actual, "VERSION_REGLA", None)
    catalogo = getattr(motor_actual, "TECNICAS", None)
    _exigir(type(regla) is str and regla == VERSION_REGLA_V1
            and type(catalogo) is tuple and all(type(t) is str for t in catalogo), "version_no_soportada")
    tecnica = pronostico["tecnica"]
    if type(tecnica) is str and tecnica in TECNICAS_V1:
        _exigir(tecnica in catalogo, "version_no_soportada")


def preparar_snapshot(pronostico, detalles, procedencia, escenarios, resultado, *, modalidad) -> dict:
    """Construye columnas históricas v1 desde entradas y motores YA ejecutados.

    Procedencia: {conjunto_id, serie:{id,store_nbr,family}}. Escenarios list/tuple
    A o A/B[/C]. Individual recibe salida íntegra de calcular_trayectoria;
    comparación recibe comparar_escenarios o su integración con procedencia.
    Cantidades de entrada: int/float exactos, sin bool ni conversiones numéricas
    externas. Fechas date exacto se adaptan explícitamente a ISO; las secuencias
    de entrada list/tuple a listas. Los documentos conservan tipos numéricos y
    -0.0 originales; las salidas publicadas del motor no se modifican.
    El escritor v1 exige motor vigente compatible y resultados con el contrato
    histórico cerrado. El lector no depende de esa comprobación del escritor.
    """
    _exigir(type(pronostico) is dict)
    campos = set(_IDENTIDAD) - {"pronostico_id", "experimento_id"}
    _exigir({"id", *campos} <= set(pronostico))
    permitidos = {"id", "pronostico_id", "experimento_id", *campos, "tipo", "fecha_generacion",
                  "inicio_entrenamiento", "fin_entrenamiento", "duracion_segundos"}
    _exigir(set(pronostico) <= permitidos)
    _admitir_escritura_v1(pronostico)
    _objeto(procedencia, {"conjunto_id", "serie"})
    _exigir(type(detalles) in (list, tuple) and type(escenarios) in (list, tuple))
    _exigir(type(modalidad) is str and modalidad in ("individual", "comparacion"))
    # Rechazar objetos externos antes de deepcopy: no ejecutar sus protocolos.
    for valor in (procedencia, list(escenarios), resultado):
        _json(valor)
    corte = pronostico["fecha_corte"]
    corte = corte.isoformat() if type(corte) is date else corte
    fecha_corte = _fecha(corte)
    if "tipo" in pronostico:
        _exigir(type(pronostico["tipo"]) is str and pronostico["tipo"] in ("validacion", "prueba", "futuro"))
    if "inicio_entrenamiento" in pronostico or "fin_entrenamiento" in pronostico:
        _exigir({"inicio_entrenamiento", "fin_entrenamiento"} <= set(pronostico))
        limites = [pronostico[k] for k in ("inicio_entrenamiento", "fin_entrenamiento")]
        inicio, fin = [_fecha(v.isoformat() if type(v) is date else v) for v in limites]
        _exigir(inicio <= fin <= fecha_corte)
    if "fecha_generacion" in pronostico:
        _exigir(type(pronostico["fecha_generacion"]) is str)
        try:
            datetime.fromisoformat(pronostico["fecha_generacion"])
        except ValueError as error:
            raise ErrorSnapshotSimulacion("estructura_invalida") from error
    if pronostico.get("duracion_segundos") is not None:
        _cantidad(pronostico["duracion_segundos"])
    origen = {**deepcopy(procedencia), "pronostico_id": pronostico["id"],
              "experimento_id": pronostico.get("experimento_id"), "tecnica": pronostico["tecnica"],
              "fecha_corte": corte, "horizonte": pronostico["horizonte"], "detalles": []}
    if "pronostico_id" in pronostico:
        _coincide(pronostico["pronostico_id"], origen["pronostico_id"])
    for detalle in detalles:
        _exigir(type(detalle) is dict and set(detalle) in ({"paso", "fecha_objetivo", "venta_estimada"},
                 {"paso", "fecha_objetivo", "venta_estimada", "pronostico_id"}))
        if "pronostico_id" in detalle:
            _coincide(detalle["pronostico_id"], origen["pronostico_id"])
        f = detalle["fecha_objetivo"]
        origen["detalles"].append({"paso": detalle["paso"], "fecha_objetivo": f.isoformat() if type(f) is date else f,
                                    "venta_estimada": detalle["venta_estimada"]})
    solicitud = {"pronostico_id": pronostico["id"], "modalidad": modalidad, "escenarios": deepcopy(list(escenarios))}
    if modalidad == "individual":
        _exigir(type(resultado) is dict)
        resultados = [{"etiqueta": "A", "simulacion": deepcopy(resultado)}]
        advertencias = resultado.get("advertencias")
        contexto = {**{k: origen[k] for k in _IDENTIDAD}, "version_regla": resultado.get("version_regla"),
                    "fechas_objetivo": [d["fecha_objetivo"] for d in origen["detalles"]]}
    else:
        base = {"contexto", "resultados", "advertencias_comparacion"}
        _exigir(type(resultado) is dict and set(resultado) in (base, base | {"conjunto_id", "serie"}))
        if "conjunto_id" in resultado:
            _coincide({k: resultado[k] for k in procedencia}, procedencia)
        contexto, resultados, advertencias = (deepcopy(resultado[k]) for k in ("contexto", "resultados", "advertencias_comparacion"))
    documentos = {"solicitud": solicitud, "origen": origen, "contexto": contexto, "advertencias": deepcopy(advertencias)}
    _validar_documentos(**documentos, resultados=resultados)
    paquete = {"version_documento": VERSION_DOCUMENTO, "version_regla": contexto["version_regla"],
               "identificador_implementacion": IDENTIFICADOR_IMPLEMENTACION, "version_politica_numerica": VERSION_POLITICA_NUMERICA,
               "modalidad": modalidad, "numero_escenarios": len(resultados), "conjunto_id": origen["conjunto_id"],
               "serie_id": origen["serie"]["id"], "pronostico_id": origen["pronostico_id"], "escenarios": []}
    for nombre, documento in documentos.items():
        texto = serializar_documento(documento)
        paquete[nombre + "_json"], paquete["sha256_" + nombre] = texto, calcular_sha256(texto)
    for posicion, entrada in enumerate(resultados, 1):
        texto = serializar_documento(entrada["simulacion"])
        paquete["escenarios"].append({"etiqueta": entrada["etiqueta"], "posicion": posicion,
                                     "resultado_json": texto, "sha256_resultado": calcular_sha256(texto)})
    verificar_paquete(paquete)
    return paquete
