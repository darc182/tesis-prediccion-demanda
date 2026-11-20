"""Historial GET y registro POST explícito, sin SQL ni motores propios."""

import json
from math import isfinite
from pathlib import Path
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Path as ParametroRuta, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services.consulta_historial_simulaciones import (
    MAX_ENTERO_SQLITE, ErrorConsultaHistorialSimulaciones,
    consultar_ejecucion_simulacion, listar_ejecuciones_simulaciones,
)

from backend.app.services.registro_simulaciones import (
    ErrorRegistroSimulacion, registrar_comparacion_simulaciones, registrar_simulacion,
)


ERRORES = {
    "entrada_invalida": (422, "La consulta no admite cuerpo, parámetros desconocidos ni parámetros repetidos."),
    "ejecucion_invalida": (422, "El identificador de ejecución debe ser un entero decimal ASCII positivo dentro del límite SQLite."),
    "filtro_invalido": (422, "Los filtros deben contener identificadores decimales ASCII positivos o una modalidad admitida."),
    "paginacion_invalida": (422, "Indique un límite entre 1 y 50 y un desplazamiento decimal ASCII no negativo dentro del límite SQLite."),
    "ejecucion_inexistente": (404, "La ejecución histórica solicitada no existe."),
    "esquema_no_preparado": (409, "El esquema de simulaciones necesita preparación explícita."),
    "esquema_incompatible": (409, "El esquema de simulaciones no es compatible."),
    "version_no_soportada": (409, "La versión del esquema histórico no está soportada."),
    "snapshot_corrupto": (409, "La ejecución histórica está incompleta o contiene documentos inconsistentes."),
    "snapshot_incompatible": (409, "El formato histórico no está soportado."),
    "base_no_disponible": (503, "La base de datos existente no está disponible."),
    "error_interno": (500, "No se pudo completar la consulta del historial de simulaciones."),
}
_QUERY_LISTADO = frozenset({"limite", "desplazamiento", "pronostico_id", "conjunto_id", "serie_id", "modalidad"})


# El catálogo GET permanece intacto. Las operaciones de escritura tienen su
# propia entrada y mensajes; nunca se publican mensajes recibidos del servicio.
ERRORES_REGISTRO = {
    **{c: ERRORES[c] for c in ("esquema_no_preparado", "esquema_incompatible", "version_no_soportada",
                              "snapshot_corrupto", "snapshot_incompatible", "base_no_disponible")},
    "entrada_invalida": (422, "Se requiere un objeto JSON UTF-8 de hasta 16384 bytes, sin claves duplicadas ni query."),
    "solicitud_invalida": (422, "El objeto debe contener exactamente los campos obligatorios documentados."),
    "pronostico_invalido": (422, "El pronóstico debe ser un entero positivo dentro del límite SQLite."),
    "clave_idempotencia_invalida": (422, "Indique una clave de texto no vacío de hasta 128 bytes UTF-8, sin nulos."),
    "escenarios_invalidos": (422, "Indique dos o tres escenarios completos A/B o A/B/C en ese orden."),
    "parametros_invalidos": (422, "El inventario, la reserva o el plazo hipotéticos no son válidos."),
    "precision_no_representable": (422, "Una cantidad no conserva la precisión numérica admitida."),
    "tipo_no_evaluable": (422, "Solo se registran simulaciones de pronósticos de validación."),
    "corte_no_autorizado": (422, "El corte no pertenece a las tres ventanas autorizadas."),
    "pronostico_inexistente": (404, "El pronóstico solicitado no existe."),
    "pronostico_inactivo": (404, "El pronóstico no pertenece al conjunto activo."),
    "conflicto_idempotencia": (409, "La clave ya corresponde a una solicitud diferente."),
    "sin_conjunto_activo": (409, "No existe un conjunto de datos activo para una ejecución nueva."),
    "pronostico_corrupto": (409, "El pronóstico o su procedencia contienen datos inconsistentes."),
    "resultados_incompatibles": (409, "Los resultados no corresponden a una simulación coherente."),
    "resultado_no_finito": (409, "El resultado excede el rango numérico admitido."),
    "origen_modificado": (409, "El origen cambió durante el cálculo; no se guardó la ejecución."),
    "error_persistencia": (503, "No se pudo confirmar una ejecución completa."),
    "error_interno": (500, "No se pudo completar la solicitud de registro de simulación."),
}
MAX_BYTES_CUERPO = 16384
MAX_PROFUNDIDAD_JSON = 32
_TIPO_JSON = re.compile(r'application/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?', re.IGNORECASE)


class _ErrorEntradaJSON(ValueError):
    def __init__(self, codigo="entrada_invalida"):
        self.codigo = codigo


async def _leer_json_registro(request):
    """Límite sobre bytes recibidos, incluso sin Content-Length.

    float interpreta binary64 nativo. Solo se impide el cero falso por
    subdesbordamiento; no se aplican tolerancias ni aritmética de simulación.
    Los enteros se conservan como int para la política del servicio publicado.
    """
    tipos = request.headers.getlist("content-type")
    if len(tipos) != 1 or not _TIPO_JSON.fullmatch(tipos[0].strip()):
        raise _ErrorEntradaJSON()
    cuerpo = bytearray()
    async for bloque in request.stream():
        if len(cuerpo) + len(bloque) > MAX_BYTES_CUERPO:
            raise _ErrorEntradaJSON()
        cuerpo.extend(bloque)
    def pares(elementos):
        objeto = {}
        for clave, valor in elementos:
            if clave in objeto:
                raise _ErrorEntradaJSON()
            objeto[clave] = valor
        return objeto
    def constante(_):
        raise _ErrorEntradaJSON()
    def entero(texto):
        try:
            return int(texto)
        except ValueError as error:
            raise _ErrorEntradaJSON("parametros_invalidos") from error
    def flotante(texto):
        numero = float(texto)
        if not isfinite(numero):
            raise _ErrorEntradaJSON("parametros_invalidos")
        significando = texto.lower().split("e")[0]
        if numero == 0 and any(c in "123456789" for c in significando):
            raise _ErrorEntradaJSON("precision_no_representable")
        return numero
    try:
        datos = json.loads(bytes(cuerpo).decode("utf-8"), object_pairs_hook=pares,
                           parse_int=entero, parse_float=flotante, parse_constant=constante)
    except _ErrorEntradaJSON:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _ErrorEntradaJSON() from error
    pendientes = [(datos, 0)]
    while pendientes:
        valor, nivel = pendientes.pop()
        if nivel > MAX_PROFUNDIDAD_JSON:
            raise _ErrorEntradaJSON()
        if type(valor) is dict:
            pendientes.extend((v, nivel + 1) for v in valor.values())
        elif type(valor) is list:
            pendientes.extend((v, nivel + 1) for v in valor)
    if type(datos) is not dict:
        raise _ErrorEntradaJSON("solicitud_invalida")
    # Starlette conserva body/json en estos cachés. Se suministra la lectura
    # acotada y estricta para evitar una segunda decodificación permisiva.
    request._body = bytes(cuerpo)
    request._json = datos
    request.state.solicitud_simulacion = datos



def _error(codigo, *, registro=False):
    catalogo = ERRORES_REGISTRO if registro else ERRORES
    if type(codigo) is not str or codigo not in catalogo:
        codigo = "error_interno"
    estado, mensaje = catalogo[codigo]
    return JSONResponse(status_code=estado, content={"detail": {"codigo": codigo, "mensaje": mensaje}})


class RutaHistorial(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        es_registro = self.name in ("registrar_individual", "registrar_comparacion") and "POST" in self.methods
        permitidos = _QUERY_LISTADO if self.name == "listar_historial" else frozenset()

        async def manejar(request: Request):
            try:
                pares = request.query_params.multi_items()
                nombres = [nombre for nombre, _ in pares]
                if es_registro:
                    if nombres:
                        raise _ErrorEntradaJSON()
                    await _leer_json_registro(request)
                    respuesta = await original(request)
                elif (set(nombres) - permitidos or len(nombres) != len(set(nombres))
                        or await request.body()):
                    respuesta = _error("entrada_invalida")
                else:
                    respuesta = await original(request)
            except RequestValidationError as error:
                lugares = [e["loc"] for e in error.errors()]
                codigo = (_codigo_entrada_registro(error) if es_registro else "ejecucion_invalida" if any(l[0] == "path" for l in lugares)
                          else "paginacion_invalida" if any(l[:2] in (("query", "limite"), ("query", "desplazamiento")) for l in lugares)
                          else "filtro_invalido")
                respuesta = _error(codigo, registro=es_registro)
            except _ErrorEntradaJSON as error:
                respuesta = _error(error.codigo, registro=es_registro)
            except ErrorRegistroSimulacion as error:
                respuesta = _error(error.codigo if es_registro else "error_interno", registro=es_registro)
            except ErrorConsultaHistorialSimulaciones as error:
                respuesta = _error("error_interno" if es_registro else error.codigo, registro=es_registro)
            except Exception:
                # Frontera HTTP: nunca publicar textos de excepciones ni validación.
                respuesta = _error("error_interno", registro=es_registro)
            respuesta.headers["Cache-Control"] = "no-store"
            return respuesta

        return manejar


def _decimal_ascii(valor):
    # FastAPI valida también los defaults Python ya enteros; los valores HTTP
    # explícitos llegan como str y siempre recorren la comprobación ASCII.
    if type(valor) is int and 0 <= valor <= MAX_ENTERO_SQLITE:
        return valor
    if type(valor) is not str or not valor.isascii() or not valor.isdecimal():
        raise ValueError("Entero decimal ASCII requerido.")
    digitos = valor.lstrip("0") or "0"
    maximo = str(MAX_ENTERO_SQLITE)
    if len(digitos) > len(maximo) or (len(digitos) == len(maximo) and digitos > maximo):
        raise ValueError("Entero fuera de rango.")
    return int(digitos)


EnteroHTTP = Annotated[int, BeforeValidator(_decimal_ascii)]


class ModeloHistorial(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SerieHistorica(ModeloHistorial):
    id: int = Field(gt=0)
    store_nbr: int = Field(gt=0)
    family: str


class CabeceraHistorica(ModeloHistorial):
    ejecucion_id: int = Field(gt=0)
    fecha_registro: str = Field(description="Fecha UTC original YYYY-MM-DDTHH:MM:SSZ.")
    modalidad: Literal["individual", "comparacion"]
    numero_escenarios: int = Field(ge=1, le=3)


class ResumenHistorico(CabeceraHistorica):
    conjunto_id: int = Field(gt=0)
    serie: SerieHistorica
    pronostico_id: int = Field(gt=0)
    experimento_id: int | None
    tecnica: str
    fecha_corte: str
    etiquetas: list[Literal["A", "B", "C"]] = Field(min_length=1, max_length=3)


class ListadoHistorial(ModeloHistorial):
    total: int = Field(ge=0)
    limite: int = Field(ge=1, le=50)
    desplazamiento: int = Field(ge=0, le=MAX_ENTERO_SQLITE)
    ejecuciones: list[ResumenHistorico] = Field(max_length=50)


class ResultadoHistorico(ModeloHistorial):
    etiqueta: Literal["A", "B", "C"]
    simulacion: dict[str, JsonValue] = Field(description="Resultado íntegro publicado: parámetros, Q, recepción, periodo, quince filas, resumen, supuestos y advertencias. Sin recálculo.")


class PaqueteHistorico(ModeloHistorial):
    # Los documentos internos ya tienen un lector v1 estricto en S4-A4-04.
    # No duplicarlo aquí ni usar modelos float que conviertan números históricos.
    version_documento: Literal[1]
    version_regla: str
    identificador_implementacion: str
    version_politica_numerica: str
    solicitud: dict[str, JsonValue] = Field(description="Solicitud original con escenarios en orden y parámetros con sus tipos originales.")
    origen: dict[str, JsonValue] = Field(description="Procedencia pública y las quince estimaciones originales.")
    contexto: dict[str, JsonValue] = Field(description="Identidad, regla y quince fechas compartidas.")
    advertencias: list[str]
    resultados: list[ResultadoHistorico] = Field(min_length=1, max_length=3)


class DetalleHistorial(CabeceraHistorica):
    historico: PaqueteHistorico


class DetalleError(ModeloHistorial):
    codigo: str
    mensaje: str


class ErrorHistorial(ModeloHistorial):
    detail: DetalleError


_CACHE = {"Cache-Control": {"description": "No almacenar la respuesta.", "schema": {"type": "string", "const": "no-store"}}}
RESPUESTAS = {
    200: {"description": "Historial original verificado; simulaciones hipotéticas, no inventario observado ni órdenes reales.", "headers": _CACHE},
    **{estado: {"model": ErrorHistorial, "description": ", ".join(c for c, (e, _) in ERRORES.items() if e == estado), "headers": _CACHE}
       for estado in (422, 404, 409, 503, 500)},
}
router = APIRouter(prefix="/api/simulaciones", tags=["simulaciones"], route_class=RutaHistorial)


@router.get("/historial", response_model=ListadoHistorial, responses=RESPUESTAS,
            summary="Listar ejecuciones históricas verificadas")
def listar_historial(
    limite: Annotated[EnteroHTTP, Query(ge=1, le=50, description="Entero decimal ASCII, una sola vez.")] = 20,
    desplazamiento: Annotated[EnteroHTTP, Query(ge=0, le=MAX_ENTERO_SQLITE)] = 0,
    pronostico_id: Annotated[EnteroHTTP | None, Query(gt=0, le=MAX_ENTERO_SQLITE)] = None,
    conjunto_id: Annotated[EnteroHTTP | None, Query(gt=0, le=MAX_ENTERO_SQLITE)] = None,
    serie_id: Annotated[EnteroHTTP | None, Query(gt=0, le=MAX_ENTERO_SQLITE)] = None,
    modalidad: Annotated[Literal["individual", "comparacion"] | None, Query()] = None,
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Una llamada al servicio publicado; sin filtro de conjunto activo.

    Solo admite los seis parámetros documentados, una vez cada uno y sin cuerpo.
    COUNT y página comparten la instantánea de lectura del servicio. Se verifica
    íntegramente cada ejecución de la página; total cuenta cabeceras coincidentes.
    OFFSET puede cambiar entre peticiones si existen nuevas inserciones.
    No prepara esquemas, registra, recalcula, entrena ni selecciona ganadores.
    Éxitos y errores gestionados incluyen Cache-Control: no-store.
    """
    resultado = listar_ejecuciones_simulaciones(
        ruta_base, limite=limite, desplazamiento=desplazamiento,
        pronostico_id=pronostico_id, conjunto_id=conjunto_id, serie_id=serie_id, modalidad=modalidad,
    )
    ListadoHistorial.model_validate(resultado)
    return JSONResponse(content=resultado)


@router.get("/historial/{ejecucion_id}", response_model=DetalleHistorial, responses=RESPUESTAS,
            summary="Consultar una ejecución histórica íntegra por ID")
def consultar_historial(
    ejecucion_id: Annotated[EnteroHTTP, ParametroRuta(gt=0, le=MAX_ENTERO_SQLITE)],
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Recupera snapshots originales, sin leer ni recalcular el pronóstico actual.

    No exige conjunto activo. Conserva solicitud, procedencia, quince estimaciones,
    versiones, advertencias y resultados A/B/C completos. Los documentos internos
    se verifican en el servicio mediante el lector histórico v1 publicado.
    El modelo documenta y comprueba la envoltura; se serializa el objeto original
    con JSONResponse para conservar int/float, -0.0 y binary64 sin coerciones.
    No admite cuerpo ni query. Éxitos y errores incluyen Cache-Control: no-store.
    """
    resultado = consultar_ejecucion_simulacion(ruta_base, ejecucion_id)
    DetalleHistorial.model_validate(resultado)
    return JSONResponse(content=resultado)


# Tipos HTTP estrictos, sin reglas matemáticas ni validadores vigentes del motor.
# La instancia validada sirve de barrera estructural; se delegan los originales
# del lector JSON, no model_dump() ni una conversión a float.
class ParametrosRegistro(ModeloHistorial):
    inventario_disponible: int | float = Field(description="Cantidad no negativa, con tipo int/float original; validación numérica del servicio.", json_schema_extra={"minimum": 0})
    tiempo_abastecimiento: int = Field(description="Plazo entero de 0 a 15 días.", json_schema_extra={"minimum": 0, "maximum": 15})
    reserva: int | float = Field(description="Reserva hipotética no negativa, sin redondeo.", json_schema_extra={"minimum": 0})


class SolicitudIndividual(ParametrosRegistro):
    pronostico_id: int = Field(gt=0, le=MAX_ENTERO_SQLITE)
    clave_idempotencia: str = Field(description="Obligatoria, no vacía ni solo espacios, sin nulos; máximo 128 bytes UTF-8. No se genera automáticamente.")


class EscenarioRegistro(ParametrosRegistro):
    etiqueta: Literal["A", "B", "C"]


class SolicitudComparacion(ModeloHistorial):
    pronostico_id: int = Field(gt=0, le=MAX_ENTERO_SQLITE)
    escenarios: list[EscenarioRegistro] = Field(min_length=2, max_length=3, description="A/B o A/B/C, completos y en ese orden. Se permiten parámetros iguales.")
    clave_idempotencia: str = Field(description="Clave explícita de hasta 128 bytes UTF-8, no vacía ni solo espacios, sin nulos.")


class RegistroSimulacion(DetalleHistorial):
    estado_registro: Literal["creado", "reutilizado"]
    hubo_escritura: bool


def _codigo_entrada_registro(error):
    errores = error.errors()
    if any(e["type"] in ("missing", "extra_forbidden") and e["loc"][:2] != ("body", "escenarios") for e in errores):
        return "solicitud_invalida"
    for campo, codigo in (("pronostico_id", "pronostico_invalido"), ("clave_idempotencia", "clave_idempotencia_invalida")):
        if any(e["loc"][:2] == ("body", campo) for e in errores):
            return codigo
    if any(e["loc"][:2] == ("body", "escenarios") and
           (len(e["loc"]) <= 3 or "etiqueta" in e["loc"] or e["type"] in ("missing", "extra_forbidden")) for e in errores):
        return "escenarios_invalidos"
    return "parametros_invalidos"


RESPUESTAS_REGISTRO = {
    200: {"model": RegistroSimulacion, "description": "Ejecución original reutilizada; sin nuevas escrituras ni recálculo.", "headers": _CACHE},
    201: {"description": "Ejecución hipotética nueva confirmada por el servicio transaccional.", "headers": _CACHE},
    **{estado: {"model": ErrorHistorial, "description": ", ".join(c for c, (e, _) in ERRORES_REGISTRO.items() if e == estado), "headers": _CACHE}
       for estado in (422, 404, 409, 503, 500)},
}


def _respuesta_registro(resultado):
    validado = RegistroSimulacion.model_validate(resultado)
    if validado.hubo_escritura != (validado.estado_registro == "creado"):
        raise ValueError("Estado de registro contradictorio.")
    return JSONResponse(status_code=201 if validado.estado_registro == "creado" else 200, content=resultado)


@router.post("/individual", status_code=201, response_model=RegistroSimulacion, responses=RESPUESTAS_REGISTRO,
             summary="Registrar explícitamente una simulación hipotética individual")
def registrar_individual(request: Request, solicitud: Annotated[SolicitudIndividual, Body()],
                        ruta_base: Path = Depends(obtener_ruta_base)):
    """JSON application/json UTF-8, máximo 16384 bytes y 32 niveles, sin query ni duplicados.

    Preserva int/float y -0.0 originales. El servicio decide identidad histórica,
    precisión, elegibilidad, cálculo, revalidación y commit/rollback. Un reintento
    válido recupera el original incluso con conjunto inactivo; no recalcula.
    Clave nueva puede crear otra ejecución con parámetros iguales. No reintentar
    automáticamente ante un fallo de red: la transacción podría haberse confirmado.
    Son escenarios hipotéticos, no órdenes reales. Cache-Control: no-store.
    """
    datos = request.state.solicitud_simulacion
    return _respuesta_registro(registrar_simulacion(
        ruta_base, datos["pronostico_id"], datos["inventario_disponible"],
        datos["tiempo_abastecimiento"], datos["reserva"], clave_idempotencia=datos["clave_idempotencia"],
    ))


@router.post("/comparacion", status_code=201, response_model=RegistroSimulacion, responses=RESPUESTAS_REGISTRO,
             summary="Registrar explícitamente escenarios hipotéticos A/B o A/B/C")
def registrar_comparacion(request: Request, solicitud: Annotated[SolicitudComparacion, Body()],
                         ruta_base: Path = Depends(obtener_ruta_base)):
    """Una llamada al servicio, sin query, migraciones ni transacciones propias.

    Objeto JSON UTF-8 de hasta 16384 bytes y 32 niveles, sin duplicados ni no finitos.
    Conserva los escenarios y sus números originales, en orden A/B o A/B/C.
    201 implica registro nuevo confirmado; 200 reutiliza el histórico íntegro.
    Los conflictos no sobrescriben. No son órdenes operativas ni se selecciona
    un ganador. No repetir automáticamente ante errores de red.
    Éxitos y errores gestionados llevan Cache-Control: no-store.
    """
    datos = request.state.solicitud_simulacion
    return _respuesta_registro(registrar_comparacion_simulaciones(
        ruta_base, datos["pronostico_id"], datos["escenarios"], clave_idempotencia=datos["clave_idempotencia"],
    ))
