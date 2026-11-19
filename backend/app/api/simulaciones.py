"""GET históricos: frontera HTTP sin SQLite, migraciones ni motores propios."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path as ParametroRuta, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services.consulta_historial_simulaciones import (
    MAX_ENTERO_SQLITE, ErrorConsultaHistorialSimulaciones,
    consultar_ejecucion_simulacion, listar_ejecuciones_simulaciones,
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


def _error(codigo):
    if type(codigo) is not str or codigo not in ERRORES:
        codigo = "error_interno"
    estado, mensaje = ERRORES[codigo]
    return JSONResponse(status_code=estado, content={"detail": {"codigo": codigo, "mensaje": mensaje}})


class RutaHistorial(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        permitidos = _QUERY_LISTADO if self.name == "listar_historial" else frozenset()

        async def manejar(request: Request):
            try:
                pares = request.query_params.multi_items()
                nombres = [nombre for nombre, _ in pares]
                if (set(nombres) - permitidos or len(nombres) != len(set(nombres))
                        or await request.body()):
                    respuesta = _error("entrada_invalida")
                else:
                    respuesta = await original(request)
            except RequestValidationError as error:
                lugares = [e["loc"] for e in error.errors()]
                codigo = ("ejecucion_invalida" if any(l[0] == "path" for l in lugares)
                          else "paginacion_invalida" if any(l[:2] in (("query", "limite"), ("query", "desplazamiento")) for l in lugares)
                          else "filtro_invalido")
                respuesta = _error(codigo)
            except ErrorConsultaHistorialSimulaciones as error:
                respuesta = _error(error.codigo)
            except Exception:
                # Frontera HTTP: nunca publicar textos de excepciones ni validación.
                respuesta = _error("error_interno")
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
