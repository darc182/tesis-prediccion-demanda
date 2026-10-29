"""API explícita de generación y consulta; no ejecuta operaciones al importar."""

from dataclasses import asdict
from pathlib import Path
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Path as ParametroRuta, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services.consulta_pronosticos import (
    ErrorConsultaPronostico, consultar_pronostico, listar_pronosticos,
)
from backend.app.services.registro_pronosticos import (
    TECNICAS, ErrorRegistroPronostico, registrar_pronostico,
)


# Solo mensajes públicos fijos: nunca se devuelve str(error) de los servicios.
ERRORES = {
    "entrada_invalida": (422, "Revise los identificadores y el cuerpo: técnica y fecha de corte YYYY-MM-DD son obligatorias."),
    "serie_invalida": (422, "El identificador de serie debe ser un entero positivo."),
    "tecnica_invalida": (422, "La técnica solicitada no está disponible."),
    "corte_invalido": (422, "La fecha de corte debe ser una fecha válida con formato YYYY-MM-DD."),
    "corte_no_autorizado": (422, "El corte no está autorizado para generar pronósticos de validación."),
    "prueba_reservada": (422, "La prueba final permanece reservada; no se permite generar ese horizonte."),
    "serie_inexistente": (404, "La serie no existe en el conjunto activo."),
    "serie_inactiva": (404, "La serie no pertenece al conjunto activo."),
    "pronostico_inexistente": (404, "El pronóstico no existe en el conjunto activo."),
    "sin_conjunto_activo": (409, "No existe un conjunto de datos activo para esta operación."),
    "historial_cambiado": (409, "Los datos cambiaron durante la generación. Vuelva a solicitar el pronóstico."),
    "conflicto": (409, "Existe un pronóstico conflictivo o incompleto; no se sobrescribió."),
    "pronostico_corrupto": (409, "El pronóstico almacenado está incompleto o presenta datos inconsistentes."),
    "sin_historial": (422, "La serie no tiene observaciones disponibles hasta el corte solicitado."),
    "error_calculo": (422, "El histórico no permite generar el pronóstico solicitado."),
    "base_no_disponible": (503, "La base de datos no está disponible o no tiene el esquema necesario."),
    "error_persistencia": (503, "No se pudo completar el registro en la base de datos. La operación se revirtió."),
    "resultado_invalido": (500, "No se pudo obtener un pronóstico válido. No se guardaron resultados."),
    "error_interno": (500, "Ocurrió un error interno al procesar la solicitud."),
}


def _respuesta_error(codigo):
    if codigo not in ERRORES:
        codigo = "error_interno"
    estado, mensaje = ERRORES[codigo]
    return JSONResponse(status_code=estado, content={"detail": {"codigo": codigo, "mensaje": mensaje}})


class RutaPronosticos(APIRoute):
    """Traduce errores solo en este router; conserva los contratos de Sprint 1."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def manejar(request: Request):
            try:
                return await original(request)
            except RequestValidationError:
                return _respuesta_error("entrada_invalida")
            except (ErrorRegistroPronostico, ErrorConsultaPronostico) as error:
                return _respuesta_error(error.codigo)
            except (sqlite3.Error, OSError):
                return _respuesta_error("base_no_disponible")
            except Exception:
                return _respuesta_error("error_interno")

        return manejar


class SolicitudPronostico(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tecnica: str = Field(strict=True, min_length=1, json_schema_extra={"enum": list(TECNICAS)})
    fecha_corte: str = Field(strict=True, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class Serie(BaseModel):
    id: int
    store_nbr: int
    family: str


class MetadatosPronostico(BaseModel):
    id: int
    experimento_id: int
    tecnica: str
    fecha_corte: str
    tipo: str
    horizonte: int
    fecha_generacion: str
    inicio_entrenamiento: str
    fin_entrenamiento: str
    duracion_segundos: float | None


class ListaPronosticos(BaseModel):
    conjunto_id: int
    serie: Serie
    pronosticos: list[MetadatosPronostico]


class Detalle(BaseModel):
    paso: int
    fecha_objetivo: str
    venta_estimada: float


class ConsultaPronostico(BaseModel):
    conjunto_id: int
    serie: Serie
    pronostico: MetadatosPronostico
    detalles: list[Detalle]


class RegistroPronostico(BaseModel):
    conjunto_id: int
    serie_id: int
    experimento_id: int
    pronostico_id: int
    tecnica: str
    fecha_corte: str
    horizonte: int
    cantidad_detalles: int
    nuevo: bool
    reutilizado: bool


class DetalleError(BaseModel):
    codigo: str
    mensaje: str


class RespuestaError(BaseModel):
    detail: DetalleError


RESPUESTAS_ERROR = {codigo: {"model": RespuestaError, "description": descripcion}
                    for codigo, descripcion in (
                        (422, "Solicitud inválida"), (404, "Recurso no disponible en el conjunto activo"),
                        (409, "Conflicto o datos inconsistentes"), (503, "Base no disponible"),
                        (500, "Error interno"))}

router = APIRouter(prefix="/api/pronosticos", tags=["pronosticos"], route_class=RutaPronosticos)
Identificador = Annotated[int, ParametroRuta(gt=0)]


@router.get("/active/series/{serie_id}", response_model=ListaPronosticos, responses=RESPUESTAS_ERROR)
def listar(serie_id: Identificador, ruta_base: Path = Depends(obtener_ruta_base)):
    """Lista pronósticos ya almacenados de una serie del conjunto activo."""
    return listar_pronosticos(ruta_base, serie_id)


@router.get("/active/{pronostico_id}", response_model=ConsultaPronostico, responses=RESPUESTAS_ERROR)
def detalle(pronostico_id: Identificador, ruta_base: Path = Depends(obtener_ruta_base)):
    """Consulta un pronóstico completo y valida su integridad sin generar datos."""
    return consultar_pronostico(ruta_base, pronostico_id)


@router.post("/active/series/{serie_id}", status_code=201, response_model=RegistroPronostico,
             responses={**RESPUESTAS_ERROR, 200: {"model": RegistroPronostico, "description": "Pronóstico idéntico reutilizado"}})
def generar(serie_id: Identificador, solicitud: SolicitudPronostico,
            ruta_base: Path = Depends(obtener_ruta_base)):
    """Genera solo la serie, técnica y corte solicitados, delegando en HU004-D1."""
    resultado = registrar_pronostico(ruta_base, serie_id, solicitud.tecnica, solicitud.fecha_corte)
    contenido = RegistroPronostico(**asdict(resultado), reutilizado=resultado.reutilizado)
    return JSONResponse(status_code=201 if resultado.nuevo else 200, content=contenido.model_dump())
