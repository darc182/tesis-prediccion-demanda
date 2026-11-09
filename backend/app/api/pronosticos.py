"""API explícita de generación y consulta; no ejecuta operaciones al importar."""

from dataclasses import asdict
from pathlib import Path
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path as ParametroRuta, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services import configuracion_experimentos as configuraciones
from backend.app.services.consulta_pronosticos import (
    ErrorConsultaPronostico, consultar_configuracion_pronostico, consultar_pronostico, listar_pronosticos,
)
from backend.app.services.registro_pronosticos import (
    TECNICAS, ErrorRegistroPronostico, registrar_pronostico,
)


# Solo mensajes públicos fijos: nunca se devuelve str(error) de los servicios.
ERRORES = {
    "entrada_invalida": (422, "Revise los identificadores y el cuerpo: técnica y fecha de corte YYYY-MM-DD son obligatorias."),
    "configuracion_invalida": (422, "La configuración debe ser un objeto válido para la técnica solicitada y conservar las variables y reglas admitidas."),
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
            except RequestValidationError as error:
                if any(e["loc"][:2] == ("body", "configuracion") for e in error.errors()):
                    return _respuesta_error("configuracion_invalida")
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
    # model_fields_set distingue omisión de un objeto vacío explícito (inválido).
    # Any preserva los tipos JSON de los parámetros para la validación de F2.
    configuracion: dict[str, Any] = Field(
        default_factory=dict, strict=True,
        description="Opcional y no nulo. Objeto del contrato F2 con técnica interna coincidente; "
                    "admite los grupos canónicos 1/1 (seis variables) y 2/2 (siete variables). "
                    "Si se omite, se usan los defaults clásicos 1/1.",
        examples=[{"tecnica": "random_forest", "parametros": {"n_estimators": 200, "max_depth": 4, "random_state": 7}}],
    )


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


class ConfiguracionPredictiva(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_contrato: int
    tecnica: str
    parametros: dict[str, Any]  # conservar int/float/bool/null normalizados por F2
    variables: list[str]
    horizonte: int
    version_reglas: int


class ConfiguracionPronostico(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conjunto_id: int
    serie: Serie
    pronostico_id: int
    experimento_id: int
    fecha_corte: str
    configuracion: ConfiguracionPredictiva
    es_predeterminada: bool = Field(description="Equivalencia con los parámetros efectivos predeterminados; "
                                               "no indica cómo se formuló la solicitud original.")


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


@router.get("/active/{pronostico_id}/configuracion", response_model=ConfiguracionPronostico,
            responses=RESPUESTAS_ERROR)
def configuracion_guardada(pronostico_id: Identificador, ruta_base: Path = Depends(obtener_ruta_base)):
    """Consulta la configuración efectiva del experimento en el conjunto activo, sin entrenar."""
    return consultar_configuracion_pronostico(ruta_base, pronostico_id)


@router.post("/active/series/{serie_id}", status_code=201, response_model=RegistroPronostico,
             responses={**RESPUESTAS_ERROR, 200: {"model": RegistroPronostico, "description": "Pronóstico idéntico reutilizado"}})
def generar(serie_id: Identificador, solicitud: SolicitudPronostico,
            ruta_base: Path = Depends(obtener_ruta_base)):
    """Valida la opción F2/F3 y delega una única generación en el servicio."""
    if "configuracion" not in solicitud.model_fields_set:
        resultado = registrar_pronostico(ruta_base, serie_id, solicitud.tecnica, solicitud.fecha_corte)
    else:
        if solicitud.tecnica not in TECNICAS:
            return _respuesta_error("tecnica_invalida")
        try:
            efectiva = configuraciones.normalizar_configuracion(solicitud.configuracion)
            if efectiva["tecnica"] != solicitud.tecnica:
                raise ValueError("Técnicas diferentes.")
            if solicitud.tecnica != configuraciones.BASE:
                # Importación diferida: validar no construye ni entrena estimadores.
                from scripts.modelos_prediccion import normalizar_configuracion_entrenamiento
                efectiva = normalizar_configuracion_entrenamiento(solicitud.tecnica, efectiva)
        except (ValueError, TypeError):
            return _respuesta_error("configuracion_invalida")
        resultado = registrar_pronostico(
            ruta_base, serie_id, solicitud.tecnica, solicitud.fecha_corte,
            configuracion_solicitada=efectiva,
        )
    contenido = RegistroPronostico(**asdict(resultado), reutilizado=resultado.reutilizado)
    return JSONResponse(status_code=201 if resultado.nuevo else 200, content=contenido.model_dump())
