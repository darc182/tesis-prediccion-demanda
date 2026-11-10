"""API explícita de generación y consulta; no ejecuta operaciones al importar."""

from dataclasses import asdict
from copy import deepcopy
from datetime import date
from pathlib import Path
import sqlite3
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path as ParametroRuta, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services import configuracion_experimentos as configuraciones
from backend.app.services.consulta_filas_entrenamiento import (
    ErrorConsultaFilas, consultar_filas_entrenamiento,
)
from backend.app.services.consulta_pronosticos import (
    ErrorConsultaPronostico, consultar_configuracion_pronostico, consultar_pronostico, listar_pronosticos,
)
from backend.app.services.registro_pronosticos import (
    TECNICAS, ErrorRegistroPronostico, registrar_pronostico,
)
from backend.app.services.ejecucion_bateria_pronosticos import ErrorEjecucionBateria, ejecutar_bateria
from backend.app.services.validacion_csv import MAX_ENTERO_SQLITE


# Solo mensajes públicos fijos: nunca se devuelve str(error) de los servicios.
ERRORES = {
    "entrada_bateria_invalida": (422, "Indique únicamente conjunto_id y fecha_corte con tipos válidos."),
    "conjunto_invalido": (422, "El identificador de conjunto debe ser un entero positivo válido."),
    "conjunto_inactivo": (409, "El conjunto solicitado no es el activo. Actualice la selección."),
    "grupo_invalido": (422, "El grupo de variables debe ser clasico o ampliado."),
    "entrada_filas_invalida": (422, "Indique únicamente fecha_corte y grupo, una vez cada uno."),
    "historial_invalido": (409, "Las observaciones históricas presentan datos inconsistentes."),
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
                if self.name == "generar_bateria":
                    ubicaciones = [e["loc"] for e in error.errors()]
                    if ("path", "serie_id") in ubicaciones:
                        return _respuesta_error("serie_invalida")
                    if ("body", "conjunto_id") in ubicaciones:
                        return _respuesta_error("conjunto_invalido")
                    if ("body", "fecha_corte") in ubicaciones:
                        return _respuesta_error("corte_invalido")
                    return _respuesta_error("entrada_bateria_invalida")
                if self.name == "filas_entrenamiento":
                    ubicaciones = [e["loc"] for e in error.errors()]
                    if ("path", "serie_id") in ubicaciones:
                        return _respuesta_error("serie_invalida")
                    if ("query", "fecha_corte") in ubicaciones:
                        return _respuesta_error("corte_invalido")
                    return _respuesta_error("grupo_invalido")
                if any(e["loc"][:2] == ("body", "configuracion") for e in error.errors()):
                    return _respuesta_error("configuracion_invalida")
                return _respuesta_error("entrada_invalida")
            except (ErrorRegistroPronostico, ErrorConsultaPronostico, ErrorConsultaFilas) as error:
                return _respuesta_error(error.codigo)
            except ErrorEjecucionBateria as error:
                return _respuesta_error(error.codigo)
            except (sqlite3.Error, OSError):
                return _respuesta_error("base_no_disponible")
            except Exception:
                return _respuesta_error("error_interno")

        if self.name in {"filas_entrenamiento", "generar_bateria"}:
            async def sin_cache(request: Request):
                respuesta = await manejar(request)
                respuesta.headers["Cache-Control"] = "no-store"
                return respuesta
            return sin_cache
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


class FilasEntrenamiento(BaseModel):
    conjunto_id: int
    serie: Serie
    fecha_corte: str
    grupo: Literal["clasico", "ampliado"]
    version_contrato: int
    version_reglas: int
    variables: list[str]
    observaciones_historicas: int = Field(ge=0)
    filas_aptas: int = Field(ge=0)
    filas_descartadas: int = Field(ge=0)
    descartes_por_variable: dict[str, int] = Field(
        description="Filas con cada variable ausente. Los conteos pueden solaparse y no deben sumarse.")


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


class SolicitudBateria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conjunto_id: int = Field(strict=True, gt=0, le=MAX_ENTERO_SQLITE)
    fecha_corte: str = Field(strict=True, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

    @field_validator("fecha_corte")
    @classmethod
    def fecha_canonica(cls, valor):
        if date.fromisoformat(valor).isoformat() != valor:
            raise ValueError("Fecha no canónica.")
        return valor


class ElementoBateria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orden: int = Field(ge=1, le=5)
    conjunto_id: int
    serie_id: int
    fecha_corte: str
    tecnica: str
    configuracion: ConfiguracionPredictiva
    variables: list[str]
    version_contrato: int
    version_reglas: int
    horizonte: Literal[15]
    identidad_predictiva: str = Field(pattern=r"^[0-9a-f]{64}$")
    estado: Literal["creado", "reutilizado", "fallido", "no_ejecutado"]
    duracion_segundos: float | None = Field(ge=0, allow_inf_nan=False)
    experimento_id: int | None = Field(default=None, strict=True, gt=0)
    pronostico_id: int | None = Field(default=None, strict=True, gt=0)
    error: DetalleError | None = None


class ResumenBateria(BaseModel):
    creado: int = Field(ge=0, le=5)
    reutilizado: int = Field(ge=0, le=5)
    fallido: int = Field(ge=0, le=5)
    no_ejecutado: int = Field(ge=0, le=5)


class ResultadoBateria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conjunto_id: int
    serie_id: int
    fecha_corte: str
    horizonte: Literal[15]
    estado: Literal["completa", "parcial", "interrumpida"]
    contexto_verificado: bool
    resultados: list[ElementoBateria] = Field(min_length=5, max_length=5)
    resumen: ResumenBateria
    interrupcion: DetalleError | None
    duracion_segundos: float = Field(ge=0, allow_inf_nan=False)


def _error_publico_bateria(error):
    codigo = error.get("codigo")
    publico = ErrorEjecucionBateria(codigo if type(codigo) is str else "error_interno")
    return DetalleError(codigo=publico.codigo, mensaje=str(publico))


def _bateria_publica(resultado):
    """Proyección por campos: no serializa auditoría ni modifica el servicio."""
    elementos = []
    for origen in resultado["resultados"]:
        cfg = origen["configuracion"]
        parametros = configuraciones.configuracion_predeterminada(cfg["tecnica"])["parametros"]
        configuracion = ConfiguracionPredictiva(
            tecnica=cfg["tecnica"], version_contrato=cfg["version_contrato"],
            version_reglas=cfg["version_reglas"], horizonte=cfg["horizonte"],
            variables=list(cfg["variables"]),
            parametros={k: deepcopy(cfg["parametros"][k]) for k in parametros})
        datos = {k: deepcopy(origen[k]) for k in (
            "orden", "conjunto_id", "serie_id", "fecha_corte", "tecnica", "variables",
            "version_contrato", "version_reglas", "horizonte", "identidad_predictiva",
            "estado", "duracion_segundos")}
        datos["configuracion"] = configuracion
        if origen["estado"] in {"creado", "reutilizado"}:
            datos.update({k: origen[k] for k in ("experimento_id", "pronostico_id")})
        elif origen["estado"] == "fallido":
            datos["error"] = _error_publico_bateria(origen["error"])
        elementos.append(ElementoBateria(**datos))
    return ResultadoBateria(
        **{k: resultado[k] for k in ("conjunto_id", "serie_id", "fecha_corte", "horizonte",
                                    "estado", "contexto_verificado", "duracion_segundos")},
        resultados=elementos,
        resumen=ResumenBateria(**{k: resultado["resumen"][k]
                                for k in ("creado", "reutilizado", "fallido", "no_ejecutado")}),
        interrupcion=None if resultado["interrupcion"] is None else _error_publico_bateria(resultado["interrupcion"]))


RESPUESTAS_ERROR = {codigo: {"model": RespuestaError, "description": descripcion}
                    for codigo, descripcion in (
                        (422, "Solicitud inválida"), (404, "Recurso no disponible en el conjunto activo"),
                        (409, "Conflicto o datos inconsistentes"), (503, "Base no disponible"),
                        (500, "Error interno"))}

router = APIRouter(prefix="/api/pronosticos", tags=["pronosticos"], route_class=RutaPronosticos)
Identificador = Annotated[int, ParametroRuta(gt=0)]


def _entero_ruta_bateria(valor):
    if type(valor) is not str or not valor.isascii() or not valor.isdecimal():
        raise ValueError("Identificador no entero.")
    return valor


@router.post("/active/series/{serie_id}/bateria", response_model=ResultadoBateria,
             response_model_exclude_unset=True, responses=RESPUESTAS_ERROR)
def generar_bateria(
    serie_id: Annotated[int, ParametroRuta(gt=0, le=MAX_ENTERO_SQLITE), BeforeValidator(_entero_ruta_bateria)],
    solicitud: SolicitudBateria, ruta_base: Path = Depends(obtener_ruta_base),
):
    """Ejecuta explícitamente cinco configuraciones, de forma síncrona.

    HTTP 200 conserva completa/parcial/interrumpida; no es una transacción
    global. Reutilizar puede volver a entrenar. No hay cancelación inmediata:
    una desconexión no acredita ni parada ni finalización. Los registros
    individuales confirmados pueden consultarse por GET si siguen activos.
    No se garantiza que baterías simultáneas tengan el mismo reparto de estados.
    Respuestas y errores llevan Cache-Control: no-store.
    """
    resultado = ejecutar_bateria(ruta_base, solicitud.conjunto_id, serie_id, solicitud.fecha_corte)
    return _bateria_publica(resultado)


@router.get("/active/series/{serie_id}/filas-entrenamiento", response_model=FilasEntrenamiento,
            responses=RESPUESTAS_ERROR)
def filas_entrenamiento(
    request: Request, serie_id: Identificador,
    fecha_corte: Annotated[str, Query(description="Corte de validación autorizado en formato YYYY-MM-DD.")],
    grupo: Annotated[Literal["clasico", "ampliado"], Query()],
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Consulta disponibilidad de ejemplos de regresión sin entrenar ni escribir."""
    if (set(request.query_params) != {"fecha_corte", "grupo"}
            or any(len(request.query_params.getlist(k)) != 1 for k in ("fecha_corte", "grupo"))):
        return _respuesta_error("entrada_filas_invalida")
    return consultar_filas_entrenamiento(ruta_base, serie_id, fecha_corte, grupo)


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
