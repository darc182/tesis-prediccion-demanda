"""Consulta actual y registro explícito de evaluaciones; sin operaciones al importar."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path as ParametroRuta, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services.consulta_evaluaciones import ErrorConsultaEvaluacion, consultar_evaluacion
from backend.app.services.registro_evaluaciones import ErrorRegistroEvaluacion, registrar_evaluacion
from backend.app.services.validacion_csv import MAX_ENTERO_SQLITE


# Solo se publican estos mensajes; los mensajes recibidos del servicio se descartan.
ERRORES = {
    "pronostico_invalido": (422, "El identificador de pronóstico debe ser un entero positivo dentro del límite SQLite."),
    "entrada_invalida": (422, "Esta consulta no admite cuerpo ni parámetros de consulta adicionales."),
    "tipo_no_evaluable": (422, "Solo se consultan métricas de pronósticos de validación."),
    "corte_no_autorizado": (422, "El corte no pertenece a las tres ventanas de validación autorizadas."),
    "pronostico_inexistente": (404, "El pronóstico solicitado no existe."),
    "pronostico_inactivo": (404, "El pronóstico no pertenece al conjunto de datos activo."),
    "sin_conjunto_activo": (409, "No existe un conjunto de datos activo para esta consulta."),
    "datos_inconsistentes": (409, "Los registros del pronóstico o de sus ventas observadas son inconsistentes."),
    "base_no_disponible": (503, "La base de datos no está disponible o no tiene el esquema necesario."),
    "error_interno": (500, "No se pudo completar la consulta de métricas."),
}


ERRORES_REGISTRO = {
    **ERRORES,
    "evaluacion_desactualizada": (409, "La evaluación guardada no coincide con los datos actuales; no se sobrescribió."),
    "error_persistencia": (503, "No se pudo completar el registro de la evaluación."),
    "error_interno": (500, "No se pudo completar la solicitud de registro de evaluación."),
}


def _respuesta_error(codigo, *, registro=False):
    errores = ERRORES_REGISTRO if registro else ERRORES
    if type(codigo) is not str or codigo not in errores:
        codigo = "error_interno"
    estado, mensaje = errores[codigo]
    return JSONResponse(status_code=estado, content={"detail": {"codigo": codigo, "mensaje": mensaje}})


class RutaEvaluaciones(APIRoute):
    """Control de entrada, errores y caché limitado al router de evaluaciones."""

    def get_route_handler(self):
        original = super().get_route_handler()
        es_registro = "POST" in self.methods

        async def manejar(request: Request):
            try:
                if request.query_params or await request.body():
                    respuesta = _respuesta_error("entrada_invalida", registro=es_registro)
                else:
                    respuesta = await original(request)
            except RequestValidationError:
                respuesta = _respuesta_error("pronostico_invalido", registro=es_registro)
            except ErrorConsultaEvaluacion as error:
                respuesta = _respuesta_error(error.codigo, registro=es_registro)
            except ErrorRegistroEvaluacion as error:
                respuesta = _respuesta_error(error.codigo if es_registro else "error_interno", registro=es_registro)
            except Exception:
                respuesta = _respuesta_error("error_interno", registro=es_registro)
            respuesta.headers["Cache-Control"] = "no-store"
            return respuesta

        return manejar


class ModeloEvaluaciones(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SerieEvaluacion(ModeloEvaluaciones):
    id: int = Field(gt=0)
    store_nbr: int = Field(gt=0)
    family: str


class DetalleDiarioEvaluacion(ModeloEvaluaciones):
    paso: int = Field(ge=1, le=15)
    fecha_objetivo: str = Field(description="Fecha objetivo YYYY-MM-DD, en orden cronológico.")
    venta_estimada: float = Field(ge=0)
    venta_real: float | None = Field(ge=0, description="Venta observada; null si la fecha no tiene registro.")
    error_absoluto: float | None = Field(ge=0, description="Error absoluto; null si falta la venta real.")


class MetricasEvaluacion(ModeloEvaluaciones):
    fecha_corte: str
    horizonte: Literal[15]
    observaciones_evaluadas: int = Field(ge=0, le=15)
    fechas_sin_registro: int = Field(ge=0, le=15)
    cobertura: float = Field(ge=0, le=1, description="Proporción de observaciones evaluadas respecto a quince fechas.")
    definicion_cobertura: str
    mae: float | None = Field(ge=0, description="Error absoluto medio sobre las observaciones disponibles.")
    wape: float | None = Field(ge=0, description="Error porcentual absoluto ponderado, no porcentaje de exactitud. "
                                               "Puede superar 100; null sin reales o si suman cero.")
    rmse: float | None = Field(ge=0, description="Raíz del error cuadrático medio sobre las mismas fechas observadas.")
    detalles: list[DetalleDiarioEvaluacion] = Field(min_length=15, max_length=15)
    fechas_ausentes: list[str] = Field(max_length=15)


class ConsultaEvaluacion(ModeloEvaluaciones):
    conjunto_id: int = Field(gt=0)
    serie: SerieEvaluacion
    pronostico_id: int = Field(gt=0)
    experimento_id: int = Field(gt=0)
    tecnica: str
    fecha_corte: str
    horizonte: Literal[15]
    estado_cobertura: Literal["completa", "parcial", "sin_observaciones"] = Field(
        description="Disponibilidad de observaciones, sin interpretar la calidad predictiva.")
    alcance_metricas: str
    evaluacion: MetricasEvaluacion


class RegistroEvaluacion(ConsultaEvaluacion):
    estado_registro: Literal["creado", "reutilizado", "no_guardado"]
    hubo_escritura: bool
    motivo: Literal["sin_observaciones"] | None


class DetalleErrorEvaluacion(ModeloEvaluaciones):
    codigo: str
    mensaje: str


class ErrorEvaluacion(ModeloEvaluaciones):
    detail: DetalleErrorEvaluacion


CABECERA_CACHE = {"Cache-Control": {
    "description": "La respuesta no debe almacenarse en caché.", "schema": {"type": "string", "const": "no-store"}}}
RESPUESTAS = {
    200: {"description": "Cálculo actual sobre las observaciones disponibles.", "headers": CABECERA_CACHE},
    **{estado: {"model": ErrorEvaluacion, "description": descripcion, "headers": CABECERA_CACHE}
       for estado, descripcion in (
           (422, "pronostico_invalido, entrada_invalida, tipo_no_evaluable o corte_no_autorizado."),
           (404, "pronostico_inexistente o pronostico_inactivo."),
           (409, "sin_conjunto_activo o datos_inconsistentes."),
           (503, "base_no_disponible."), (500, "error_interno; mensaje público genérico."))},
}


def _entero_ruta(valor):
    if type(valor) is not str or not valor.isascii() or not valor.isdecimal():
        raise ValueError("Identificador no entero.")
    return valor


RESPUESTAS_REGISTRO = {
    **{estado: contenido for estado, contenido in RESPUESTAS.items() if estado != 200},
    200: {"model": RegistroEvaluacion, "description": "Evaluación reutilizada, o no_guardado por falta de observaciones; "
                                                    "no implica que se haya insertado una fila.",
          "headers": CABECERA_CACHE},
    201: {"description": "Nueva evaluación insertada y confirmada por B1.", "headers": CABECERA_CACHE},
    409: {**RESPUESTAS[409], "description": "sin_conjunto_activo, datos_inconsistentes o evaluacion_desactualizada; "
                                          "sin sobrescribir ni eliminar la evaluación anterior."},
    503: {**RESPUESTAS[503], "description": "base_no_disponible o error_persistencia."},
}


router = APIRouter(prefix="/api/evaluaciones", tags=["evaluaciones"], route_class=RutaEvaluaciones)


@router.get("/active/{pronostico_id}", response_model=ConsultaEvaluacion, responses=RESPUESTAS,
            summary="Consultar métricas actuales de un pronóstico de validación")
def evaluacion_actual(
    pronostico_id: Annotated[int, ParametroRuta(gt=0, le=MAX_ENTERO_SQLITE), BeforeValidator(_entero_ruta)],
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Calcula MAE, WAPE, RMSE y cobertura desde las ventas actualmente observadas.

    Delega una sola llamada a consultar_evaluacion. No admite cuerpo ni
    parámetros de consulta adicionales. No imputa ausencias ni redondea métricas.
    Con cobertura parcial, las métricas corresponden a observaciones disponibles.

    Este GET no consulta ni utiliza la tabla evaluacion para decidir vigencia,
    no guarda resultados y puede usarse aunque nunca se haya ejecutado B1.
    Si las ventas cambian después de un registro, devuelve el cálculo actual
    sin modificar la fila guardada. No informa estados de registro o vigencia.
    Los éxitos y errores de la consulta llevan Cache-Control: no-store.
    """
    return consultar_evaluacion(ruta_base, pronostico_id)


@router.post("/active/{pronostico_id}", status_code=201, response_model=RegistroEvaluacion,
             responses=RESPUESTAS_REGISTRO, summary="Registrar explícitamente una evaluación de validación")
def guardar_evaluacion(
    pronostico_id: Annotated[int, ParametroRuta(gt=0, le=MAX_ENTERO_SQLITE), BeforeValidator(_entero_ruta)],
    respuesta: Response,
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Delega exactamente una vez en registrar_evaluacion, sin SQL ni cálculo propio.

    Acción explícita sin cuerpo ni parámetros adicionales. B1 guarda coberturas
    completas o parciales y devuelve los conteos y fechas actuales. HTTP 201
    significa creado; HTTP 200 puede ser reutilizado o no_guardado. Sin reales
    no se guarda y las métricas son null. WAPE indefinido se conserva como null.
    Los conflictos devuelven 409 sin sobrescribir ni eliminar filas anteriores.

    Reutilizar métricas numéricamente iguales no acredita identidad histórica
    ni vigencia permanente. El GET conserva su cálculo actual independiente.
    No repetir solicitudes automáticamente ante un fallo de red: la transacción
    podría haberse confirmado aunque el cliente no haya recibido la respuesta.
    Éxitos y errores gestionados incluyen Cache-Control: no-store.
    """
    resultado = RegistroEvaluacion.model_validate(registrar_evaluacion(ruta_base, pronostico_id))
    respuesta.status_code = 201 if resultado.estado_registro == "creado" else 200
    return resultado
