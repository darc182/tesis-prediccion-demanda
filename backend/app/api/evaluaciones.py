"""Consulta actual y registro explícito de evaluaciones; sin operaciones al importar."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path as ParametroRuta, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from backend.app.api.datasets import obtener_ruta_base
from backend.app.services.comparacion_evaluaciones import ErrorComparacionEvaluaciones, comparar_evaluaciones
from backend.app.services.consolidacion_evaluaciones import ErrorConsolidacionEvaluaciones, consolidar_evaluaciones
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


ERRORES_COMPARACION = {
    **ERRORES,
    "entrada_invalida": (422, "Indique únicamente pronostico_id repetido dos o tres veces, con IDs distintos y sin cuerpo."),
    "pronostico_invalido": (422, "Cada pronostico_id debe ser un entero decimal ASCII positivo dentro del límite SQLite."),
    "pronosticos_no_comparables": (409, "Los pronósticos no comparten conjunto, serie, periodo y observaciones reales."),
    "error_interno": (500, "No se pudo completar la comparación de evaluaciones."),
}


ERRORES_CONSOLIDACION = {
    **ERRORES,
    "entrada_invalida": (422, "Indique únicamente dos o tres parámetros grupo, cada uno con tres IDs distintos separados por comas, sin cuerpo."),
    "pronostico_invalido": (422, "Cada ID del grupo debe ser un entero decimal ASCII positivo dentro del límite SQLite."),
    "configuraciones_no_equivalentes": (409, "Los tres pronósticos de un grupo no comparten identidad predictiva."),
    "configuraciones_duplicadas": (409, "Los grupos deben representar configuraciones predictivas diferentes."),
    "versiones_incompatibles": (409, "Las versiones de ejecución de un grupo son incompatibles o no acreditables."),
    "pronosticos_no_comparables": (409, "Los pronósticos no comparten conjunto, serie, cortes ordenados y observaciones reales."),
    "resultado_no_finito": (409, "Las sumas o métricas consolidadas exceden el rango numérico representable."),
    "error_interno": (500, "No se pudo completar la consolidación temporal de evaluaciones."),
}


def _respuesta_error(codigo, *, registro=False, comparacion=False, consolidacion=False):
    errores = (ERRORES_CONSOLIDACION if consolidacion else ERRORES_COMPARACION if comparacion
               else ERRORES_REGISTRO if registro else ERRORES)
    if type(codigo) is not str or codigo not in errores:
        codigo = "error_interno"
    estado, mensaje = errores[codigo]
    return JSONResponse(status_code=estado, content={"detail": {"codigo": codigo, "mensaje": mensaje}})


class RutaEvaluaciones(APIRoute):
    """Control de entrada, errores y caché limitado al router de evaluaciones."""

    def get_route_handler(self):
        original = super().get_route_handler()
        es_registro = "POST" in self.methods
        es_comparacion = self.name == "comparacion_actual"
        es_consolidacion = self.name == "consolidacion_actual"
        opciones_error = {"registro": es_registro, "comparacion": es_comparacion, "consolidacion": es_consolidacion}

        async def manejar(request: Request):
            try:
                if es_consolidacion:
                    query_invalida = bool(set(request.query_params) - {"grupo"})
                elif es_comparacion:
                    query_invalida = bool(set(request.query_params) - {"pronostico_id"})
                else:
                    query_invalida = bool(request.query_params)
                if query_invalida or await request.body():
                    respuesta = _respuesta_error("entrada_invalida", **opciones_error)
                else:
                    respuesta = await original(request)
            except RequestValidationError as error:
                codigo = "pronostico_invalido"
                if es_consolidacion or (es_comparacion and not any(
                    e["loc"][:2] == ("query", "pronostico_id") and len(e["loc"]) > 2 for e in error.errors()
                )):
                    codigo = "entrada_invalida"
                respuesta = _respuesta_error(codigo, **opciones_error)
            except ErrorConsultaEvaluacion as error:
                respuesta = _respuesta_error("error_interno" if es_comparacion or es_consolidacion else error.codigo,
                                             **opciones_error)
            except ErrorRegistroEvaluacion as error:
                respuesta = _respuesta_error(error.codigo if es_registro else "error_interno",
                                             **opciones_error)
            except ErrorComparacionEvaluaciones as error:
                respuesta = _respuesta_error(error.codigo if es_comparacion else "error_interno",
                                             **opciones_error)
            except ErrorConsolidacionEvaluaciones as error:
                respuesta = _respuesta_error(error.codigo if es_consolidacion else "error_interno",
                                             **opciones_error)
            except Exception:
                respuesta = _respuesta_error("error_interno", **opciones_error)
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


class ResultadoComparacion(ModeloEvaluaciones):
    pronostico_id: int = Field(gt=0)
    experimento_id: int = Field(gt=0)
    tecnica: str
    evaluacion: MetricasEvaluacion


class ComparacionEvaluaciones(ModeloEvaluaciones):
    conjunto_id: int = Field(gt=0)
    serie: SerieEvaluacion
    fecha_corte: str
    horizonte: Literal[15]
    estado_cobertura: Literal["completa", "parcial", "sin_observaciones"]
    alcance_metricas: str
    observaciones_evaluadas: int = Field(ge=0, le=15)
    cobertura: float = Field(ge=0, le=1)
    fechas_ausentes: list[str] = Field(max_length=15)
    resultados: list[ResultadoComparacion] = Field(min_length=2, max_length=3)


class ConfiguracionConsolidacion(ModeloEvaluaciones):
    version_contrato: int = Field(ge=1)
    tecnica: Literal["estacional_semanal_recursiva", "random_forest", "hist_gradient_boosting"]
    parametros: dict[str, int | float | bool | str | None] = Field(
        description="Parámetros completos F2; conserva números, booleanos, cadenas y null con sus tipos JSON.")
    variables: list[str]
    horizonte: Literal[15]
    version_reglas: int = Field(ge=1)


class VersionesBaseConsolidacion(ModeloEvaluaciones):
    python: str
    pandas: str


class VersionesRegresorConsolidacion(VersionesBaseConsolidacion):
    scikit_learn: str


class RMSEEscaladoConsolidacion(ModeloEvaluaciones):
    escala_errores: float = Field(ge=0)
    suma_cuadrados_normalizados: float = Field(ge=0, le=45)


class MetricasConsolidadas(ModeloEvaluaciones):
    mae: float | None = Field(ge=0)
    wape: float | None = Field(ge=0, description="Porcentaje global; no se promedian los WAPE por corte.")
    rmse: float | None = Field(ge=0)
    suma_errores_absolutos: float = Field(ge=0)
    suma_ventas_reales: float = Field(ge=0)
    observaciones_evaluadas: int = Field(ge=0, le=45)
    fechas_sin_registro: int = Field(ge=0, le=45)
    cobertura: float = Field(ge=0, le=1)
    definicion_cobertura: str
    estado_cobertura: Literal["completa", "parcial", "sin_observaciones"]
    alcance_metricas: str
    rmse_escalado: RMSEEscaladoConsolidacion = Field(
        description="RMSE = escala_errores * sqrt(suma_cuadrados_normalizados / observaciones_evaluadas), si hay pares.")


class AusenciasCorteConsolidacion(ModeloEvaluaciones):
    fecha_corte: Literal["2017-06-16", "2017-07-01", "2017-07-16"]
    fechas_ausentes: list[str] = Field(max_length=15)


class ResultadoConsolidacion(ModeloEvaluaciones):
    tecnica: Literal["estacional_semanal_recursiva", "random_forest", "hist_gradient_boosting"]
    identidad_configuracion: str = Field(pattern=r"^[0-9a-f]{64}$", description="Identidad predictiva SHA256 de F2.")
    configuracion: ConfiguracionConsolidacion
    versiones_ejecucion: VersionesBaseConsolidacion | VersionesRegresorConsolidacion
    evaluaciones: list[ConsultaEvaluacion] = Field(min_length=3, max_length=3)
    consolidado: MetricasConsolidadas


class ConsolidacionEvaluaciones(ModeloEvaluaciones):
    conjunto_id: int = Field(gt=0)
    serie: SerieEvaluacion
    cortes: list[Literal["2017-06-16", "2017-07-01", "2017-07-16"]] = Field(min_length=3, max_length=3)
    horizonte_por_corte: Literal[15]
    fechas_esperadas_total: Literal[45]
    observaciones_evaluadas_total: int = Field(ge=0, le=45)
    cobertura_total: float = Field(ge=0, le=1)
    estado_cobertura: Literal["completa", "parcial", "sin_observaciones"]
    alcance_metricas: str
    fechas_ausentes_por_corte: list[AusenciasCorteConsolidacion] = Field(min_length=3, max_length=3)
    resultados: list[ResultadoConsolidacion] = Field(min_length=2, max_length=3)


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


IdentificadorComparacion = Annotated[int, Field(gt=0, le=MAX_ENTERO_SQLITE), BeforeValidator(_entero_ruta)]


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


RESPUESTAS_COMPARACION = {
    **RESPUESTAS,
    200: {"description": "Comparación actual de dos o tres pronósticos sobre las mismas observaciones.",
          "headers": CABECERA_CACHE},
    409: {**RESPUESTAS[409], "description": "sin_conjunto_activo, pronosticos_no_comparables o datos_inconsistentes; "
                                          "sin resultados parciales."},
}


RESPUESTAS_CONSOLIDACION = {
    200: {"description": "Consolidado actual de dos o tres configuraciones sobre tres cortes; sin escrituras.",
          "headers": CABECERA_CACHE},
    **{estado: {"model": ErrorEvaluacion, "description": descripcion, "headers": CABECERA_CACHE}
       for estado, descripcion in (
           (422, "entrada_invalida, pronostico_invalido, tipo_no_evaluable o corte_no_autorizado."),
           (404, "pronostico_inexistente o pronostico_inactivo."),
           (409, "sin_conjunto_activo, configuraciones_no_equivalentes, configuraciones_duplicadas, "
                 "versiones_incompatibles, pronosticos_no_comparables, datos_inconsistentes o resultado_no_finito."),
           (503, "base_no_disponible."), (500, "error_interno; mensaje público genérico."))},
}


router = APIRouter(prefix="/api/evaluaciones", tags=["evaluaciones"], route_class=RutaEvaluaciones)


@router.get("/consolidacion/active", response_model=ConsolidacionEvaluaciones, responses=RESPUESTAS_CONSOLIDACION,
            summary="Consolidar evaluaciones actuales de dos o tres configuraciones sobre tres cortes")
def consolidacion_actual(
    grupo: Annotated[list[str], Query(
        min_length=2, max_length=3,
        description="Repita grupo dos o tres veces. Cada valor contiene tres IDs ASCII positivos separados solo por "
                    "comas, en orden de cortes 2017-06-16, 2017-07-01, 2017-07-16. Sin espacios ni IDs repetidos "
                    f"entre grupos. Cada ID no supera {MAX_ENTERO_SQLITE}. Se conserva el orden solicitado.",
        examples=[["101,102,103", "201,202,203"], ["101,102,103", "201,202,203", "301,302,303"]])],
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Delega exactamente una llamada a consolidar_evaluaciones(ruta_base, grupos).

    Solo admite grupo repetido dos o tres veces, sin cuerpo ni parámetros
    adicionales. Los seis o nueve IDs son enteros decimales ASCII positivos,
    distintos y dentro del límite SQLite. Cada grupo sigue los tres cortes en
    orden; el servicio verifica identidad, versiones y observaciones equivalentes.

    E2-A calcula en una única instantánea SQLite de solo lectura; no utiliza la
    tabla evaluacion almacenada. No registra, entrena, genera ni selecciona
    ganadores. La cobertura consolidada usa las 45 fechas esperadas. Con cobertura
    parcial las métricas solo describen las observaciones disponibles. Sin reales
    las tres métricas son null; con suma real cero WAPE es null. No se redondean
    números ni se promedian porcentajes WAPE por corte. Las versiones de ejecución
    acreditadas se conservan por grupo, sin inventar scikit_learn para la línea base.
    Éxitos y errores gestionados incluyen Cache-Control: no-store.
    """
    grupos = []
    limite = str(MAX_ENTERO_SQLITE)
    for valor in grupo:
        partes = valor.split(",")
        if len(partes) != 3:
            return _respuesta_error("entrada_invalida", consolidacion=True)
        ids = []
        for parte in partes:
            # Comprobar el rango como texto evita convertir enteros arbitrariamente
            # largos; los ceros iniciales válidos no alteran identidad ni duplicados.
            numero = parte.lstrip("0")
            if (not parte.isascii() or not parte.isdecimal() or not numero
                    or len(numero) > len(limite) or (len(numero) == len(limite) and numero > limite)):
                return _respuesta_error("pronostico_invalido", consolidacion=True)
            ids.append(int(numero))
        grupos.append(ids)
    seleccion = [i for ids in grupos for i in ids]
    if len(set(seleccion)) != len(seleccion):
        return _respuesta_error("entrada_invalida", consolidacion=True)
    return ConsolidacionEvaluaciones.model_validate(consolidar_evaluaciones(ruta_base, grupos))


@router.get("/comparacion/active", response_model=ComparacionEvaluaciones, responses=RESPUESTAS_COMPARACION,
            summary="Comparar métricas actuales de dos o tres pronósticos")
def comparacion_actual(
    pronostico_id: Annotated[list[IdentificadorComparacion], Query(
        min_length=2, max_length=3, json_schema_extra={"uniqueItems": True},
        description="Repita pronostico_id dos o tres veces. IDs decimales ASCII positivos, distintos y dentro del "
                    "límite SQLite; se conserva el orden solicitado.", examples=[[102, 101], [103, 101, 102]])],
    ruta_base: Path = Depends(obtener_ruta_base),
):
    """Delega una sola llamada a comparar_evaluaciones con los IDs en el orden recibido.

    Solo admite pronostico_id repetido dos o tres veces, sin cuerpo ni parámetros
    adicionales. D1 verifica conjunto activo, serie, corte, horizonte y fechas
    reales equivalentes, y calcula en una única instantánea SQLite de solo lectura.
    Una incompatibilidad rechaza toda la comparación con HTTP 409.

    Devuelve métricas actuales, incluidas coberturas parciales y fechas ausentes.
    Sin observaciones las métricas son null; no se evalúa el desempeño predictivo.
    No lee evaluaciones guardadas ni configuraciones privadas, no registra datos,
    no entrena y no selecciona ganadores. No redondea ni sustituye null por cero.
    Éxitos y errores gestionados incluyen Cache-Control: no-store.
    """
    if len(set(pronostico_id)) != len(pronostico_id):
        return _respuesta_error("entrada_invalida", comparacion=True)
    return comparar_evaluaciones(ruta_base, pronostico_id)


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
