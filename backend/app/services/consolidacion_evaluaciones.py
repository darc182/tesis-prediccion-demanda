"""Consolidación temporal de configuraciones, sin persistencia ni selección."""

import json
from math import fsum, isfinite, nextafter, sqrt
from pathlib import Path
import re

from backend.app.services import configuracion_experimentos as configuraciones
from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import consulta_pronosticos as pronosticos
from backend.app.services import metricas_evaluacion as metricas


CORTES = tuple(c.isoformat() for c in metricas.CORTES_AUTORIZADOS)
FECHAS_ESPERADAS = 3 * metricas.HORIZONTE_DIAS
_MINIMO_FLOAT = nextafter(0.0, 1.0)
_VERSION_EJECUCION = re.compile(
    r"[0-9]+(?:\.[0-9]+)+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?",
    flags=re.ASCII,
)
_MENSAJES = {
    "entrada_invalida": "Seleccione dos o tres grupos de tres identificadores distintos.",
    "pronostico_invalido": "Cada identificador debe ser un entero Python positivo dentro del límite SQLite.",
    "pronostico_inexistente": "Uno de los pronósticos seleccionados no existe.",
    "pronostico_inactivo": "Uno de los pronósticos no pertenece al conjunto activo.",
    "sin_conjunto_activo": "No existe un conjunto de datos activo.",
    "tipo_no_evaluable": "Solo se consolidan pronósticos de validación.",
    "corte_no_autorizado": "Uno de los cortes no pertenece a las ventanas autorizadas.",
    "configuraciones_no_equivalentes": "Los tres pronósticos del grupo no comparten identidad predictiva.",
    "configuraciones_duplicadas": "Los grupos deben representar configuraciones predictivas diferentes.",
    "versiones_incompatibles": "Las versiones de ejecución del grupo son incompatibles o no acreditables.",
    "pronosticos_no_comparables": "Los pronósticos no comparten serie, cortes ordenados y observaciones reales.",
    "datos_inconsistentes": "Los registros necesarios para la consolidación son inconsistentes.",
    "resultado_no_finito": "Las sumas o métricas consolidadas exceden el rango numérico representable.",
    "base_no_disponible": "La base de datos no está disponible o no tiene el esquema necesario.",
}


class ErrorConsolidacionEvaluaciones(ValueError):
    """Código estable y mensaje constante, sin incorporar texto del almacenamiento."""

    def __init__(self, codigo: str):
        self.codigo = codigo if type(codigo) is str and codigo in _MENSAJES else "datos_inconsistentes"
        super().__init__(_MENSAJES[self.codigo])


def _seleccion(grupos):
    if not isinstance(grupos, (list, tuple)) or len(grupos) not in (2, 3):
        raise ErrorConsolidacionEvaluaciones("entrada_invalida")
    if any(not isinstance(g, (list, tuple)) or len(g) != 3 for g in grupos):
        raise ErrorConsolidacionEvaluaciones("entrada_invalida")
    seleccion = tuple(tuple(g) for g in grupos)
    ids = [i for grupo in seleccion for i in grupo]
    if any(type(i) is not int or not 0 < i <= 2**63 - 1 for i in ids):
        raise ErrorConsolidacionEvaluaciones("pronostico_invalido")
    if len(set(ids)) != len(ids):
        raise ErrorConsolidacionEvaluaciones("entrada_invalida")
    return seleccion


def _versiones(registro, tecnica):
    """Igualdad exacta de versiones declaradas, sin inferir el entorno instalado.

    Python y pandas siempre son relevantes; scikit_learn solo para regresores.
    Se admiten versiones numéricas con sufijos de desarrollo/locales habituales.
    Otros metadatos no se publican ni acreditan equivalencia de entornos.
    """
    versiones = registro.get("versiones")
    campos = ("python", "pandas") + (() if tecnica == configuraciones.BASE else ("scikit_learn",))
    if type(versiones) is not dict or any(
        type(versiones.get(k)) is not str
        or _VERSION_EJECUCION.fullmatch(versiones[k]) is None
        for k in campos
    ):
        raise ErrorConsolidacionEvaluaciones("versiones_incompatibles")
    return {k: versiones[k] for k in campos}


def _configuracion(conexion, evaluacion):
    fila = conexion.execute(
        "SELECT configuracion_json FROM experimento WHERE id = ?",
        (evaluacion["experimento_id"],),
    ).fetchone()
    if fila is None:
        raise ErrorConsolidacionEvaluaciones("datos_inconsistentes")
    try:
        if type(fila["configuracion_json"]) is not str:
            raise ValueError("El registro debe ser texto JSON")
        registro = json.loads(fila["configuracion_json"], object_pairs_hook=pronosticos._objeto_json_univoco)
        json.dumps(registro, allow_nan=False)  # también rechaza infinitos de la auditoría
        efectiva = configuraciones.recuperar_configuracion(registro)
        if efectiva["tecnica"] != evaluacion["tecnica"] or efectiva["horizonte"] != evaluacion["horizonte"]:
            raise ValueError("Cabecera contradictoria")
        identidad = configuraciones.identidad_configuracion(efectiva)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ErrorConsolidacionEvaluaciones("datos_inconsistentes") from error
    return efectiva, identidad, _versiones(registro, efectiva["tecnica"])


def _condiciones(evaluacion):
    # None identifica ausencia; una venta cero sí es una observación.
    return (evaluacion["conjunto_id"], evaluacion["serie"], evaluacion["fecha_corte"],
            evaluacion["horizonte"], tuple((d["fecha_objetivo"], d["venta_real"])
                                         for d in evaluacion["evaluacion"]["detalles"]))


def _cobertura(cantidad):
    if cantidad == FECHAS_ESPERADAS:
        return "completa", "Métricas sobre las 45 observaciones de los tres horizontes."
    if cantidad:
        return "parcial", "Métricas sobre observaciones disponibles; no describen las fechas sin registro."
    return "sin_observaciones", "Sin observaciones reales; no se puede evaluar el desempeño predictivo."


def _consolidado(evaluaciones):
    """Acumula contribuciones diarias A1; nunca promedia métricas por corte.

    RMSE = escala * sqrt(suma_cuadrados_normalizados / N). Su representación
    auxiliar evita publicar cuadrados gigantes. Las sumas se mantienen como
    float finito, igual que A1; no se rescatan sumas desbordadas. Se rechaza
    también una media positiva inferior al mínimo subnormal representable.
    """
    pares = [d for e in evaluaciones for d in e["evaluacion"]["detalles"] if d["venta_real"] is not None]
    cantidad = len(pares)
    mae = wape = rmse = None
    suma_errores = suma_reales = escala = cuadrados = 0.0
    if cantidad:
        try:
            errores = [d["error_absoluto"] for d in pares]
            suma_errores = fsum(errores)
            suma_reales = fsum(d["venta_real"] for d in pares)
            mae = suma_errores / cantidad
            if 0 < suma_errores < _MINIMO_FLOAT * cantidad:
                raise ValueError("Media fuera de rango")
            wape = metricas._wape_estable(suma_errores, suma_reales)
            escala = max(errores)
            cuadrados = fsum((e / escala) ** 2 for e in errores) if escala else 0.0
            rmse = escala * sqrt(cuadrados / cantidad) if escala else 0.0
            # En este extremo cada cociente es un entero pequeño exacto.
            if escala and rmse <= _MINIMO_FLOAT and fsum((e / _MINIMO_FLOAT) ** 2 for e in errores) < cantidad:
                raise ValueError("Raíz media fuera de rango")
            valores = (suma_errores, suma_reales, mae, rmse, escala, cuadrados)
            if not all(isfinite(v) for v in valores + (() if wape is None else (wape,))):
                raise ValueError("Resultado no finito")
        except (OverflowError, ValueError) as error:
            raise ErrorConsolidacionEvaluaciones("resultado_no_finito") from error
    estado, alcance = _cobertura(cantidad)
    return {
        "mae": mae, "wape": wape, "rmse": rmse,
        "suma_errores_absolutos": suma_errores, "suma_ventas_reales": suma_reales,
        "observaciones_evaluadas": cantidad, "fechas_sin_registro": FECHAS_ESPERADAS - cantidad,
        "cobertura": cantidad / FECHAS_ESPERADAS,
        "definicion_cobertura": "observaciones_evaluadas / 45 fechas naturales esperadas",
        "estado_cobertura": estado, "alcance_metricas": alcance,
        "rmse_escalado": {"escala_errores": escala, "suma_cuadrados_normalizados": cuadrados},
    }


def consolidar_evaluaciones(ruta_base: str | Path, grupos: list | tuple) -> dict:
    """Consolida dos o tres configuraciones sobre los tres cortes autorizados.

    Entrada: lista/tupla de 2 o 3 grupos, cada uno con 3 IDs Python int distintos
    globalmente, en orden 2017-06-16, 2017-07-01, 2017-07-16. Se valida antes de
    abrir SQLite. Cada grupo requiere identidad F2 única y versiones relevantes
    exactamente iguales entre cortes; distintos grupos deben tener identidades
    distintas. Esta política acredita las versiones declaradas, no la igualdad
    integral de máquinas ni una reproducibilidad histórica permanente.

    Salida común: conjunto_id, serie, cortes, horizonte_por_corte,
    fechas_esperadas_total, observaciones_evaluadas_total, cobertura_total,
    estado_cobertura, alcance_metricas, fechas_ausentes_por_corte y resultados.
    Cada resultado contiene tecnica, identidad_configuracion (SHA256 F2),
    configuracion (contrato efectivo público completo, incluidas sus versiones
    de contrato/reglas), versiones_ejecucion, evaluaciones (las tres respuestas
    A2 íntegras con IDs, experimentos, cortes y detalle A1) y consolidado.

    Una conexión mode=ro y un BEGIN de _conexion_lectura fijan la instantánea.
    A2 valida tipo/corte antes de leer ventas. Se exige misma serie/conjunto y,
    por corte, idénticas fechas objetivo y ventas observadas. No se lee la tabla
    evaluacion, ni rutas de modelos, ni históricos para reconstruir pronósticos.
    No hay escrituras, selección, ranking ni resultados parciales ante errores.
    Los mensajes de ErrorConsolidacionEvaluaciones son siempre constantes.
    """
    seleccion = _seleccion(grupos)
    try:
        if not isinstance(ruta_base, (str, Path)) or not str(ruta_base) or "\0" in str(ruta_base):
            raise ValueError("Ruta inválida")
        ruta = Path(ruta_base).resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ErrorConsolidacionEvaluaciones("base_no_disponible") from error
    try:
        with pronosticos._conexion_lectura(ruta) as conexion:
            resultados, referencias, identidades = [], [], set()
            contexto = None
            for grupo in seleccion:
                evaluaciones = []
                identidad_grupo = versiones_grupo = efectiva_grupo = None
                for indice, identificador in enumerate(grupo):
                    actual = consulta._evaluar_en_conexion(conexion, identificador)
                    actual_contexto = (actual["conjunto_id"], actual["serie"], actual["horizonte"])
                    if actual["fecha_corte"] != CORTES[indice] or (contexto is not None and actual_contexto != contexto):
                        raise ErrorConsolidacionEvaluaciones("pronosticos_no_comparables")
                    contexto = actual_contexto
                    condiciones = _condiciones(actual)
                    if len(referencias) <= indice:
                        referencias.append(condiciones)
                    elif condiciones != referencias[indice]:
                        raise ErrorConsolidacionEvaluaciones("pronosticos_no_comparables")
                    efectiva, identidad, versiones = _configuracion(conexion, actual)
                    if identidad_grupo is None:
                        identidad_grupo, versiones_grupo, efectiva_grupo = identidad, versiones, efectiva
                    elif identidad != identidad_grupo:
                        raise ErrorConsolidacionEvaluaciones("configuraciones_no_equivalentes")
                    elif versiones != versiones_grupo:
                        raise ErrorConsolidacionEvaluaciones("versiones_incompatibles")
                    evaluaciones.append(actual)
                if identidad_grupo in identidades:
                    raise ErrorConsolidacionEvaluaciones("configuraciones_duplicadas")
                identidades.add(identidad_grupo)
                resultados.append({
                    "tecnica": efectiva_grupo["tecnica"], "identidad_configuracion": identidad_grupo,
                    "configuracion": efectiva_grupo, "versiones_ejecucion": versiones_grupo,
                    "evaluaciones": evaluaciones, "consolidado": _consolidado(evaluaciones),
                })
            primera = resultados[0]
            total = primera["consolidado"]
            return {
                "conjunto_id": contexto[0], "serie": dict(contexto[1]), "cortes": list(CORTES),
                "horizonte_por_corte": metricas.HORIZONTE_DIAS, "fechas_esperadas_total": FECHAS_ESPERADAS,
                "observaciones_evaluadas_total": total["observaciones_evaluadas"],
                "cobertura_total": total["cobertura"], "estado_cobertura": total["estado_cobertura"],
                "alcance_metricas": total["alcance_metricas"],
                "fechas_ausentes_por_corte": [{"fecha_corte": e["fecha_corte"],
                                               "fechas_ausentes": list(e["evaluacion"]["fechas_ausentes"])}
                                              for e in primera["evaluaciones"]],
                "resultados": resultados,
            }
    except (consulta.ErrorConsultaEvaluacion, pronosticos.ErrorConsultaPronostico) as error:
        raise ErrorConsolidacionEvaluaciones(error.codigo) from error
