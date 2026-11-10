"""Coordinación síncrona I1-B1; cada registro conserva su propia transacción.

No hay atomicidad de batería, búsqueda rápida de duplicados ni cancelación de
un ajuste iniciado. Las lecturas ro delimitan cada intento y contrastan además
el histórico acreditado por el registro con la instantánea inicial. No bloquean
a otros escritores durante los cálculos ni detectan cambios transitorios que
se reviertan entre lecturas y no afecten a ningún registro.
"""

from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from backend.app.services import configuracion_experimentos as contrato
from backend.app.services import plan_bateria_pronosticos as planificador
from backend.app.services import registro_pronosticos as registro


_MENSAJES = {
    "conjunto_invalido": "El identificador de conjunto no es válido.",
    "serie_invalida": "El identificador de serie no es válido.",
    "corte_invalido": "La fecha de corte no es válida.",
    "corte_no_autorizado": "El corte de validación no está autorizado.",
    "prueba_reservada": "La prueba final permanece reservada.",
    "sin_conjunto_activo": "No existe un conjunto activo.",
    "conjunto_inactivo": "El conjunto solicitado no es el activo.",
    "serie_inexistente": "La serie no existe.",
    "serie_inactiva": "La serie no pertenece al conjunto activo.",
    "sin_historial": "La serie no tiene observaciones hasta el corte.",
    "historial_invalido": "Las observaciones históricas presentan datos inconsistentes.",
    "historial_cambiado": "El conjunto, la serie o el histórico cambiaron; la batería se detuvo.",
    "base_no_disponible": "La base de datos no está disponible o no tiene el esquema necesario.",
    "error_calculo": "El histórico no permite calcular este pronóstico.",
    "resultado_invalido": "El registro no acreditó un resultado válido para la configuración solicitada.",
    "conflicto": "Existe un pronóstico conflictivo; no se sobrescribió.",
    "error_persistencia": "No se pudo completar el registro de este pronóstico.",
    "error_interno": "No se pudo completar la ejecución de la batería.",
}


class ErrorEjecucionBateria(ValueError):
    """Rechazo previo al primer intento; siempre con un mensaje público fijo."""

    def __init__(self, codigo):
        self.codigo = codigo if codigo in _MENSAJES else "error_interno"
        super().__init__(_MENSAJES[self.codigo])


def _error(codigo):
    error = ErrorEjecucionBateria(codigo)
    return {"codigo": error.codigo, "mensaje": str(error)}


@dataclass(frozen=True)
class _Contexto:
    conjunto_id: int
    hash_conjunto: str
    serie_id: int
    store_nbr: int
    family: str
    filas: tuple

    def historial(self):
        huella = hashlib.sha256(json.dumps(
            self.filas, ensure_ascii=False, allow_nan=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        return {"inicio_disponible": self.filas[0][0],
                "fin_disponible": self.filas[-1][0], "cantidad": len(self.filas),
                "sha256_observaciones": huella, "conjunto_id": self.conjunto_id,
                "hash_conjunto": self.hash_conjunto}


def _comprobar_contexto(ruta, plan, inicial=None, resultado=None, elemento=None):
    """Una instantánea por comprobación, cerrada antes de llamar al registro."""
    try:
        with closing(sqlite3.connect(f"{ruta.as_uri()}?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            activo = c.execute("SELECT id, hash FROM conjunto_datos WHERE activo=1 "
                               "ORDER BY id DESC LIMIT 1").fetchone()
            if activo is None:
                raise ErrorEjecucionBateria("sin_conjunto_activo")
            if activo[0] != plan["conjunto_id"]:
                raise ErrorEjecucionBateria("conjunto_inactivo")
            serie = c.execute("SELECT conjunto_id, store_nbr, family FROM serie_temporal "
                              "WHERE id=?", (plan["serie_id"],)).fetchone()
            if serie is None:
                raise ErrorEjecucionBateria("serie_inexistente")
            if serie[0] != activo[0]:
                raise ErrorEjecucionBateria("serie_inactiva")
            filas = tuple(c.execute("SELECT fecha, sales, onpromotion FROM venta_diaria "
                                   "WHERE serie_id=? AND fecha<=? ORDER BY fecha",
                                   (plan["serie_id"], plan["fecha_corte"])).fetchall())
            if not filas:
                raise ErrorEjecucionBateria("sin_historial")
            actual = _Contexto(activo[0], activo[1], plan["serie_id"], *serie[1:], filas)
            try:
                actual.historial()
            except (ValueError, TypeError, OverflowError) as error:
                raise ErrorEjecucionBateria("historial_invalido") from error
            if inicial is not None and actual != inicial:
                raise ErrorEjecucionBateria("historial_cambiado")
            if resultado is not None:
                # Acreditar el histórico realmente usado, incluso si cambió y se
                # restauró entre la comprobación previa y el retorno del registro.
                guardado = c.execute(
                    "SELECT e.configuracion_json FROM experimento e JOIN pronostico p "
                    "ON p.experimento_id=e.id WHERE e.id=? AND p.id=? AND e.serie_id=? "
                    "AND e.tecnica=? AND p.fecha_corte=? AND p.horizonte=?",
                    (resultado.experimento_id, resultado.pronostico_id, plan["serie_id"],
                     elemento["tecnica"], plan["fecha_corte"], elemento["horizonte"])).fetchone()
                if guardado is None:
                    raise ErrorEjecucionBateria("resultado_invalido")
                try:
                    config = registro._leer_configuracion_estricta(guardado[0])
                    identidad = contrato.identidad_configuracion(contrato.recuperar_configuracion(config))
                    if identidad != elemento["identidad_predictiva"]:
                        raise ValueError("Identidad distinta")
                    if config.get("historial") != inicial.historial():
                        raise ErrorEjecucionBateria("historial_cambiado")
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
                    if isinstance(error, ErrorEjecucionBateria):
                        raise
                    raise ErrorEjecucionBateria("resultado_invalido") from error
            return actual
    except ErrorEjecucionBateria as error:
        if inicial is not None and error.codigo in {
                "sin_conjunto_activo", "conjunto_inactivo", "serie_inexistente",
                "serie_inactiva", "sin_historial"}:
            raise ErrorEjecucionBateria("historial_cambiado") from error
        raise
    except (sqlite3.Error, OSError) as error:
        raise ErrorEjecucionBateria("base_no_disponible") from error


def _validar_resultado(resultado, plan, elemento):
    if (not isinstance(resultado, registro.ResultadoRegistro)
            or any(type(i) is not int or i <= 0 for i in (
                resultado.experimento_id, resultado.pronostico_id))
            or type(resultado.nuevo) is not bool
            or (resultado.conjunto_id, resultado.serie_id, resultado.tecnica,
                resultado.fecha_corte, resultado.horizonte, resultado.cantidad_detalles) != (
                plan["conjunto_id"], plan["serie_id"], elemento["tecnica"],
                plan["fecha_corte"], elemento["horizonte"], elemento["horizonte"])):
        raise ErrorEjecucionBateria("resultado_invalido")


def ejecutar_bateria(ruta_base: str | Path, conjunto_id: int, serie_id: int,
                    fecha_corte: date | str) -> dict:
    """Ejecuta cinco intentos secuenciales; no acepta planes externos ni opciones.

    Un rechazo inicial lanza ErrorEjecucionBateria. Después de comenzar devuelve
    resultados observados, resumen, tiempos y eventual interrupción. Reutilizado
    significa registro reutilizado, no cálculo evitado. No revierte éxitos previos.
    """
    inicio_total = perf_counter()
    try:
        plan = planificador.construir_plan_bateria(conjunto_id, serie_id, fecha_corte)
    except planificador.ErrorPlanBateria as error:
        raise ErrorEjecucionBateria(error.codigo) from error
    try:
        ruta = Path(ruta_base).resolve()
    except (TypeError, ValueError, OSError) as error:
        raise ErrorEjecucionBateria("base_no_disponible") from error
    inicial = _comprobar_contexto(ruta, plan)
    resultados = []
    for posicion, elemento in enumerate(plan["elementos"], 1):
        resultados.append({**deepcopy(elemento), "orden": posicion,
                           "conjunto_id": conjunto_id, "serie_id": serie_id,
                           "fecha_corte": plan["fecha_corte"],
                           "estado": "no_ejecutado", "duracion_segundos": None})
    interrupcion = None
    for elemento in resultados:
        try:
            _comprobar_contexto(ruta, plan, inicial)
        except ErrorEjecucionBateria as error:
            interrupcion = _error(error.codigo)
            break
        inicio = perf_counter()
        try:
            resultado = registro.registrar_pronostico(
                ruta, serie_id, elemento["tecnica"], plan["fecha_corte"],
                configuracion_solicitada=deepcopy(elemento["configuracion"]))
            _validar_resultado(resultado, plan, elemento)
            elemento.update({"estado": "creado" if resultado.nuevo else "reutilizado",
                             "experimento_id": resultado.experimento_id,
                             "pronostico_id": resultado.pronostico_id})
        except (registro.ErrorRegistroPronostico, ErrorEjecucionBateria) as error:
            elemento.update({"estado": "fallido", "error": _error(error.codigo)})
            if error.codigo != "error_calculo":
                interrupcion = elemento["error"].copy()
        except (sqlite3.Error, OSError):
            elemento.update({"estado": "fallido", "error": _error("base_no_disponible")})
            interrupcion = elemento["error"].copy()
        except Exception:
            elemento.update({"estado": "fallido", "error": _error("error_interno")})
            interrupcion = elemento["error"].copy()
        finally:
            elemento["duracion_segundos"] = perf_counter() - inicio
        try:
            _comprobar_contexto(ruta, plan, inicial,
                               resultado if elemento["estado"] in {"creado", "reutilizado"} else None,
                               elemento)
        except ErrorEjecucionBateria as error:
            interrupcion = _error(error.codigo)
            if error.codigo == "resultado_invalido":
                elemento.update({"estado": "fallido", "error": interrupcion.copy()})
                elemento.pop("experimento_id", None)
                elemento.pop("pronostico_id", None)
        if interrupcion is not None:
            break
    resumen = {estado: sum(e["estado"] == estado for e in resultados)
               for estado in ("creado", "reutilizado", "fallido", "no_ejecutado")}
    return {"conjunto_id": conjunto_id, "serie_id": serie_id,
            "fecha_corte": plan["fecha_corte"], "horizonte": resultados[0]["horizonte"],
            "historial_inicial": inicial.historial(),
            "estado": "interrumpida" if interrupcion else ("parcial" if resumen["fallido"] else "completa"),
            "contexto_verificado": interrupcion is None,
            "resultados": resultados, "resumen": resumen, "interrupcion": interrupcion,
            "duracion_segundos": perf_counter() - inicio_total}
