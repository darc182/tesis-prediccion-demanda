"""Contrato predictivo v1: funciones puras, sin estimadores, SQL ni entrenamiento.

Los defaults completos son los efectivos de HU004 (comprobados por tests contra
crear_estimador). No dependen de la versión de sklearn instalada al consultar.
random_state representa la semilla; la línea base no tiene semilla ni X.
Las alternativas admitidas preparan F3: validarlas NO implica entrenarlas.
Cambiar la matemática temporal requiere una nueva VERSION_REGLAS y su lector.
"""

from copy import deepcopy
import hashlib
import json
from math import isfinite


VERSION_CONTRATO = 1
VERSION_REGLAS = 1
BASE = "estacional_semanal_recursiva"
VARIABLES = ("lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes")

_PARAMETROS = {
    BASE: {"periodo_estacional_dias": 7},
    "random_forest": {
        "bootstrap": True, "ccp_alpha": 0.0, "criterion": "squared_error",
        "max_depth": None, "max_features": 1.0, "max_leaf_nodes": None,
        "max_samples": None, "min_impurity_decrease": 0.0,
        "min_samples_leaf": 5, "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0, "monotonic_cst": None,
        "n_estimators": 100, "n_jobs": 1, "oob_score": False,
        "random_state": 42, "verbose": 0, "warm_start": False,
    },
    "hist_gradient_boosting": {
        "categorical_features": "from_dtype", "early_stopping": False,
        "interaction_cst": None, "l2_regularization": 0.0, "learning_rate": 0.05,
        "loss": "squared_error", "max_bins": 255, "max_depth": None,
        "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 15,
        "min_samples_leaf": 20, "monotonic_cst": None, "n_iter_no_change": 10,
        "quantile": None, "random_state": 42, "scoring": "loss", "tol": 1e-7,
        "validation_fraction": 0.1, "verbose": 0, "warm_start": False,
    },
}


class ErrorConfiguracion(ValueError):
    """Solicitud inválida o registro cuya identidad no puede reconstruirse."""


def configuracion_predeterminada(tecnica: str) -> dict:
    if type(tecnica) is not str or tecnica not in _PARAMETROS:
        raise ErrorConfiguracion("Técnica desconocida.")
    return {"version_contrato": VERSION_CONTRATO, "tecnica": tecnica,
            "parametros": deepcopy(_PARAMETROS[tecnica]),
            "variables": [] if tecnica == BASE else list(VARIABLES),
            "horizonte": 15, "version_reglas": VERSION_REGLAS}


def _entero(valor, minimo, maximo):
    if type(valor) is not int or not minimo <= valor <= maximo:
        raise ErrorConfiguracion(f"Se requiere entero entre {minimo} y {maximo}.")
    return valor


def _real(valor, minimo, maximo):
    if type(valor) not in (int, float) or not minimo <= valor <= maximo or not isfinite(valor):
        raise ErrorConfiguracion(f"Se requiere número finito entre {minimo} y {maximo}.")
    # JSON distingue -0.0 de 0.0; ambos ceros válidos tienen una sola identidad.
    return 0.0 if valor == 0 else float(valor)


def _parametro(tecnica, nombre, valor, defecto):
    # Superficie acotada para F3; el resto se reconoce pero permanece fijo.
    if tecnica != BASE:
        if nombre == "random_state":
            return _entero(valor, 0, 2**32 - 1)
        if nombre == "max_depth":
            return None if valor is None else _entero(valor, 1, 50)
        if nombre == "min_samples_leaf":
            return _entero(valor, 1, 100)
        if nombre in ("n_estimators", "max_iter"):
            return _entero(valor, 1, 500)
        if nombre == "max_leaf_nodes":
            if tecnica == "random_forest" and valor is None:
                return None
            return _entero(valor, 2, 255)
        if nombre == "min_samples_split":
            return _entero(valor, 2, 100)
        if nombre == "learning_rate":
            return _real(valor, 0.001, 1.0)
        if nombre == "l2_regularization":
            return _real(valor, 0.0, 100.0)
        if nombre == "max_features":
            if tecnica == "random_forest":
                if type(valor) is str and valor in ("sqrt", "log2"):
                    return valor
                if type(valor) is int:
                    return _entero(valor, 1, len(VARIABLES))
            return _real(valor, 0.01, 1.0)
    # Comparación tipada: True no equivale a 1; 0.0 no equivale a False.
    if type(defecto) is float:
        valor = _real(valor, 0.0, 1.0)
    if type(valor) is not type(defecto) or valor != defecto:
        raise ErrorConfiguracion(f"El parámetro {nombre} permanece fijo en {defecto!r}.")
    return valor


def normalizar_configuracion(solicitud: dict) -> dict:
    """Completa defaults explícitos del contrato; devuelve una copia normalizada.

    Las seis variables son obligatorias en regresores; se admite permutarlas y
    su orden distingue identidades. No admite onpromotion ni subconjuntos.
    max_features entero (número) y flotante (fracción) NO son intercambiables.
    """
    if type(solicitud) is not dict:
        raise ErrorConfiguracion("La configuración debe ser un objeto.")
    resultado = configuracion_predeterminada(solicitud.get("tecnica"))
    if set(solicitud) - set(resultado):
        raise ErrorConfiguracion("Campos de configuración desconocidos.")
    for nombre in ("version_contrato", "horizonte", "version_reglas"):
        valor = solicitud.get(nombre, resultado[nombre])
        if type(valor) is not int or valor != resultado[nombre]:
            raise ErrorConfiguracion(f"{nombre} no soportado.")
    parametros = solicitud.get("parametros", {})
    if type(parametros) is not dict or set(parametros) - set(resultado["parametros"]):
        raise ErrorConfiguracion("Parámetros desconocidos o estructura inválida.")
    for nombre, defecto in resultado["parametros"].items():
        resultado["parametros"][nombre] = _parametro(
            resultado["tecnica"], nombre, parametros.get(nombre, defecto), defecto)
    variables = solicitud.get("variables", resultado["variables"])
    if (type(variables) not in (list, tuple) or any(type(v) is not str for v in variables)
            or len(variables) != len(set(variables))
            or set(variables) != set(resultado["variables"])):
        raise ErrorConfiguracion("Variables requeridas ausentes, desconocidas o duplicadas.")
    resultado["variables"] = list(variables)
    return resultado


def validar_configuracion(solicitud: dict) -> None:
    normalizar_configuracion(solicitud)


def _json(valor):
    return json.dumps(valor, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def json_canonico(solicitud: dict) -> str:
    return _json(normalizar_configuracion(solicitud))


def identidad_configuracion(solicitud: dict) -> str:
    return hashlib.sha256(json_canonico(solicitud).encode("utf-8")).hexdigest()


def reglas_heredadas(tecnica: str, variables=None) -> dict:
    """Reglas textuales exactas de HU004/F1 que respaldan VERSION_REGLAS=1."""
    configuracion_predeterminada(tecnica)
    reglas = {"horizonte_dias": 15, "ventas_reales_hasta_corte_inclusive": True,
              "ausencias_no_son_ceros": True, "redondeo_interno": False}
    if tecnica == BASE:
        reglas.update({"referencia": "objetivo menos 7 días; real hasta corte o predicción previa",
                       "respaldos": ["última observación real del mismo día semanal",
                                     "última observación real disponible"], "sin_historial": "error"})
    else:
        reglas.update({"predictoras": list(VARIABLES if variables is None else variables),
                       "onpromotion_predictor": False, "dias_naturales_exactos": True,
                       "media_7": "siete observaciones de los siete días anteriores completos",
                       "dia_semana": "lunes=0, domingo=6", "mes": "enero=1, diciembre=12",
                       "entrenamiento_incompleto": "excluir fila",
                       "inferencia_incompleta": "error sin imputación",
                       "recursion": "incorporar solo estimaciones previas al objetivo",
                       "negativos": "comprobar finitud y aplicar max(0, predicción)", "hilos_numericos": 1})
    return reglas


def recuperar_configuracion(registro: dict) -> dict:
    """Lee v1/v2 sin mutarlos ni completar evidencia predictiva ausente.

    v1 exige TODOS los parámetros efectivos y reglas reconocidas. v2 exige el
    contrato completo y coherencia con los campos técnicos conservados.
    Versiones y estadísticas se conservan como auditoría, nunca como identidad.
    """
    if type(registro) is not dict or type(registro.get("version_registro")) is not int:
        raise ErrorConfiguracion("Formato de registro no reconocido.")
    campos = {"version_registro", "tecnica", "parametros", "reglas", "versiones",
              "entrenamiento", "auditoria", "historial", "configuracion_predictiva",
              "duracion_segundos", "fecha_generacion"}
    if set(registro) - campos:
        raise ErrorConfiguracion("Campos de registro no reconocidos; identidad incierta.")
    version = registro["version_registro"]
    tecnica = registro.get("tecnica")
    defecto = configuracion_predeterminada(tecnica)
    parametros = registro.get("parametros")
    reglas = registro.get("reglas")
    if type(parametros) is not dict or set(parametros) != set(defecto["parametros"]):
        raise ErrorConfiguracion("Parámetros efectivos antiguos incompletos o desconocidos.")
    if type(reglas) is not dict:
        raise ErrorConfiguracion("Reglas ausentes.")
    if version == 1:
        if "configuracion_predictiva" in registro:
            raise ErrorConfiguracion("Formato de registro contradictorio.")
        solicitud = {"tecnica": tecnica, "parametros": parametros,
                     "variables": [] if tecnica == BASE else reglas.get("predictoras")}
    elif version == 2:
        solicitud = registro.get("configuracion_predictiva")
        if (type(solicitud) is not dict or set(solicitud) != set(defecto)
                or type(solicitud.get("parametros")) is not dict
                or set(solicitud["parametros"]) != set(defecto["parametros"])):
            raise ErrorConfiguracion("Contrato almacenado incompleto.")
    else:
        raise ErrorConfiguracion("Versión de registro no soportada.")
    config = normalizar_configuracion(solicitud)
    if (config["tecnica"] != tecnica
            or _json(config["parametros"]) != _json(normalizar_configuracion(
                {"tecnica": tecnica, "parametros": parametros})["parametros"])
            or _json(reglas) != _json(reglas_heredadas(tecnica, config["variables"]))):
        raise ErrorConfiguracion("Reglas o parámetros no reconocibles de manera inequívoca.")
    return config
