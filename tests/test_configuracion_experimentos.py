"""Contrato puro y compatibilidad; ninguna prueba usa la base principal."""

from copy import deepcopy
import hashlib
from itertools import permutations
import json
import unittest

from backend.app.services import configuracion_experimentos as config
from scripts import modelos_prediccion as modelos
from scripts.variables_predictoras import COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS


RF = "random_forest"
HGB = "hist_gradient_boosting"
PARAMETROS_CERO_REAL = ((RF, "ccp_alpha"), (RF, "min_impurity_decrease"),
                        (RF, "min_weight_fraction_leaf"), (HGB, "l2_regularization"))


def registro_antiguo(tecnica=RF):
    """Forma HU004/F1, sin bloque canónico ni valores inferidos del software."""
    return {"version_registro": 1, "tecnica": tecnica,
            "parametros": config.configuracion_predeterminada(tecnica)["parametros"],
            "reglas": config.reglas_heredadas(tecnica),
            "versiones": {"python": "anterior", "pandas": "anterior", "scikit_learn": "anterior"},
            "entrenamiento": {"filas_validas": 46, "filas_descartadas": 14},
            "auditoria": {"ajustes_a_cero": 0},
            "historial": {"sha256_observaciones": "a" * 64}}


class ConfiguracionExperimentosTests(unittest.TestCase):
    def test_default_linea_base_sin_semilla_ni_variables_artificiales(self):
        c = config.configuracion_predeterminada(config.BASE)
        self.assertEqual(c, {"version_contrato": 1, "version_reglas": 1,
                             "tecnica": config.BASE, "parametros": {"periodo_estacional_dias": 7},
                             "variables": [], "horizonte": 15})
        self.assertEqual(config.reglas_heredadas(config.BASE)["respaldos"],
                         ["última observación real del mismo día semanal",
                          "última observación real disponible"])

    def test_default_random_forest(self):
        c = config.configuracion_predeterminada(RF)
        self.assertEqual([c["parametros"][k] for k in
                          ("n_estimators", "min_samples_leaf", "random_state", "n_jobs")],
                         [100, 5, 42, 1])
        self.assertEqual(c["variables"], list(COLUMNAS_PREDICTORAS))
        self.assertNotIn("onpromotion", c["variables"])

    def test_default_hist_gradient_boosting(self):
        c = config.configuracion_predeterminada(HGB)
        self.assertEqual([c["parametros"][k] for k in
                          ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                           "early_stopping", "random_state")], [100, 0.05, 15, 20, False, 42])
        self.assertEqual(c["variables"], ["lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes"])

    def test_defaults_completos_coinciden_con_estimadores_reales(self):
        for tecnica in (RF, HGB):
            with self.subTest(tecnica=tecnica):
                real = modelos.crear_estimador(tecnica).get_params(deep=False)
                esperado = config.configuracion_predeterminada(tecnica)["parametros"]
                self.assertEqual(set(real), set(esperado))
                # JSON tipado: evita que True=1 o 1=1.0 oculten discrepancias.
                self.assertEqual(json.dumps(real, sort_keys=True), json.dumps(esperado, sort_keys=True))

    def test_defaults_explicitos_e_implicitos_equivalentes(self):
        for tecnica in (config.BASE, RF, HGB):
            with self.subTest(tecnica=tecnica):
                completo = config.configuracion_predeterminada(tecnica)
                self.assertEqual(config.normalizar_configuracion({"tecnica": tecnica}), completo)
                self.assertEqual(config.identidad_configuracion({"tecnica": tecnica}),
                                 config.identidad_configuracion(completo))
                self.assertIsNone(config.validar_configuracion(completo))

    def test_identidad_es_sha256_determinista(self):
        solicitud = {"tecnica": RF, "parametros": {"max_depth": 10}}
        texto = config.json_canonico(solicitud)
        self.assertEqual(config.identidad_configuracion(solicitud),
                         hashlib.sha256(texto.encode("utf-8")).hexdigest())
        self.assertEqual(config.identidad_configuracion(solicitud), config.identidad_configuracion(solicitud))
        self.assertEqual(len(config.identidad_configuracion(solicitud)), 64)

    def test_orden_de_claves_json_no_importa(self):
        a = config.configuracion_predeterminada(RF)
        b = dict(reversed(list(a.items())))
        b["parametros"] = dict(reversed(list(b["parametros"].items())))
        b = json.loads(json.dumps(b, indent=4))
        self.assertEqual(config.json_canonico(a), config.json_canonico(b))

    def test_normalizacion_numerica_sin_confundir_numero_y_fraccion(self):
        self.assertEqual(config.identidad_configuracion({"tecnica": HGB, "parametros": {"learning_rate": 1}}),
                         config.identidad_configuracion({"tecnica": HGB, "parametros": {"learning_rate": 1.0}}))
        self.assertNotEqual(config.identidad_configuracion({"tecnica": RF, "parametros": {"max_features": 1}}),
                            config.identidad_configuracion({"tecnica": RF, "parametros": {"max_features": 1.0}}))

    def test_ceros_reales_con_ambos_signos_tienen_identidad_y_json_iguales(self):
        for tecnica, parametro in PARAMETROS_CERO_REAL:
            with self.subTest(tecnica=tecnica, parametro=parametro):
                positiva = {"tecnica": tecnica, "parametros": {parametro: 0.0}}
                negativa = {"tecnica": tecnica, "parametros": {parametro: -0.0}}
                original = json.dumps(negativa, sort_keys=True)
                self.assertEqual(config.identidad_configuracion(positiva),
                                 config.identidad_configuracion(negativa))
                texto = config.json_canonico(negativa)
                self.assertEqual(texto, config.json_canonico(positiva))
                self.assertIn(f'"{parametro}":0.0', texto)
                self.assertNotIn("-0.0", texto)
                self.assertEqual(json.dumps(negativa, sort_keys=True), original)
                self.assertIn("-0.0", original)  # la solicitud conserva su signo original

    def test_ceros_se_normalizan_solo_despues_de_validar(self):
        for valor in (True, False, "0.0", None, -5e-324, float("nan"),
                      float("inf"), -float("inf")):
            with self.subTest(valor=valor), self.assertRaises(config.ErrorConfiguracion):
                config._real(valor, 0.0, 1.0)
        for tecnica, parametro in ((HGB, "learning_rate"), (RF, "max_features")):
            for valor in (0.0, -0.0):
                with self.subTest(parametro=parametro, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({"tecnica": tecnica, "parametros": {parametro: valor}})

    def test_parametros_con_cero_siguen_rechazando_negativos_y_no_finitos(self):
        for tecnica, parametro in PARAMETROS_CERO_REAL:
            for valor in (-0.1, -5e-324, float("nan"), float("inf"), -float("inf")):
                with self.subTest(parametro=parametro, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({"tecnica": tecnica, "parametros": {parametro: valor}})

    def test_valores_no_nulos_y_restricciones_fijas_no_cambian(self):
        for valor in (-0.5, 0.125, 0.5, 1.0):
            self.assertEqual(config._real(valor, -1.0, 1.0), valor)
        identidades = []
        for valor in (0.0, 0.125, 0.5):
            solicitud = {"tecnica": HGB, "parametros": {"l2_regularization": valor}}
            self.assertEqual(config.normalizar_configuracion(solicitud)["parametros"]["l2_regularization"], valor)
            identidades.append(config.identidad_configuracion(solicitud))
        self.assertEqual(len(set(identidades)), 3)
        # Estos parámetros RF permanecen fijos en cero: la corrección no amplía F3.
        for tecnica, parametro in PARAMETROS_CERO_REAL[:3]:
            with self.subTest(parametro=parametro), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": tecnica, "parametros": {parametro: 0.125}})

    def test_identidades_predeterminadas_anteriores_permanecen_iguales(self):
        # Huellas capturadas antes de corregir el cero con signo.
        anteriores = {
            config.BASE: "e3d39aeca795751d4827fdd1cd63654f6ccb67ce81c02bae4a25850ef54e3432",
            RF: "93d545ae239542cb0fa00d2fa6549e21365ba89e4c1feca3c56f222a4874ebbb",
            HGB: "985fe1ea61f81e82d4d66c2ee2cd6ffbf7aab3f78ea8f75e60bed43561aad6b9",
        }
        for tecnica, identidad in anteriores.items():
            with self.subTest(tecnica=tecnica):
                self.assertEqual(config.identidad_configuracion({"tecnica": tecnica}), identidad)

    def test_registros_anteriores_con_cero_negativo_se_leen_sin_mutacion(self):
        """El JSON persiste intacto; el hash recalculado converge al del cero positivo.

        F2 no persiste hashes: registro reconstruye la identidad desde el JSON.
        Si coexistían ambos signos en filas equivalentes, el servicio detectará
        su ambigüedad en lugar de elegir una. No migrar ni reescribir registros.
        """
        for tecnica, parametro in PARAMETROS_CERO_REAL:
            for version in (1, 2):
                with self.subTest(tecnica=tecnica, parametro=parametro, version=version):
                    registro = registro_antiguo(tecnica)
                    registro["parametros"][parametro] = -0.0
                    if version == 2:
                        registro["version_registro"] = 2
                        registro["configuracion_predictiva"] = config.configuracion_predeterminada(tecnica)
                        registro["configuracion_predictiva"]["parametros"][parametro] = -0.0
                    texto = json.dumps(registro, sort_keys=True)
                    # Reconstrucción exacta del JSON canónico previo a la corrección.
                    previa = config.configuracion_predeterminada(tecnica)
                    previa["parametros"][parametro] = -0.0
                    json_previo = json.dumps(previa, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    hash_previo = hashlib.sha256(json_previo.encode("utf-8")).hexdigest()
                    recuperada = config.recuperar_configuracion(registro)
                    identidad = config.identidad_configuracion(recuperada)
                    self.assertNotEqual(identidad, hash_previo)
                    self.assertEqual(identidad, config.identidad_configuracion({"tecnica": tecnica}))
                    self.assertNotIn("-0.0", config.json_canonico(recuperada))
                    self.assertEqual(json.dumps(registro, sort_keys=True), texto)
                    self.assertIn("-0.0", texto)

    def test_parametros_predictivos_distintos_diferencian_identidades(self):
        casos = ((RF, "n_estimators", 110), (RF, "min_samples_leaf", 6), (RF, "max_depth", 10),
                 (RF, "max_features", "sqrt"), (HGB, "max_iter", 110),
                 (HGB, "learning_rate", 0.1), (HGB, "max_leaf_nodes", 16),
                 (HGB, "l2_regularization", 1.5))
        for tecnica, nombre, valor in casos:
            with self.subTest(tecnica=tecnica, nombre=nombre):
                self.assertNotEqual(config.identidad_configuracion({"tecnica": tecnica}),
                                    config.identidad_configuracion({"tecnica": tecnica,
                                                                    "parametros": {nombre: valor}}))

    def test_semillas_distintas_diferencian_identidades(self):
        for tecnica in (RF, HGB):
            self.assertNotEqual(config.identidad_configuracion({"tecnica": tecnica}),
                                config.identidad_configuracion({"tecnica": tecnica,
                                                                "parametros": {"random_state": 43}}))

    def test_orden_de_variables_forma_parte_de_identidad(self):
        a = config.configuracion_predeterminada(RF)
        b = deepcopy(a)
        b["variables"].reverse()
        self.assertNotEqual(config.identidad_configuracion(a), config.identidad_configuracion(b))

    def test_no_muta_solicitudes_ni_defaults(self):
        a = config.configuracion_predeterminada(RF)
        copia = deepcopy(a)
        normalizada = config.normalizar_configuracion(a)
        normalizada["parametros"]["n_estimators"] = 120
        normalizada["variables"].reverse()
        self.assertEqual(a, copia)
        self.assertEqual(config.configuracion_predeterminada(RF), copia)

    def test_rechaza_tecnicas_desconocidas_y_estructuras_incorrectas(self):
        for solicitud in ({"tecnica": "otra"}, {}, None, [], {"tecnica": True}, {"tecnica": []}):
            with self.subTest(solicitud=solicitud), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion(solicitud)

    def test_rechaza_parametros_y_campos_desconocidos(self):
        for solicitud in ({"tecnica": RF, "parametros": {"inventado": 1}},
                          {"tecnica": RF, "versiones": {}}, {"tecnica": RF, "parametros": []},
                          {"tecnica": config.BASE, "parametros": {"random_state": 42}}):
            with self.subTest(solicitud=solicitud), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion(solicitud)

    def test_rechaza_tipos_invalidos_y_booleanos_como_enteros(self):
        casos = ((RF, "n_estimators", True), (RF, "n_estimators", 100.0),
                 (RF, "random_state", False), (RF, "min_samples_leaf", "5"),
                 (RF, "bootstrap", 1), (RF, "n_jobs", True),
                 (RF, "max_features", True), (RF, "max_depth", 2.5),
                 (HGB, "early_stopping", 0), (HGB, "learning_rate", True),
                 (HGB, "max_leaf_nodes", None), (HGB, "random_state", None))
        for tecnica, nombre, valor in casos:
            with self.subTest(nombre=nombre, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": tecnica, "parametros": {nombre: valor}})

    def test_rechaza_nan_infinity_y_desbordamientos(self):
        for valor in (float("nan"), float("inf"), -float("inf"), 10**400):
            for tecnica, nombre in ((RF, "n_estimators"), (RF, "ccp_alpha"),
                                     (HGB, "learning_rate"), (HGB, "l2_regularization")):
                with self.subTest(tecnica=tecnica, nombre=nombre, valor=valor), \
                        self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({"tecnica": tecnica, "parametros": {nombre: valor}})

    def test_limites_acotados_para_f3(self):
        casos = ((RF, "n_estimators", 0), (RF, "n_estimators", 501),
                 (RF, "min_samples_leaf", -1), (RF, "min_samples_leaf", 101),
                 (RF, "min_samples_split", 1), (RF, "max_depth", 51),
                 (RF, "max_features", 0), (RF, "max_features", 7),
                 (RF, "max_features", 1.1), (RF, "max_leaf_nodes", 256),
                 (RF, "random_state", -1), (RF, "random_state", 2**32),
                 (HGB, "learning_rate", 0), (HGB, "learning_rate", 1.1),
                 (HGB, "l2_regularization", -0.1), (HGB, "max_iter", 501))
        for tecnica, nombre, valor in casos:
            with self.subTest(nombre=nombre, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": tecnica, "parametros": {nombre: valor}})

    def test_parametros_fuera_de_superficie_f3_permanecen_fijos(self):
        for tecnica, nombre, valor in ((RF, "n_jobs", 2), (RF, "criterion", "absolute_error"),
                                       (RF, "bootstrap", False), (RF, "warm_start", True),
                                       (HGB, "early_stopping", True),
                                       (config.BASE, "periodo_estacional_dias", 14)):
            with self.subTest(nombre=nombre), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": tecnica, "parametros": {nombre: valor}})

    def test_rechaza_variables_ausentes_duplicadas_desconocidas_y_onpromotion(self):
        for variables in ([], list(config.VARIABLES[:-1]), [*config.VARIABLES, "lag_1"],
                          [*config.VARIABLES, "onpromotion"], "lag_1", [1], None):
            with self.subTest(variables=variables), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": RF, "variables": variables})

    def test_horizonte_y_versiones_del_contrato_son_explicitos(self):
        for campo, valor in (("horizonte", 30), ("horizonte", 15.0),
                              ("version_contrato", 2), ("version_contrato", True),
                              ("version_reglas", 2), ("version_reglas", True)):
            with self.subTest(campo=campo), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion({"tecnica": RF, campo: valor})

    def test_compatibilidad_identidad_antigua_y_nueva_sin_mutar_json(self):
        for tecnica in (config.BASE, RF, HGB):
            with self.subTest(tecnica=tecnica):
                antiguo = registro_antiguo(tecnica)
                anterior = deepcopy(antiguo)
                esperado = config.configuracion_predeterminada(tecnica)
                self.assertEqual(config.recuperar_configuracion(antiguo), esperado)
                nuevo = {**antiguo, "version_registro": 2, "configuracion_predictiva": esperado}
                self.assertEqual(config.recuperar_configuracion(nuevo), esperado)
                self.assertEqual(antiguo, anterior)

    def test_versiones_duracion_y_auditoria_no_cambian_identidad(self):
        a = registro_antiguo()
        b = deepcopy(a)
        b.update({"versiones": {"python": "futura", "pandas": "futura"},
                  "entrenamiento": {"filas_validas": 999},
                  "auditoria": {"ajustes_a_cero": 42, "otro_dato": "técnico"},
                  "duracion_segundos": 123.4, "fecha_generacion": "2030-01-01"})
        self.assertEqual(config.identidad_configuracion(config.recuperar_configuracion(a)),
                         config.identidad_configuracion(config.recuperar_configuracion(b)))

    def test_historial_no_es_identidad_predictiva_pero_se_conserva(self):
        a = registro_antiguo()
        b = deepcopy(a)
        b["historial"]["sha256_observaciones"] = "b" * 64
        self.assertEqual(config.recuperar_configuracion(a), config.recuperar_configuracion(b))
        self.assertNotEqual(a["historial"], b["historial"])

    def test_antiguos_no_reconocibles_no_reciben_defaults_inventados(self):
        a = registro_antiguo()
        casos = []
        for campo in ("parametros", "reglas", "version_registro", "tecnica"):
            copia = deepcopy(a)
            del copia[campo]
            casos.append(copia)
        copia = deepcopy(a)
        del copia["parametros"]["max_depth"]
        casos.append(copia)
        copia = deepcopy(a)
        copia["parametros"]["desconocido"] = 1
        casos.append(copia)
        copia = deepcopy(a)
        copia["reglas"]["recursion"] = "otra regla matemática"
        casos.append(copia)
        copia = deepcopy(a)
        copia["reglas"]["ausencias_no_son_ceros"] = 1
        casos.append(copia)
        copia = deepcopy(a)
        copia["estrategia_predictiva_desconocida"] = "no ignorar"
        casos.append(copia)
        for c in casos:
            with self.subTest(registro=c), self.assertRaises(config.ErrorConfiguracion):
                config.recuperar_configuracion(c)

    def test_antiguo_alternativo_reconocido_conserva_parametros_reales_registrados(self):
        a = registro_antiguo()
        a["parametros"]["n_estimators"] = 110
        c = config.recuperar_configuracion(a)
        self.assertEqual(c["parametros"]["n_estimators"], 110)
        self.assertNotEqual(c, config.configuracion_predeterminada(RF))

    def test_rechaza_contrato_nuevo_incompleto_o_contradictorio(self):
        a = registro_antiguo()
        a.update({"version_registro": 2, "configuracion_predictiva": config.configuracion_predeterminada(RF)})
        casos = []
        for campo in ("variables", "version_reglas", "version_contrato", "parametros"):
            c = deepcopy(a)
            del c["configuracion_predictiva"][campo]
            casos.append(c)
        c = deepcopy(a)
        del c["configuracion_predictiva"]["parametros"]["max_depth"]
        casos.append(c)
        c = deepcopy(a)
        c["configuracion_predictiva"]["parametros"]["n_estimators"] = 110
        casos.append(c)
        c = deepcopy(a)
        c["version_registro"] = 3
        casos.append(c)
        c = deepcopy(a)
        c["version_registro"] = 1
        casos.append(c)
        for c in casos:
            with self.subTest(registro=c), self.assertRaises(config.ErrorConfiguracion):
                config.recuperar_configuracion(c)


# Fixtures capturados del contrato previo (aea781e); no se derivan del código probado.
VARIABLES_V1 = ("lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes")
VARIABLES_V2 = (*VARIABLES_V1, "media_14")
JSON_CLASICOS = {
    config.BASE: ('{"horizonte":15,"parametros":{"periodo_estacional_dias":7},'
                  '"tecnica":"estacional_semanal_recursiva","variables":[],"version_contrato":1,"version_reglas":1}'),
    RF: ('{"horizonte":15,"parametros":{"bootstrap":true,"ccp_alpha":0.0,"criterion":"squared_error",'
         '"max_depth":null,"max_features":1.0,"max_leaf_nodes":null,"max_samples":null,'
         '"min_impurity_decrease":0.0,"min_samples_leaf":5,"min_samples_split":2,'
         '"min_weight_fraction_leaf":0.0,"monotonic_cst":null,"n_estimators":100,"n_jobs":1,'
         '"oob_score":false,"random_state":42,"verbose":0,"warm_start":false},"tecnica":"random_forest",'
         '"variables":["lag_1","lag_7","lag_14","media_7","dia_semana","mes"],'
         '"version_contrato":1,"version_reglas":1}'),
    HGB: ('{"horizonte":15,"parametros":{"categorical_features":"from_dtype","early_stopping":false,'
          '"interaction_cst":null,"l2_regularization":0.0,"learning_rate":0.05,"loss":"squared_error",'
          '"max_bins":255,"max_depth":null,"max_features":1.0,"max_iter":100,"max_leaf_nodes":15,'
          '"min_samples_leaf":20,"monotonic_cst":null,"n_iter_no_change":10,"quantile":null,'
          '"random_state":42,"scoring":"loss","tol":1e-07,"validation_fraction":0.1,"verbose":0,'
          '"warm_start":false},"tecnica":"hist_gradient_boosting",'
          '"variables":["lag_1","lag_7","lag_14","media_7","dia_semana","mes"],'
          '"version_contrato":1,"version_reglas":1}'),
}
REGLAS_V1_COMUNES = {"horizonte_dias": 15, "ventas_reales_hasta_corte_inclusive": True,
                     "ausencias_no_son_ceros": True, "redondeo_interno": False}
REGLAS_V1_REGRESORES = {
    **REGLAS_V1_COMUNES, "predictoras": list(VARIABLES_V1),
    "onpromotion_predictor": False, "dias_naturales_exactos": True,
    "media_7": "siete observaciones de los siete días anteriores completos",
    "dia_semana": "lunes=0, domingo=6", "mes": "enero=1, diciembre=12",
    "entrenamiento_incompleto": "excluir fila", "inferencia_incompleta": "error sin imputación",
    "recursion": "incorporar solo estimaciones previas al objetivo",
    "negativos": "comprobar finitud y aplicar max(0, predicción)", "hilos_numericos": 1,
}


def solicitud_alternativa(tecnica=RF, parametros=None):
    return {"tecnica": tecnica, "version_contrato": 2, "version_reglas": 2,
            "variables": list(VARIABLES_V2), "parametros": {} if parametros is None else deepcopy(parametros)}


def registro_actual(efectiva):
    """Registro sintético completo en memoria; nunca escribe ni abre SQLite."""
    return {"version_registro": 2, "tecnica": efectiva["tecnica"],
            "parametros": deepcopy(efectiva["parametros"]),
            "configuracion_predictiva": deepcopy(efectiva),
            "reglas": config.reglas_por_configuracion(efectiva)}


class GruposPredictivosTests(unittest.TestCase):
    def test_defaults_y_json_clasicos_exactos_anteriores(self):
        for tecnica, anterior in JSON_CLASICOS.items():
            with self.subTest(tecnica=tecnica):
                esperado = json.loads(anterior)
                self.assertEqual(config.configuracion_predeterminada(tecnica), esperado)
                self.assertEqual(config.json_canonico({"tecnica": tecnica}), anterior)
                self.assertEqual(config.json_canonico(esperado), anterior)
                self.assertEqual(config.identidad_configuracion(esperado),
                                 hashlib.sha256(anterior.encode("utf-8")).hexdigest())
                # El JSON exacto también fija tipos (1 / 1.0 / True).
                self.assertEqual(config.json_canonico(config.configuracion_predeterminada(tecnica)), anterior)

    def test_reglas_historicas_conservan_json_exacto(self):
        base = {**REGLAS_V1_COMUNES,
                "referencia": "objetivo menos 7 días; real hasta corte o predicción previa",
                "respaldos": ["última observación real del mismo día semanal",
                              "última observación real disponible"], "sin_historial": "error"}
        for tecnica in (config.BASE, RF, HGB):
            esperado = base if tecnica == config.BASE else REGLAS_V1_REGRESORES
            with self.subTest(tecnica=tecnica):
                for reglas in (config.reglas_heredadas(tecnica),
                               config.reglas_por_configuracion({"tecnica": tecnica})):
                    self.assertEqual(json.dumps(reglas, sort_keys=True), json.dumps(esperado, sort_keys=True))

    def test_catalogo_versionado_coincide_con_h2a(self):
        self.assertEqual((config.VERSION_CONTRATO, config.VERSION_REGLAS), (1, 1))
        self.assertEqual((config.VERSION_CONTRATO_ALTERNATIVO, config.VERSION_REGLAS_ALTERNATIVO), (2, 2))
        self.assertEqual(config.VARIABLES, VARIABLES_V1)
        self.assertEqual(config.VARIABLES_ALTERNATIVAS_V2, VARIABLES_V2)
        self.assertEqual(config.VARIABLES_ALTERNATIVAS_V2, COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        self.assertEqual(config.VARIABLES, COLUMNAS_PREDICTORAS)

    def test_parametros_clasicos_alternativos_siguen_validos(self):
        casos = ((RF, {"n_estimators": 500, "max_depth": 50, "min_samples_leaf": 100,
                        "max_features": 6, "max_leaf_nodes": 255, "min_samples_split": 100,
                        "random_state": 2**32 - 1}),
                 (HGB, {"max_iter": 500, "learning_rate": 0.001, "max_depth": 50,
                        "max_leaf_nodes": 255, "min_samples_leaf": 100,
                        "l2_regularization": 100.0, "max_features": 0.01, "random_state": 0}))
        for tecnica, parametros in casos:
            with self.subTest(tecnica=tecnica):
                c = config.normalizar_configuracion({"tecnica": tecnica, "parametros": parametros})
                self.assertEqual(c, {**json.loads(JSON_CLASICOS[tecnica]),
                                     "parametros": {**json.loads(JSON_CLASICOS[tecnica])["parametros"], **parametros}})

    def test_todas_las_permutaciones_v1_se_recuperan_sin_conversion(self):
        for tecnica in (RF, HGB):
            for variables in permutations(VARIABLES_V1):
                with self.subTest(tecnica=tecnica, variables=variables):
                    esperado = json.loads(JSON_CLASICOS[tecnica])
                    esperado["variables"] = list(variables)
                    self.assertEqual(config.normalizar_configuracion(esperado), esperado)
                    antiguo = registro_antiguo(tecnica)
                    antiguo["reglas"]["predictoras"] = list(variables)
                    for registro in (antiguo, registro_actual(esperado)):
                        self.assertEqual(config.recuperar_configuracion(registro), esperado)
                    if variables != VARIABLES_V1:
                        self.assertNotEqual(config.identidad_configuracion(esperado),
                                            config.identidad_configuracion({"tecnica": tecnica}))

    def test_nueva_configuracion_random_forest(self):
        c = config.normalizar_configuracion(solicitud_alternativa(RF))
        self.assertEqual(c, {**json.loads(JSON_CLASICOS[RF]), "version_contrato": 2,
                             "version_reglas": 2, "variables": list(VARIABLES_V2)})

    def test_nueva_configuracion_hist_gradient_boosting(self):
        c = config.normalizar_configuracion(solicitud_alternativa(HGB))
        self.assertEqual(c, {**json.loads(JSON_CLASICOS[HGB]), "version_contrato": 2,
                             "version_reglas": 2, "variables": list(VARIABLES_V2)})

    def test_defaults_v2_implicitos_y_explicitos_equivalentes(self):
        for tecnica in (RF, HGB):
            solicitud = solicitud_alternativa(tecnica)
            del solicitud["parametros"]
            completa = config.normalizar_configuracion(solicitud)
            self.assertIsNone(config.validar_configuracion(completa))
            self.assertEqual(config.json_canonico(solicitud), config.json_canonico(completa))
            self.assertEqual(config.identidad_configuracion(solicitud), config.identidad_configuracion(completa))
            solicitud["variables"] = tuple(solicitud["variables"])
            self.assertEqual(config.normalizar_configuracion(solicitud), completa)

    def test_rechaza_versiones_mixtas_desconocidas_y_tipos_invalidos(self):
        for tecnica in (RF, HGB):
            for campo in ("version_contrato", "version_reglas"):
                for valor in (1, 0, -1, 3, True, False, 1.0, 2.0, 2.5, "2", None, [], {},
                              float("nan"), float("inf")):
                    solicitud = solicitud_alternativa(tecnica)
                    solicitud[campo] = valor
                    with self.subTest(campo=campo, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                        config.normalizar_configuracion(solicitud)
            for versiones in ((1, 2), (2, 1)):
                solicitud = {**json.loads(JSON_CLASICOS[tecnica]),
                             "version_contrato": versiones[0], "version_reglas": versiones[1]}
                with self.subTest(versiones=versiones), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion(solicitud)

    def test_grupo_v2_requiere_seleccion_completa_explicita(self):
        for campo in ("tecnica", "version_contrato", "version_reglas", "variables"):
            solicitud = solicitud_alternativa()
            del solicitud[campo]
            with self.subTest(campo=campo), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion(solicitud)
        for solicitud in ({"tecnica": RF, "variables": list(VARIABLES_V2)},
                          {"tecnica": RF, "grupo": "alternativo"},
                          {**solicitud_alternativa(), "grupo": "desconocido"},
                          solicitud_alternativa("desconocida")):
            with self.subTest(solicitud=solicitud), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion(solicitud)

    def test_variables_v2_rechazan_subconjuntos_duplicados_y_ordenes(self):
        casos = [[], list(VARIABLES_V1), list(reversed(VARIABLES_V2)),
                 [*VARIABLES_V2, "lag_1"], [*VARIABLES_V1, "lag_1"],
                 [*VARIABLES_V1, "onpromotion"], [*VARIABLES_V2, "onpromotion"],
                 ["media_14", *VARIABLES_V1], "media_14", None, {}, [True], [["lag_1"]]]
        casos.extend(list(VARIABLES_V2[:i] + VARIABLES_V2[i + 1:]) for i in range(7))
        for tecnica in (RF, HGB):
            for variables in casos:
                solicitud = {**solicitud_alternativa(tecnica), "variables": variables}
                with self.subTest(variables=variables), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion(solicitud)
            for i in range(6):
                solicitud = solicitud_alternativa(tecnica)
                solicitud["variables"][i], solicitud["variables"][i + 1] = (
                    solicitud["variables"][i + 1], solicitud["variables"][i])
                with self.subTest(posicion=i), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion(solicitud)

    def test_linea_base_permanece_exclusivamente_v1_sin_variables(self):
        for variables in ([], list(VARIABLES_V1), list(VARIABLES_V2)):
            solicitud = {**solicitud_alternativa(config.BASE), "variables": variables}
            with self.subTest(variables=variables), self.assertRaises(config.ErrorConfiguracion):
                config.normalizar_configuracion(solicitud)
        self.assertEqual(config.json_canonico({"tecnica": config.BASE}), JSON_CLASICOS[config.BASE])

    def test_max_features_rf_entero_depende_de_grupo_resuelto(self):
        for version, maximo in ((1, 6), (2, 7)):
            solicitud = {"tecnica": RF} if version == 1 else solicitud_alternativa()
            for valor in range(1, maximo + 1):
                c = config.normalizar_configuracion({**solicitud, "parametros": {"max_features": valor}})
                self.assertIs(type(c["parametros"]["max_features"]), int)
                self.assertEqual(c["parametros"]["max_features"], valor)
            for valor in (0, maximo + 1, 6.0, 7.0):
                with self.subTest(version=version, valor=valor), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({**solicitud, "parametros": {"max_features": valor}})

    def test_max_features_rf_uno_entero_difiere_de_fraccion_uno(self):
        for solicitud in ({"tecnica": RF}, solicitud_alternativa()):
            numero = {**solicitud, "parametros": {"max_features": 1}}
            fraccion = {**solicitud, "parametros": {"max_features": 1.0}}
            self.assertIs(type(config.normalizar_configuracion(numero)["parametros"]["max_features"]), int)
            self.assertIs(type(config.normalizar_configuracion(fraccion)["parametros"]["max_features"]), float)
            self.assertNotEqual(config.identidad_configuracion(numero), config.identidad_configuracion(fraccion))

    def test_max_features_fracciones_y_cadenas_conservan_reglas(self):
        for tecnica in (RF, HGB):
            for solicitud in ({"tecnica": tecnica}, solicitud_alternativa(tecnica)):
                for valor in (0.01, 0.125, 0.5, 1.0):
                    c = config.normalizar_configuracion({**solicitud, "parametros": {"max_features": valor}})
                    self.assertEqual(c["parametros"]["max_features"], valor)
                for valor in ("sqrt", "log2", 6, 7):
                    c = {**solicitud, "parametros": {"max_features": valor}}
                    if tecnica == HGB:
                        with self.assertRaises(config.ErrorConfiguracion):
                            config.normalizar_configuracion(c)
                    elif type(valor) is str:
                        self.assertEqual(config.normalizar_configuracion(c)["parametros"]["max_features"], valor)

    def test_v2_no_amplia_otros_parametros_ni_horizonte(self):
        for tecnica in (RF, HGB):
            for parametros in ({"inventado": 1}, {"warm_start": True}, {"max_depth": 51},
                               {"min_samples_leaf": 101}, {"random_state": 2**32}):
                with self.subTest(parametros=parametros), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion(solicitud_alternativa(tecnica, parametros))
            for horizonte in (True, 15.0, 14, 16, "15", None):
                with self.subTest(horizonte=horizonte), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({**solicitud_alternativa(tecnica), "horizonte": horizonte})

    def test_v2_rechaza_booleanos_no_finitos_y_tipos_incorrectos(self):
        for tecnica in (RF, HGB):
            for parametro in ("max_features", "max_depth", "random_state", "min_samples_leaf"):
                for valor in (True, False, float("nan"), float("inf"), -float("inf"), "1", [], {}, 10**400):
                    with self.subTest(tecnica=tecnica, parametro=parametro, valor=valor), \
                            self.assertRaises(config.ErrorConfiguracion):
                        config.normalizar_configuracion(solicitud_alternativa(tecnica, {parametro: valor}))
            for parametros in (None, [], "parametros"):
                with self.subTest(parametros=parametros), self.assertRaises(config.ErrorConfiguracion):
                    config.normalizar_configuracion({**solicitud_alternativa(tecnica), "parametros": parametros})

    def test_reglas_v2_explicitas_coherentes_con_h2a(self):
        esperado = {**REGLAS_V1_REGRESORES, "predictoras": list(VARIABLES_V2), "version_reglas": 2,
                    "media_14": "catorce observaciones de las catorce fechas naturales anteriores exactas",
                    "ventas_objetivo_predictor": False,
                    "calculo_medias": "math.fsum(venta / ventana); dividir antes de sumar, sin redondeo",
                    "media_14_finita": True}
        for tecnica in (RF, HGB):
            self.assertEqual(config.reglas_por_configuracion(solicitud_alternativa(tecnica)), esperado)

    def test_recuperacion_sintetica_v2_con_contrato_2_y_parametros_efectivos(self):
        for tecnica, parametros in ((RF, {}), (RF, {"max_features": 7, "n_estimators": 123}),
                                    (HGB, {}), (HGB, {"max_features": 0.5, "max_iter": 123})):
            with self.subTest(tecnica=tecnica, parametros=parametros):
                efectiva = config.normalizar_configuracion(solicitud_alternativa(tecnica, parametros))
                registro = registro_actual(efectiva)
                anterior = json.dumps(registro, sort_keys=True)
                recuperada = config.recuperar_configuracion(registro)
                self.assertEqual(config.json_canonico(recuperada), config.json_canonico(efectiva))
                recuperada["variables"].reverse()
                recuperada["parametros"]["random_state"] = 99
                self.assertEqual(json.dumps(registro, sort_keys=True), anterior)

    def test_registros_v2_rechazan_reglas_contradictorias_o_parciales(self):
        for tecnica in (RF, HGB):
            registro = registro_actual(config.normalizar_configuracion(solicitud_alternativa(tecnica)))
            for campo in registro["reglas"]:
                for borrar in (False, True):
                    c = deepcopy(registro)
                    if borrar:
                        del c["reglas"][campo]
                    else:
                        c["reglas"][campo] = "regla contradictoria"
                    with self.subTest(campo=campo, borrar=borrar), self.assertRaises(config.ErrorConfiguracion):
                        config.recuperar_configuracion(c)
            for reglas in (config.reglas_heredadas(tecnica),
                           {**registro["reglas"], "predictoras": list(reversed(VARIABLES_V2))},
                           {**registro["reglas"], "version_reglas": 2.0},
                           {**registro["reglas"], "ausencias_no_son_ceros": 1}):
                with self.subTest(reglas=reglas), self.assertRaises(config.ErrorConfiguracion):
                    config.recuperar_configuracion({**registro, "reglas": reglas})
            clasico = registro_actual(config.configuracion_predeterminada(tecnica))
            clasico["reglas"] = registro["reglas"]
            with self.assertRaises(config.ErrorConfiguracion):
                config.recuperar_configuracion(clasico)

    def test_registros_v2_rechazan_parametros_efectivos_contradictorios(self):
        for tecnica in (RF, HGB):
            registro = registro_actual(config.normalizar_configuracion(solicitud_alternativa(tecnica)))
            for bloque in ("exterior", "predictivo"):
                c = deepcopy(registro)
                parametros = c["parametros"] if bloque == "exterior" else c["configuracion_predictiva"]["parametros"]
                parametros["random_state"] = 43
                with self.subTest(bloque=bloque), self.assertRaises(config.ErrorConfiguracion):
                    config.recuperar_configuracion(c)
        registro = registro_actual(config.normalizar_configuracion(solicitud_alternativa()))
        registro["parametros"]["max_features"] = 1
        with self.assertRaises(config.ErrorConfiguracion):
            config.recuperar_configuracion(registro)  # el contrato conserva 1.0

    def test_registros_v2_parciales_nunca_reciben_defaults(self):
        for tecnica in (RF, HGB):
            registro = registro_actual(config.normalizar_configuracion(solicitud_alternativa(tecnica)))
            for campo in registro["configuracion_predictiva"]:
                c = deepcopy(registro)
                del c["configuracion_predictiva"][campo]
                with self.subTest(campo=campo), self.assertRaises(config.ErrorConfiguracion):
                    config.recuperar_configuracion(c)
            for campo in registro["parametros"]:
                for bloque in ("exterior", "predictivo"):
                    c = deepcopy(registro)
                    parametros = c["parametros"] if bloque == "exterior" else c["configuracion_predictiva"]["parametros"]
                    del parametros[campo]
                    with self.subTest(campo=campo, bloque=bloque), self.assertRaises(config.ErrorConfiguracion):
                        config.recuperar_configuracion(c)

    def test_registros_rechazan_versiones_inconsistentes_y_combinaciones_imposibles(self):
        registro = registro_actual(config.normalizar_configuracion(solicitud_alternativa()))
        casos = []
        for version in (1, 3, True, 2.0, None):
            casos.append({**deepcopy(registro), "version_registro": version})
        sin_bloque = deepcopy(registro)
        sin_bloque["version_registro"] = 1
        del sin_bloque["configuracion_predictiva"]
        casos.append(sin_bloque)
        for campo, valor in (("version_contrato", 1), ("version_reglas", 1), ("tecnica", HGB),
                              ("variables", list(VARIABLES_V1)), ("horizonte", 14)):
            c = deepcopy(registro)
            c["configuracion_predictiva"][campo] = valor
            casos.append(c)
        for tecnica in (config.BASE, HGB):
            casos.append({**deepcopy(registro), "tecnica": tecnica})
        for c in casos:
            with self.subTest(registro=c), self.assertRaises(config.ErrorConfiguracion):
                config.recuperar_configuracion(c)

    def test_v2_ceros_conservan_normalizacion_y_entradas_intactas(self):
        for tecnica, parametro in PARAMETROS_CERO_REAL:
            solicitud = solicitud_alternativa(tecnica, {parametro: -0.0})
            registro = registro_actual(config.normalizar_configuracion(solicitud))
            registro["parametros"][parametro] = -0.0
            registro["configuracion_predictiva"]["parametros"][parametro] = -0.0
            antes = json.dumps([solicitud, registro], sort_keys=True)
            positiva = solicitud_alternativa(tecnica, {parametro: 0.0})
            self.assertEqual(config.identidad_configuracion(solicitud), config.identidad_configuracion(positiva))
            self.assertEqual(config.identidad_configuracion(config.recuperar_configuracion(registro)),
                             config.identidad_configuracion(positiva))
            self.assertEqual(json.dumps([solicitud, registro], sort_keys=True), antes)

    def test_identidad_v2_versiones_parametros_y_claves_json(self):
        for tecnica in (RF, HGB):
            a = config.normalizar_configuracion(solicitud_alternativa(tecnica))
            b = dict(reversed(list(a.items())))
            b["parametros"] = dict(reversed(list(b["parametros"].items())))
            texto = config.json_canonico(a)
            self.assertEqual(texto, config.json_canonico(json.loads(json.dumps(b))))
            self.assertEqual(config.identidad_configuracion(a), hashlib.sha256(texto.encode("utf-8")).hexdigest())
            self.assertNotEqual(config.identidad_configuracion(a), config.identidad_configuracion({"tecnica": tecnica}))
            self.assertIn('"version_contrato":2', texto)
            self.assertIn('"version_reglas":2', texto)
            self.assertIn('"variables":["lag_1","lag_7","lag_14","media_7","dia_semana","mes","media_14"]', texto)
            b["parametros"]["random_state"] = 43
            self.assertNotEqual(config.identidad_configuracion(a), config.identidad_configuracion(b))

    def test_solicitudes_y_reglas_v2_no_comparten_objetos(self):
        solicitud = solicitud_alternativa()
        anterior = deepcopy(solicitud)
        normalizada = config.normalizar_configuracion(solicitud)
        reglas = config.reglas_por_configuracion(solicitud)
        config.json_canonico(solicitud)
        config.identidad_configuracion(solicitud)
        normalizada["variables"].reverse()
        normalizada["parametros"]["random_state"] = 99
        reglas["predictoras"].reverse()
        self.assertEqual(solicitud, anterior)
        self.assertEqual(config.reglas_por_configuracion(solicitud)["predictoras"], list(VARIABLES_V2))
        invalida = {**solicitud, "variables": list(reversed(VARIABLES_V2))}
        anterior = deepcopy(invalida)
        with self.assertRaises(config.ErrorConfiguracion):
            config.normalizar_configuracion(invalida)
        self.assertEqual(invalida, anterior)

    def test_motor_f3_acepta_v2_canonico_y_rechaza_permutaciones_e_invalidas_sin_entrenar(self):
        from datetime import date
        from unittest.mock import patch

        with patch.object(modelos, "crear_estimador") as crear:
            for tecnica in (RF, HGB):
                solicitud = solicitud_alternativa(tecnica)
                self.assertEqual(modelos.normalizar_configuracion_entrenamiento(tecnica, solicitud),
                                 config.normalizar_configuracion(solicitud))
                permutada = {"tecnica": tecnica, "variables": list(reversed(VARIABLES_V1))}
                self.assertIsNone(config.validar_configuracion(permutada))
                for invalida in (permutada, {**solicitud, "version_reglas": 1},
                                 {**solicitud, "variables": list(VARIABLES_V1)},
                                 {**solicitud, "variables": list(reversed(VARIABLES_V2))}):
                    with self.subTest(tecnica=tecnica, solicitud=invalida), self.assertRaises(ValueError):
                        modelos.entrenar_modelo(None, 1, "A", tecnica, date(2017, 6, 16),
                                                configuracion_solicitada=invalida)
            crear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
