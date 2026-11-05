"""Contrato puro y compatibilidad; ninguna prueba usa la base principal."""

from copy import deepcopy
import hashlib
import json
import unittest

from backend.app.services import configuracion_experimentos as config
from scripts import modelos_prediccion as modelos
from scripts.variables_predictoras import COLUMNAS_PREDICTORAS


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


if __name__ == "__main__":
    unittest.main()
