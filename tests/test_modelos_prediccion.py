"""Pruebas temporales con regresores reales pequeños y estimadores simulados."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from scripts import modelos_prediccion as modelos
from backend.app.services import configuracion_experimentos as configuracion
from scripts.preparar_series_modelado import leer_configuracion
from threadpoolctl import threadpool_info

from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS, construir_variables_futuras,
)


CORTE = date(2017, 6, 16)


def observaciones(tienda=1, familia="A", escala=1.0):
    return pd.DataFrame({
        "date": [(CORTE - timedelta(days=59-i)).isoformat() for i in range(60)],
        "store_nbr": tienda, "family": familia,
        "sales": [(1 + i * 0.125) * escala for i in range(60)], "onpromotion": 0,
    })


def con_futuro(datos, valor):
    futuro = pd.concat([datos.iloc[[-1]]] * 15, ignore_index=True)
    futuro["date"] = [(CORTE + timedelta(days=i)).isoformat() for i in range(1, 16)]
    futuro["sales"] = valor
    return pd.concat([datos, futuro], ignore_index=True)


class EstimadorSimulado:
    def __init__(self, valor=None):
        self.valor = valor
        self.entradas = []

    def fit(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        return self

    def predict(self, x):
        self.entradas.append(x.copy())
        return [float(x.iloc[0]["lag_1"]) + 0.25 if self.valor is None else self.valor]


def modelo_simulado(estimador):
    return modelos.ModeloEntrenado(1, "A", "random_forest", CORTE,
                                   estimador, 46, 14, CORTE, 0.0)


class ModelosPrediccionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos = observaciones()
        cls.entrenados = {t: modelos.entrenar_modelo(cls.datos, 1, "A", t, CORTE)
                          for t in modelos.TECNICAS}
        cls.solicitudes = {
            "random_forest": {"tecnica": "random_forest", "parametros": {
                "n_estimators": 200, "max_depth": 4, "min_samples_leaf": 2,
                "min_samples_split": 4, "max_leaf_nodes": 12, "max_features": "sqrt", "random_state": 7}},
            "hist_gradient_boosting": {"tecnica": "hist_gradient_boosting", "parametros": {
                "max_iter": 150, "learning_rate": 0.1, "max_depth": 3, "max_leaf_nodes": 7,
                "min_samples_leaf": 5, "l2_regularization": 0.5, "max_features": 0.8, "random_state": 11}},
        }
        cls.alternativos = {t: modelos.entrenar_modelo(cls.datos, 1, "A", t, CORTE,
                                                      configuracion_solicitada=cls.solicitudes[t])
                            for t in modelos.TECNICAS}

    def test_random_forest_con_seis_variables(self):
        m = self.entrenados["random_forest"]
        self.assertIsInstance(m.estimador, RandomForestRegressor)
        self.assertEqual(tuple(m.estimador.feature_names_in_), COLUMNAS_PREDICTORAS)
        self.assertEqual(m.filas_validas, 46)
        self.assertEqual(m.filas_descartadas, 14)

    def test_hist_gradient_boosting_con_seis_variables(self):
        m = self.entrenados["hist_gradient_boosting"]
        self.assertIsInstance(m.estimador, HistGradientBoostingRegressor)
        self.assertEqual(tuple(m.estimador.feature_names_in_), COLUMNAS_PREDICTORAS)
        self.assertEqual(m.estimador.n_iter_, 100)

    def test_configuraciones_fijas(self):
        esperado_rf = dict(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)
        esperado_hgb = dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                            min_samples_leaf=20, early_stopping=False, random_state=42)
        for tecnica, parametros in zip(modelos.TECNICAS, (esperado_rf, esperado_hgb)):
            actual = self.entrenados[tecnica].estimador.get_params()
            self.assertEqual({k: actual[k] for k in parametros}, parametros)

    def test_random_forest_alternativo_entrena_200_arboles_con_parametros_solicitados(self):
        m = self.alternativos["random_forest"]
        self.assertIsInstance(m.estimador, RandomForestRegressor)
        self.assertEqual(len(m.estimador.estimators_), 200)
        for nombre, valor in self.solicitudes["random_forest"]["parametros"].items():
            self.assertEqual(m.estimador.get_params()[nombre], valor)
        self.assertEqual(m.estimador.n_jobs, 1)
        self.assertTrue(all(a.get_depth() <= 4 for a in m.estimador.estimators_))

    def test_hist_gradient_boosting_alternativo_entrena_150_iteraciones_y_regularizacion(self):
        m = self.alternativos["hist_gradient_boosting"]
        self.assertIsInstance(m.estimador, HistGradientBoostingRegressor)
        self.assertEqual(m.estimador.n_iter_, 150)
        for nombre, valor in self.solicitudes["hist_gradient_boosting"]["parametros"].items():
            self.assertEqual(m.estimador.get_params()[nombre], valor)
        self.assertIs(m.estimador.early_stopping, False)

    def test_get_params_completo_coincide_con_contrato_en_modelos_realmente_entrenados(self):
        for t, m in self.alternativos.items():
            esperado = configuracion.normalizar_configuracion(self.solicitudes[t])["parametros"]
            self.assertEqual(json.dumps(m.estimador.get_params(deep=False), sort_keys=True),
                             json.dumps(esperado, sort_keys=True))
            self.assertEqual(tuple(m.estimador.feature_names_in_), COLUMNAS_PREDICTORAS)
            self.assertEqual((m.filas_validas, m.filas_descartadas, m.ultima_fecha_entrenamiento),
                             (46, 14, CORTE))
            self.assertIsNot(m.estimador, self.entrenados[t].estimador)

    def test_defaults_explicitos_conservan_resultados_originales(self):
        for t, anterior in self.entrenados.items():
            with self.subTest(tecnica=t):
                actual = modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE,
                    configuracion_solicitada=configuracion.configuracion_predeterminada(t))
                self.assertEqual(modelos.pronosticar_recursivo(actual, self.datos),
                                 modelos.pronosticar_recursivo(anterior, self.datos))

    def test_defaults_coinciden_con_los_constructores_originales_anteriores_a_f3(self):
        originales = {
            "random_forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1),
            "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05,
                max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False, random_state=42),
        }
        tabla = modelos.construir_tabla_serie(self.datos, 1, "A", CORTE).tabla
        for t, estimador in originales.items():
            with self.subTest(tecnica=t):
                self.assertEqual(json.dumps(estimador.get_params(deep=False), sort_keys=True),
                                 json.dumps(self.entrenados[t].estimador.get_params(deep=False), sort_keys=True))
                with modelos.threadpool_limits(limits=1):
                    estimador.fit(tabla.loc[:, list(COLUMNAS_PREDICTORAS)], tabla["sales"])
                anterior = modelos.ModeloEntrenado(1, "A", t, CORTE, estimador, 46, 14, CORTE, 0.0)
                self.assertEqual(modelos.pronosticar_recursivo(anterior, self.datos),
                                 modelos.pronosticar_recursivo(self.entrenados[t], self.datos))

    def test_configuracion_alternativa_es_determinista_y_no_comparte_instancia(self):
        for t, anterior in self.alternativos.items():
            with self.subTest(tecnica=t):
                actual = modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE,
                                                 configuracion_solicitada=self.solicitudes[t])
                self.assertIsNot(actual.estimador, anterior.estimador)
                self.assertEqual(modelos.pronosticar_recursivo(actual, self.datos),
                                 modelos.pronosticar_recursivo(anterior, self.datos))

    def test_alternativos_generan_quince_fracciones_finitas_con_variables_originales_y_recursion(self):
        original = construir_variables_futuras
        for t, m in self.alternativos.items():
            consultas = []

            def observar(historial, tienda, familia, corte, objetivo, previas):
                x = original(historial, tienda, familia, corte, objetivo, previas)
                self.assertEqual(tuple(x.columns), COLUMNAS_PREDICTORAS)
                self.assertNotIn("onpromotion", x.columns)
                consultas.append(dict(previas))
                return x

            with self.subTest(tecnica=t), patch.object(modelos, "construir_variables_futuras", side_effect=observar):
                p = modelos.pronosticar_recursivo(m, self.datos)
            self.assertEqual([v.fecha_objetivo for v in p], [CORTE + timedelta(days=i) for i in range(1, 16)])
            self.assertEqual([v.paso for v in p], list(range(1, 16)))
            self.assertTrue(all(isfinite(v.venta_estimada) and v.venta_estimada >= 0 for v in p))
            self.assertTrue(any(v.venta_estimada != int(v.venta_estimada) for v in p))
            self.assertEqual([len(v) for v in consultas], list(range(15)))
            self.assertEqual(consultas[-1], {v.fecha_objetivo: v.venta_estimada for v in p[:-1]})

    def test_alternativos_ignoran_ventas_y_promociones_futuras_sin_modificar_dataframe(self):
        for t, anterior in self.alternativos.items():
            with self.subTest(tecnica=t):
                entrada = con_futuro(self.datos, "NO_LEER_VENTAS")
                entrada["onpromotion"] = entrada["onpromotion"].astype(object)
                entrada.loc[entrada["date"] > CORTE.isoformat(), "onpromotion"] = "NO_LEER_PROMOCIONES"
                copia = entrada.copy(deep=True)
                solicitud = deepcopy(self.solicitudes[t])
                actual = modelos.entrenar_modelo(entrada, 1, "A", t, CORTE,
                                                 configuracion_solicitada=solicitud)
                self.assertEqual(modelos.pronosticar_recursivo(actual, entrada),
                                 modelos.pronosticar_recursivo(anterior, self.datos))
                self.assertEqual(actual.ultima_fecha_entrenamiento, CORTE)
                pd.testing.assert_frame_equal(entrada, copia)
                self.assertEqual(solicitud, self.solicitudes[t])

    def test_configuraciones_invalidas_se_rechazan_antes_de_preparar_o_entrenar(self):
        casos = [
            {"tecnica": "random_forest", "parametros": {"n_estimators": True}},
            {"tecnica": "random_forest", "parametros": {"n_jobs": 2}},
            {"tecnica": "random_forest", "parametros": {"desconocido": 1}},
            {"tecnica": "random_forest", "parametros": {"max_depth": -1}},
            {"tecnica": "hist_gradient_boosting"},
            {"tecnica": "random_forest", "variables": list(reversed(COLUMNAS_PREDICTORAS))},
            {"tecnica": "random_forest", "variables": list(COLUMNAS_PREDICTORAS[:-1])},
            {"tecnica": "random_forest", "variables": [*COLUMNAS_PREDICTORAS, "onpromotion"]},
        ]
        with patch.object(modelos, "construir_tabla_serie") as preparar, \
                patch.object(RandomForestRegressor, "fit") as ajustar:
            for solicitud in casos:
                with self.subTest(solicitud=solicitud), self.assertRaises(ValueError):
                    modelos.entrenar_modelo(self.datos, 1, "A", "random_forest", CORTE,
                                            configuracion_solicitada=solicitud)
                with self.assertRaises(ValueError):
                    modelos.crear_estimador("random_forest", configuracion_solicitada=solicitud)
            preparar.assert_not_called()
            ajustar.assert_not_called()
        for parametros in ({"early_stopping": True}, {"learning_rate": float("nan")}, {"max_iter": 501}):
            with patch.object(HistGradientBoostingRegressor, "fit") as ajustar, self.assertRaises(ValueError):
                modelos.entrenar_modelo(self.datos, 1, "A", "hist_gradient_boosting", CORTE,
                    configuracion_solicitada={"tecnica": "hist_gradient_boosting", "parametros": parametros})
            ajustar.assert_not_called()

    def test_comprobacion_efectiva_distingue_numero_y_fraccion_max_features(self):
        t = "random_forest"
        entera = {"tecnica": t, "parametros": {"n_estimators": 20, "max_features": 1}}
        fraccion = {"tecnica": t, "parametros": {"n_estimators": 20, "max_features": 1.0}}
        a = modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE, configuracion_solicitada=entera)
        b = modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE, configuracion_solicitada=fraccion)
        self.assertIs(type(a.estimador.max_features), int)
        self.assertIs(type(b.estimador.max_features), float)
        with self.assertRaisesRegex(ValueError, "parámetros reales"):
            modelos.comprobar_parametros_efectivos(a.estimador, configuracion.normalizar_configuracion(fraccion))

    def test_no_entrena_si_constructor_ignora_parametros_solicitados(self):
        t = "random_forest"
        with patch.object(modelos, "crear_estimador", return_value=RandomForestRegressor(
                n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)), \
                patch.object(RandomForestRegressor, "fit") as ajustar:
            with self.assertRaisesRegex(ValueError, "parámetros reales"):
                modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE,
                    configuracion_solicitada={"tecnica": t, "parametros": {"n_estimators": 200}})
            ajustar.assert_not_called()

    def test_alternativos_mantienen_limite_de_un_hilo_en_fit_y_prediccion(self):
        limitar = modelos.threadpool_limits
        for t, parametros in (("random_forest", {"n_estimators": 20}),
                               ("hist_gradient_boosting", {"max_iter": 20})):
            with self.subTest(tecnica=t), patch.object(modelos, "threadpool_limits", wraps=limitar) as limites:
                m = modelos.entrenar_modelo(self.datos, 1, "A", t, CORTE,
                    configuracion_solicitada={"tecnica": t, "parametros": parametros})
                modelos.pronosticar_recursivo(m, self.datos)
            self.assertEqual(len(limites.call_args_list), 2)
            self.assertTrue(all(c.kwargs == {"limits": 1} for c in limites.call_args_list))

    def test_entrenamiento_independiente_por_tienda_familia_y_tecnica(self):
        claves = ((1, "A"), (1, "B"), (2, "A"))
        datos = pd.concat([observaciones(t, f, i) for i, (t, f) in enumerate(claves, 1)],
                          ignore_index=True)
        with patch.object(modelos, "crear_estimador", side_effect=lambda _: EstimadorSimulado()):
            resultados = modelos.ejecutar_modelos(datos, CORTE, claves)
        self.assertEqual(len({id(r.modelo.estimador) for r in resultados}), 6)
        for i, (t, f) in enumerate(claves, 1):
            grupo = [r for r in resultados if (r.modelo.store_nbr, r.modelo.family) == (t, f)]
            self.assertEqual(len(grupo), 2)
            for r in grupo:
                self.assertAlmostEqual(r.modelo.estimador.y.iloc[-1], 8.375 * i)
                self.assertEqual(tuple(r.modelo.estimador.x.columns), COLUMNAS_PREDICTORAS)

    def test_horizonte_y_fechas_consecutivas(self):
        for m in self.entrenados.values():
            p = modelos.pronosticar_recursivo(m, self.datos)
            self.assertEqual(len(p), 15)
            self.assertEqual([f.paso for f in p], list(range(1, 16)))
            self.assertEqual([f.fecha_objetivo for f in p],
                             [CORTE + timedelta(days=i) for i in range(1, 16)])

    def test_recursion_reutiliza_estimados(self):
        estimador = EstimadorSimulado()
        p = modelos.pronosticar_recursivo(modelo_simulado(estimador), self.datos)
        ultimo = self.datos["sales"].iloc[-1]
        self.assertEqual([f.venta_estimada for f in p], [ultimo + i * 0.25 for i in range(1, 16)])
        self.assertEqual(estimador.entradas[1].iloc[0]["lag_1"], p[0].venta_estimada)
        self.assertEqual(estimador.entradas[7].iloc[0]["lag_7"], p[0].venta_estimada)
        self.assertEqual(estimador.entradas[14].iloc[0]["lag_14"], p[0].venta_estimada)

    def test_ventas_futuras_no_afectan_fit_ni_pronostico(self):
        entrada = con_futuro(self.datos, 1e12)
        with patch.object(modelos, "crear_estimador", side_effect=lambda _: EstimadorSimulado()):
            a = modelos.entrenar_modelo(self.datos, 1, "A", "random_forest", CORTE)
            b = modelos.entrenar_modelo(entrada, 1, "A", "random_forest", CORTE)
        pd.testing.assert_frame_equal(a.estimador.x, b.estimador.x)
        pd.testing.assert_series_equal(a.estimador.y, b.estimador.y)
        self.assertEqual(a.ultima_fecha_entrenamiento, CORTE)
        self.assertEqual(modelos.pronosticar_recursivo(a, self.datos),
                         modelos.pronosticar_recursivo(b, entrada))

    def test_lector_limitado_al_corte(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            self.datos.to_csv(ruta, index=False)
            with ruta.open("a", encoding="utf-8") as archivo:
                archivo.write("2017-06-17,1,A,NO_LEER,NO_LEER\n")
            historico = modelos.cargar_historico_csv(ruta, ((1, "A"),), CORTE, tamano_bloque=11)
        self.assertEqual(len(historico), 60)
        self.assertEqual(historico["date"].max().date(), CORTE)
        self.assertEqual(historico["sales"].iloc[-1], 8.375)

    def test_fracciones_de_respuesta_sin_conversion_a_entero(self):
        estimador = EstimadorSimulado()
        with patch.object(modelos, "crear_estimador", return_value=estimador):
            modelos.entrenar_modelo(self.datos, 1, "A", "random_forest", CORTE)
        self.assertEqual(estimador.y.iloc[-1], 8.375)
        self.assertTrue(pd.api.types.is_float_dtype(estimador.y))

    def test_predicciones_reales_finitas_no_negativas(self):
        for m in self.entrenados.values():
            self.assertTrue(all(isfinite(p.venta_estimada) and p.venta_estimada >= 0
                                for p in modelos.pronosticar_recursivo(m, self.datos)))

    def test_ajuste_negativo_y_reutilizacion_del_cero(self):
        estimador = EstimadorSimulado(-0.125)
        m = modelo_simulado(estimador)
        p = modelos.pronosticar_recursivo(m, self.datos)
        self.assertTrue(all(f.venta_estimada == 0.0 and f.ajustada_a_cero for f in p))
        self.assertEqual(modelos.Resultado(m, p, 0.0).ajustes_a_cero, 15)
        self.assertEqual(estimador.entradas[1].iloc[0]["lag_1"], 0.0)

    def test_rechaza_nan_inf_y_valores_no_numericos(self):
        for valor in (float("nan"), float("inf"), -float("inf"), "3.25", None, True):
            # None significa recursivo en el doble; se sustituye por una salida
            # explícita para comprobar el contrato de predict.
            estimador = EstimadorSimulado(valor)
            if valor is None:
                estimador.predict = lambda x: [None]
            with self.subTest(valor=valor), self.assertRaisesRegex(ValueError, "objetivo 2017-06-17"):
                modelos.pronosticar_recursivo(modelo_simulado(estimador), self.datos)

    def test_error_sin_filas_completas(self):
        with self.assertRaisesRegex(ValueError, "1/A.*2017-06-16.*No hay filas completas"):
            modelos.entrenar_modelo(self.datos.iloc[-7:], 1, "A", "random_forest", CORTE)

    def test_error_historial_incompleto_para_pronosticar(self):
        datos = self.datos.drop(self.datos.index[-1])
        with self.assertRaisesRegex(ValueError, "1/A.*objetivo 2017-06-17.*lag_1"):
            modelos.pronosticar_recursivo(self.entrenados["random_forest"], datos)

    def test_determinismo_de_ambas_tecnicas(self):
        for tecnica, primero in self.entrenados.items():
            segundo = modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE)
            self.assertIsNot(primero.estimador, segundo.estimador)
            self.assertEqual(modelos.pronosticar_recursivo(primero, self.datos),
                             modelos.pronosticar_recursivo(segundo, self.datos))

    def test_metadatos_estructurados(self):
        for tecnica, modelo in self.entrenados.items():
            p = modelos.pronosticar_recursivo(modelo, self.datos)
            self.assertTrue(all((f.store_nbr, f.family, f.tecnica, f.fecha_corte)
                                == (1, "A", tecnica, CORTE) for f in p))

    def test_diez_series_dos_tecnicas_trescientas_predicciones(self):
        claves = leer_configuracion()
        datos = pd.concat([observaciones(t, f) for t, f in claves], ignore_index=True)
        with patch.object(modelos, "crear_estimador", side_effect=lambda _: EstimadorSimulado()):
            resultados = modelos.ejecutar_modelos(datos, CORTE)
        self.assertEqual(len(resultados), 20)
        self.assertEqual(sum(len(r.predicciones) for r in resultados), 300)
        self.assertTrue(all(len(r.predicciones) == 15 for r in resultados))
        self.assertEqual({(r.modelo.store_nbr, r.modelo.family) for r in resultados}, set(claves))

    def test_nunca_entrega_ventas_reales_objetivo_a_variables(self):
        entrada = con_futuro(self.datos, "VENTA_PROHIBIDA")
        original = entrada.copy(deep=True)
        consultas = []

        def verificar(historial, tienda, familia, corte, objetivo, previas):
            self.assertTrue(historial["date"].le(pd.Timestamp(corte)).all())
            self.assertFalse(historial["date"].eq(pd.Timestamp(objetivo)).any())
            consultas.append((objetivo, dict(previas)))
            return construir_variables_futuras(historial, tienda, familia, corte, objetivo, previas)

        with patch.object(modelos, "construir_variables_futuras", side_effect=verificar):
            p = modelos.pronosticar_recursivo(modelo_simulado(EstimadorSimulado()), entrada)
        self.assertEqual(len(consultas), 15)
        self.assertEqual([len(previas) for _, previas in consultas], list(range(15)))
        self.assertEqual(len(p), 15)
        pd.testing.assert_frame_equal(entrada, original)

    def test_corte_alternativo(self):
        corte = CORTE - timedelta(days=2)
        modelo = modelos.entrenar_modelo(self.datos, 1, "A", "random_forest", corte)
        p = modelos.pronosticar_recursivo(modelo, self.datos)
        self.assertEqual(modelo.ultima_fecha_entrenamiento, corte)
        self.assertEqual(p[0].fecha_objetivo, corte + timedelta(days=1))
        self.assertTrue(all(f.fecha_corte == corte for f in p))

    def test_error_serie_y_tecnica_desconocidas(self):
        with self.assertRaisesRegex(ValueError, "99/Z.*corte 2017-06-16"):
            modelos.entrenar_modelo(self.datos, 99, "Z", "random_forest", CORTE)
        with self.assertRaisesRegex(ValueError, "Técnica desconocida"):
            modelos.crear_estimador("otra")


class GruposMotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos = observaciones()
        cls.datos.loc[cls.datos.index % 9 == 0, "sales"] = 0.0
        cls.solicitudes = {
            t: {"tecnica": t, "version_contrato": 2, "version_reglas": 2,
                "variables": list(COLUMNAS_PREDICTORAS_ALTERNATIVAS),
                "parametros": {"random_state": 13, "min_samples_leaf": 2,
                    **({"n_estimators": 23, "max_features": 7} if t == "random_forest"
                       else {"max_iter": 25, "learning_rate": 0.1})}}
            for t in modelos.TECNICAS}
        cls.entrenados = {t: modelos.entrenar_modelo(cls.datos, 1, "A", t, CORTE,
            configuracion_solicitada=cls.solicitudes[t]) for t in modelos.TECNICAS}

    def comprobar_estimador_v2(self, tecnica, clase):
        m = self.entrenados[tecnica]
        self.assertIsInstance(m.estimador, clase)
        self.assertEqual(m.columnas_predictoras, COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        self.assertEqual(tuple(m.estimador.feature_names_in_), m.columnas_predictoras)
        self.assertEqual(m.estimador.n_features_in_, 7)
        esperada = configuracion.normalizar_configuracion(self.solicitudes[tecnica])
        self.assertEqual(json.dumps(m.estimador.get_params(deep=False), sort_keys=True),
                         json.dumps(esperada["parametros"], sort_keys=True))
        self.assertEqual((m.filas_validas, m.filas_descartadas, m.ultima_fecha_entrenamiento),
                         (46, 14, CORTE))

    def test_rf_real_siete_variables_y_max_features_entero_siete(self):
        self.comprobar_estimador_v2("random_forest", RandomForestRegressor)
        self.assertIs(type(self.entrenados["random_forest"].estimador.max_features), int)
        self.assertEqual(self.entrenados["random_forest"].estimador.max_features, 7)

    def test_hgb_real_siete_variables(self):
        self.comprobar_estimador_v2("hist_gradient_boosting", HistGradientBoostingRegressor)

    def test_fit_y_quince_predict_reales_mismo_esquema_respuesta_y_un_hilo(self):
        for tecnica, clase in (("random_forest", RandomForestRegressor),
                               ("hist_gradient_boosting", HistGradientBoostingRegressor)):
            for columnas in (COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS):
                with self.subTest(tecnica=tecnica, columnas=columnas):
                    solicitud = self.solicitudes[tecnica] if len(columnas) == 7 else {"tecnica": tecnica}
                    tabla = modelos.construir_tabla_serie(self.datos, 1, "A", CORTE,
                        columnas_predictoras=columnas).tabla
                    ajustar, predecir = clase.fit, clase.predict
                    entradas = []
                    hilos = []

                    def fit(estimador, x, y):
                        pd.testing.assert_frame_equal(x, tabla.loc[:, list(columnas)])
                        pd.testing.assert_series_equal(y, tabla["sales"])
                        self.assertEqual(y.name, "sales")
                        self.assertTrue((y == 0).any())
                        self.assertTrue((y % 1 != 0).any())
                        self.assertTrue(tabla["fecha_objetivo"].le(CORTE).all())
                        self.assertNotIn("sales", x.columns)
                        self.assertNotIn("onpromotion", x.columns)
                        hilos.extend(threadpool_info())
                        return ajustar(estimador, x, y)

                    def predict(estimador, x):
                        self.assertEqual(tuple(x.columns), columnas)
                        self.assertEqual(tuple(estimador.feature_names_in_), columnas)
                        self.assertEqual(estimador.n_features_in_, len(columnas))
                        entradas.append(x.copy())
                        hilos.extend(threadpool_info())
                        return predecir(estimador, x)

                    with patch.object(clase, "fit", new=fit), patch.object(clase, "predict", new=predict), \
                            patch.object(modelos, "comprobar_parametros_efectivos",
                                wraps=modelos.comprobar_parametros_efectivos) as parametros:
                        m = modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                                   configuracion_solicitada=solicitud)
                        p = modelos.pronosticar_recursivo(m, self.datos)
                    self.assertGreaterEqual(parametros.call_count, 3)  # constructor, antes y después de fit
                    self.assertEqual(len(entradas), 15)
                    self.assertEqual([v.fecha_objetivo for v in p],
                                     [CORTE + timedelta(days=i) for i in range(1, 16)])
                    self.assertEqual([v.paso for v in p], list(range(1, 16)))
                    self.assertTrue(all(isfinite(v.venta_estimada) and v.venta_estimada >= 0 for v in p))
                    self.assertTrue(any(v.venta_estimada % 1 != 0 for v in p))
                    self.assertTrue(hilos)
                    self.assertTrue(all(info["num_threads"] == 1 for info in hilos))

    def test_v2_reproducible_instancias_nuevas_y_entradas_intactas(self):
        for tecnica, anterior in self.entrenados.items():
            with self.subTest(tecnica=tecnica):
                entrada = con_futuro(self.datos, "VENTA_FUTURA_PROHIBIDA")
                entrada["onpromotion"] = entrada["onpromotion"].astype(object)
                entrada.loc[entrada["date"] > CORTE.isoformat(), "onpromotion"] = "NO_LEER"
                copia = entrada.copy(deep=True)
                solicitud = deepcopy(self.solicitudes[tecnica])
                actual = modelos.entrenar_modelo(entrada, 1, "A", tecnica, CORTE,
                                                 configuracion_solicitada=solicitud)
                self.assertIsNot(actual.estimador, anterior.estimador)
                self.assertEqual(modelos.pronosticar_recursivo(actual, entrada),
                                 modelos.pronosticar_recursivo(anterior, self.datos))
                self.assertEqual((actual.filas_validas, actual.filas_descartadas), (46, 14))
                pd.testing.assert_frame_equal(entrada, copia)
                self.assertEqual(solicitud, self.solicitudes[tecnica])

    def test_recursion_v2_media_14_paso_15_y_predicciones_previas_explicitas(self):
        for tecnica, modelo in self.entrenados.items():
            consultas = []
            xs = []
            def observar(historial, tienda, familia, corte, objetivo, previas, **kwargs):
                self.assertEqual(kwargs, {"columnas_predictoras": COLUMNAS_PREDICTORAS_ALTERNATIVAS})
                self.assertTrue(historial["date"].le(pd.Timestamp(corte)).all())
                self.assertFalse(historial["date"].eq(pd.Timestamp(objetivo)).any())
                consultas.append(dict(previas))
                x = construir_variables_futuras(historial, tienda, familia, corte, objetivo, previas, **kwargs)
                xs.append(x)
                return x
            with self.subTest(tecnica=tecnica), \
                    patch.object(modelos, "construir_variables_futuras", side_effect=observar):
                p = modelos.pronosticar_recursivo(modelo, con_futuro(self.datos, "PROHIBIDA"))
            self.assertEqual([len(v) for v in consultas], list(range(15)))
            for i, previas in enumerate(consultas):
                self.assertEqual(previas, {v.fecha_objetivo: v.venta_estimada for v in p[:i]})
            self.assertAlmostEqual(xs[14].iloc[0]["media_14"], sum(v.venta_estimada / 14 for v in p[:14]))
            self.assertEqual(xs[1].iloc[0]["lag_1"], p[0].venta_estimada)
            self.assertEqual(xs[7].iloc[0]["lag_7"], p[0].venta_estimada)
            self.assertEqual(xs[14].iloc[0]["lag_14"], p[0].venta_estimada)

    def test_venta_objetivo_no_integra_x_de_entrenamiento(self):
        for columnas in (COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS):
            original = modelos.construir_tabla_serie(self.datos, 1, "A", CORTE,
                                                    columnas_predictoras=columnas).tabla
            cambiada = self.datos.copy(deep=True)
            cambiada.loc[cambiada["date"] == CORTE.isoformat(), "sales"] = 99999.75
            actual = modelos.construir_tabla_serie(cambiada, 1, "A", CORTE,
                                                  columnas_predictoras=columnas).tabla
            pd.testing.assert_frame_equal(original.loc[:, list(columnas)], actual.loc[:, list(columnas)])
            self.assertEqual(actual["sales"].iloc[-1], 99999.75)

    def test_ausencia_t_menos_10_descarta_objetivo_solo_en_alternativo_sin_imputar(self):
        datos = self.datos.loc[self.datos["date"] != (CORTE - timedelta(days=10)).isoformat()].copy()
        for tecnica in modelos.TECNICAS:
            for columnas in (COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS):
                with self.subTest(tecnica=tecnica, columnas=columnas):
                    preparado = modelos.construir_tabla_serie(datos, 1, "A", CORTE,
                                                             columnas_predictoras=columnas)
                    solicitud = self.solicitudes[tecnica] if len(columnas) == 7 else {"tecnica": tecnica}
                    m = modelos.entrenar_modelo(datos, 1, "A", tecnica, CORTE,
                                               configuracion_solicitada=solicitud)
                    self.assertEqual(m.filas_validas + m.filas_descartadas, 59)
                    self.assertEqual(m.filas_validas, len(preparado.tabla))
                    self.assertEqual(m.filas_descartadas, len(preparado.descartes))
                    self.assertEqual(CORTE in set(preparado.tabla["fecha_objetivo"]), len(columnas) == 6)
                    self.assertNotIn(CORTE - timedelta(days=10), preparado.tabla["fecha_objetivo"].values)
                    if len(columnas) == 7:
                        descarte = next(d for d in preparado.descartes if d.fecha_objetivo == CORTE)
                        self.assertEqual(set(descarte.faltantes), {"media_14"})

    def test_historial_incompleto_v2_error_controlado_en_fit_y_predict(self):
        for tecnica, m in self.entrenados.items():
            with self.subTest(tecnica=tecnica):
                with self.assertRaisesRegex(ValueError, "No hay filas completas"):
                    modelos.entrenar_modelo(self.datos.iloc[-14:], 1, "A", tecnica, CORTE,
                                           configuracion_solicitada=self.solicitudes[tecnica])
                datos = self.datos.loc[self.datos["date"] != (CORTE - timedelta(days=9)).isoformat()]
                with patch.object(m.estimador, "predict") as predecir:
                    with self.assertRaisesRegex(ValueError, "objetivo 2017-06-17.*media_14"):
                        modelos.pronosticar_recursivo(m, datos)
                    predecir.assert_not_called()

    def test_minimo_tecnico_una_fila_completa_ambos_grupos_y_regresores(self):
        datos = self.datos.iloc[-15:]
        for tecnica in modelos.TECNICAS:
            for solicitud in ({"tecnica": tecnica}, self.solicitudes[tecnica]):
                with self.subTest(solicitud=solicitud):
                    m = modelos.entrenar_modelo(datos, 1, "A", tecnica, CORTE,
                                               configuracion_solicitada=solicitud)
                    self.assertEqual((m.filas_validas, m.filas_descartadas), (1, 14))

    def test_grupos_invalidos_rechazados_antes_de_preparar_o_fit(self):
        for tecnica, solicitud in self.solicitudes.items():
            casos = [{**solicitud, "variables": list(v)} for v in (
                COLUMNAS_PREDICTORAS_ALTERNATIVAS[:-1], tuple(reversed(COLUMNAS_PREDICTORAS_ALTERNATIVAS)),
                (*COLUMNAS_PREDICTORAS_ALTERNATIVAS, "lag_1"), (*COLUMNAS_PREDICTORAS, "onpromotion"))]
            casos += [{**solicitud, "grupo": "desconocido"}, {**solicitud, "version_contrato": 1},
                      {**solicitud, "version_reglas": 1}, {**solicitud, "version_reglas": 3},
                      {"tecnica": tecnica, "variables": list(reversed(COLUMNAS_PREDICTORAS))},
                      {**solicitud, "tecnica": next(t for t in modelos.TECNICAS if t != tecnica)}]
            if tecnica == "random_forest":
                casos.append({"tecnica": tecnica, "parametros": {"max_features": 7}})
            with patch.object(modelos, "construir_tabla_serie") as preparar, \
                    patch.object(modelos, "crear_estimador") as crear:
                for invalida in casos:
                    copia = deepcopy(invalida)
                    with self.subTest(solicitud=invalida), self.assertRaises(ValueError):
                        modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                               configuracion_solicitada=invalida)
                    self.assertEqual(invalida, copia)
                preparar.assert_not_called()
                crear.assert_not_called()

    def test_metadatos_inmutables_defaults_y_constructor_posicional_historico(self):
        antiguo = modelo_simulado(EstimadorSimulado())
        self.assertEqual(antiguo.columnas_predictoras, COLUMNAS_PREDICTORAS)
        columnas = list(COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        nuevo = replace(antiguo, columnas_predictoras=columnas)
        columnas.reverse()
        self.assertIs(type(nuevo.columnas_predictoras), tuple)
        self.assertEqual(nuevo.columnas_predictoras, COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        with self.assertRaises(FrozenInstanceError):
            nuevo.columnas_predictoras = COLUMNAS_PREDICTORAS
        with self.assertRaises(TypeError):
            modelos.ModeloEntrenado(1, "A", "random_forest", CORTE, EstimadorSimulado(),
                                   46, 14, CORTE, 0.0, COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        for columnas in ((), COLUMNAS_PREDICTORAS[:-1], tuple(reversed(COLUMNAS_PREDICTORAS)),
                          (*COLUMNAS_PREDICTORAS, "onpromotion"), "lag_1"):
            with self.subTest(columnas=columnas), self.assertRaises(ValueError):
                replace(antiguo, columnas_predictoras=columnas)

    def test_no_infiere_grupo_alternativo_en_objeto_antiguo(self):
        m = self.entrenados["random_forest"]
        antiguo = modelos.ModeloEntrenado(1, "A", m.tecnica, CORTE, m.estimador, 46, 14, CORTE, 0.0)
        with patch.object(m.estimador, "predict") as predecir, self.assertRaisesRegex(ValueError, "n_features_in_"):
            modelos.pronosticar_recursivo(antiguo, self.datos)
        predecir.assert_not_called()

    def test_rechaza_atributos_sklearn_incoherentes_antes_de_predict(self):
        for tecnica, m in self.entrenados.items():
            for atributo, valor in (("n_features_in_", 6),
                                    ("feature_names_in_", list(reversed(m.columnas_predictoras)))):
                with self.subTest(tecnica=tecnica, atributo=atributo), \
                        patch.object(m.estimador, atributo, valor), patch.object(m.estimador, "predict") as predecir:
                    with self.assertRaisesRegex(ValueError, atributo):
                        modelos.pronosticar_recursivo(m, self.datos)
                    predecir.assert_not_called()

    def test_rechaza_x_predict_incompleta_o_reordenada_sin_corregir_silenciosamente(self):
        for m in self.entrenados.values():
            original = construir_variables_futuras(self.datos, 1, "A", CORTE, CORTE + timedelta(days=1),
                                                  columnas_predictoras=m.columnas_predictoras)
            for x in (original.iloc[:, :-1], original.iloc[:, ::-1]):
                with patch.object(modelos, "construir_variables_futuras", return_value=x), \
                        patch.object(m.estimador, "predict") as predecir:
                    with self.assertRaisesRegex(ValueError, "columnas de predict"):
                        modelos.pronosticar_recursivo(m, self.datos)
                    predecir.assert_not_called()

    def test_rechaza_tabla_fit_incompleta_reordenada_o_grupo_distinto(self):
        preparado = modelos.construir_tabla_serie(self.datos, 1, "A", CORTE,
            columnas_predictoras=COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        casos = [replace(preparado, tabla=preparado.tabla.drop(columns="media_14")),
                 replace(preparado, tabla=preparado.tabla.iloc[:, ::-1]),
                 replace(preparado, columnas_predictoras=COLUMNAS_PREDICTORAS)]
        for tecnica in modelos.TECNICAS:
            for invalido in casos:
                with patch.object(modelos, "construir_tabla_serie", return_value=invalido), \
                        patch.object(modelos, "crear_estimador") as crear:
                    with self.assertRaisesRegex(ValueError, "columnas preparadas"):
                        modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                               configuracion_solicitada=self.solicitudes[tecnica])
                    crear.assert_not_called()

    def test_comprueba_parametros_antes_y_despues_de_fit_incluso_defaults(self):
        for tecnica, clase in (("random_forest", RandomForestRegressor),
                               ("hist_gradient_boosting", HistGradientBoostingRegressor)):
            for solicitud in (None, self.solicitudes[tecnica]):
                ajustar = clase.fit
                def fit(estimador, x, y):
                    resultado = ajustar(estimador, x, y)
                    estimador.set_params(random_state=123)
                    return resultado
                with patch.object(clase, "fit", new=fit), self.assertRaisesRegex(ValueError, "parámetros reales"):
                    modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                           configuracion_solicitada=solicitud)
                equivocado = modelos.crear_estimador(tecnica)
                equivocado.set_params(random_state=123)
                with patch.object(modelos, "crear_estimador", return_value=equivocado), \
                        patch.object(clase, "fit") as ajustar_mock:
                    with self.assertRaisesRegex(ValueError, "parámetros reales"):
                        modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                               configuracion_solicitada=solicitud)
                    ajustar_mock.assert_not_called()

    def test_rechaza_esquema_real_incoherente_inmediatamente_despues_de_fit(self):
        for tecnica, clase in (("random_forest", RandomForestRegressor),
                               ("hist_gradient_boosting", HistGradientBoostingRegressor)):
            for atributo in ("n_features_in_", "feature_names_in_"):
                ajustar = clase.fit
                def fit(estimador, x, y):
                    ajustar(estimador, x, y)
                    setattr(estimador, atributo, 6 if atributo == "n_features_in_" else list(reversed(x.columns)))
                    return estimador
                with self.subTest(tecnica=tecnica, atributo=atributo), \
                        patch.object(clase, "fit", new=fit), self.assertRaisesRegex(ValueError, atributo):
                    modelos.entrenar_modelo(self.datos, 1, "A", tecnica, CORTE,
                                           configuracion_solicitada=self.solicitudes[tecnica])

    def test_h2a_recibe_grupo_explicito_solo_en_ruta_alternativa(self):
        for solicitud in (None, {"tecnica": "random_forest"}, self.solicitudes["random_forest"]):
            with patch.object(modelos, "construir_tabla_serie", wraps=modelos.construir_tabla_serie) as preparar, \
                    patch.object(modelos, "construir_variables_futuras", wraps=construir_variables_futuras) as futuras:
                m = modelos.entrenar_modelo(self.datos, 1, "A", "random_forest", CORTE,
                                           configuracion_solicitada=solicitud)
                modelos.pronosticar_recursivo(m, self.datos)
            esperado = {"columnas_predictoras": COLUMNAS_PREDICTORAS_ALTERNATIVAS} if len(m.columnas_predictoras) == 7 else {}
            self.assertEqual(preparar.call_args.kwargs, esperado)
            self.assertEqual(len(futuras.call_args_list), 15)
            self.assertTrue(all(c.kwargs == esperado for c in futuras.call_args_list))

    def test_v2_no_finitos_rechazados_y_negativos_ajustados_reutilizados(self):
        for valor in (float("nan"), float("inf"), -float("inf")):
            m = replace(modelo_simulado(EstimadorSimulado(valor)),
                        columnas_predictoras=COLUMNAS_PREDICTORAS_ALTERNATIVAS)
            with self.subTest(valor=valor), self.assertRaisesRegex(ValueError, "no es finita"):
                modelos.pronosticar_recursivo(m, self.datos)
        estimador = EstimadorSimulado(-0.125)
        m = replace(modelo_simulado(estimador), columnas_predictoras=COLUMNAS_PREDICTORAS_ALTERNATIVAS)
        p = modelos.pronosticar_recursivo(m, self.datos)
        self.assertTrue(all(v.venta_estimada == 0.0 and v.ajustada_a_cero for v in p))
        self.assertEqual(estimador.entradas[1].iloc[0]["lag_1"], 0.0)
        self.assertEqual(estimador.entradas[14].iloc[0]["media_14"], 0.0)

    def test_diez_series_v2_estimadores_reales_independientes_sin_contaminacion(self):
        claves = leer_configuracion()
        datos = pd.concat([observaciones(t, f, i) for i, (t, f) in enumerate(claves, 1)], ignore_index=True)
        copia = datos.copy(deep=True)
        for tecnica in modelos.TECNICAS:
            estimadores = []
            for i, (tienda, familia) in enumerate(claves, 1):
                solicitud = deepcopy(self.solicitudes[tecnica])
                solicitud["parametros"].update({"n_estimators": 3} if tecnica == "random_forest" else {"max_iter": 3})
                m = modelos.entrenar_modelo(datos, tienda, familia, tecnica, CORTE,
                                           configuracion_solicitada=solicitud)
                estimadores.append(m.estimador)
                p = modelos.pronosticar_recursivo(m, datos)
                aislado = datos.loc[(datos["store_nbr"] == tienda) & (datos["family"] == familia)]
                self.assertEqual(p, modelos.pronosticar_recursivo(m, aislado))
                self.assertTrue(all((v.store_nbr, v.family, v.tecnica) == (tienda, familia, tecnica) for v in p))
                self.assertEqual(m.filas_validas, 46)
                self.assertAlmostEqual(m.estimador.feature_names_in_.size, 7)
            self.assertEqual(len({id(e) for e in estimadores}), 10)
        pd.testing.assert_frame_equal(datos, copia)


if __name__ == "__main__":
    unittest.main()
