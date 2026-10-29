"""Pruebas temporales con regresores reales pequeños y estimadores simulados."""

from datetime import date, timedelta
from math import isfinite
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from scripts import modelos_prediccion as modelos
from scripts.preparar_series_modelado import leer_configuracion
from scripts.variables_predictoras import COLUMNAS_PREDICTORAS, construir_variables_futuras


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


if __name__ == "__main__":
    unittest.main()
