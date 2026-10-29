"""Pruebas sintéticas de retardos por calendario y disponibilidad temporal."""

from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.linea_base_prediccion import cargar_historico_csv
from scripts.preparar_series_modelado import leer_configuracion
from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_TABLA, HistorialInsuficiente,
    construir_tabla_serie, construir_tablas, construir_variables_futuras,
)


CORTE = date(2017, 6, 16)


def observaciones(tienda=1, familia="A", escala=1.0):
    return pd.DataFrame({
        "date": [f"2017-06-{d:02d}" for d in range(1, 17)],
        "store_nbr": tienda, "family": familia,
        "sales": [d * escala for d in range(1, 17)], "onpromotion": 0,
    })


class VariablesPredictorasTests(unittest.TestCase):
    def construir(self, datos=None):
        return construir_tabla_serie(observaciones() if datos is None else datos, 1, "A", CORTE)

    def test_retardos_exactos_y_calendario(self):
        fila = self.construir().tabla.iloc[0]
        self.assertEqual(fila["fecha_objetivo"], date(2017, 6, 15))
        self.assertEqual([fila[c] for c in ("lag_1", "lag_7", "lag_14")], [14, 8, 1])
        self.assertEqual(fila["dia_semana"], 3)
        self.assertEqual(fila["mes"], 6)

    def test_media_siete_dias_anteriores(self):
        self.assertAlmostEqual(self.construir().tabla.iloc[0]["media_7"], 11.0)

    def test_venta_objetivo_no_es_predictor(self):
        datos = observaciones()
        datos.loc[14, "sales"] = 1e12
        original = self.construir().tabla.iloc[[0]]
        modificado = self.construir(datos).tabla.iloc[[0]]
        pd.testing.assert_frame_equal(original.loc[:, list(COLUMNAS_PREDICTORAS)],
                                      modificado.loc[:, list(COLUMNAS_PREDICTORAS)])
        self.assertEqual(modificado.iloc[0]["sales"], 1e12)

    def test_ventas_posteriores_al_corte_no_cambian_entrenamiento(self):
        datos = observaciones()
        futuro = datos.iloc[[-1]].copy()
        futuro["date"] = "2017-06-17"
        futuro["sales"] = 1e12
        entrada = pd.concat([datos, futuro], ignore_index=True)
        copia = entrada.copy(deep=True)
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(entrada).tabla)
        pd.testing.assert_frame_equal(entrada, copia)

    def test_dia_ausente_no_se_desplaza_a_otro_registro(self):
        resultado = self.construir(observaciones().drop(index=13))
        descartada = next(d for d in resultado.descartes if d.fecha_objetivo == date(2017, 6, 15))
        self.assertEqual(descartada.faltantes["lag_1"], (date(2017, 6, 14),))
        self.assertEqual(descartada.faltantes["media_7"], (date(2017, 6, 14),))
        self.assertFalse(any(d.fecha_objetivo == date(2017, 6, 14) for d in resultado.descartes))
        self.assertEqual(resultado.observaciones_historicas, 15)

    def test_cero_real_se_conserva(self):
        datos = observaciones()
        datos.loc[13, "sales"] = 0
        datos.loc[14, "sales"] = 0
        resultado = self.construir(datos)
        self.assertEqual(len(resultado.tabla), 2)
        self.assertEqual(resultado.tabla.iloc[0]["lag_1"], 0)
        self.assertEqual(resultado.tabla.iloc[0]["sales"], 0)

    def test_exclusion_y_contabilidad_de_motivos(self):
        resultado = self.construir()
        self.assertEqual(resultado.observaciones_historicas, 16)
        self.assertEqual(len(resultado.tabla), 2)
        self.assertEqual(len(resultado.descartes), 14)
        self.assertEqual(resultado.descartes_por_variable,
                         {"lag_1": 1, "lag_7": 7, "lag_14": 14,
                          "media_7": 7, "dia_semana": 0, "mes": 0})
        self.assertEqual(resultado.observaciones_historicas,
                         len(resultado.tabla) + len(resultado.descartes))

    def test_conservacion_de_fracciones(self):
        fila = self.construir(observaciones(escala=0.125)).tabla.iloc[0]
        self.assertEqual(fila["lag_1"], 1.75)
        self.assertEqual(fila["lag_14"], 0.125)
        self.assertEqual(fila["sales"], 1.875)
        self.assertAlmostEqual(fila["media_7"], 1.375)

    def test_separacion_por_tienda_y_familia(self):
        claves = ((1, "A"), (1, "B"), (2, "A"))
        datos = pd.concat([observaciones(t, f, escala=i)
                           for i, (t, f) in enumerate(claves, 1)], ignore_index=True)
        resultados = construir_tablas(datos, CORTE, claves)
        for i, resultado in enumerate(resultados, 1):
            fila = resultado.tabla.iloc[0]
            self.assertEqual(fila["lag_1"], 14 * i)
            self.assertEqual(fila["lag_7"], 8 * i)
            self.assertEqual(fila["lag_14"], i)
            self.assertAlmostEqual(fila["media_7"], 11 * i)

    def test_orden_cronologico_y_columnas_estables(self):
        tabla = self.construir(observaciones().iloc[::-1]).tabla
        self.assertEqual(tuple(tabla.columns), COLUMNAS_TABLA)
        self.assertEqual(tabla["fecha_objetivo"].tolist(), [date(2017, 6, 15), CORTE])
        pd.testing.assert_frame_equal(tabla, self.construir().tabla)

    def test_coincidencia_entrenamiento_inferencia(self):
        tabla = self.construir().tabla
        for i, fila in tabla.iterrows():
            objetivo = fila["fecha_objetivo"]
            futuro = construir_variables_futuras(observaciones(), 1, "A",
                                                 objetivo - timedelta(days=1), objetivo)
            pd.testing.assert_frame_equal(tabla.loc[[i], list(COLUMNAS_PREDICTORAS)].reset_index(drop=True),
                                          futuro)

    def test_predicciones_previas_explicitas_sin_ventas_reales_futuras(self):
        datos = observaciones()
        futuro = datos.iloc[[-1, -1, -1]].copy()
        futuro["date"] = ["2017-06-17", "2017-06-18", "2017-06-19"]
        futuro["sales"] = 1e12
        entrada = pd.concat([datos, futuro], ignore_index=True)
        previas = {date(2017, 6, 17): 100.25, date(2017, 6, 18): 200.5}
        x = construir_variables_futuras(entrada, 1, "A", CORTE, date(2017, 6, 19), previas)
        self.assertEqual(x.iloc[0]["lag_1"], 200.5)
        self.assertEqual(x.iloc[0]["lag_7"], 12)
        self.assertEqual(x.iloc[0]["lag_14"], 5)
        self.assertAlmostEqual(x.iloc[0]["media_7"], (12 + 13 + 14 + 15 + 16 + 100.25 + 200.5) / 7)
        pd.testing.assert_frame_equal(x, construir_variables_futuras(
            datos, 1, "A", CORTE, date(2017, 6, 19), previas))
        with self.assertRaises(HistorialInsuficiente):
            construir_variables_futuras(entrada, 1, "A", CORTE, date(2017, 6, 19))

    def test_error_inferencia_sin_historial_suficiente(self):
        with self.assertRaisesRegex(HistorialInsuficiente, "lag_14") as contexto:
            construir_variables_futuras(observaciones().iloc[-7:], 1, "A", CORTE, date(2017, 6, 17))
        self.assertEqual(contexto.exception.faltantes["lag_14"], (date(2017, 6, 3),))
        with self.assertRaises(HistorialInsuficiente):
            construir_variables_futuras(observaciones().iloc[:0], 1, "A", CORTE, date(2017, 6, 17))

    def test_lectura_y_procesamiento_diez_series_configuradas(self):
        claves = leer_configuracion()
        datos = pd.concat([observaciones(t, f) for t, f in claves], ignore_index=True)
        datos = datos.sort_values(["date", "store_nbr", "family"])
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            datos.to_csv(ruta, index=False)
            with ruta.open("a", encoding="utf-8") as archivo:
                t, f = claves[0]
                archivo.write(f"2017-06-17,{t},{f},NO_LEER,NO_LEER\n")
            historico = cargar_historico_csv(ruta, claves, CORTE, tamano_bloque=23)
            resultados = construir_tablas(historico, CORTE)
        self.assertEqual(len(resultados), 10)
        self.assertEqual(tuple((r.store_nbr, r.family) for r in resultados), claves)
        self.assertTrue(all(r.observaciones_historicas == 16 and len(r.tabla) == 2
                            and len(r.descartes) == 14 for r in resultados))

    def test_predicciones_previas_invalidas(self):
        for previas in ({CORTE: 1}, {date(2017, 6, 19): 1},
                        {date(2017, 6, 17): -1}, {date(2017, 6, 17): float("nan")},
                        {date(2017, 6, 17): float("inf")}):
            with self.subTest(previas=previas), self.assertRaises(ValueError):
                construir_variables_futuras(observaciones(), 1, "A", CORTE,
                                            date(2017, 6, 19), previas)

    def test_onpromotion_no_es_predictor(self):
        datos = observaciones()
        datos["onpromotion"] = 9999
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(datos).tabla)
        self.assertNotIn("onpromotion", COLUMNAS_TABLA)

    def test_historial_corto_descarta_sin_fabricar_filas(self):
        resultado = self.construir(observaciones().iloc[:3])
        self.assertTrue(resultado.tabla.empty)
        self.assertEqual(tuple(resultado.tabla.columns), COLUMNAS_TABLA)
        self.assertEqual(len(resultado.descartes), 3)


if __name__ == "__main__":
    unittest.main()
