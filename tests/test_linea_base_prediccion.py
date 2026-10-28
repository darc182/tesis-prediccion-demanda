"""Contratos de la línea base semanal con observaciones sintéticas."""

from datetime import date, timedelta
from math import isfinite
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.linea_base_prediccion import (
    TECNICA, cargar_historico_csv, fechas_consecutivas,
    pronosticar_serie, pronosticar_series,
)
from scripts.preparar_series_modelado import leer_configuracion


CORTE = date(2017, 6, 16)


def datos_serie(fechas=None, ventas=None, tienda=1, familia="A"):
    if fechas is None:
        fechas = [CORTE - timedelta(days=d) for d in range(6, -1, -1)]
    if ventas is None:
        ventas = [10, 20, 30, 40, 50, 60, 70]
    return pd.DataFrame({
        "date": [d.isoformat() for d in fechas], "store_nbr": tienda,
        "family": familia, "sales": ventas, "onpromotion": 0,
    })


class LineaBasePrediccionTests(unittest.TestCase):
    def pronostico(self, datos=None):
        return pronosticar_serie(datos if datos is not None else datos_serie(), 1, "A", CORTE)

    def test_prediccion_semanal_conocida(self):
        self.assertEqual([p.venta_estimada for p in self.pronostico()[:7]],
                         [10, 20, 30, 40, 50, 60, 70])

    def test_horizonte_exactamente_quince_dias(self):
        self.assertEqual(len(self.pronostico()), 15)

    def test_recursion_despues_de_siete_dias(self):
        predicciones = self.pronostico()
        self.assertEqual([p.venta_estimada for p in predicciones],
                         [10, 20, 30, 40, 50, 60, 70] * 2 + [10])
        self.assertTrue(all(p.regla == "recursiva" for p in predicciones[7:]))
        self.assertEqual(predicciones[14].fecha_origen_real, date(2017, 6, 10))

    def test_solo_utiliza_origenes_hasta_corte(self):
        self.assertTrue(all(p.fecha_origen_real <= CORTE for p in self.pronostico()))

    def test_ventas_futuras_elevadas_no_cambian_resultados(self):
        futuro = datos_serie([CORTE + timedelta(days=d) for d in range(1, 16)], [1e12] * 15)
        entrada = pd.concat([datos_serie(), futuro], ignore_index=True)
        original = entrada.copy(deep=True)
        self.assertEqual(self.pronostico(entrada), self.pronostico())
        pd.testing.assert_frame_equal(entrada, original)

    def test_conserva_cero_real(self):
        datos = datos_serie(ventas=[0, 20, 30, 40, 50, 60, 70])
        p = self.pronostico(datos)
        self.assertEqual([p[i].venta_estimada for i in (0, 7, 14)], [0, 0, 0])
        self.assertFalse(p[0].uso_respaldo)

    def test_conserva_fraccion_sin_redondear(self):
        valor = 2.123456789
        p = self.pronostico(datos_serie(ventas=[valor, 20, 30, 40, 50, 60, 70]))
        self.assertEqual([p[i].venta_estimada for i in (0, 7, 14)], [valor] * 3)

    def test_respaldo_mismo_dia_semanal_mas_reciente(self):
        datos = datos_serie(
            [date(2017, 5, 27), date(2017, 6, 3), CORTE], [99, 2.75, 4])
        p = self.pronostico(datos)
        self.assertEqual(p[0].regla, "respaldo_mismo_dia")
        self.assertTrue(p[0].uso_respaldo)
        self.assertEqual(p[0].fecha_referencia, date(2017, 6, 10))
        self.assertEqual(p[0].fecha_origen_real, date(2017, 6, 3))
        self.assertEqual(p[0].venta_estimada, 2.75)
        self.assertEqual(p[7].venta_estimada, 2.75)
        self.assertEqual(p[14].venta_estimada, 2.75)

    def test_respaldo_ultimo_valor_real(self):
        p = self.pronostico(datos_serie([CORTE], [3.25]))
        self.assertEqual(p[0].regla, "respaldo_ultimo_valor")
        self.assertEqual(p[0].fecha_origen_real, CORTE)
        self.assertEqual(p[0].venta_estimada, 3.25)

    def test_error_sin_historial_utilizable(self):
        with self.assertRaisesRegex(ValueError, "Sin historial utilizable"):
            self.pronostico(datos_serie([CORTE + timedelta(days=1)], [999]))

    def test_error_entrada_vacia(self):
        with self.assertRaisesRegex(ValueError, "Serie no encontrada"):
            self.pronostico(datos_serie().iloc[:0])

    def test_admite_otro_corte_explicito(self):
        corte = CORTE - timedelta(days=1)
        p = pronosticar_serie(datos_serie(), 1, "A", corte)
        self.assertEqual(p[0].fecha_objetivo, CORTE)
        self.assertEqual(p[0].venta_estimada, 60)
        self.assertTrue(p[0].uso_respaldo)
        self.assertTrue(all(f.fecha_origen_real <= corte for f in p))

    def test_error_serie_no_encontrada(self):
        with self.assertRaisesRegex(ValueError, "Serie no encontrada"):
            pronosticar_serie(datos_serie(), 2, "B", CORTE)

    def test_determinismo_incluso_con_entrada_desordenada(self):
        datos = datos_serie()
        self.assertEqual(self.pronostico(datos), self.pronostico(datos))
        self.assertEqual(self.pronostico(datos), self.pronostico(datos.iloc[::-1]))

    def test_fechas_pasos_valores_y_metadatos(self):
        p = self.pronostico()
        self.assertTrue(fechas_consecutivas(p))
        self.assertEqual([f.paso for f in p], list(range(1, 16)))
        self.assertEqual([f.fecha_objetivo for f in p],
                         [CORTE + timedelta(days=d) for d in range(1, 16)])
        self.assertTrue(all(isfinite(f.venta_estimada) and f.venta_estimada >= 0 for f in p))
        self.assertTrue(all((f.store_nbr, f.family, f.tecnica, f.fecha_corte)
                            == (1, "A", TECNICA, CORTE) for f in p))

    def test_diez_series_de_configuracion_ciento_cincuenta_predicciones(self):
        claves = leer_configuracion()
        datos = pd.concat([datos_serie(tienda=t, familia=f) for t, f in claves], ignore_index=True)
        predicciones = pronosticar_series(datos, CORTE)
        self.assertEqual(len(predicciones), 150)
        self.assertEqual(set((p.store_nbr, p.family) for p in predicciones), set(claves))
        for t, f in claves:
            grupo = tuple(p for p in predicciones if (p.store_nbr, p.family) == (t, f))
            self.assertTrue(fechas_consecutivas(grupo))

    def test_lector_no_materializa_ventas_futuras(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            ruta.write_text(
                "date,store_nbr,family,sales,onpromotion\n"
                "2017-06-16,1,A,2.75,3\n"
                "2017-06-17,1,A,NO_LEER,NO_LEER\n", encoding="utf-8")
            datos = cargar_historico_csv(ruta, ((1, "A"),), CORTE, tamano_bloque=1)
            self.assertEqual(datos["sales"].tolist(), [2.75])
            self.assertLessEqual(datos["date"].max().date(), CORTE)
            self.assertEqual(len(self.pronostico(datos)), 15)

    def test_corte_explicito_y_errores_de_datos_historicos(self):
        with self.assertRaisesRegex(ValueError, "fecha_corte"):
            pronosticar_serie(datos_serie(), 1, "A", "2017-06-16")
        for valor in (-1, float("inf"), float("nan")):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                self.pronostico(datos_serie([CORTE], [valor]))
        with self.assertRaisesRegex(ValueError, "duplicada"):
            self.pronostico(datos_serie([CORTE, CORTE], [1, 2]))


if __name__ == "__main__":
    unittest.main()
