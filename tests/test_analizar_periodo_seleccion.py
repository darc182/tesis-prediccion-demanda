"""Pruebas pequeñas de corte temporal y calendario del periodo inicial."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.analizar_periodo_seleccion import (
    FIN_SELECCION,
    HORIZONTE_DIAS,
    VENTANAS_POSTERIORES,
    calcular_caracteristicas,
    contar_prefijo_fechas,
    dias_sin_registro,
    leer_bloques_hasta_corte,
    validar_ventanas,
)


class AnalizarPeriodoSeleccionTests(unittest.TestCase):
    def test_ventas_y_promociones_posteriores_no_entran_en_calculos(self) -> None:
        datos = pd.DataFrame({
            "date": ["2017-06-15", "2017-06-16", "2017-06-17"],
            "store_nbr": ["1", "1", "1"],
            "family": ["A", "A", "A"],
            "sales": [0.0, 2.5, 1_000_000.0],
            "onpromotion": [0, 1, 999],
        })
        resultado = calcular_caracteristicas([datos])
        serie = resultado.series.iloc[0]
        self.assertEqual(resultado.filas_seleccion, 2)
        self.assertEqual(resultado.filas_fuera_intervalo, 1)
        self.assertEqual(serie["observaciones"], 2)
        self.assertAlmostEqual(serie["volumen_total"], 2.5)
        self.assertAlmostEqual(serie["media"], 1.25)
        self.assertAlmostEqual(serie["porcentaje_cero"], 50.0)
        self.assertAlmostEqual(serie["porcentaje_fraccionarias"], 50.0)
        self.assertAlmostEqual(serie["porcentaje_promocion"], 50.0)

    def test_lectura_csv_no_materializa_valores_posteriores(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            ruta.write_text(
                "date,store_nbr,family,sales,onpromotion\n"
                "2017-06-16,1,A,2.5,1\n"
                "2017-06-17,1,A,NO_LEER,NO_LEER\n",
                encoding="utf-8",
            )
            prefijo, total = contar_prefijo_fechas(ruta, tamano_bloque=1)
            bloques = list(leer_bloques_hasta_corte(ruta, prefijo, tamano_bloque=1))
            self.assertEqual((prefijo, total), (1, 2))
            self.assertEqual(len(bloques), 1)
            self.assertEqual(bloques[0]["sales"].tolist(), ["2.5"])

    def test_ventanas_de_quince_dias_sin_solapamiento(self) -> None:
        validar_ventanas()
        self.assertEqual(len(VENTANAS_POSTERIORES), 4)
        self.assertTrue(all(v.dias == HORIZONTE_DIAS for v in VENTANAS_POSTERIORES))
        anterior = FIN_SELECCION
        for ventana in VENTANAS_POSTERIORES:
            self.assertGreater(ventana.inicio, anterior)
            anterior = ventana.fin

    def test_dia_ausente_no_es_venta_cero(self) -> None:
        datos = pd.DataFrame({
            "date": ["2017-06-14", "2017-06-16"],
            "store_nbr": ["1", "1"],
            "family": ["A", "A"],
            "sales": [0.0, 5.0],
            "onpromotion": [0, 0],
        })
        inicio, fin = date(2017, 6, 14), date(2017, 6, 16)
        resultado = calcular_caracteristicas([datos], inicio=inicio, fin=fin)
        serie = resultado.series.iloc[0]
        self.assertEqual(
            dias_sin_registro(resultado.fechas_observadas, inicio, fin),
            [date(2017, 6, 15)],
        )
        self.assertEqual(serie["observaciones"], 2)
        self.assertEqual(serie["dias_distintos"], 2)
        self.assertEqual(serie["ventas_cero"], 1)
        self.assertEqual(serie["porcentaje_cero"], 50.0)


if __name__ == "__main__":
    unittest.main()
