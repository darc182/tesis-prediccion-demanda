"""Pruebas sintéticas de preparación y auditoría temporal."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from scripts.analizar_periodo_seleccion import (
    FIN_SELECCION, HORIZONTE_DIAS, INICIO_SELECCION,
    VENTANAS_POSTERIORES,
)
from scripts.preparar_series_modelado import (
    cargar_series_csv, historico_hasta, informar_cobertura,
    leer_configuracion, preparar_observaciones,
)


def filas(fechas, ventas, tienda=1, familia="A", promociones=None):
    return pd.DataFrame({
        "date": fechas,
        "store_nbr": [tienda] * len(fechas),
        "family": [familia] * len(fechas),
        "sales": ventas,
        "onpromotion": promociones if promociones is not None else [0] * len(fechas),
    })


class PreparacionModeladoTests(unittest.TestCase):
    def test_configuracion_lee_solo_las_combinaciones_registradas(self):
        claves = leer_configuracion()
        config = json.loads((Path(__file__).resolve().parents[1] / "config" /
                             "series_seleccionadas.json").read_text(encoding="utf-8"))
        self.assertEqual(claves, tuple((s["store_nbr"], s["family"])
                                       for s in config["series"]))
        self.assertEqual(len(claves), 10)

    def test_configuracion_incoherente(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "config" /
                             "series_seleccionadas.json").read_text(encoding="utf-8"))
        config["horizonte_prediccion_dias"] = 14
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "config.json"
            ruta.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "horizonte"):
                leer_configuracion(ruta)

    def test_orden_y_conservacion_de_ventas_y_promociones(self):
        datos = preparar_observaciones(filas(
            ["2017-06-16", "2017-06-14"], [2.5, 0], promociones=[3, 0]
        ))
        self.assertEqual(datos["date"].dt.date.tolist(),
                         [date(2017, 6, 14), date(2017, 6, 16)])
        self.assertEqual(datos["sales"].tolist(), [0.0, 2.5])
        self.assertEqual(datos["onpromotion"].tolist(), [0, 3])
        self.assertTrue(pd.api.types.is_float_dtype(datos["sales"]))

        dos_series = pd.concat([
            filas(["2017-06-16"], [1], tienda=1),
            filas(["2017-06-14"], [2], tienda=2),
        ], ignore_index=True)
        ordenadas = preparar_observaciones(dos_series)
        self.assertEqual(ordenadas["store_nbr"].tolist(), [2, 1])

    def test_ventanas_de_quince_dias_consecutivos_sin_solapamiento(self):
        anterior = FIN_SELECCION
        self.assertEqual(len(VENTANAS_POSTERIORES), 4)
        for ventana in VENTANAS_POSTERIORES:
            self.assertEqual(ventana.dias, HORIZONTE_DIAS)
            self.assertEqual(ventana.inicio, anterior + timedelta(days=1))
            anterior = ventana.fin
        self.assertEqual(anterior, date(2017, 8, 15))

    def test_corte_estricto_y_ausencia_de_fuga_futura(self):
        datos = preparar_observaciones(filas(
            ["2017-06-15", "2017-06-16", "2017-06-17"],
            [0, 2.5, 999999], promociones=[0, 2, 999]
        ))
        historico = historico_hasta(datos, FIN_SELECCION)
        self.assertEqual(historico["sales"].tolist(), [0, 2.5])
        self.assertEqual(historico["onpromotion"].tolist(), [0, 2])
        self.assertTrue(historico["date"].le(pd.Timestamp(FIN_SELECCION)).all())
        informe = informar_cobertura(datos, ((1, "A"),),
                                    inicio=date(2017, 6, 15), fin=FIN_SELECCION)[0]
        self.assertEqual(informe.registros_entrenamiento, 2)
        self.assertEqual(informe.ventas_cero, 1)
        self.assertTrue(informe.ventas_fraccionarias)
        self.assertEqual(informe.ventanas[0].disponibles, (date(2017, 6, 17),))
        self.assertFalse(hasattr(informe.ventanas[0], "sales"))

    def test_ausencia_de_fecha_es_distinta_de_venta_cero(self):
        datos = preparar_observaciones(filas(
            ["2017-06-14", "2017-06-16"], [0, 2.5]
        ))
        informe = informar_cobertura(datos, ((1, "A"),),
                                    inicio=date(2017, 6, 14), fin=FIN_SELECCION)[0]
        self.assertEqual(informe.dias_esperados, 3)
        self.assertEqual(informe.registros_entrenamiento, 2)
        self.assertEqual(informe.dias_sin_registro, 1)
        self.assertEqual(informe.ventas_cero, 1)
        self.assertFalse(informe.entrenamiento_completo)

    def test_datos_invalidos_y_duplicados(self):
        base = filas(["2017-06-16"], [2.5], promociones=[3])
        for columna, invalido in (
            ("date", "2017-02-30"), ("store_nbr", 0),
            ("family", " "), ("sales", "NaN"), ("sales", -1),
            ("onpromotion", -1), ("onpromotion", 1.5),
        ):
            with self.subTest(columna=columna, valor=invalido):
                datos = base.copy()
                datos[columna] = pd.Series([invalido], dtype="object")
                with self.assertRaises(ValueError):
                    preparar_observaciones(datos)
        with self.assertRaisesRegex(ValueError, "duplicada"):
            preparar_observaciones(pd.concat([base, base], ignore_index=True))

    def test_series_no_encontradas_y_csv_por_bloques(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            ruta.write_text(
                "date,store_nbr,family,sales,onpromotion\n"
                "2017-06-15,2,B,4,0\n"
                "2017-06-16,1,A,0,2\n"
                "2017-06-17,1,A,2.5,3\n", encoding="utf-8",
            )
            datos = cargar_series_csv(ruta, ((1, "A"),), tamano_bloque=1)
            self.assertEqual(datos["sales"].tolist(), [0, 2.5])
            self.assertEqual(len(datos), 2)
            with self.assertRaisesRegex(ValueError, "no encontradas"):
                cargar_series_csv(ruta, ((3, "C"),), tamano_bloque=1)

    def test_falta_de_columnas_y_archivo(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            with self.assertRaises(FileNotFoundError):
                cargar_series_csv(ruta, ((1, "A"),))
            ruta.write_text("date,store_nbr,family,sales\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columnas"):
                cargar_series_csv(ruta, ((1, "A"),))


if __name__ == "__main__":
    unittest.main()
