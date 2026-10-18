"""Pruebas sintéticas del servicio de validación CSV, sin usar SQLite."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from backend.app.services.validacion_csv import validar_csv


CABECERAS = ["date", "store_nbr", "family", "sales", "onpromotion"]
FILA = ["2017-06-16", "1", "GROCERY I", "2.5", "0"]


class ValidacionCSVTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name) / "ventas.csv"

    def escribir(self, filas: list[list[str]],
                cabeceras: list[str] = CABECERAS) -> None:
        with self.ruta.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(cabeceras)
            escritor.writerows(filas)

    def test_csv_valido_con_decimales_ceros_y_columna_adicional(self) -> None:
        self.escribir([
            ["2013-01-01", "1", "GROCERY I", "0", "0", "a"],
            ["2017-06-16", "1", "GROCERY I", "2.75", "3", "b"],
        ], CABECERAS + ["id"])
        antes = self.ruta.read_bytes()
        fecha_modificacion = self.ruta.stat().st_mtime_ns
        resultado = validar_csv(self.ruta, tamano_bloque=1)
        self.assertTrue(resultado.valido)
        self.assertEqual(resultado.registros_examinados, 2)
        self.assertEqual(resultado.registros_invalidos, 0)
        self.assertEqual(resultado.errores, ())
        self.assertEqual((resultado.fecha_minima, resultado.fecha_maxima),
                         ("2013-01-01", "2017-06-16"))
        self.assertEqual(resultado.sha256, hashlib.sha256(antes).hexdigest())
        self.assertEqual(self.ruta.read_bytes(), antes)
        self.assertEqual(self.ruta.stat().st_mtime_ns, fecha_modificacion)

    def test_columna_obligatoria_faltante(self) -> None:
        self.escribir([["2017-06-16", "1", "GROCERY I", "2.5"]],
                     CABECERAS[:-1])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_examinados, 0)
        self.assertIn("onpromotion", resultado.errores[0])

    def test_archivo_vacio_y_archivo_inexistente(self) -> None:
        self.ruta.write_bytes(b"")
        vacio = validar_csv(self.ruta)
        self.assertFalse(vacio.valido)
        self.assertEqual(vacio.registros_examinados, 0)
        self.assertEqual(vacio.sha256, hashlib.sha256(b"").hexdigest())
        self.assertIn("vacío", vacio.errores[0])

        ausente = validar_csv(self.ruta.with_name("no_existe.csv"))
        self.assertFalse(ausente.valido)
        self.assertIsNone(ausente.sha256)
        self.assertTrue(ausente.errores)

    def test_fecha_invalida_y_rango_determinable(self) -> None:
        self.escribir([
            ["2017-02-30", "1", "A", "1", "0"],
            ["2017/06/01", "1", "B", "1", "0"],
            ["2017-06-16", "1", "C", "1", "0"],
        ])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 2)
        self.assertEqual((resultado.fecha_minima, resultado.fecha_maxima),
                         ("2017-06-16", "2017-06-16"))

    def test_familia_vacia(self) -> None:
        self.escribir([["2017-06-16", "1", "   ", "1", "0"]])
        resultado = validar_csv(self.ruta)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertIn("family", resultado.errores[0])

    def test_tienda_debe_ser_entero_positivo(self) -> None:
        self.escribir([
            ["2017-06-16", "0", "A", "1", "0"],
            ["2017-06-17", "-1", "A", "1", "0"],
            ["2017-06-18", "1.5", "A", "1", "0"],
        ])
        resultado = validar_csv(self.ruta)
        self.assertEqual(resultado.registros_invalidos, 3)
        self.assertTrue(all("store_nbr" in error for error in resultado.errores))

    def test_ventas_negativas_o_no_finitas(self) -> None:
        self.escribir([
            ["2017-06-16", "1", "A", "-0.1", "0"],
            ["2017-06-17", "1", "A", "NaN", "0"],
            ["2017-06-18", "1", "A", "Infinity", "0"],
        ])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 3)
        self.assertTrue(all("sales" in error for error in resultado.errores))

    def test_venta_decimal_finita_que_desborda_sqlite_real(self) -> None:
        self.escribir([["2017-06-16", "1", "A", "1e309", "0"]])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertIn("REAL finito", resultado.errores[0])

    def test_venta_positiva_que_se_convierte_en_cero(self) -> None:
        self.escribir([["2017-06-16", "1", "A", "1e-400", "0"]])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertIn("convertirse en cero", resultado.errores[0])

    def test_tienda_fuera_del_rango_integer_de_sqlite(self) -> None:
        self.escribir([["2017-06-16", "9223372036854775808", "A", "1", "0"]])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertIn("store_nbr", resultado.errores[0])

    def test_promocion_fuera_del_rango_integer_de_sqlite(self) -> None:
        self.escribir([["2017-06-16", "1", "A", "1", "9223372036854775808"]])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertIn("onpromotion", resultado.errores[0])

    def test_limite_integer_y_ventas_normales_siguen_validos(self) -> None:
        self.escribir([
            ["2017-06-16", "9223372036854775807", "A", "0", "9223372036854775807"],
            ["2017-06-17", "1", "B", "2.75", "3"],
        ])
        resultado = validar_csv(self.ruta, tamano_bloque=1)
        self.assertTrue(resultado.valido)
        self.assertEqual(resultado.registros_examinados, 2)
        self.assertEqual(resultado.registros_invalidos, 0)

    def test_promociones_negativas_o_fraccionarias(self) -> None:
        self.escribir([
            ["2017-06-16", "1", "A", "1", "-1"],
            ["2017-06-17", "1", "A", "1", "1.5"],
            ["2017-06-18", "1", "A", "1", "1.0"],
        ])
        resultado = validar_csv(self.ruta)
        self.assertEqual(resultado.registros_invalidos, 3)
        self.assertTrue(all("onpromotion" in error for error in resultado.errores))

    def test_claves_duplicadas_entre_bloques(self) -> None:
        self.escribir([
            FILA,
            ["2017-06-17", "1", "GROCERY I", "3", "0"],
            ["2017-06-16", "01", "GROCERY I", "4", "0"],
        ])
        resultado = validar_csv(self.ruta, tamano_bloque=1)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_examinados, 3)
        self.assertEqual(resultado.registros_invalidos, 1)
        self.assertTrue(any("duplicada" in error for error in resultado.errores))

    def test_fila_invalida_y_duplicada_se_cuenta_una_vez(self) -> None:
        self.escribir([
            ["2017-06-16", "1", "A", "-1", "0"],
            ["2017-06-16", "1", "A", "2", "0"],
        ])
        resultado = validar_csv(self.ruta, tamano_bloque=1)
        self.assertEqual(resultado.registros_invalidos, 2)

    def test_fechas_sin_registro_y_posteriores_a_2017_permitidas(self) -> None:
        self.escribir([
            ["2017-06-16", "1", "A", "1", "0"],
            ["2020-01-01", "1", "A", "0", "0"],
        ])
        resultado = validar_csv(self.ruta)
        self.assertTrue(resultado.valido)
        self.assertEqual(resultado.fecha_maxima, "2020-01-01")

    def test_limita_mensajes_sin_perder_conteo(self) -> None:
        self.escribir([
            [f"2017-06-{dia:02d}", "1", "A", "-1", "0"]
            for dia in range(1, 8)
        ])
        resultado = validar_csv(self.ruta, max_errores=2)
        self.assertEqual(resultado.registros_invalidos, 7)
        self.assertEqual(len(resultado.errores), 2)
        self.assertIn("omitieron", resultado.errores[-1])

    def test_cabeceras_sin_filas_no_es_csv_de_ventas_valido(self) -> None:
        self.escribir([])
        resultado = validar_csv(self.ruta)
        self.assertFalse(resultado.valido)
        self.assertEqual(resultado.registros_examinados, 0)
        self.assertIsNone(resultado.fecha_minima)


if __name__ == "__main__":
    unittest.main()
