"""Pruebas de la carga transaccional con CSV y bases SQLite temporales."""

import csv
import hashlib
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.app.services.carga_datos import cargar_csv
from backend.app.services.validacion_csv import validar_csv
from backend.app.storage.db import conectar, inicializar_base


RAIZ = Path(__file__).resolve().parents[1]
CABECERAS = ["date", "store_nbr", "family", "sales", "onpromotion"]
PREPARADOS = (17, "PREPARED FOODS")
PANADERIA = (8, "BREAD/BAKERY")


class CargaDatosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        directorio = Path(self.temporal.name)
        self.csv = directorio / "ventas.csv"
        self.base = directorio / "prueba.sqlite3"
        self.config = directorio / "series.json"
        self.config.write_text(
            (RAIZ / "config" / "series_seleccionadas.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        inicializar_base(self.base)

    def escribir(self, filas: list[list[str]]) -> None:
        with self.csv.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(CABECERAS)
            escritor.writerows(filas)

    def cargar(self, lote: int = 2):
        return cargar_csv(self.csv, self.base, self.config, tamano_lote=lote)

    def estado(self) -> tuple[list[tuple], list[tuple], list[tuple]]:
        with closing(conectar(self.base)) as conexion:
            conjuntos = conexion.execute(
                "SELECT id, hash, activo FROM conjunto_datos ORDER BY id"
            ).fetchall()
            series = conexion.execute(
                "SELECT id, conjunto_id, store_nbr, family, criterio_seleccion "
                "FROM serie_temporal ORDER BY id"
            ).fetchall()
            ventas = conexion.execute(
                "SELECT id, serie_id, fecha, sales, onpromotion "
                "FROM venta_diaria ORDER BY id"
            ).fetchall()
            return conjuntos, series, ventas

    def test_primera_carga_con_ceros_fracciones_y_solo_series_habilitadas(self) -> None:
        self.escribir([
            ["2017-06-14", "17", "PREPARED FOODS", "0", "0"],
            ["2017-06-16", "17", "PREPARED FOODS", "2.75", "1"],
            ["2017-06-16", "8", "BREAD/BAKERY", "1.5", "0"],
            ["2017-06-16", "1", "NO HABILITADA", "99", "0"],
        ])
        origen = self.csv.read_bytes()
        resultado = self.cargar(lote=1)
        self.assertTrue(resultado.exito, resultado.errores)
        self.assertIsNotNone(resultado.conjunto_id)
        self.assertEqual(resultado.sha256, hashlib.sha256(origen).hexdigest())
        self.assertEqual((resultado.series_importadas,
                          resultado.observaciones_importadas), (2, 3))
        self.assertEqual(len(resultado.advertencias), 8)
        self.assertEqual(self.csv.read_bytes(), origen)

        conjuntos, series, ventas = self.estado()
        self.assertEqual(len(conjuntos), 1)
        self.assertEqual(conjuntos[0], (resultado.conjunto_id, resultado.sha256, 1))
        self.assertEqual({(s[2], s[3]) for s in series}, {PREPARADOS, PANADERIA})
        self.assertTrue(all("1.0; grupo A" in s[4] for s in series))
        self.assertEqual(len(ventas), 3)
        self.assertEqual(sorted(v[3] for v in ventas), [0.0, 1.5, 2.75])
        self.assertNotIn("2017-06-15", {v[2] for v in ventas})
        with closing(conectar(self.base)) as conexion:
            self.assertEqual(conexion.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(conexion.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_csv_invalido_conserva_conjunto_activo(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "1", "0"]])
        self.assertTrue(self.cargar().exito)
        anterior = self.estado()
        self.escribir([["2017-06-17", "17", "PREPARED FOODS", "-1", "0"]])
        resultado = self.cargar()
        self.assertFalse(resultado.exito)
        self.assertIsNone(resultado.conjunto_id)
        self.assertTrue(resultado.errores)
        self.assertEqual(self.estado(), anterior)

    def test_segunda_carga_desactiva_la_anterior_sin_borrarla(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "1", "0"]])
        primero = self.cargar()
        self.escribir([["2017-06-17", "8", "BREAD/BAKERY", "3.5", "2"]])
        segundo = self.cargar()
        self.assertTrue(primero.exito and segundo.exito)
        self.assertNotEqual(primero.conjunto_id, segundo.conjunto_id)
        conjuntos, series, ventas = self.estado()
        self.assertEqual([(c[0], c[2]) for c in conjuntos],
                         [(primero.conjunto_id, 0), (segundo.conjunto_id, 1)])
        self.assertEqual(len(series), 2)
        self.assertEqual(len(ventas), 2)
        self.assertEqual({v[3] for v in ventas}, {1.0, 3.5})

    def test_error_de_insercion_revierte_todo_y_conserva_anterior(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "1", "0"]])
        self.assertTrue(self.cargar().exito)
        with closing(conectar(self.base)) as conexion:
            conexion.execute(
                "CREATE TRIGGER fallo_venta BEFORE INSERT ON venta_diaria "
                "WHEN NEW.sales = 99 BEGIN "
                "SELECT RAISE(ABORT, 'fallo simulado de inserción'); END"
            )
            conexion.commit()
        anterior = self.estado()
        self.escribir([
            ["2017-06-17", "17", "PREPARED FOODS", "2", "0"],
            ["2017-06-18", "17", "PREPARED FOODS", "99", "0"],
        ])
        resultado = self.cargar(lote=1)
        self.assertFalse(resultado.exito)
        self.assertIn("fallo simulado", resultado.errores[0])
        self.assertEqual(self.estado(), anterior)

    def test_error_de_activacion_revierte_todo(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "1", "0"]])
        self.assertTrue(self.cargar().exito)
        with closing(conectar(self.base)) as conexion:
            conexion.execute(
                "CREATE TRIGGER fallo_activacion BEFORE UPDATE OF activo "
                "ON conjunto_datos WHEN NEW.activo = 1 AND OLD.activo = 0 "
                "BEGIN SELECT RAISE(ABORT, 'fallo simulado de activación'); END"
            )
            conexion.commit()
        anterior = self.estado()
        self.escribir([["2017-06-17", "8", "BREAD/BAKERY", "2", "0"]])
        resultado = self.cargar()
        self.assertFalse(resultado.exito)
        self.assertIn("fallo simulado", resultado.errores[0])
        self.assertEqual(self.estado(), anterior)

    def test_sin_series_habilitadas_no_modifica_la_base(self) -> None:
        self.escribir([["2017-06-15", "17", "PREPARED FOODS", "2", "0"]])
        self.assertTrue(self.cargar().exito)
        self.escribir([["2017-06-16", "1", "OTRA", "1", "0"]])
        anterior = self.estado()
        resultado = self.cargar()
        self.assertFalse(resultado.exito)
        self.assertIn("no hay series utilizables", resultado.errores[0])
        self.assertEqual(self.estado(), anterior)

    def test_subconjunto_informa_las_nueve_series_ausentes(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "0", "0"]])
        resultado = self.cargar()
        self.assertTrue(resultado.exito)
        self.assertEqual((resultado.series_importadas,
                          resultado.observaciones_importadas), (1, 1))
        self.assertEqual(len(resultado.advertencias), 9)
        self.assertTrue(any("BREAD/BAKERY" in aviso for aviso in resultado.advertencias))
        self.assertFalse(any("PREPARED FOODS" in aviso
                             for aviso in resultado.advertencias))

    def test_validacion_e_importacion_usan_la_misma_copia_temporal(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "2.5", "0"]])
        hash_inicial = hashlib.sha256(self.csv.read_bytes()).hexdigest()
        rutas_validadas: list[Path] = []

        def validar_y_cambiar_origen(copia: Path):
            rutas_validadas.append(copia)
            resultado = validar_csv(copia)
            self.escribir([["2017-06-16", "17", "PREPARED FOODS", "99", "0"]])
            return resultado

        with patch("backend.app.services.carga_datos.validar_csv",
                   side_effect=validar_y_cambiar_origen):
            resultado = self.cargar()
        self.assertTrue(resultado.exito)
        self.assertNotEqual(rutas_validadas[0], self.csv)
        self.assertFalse(rutas_validadas[0].exists())
        self.assertEqual(resultado.sha256, hash_inicial)
        self.assertEqual(self.estado()[2][0][3], 2.5)

    def test_no_crea_base_no_inicializada(self) -> None:
        self.escribir([["2017-06-16", "17", "PREPARED FOODS", "1", "0"]])
        inexistente = self.base.with_name("inexistente.sqlite3")
        resultado = cargar_csv(self.csv, inexistente, self.config)
        self.assertFalse(resultado.exito)
        self.assertFalse(inexistente.exists())


if __name__ == "__main__":
    unittest.main()
