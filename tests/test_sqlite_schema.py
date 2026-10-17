"""Verificaciones del esquema inicial sin datos reales."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.app.storage.db import conectar, inicializar_base


TABLAS = {
    "conjunto_datos",
    "serie_temporal",
    "venta_diaria",
    "experimento",
    "pronostico",
    "detalle_pronostico",
    "evaluacion",
    "simulacion",
}


class SQLiteSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name) / "prueba.sqlite3"
        inicializar_base(self.ruta)
        self.conexion = conectar(self.ruta)
        self.addCleanup(self.conexion.close)

    def insertar_conjunto_y_serie(self) -> None:
        self.conexion.execute(
            "INSERT INTO conjunto_datos "
            "(id, nombre_archivo, hash, fecha_carga) VALUES (1, ?, ?, ?)",
            ("ventas.csv", "hash-de-prueba", "2026-01-01"),
        )
        self.conexion.execute(
            "INSERT INTO serie_temporal "
            "(id, conjunto_id, store_nbr, family, criterio_seleccion) "
            "VALUES (1, 1, 1, 'GROCERY I', 'serie representativa')"
        )

    def insertar_pronostico(self, tipo: str = "validacion") -> None:
        self.insertar_conjunto_y_serie()
        self.conexion.execute(
            "INSERT INTO experimento "
            "(id, serie_id, tecnica, configuracion_json, "
            "inicio_entrenamiento, fin_entrenamiento) "
            "VALUES (1, 1, 'base', '{}', '2017-01-01', '2017-07-01')"
        )
        self.conexion.execute(
            "INSERT INTO pronostico "
            "(id, experimento_id, fecha_corte, tipo, fecha_generacion) "
            "VALUES (1, 1, '2017-07-01', ?, '2026-01-01')",
            (tipo,),
        )

    def test_crea_exactamente_ocho_tablas(self) -> None:
        tablas = {
            fila[0]
            for fila in self.conexion.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        self.assertEqual(tablas, TABLAS)

    def test_claves_foraneas_activas_y_definidas(self) -> None:
        self.assertEqual(
            self.conexion.execute("PRAGMA foreign_keys").fetchone()[0], 1
        )
        esperadas = {
            "serie_temporal": "conjunto_datos",
            "venta_diaria": "serie_temporal",
            "experimento": "serie_temporal",
            "pronostico": "experimento",
            "detalle_pronostico": "pronostico",
            "evaluacion": "pronostico",
            "simulacion": "pronostico",
        }
        for tabla, padre in esperadas.items():
            with self.subTest(tabla=tabla):
                referencias = {
                    fila[2]
                    for fila in self.conexion.execute(
                        f"PRAGMA foreign_key_list({tabla})"
                    )
                }
                self.assertEqual(referencias, {padre})

    def test_rechaza_series_y_ventas_duplicadas(self) -> None:
        self.insertar_conjunto_y_serie()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conexion.execute(
                "INSERT INTO serie_temporal "
                "(conjunto_id, store_nbr, family, criterio_seleccion) "
                "VALUES (1, 1, 'GROCERY I', 'otro criterio')"
            )
        self.conexion.execute(
            "INSERT INTO venta_diaria (serie_id, fecha, sales, onpromotion) "
            "VALUES (1, '2017-07-01', 1.5, 0)"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conexion.execute(
                "INSERT INTO venta_diaria (serie_id, fecha, sales, onpromotion) "
                "VALUES (1, '2017-07-01', 2.25, 1)"
            )
        self.assertEqual(
            self.conexion.execute("SELECT sales FROM venta_diaria").fetchone()[0],
            1.5,
        )

    def test_rechaza_referencias_inexistentes(self) -> None:
        inserciones = {
            "serie_temporal": (
                "INSERT INTO serie_temporal "
                "(conjunto_id, store_nbr, family, criterio_seleccion) "
                "VALUES (999, 1, 'GROCERY I', 'prueba')"
            ),
            "venta_diaria": (
                "INSERT INTO venta_diaria (serie_id, fecha, sales, onpromotion) "
                "VALUES (999, '2017-07-01', 1.5, 0)"
            ),
            "experimento": (
                "INSERT INTO experimento "
                "(serie_id, tecnica, configuracion_json, "
                "inicio_entrenamiento, fin_entrenamiento) "
                "VALUES (999, 'base', '{}', '2017-01-01', '2017-07-01')"
            ),
            "pronostico": (
                "INSERT INTO pronostico "
                "(experimento_id, fecha_corte, tipo, fecha_generacion) "
                "VALUES (999, '2017-07-01', 'prueba', '2026-01-01')"
            ),
            "detalle_pronostico": (
                "INSERT INTO detalle_pronostico "
                "(pronostico_id, paso, fecha_objetivo, venta_estimada) "
                "VALUES (999, 1, '2017-07-02', 1.5)"
            ),
            "evaluacion": (
                "INSERT INTO evaluacion (pronostico_id, mae) VALUES (999, 1.5)"
            ),
            "simulacion": (
                "INSERT INTO simulacion "
                "(pronostico_id, inventario_disponible, tiempo_abastecimiento, "
                "reserva, demanda_considerada, reposicion, regla_aplicada, fecha) "
                "VALUES (999, 10, 1, 2, 5, 0, 'prueba', '2026-01-01')"
            ),
        }
        for tabla, sql in inserciones.items():
            with self.subTest(tabla=tabla):
                with self.assertRaises(sqlite3.IntegrityError):
                    self.conexion.execute(sql)

    def test_wape_y_ruta_modelo_aceptan_null(self) -> None:
        self.insertar_pronostico()
        self.conexion.execute(
            "INSERT INTO evaluacion (pronostico_id, mae, wape) "
            "VALUES (1, 1.25, NULL)"
        )
        self.assertEqual(
            self.conexion.execute(
                "SELECT wape, ruta_modelo FROM evaluacion "
                "JOIN pronostico ON pronostico.id = evaluacion.pronostico_id "
                "JOIN experimento ON experimento.id = pronostico.experimento_id"
            ).fetchone(),
            (None, None),
        )

    def test_tipo_y_horizonte_del_pronostico(self) -> None:
        self.insertar_pronostico(tipo="futuro")
        self.assertEqual(
            self.conexion.execute(
                "SELECT tipo, horizonte FROM pronostico"
            ).fetchone(),
            ("futuro", 15),
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conexion.execute(
                "INSERT INTO pronostico "
                "(experimento_id, fecha_corte, tipo, horizonte, fecha_generacion) "
                "VALUES (1, '2017-07-01', 'futuro', 14, '2026-01-01')"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conexion.execute(
                "INSERT INTO detalle_pronostico "
                "(pronostico_id, paso, fecha_objetivo, venta_estimada) "
                "VALUES (1, 16, '2017-07-17', 3.5)"
            )

    def test_tiempo_abastecimiento_entre_cero_y_quince_dias_enteros(self) -> None:
        self.insertar_pronostico()
        sql = (
            "INSERT INTO simulacion "
            "(pronostico_id, inventario_disponible, tiempo_abastecimiento, "
            "reserva, demanda_considerada, reposicion, regla_aplicada, fecha) "
            "VALUES (1, 10, ?, 2, 5, 0, 'prueba', '2026-01-01')"
        )
        for dias in (0, 15):
            with self.subTest(dias=dias):
                self.conexion.execute(sql, (dias,))

        tiempos = [
            fila[0]
            for fila in self.conexion.execute(
                "SELECT tiempo_abastecimiento FROM simulacion ORDER BY id"
            )
        ]
        self.assertEqual(tiempos, [0, 15])

        for dias in (16, 1.5):
            with self.subTest(dias=dias):
                with self.assertRaises(sqlite3.IntegrityError):
                    self.conexion.execute(sql, (dias,))


if __name__ == "__main__":
    unittest.main()
