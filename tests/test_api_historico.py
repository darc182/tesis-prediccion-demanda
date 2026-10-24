"""Pruebas HTTP del histórico de ventas con bases SQLite temporales."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.storage.db import conectar, inicializar_base


class ApiHistoricoTests(unittest.TestCase):
    def setUp(self) -> None:
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.base = Path(temporal.name) / "prueba.sqlite3"
        inicializar_base(self.base)
        app.dependency_overrides[obtener_ruta_base] = lambda: self.base
        self.addCleanup(app.dependency_overrides.clear)
        self.cliente = TestClient(app)
        self.addCleanup(self.cliente.close)

    def insertar_conjunto(self, conjunto_id: int, activo: int) -> None:
        with closing(conectar(self.base)) as conexion:
            conexion.execute(
                "INSERT INTO conjunto_datos "
                "(id, nombre_archivo, hash, fecha_carga, activo) VALUES (?, ?, ?, ?, ?)",
                (conjunto_id, f"datos_{conjunto_id}.csv", f"hash_{conjunto_id}",
                 "2026-01-01", activo),
            )
            conexion.commit()

    def insertar_serie(self, serie_id: int, conjunto_id: int) -> None:
        with closing(conectar(self.base)) as conexion:
            conexion.execute(
                "INSERT INTO serie_temporal "
                "(id, conjunto_id, store_nbr, family, criterio_seleccion) "
                "VALUES (?, ?, ?, ?, ?)",
                (serie_id, conjunto_id, 17, "PREPARED FOODS", "prueba"),
            )
            conexion.commit()

    def preparar_historico(self) -> None:
        self.insertar_conjunto(2, 1)
        self.insertar_serie(7, 2)
        with closing(conectar(self.base)) as conexion:
            conexion.executemany(
                "INSERT INTO venta_diaria (serie_id, fecha, sales, onpromotion) "
                "VALUES (?, ?, ?, ?)",
                [
                    (7, "2017-06-18", 3.5, 0),
                    (7, "2017-06-14", 0.0, 0),
                    (7, "2017-06-16", 2.75, 1),
                ],
            )
            conexion.commit()

    def consultar(self, serie_id: int = 7, **params: str):
        return self.cliente.get(f"/api/datasets/active/series/{serie_id}/sales", params=params)

    def test_serie_valida_ordenada_con_ceros_fracciones_y_dias_ausentes(self) -> None:
        self.preparar_historico()
        respuesta = self.consultar()
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json(), {
            "conjunto_id": 2,
            "serie": {"id": 7, "store_nbr": 17, "family": "PREPARED FOODS"},
            "observaciones": [
                {"fecha": "2017-06-14", "sales": 0.0, "onpromotion": 0},
                {"fecha": "2017-06-16", "sales": 2.75, "onpromotion": 1},
                {"fecha": "2017-06-18", "sales": 3.5, "onpromotion": 0},
            ],
        })

    def test_filtros_inclusivos_y_limites_independientes(self) -> None:
        self.preparar_historico()
        casos = [
            ({"desde": "2017-06-14", "hasta": "2017-06-16"},
             ["2017-06-14", "2017-06-16"]),
            ({"desde": "2017-06-16"}, ["2017-06-16", "2017-06-18"]),
            ({"hasta": "2017-06-16"}, ["2017-06-14", "2017-06-16"]),
            ({"desde": "2017-06-16", "hasta": "2017-06-16"}, ["2017-06-16"]),
        ]
        for params, fechas in casos:
            with self.subTest(params=params):
                respuesta = self.consultar(**params)
                self.assertEqual(respuesta.status_code, 200, respuesta.text)
                self.assertEqual(
                    [fila["fecha"] for fila in respuesta.json()["observaciones"]], fechas,
                )

    def test_fechas_invalidas_y_rango_invertido(self) -> None:
        self.preparar_historico()
        for params in (
            {"desde": "2017-6-14"},
            {"desde": "2017-02-29"},
            {"hasta": "2017-13-01"},
            {"hasta": "20170616"},
            {"desde": "2017-06-18", "hasta": "2017-06-14"},
        ):
            with self.subTest(params=params):
                respuesta = self.consultar(**params)
                self.assertEqual(respuesta.status_code, 422, respuesta.text)
                self.assertTrue(respuesta.json()["detail"])

    def test_periodo_sin_observaciones_conserva_la_serie(self) -> None:
        self.preparar_historico()
        respuesta = self.consultar(desde="2017-06-15", hasta="2017-06-15")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json(), {
            "conjunto_id": 2,
            "serie": {"id": 7, "store_nbr": 17, "family": "PREPARED FOODS"},
            "observaciones": [],
        })

    def test_serie_sin_ventas_tambien_devuelve_lista_vacia(self) -> None:
        self.insertar_conjunto(2, 1)
        self.insertar_serie(7, 2)
        respuesta = self.consultar()
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json()["observaciones"], [])

    def test_serie_inexistente_o_de_conjunto_inactivo(self) -> None:
        self.insertar_conjunto(1, 0)
        self.insertar_serie(6, 1)
        self.insertar_conjunto(2, 1)
        self.insertar_serie(7, 2)
        for serie_id in (6, 999):
            with self.subTest(serie_id=serie_id):
                respuesta = self.consultar(serie_id)
                self.assertEqual(respuesta.status_code, 404, respuesta.text)

    def test_sin_conjunto_activo_devuelve_404(self) -> None:
        self.insertar_conjunto(1, 0)
        self.insertar_serie(7, 1)
        respuesta = self.consultar()
        self.assertEqual(respuesta.status_code, 404, respuesta.text)

    def test_base_ausente_danada_o_sin_esquema_devuelve_503(self) -> None:
        directorio = self.base.parent
        ausente = directorio / "ausente.sqlite3"
        danada = directorio / "danada.sqlite3"
        danada.write_bytes(b"no es sqlite")
        vacia = directorio / "sin_esquema.sqlite3"
        vacia.touch()
        parcial = directorio / "sin_ventas.sqlite3"
        with closing(sqlite3.connect(parcial)) as conexion:
            conexion.executescript(
                "CREATE TABLE conjunto_datos (id INTEGER, activo INTEGER);"
                "CREATE TABLE serie_temporal (id INTEGER, conjunto_id INTEGER, "
                "store_nbr INTEGER, family TEXT);"
            )
        for ruta in (ausente, danada, vacia, parcial):
            with self.subTest(ruta=ruta.name):
                contenido_antes = ruta.read_bytes() if ruta.exists() else None
                app.dependency_overrides[obtener_ruta_base] = lambda ruta=ruta: ruta
                respuesta = self.consultar()
                self.assertEqual(respuesta.status_code, 503, respuesta.text)
                self.assertIn("base de datos", respuesta.json()["detail"])
                contenido_despues = ruta.read_bytes() if ruta.exists() else None
                self.assertEqual(contenido_despues, contenido_antes)

    def test_consulta_no_modifica_sqlite(self) -> None:
        self.preparar_historico()
        contenido_antes = self.base.read_bytes()
        self.assertEqual(self.consultar(desde="2017-06-14").status_code, 200)
        self.assertEqual(self.base.read_bytes(), contenido_antes)

    def test_endpoints_anteriores_y_openapi(self) -> None:
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        self.assertEqual(
            self.cliente.get("/api/datasets/active/series").json(),
            {"conjunto_id": None, "series": []},
        )
        csv = b"date,store_nbr,family,sales,onpromotion\n2017-06-16,17,PREPARED FOODS,2.75,1\n"
        respuesta = self.cliente.post(
            "/api/datasets/upload", files={"file": ("ventas.csv", csv, "text/csv")},
        )
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertTrue(respuesta.json()["exito"])
        self.assertEqual(len(self.cliente.get("/api/datasets/active/series").json()["series"]), 1)
        esquema = self.cliente.get("/openapi.json").json()["paths"]
        self.assertIn("get", esquema["/api/datasets/active/series/{serie_id}/sales"])
        self.assertIn("get", esquema["/api/datasets/active/series"])
        self.assertIn("post", esquema["/api/datasets/upload"])


if __name__ == "__main__":
    unittest.main()
