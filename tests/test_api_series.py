"""Pruebas HTTP de la consulta de series del conjunto activo."""

from contextlib import closing
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.storage.db import conectar, inicializar_base


class ApiSeriesTests(unittest.TestCase):
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

    def insertar_serie(self, serie_id: int, conjunto_id: int,
                      tienda: int, familia: str) -> None:
        with closing(conectar(self.base)) as conexion:
            conexion.execute(
                "INSERT INTO serie_temporal "
                "(id, conjunto_id, store_nbr, family, criterio_seleccion) "
                "VALUES (?, ?, ?, ?, ?)",
                (serie_id, conjunto_id, tienda, familia, "prueba"),
            )
            conexion.commit()

    def test_consulta_ordenada_y_excluye_conjuntos_inactivos(self) -> None:
        self.insertar_conjunto(1, 0)
        self.insertar_conjunto(2, 1)
        self.insertar_serie(1, 1, 1, "INACTIVA")
        self.insertar_serie(2, 2, 17, "PREPARED FOODS")
        self.insertar_serie(3, 2, 1, "BREAD/BAKERY")
        self.insertar_serie(4, 2, 1, "BEVERAGES")

        respuesta = self.cliente.get("/api/datasets/active/series")

        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json(), {
            "conjunto_id": 2,
            "series": [
                {"id": 4, "store_nbr": 1, "family": "BEVERAGES"},
                {"id": 3, "store_nbr": 1, "family": "BREAD/BAKERY"},
                {"id": 2, "store_nbr": 17, "family": "PREPARED FOODS"},
            ],
        })

    def test_sin_conjunto_activo(self) -> None:
        self.insertar_conjunto(1, 0)
        self.insertar_serie(1, 1, 17, "PREPARED FOODS")
        respuesta = self.cliente.get("/api/datasets/active/series")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json(), {"conjunto_id": None, "series": []})

    def test_conjunto_activo_sin_series(self) -> None:
        self.insertar_conjunto(2, 1)
        respuesta = self.cliente.get("/api/datasets/active/series")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json(), {"conjunto_id": 2, "series": []})

    def test_base_ausente_sin_esquema_o_danada_devuelve_503(self) -> None:
        directorio = self.base.parent
        ausente = directorio / "ausente.sqlite3"
        vacia = directorio / "vacia.sqlite3"
        vacia.touch()
        danada = directorio / "danada.sqlite3"
        danada.write_bytes(b"no es sqlite")
        for ruta in (ausente, vacia, danada):
            with self.subTest(ruta=ruta.name):
                contenido_antes = ruta.read_bytes() if ruta.exists() else None
                app.dependency_overrides[obtener_ruta_base] = lambda ruta=ruta: ruta
                respuesta = self.cliente.get("/api/datasets/active/series")
                self.assertEqual(respuesta.status_code, 503, respuesta.text)
                self.assertIn("base de datos", respuesta.json()["detail"])
                contenido_despues = ruta.read_bytes() if ruta.exists() else None
                self.assertEqual(contenido_despues, contenido_antes)

    def test_consulta_no_modifica_sqlite(self) -> None:
        self.insertar_conjunto(2, 1)
        self.insertar_serie(7, 2, 17, "PREPARED FOODS")
        contenido_antes = self.base.read_bytes()
        respuesta = self.cliente.get("/api/datasets/active/series")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(self.base.read_bytes(), contenido_antes)

    def test_upload_health_y_openapi_siguen_disponibles(self) -> None:
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        csv = b"date,store_nbr,family,sales,onpromotion\n2017-06-16,17,PREPARED FOODS,2.75,1\n"
        respuesta = self.cliente.post(
            "/api/datasets/upload", files={"file": ("ventas.csv", csv, "text/csv")},
        )
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertTrue(respuesta.json()["exito"])
        esquema = self.cliente.get("/openapi.json").json()
        self.assertIn("get", esquema["paths"]["/api/datasets/active/series"])
        self.assertIn("post", esquema["paths"]["/api/datasets/upload"])


if __name__ == "__main__":
    unittest.main()
