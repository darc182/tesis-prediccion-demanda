"""Pruebas HTTP del endpoint de carga con bases SQLite temporales."""

from contextlib import closing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services.carga_datos import ResultadoCargaDatos, cargar_csv
from backend.app.storage.db import conectar, inicializar_base


CABECERA = "date,store_nbr,family,sales,onpromotion\n"
VALIDO = (
    CABECERA
    + "2017-06-14,17,PREPARED FOODS,0,0\n"
    + "2017-06-16,17,PREPARED FOODS,2.75,1\n"
    + "2017-06-16,1,NO HABILITADA,99,0\n"
).encode("utf-8")


class ApiCargaCSVTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.base = Path(self.temporal.name) / "prueba.sqlite3"
        inicializar_base(self.base)
        app.dependency_overrides[obtener_ruta_base] = lambda: self.base
        self.addCleanup(app.dependency_overrides.clear)
        self.cliente = TestClient(app)
        self.addCleanup(self.cliente.close)

    def subir(self, contenido: bytes = VALIDO, nombre: str = "ventas.csv",
              tipo: str = "text/csv"):
        return self.cliente.post(
            "/api/datasets/upload",
            files={"file": (nombre, contenido, tipo)},
        )

    def test_multipart_valido_importa_solo_series_habilitadas(self) -> None:
        respuesta = self.subir(tipo="application/octet-stream")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        datos = respuesta.json()
        self.assertEqual(set(datos), {
            "exito", "conjunto_id", "sha256", "series_importadas",
            "observaciones_importadas", "errores", "advertencias",
        })
        self.assertTrue(datos["exito"])
        self.assertEqual((datos["series_importadas"],
                          datos["observaciones_importadas"]), (1, 2))
        self.assertEqual(len(datos["advertencias"]), 9)
        self.assertEqual(datos["errores"], [])
        with closing(conectar(self.base)) as conexion:
            nombre = conexion.execute(
                "SELECT nombre_archivo FROM conjunto_datos WHERE id = ?",
                (datos["conjunto_id"],),
            ).fetchone()[0]
            claves = conexion.execute(
                "SELECT store_nbr, family FROM serie_temporal"
            ).fetchall()
            ventas = conexion.execute(
                "SELECT fecha, sales FROM venta_diaria ORDER BY fecha"
            ).fetchall()
        self.assertEqual(nombre, "ventas.csv")
        self.assertEqual(claves, [(17, "PREPARED FOODS")])
        self.assertEqual(ventas, [("2017-06-14", 0.0), ("2017-06-16", 2.75)])

    def test_nombre_seguro_conserva_solo_el_basename(self) -> None:
        respuesta = self.subir(nombre="..\\carpeta\\origen.csv")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        with closing(conectar(self.base)) as conexion:
            nombre = conexion.execute(
                "SELECT nombre_archivo FROM conjunto_datos"
            ).fetchone()[0]
        self.assertEqual(nombre, "origen.csv")

    def test_nombre_y_extension_invalidos(self) -> None:
        for nombre in ("ventas.txt", "con?.csv", ".csv"):
            with self.subTest(nombre=nombre):
                respuesta = self.subir(nombre=nombre)
                self.assertEqual(respuesta.status_code, 422)
                self.assertFalse(respuesta.json()["exito"])
        with closing(conectar(self.base)) as conexion:
            self.assertEqual(conexion.execute(
                "SELECT COUNT(*) FROM conjunto_datos"
            ).fetchone()[0], 0)

    def test_csv_invalido_y_archivo_vacio(self) -> None:
        invalido = CABECERA.encode("utf-8") + b"2017-06-16,17,PREPARED FOODS,-1,0\n"
        for contenido in (invalido, b""):
            with self.subTest(contenido=contenido):
                respuesta = self.subir(contenido)
                self.assertEqual(respuesta.status_code, 422)
                datos = respuesta.json()
                self.assertFalse(datos["exito"])
                self.assertTrue(datos["errores"])
                self.assertIsNone(datos["conjunto_id"])

    def test_rechazo_conserva_conjunto_activo(self) -> None:
        primera = self.subir()
        self.assertEqual(primera.status_code, 200)
        anterior = primera.json()["conjunto_id"]
        invalido = CABECERA.encode("utf-8") + b"2017-06-17,17,PREPARED FOODS,-2,0\n"
        rechazo = self.subir(invalido)
        self.assertEqual(rechazo.status_code, 422)
        with closing(conectar(self.base)) as conexion:
            activos = conexion.execute(
                "SELECT id FROM conjunto_datos WHERE activo = 1"
            ).fetchall()
            ventas = conexion.execute("SELECT COUNT(*) FROM venta_diaria").fetchone()[0]
        self.assertEqual(activos, [(anterior,)])
        self.assertEqual(ventas, 2)

    def test_csv_sin_series_utilizables_devuelve_422(self) -> None:
        contenido = CABECERA.encode("utf-8") + b"2017-06-16,1,OTRA,1,0\n"
        respuesta = self.subir(contenido)
        self.assertEqual(respuesta.status_code, 422)
        self.assertIn("no hay series utilizables", " ".join(respuesta.json()["errores"]))

    def test_base_inexistente_o_no_inicializada_devuelve_503(self) -> None:
        ausente = Path(self.temporal.name) / "ausente.sqlite3"
        app.dependency_overrides[obtener_ruta_base] = lambda: ausente
        respuesta = self.subir()
        self.assertEqual(respuesta.status_code, 503, respuesta.text)
        self.assertFalse(ausente.exists())
        self.assertEqual(set(respuesta.json()), {
            "exito", "conjunto_id", "sha256", "series_importadas",
            "observaciones_importadas", "errores", "advertencias",
        })

        corrupta = Path(self.temporal.name) / "corrupta.sqlite3"
        corrupta.write_bytes(b"no es una base sqlite")
        app.dependency_overrides[obtener_ruta_base] = lambda: corrupta
        respuesta = self.subir()
        self.assertEqual(respuesta.status_code, 503, respuesta.text)

        vacia = Path(self.temporal.name) / "sin_esquema.sqlite3"
        vacia.touch()
        app.dependency_overrides[obtener_ruta_base] = lambda: vacia
        respuesta = self.subir()
        self.assertEqual(respuesta.status_code, 503, respuesta.text)

    def test_archivo_temporal_se_elimina_en_exito_y_fallo(self) -> None:
        rutas: list[Path] = []

        def observar(ruta: Path, base: Path):
            rutas.append(ruta)
            self.assertTrue(ruta.exists())
            return cargar_csv(ruta, base)

        with patch("backend.app.api.datasets.cargar_csv", side_effect=observar):
            self.assertEqual(self.subir().status_code, 200)
            self.assertEqual(self.subir(b"").status_code, 422)
        self.assertEqual(len(rutas), 2)
        self.assertTrue(all(not ruta.exists() for ruta in rutas))

    def test_fallo_interno_devuelve_500_con_misma_estructura(self) -> None:
        rutas: list[Path] = []

        def fallar(ruta: Path, _base: Path):
            rutas.append(ruta)
            raise RuntimeError("fallo simulado")

        with patch("backend.app.api.datasets.cargar_csv",
                   side_effect=fallar):
            respuesta = self.subir()
        self.assertEqual(respuesta.status_code, 500)
        self.assertFalse(respuesta.json()["exito"])
        self.assertTrue(respuesta.json()["errores"])
        self.assertEqual(len(rutas), 1)
        self.assertFalse(rutas[0].exists())

    def test_falta_campo_file_devuelve_json_consistente(self) -> None:
        respuesta = self.cliente.post("/api/datasets/upload", data={})
        self.assertEqual(respuesta.status_code, 422)
        self.assertEqual(set(respuesta.json()), {
            "exito", "conjunto_id", "sha256", "series_importadas",
            "observaciones_importadas", "errores", "advertencias",
        })

    def test_estado_http_depende_del_codigo_y_no_del_mensaje(self) -> None:
        esperado = {
            "csv_invalido": 422,
            "sin_series_utilizables": 422,
            "base_no_disponible": 503,
            "configuracion_invalida": 500,
            "fallo_interno": 500,
            None: 500,
            "codigo_desconocido": 500,
        }
        claves_publicas = {
            "exito", "conjunto_id", "sha256", "series_importadas",
            "observaciones_importadas", "errores", "advertencias",
        }
        for codigo, estado in esperado.items():
            for mensaje in ("Texto libre", "no hay series utilizables; no such table"):
                with self.subTest(codigo=codigo, mensaje=mensaje):
                    resultado = ResultadoCargaDatos(
                        False, None, None, 0, 0, (mensaje,), (), codigo,
                    )
                    with patch("backend.app.api.datasets.cargar_csv", return_value=resultado):
                        respuesta = self.subir()
                    self.assertEqual(respuesta.status_code, estado)
                    self.assertEqual(respuesta.json()["errores"], [mensaje])
                    self.assertEqual(set(respuesta.json()), claves_publicas)

    def test_ruta_por_defecto_no_depende_del_cwd(self) -> None:
        with patch.dict(os.environ, {"TESIS_DB_PATH": ""}):
            ruta = obtener_ruta_base()
        esperado = Path(__file__).resolve().parents[1] / "data/local/tesis.sqlite3"
        self.assertEqual(ruta, esperado)
        with patch.dict(os.environ, {"TESIS_DB_PATH": "data/local/otra.sqlite3"}):
            self.assertEqual(obtener_ruta_base(), esperado.with_name("otra.sqlite3"))

    def test_health_y_docs(self) -> None:
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.cliente.get("/docs").status_code, 200)
        esquema = self.cliente.get("/openapi.json").json()
        operacion = esquema["paths"]["/api/datasets/upload"]["post"]
        self.assertIn("multipart/form-data", str(operacion))


if __name__ == "__main__":
    unittest.main()
