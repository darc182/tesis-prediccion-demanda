"""G5-B3: HTTP real, procedencia global e instantánea WAL; solo datos temporales."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from backend.app.api import datasets as api
from backend.app.main import app
from backend.app.services import perfil_global_conjunto as persistencia
from backend.app.storage.db import SCHEMA_PATH
from tests import test_integridad_exploracion as fixture


URL = "/api/datasets/active/perfil-global"
TABLAS = fixture.TABLAS + ("perfil_global_conjunto",)


class ApiPerfilGlobalTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixture.IntegridadExploracionTests()
        self.addCleanup(self.datos.doCleanups)
        self.datos.setUp()
        self.ruta = self.datos.ruta
        self.cliente = self.datos.cliente

    def get(self):
        return self.cliente.get(URL)

    def perfil(self, conjunto=None):
        with closing(sqlite3.connect(self.ruta.as_uri() + "?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            return persistencia.recuperar_perfil_global(c, conjunto or self.datos.conjunto)

    def estado(self):
        return tuple(tuple(self.datos.sql(f"SELECT * FROM {t} ORDER BY 1")) for t in TABLAS)

    def error(self, estado):
        r = self.get()
        self.assertEqual(r.status_code, estado, r.text)
        self.assertEqual(set(r.json()), {"detail"})
        self.assertIsInstance(r.json()["detail"], str)
        self.assertEqual(r.headers["cache-control"], "no-store")
        for privado in (str(self.ruta), "SELECT", "Traceback", "PRIVADO", "reporte_json"):
            self.assertNotIn(privado, r.text)
        return r

    def alterar(self, columna, valor):
        # Solo la fixture temporal puede vulnerar CHECK para simular corrupción.
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute(f"UPDATE perfil_global_conjunto SET {columna}=? WHERE conjunto_id=?",
                      (valor, self.datos.conjunto))

    def test_200_identico_al_servicio_y_g1_completo(self):
        esperado = self.perfil()
        r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), esperado)
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertEqual(set(r.json()), {"conjunto_id", "version_perfil", "sha256_csv",
                                        "series_importadas", "observaciones_importadas", "reporte"})
        self.assertEqual(esperado["sha256_csv"], hashlib.sha256(self.datos.contenido).hexdigest())
        from scripts.perfil_global_csv import perfilar_csv_global
        self.assertEqual(esperado["reporte"], perfilar_csv_global(self.datos.csv, tamano_bloque=3))

    def test_procedencia_global_distinta_del_subconjunto(self):
        p = self.get().json()
        g = p["reporte"]
        self.assertEqual((p["series_importadas"], p["observaciones_importadas"]), (10, 80))
        self.assertEqual((g["volumen"]["filas_examinadas"], g["volumen"]["combinaciones_observadas"]), (88, 11))
        self.assertEqual(g["ventas"]["suma"], 1480)
        self.assertEqual(g["promociones"]["suma_conteos_onpromotion"], 140)
        self.assertEqual(self.datos.sql("SELECT SUM(sales), SUM(onpromotion) FROM venta_diaria"), [(680, 100)])
        self.assertEqual(g["fuente"]["nombre_archivo"], "exploracion_sintetica.csv")
        self.assertEqual(p["sha256_csv"], self.datos.carga["sha256"])

    def test_orden_fracciones_ausencias_y_declaracion_retrospectiva(self):
        g = self.get().json()["reporte"]
        self.assertEqual(g, self.perfil()["reporte"])
        self.assertEqual(g["ventas"]["registros_fraccionarios"], 40)
        self.assertEqual(g["ventas"]["registros_cero"], 10)
        faltantes = g["temporalidad"]["fechas_sin_registro_global"]
        self.assertEqual(faltantes, sorted(faltantes))
        self.assertEqual(len(faltantes), 198)
        self.assertIn("2016-12-25", faltantes)
        self.assertTrue(g["interpretacion_temporal"]["retrospectivo"])
        self.assertFalse(g["interpretacion_temporal"]["autorizado_para_seleccion_de_modelos"])
        json.dumps(g, allow_nan=False)

    def test_cambio_activo_conserva_perfiles_independientes(self):
        primero = self.get().json()
        segundo = self.cliente.post("/api/datasets/upload", files={
            "file": ("otro.csv", self.datos.contenido.replace(b",100,5", b",101,5"), "text/csv")})
        self.assertEqual(segundo.status_code, 200, segundo.text)
        nuevo = self.get().json()
        self.assertNotEqual(nuevo["conjunto_id"], primero["conjunto_id"])
        self.assertNotEqual(nuevo["sha256_csv"], primero["sha256_csv"])
        self.assertEqual(nuevo["reporte"]["fuente"]["nombre_archivo"], "otro.csv")
        self.assertEqual(self.perfil(primero["conjunto_id"]), primero)
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.datos.sql("UPDATE conjunto_datos SET activo=1 WHERE id=?", (primero["conjunto_id"],))
        self.assertEqual(self.get().json(), primero)

    def test_activo_sin_perfil_404_sin_rescatar_el_anterior(self):
        nuevo = self.datos.subir()["conjunto_id"]
        self.datos.sql("DELETE FROM perfil_global_conjunto WHERE conjunto_id=?", (nuevo,))
        self.error(404)

    def test_legado_ocho_tablas_sin_perfil_404_sin_migracion(self):
        legado = self.ruta.with_name("legado.sqlite3")
        with closing(sqlite3.connect(legado)) as c:
            c.executescript(SCHEMA_PATH.read_text(encoding="utf-8").split("-- Identidad compartida")[0] + "COMMIT;")
            c.execute("ATTACH DATABASE ? AS origen", (str(self.ruta),))
            c.execute("INSERT INTO conjunto_datos SELECT * FROM origen.conjunto_datos")
            c.commit()
        app.dependency_overrides[api.obtener_ruta_base] = lambda: legado
        antes = legado.read_bytes()
        self.error(404)
        self.assertEqual(legado.read_bytes(), antes)
        with closing(sqlite3.connect(legado)) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0], 8)

    def test_sin_conjunto_activo_409(self):
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.assertEqual(self.error(409).json()["detail"], "No existe un conjunto de datos activo.")

    def test_base_ausente_503_sin_crear_archivos(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        app.dependency_overrides[api.obtener_ruta_base] = lambda: ausente
        self.error(503)
        self.assertFalse(ausente.exists())

    def test_base_sin_esquema_503(self):
        vacia = self.ruta.with_name("vacia.sqlite3")
        with closing(sqlite3.connect(vacia)):
            pass
        app.dependency_overrides[api.obtener_ruta_base] = lambda: vacia
        self.error(503)

    def test_base_danada_503(self):
        mala = self.ruta.with_name("danada.sqlite3")
        mala.write_bytes(b"No es SQLite")
        app.dependency_overrides[api.obtener_ruta_base] = lambda: mala
        self.error(503)

    def test_nombre_y_sha_inconsistentes_409(self):
        for columna, valor in (("nombre_archivo", "otro.csv"), ("hash", "0" * 64)):
            with self.subTest(columna=columna):
                anterior = self.datos.sql(f"SELECT {columna} FROM conjunto_datos")[0][0]
                self.datos.sql(f"UPDATE conjunto_datos SET {columna}=?", (valor,))
                self.error(409)
                self.datos.sql(f"UPDATE conjunto_datos SET {columna}=?", (anterior,))

    def test_checksum_corrupto_409(self):
        self.alterar("sha256_reporte", "0" * 64)
        self.error(409)

    def test_json_truncado_y_duplicado_409(self):
        for texto in ('{"privado":', '{"ventas":{},"ventas":{}}'):
            with self.subTest(texto=texto):
                self.alterar("reporte_json", texto)
                self.alterar("sha256_reporte", hashlib.sha256(texto.encode()).hexdigest())
                self.error(409)

    def test_json_contradictorio_con_checksum_correcto_409(self):
        g = self.perfil()["reporte"]
        g["volumen"]["filas_validas"] += 1
        texto = json.dumps(g, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.alterar("reporte_json", texto)
        self.alterar("sha256_reporte", hashlib.sha256(texto.encode()).hexdigest())
        self.error(409)

    def test_version_no_soportada_409(self):
        self.alterar("version_perfil", 99)
        self.error(409)

    def test_reporte_excesivo_409_antes_de_interpretar(self):
        self.alterar("reporte_json", " " * (persistencia.MAX_BYTES_REPORTE + 1))
        with patch.object(persistencia, "_leer_json", side_effect=AssertionError("No interpretar")):
            self.error(409)

    def test_mapeo_explicito_de_todos_los_errores_de_dominio(self):
        for codigo, estado in (("perfil_no_disponible", 404), ("perfil_corrupto", 409),
                               ("perfil_invalido", 409), ("version_no_soportada", 409),
                               ("perfil_demasiado_grande", 409), ("almacenamiento_no_disponible", 503)):
            with self.subTest(codigo=codigo), patch.object(api, "recuperar_perfil_global",
                                                         side_effect=persistencia.ErrorPerfilGlobal(codigo)):
                self.error(estado)

    def test_sqlite_falla_503_seguro(self):
        with patch.object(api, "recuperar_perfil_global", side_effect=sqlite3.OperationalError("PRIVADO SELECT")):
            self.error(503)

    def test_excepcion_inesperada_500_incluso_valueerror(self):
        for error in (RuntimeError("PRIVADO SELECT"), ValueError("PRIVADO")):
            with self.subTest(error=type(error)), patch.object(api, "recuperar_perfil_global", side_effect=error):
                self.error(500)

    def test_conexion_se_cierra_incluso_si_falla(self):
        real = sqlite3.connect
        conexiones = []
        def abrir(destino, *args, **kwargs):
            c = real(destino, *args, **kwargs)
            conexiones.append(c)
            return c
        with patch.object(sqlite3, "connect", side_effect=abrir), patch.object(
                api, "recuperar_perfil_global", side_effect=ValueError("PRIVADO")):
            self.error(500)
        self.assertEqual(len(conexiones), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            conexiones[0].execute("SELECT 1")

    def test_solo_lectura_ro_begin_y_contenido_intacto_delete_y_wal(self):
        self.datos.crear_base()
        real = sqlite3.connect
        for modo in ("DELETE", "WAL"):
            with self.subTest(modo=modo):
                self.datos.sql(f"PRAGMA journal_mode={modo}")
                antes, fisico = self.estado(), self.ruta.read_bytes()
                trazas, conexiones = [], []
                def abrir(destino, *args, **kwargs):
                    self.assertIn("mode=ro", str(destino))
                    self.assertTrue(kwargs.get("uri"))
                    c = real(destino, *args, **kwargs)
                    conexiones.append(c)
                    c.set_trace_callback(trazas.append)
                    return c
                with patch.object(sqlite3, "connect", side_effect=abrir):
                    self.assertEqual(self.get().status_code, 200)
                self.assertEqual(len(conexiones), 1)
                self.assertEqual(trazas[0], "BEGIN")
                self.assertIn("WHERE activo = 1 ORDER BY id DESC LIMIT 1", trazas[1])
                self.assertTrue(any("FROM perfil_global_conjunto" in s for s in trazas))
                self.assertTrue(any("SELECT nombre_archivo,hash" in s for s in trazas))
                self.assertTrue(all(s.lstrip().upper().startswith(("SELECT", "BEGIN")) for s in trazas))
                self.assertEqual(self.estado(), antes)
                if modo == "DELETE":
                    self.assertEqual(self.ruta.read_bytes(), fisico)
                self.assertEqual(self.datos.sql("PRAGMA foreign_key_check"), [])

    def test_sin_csv_ni_g1_inicializacion_entrenamiento_o_registro(self):
        esperado = self.perfil()
        self.datos.csv.unlink()  # El CSV temporal ya no existe: solo se lee SQLite.
        with patch.object(Path, "open", side_effect=AssertionError("GET abre archivo")), \
                patch("scripts.perfil_global_csv.perfilar_csv_global", side_effect=AssertionError("GET G1")), \
                patch("backend.app.storage.db.inicializar_base", side_effect=AssertionError("GET DDL")), \
                patch("backend.app.services.carga_datos.perfilar_csv_global", side_effect=AssertionError("GET carga")), \
                patch("scripts.modelos_prediccion.entrenar_modelo", side_effect=AssertionError("GET entrena")), \
                patch("backend.app.api.pronosticos.registrar_pronostico", side_effect=AssertionError("GET registra")):
            self.assertEqual(self.get().json(), esperado)

    def test_wal_intercalado_misma_conexion_y_snapshot_incluso_metadatos(self):
        primero = self.get().json()
        segundo_id = self.datos.subir()["conjunto_id"]
        segundo = self.perfil(segundo_id)
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.datos.sql("UPDATE conjunto_datos SET activo=1 WHERE id=?", (primero["conjunto_id"],))
        self.datos.sql("PRAGMA journal_mode=WAL")
        real_abrir, real_leer = sqlite3.connect, api.recuperar_perfil_global
        lectores = []
        def abrir(destino, *args, **kwargs):
            c = real_abrir(destino, *args, **kwargs)
            lectores.append(c)
            return c
        def intercalar(c, conjunto):
            self.assertEqual(conjunto, primero["conjunto_id"])
            self.assertEqual(lectores, [c])
            self.assertTrue(c.in_transaction)
            with closing(real_abrir(self.ruta)) as escritor, escritor:
                escritor.execute("UPDATE conjunto_datos SET activo=0")
                escritor.execute("UPDATE conjunto_datos SET activo=1 WHERE id=?", (segundo_id,))
                # Una segunda conexión lectora vería metadatos incompatibles.
                escritor.execute("UPDATE conjunto_datos SET nombre_archivo='cambiado.csv' WHERE id=?", (conjunto,))
            return real_leer(c, conjunto)
        with patch.object(sqlite3, "connect", side_effect=abrir), \
                patch.object(api, "recuperar_perfil_global", side_effect=intercalar):
            r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), primero)
        self.assertEqual(len(lectores), 1)
        self.assertEqual(self.get().json(), segundo)

    def test_contratos_anteriores_y_pronostico_permanecen_intactos(self):
        p = self.datos.crear_base()
        sid, pid = self.datos.serie["id"], p["pronostico_id"]
        rutas = ["/health", "/api/datasets/active/series",
                 f"/api/datasets/active/series/{sid}/sales",
                 f"/api/datasets/active/series/{sid}/perfil?fecha_corte=2017-06-16",
                 f"/api/pronosticos/active/series/{sid}",
                 f"/api/pronosticos/active/{pid}", f"/api/pronosticos/active/{pid}/configuracion"]
        antes = [(r, self.cliente.get(r).json()) for r in rutas]
        self.assertEqual(self.get().status_code, 200)
        for ruta, contenido in antes:
            r = self.cliente.get(ruta)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json(), contenido)
        self.assertEqual(set(self.datos.carga), {"exito", "conjunto_id", "sha256", "series_importadas",
                                               "observaciones_importadas", "errores", "advertencias"})

    def test_openapi_documenta_ruta_sin_parametros_ni_consultas(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("OpenAPI consulta")):
            esquema = self.cliente.get("/openapi.json").json()
            self.assertEqual(self.cliente.get("/docs").status_code, 200)
            self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        operacion = esquema["paths"][URL]["get"]
        self.assertFalse(operacion.get("parameters"))
        self.assertEqual(set(operacion["responses"]), {"200", "404", "409", "503", "500"})


if __name__ == "__main__":
    unittest.main()
