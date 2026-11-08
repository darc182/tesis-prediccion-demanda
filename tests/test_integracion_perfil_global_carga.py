"""G5-B2: HTTP/carga/G1/SQLite reales; exclusivamente fixtures temporales.

La fixture G4 incluye 88 filas y 1.480 ventas globales, 80 filas y 680 ventas
importadas. Los dobles solo observan funciones reales, intercalan cambios
controlados o provocan errores. Los triggers de fallo existen solo en SQLite
sintética. No se añaden observaciones de agosto ni se seleccionan modelos.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Barrier
import unittest
from unittest.mock import patch

from backend.app.api import datasets as api
from backend.app.services import carga_datos as carga
from backend.app.services.perfil_global_conjunto import ErrorPerfilGlobal, recuperar_perfil_global
from backend.app.storage.db import SCHEMA_PATH, conectar
from scripts.perfil_global_csv import ErrorPerfilCSV, perfilar_csv_global
from tests import test_integridad_exploracion as fixture


TABLAS = fixture.TABLAS + ("perfil_global_conjunto",)
CLAVES_POST = {"exito", "conjunto_id", "sha256", "series_importadas", "observaciones_importadas", "errores", "advertencias"}
CSV_IDENTIDAD = (b"date,store_nbr,family,sales,onpromotion\n"
                 b"2017-06-14,17,PREPARED FOODS,2.5,0\n"
                 b"2017-06-15,17,PREPARED FOODS,4.5,0\n")


class IntegracionPerfilGlobalCargaTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixture.IntegridadExploracionTests()
        self.addCleanup(self.datos.doCleanups)
        self.datos.setUp()  # primera carga HTTP real: ya incluye perfil obligatorio
        self.ruta, self.cliente = self.datos.ruta, self.datos.cliente
        self.contenido, self.conjunto = self.datos.contenido, self.datos.conjunto

    def subir(self, contenido=None, nombre="exploracion_sintetica.csv"):
        r = self.cliente.post("/api/datasets/upload", files={
            "file": (nombre, self.contenido if contenido is None else contenido, "text/csv")})
        self.assertEqual(set(r.json()), CLAVES_POST)
        return r

    def estado(self, ruta=None, tablas=TABLAS):
        with closing(conectar(ruta or self.ruta)) as c:
            return tuple(tuple(c.execute(f"SELECT * FROM {t} ORDER BY 1,2")) for t in tablas)

    def perfil(self, conjunto, ruta=None):
        with closing(sqlite3.connect((ruta or self.ruta).as_uri() + "?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            return recuperar_perfil_global(c, conjunto)

    def rechazo(self, respuesta, estado, anterior):
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertFalse(respuesta.json()["exito"])
        self.assertIsNone(respuesta.json()["conjunto_id"])
        self.assertEqual((respuesta.json()["series_importadas"], respuesta.json()["observaciones_importadas"]), (0, 0))
        for privado in (str(self.ruta), "PRIVADO", "SELECT", "RAISE", "Traceback"):
            self.assertNotIn(privado, respuesta.text)
        self.assertEqual(self.estado(), anterior)

    def test_carga_http_guarda_panorama_completo_y_subconjunto_real(self):
        p = self.perfil(self.conjunto)
        esperado = perfilar_csv_global(self.datos.csv, 3)
        self.assertEqual(p["reporte"], esperado)
        self.assertEqual((p["reporte"]["volumen"]["filas_examinadas"], p["reporte"]["volumen"]["combinaciones_observadas"]), (88, 11))
        self.assertEqual((p["reporte"]["ventas"]["suma"], p["reporte"]["promociones"]["suma_conteos_onpromotion"]), (1480, 140))
        self.assertEqual((p["series_importadas"], p["observaciones_importadas"]), (10, 80))
        with closing(conectar(self.ruta)) as c:
            self.assertEqual(c.execute("SELECT SUM(sales),SUM(onpromotion) FROM venta_diaria").fetchone(), (680, 100))
            fila = c.execute("SELECT nombre_archivo,hash FROM conjunto_datos WHERE id=?", (self.conjunto,)).fetchone()
            self.assertEqual(fila, (p["reporte"]["fuente"]["nombre_archivo"], p["sha256_csv"]))
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(p["sha256_csv"], hashlib.sha256(self.contenido).hexdigest())
        self.assertEqual({(s["store_nbr"], s["family"]) for s in self.datos.activas()["series"]},
                         {(s["store_nbr"], s["family"]) for s in self.datos.series})

    def test_secuencia_real_sin_bloqueo_g1_y_sin_publicacion_antes_del_commit(self):
        eventos, visibilidad, temporales = [], [], []
        g1, preparar, abrir, guardar = carga.perfilar_csv_global, carga.preparar_perfil_global, carga._abrir_escritura, carga.guardar_perfil_global
        sha_final = carga._sha256_copia

        def comprobar_sin_bloqueo():
            with closing(conectar(self.ruta)) as c:
                c.execute("BEGIN IMMEDIATE")
                c.rollback()

        def observar_g1(ruta):
            eventos.append("G1")
            temporales.append(ruta)
            comprobar_sin_bloqueo()
            self.assertEqual(ruta.read_bytes(), self.contenido)
            return g1(ruta)

        def observar_preparar(*args):
            eventos.append("PREPARACION")
            comprobar_sin_bloqueo()
            return preparar(*args)

        def trazar(sql):
            s = sql.strip().upper()
            if s == "BEGIN IMMEDIATE":
                eventos.append(s)
            elif s.startswith("INSERT INTO VENTA_DIARIA") and "IMPORTACION" not in eventos:
                eventos.append("IMPORTACION")
            elif s.startswith("UPDATE CONJUNTO_DATOS SET ACTIVO") and "ACTIVACION" not in eventos:
                eventos.append("ACTIVACION")
            elif s == "COMMIT":
                eventos.append("COMMIT")
                # La traza se llama antes de ejecutar COMMIT; observar, no afirmar
                # dentro del callback (SQLite puede silenciar sus excepciones).
                with closing(conectar(self.ruta)) as otra:
                    visibilidad.append((otra.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone()[0],
                                        otra.execute("SELECT COUNT(*) FROM venta_diaria").fetchone()[0],
                                        otra.execute("SELECT id FROM conjunto_datos WHERE activo=1").fetchone()[0]))

        def observar_abrir(ruta):
            c = abrir(ruta)
            c.set_trace_callback(trazar)
            return c

        def observar_guardar(c, conjunto, *args):
            eventos.append("GUARDAR_PERFIL")
            self.assertTrue(c.in_transaction)
            self.assertEqual(c.execute("SELECT activo FROM conjunto_datos WHERE id=?", (conjunto,)).fetchone(), (0,))
            self.assertEqual(c.execute("SELECT COUNT(*) FROM venta_diaria WHERE serie_id IN "
                                       "(SELECT id FROM serie_temporal WHERE conjunto_id=?)", (conjunto,)).fetchone(), (80,))
            guardar(c, conjunto, *args)
            eventos.append("PERFIL_GUARDADO")

        def observar_sha(ruta):
            eventos.append("VERIFICAR_SHA")
            return sha_final(ruta)

        with patch.object(carga, "perfilar_csv_global", side_effect=observar_g1), \
                patch.object(carga, "preparar_perfil_global", side_effect=observar_preparar), \
                patch.object(carga, "_abrir_escritura", side_effect=observar_abrir), \
                patch.object(carga, "_sha256_copia", side_effect=observar_sha), \
                patch.object(carga, "guardar_perfil_global", side_effect=observar_guardar):
            r = self.subir()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(eventos, ["G1", "PREPARACION", "BEGIN IMMEDIATE", "IMPORTACION", "VERIFICAR_SHA", "GUARDAR_PERFIL", "PERFIL_GUARDADO", "ACTIVACION", "COMMIT"])
        self.assertEqual(visibilidad, [(1, 80, self.conjunto)])
        self.assertEqual(len(temporales), 1)
        self.assertFalse(temporales[0].exists())
        self.assertEqual(self.perfil(r.json()["conjunto_id"])["observaciones_importadas"], 80)

    def test_sha_final_normal_por_bloques_perfil_y_ventas_coinciden(self):
        lecturas = []
        sha_real, abrir_real = carga._sha256_copia, Path.open

        class LectorObservado:
            def __init__(self, archivo):
                self.archivo = archivo

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.archivo.close()

            def read(self, tamano):
                bloque = self.archivo.read(tamano)
                lecturas.append((tamano, len(bloque)))
                return bloque

        def abrir_observado(ruta, *args, **kwargs):
            return LectorObservado(abrir_real(ruta, *args, **kwargs))

        def observar_sha(ruta):
            with patch.object(Path, "open", abrir_observado):
                return sha_real(ruta)

        # Bloque reducido solo en la prueba para observar varias lecturas reales.
        with patch.object(carga, "TAMANO_HASH", 32), patch.object(carga, "_sha256_copia", side_effect=observar_sha):
            r = self.subir(CSV_IDENTIDAD, "identidad.csv")
        self.assertEqual(r.status_code, 200, r.text)
        p = self.perfil(r.json()["conjunto_id"])
        self.assertEqual(p["sha256_csv"], hashlib.sha256(CSV_IDENTIDAD).hexdigest())
        self.assertEqual(r.json()["sha256"], p["sha256_csv"])
        self.assertEqual((p["reporte"]["volumen"]["filas_examinadas"], p["reporte"]["ventas"]["suma"]), (2, 7))
        self.assertEqual(self.datos.sql("SELECT v.sales FROM venta_diaria v JOIN serie_temporal s ON s.id=v.serie_id "
                                       "WHERE s.conjunto_id=? ORDER BY v.fecha", (p["conjunto_id"],)), [(2.5,), (4.5,)])
        self.assertGreater(len(lecturas), 2)
        self.assertTrue(all(tamano == 32 and leidos <= 32 for tamano, leidos in lecturas))
        self.assertEqual(sum(leidos for _, leidos in lecturas), len(CSV_IDENTIDAD))
        self.assertEqual(lecturas[-1], (32, 0))

    def _rechazar_copia_alterada(self, antes_de_g1):
        self.datos.crear_base()  # pronóstico previo real, debe conservarse byte a byte en las filas
        anterior = self.estado()
        copia_alterada = CSV_IDENTIDAD.replace(b"2.5", b"9.5", 1)
        self.assertEqual(len(copia_alterada), len(CSV_IDENTIDAD))
        rutas, fuentes, totales, huellas, origen_intacto = [], [], [], [], []
        g1, sha_real, carga_real = carga.perfilar_csv_global, carga._sha256_copia, carga.cargar_csv

        def alterar(ruta):
            rutas.append(ruta)
            if antes_de_g1:
                ruta.write_bytes(copia_alterada)
            reporte = g1(ruta)
            totales.append(reporte["ventas"]["suma"])
            if not antes_de_g1:
                ruta.write_bytes(copia_alterada)
            return reporte

        def observar_sha(ruta):
            valor = sha_real(ruta)
            huellas.append(valor)
            return valor

        def observar_origen(ruta, base):
            fuentes.append(ruta)
            resultado = carga_real(ruta, base)
            origen_intacto.append(ruta.read_bytes() == CSV_IDENTIDAD)
            return resultado

        with patch.object(carga, "perfilar_csv_global", side_effect=alterar), \
                patch.object(carga, "_sha256_copia", side_effect=observar_sha), \
                patch.object(api, "cargar_csv", side_effect=observar_origen), \
                patch.object(carga, "guardar_perfil_global", wraps=carga.guardar_perfil_global) as guardar:
            r = self.subir(CSV_IDENTIDAD, "identidad.csv")
        self.rechazo(r, 500, anterior)
        self.assertEqual(r.json()["errores"], [carga._ERROR_INTERNO])
        self.assertIsNone(r.json()["sha256"])
        self.assertEqual(totales, [14 if antes_de_g1 else 7])
        self.assertEqual(huellas, [hashlib.sha256(copia_alterada).hexdigest()])
        guardar.assert_not_called()
        for huella in (hashlib.sha256(CSV_IDENTIDAD).hexdigest(), huellas[0]):
            self.assertNotIn(huella, r.text)
        self.assertEqual(origen_intacto, [True])
        self.assertEqual(len(rutas), 1)
        self.assertEqual(len(fuentes), 1)
        self.assertTrue(all(not p.exists() and not p.parent.exists() for p in rutas + fuentes))
        self.assertEqual(self.datos.sql("PRAGMA foreign_key_check"), [])
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM perfil_global_conjunto p LEFT JOIN conjunto_datos c "
                                       "ON c.id=p.conjunto_id WHERE c.id IS NULL"), [(0,)])

    def test_alteracion_despues_de_g1_rechaza_sha_y_revierte_todo(self):
        self._rechazar_copia_alterada(antes_de_g1=False)

    def test_alteracion_antes_de_g1_rechaza_sha_y_revierte_todo(self):
        self._rechazar_copia_alterada(antes_de_g1=True)

    def test_csv_invalido_no_perfila_ni_modifica_registros(self):
        anterior = self.estado()
        invalido = b"date,store_nbr,family,sales,onpromotion\n2017-06-16,17,PREPARED FOODS,-1,0\n"
        with patch.object(carga, "perfilar_csv_global") as g1:
            self.rechazo(self.subir(invalido), 422, anterior)
            g1.assert_not_called()

    def test_rechazos_reales_de_g1_y_de_rango_publico_son_422(self):
        anterior = self.estado()
        cabecera = "date,store_nbr,family,sales,onpromotion\n"
        for filas in ("2000-01-01,17,PREPARED FOODS,1,0\n2101-01-01,17,PREPARED FOODS,1,0\n",
                      "2017-06-16,17,PREPARED FOODS,1,9223372036854775807\n",
                      "2017-06-15,17,PREPARED FOODS,1e308,0\n2017-06-16,17,PREPARED FOODS,1e308,0\n"):
            with self.subTest(filas=filas):
                self.rechazo(self.subir((cabecera + filas).encode()), 422, anterior)

    def test_errores_esperados_de_g1_y_preparacion_no_abren_escritura(self):
        anterior = self.estado()
        for funcion, error in (("perfilar_csv_global", ErrorPerfilCSV("PRIVADO C:/datos")),
                               ("preparar_perfil_global", ErrorPerfilGlobal("perfil_demasiado_grande"))):
            with self.subTest(funcion=funcion), patch.object(carga, funcion, side_effect=error), \
                    patch.object(carga, "_abrir_escritura") as abrir:
                self.rechazo(self.subir(), 422, anterior)
                abrir.assert_not_called()

    def test_errores_inesperados_no_se_clasifican_como_datos_invalidos(self):
        anterior = self.estado()
        for funcion in ("perfilar_csv_global", "preparar_perfil_global"):
            for error in (RuntimeError("PRIVADO"), ValueError("SELECT C:/privado"), TypeError("PRIVADO")):
                with self.subTest(funcion=funcion, error=type(error).__name__), patch.object(carga, funcion, side_effect=error):
                    self.rechazo(self.subir(), 500, anterior)

    def test_incoherencia_de_filas_o_fechas_entre_validador_y_g1_no_se_publica(self):
        anterior = self.estado()
        validar = carga.validar_csv
        for cambios in ({"registros_examinados": 89}, {"fecha_minima": "2016-12-23"}, {"fecha_maxima": "2017-07-18"}):
            with self.subTest(cambios=cambios), patch.object(carga, "validar_csv", side_effect=lambda p: replace(validar(p), **cambios)):
                self.rechazo(self.subir(), 500, anterior)

    def test_triggers_reales_ventas_perfil_y_activacion_revierten_todo(self):
        self.datos.crear_base()  # pronóstico previo real, debe sobrevivir intacto
        anterior = self.estado()
        casos = ("BEFORE INSERT ON venta_diaria WHEN NEW.fecha='2017-06-16'",
                 "BEFORE INSERT ON perfil_global_conjunto",
                 "BEFORE UPDATE OF activo ON conjunto_datos WHEN NEW.activo=1 AND OLD.activo=0")
        for condicion in casos:
            with self.subTest(condicion=condicion):
                self.datos.sql(f"CREATE TRIGGER fallo_b2 {condicion} BEGIN SELECT RAISE(ABORT,'PRIVADO SELECT C:/secreto'); END")
                try:
                    self.rechazo(self.subir(), 500, anterior)
                finally:
                    self.datos.sql("DROP TRIGGER fallo_b2")
        self.assertEqual(self.datos.sql("PRAGMA foreign_key_check"), [])

    def test_trigger_de_activacion_exige_que_el_perfil_ya_exista(self):
        self.datos.sql("CREATE TRIGGER requiere_perfil BEFORE UPDATE OF activo ON conjunto_datos "
                       "WHEN NEW.activo=1 AND NOT EXISTS (SELECT 1 FROM perfil_global_conjunto WHERE conjunto_id=NEW.id) "
                       "BEGIN SELECT RAISE(ABORT,'Falta perfil'); END")
        r = self.subir()
        self.assertEqual(r.status_code, 200, r.text)

    def test_base_antigua_o_ausente_rechaza_sin_perfilado_ni_migracion(self):
        vieja = self.ruta.with_name("legado_ocho.sqlite3")
        with closing(conectar(vieja)) as c:
            c.executescript(SCHEMA_PATH.read_text(encoding="utf-8").split("-- Identidad compartida")[0] + "COMMIT;")
            with closing(conectar(self.ruta)) as origen:
                for tabla in fixture.TABLAS:
                    filas = origen.execute(f"SELECT * FROM {tabla}").fetchall()
                    if filas:
                        c.executemany(f"INSERT INTO {tabla} VALUES ({','.join('?' for _ in filas[0])})", filas)
            c.commit()
        bytes_antes = vieja.read_bytes()
        for base in (vieja, self.ruta.with_name("inexistente.sqlite3")):
            with self.subTest(base=base.name), patch.object(carga, "perfilar_csv_global") as g1, \
                    patch.object(carga, "_abrir_escritura") as escritura:
                self.cliente.app.dependency_overrides[api.obtener_ruta_base] = lambda: base
                self.assertEqual(self.subir().status_code, 503)
                g1.assert_not_called()
                escritura.assert_not_called()
        self.assertEqual(vieja.read_bytes(), bytes_antes)
        self.assertFalse(self.ruta.with_name("inexistente.sqlite3").exists())
        with closing(conectar(vieja)) as c:
            self.assertEqual(len(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()), 8)

    def test_revalidacion_del_esquema_bajo_transaccion_detecta_cambio_durante_g1(self):
        antes = self.estado(tablas=fixture.TABLAS)
        real = carga.perfilar_csv_global
        def cambiar(ruta):
            r = real(ruta)
            self.datos.sql("ALTER TABLE perfil_global_conjunto RENAME TO perfil_resguardado")
            return r
        with patch.object(carga, "perfilar_csv_global", side_effect=cambiar):
            self.assertEqual(self.subir().status_code, 503)
        self.assertEqual(self.estado(tablas=fixture.TABLAS), antes)
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM perfil_resguardado"), [(1,)])
        self.assertEqual(self.datos.sql("SELECT name FROM sqlite_master WHERE name='perfil_global_conjunto'"), [])

    def test_base_que_desaparece_durante_g1_no_se_recrea(self):
        real = carga.perfilar_csv_global
        respaldo = self.ruta.with_name("resguardo.sqlite3")
        def mover(ruta):
            r = real(ruta)
            assert self.ruta.parent.name.startswith("hu004_g4_")
            self.ruta.rename(respaldo)  # solo archivo sintético conocido, mismo directorio temporal
            return r
        try:
            with patch.object(carga, "perfilar_csv_global", side_effect=mover):
                self.assertEqual(self.subir().status_code, 503)
            self.assertFalse(self.ruta.exists())
            self.assertTrue(respaldo.is_file())
        finally:
            if respaldo.exists():
                respaldo.rename(self.ruta)

    def test_fuente_intacta_y_misma_copia_privada_para_validacion_y_g1(self):
        rutas = []
        validar, g1 = carga.validar_csv, carga.perfilar_csv_global
        def observar_validar(ruta):
            rutas.append(ruta)
            return validar(ruta)
        def observar_g1(ruta):
            rutas.append(ruta)
            return g1(ruta)
        with patch.object(carga, "validar_csv", side_effect=observar_validar), \
                patch.object(carga, "perfilar_csv_global", side_effect=observar_g1):
            resultado = carga.cargar_csv(self.datos.csv, self.ruta)
        self.assertTrue(resultado.exito, resultado.errores)
        self.assertEqual(rutas[0], rutas[1])
        self.assertNotEqual(rutas[0], self.datos.csv)
        self.assertFalse(rutas[0].exists())
        self.assertEqual(self.datos.csv.read_bytes(), self.contenido)

    def test_copias_temporales_se_eliminan_si_g1_falla(self):
        rutas = []
        def fallar(ruta):
            rutas.append(ruta)
            raise ErrorPerfilCSV("PRIVADO")
        with patch.object(carga, "perfilar_csv_global", side_effect=fallar):
            self.assertEqual(self.subir().status_code, 422)
        self.assertEqual(len(rutas), 1)
        self.assertFalse(rutas[0].exists())

    def test_importaciones_identicas_comparten_hash_pero_no_identidad(self):
        resultados = [self.subir().json(), self.subir().json()]
        ids = {self.conjunto, *(r["conjunto_id"] for r in resultados)}
        self.assertEqual(len(ids), 3)
        perfiles = [self.perfil(i) for i in ids]
        self.assertEqual({p["sha256_csv"] for p in perfiles}, {hashlib.sha256(self.contenido).hexdigest()})
        self.assertTrue(all(p["observaciones_importadas"] == 80 for p in perfiles))
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM perfil_global_conjunto"), [(3,)])

    def test_cargas_concurrentes_sin_mezclar_reportes_ni_suponer_orden(self):
        barrera = Barrier(2)
        real = carga.perfilar_csv_global
        distinto = self.contenido.replace(b",0.5,1", b",0.75,1")
        def sincronizar(ruta):
            r = real(ruta)
            barrera.wait(timeout=20)
            return r
        with patch.object(carga, "perfilar_csv_global", side_effect=sincronizar):
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(self.subir, self.contenido, "A.csv")
                b = pool.submit(self.subir, distinto, "B.csv")
                respuestas = [a.result(timeout=30), b.result(timeout=30)]
        self.assertEqual([r.status_code for r in respuestas], [200, 200])
        ids = {r.json()["conjunto_id"] for r in respuestas}
        self.assertEqual(len(ids), 2)
        for respuesta, nombre, datos, total in zip(respuestas, ("A.csv", "B.csv"), (self.contenido, distinto), (1480, 1482.5)):
            p = self.perfil(respuesta.json()["conjunto_id"])
            self.assertEqual(p["sha256_csv"], hashlib.sha256(datos).hexdigest())
            self.assertEqual(p["reporte"]["fuente"]["nombre_archivo"], nombre)
            self.assertEqual(p["reporte"]["ventas"]["suma"], total)
        activos = self.datos.sql("SELECT id FROM conjunto_datos WHERE activo=1")
        self.assertEqual(len(activos), 1)
        self.assertIn(activos[0][0], ids)
        self.assertEqual(self.datos.sql("PRAGMA foreign_key_check"), [])

    def test_carga_no_entrena_no_genera_pronosticos_y_reporte_es_retrospectivo(self):
        with patch.object(fixture.modelos, "entrenar_modelo", side_effect=AssertionError("entrena")), \
                patch.object(fixture.registro, "_calcular", side_effect=AssertionError("genera pronóstico")):
            r = self.subir()
        self.assertEqual(r.status_code, 200, r.text)
        p = self.perfil(r.json()["conjunto_id"])
        self.assertFalse(p["reporte"]["interpretacion_temporal"]["autorizado_para_seleccion_de_modelos"])
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM pronostico"), [(0,)])
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM venta_diaria WHERE fecha>='2017-08-01'"), [(0,)])
        for corte in ("2017-08-01", "2017-08-15"):
            self.assertEqual(self.datos.perfil_http(corte).status_code, 422)


if __name__ == "__main__":
    unittest.main()
