"""G4: procedencia CSV -> carga -> perfil hasta corte -> consultas HTTP.

Solo CSV y SQLite temporales. Las cifras esperadas son constantes de una fixture
de ocho fechas: 88 filas globales/1.480 ventas frente a 80/680 persistidas.
No se generan observaciones de agosto. Los dobles vigilan operaciones o fuerzan
un cambio concurrente determinista; el perfil, carga y modelos son reales.
La revisión de React y del plan maestro se informa aparte como inspección,
no como pruebas visuales. Ejecutar: python -m unittest tests.test_integridad_exploracion -v
"""

from contextlib import closing
from datetime import date, timedelta
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import configuracion_experimentos as configuracion
from backend.app.services import registro_pronosticos as registro
from backend.app.services.perfil_series import perfilar_serie
from backend.app.storage.db import conectar, inicializar_base
from scripts import modelos_prediccion as modelos
from scripts import perfil_global_csv as global_csv
from scripts.analizar_periodo_seleccion import (
    calcular_caracteristicas, contar_prefijo_fechas, leer_bloques_hasta_corte,
)
from scripts.preparar_series_modelado import cargar_series_csv, historico_hasta
from scripts.seleccionar_series import preparar_candidatos
from scripts.variables_predictoras import COLUMNAS_PREDICTORAS, construir_tabla_serie


RAIZ = Path(__file__).resolve().parents[1]
CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")
FILAS = (
    ("2016-12-24", 0, 0), ("2016-12-26", 0.5, 1),
    ("2017-06-16", 1.5, 0), ("2017-06-17", 10, 2),
    ("2017-07-01", 2.5, 0), ("2017-07-02", 20, 3),
    ("2017-07-16", 3.5, 0), ("2017-07-17", 30, 4),
)
TABLAS = ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento",
          "pronostico", "detalle_pronostico", "evaluacion", "simulacion")


class IntegridadExploracionTests(unittest.TestCase):
    def setUp(self):
        temporal = self.enterContext(tempfile.TemporaryDirectory(prefix="hu004_g4_"))
        self.ruta = Path(temporal) / "sintetica.sqlite3"
        self.csv = Path(temporal) / "exploracion_sintetica.csv"
        self.config = (RAIZ / "config/series_seleccionadas.json").read_bytes()
        self.series = json.loads(self.config)["series"]
        salida = io.StringIO(newline="")
        escritor = csv.writer(salida)
        escritor.writerow(("date", "store_nbr", "family", "sales", "onpromotion"))
        for fecha, venta, promo in FILAS:
            for serie in self.series:
                escritor.writerow((fecha, serie["store_nbr"], serie["family"], venta, promo))
            escritor.writerow((fecha, 999, "NO HABILITADA G4", 100, 5))
        self.contenido = salida.getvalue().encode("utf-8")
        self.csv.write_bytes(self.contenido)
        inicializar_base(self.ruta)
        anteriores = app.dependency_overrides.copy()
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))
        self.carga = self.subir()
        self.conjunto = self.carga["conjunto_id"]
        self.serie = next(s for s in self.activas()["series"] if s["store_nbr"] == 17)

    def subir(self):
        r = self.cliente.post("/api/datasets/upload", files={
            "file": ("exploracion_sintetica.csv", self.contenido, "text/csv")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["exito"])
        return r.json()

    def activas(self):
        r = self.cliente.get("/api/datasets/active/series")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def sql(self, consulta, valores=()):
        with closing(conectar(self.ruta)) as c, c:
            return c.execute(consulta, valores).fetchall()

    def estado(self):
        # Contenido lógico, válido también con WAL; no depende solo de bytes físicos.
        with closing(sqlite3.connect(self.ruta.as_uri() + "?mode=ro", uri=True)) as c:
            return tuple(tuple(c.execute(f"SELECT * FROM {t} ORDER BY 1,2")) for t in TABLAS)

    def perfil_http(self, corte=CORTES[0], serie=None):
        return self.cliente.get(f"/api/datasets/active/series/{serie or self.serie['id']}/perfil",
                                params={"fecha_corte": corte})

    def crear_base(self):
        r = self.cliente.post(f"/api/pronosticos/active/series/{self.serie['id']}",
                              json={"tecnica": configuracion.BASE, "fecha_corte": CORTES[0]})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def test_panorama_global_completo_con_cifras_independientes(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("G1 abre SQLite")), \
                patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("G1 entrena")):
            g = global_csv.perfilar_csv_global(self.csv, tamano_bloque=3)
        self.assertEqual(g["volumen"], {"filas_examinadas": 88, "filas_validas": 88,
                                      "filas_invalidas": 0, "tiendas_distintas": 9,
                                      "familias_distintas": 11, "combinaciones_observadas": 11})
        t = g["temporalidad"]
        self.assertEqual((t["primera_fecha"], t["ultima_fecha"], t["fechas_distintas"],
                          t["dias_calendario_inclusivos"]), ("2016-12-24", "2017-07-17", 8, 206))
        self.assertEqual(len(t["fechas_sin_registro_global"]), 198)
        self.assertIn("2016-12-25", t["fechas_sin_registro_global"])
        self.assertEqual(t["filas_por_anio"], [{"anio": 2016, "filas": 22}, {"anio": 2017, "filas": 66}])
        self.assertEqual((g["ventas"]["suma"], g["ventas"]["registros_cero"],
                          g["ventas"]["registros_fraccionarios"]), (1480, 10, 40))
        self.assertEqual((g["promociones"]["registros_positivos"], g["promociones"]["registros_cero"],
                          g["promociones"]["suma_conteos_onpromotion"]), (48, 40, 140))
        self.assertEqual(g["alcance"], "panorama_descriptivo_global")
        self.assertTrue(g["interpretacion_temporal"]["retrospectivo"])
        self.assertTrue(g["interpretacion_temporal"]["incluye_todas_las_fechas_del_archivo"])
        self.assertFalse(g["interpretacion_temporal"]["autorizado_para_seleccion_de_modelos"])
        json.dumps(g, allow_nan=False)
        self.assertEqual(self.csv.read_bytes(), self.contenido)

    def test_carga_real_solo_diez_series_y_procedencia_de_cada_fila(self):
        self.assertEqual((self.carga["series_importadas"], self.carga["observaciones_importadas"]), (10, 80))
        self.assertEqual(self.carga["advertencias"], [])
        claves = {(s["store_nbr"], s["family"]) for s in self.series}
        self.assertEqual({(s["store_nbr"], s["family"]) for s in self.activas()["series"]}, claves)
        esperadas = sorted((t, f, fecha, venta, promo) for t, f in claves for fecha, venta, promo in FILAS)
        guardadas = self.sql("SELECT s.store_nbr,s.family,v.fecha,v.sales,v.onpromotion FROM venta_diaria v "
                             "JOIN serie_temporal s ON s.id=v.serie_id ORDER BY 1,2,3")
        self.assertEqual(guardadas, esperadas)
        self.assertEqual(self.sql("SELECT SUM(sales),SUM(onpromotion) FROM venta_diaria"), [(680, 100)])
        self.assertEqual(self.sql("SELECT COUNT(*) FROM serie_temporal WHERE store_nbr=999"), [(0,)])
        self.assertEqual(self.sql("SELECT COUNT(*) FROM venta_diaria WHERE fecha>'2017-06-16'"), [(50,)])
        self.assertEqual(self.sql("SELECT COUNT(*) FROM experimento"), [(0,)])
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])
        self.assertEqual((RAIZ / "config/series_seleccionadas.json").read_bytes(), self.config)

    def test_tres_cortes_g2_http_y_calendario_con_resultados_conocidos(self):
        casos = ((CORTES[0], 3, 2, 175, 172, 0.25, 1.0),
                 (CORTES[1], 5, 14.5, 190, 185, 0.5, 2.5),
                 (CORTES[2], 7, 38, 205, 198, 1.0, 6.75))
        for corte, n, total, dias, ausentes, q1, q3 in casos:
            with self.subTest(corte=corte):
                p = perfilar_serie(self.ruta, self.serie["id"], corte)
                respuesta = self.perfil_http(corte)
                self.assertEqual(respuesta.status_code, 200, respuesta.text)
                self.assertEqual(respuesta.json(), p)
                self.assertEqual((p["conjunto_id"], p["serie"]), (self.conjunto, self.serie))
                c, v = p["cobertura"], p["ventas"]
                self.assertEqual((c["observaciones"], c["dias_naturales"], c["fechas_sin_registro"]), (n, dias, ausentes))
                self.assertEqual(c["periodo_utilizado"], {"inicio": "2016-12-24", "fin": corte, "ambos_inclusive": True})
                self.assertEqual(c["ultima_fecha_observada"], corte)
                self.assertIn("2016-12-25", c["fechas_ausentes"])
                self.assertEqual((v["total"], v["ventas_cero"], v["percentil_25"], v["percentil_75"]), (total, 1, q1, q3))
                self.assertAlmostEqual(v["promedio_por_observacion"], total / n)
                self.assertEqual(p["valores_extremos"]["limite_inferior"], q1 - 1.5 * (q3 - q1))
                self.assertEqual(p["valores_extremos"]["limite_superior"], q3 + 1.5 * (q3 - q1))

    def test_mutaciones_y_extremos_futuros_no_cambian_ninguna_seccion(self):
        for corte in CORTES:
            with self.subTest(corte=corte):
                antes = self.perfil_http(corte).json()
                originales = self.sql("SELECT sales,onpromotion,id FROM venta_diaria WHERE serie_id=? AND fecha>?",
                                      (self.serie["id"], corte))
                nueva = (date.fromisoformat(corte) + timedelta(days=2)).isoformat()
                try:
                    self.sql("UPDATE venta_diaria SET sales=1000000000 WHERE serie_id=? AND fecha>?",
                             (self.serie["id"], corte))
                    self.assertEqual(self.perfil_http(corte).json(), antes)
                    self.sql("UPDATE venta_diaria SET onpromotion=999999 WHERE serie_id=? AND fecha>?",
                             (self.serie["id"], corte))
                    self.assertEqual(self.perfil_http(corte).json(), antes)
                    self.sql("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,1e100,999999)",
                             (self.serie["id"], nueva))
                    self.assertEqual(self.perfil_http(corte).json(), antes)
                    self.assertEqual(perfilar_serie(self.ruta, self.serie["id"], corte), antes)
                finally:
                    self.sql("DELETE FROM venta_diaria WHERE serie_id=? AND fecha=?", (self.serie["id"], nueva))
                    with closing(conectar(self.ruta)) as c, c:
                        c.executemany("UPDATE venta_diaria SET sales=?,onpromotion=? WHERE id=?", originales)

    def test_seleccion_temporal_y_preparacion_no_confunden_ambitos(self):
        prefijo, total = contar_prefijo_fechas(self.csv, tamano_bloque=3)
        self.assertEqual((prefijo, total), (33, 88))
        caracteristicas = calcular_caracteristicas(leer_bloques_hasta_corte(self.csv, prefijo, tamano_bloque=2))
        candidatos = preparar_candidatos(caracteristicas.series)
        self.assertEqual(caracteristicas.filas_seleccion, 33)
        self.assertEqual(len(candidatos), 11)  # candidatos descriptivos != lista oficial importada
        oficial = candidatos.loc[candidatos["store_nbr"].astype(int).eq(17)].iloc[0]
        self.assertEqual((oficial["observaciones"], oficial["volumen_total"]), (3, 2))
        claves = tuple((s["store_nbr"], s["family"]) for s in self.series)
        preparado = cargar_series_csv(self.csv, claves, tamano_bloque=3)
        limitado = historico_hasta(preparado, date(2017, 6, 16))
        self.assertEqual((len(preparado), len(limitado)), (80, 30))
        self.assertEqual(limitado["sales"].sum(), 20)
        self.assertFalse(preparado["date"].eq(pd.Timestamp("2016-12-25")).any())
        self.assertEqual(int(limitado["sales"].eq(0).sum()), 10)

    def test_variables_y_estimador_real_independientes_de_ventas_y_promociones_futuras(self):
        filas = [{"date": (date(2017, 5, 1) + timedelta(days=i)).isoformat(),
                  "store_nbr": 17, "family": "PREPARED FOODS", "sales": 0.25 + i % 7, "onpromotion": i % 3}
                 for i in range(47) if i != 19]
        filas += [{"date": "2017-06-17", "store_nbr": 17, "family": "PREPARED FOODS", "sales": 10, "onpromotion": 2}]
        datos = pd.DataFrame(filas)
        alterados = datos.copy(deep=True)
        alterados.loc[alterados["date"].gt(CORTES[0]), ["sales", "onpromotion"]] = [1e100, 999999]
        original = datos.copy(deep=True)
        tablas = [construir_tabla_serie(d, 17, "PREPARED FOODS", date(2017, 6, 16)).tabla for d in (datos, alterados)]
        pd.testing.assert_frame_equal(*tablas)
        self.assertNotIn("onpromotion", COLUMNAS_PREDICTORAS)
        self.assertEqual(COLUMNAS_PREDICTORAS, ("lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes"))
        self.assertTrue(all(f <= date(2017, 6, 16) for f in tablas[0]["fecha_objetivo"]))
        self.assertNotIn(date(2017, 5, 20), tablas[0]["fecha_objetivo"].tolist())
        resultados = []
        for d in (datos, alterados):
            m = modelos.entrenar_modelo(d, 17, "PREPARED FOODS", "random_forest", date(2017, 6, 16),
                                       configuracion_solicitada={"tecnica": "random_forest", "parametros": {"n_estimators": 5}})
            self.assertEqual(m.estimador.get_params(deep=False)["n_estimators"], 5)
            resultados.append([p.venta_estimada for p in modelos.pronosticar_recursivo(m, d)])
        self.assertEqual(resultados[0], resultados[1])
        self.assertEqual(len(resultados[0]), 15)
        pd.testing.assert_frame_equal(datos, original)

    def test_perfiles_no_disparan_entrenamiento_registro_ni_reconfiguracion(self):
        antes = self.estado()
        identidad = configuracion.identidad_configuracion(configuracion.configuracion_predeterminada("random_forest"))
        with patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("perfil entrena")), \
                patch.object(modelos, "crear_estimador", side_effect=AssertionError("perfil crea estimador")), \
                patch.object(registro, "_calcular", side_effect=AssertionError("perfil genera")):
            global_csv.perfilar_csv_global(self.csv, 2)
            for corte in CORTES:
                self.assertEqual(self.perfil_http(corte).status_code, 200)
        self.assertEqual(self.estado(), antes)
        self.assertEqual(configuracion.identidad_configuracion(configuracion.configuracion_predeterminada("random_forest")), identidad)

    def test_get_ro_begin_sin_dml_conserva_pronosticos_en_wal(self):
        self.crear_base()
        self.assertEqual(self.sql("PRAGMA journal_mode=WAL"), [("wal",)])
        antes = self.estado()
        conectar_real = sqlite3.connect
        trazas, aperturas, escrituras = [], [], []

        def abrir(destino, *args, **kwargs):
            self.assertIn("mode=ro", destino)
            self.assertTrue(kwargs.get("uri"))
            aperturas.append(destino)
            c = conectar_real(destino, *args, **kwargs)
            c.set_trace_callback(trazas.append)

            def autorizar(accion, tabla, *_):
                if accion in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
                    escrituras.append((accion, tabla))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            c.set_authorizer(autorizar)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir), \
                patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("GET entrena")):
            for corte in CORTES:
                self.assertEqual(self.perfil_http(corte).status_code, 200)
        self.assertEqual(len(aperturas), 3)
        self.assertEqual(sum(s.strip().upper() == "BEGIN" for s in trazas), 3)
        self.assertEqual(escrituras, [])
        self.assertEqual(self.estado(), antes)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_instantanea_wal_consistente_durante_cambio_real_del_activo(self):
        segundo = self.subir()["conjunto_id"]
        nueva = next(s for s in self.activas()["series"] if s["store_nbr"] == 17)
        esperado = perfilar_serie(self.ruta, nueva["id"], CORTES[0])
        self.assertEqual(self.sql("PRAGMA journal_mode=WAL"), [("wal",)])
        real = sqlite3.connect
        cambios = []

        def abrir(*args, **kwargs):
            c = real(*args, **kwargs)

            def trazar(sentencia):
                if sentencia.startswith("SELECT fecha, sales, onpromotion") and not cambios:
                    # Fixture: escritura por una conexión distinta DESPUÉS de leer activo y serie.
                    with closing(real(self.ruta)) as escritor, escritor:
                        escritor.execute("UPDATE conjunto_datos SET activo=(id=?)", (self.conjunto,))
                        escritor.execute("UPDATE venta_diaria SET sales=9999 WHERE serie_id=? AND fecha='2016-12-24'", (nueva["id"],))
                    cambios.append(segundo)
            c.set_trace_callback(trazar)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir):
            observado = perfilar_serie(self.ruta, nueva["id"], CORTES[0])
        self.assertEqual(cambios, [segundo])
        self.assertEqual(observado, esperado)  # ni activo ni ventas se mezclan entre instantáneas
        self.assertEqual(self.perfil_http(serie=nueva["id"]).status_code, 404)
        self.assertEqual(self.perfil_http().status_code, 200)

    def test_cambio_activo_excluye_perfiles_historicos_y_pronosticos_antiguos(self):
        p = self.crear_base()
        previo = self.estado()
        nuevo = self.subir()["conjunto_id"]
        self.assertNotEqual(nuevo, self.conjunto)
        self.assertEqual(self.perfil_http().status_code, 404)
        r = self.cliente.get(f"/api/datasets/active/series/{self.serie['id']}/sales")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.cliente.get(f"/api/pronosticos/active/{p['pronostico_id']}").status_code, 404)
        actual = self.estado()
        self.assertEqual(actual[3:], previo[3:])  # experimento y resultados anteriores intactos
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_base_ausente_no_se_crea_y_prueba_final_sigue_bloqueada(self):
        ausente = self.ruta.with_name("inexistente.sqlite3")
        app.dependency_overrides[obtener_ruta_base] = lambda: ausente
        self.assertEqual(self.perfil_http().status_code, 503)
        self.assertFalse(ausente.exists())
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        antes = self.estado()
        with patch.object(sqlite3, "connect", side_effect=AssertionError("corte prohibido abre base")):
            for corte in ("2017-07-31", "2017-08-01", "2017-08-15"):
                self.assertEqual(self.perfil_http(corte).status_code, 422)
                r = self.cliente.post(f"/api/pronosticos/active/series/{self.serie['id']}",
                                     json={"tecnica": configuracion.BASE, "fecha_corte": corte})
                self.assertEqual(r.status_code, 422)
        self.assertEqual(self.estado(), antes)
        self.assertEqual(self.sql("SELECT COUNT(*) FROM venta_diaria WHERE fecha>='2017-08-01'"), [(0,)])

    def test_contratos_http_anteriores_iguales_antes_y_despues_del_perfil(self):
        p = self.crear_base()
        rutas = ("/health", "/api/datasets/active/series",
                 f"/api/datasets/active/series/{self.serie['id']}/sales",
                 f"/api/pronosticos/active/series/{self.serie['id']}",
                 f"/api/pronosticos/active/{p['pronostico_id']}",
                 f"/api/pronosticos/active/{p['pronostico_id']}/configuracion")
        anteriores = [(r.status_code, r.json()) for r in (self.cliente.get(u) for u in rutas)]
        self.assertTrue(all(estado == 200 for estado, _ in anteriores))
        antes = self.estado()
        global_csv.perfilar_csv_global(self.csv, 3)
        for corte in CORTES:
            self.assertEqual(self.perfil_http(corte).status_code, 200)
        posteriores = [(r.status_code, r.json()) for r in (self.cliente.get(u) for u in rutas)]
        self.assertEqual(posteriores, anteriores)
        self.assertEqual(self.estado(), antes)
        self.assertEqual(self.sql("SELECT COUNT(*) FROM detalle_pronostico"), [(15,)])


if __name__ == "__main__":
    unittest.main()
