"""G2: una serie por corte, únicamente SQLite temporal y datos sintéticos."""

from contextlib import closing
from datetime import date, timedelta
import importlib.util
import json
from math import sqrt
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import perfil_series as perfil
from backend.app.services.consulta_pronosticos import consultar_pronostico
from backend.app.services.registro_pronosticos import registrar_pronostico
from backend.app.storage.db import conectar, inicializar_base


CORTE = "2017-06-16"
FILAS = [("2016-12-24", 0.0, 0), ("2016-12-26", 0.5, 1),
         ("2017-01-02", 1.5, 0), ("2017-06-12", 2.5, 2),
         ("2017-06-13", 3.5, 0), ("2017-06-15", 20.0, 3)]


class PerfilSeriesTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory(prefix="perfil_g2_")
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "sintetica.sqlite3"
        inicializar_base(self.ruta)
        with closing(conectar(self.ruta)) as c, c:
            c.executemany("INSERT INTO conjunto_datos VALUES (?,?,?,?,?)",
                          [(1, "anterior.csv", "privado1", "2026-01-01", 0),
                           (2, "actual.csv", "privado2", "2026-01-02", 1)])
            c.executemany("INSERT INTO serie_temporal VALUES (?,?,?,?,?)",
                          [(11, 1, 17, "PREPARED FOODS", "sintética"),
                           (21, 2, 17, "PREPARED FOODS", "sintética"),
                           (22, 2, 8, "BREAD/BAKERY", "sintética")])
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                          [(21, *fila) for fila in reversed(FILAS)]
                          + [(21, "2017-06-17", 1e9, 900), (21, "2017-07-02", 2e9, 901),
                             (11, CORTE, 999, 99), (22, CORTE, 777, 77)])

    def sql(self, texto, parametros=(), *, corrupta=False):
        with closing(conectar(self.ruta)) as c, c:
            if corrupta:
                c.execute("PRAGMA ignore_check_constraints=ON")  # solo esta fixture temporal
            return c.execute(texto, parametros).fetchall()

    def estado(self):
        with closing(conectar(self.ruta)) as c:
            return tuple(tuple(c.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()) for t in
                         ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento", "pronostico", "detalle_pronostico"))

    def analizar(self, serie=21, corte=CORTE):
        return perfil.perfilar_serie(self.ruta, serie, corte)

    def error(self, codigo, llamada=None):
        with self.assertRaises(perfil.ErrorPerfilSerie) as capturada:
            (llamada or self.analizar)()
        self.assertEqual(capturada.exception.codigo, codigo)
        mensaje = str(capturada.exception)
        for privado in (str(self.ruta), "SELECT", "privado1", "privado2", "Traceback"):
            self.assertNotIn(privado, mensaje)

    def reemplazar(self, filas):
        self.sql("DELETE FROM venta_diaria WHERE serie_id=21")
        with closing(conectar(self.ruta)) as c, c:
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (21,?,?,?)", filas)

    def test_serie_activa_contrato_publico_json_y_sin_rutas(self):
        r = self.analizar()
        self.assertEqual(set(r), {"alcance", "conjunto_id", "serie", "fecha_corte", "cobertura", "ventas",
                                  "estacionalidad", "promociones", "valores_extremos", "advertencias"})
        self.assertEqual(r["alcance"], "perfil_descriptivo_serie_hasta_corte")
        self.assertEqual((r["conjunto_id"], r["serie"], r["fecha_corte"]),
                         (2, {"id": 21, "store_nbr": 17, "family": "PREPARED FOODS"}, CORTE))
        texto = json.dumps(r, allow_nan=False)
        self.assertEqual(json.loads(texto), r)
        for privado in (str(self.ruta), "hash", "privado2", "SELECT", "BREAD/BAKERY"):
            self.assertNotIn(privado, texto)

    def test_cobertura_desde_primera_observacion_hasta_corte_incluye_hueco_final(self):
        c = self.analizar()["cobertura"]
        self.assertEqual(c["periodo_utilizado"], {"inicio": "2016-12-24", "fin": CORTE, "ambos_inclusive": True})
        self.assertEqual((c["primera_fecha_observada"], c["ultima_fecha_observada"]), ("2016-12-24", "2017-06-15"))
        self.assertEqual((c["observaciones"], c["dias_naturales"], c["fechas_observadas"], c["fechas_sin_registro"]),
                         (6, 175, 6, 169))
        self.assertAlmostEqual(c["porcentaje_cobertura_calendario"], 24 / 7)
        self.assertIn(CORTE, c["fechas_ausentes"])
        self.assertNotIn("2016-12-23", c["fechas_ausentes"])
        self.assertEqual(c["fechas_ausentes"], sorted(c["fechas_ausentes"]))

    def test_24_y_26_diciembre_no_inventan_cero_el_25(self):
        r = self.analizar()
        self.assertIn("2016-12-25", r["cobertura"]["fechas_ausentes"])
        self.assertNotIn("2016-12-24", r["cobertura"]["fechas_ausentes"])
        self.assertNotIn("2016-12-26", r["cobertura"]["fechas_ausentes"])
        self.assertEqual(r["ventas"]["ventas_cero"], 1)
        self.assertEqual(r["ventas"]["promedio_por_observacion"], 28 / 6)
        self.assertEqual(self.sql("SELECT COUNT(*) FROM venta_diaria WHERE serie_id=21 AND fecha='2016-12-25'"), [(0,)])

    def test_suma_media_mediana_fracciones_y_ceros_conocidos(self):
        v = self.analizar()["ventas"]
        self.assertEqual((v["total"], v["mediana"], v["minimo"], v["maximo"]), (28.0, 2.0, 0.0, 20.0))
        self.assertAlmostEqual(v["promedio_por_observacion"], 14 / 3)
        self.assertEqual((v["ventas_cero"], v["ventas_positivas"], v["ventas_fraccionarias"]), (1, 5, 4))
        self.assertAlmostEqual(v["porcentaje_cero"], 100 / 6)
        self.assertAlmostEqual(v["porcentaje_fraccionarias"], 400 / 6)

    def test_percentiles_lineales_y_desviacion_muestral_conocidos(self):
        v = self.analizar()["ventas"]
        self.assertEqual((v["percentil_25"], v["percentil_75"], v["rango_intercuartilico"]), (0.75, 3.25, 2.5))
        # Para estos seis valores: varianza muestral exacta = 871/15.
        self.assertAlmostEqual(v["desviacion_estandar_muestral"], sqrt(871 / 15), delta=1e-12)
        self.assertIn("(n-1)*p", v["metodo_percentiles"])

    def test_unica_observacion_desviacion_null_y_grupos_vacios(self):
        self.reemplazar([(CORTE, 2.75, 0)])
        r = self.analizar()
        self.assertEqual((r["cobertura"]["dias_naturales"], r["cobertura"]["fechas_sin_registro"]), (1, 0))
        self.assertEqual(r["cobertura"]["porcentaje_cobertura_calendario"], 100.0)
        v = r["ventas"]
        self.assertIsNone(v["desviacion_estandar_muestral"])
        self.assertEqual((v["mediana"], v["percentil_25"], v["percentil_75"]), (2.75, 2.75, 2.75))
        self.assertEqual(r["promociones"]["ventas_con_promocion"],
                         {"observaciones": 0, "total_ventas": 0.0, "promedio_por_observacion": None, "ventas_cero": 0})
        self.assertEqual(r["valores_extremos"]["cantidad"], 0)
        json.dumps(r, allow_nan=False)

    def test_grupos_dia_semana_solo_observados_en_orden(self):
        self.assertEqual(self.analizar()["estacionalidad"]["por_dia_semana"], [
            {"dia_semana": 0, "observaciones": 3, "total_ventas": 4.5, "promedio_por_observacion": 1.5, "ventas_cero": 0},
            {"dia_semana": 1, "observaciones": 1, "total_ventas": 3.5, "promedio_por_observacion": 3.5, "ventas_cero": 0},
            {"dia_semana": 3, "observaciones": 1, "total_ventas": 20.0, "promedio_por_observacion": 20.0, "ventas_cero": 0},
            {"dia_semana": 5, "observaciones": 1, "total_ventas": 0.0, "promedio_por_observacion": 0.0, "ventas_cero": 1}])

    def test_grupos_mes_solo_observados_en_orden(self):
        self.assertEqual(self.analizar()["estacionalidad"]["por_mes"], [
            {"mes": 1, "observaciones": 1, "total_ventas": 1.5, "promedio_por_observacion": 1.5, "ventas_cero": 0},
            {"mes": 6, "observaciones": 3, "total_ventas": 26.0, "promedio_por_observacion": 26 / 3, "ventas_cero": 0},
            {"mes": 12, "observaciones": 2, "total_ventas": 0.5, "promedio_por_observacion": 0.25, "ventas_cero": 1}])

    def test_promociones_y_ventas_asociadas_sin_causalidad(self):
        p = self.analizar()["promociones"]
        self.assertEqual((p["observaciones_sin_promocion"], p["observaciones_con_promocion"],
                          p["porcentaje_promocionadas"], p["suma_conteos_onpromotion"]), (3, 3, 50.0, 6))
        self.assertEqual(p["ventas_con_promocion"],
                         {"observaciones": 3, "total_ventas": 23.0, "promedio_por_observacion": 23 / 3, "ventas_cero": 0})
        self.assertEqual(p["ventas_sin_promocion"]["total_ventas"], 5.0)

    def test_limites_iqr_y_ejemplo_descriptivo_no_modifica_extremos(self):
        anterior = self.estado()
        e = self.analizar()["valores_extremos"]
        self.assertEqual((e["limite_inferior"], e["limite_superior"], e["cantidad"]), (-3.0, 7.0, 1))
        self.assertEqual(e["ejemplos"], [{"fecha": "2017-06-15", "venta": 20.0}])
        self.assertEqual(self.estado(), anterior)

    def test_iqr_cero_y_maximo_diez_ejemplos_en_orden_cronologico(self):
        inicio = date(2017, 2, 1)
        filas = [((inicio + timedelta(days=i)).isoformat(), 0 if i < 100 else i - 99, 0) for i in range(112)]
        self.reemplazar(list(reversed(filas)))
        e = self.analizar()["valores_extremos"]
        self.assertEqual((e["limite_inferior"], e["limite_superior"], e["cantidad"]), (0.0, 0.0, 12))
        self.assertEqual(len(e["ejemplos"]), 10)
        self.assertEqual(e["ejemplos"][0], {"fecha": "2017-05-12", "venta": 1.0})
        self.assertEqual(e["ejemplos"][-1], {"fecha": "2017-05-21", "venta": 10.0})
        self.assertIn("IQR=0", e["interpretacion_iqr_cero"])

    def test_valor_igual_al_limite_iqr_no_se_cuenta_fuera(self):
        valores = (0, 1, 1, 1, 2, 2, 3.5)  # Q1=1, Q3=2; límite superior=3.5
        self.reemplazar([(f"2017-06-{i:02d}", v, 0) for i, v in enumerate(valores, 1)])
        e = self.analizar()["valores_extremos"]
        self.assertEqual((e["limite_inferior"], e["limite_superior"]), (-0.5, 3.5))
        self.assertEqual((e["cantidad"], e["ejemplos"]), (0, []))

    def test_serie_de_ceros_y_todas_promocionadas_sin_nan(self):
        self.reemplazar([("2017-06-14", 0, 1), ("2017-06-15", 0, 2), (CORTE, 0, 3)])
        r = self.analizar()
        self.assertEqual((r["ventas"]["total"], r["ventas"]["mediana"], r["ventas"]["desviacion_estandar_muestral"]), (0.0, 0.0, 0.0))
        self.assertEqual((r["ventas"]["porcentaje_cero"], r["ventas"]["ventas_positivas"]), (100.0, 0))
        self.assertEqual(r["valores_extremos"]["cantidad"], 0)
        self.assertIsNone(r["promociones"]["ventas_sin_promocion"]["promedio_por_observacion"])
        json.dumps(r, allow_nan=False)

    def test_identificador_invalido_se_rechaza_antes_de_abrir_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("Entrada inválida abre base")):
            for serie in (0, -1, True, 1.0, "21", None, 2**63):
                with self.subTest(serie=serie):
                    self.error("serie_invalida", lambda: self.analizar(serie=serie))

    def test_fechas_invalidas_no_canonicas_y_tipo_incorrecto(self):
        for corte in ("2017-02-30", "2017-6-16", "20170616", "2017-06-16T00:00:00", "", None, date(2017, 6, 16)):
            with self.subTest(corte=corte):
                self.error("corte_invalido", lambda: self.analizar(corte=corte))

    def test_solo_tres_cortes_y_reserva_final_bloqueada(self):
        for corte, observaciones, dias in ((CORTE, 6, 175), ("2017-07-01", 7, 190), ("2017-07-16", 8, 205)):
            with self.subTest(corte=corte):
                r = self.analizar(corte=corte)
                self.assertEqual(r["fecha_corte"], corte)
                self.assertEqual((r["cobertura"]["observaciones"], r["cobertura"]["dias_naturales"]), (observaciones, dias))
        with patch.object(sqlite3, "connect", side_effect=AssertionError("Corte inválido abre base")):
            for corte in ("2017-06-17", "2017-07-17", "2017-07-31", "2018-01-01"):
                self.error("corte_no_autorizado", lambda: self.analizar(corte=corte))
            for corte in ("2017-08-01", "2017-08-07", "2017-08-15"):
                self.error("prueba_reservada", lambda: self.analizar(corte=corte))

    def test_futuro_extremo_o_corrupto_no_altera_ningun_estadistico(self):
        anterior = self.analizar()
        self.assertEqual(anterior["ventas"]["maximo"], 20.0)
        self.sql("UPDATE venta_diaria SET sales=1e100,onpromotion=100000 WHERE serie_id=21 AND fecha>?", (CORTE,))
        self.assertEqual(self.analizar(), anterior)
        self.sql("UPDATE venta_diaria SET sales='FUTURO INVALIDO',onpromotion=1.5 WHERE serie_id=21 AND fecha>?", (CORTE,))
        self.assertEqual(self.analizar(), anterior)

    def test_no_incluye_otra_serie_o_conjunto_inactivo(self):
        self.sql("UPDATE venta_diaria SET sales='OTRA SERIE INVALIDA' WHERE serie_id<>21")
        self.assertEqual(self.analizar()["ventas"]["total"], 28.0)
        self.assertEqual(self.analizar()["promociones"]["suma_conteos_onpromotion"], 6)

    def test_serie_inexistente_e_inactiva_no_consultan_sus_ventas(self):
        for serie in (9999, 11):
            sentencias = []
            abrir = sqlite3.connect
            def observar(*args, **kwargs):
                c = abrir(*args, **kwargs)
                c.set_trace_callback(sentencias.append)
                return c
            with patch.object(sqlite3, "connect", side_effect=observar):
                self.error("serie_inexistente", lambda: self.analizar(serie=serie))
            self.assertFalse(any("FROM venta_diaria" in s for s in sentencias))

    def test_sin_conjunto_activo(self):
        self.sql("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_base_ausente_danada_sin_esquema_y_ruta_invalida(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        self.error("base_no_disponible", lambda: perfil.perfilar_serie(ausente, 21, CORTE))
        self.assertFalse(ausente.exists())
        danada = self.ruta.with_name("danada.sqlite3")
        danada.write_bytes(b"no es SQLite")
        vacia = self.ruta.with_name("vacia.sqlite3")
        vacia.touch()
        for ruta in (danada, vacia, None, "\0", Path("\0")):
            self.error("base_no_disponible", lambda: perfil.perfilar_serie(ruta, 21, CORTE))

    def test_historial_vacio_o_exclusivamente_futuro(self):
        self.reemplazar([])
        self.error("historial_insuficiente")
        self.reemplazar([("2017-06-17", 3, 1)])
        self.error("historial_insuficiente")

    def test_ventas_almacenadas_invalidas_rechazadas(self):
        for venta in (-0.1, float("inf"), "NaN", "texto"):
            with self.subTest(venta=venta):
                self.sql("UPDATE venta_diaria SET sales=? WHERE serie_id=21 AND fecha='2016-12-24'", (venta,))
                self.error("datos_inconsistentes")

    def test_promociones_almacenadas_invalidas_rechazadas(self):
        for promo in (-1, 1.5, "texto", float("inf")):
            with self.subTest(promocion=promo):
                self.sql("UPDATE venta_diaria SET onpromotion=? WHERE serie_id=21 AND fecha='2016-12-24'", (promo,), corrupta=True)
                self.error("datos_inconsistentes")

    def test_fechas_almacenadas_invalidas_rechazadas(self):
        for texto in ("2016-12-2", "2016-12-24T00:00:00", "2017-02-30"):
            with self.subTest(fecha=texto):
                self.reemplazar([(texto, 1, 0)])
                self.error("datos_inconsistentes")

    def test_metadatos_de_serie_inconsistentes_rechazados(self):
        self.sql("UPDATE serie_temporal SET family='' WHERE id=21", corrupta=True)
        self.error("datos_inconsistentes")
        self.sql("UPDATE serie_temporal SET family='A',store_nbr=0 WHERE id=21", corrupta=True)
        self.error("datos_inconsistentes")

    def test_fechas_duplicadas_bloqueadas_por_schema_y_por_validacion(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.sql("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (21,'2016-12-24',1,0)")
        # Simula un resultado corrupto que no puede producir el esquema sano.
        abrir = sqlite3.connect
        class ConexionDuplicada(sqlite3.Connection):
            def execute(self, sql, parametros=()):
                cursor = super().execute(sql, parametros)
                if sql.startswith("SELECT fecha, sales, onpromotion"):
                    filas = cursor.fetchall()
                    class Duplicadas:
                        def fetchall(self):
                            return [filas[0], *filas]
                    return Duplicadas()
                return cursor
        with patch.object(sqlite3, "connect", side_effect=lambda *a, **k: abrir(*a, **k, factory=ConexionDuplicada)):
            self.error("datos_inconsistentes")

    def test_limite_de_observaciones_aplicado_en_consulta(self):
        sentencias = []
        abrir = sqlite3.connect
        def observar(*args, **kwargs):
            c = abrir(*args, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c
        with patch.object(perfil, "MAX_OBSERVACIONES", 5), patch.object(sqlite3, "connect", side_effect=observar):
            self.error("limite_observaciones")
        self.assertTrue(any("LIMIT 6" in s for s in sentencias))

    def test_calendario_excesivo_rechazado_antes_de_enumerar_ausencias(self):
        with patch.object(perfil, "MAX_DIAS_CALENDARIO", 174):
            self.error("periodo_excesivo")

    def test_sumatoria_y_limites_desbordados_rechazados_sin_infinity(self):
        for valores in ((1e308, 1e308), (0.0, 1.7e308)):
            with self.subTest(valores=valores):
                self.reemplazar([("2017-06-15", valores[0], 0), (CORTE, valores[1], 0)])
                self.error("resultado_no_finito")

    def test_lectura_ro_begin_sin_dml_entrenamiento_ni_cambios_fisicos(self):
        anterior, contenido = self.estado(), self.ruta.read_bytes()
        abrir = sqlite3.connect
        rutas, sentencias = [], []
        def observar(ruta, *args, **kwargs):
            rutas.append((str(ruta), kwargs.get("uri")))
            c = abrir(ruta, *args, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c
        with patch.object(sqlite3, "connect", side_effect=observar), \
                patch("scripts.modelos_prediccion.entrenar_modelo", side_effect=AssertionError("Perfil entrena")), \
                patch("backend.app.services.registro_pronosticos.registrar_pronostico", side_effect=AssertionError("Perfil registra")):
            self.analizar()
        self.assertEqual(len(rutas), 1)
        self.assertTrue("mode=ro" in rutas[0][0] and rutas[0][1])
        self.assertEqual(sentencias[0], "BEGIN")
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE")) for s in sentencias))
        self.assertTrue(any("WHERE serie_id=21 AND fecha<='2017-06-16'" in s for s in sentencias))
        self.assertEqual(self.estado(), anterior)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_wal_consulta_sin_cambios_de_contenido_logico(self):
        self.sql("PRAGMA journal_mode=WAL")
        anterior = self.estado()
        self.analizar()
        self.assertEqual(self.estado(), anterior)

    def test_instantanea_consistente_y_cambio_del_activo_entre_consultas(self):
        self.sql("PRAGMA journal_mode=WAL")
        activo = perfil._activo
        def intercalar(c):
            conjunto = activo(c)
            with closing(conectar(self.ruta)) as escritor, escritor:
                escritor.execute("UPDATE conjunto_datos SET activo=0 WHERE id=2")
                escritor.execute("UPDATE conjunto_datos SET activo=1 WHERE id=1")
                escritor.execute("UPDATE venta_diaria SET sales=500 WHERE serie_id=21 AND fecha='2016-12-24'")
            return conjunto
        with patch.object(perfil, "_activo", side_effect=intercalar):
            r = self.analizar()
        self.assertEqual((r["conjunto_id"], r["ventas"]["total"]), (2, 28.0))
        self.error("serie_inexistente")
        siguiente = self.analizar(serie=11)
        self.assertEqual((siguiente["conjunto_id"], siguiente["ventas"]["total"]), (1, 999.0))

    def test_importacion_sin_abrir_base(self):
        nombre = "perfil_g2_importacion_aislada"
        spec = importlib.util.spec_from_file_location(nombre, Path(perfil.__file__))
        modulo = importlib.util.module_from_spec(spec)
        with patch.object(sqlite3, "connect", side_effect=AssertionError("Importación abre base")):
            spec.loader.exec_module(modulo)
        self.assertTrue(callable(modulo.perfilar_serie))

    def test_coexiste_con_pronostico_y_no_cambia_contratos_http(self):
        registrado = registrar_pronostico(self.ruta, 21, "estacional_semanal_recursiva", CORTE)
        anteriores = app.dependency_overrides.copy()
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        urls = ["/health", "/openapi.json", "/api/datasets/active/series", "/api/datasets/active/series/21/sales",
                "/api/pronosticos/active/series/21", f"/api/pronosticos/active/{registrado.pronostico_id}",
                f"/api/pronosticos/active/{registrado.pronostico_id}/configuracion"]
        with TestClient(app) as cliente:
            antes = [cliente.get(url).json() for url in urls]
            estado = self.estado()
            r = self.analizar()
            self.assertEqual(r["ventas"]["total"], 28.0)
            self.assertEqual([cliente.get(url).json() for url in urls], antes)
        self.assertEqual(self.estado(), estado)
        self.assertEqual(len(consultar_pronostico(self.ruta, registrado.pronostico_id)["detalles"]), 15)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])


if __name__ == "__main__":
    unittest.main()
