"""G3-A: HTTP y servicio G2 real con la fixture SQLite sintética de G2.

Los únicos dobles provocan errores o vigilan operaciones. Los pronósticos de
compatibilidad se siembran como registros SINTÉTICOS, sin entrenar ni calcularlos.
"""

from contextlib import closing
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import datasets as api
from backend.app.main import app
from backend.app.services import configuracion_experimentos as contrato
from backend.app.services import perfil_series as perfil
from backend.app.storage.db import conectar
from tests import test_perfil_series as fixtures


CORTE = "2017-06-16"
PREFIJO = "/api/datasets/active/series"
RUTA_OPENAPI = PREFIJO + "/{serie_id}/perfil"


class ApiPerfilSeriesTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.PerfilSeriesTests("test_serie_activa_contrato_publico_json_y_sin_rutas")
        self.addCleanup(self.datos.doCleanups)
        self.datos.setUp()  # inicializador existente; únicamente base y observaciones sintéticas
        self.ruta = self.datos.ruta
        anteriores = app.dependency_overrides.copy()
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides[api.obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))

    def get(self, serie=21, corte=CORTE):
        return self.cliente.get(f"{PREFIJO}/{serie}/perfil", params={"fecha_corte": corte})

    def error(self, respuesta, estado, mensaje):
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertEqual(respuesta.json(), {"detail": mensaje})
        for privado in (str(self.ruta), "SELECT", "Traceback", "PRIVADO", "privado1", "privado2"):
            self.assertNotIn(privado, respuesta.text)

    def sembrar_pronostico_sintetico(self):
        efectiva = contrato.configuracion_predeterminada(contrato.BASE)
        config = {"version_registro": 2, "tecnica": contrato.BASE, "configuracion_predictiva": efectiva,
                  "parametros": efectiva["parametros"], "reglas": contrato.reglas_heredadas(contrato.BASE),
                  "versiones": {}, "entrenamiento": {}, "auditoria": {}, "historial": {}}
        with closing(conectar(self.ruta)) as c, c:
            eid = c.execute("INSERT INTO experimento (serie_id,tecnica,configuracion_json,inicio_entrenamiento,fin_entrenamiento) "
                            "VALUES (21,?,?,?,?)", (contrato.BASE, json.dumps(config), "2016-12-24", "2017-06-15")).lastrowid
            pid = c.execute("INSERT INTO pronostico (experimento_id,fecha_corte,tipo,horizonte,fecha_generacion,duracion_segundos) "
                            "VALUES (?,?,'validacion',15,'2026-01-01T00:00:00+00:00',0)", (eid, CORTE)).lastrowid
            c.executemany("INSERT INTO detalle_pronostico VALUES (?,?,?,?)",
                          [(pid, i, (date(2017, 6, 16) + timedelta(days=i)).isoformat(), i * 0.25) for i in range(1, 16)])
        return pid

    def test_tres_cortes_json_identico_a_g2_y_una_sola_delegacion(self):
        for corte, observaciones in ((CORTE, 6), ("2017-07-01", 7), ("2017-07-16", 8)):
            with self.subTest(corte=corte):
                esperado = perfil.perfilar_serie(self.ruta, 21, corte)
                with patch.object(api, "perfilar_serie", wraps=perfil.perfilar_serie) as llamada:
                    r = self.get(corte=corte)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json(), esperado)
                self.assertEqual(json.dumps(r.json(), sort_keys=True, allow_nan=False),
                                 json.dumps(esperado, sort_keys=True, allow_nan=False))
                self.assertEqual(r.json()["cobertura"]["observaciones"], observaciones)
                llamada.assert_called_once_with(self.ruta, 21, corte)

    def test_identificacion_y_estadisticas_sinteticas_conocidas(self):
        r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        datos = r.json()
        self.assertEqual((datos["conjunto_id"], datos["serie"], datos["fecha_corte"]),
                         (2, {"id": 21, "store_nbr": 17, "family": "PREPARED FOODS"}, CORTE))
        self.assertEqual((datos["ventas"]["total"], datos["ventas"]["mediana"], datos["ventas"]["percentil_25"],
                          datos["ventas"]["percentil_75"]), (28.0, 2.0, 0.75, 3.25))
        self.assertEqual(datos["ventas"]["promedio_por_observacion"], 28 / 6)
        self.assertEqual(json.loads(json.dumps(datos, allow_nan=False)), datos)

    def test_ausencia_del_25_diciembre_diferente_de_venta_cero(self):
        datos = self.get().json()
        c = datos["cobertura"]
        self.assertEqual((c["observaciones"], c["dias_naturales"], c["fechas_sin_registro"]), (6, 175, 169))
        self.assertIn("2016-12-25", c["fechas_ausentes"])
        for fecha in ("2016-12-24", "2016-12-26"):
            self.assertNotIn(fecha, c["fechas_ausentes"])
        self.assertEqual(datos["ventas"]["ventas_cero"], 1)
        self.assertEqual(c["fechas_ausentes"], sorted(c["fechas_ausentes"]))

    def test_null_y_tipos_permitidos_se_preservan(self):
        self.datos.reemplazar([(CORTE, 2.75, 0)])
        r = self.get()
        self.assertEqual(r.status_code, 200)
        datos = r.json()
        self.assertIsNone(datos["ventas"]["desviacion_estandar_muestral"])
        self.assertIsNone(datos["promociones"]["ventas_con_promocion"]["promedio_por_observacion"])
        self.assertIs(type(datos["ventas"]["total"]), float)
        self.assertIs(type(datos["cobertura"]["observaciones"]), int)
        self.assertEqual(datos, perfil.perfilar_serie(self.ruta, 21, CORTE))
        json.dumps(datos, allow_nan=False)

    def test_fecha_corte_obligatoria_sin_valor_por_defecto(self):
        with patch.object(api, "perfilar_serie") as servicio, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("Falta corte pero abre base")):
            r = self.cliente.get(f"{PREFIJO}/21/perfil")
        self.assertEqual(r.status_code, 422)
        self.assertIsInstance(r.json()["detail"], list)  # validación nativa, como el router anterior
        servicio.assert_not_called()

    def test_identificadores_invalidos_sin_abrir_base(self):
        with patch.object(api, "perfilar_serie") as servicio, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("ID inválido abre base")):
            for serie in (-1, 0, 2**63, "abc", "true"):
                with self.subTest(serie=serie):
                    self.assertEqual(self.get(serie=serie).status_code, 422)
        servicio.assert_not_called()

    def test_fechas_invalidas_no_canonicas_y_vacias(self):
        for corte in ("2017-02-30", "2017-6-16", "20170616", "2017-06-16T00:00:00", "", "null"):
            with self.subTest(corte=corte):
                self.error(self.get(corte=corte), 422, "La fecha de corte debe ser real y canónica YYYY-MM-DD.")

    def test_cortes_no_autorizados_y_prueba_final_bloqueados(self):
        for corte in ("2017-06-17", "2017-07-17", "2017-07-31", "2018-01-01"):
            self.error(self.get(corte=corte), 422, "El corte no está autorizado para perfilar la serie.")
        for corte in ("2017-08-01", "2017-08-07", "2017-08-15"):
            self.error(self.get(corte=corte), 422, "La prueba final permanece reservada.")

    def test_serie_inexistente_e_inactiva_son_404(self):
        for serie in (9999, 11):
            with self.subTest(serie=serie):
                self.error(self.get(serie=serie), 404, "La serie no existe en el conjunto activo.")

    def test_base_ausente_danada_y_sin_esquema_503_sin_creacion(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        danada = self.ruta.with_name("danada.sqlite3")
        danada.write_bytes(b"no es SQLite")
        vacia = self.ruta.with_name("sin_esquema.sqlite3")
        vacia.touch()
        for ruta in (ausente, danada, vacia):
            with self.subTest(ruta=ruta.name):
                app.dependency_overrides[api.obtener_ruta_base] = lambda: ruta
                self.error(self.get(), 503, "La base de datos no está disponible o no tiene el esquema necesario.")
        self.assertFalse(ausente.exists())
        self.assertEqual(vacia.stat().st_size, 0)
        self.assertEqual(danada.read_bytes(), b"no es SQLite")

    def test_sin_conjunto_activo_409(self):
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.error(self.get(), 409, "No existe un conjunto de datos activo.")

    def test_historial_insuficiente_409(self):
        for filas in ([], [("2017-06-17", 1, 0)]):
            self.datos.reemplazar(filas)
            self.error(self.get(), 409, "La serie no tiene observaciones hasta el corte solicitado.")

    def test_datos_almacenados_inconsistentes_409(self):
        for fila in ((CORTE, -1, 0), ("2017-02-30", 1, 0), (CORTE, 1, 1.5)):
            with self.subTest(fila=fila):
                self.datos.reemplazar([fila])
                self.error(self.get(), 409, "La serie contiene observaciones o metadatos inconsistentes.")

    def test_acumulados_y_limites_no_finitos_409(self):
        for valores in ((1e308, 1e308), (0, 1.7e308)):
            with self.subTest(valores=valores):
                self.datos.reemplazar([("2017-06-15", valores[0], 0), (CORTE, valores[1], 0)])
                self.error(self.get(), 409, "Los datos no permiten obtener estadísticas finitas.")

    def test_limite_de_observaciones_409(self):
        with patch.object(perfil, "MAX_OBSERVACIONES", 5):
            self.error(self.get(), 409, "La serie excede el límite de observaciones permitido.")

    def test_periodo_excesivo_409(self):
        with patch.object(perfil, "MAX_DIAS_CALENDARIO", 174):
            self.error(self.get(), 409, "El periodo de cobertura excede el límite permitido.")

    def test_tabla_completa_de_errores_tipificados_con_mensajes_fijos(self):
        casos = [
            ("serie_invalida", 422, "El identificador de serie debe ser un entero positivo válido."),
            ("corte_invalido", 422, "La fecha de corte debe ser real y canónica YYYY-MM-DD."),
            ("corte_no_autorizado", 422, "El corte no está autorizado para perfilar la serie."),
            ("prueba_reservada", 422, "La prueba final permanece reservada."),
            ("serie_inexistente", 404, "La serie no existe en el conjunto activo."),
            ("sin_conjunto_activo", 409, "No existe un conjunto de datos activo."),
            ("historial_insuficiente", 409, "La serie no tiene observaciones hasta el corte solicitado."),
            ("datos_inconsistentes", 409, "La serie contiene observaciones o metadatos inconsistentes."),
            ("limite_observaciones", 409, "La serie excede el límite de observaciones permitido."),
            ("periodo_excesivo", 409, "El periodo de cobertura excede el límite permitido."),
            ("resultado_no_finito", 409, "Los datos no permiten obtener estadísticas finitas."),
            ("base_no_disponible", 503, "La base de datos no está disponible o no tiene el esquema necesario."),
        ]
        for codigo, estado, mensaje in casos:
            with self.subTest(codigo=codigo):
                error = perfil.ErrorPerfilSerie(codigo)
                error.args = (f"SELECT PRIVADO {self.ruta}",)  # nunca publicar str(error)
                with patch.object(api, "perfilar_serie", side_effect=error):
                    self.error(self.get(), estado, mensaje)

    def test_errores_inesperados_o_codigo_desconocido_no_filtran_internos(self):
        desconocido = perfil.ErrorPerfilSerie("datos_inconsistentes")
        desconocido.codigo = "desconocido"
        for error, estado, mensaje in (
            (desconocido, 500, "No se pudo obtener el perfil solicitado."),
            (RuntimeError(f"SELECT PRIVADO {self.ruta}"), 500, "No se pudo obtener el perfil solicitado."),
            (sqlite3.DatabaseError(f"SELECT PRIVADO {self.ruta}"), 503,
             "La base de datos no está disponible o no tiene el esquema necesario."),
        ):
            with patch.object(api, "perfilar_serie", side_effect=error):
                self.error(self.get(), estado, mensaje)

    def test_extremos_y_promociones_futuras_no_alteran_json(self):
        antes = self.get().json()
        self.assertEqual((antes["ventas"]["maximo"], antes["promociones"]["suma_conteos_onpromotion"]), (20.0, 6))
        for venta, promo in ((1e100, 100000), ("FUTURO INVALIDO", 1.5)):
            self.datos.sql("UPDATE venta_diaria SET sales=?,onpromotion=? WHERE serie_id=21 AND fecha>?", (venta, promo, CORTE))
            self.assertEqual(self.get().json(), antes)

    def test_mode_ro_begin_sin_dml_entrenamiento_y_datos_intactos(self):
        pid = self.sembrar_pronostico_sintetico()
        for modo in ("DELETE", "WAL"):
            with self.subTest(modo=modo):
                self.datos.sql(f"PRAGMA journal_mode={modo}")
                estado, contenido = self.datos.estado(), self.ruta.read_bytes()
                abrir = sqlite3.connect
                rutas, sentencias = [], []
                def observar(ruta, *args, **kwargs):
                    rutas.append((str(ruta), kwargs.get("uri")))
                    c = abrir(ruta, *args, **kwargs)
                    c.set_trace_callback(sentencias.append)
                    return c
                with patch.object(sqlite3, "connect", side_effect=observar), \
                        patch("scripts.modelos_prediccion.entrenar_modelo", side_effect=AssertionError("GET entrena")), \
                        patch("backend.app.api.pronosticos.registrar_pronostico", side_effect=AssertionError("GET registra")), \
                        patch("backend.app.services.registro_pronosticos._calcular", side_effect=AssertionError("GET calcula pronóstico")):
                    r = self.get()
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(len(rutas), 1)
                self.assertTrue("mode=ro" in rutas[0][0] and rutas[0][1])
                self.assertEqual(sentencias[0], "BEGIN")
                self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE")) for s in sentencias))
                self.assertTrue(any("WHERE serie_id=21 AND fecha<='2017-06-16'" in s for s in sentencias))
                self.assertEqual(self.datos.estado(), estado)
                if modo == "DELETE":
                    self.assertEqual(self.ruta.read_bytes(), contenido)
                self.assertEqual(self.datos.sql("PRAGMA foreign_key_check"), [])
        self.assertEqual(len(self.cliente.get(f"/api/pronosticos/active/{pid}").json()["detalles"]), 15)

    def test_cambio_de_activo_entre_solicitudes_sin_cache(self):
        self.assertEqual(self.get().json()["conjunto_id"], 2)
        self.datos.sql("UPDATE conjunto_datos SET activo=0 WHERE id=2")
        self.datos.sql("UPDATE conjunto_datos SET activo=1 WHERE id=1")
        self.error(self.get(), 404, "La serie no existe en el conjunto activo.")
        r = self.get(serie=11)
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["conjunto_id"], r.json()["serie"]["id"], r.json()["ventas"]["total"]), (1, 11, 999.0))

    def test_instantanea_http_consistente_durante_cambio_de_activo_en_wal(self):
        self.datos.sql("PRAGMA journal_mode=WAL")
        activo = perfil._activo
        def intercalar(c):
            conjunto = activo(c)
            with closing(conectar(self.ruta)) as escritor, escritor:
                escritor.execute("UPDATE conjunto_datos SET activo=0 WHERE id=2")
                escritor.execute("UPDATE conjunto_datos SET activo=1 WHERE id=1")
                escritor.execute("UPDATE venta_diaria SET sales=500 WHERE serie_id=21 AND fecha='2016-12-24'")
            return conjunto
        with patch.object(perfil, "_activo", side_effect=intercalar):
            r = self.get()
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["conjunto_id"], r.json()["ventas"]["total"]), (2, 28.0))
        self.error(self.get(), 404, "La serie no existe en el conjunto activo.")

    def test_respuesta_sin_rutas_hashes_sql_ni_datos_ajenos(self):
        r = self.get()
        self.assertEqual(r.status_code, 200)
        for privado in (str(self.ruta), "hash", "privado1", "privado2", "SELECT", "Traceback", "ruta_modelo", "BREAD/BAKERY"):
            self.assertNotIn(privado, r.text)
        self.assertEqual(r.json(), perfil.perfilar_serie(self.ruta, 21, CORTE))

    def test_respuestas_anteriores_y_pronosticos_sinteticos_no_cambian(self):
        pid = self.sembrar_pronostico_sintetico()
        urls = ["/health", PREFIJO, PREFIJO + "/21/sales", "/api/pronosticos/active/series/21",
                f"/api/pronosticos/active/{pid}", f"/api/pronosticos/active/{pid}/configuracion"]
        antes = [self.cliente.get(url) for url in urls]
        self.assertTrue(all(r.status_code == 200 for r in antes))
        estado = self.datos.estado()
        self.assertEqual(self.get().status_code, 200)
        despues = [self.cliente.get(url) for url in urls]
        self.assertEqual([r.json() for r in antes], [r.json() for r in despues])
        self.assertEqual([r.content for r in antes], [r.content for r in despues])
        self.assertEqual(self.datos.estado(), estado)

    def test_carga_csv_anterior_sigue_funcionando_y_perfil_usa_nuevo_activo(self):
        csv = b"date,store_nbr,family,sales,onpromotion\n2017-06-14,17,PREPARED FOODS,0.25,0\n2017-06-16,17,PREPARED FOODS,0.75,1\n"
        r = self.cliente.post("/api/datasets/upload", files={"file": ("sintetico.csv", csv, "text/csv")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["series_importadas"], r.json()["observaciones_importadas"]), (1, 2))
        nuevas = self.cliente.get(PREFIJO).json()
        serie = nuevas["series"][0]["id"]
        p = self.get(serie=serie)
        self.assertEqual(p.status_code, 200)
        self.assertEqual((p.json()["conjunto_id"], p.json()["ventas"]["total"]), (r.json()["conjunto_id"], 1.0))
        self.error(self.get(), 404, "La serie no existe en el conjunto activo.")
        self.assertEqual(self.datos.sql("SELECT COUNT(*) FROM experimento"), [(0,)])

    def test_openapi_documenta_ruta_corte_obligatorio_y_estados(self):
        esquema = self.cliente.get("/openapi.json").json()
        ruta = esquema["paths"][RUTA_OPENAPI]
        self.assertEqual(set(ruta), {"get"})
        operacion = ruta["get"]
        parametros = {p["name"]: p for p in operacion["parameters"]}
        self.assertEqual(set(parametros), {"serie_id", "fecha_corte"})
        self.assertTrue(parametros["fecha_corte"]["required"])
        self.assertEqual(parametros["fecha_corte"]["in"], "query")
        self.assertEqual(parametros["fecha_corte"]["schema"]["enum"], [CORTE, "2017-07-01", "2017-07-16"])
        self.assertNotIn("default", parametros["fecha_corte"]["schema"])
        self.assertEqual(parametros["serie_id"]["schema"]["exclusiveMinimum"], 0)
        self.assertTrue({"200", "422", "404", "409", "503"}.issubset(operacion["responses"]))
        self.assertEqual(operacion["responses"]["200"]["content"]["application/json"]["schema"]["type"], "object")

    def test_importacion_arranque_health_y_docs_no_consultan_datos(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("Arranque consulta base")), \
                patch.object(api, "perfilar_serie", side_effect=AssertionError("Arranque calcula perfil")), \
                patch("scripts.modelos_prediccion.entrenar_modelo", side_effect=AssertionError("Arranque entrena")):
            for nombre, archivo in (("datasets_importacion_g3", Path(api.__file__)),
                                     ("main_importacion_g3", Path(__file__).resolve().parents[1] / "backend/app/main.py")):
                spec = importlib.util.spec_from_file_location(nombre, archivo)
                modulo = importlib.util.module_from_spec(spec)
                sys.modules[nombre] = modulo
                self.addCleanup(lambda nombre=nombre: sys.modules.pop(nombre, None))
                spec.loader.exec_module(modulo)
            with TestClient(modulo.app) as cliente:
                self.assertEqual(cliente.get("/health").json(), {"status": "ok"})
                self.assertEqual(cliente.get("/docs").status_code, 200)
                self.assertIn(RUTA_OPENAPI, cliente.get("/openapi.json").json()["paths"])


if __name__ == "__main__":
    unittest.main()
