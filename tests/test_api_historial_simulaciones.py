"""S4-A5-01: HTTP histórico exclusivamente con SQLite temporal sintética."""

from contextlib import contextmanager
from copy import deepcopy
import json
import sqlite3
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import simulaciones as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import consulta_historial_simulaciones as consulta
from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.storage import migraciones_simulacion as esquema
from tests import test_consulta_historial_simulaciones as fixtures


BASE = "/api/simulaciones/historial"
MAX = 2**63 - 1


class ApiHistorialSimulacionesTests(unittest.TestCase):
    def setUp(self):
        # Composición sin herencia: preparar solo esta SQLite temporal.
        self.datos = fixtures.ConsultaHistorialSimulacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        anteriores = dict(app.dependency_overrides)
        def restaurar():
            app.dependency_overrides.clear()
            app.dependency_overrides.update(anteriores)
        self.addCleanup(restaurar)
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))

    def get(self, sufijo="", **opciones):
        r = self.cliente.request("GET", BASE + sufijo, **opciones)
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        return r

    def error(self, r, codigo):
        self.assertEqual(r.status_code, api.ERRORES[codigo][0], r.text)
        self.assertEqual(r.json(), {"detail": {"codigo": codigo, "mensaje": api.ERRORES[codigo][1]}})
        for privado in (str(self.ruta), "PRIVADO", "Traceback", "SELECT", "ruta_modelo", "configuracion_json", "privada-"):
            self.assertNotIn(privado, r.text)

    def exacto(self, recibido, esperado):
        # El oráculo canónico comprueba tipos int/float y bits de -0.0 además del valor.
        self.assertEqual(snapshot.serializar_documento(recibido), snapshot.serializar_documento(esperado))

    def individual(self, **opciones):
        return self.datos.datos.individual(**opciones)

    @contextmanager
    def vigilar_sql(self):
        abrir = sqlite3.connect
        conexiones, sql, denegadas = [], [], []
        permitidas = {"sqlite_master", "sqlite_schema", *esquema.TABLAS_NUEVAS}
        escrituras = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                      sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TRIGGER,
                      sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX,
                      sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_ALTER_TABLE}
        def autorizar(accion, tabla, columna, base, origen):
            if accion in escrituras or (accion == sqlite3.SQLITE_READ and tabla not in permitidas):
                denegadas.append((accion, tabla, columna))
                return sqlite3.SQLITE_DENY
            if accion == sqlite3.SQLITE_PRAGMA and tabla in ("journal_mode", "user_version"):
                denegadas.append((accion, tabla, columna))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        class Vigilada(sqlite3.Connection):
            cerrada = False
            def execute(c, texto, parametros=()):
                sql.append((texto, tuple(parametros)))
                return super().execute(texto, parametros)
            def close(c):
                c.cerrada = True
                return super().close()
        def conectar(ruta, **opciones):
            self.assertEqual(ruta, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertEqual(opciones, {"uri": True})
            c = abrir(ruta, factory=Vigilada, **opciones)
            c.set_authorizer(autorizar)
            conexiones.append(c)
            return c
        with patch.object(sqlite3, "connect", side_effect=conectar):
            yield
        self.assertEqual(len(conexiones), 1)
        self.assertTrue(all(c.cerrada for c in conexiones))
        self.assertEqual([s for s, _ in sql if s.startswith("BEGIN")], ["BEGIN"])
        self.assertFalse(denegadas)

    def test_historial_vacio(self):
        r = self.get()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"total": 0, "limite": 20, "desplazamiento": 0, "ejecuciones": []})

    def test_listado_individual_ab_abc_y_orden(self):
        creados = self.datos.basicos()
        r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        self.exacto(r.json(), consulta.listar_ejecuciones_simulaciones(self.ruta))
        self.assertEqual([e["ejecucion_id"] for e in r.json()["ejecuciones"]], [x["ejecucion_id"] for x in reversed(creados)])
        self.assertEqual([e["etiquetas"] for e in r.json()["ejecuciones"]], [["A", "B", "C"], ["A", "B"], ["A"]])

    def test_detalle_individual_y_fecha_original(self):
        r = self.individual()
        respuesta = self.get("/" + str(r["ejecucion_id"]))
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.exacto(respuesta.json(), consulta.consultar_ejecucion_simulacion(self.ruta, r["ejecucion_id"]))
        self.assertEqual(respuesta.json()["fecha_registro"], r["fecha_registro"])
        self.assertRegex(respuesta.json()["fecha_registro"], r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

    def test_detalle_ab_abc_con_quince_filas(self):
        for n in (2, 3):
            r = self.datos.datos.comparar(clave=str(n), n=n)
            respuesta = self.get("/" + str(r["ejecucion_id"]))
            self.assertEqual(respuesta.status_code, 200, respuesta.text)
            self.exacto(respuesta.json()["historico"], r["historico"])
            h = respuesta.json()["historico"]
            self.assertEqual(len(h["origen"]["detalles"]), 15)
            self.assertEqual([e["etiqueta"] for e in h["resultados"]], list("ABC"[:n]))
            self.assertTrue(all(len(e["simulacion"]["trayectoria"]) == 15 for e in h["resultados"]))

    def test_filtros_individuales(self):
        self.datos.todos()
        for filtros, total in (({"pronostico_id": 101}, 3), ({"conjunto_id": 2}, 1), ({"serie_id": 84}, 1),
                               ({"modalidad": "individual"}, 3), ({"modalidad": "comparacion"}, 2)):
            with self.subTest(filtros=filtros):
                r = self.get(params=filtros)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["total"], total)
                self.exacto(r.json(), consulta.listar_ejecuciones_simulaciones(self.ruta, **filtros))

    def test_filtros_combinados(self):
        self.datos.todos()
        filtros = dict(pronostico_id=101, conjunto_id=7, serie_id=42, modalidad="comparacion")
        r = self.get(params=filtros)
        self.assertEqual(r.status_code, 200, r.text)
        self.exacto(r.json(), consulta.listar_ejecuciones_simulaciones(self.ruta, **filtros))
        self.assertEqual(self.get(params={"pronostico_id": MAX}).json()["total"], 0)

    def test_paginacion_total_y_limites(self):
        r = self.datos.basicos()
        pagina = self.get(params={"limite": 1, "desplazamiento": 1})
        self.assertEqual(pagina.status_code, 200)
        self.assertEqual(pagina.json()["total"], 3)
        self.assertEqual(pagina.json()["ejecuciones"][0]["ejecucion_id"], r[1]["ejecucion_id"])
        extrema = self.get(params={"limite": 50, "desplazamiento": MAX})
        self.assertEqual(extrema.status_code, 200)
        self.assertEqual(extrema.json()["ejecuciones"], [])

    def test_delegacion_listado_exactamente_una_vez(self):
        real = consulta.listar_ejecuciones_simulaciones
        with patch.object(api, "listar_ejecuciones_simulaciones", wraps=real) as espia:
            r = self.get(params={"limite": "02", "desplazamiento": "0", "conjunto_id": "7"})
        self.assertEqual(r.status_code, 200, r.text)
        espia.assert_called_once_with(self.ruta, limite=2, desplazamiento=0, pronostico_id=None,
                                     conjunto_id=7, serie_id=None, modalidad=None)

    def test_delegacion_detalle_exactamente_una_vez(self):
        r = self.individual()
        with patch.object(api, "consultar_ejecucion_simulacion", wraps=consulta.consultar_ejecucion_simulacion) as espia:
            respuesta = self.get("/" + str(r["ejecucion_id"]))
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        espia.assert_called_once_with(self.ruta, r["ejecucion_id"])

    def test_historico_inactivo_y_origen_modificado(self):
        r = self.individual()
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.datos.sql("UPDATE detalle_pronostico SET venta_estimada=999 WHERE pronostico_id=101")
        self.datos.sql("UPDATE experimento SET tecnica='tecnica_futura' WHERE id=201")
        with self.datos.sin_calculos_actuales(), self.vigilar_sql():
            respuesta = self.get("/" + str(r["ejecucion_id"]))
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.exacto(respuesta.json()["historico"], r["historico"])

    def test_listado_sin_validadores_vigentes(self):
        self.datos.basicos()
        with self.datos.sin_calculos_actuales(), self.vigilar_sql():
            r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["total"], 3)

    def test_tipos_cero_float_y_menos_cero(self):
        for i, numero in enumerate((0, 0.0, -0.0)):
            r = self.individual(clave=str(i), inventario=numero)
            respuesta = self.get("/" + str(r["ejecucion_id"]))
            self.assertEqual(respuesta.status_code, 200, respuesta.text)
            self.exacto(respuesta.json()["historico"], r["historico"])
            valor = respuesta.json()["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]
            self.assertIs(type(valor), type(numero))
            if type(numero) is float:
                self.assertEqual(valor.hex(), numero.hex())

    def test_subnormal_y_residuo_binary64(self):
        for i, valor in enumerate((float.fromhex("0x0.0000000000001p-1022"), 0.1, 0.3)):
            r = self.individual(clave=str(i), inventario=valor, reserva=0.1)
            respuesta = self.get("/" + str(r["ejecucion_id"]))
            self.assertEqual(respuesta.status_code, 200, respuesta.text)
            self.exacto(respuesta.json()["historico"], r["historico"])
            parametro = respuesta.json()["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]
            self.assertEqual(parametro.hex(), valor.hex())

    def test_id_invalido_antes_del_servicio(self):
        with patch.object(api, "consultar_ejecucion_simulacion") as espia:
            for valor in ("0", "-1", "1.0", "1e2", "true", "NaN", "１２", "١", " 1", "+1", str(MAX + 1), "9" * 5000):
                with self.subTest(valor=valor[:30]):
                    self.error(self.get("/" + valor), "ejecucion_invalida")
        espia.assert_not_called()

    def test_id_inexistente(self):
        self.error(self.get("/1"), "ejecucion_inexistente")

    def test_filtros_invalidos_antes_del_servicio(self):
        with patch.object(api, "listar_ejecuciones_simulaciones") as espia:
            for campo in ("pronostico_id", "conjunto_id", "serie_id"):
                for valor in ("", "0", "-1", "true", "1.5", "1e2", "１２", "+1", " 1", str(MAX + 1), "9" * 5000):
                    with self.subTest(campo=campo, valor=valor[:30]):
                        self.error(self.get(params={campo: valor}), "filtro_invalido")
            for valor in ("", "Individual", "true", "desconocida", " individual"):
                self.error(self.get(params={"modalidad": valor}), "filtro_invalido")
        espia.assert_not_called()

    def test_paginacion_invalida_antes_del_servicio(self):
        with patch.object(api, "listar_ejecuciones_simulaciones") as espia:
            for campo, valores in (("limite", ("0", "51", "-1", "1.0", "1e1", "true", "１２", "+1", " 1", "")),
                                   ("desplazamiento", ("-1", str(MAX + 1), "0.0", "1e2", "false", "١", "", "9" * 5000))):
                for valor in valores:
                    with self.subTest(campo=campo, valor=valor[:30]):
                        self.error(self.get(params={campo: valor}), "paginacion_invalida")
        espia.assert_not_called()

    def test_parametros_desconocidos_y_repetidos(self):
        with patch.object(api, "listar_ejecuciones_simulaciones") as listado, patch.object(api, "consultar_ejecucion_simulacion") as detalle:
            self.error(self.get(params={"activo": 1}), "entrada_invalida")
            for nombre in ("limite", "desplazamiento", "pronostico_id", "conjunto_id", "serie_id", "modalidad"):
                self.error(self.get(params=[(nombre, "1"), (nombre, "1")]), "entrada_invalida")
            self.error(self.get("/1", params={"limite": 1}), "entrada_invalida")
        listado.assert_not_called()
        detalle.assert_not_called()

    def test_cuerpos_get_rechazados(self):
        with patch.object(api, "listar_ejecuciones_simulaciones") as listado, patch.object(api, "consultar_ejecucion_simulacion") as detalle:
            for sufijo in ("", "/1"):
                for cuerpo in (b"{}", b"null", b" ", b"texto"):
                    self.error(self.get(sufijo, content=cuerpo), "entrada_invalida")
        listado.assert_not_called()
        detalle.assert_not_called()

    def test_snapshot_hash_corrupto_en_listado_y_detalle(self):
        r = self.individual()
        self.datos.sql("UPDATE escenario_simulacion SET sha256_resultado=?", ("0" * 64,))
        for sufijo in ("", "/" + str(r["ejecucion_id"])):
            with self.vigilar_sql():
                self.error(self.get(sufijo), "snapshot_corrupto")

    def test_documento_adulterado_con_hash_recalculado(self):
        r = self.individual()
        texto = self.datos.sql("SELECT resultado_json FROM escenario_simulacion")[0][0]
        d = json.loads(texto)
        d["pronostico_id"] = 999
        adulterado = snapshot.serializar_documento(d)
        self.datos.sql("UPDATE escenario_simulacion SET resultado_json=?,sha256_resultado=?", (adulterado, snapshot.calcular_sha256(adulterado)))
        self.error(self.get("/" + str(r["ejecucion_id"])), "snapshot_corrupto")

    def test_hijo_faltante_sin_respuesta_parcial(self):
        r = self.datos.datos.comparar(n=3)
        self.datos.sql("DELETE FROM escenario_simulacion WHERE etiqueta='C'")
        self.error(self.get(), "snapshot_corrupto")
        self.error(self.get("/" + str(r["ejecucion_id"])), "snapshot_corrupto")

    def test_posicion_adulterada(self):
        r = self.individual()
        self.datos.datos.datos.corromper("UPDATE escenario_simulacion SET posicion=2")
        self.error(self.get("/" + str(r["ejecucion_id"])), "snapshot_corrupto")

    def test_version_historica_incompatible(self):
        r = self.individual()
        self.datos.sql("UPDATE ejecucion_simulacion SET version_documento=2")
        self.error(self.get("/" + str(r["ejecucion_id"])), "snapshot_incompatible")

    def test_esquema_no_preparado(self):
        for t in reversed(esquema.TABLAS_NUEVAS): self.datos.sql("DROP TABLE " + t)
        self.error(self.get(), "esquema_no_preparado")

    def test_esquema_parcial_y_version_no_soportada(self):
        self.datos.sql("UPDATE version_esquema_simulaciones SET version=2")
        self.error(self.get(), "version_no_soportada")
        self.datos.sql("DROP TABLE escenario_simulacion")
        self.error(self.get(), "esquema_incompatible")

    def test_base_ausente_y_danada(self):
        for nombre, contenido in (("ausente.sqlite3", None), ("danada.sqlite3", b"no sqlite sintetica")):
            ruta = self.ruta.with_name(nombre)
            if contenido is not None: ruta.write_bytes(contenido)
            app.dependency_overrides[obtener_ruta_base] = lambda: ruta
            self.error(self.get(), "base_no_disponible")
            if contenido is None: self.assertFalse(ruta.exists())

    def test_catalogo_errores_y_mensajes_internos(self):
        for codigo in api.ERRORES:
            if codigo == "error_interno": continue
            error = consulta.ErrorConsultaHistorialSimulaciones(codigo) if codigo != "entrada_invalida" else None
            if error is None: continue
            error.args = ("PRIVADO SELECT C:/privada- ruta_modelo",)
            for funcion, sufijo in (("listar_ejecuciones_simulaciones", ""), ("consultar_ejecucion_simulacion", "/1")):
                with self.subTest(codigo=codigo, funcion=funcion), patch.object(api, funcion, side_effect=error):
                    self.error(self.get(sufijo), codigo)

    def test_error_inesperado_y_codigo_desconocido(self):
        error = consulta.ErrorConsultaHistorialSimulaciones("snapshot_corrupto")
        error.codigo = "PRIVADO"
        for causa in (RuntimeError("PRIVADO SELECT C:/datos"), error):
            for funcion, sufijo in (("listar_ejecuciones_simulaciones", ""), ("consultar_ejecucion_simulacion", "/1")):
                with patch.object(api, funcion, side_effect=causa):
                    self.error(self.get(sufijo), "error_interno")

    def test_respuesta_incompatible_no_expone_validacion(self):
        with patch.object(api, "consultar_ejecucion_simulacion", return_value={"ruta_modelo": "PRIVADO"}):
            self.error(self.get("/1"), "error_interno")

    def test_sin_clave_ni_campos_privados(self):
        self.individual(clave="privada-idempotencia")
        for sufijo in ("", "/1"):
            r = self.get(sufijo)
            self.assertEqual(r.status_code, 200, r.text)
            for campo in ("clave_idempotencia", "privada-idempotencia", "configuracion_json", "ruta_modelo"):
                self.assertNotIn(campo, r.text)

    def test_get_sin_dml_ddl_y_base_intacta(self):
        self.datos.basicos()
        antes = self.ruta.read_bytes()
        tablas = self.datos.datos.tablas()
        antiguas = self.datos.datos.antiguas()
        for sufijo in ("", "/1", "/3"):
            with self.vigilar_sql(), self.datos.sin_calculos_actuales():
                r = self.get(sufijo)
            self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.ruta.read_bytes(), antes)
        self.assertEqual(self.datos.datos.tablas(), tablas)
        self.assertEqual(self.datos.datos.antiguas(), antiguas)

    def test_importacion_router_main_y_openapi_sin_efectos(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir SQLite")), self.datos.sin_calculos_actuales():
            for archivo in (Path(api.__file__), Path("backend/app/main.py")):
                modulo = ModuleType("auditoria_importacion")
                exec(compile(archivo.read_text(encoding="utf-8"), str(archivo), "exec"), modulo.__dict__)
            documento = app.openapi()
        self.assertIn(BASE, documento["paths"])

    def test_openapi_contratos_y_routers_previos(self):
        doc = self.cliente.get("/openapi.json").json()
        for ruta in (BASE, BASE + "/{ejecucion_id}"):
            self.assertEqual(set(doc["paths"][ruta]), {"get"})
            operacion = doc["paths"][ruta]["get"]
            self.assertNotIn("requestBody", operacion)
            self.assertEqual(set(operacion["responses"]), {"200", "422", "404", "409", "503", "500"})
            for respuesta in operacion["responses"].values():
                self.assertIn("Cache-Control", respuesta["headers"])
        parametros = doc["paths"][BASE]["get"]["parameters"]
        self.assertEqual({p["name"] for p in parametros}, api._QUERY_LISTADO)
        for p in parametros:
            if p["name"] in ("limite", "desplazamiento"):
                self.assertEqual(p["schema"]["default"], {"limite": 20, "desplazamiento": 0}[p["name"]])
        for ruta in ("/health", "/api/evaluaciones/active/{pronostico_id}", "/api/evaluaciones/comparacion/active",
                     "/api/evaluaciones/consolidacion/active", "/api/pronosticos/active/{pronostico_id}"):
            self.assertIn(ruta, doc["paths"])
        self.assertTrue(any(r.startswith("/api/datasets/") for r in doc["paths"]))
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        for nombre in ("ListadoHistorial", "DetalleHistorial", "PaqueteHistorico", "ErrorHistorial"):
            self.assertFalse(doc["components"]["schemas"][nombre]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
