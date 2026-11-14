"""HU005-C1/C2: GET y POST con TestClient y SQLite exclusivamente sintéticas."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from math import sqrt
from pathlib import Path
import sqlite3
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import datasets, evaluaciones as api, pronosticos
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import registro_evaluaciones as registro
from backend.app.services import registro_pronosticos as generacion
from scripts import linea_base_prediccion, modelos_prediccion
from tests import test_consulta_evaluaciones as fixtures


PREFIJO = "/api/evaluaciones/active"
MAX_ID = 2**63 - 1


class ApiEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.ConsultaEvaluacionesTests()
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

    def obtener(self, identificador=101, **opciones):
        r = self.cliente.get(f"{PREFIJO}/{identificador}", **opciones)
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        return r

    def consultar(self, identificador=101):
        contenido = self.ruta.read_bytes()
        r = self.obtener(identificador)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertNotIn("PRIVADO", r.text)
        datos = r.json()
        self.comprobar_sin_estados_guardados(datos)
        e = datos["evaluacion"]
        self.assertEqual(e["observaciones_evaluadas"] + e["fechas_sin_registro"], 15)
        self.assertEqual(len(e["detalles"]), 15)
        self.assertEqual([d["paso"] for d in e["detalles"]], list(range(1, 16)))
        self.assertEqual([d["fecha_objetivo"] for d in e["detalles"]],
                         sorted(d["fecha_objetivo"] for d in e["detalles"]))
        return datos

    def comprobar_sin_estados_guardados(self, valor):
        if isinstance(valor, dict):
            self.assertTrue({"registrado", "vigente", "reutilizado", "estado_registro", "hubo_escritura"}.isdisjoint(valor))
            for hijo in valor.values():
                self.comprobar_sin_estados_guardados(hijo)
        elif isinstance(valor, list):
            for hijo in valor:
                self.comprobar_sin_estados_guardados(hijo)

    def comprobar_error(self, respuesta, estado, codigo, *, errores=None):
        errores = api.ERRORES if errores is None else errores
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertEqual(respuesta.headers.get("cache-control"), "no-store")
        self.assertEqual(respuesta.json(), {"detail": {"codigo": codigo, "mensaje": errores[codigo][1]}})
        for privado in ("PRIVADO", "SELECT", "Traceback", str(self.ruta)):
            self.assertNotIn(privado, respuesta.text)

    @contextmanager
    def vigilar_sql(self):
        abrir_real = sqlite3.connect
        llamadas, aperturas, cierres, denegadas = [], [], [], []

        class Lectura(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                llamadas.append((sql, parametros, conexion.in_transaction))
                return super().execute(sql, parametros)

            def close(conexion):
                cierres.append(True)
                return super().close()

        def abrir(ruta_uri, **opciones):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertTrue(opciones["uri"])
            c = abrir_real(ruta_uri, factory=Lectura, **opciones)
            aperturas.append(True)

            def autorizar(accion, tabla, columna, *otros):
                permitidas = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                              sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION)
                if (accion not in permitidas or (accion == sqlite3.SQLITE_READ
                        and (tabla == "evaluacion" or columna in ("configuracion_json", "ruta_modelo")))):
                    denegadas.append((accion, tabla, columna))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            c.set_authorizer(autorizar)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir):
            yield llamadas
        self.assertEqual(aperturas, [True])
        self.assertEqual(cierres, [True])
        self.assertEqual(denegadas, [])

    def test_get_completo_metricas_exactas_y_contrato_a2_integro(self):
        directo = consulta.consultar_evaluacion(self.ruta, 101)
        r = self.consultar()
        self.assertEqual(r, directo)
        self.assertEqual((r["conjunto_id"], r["pronostico_id"], r["experimento_id"]), (7, 101, 201))
        self.assertEqual(r["serie"], {"id": 83, "store_nbr": 17, "family": "FAMILIA A"})
        e = r["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (2, 10, sqrt(20 / 3)))
        self.assertEqual([d["error_absoluto"] for d in e["detalles"]], [2, 4, 0] * 5)
        self.assertEqual((r["estado_cobertura"], e["cobertura"]), ("completa", 1))

    def test_delega_exactamente_una_llamada_al_servicio_publico_a2(self):
        with patch.object(api, "consultar_evaluacion", wraps=consulta.consultar_evaluacion) as consultar:
            self.consultar()
        consultar.assert_called_once_with(self.ruta, 101)

    def test_cobertura_parcial_y_fechas_interiores_ausentes(self):
        faltan = (2, 8, 11)
        for paso in faltan:
            self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha=?", (fixtures.fecha_objetivo(fixtures.CORTES[0], paso),))
        r = self.consultar()
        e = r["evaluacion"]
        self.assertEqual(r["estado_cobertura"], "parcial")
        self.assertEqual(r["alcance_metricas"], "Métricas sobre observaciones disponibles.")
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (12, 3, 12 / 15))
        self.assertEqual(e["fechas_ausentes"], [fixtures.fecha_objetivo(fixtures.CORTES[0], p) for p in faltan])
        for paso in faltan:
            self.assertIsNone(e["detalles"][paso - 1]["venta_real"])
            self.assertIsNone(e["detalles"][paso - 1]["error_absoluto"])
        self.assertEqual(e["mae"], 18 / 12)
        self.assertAlmostEqual(e["rmse"], sqrt(52 / 12))

    def test_sin_observaciones_metricas_null(self):
        self.datos.ejecutar("DELETE FROM venta_diaria")
        r = self.consultar()
        e = r["evaluacion"]
        self.assertEqual(r["estado_cobertura"], "sin_observaciones")
        self.assertEqual(tuple(e[c] for c in ("mae", "wape", "rmse")), (None, None, None))
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (0, 15, 0))

    def test_reales_cero_wape_null_mae_y_rmse_definidos(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        e = self.consultar()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (2, None, 2))
        self.assertEqual(e["cobertura"], 1)

    def test_valores_fraccionarios_y_wape_superior_a_cien(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0.25")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=0.75")
        e = self.consultar()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (0.5, 200, 0.5))
        self.assertEqual(e["detalles"][0]["venta_real"], 0.25)

    def test_tres_cortes_autorizados(self):
        for i, corte in enumerate(fixtures.CORTES):
            with self.subTest(corte=corte):
                if i:
                    self.datos.crear_pronostico(101 + i, 201 + i, corte=corte)
                    self.datos.insertar_reales(corte=corte)
                r = self.consultar(101 + i)
                self.assertEqual(r["fecha_corte"], corte)
                self.assertEqual(tuple(r["evaluacion"][c] for c in ("mae", "wape", "rmse")),
                                 (2, 10, sqrt(20 / 3)))
                self.assertEqual(r["evaluacion"]["detalles"][-1]["fecha_objetivo"], fixtures.fecha_objetivo(corte, 15))

    def test_tecnicas_y_series_distintas_sin_mezclar_observaciones(self):
        self.datos.crear_pronostico(102, 202, tecnica="random_forest", estimadas=fixtures.REALES)
        self.datos.crear_pronostico(103, 203, serie=84, tecnica="hist_gradient_boosting", estimadas=[100] * 15)
        self.datos.insertar_reales(serie=84, valores=[100] * 15)
        a, b, c = self.consultar(), self.consultar(102), self.consultar(103)
        self.assertEqual((a["evaluacion"]["mae"], b["evaluacion"]["mae"], c["evaluacion"]["mae"]), (2, 0, 0))
        self.assertEqual([d["venta_real"] for d in a["evaluacion"]["detalles"]],
                         [d["venta_real"] for d in b["evaluacion"]["detalles"]])
        self.assertEqual([d["venta_real"] for d in c["evaluacion"]["detalles"]], [100] * 15)
        self.assertEqual(c["serie"]["id"], 84)

    def test_pronostico_inactivo_y_cambio_de_conjunto_activo(self):
        self.datos.crear_pronostico(102, 202, serie=24)
        self.comprobar_error(self.obtener(102), 404, "pronostico_inactivo")
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
        self.assertEqual(self.consultar(102)["conjunto_id"], 2)
        self.comprobar_error(self.obtener(), 404, "pronostico_inactivo")

    def test_sin_conjunto_activo_es_409(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.comprobar_error(self.obtener(), 409, "sin_conjunto_activo")

    def test_identificadores_invalidos_422_sin_llamar_al_servicio(self):
        with patch.object(api, "consultar_evaluacion") as consultar:
            for valor in ("0", "-1", "1.5", "101.0", "1e2", "true", "abc", "+101", " 101", "１２", "١", str(MAX_ID + 1), "9" * 100):
                with self.subTest(valor=valor):
                    self.comprobar_error(self.obtener(valor), 422, "pronostico_invalido")
            consultar.assert_not_called()

    def test_limite_sqlite_se_admite_y_no_se_trunca(self):
        with patch.object(api, "consultar_evaluacion", wraps=consulta.consultar_evaluacion) as consultar:
            self.comprobar_error(self.obtener(MAX_ID), 404, "pronostico_inexistente")
        consultar.assert_called_once_with(self.ruta, MAX_ID)

    def test_pronostico_inexistente_es_404(self):
        self.comprobar_error(self.obtener(999), 404, "pronostico_inexistente")

    def test_tipo_prueba_futuro_y_cortes_no_autorizados(self):
        for tipo, corte, codigo in (("prueba", fixtures.CORTES[0], "tipo_no_evaluable"),
                                    ("futuro", "2017-07-31", "tipo_no_evaluable"),
                                    ("validacion", "2017-07-31", "corte_no_autorizado"),
                                    ("validacion", "2017-06-17", "corte_no_autorizado")):
            self.datos.ejecutar("UPDATE pronostico SET tipo=?,fecha_corte=?", (tipo, corte))
            with self.subTest(tipo=tipo, corte=corte), self.vigilar_sql() as llamadas:
                self.comprobar_error(self.obtener(), 422, codigo)
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_detalle_omitido_es_409(self):
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE paso=8")
        self.comprobar_error(self.obtener(), 409, "datos_inconsistentes")

    def test_fechas_valores_y_metadatos_corruptos_son_409(self):
        casos = (("UPDATE detalle_pronostico SET fecha_objetivo=? WHERE paso=8", "2017-06-16",
                  "UPDATE detalle_pronostico SET fecha_objetivo='2017-06-24' WHERE paso=8"),
                 ("UPDATE detalle_pronostico SET venta_estimada=? WHERE paso=1", -1,
                  "UPDATE detalle_pronostico SET venta_estimada=8 WHERE paso=1"),
                 ("UPDATE venta_diaria SET sales=? WHERE fecha='2017-06-17'", "PRIVADO NO_NUMERICO",
                  "UPDATE venta_diaria SET sales=10 WHERE fecha='2017-06-17'"),
                 ("UPDATE pronostico SET horizonte=?", 14, "UPDATE pronostico SET horizonte=15"))
        for sql, valor, restaurar in casos:
            with self.subTest(sql=sql):
                self.datos.corromper(sql, (valor,))
                self.comprobar_error(self.obtener(), 409, "datos_inconsistentes")
                self.datos.ejecutar(restaurar)

    def test_base_inexistente_danada_o_sin_esquema_es_503(self):
        ausente = self.ruta.parent / "PRIVADO_ausente.sqlite3"
        app.dependency_overrides[obtener_ruta_base] = lambda: ausente
        self.comprobar_error(self.obtener(), 503, "base_no_disponible")
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("corrupta.sqlite3", b"Esto no es SQLite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            app.dependency_overrides[obtener_ruta_base] = lambda: ruta
            self.comprobar_error(self.obtener(), 503, "base_no_disponible")
            self.assertEqual(ruta.read_bytes(), contenido)

    def test_mapeo_de_errores_descarta_mensajes_internos_inyectados(self):
        codigos = (("pronostico_invalido", 422), ("tipo_no_evaluable", 422), ("corte_no_autorizado", 422),
                   ("pronostico_inexistente", 404), ("pronostico_inactivo", 404),
                   ("sin_conjunto_activo", 409), ("datos_inconsistentes", 409), ("base_no_disponible", 503))
        for codigo, estado in codigos:
            with self.subTest(codigo=codigo), patch.object(api, "consultar_evaluacion", side_effect=
                    consulta.ErrorConsultaEvaluacion(codigo, "PRIVADO SELECT C:/secreto Traceback")):
                self.comprobar_error(self.obtener(), estado, codigo)

    def test_codigos_desconocidos_y_excepciones_inesperadas_son_500_generico(self):
        errores = [consulta.ErrorConsultaEvaluacion(c, "PRIVADO") for c in ("PRIVADO codigo", None, [], {})]
        errores += [RuntimeError("PRIVADO Traceback"), sqlite3.OperationalError("PRIVADO SELECT"), OSError("PRIVADO ruta")]
        for error in errores:
            with self.subTest(error=type(error).__name__), patch.object(api, "consultar_evaluacion", side_effect=error):
                self.comprobar_error(self.obtener(), 500, "error_interno")

    def test_error_inesperado_de_dependencia_se_sanea(self):
        def fallar():
            raise RuntimeError("PRIVADO ruta interna")

        app.dependency_overrides[obtener_ruta_base] = fallar
        with patch.object(api, "consultar_evaluacion") as consultar:
            self.comprobar_error(self.obtener(), 500, "error_interno")
            consultar.assert_not_called()

    def test_no_admite_parametros_de_consulta(self):
        with patch.object(api, "consultar_evaluacion") as consultar:
            for parametros in ({"fecha_corte": "2017-06-16"}, {"otro": "PRIVADO"},
                               [("otro", "1"), ("otro", "2")], {"otro": ""}):
                with self.subTest(parametros=parametros):
                    self.comprobar_error(self.obtener(params=parametros), 422, "entrada_invalida")
            consultar.assert_not_called()

    def test_no_admite_cuerpo_incluso_vacio_json_o_malformado(self):
        with patch.object(api, "consultar_evaluacion") as consultar:
            for contenido in (b"{}", b"[]", b"null", b" ", b"PRIVADO JSON malformado"):
                with self.subTest(contenido=contenido):
                    r = self.cliente.request("GET", f"{PREFIJO}/101", content=contenido,
                                             headers={"Content-Type": "application/json"})
                    self.comprobar_error(r, 422, "entrada_invalida")
            consultar.assert_not_called()

    def test_validacion_de_respuesta_no_expone_contenido_incorrecto(self):
        valido = consulta.consultar_evaluacion(self.ruta, 101)
        incorrecto = deepcopy(valido)
        incorrecto["PRIVADO"] = "ruta interna"
        with patch.object(api, "consultar_evaluacion", return_value=incorrecto):
            self.comprobar_error(self.obtener(), 500, "error_interno")

    def test_precision_numerica_y_nulos_sin_redondear(self):
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha>'2017-06-19'")
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=CASE WHEN paso=1 THEN 1 ELSE 0 END")
        e = self.consultar()["evaluacion"]
        self.assertEqual(e["mae"], 1 / 3)
        self.assertEqual(e["rmse"], sqrt(1 / 3))
        self.assertIsNone(e["wape"])

    def test_get_ro_una_instantanea_sin_dml_ni_lectura_de_tabla_evaluacion(self):
        contenido = self.ruta.read_bytes()
        with self.vigilar_sql() as llamadas:
            self.consultar()
        self.assertEqual(llamadas[0][:2], ("BEGIN", ()))
        self.assertEqual(len(llamadas), 5)
        self.assertTrue(all(en_transaccion for _, _, en_transaccion in llamadas[1:]))
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
                             for s, _, _ in llamadas))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.datos.ejecutar("SELECT COUNT(*) FROM evaluacion"), [(0,)])

    def test_get_no_entrena_genera_ni_registra_evaluaciones(self):
        objetivos = ((registro, "registrar_evaluacion"), (generacion, "registrar_pronostico"),
                     (generacion, "_calcular"), (modelos_prediccion, "entrenar_modelo"),
                     (modelos_prediccion, "crear_estimador"), (modelos_prediccion, "pronosticar_recursivo"),
                     (linea_base_prediccion, "pronosticar_serie"), (pronosticos, "registrar_pronostico"))
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            self.consultar()
            for doble in dobles:
                doble.assert_not_called()

    def test_b1_guardada_y_ventas_cambiadas_get_recalcula_sin_alterar_fila(self):
        registro.registrar_evaluacion(self.ruta, 101)
        guardada = self.datos.ejecutar("SELECT * FROM evaluacion")
        self.datos.ejecutar("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        with self.vigilar_sql():
            r = self.consultar()
        e = r["evaluacion"]
        self.assertEqual(e["mae"], 31 / 15)
        self.assertEqual(e["wape"], (31 / 301) * 100)
        self.assertAlmostEqual(e["rmse"], sqrt(7))
        self.assertEqual(self.datos.ejecutar("SELECT * FROM evaluacion"), guardada)
        self.comprobar_sin_estados_guardados(r)

    def test_fila_guardada_corrupta_no_interviene_en_el_calculo_actual(self):
        esperado = self.consultar()
        registro.registrar_evaluacion(self.ruta, 101)
        self.datos.corromper("UPDATE evaluacion SET mae='PRIVADO',wape=NULL,rmse=NULL")
        guardada = self.datos.ejecutar("SELECT * FROM evaluacion")
        with self.vigilar_sql():
            self.assertEqual(self.consultar(), esperado)
        self.assertEqual(self.datos.ejecutar("SELECT * FROM evaluacion"), guardada)

    def test_openapi_contrato_errores_cache_y_distincion_de_persistencia(self):
        esquema = self.cliente.get("/openapi.json").json()
        ruta = esquema["paths"][f"{PREFIJO}/{{pronostico_id}}"]
        self.assertEqual(set(ruta), {"get", "post"})
        operacion = ruta["get"]
        self.assertNotIn("requestBody", operacion)
        self.assertEqual([(p["name"], p["in"]) for p in operacion["parameters"]], [("pronostico_id", "path")])
        parametro = operacion["parameters"][0]
        self.assertTrue(parametro["required"])
        self.assertEqual(parametro["schema"]["type"], "integer")
        self.assertEqual(parametro["schema"]["maximum"], MAX_ID)
        self.assertEqual(parametro["schema"]["exclusiveMinimum"], 0)
        self.assertIn("ventas actualmente observadas", operacion["description"])
        self.assertIn("tabla evaluacion", operacion["description"])
        self.assertIn("sin modificar la fila guardada", operacion["description"])
        self.assertEqual(set(operacion["responses"]), {"200", "422", "404", "409", "503", "500"})
        schemas = esquema["components"]["schemas"]

        def resolver(ref):
            return schemas[ref["$ref"].rsplit("/", 1)[-1]]

        respuesta = resolver(operacion["responses"]["200"]["content"]["application/json"]["schema"])
        self.assertEqual(set(respuesta["properties"]), set(consulta.consultar_evaluacion(self.ruta, 101)))
        self.assertFalse(respuesta["additionalProperties"])
        evaluacion = resolver(respuesta["properties"]["evaluacion"])
        self.assertEqual(evaluacion["properties"]["detalles"]["minItems"], 15)
        self.assertEqual(evaluacion["properties"]["detalles"]["maxItems"], 15)
        for campo in ("mae", "wape", "rmse"):
            self.assertEqual({v["type"] for v in evaluacion["properties"][campo]["anyOf"]}, {"number", "null"})
        for codigo, contenido in operacion["responses"].items():
            self.assertEqual(contenido["headers"]["Cache-Control"]["schema"]["const"], "no-store")
            if codigo != "200":
                error = resolver(contenido["content"]["application/json"]["schema"])
                detalle = resolver(error["properties"]["detail"])
                self.assertEqual(set(detalle["properties"]), {"codigo", "mensaje"})

    def test_openapi_no_consulta_sqlite_ni_ejecuta_servicios(self):
        anterior = app.openapi_schema
        app.openapi_schema = None
        try:
            with patch.object(api, "consultar_evaluacion") as consultar, \
                    patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir para OpenAPI")):
                self.assertEqual(self.cliente.get("/openapi.json").status_code, 200)
                consultar.assert_not_called()
        finally:
            app.openapi_schema = anterior

    def test_routers_previos_y_health_conservados(self):
        rutas = app.openapi()["paths"]
        for router in (datasets.router, pronosticos.router):
            for anterior in router.routes:
                for metodo in anterior.methods:
                    self.assertIn(metodo.lower(), rutas[anterior.path])
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.cliente.get("/api/datasets/active/series").status_code, 200)
        self.assertEqual(self.cliente.get("/api/pronosticos/active/101").status_code, 200)
        self.assertEqual(self.cliente.put(f"{PREFIJO}/101").status_code, 405)

    def guardar(self, identificador=101, **opciones):
        r = self.cliente.post(f"{PREFIJO}/{identificador}", **opciones)
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        return r

    def comprobar_error_post(self, respuesta, estado, codigo):
        self.comprobar_error(respuesta, estado, codigo, errores=api.ERRORES_REGISTRO)

    def filas_guardadas(self):
        return self.datos.ejecutar("SELECT pronostico_id,mae,wape,rmse FROM evaluacion ORDER BY pronostico_id")

    @contextmanager
    def vigilar_registro_sql(self):
        abrir_real = sqlite3.connect
        trazas, aperturas, cierres = [], [], []

        class Escritura(sqlite3.Connection):
            def close(conexion):
                cierres.append(True)
                return super().close()

        def abrir(ruta_uri, **opciones):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=rw")
            self.assertTrue(opciones["uri"])
            c = abrir_real(ruta_uri, factory=Escritura, **opciones)
            aperturas.append(True)
            c.set_trace_callback(trazas.append)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir):
            yield trazas
        self.assertEqual(aperturas, [True])
        self.assertEqual(cierres, [True])

    def test_post_nuevo_201_una_fila_y_resultado_completo(self):
        esperado = consulta.consultar_evaluacion(self.ruta, 101)
        r = self.guardar()
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json(), {**esperado, "estado_registro": "creado", "hubo_escritura": True, "motivo": None})
        self.assertEqual(self.filas_guardadas(), [(101, 2, 10, sqrt(20 / 3))])
        self.assertNotIn("PRIVADO", r.text)

    def test_post_repetido_200_reutiliza_sin_dml(self):
        primero = self.guardar().json()
        contenido, guardada = self.ruta.read_bytes(), self.filas_guardadas()
        with self.vigilar_registro_sql() as trazas:
            segundo = self.guardar()
        self.assertEqual(segundo.status_code, 200, segundo.text)
        self.assertEqual(segundo.json(), {**primero, "estado_registro": "reutilizado", "hubo_escritura": False})
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in trazas))
        self.assertEqual(self.filas_guardadas(), guardada)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_sin_observaciones_200_no_guardado(self):
        self.datos.ejecutar("DELETE FROM venta_diaria")
        contenido = self.ruta.read_bytes()
        with self.vigilar_registro_sql() as trazas:
            r = self.guardar()
        self.assertEqual(r.status_code, 200, r.text)
        datos = r.json()
        self.assertEqual((datos["estado_registro"], datos["hubo_escritura"], datos["motivo"]),
                         ("no_guardado", False, "sin_observaciones"))
        self.assertEqual(datos["estado_cobertura"], "sin_observaciones")
        e = datos["evaluacion"]
        self.assertEqual(tuple(e[c] for c in ("mae", "wape", "rmse")), (None, None, None))
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (0, 15, 0))
        self.assertEqual(self.filas_guardadas(), [])
        self.assertFalse(any(s.lstrip().upper().startswith("INSERT") for s in trazas))
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_sin_observaciones_con_fila_antigua_409(self):
        self.guardar()
        guardada = self.filas_guardadas()
        self.datos.ejecutar("DELETE FROM venta_diaria")
        contenido = self.ruta.read_bytes()
        self.comprobar_error_post(self.guardar(), 409, "evaluacion_desactualizada")
        self.assertEqual(self.filas_guardadas(), guardada)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_cobertura_parcial_con_ausencias_interiores(self):
        faltan = (2, 8, 11)
        for paso in faltan:
            self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha=?", (fixtures.fecha_objetivo(fixtures.CORTES[0], paso),))
        esperado = consulta.consultar_evaluacion(self.ruta, 101)
        r = self.guardar()
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json(), {**esperado, "estado_registro": "creado", "hubo_escritura": True, "motivo": None})
        e = r.json()["evaluacion"]
        self.assertEqual((e["observaciones_evaluadas"], e["cobertura"]), (12, 12 / 15))
        self.assertEqual(e["fechas_ausentes"], [fixtures.fecha_objetivo(fixtures.CORTES[0], p) for p in faltan])
        self.assertEqual(self.filas_guardadas(), [(101, e["mae"], e["wape"], e["rmse"])])

    def test_post_ventas_cero_guarda_wape_sql_null(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        r = self.guardar()
        self.assertEqual(r.status_code, 201, r.text)
        e = r.json()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (2, None, 2))
        self.assertEqual(self.filas_guardadas(), [(101, 2, None, 2)])
        self.assertEqual(self.datos.ejecutar("SELECT typeof(wape) FROM evaluacion"), [("null",)])

    def test_post_cambio_de_ventas_conflicto_sin_sobrescribir(self):
        self.guardar()
        guardada = self.filas_guardadas()
        self.datos.ejecutar("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        contenido = self.ruta.read_bytes()
        self.comprobar_error_post(self.guardar(), 409, "evaluacion_desactualizada")
        self.assertEqual(self.filas_guardadas(), guardada)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_cambio_de_prediccion_conflicto_sin_sobrescribir(self):
        self.guardar()
        guardada = self.filas_guardadas()
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=9 WHERE paso=1")
        self.comprobar_error_post(self.guardar(), 409, "evaluacion_desactualizada")
        self.assertEqual(self.filas_guardadas(), guardada)

    def test_post_fila_previa_corrupta_conflicto_sin_filtraciones(self):
        self.guardar()
        for campo, valor in (("mae", "PRIVADO SELECT"), ("mae", -1), ("wape", None), ("rmse", None)):
            with self.subTest(campo=campo, valor=valor):
                self.datos.ejecutar("UPDATE evaluacion SET mae=?,wape=?,rmse=?", (2, 10, sqrt(20 / 3)))
                self.datos.corromper(f"UPDATE evaluacion SET {campo}=?", (valor,))
                guardada = self.filas_guardadas()
                contenido = self.ruta.read_bytes()
                self.comprobar_error_post(self.guardar(), 409, "evaluacion_desactualizada")
                self.assertEqual(self.filas_guardadas(), guardada)
                self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_pronostico_inexistente_o_inactivo(self):
        self.comprobar_error_post(self.guardar(999), 404, "pronostico_inexistente")
        self.datos.crear_pronostico(102, 202, serie=24)
        self.comprobar_error_post(self.guardar(102), 404, "pronostico_inactivo")
        self.assertEqual(self.filas_guardadas(), [])

    def test_post_sin_conjunto_activo(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.comprobar_error_post(self.guardar(), 409, "sin_conjunto_activo")

    def test_post_tipo_y_corte_no_autorizados_sin_leer_ventas(self):
        for tipo, corte, codigo in (("prueba", fixtures.CORTES[0], "tipo_no_evaluable"),
                                    ("futuro", "2017-07-31", "tipo_no_evaluable"),
                                    ("validacion", "2017-07-31", "corte_no_autorizado"),
                                    ("validacion", "2017-06-17", "corte_no_autorizado")):
            self.datos.ejecutar("UPDATE pronostico SET tipo=?,fecha_corte=?", (tipo, corte))
            with self.subTest(tipo=tipo, corte=corte), self.vigilar_registro_sql() as trazas:
                self.comprobar_error_post(self.guardar(), 422, codigo)
            self.assertFalse(any("FROM venta_diaria" in sql for sql in trazas))
        self.assertEqual(self.filas_guardadas(), [])

    def test_post_ids_invalidos_ascii_y_limite_sqlite(self):
        with patch.object(api, "registrar_evaluacion") as guardar, patch.object(api, "consultar_evaluacion") as consultar:
            for valor in ("0", "-1", "1.5", "101.0", "1e2", "true", "abc", "+101", " 101", "１２", "١", str(MAX_ID + 1)):
                with self.subTest(valor=valor):
                    self.comprobar_error_post(self.guardar(valor), 422, "pronostico_invalido")
            guardar.assert_not_called()
            consultar.assert_not_called()
        with patch.object(api, "registrar_evaluacion", wraps=registro.registrar_evaluacion) as guardar:
            self.comprobar_error_post(self.guardar(MAX_ID), 404, "pronostico_inexistente")
        guardar.assert_called_once_with(self.ruta, MAX_ID)

    def test_post_cuerpos_incluso_json_vacio_rechazados(self):
        with patch.object(api, "registrar_evaluacion") as guardar:
            for cuerpo in (b"{}", b"[]", b"null", b" ", b"PRIVADO JSON malformado"):
                with self.subTest(cuerpo=cuerpo):
                    self.comprobar_error_post(self.guardar(content=cuerpo, headers={"Content-Type": "application/json"}),
                                             422, "entrada_invalida")
            guardar.assert_not_called()

    def test_post_parametros_adicionales_rechazados(self):
        with patch.object(api, "registrar_evaluacion") as guardar:
            for parametros in ({"fecha_corte": fixtures.CORTES[0]}, {"otro": "PRIVADO"},
                               [("otro", "1"), ("otro", "2")], {"otro": ""}):
                with self.subTest(parametros=parametros):
                    self.comprobar_error_post(self.guardar(params=parametros), 422, "entrada_invalida")
            guardar.assert_not_called()

    def test_post_base_inexistente_corrupta_o_sin_esquema(self):
        ausente = self.ruta.parent / "PRIVADO_ausente.sqlite3"
        app.dependency_overrides[obtener_ruta_base] = lambda: ausente
        self.comprobar_error_post(self.guardar(), 503, "base_no_disponible")
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("corrupta.sqlite3", b"Esto no es SQLite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            app.dependency_overrides[obtener_ruta_base] = lambda: ruta
            self.comprobar_error_post(self.guardar(), 503, "base_no_disponible")
            self.assertEqual(ruta.read_bytes(), contenido)

    def test_post_error_de_persistencia_503_con_rollback(self):
        self.datos.ejecutar("CREATE TRIGGER fallo BEFORE INSERT ON evaluacion "
                            "BEGIN SELECT RAISE(ABORT,'PRIVADO INSERT C:/secreto'); END")
        contenido = self.ruta.read_bytes()
        with self.vigilar_registro_sql() as trazas:
            self.comprobar_error_post(self.guardar(), 503, "error_persistencia")
        self.assertIn("ROLLBACK", trazas)
        self.assertEqual(self.filas_guardadas(), [])
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_fallo_tras_insert_revierte_fila_pendiente(self):
        abrir_real = sqlite3.connect
        trazas = []

        class Fallida(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                cursor = super().execute(sql, parametros)
                if sql.startswith("INSERT INTO evaluacion"):
                    raise sqlite3.OperationalError("PRIVADO fallo tras INSERT")
                return cursor

        def abrir(ruta_uri, **opciones):
            c = abrir_real(ruta_uri, factory=Fallida, **opciones)
            c.set_trace_callback(trazas.append)
            return c

        contenido = self.ruta.read_bytes()
        with patch.object(sqlite3, "connect", side_effect=abrir):
            self.comprobar_error_post(self.guardar(), 503, "error_persistencia")
        self.assertIn("ROLLBACK", trazas)
        self.assertEqual(self.filas_guardadas(), [])
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_delega_exactamente_una_vez_en_b1_sin_consulta_publica_a2(self):
        with patch.object(api, "registrar_evaluacion", wraps=registro.registrar_evaluacion) as guardar, \
                patch.object(api, "consultar_evaluacion", side_effect=AssertionError("No consultar por GET")) as consultar:
            self.assertEqual(self.guardar().status_code, 201)
            guardar.assert_called_once_with(self.ruta, 101)
            consultar.assert_not_called()

    def test_post_modelo_pydantic_rechaza_datos_privados_y_tipos_invalidos(self):
        valido = {**consulta.consultar_evaluacion(self.ruta, 101),
                  "estado_registro": "creado", "hubo_escritura": True, "motivo": None}
        for campo, valor in (("PRIVADO", "ruta interna"), ("estado_registro", "PRIVADO"),
                              ("hubo_escritura", "true"), ("motivo", "PRIVADO")):
            incorrecto = {**deepcopy(valido), campo: valor}
            with self.subTest(campo=campo), patch.object(api, "registrar_evaluacion", return_value=incorrecto) as guardar:
                self.comprobar_error_post(self.guardar(), 500, "error_interno")
                guardar.assert_called_once_with(self.ruta, 101)
        self.assertEqual(self.filas_guardadas(), [])

    def test_post_mapeo_constante_y_proteccion_de_mensajes_internos(self):
        codigos = (("pronostico_invalido", 422), ("tipo_no_evaluable", 422), ("corte_no_autorizado", 422),
                   ("pronostico_inexistente", 404), ("pronostico_inactivo", 404), ("sin_conjunto_activo", 409),
                   ("datos_inconsistentes", 409), ("evaluacion_desactualizada", 409),
                   ("base_no_disponible", 503), ("error_persistencia", 503))
        for codigo, estado in codigos:
            with self.subTest(codigo=codigo), patch.object(api, "registrar_evaluacion", side_effect=
                    registro.ErrorRegistroEvaluacion(codigo, "PRIVADO SELECT C:/secreto Traceback")):
                self.comprobar_error_post(self.guardar(), estado, codigo)

    def test_post_codigos_desconocidos_y_errores_inesperados_500(self):
        errores = [registro.ErrorRegistroEvaluacion(c, "PRIVADO") for c in ("PRIVADO codigo", None, [], {})]
        errores += [RuntimeError("PRIVADO Traceback"), sqlite3.OperationalError("PRIVADO SQL"), OSError("PRIVADO ruta")]
        for error in errores:
            with self.subTest(error=type(error).__name__), patch.object(api, "registrar_evaluacion", side_effect=error) as guardar:
                self.comprobar_error_post(self.guardar(), 500, "error_interno")
                guardar.assert_called_once()

    def test_post_openapi_documenta_200_201_errores_y_esquema_heredado(self):
        esquema = self.cliente.get("/openapi.json").json()
        ruta = esquema["paths"][f"{PREFIJO}/{{pronostico_id}}"]
        self.assertEqual(set(ruta), {"get", "post"})
        operacion = ruta["post"]
        self.assertNotIn("requestBody", operacion)
        self.assertEqual([(p["name"], p["in"]) for p in operacion["parameters"]], [("pronostico_id", "path")])
        self.assertEqual(operacion["parameters"][0]["schema"]["maximum"], MAX_ID)
        self.assertEqual(set(operacion["responses"]), {"200", "201", "422", "404", "409", "503", "500"})
        self.assertIn("no_guardado", operacion["responses"]["200"]["description"])
        self.assertIn("evaluacion_desactualizada", operacion["responses"]["409"]["description"])
        self.assertIn("error_persistencia", operacion["responses"]["503"]["description"])
        self.assertIn("no acredita identidad histórica", operacion["description"])
        self.assertIn("No repetir solicitudes automáticamente", operacion["description"])
        schemas = esquema["components"]["schemas"]
        refs = [operacion["responses"][estado]["content"]["application/json"]["schema"] for estado in ("200", "201")]
        self.assertEqual(refs[0], refs[1])
        modelo = schemas[refs[0]["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(set(modelo["properties"]), set(consulta.consultar_evaluacion(self.ruta, 101))
                         | {"estado_registro", "hubo_escritura", "motivo"})
        self.assertFalse(modelo["additionalProperties"])
        self.assertEqual(modelo["properties"]["estado_registro"]["enum"], ["creado", "reutilizado", "no_guardado"])
        self.assertEqual(modelo["properties"]["hubo_escritura"]["type"], "boolean")
        self.assertEqual({x["type"] for x in modelo["properties"]["motivo"]["anyOf"]}, {"string", "null"})
        for contenido in operacion["responses"].values():
            self.assertEqual(contenido["headers"]["Cache-Control"]["schema"]["const"], "no-store")

    def test_post_get_posterior_conserva_contrato_ro_y_no_llama_b1(self):
        esperado = self.consultar()
        self.assertEqual(self.guardar().status_code, 201)
        guardada = self.filas_guardadas()
        with patch.object(api, "registrar_evaluacion", side_effect=AssertionError("No registrar en GET")) as guardar, \
                self.vigilar_sql():
            self.assertEqual(self.consultar(), esperado)
            guardar.assert_not_called()
        self.assertEqual(self.filas_guardadas(), guardada)

    def test_post_get_tras_cambiar_ventas_refleja_calculo_actual(self):
        self.guardar()
        guardada = self.filas_guardadas()
        self.datos.ejecutar("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        with self.vigilar_sql():
            actual = self.consultar()
        self.assertEqual(actual["evaluacion"]["mae"], 31 / 15)
        self.assertEqual(actual["evaluacion"]["wape"], (31 / 301) * 100)
        self.assertAlmostEqual(actual["evaluacion"]["rmse"], sqrt(7))
        self.assertEqual(self.filas_guardadas(), guardada)
        self.comprobar_error_post(self.guardar(), 409, "evaluacion_desactualizada")

    def test_post_y_get_no_ejecutan_modelos_predictivos(self):
        objetivos = ((generacion, "registrar_pronostico"), (generacion, "_calcular"),
                     (modelos_prediccion, "entrenar_modelo"), (modelos_prediccion, "crear_estimador"),
                     (modelos_prediccion, "pronosticar_recursivo"), (linea_base_prediccion, "pronosticar_serie"),
                     (pronosticos, "registrar_pronostico"))
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            self.assertEqual(self.guardar().status_code, 201)
            self.consultar()
            for doble in dobles:
                doble.assert_not_called()

    def test_post_importacion_arranque_y_openapi_no_ejecutan_registros(self):
        codigo = compile(Path(api.__file__).read_text(encoding="utf-8"), api.__file__, "exec")
        aislado = ModuleType("api_evaluaciones_c2_en_prueba")
        with patch.dict(sys.modules, {aislado.__name__: aislado}), \
                patch.object(registro, "registrar_evaluacion", side_effect=AssertionError("No registrar")) as guardar, \
                patch.object(consulta, "consultar_evaluacion", side_effect=AssertionError("No consultar")) as consultar, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir SQLite")):
            exec(codigo, aislado.__dict__)
            aplicacion = FastAPI()
            aplicacion.include_router(aislado.router)
            with TestClient(aplicacion) as cliente:
                self.assertEqual(cliente.get("/openapi.json").status_code, 200)
            guardar.assert_not_called()
            consultar.assert_not_called()
        self.assertEqual(self.filas_guardadas(), [])

    def test_post_fallo_de_respuesta_despues_de_commit_no_reintenta(self):
        def confirmar_y_fallar(*args):
            registro.registrar_evaluacion(*args)
            raise RuntimeError("PRIVADO fallo de respuesta tras confirmar")

        with patch.object(api, "registrar_evaluacion", side_effect=confirmar_y_fallar) as guardar:
            self.comprobar_error_post(self.guardar(), 500, "error_interno")
            guardar.assert_called_once_with(self.ruta, 101)
        self.assertEqual(self.filas_guardadas(), [(101, 2, 10, sqrt(20 / 3))])

    def test_post_errores_nuevos_no_cambian_mapeo_del_get(self):
        for error in (consulta.ErrorConsultaEvaluacion("evaluacion_desactualizada", "PRIVADO"),
                      consulta.ErrorConsultaEvaluacion("error_persistencia", "PRIVADO"),
                      registro.ErrorRegistroEvaluacion("base_no_disponible", "PRIVADO")):
            with self.subTest(error=type(error).__name__), patch.object(api, "consultar_evaluacion", side_effect=error):
                self.comprobar_error(self.obtener(), 500, "error_interno")

    def test_post_tres_cortes_autorizados(self):
        for i, corte in enumerate(fixtures.CORTES):
            with self.subTest(corte=corte):
                if i:
                    self.datos.crear_pronostico(101 + i, 201 + i, corte=corte)
                    self.datos.insertar_reales(corte=corte)
                r = self.guardar(101 + i)
                self.assertEqual(r.status_code, 201, r.text)
                self.assertEqual(r.json()["fecha_corte"], corte)
                self.assertEqual(tuple(r.json()["evaluacion"][c] for c in ("mae", "wape", "rmse")),
                                 (2, 10, sqrt(20 / 3)))
        self.assertEqual(len(self.filas_guardadas()), 3)

    def test_post_fracciones_y_precision_sin_redondeo(self):
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha>'2017-06-19'")
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=CASE WHEN paso=1 THEN 1 ELSE 0 END")
        r = self.guardar()
        self.assertEqual(r.status_code, 201, r.text)
        e = r.json()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (1 / 3, None, sqrt(1 / 3)))
        self.assertEqual(self.filas_guardadas(), [(101, e["mae"], None, e["rmse"])])

    def test_post_reutiliza_metricas_iguales_con_cobertura_actual_distinta(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=8")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=10")
        self.assertEqual(self.guardar().status_code, 201)
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        contenido = self.ruta.read_bytes()
        r = self.guardar()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["estado_registro"], r.json()["estado_cobertura"]), ("reutilizado", "parcial"))
        self.assertEqual(r.json()["evaluacion"]["observaciones_evaluadas"], 14)
        self.assertEqual(self.filas_guardadas(), [(101, 2, 25, 2)])
        self.assertEqual(self.ruta.read_bytes(), contenido)


if __name__ == "__main__":
    unittest.main()
