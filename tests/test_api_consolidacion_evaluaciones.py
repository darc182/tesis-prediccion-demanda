"""HU005-E2-B: HTTP con SQLite temporales y datos exclusivamente sintéticos."""

from contextlib import ExitStack
from copy import deepcopy
import json
from math import sqrt
from pathlib import Path
import sqlite3
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.api import evaluaciones as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.services import comparacion_evaluaciones as comparacion
from backend.app.services import configuracion_experimentos as config
from backend.app.services import consolidacion_evaluaciones as servicio
from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import registro_evaluaciones as registro
from tests import test_consolidacion_evaluaciones as fixtures


RUTA = "/api/evaluaciones/consolidacion/active"
GRUPOS = fixtures.GRUPOS


class ApiConsolidacionEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        # Composición de la fixture; no se heredan ni descubren sus 56 tests.
        self.datos = fixtures.ConsolidacionEvaluacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        self.app = FastAPI()
        self.app.include_router(api.router)
        # La dependencia queda sustituida antes de iniciar TestClient.
        self.app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(self.app))

    def obtener(self, grupos=GRUPOS, **opciones):
        parametros = opciones.pop("params", [("grupo", ",".join(str(i) for i in g)) for g in grupos])
        r = self.cliente.request("GET", RUTA, params=parametros, **opciones)
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        return r

    def consultar(self, grupos=GRUPOS):
        antes = self.ruta.read_bytes()
        r = self.obtener(grupos)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.ruta.read_bytes(), antes)
        self.assertNotIn("PRIVADO", r.text)
        datos = r.json()
        self.assertEqual(api.ConsolidacionEvaluaciones.model_validate(datos).model_dump(), datos)
        self.assertEqual([tuple(e["pronostico_id"] for e in g["evaluaciones"]) for g in datos["resultados"]],
                         [tuple(g) for g in grupos])
        self.assertTrue({"ganador", "ranking", "mejor_modelo", "exactitud", "registrado", "vigente"}.isdisjoint(datos))
        return datos

    def comprobar_error(self, r, codigo, estado=None, errores=None):
        errores = api.ERRORES_CONSOLIDACION if errores is None else errores
        esperado, mensaje = errores[codigo]
        self.assertEqual(r.status_code, esperado if estado is None else estado, r.text)
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        self.assertEqual(r.json(), {"detail": {"codigo": codigo, "mensaje": mensaje}})
        for secreto in ("PRIVADO", "SELECT", "Traceback", str(self.ruta)):
            self.assertNotIn(secreto, r.text)

    def error(self, codigo, grupos=GRUPOS, **opciones):
        antes = self.ruta.read_bytes()
        self.comprobar_error(self.obtener(grupos, **opciones), codigo)
        self.assertEqual(self.ruta.read_bytes(), antes)

    def test_dos_grupos_respuesta_e2a_integra_y_metricas_exactas(self):
        esperado = servicio.consolidar_evaluaciones(self.ruta, GRUPOS)
        r = self.consultar()
        self.assertEqual(r, esperado)
        c = r["resultados"][0]["consolidado"]
        self.assertEqual((c["mae"], c["wape"], c["observaciones_evaluadas"], c["cobertura"]), (2, 10, 45, 1))
        self.assertAlmostEqual(c["rmse"], sqrt(20 / 3))
        self.assertEqual((c["suma_errores_absolutos"], c["suma_ventas_reales"]), (90, 900))
        self.assertEqual(set(r), set(esperado))

    def test_tres_grupos_tres_tecnicas_orden_solicitado(self):
        tercero = self.datos.crear_grupo(3, "hist_gradient_boosting")
        r = self.consultar([tercero, GRUPOS[1], GRUPOS[0]])
        self.assertEqual([g["tecnica"] for g in r["resultados"]], ["hist_gradient_boosting", "random_forest", config.BASE])

    def test_dos_configuraciones_de_la_misma_tecnica(self):
        efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "parametros": {"n_estimators": 200}})
        tercero = self.datos.crear_grupo(3, "random_forest", efectiva=efectiva)
        r = self.consultar([tercero, GRUPOS[1]])
        self.assertEqual([g["configuracion"]["parametros"]["n_estimators"] for g in r["resultados"]], [200, 100])

    def test_tres_configuraciones_de_la_misma_tecnica(self):
        grupos = [GRUPOS[1]]
        for numero, arboles in ((3, 200), (4, 300)):
            efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "parametros": {"n_estimators": arboles}})
            grupos.append(self.datos.crear_grupo(numero, "random_forest", efectiva=efectiva))
        self.assertEqual(len(self.consultar(grupos)["resultados"]), 3)

    def test_delegacion_unica_grupos_enteros_sin_conexion_del_router(self):
        esperado = servicio.consolidar_evaluaciones(self.ruta, [GRUPOS[1], GRUPOS[0]])
        with patch.object(api, "consolidar_evaluaciones", return_value=esperado) as consolidar, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("Conexión del router")), \
                patch.object(api, "consultar_evaluacion", side_effect=AssertionError("Consulta individual")), \
                patch.object(api, "comparar_evaluaciones", side_effect=AssertionError("Comparación individual")), \
                patch.object(api, "registrar_evaluacion", side_effect=AssertionError("Registro")):
            self.assertEqual(self.consultar([GRUPOS[1], GRUPOS[0]]), esperado)
            consolidar.assert_called_once_with(self.ruta, [[201, 202, 203], [101, 102, 103]])
        self.assertTrue(all(type(i) is int for grupo in consolidar.call_args.args[1] for i in grupo))

    def test_identidad_configuracion_completa_y_tipos_json_f2(self):
        r = self.consultar()
        for g in r["resultados"]:
            self.assertEqual(g["configuracion"], config.configuracion_predeterminada(g["tecnica"]))
            self.assertEqual(g["identidad_configuracion"], config.identidad_configuracion(g["configuracion"]))
        p = r["resultados"][1]["configuracion"]["parametros"]
        for clave, tipo in (("bootstrap", bool), ("n_estimators", int), ("max_features", float), ("criterion", str)):
            self.assertIs(type(p[clave]), tipo)
        self.assertIn("max_depth", p)
        self.assertIsNone(p["max_depth"])

    def test_versiones_sin_inventar_null_para_linea_base(self):
        r = self.consultar()
        self.assertEqual(r["resultados"][0]["versiones_ejecucion"], {"python": "3.12.10", "pandas": "2.2.3"})
        self.assertEqual(r["resultados"][1]["versiones_ejecucion"],
                         {"python": "3.12.10", "pandas": "2.2.3", "scikit_learn": "1.6.1"})

    def test_tres_evaluaciones_a2_con_quince_detalles_y_cortes(self):
        r = self.consultar()
        for grupo in r["resultados"]:
            self.assertEqual([e["fecha_corte"] for e in grupo["evaluaciones"]], list(fixtures.CORTES))
            for e in grupo["evaluaciones"]:
                self.assertEqual(e, consulta.consultar_evaluacion(self.ruta, e["pronostico_id"]))
                self.assertEqual([d["paso"] for d in e["evaluacion"]["detalles"]], list(range(1, 16)))

    def test_wape_global_no_es_promedio_y_referencia_decimal(self):
        self.datos.datos_numericos([[1.0] * 15, [10.0] * 15, [100.0] * 15],
                                   [[2.0] * 15, [11.0] * 15, [101.0] * 15])
        g = self.consultar()["resultados"][0]
        c = g["consolidado"]
        self.assertEqual((c["mae"], c["rmse"]), (1, 1))
        self.assertAlmostEqual(c["wape"], 100 / 37)
        self.assertNotAlmostEqual(c["wape"], (100 + 10 + 1) / 3)
        for nombre, esperado in zip(("mae", "wape", "rmse"), fixtures.referencia_decimal(g["evaluaciones"])):
            self.assertEqual(c[nombre], esperado)

    def test_cobertura_parcial_y_ausencias_interiores(self):
        self.datos.sql("DELETE FROM venta_diaria WHERE fecha IN ('2017-06-24','2017-07-09','2017-07-24')")
        r = self.consultar()
        self.assertEqual((r["estado_cobertura"], r["observaciones_evaluadas_total"], r["cobertura_total"]), ("parcial", 42, 42 / 45))
        self.assertEqual([x["fechas_ausentes"] for x in r["fechas_ausentes_por_corte"]],
                         [["2017-06-24"], ["2017-07-09"], ["2017-07-24"]])
        d = r["resultados"][0]["evaluaciones"][0]["evaluacion"]["detalles"][7]
        self.assertIsNone(d["venta_real"])
        self.assertIsNone(d["error_absoluto"])

    def test_un_corte_sin_observaciones(self):
        self.datos.sql("DELETE FROM venta_diaria WHERE fecha>'2017-07-01' AND fecha<='2017-07-16'")
        r = self.consultar()
        self.assertEqual(r["observaciones_evaluadas_total"], 30)
        for g in r["resultados"]:
            self.assertEqual([g["evaluaciones"][1]["evaluacion"][k] for k in ("mae", "wape", "rmse")], [None] * 3)
            self.assertIsNotNone(g["consolidado"]["mae"])

    def test_tres_cortes_sin_observaciones_null_conservados(self):
        self.datos.sql("DELETE FROM venta_diaria")
        r = self.consultar()
        self.assertEqual((r["observaciones_evaluadas_total"], r["cobertura_total"], r["estado_cobertura"]), (0, 0, "sin_observaciones"))
        for g in r["resultados"]:
            for k in ("mae", "wape", "rmse"):
                self.assertIn(k, g["consolidado"])
                self.assertIsNone(g["consolidado"][k])
            self.assertEqual(g["consolidado"]["fechas_sin_registro"], 45)

    def test_reales_cero_wape_null_mae_rmse_validos_y_ausencia_distinta(self):
        self.datos.datos_numericos([[0.0] * 15] * 3, [[2.0] * 15] * 3)
        self.datos.sql("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        r = self.consultar()
        self.assertEqual(r["observaciones_evaluadas_total"], 44)
        c = r["resultados"][0]["consolidado"]
        self.assertEqual((c["mae"], c["wape"], c["rmse"]), (2, None, 2))

    def test_un_corte_con_wape_null_no_se_excluye_del_global(self):
        self.datos.datos_numericos([[0.0] * 15, [1.0] * 15, [1.0] * 15], [[2.0] * 15] * 3)
        g = self.consultar()["resultados"][0]
        self.assertIsNone(g["evaluaciones"][0]["evaluacion"]["wape"])
        self.assertEqual(g["consolidado"]["wape"], 200)

    def test_wape_subnormal_positivo_sin_redondear(self):
        self.datos.datos_numericos([[1e307] + [0.0] * 14] * 3, [[1e307, 1e-18] + [0.0] * 13] * 3)
        for g in self.consultar()["resultados"]:
            self.assertGreater(g["consolidado"]["wape"], 0)
            self.assertEqual(g["consolidado"]["wape"], fixtures.referencia_decimal(g["evaluaciones"])[1])

    def test_fraccionarios_wape_superior_a_cien_y_detalles(self):
        self.datos.datos_numericos([[0.125, 0.25, 0.5] * 5] * 3, [[1.125, 2.25, 3.5] * 5] * 3)
        g = self.consultar()["resultados"][0]
        self.assertGreater(g["consolidado"]["wape"], 100)
        self.assertEqual(g["evaluaciones"][0]["evaluacion"]["detalles"][0]["venta_real"], 0.125)
        for nombre, esperado in zip(("mae", "wape", "rmse"), fixtures.referencia_decimal(g["evaluaciones"])):
            self.assertAlmostEqual(g["consolidado"][nombre], esperado)

    def test_cantidad_grupos_invalida_sin_delegar(self):
        for parametros in ([], [("grupo", "101,102,103")], [("grupo", "101,102,103")] * 4):
            with self.subTest(parametros=parametros), patch.object(api, "consolidar_evaluaciones") as consolidar:
                self.error("entrada_invalida", params=parametros)
                consolidar.assert_not_called()

    def test_cantidad_ids_por_grupo_y_formatos_no_admitidos(self):
        for valor in ("", "101", "101,102", "101,102,103,104", "101;102;103", '[[101,102,103]]', '{"ids":[101,102,103]}'):
            with self.subTest(valor=valor), patch.object(api, "consolidar_evaluaciones") as consolidar:
                r = self.obtener(params=[("grupo", valor), ("grupo", "201,202,203")])
                self.assertEqual(r.status_code, 422)
                self.assertIn(r.json()["detail"]["codigo"], ("entrada_invalida", "pronostico_invalido"))
                consolidar.assert_not_called()

    def test_ids_invalidos_ascii_rango_espacios_signos_y_vacios(self):
        for valor in ("0", "-1", "+101", "1.0", "1e2", "true", "NaN", "Infinity", "١٠١", "１０１", "²", " 101", "101 ",
                      "101\n", "101\t", "", str(2**63), "9" * 5000):
            with self.subTest(valor=valor[:30]), patch.object(api, "consolidar_evaluaciones") as consolidar, \
                    patch.object(sqlite3, "connect", side_effect=AssertionError("Abrir entrada inválida")):
                self.error("pronostico_invalido", params=[("grupo", f"{valor},102,103"), ("grupo", "201,202,203")])
                consolidar.assert_not_called()

    def test_duplicados_dentro_entre_grupos_y_ceros_iniciales(self):
        for grupos in ([(101, 101, 103), GRUPOS[1]], [GRUPOS[0], (201, 202, 101)]):
            with patch.object(api, "consolidar_evaluaciones") as consolidar:
                self.error("entrada_invalida", grupos)
                consolidar.assert_not_called()
        self.error("entrada_invalida", params=[("grupo", "00101,102,103"), ("grupo", "201,202,101")])

    def test_ceros_iniciales_validos_preservan_valor_y_orden(self):
        esperado = servicio.consolidar_evaluaciones(self.ruta, GRUPOS)
        with patch.object(api, "consolidar_evaluaciones", return_value=esperado) as consolidar:
            r = self.obtener(params=[("grupo", "00101,00102,00103"), ("grupo", "201,202,203")])
            self.assertEqual(r.status_code, 200)
            consolidar.assert_called_once_with(self.ruta, [[101, 102, 103], [201, 202, 203]])

    def test_maximo_sqlite_admitido_sin_conversion_silenciosa(self):
        with patch.object(api, "consolidar_evaluaciones", side_effect=servicio.ErrorConsolidacionEvaluaciones("pronostico_inexistente")) as consolidar:
            self.error("pronostico_inexistente", [GRUPOS[0], (201, 202, api.MAX_ENTERO_SQLITE)])
            consolidar.assert_called_once_with(self.ruta, [[101, 102, 103], [201, 202, 2**63 - 1]])

    def test_parametros_desconocidos_y_grupos_json_no_se_interpretan(self):
        for extra in (("otro", "1"), ("pronostico_id", "101"), ("Grupo", "101,102,103"), ("grupo[]", "101,102,103")):
            with patch.object(api, "consolidar_evaluaciones") as consolidar:
                self.error("entrada_invalida", params=[("grupo", "101,102,103"), ("grupo", "201,202,203"), extra])
                consolidar.assert_not_called()

    def test_cuerpo_http_incluso_json_vacio_o_bytes_rechazado(self):
        for contenido in (b"{}", b"[]", b"null", b" ", b"PRIVADO", b'{"grupo":[]}'):
            with patch.object(api, "consolidar_evaluaciones") as consolidar:
                self.error("entrada_invalida", content=contenido, headers={"Content-Type": "application/json"})
                consolidar.assert_not_called()

    def test_pronostico_inexistente_o_inactivo(self):
        self.error("pronostico_inexistente", [GRUPOS[0], (201, 202, 999999)])
        self.datos.sql("UPDATE experimento SET serie_id=24 WHERE id=1203")
        self.error("pronostico_inactivo")

    def test_sin_conjunto_activo(self):
        self.datos.sql("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_tipos_prueba_futuro_y_corte_reservado_antes_de_ventas(self):
        for tipo, corte, codigo in (("prueba", "2017-06-16", "tipo_no_evaluable"), ("futuro", "2017-06-16", "tipo_no_evaluable"),
                                    ("validacion", "2017-07-31", "corte_no_autorizado")):
            self.datos.sql("UPDATE pronostico SET tipo=?,fecha_corte=? WHERE id=101", (tipo, corte))
            with self.datos.vigilar() as llamadas:
                self.error(codigo)
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_cortes_desordenados_sin_resultados_parciales(self):
        self.error("pronosticos_no_comparables", [(102, 101, 103), GRUPOS[1]])

    def test_series_distintas(self):
        self.datos.sql("UPDATE experimento SET serie_id=84 WHERE id=1203")
        self.error("pronosticos_no_comparables")

    def test_identidad_no_equivalente_dentro_del_grupo(self):
        def cambiar(r):
            r["parametros"]["n_estimators"] = r["configuracion_predictiva"]["parametros"]["n_estimators"] = 200
        self.datos.cambiar_registro(202, cambiar)
        self.error("configuraciones_no_equivalentes")

    def test_configuraciones_repetidas_entre_grupos(self):
        tercero = self.datos.crear_grupo(3, "random_forest")
        self.error("configuraciones_duplicadas", [GRUPOS[1], tercero])

    def test_versiones_incompatibles_o_no_acreditables(self):
        self.datos.cambiar_registro(202, lambda r: r["versiones"].update(python="3.13.1"))
        self.error("versiones_incompatibles")
        self.datos.cambiar_registro(202, lambda r: r["versiones"].pop("pandas"))
        self.error("versiones_incompatibles")

    def test_json_ambiguo_o_configuracion_historica_inconsistente(self):
        texto = self.datos.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
        for invalido in (texto[:-1] + ',"tecnica":"random_forest"}', '{"version_registro":77}'):
            self.datos.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (invalido,))
            self.error("datos_inconsistentes")

    def test_registro_historico_reconocido_no_se_reescribe(self):
        efectiva = config.configuracion_predeterminada("random_forest")
        self.datos.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202",
                       (json.dumps(fixtures.registro(efectiva, fixtures.CORTES[1], version=1)),))
        self.consultar()

    def test_detalles_corruptos(self):
        self.datos.sql("DELETE FROM detalle_pronostico WHERE pronostico_id=203 AND paso=15")
        self.error("datos_inconsistentes")

    def test_observaciones_incompatibles_mismas_fechas(self):
        contador = []

        def alterar(parametros, filas):
            contador.append(True)
            if len(contador) == 4:
                filas[0]["sales"] += 1
            return filas

        with self.datos.vigilar(alterar):
            self.error("pronosticos_no_comparables")

    def test_observaciones_incompatibles_igual_cantidad_distintas_fechas(self):
        self.datos.sql("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        contador = []

        def alterar(parametros, filas):
            contador.append(True)
            if len(contador) == 4:
                filas[0]["fecha"] = "2017-06-24"
            return filas

        with self.datos.vigilar(alterar):
            self.error("pronosticos_no_comparables")

    def test_resultado_numerico_no_representable(self):
        self.datos.datos_numericos([[0.0] * 15] * 3, [[7e307] + [0.0] * 14] * 3)
        self.error("resultado_no_finito")

    def test_base_inexistente_danada_o_sin_esquema(self):
        ausente = self.ruta.parent / "ausente.sqlite3"
        rutas = [ausente]
        for nombre, contenido in (("danada.sqlite3", b"no es sqlite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            rutas.append(ruta)
        for ruta in rutas:
            self.app.dependency_overrides[obtener_ruta_base] = lambda ruta=ruta: ruta
            self.comprobar_error(self.obtener(), "base_no_disponible")
        self.assertFalse(ausente.exists())

    def test_catalogo_completo_codigos_publicos_y_mensajes_internos_descartados(self):
        codigos = {
            "entrada_invalida": 422, "pronostico_invalido": 422, "tipo_no_evaluable": 422, "corte_no_autorizado": 422,
            "pronostico_inexistente": 404, "pronostico_inactivo": 404, "sin_conjunto_activo": 409,
            "configuraciones_no_equivalentes": 409, "configuraciones_duplicadas": 409, "versiones_incompatibles": 409,
            "pronosticos_no_comparables": 409, "datos_inconsistentes": 409, "resultado_no_finito": 409, "base_no_disponible": 503,
        }
        for codigo, estado in codigos.items():
            error = servicio.ErrorConsolidacionEvaluaciones(codigo)
            error.args = ("PRIVADO SELECT C:/secreto Traceback",)
            with self.subTest(codigo=codigo), patch.object(api, "consolidar_evaluaciones", side_effect=error):
                self.comprobar_error(self.obtener(), codigo, estado)

    def test_codigos_desconocidos_y_errores_inesperados_genericos(self):
        for codigo in ("PRIVADO", None, [], {}):
            error = servicio.ErrorConsolidacionEvaluaciones("datos_inconsistentes")
            error.codigo = codigo
            with patch.object(api, "consolidar_evaluaciones", side_effect=error):
                self.error("error_interno")
        for error in (RuntimeError("PRIVADO SELECT"), sqlite3.OperationalError("PRIVADO"),
                      consulta.ErrorConsultaEvaluacion("pronostico_inexistente", "PRIVADO"),
                      comparacion.ErrorComparacionEvaluaciones("datos_inconsistentes"),
                      registro.ErrorRegistroEvaluacion("base_no_disponible", "PRIVADO")):
            with patch.object(api, "consolidar_evaluaciones", side_effect=error):
                self.error("error_interno")

    def test_error_inesperado_de_dependencia(self):
        def fallar():
            raise RuntimeError("PRIVADO SELECT C:/secreto")
        self.app.dependency_overrides[obtener_ruta_base] = fallar
        self.error("error_interno")

    def test_modelos_estrictos_rechazan_extras_tipos_no_finitos_y_campos_ausentes(self):
        original = servicio.consolidar_evaluaciones(self.ruta, GRUPOS)
        cambios = (
            lambda r: r.update(privado="PRIVADO"),
            lambda r: r.update(detail={"codigo": "datos_inconsistentes", "mensaje": "PRIVADO SELECT"}),
            lambda r: r.update(conjunto_id="7"),
            lambda r: r.update(observaciones_evaluadas_total=True),
            lambda r: r.update(cobertura_total=float("nan")),
            lambda r: r.update(horizonte_por_corte=30),
            lambda r: r.update(fechas_esperadas_total=44),
            lambda r: r["resultados"].pop(),
            lambda r: r["resultados"][0]["evaluaciones"].pop(),
            lambda r: r["resultados"][0]["evaluaciones"][0]["evaluacion"]["detalles"].pop(),
            lambda r: r["resultados"][0]["evaluaciones"][0].update(privado="PRIVADO"),
            lambda r: r["resultados"][0]["consolidado"].update(mae=float("inf")),
            lambda r: r["resultados"][0]["consolidado"].pop("wape"),
            lambda r: r["resultados"][1]["configuracion"]["parametros"].update(max_features=float("nan")),
            lambda r: r["resultados"][1]["configuracion"]["parametros"].update(max_features=["PRIVADO"]),
            lambda r: r["resultados"][0]["versiones_ejecucion"].update(sistema="PRIVADO"),
            lambda r: r["resultados"][0]["versiones_ejecucion"].pop("python"),
            lambda r: r["resultados"][0].update(identidad_configuracion="PRIVADO"),
        )
        for indice, cambiar in enumerate(cambios):
            invalido = deepcopy(original)
            cambiar(invalido)
            with self.subTest(caso=indice):
                with self.assertRaises(ValidationError):
                    api.ConsolidacionEvaluaciones.model_validate(invalido)
                with patch.object(api, "consolidar_evaluaciones", return_value=invalido):
                    self.error("error_interno")
        # Un objeto Response evita la validación automática de FastAPI; el
        # endpoint debe validar el contrato del servicio antes de devolverlo.
        for invalido in (None, [], "PRIVADO", api.JSONResponse(status_code=409, content={
            "detail": {"codigo": "datos_inconsistentes", "mensaje": "PRIVADO SELECT C:/secreto"}
        })):
            with patch.object(api, "consolidar_evaluaciones", return_value=invalido):
                self.error("error_interno")
        for modelo in (api.ConfiguracionConsolidacion, api.VersionesBaseConsolidacion, api.VersionesRegresorConsolidacion,
                       api.RMSEEscaladoConsolidacion, api.MetricasConsolidadas, api.AusenciasCorteConsolidacion,
                       api.ResultadoConsolidacion, api.ConsolidacionEvaluaciones):
            self.assertEqual(modelo.model_config, {"extra": "forbid", "strict": True, "allow_inf_nan": False})

    def test_conexion_unica_ro_begin_ventas_acotadas_y_sin_escrituras(self):
        with self.datos.vigilar() as llamadas, patch.object(api, "consolidar_evaluaciones", wraps=servicio.consolidar_evaluaciones) as consolidar:
            self.consultar()
            consolidar.assert_called_once_with(self.ruta, [[101, 102, 103], [201, 202, 203]])
        self.assertEqual([sql for sql, _, _ in llamadas if sql == "BEGIN"], ["BEGIN"])
        self.assertTrue(all(activa for sql, _, activa in llamadas if sql != "BEGIN"))
        ventas = [(sql, p) for sql, p, _ in llamadas if "FROM venta_diaria" in sql]
        self.assertEqual(len(ventas), 6)
        for indice, (sql, parametros) in enumerate(ventas):
            corte = fixtures.CORTES[indice % 3]
            self.assertEqual(parametros, (83, corte, fixtures.fecha(corte, 15)))
            self.assertIn("fecha > ? AND fecha <= ?", sql)
            self.assertLessEqual(parametros[2], "2017-07-31")

    def test_tres_grupos_una_conexion_nueve_lecturas_de_ventas(self):
        tercero = self.datos.crear_grupo(3, "hist_gradient_boosting")
        with self.datos.vigilar() as llamadas:
            self.consultar([*GRUPOS, tercero])
        self.assertEqual(sum("FROM venta_diaria" in sql for sql, _, _ in llamadas), 9)
        self.assertEqual(sum(sql == "BEGIN" for sql, _, _ in llamadas), 1)

    def test_fila_evaluacion_guardada_no_se_lee_ni_modifica(self):
        self.datos.sql("INSERT INTO evaluacion VALUES (101,999999,999999,999999)")
        with self.datos.vigilar():
            self.assertEqual(self.consultar()["resultados"][0]["consolidado"]["mae"], 2)
        self.assertEqual(self.datos.sql("SELECT mae,wape,rmse FROM evaluacion WHERE pronostico_id=101"), [(999999.0,) * 3])

    def test_importacion_arranque_y_openapi_sin_operaciones_automaticas(self):
        codigo = compile(Path(api.__file__).read_text(encoding="utf-8"), api.__file__, "exec")
        aislado = ModuleType("api_consolidacion_aislada")
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {aislado.__name__: aislado}))
            dobles = [stack.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("Operación automática")))
                      for modulo, nombre in ((servicio, "consolidar_evaluaciones"), (consulta, "consultar_evaluacion"),
                                              (comparacion, "comparar_evaluaciones"), (registro, "registrar_evaluacion"))]
            stack.enter_context(patch.object(sqlite3, "connect", side_effect=AssertionError("SQL automático")))
            exec(codigo, aislado.__dict__)
            aplicacion = FastAPI()
            aplicacion.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
            aplicacion.include_router(aislado.router)
            with TestClient(aplicacion) as cliente:
                self.assertEqual(cliente.get("/openapi.json").status_code, 200)
            for doble in dobles:
                doble.assert_not_called()

    def test_get_sin_entrenamiento_generacion_registro_ni_consultas_individuales(self):
        from backend.app.services import registro_pronosticos as generacion
        from scripts import linea_base_prediccion, modelos_prediccion
        with ExitStack() as stack:
            for modulo, nombre in ((generacion, "registrar_pronostico"), (generacion, "_calcular"),
                                    (modelos_prediccion, "entrenar_modelo"), (modelos_prediccion, "pronosticar_recursivo"),
                                    (linea_base_prediccion, "pronosticar_serie"), (api, "registrar_evaluacion"),
                                    (registro, "registrar_evaluacion"), (api, "consultar_evaluacion"), (api, "comparar_evaluaciones")):
                stack.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("Operación prohibida")))
            self.consultar()

    def test_aislamiento_de_queries_c1_c2_d2_y_consolidacion(self):
        with patch.object(api, "consolidar_evaluaciones") as consolidar:
            for metodo in ("GET", "POST"):
                r = self.cliente.request(metodo, "/api/evaluaciones/active/101", params={"grupo": "101,102,103"})
                self.comprobar_error(r, "entrada_invalida", errores=api.ERRORES_REGISTRO if metodo == "POST" else api.ERRORES)
            r = self.cliente.get("/api/evaluaciones/comparacion/active", params={"grupo": "101,102,103"})
            self.comprobar_error(r, "entrada_invalida", errores=api.ERRORES_COMPARACION)
            self.error("entrada_invalida", params=[("pronostico_id", "101"), ("pronostico_id", "201")])
            consolidar.assert_not_called()

    def test_c1_c2_y_d2_conservan_respuestas_y_delegacion(self):
        individual = consulta.consultar_evaluacion(self.ruta, 101)
        comparada = comparacion.comparar_evaluaciones(self.ruta, [201, 101])
        guardada = {**individual, "estado_registro": "creado", "hubo_escritura": True, "motivo": None}
        antes = self.ruta.read_bytes()
        with patch.object(api, "consolidar_evaluaciones") as consolidar, \
                patch.object(api, "consultar_evaluacion", return_value=individual) as consultar, \
                patch.object(api, "registrar_evaluacion", return_value=guardada) as guardar, \
                patch.object(api, "comparar_evaluaciones", return_value=comparada) as comparar:
            r = self.cliente.get("/api/evaluaciones/active/101")
            self.assertEqual((r.status_code, r.json()), (200, individual))
            r = self.cliente.post("/api/evaluaciones/active/101")
            self.assertEqual((r.status_code, r.json()), (201, guardada))
            r = self.cliente.get("/api/evaluaciones/comparacion/active", params=[("pronostico_id", "201"), ("pronostico_id", "101")])
            self.assertEqual((r.status_code, r.json()), (200, comparada))
            consultar.assert_called_once_with(self.ruta, 101)
            guardar.assert_called_once_with(self.ruta, 101)
            comparar.assert_called_once_with(self.ruta, [201, 101])
            consolidar.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), antes)

    def test_error_e2a_fuera_de_nueva_ruta_es_generico_de_operacion_original(self):
        for metodo, nombre, url, opciones, errores in (
            ("GET", "consultar_evaluacion", "/api/evaluaciones/active/101", {}, api.ERRORES),
            ("POST", "registrar_evaluacion", "/api/evaluaciones/active/101", {}, api.ERRORES_REGISTRO),
            ("GET", "comparar_evaluaciones", "/api/evaluaciones/comparacion/active",
             {"params": [("pronostico_id", "101"), ("pronostico_id", "201")]}, api.ERRORES_COMPARACION),
        ):
            with patch.object(api, nombre, side_effect=servicio.ErrorConsolidacionEvaluaciones("versiones_incompatibles")):
                self.comprobar_error(self.cliente.request(metodo, url, **opciones), "error_interno", errores=errores)

    def test_openapi_contrato_parametros_errores_modelos_y_cache(self):
        with patch.object(api, "consolidar_evaluaciones") as consolidar, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("SQL en OpenAPI")):
            esquema = self.cliente.get("/openapi.json").json()
            consolidar.assert_not_called()
        ruta = esquema["paths"][RUTA]
        self.assertEqual(set(ruta), {"get"})
        op = ruta["get"]
        self.assertNotIn("requestBody", op)
        self.assertEqual(len(op["parameters"]), 1)
        parametro = op["parameters"][0]
        self.assertEqual((parametro["name"], parametro["in"], parametro["required"]), ("grupo", "query", True))
        self.assertEqual((parametro["schema"]["type"], parametro["schema"]["minItems"], parametro["schema"]["maxItems"]), ("array", 2, 3))
        self.assertEqual(parametro["schema"]["items"]["type"], "string")
        for corte in fixtures.CORTES:
            self.assertIn(corte, parametro["description"])
        self.assertEqual(set(op["responses"]), {"200", "422", "404", "409", "503", "500"})
        for texto in ("solo lectura", "parcial", "null", "ganadores", "tabla evaluacion"):
            self.assertIn(texto, op["description"])
        for contenido in op["responses"].values():
            self.assertEqual(contenido["headers"]["Cache-Control"]["schema"]["const"], "no-store")
        for codigo, (estado, _) in api.ERRORES_CONSOLIDACION.items():
            self.assertIn(codigo, op["responses"][str(estado)]["description"])
        schemas = esquema["components"]["schemas"]
        modelo = schemas[op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(set(modelo["properties"]), set(servicio.consolidar_evaluaciones(self.ruta, GRUPOS)))
        self.assertFalse(modelo["additionalProperties"])
        self.assertEqual(modelo["properties"]["serie"]["$ref"], "#/components/schemas/SerieEvaluacion")
        resultado = schemas["ResultadoConsolidacion"]
        evaluaciones = resultado["properties"]["evaluaciones"]
        self.assertEqual((evaluaciones["minItems"], evaluaciones["maxItems"]), (3, 3))
        self.assertEqual(evaluaciones["items"]["$ref"], "#/components/schemas/ConsultaEvaluacion")
        parametros = schemas["ConfiguracionConsolidacion"]["properties"]["parametros"]["additionalProperties"]["anyOf"]
        self.assertEqual({p["type"] for p in parametros}, {"integer", "number", "boolean", "string", "null"})
        self.assertNotIn("scikit_learn", schemas["VersionesBaseConsolidacion"]["properties"])
        self.assertIn("scikit_learn", schemas["VersionesRegresorConsolidacion"]["required"])
        for nombre in ("ConsolidacionEvaluaciones", "ResultadoConsolidacion", "MetricasConsolidadas", "RMSEEscaladoConsolidacion",
                       "ConfiguracionConsolidacion", "VersionesBaseConsolidacion", "VersionesRegresorConsolidacion"):
            self.assertFalse(schemas[nombre]["additionalProperties"])

    def test_openapi_de_operaciones_previas_conserva_contratos(self):
        esquema = self.app.openapi()
        for ruta, metodos in (("/api/evaluaciones/active/{pronostico_id}", {"get", "post"}),
                              ("/api/evaluaciones/comparacion/active", {"get"})):
            self.assertEqual(set(esquema["paths"][ruta]), metodos)
        rutas = {r.name: r.path for r in api.router.routes}
        self.assertEqual(rutas["consolidacion_actual"], RUTA)
        self.assertEqual(rutas["comparacion_actual"], "/api/evaluaciones/comparacion/active")


if __name__ == "__main__":
    unittest.main()
