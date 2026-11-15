"""HU005-E2-A: consolidación con SQLite temporales y referencias independientes."""

from contextlib import closing, contextmanager, ExitStack
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
import builtins
import io
import json
from math import isfinite, nextafter, sqrt
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import configuracion_experimentos as config
from backend.app.services import consolidacion_evaluaciones as servicio
from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import consulta_pronosticos as pronosticos
from backend.app.storage.db import inicializar_base


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")
GRUPOS = ((101, 102, 103), (201, 202, 203))
REALES = [10.0, 20.0, 30.0] * 5
ESTIMADAS = [8.0, 24.0, 30.0] * 5


def fecha(corte, paso):
    return (date.fromisoformat(corte) + timedelta(days=paso)).isoformat()


def registro(efectiva, corte, version=2):
    r = {"version_registro": version, "tecnica": efectiva["tecnica"],
         "parametros": deepcopy(efectiva["parametros"]), "reglas": config.reglas_por_configuracion(efectiva),
         "versiones": {"python": "3.12.10", "pandas": "2.2.3"},
         "entrenamiento": {"filas_validas": CORTES.index(corte) + 40},
         "auditoria": {"ajustes_a_cero": 0},
         "historial": {"fin_disponible": corte, "sha256_observaciones": "PRIVADO " + corte}}
    if efectiva["tecnica"] != config.BASE:
        r["versiones"]["scikit_learn"] = "1.6.1"
    if version == 2:
        r["configuracion_predictiva"] = deepcopy(efectiva)
    return r


def referencia_decimal(evaluaciones):
    """Desde pares binarios originales, sin métricas ni helper de producción."""
    with localcontext() as contexto:
        contexto.prec = 2200
        pares = [d for e in evaluaciones for d in e["evaluacion"]["detalles"] if d["venta_real"] is not None]
        if not pares:
            return None, None, None
        errores = [abs(Decimal.from_float(d["venta_real"]) - Decimal.from_float(d["venta_estimada"])) for d in pares]
        suma = sum(errores, Decimal(0))
        reales = sum((Decimal.from_float(d["venta_real"]) for d in pares), Decimal(0))
        n = Decimal(len(pares))
        return (float(suma / n), float(100 * suma / reales) if reales else None,
                float((sum((e * e for e in errores), Decimal(0)) / n).sqrt()))


class ConsolidacionEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "síntesis # consolidación.sqlite3"
        inicializar_base(self.ruta)
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.executemany("INSERT INTO conjunto_datos VALUES (?,?,?,?,?)",
                          [(7, "sintetico", "hash", "2026-01-01", 1), (2, "anterior", "hash2", "2026-01-01", 0)])
            c.executemany("INSERT INTO serie_temporal VALUES (?,?,?,?, 'sintética')",
                          [(83, 7, 17, "FAMILIA A"), (84, 7, 18, "FAMILIA B"), (24, 2, 17, "FAMILIA A")])
        self.crear_grupo(1, config.BASE, ESTIMADAS)
        self.crear_grupo(2, "random_forest", REALES)
        for corte in CORTES:
            self.ventas(corte, REALES)

    def sql(self, texto, parametros=()):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA foreign_keys=ON")
            return c.execute(texto, parametros).fetchall()

    def crear_grupo(self, numero, tecnica, estimadas=None, efectiva=None, serie=83):
        efectiva = config.configuracion_predeterminada(tecnica) if efectiva is None else efectiva
        ids = tuple(numero * 100 + i for i in (1, 2, 3))
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA foreign_keys=ON")
            for identificador, corte in zip(ids, CORTES):
                c.execute("INSERT INTO experimento VALUES (?,?,?,?,?,?,?)",
                          (identificador + 1000, serie, tecnica, json.dumps(registro(efectiva, corte)),
                           "2017-01-01", corte, "PRIVADO C:/modelo"))
                c.execute("INSERT INTO pronostico VALUES (?,?,?,'validacion',15,'2026-01-01T00:00:00',NULL)",
                          (identificador, identificador + 1000, corte))
                valores = ESTIMADAS if estimadas is None else estimadas
                c.executemany("INSERT INTO detalle_pronostico VALUES (?,?,?,?)",
                              [(identificador, paso, fecha(corte, paso), valor)
                               for paso, valor in enumerate(valores, 1)])
        return ids

    def ventas(self, corte, valores, serie=83):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,0)",
                          [(serie, fecha(corte, paso), valor) for paso, valor in reversed(list(enumerate(valores, 1)))
                           if valor is not None])

    def cambiar_registro(self, identificador, transformar):
        r = json.loads(self.sql("SELECT configuracion_json FROM experimento WHERE id=?", (identificador + 1000,))[0][0])
        transformar(r)
        self.sql("UPDATE experimento SET configuracion_json=? WHERE id=?", (json.dumps(r), identificador + 1000))

    def datos_numericos(self, reales, estimadas):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("DELETE FROM venta_diaria")
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (83,?,?,0)",
                          [(fecha(corte, paso), valor) for corte, valores in zip(CORTES, reales)
                           for paso, valor in enumerate(valores, 1) if valor is not None])
            c.executemany("UPDATE detalle_pronostico SET venta_estimada=? WHERE pronostico_id=? AND paso=?",
                          [(valor, grupo[indice], paso) for indice in range(3) for grupo in GRUPOS
                           for paso, valor in enumerate(estimadas[indice], 1)])

    def consolidar(self, grupos=GRUPOS):
        antes = self.ruta.read_bytes()
        r = servicio.consolidar_evaluaciones(self.ruta, grupos)
        self.assertEqual(self.ruta.read_bytes(), antes)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        self.assertNotIn("PRIVADO", json.dumps(r))
        self.assertEqual([tuple(e["pronostico_id"] for e in g["evaluaciones"]) for g in r["resultados"]],
                         [tuple(g) for g in grupos])
        for g in r["resultados"]:
            c = g["consolidado"]
            self.assertEqual(c["observaciones_evaluadas"] + c["fechas_sin_registro"], 45)
            self.assertEqual(c["cobertura"], c["observaciones_evaluadas"] / 45)
            self.assertEqual(c["observaciones_evaluadas"], r["observaciones_evaluadas_total"])
            for e in g["evaluaciones"]:
                self.assertEqual(len(e["evaluacion"]["detalles"]), 15)
        return r

    def error(self, codigo, grupos=GRUPOS, ruta=None):
        ruta = self.ruta if ruta is None else ruta
        antes = ruta.read_bytes() if isinstance(ruta, Path) and ruta.is_file() else None
        with self.assertRaises(servicio.ErrorConsolidacionEvaluaciones) as error:
            servicio.consolidar_evaluaciones(ruta, grupos)
        self.assertEqual(error.exception.codigo, codigo)
        self.assertFalse(hasattr(error.exception, "resultados"))
        for secreto in ("PRIVADO", "SELECT", "Traceback", str(self.ruta)):
            self.assertNotIn(secreto, str(error.exception))
        if antes is not None:
            self.assertEqual(ruta.read_bytes(), antes)

    @contextmanager
    def vigilar(self, alterar=None):
        abrir_real = sqlite3.connect
        llamadas, conexiones, denegadas = [], [], []

        class Filas:
            def __init__(self, filas):
                self.filas = filas

            def fetchall(self):
                return self.filas

        class Lectura(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                llamadas.append((sql, parametros, c.in_transaction))
                cursor = super().execute(sql, parametros)
                if alterar and "FROM venta_diaria" in sql:
                    return Filas(alterar(parametros, [dict(f) for f in cursor.fetchall()]))
                return cursor

        def abrir(ruta_uri, **opciones):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertTrue(opciones["uri"])
            c = abrir_real(ruta_uri, factory=Lectura, **opciones)
            conexiones.append(c)

            def autorizar(accion, tabla, columna, *otros):
                permitidas = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION)
                if accion not in permitidas or (accion == sqlite3.SQLITE_READ and (tabla == "evaluacion" or columna == "ruta_modelo")):
                    denegadas.append((accion, tabla, columna))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            c.set_authorizer(autorizar)
            return c

        with ExitStack() as stack:
            stack.enter_context(patch.object(sqlite3, "connect", side_effect=abrir))
            for modulo, nombre in ((consulta, "consultar_evaluacion"), (pronosticos, "consultar_configuracion_pronostico"),
                                    (pronosticos, "consultar_pronostico")):
                stack.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("Otra conexión")))
            yield llamadas
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(denegadas, [])
        with self.assertRaises(sqlite3.ProgrammingError):
            sqlite3.Connection.execute(conexiones[0], "SELECT 1")

    def test_dos_configuraciones_contrato_y_cobertura_completa(self):
        r = self.consolidar()
        self.assertEqual(set(r), {"conjunto_id", "serie", "cortes", "horizonte_por_corte", "fechas_esperadas_total",
                                  "observaciones_evaluadas_total", "cobertura_total", "estado_cobertura", "alcance_metricas",
                                  "fechas_ausentes_por_corte", "resultados"})
        self.assertEqual((r["conjunto_id"], r["serie"], r["cortes"]),
                         (7, {"id": 83, "store_nbr": 17, "family": "FAMILIA A"}, list(CORTES)))
        self.assertEqual((r["horizonte_por_corte"], r["fechas_esperadas_total"], r["observaciones_evaluadas_total"],
                          r["cobertura_total"], r["estado_cobertura"]), (15, 45, 45, 1, "completa"))
        self.assertEqual([e["fechas_ausentes"] for e in r["fechas_ausentes_por_corte"]], [[], [], []])
        self.assertEqual(set(r["resultados"][0]), {"tecnica", "identidad_configuracion", "configuracion",
                                                  "versiones_ejecucion", "evaluaciones", "consolidado"})

    def test_tres_tecnicas_en_orden_solicitado_sin_ranking(self):
        tercero = self.crear_grupo(3, "hist_gradient_boosting")
        r = self.consolidar([tercero, GRUPOS[1], GRUPOS[0]])
        self.assertEqual([g["tecnica"] for g in r["resultados"]], ["hist_gradient_boosting", "random_forest", config.BASE])
        self.assertTrue({"ganador", "ranking", "mejor_modelo", "exactitud", "vigente"}.isdisjoint(r))

    def test_misma_tecnica_configuraciones_diferentes(self):
        efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "parametros": {"n_estimators": 200}})
        tercero = self.crear_grupo(3, "random_forest", efectiva=efectiva)
        r = self.consolidar([GRUPOS[1], tercero])
        self.assertEqual([g["configuracion"]["parametros"]["n_estimators"] for g in r["resultados"]], [100, 200])

    def test_historial_entrenamiento_y_experimento_no_definen_identidad(self):
        r = self.consolidar()
        for g in r["resultados"]:
            self.assertEqual(g["identidad_configuracion"], config.identidad_configuracion(g["configuracion"]))
            self.assertEqual(len({e["experimento_id"] for e in g["evaluaciones"]}), 3)
            self.assertNotIn("historial", g)
            self.assertNotIn("entrenamiento", g)

    def test_cambio_hiperparametro_dentro_del_grupo(self):
        def cambiar(r):
            r["parametros"]["n_estimators"] = r["configuracion_predictiva"]["parametros"]["n_estimators"] = 200
        self.cambiar_registro(202, cambiar)
        self.error("configuraciones_no_equivalentes")

    def test_cambio_orden_variables_es_otra_identidad(self):
        def cambiar(r):
            r["configuracion_predictiva"]["variables"].reverse()
            r["reglas"]["predictoras"].reverse()
        self.cambiar_registro(202, cambiar)
        self.error("configuraciones_no_equivalentes")

    def test_cambio_grupo_variables_y_version_contrato(self):
        efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "version_contrato": 2,
                                                    "version_reglas": 2, "variables": list(config.VARIABLES_ALTERNATIVAS_V2)})
        self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (json.dumps(registro(efectiva, CORTES[1])),))
        self.error("configuraciones_no_equivalentes")

    def test_configuracion_contrato_v2_reconocida_en_tres_cortes(self):
        efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "version_contrato": 2,
                                                    "version_reglas": 2, "variables": list(config.VARIABLES_ALTERNATIVAS_V2)})
        tercero = self.crear_grupo(3, "random_forest", efectiva=efectiva)
        r = self.consolidar([GRUPOS[0], tercero])
        self.assertEqual(r["resultados"][1]["configuracion"], efectiva)

    def test_configuraciones_duplicadas_aunque_ids_distintos(self):
        tercero = self.crear_grupo(3, "random_forest", estimadas=[1.0] * 15)
        self.error("configuraciones_duplicadas", [GRUPOS[1], tercero])

    def test_parametro_numero_y_fraccion_no_son_identicos(self):
        efectiva = config.normalizar_configuracion({"tecnica": "random_forest", "parametros": {"max_features": 1}})
        tercero = self.crear_grupo(3, "random_forest", efectiva=efectiva)
        r = self.consolidar([GRUPOS[1], tercero])
        self.assertNotEqual(*[g["identidad_configuracion"] for g in r["resultados"]])

    def test_versiones_ejecucion_distintas_no_cambian_identidad_pero_rechazan_grupo(self):
        identidad = config.identidad_configuracion(config.configuracion_predeterminada("random_forest"))
        for campo in ("python", "pandas", "scikit_learn"):
            with self.subTest(campo=campo):
                anterior = self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
                self.cambiar_registro(202, lambda r: r["versiones"].update({campo: "99.0.0"}))
                r = json.loads(self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0])
                self.assertEqual(config.identidad_configuracion(config.recuperar_configuracion(r)), identidad)
                self.error("versiones_incompatibles")
                self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (anterior,))

    def test_versiones_ausentes_o_no_acreditables(self):
        anterior = self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
        for versiones in ({}, None, "3.12.10", {"python": "3.12.10", "pandas": "2.2.3"},
                          {"python": "desconocida", "pandas": "2.2.3", "scikit_learn": "1.6.1"},
                          {"python": True, "pandas": "2.2.3", "scikit_learn": "1.6.1"}):
            with self.subTest(versiones=versiones):
                self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (anterior,))
                self.cambiar_registro(202, lambda r: r.update(versiones=versiones))
                self.error("versiones_incompatibles")

    def test_linea_base_sin_sklearn_y_versiones_por_grupo_explicitas(self):
        for identificador in GRUPOS[1]:
            self.cambiar_registro(identificador, lambda r: r["versiones"].update(python="3.13.1"))
        r = self.consolidar()
        self.assertEqual(r["resultados"][0]["versiones_ejecucion"], {"python": "3.12.10", "pandas": "2.2.3"})
        self.assertEqual(r["resultados"][1]["versiones_ejecucion"]["python"], "3.13.1")

    def test_linea_base_exige_python_y_pandas(self):
        self.cambiar_registro(102, lambda r: r["versiones"].pop("pandas"))
        self.error("versiones_incompatibles")

    def test_versiones_con_sufijos_reconocidos_y_formatos_invalidos(self):
        for identificador in GRUPOS[1]:
            self.cambiar_registro(identificador, lambda r: r["versiones"].update(
                python="3.14.0rc1", pandas="2.3.0.dev0+git.abc", scikit_learn="1.6.1.post1"))
        self.consolidar()
        for invalida in ("", " 3.12.10", "3.12.10PRIVADO", "３.１２.１０", "3.12.10/privado"):
            self.cambiar_registro(202, lambda r: r["versiones"].update(python=invalida))
            self.error("versiones_incompatibles")

    def test_registros_historicos_v1_y_v2_equivalentes(self):
        for grupo, tecnica in zip(GRUPOS, (config.BASE, "random_forest")):
            efectiva = config.configuracion_predeterminada(tecnica)
            self.sql("UPDATE experimento SET configuracion_json=? WHERE id=?",
                     (json.dumps(registro(efectiva, CORTES[1], version=1)), grupo[1] + 1000))
        self.consolidar()

    def test_json_ambiguo_no_finito_o_formato_desconocido(self):
        anterior = self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
        casos = [anterior[:-1] + ',"tecnica":"random_forest"}', '{"version_registro":77}', '[]', 'null',
                 anterior.replace('"ajustes_a_cero": 0', '"ajustes_a_cero": 1e999'),
                 anterior.replace('"version_registro": 2', '"version_registro": true')]
        for texto in casos:
            with self.subTest(texto=texto[:25]):
                self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (texto,))
                self.error("datos_inconsistentes")

    def test_json_nan_o_malformado_incluso_en_auditoria(self):
        for texto in ('{"version_registro":2,"auditoria":{"x":NaN}}', '{"incompleto":'):
            with self.subTest(texto=texto), closing(sqlite3.connect(self.ruta)) as c, c:
                c.execute("PRAGMA ignore_check_constraints=ON")
                c.execute("UPDATE experimento SET configuracion_json=? WHERE id=1202", (texto,))
            self.error("datos_inconsistentes")

    def test_configuracion_incompleta_contradictoria_o_desconocida(self):
        anterior = self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
        cambios = (lambda r: r["parametros"].pop("n_estimators"),
                   lambda r: r["parametros"].update(n_estimators=200),
                   lambda r: r["reglas"].update(ausencias_no_son_ceros=False),
                   lambda r: r.update(campo_desconocido="PRIVADO"),
                   lambda r: r.update(tecnica="desconocida"))
        for cambiar in cambios:
            self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (anterior,))
            self.cambiar_registro(202, cambiar)
            self.error("datos_inconsistentes")

    def test_configuracion_contradice_tecnica_de_cabecera(self):
        self.sql("UPDATE experimento SET tecnica='hist_gradient_boosting' WHERE id=1202")
        self.error("datos_inconsistentes")

    def test_json_binario_no_es_formato_textual_admitido(self):
        texto = self.sql("SELECT configuracion_json FROM experimento WHERE id=1202")[0][0]
        self.sql("UPDATE experimento SET configuracion_json=? WHERE id=1202", (texto.encode("utf-8"),))
        self.error("datos_inconsistentes")

    def test_series_distintas(self):
        self.sql("UPDATE experimento SET serie_id=84 WHERE id=1203")
        self.error("pronosticos_no_comparables")

    def test_conjunto_inactivo_y_sin_activo(self):
        self.sql("UPDATE experimento SET serie_id=24 WHERE id=1203")
        self.error("pronostico_inactivo")
        self.sql("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_cortes_invertidos_repetidos_y_faltantes(self):
        for grupos in (((102, 101, 103), GRUPOS[1]), ((101, 102, 102), GRUPOS[1]), ((101, 102), GRUPOS[1])):
            self.error("pronosticos_no_comparables" if grupos[0] == (102, 101, 103) else "entrada_invalida", grupos)
        self.sql("UPDATE pronostico SET fecha_corte=? WHERE id=103", (CORTES[1],))
        self.sql("UPDATE experimento SET fin_entrenamiento=? WHERE id=1103", (CORTES[1],))
        self.error("datos_inconsistentes")

    def test_estructuras_y_cantidades_invalidas_antes_de_sqlite(self):
        for grupos in (None, "101", {}, [], [GRUPOS[0]], [GRUPOS[0]] * 4, [None, GRUPOS[1]],
                       [[101, 102], GRUPOS[1]], [[101, 102, 103, 104], GRUPOS[1]], ["abc", GRUPOS[1]]):
            with self.subTest(grupos=grupos), patch.object(pronosticos, "_conexion_lectura", side_effect=AssertionError):
                self.error("entrada_invalida", grupos)

    def test_corte_repetido_con_detalles_validos_tampoco_es_comparable(self):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("UPDATE pronostico SET fecha_corte=? WHERE id=103", (CORTES[1],))
            c.execute("UPDATE experimento SET fin_entrenamiento=? WHERE id=1103", (CORTES[1],))
            c.executemany("UPDATE detalle_pronostico SET fecha_objetivo=? WHERE pronostico_id=103 AND paso=?",
                          [(fecha(CORTES[1], paso), paso) for paso in range(1, 16)])
        self.error("pronosticos_no_comparables")

    def test_ids_invalidos_y_duplicados_globales_antes_de_sqlite(self):
        for invalido in (True, False, "101", 1.5, 0, -1, 2**63, None, [], float("nan"), float("inf")):
            with self.subTest(invalido=invalido), patch.object(pronosticos, "_conexion_lectura", side_effect=AssertionError):
                self.error("pronostico_invalido", [[101, 102, invalido], GRUPOS[1]])
        with patch.object(pronosticos, "_conexion_lectura", side_effect=AssertionError):
            self.error("entrada_invalida", [GRUPOS[0], [201, 202, 101]])

    def test_id_limite_sqlite_valido_pero_inexistente(self):
        self.error("pronostico_inexistente", [GRUPOS[0], [201, 202, 2**63 - 1]])

    def test_fechas_reales_distintas_aunque_igual_cantidad(self):
        self.sql("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        contador = []

        def alterar(parametros, filas):
            contador.append(True)
            if len(contador) == 4:
                filas[0]["fecha"] = "2017-06-24"
            return filas

        with self.vigilar(alterar):
            self.error("pronosticos_no_comparables")

    def test_valores_reales_distintos_aunque_mismas_fechas(self):
        contador = []

        def alterar(parametros, filas):
            contador.append(True)
            if len(contador) == 4:
                filas[0]["sales"] += 1
            return filas

        with self.vigilar(alterar):
            self.error("pronosticos_no_comparables")

    def test_metricas_exactas_y_referencia_decimal(self):
        r = self.consolidar()
        c = r["resultados"][0]["consolidado"]
        self.assertEqual((c["suma_errores_absolutos"], c["suma_ventas_reales"], c["mae"], c["wape"]), (90, 900, 2, 10))
        self.assertAlmostEqual(c["rmse"], sqrt(20 / 3))
        for g in r["resultados"]:
            for nombre, esperado in zip(("mae", "wape", "rmse"), referencia_decimal(g["evaluaciones"])):
                self.assertAlmostEqual(g["consolidado"][nombre], esperado)

    def test_wape_global_no_es_media_de_porcentajes_por_corte(self):
        self.datos_numericos([[1.0] * 15, [10.0] * 15, [100.0] * 15],
                             [[2.0] * 15, [11.0] * 15, [101.0] * 15])
        g = self.consolidar()["resultados"][0]
        self.assertAlmostEqual(g["consolidado"]["wape"], 100 / 37)
        self.assertEqual(g["consolidado"]["mae"], 1)
        self.assertEqual(g["consolidado"]["rmse"], 1)
        self.assertNotAlmostEqual(g["consolidado"]["wape"], sum(e["evaluacion"]["wape"] for e in g["evaluaciones"]) / 3)

    def test_cobertura_parcial_ausencias_interiores_y_denominadores(self):
        self.sql("DELETE FROM venta_diaria WHERE fecha IN ('2017-06-24','2017-07-09','2017-07-24')")
        r = self.consolidar()
        self.assertEqual((r["observaciones_evaluadas_total"], r["cobertura_total"], r["estado_cobertura"]), (42, 42 / 45, "parcial"))
        self.assertEqual([x["fechas_ausentes"] for x in r["fechas_ausentes_por_corte"]],
                         [["2017-06-24"], ["2017-07-09"], ["2017-07-24"]])
        for g in r["resultados"]:
            for nombre, esperado in zip(("mae", "wape", "rmse"), referencia_decimal(g["evaluaciones"])):
                self.assertAlmostEqual(g["consolidado"][nombre], esperado)

    def test_un_corte_sin_observaciones_y_medias_ponderadas(self):
        self.sql("DELETE FROM venta_diaria WHERE fecha>'2017-07-01' AND fecha<='2017-07-16'")
        self.sql("DELETE FROM venta_diaria WHERE fecha>'2017-06-18' AND fecha<='2017-07-01'")
        g = self.consolidar()["resultados"][0]
        self.assertEqual(g["consolidado"]["observaciones_evaluadas"], 17)
        self.assertIsNone(g["evaluaciones"][1]["evaluacion"]["mae"])
        for nombre, esperado in zip(("mae", "wape", "rmse"), referencia_decimal(g["evaluaciones"])):
            self.assertAlmostEqual(g["consolidado"][nombre], esperado)
        self.assertNotEqual(g["consolidado"]["mae"], (g["evaluaciones"][0]["evaluacion"]["mae"] + 2) / 2)

    def test_tres_cortes_sin_observaciones(self):
        self.sql("DELETE FROM venta_diaria")
        r = self.consolidar()
        self.assertEqual((r["observaciones_evaluadas_total"], r["cobertura_total"], r["estado_cobertura"]), (0, 0, "sin_observaciones"))
        for g in r["resultados"]:
            self.assertEqual([g["consolidado"][k] for k in ("mae", "wape", "rmse")], [None] * 3)
            self.assertEqual(g["consolidado"]["fechas_sin_registro"], 45)

    def test_cero_es_observado_y_ausencia_no_es_cero(self):
        self.datos_numericos([[0.0] * 15] * 3, [[2.0] * 15] * 3)
        self.sql("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        r = self.consolidar()
        self.assertEqual(r["observaciones_evaluadas_total"], 44)
        c = r["resultados"][0]["consolidado"]
        self.assertEqual((c["mae"], c["rmse"], c["wape"], c["suma_errores_absolutos"]), (2, 2, None, 88))
        detalles = r["resultados"][0]["evaluaciones"][0]["evaluacion"]["detalles"]
        self.assertEqual(detalles[0]["venta_real"], 0)
        self.assertIsNone(detalles[7]["venta_real"])
        self.assertIsNone(detalles[7]["error_absoluto"])

    def test_fraccionarios_y_wape_superior_a_cien(self):
        self.datos_numericos([[0.125, 0.25, 0.5] * 5] * 3, [[1.125, 2.25, 3.5] * 5] * 3)
        for g in self.consolidar()["resultados"]:
            self.assertGreater(g["consolidado"]["wape"], 100)
            for nombre, esperado in zip(("mae", "wape", "rmse"), referencia_decimal(g["evaluaciones"])):
                self.assertAlmostEqual(g["consolidado"][nombre], esperado)

    def test_rmse_grande_estable_y_auxiliar_serializable(self):
        self.datos_numericos([[0.0] * 15] * 3, [[1e300, 2e300, 0.0] * 5] * 3)
        for g in self.consolidar()["resultados"]:
            c = g["consolidado"]
            esperado = referencia_decimal(g["evaluaciones"])[2]
            self.assertLessEqual(abs(c["rmse"] - esperado) / esperado, 2e-15)
            self.assertIsNone(c["wape"])
            self.assertEqual(c["rmse_escalado"], {"escala_errores": 2e300, "suma_cuadrados_normalizados": 18.75})
            self.assertTrue(isfinite(c["rmse"]))

    def test_wape_subnormal_estable_y_repetible(self):
        self.datos_numericos([[1e307] + [0.0] * 14] * 3, [[1e307, 1e-18] + [0.0] * 13] * 3)
        r = self.consolidar()
        self.assertEqual(self.consolidar(), r)
        for g in r["resultados"]:
            self.assertGreater(g["consolidado"]["wape"], 0)
            self.assertEqual(g["consolidado"]["wape"], referencia_decimal(g["evaluaciones"])[1])

    def test_sumas_errores_consolidadas_desbordadas_rechazadas(self):
        self.datos_numericos([[0.0] * 15] * 3, [[7e307] + [0.0] * 14] * 3)
        self.error("resultado_no_finito")

    def test_sumas_reales_consolidadas_desbordadas_rechazadas(self):
        self.datos_numericos([[7e307] + [0.0] * 14] * 3, [[7e307] + [0.0] * 14] * 3)
        self.error("resultado_no_finito")

    def test_wape_consolidado_fuera_de_rango_aunque_cortes_validos(self):
        # Los cortes primero y tercero son perfectos; el segundo tiene un WAPE
        # subnormal representable. El denominador global hace caer el porcentaje.
        self.datos_numericos([[5e307] + [0.0] * 14, [1e-300] + [0.0] * 14, [5e307] + [0.0] * 14],
                             [[5e307] + [0.0] * 14, [1e-300, 1e-300] + [0.0] * 13, [5e307] + [0.0] * 14])
        self.error("resultado_no_finito")

    def test_media_consolidada_subnormal_fuera_de_rango_no_es_cero(self):
        minimo = nextafter(0.0, 1.0)
        self.datos_numericos([[None] * 15, [None] * 15, [None] * 15], [[0.0] * 15] * 3)
        self.ventas(CORTES[0], [0.0])
        self.ventas(CORTES[1], [0.0])
        self.ventas(CORTES[2], [0.0])
        for grupo in GRUPOS:
            self.sql("UPDATE detalle_pronostico SET venta_estimada=? WHERE pronostico_id=? AND paso=1", (minimo, grupo[0]))
        self.error("resultado_no_finito")

    def test_base_ausente_danada_sin_esquema_y_rutas_invalidas(self):
        ausente = self.ruta.parent / "ausente.sqlite3"
        self.error("base_no_disponible", ruta=ausente)
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("danada.sqlite3", b"no es sqlite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            self.error("base_no_disponible", ruta=ruta)
        for ruta in ("", "\0", 12, [], self.ruta.parent):
            self.error("base_no_disponible", ruta=ruta)

    def test_detalle_y_metadatos_corruptos(self):
        self.sql("DELETE FROM detalle_pronostico WHERE pronostico_id=203 AND paso=8")
        self.error("datos_inconsistentes")

    def test_detalle_fecha_incorrecta_o_valor_invalido(self):
        self.sql("UPDATE detalle_pronostico SET fecha_objetivo='2017-07-16' WHERE pronostico_id=203 AND paso=8")
        self.error("datos_inconsistentes")
        self.sql("UPDATE detalle_pronostico SET fecha_objetivo='2017-07-24',venta_estimada=-1 WHERE pronostico_id=203 AND paso=8")
        self.error("datos_inconsistentes")

    def test_tipo_prueba_futuro_y_reserva_antes_de_ventas_del_id(self):
        for tipo, corte, codigo in (("prueba", CORTES[0], "tipo_no_evaluable"), ("futuro", CORTES[0], "tipo_no_evaluable"),
                                    ("validacion", "2017-07-31", "corte_no_autorizado")):
            self.sql("UPDATE pronostico SET tipo=?,fecha_corte=? WHERE id=101", (tipo, corte))
            with self.vigilar() as llamadas:
                self.error(codigo)
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_conexion_unica_ro_begin_parametrizacion_y_cierre(self):
        with self.vigilar() as llamadas:
            self.consolidar()
        self.assertEqual([sql for sql, _, _ in llamadas if sql == "BEGIN"], ["BEGIN"])
        self.assertTrue(all(en_transaccion for sql, _, en_transaccion in llamadas if sql != "BEGIN"))
        ventas = [(sql, p) for sql, p, _ in llamadas if "FROM venta_diaria" in sql]
        self.assertEqual(len(ventas), 6)
        for indice, (sql, parametros) in enumerate(ventas):
            corte = CORTES[indice % 3]
            self.assertEqual(parametros, (83, corte, fecha(corte, 15)))
            self.assertIn("fecha > ? AND fecha <= ?", sql)
            self.assertLessEqual(parametros[2], "2017-07-31")
        self.assertEqual(len([sql for sql, _, _ in llamadas if "SELECT configuracion_json" in sql]), 6)

    def test_ninguna_escritura_ni_lectura_de_evaluacion_o_ruta_modelo(self):
        self.sql("INSERT INTO evaluacion VALUES (101,999999,999999,999999)")
        with self.vigilar() as llamadas:
            self.assertEqual(self.consolidar()["resultados"][0]["consolidado"]["mae"], 2)
        self.assertFalse(any("FROM evaluacion" in sql or "ruta_modelo" in sql for sql, _, _ in llamadas))
        self.assertEqual(self.sql("SELECT mae FROM evaluacion WHERE pronostico_id=101"), [(999999.0,)])

    def test_preserva_evaluaciones_individuales_e_inputs(self):
        grupos = [list(GRUPOS[1]), list(GRUPOS[0])]
        antes = deepcopy(grupos)
        individuales = {i: consulta.consultar_evaluacion(self.ruta, i) for grupo in grupos for i in grupo}
        r = self.consolidar(grupos)
        self.assertEqual(grupos, antes)
        for g in r["resultados"]:
            for e in g["evaluaciones"]:
                self.assertEqual(e, individuales[e["pronostico_id"]])

    def test_ningun_resultado_parcial_si_falla_el_noveno(self):
        tercero = self.crear_grupo(3, "hist_gradient_boosting")
        self.sql("DELETE FROM detalle_pronostico WHERE pronostico_id=303 AND paso=15")
        self.error("datos_inconsistentes", [*GRUPOS, tercero])

    def test_errores_publicos_constantes_y_programacion_no_oculta(self):
        for codigo in ("pronostico_inexistente", "pronostico_inactivo", "sin_conjunto_activo", "tipo_no_evaluable",
                       "corte_no_autorizado", "datos_inconsistentes", "base_no_disponible", "PRIVADO", None, []):
            with patch.object(consulta, "_evaluar_en_conexion", side_effect=consulta.ErrorConsultaEvaluacion(codigo, "PRIVADO SELECT")):
                self.error(codigo if type(codigo) is str and codigo in servicio._MENSAJES else "datos_inconsistentes")
        with patch.object(consulta, "_evaluar_en_conexion", side_effect=RuntimeError("error de programación")):
            with self.assertRaisesRegex(RuntimeError, "programación"):
                servicio.consolidar_evaluaciones(self.ruta, GRUPOS)

    def escritor_wal(self, modo):
        codigo = """
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
with closing(sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + '?mode=rw', uri=True)) as c, c:
    if sys.argv[2] == 'ventas':
        c.execute("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        c.execute("DELETE FROM venta_diaria WHERE fecha='2017-07-09'")
    elif sys.argv[2] == 'activo':
        c.execute('UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END')
    else:
        import json
        r = json.loads(c.execute('SELECT configuracion_json FROM experimento WHERE id=1203').fetchone()[0])
        r['versiones']['python'] = '99.0.0'
        c.execute('UPDATE experimento SET configuracion_json=? WHERE id=1203', (json.dumps(r),))
"""
        proceso = subprocess.run([sys.executable, "-B", "-c", codigo, str(self.ruta), modo],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(proceso.returncode, 0, proceso.stderr)

    def verificar_wal(self, modo):
        self.assertEqual(self.sql("PRAGMA journal_mode=WAL")[0][0], "wal")
        esperado = self.consolidar()
        helper_real = consulta._evaluar_en_conexion

        def intercalar(conexion, identificador):
            r = helper_real(conexion, identificador)
            if identificador == 101:
                self.escritor_wal(modo)
            return r

        with patch.object(consulta, "_evaluar_en_conexion", side_effect=intercalar):
            # El escritor cambia legítimamente el archivo temporal: aquí no se
            # compara el hash global, sino la instantánea lógica completa.
            self.assertEqual(servicio.consolidar_evaluaciones(self.ruta, GRUPOS), esperado)

    def test_snapshot_wal_proceso_independiente_cambia_ventas(self):
        self.verificar_wal("ventas")
        r = self.consolidar()
        self.assertEqual(r["observaciones_evaluadas_total"], 44)
        self.assertEqual(r["resultados"][0]["evaluaciones"][0]["evaluacion"]["detalles"][0]["venta_real"], 11)

    def test_snapshot_wal_proceso_independiente_cambia_activo(self):
        self.verificar_wal("activo")
        self.error("pronostico_inactivo")

    def test_snapshot_wal_proceso_independiente_cambia_versiones(self):
        self.verificar_wal("versiones")
        self.error("versiones_incompatibles")

    def test_importacion_y_calculo_sin_modelos_registro_ni_efectos_externos(self):
        codigo = compile(Path(servicio.__file__).read_text(encoding="utf-8"), servicio.__file__, "exec")
        aislado = ModuleType("consolidacion_aislada")
        importar_real = builtins.__import__
        prohibidos = ("scripts.modelos_prediccion", "scripts.linea_base_prediccion", "sklearn", "fastapi", "uvicorn",
                      "backend.app.services.registro_pronosticos", "backend.app.services.registro_evaluaciones")

        def importar(nombre, globals=None, locals=None, fromlist=(), level=0):
            candidatos = (nombre,) + tuple(f"{nombre}.{n}" for n in (fromlist or ()))
            if any(c == p or c.startswith(p + ".") for c in candidatos for p in prohibidos):
                raise AssertionError("Dependencia de ejecución")
            return importar_real(nombre, globals, locals, fromlist, level)

        with ExitStack() as stack:
            stack.enter_context(patch.object(builtins, "__import__", side_effect=importar))
            for modulo, nombre in ((builtins, "open"), (io, "open"), (Path, "open"), (os, "open"),
                                    (socket, "socket"), (subprocess, "Popen")):
                stack.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("Efecto externo")))
            with patch.object(sqlite3, "connect", side_effect=AssertionError("E/S al importar")):
                exec(codigo, aislado.__dict__)
            self.assertEqual(aislado.consolidar_evaluaciones(self.ruta, GRUPOS)["observaciones_evaluadas_total"], 45)


if __name__ == "__main__":
    unittest.main()
