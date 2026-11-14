"""HU005-D1: comparación actual con SQLite temporales y datos sintéticos."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import builtins
import io
import json
from math import inf, nextafter, sqrt
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import comparacion_evaluaciones as servicio
from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import consulta_pronosticos as pronosticos
from backend.app.services import metricas_evaluacion as metricas
from tests import test_consulta_evaluaciones as fixtures


class ComparacionEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.ConsultaEvaluacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        self.datos.crear_pronostico(102, 202, tecnica="random_forest", estimadas=fixtures.REALES)

    def tercero(self, **opciones):
        self.datos.crear_pronostico(103, 203, tecnica="hist_gradient_boosting",
                                   estimadas=[12, 16, 30] * 5, **opciones)

    def comparar(self, ids=(101, 102)):
        contenido = self.ruta.read_bytes()
        r = servicio.comparar_evaluaciones(self.ruta, ids)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        self.assertNotIn("PRIVADO", json.dumps(r))
        self.assertEqual([x["pronostico_id"] for x in r["resultados"]], list(ids))
        self.assertEqual(r["observaciones_evaluadas"] + len(r["fechas_ausentes"]), 15)
        for resultado in r["resultados"]:
            e = resultado["evaluacion"]
            self.assertEqual(len(e["detalles"]), 15)
            self.assertEqual(e["observaciones_evaluadas"], r["observaciones_evaluadas"])
            self.assertEqual(e["fechas_ausentes"], r["fechas_ausentes"])
            self.assertEqual(e["cobertura"], r["cobertura"])
        return r

    def error(self, codigo, ids=(101, 102), ruta=None):
        ruta = self.ruta if ruta is None else ruta
        contenido = ruta.read_bytes() if isinstance(ruta, Path) and ruta.is_file() else None
        with self.assertRaises(servicio.ErrorComparacionEvaluaciones) as error:
            servicio.comparar_evaluaciones(ruta, ids)
        self.assertEqual(error.exception.codigo, codigo)
        self.assertFalse(hasattr(error.exception, "resultados"))
        for privado in ("PRIVADO", "SELECT", "Traceback", str(self.ruta)):
            self.assertNotIn(privado, str(error.exception))
        if contenido is not None:
            self.assertEqual(ruta.read_bytes(), contenido)

    @contextmanager
    def lectura_vigilada(self, alterar_ventas=None):
        abrir_real = sqlite3.connect
        llamadas, conexiones, denegadas, lecturas_ventas = [], [], [], []

        class Filas:
            def __init__(self, filas):
                self.filas = filas

            def fetchall(self):
                return self.filas

        class Lectura(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                llamadas.append((sql, parametros, conexion.in_transaction))
                cursor = super().execute(sql, parametros)
                if alterar_ventas and "FROM venta_diaria" in sql:
                    lecturas_ventas.append(True)
                    filas = [dict(f) for f in cursor.fetchall()]
                    return Filas(alterar_ventas(len(lecturas_ventas), filas))
                return cursor

        def abrir(ruta_uri, **opciones):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertTrue(opciones["uri"])
            c = abrir_real(ruta_uri, factory=Lectura, **opciones)
            conexiones.append(c)

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

        with ExitStack() as stack:
            stack.enter_context(patch.object(sqlite3, "connect", side_effect=abrir))
            stack.enter_context(patch.object(consulta, "consultar_evaluacion", side_effect=AssertionError("Otra conexión")))
            for nombre in ("consultar_pronostico", "listar_pronosticos", "consultar_configuracion_pronostico"):
                stack.enter_context(patch.object(pronosticos, nombre, side_effect=AssertionError(nombre)))
            yield llamadas
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(denegadas, [])
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                sqlite3.Connection.execute(c, "SELECT 1")

    def test_dos_tecnicas_y_contrato_publico(self):
        r = self.comparar()
        self.assertEqual(set(r), {"conjunto_id", "serie", "fecha_corte", "horizonte", "estado_cobertura",
                                  "alcance_metricas", "observaciones_evaluadas", "cobertura", "fechas_ausentes", "resultados"})
        self.assertEqual((r["conjunto_id"], r["fecha_corte"], r["horizonte"]), (7, fixtures.CORTES[0], 15))
        self.assertEqual(r["serie"], {"id": 83, "store_nbr": 17, "family": "FAMILIA A"})
        self.assertEqual([x["tecnica"] for x in r["resultados"]], [fixtures.BASE, "random_forest"])
        for x in r["resultados"]:
            self.assertEqual(set(x), {"pronostico_id", "experimento_id", "tecnica", "evaluacion"})
        self.assertEqual((r["estado_cobertura"], r["observaciones_evaluadas"], r["cobertura"]), ("completa", 15, 1))
        self.assertEqual(r["fechas_ausentes"], [])

    def test_tres_tecnicas_sin_clasificar_ni_escoger_ganador(self):
        self.tercero()
        r = self.comparar((101, 102, 103))
        self.assertEqual([x["tecnica"] for x in r["resultados"]],
                         [fixtures.BASE, "random_forest", "hist_gradient_boosting"])
        self.assertEqual([x["evaluacion"]["mae"] for x in r["resultados"]], [2, 0, 2])
        self.assertTrue({"ganador", "ranking", "mejor_tecnica", "vigente", "registrado"}.isdisjoint(r))

    def test_configuraciones_distintas_de_la_misma_tecnica(self):
        self.datos.crear_pronostico(103, 203, tecnica="random_forest", estimadas=[12, 16, 30] * 5)
        for identificador, arboles in ((202, 100), (203, 200)):
            self.datos.ejecutar("UPDATE experimento SET configuracion_json=? WHERE id=?",
                              (json.dumps({"tecnica": "random_forest", "parametros": {"n_estimators": arboles}}), identificador))
        with self.lectura_vigilada():
            r = self.comparar((102, 103))
        self.assertEqual([x["tecnica"] for x in r["resultados"]], ["random_forest", "random_forest"])
        self.assertEqual([x["experimento_id"] for x in r["resultados"]], [202, 203])

    def test_preserva_orden_lista_y_tupla(self):
        self.tercero()
        for seleccion in ([102, 101], (103, 101, 102), [102, 103, 101]):
            with self.subTest(seleccion=seleccion):
                anterior = deepcopy(seleccion)
                self.comparar(seleccion)
                self.assertEqual(seleccion, anterior)

    def test_metricas_exactas_y_resultado_a1_integro(self):
        esperados = {i: consulta.consultar_evaluacion(self.ruta, i)["evaluacion"] for i in (101, 102)}
        r = self.comparar()
        for x in r["resultados"]:
            self.assertEqual(x["evaluacion"], esperados[x["pronostico_id"]])
        a, b = [x["evaluacion"] for x in r["resultados"]]
        self.assertEqual((a["mae"], a["wape"], a["rmse"]), (2, 10, sqrt(20 / 3)))
        self.assertEqual((b["mae"], b["wape"], b["rmse"]), (0, 0, 0))

    def test_cobertura_parcial_compartida_y_ausencias_interiores(self):
        faltan = (2, 8, 11)
        for paso in faltan:
            self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha=?", (fixtures.fecha_objetivo(fixtures.CORTES[0], paso),))
        r = self.comparar()
        self.assertEqual((r["estado_cobertura"], r["observaciones_evaluadas"], r["cobertura"]), ("parcial", 12, 12 / 15))
        self.assertEqual(r["alcance_metricas"], "Métricas sobre observaciones disponibles.")
        self.assertEqual(r["fechas_ausentes"], [fixtures.fecha_objetivo(fixtures.CORTES[0], p) for p in faltan])
        a, b = [x["evaluacion"] for x in r["resultados"]]
        self.assertEqual(a["mae"], 18 / 12)
        self.assertAlmostEqual(a["rmse"], sqrt(52 / 12))
        self.assertEqual(b["mae"], 0)
        for e in (a, b):
            for paso in faltan:
                self.assertIsNone(e["detalles"][paso - 1]["venta_real"])
                self.assertIsNone(e["detalles"][paso - 1]["error_absoluto"])

    def test_sin_observaciones_no_afirma_desempeno(self):
        self.datos.ejecutar("DELETE FROM venta_diaria")
        r = self.comparar()
        self.assertEqual((r["estado_cobertura"], r["observaciones_evaluadas"], r["cobertura"]), ("sin_observaciones", 0, 0))
        self.assertIn("métricas no definidas", r["alcance_metricas"])
        self.assertEqual(len(r["fechas_ausentes"]), 15)
        for x in r["resultados"]:
            self.assertEqual(tuple(x["evaluacion"][c] for c in ("mae", "wape", "rmse")), (None, None, None))

    def test_ventas_cero_wape_none_y_mae_rmse_validos(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=CASE pronostico_id WHEN 101 THEN 2 ELSE 4 END")
        r = self.comparar()
        self.assertEqual(r["estado_cobertura"], "completa")
        self.assertEqual([tuple(x["evaluacion"][c] for c in ("mae", "wape", "rmse")) for x in r["resultados"]],
                         [(2, None, 2), (4, None, 4)])

    def test_valores_fraccionarios_y_porcentajes_sin_redondear(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0.25")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=CASE pronostico_id WHEN 101 THEN 0.75 ELSE 0.5 END")
        r = self.comparar()
        self.assertEqual([tuple(x["evaluacion"][c] for c in ("mae", "wape", "rmse")) for x in r["resultados"]],
                         [(0.5, 200, 0.5), (0.25, 100, 0.25)])

    def test_los_tres_cortes_por_separado(self):
        for i, corte in enumerate(fixtures.CORTES):
            ids = (101 + 2 * i, 102 + 2 * i)
            if i:
                self.datos.crear_pronostico(ids[0], 201 + 2 * i, corte=corte)
                self.datos.crear_pronostico(ids[1], 202 + 2 * i, corte=corte, tecnica="random_forest", estimadas=fixtures.REALES)
                self.datos.insertar_reales(corte=corte)
            with self.subTest(corte=corte):
                r = self.comparar(ids)
                self.assertEqual(r["fecha_corte"], corte)
                self.assertEqual([x["evaluacion"]["mae"] for x in r["resultados"]], [2, 0])
                self.assertEqual(r["resultados"][0]["evaluacion"]["detalles"][-1]["fecha_objetivo"],
                                 fixtures.fecha_objetivo(corte, 15))

    def test_series_distintas_no_son_comparables_aunque_las_ventas_coincidan(self):
        self.tercero(serie=84)
        self.datos.insertar_reales(serie=84)
        for seleccion in ((101, 103), (103, 101)):
            self.error("pronosticos_no_comparables", seleccion)

    def test_cortes_distintos_no_se_agregan(self):
        self.tercero(corte=fixtures.CORTES[1])
        self.datos.insertar_reales(corte=fixtures.CORTES[1])
        self.error("pronosticos_no_comparables", (101, 103))

    def test_pronosticos_de_conjunto_anterior_son_inactivos(self):
        self.tercero(serie=24)
        self.error("pronostico_inactivo", (101, 103))
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
        self.error("pronostico_inactivo", (103, 101))

    def test_sin_conjunto_activo(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_estructura_cantidad_y_duplicados_invalidos_antes_de_sqlite(self):
        casos = (None, {}, {101, 102}, "101,102", iter((101, 102)), [], [101], [101, 102, 103, 104],
                 [101, 102, 103, 104, 105], [101, 101], (101, 102, 101))
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")) as abrir:
            for ids in casos:
                with self.subTest(ids=ids):
                    self.error("entrada_invalida", ids)
            abrir.assert_not_called()

    def test_valida_todos_los_identificadores_antes_de_abrir(self):
        class EnteroDerivado(int):
            pass

        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")) as abrir:
            for valor in (None, True, False, 0, -1, 101.0, "102", "102 OR 1=1", [], {}, float("nan"), inf,
                          2**63, EnteroDerivado(102)):
                for ids in ([101, valor], [101, 102, valor]):
                    with self.subTest(ids=ids):
                        self.error("pronostico_invalido", ids)
            abrir.assert_not_called()

    def test_rutas_invalidas_controladas_sin_conexion(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")) as abrir:
            for ruta in (None, 5, [], "", "PRIVADO\0ruta"):
                with self.subTest(ruta=ruta), self.assertRaises(servicio.ErrorComparacionEvaluaciones) as error:
                    servicio.comparar_evaluaciones(ruta, [101, 102])
                self.assertEqual(error.exception.codigo, "base_no_disponible")
                self.assertNotIn("PRIVADO", str(error.exception))
            abrir.assert_not_called()

    def test_inexistente_y_limite_sqlite_sin_resultados_parciales(self):
        for ids in ((101, 999), (101, 2**63 - 1), (101, 102, 999)):
            with self.subTest(ids=ids), self.lectura_vigilada():
                self.error("pronostico_inexistente", ids)

    def test_tipos_y_cortes_no_autorizados_no_consultan_sus_ventas(self):
        for tipo, corte, codigo in (("prueba", fixtures.CORTES[0], "tipo_no_evaluable"),
                                    ("futuro", "2017-07-31", "tipo_no_evaluable"),
                                    ("validacion", "2017-07-31", "corte_no_autorizado"),
                                    ("validacion", "2017-06-17", "corte_no_autorizado")):
            self.datos.ejecutar("UPDATE pronostico SET tipo=?,fecha_corte=? WHERE id=102", (tipo, corte))
            with self.subTest(tipo=tipo, corte=corte), self.lectura_vigilada() as llamadas:
                self.error(codigo)
            ventas = [p for sql, p, _ in llamadas if "FROM venta_diaria" in sql]
            self.assertEqual(ventas, [(83, "2017-06-16", "2017-07-01")])

    def test_detalle_omitido_rechaza_toda_la_comparacion(self):
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE pronostico_id=102 AND paso=8")
        self.error("datos_inconsistentes")

    def test_detalles_y_metadatos_corruptos(self):
        casos = (("UPDATE detalle_pronostico SET fecha_objetivo=? WHERE pronostico_id=102 AND paso=8", "2017-06-16",
                  "UPDATE detalle_pronostico SET fecha_objetivo='2017-06-24' WHERE pronostico_id=102 AND paso=8"),
                 ("UPDATE detalle_pronostico SET venta_estimada=? WHERE pronostico_id=102 AND paso=1", -1,
                  "UPDATE detalle_pronostico SET venta_estimada=10 WHERE pronostico_id=102 AND paso=1"),
                 ("UPDATE pronostico SET horizonte=? WHERE id=102", 14, "UPDATE pronostico SET horizonte=15 WHERE id=102"),
                 ("UPDATE pronostico SET fecha_generacion=? WHERE id=102", "PRIVADO fecha",
                  "UPDATE pronostico SET fecha_generacion='2026-01-03' WHERE id=102"))
        for sql, valor, restaurar in casos:
            with self.subTest(sql=sql):
                self.datos.corromper(sql, (valor,))
                self.error("datos_inconsistentes")
                self.datos.ejecutar(restaurar)

    def test_ventas_invalidas_no_se_ocultan(self):
        for valor in (-1, inf, "PRIVADO NO_NUMERICO"):
            self.datos.ejecutar("UPDATE venta_diaria SET sales=? WHERE fecha='2017-06-17'", (valor,))
            self.error("datos_inconsistentes")

    def test_misma_cantidad_pero_fechas_reales_distintas_no_basta(self):
        # Una instantánea correcta ya impide esta divergencia para una misma
        # serie/corte. El doble simula un defecto de lectura para verificar la
        # comprobación explícita adicional, manteniendo dos resultados A1 válidos.
        cantidades = []

        def distintas(numero, filas):
            fecha = "2017-06-18" if numero == 1 else "2017-06-24"
            filas = [f for f in filas if f["fecha"] != fecha]
            cantidades.append(len(filas))
            return filas

        with self.lectura_vigilada(distintas):
            self.error("pronosticos_no_comparables")
        self.assertEqual(cantidades, [14, 14])

    def test_mismas_fechas_pero_venta_real_distinta_sin_tolerancias(self):
        def distintas(numero, filas):
            if numero == 2:
                filas[0]["sales"] = nextafter(filas[0]["sales"], inf)
            return filas

        with self.lectura_vigilada(distintas):
            self.error("pronosticos_no_comparables")

    def test_cero_y_fecha_ausente_no_son_observaciones_equivalentes(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")

        def distintas(numero, filas):
            return filas[1:] if numero == 1 else filas

        with self.lectura_vigilada(distintas):
            self.error("pronosticos_no_comparables")

    def test_fechas_objetivo_distintas_se_rechazan_explicitamente(self):
        helper_real = consulta._evaluar_en_conexion

        def alterar(conexion, identificador):
            r = helper_real(conexion, identificador)
            if identificador == 102:
                r["evaluacion"]["detalles"][0]["fecha_objetivo"] = "2017-06-16"
            return r

        # También comprueba la guarda del comparador ante una regresión del
        # contrato del helper; la corrupción física se prueba arriba con A2 real.
        with patch.object(consulta, "_evaluar_en_conexion", side_effect=alterar):
            self.error("pronosticos_no_comparables")

    def test_delega_en_helper_a2_con_una_conexion_y_a1_sin_otra_integracion(self):
        self.tercero()
        with self.lectura_vigilada() as llamadas, \
                patch.object(consulta, "_evaluar_en_conexion", wraps=consulta._evaluar_en_conexion) as evaluar, \
                patch.object(metricas, "calcular_metricas", wraps=metricas.calcular_metricas) as calcular:
            self.comparar((103, 101, 102))
        self.assertEqual([c.args[1] for c in evaluar.call_args_list], [103, 101, 102])
        self.assertEqual(len({id(c.args[0]) for c in evaluar.call_args_list}), 1)
        self.assertEqual(calcular.call_count, 3)
        self.assertEqual(llamadas[0][:2], ("BEGIN", ()))
        self.assertEqual(len(llamadas), 13)  # BEGIN y cuatro SELECT por pronóstico.
        self.assertTrue(all(en_transaccion for _, _, en_transaccion in llamadas[1:]))
        ventas = [(sql, p) for sql, p, _ in llamadas if "FROM venta_diaria" in sql]
        self.assertEqual(len(ventas), 3)
        for sql, parametros in ventas:
            self.assertIn("serie_id = ? AND fecha > ? AND fecha <= ?", sql)
            self.assertEqual(parametros, (83, "2017-06-16", "2017-07-01"))

    def test_todas_las_tablas_y_fila_guardada_permanecen_intactas(self):
        self.datos.ejecutar("INSERT INTO evaluacion VALUES (101,999999,NULL,NULL)")
        tablas = ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento", "pronostico",
                  "detalle_pronostico", "evaluacion", "simulacion", "perfil_global_conjunto")
        antes = [self.datos.ejecutar(f"SELECT * FROM {t} ORDER BY 1,2") for t in tablas]
        with self.lectura_vigilada():
            self.assertEqual(self.comparar()["resultados"][0]["evaluacion"]["mae"], 2)
        despues = [self.datos.ejecutar(f"SELECT * FROM {t} ORDER BY 1,2") for t in tablas]
        self.assertEqual(despues, antes)

    def test_base_inexistente_corrupta_o_sin_esquema_no_se_crea_ni_inicializa(self):
        ausente = self.ruta.parent / "PRIVADO_ausente.sqlite3"
        self.error("base_no_disponible", ruta=ausente)
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("corrupta.sqlite3", b"Esto no es SQLite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            self.error("base_no_disponible", ruta=ruta)
            self.assertEqual(ruta.read_bytes(), contenido)

    def test_errores_a2_se_traducen_sin_mensajes_internos(self):
        for codigo in ("pronostico_inexistente", "pronostico_inactivo", "sin_conjunto_activo", "tipo_no_evaluable",
                       "corte_no_autorizado", "datos_inconsistentes", "base_no_disponible"):
            with self.subTest(codigo=codigo), patch.object(consulta, "_evaluar_en_conexion", side_effect=
                    consulta.ErrorConsultaEvaluacion(codigo, "PRIVADO SELECT C:/secreto Traceback")):
                self.error(codigo)
        for codigo in ("PRIVADO codigo", None, [], {}):
            with self.subTest(codigo=codigo), patch.object(consulta, "_evaluar_en_conexion", side_effect=
                    consulta.ErrorConsultaEvaluacion(codigo, "PRIVADO")):
                self.error("datos_inconsistentes")

    def test_entradas_intactas_tras_rechazo_y_resultados_independientes(self):
        ids = [101, 101]
        anterior = list(ids)
        self.error("entrada_invalida", ids)
        self.assertEqual(ids, anterior)
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        r = self.comparar()
        esperado = deepcopy(r)
        r["serie"]["family"] = "ALTERADA"
        r["fechas_ausentes"].append("otra")
        r["resultados"][0]["evaluacion"]["detalles"][0]["venta_real"] = 999
        self.assertEqual(self.comparar(), esperado)
        self.assertEqual(r["resultados"][1]["evaluacion"], esperado["resultados"][1]["evaluacion"])
        self.assertEqual(r["resultados"][0]["evaluacion"]["fechas_ausentes"], esperado["fechas_ausentes"])

    def escritor_wal(self, modo):
        # Proceso Python independiente. Solo escribe en la SQLite temporal
        # existente que recibe como argumento; no crea archivos de código.
        codigo = """
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
uri = Path(sys.argv[1]).resolve().as_uri() + '?mode=rw'
with closing(sqlite3.connect(uri, uri=True)) as c, c:
    c.execute('PRAGMA foreign_keys = ON')
    if sys.argv[2] == 'ventas':
        c.execute("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        c.execute("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
    else:
        c.execute('UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END')
"""
        proceso = subprocess.run([sys.executable, "-B", "-c", codigo, str(self.ruta), modo],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(proceso.returncode, 0, proceso.stderr)

    def test_snapshot_wal_con_otro_proceso_que_modifica_ventas(self):
        self.datos.activar_wal()
        esperado = self.comparar()
        helper_real = consulta._evaluar_en_conexion

        def intercalar(conexion, identificador):
            r = helper_real(conexion, identificador)
            if identificador == 101:
                self.escritor_wal("ventas")
            return r

        with patch.object(consulta, "_evaluar_en_conexion", side_effect=intercalar):
            self.assertEqual(servicio.comparar_evaluaciones(self.ruta, [101, 102]), esperado)
        actual = self.comparar()
        self.assertEqual((actual["estado_cobertura"], actual["observaciones_evaluadas"]), ("parcial", 14))
        self.assertEqual(actual["fechas_ausentes"], ["2017-06-24"])
        for resultado in actual["resultados"]:
            self.assertEqual(resultado["evaluacion"]["detalles"][0]["venta_real"], 11)

    def test_snapshot_wal_con_otro_proceso_que_cambia_conjunto_activo(self):
        self.datos.activar_wal()
        esperado = self.comparar()
        helper_real = consulta._evaluar_en_conexion

        def intercalar(conexion, identificador):
            r = helper_real(conexion, identificador)
            if identificador == 101:
                self.escritor_wal("activo")
            return r

        with patch.object(consulta, "_evaluar_en_conexion", side_effect=intercalar):
            self.assertEqual(servicio.comparar_evaluaciones(self.ruta, [101, 102]), esperado)
        self.error("pronostico_inactivo")

    def test_importacion_y_comparacion_sin_e_s_modelos_registros_ni_servidores(self):
        codigo = compile(Path(servicio.__file__).read_text(encoding="utf-8"), servicio.__file__, "exec")
        aislado = ModuleType("comparacion_evaluaciones_en_prueba")
        importar_real = builtins.__import__
        prohibidos = ("backend.app.services.registro_pronosticos", "backend.app.services.registro_evaluaciones",
                      "backend.app.api", "scripts.modelos_prediccion", "scripts.linea_base_prediccion",
                      "sklearn", "fastapi", "uvicorn")

        def importar(nombre, globals=None, locals=None, fromlist=(), level=0):
            candidatos = (nombre,) + tuple(f"{nombre}.{item}" for item in (fromlist or ()))
            if any(c == p or c.startswith(p + ".") for c in candidatos for p in prohibidos):
                raise AssertionError(f"Dependencia de ejecución: {nombre}")
            return importar_real(nombre, globals, locals, fromlist, level)

        objetivos = ((builtins, "open"), (io, "open"), (Path, "open"), (os, "open"),
                     (socket, "socket"), (subprocess, "Popen"))
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            stack.enter_context(patch.object(builtins, "__import__", side_effect=importar))
            with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir al importar")):
                exec(codigo, aislado.__dict__)
            with self.lectura_vigilada():
                r = aislado.comparar_evaluaciones(self.ruta, [101, 102])
                self.assertEqual(r["resultados"][0]["evaluacion"]["mae"], 2)
            for doble in dobles:
                doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
