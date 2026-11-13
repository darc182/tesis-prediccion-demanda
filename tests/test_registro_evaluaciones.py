"""HU005-B1: persistencia en SQLite temporales, sin datos ni modelos reales."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, ExitStack
import builtins
import io
import json
from math import inf, nextafter, sqrt
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
from threading import Barrier, Event, Lock
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import consulta_evaluaciones as consulta
from backend.app.services import registro_evaluaciones as registro
from tests import test_consulta_evaluaciones as fixtures


class RegistroEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        # Se reutiliza solo la preparación sintética de A2; no se heredan tests.
        self.datos = fixtures.ConsultaEvaluacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta

    def filas(self):
        return self.datos.ejecutar("SELECT pronostico_id,mae,wape,rmse FROM evaluacion ORDER BY pronostico_id")

    def ajenos(self):
        tablas = ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento", "pronostico",
                  "detalle_pronostico", "simulacion", "perfil_global_conjunto")
        return [self.datos.ejecutar(f"SELECT * FROM {tabla} ORDER BY 1,2") for tabla in tablas]

    def registrar(self, identificador=101):
        r = registro.registrar_evaluacion(self.ruta, identificador)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        self.assertNotIn("PRIVADO", json.dumps(r))
        self.assertEqual(r["hubo_escritura"], r["estado_registro"] == "creado")
        e = r["evaluacion"]
        self.assertEqual(e["observaciones_evaluadas"] + e["fechas_sin_registro"], 15)
        return r

    def error(self, codigo, identificador=101, ruta=None):
        ruta = self.ruta if ruta is None else ruta
        contenido = ruta.read_bytes() if isinstance(ruta, Path) and ruta.is_file() else None
        with self.assertRaises(registro.ErrorRegistroEvaluacion) as error:
            registro.registrar_evaluacion(ruta, identificador)
        self.assertEqual(error.exception.codigo, codigo)
        for privado in ("PRIVADO", "SELECT", "INSERT", "Traceback", str(self.ruta)):
            self.assertNotIn(privado, str(error.exception))
        if contenido is not None:
            self.assertEqual(ruta.read_bytes(), contenido)

    @contextmanager
    def vigilar(self):
        abrir_real = sqlite3.connect
        conexiones, llamadas, trazas, escrituras, denegadas = [], [], [], [], []

        class Vigilada(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                llamadas.append((sql, parametros, conexion.in_transaction))
                return super().execute(sql, parametros)

        def abrir(ruta_uri, **kwargs):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=rw")
            self.assertTrue(kwargs["uri"])
            c = abrir_real(ruta_uri, factory=Vigilada, **kwargs)
            conexiones.append(c)
            c.set_trace_callback(trazas.append)

            def autorizar(accion, tabla, columna, *otros):
                permitidas = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION,
                              sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_PRAGMA, sqlite3.SQLITE_INSERT)
                if accion == sqlite3.SQLITE_INSERT:
                    escrituras.append(tabla)
                if (accion not in permitidas or (accion == sqlite3.SQLITE_INSERT and tabla != "evaluacion")
                        or (accion == sqlite3.SQLITE_PRAGMA and tabla != "foreign_keys")
                        or (accion == sqlite3.SQLITE_READ and columna in ("configuracion_json", "ruta_modelo"))):
                    denegadas.append((accion, tabla, columna))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            c.set_authorizer(autorizar)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir), \
                patch.object(consulta, "consultar_evaluacion", side_effect=AssertionError("Segunda conexión")):
            yield llamadas, trazas, escrituras
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(denegadas, [])
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                sqlite3.Connection.execute(c, "SELECT 1")

    def test_registro_completo_y_procedencia(self):
        ajenos = self.ajenos()
        esperado = consulta.consultar_evaluacion(self.ruta, 101)
        r = self.registrar()
        self.assertEqual({k: r[k] for k in esperado}, esperado)
        self.assertEqual((r["estado_registro"], r["hubo_escritura"], r["motivo"]), ("creado", True, None))
        self.assertEqual(r["estado_cobertura"], "completa")
        self.assertEqual(self.filas(), [(101, 2, 10, sqrt(20 / 3))])
        self.assertEqual(self.ajenos(), ajenos)

    def test_registro_parcial_con_fechas_ausentes(self):
        faltan = (2, 8, 11)
        for paso in faltan:
            self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha=?", (fixtures.fecha_objetivo(fixtures.CORTES[0], paso),))
        r = self.registrar()
        e = r["evaluacion"]
        self.assertEqual((r["estado_registro"], r["estado_cobertura"]), ("creado", "parcial"))
        self.assertEqual(r["alcance_metricas"], "Métricas sobre observaciones disponibles.")
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (12, 3, 12 / 15))
        self.assertEqual(e["fechas_ausentes"], [fixtures.fecha_objetivo(fixtures.CORTES[0], p) for p in faltan])
        self.assertEqual(self.filas(), [(101, e["mae"], e["wape"], e["rmse"])])

    def test_sin_observaciones_no_inserta(self):
        self.datos.ejecutar("DELETE FROM venta_diaria")
        contenido = self.ruta.read_bytes()
        with self.vigilar() as (_, trazas, escrituras):
            r = self.registrar()
        self.assertEqual((r["estado_registro"], r["motivo"], r["hubo_escritura"]),
                         ("no_guardado", "sin_observaciones", False))
        self.assertEqual(r["estado_cobertura"], "sin_observaciones")
        self.assertEqual(tuple(r["evaluacion"][c] for c in ("mae", "wape", "rmse")), (None, None, None))
        self.assertEqual(self.filas(), [])
        self.assertEqual(escrituras, [])
        self.assertIn("ROLLBACK", trazas)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_sin_observaciones_con_fila_anterior_es_conflicto(self):
        self.registrar()
        anterior = self.filas()
        self.datos.ejecutar("DELETE FROM venta_diaria")
        with self.vigilar() as (_, _, escrituras):
            self.error("evaluacion_desactualizada")
        self.assertEqual(escrituras, [])
        self.assertEqual(self.filas(), anterior)

    def test_ventas_cero_guardan_wape_null_mae_y_rmse_validos(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        r = self.registrar()
        self.assertEqual(self.filas(), [(101, 2, None, 2)])
        self.assertEqual(self.datos.ejecutar("SELECT typeof(wape) FROM evaluacion"), [("null",)])
        self.assertEqual((r["evaluacion"]["mae"], r["evaluacion"]["rmse"]), (2, 2))
        with self.vigilar() as (_, _, escrituras):
            self.assertEqual(self.registrar()["estado_registro"], "reutilizado")
        self.assertEqual(escrituras, [])

    def test_un_solo_real_cero_admite_registro_parcial(self):
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha<>'2017-06-17'")
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        r = self.registrar()
        self.assertEqual(r["estado_cobertura"], "parcial")
        self.assertEqual(r["evaluacion"]["observaciones_evaluadas"], 1)
        self.assertEqual(self.filas(), [(101, 2, None, 2)])

    def test_reutilizacion_idempotente_sin_dml(self):
        primero = self.registrar()
        contenido, filas = self.ruta.read_bytes(), self.filas()
        with self.vigilar() as (_, trazas, escrituras):
            segundo = self.registrar()
        self.assertEqual(segundo, {**primero, "estado_registro": "reutilizado", "hubo_escritura": False})
        self.assertEqual(self.filas(), filas)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(escrituras, [])
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in trazas))

    def test_cambio_de_venta_real_detecta_conflicto(self):
        self.registrar()
        anterior = self.filas()
        self.datos.ejecutar("UPDATE venta_diaria SET sales=11 WHERE fecha='2017-06-17'")
        self.error("evaluacion_desactualizada")
        self.assertEqual(self.filas(), anterior)

    def test_cambio_de_estimacion_detecta_conflicto(self):
        self.registrar()
        anterior = self.filas()
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=9 WHERE paso=1")
        self.error("evaluacion_desactualizada")
        self.assertEqual(self.filas(), anterior)

    def test_fila_previa_corrupta_o_contradictoria_no_se_sobrescribe(self):
        self.registrar()
        for campo, valor in (("mae", -1), ("wape", -1), ("rmse", -1),
                              ("mae", inf), ("wape", inf), ("rmse", inf),
                              ("mae", "PRIVADO NO_NUMERICO"), ("mae", b"2"),
                              ("wape", None), ("rmse", None), ("rmse", 99)):
            with self.subTest(campo=campo, valor=valor):
                self.datos.ejecutar("UPDATE evaluacion SET mae=?,wape=?,rmse=?", (2, 10, sqrt(20 / 3)))
                self.datos.corromper(f"UPDATE evaluacion SET {campo}=?", (valor,))
                anterior = self.filas()
                self.error("evaluacion_desactualizada")
                self.assertEqual(self.filas(), anterior)

    def test_diferencia_de_un_ulp_no_se_acepta_por_tolerancia(self):
        self.registrar()
        for campo, original in (("mae", 2.0), ("wape", 10.0), ("rmse", sqrt(20 / 3))):
            with self.subTest(campo=campo):
                self.datos.ejecutar(f"UPDATE evaluacion SET {campo}=?", (nextafter(original, inf),))
                self.error("evaluacion_desactualizada")
                self.datos.ejecutar(f"UPDATE evaluacion SET {campo}=?", (original,))

    def test_representaciones_numericas_validas_reutilizan(self):
        self.datos.ejecutar("INSERT INTO evaluacion (pronostico_id,mae,wape,rmse) VALUES (101,2,10,?)", (sqrt(20 / 3),))
        with self.vigilar() as (_, _, escrituras):
            self.assertEqual(self.registrar()["estado_registro"], "reutilizado")
        self.assertEqual(escrituras, [])

    def test_wape_null_no_equivale_a_cero(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        self.datos.ejecutar("INSERT INTO evaluacion VALUES (101,2,0,2)")
        self.error("evaluacion_desactualizada")

    def test_predicciones_perfectas_guardan_ceros(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=10")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=10")
        self.registrar()
        self.assertEqual(self.filas(), [(101, 0, 0, 0)])
        self.datos.ejecutar("UPDATE evaluacion SET mae=-0.0,wape=0,rmse=0.0")
        self.assertEqual(self.registrar()["estado_registro"], "reutilizado")

    def test_metricas_iguales_no_demuestran_cobertura_historica_igual(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=8")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=10")
        self.assertEqual(self.registrar()["estado_cobertura"], "completa")
        self.datos.ejecutar("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
        r = self.registrar()
        self.assertEqual((r["estado_registro"], r["estado_cobertura"]), ("reutilizado", "parcial"))
        self.assertEqual(r["evaluacion"]["observaciones_evaluadas"], 14)
        self.assertEqual(self.filas(), [(101, 2, 25, 2)])

    def test_intentos_simultaneos_serializan_y_no_duplican(self):
        abrir_real = sqlite3.connect
        helper_real = consulta._evaluar_en_conexion
        barrera, segundo_intento, candado = Barrier(2), Event(), Lock()
        intentos = []

        class Concurrente(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                if sql == "BEGIN IMMEDIATE":
                    with candado:
                        intentos.append(True)
                        if len(intentos) == 2:
                            segundo_intento.set()
                return super().execute(sql, parametros)

        def abrir(ruta_uri, **kwargs):
            self.assertTrue(ruta_uri.endswith("?mode=rw"))
            return abrir_real(ruta_uri, factory=Concurrente, **kwargs)

        def calcular(conexion, identificador):
            # El segundo intento se anuncia ANTES de bloquearse en BEGIN.
            # No se espera a dos transacciones que posean el bloqueo a la vez.
            self.assertTrue(segundo_intento.wait(timeout=10))
            return helper_real(conexion, identificador)

        def trabajo():
            barrera.wait(timeout=10)
            return registro.registrar_evaluacion(self.ruta, 101)

        with patch.object(sqlite3, "connect", side_effect=abrir), \
                patch.object(consulta, "_evaluar_en_conexion", side_effect=calcular), \
                ThreadPoolExecutor(max_workers=2) as ejecutor:
            futuros = [ejecutor.submit(trabajo) for _ in range(2)]
            resultados = [f.result(timeout=30) for f in futuros]
        self.assertEqual(len(intentos), 2)
        self.assertEqual(sorted(r["estado_registro"] for r in resultados), ["creado", "reutilizado"])
        self.assertEqual(sum(r["hubo_escritura"] for r in resultados), 1)
        self.assertEqual({r["pronostico_id"] for r in resultados}, {101})
        self.assertEqual(self.filas(), [(101, 2, 10, sqrt(20 / 3))])

    def test_bloqueo_cubre_calculo_y_lectura_de_ventas(self):
        helper_real = consulta._evaluar_en_conexion

        def calcular(conexion, identificador):
            self.assertTrue(conexion.in_transaction)
            self.assertEqual(conexion.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            with closing(sqlite3.connect(self.ruta, timeout=0)) as otro:
                with self.assertRaises(sqlite3.OperationalError):
                    otro.execute("UPDATE venta_diaria SET sales=999")
            return helper_real(conexion, identificador)

        with patch.object(consulta, "_evaluar_en_conexion", side_effect=calcular) as helper:
            self.assertEqual(self.registrar()["evaluacion"]["mae"], 2)
        helper.assert_called_once()

    def test_fallo_durante_insert_revierte_y_conserva_fila_ajena(self):
        self.datos.crear_pronostico(102, 202, estimadas=fixtures.REALES)
        self.registrar(102)
        self.datos.ejecutar("CREATE TRIGGER fallo BEFORE INSERT ON evaluacion WHEN NEW.pronostico_id=101 "
                            "BEGIN SELECT RAISE(ABORT,'PRIVADO fallo INSERT'); END")
        anterior, ajenos = self.filas(), self.ajenos()
        self.error("error_persistencia")
        self.assertEqual(self.filas(), anterior)
        self.assertEqual(self.ajenos(), ajenos)

    def test_error_despues_del_insert_revierte_fila_pendiente(self):
        abrir_real = sqlite3.connect
        trazas = []

        class Fallida(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                cursor = super().execute(sql, parametros)
                if sql.startswith("INSERT INTO evaluacion"):
                    raise sqlite3.OperationalError("PRIVADO fallo tras INSERT")
                return cursor

        def abrir(ruta_uri, **kwargs):
            c = abrir_real(ruta_uri, factory=Fallida, **kwargs)
            c.set_trace_callback(trazas.append)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir):
            self.error("error_persistencia")
        self.assertIn("ROLLBACK", trazas)
        self.assertEqual(self.filas(), [])

    def test_insert_ignorado_no_devuelve_creado(self):
        self.datos.ejecutar("CREATE TRIGGER ignorar BEFORE INSERT ON evaluacion BEGIN SELECT RAISE(IGNORE); END")
        self.error("error_persistencia")
        self.assertEqual(self.filas(), [])

    def test_trigger_con_escritura_adicional_revierte_toda_la_operacion(self):
        self.datos.ejecutar("CREATE TRIGGER alterar AFTER INSERT ON evaluacion "
                            "BEGIN UPDATE venta_diaria SET sales=999 WHERE fecha='2017-06-17'; END")
        ajenos = self.ajenos()
        self.error("error_persistencia")
        self.assertEqual(self.filas(), [])
        self.assertEqual(self.ajenos(), ajenos)

    def test_pronostico_inexistente_o_de_conjunto_inactivo(self):
        self.error("pronostico_inexistente", 999)
        self.datos.crear_pronostico(102, 202, serie=24)
        self.error("pronostico_inactivo", 102)
        self.assertEqual(self.filas(), [])

    def test_sin_conjunto_activo(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")
        self.assertEqual(self.filas(), [])

    def test_corte_no_autorizado_o_tipo_no_evaluable_no_leen_ventas(self):
        casos = (("2017-07-31", "validacion", "corte_no_autorizado"),
                 ("2017-06-17", "validacion", "corte_no_autorizado"),
                 (fixtures.CORTES[0], "prueba", "tipo_no_evaluable"),
                 ("2017-07-31", "futuro", "tipo_no_evaluable"))
        for corte, tipo, codigo in casos:
            self.datos.ejecutar("UPDATE pronostico SET fecha_corte=?,tipo=?", (corte, tipo))
            with self.subTest(corte=corte, tipo=tipo), self.vigilar() as (llamadas, _, escrituras):
                self.error(codigo)
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))
            self.assertEqual(escrituras, [])

    def test_detalles_o_reales_invalidos_no_persisten(self):
        self.datos.ejecutar("UPDATE venta_diaria SET sales=-1 WHERE fecha='2017-06-17'")
        self.error("datos_inconsistentes")
        self.datos.ejecutar("UPDATE venta_diaria SET sales=10 WHERE fecha='2017-06-17'")
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE paso=8")
        self.error("datos_inconsistentes")
        self.assertEqual(self.filas(), [])

    def test_base_ausente_danada_y_sin_esquema(self):
        ausente = self.ruta.parent / "PRIVADO_ausente.sqlite3"
        self.error("base_no_disponible", ruta=ausente)
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("corrupta.sqlite3", b"Esto no es SQLite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            self.error("base_no_disponible", ruta=ruta)
            self.assertEqual(ruta.read_bytes(), contenido)

    def test_id_y_ruta_invalidos_controlados_antes_de_abrir(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")) as abrir:
            for identificador in (None, True, False, 0, -1, 101.0, "101", "101 OR 1=1", [], {}, 2**63):
                with self.subTest(identificador=identificador):
                    self.error("pronostico_invalido", identificador)
            for ruta in (None, 5, [], "", "PRIVADO\0ruta"):
                with self.subTest(ruta=ruta), self.assertRaises(registro.ErrorRegistroEvaluacion) as error:
                    registro.registrar_evaluacion(ruta, 101)
                self.assertEqual(error.exception.codigo, "base_no_disponible")
                self.assertNotIn("PRIVADO", str(error.exception))
            abrir.assert_not_called()

    def test_series_y_conjuntos_estrictamente_separados(self):
        self.datos.crear_pronostico(102, 202, serie=84, tecnica="random_forest", estimadas=[100] * 15)
        self.datos.insertar_reales(serie=84, valores=[100] * 15)
        self.datos.crear_pronostico(103, 203, serie=24, estimadas=[50] * 15)
        self.datos.insertar_reales(serie=24, valores=[50] * 15)
        a, b = self.registrar(), self.registrar(102)
        self.assertEqual((a["serie"]["id"], b["serie"]["id"]), (83, 84))
        self.assertEqual((a["evaluacion"]["mae"], b["evaluacion"]["mae"]), (2, 0))
        self.error("pronostico_inactivo", 103)
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
        self.assertEqual(self.registrar(103)["conjunto_id"], 2)
        self.error("pronostico_inactivo", 101)
        self.assertEqual(len(self.filas()), 3)

    def test_una_conexion_begin_immediate_antes_de_select_y_solo_insert_evaluacion(self):
        ajenos = self.ajenos()
        with self.vigilar() as (llamadas, trazas, escrituras):
            self.registrar()
        self.assertEqual(escrituras, ["evaluacion"])
        self.assertEqual(llamadas[0][:2], ("PRAGMA foreign_keys = ON", ()))
        self.assertEqual(llamadas[1][:2], ("BEGIN IMMEDIATE", ()))
        self.assertTrue(all(en_transaccion for _, _, en_transaccion in llamadas[2:]))
        self.assertEqual(trazas[-1], "COMMIT")
        inserts = [(sql, parametros) for sql, parametros, _ in llamadas if sql.startswith("INSERT")]
        self.assertEqual(inserts, [("INSERT INTO evaluacion (pronostico_id, mae, wape, rmse) VALUES (?, ?, ?, ?)",
                                    (101, 2, 10, sqrt(20 / 3)))])
        ventas = [(sql, p) for sql, p, _ in llamadas if "FROM venta_diaria" in sql]
        self.assertEqual(len(ventas), 1)
        self.assertIn("serie_id = ? AND fecha > ? AND fecha <= ?", ventas[0][0])
        self.assertEqual(ventas[0][1], (83, "2017-06-16", "2017-07-01"))
        self.assertEqual(self.ajenos(), ajenos)

    def test_los_tres_cortes_autorizados(self):
        for i, corte in enumerate(fixtures.CORTES):
            with self.subTest(corte=corte):
                if i:
                    self.datos.crear_pronostico(101 + i, 201 + i, corte=corte)
                    self.datos.insertar_reales(corte=corte)
                r = self.registrar(101 + i)
                self.assertEqual(r["fecha_corte"], corte)
                self.assertEqual((r["evaluacion"]["mae"], r["evaluacion"]["wape"], r["evaluacion"]["rmse"]),
                                 (2, 10, sqrt(20 / 3)))
        self.assertEqual(len(self.filas()), 3)

    def test_consulta_a2_sigue_ro_y_no_cambia_su_respuesta(self):
        esperado = self.datos.consultar()
        self.registrar()
        contenido = self.ruta.read_bytes()
        with self.datos.lectura_vigilada():
            self.assertEqual(consulta.consultar_evaluacion(self.ruta, 101), esperado)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_importacion_y_registro_sin_modelos_red_archivos_ni_subprocesos(self):
        codigo = compile(Path(registro.__file__).read_text(encoding="utf-8"), registro.__file__, "exec")
        aislado = ModuleType("registro_evaluaciones_en_prueba")
        importar_real = builtins.__import__
        prohibidos = ("backend.app.services.registro_pronosticos", "scripts.modelos_prediccion",
                      "scripts.linea_base_prediccion", "sklearn", "fastapi", "uvicorn")

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
            with patch.object(sqlite3, "connect", side_effect=AssertionError("Conexión al importar")):
                exec(codigo, aislado.__dict__)
            with self.vigilar():
                r = aislado.registrar_evaluacion(self.ruta, 101)
                self.assertEqual(r["estado_registro"], "creado")
            for doble in dobles:
                doble.assert_not_called()
        self.assertEqual(self.filas(), [(101, 2, 10, sqrt(20 / 3))])


if __name__ == "__main__":
    unittest.main()
