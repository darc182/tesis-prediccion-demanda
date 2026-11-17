"""S4-A4-03: registro exclusivamente en bases temporales sintéticas."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
from threading import Barrier
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import registro_simulaciones as registro
from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.services import trayectoria_simulacion as trayectoria
from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.services import simulacion_desde_pronostico as lectura
from backend.app.services import simulacion_reposicion as motor
from backend.app.storage import migraciones_simulacion as esquema
from tests import test_consulta_evaluaciones as fixtures


def escenarios(n=2):
    return [{"etiqueta": e, "inventario_disponible": 0, "tiempo_abastecimiento": l, "reserva": 0}
            for e, l in zip("ABC"[:n], (0, 2, 15))]


def proceso_registro(ruta, clave, barrera, salida):
    original = trayectoria.calcular_trayectoria
    def calcular(*args):
        barrera.wait(timeout=30)
        return original(*args)
    try:
        with patch.object(trayectoria, "calcular_trayectoria", side_effect=calcular):
            r = registro.registrar_simulacion(ruta, 101, 0, 0, 0, clave_idempotencia=clave)
        salida.put(("ok", r["estado_registro"], r["ejecucion_id"]))
    except registro.ErrorRegistroSimulacion as error:
        salida.put(("rechazo", error.codigo))
    except Exception as error:
        salida.put(("inesperado", type(error).__name__))


def proceso_cambio(ruta, modo, permiso, terminado, salida):
    try:
        if not permiso.wait(30):
            raise TimeoutError("Coordinacion temporal agotada")
        with closing(sqlite3.connect(Path(ruta).resolve().as_uri() + "?mode=rw", uri=True)) as c, c:
            if modo == "activo":
                c.execute("UPDATE conjunto_datos SET activo=0 WHERE id=7")
            elif modo == "detalle":
                c.execute("UPDATE detalle_pronostico SET venta_estimada=11 WHERE pronostico_id=101 AND paso=1")
            else:
                c.execute("UPDATE pronostico SET fecha_generacion='2026-01-04T00:00:00' WHERE id=101")
        salida.put("ok")
    except Exception as error:
        salida.put(type(error).__name__)
    finally:
        terminado.set()


class RegistroSimulacionesTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.ConsultaEvaluacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        esquema.preparar_esquema_simulaciones(self.ruta)
        self.datos.ejecutar("INSERT INTO simulacion VALUES (1,101,0,0,0,0,0,'legado','2017-06-16')")

    def individual(self, clave="accion", identificador=101, inventario=30, plazo=0, reserva=5, ruta=None):
        return registro.registrar_simulacion(self.ruta if ruta is None else ruta, identificador, inventario, plazo, reserva,
                                            clave_idempotencia=clave)

    def comparar(self, clave="accion", n=2, seleccion=None):
        return registro.registrar_comparacion_simulaciones(self.ruta, 101, escenarios(n) if seleccion is None else seleccion,
                                                         clave_idempotencia=clave)

    def tablas(self):
        return [self.datos.ejecutar("SELECT * FROM ejecucion_simulacion ORDER BY id"),
                self.datos.ejecutar("SELECT * FROM escenario_simulacion ORDER BY ejecucion_id,posicion")]

    def antiguas(self):
        return [self.datos.ejecutar(f'SELECT * FROM "{t}" ORDER BY 1,2') for t in registro._BASE]

    def error(self, codigo, funcion=None, **kwargs):
        with self.assertRaises(registro.ErrorRegistroSimulacion) as capturado:
            (self.individual if funcion is None else funcion)(**kwargs)
        self.assertEqual(capturado.exception.codigo, codigo)
        for texto in ("PRIVADO", str(self.ruta), "INSERT", "SELECT", "Traceback"):
            self.assertNotIn(texto, str(capturado.exception))

    @contextmanager
    def vigilar(self, clase=sqlite3.Connection):
        abrir, conexiones, llamadas, denegadas = sqlite3.connect, [], [], []
        def autorizar(accion, tabla, columna, base, origen):
            prohibido = ((accion == sqlite3.SQLITE_READ and (tabla in ("venta_diaria", "evaluacion", "simulacion")
                          or columna in ("configuracion_json", "ruta_modelo")))
                         or (accion in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE)
                             and tabla not in ("ejecucion_simulacion", "escenario_simulacion")))
            if prohibido:
                denegadas.append((accion, tabla, columna))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        def conectar(ruta, **opciones):
            if opciones.get("uri"):
                c = abrir(ruta, factory=clase, **opciones)
                conexiones.append((ruta, c))
                c.set_authorizer(autorizar)
                c.set_trace_callback(lambda sql: llamadas.append((ruta, sql)))
                return c
            return abrir(ruta, **opciones)
        with patch.object(registro.sqlite3, "connect", side_effect=conectar):
            yield conexiones, llamadas, denegadas
        for _, c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                c.execute("SELECT 1")

    def test_individual_creado_y_manual(self):
        antes = self.antiguas()
        r = self.individual()
        self.assertEqual((r["estado_registro"], r["hubo_escritura"], r["modalidad"], r["numero_escenarios"]), ("creado", True, "individual", 1))
        self.assertEqual(r["historico"]["resultados"][0]["simulacion"]["reposicion"], 285)
        self.assertEqual(r["historico"]["resultados"][0]["simulacion"]["resumen"]["balance_neto_final"], 5)
        self.assertEqual([len(t) for t in self.tablas()], [1, 1])
        self.assertEqual(self.antiguas(), antes)

    def test_comparacion_ab(self):
        r = self.comparar()
        self.assertEqual([e["etiqueta"] for e in r["historico"]["resultados"]], ["A", "B"])
        self.assertEqual([e["simulacion"]["reposicion"] for e in r["historico"]["resultados"]], [310, 310])
        self.assertEqual(r["historico"]["resultados"][1]["simulacion"]["resumen"]["mayor_demanda_pendiente_actual"], 32)
        self.assertEqual([len(t) for t in self.tablas()], [1, 2])

    def test_comparacion_abc_y_plazos(self):
        r = self.comparar(n=3)
        resultados = r["historico"]["resultados"]
        self.assertEqual([e["simulacion"]["reposicion"] for e in resultados], [310]*3)
        self.assertEqual(resultados[2]["simulacion"]["resumen"]["balance_neto_final"], -310)
        self.assertFalse(resultados[2]["simulacion"]["recepcion_dentro_horizonte"])
        self.assertEqual([len(t) for t in self.tablas()], [1, 3])

    def test_idempotencia_documentos_ids_y_sin_dml(self):
        a = self.individual()
        antes = self.tablas()
        with self.vigilar() as (conexiones, llamadas, denegadas), \
             patch.object(trayectoria, "calcular_trayectoria", side_effect=AssertionError("No recalcular")):
            b = self.individual()
        self.assertEqual(a["ejecucion_id"], b["ejecucion_id"])
        self.assertEqual(a["historico"], b["historico"])
        self.assertEqual(b["estado_registro"], "reutilizado")
        self.assertFalse(b["hubo_escritura"])
        self.assertEqual(self.tablas(), antes)
        self.assertEqual(len(conexiones), 1)
        self.assertFalse(denegadas)
        self.assertFalse(any(sql.lstrip().startswith(("INSERT", "UPDATE", "DELETE")) for _, sql in llamadas))

    def test_reintento_inactivo_y_origen_modificado_sin_leerlo(self):
        a = self.comparar(n=3)
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=999 WHERE pronostico_id=101")
        with patch.object(lectura, "_leer_pronostico_simulable_en_conexion", side_effect=AssertionError("No leer origen")), \
             patch.object(comparacion, "comparar_escenarios", side_effect=AssertionError("No recalcular")):
            b = self.comparar(n=3)
        self.assertEqual(a["historico"], b["historico"])
        self.assertEqual(a["ejecucion_id"], b["ejecucion_id"])

    def test_reintento_no_depende_del_motor_vigente(self):
        a = self.individual()
        with patch.object(motor, "VERSION_REGLA", "futura"), patch.object(motor, "TECNICAS", ()), \
             patch.object(snapshot, "preparar_snapshot", side_effect=AssertionError("No preparar")):
            b = self.individual()
        self.assertEqual(a["historico"], b["historico"])

    def comprobar_reintento_futuro(self, operacion, cambios):
        original = operacion()
        antes = self.tablas()
        with self.vigilar() as (conexiones, llamadas, denegadas), ExitStack() as pila:
            validadores = [pila.enter_context(patch.object(modulo, nombre, side_effect=error))
                           for modulo, nombre, error in cambios]
            for modulo, nombre in ((trayectoria, "calcular_trayectoria"), (comparacion, "comparar_escenarios"),
                                   (snapshot, "preparar_snapshot"), (lectura, "_leer_pronostico_simulable_en_conexion")):
                pila.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("No ejecutar en reintento")))
            recuperado = operacion()
            for validar in validadores:
                validar.assert_not_called()
        self.assertEqual(original["ejecucion_id"], recuperado["ejecucion_id"])
        self.assertEqual(snapshot.serializar_documento(original["historico"]), snapshot.serializar_documento(recuperado["historico"]))
        self.assertEqual(recuperado["estado_registro"], "reutilizado")
        self.assertFalse(recuperado["hubo_escritura"])
        self.assertEqual(self.tablas(), antes)  # Incluye textos y hashes exactos.
        self.assertEqual(len(conexiones), 1)
        self.assertFalse(denegadas)
        self.assertFalse(any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for _, sql in llamadas))

    def validadores_futuros(self):
        return [(motor, "_numero", motor.ErrorSimulacionReposicion("valor_invalido")),
                (comparacion, "_seleccion", comparacion.ErrorComparacionSimulaciones("escenarios_invalidos")),
                (snapshot, "_parametros", snapshot.ErrorSnapshotSimulacion("estructura_invalida"))]

    def test_historico_individual_numero_vigente_alterado(self):
        self.comprobar_reintento_futuro(self.individual, self.validadores_futuros()[:1])

    def test_historico_ab_y_abc_seleccion_vigente_alterada(self):
        for n in (2, 3):
            with self.subTest(n=n):
                self.comprobar_reintento_futuro(lambda: self.comparar(clave=str(n), n=n), self.validadores_futuros()[1:2])

    def test_historico_parametros_snapshot_vigente_alterados(self):
        for n in (1, 2, 3):
            with self.subTest(n=n):
                operacion = (lambda: self.individual(clave=str(n))) if n == 1 else (lambda: self.comparar(clave=str(n), n=n))
                self.comprobar_reintento_futuro(operacion, self.validadores_futuros()[2:])

    def paquete_guardado(self, clave):
        with closing(sqlite3.connect(self.ruta)) as c:
            c.row_factory = sqlite3.Row
            fila = c.execute("SELECT * FROM ejecucion_simulacion WHERE clave_idempotencia=?", (clave,)).fetchone()
            p = {k: fila[k] for k in registro._PAQUETE}
            p["escenarios"] = [dict(f) for f in c.execute(
                "SELECT etiqueta,posicion,resultado_json,sha256_resultado FROM escenario_simulacion WHERE ejecucion_id=? ORDER BY posicion", (fila["id"],))]
            return p

    def test_lectura_directa_v1_sin_validadores_vigentes(self):
        original = self.comparar(n=3)
        p = self.paquete_guardado("accion")
        antes = deepcopy(p)
        with ExitStack() as pila:
            for modulo, nombre, error in self.validadores_futuros():
                pila.enter_context(patch.object(modulo, nombre, side_effect=error))
            self.assertEqual(snapshot.serializar_documento(snapshot.verificar_paquete(p)),
                             snapshot.serializar_documento(original["historico"]))
        self.assertEqual(p, antes)

    def test_conflicto_historico_bajo_todos_validadores_futuros(self):
        self.individual()
        antes = self.tablas()
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        with ExitStack() as pila:
            for modulo, nombre, error in self.validadores_futuros():
                pila.enter_context(patch.object(modulo, nombre, side_effect=error))
            pila.enter_context(patch.object(lectura, "_leer_pronostico_simulable_en_conexion", side_effect=AssertionError("No consultar origen")))
            self.error("conflicto_idempotencia", inventario=31)
        self.assertEqual(self.tablas(), antes)

    def test_clave_nueva_conserva_cada_validacion_vigente(self):
        self.individual(clave="existente")
        antes = self.tablas()
        for i, (modulo, nombre, error) in enumerate(self.validadores_futuros()):
            with self.subTest(nombre=nombre), patch.object(modulo, nombre, side_effect=error) as validar, \
                 patch.object(lectura, "_leer_pronostico_simulable_en_conexion", side_effect=AssertionError("Validar antes de leer origen")):
                self.error("escenarios_invalidos" if nombre == "_seleccion" else "parametros_invalidos",
                           funcion=self.comparar if nombre == "_seleccion" else self.individual, clave="nueva"+str(i))
                validar.assert_called_once()
            self.assertEqual(self.tablas(), antes)

    def test_preparador_nuevo_conserva_parametros_vigentes(self):
        self.individual()
        p = self.paquete_guardado("accion")
        h = snapshot.verificar_paquete(p)
        origen = h["origen"]
        cabecera = {"id": origen["pronostico_id"], **{k: origen[k] for k in ("experimento_id", "tecnica", "fecha_corte", "horizonte")}}
        with patch.object(snapshot, "_parametros", side_effect=snapshot.ErrorSnapshotSimulacion("estructura_invalida")) as validar:
            with self.assertRaises(snapshot.ErrorSnapshotSimulacion):
                snapshot.preparar_snapshot(cabecera, origen["detalles"], {k: origen[k] for k in ("conjunto_id", "serie")},
                                           h["solicitud"]["escenarios"], h["resultados"][0]["simulacion"], modalidad="individual")
            self.assertTrue(validar.called)

    def test_estructura_v1_invalida_antes_de_sqlite_y_validadores(self):
        with ExitStack() as pila:
            for modulo, nombre, _ in self.validadores_futuros():
                pila.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("No ejecutar validadores vigentes")))
            pila.enter_context(patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir SQLite")))
            for opciones in ({"inventario": True}, {"inventario": "1"}, {"reserva": -1}, {"reserva": math.nan},
                            {"reserva": math.inf}, {"plazo": True}, {"plazo": 16}, {"inventario": 10**400}):
                with self.subTest(opciones=opciones):
                    self.error("parametros_invalidos", **opciones)
            self.error("precision_no_representable", inventario=2**53+1)
            for seleccion in ([], escenarios(3)[::-1], [{"etiqueta": "A"}], escenarios()+[escenarios()[0]],
                              [{**escenarios()[0], "campo_ajeno": 0}, escenarios()[1]]):
                self.error("escenarios_invalidos", funcion=self.comparar, seleccion=seleccion)

    def test_identidad_ceros_v1_bajo_validadores_futuros(self):
        self.individual(inventario=0)
        antes = self.tablas()
        with ExitStack() as pila:
            for modulo, nombre, error in self.validadores_futuros():
                pila.enter_context(patch.object(modulo, nombre, side_effect=error))
            self.assertEqual(self.individual(inventario=0)["estado_registro"], "reutilizado")
            for valor in (0.0, -0.0):
                self.error("conflicto_idempotencia", inventario=valor)
        self.assertEqual(self.tablas(), antes)

    def test_binary64_historico_bajo_validadores_futuros(self):
        casos = (([math.ulp(0.0)]+[0.0]*14, 0), ([0.1]*3+[0.0]*12, 0.3), ([1e308]+[0.0]*14, 1e308))
        for i, (valores, inventario) in enumerate(casos):
            self.datos.crear_pronostico(102+i, 202+i, estimadas=valores)
            with self.subTest(i=i):
                self.comprobar_reintento_futuro(
                    lambda: self.individual(identificador=102+i, inventario=inventario, reserva=0, clave=str(i)), self.validadores_futuros())

    def test_integridad_v1_con_validador_vigente_permisivo(self):
        self.individual()
        base = self.paquete_guardado("accion")
        for campo, valor in (("reserva", True), ("reserva", -1), ("inventario_disponible", 2**53+1),
                             ("tiempo_abastecimiento", True), ("tiempo_abastecimiento", 16)):
            p = deepcopy(base)
            solicitud = json.loads(p["solicitud_json"])
            solicitud["escenarios"][0][campo] = valor
            p["solicitud_json"] = snapshot.serializar_documento(solicitud)
            p["sha256_solicitud"] = snapshot.calcular_sha256(p["solicitud_json"])
            with self.subTest(campo=campo, valor=valor), patch.object(snapshot, "_parametros", return_value=None):
                with self.assertRaises(snapshot.ErrorSnapshotSimulacion):
                    snapshot.verificar_paquete(p)

    def test_dos_claves_mismos_parametros_crean_dos(self):
        a, b = self.individual(clave="uno"), self.individual(clave="dos")
        self.assertNotEqual(a["ejecucion_id"], b["ejecucion_id"])
        self.assertEqual(a["historico"], b["historico"])

    def test_conflicto_parametros_modalidad_y_pronostico(self):
        self.individual()
        self.datos.crear_pronostico(102, 202)
        for opciones in ({"inventario": 31}, {"reserva": 6}, {"plazo": 1}, {"identificador": 102}):
            with self.subTest(opciones=opciones):
                self.error("conflicto_idempotencia", **opciones)
        self.error("conflicto_idempotencia", funcion=self.comparar)
        self.assertEqual([len(t) for t in self.tablas()], [1, 1])

    def test_solicitudes_cero_float_y_menos_cero_distintas(self):
        self.individual(inventario=0)
        for valor in (0.0, -0.0):
            self.error("conflicto_idempotencia", inventario=valor)
        a = self.individual(inventario=0.0, clave="float")
        b = self.individual(inventario=-0.0, clave="negativo")
        self.assertNotEqual(snapshot.serializar_documento(a["historico"]["solicitud"]), snapshot.serializar_documento(b["historico"]["solicitud"]))

    def test_subnormal_residuo_y_magnitudes_grandes(self):
        for i, (valores, inventario) in enumerate((([math.ulp(0.0)]+[0.0]*14, 0), ([0.1]*3+[0.0]*12, 0.3), ([1e308]+[0.0]*14, 1e308))):
            pid = 102+i
            self.datos.crear_pronostico(pid, 202+i, estimadas=valores)
            r = self.individual(identificador=pid, inventario=inventario, reserva=0, clave=str(i))
            q = r["historico"]["resultados"][0]["simulacion"]["reposicion"]
            if i < 2:
                self.assertGreater(q, 0)
            self.assertEqual(r["historico"], self.individual(identificador=pid, inventario=inventario, reserva=0, clave=str(i))["historico"])

    def test_clave_utf8_limite_y_rechazos(self):
        self.individual(clave="é"*64)
        for clave in (None, 1, True, "", " ", "a\0b", "é"*65, "\ud800"):
            with self.subTest(clave=clave):
                self.error("clave_idempotencia_invalida", clave=clave)

    def test_identificadores_invalidos_antes_de_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for pid in (0, -1, True, "101", 1.5, 2**63):
                self.error("pronostico_invalido", identificador=pid)

    def test_parametros_invalidos_antes_de_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for opciones in ({"inventario": True}, {"inventario": "1"}, {"inventario": -1}, {"reserva": math.nan}, {"reserva": math.inf}, {"plazo": True}, {"plazo": 16}):
                self.error("parametros_invalidos", **opciones)
            self.error("precision_no_representable", inventario=2**53+1)

    def test_escenarios_invalidos(self):
        for seleccion in ([], [{"etiqueta": "A"}], escenarios(3)[::-1], escenarios()+[escenarios()[0]]):
            self.error("escenarios_invalidos", funcion=self.comparar, seleccion=seleccion)

    def test_tres_cortes_y_tres_tecnicas_y_aislamiento(self):
        for i, (corte, tecnica) in enumerate(zip(fixtures.CORTES, motor.TECNICAS)):
            self.datos.crear_pronostico(102+i, 202+i, serie=84, corte=corte, tecnica=tecnica)
            r = self.individual(identificador=102+i, clave=str(i))
            self.assertEqual(r["historico"]["origen"]["serie"]["id"], 84)
            self.assertEqual(r["historico"]["origen"]["tecnica"], tecnica)
            self.assertEqual(r["historico"]["origen"]["fecha_corte"], corte)

    def test_suficiente_reserva_y_plazo_catorce(self):
        r = self.individual(inventario=400, reserva=0, plazo=14)
        self.assertEqual(r["historico"]["resultados"][0]["simulacion"]["reposicion"], 0)
        self.assertEqual(r["historico"]["resultados"][0]["simulacion"]["fecha_recepcion"], "2017-07-01")

    def test_inexistente_inactivo_y_sin_conjunto(self):
        self.error("pronostico_inexistente", identificador=999)
        self.datos.crear_pronostico(102, 202, serie=24)
        self.error("pronostico_inactivo", identificador=102)
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_tipos_y_corte_no_elegibles(self):
        for i, tipo in enumerate(("prueba", "futuro")):
            self.datos.crear_pronostico(102+i, 202+i, tipo=tipo)
            self.error("tipo_no_evaluable", identificador=102+i)
        self.datos.crear_pronostico(104, 204, corte="2017-06-15")
        self.error("corte_no_autorizado", identificador=104)

    def test_detalles_incompletos_y_corruptos(self):
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE pronostico_id=101 AND paso=15")
        self.error("pronostico_corrupto")
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])

    def test_detalle_fecha_incorrecta_y_negativo(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET fecha_objetivo='2017-06-16' WHERE paso=1")
        self.error("pronostico_corrupto")
        self.datos.ejecutar("UPDATE detalle_pronostico SET fecha_objetivo='2017-06-17',venta_estimada=-1 WHERE paso=1")
        self.error("pronostico_corrupto")

    def test_fuente_no_finita_y_referencia_rota(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=? WHERE paso=1", (math.inf,))
        self.error("pronostico_corrupto")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=8 WHERE paso=1")
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("UPDATE pronostico SET experimento_id=999 WHERE id=101")
        self.error("pronostico_corrupto")

    def test_precision_y_desbordamiento_del_resultado(self):
        for i, (valores, codigo) in enumerate((([1e16, 1.0]+[0.0]*13, "precision_no_representable"),
                                               ([1e308, 1e308]+[0.0]*13, "resultado_no_finito"))):
            self.datos.crear_pronostico(102+i, 202+i, estimadas=valores)
            self.error(codigo, identificador=102+i, inventario=0, reserva=0)
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])

    def test_conflicto_se_resuelve_antes_de_actividad_actual(self):
        self.individual()
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        with patch.object(lectura, "_leer_pronostico_simulable_en_conexion", side_effect=AssertionError("No leer origen")):
            self.error("conflicto_idempotencia", inventario=31)

    def test_base_ausente_danada_sin_esquema(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        self.error("base_no_disponible", ruta=ausente)
        self.assertFalse(ausente.exists())
        danada = self.ruta.with_name("danada.sqlite3")
        danada.write_bytes(b"sintetica corrupta")
        self.error("base_no_disponible", ruta=danada)
        vacia = self.ruta.with_name("vacia.sqlite3")
        with closing(sqlite3.connect(vacia)):
            pass
        self.error("esquema_no_preparado", ruta=vacia)

    def test_esquema_sin_preparar_y_parcial(self):
        for t in reversed(esquema.TABLAS_NUEVAS):
            self.datos.ejecutar(f"DROP TABLE {t}")
        self.error("esquema_no_preparado")
        self.datos.ejecutar(esquema._DDL[0])
        self.error("esquema_incompatible")

    def test_version_futura_y_marcador_inconsistente(self):
        self.datos.ejecutar("UPDATE version_esquema_simulaciones SET version=2")
        self.error("version_no_soportada")
        self.datos.ejecutar("DELETE FROM version_esquema_simulaciones")
        self.error("esquema_incompatible")

    def test_esquema_indice_adulterado(self):
        self.datos.ejecutar("CREATE UNIQUE INDEX ajeno ON ejecucion_simulacion(pronostico_id)")
        self.error("esquema_incompatible")

    def test_esquema_fk_adulterada_y_columnas_originales(self):
        self.datos.ejecutar("DROP TABLE escenario_simulacion")
        self.datos.ejecutar(esquema._DDL[2].replace("ON DELETE RESTRICT", "ON DELETE CASCADE"))
        self.error("esquema_incompatible")
        self.datos.ejecutar("DROP TABLE escenario_simulacion")
        self.datos.ejecutar(esquema._DDL[2])
        self.datos.ejecutar("ALTER TABLE pronostico RENAME COLUMN duracion_segundos TO ajeno")
        self.error("esquema_incompatible")

    def test_trigger_mayusculas_rechazado(self):
        self.datos.ejecutar('CREATE TRIGGER ajeno AFTER INSERT ON "ESCENARIO_SIMULACION" BEGIN DELETE FROM escenario_simulacion; END')
        self.error("esquema_incompatible")

    def test_registro_parcial_y_hash_corrupto(self):
        self.comparar(n=3)
        self.datos.ejecutar("DELETE FROM escenario_simulacion WHERE etiqueta='C'")
        self.error("snapshot_corrupto", funcion=self.comparar, n=3)
        self.datos.ejecutar("UPDATE ejecucion_simulacion SET sha256_origen=?", ("0"*64,))
        self.error("snapshot_corrupto", funcion=self.comparar, n=3)

    def test_version_documento_y_fecha_adulteradas(self):
        self.individual()
        self.datos.ejecutar("UPDATE ejecucion_simulacion SET version_documento=2")
        self.error("snapshot_incompatible")
        self.datos.corromper("UPDATE ejecucion_simulacion SET version_documento=1,fecha_registro='PRIVADO'")
        self.error("snapshot_corrupto")

    def test_snapshot_adulterado_con_hash_nuevo(self):
        self.individual()
        texto = self.datos.ejecutar("SELECT origen_json FROM ejecucion_simulacion")[0][0]
        d = json.loads(texto)
        d["detalles"][0]["venta_estimada"] = 999
        texto = snapshot.serializar_documento(d)
        self.datos.ejecutar("UPDATE ejecucion_simulacion SET origen_json=?,sha256_origen=?", (texto, snapshot.calcular_sha256(texto)))
        self.error("snapshot_corrupto")

    def test_delegacion_unica_y_equivalencia_pura(self):
        with patch.object(trayectoria, "calcular_trayectoria", wraps=trayectoria.calcular_trayectoria) as calcular:
            r = self.individual()
        calcular.assert_called_once()
        self.assertEqual(r["historico"]["resultados"][0]["simulacion"],
                         trayectoria.calcular_trayectoria(*calcular.call_args.args))
        seleccion = escenarios(3)
        antes = deepcopy(seleccion)
        with patch.object(comparacion, "comparar_escenarios", wraps=comparacion.comparar_escenarios) as comparar:
            self.comparar(clave="comparada", seleccion=seleccion)
        comparar.assert_called_once()
        self.assertEqual(seleccion, antes)

    def test_lectura_ro_calculo_fuera_transaccion_y_write(self):
        antes = self.antiguas()
        original = trayectoria.calcular_trayectoria
        with self.vigilar() as (conexiones, llamadas, denegadas):
            def calcular(*args):
                self.assertEqual(len(conexiones), 1)
                with self.assertRaises(sqlite3.ProgrammingError):
                    conexiones[0][1].execute("SELECT 1")
                return original(*args)
            with patch.object(trayectoria, "calcular_trayectoria", side_effect=calcular), \
                 patch.object(esquema, "preparar_esquema_simulaciones", side_effect=AssertionError("No preparar")):
                self.individual()
        self.assertEqual([uri.rsplit("?", 1)[1] for uri, _ in conexiones], ["mode=ro", "mode=rw"])
        self.assertEqual([s for _, s in llamadas if s.startswith("BEGIN")], ["BEGIN", "BEGIN IMMEDIATE"])
        self.assertFalse(denegadas)
        self.assertFalse(any(s.startswith(("CREATE", "ALTER", "DROP")) for _, s in llamadas))
        self.assertEqual(self.antiguas(), antes)

    def cambio_durante_calculo(self, sql):
        original = trayectoria.calcular_trayectoria
        def calcular(*args):
            self.datos.ejecutar(sql)
            return original(*args)
        with patch.object(trayectoria, "calcular_trayectoria", side_effect=calcular):
            self.error("origen_modificado")
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])

    def test_cambio_conjunto_durante_calculo(self):
        self.cambio_durante_calculo("UPDATE conjunto_datos SET activo=0")

    def test_cambio_detalle_durante_calculo(self):
        self.cambio_durante_calculo("UPDATE detalle_pronostico SET venta_estimada=11 WHERE paso=1")

    def test_cambio_metadato_durante_calculo(self):
        self.cambio_durante_calculo("UPDATE pronostico SET fecha_generacion='2026-01-04T00:00:00' WHERE id=101")

    def test_cambio_serie_durante_calculo(self):
        self.cambio_durante_calculo("UPDATE serie_temporal SET family='CAMBIADA' WHERE id=83")

    def fallo(self, momento):
        self.individual(clave="conservar")
        antes, antiguas = self.tablas(), self.antiguas()
        class Fallida(sqlite3.Connection):
            hijos = 0
            def execute(c, sql, parametros=()):
                if momento == "antes_cabecera" and sql.startswith("INSERT INTO ejecucion_simulacion"):
                    raise sqlite3.OperationalError("PRIVADO")
                cursor = super().execute(sql, parametros)
                if momento == "despues_cabecera" and sql.startswith("INSERT INTO ejecucion_simulacion"):
                    raise sqlite3.OperationalError("PRIVADO")
                if sql.startswith("INSERT INTO escenario_simulacion"):
                    c.hijos += 1
                    if (momento == "primer_hijo" and c.hijos == 1) or (momento == "segundo_hijo" and c.hijos == 2):
                        raise sqlite3.OperationalError("PRIVADO")
                return cursor
            def commit(c):
                if momento == "commit":
                    raise sqlite3.OperationalError("PRIVADO")
                return super().commit()
        with self.vigilar(Fallida) as (_, llamadas, _):
            self.error("error_persistencia", funcion=self.comparar, clave="revertida", n=3)
        self.assertTrue(any(s == "ROLLBACK" for _, s in llamadas))
        self.assertEqual(self.tablas(), antes)
        self.assertEqual(self.antiguas(), antiguas)
        self.assertEqual(self.comparar(clave="revertida", n=3)["estado_registro"], "creado")

    def test_rollback_antes_insert_cabecera(self): self.fallo("antes_cabecera")
    def test_rollback_despues_insert_cabecera(self): self.fallo("despues_cabecera")
    def test_rollback_primer_escenario(self): self.fallo("primer_hijo")
    def test_rollback_segundo_escenario(self): self.fallo("segundo_hijo")
    def test_rollback_antes_commit(self): self.fallo("commit")

    def test_rollback_integridad_readback(self):
        recuperar = registro._recuperar_en_conexion
        llamadas = 0
        def alterada(*args):
            nonlocal llamadas
            llamadas += 1
            if llamadas == 3:
                raise registro.ErrorRegistroSimulacion("snapshot_corrupto")
            return recuperar(*args)
        with patch.object(registro, "_recuperar_en_conexion", side_effect=alterada):
            self.error("error_persistencia")
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])
        self.assertEqual(self.individual()["estado_registro"], "creado")

    def test_rollback_insercion_ignorada(self):
        class Ignorada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                if sql.startswith("INSERT INTO escenario_simulacion"):
                    return super().execute("SELECT 1")
                return super().execute(sql, parametros)
        with self.vigilar(Ignorada):
            self.error("error_persistencia")
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])
        self.assertEqual(self.individual()["estado_registro"], "creado")

    def carrera(self, iguales=True, claves_iguales=True, pronosticos_iguales=True):
        if not pronosticos_iguales:
            self.datos.crear_pronostico(102, 202)
        barrera = Barrier(2)
        original = trayectoria.calcular_trayectoria
        def calcular(*args):
            barrera.wait(timeout=15)
            return original(*args)
        def ejecutar(i):
            try:
                return self.individual(clave="misma" if claves_iguales else str(i), inventario=0 if iguales else i,
                                       identificador=101 if pronosticos_iguales or i == 0 else 102)
            except registro.ErrorRegistroSimulacion as error:
                return error.codigo
        with patch.object(trayectoria, "calcular_trayectoria", side_effect=calcular), ThreadPoolExecutor(max_workers=2) as pool:
            futuros = [pool.submit(ejecutar, i) for i in range(2)]
            return [f.result(timeout=25) for f in futuros]

    def test_carrera_misma_clave_y_solicitud(self):
        r = self.carrera()
        self.assertEqual(sorted(x["estado_registro"] for x in r), ["creado", "reutilizado"])
        self.assertEqual(r[0]["ejecucion_id"], r[1]["ejecucion_id"])
        self.assertEqual([len(t) for t in self.tablas()], [1, 1])

    def test_carrera_clave_igual_parametros_distintos(self):
        r = self.carrera(iguales=False)
        self.assertEqual(sum(x == "conflicto_idempotencia" for x in r), 1)
        self.assertEqual([len(t) for t in self.tablas()], [1, 1])

    def test_carrera_clave_igual_pronosticos_distintos(self):
        r = self.carrera(pronosticos_iguales=False)
        self.assertEqual(sum(x == "conflicto_idempotencia" for x in r), 1)
        self.assertEqual([len(t) for t in self.tablas()], [1, 1])

    def test_carrera_claves_distintas_parametros_iguales(self):
        r = self.carrera(claves_iguales=False)
        self.assertTrue(all(x["estado_registro"] == "creado" for x in r))
        self.assertEqual([len(t) for t in self.tablas()], [2, 2])

    def terminar_procesos(self, procesos):
        for p in procesos:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                self.fail("Proceso sintético de prueba agotó el tiempo")
            self.assertEqual(p.exitcode, 0)

    def test_dos_procesos_default_y_wal(self):
        contexto = multiprocessing.get_context("spawn")
        for journal in ("delete", "wal"):
            with self.subTest(journal=journal):
                modo = self.datos.ejecutar("PRAGMA journal_mode=" + journal)[0][0]
                self.assertEqual(modo, journal)
                antes = len(self.tablas()[0])
                barrera, salida = contexto.Barrier(2), contexto.Queue()
                procesos = [contexto.Process(target=proceso_registro, args=(str(self.ruta), journal, barrera, salida)) for _ in range(2)]
                for p in procesos: p.start()
                self.terminar_procesos(procesos)
                resultados = [salida.get(timeout=5) for _ in range(2)]
                salida.close()
                salida.join_thread()
                self.assertTrue(all(r[0] == "ok" for r in resultados), resultados)
                self.assertEqual(sorted(r[1] for r in resultados), ["creado", "reutilizado"])
                self.assertEqual(resultados[0][2], resultados[1][2])
                self.assertEqual(len(self.tablas()[0]), antes + 1)

    def test_wal_proceso_cambia_activo_detalles_metadatos(self):
        self.assertEqual(self.datos.ejecutar("PRAGMA journal_mode=WAL")[0][0], "wal")
        contexto = multiprocessing.get_context("spawn")
        original = trayectoria.calcular_trayectoria
        for modo in ("detalle", "metadato", "activo"):
            with self.subTest(modo=modo):
                permiso, terminado, salida = contexto.Event(), contexto.Event(), contexto.Queue()
                proceso = contexto.Process(target=proceso_cambio, args=(str(self.ruta), modo, permiso, terminado, salida))
                proceso.start()
                def calcular(*args):
                    permiso.set()
                    self.assertTrue(terminado.wait(30))
                    self.assertEqual(salida.get(timeout=5), "ok")
                    return original(*args)
                try:
                    with patch.object(trayectoria, "calcular_trayectoria", side_effect=calcular):
                        self.error("origen_modificado", clave=modo)
                finally:
                    self.terminar_procesos([proceso])
                    salida.close()
                    salida.join_thread()
                self.assertEqual([len(t) for t in self.tablas()], [0, 0])

    def test_sin_efectos_importacion_y_json(self):
        codigo = Path(registro.__file__).read_text(encoding="utf-8")
        aislado = ModuleType("registro_aislado")
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            exec(compile(codigo, "registro_aislado", "exec"), aislado.__dict__)
        r = self.individual()
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        for privado in (str(self.ruta), "PRIVADO", "configuracion_json", "ruta_modelo"):
            self.assertNotIn(privado, json.dumps(r))

    def test_error_programacion_no_se_oculta(self):
        with patch.object(trayectoria, "calcular_trayectoria", side_effect=RuntimeError("programacion")):
            with self.assertRaises(RuntimeError):
                self.individual()
        self.assertEqual([len(t) for t in self.tablas()], [0, 0])


if __name__ == "__main__":
    unittest.main()
