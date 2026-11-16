"""S4-A2: consultas a SQLite temporales sintéticas, sin datos originales."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import simulacion_desde_pronostico as servicio
from tests import test_consulta_evaluaciones as fixtures


class SimulacionDesdePronosticoTests(unittest.TestCase):
    def setUp(self):
        # Composición: reutilizar datos sintéticos no hereda las pruebas A2.
        self.datos = fixtures.ConsultaEvaluacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta

    def simular(self, identificador=101, inventario=30, plazo=1, reserva=5):
        antes = self.ruta.read_bytes()
        r = servicio.simular_pronostico(self.ruta, identificador, inventario, plazo, reserva)
        self.assertEqual(self.ruta.read_bytes(), antes)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        for secreto in ("PRIVADO", str(self.ruta), "configuracion_json", "ruta_modelo"):
            self.assertNotIn(secreto, json.dumps(r))
        return r

    def error(self, codigo, identificador=101, ruta=None, inventario=30, plazo=1, reserva=5):
        with self.assertRaises(servicio.ErrorSimulacionDesdePronostico) as e:
            servicio.simular_pronostico(self.ruta if ruta is None else ruta, identificador, inventario, plazo, reserva)
        self.assertEqual(e.exception.codigo, codigo)
        for secreto in ("PRIVADO", str(self.ruta), "SELECT", "Traceback"):
            self.assertNotIn(secreto, str(e.exception))

    @contextmanager
    def vigilar(self, transformar=None):
        original = sqlite3.connect
        conexiones, llamadas, denegadas = [], [], []

        class Filas:
            def __init__(self, filas): self.filas = filas
            def fetchall(self): return self.filas

        class Lectura(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                llamadas.append((sql, parametros, c.in_transaction))
                cursor = super().execute(sql, parametros)
                if transformar and "FROM detalle_pronostico" in sql:
                    return Filas(transformar([dict(f) for f in cursor.fetchall()]))
                return cursor

        def abrir(ruta_uri, **opciones):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertTrue(opciones["uri"])
            c = original(ruta_uri, factory=Lectura, **opciones)
            conexiones.append(c)

            def autorizar(accion, tabla, columna, *otros):
                permitidas = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION)
                if accion not in permitidas or (accion == sqlite3.SQLITE_READ and
                        (tabla not in ("conjunto_datos", "serie_temporal", "experimento", "pronostico", "detalle_pronostico")
                         or columna in ("configuracion_json", "ruta_modelo"))):
                    denegadas.append((accion, tabla, columna))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            c.set_authorizer(autorizar)
            return c

        with patch.object(sqlite3, "connect", side_effect=abrir):
            yield llamadas
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(denegadas, [])
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError): c.execute("SELECT 1")

    def test_resultado_y_procedencia_sin_duplicaciones(self):
        r = self.simular()
        self.assertEqual(r["conjunto_id"], 7)
        self.assertEqual(r["serie"], {"id": 83, "store_nbr": 17, "family": "FAMILIA A"})
        self.assertEqual((r["pronostico_id"], r["experimento_id"]), (101, 201))
        self.assertEqual((r["demanda_considerada"], r["reposicion"]), (310, 285))
        self.assertNotIn("simulacion", r)

    def test_coincide_integramente_con_motor(self):
        # El GET de referencia se ejecuta fuera de la llamada vigilada.
        datos = consulta.consultar_pronostico(self.ruta, 101)
        esperado = motor.calcular_simulacion(datos["pronostico"], datos["detalles"], 30, 1, 5)
        actual = self.simular()
        self.assertEqual({k: actual[k] for k in esperado}, esperado)
        self.assertEqual(set(actual)-set(esperado), {"conjunto_id", "serie"})

    def test_tres_cortes_y_tres_tecnicas(self):
        for i, (corte, tecnica) in enumerate(zip(fixtures.CORTES, motor.TECNICAS), 2):
            self.datos.crear_pronostico(100+i, 200+i, corte=corte, tecnica=tecnica)
            r = self.simular(100+i)
            self.assertEqual((r["fecha_corte"], r["tecnica"]), (corte, tecnica))

    def test_parametros_distintos_inventario_suficiente_y_reservas(self):
        self.assertEqual(self.simular(inventario=400, reserva=0)["reposicion"], 0)
        self.assertEqual(self.simular(inventario=0, reserva=0)["reposicion"], 310)
        self.assertEqual(self.simular(inventario=0, reserva=5)["reposicion"], 315)

    def test_plazos_cero_catorce_quince(self):
        for plazo, fecha, dentro in ((0,"2017-06-17",True),(14,"2017-07-01",True),(15,"2017-07-02",False)):
            r = self.simular(plazo=plazo)
            self.assertEqual((r["fecha_recepcion"],r["recepcion_dentro_horizonte"],r["reposicion"]), (fecha,dentro,285))

    def test_fracciones(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=0.25")
        r = self.simular(inventario=1.125, reserva=0.5)
        self.assertEqual((r["demanda_considerada"],r["reposicion"]),(3.75,3.125))

    def test_id_invalido_antes_de_abrir(self):
        with patch.object(consulta,"_conexion_lectura",side_effect=AssertionError("No abrir")):
            for i in (True,False,0,-1,1.0,"101",None,2**63,[]): self.error("pronostico_invalido",i)

    def test_inexistente_y_limite_sqlite(self):
        for i in (999,2**63-1): self.error("pronostico_inexistente",i)

    def test_pronostico_inactivo(self):
        self.datos.crear_pronostico(102,202,serie=24)
        self.error("pronostico_inactivo",102)

    def test_sin_conjunto_activo(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_tipos_no_elegibles_antes_de_detalles(self):
        for i,tipo in enumerate(("prueba","futuro"),102):
            self.datos.crear_pronostico(i,i+100,tipo=tipo)
            with patch.object(consulta,"_detalles_validos",side_effect=AssertionError("No leer detalles")):
                self.error("tipo_no_evaluable",i)

    def test_cortes_no_autorizados_antes_de_detalles(self):
        # Metadatos reservados sintéticos; no se crean ventas ni se lee su horizonte.
        self.datos.ejecutar("UPDATE pronostico SET fecha_corte='2017-07-31'")
        with patch.object(consulta,"_detalles_validos",side_effect=AssertionError("No leer detalles")):
            self.error("corte_no_autorizado")

    def test_detalles_incompletos(self):
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE paso=8")
        self.error("pronostico_corrupto")

    def test_detalles_duplicados_vigilados(self):
        def duplicar(filas): filas[7]=deepcopy(filas[6]); return filas
        with self.vigilar(duplicar): self.error("pronostico_corrupto")

    def test_fechas_incorrectas(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET fecha_objetivo='2017-06-16' WHERE paso=1")
        self.error("pronostico_corrupto")

    def test_estimaciones_corruptas(self):
        for v in (-1,float("inf"),"texto"):
            self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=? WHERE paso=1",(v,))
            self.error("pronostico_corrupto")

    def test_nan_en_detalle_rechazado(self):
        def alterar(f): f[0]["venta_estimada"]=float("nan"); return f
        with self.vigilar(alterar): self.error("pronostico_corrupto")

    def test_metadatos_corruptos(self):
        for campo,v in (("fecha_generacion","no-fecha"),("horizonte",14)):
            self.datos.corromper(f"UPDATE pronostico SET {campo}=?",(v,))
            self.error("pronostico_corrupto")
        self.datos.corromper("UPDATE pronostico SET horizonte=15,fecha_generacion='2026-01-01'")
        self.datos.ejecutar("UPDATE experimento SET tecnica='desconocida'")
        self.error("pronostico_corrupto")

    def test_referencias_rotas(self):
        with sqlite3.connect(self.ruta) as c: c.execute("UPDATE pronostico SET experimento_id=999")
        self.error("pronostico_corrupto")

    def test_aislamiento_series(self):
        self.datos.crear_pronostico(102,202,serie=84,estimadas=[1]*15)
        r=self.simular(102,inventario=0,reserva=0)
        self.assertEqual((r["serie"]["id"],r["demanda_considerada"]),(84,15))
        self.assertEqual(self.simular()["demanda_considerada"],310)

    def test_base_ausente_no_se_crea(self):
        ruta=self.ruta.parent/"ausente.sqlite3"
        self.error("base_no_disponible",ruta=ruta)
        self.assertFalse(ruta.exists())

    def test_base_danada_y_sin_esquema(self):
        for nombre,contenido in (("danada",b"no sqlite"),("vacia",b"")):
            ruta=self.ruta.parent/nombre; ruta.write_bytes(contenido)
            self.error("base_no_disponible",ruta=ruta)
            self.assertEqual(ruta.read_bytes(),contenido)

    def test_ruta_invalida(self):
        for ruta in ("",123,"\0"):
            self.error("base_no_disponible",ruta=ruta)

    def test_una_conexion_ro_begin_sin_dml_ventas_evaluacion_ni_configuracion(self):
        with self.vigilar() as llamadas:
            self.simular()
        self.assertEqual([s for s,_,_ in llamadas if s=="BEGIN"],["BEGIN"])
        self.assertTrue(all(t for s,_,t in llamadas if s!="BEGIN"))
        self.assertTrue(any("WHERE p.id=?" in s and p==(101,) for s,p,_ in llamadas))

    def test_motor_unico_y_sin_servicios_publicos(self):
        with patch.object(motor,"calcular_simulacion",wraps=motor.calcular_simulacion) as calcular, \
                patch.object(consulta,"consultar_pronostico",side_effect=AssertionError("Otra conexión")), \
                patch.object(consulta,"listar_pronosticos",side_effect=AssertionError("Otra conexión")):
            self.simular(); calcular.assert_called_once()
            self.assertEqual(calcular.call_args.args[0]["id"],101)
            self.assertEqual(len(calcular.call_args.args[1]),15)

    def test_parametros_invalidos_delegados_al_motor(self):
        for kwargs in ({"inventario":-1},{"inventario":True},{"reserva":"5"},{"plazo":1.5}):
            self.error("parametros_invalidos",**kwargs)

    def test_precision_entrada_y_total_controladas(self):
        self.error("precision_no_representable",reserva=2**53+1)
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=1e16 WHERE paso=1")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=1 WHERE paso=2")
        self.error("precision_no_representable",inventario=1e16,reserva=0)

    def test_desbordamiento_controlado(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=1e308")
        self.error("resultado_no_finito")

    def test_error_interno_no_se_filtra_en_error_controlado(self):
        with patch.object(consulta,"_metadatos",side_effect=consulta.ErrorConsultaPronostico("pronostico_corrupto","PRIVADO SELECT")):
            self.error("pronostico_corrupto")
        with patch.object(motor,"calcular_simulacion",side_effect=RuntimeError("programación")):
            with self.assertRaises(RuntimeError): self.simular()

    def test_ventas_y_evaluaciones_ausentes_o_corruptas_no_intervienen(self):
        esperado=self.simular()
        self.datos.ejecutar("DELETE FROM venta_diaria")
        self.datos.ejecutar("INSERT INTO evaluacion VALUES (101,999,999,999)")
        self.assertEqual(self.simular(),esperado)

    def wal(self, modo):
        self.datos.ejecutar("PRAGMA journal_mode=WAL")
        original=consulta._activo
        def activo(c):
            resultado=original(c)
            codigo="""
import sqlite3,sys
with sqlite3.connect(sys.argv[1]) as c:
    if sys.argv[2]=='detalle': c.execute('UPDATE detalle_pronostico SET venta_estimada=1')
    else: c.execute('UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END')
"""
            p=subprocess.run([sys.executable,"-B","-c",codigo,str(self.ruta),modo],capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stderr)
            return resultado
        with patch.object(consulta,"_activo",side_effect=activo):
            r=servicio.simular_pronostico(self.ruta,101,0,0,0)
            self.assertEqual((r["conjunto_id"],r["demanda_considerada"]),(7,310))
        if modo=="detalle": self.assertEqual(self.simular()["demanda_considerada"],15)
        else: self.error("pronostico_inactivo")

    def test_wal_proceso_independiente_cambia_detalles(self): self.wal("detalle")

    def test_wal_proceso_independiente_cambia_activo(self): self.wal("activo")

    def test_repetibilidad(self): self.assertEqual(self.simular(),self.simular())



    def test_sin_entrenamiento_generacion_registro_ni_metricas(self):
        from backend.app.services import registro_pronosticos, registro_evaluaciones, consulta_evaluaciones, metricas_evaluacion
        with patch.object(registro_pronosticos, "registrar_pronostico", side_effect=AssertionError("No generar")), \
                patch.object(registro_evaluaciones, "registrar_evaluacion", side_effect=AssertionError("No registrar")), \
                patch.object(consulta_evaluaciones, "_evaluar_en_conexion", side_effect=AssertionError("No evaluar")), \
                patch.object(metricas_evaluacion, "calcular_metricas", side_effect=AssertionError("No calcular métricas")):
            with self.vigilar(): self.simular()


if __name__ == "__main__":
    unittest.main()
