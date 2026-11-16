"""S4-A3-03: comparación en SQLite temporales y datos sintéticos exclusivamente."""

from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch

from backend.app.services import comparacion_desde_pronostico as servicio
from backend.app.services import comparacion_simulaciones as puro
from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import simulacion_desde_pronostico as lectura
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import trayectoria_simulacion as trayectoria
from tests import test_simulacion_desde_pronostico as fixtures
from tests.test_comparacion_simulaciones import escenarios


class ComparacionDesdePronosticoTests(unittest.TestCase):
    def setUp(self):
        # Composición de la preparación y vigilancia existentes; sin herencia
        # ni descubrimiento accidental de sus 34 pruebas publicadas.
        self.fixture=fixtures.SimulacionDesdePronosticoTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.datos=self.fixture.datos
        self.ruta=self.fixture.ruta

    def comparar(self, seleccion=None, identificador=101):
        antes=self.ruta.read_bytes()
        r=servicio.comparar_escenarios_pronostico(self.ruta,identificador,escenarios() if seleccion is None else seleccion)
        self.assertEqual(self.ruta.read_bytes(),antes)
        self.assertEqual(json.loads(json.dumps(r,allow_nan=False)),r)
        for secreto in ("PRIVADO",str(self.ruta),"configuracion_json","ruta_modelo"):
            self.assertNotIn(secreto,json.dumps(r))
        self.assertEqual(set(r),{"contexto","resultados","advertencias_comparacion","conjunto_id","serie"})
        return r

    def error(self, codigo, seleccion=None, identificador=101, ruta=None):
        with self.assertRaises(servicio.ErrorComparacionDesdePronostico) as e:
            servicio.comparar_escenarios_pronostico(self.ruta if ruta is None else ruta,identificador,escenarios() if seleccion is None else seleccion)
        self.assertEqual(e.exception.codigo,codigo)
        for secreto in ("PRIVADO",str(self.ruta),"SELECT","Traceback"):
            self.assertNotIn(secreto,str(e.exception))
        self.assertFalse(hasattr(e.exception,"resultados"))

    def test_ab_procedencia_y_resultado_manual(self):
        r=self.comparar()
        self.assertEqual(r["conjunto_id"],7)
        self.assertEqual(r["serie"],{"id":83,"store_nbr":17,"family":"FAMILIA A"})
        self.assertEqual((r["contexto"]["pronostico_id"],r["contexto"]["experimento_id"]),(101,201))
        self.assertEqual([e["etiqueta"] for e in r["resultados"]],["A","B"])
        self.assertEqual([e["simulacion"]["reposicion"] for e in r["resultados"]],[310,310])
        self.assertEqual(r["resultados"][1]["simulacion"]["resumen"]["mayor_demanda_pendiente_actual"],32)

    def test_abc_plazos_cero_catorce_quince(self):
        s=escenarios(True);s[1]["tiempo_abastecimiento"]=14
        r=self.comparar(s)
        self.assertEqual([e["etiqueta"] for e in r["resultados"]],["A","B","C"])
        self.assertEqual([e["simulacion"]["fecha_recepcion"] for e in r["resultados"]],["2017-06-17","2017-07-01","2017-07-02"])
        c=r["resultados"][2]["simulacion"]
        self.assertFalse(c["recepcion_dentro_horizonte"])
        self.assertTrue(all(d["recepcion_simulada"]==0 for d in c["trayectoria"]))
        self.assertEqual(c["resumen"]["balance_neto_final"],-310)

    def test_tres_tecnicas_y_tres_cortes(self):
        for i,(tecnica,corte) in enumerate(zip(motor.TECNICAS,lectura.CORTES_AUTORIZADOS),102):
            self.datos.crear_pronostico(i,i+100,tecnica=tecnica,corte=corte)
            r=self.comparar(identificador=i)
            self.assertEqual((r["contexto"]["tecnica"],r["contexto"]["fecha_corte"]),(tecnica,corte))

    def test_parametros_distintos_inventario_y_reserva(self):
        s=escenarios(True);s[0]["inventario_disponible"]=400;s[1]["reserva"]=5
        r=self.comparar(s)
        self.assertEqual([e["simulacion"]["reposicion"] for e in r["resultados"]],[0,315,310])

    def test_igual_q_por_plazo_pendiente_aparece_y_desaparece(self):
        r=self.comparar()
        a,b=[e["simulacion"] for e in r["resultados"]]
        self.assertEqual(a["reposicion"],b["reposicion"])
        self.assertNotEqual(a["trayectoria"],b["trayectoria"])
        self.assertEqual([f["demanda_pendiente_actual"] for f in b["trayectoria"][:3]],[8,32,0])

    def test_resultado_integro_equivalente_al_comparador_puro(self):
        referencia=consulta.consultar_pronostico(self.ruta,101)
        s=escenarios(True)
        esperado=puro.comparar_escenarios(referencia["pronostico"],referencia["detalles"],s)
        r=self.comparar(s)
        self.assertEqual({k:r[k] for k in esperado},esperado)

    def test_una_conexion_begin_cabecera_detalles_y_delegacion(self):
        helper=lectura._leer_pronostico_simulable_en_conexion
        real=puro.comparar_escenarios
        with self.fixture.vigilar() as llamadas, patch.object(lectura,"_leer_pronostico_simulable_en_conexion",wraps=helper) as leer:
            def comparar(p,d,s):
                self.assertTrue(leer.call_args.args[0].in_transaction)
                self.assertEqual(len(d),15)
                return real(p,d,s)
            with patch.object(puro,"comparar_escenarios",side_effect=comparar) as calcular, \
                    patch.object(lectura,"simular_pronostico",side_effect=AssertionError("Otra conexión")):
                self.comparar(escenarios(True));calcular.assert_called_once();leer.assert_called_once()
        self.assertEqual([sql for sql,_,_ in llamadas if sql=="BEGIN"],["BEGIN"])
        self.assertEqual(sum("FROM detalle_pronostico" in sql for sql,_,_ in llamadas),1)
        self.assertEqual(sum("FROM pronostico p" in sql for sql,_,_ in llamadas),1)
        self.assertTrue(all(t for sql,_,t in llamadas if sql!="BEGIN"))

    def test_mismos_objetos_y_detalles_en_tres_escenarios(self):
        real=trayectoria.calcular_trayectoria
        with patch.object(trayectoria,"calcular_trayectoria",wraps=real) as calcular:
            self.comparar(escenarios(True))
        self.assertEqual(calcular.call_count,3)
        p,d=calcular.call_args_list[0].args[:2]
        for llamada in calcular.call_args_list:
            self.assertIs(llamada.args[0],p);self.assertIs(llamada.args[1],d)
        self.assertEqual(len(d),15)

    def test_id_invalido_antes_de_conexion(self):
        with patch.object(consulta,"_conexion_lectura",side_effect=AssertionError("No abrir")):
            for i in (True,0,-1,1.0,"101",None,2**63,[]):self.error("pronostico_invalido",identificador=i)

    def test_inexistente(self):self.error("pronostico_inexistente",identificador=999)

    def test_inactivo(self):
        self.datos.crear_pronostico(102,202,serie=24)
        self.error("pronostico_inactivo",identificador=102)

    def test_sin_activo(self):
        self.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_tipos_prueba_futuro_antes_de_detalles(self):
        for i,tipo in enumerate(("prueba","futuro"),102):
            self.datos.crear_pronostico(i,i+100,tipo=tipo)
            with patch.object(consulta,"_detalles_validos",side_effect=AssertionError("No leer")):
                self.error("tipo_no_evaluable",identificador=i)

    def test_corte_no_autorizado_antes_de_detalles(self):
        self.datos.ejecutar("UPDATE pronostico SET fecha_corte='2017-07-31'")
        with patch.object(consulta,"_detalles_validos",side_effect=AssertionError("No leer holdout")):
            self.error("corte_no_autorizado")

    def test_detalles_incompletos(self):
        self.datos.ejecutar("DELETE FROM detalle_pronostico WHERE paso=9")
        self.error("pronostico_corrupto")

    def test_duplicados_y_nan_vigilados(self):
        for modo in ("duplicado","nan"):
            def alterar(f):
                if modo=="duplicado":f[7]=deepcopy(f[6])
                else:f[0]["venta_estimada"]=float("nan")
                return f
            with self.fixture.vigilar(alterar):self.error("pronostico_corrupto")

    def test_fecha_inconsistente(self):
        self.datos.ejecutar("UPDATE detalle_pronostico SET fecha_objetivo='2017-06-16' WHERE paso=1")
        self.error("pronostico_corrupto")

    def test_estimaciones_corruptas(self):
        for v in (-1,"texto",float("inf")):
            self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=? WHERE paso=1",(v,))
            self.error("pronostico_corrupto")

    def test_referencias_rotas(self):
        with closing(sqlite3.connect(self.ruta)) as c,c:c.execute("UPDATE experimento SET serie_id=999")
        self.error("pronostico_corrupto")

    def test_metadatos_inconsistentes(self):
        self.datos.ejecutar("UPDATE experimento SET tecnica='desconocida'")
        self.error("pronostico_corrupto")

    def test_escenarios_invalidos_validacion_delegada(self):
        for s in ([],escenarios()[::-1],escenarios()+[{"etiqueta":"C"}]):
            self.error("escenarios_invalidos",s)

    def test_parametros_invalidos(self):
        for campo,valor in (("reserva",-1),("inventario_disponible",True),("tiempo_abastecimiento",1.0)):
            s=escenarios();s[1][campo]=valor;self.error("parametros_invalidos",s)

    def test_precision_y_desbordamiento(self):
        s=escenarios();s[1]["reserva"]=2**53+1
        self.error("precision_no_representable",s)
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=1e308")
        self.error("resultado_no_finito")

    def test_rango_trayectoria(self):
        from sys import float_info
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=0")
        self.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=? WHERE paso=2",(float_info.max,))
        s=escenarios();s[1].update(inventario_disponible=float_info.max,reserva=float_info.max,tiempo_abastecimiento=0)
        self.error("resultado_no_finito",s)

    def test_base_ausente_no_se_crea(self):
        ruta=self.ruta.parent/"ausente.sqlite3";self.error("base_no_disponible",ruta=ruta)
        self.assertFalse(ruta.exists())

    def test_base_danada_o_sin_esquema(self):
        for nombre,contenido in (("danada",b"no sqlite"),("vacia",b"")):
            ruta=self.ruta.parent/nombre;ruta.write_bytes(contenido)
            self.error("base_no_disponible",ruta=ruta);self.assertEqual(ruta.read_bytes(),contenido)

    def test_sin_dml_ventas_evaluaciones_configuracion_o_generacion(self):
        from backend.app.services import registro_pronosticos,registro_evaluaciones,metricas_evaluacion
        with self.fixture.vigilar(), \
                patch.object(registro_pronosticos,"registrar_pronostico",side_effect=AssertionError("No generar")), \
                patch.object(registro_evaluaciones,"registrar_evaluacion",side_effect=AssertionError("No guardar")), \
                patch.object(metricas_evaluacion,"calcular_metricas",side_effect=AssertionError("No evaluar")):
            self.comparar()

    def test_repetibilidad_y_entradas_intactas(self):
        s=escenarios(True);antes=deepcopy(s)
        self.assertEqual(self.comparar(s),self.comparar(s));self.assertEqual(s,antes)

    def test_error_publico_sanitizado_y_sin_parciales(self):
        s=escenarios(True);s[2]["reserva"]=-1;antes=deepcopy(s)
        self.error("parametros_invalidos",s);self.assertEqual(s,antes)
        for error,codigo in ((lectura.ErrorSimulacionDesdePronostico("pronostico_corrupto"),"pronostico_corrupto"),
                             (consulta.ErrorConsultaPronostico("base_no_disponible","PRIVADO SELECT"),"base_no_disponible")):
            with patch.object(lectura,"_leer_pronostico_simulable_en_conexion",side_effect=error):self.error(codigo)

    def test_resultados_incompatibles_y_programacion_no_oculta(self):
        with patch.object(puro,"comparar_escenarios",side_effect=puro.ErrorComparacionSimulaciones("resultados_incompatibles")):
            self.error("resultados_incompatibles")
        with patch.object(puro,"comparar_escenarios",side_effect=RuntimeError("programación")):
            with self.assertRaises(RuntimeError):self.comparar()

    def test_aislamiento_series_y_conjuntos(self):
        self.datos.crear_pronostico(102,202,serie=84,estimadas=[1]*15)
        r=self.comparar(identificador=102)
        self.assertEqual(r["serie"]["id"],84)
        self.assertEqual(r["resultados"][0]["simulacion"]["demanda_considerada"],15)
        self.assertEqual(self.comparar()["serie"]["id"],83)
        self.datos.crear_pronostico(103,203,serie=24)
        self.error("pronostico_inactivo",identificador=103)

    def wal(self,modo):
        self.datos.ejecutar("PRAGMA journal_mode=WAL")
        esperado=self.comparar(escenarios(True))
        original=consulta._activo
        def activo(c):
            r=original(c)
            codigo="""
import sqlite3,sys
from contextlib import closing
with closing(sqlite3.connect(sys.argv[1])) as c,c:
    if sys.argv[2]=='activo':c.execute('UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END')
    elif sys.argv[2]=='detalles':c.execute('UPDATE detalle_pronostico SET venta_estimada=1')
    else:
        c.execute("UPDATE experimento SET tecnica='random_forest'")
        c.execute("UPDATE serie_temporal SET family='CAMBIADA' WHERE id=83")
"""
            p=subprocess.run([sys.executable,"-B","-c",codigo,str(self.ruta),modo],capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stderr)
            return r
        with patch.object(consulta,"_activo",side_effect=activo):
            self.assertEqual(servicio.comparar_escenarios_pronostico(self.ruta,101,escenarios(True)),esperado)
        if modo=="activo":self.error("pronostico_inactivo")
        elif modo=="detalles":self.assertEqual(self.comparar()["resultados"][0]["simulacion"]["demanda_considerada"],15)
        else:
            nuevo=self.comparar();self.assertEqual(nuevo["serie"]["family"],"CAMBIADA")
            self.assertEqual(nuevo["contexto"]["tecnica"],"random_forest")

    def test_wal_activo_proceso_independiente(self):self.wal("activo")

    def test_wal_detalles_proceso_independiente(self):self.wal("detalles")

    def test_wal_metadatos_proceso_independiente(self):self.wal("metadatos")


if __name__ == "__main__":
    unittest.main()
