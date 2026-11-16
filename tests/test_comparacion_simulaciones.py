"""S4-A3-02: comparación pura con datos sintéticos y sin conexiones externas."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date
from fractions import Fraction
import builtins
import io
import json
from math import nextafter
import os
from pathlib import Path
import random
import socket
import sqlite3
import subprocess
from sys import float_info
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import comparacion_simulaciones as servicio
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import trayectoria_simulacion as trayectoria
from tests.test_simulacion_reposicion import datos


def escenarios(tres=False):
    return [{"etiqueta":e,"inventario_disponible":0,"tiempo_abastecimiento":l,"reserva":0}
            for e,l in zip(("A","B","C"),(0,2,15))][:3 if tres else 2]


class ComparacionSimulacionesTests(unittest.TestCase):
    def comparar(self, seleccion=None, valores=None):
        p,d=datos(valores)
        r=servicio.comparar_escenarios(p,d,escenarios() if seleccion is None else seleccion)
        self.assertEqual(json.loads(json.dumps(r,allow_nan=False)),r)
        self.assertTrue({"ganador","ranking","mejor_escenario"}.isdisjoint(r))
        return r

    def rechazar_estructura(self, seleccion):
        p,d=datos()
        with patch.object(trayectoria,"calcular_trayectoria",side_effect=AssertionError("No calcular")) as calcular:
            with self.assertRaises(servicio.ErrorComparacionSimulaciones) as e:
                servicio.comparar_escenarios(p,d,seleccion)
            self.assertEqual(e.exception.codigo,"escenarios_invalidos")
            calcular.assert_not_called()

    def test_dos_escenarios_orden_y_contexto(self):
        r=self.comparar()
        self.assertEqual([x["etiqueta"] for x in r["resultados"]],["A","B"])
        self.assertEqual(r["contexto"]["pronostico_id"],101)
        self.assertEqual(r["contexto"]["experimento_id"],201)
        self.assertEqual(r["contexto"]["version_regla"],motor.VERSION_REGLA)
        self.assertEqual(len(r["contexto"]["fechas_objetivo"]),15)

    def test_tres_escenarios_manuales(self):
        r=self.comparar(escenarios(True))
        a,b,c=[x["simulacion"] for x in r["resultados"]]
        self.assertEqual([x["etiqueta"] for x in r["resultados"]],["A","B","C"])
        self.assertEqual([s["reposicion"] for s in (a,b,c)],[150]*3)
        self.assertEqual([s["resumen"]["balance_neto_final"] for s in (a,b,c)],[0,0,-150])
        self.assertEqual([s["resumen"]["mayor_demanda_pendiente_actual"] for s in (a,b,c)],[0,20,150])
        self.assertEqual([s["resumen"]["dias_con_balance_negativo"] for s in (a,b,c)],[0,2,15])

    def test_tupla_aceptada(self): self.assertEqual(len(self.comparar(tuple(escenarios()))["resultados"]),2)

    def test_cantidades_incorrectas(self):
        for n in (0,1,4):self.rechazar_estructura(escenarios(True)[:n] if n<4 else escenarios(True)+[escenarios()[0]])

    def test_contenedor_invalido(self):
        for v in (None,{},"AB",iter(escenarios())):self.rechazar_estructura(v)

    def test_etiquetas_desordenadas_duplicadas_desconocidas(self):
        for labels in (("B","A"),("A","A"),("A","C"),("a","B"),(1,"B"),(True,"B"),("A","B","B")):
            seleccion=escenarios(len(labels)==3)
            for s,e in zip(seleccion,labels):s["etiqueta"]=e
            self.rechazar_estructura(seleccion)

    def test_campos_ausentes_y_adicionales(self):
        for campo in escenarios()[0]:
            seleccion=escenarios();del seleccion[-1][campo];self.rechazar_estructura(seleccion)
        seleccion=escenarios();seleccion[-1]["extra"]=1;self.rechazar_estructura(seleccion)

    def test_objeto_escenario_invalido(self):
        for valor in (None,[],"B",1):
            seleccion=escenarios();seleccion[-1]=valor;self.rechazar_estructura(seleccion)

    def test_estructura_del_tercero_validada_antes_de_cualquier_motor(self):
        seleccion=escenarios(True);del seleccion[2]["reserva"];self.rechazar_estructura(seleccion)

    def test_parametros_identicos_permitidos(self):
        seleccion=escenarios();seleccion[1]={**seleccion[0],"etiqueta":"B"}
        a,b=self.comparar(seleccion)["resultados"]
        self.assertEqual(a["simulacion"],b["simulacion"])

    def test_igual_q_distinto_plazo_y_trayectoria(self):
        a,b=[x["simulacion"] for x in self.comparar()["resultados"]]
        self.assertEqual(a["reposicion"],b["reposicion"])
        self.assertNotEqual(a["fecha_recepcion"],b["fecha_recepcion"])
        self.assertNotEqual(a["trayectoria"],b["trayectoria"])
        self.assertEqual([f["demanda_pendiente_actual"] for f in b["trayectoria"][:3]],[10,20,0])

    def test_l15_sin_recepciones(self):
        c=self.comparar(escenarios(True))["resultados"][2]["simulacion"]
        self.assertFalse(c["recepcion_dentro_horizonte"])
        self.assertTrue(all(f["recepcion_simulada"]==0 for f in c["trayectoria"]))

    def test_inventario_suficiente_y_reservas_distintas(self):
        seleccion=escenarios(True)
        seleccion[0]["inventario_disponible"]=200
        seleccion[1]["reserva"]=5
        r=self.comparar(seleccion)
        self.assertEqual([x["simulacion"]["reposicion"] for x in r["resultados"]],[0,155,150])

    def test_demanda_cero(self):
        r=self.comparar(valores=[0]*15)
        self.assertTrue(all(x["simulacion"]["reposicion"]==0 for x in r["resultados"]))

    def test_fracciones(self):
        seleccion=escenarios();seleccion[0].update(inventario_disponible=1.125,reserva=.5)
        r=self.comparar(seleccion,[.25]*15)
        self.assertEqual(r["resultados"][0]["simulacion"]["reposicion"],3.125)

    def test_resultados_integros_iguales_a_trayectoria(self):
        p,d=datos();seleccion=escenarios(True)
        esperado=[trayectoria.calcular_trayectoria(p,d,s["inventario_disponible"],s["tiempo_abastecimiento"],s["reserva"]) for s in seleccion]
        r=servicio.comparar_escenarios(p,d,seleccion)
        self.assertEqual([x["simulacion"] for x in r["resultados"]],esperado)

    def test_una_invocacion_por_escenario_mismos_objetos(self):
        p,d=datos();seleccion=escenarios(True)
        with patch.object(trayectoria,"calcular_trayectoria",wraps=trayectoria.calcular_trayectoria) as calcular:
            servicio.comparar_escenarios(p,d,seleccion)
        self.assertEqual(calcular.call_count,3)
        for llamada,s in zip(calcular.call_args_list,seleccion):
            self.assertIs(llamada.args[0],p);self.assertIs(llamada.args[1],d)
            self.assertEqual(llamada.args[2:],(s["inventario_disponible"],s["tiempo_abastecimiento"],s["reserva"]))

    def test_fechas_date_y_experimento_opcional(self):
        p,d=datos();del p["experimento_id"];p["fecha_corte"]=date.fromisoformat(p["fecha_corte"])
        for f in d:f["fecha_objetivo"]=date.fromisoformat(f["fecha_objetivo"])
        r=servicio.comparar_escenarios(p,tuple(d),escenarios())
        self.assertIsNone(r["contexto"]["experimento_id"])
        self.assertEqual(r["contexto"]["fechas_objetivo"][0],"2017-06-17")

    def test_parametros_invalidos_sin_conversion(self):
        for k,v in (("inventario_disponible","1"),("reserva",True),("tiempo_abastecimiento",1.0),("reserva",float("nan"))):
            seleccion=escenarios();seleccion[0][k]=v
            with self.assertRaises(motor.ErrorSimulacionReposicion):self.comparar(seleccion)

    def test_fallo_en_segundo_no_devuelve_parciales_ni_ejecuta_tercero(self):
        p,d=datos();seleccion=escenarios(True);seleccion[1]["reserva"]=-1
        with patch.object(trayectoria,"calcular_trayectoria",wraps=trayectoria.calcular_trayectoria) as calcular:
            with self.assertRaises(motor.ErrorSimulacionReposicion) as e:servicio.comparar_escenarios(p,d,seleccion)
        self.assertEqual(calcular.call_count,2)
        self.assertFalse(hasattr(e.exception,"resultados"))

    def test_fallo_en_tercero_rechaza_comparacion(self):
        seleccion=escenarios(True);seleccion[2]["reserva"]=2**53+1
        with self.assertRaises(motor.ErrorSimulacionReposicion) as e:self.comparar(seleccion)
        self.assertEqual(e.exception.codigo,"precision_no_representable")

    def test_rango_y_desbordamiento(self):
        with self.assertRaises(motor.ErrorSimulacionReposicion):self.comparar(valores=[1e308]*15)
        seleccion=escenarios();seleccion[0].update(inventario_disponible=float_info.max,reserva=float_info.max)
        with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion):self.comparar(seleccion,[0,float_info.max]+[0]*13)

    def test_subnormal_y_residuos_no_se_sustituyen_por_cero(self):
        r=self.comparar(escenarios(True),[nextafter(0.,1.)]+[0]*14)
        self.assertGreater(r["resultados"][2]["simulacion"]["resumen"]["demanda_pendiente_final"],0)
        r=self.comparar(valores=[.1]*15)
        self.assertGreater(r["resultados"][0]["simulacion"]["resumen"]["demanda_pendiente_final"],0)

    def test_contradicciones_entre_resultados(self):
        p,d=datos();seleccion=escenarios();b=seleccion[1]
        original=trayectoria.calcular_trayectoria(p,d,b["inventario_disponible"],b["tiempo_abastecimiento"],b["reserva"])
        cambios=[lambda r:r.update(pronostico_id=999),lambda r:r.update(experimento_id=999),lambda r:r.update(tecnica="otra"),
                 lambda r:r.update(fecha_corte="2017-07-01"),lambda r:r.update(horizonte=14),lambda r:r.update(version_regla="otra"),
                 lambda r:r.update(demanda_considerada=151),lambda r:r["trayectoria"][0].update(fecha_objetivo="2017-06-18"),
                 lambda r:r["trayectoria"][0].update(demanda_pronosticada=11),lambda r:r["trayectoria"].pop(),
                 lambda r:r["parametros"].update(reserva=99),lambda r:r.update(pronostico_id=101.0)]
        real=trayectoria.calcular_trayectoria
        for cambiar in cambios:
            modificado=deepcopy(original);cambiar(modificado);n=[0]
            def calcular(*args):
                n[0]+=1;return real(*args) if n[0]==1 else modificado
            with patch.object(trayectoria,"calcular_trayectoria",side_effect=calcular):
                with self.assertRaises(servicio.ErrorComparacionSimulaciones) as e:servicio.comparar_escenarios(p,d,seleccion)
            self.assertEqual(e.exception.codigo,"resultados_incompatibles")

    def test_primer_resultado_debe_corresponder_a_entrada(self):
        p,d=datos();r=trayectoria.calcular_trayectoria(p,d,0,0,0);r["pronostico_id"]=999
        with patch.object(trayectoria,"calcular_trayectoria",return_value=r):
            with self.assertRaises(servicio.ErrorComparacionSimulaciones):servicio.comparar_escenarios(p,d,escenarios())

    def test_resultado_no_finito_ni_objetos_no_json(self):
        p,d=datos();r=trayectoria.calcular_trayectoria(p,d,0,0,0)
        for valor in (float("nan"),float("inf"),Fraction(1,3)):
            modificado=deepcopy(r);modificado["trayectoria"][0]["balance_neto"]=valor
            with patch.object(trayectoria,"calcular_trayectoria",return_value=modificado):
                with self.assertRaises(servicio.ErrorComparacionSimulaciones) as e:servicio.comparar_escenarios(p,d,escenarios())
            self.assertEqual(e.exception.codigo,"resultado_no_finito")

    def test_entradas_y_resultados_del_motor_no_modificados(self):
        p,d=datos();s=escenarios();antes=deepcopy((p,d,s))
        servicio.comparar_escenarios(p,d,s);self.assertEqual((p,d,s),antes)
        iguales=escenarios();iguales[1]={**iguales[0],"etiqueta":"B"}
        base=trayectoria.calcular_trayectoria(p,d,0,0,0);original=deepcopy(base)
        with patch.object(trayectoria,"calcular_trayectoria",return_value=base):r=servicio.comparar_escenarios(p,d,iguales)
        r["resultados"][0]["simulacion"]["parametros"]["reserva"]=99
        r["resultados"][0]["simulacion"]["trayectoria"][0]["balance_neto"]=999
        r["resultados"][0]["simulacion"]["supuestos"].clear()
        r["contexto"]["fechas_objetivo"].clear()
        self.assertEqual(base,original)
        self.assertEqual(r["resultados"][1]["simulacion"],original)

    def test_rechazo_conserva_entradas(self):
        p,d=datos();s=escenarios();s[1]["reserva"]=-1;antes=deepcopy((p,d,s))
        with self.assertRaises(motor.ErrorSimulacionReposicion):servicio.comparar_escenarios(p,d,s)
        self.assertEqual((p,d,s),antes)

    def test_repetibilidad(self):self.assertEqual(self.comparar(),self.comparar())

    def test_conservacion_sintetica_independiente(self):
        azar=random.Random(502)
        for _ in range(30):
            valores=[azar.randrange(20)/4 for _ in range(15)];s=escenarios(True)
            for e in s:e["inventario_disponible"]=azar.randrange(40)/4;e["reserva"]=azar.randrange(12)/4
            r=self.comparar(s,valores)
            for e in r["resultados"]:
                x=e["simulacion"];balance=Fraction.from_float(x["parametros"]["inventario_disponible"])
                for fila in x["trayectoria"]:balance+=Fraction.from_float(fila["recepcion_simulada"])-Fraction.from_float(fila["demanda_pronosticada"])
                self.assertEqual(x["resumen"]["balance_neto_final"],float(balance))

    def test_importacion_y_calculo_sin_efectos_secundarios(self):
        codigo=compile(Path(servicio.__file__).read_text(encoding="utf-8"),servicio.__file__,"exec")
        aislado=ModuleType("comparacion_aislada");p,d=datos()
        with ExitStack() as stack:
            dobles=[stack.enter_context(patch.object(m,n,side_effect=AssertionError("E/S prohibida")))
                    for m,n in ((builtins,"open"),(io,"open"),(os,"open"),(sqlite3,"connect"),(socket,"socket"),(subprocess,"Popen"),(builtins,"print"))]
            exec(codigo,aislado.__dict__)
            self.assertEqual(len(aislado.comparar_escenarios(p,d,escenarios())["resultados"]),2)
            for doble in dobles:doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
