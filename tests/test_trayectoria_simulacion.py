"""S4-A3-01: trayectorias sintéticas y referencias exactas independientes."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, timedelta
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

from backend.app.services import simulacion_reposicion as motor
from backend.app.services import trayectoria_simulacion as trayectoria
from tests.test_simulacion_reposicion import datos


class TrayectoriaSimulacionTests(unittest.TestCase):
    def calcular(self, valores=None, inventario=30, plazo=1, reserva=5):
        p, d = datos(valores)
        r = trayectoria.calcular_trayectoria(p, d, inventario, plazo, reserva)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        self.assertEqual(len(r["trayectoria"]), 15)
        self.assertEqual([f["paso"] for f in r["trayectoria"]], list(range(1, 16)))
        return r

    def test_demanda_constante_manual(self):
        r = self.calcular()
        self.assertEqual(r["reposicion"], 125)
        self.assertEqual(r["trayectoria"][0]["balance_neto"], 20)
        self.assertEqual(r["trayectoria"][1]["balance_neto"], 135)
        self.assertEqual(r["resumen"], {
            "fecha_primera_demanda_pendiente": None, "mayor_demanda_pendiente_actual": 0,
            "dias_con_balance_negativo": 0, "balance_neto_final": 5,
            "disponible_hipotetico_final": 5, "demanda_pendiente_final": 0,
            "recepcion_dentro_horizonte": True})

    def test_demanda_variable(self):
        r = self.calcular(list(range(1,16)), inventario=0, plazo=2, reserva=0)
        self.assertEqual([f["balance_neto"] for f in r["trayectoria"][:3]], [-1,-3,114])
        self.assertEqual(r["resumen"]["balance_neto_final"], 0)

    def test_demanda_cero(self):
        r = self.calcular([0]*15, inventario=0, reserva=0)
        self.assertTrue(all(f["balance_neto"] == 0 for f in r["trayectoria"]))
        self.assertIsNone(r["resumen"]["fecha_primera_demanda_pendiente"])

    def test_inventario_suficiente_sin_pedido(self):
        r = self.calcular(inventario=200,reserva=0)
        self.assertEqual(r["reposicion"],0)
        self.assertEqual(r["resumen"]["balance_neto_final"],50)
        self.assertTrue(all(f["recepcion_simulada"]==0 for f in r["trayectoria"]))

    def test_inventario_insuficiente_demanda_pendiente(self):
        r = self.calcular(inventario=0,plazo=2,reserva=0)
        self.assertEqual(r["resumen"]["fecha_primera_demanda_pendiente"],"2017-06-17")
        self.assertEqual(r["resumen"]["mayor_demanda_pendiente_actual"],20)
        self.assertEqual(r["resumen"]["dias_con_balance_negativo"],2)

    def test_recepcion_reduce_pendiente(self):
        r = self.calcular(inventario=0,plazo=2,reserva=0)
        self.assertEqual([f["demanda_pendiente_actual"] for f in r["trayectoria"][:3]],[10,20,0])
        self.assertEqual(r["resumen"]["mayor_demanda_pendiente_actual"],20)
        self.assertNotIn("suma_demanda_pendiente",r["resumen"])

    def test_balance_negativo_no_es_stock_negativo(self):
        r = self.calcular(inventario=0,plazo=15,reserva=0)
        for f in r["trayectoria"]:
            self.assertLess(f["balance_neto"],0)
            self.assertEqual(f["disponible_hipotetico"],0)
            self.assertEqual(f["demanda_pendiente_actual"],-f["balance_neto"])

    def test_reserva_positiva_y_cero(self):
        for reserva in (0,5):
            r = self.calcular(inventario=0,plazo=0,reserva=reserva)
            self.assertEqual(r["resumen"]["balance_neto_final"],reserva)

    def test_recepcion_plazo_cero(self):
        r = self.calcular(plazo=0)
        self.assertEqual([f["recepcion_simulada"] for f in r["trayectoria"]],[125]+[0]*14)

    def test_recepcion_plazo_catorce(self):
        r = self.calcular(plazo=14)
        self.assertEqual([f["recepcion_simulada"] for f in r["trayectoria"]],[0]*14+[125])
        self.assertEqual(r["resumen"]["balance_neto_final"],5)

    def test_recepcion_plazo_quince_fuera(self):
        r = self.calcular(plazo=15)
        self.assertTrue(all(f["recepcion_simulada"]==0 for f in r["trayectoria"]))
        self.assertFalse(r["resumen"]["recepcion_dentro_horizonte"])
        self.assertEqual(r["resumen"]["balance_neto_final"],-120)

    def test_plazos_distintos_igual_q(self):
        resultados=[self.calcular(plazo=l) for l in range(16)]
        self.assertEqual({r["reposicion"] for r in resultados},{125})
        for l,r in enumerate(resultados):
            self.assertEqual(sum(f["recepcion_simulada"]!=0 for f in r["trayectoria"]), int(l<15))

    def test_cantidades_fraccionarias(self):
        r = self.calcular([.25]*15,inventario=1.125,reserva=.5,plazo=0)
        self.assertEqual(r["resumen"]["balance_neto_final"],.5)
        self.assertEqual(r["trayectoria"][0]["balance_neto"],4)

    def test_q_publicada_redondeada_residuo_no_se_borra(self):
        r = self.calcular([.1]*15,inventario=0,reserva=0,plazo=0)
        exacto=Fraction.from_float(r["reposicion"])-15*Fraction.from_float(.1)
        self.assertLess(exacto,0)
        self.assertEqual(r["resumen"]["balance_neto_final"],float(exacto))
        self.assertGreater(r["resumen"]["demanda_pendiente_final"],0)

    def test_cancelacion_no_reconstruye_q_desde_total_redondeado(self):
        r = self.calcular([0,1e16,.25]+[0]*12,inventario=1e16,reserva=0,plazo=0)
        self.assertEqual(r["demanda_considerada"],1e16)
        self.assertEqual(r["reposicion"],.25)
        self.assertEqual(r["trayectoria"][1]["balance_neto"],.25)
        self.assertEqual(r["resumen"]["balance_neto_final"],0)
        self.assertTrue(any("saldos fraccionarios" in a for a in r["advertencias"]))

    def test_conservacion_exacta_incluso_con_redondeos_publicados(self):
        valores=[0,1e16,.25]+[0]*12
        p,d=datos(valores)
        original=trayectoria._saldo
        with patch.object(trayectoria,"_saldo",wraps=original) as representar:
            r=trayectoria.calcular_trayectoria(p,d,1e16,0,0)
        acumulado=Fraction(0); entradas=Fraction(0); inicial=Fraction.from_float(1e16)
        for i,f in enumerate(r["trayectoria"]):
            acumulado+=Fraction.from_float(float(valores[i]))
            entradas+=Fraction.from_float(f["recepcion_simulada"])
            # Captura el saldo exacto ANTES de convertirlo a float.
            self.assertEqual(representar.call_args_list[3*i].args[0],inicial+entradas-acumulado)
        self.assertNotEqual(representar.call_args_list[0].args[0],Fraction.from_float(r["trayectoria"][0]["balance_neto"]))

    def test_referencia_fraccion_independiente_100_escenarios(self):
        azar=random.Random(803)
        for _ in range(100):
            valores=[azar.randrange(60)/8 for _ in range(15)]
            inventario=azar.randrange(100)/8; reserva=azar.randrange(20)/8; plazo=azar.randrange(16)
            r=self.calcular(valores,inventario,plazo,reserva)
            balance=Fraction.from_float(inventario)
            for i,f in enumerate(r["trayectoria"]):
                entrada=Fraction.from_float(r["reposicion"]) if i==plazo else Fraction(0)
                balance+=entrada-Fraction.from_float(valores[i])
                self.assertEqual(f["balance_neto"],float(balance))
            self.assertEqual(balance,Fraction.from_float(inventario)+(Fraction.from_float(r["reposicion"]) if plazo<15 else 0)-sum(map(Fraction.from_float,valores)))

    def test_extremo_positivo_desborda_sin_resultado_parcial(self):
        p,d=datos([0,float_info.max]+[0]*13)
        with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion) as e:
            trayectoria.calcular_trayectoria(p,d,float_info.max,0,float_info.max)
        self.assertEqual(e.exception.codigo,"resultado_no_finito")
        self.assertFalse(hasattr(e.exception,"trayectoria"))

    def test_intermedio_gigante_se_cancela_antes_de_representar(self):
        r=self.calcular([float_info.max]+[0]*14,inventario=float_info.max,plazo=0,reserva=float_info.max)
        self.assertEqual(r["reposicion"],float_info.max)
        self.assertTrue(all(f["balance_neto"]==float_info.max for f in r["trayectoria"]))

    def test_extremo_negativo_finito(self):
        r=self.calcular([float_info.max]+[0]*14,inventario=0,plazo=15,reserva=0)
        self.assertEqual(r["resumen"]["balance_neto_final"],-float_info.max)
        self.assertEqual(r["resumen"]["demanda_pendiente_final"],float_info.max)

    def test_representador_extremos_ambos_signos(self):
        for signo in (1,-1):
            with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion) as e:
                trayectoria._saldo(signo*(Fraction.from_float(float_info.max)+1))
            self.assertEqual(e.exception.codigo,"resultado_no_finito")

    def test_saldos_enteros_no_representables_ambos_signos(self):
        for signo in (1,-1):
            with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion) as e:
                trayectoria._saldo(signo*Fraction(2**53+1))
            self.assertEqual(e.exception.codigo,"precision_no_representable")

    def test_saldo_diario_negativo_pierde_unidad_rechazado(self):
        p,d=datos([2**53,1,1]+[0]*12)
        with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion) as e:
            trayectoria.calcular_trayectoria(p,d,0,15,0)
        self.assertEqual(e.exception.codigo,"precision_no_representable")

    def test_saldo_diario_positivo_pierde_unidad_rechazado(self):
        p,d=datos([1,1]+[0]*13)
        with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion):
            trayectoria.calcular_trayectoria(p,d,2**53+2,15,0)

    def test_subnormales_con_signo_y_sin_cero_falso(self):
        minimo=nextafter(0.0,1.0)
        r=self.calcular([minimo]+[0]*14,inventario=0,plazo=15,reserva=0)
        self.assertEqual(r["trayectoria"][0]["balance_neto"],-minimo)
        self.assertEqual(r["resumen"]["demanda_pendiente_final"],minimo)
        self.assertEqual(trayectoria._saldo(Fraction.from_float(minimo)),minimo)

    def test_subdesbordamiento_de_representacion_controlado(self):
        # No nace de sumas de binary64 válidos (múltiplos del menor subnormal).
        # Se prueba explícitamente la frontera del representador firmado.
        for signo in (1,-1):
            with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion) as e:
                trayectoria._saldo(signo*Fraction(1,2**2000))
            self.assertEqual(e.exception.codigo,"resultado_no_finito")

    def test_quince_fechas_consecutivas_y_date(self):
        p,d=datos();p["fecha_corte"]=date(2017,6,16)
        for f in d:f["fecha_objetivo"]=date.fromisoformat(f["fecha_objetivo"])
        r=trayectoria.calcular_trayectoria(p,tuple(d),30,1,5)
        self.assertEqual([f["fecha_objetivo"] for f in r["trayectoria"]],[(date(2017,6,16)+timedelta(days=i)).isoformat() for i in range(1,16)])

    def test_entradas_invalidas_delegadas(self):
        for campo,valor in (("inventario_disponible",True),("reserva",float("nan")),("tiempo_abastecimiento",16)):
            p,d=datos();params={"inventario_disponible":30,"tiempo_abastecimiento":1,"reserva":5};params[campo]=valor
            with self.assertRaises(motor.ErrorSimulacionReposicion):trayectoria.calcular_trayectoria(p,d,**params)
        for transformar in (lambda d:d.pop(),lambda d:d[0].update(paso=2),lambda d:d[0].update(venta_estimada=-1)):
            p,d=datos();transformar(d)
            with self.assertRaises(motor.ErrorSimulacionReposicion):trayectoria.calcular_trayectoria(p,d,30,1,5)

    def test_delegacion_exactamente_una_vez(self):
        p,d=datos()
        with patch.object(motor,"calcular_simulacion",wraps=motor.calcular_simulacion) as calcular:
            trayectoria.calcular_trayectoria(p,d,30,1,5)
        calcular.assert_called_once_with(p,d,30,1,5)

    def test_identificacion_y_resultado_s4a1_conservados(self):
        p,d=datos();base=motor.calcular_simulacion(p,d,30,1,5)
        r=trayectoria.calcular_trayectoria(p,d,30,1,5)
        for k,v in base.items():
            if k not in ("supuestos","advertencias"):self.assertEqual(r[k],v)
        self.assertFalse(any("Sin trayectoria diaria" in s for s in r["supuestos"]))
        self.assertTrue(any("consumidores" in s for s in r["supuestos"]))

    def test_base_del_motor_no_se_modifica_ni_se_comparte(self):
        p,d=datos();base=motor.calcular_simulacion(p,d,30,1,5);antes=deepcopy(base)
        with patch.object(motor,"calcular_simulacion",return_value=base):
            r=trayectoria.calcular_trayectoria(p,d,30,1,5)
        self.assertEqual(base,antes)
        r["parametros"]["reserva"]=100;r["advertencias"].clear()
        self.assertEqual(base,antes)

    def test_entradas_intactas_y_repetibilidad(self):
        p,d=datos();antes=deepcopy((p,d))
        a=trayectoria.calcular_trayectoria(p,d,30,1,5);b=trayectoria.calcular_trayectoria(p,d,30,1,5)
        self.assertEqual(a,b);self.assertEqual((p,d),antes)

    def test_rechazo_conserva_entradas(self):
        p,d=datos([0,float_info.max]+[0]*13);antes=deepcopy((p,d))
        with self.assertRaises(trayectoria.ErrorTrayectoriaSimulacion):trayectoria.calcular_trayectoria(p,d,float_info.max,0,float_info.max)
        self.assertEqual((p,d),antes)

    def test_importacion_y_calculo_sin_efectos_secundarios(self):
        codigo=compile(Path(trayectoria.__file__).read_text(encoding="utf-8"),trayectoria.__file__,"exec")
        aislado=ModuleType("trayectoria_aislada");p,d=datos()
        with ExitStack() as stack:
            dobles=[stack.enter_context(patch.object(m,n,side_effect=AssertionError("E/S prohibida")))
                    for m,n in ((builtins,"open"),(io,"open"),(os,"open"),(sqlite3,"connect"),(socket,"socket"),(subprocess,"Popen"),(builtins,"print"))]
            exec(codigo,aislado.__dict__)
            self.assertEqual(aislado.calcular_trayectoria(p,d,30,1,5)["resumen"]["balance_neto_final"],5)
            for doble in dobles:doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
