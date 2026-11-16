"""S4-A1: simulación pura con entradas exclusivamente sintéticas."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
import builtins
import io
import json
from math import isfinite, nextafter
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


def datos(valores=None, corte="2017-06-16"):
    p = {"id": 101, "experimento_id": 201, "tecnica": "random_forest",
         "fecha_corte": corte, "horizonte": 15, "tipo": "validacion",
         "inicio_entrenamiento": "2017-01-01", "fin_entrenamiento": corte}
    inicio = date.fromisoformat(corte)
    d = [{"paso": i, "fecha_objetivo": (inicio + timedelta(days=i)).isoformat(), "venta_estimada": v}
         for i, v in enumerate([10.0] * 15 if valores is None else valores, 1)]
    return p, d


class SimulacionReposicionTests(unittest.TestCase):
    def calcular(self, valores=None, inventario=30, plazo=1, reserva=5):
        p, d = datos(valores)
        r = motor.calcular_simulacion(p, d, inventario, plazo, reserva)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        self.assertEqual(r["version_regla"], "cobertura_horizonte_15_pendientes_v1")
        return r

    def rechazar(self, p, d, codigo=None, inventario=30, plazo=1, reserva=5):
        with self.assertRaises(motor.ErrorSimulacionReposicion) as e:
            motor.calcular_simulacion(p, d, inventario, plazo, reserva)
        if codigo:
            self.assertEqual(e.exception.codigo, codigo)

    def test_calculo_manual_y_contrato(self):
        r = self.calcular()
        self.assertEqual((r["demanda_considerada"], r["reposicion"]), (150, 125))
        self.assertEqual((r["pronostico_id"], r["experimento_id"], r["tecnica"]), (101, 201, "random_forest"))
        self.assertEqual(r["periodo_cobertura"], {"fecha_inicio": "2017-06-17", "fecha_fin": "2017-07-01", "dias": 15})
        self.assertTrue(r["supuestos"] and r["advertencias"])
        self.assertNotIn("detalle_diario", r)

    def test_demanda_cero(self):
        self.assertEqual(self.calcular([0] * 15, inventario=0, reserva=0)["reposicion"], 0)

    def test_inventario_suficiente(self):
        self.assertEqual(self.calcular(inventario=200)["reposicion"], 0)

    def test_reserva_positiva_con_demanda_cero(self):
        self.assertEqual(self.calcular([0] * 15, inventario=2, reserva=5)["reposicion"], 3)

    def test_estimaciones_fraccionarias(self):
        r = self.calcular([0.25] * 15, inventario=1, reserva=0)
        self.assertEqual((r["demanda_considerada"], r["reposicion"]), (3.75, 2.75))

    def test_inventario_y_reserva_fraccionarios(self):
        self.assertEqual(self.calcular([0.25] * 15, inventario=1.125, reserva=0.5)["reposicion"], 3.125)

    def test_plazo_cero(self):
        r = self.calcular(plazo=0)
        self.assertEqual(r["fecha_recepcion"], "2017-06-17")
        self.assertTrue(r["recepcion_dentro_horizonte"])

    def test_plazo_catorce(self):
        r = self.calcular(plazo=14)
        self.assertEqual(r["fecha_recepcion"], "2017-07-01")
        self.assertTrue(r["recepcion_dentro_horizonte"])

    def test_plazo_quince_fuera_del_horizonte(self):
        r = self.calcular(plazo=15)
        self.assertEqual(r["fecha_recepcion"], "2017-07-02")
        self.assertFalse(r["recepcion_dentro_horizonte"])
        self.assertTrue(any("fuera del horizonte" in a for a in r["advertencias"]))
        self.assertEqual(r["reposicion"], 125)

    def test_plazo_no_altera_demanda_ni_reposicion(self):
        resultados = [self.calcular(plazo=l) for l in range(16)]
        self.assertEqual({r["reposicion"] for r in resultados}, {125})
        self.assertEqual({r["demanda_considerada"] for r in resultados}, {150})
        self.assertEqual(len({r["fecha_recepcion"] for r in resultados}), 16)

    def test_tres_tecnicas_reconocidas(self):
        for t in motor.TECNICAS:
            p, d = datos(); p["tecnica"] = t
            self.assertEqual(motor.calcular_simulacion(p, d, 0, 0, 0)["tecnica"], t)

    def test_experimento_opcional(self):
        p, d = datos(); del p["experimento_id"]
        self.assertIsNone(motor.calcular_simulacion(p, d, 0, 0, 0)["experimento_id"])

    def test_fecha_date_y_tuplas_sin_conversion_de_strings_numericos(self):
        p, d = datos(); p["fecha_corte"] = date(2017, 6, 16)
        for fila in d:
            fila["fecha_objetivo"] = date.fromisoformat(fila["fecha_objetivo"])
        self.assertEqual(motor.calcular_simulacion(p, tuple(d), 0, 0, 0)["demanda_considerada"], 150)

    def test_corte_no_elegible_pero_calendario_coherente(self):
        # El núcleo no decide elegibilidad ni consulta observaciones.
        p, d = datos(corte="2020-02-20")
        self.assertEqual(motor.calcular_simulacion(p, d, 0, 0, 0)["periodo_cobertura"]["fecha_fin"], "2020-03-06")

    def test_longitudes_incorrectas(self):
        p, d = datos()
        for filas in ([], d[:-1], d + [d[-1]]):
            self.rechazar(p, filas, "alineacion_invalida")

    def test_detalle_duplicado(self):
        p, d = datos(); d[7] = deepcopy(d[6])
        self.rechazar(p, d, "alineacion_invalida")

    def test_pasos_desordenados(self):
        p, d = datos(); d[0], d[1] = d[1], d[0]
        self.rechazar(p, d, "alineacion_invalida")

    def test_pasos_invalidos_incluidos_booleanos(self):
        for paso in (True, 1.0, "1", 0, 16):
            p, d = datos(); d[0]["paso"] = paso
            self.rechazar(p, d, "alineacion_invalida")

    def test_fecha_objetivo_incorrecta(self):
        p, d = datos(); d[5]["fecha_objetivo"] = "2017-06-29"
        self.rechazar(p, d, "alineacion_invalida")

    def test_fechas_invalidas_no_canonicas_y_datetime(self):
        for fecha in ("2017-02-30", "20170616", "2017-6-16", "2017-06-16T00:00:00", datetime(2017, 6, 16), None, True):
            for cabecera in (True, False):
                p, d = datos()
                if cabecera: p["fecha_corte"] = fecha
                else: d[0]["fecha_objetivo"] = fecha
                self.rechazar(p, d, "fecha_invalida")

    def test_estimaciones_invalidas(self):
        for valor in (-1, float("nan"), float("inf"), -float("inf"), True, False, "10", None, [], complex(1), Decimal("1")):
            p, d = datos(); d[0]["venta_estimada"] = valor
            self.rechazar(p, d, "valor_invalido")

    def test_inventario_y_reserva_invalidos(self):
        for valor in (-1, float("nan"), float("inf"), True, False, "1", None, {}, complex(1)):
            for campo in ("inventario", "reserva"):
                p, d = datos()
                self.rechazar(p, d, "valor_invalido", **{campo: valor})

    def test_plazos_invalidos(self):
        for valor in (-1, 16, 1.5, 1.0, True, False, "1", None, float("nan")):
            p, d = datos(); self.rechazar(p, d, "plazo_invalido", plazo=valor)

    def test_identificadores_invalidos(self):
        for campo in ("id", "experimento_id"):
            for v in (0, -1, True, 1.0, "101", None, 2**63):
                p, d = datos(); p[campo] = v
                self.rechazar(p, d, "identificador_invalido")

    def test_limite_sqlite_aceptado_sin_afirmar_existencia(self):
        p, d = datos(); p["id"] = p["experimento_id"] = 2**63 - 1
        self.assertEqual(motor.calcular_simulacion(p, d, 0, 0, 0)["pronostico_id"], 2**63 - 1)

    def test_identificadores_contradictorios(self):
        for en_detalle in (False, True):
            p, d = datos()
            (d[4] if en_detalle else p)["pronostico_id"] = 999
            self.rechazar(p, d, "identificador_invalido")

    def test_metadatos_invalidos(self):
        for campo, valor in (("tecnica", "desconocida"), ("tecnica", True), ("horizonte", True), ("horizonte", 15.0), ("horizonte", 14), ("tipo", "otro"), ("fin_entrenamiento", "2017-06-17"), ("inicio_entrenamiento", "2017-06-17")):
            p, d = datos(); p[campo] = valor
            self.rechazar(p, d, "metadatos_invalidos")

    def test_entrenamiento_parcial_rechazado(self):
        p, d = datos(); del p["fin_entrenamiento"]
        self.rechazar(p, d, "metadatos_invalidos")

    def test_estructuras_incorrectas_y_campos_ausentes(self):
        p, d = datos()
        for v in (None, [], "cabecera", {}): self.rechazar(v, d, "estructura_invalida")
        for v in (None, {}, iter(d), "detalles"): self.rechazar(p, v, "estructura_invalida")
        for campo in ("paso", "fecha_objetivo", "venta_estimada"):
            filas = deepcopy(d); del filas[0][campo]
            self.rechazar(p, filas, "estructura_invalida")
        filas = deepcopy(d); filas[0] = None
        self.rechazar(p, filas, "estructura_invalida")

    def test_suma_demanda_desbordada(self):
        p, d = datos([1e308] * 15)
        self.rechazar(p, d, "resultado_no_finito")

    def test_q_desbordada(self):
        p, d = datos([float_info.max] + [0] * 14)
        self.rechazar(p, d, "resultado_no_finito", inventario=0, reserva=1)

    def test_cancelacion_evitar_infinito_intermedio(self):
        r = self.calcular([1e308] + [0] * 14, inventario=1e308, reserva=1e308)
        self.assertEqual(r["reposicion"], 1e308)

    def test_cancelacion_conserva_reserva_pequena(self):
        self.assertEqual(self.calcular([1e308] + [0] * 14, inventario=1e308, reserva=1)["reposicion"], 1)

    def test_subnormal_positivo_no_se_pierde(self):
        minimo = nextafter(0.0, 1.0)
        r = self.calcular([minimo] + [0] * 14, inventario=0, reserva=0)
        self.assertEqual(r["reposicion"], minimo)
        self.assertGreater(r["reposicion"], 0)

    def test_entrada_numerica_fuera_de_rango(self):
        for valor in (10**1000, Fraction(1, 10**1000)):
            p, d = datos(); self.rechazar(p, d, "valor_invalido", inventario=valor)

    def test_limite_calendario_controlado(self):
        p = {"id": 1, "tecnica": motor.TECNICAS[0], "fecha_corte": "9999-12-31", "horizonte": 15}
        self.rechazar(p, [], "resultado_no_finito")
        p, d = datos(corte="9999-12-16")
        self.rechazar(p, d, "resultado_no_finito", plazo=15)

    def test_referencia_decimal_independiente(self):
        azar = random.Random(491)
        with localcontext() as contexto:
            contexto.prec = 1200
            for _ in range(100):
                valores = [azar.randrange(1000) / 8 for _ in range(15)]
                inventario, reserva = azar.randrange(10000) / 8, azar.randrange(1000) / 8
                demanda = sum((Decimal.from_float(v) for v in valores), Decimal(0))
                esperado = max(Decimal(0), demanda + Decimal.from_float(reserva) - Decimal.from_float(inventario))
                r = self.calcular(valores, inventario=inventario, reserva=reserva)
                self.assertEqual(r["demanda_considerada"], float(demanda))
                self.assertEqual(r["reposicion"], float(esperado))
                self.assertTrue(isfinite(r["reposicion"]))

    def test_no_modifica_entradas_ni_comparte_resultados(self):
        p, d = datos(); original = deepcopy((p, d))
        a = motor.calcular_simulacion(p, d, 30, 1, 5)
        b = motor.calcular_simulacion(p, d, 30, 1, 5)
        self.assertEqual(a, b); self.assertEqual((p, d), original)
        a["parametros"]["reserva"] = 100; a["supuestos"].clear()
        self.assertEqual(b["parametros"]["reserva"], 5); self.assertTrue(b["supuestos"])

    def test_rechazo_tampoco_modifica_entradas(self):
        p, d = datos(); d[4]["paso"] = 99; original = deepcopy((p, d))
        self.rechazar(p, d); self.assertEqual((p, d), original)

    def test_importacion_y_calculo_sin_efectos_secundarios(self):
        codigo = compile(Path(motor.__file__).read_text(encoding="utf-8-sig"), motor.__file__, "exec")
        aislado = ModuleType("motor_simulacion_aislado")
        p, d = datos()
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(m, nombre, side_effect=AssertionError("E/S prohibida")))
                      for m, nombre in ((builtins, "open"), (io, "open"), (os, "open"), (sqlite3, "connect"),
                                        (socket, "socket"), (subprocess, "Popen"), (builtins, "print"))]
            exec(codigo, aislado.__dict__)
            self.assertEqual(aislado.calcular_simulacion(p, d, 30, 1, 5)["reposicion"], 125)
            for doble in dobles: doble.assert_not_called()


    def test_r1_a_total_entero_no_representable_rechazado(self):
        p, d = datos([1e16, 1] + [0] * 13)
        self.rechazar(p, d, "precision_no_representable", inventario=1e16, reserva=0)
        self.assertEqual(Fraction(10**16) + 1 - Fraction(10**16), 1)

    def test_r1_b_entrada_entera_no_representable_rechazada(self):
        p, d = datos([0] * 15)
        self.rechazar(p, d, "precision_no_representable", inventario=2**53, reserva=2**53 + 1)
        self.assertEqual((2**53 + 1) - 2**53, 1)

    def test_r1_c_binario_difiere_del_decimal(self):
        r = self.calcular([0.1] * 3 + [0] * 12, inventario=0.3, reserva=0)
        with localcontext() as c:
            c.prec = 1200
            binario = 3 * Decimal.from_float(0.1) - Decimal.from_float(0.3)
            decimal = 3 * Decimal("0.1") - Decimal("0.3")
        self.assertEqual(r["reposicion"], float(binario))
        self.assertEqual(r["reposicion"], 2**-55)
        self.assertEqual(decimal, 0)
        self.assertTrue(any("binario" in s for s in r["supuestos"]))

    def test_r1_cancelacion_antes_de_redondear_demanda(self):
        # D15 exacta no es entera: su representación final admite redondeo natural.
        valores = [1e16, 0.25] + [0] * 13
        r = self.calcular(valores, inventario=1e16, reserva=0)
        self.assertEqual(r["reposicion"], 0.25)
        self.assertEqual(r["demanda_considerada"], 1e16)

    def test_r1_enteros_grandes_exactamente_representables(self):
        r = self.calcular([0] * 15, inventario=2**53, reserva=2**53 + 2)
        self.assertEqual(r["reposicion"], 2)
        self.assertEqual(r["parametros"]["reserva"], float(2**53 + 2))

    def test_r1_estimacion_entera_no_representable(self):
        p, d = datos([2**53 + 1] + [0] * 14)
        self.rechazar(p, d, "precision_no_representable")

    def test_r1_reposicion_entera_no_representable(self):
        p, d = datos([2**53] + [0] * 14)
        self.rechazar(p, d, "precision_no_representable", inventario=0, reserva=1)

    def test_r1_real_racional_no_se_convierte_silenciosamente(self):
        p, d = datos()
        self.rechazar(p, d, "precision_no_representable", inventario=Fraction(1, 3))
        self.assertEqual(self.calcular(inventario=Fraction(1, 4))["parametros"]["inventario_disponible"], 0.25)

    def test_r1_decimal_cero_con_fracciones_binarias_exactas(self):
        self.assertEqual(self.calcular([0.125] * 3 + [0] * 12, inventario=0.375, reserva=0)["reposicion"], 0)

    def test_r1_referencia_binaria_decimal_con_cancelaciones(self):
        with localcontext() as c:
            c.prec = 1200
            for x in (0.1, 0.2, 0.7, 1e-200, 1e-300):
                valores = [x, x, x] + [0] * 12
                inventario = x * 3
                exacto = max(Decimal(0), 3 * Decimal.from_float(x) - Decimal.from_float(inventario))
                r = self.calcular(valores, inventario=inventario, reserva=0)
                self.assertEqual(r["reposicion"], float(exacto))

    def test_r1_no_depende_del_contexto_decimal(self):
        with localcontext() as c:
            c.prec = 2
            a = self.calcular([0.1] * 3 + [0] * 12, inventario=0.3, reserva=0)
        with localcontext() as c:
            c.prec = 200
            b = self.calcular([0.1] * 3 + [0] * 12, inventario=0.3, reserva=0)
        self.assertEqual(a, b)

    def test_r1_conversion_final_no_oculta_subdesbordamiento(self):
        with self.assertRaises(motor.ErrorSimulacionReposicion) as e:
            motor._representar(Fraction(1, 2**2000))
        self.assertEqual(e.exception.codigo, "resultado_no_finito")



    def test_r1_aproximacion_final_explicita(self):
        r = self.calcular([1e16, 0.25] + [0] * 13, inventario=1e16, reserva=0)
        self.assertTrue(any("aproximada" in a for a in r["advertencias"]))
        self.assertEqual(r["reposicion"], 0.25)

    def test_r1_json_y_sqlite_real_en_memoria(self):
        r = self.calcular([0.1] * 3 + [0] * 12, inventario=0.3, reserva=0)
        copia = json.loads(json.dumps(r, allow_nan=False))
        self.assertEqual(copia, r)
        with sqlite3.connect(":memory:") as c:
            c.execute("CREATE TABLE prueba (demanda REAL, reposicion REAL)")
            c.execute("INSERT INTO prueba VALUES (?,?)", (copia["demanda_considerada"], copia["reposicion"]))
            self.assertEqual(c.execute("SELECT demanda,reposicion FROM prueba").fetchone(),
                             (r["demanda_considerada"], r["reposicion"]))


if __name__ == "__main__":
    unittest.main()
