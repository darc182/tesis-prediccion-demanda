"""HU005-A1: datos sintéticos; sin CSV original, base local ni modelos."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
import builtins
import io
import json
from math import fsum, isclose, isfinite, sqrt
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import metricas_evaluacion as metricas


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")


def datos(corte=CORTES[0], estimadas=None, reales=None):
    """Quince fechas creadas en memoria; los valores predeterminados son 10."""
    inicio = date.fromisoformat(corte)
    estimadas = [10] * 15 if estimadas is None else estimadas
    reales = [10] * 15 if reales is None else reales
    detalles = [{"paso": i, "fecha_objetivo": (inicio + timedelta(days=i)).isoformat(),
                 "venta_estimada": valor} for i, valor in enumerate(estimadas, 1)]
    observaciones = [{"fecha": (inicio + timedelta(days=i)).isoformat(), "sales": valor}
                     for i, valor in enumerate(reales, 1)]
    return detalles, observaciones


class MetricasEvaluacionTests(unittest.TestCase):
    def evaluar(self, detalles, observaciones, corte=CORTES[0]):
        resultado = metricas.calcular_metricas(corte, detalles, observaciones)
        self.assertEqual(resultado["observaciones_evaluadas"] + resultado["fechas_sin_registro"], 15)
        self.assertEqual(resultado["cobertura"], len(observaciones) / 15)
        self.assertEqual(resultado["fechas_sin_registro"], len(resultado["fechas_ausentes"]))
        self.assertEqual(len(resultado["detalles"]), 15)
        self.assertEqual(sum(d["venta_real"] is not None for d in resultado["detalles"]), len(observaciones))
        self.assertEqual(json.loads(json.dumps(resultado, allow_nan=False)), resultado)
        return resultado

    def rechazar(self, detalles, observaciones, codigo=None, corte=CORTES[0]):
        with self.assertRaises(metricas.ErrorMetricasEvaluacion) as error:
            metricas.calcular_metricas(corte, detalles, observaciones)
        if codigo is not None:
            self.assertEqual(error.exception.codigo, codigo)

    def test_quince_pronosticos_perfectos(self):
        d, o = datos(estimadas=list(range(15)), reales=list(range(15)))
        r = self.evaluar(d, o)
        self.assertEqual((r["mae"], r["wape"], r["cobertura"]), (0, 0, 1))
        self.assertEqual((r["observaciones_evaluadas"], r["fechas_sin_registro"]), (15, 0))
        self.assertEqual(r["fechas_ausentes"], [])
        self.assertTrue(all(f["error_absoluto"] == 0 for f in r["detalles"]))
        self.assertIn("observaciones_evaluadas / horizonte", r["definicion_cobertura"])

    def test_errores_conocidos_comprobables_manualmente(self):
        d, o = datos(estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
        r = self.evaluar(d, o)
        # 5 * (2 + 4 + 0) = 30; ventas = 5 * 60 = 300.
        self.assertEqual(r["mae"], 30 / 15)
        self.assertEqual(r["wape"], 100 * 30 / 300)
        self.assertEqual([f["error_absoluto"] for f in r["detalles"]], [2, 4, 0] * 5)

    def test_ceros_observados_con_predicciones_positivas(self):
        d, o = datos(estimadas=[2, 10, 18] * 5, reales=[0, 10, 20] * 5)
        r = self.evaluar(d, o)
        self.assertEqual(r["mae"], 20 / 15)
        self.assertAlmostEqual(r["wape"], 100 * 20 / 150)
        self.assertEqual(r["detalles"][0]["venta_real"], 0)
        self.assertEqual(r["detalles"][0]["error_absoluto"], 2)

    def test_todos_reales_cero_wape_indefinido(self):
        for estimadas in ([2] * 15, [0] * 15):
            with self.subTest(estimadas=estimadas):
                d, o = datos(estimadas=estimadas, reales=[0] * 15)
                r = self.evaluar(d, o)
                self.assertEqual(r["mae"], estimadas[0])
                self.assertIsNone(r["wape"])
                self.assertEqual(r["cobertura"], 1)

    def test_ventas_fraccionarias(self):
        d, o = datos(estimadas=[0.25, 1.75, 2.75] * 5, reales=[0.5, 1.25, 2.75] * 5)
        r = self.evaluar(d, o)
        self.assertEqual(r["mae"], 0.25)
        self.assertAlmostEqual(r["wape"], 100 * 3.75 / 22.5)

    def test_una_fecha_ausente_entre_observadas(self):
        d, o = datos(reales=[8] * 15)
        ausente = o.pop(7)["fecha"]
        r = self.evaluar(d, o)
        self.assertEqual((r["observaciones_evaluadas"], r["mae"], r["wape"]), (14, 2, 25))
        self.assertEqual(r["fechas_ausentes"], [ausente])
        self.assertIsNone(r["detalles"][7]["venta_real"])
        self.assertIsNone(r["detalles"][7]["error_absoluto"])

    def test_varias_fechas_ausentes_y_reales_desordenados(self):
        d, o = datos(estimadas=list(range(1, 16)), reales=list(range(1, 16)))
        faltan = (2, 6, 11)
        r = self.evaluar(d, [o[i] for i in reversed(range(15)) if i not in faltan])
        self.assertEqual((r["observaciones_evaluadas"], r["mae"], r["wape"]), (12, 0, 0))
        self.assertEqual(r["fechas_ausentes"], [o[i]["fecha"] for i in faltan])
        self.assertEqual(r["detalles"][3]["venta_real"], 4)

    def test_ninguna_observacion_real(self):
        d, _ = datos()
        r = self.evaluar(d, [])
        self.assertEqual((r["observaciones_evaluadas"], r["fechas_sin_registro"], r["cobertura"]), (0, 15, 0))
        self.assertIsNone(r["mae"])
        self.assertIsNone(r["wape"])
        self.assertEqual(r["fechas_ausentes"], [f["fecha_objetivo"] for f in d])
        self.assertTrue(all(f["error_absoluto"] is None for f in r["detalles"]))

    def test_emparejamiento_por_fecha_con_lista_real_invertida(self):
        d, o = datos(estimadas=list(range(15)), reales=list(range(15)))
        self.assertEqual(self.evaluar(d, o), self.evaluar(d, list(reversed(o))))

    def test_cero_y_ausencia_tienen_consecuencias_distintas(self):
        d, o = datos(estimadas=[4, 10] + [4] * 13, reales=[0, 10] + [0] * 13)
        con_cero = self.evaluar(d, o[:2])
        sin_registro = self.evaluar(d, o[1:2])
        self.assertEqual((con_cero["cobertura"], con_cero["mae"], con_cero["wape"]), (2 / 15, 2, 40))
        self.assertEqual((sin_registro["cobertura"], sin_registro["mae"], sin_registro["wape"]), (1 / 15, 0, 0))
        self.assertEqual(con_cero["detalles"][0]["venta_real"], 0)
        self.assertIsNone(sin_registro["detalles"][0]["venta_real"])

    def test_fecha_objetivo_duplicada(self):
        d, o = datos()
        d[5]["fecha_objetivo"] = d[4]["fecha_objetivo"]
        self.rechazar(d, o, "alineacion_invalida")

    def test_pronosticos_fuera_de_orden(self):
        d, o = datos()
        for alterados in (list(reversed(d)), d[:3] + [d[4], d[3]] + d[5:]):
            self.rechazar(alterados, o, "alineacion_invalida")

    def test_fecha_objetivo_incorrecta_u_omitida(self):
        for fecha in ("2017-06-16", "2017-06-19", "2017-07-02", "2017-08-01", "2017-08-15"):
            d, o = datos()
            d[0]["fecha_objetivo"] = fecha
            with self.subTest(fecha=fecha):
                self.rechazar(d, o, "alineacion_invalida")

    def test_pasos_duplicados_omitidos_o_de_tipo_incorrecto(self):
        for paso in (1, 0, 16, 3, True, 2.0, "2", None):
            d, o = datos()
            d[1]["paso"] = paso
            with self.subTest(paso=paso):
                self.rechazar(d, o, "alineacion_invalida")

    def test_horizonte_incompleto_o_excesivo(self):
        d, o = datos()
        for alterados in ([], d[:-1], d[:7] + d[8:], d + [d[-1]]):
            with self.subTest(cantidad=len(alterados)):
                self.rechazar(alterados, o, "horizonte_invalido")

    def test_resultados_sobre_los_tres_cortes_autorizados(self):
        for corte, fin in zip(CORTES, ("2017-07-01", "2017-07-16", "2017-07-31")):
            with self.subTest(corte=corte):
                d, o = datos(corte, estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
                r = self.evaluar(d, o, corte)
                self.assertEqual((r["fecha_corte"], r["horizonte"], r["mae"], r["wape"]), (corte, 15, 2, 10))
                self.assertEqual(r["detalles"][-1]["fecha_objetivo"], fin)
                d[0]["fecha_objetivo"] = date.fromisoformat(d[0]["fecha_objetivo"])
                o[0]["fecha"] = date.fromisoformat(o[0]["fecha"])
                self.assertEqual(self.evaluar(d, o, date.fromisoformat(corte)), r)

    def test_fechas_invalidas_en_todas_las_entradas(self):
        for valor in (None, True, 20170616, [], {}, b"2017-06-16", "", "20170616", "2017-6-16",
                      "2017-02-29", "2017-06-16T00:00:00", " 2017-06-16", datetime(2017, 6, 16)):
            for campo in ("corte", "objetivo", "real"):
                d, o = datos()
                corte = CORTES[0]
                if campo == "corte":
                    corte = valor
                elif campo == "objetivo":
                    d[0]["fecha_objetivo"] = valor
                else:
                    o[0]["fecha"] = valor
                with self.subTest(campo=campo, valor=valor):
                    self.rechazar(d, o, "fecha_invalida", corte)

    def test_cortes_no_autorizados_y_periodo_final_reservado(self):
        d, o = datos()
        cortes = ["2017-06-15", "2017-06-17", "2017-06-30", "2017-07-02", "2017-07-15", "2017-08-16"]
        cortes += [(date(2017, 7, 17) + timedelta(days=i)).isoformat() for i in range(30)]
        for corte in cortes:
            with self.subTest(corte=corte):
                self.rechazar(d, o, "corte_no_autorizado", corte)

    def test_reales_duplicados_incluso_con_fecha_en_otro_tipo(self):
        d, o = datos()
        for fecha in (o[0]["fecha"], date.fromisoformat(o[0]["fecha"])):
            self.rechazar(d, o + [{"fecha": fecha, "sales": 99}], "fecha_duplicada")

    def test_observaciones_fuera_del_horizonte_no_se_ignoran(self):
        for corte in CORTES:
            d, o = datos(corte)
            inicio = date.fromisoformat(corte)
            for fecha in (inicio, inicio - timedelta(days=1), inicio + timedelta(days=16),
                          date(2017, 8, 1), date(2017, 8, 15)):
                with self.subTest(corte=corte, fecha=fecha):
                    self.rechazar(d, o + [{"fecha": fecha, "sales": 0}], "observacion_fuera_horizonte", corte)

    def test_numeros_invalidos_en_estimadas_y_reales(self):
        for valor in (-1, -0.5, True, False, float("nan"), float("inf"), -float("inf"),
                      "10", "NaN", None, [], {}, 1j, 10 ** 400):
            for campo in ("venta_estimada", "sales"):
                d, o = datos()
                (d if campo == "venta_estimada" else o)[0][campo] = valor
                with self.subTest(campo=campo, valor=valor):
                    self.rechazar(d, o, "valor_invalido")

    def test_estructuras_invalidas_y_claves_obligatorias(self):
        d, o = datos()
        for filas in (None, True, {}, "", b"", iter([]), [None], [1], [[]], [{}]):
            for campo in ("detalles", "observaciones"):
                with self.subTest(campo=campo, filas=filas):
                    self.rechazar(filas if campo == "detalles" else d,
                                  filas if campo == "observaciones" else o, "estructura_invalida")
        for campo in ("paso", "fecha_objetivo", "venta_estimada", "fecha", "sales"):
            dd, oo = datos()
            del (dd if campo in dd[0] else oo)[0][campo]
            self.rechazar(dd, oo, "estructura_invalida")
        self.assertEqual(self.evaluar(tuple(d), tuple(o))["mae"], 0)

    def test_entradas_conservadas_al_evaluar_y_resultado_independiente(self):
        d, o = datos()
        o.reverse()
        d[0]["extra"] = {"lista": [1, 2]}
        o[0]["extra"] = [3]
        antes = deepcopy((d, o))
        r = self.evaluar(d, o)
        r["detalles"][0]["venta_estimada"] = 99
        r["fechas_ausentes"].append("otro")
        self.assertEqual((d, o), antes)
        self.assertEqual(self.evaluar(d, o)["mae"], 0)

    def test_entradas_conservadas_tras_rechazo(self):
        for causa in ("duplicado", "negativo", "desbordamiento"):
            d, o = datos()
            if causa == "duplicado":
                o.append(dict(o[0]))
            elif causa == "negativo":
                o[7]["sales"] = -1
            else:
                for fila in d:
                    fila["venta_estimada"] = 1e308
            antes = deepcopy((d, o))
            self.rechazar(d, o)
            self.assertEqual((d, o), antes)

    def test_suma_cuidadosa_con_magnitudes_diferentes(self):
        errores = [1e16] + [1] * 14
        d, o = datos(estimadas=errores, reales=[0] * 15)
        r = self.evaluar(d, o)
        self.assertEqual(r["mae"], fsum(errores) / 15)
        self.assertNotEqual(r["mae"], sum(errores) / 15)

    def test_desbordamiento_de_suma_de_errores(self):
        d, o = datos(estimadas=[1e308] * 15, reales=[0] * 15)
        self.rechazar(d, o, "resultado_no_finito")

    def test_desbordamiento_de_suma_de_reales(self):
        d, o = datos(estimadas=[1e308] * 15, reales=[1e308] * 15)
        self.rechazar(d, o, "resultado_no_finito")

    def test_desbordamiento_de_wape(self):
        d, o = datos(estimadas=[1e308] + [0] * 14, reales=[1e-308])
        self.rechazar(d, o, "resultado_no_finito")

    def test_valores_extremos_finitos_sin_desbordamiento_prematuro(self):
        d, o = datos(estimadas=[2e306] * 15, reales=[1e306] * 15)
        r = self.evaluar(d, o)
        self.assertEqual(r["mae"], 1e306)
        self.assertEqual(r["wape"], 100)
        d, o = datos(estimadas=[1e308] + [0] * 14, reales=[1e308])
        self.assertEqual(self.evaluar(d, o)["mae"], 0)

    def test_no_redondea_y_wape_puede_superar_cien(self):
        d, o = datos(estimadas=[1, 0, 0] + [0] * 12, reales=[0, 0, 0])
        self.assertEqual(self.evaluar(d, o)["mae"], 1 / 3)
        d, o = datos(estimadas=[4] * 15, reales=[1] * 15)
        self.assertEqual(self.evaluar(d, o)["wape"], 300)

    def test_rmse_errores_conocidos(self):
        d, o = datos(estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
        r = self.evaluar(d, o)
        # Cuadrados: 5 * (4 + 16 + 0) = 100; media = 100 / 15.
        self.assertEqual(r["rmse"], sqrt(20 / 3))
        self.assertEqual((r["mae"], r["wape"], r["cobertura"]), (2, 10, 1))

    def test_rmse_prediccion_perfecta(self):
        d, o = datos(estimadas=list(range(15)), reales=list(range(15)))
        self.assertEqual(self.evaluar(d, o)["rmse"], 0)

    def test_rmse_todos_los_reales_cero(self):
        for predicciones, esperado in (([2] * 15, 2), ([0] * 15, 0),
                                      ([2, 4, 0] * 5, sqrt(20 / 3))):
            with self.subTest(predicciones=predicciones):
                d, o = datos(estimadas=predicciones, reales=[0] * 15)
                r = self.evaluar(d, o)
                self.assertEqual(r["rmse"], esperado)
                self.assertIsNone(r["wape"])
                self.assertEqual(r["cobertura"], 1)

    def test_rmse_sin_observaciones(self):
        d, _ = datos()
        r = self.evaluar(d, [])
        self.assertIsNone(r["rmse"])
        self.assertIsNone(r["mae"])
        self.assertIsNone(r["wape"])
        self.assertEqual(r["cobertura"], 0)

    def test_rmse_una_y_varias_fechas_ausentes(self):
        d, o = datos(estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
        for faltan in ((1,), (1, 7, 10)):
            with self.subTest(faltan=faltan):
                disponibles = [o[i] for i in reversed(range(15)) if i not in faltan]
                r = self.evaluar(d, disponibles)
                # Cada fecha omitida tenía error 4; no participa en la media.
                esperado = sqrt((100 - 16 * len(faltan)) / len(disponibles))
                self.assertAlmostEqual(r["rmse"], esperado)
                self.assertEqual(r["fechas_ausentes"], [o[i]["fecha"] for i in faltan])
                self.assertEqual(r["cobertura"], len(disponibles) / 15)
                for i in faltan:
                    self.assertIsNone(r["detalles"][i]["error_absoluto"])

    def test_rmse_valores_fraccionarios(self):
        d, o = datos(estimadas=[0.25, 1.75, 2.75] * 5, reales=[0.5, 1.25, 2.75] * 5)
        r = self.evaluar(d, o)
        self.assertEqual(r["rmse"], sqrt(5 / 48))
        self.assertEqual(r["mae"], 0.25)
        self.assertAlmostEqual(r["wape"], 100 * 3.75 / 22.5)

    def test_rmse_tres_cortes_autorizados(self):
        for corte in CORTES:
            with self.subTest(corte=corte):
                d, o = datos(corte, estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
                self.assertEqual(self.evaluar(d, o, corte)["rmse"], sqrt(20 / 3))

    def test_rmse_errores_grandes_sin_desbordar_cuadrados(self):
        d, o = datos(estimadas=[2e306] * 15, reales=[1e306] * 15)
        # La formulación directa falla aunque RMSE = 1e306 es representable.
        with self.assertRaises(OverflowError):
            _ = (1e306) ** 2
        r = self.evaluar(d, o)
        self.assertEqual(r["rmse"], 1e306)
        self.assertTrue(isfinite(r["rmse"]))
        self.assertEqual((r["mae"], r["wape"]), (1e306, 100))

    def test_rmse_referencia_decimal_independiente(self):
        casos = (([8, 24, 30] * 5, [10, 20, 30] * 5),
                 ([0.25, 1.75, 2.75] * 5, [0.5, 1.25, 2.75] * 5),
                 ([1e306, 2e306, 0] * 5, [0] * 15),
                 ([1e-200, 2e-200, 0] * 5, [0] * 15))
        for estimadas, reales in casos:
            with self.subTest(estimadas=estimadas):
                d, o = datos(estimadas=estimadas, reales=reales)
                r = self.evaluar(d, o)
                # Referencia sin escalado: cuadrados y raíz en Decimal de
                # precisión alta, también para magnitudes fuera de float**2.
                with localcontext() as contexto:
                    contexto.prec = 80
                    cuadrados = [(Decimal.from_float(float(p)) - Decimal.from_float(float(y))) ** 2
                                 for p, y in zip(estimadas, reales)]
                    referencia = float((sum(cuadrados) / Decimal(len(reales))).sqrt())
                self.assertTrue(isclose(r["rmse"], referencia, rel_tol=1e-14, abs_tol=0))

    def test_rmse_conserva_entradas_y_contrato_previo(self):
        d, o = datos(estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
        o = list(reversed(o[:1] + o[2:]))
        antes = deepcopy((d, o))
        r = self.evaluar(d, o)
        self.assertEqual((d, o), antes)
        self.assertEqual(r["rmse"], sqrt(84 / 14))
        self.assertEqual(r["mae"], 26 / 14)
        self.assertEqual(r["wape"], (26 / 280) * 100)
        self.assertEqual(r["cobertura"], 14 / 15)
        self.assertEqual(r["fechas_ausentes"], [d[1]["fecha_objetivo"]])
        por_fecha = {fila["fecha"]: fila["sales"] for fila in o}
        for fila, detalle in zip(d, r["detalles"]):
            real = por_fecha.get(fila["fecha_objetivo"])
            self.assertEqual(detalle, {**fila, "venta_real": real,
                                      "error_absoluto": None if real is None else abs(real - fila["venta_estimada"])})

    def test_importacion_y_calculo_sin_efectos_externos(self):
        # Leer solo el código de este módulo antes de activar las guardias.
        codigo = compile(Path(metricas.__file__).read_text(encoding="utf-8"), metricas.__file__, "exec")
        aislado = ModuleType("metricas_puras_en_prueba")
        d, o = datos()
        d_rmse, o_rmse = datos(estimadas=[8, 24, 30] * 5, reales=[10, 20, 30] * 5)
        importar_real = builtins.__import__

        def importar(nombre, *args, **kwargs):
            if nombre not in ("collections.abc", "datetime", "math", "numbers"):
                raise AssertionError(f"Dependencia ajena al cálculo: {nombre}")
            return importar_real(nombre, *args, **kwargs)

        objetivos = ((builtins, "open"), (io, "open"), (Path, "open"), (os, "open"),
                     (sqlite3, "connect"), (socket, "socket"), (subprocess, "Popen"))
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            stack.enter_context(patch.object(builtins, "__import__", side_effect=importar))
            exec(codigo, aislado.__dict__)
            self.assertEqual(aislado.calcular_metricas(CORTES[0], d, o)["mae"], 0)
            self.assertEqual(aislado.calcular_metricas(CORTES[0], d_rmse, o_rmse)["rmse"], sqrt(20 / 3))
            for doble in dobles:
                doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
