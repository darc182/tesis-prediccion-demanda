"""Pruebas sintéticas de retardos por calendario y disponibilidad temporal."""

from datetime import date, timedelta
from math import fsum, isfinite
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.linea_base_prediccion import cargar_historico_csv
from scripts.preparar_series_modelado import leer_configuracion
from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS, COLUMNAS_TABLA,
    HistorialInsuficiente, ResultadoSerie, _variables_por_fecha,
    construir_tabla_serie, construir_tablas, construir_variables_futuras,
)


CORTE = date(2017, 6, 16)


def observaciones(tienda=1, familia="A", escala=1.0):
    return pd.DataFrame({
        "date": [f"2017-06-{d:02d}" for d in range(1, 17)],
        "store_nbr": tienda, "family": familia,
        "sales": [d * escala for d in range(1, 17)], "onpromotion": 0,
    })


class VariablesPredictorasTests(unittest.TestCase):
    def construir(self, datos=None):
        return construir_tabla_serie(observaciones() if datos is None else datos, 1, "A", CORTE)

    def test_retardos_exactos_y_calendario(self):
        fila = self.construir().tabla.iloc[0]
        self.assertEqual(fila["fecha_objetivo"], date(2017, 6, 15))
        self.assertEqual([fila[c] for c in ("lag_1", "lag_7", "lag_14")], [14, 8, 1])
        self.assertEqual(fila["dia_semana"], 3)
        self.assertEqual(fila["mes"], 6)

    def test_media_siete_dias_anteriores(self):
        self.assertAlmostEqual(self.construir().tabla.iloc[0]["media_7"], 11.0)

    def test_venta_objetivo_no_es_predictor(self):
        datos = observaciones()
        datos.loc[14, "sales"] = 1e12
        original = self.construir().tabla.iloc[[0]]
        modificado = self.construir(datos).tabla.iloc[[0]]
        pd.testing.assert_frame_equal(original.loc[:, list(COLUMNAS_PREDICTORAS)],
                                      modificado.loc[:, list(COLUMNAS_PREDICTORAS)])
        self.assertEqual(modificado.iloc[0]["sales"], 1e12)

    def test_ventas_posteriores_al_corte_no_cambian_entrenamiento(self):
        datos = observaciones()
        futuro = datos.iloc[[-1]].copy()
        futuro["date"] = "2017-06-17"
        futuro["sales"] = 1e12
        entrada = pd.concat([datos, futuro], ignore_index=True)
        copia = entrada.copy(deep=True)
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(entrada).tabla)
        pd.testing.assert_frame_equal(entrada, copia)

    def test_dia_ausente_no_se_desplaza_a_otro_registro(self):
        resultado = self.construir(observaciones().drop(index=13))
        descartada = next(d for d in resultado.descartes if d.fecha_objetivo == date(2017, 6, 15))
        self.assertEqual(descartada.faltantes["lag_1"], (date(2017, 6, 14),))
        self.assertEqual(descartada.faltantes["media_7"], (date(2017, 6, 14),))
        self.assertFalse(any(d.fecha_objetivo == date(2017, 6, 14) for d in resultado.descartes))
        self.assertEqual(resultado.observaciones_historicas, 15)

    def test_cero_real_se_conserva(self):
        datos = observaciones()
        datos.loc[13, "sales"] = 0
        datos.loc[14, "sales"] = 0
        resultado = self.construir(datos)
        self.assertEqual(len(resultado.tabla), 2)
        self.assertEqual(resultado.tabla.iloc[0]["lag_1"], 0)
        self.assertEqual(resultado.tabla.iloc[0]["sales"], 0)

    def test_exclusion_y_contabilidad_de_motivos(self):
        resultado = self.construir()
        self.assertEqual(resultado.observaciones_historicas, 16)
        self.assertEqual(len(resultado.tabla), 2)
        self.assertEqual(len(resultado.descartes), 14)
        self.assertEqual(resultado.descartes_por_variable,
                         {"lag_1": 1, "lag_7": 7, "lag_14": 14,
                          "media_7": 7, "dia_semana": 0, "mes": 0})
        self.assertEqual(resultado.observaciones_historicas,
                         len(resultado.tabla) + len(resultado.descartes))

    def test_conservacion_de_fracciones(self):
        fila = self.construir(observaciones(escala=0.125)).tabla.iloc[0]
        self.assertEqual(fila["lag_1"], 1.75)
        self.assertEqual(fila["lag_14"], 0.125)
        self.assertEqual(fila["sales"], 1.875)
        self.assertAlmostEqual(fila["media_7"], 1.375)

    def test_separacion_por_tienda_y_familia(self):
        claves = ((1, "A"), (1, "B"), (2, "A"))
        datos = pd.concat([observaciones(t, f, escala=i)
                           for i, (t, f) in enumerate(claves, 1)], ignore_index=True)
        resultados = construir_tablas(datos, CORTE, claves)
        for i, resultado in enumerate(resultados, 1):
            fila = resultado.tabla.iloc[0]
            self.assertEqual(fila["lag_1"], 14 * i)
            self.assertEqual(fila["lag_7"], 8 * i)
            self.assertEqual(fila["lag_14"], i)
            self.assertAlmostEqual(fila["media_7"], 11 * i)

    def test_orden_cronologico_y_columnas_estables(self):
        tabla = self.construir(observaciones().iloc[::-1]).tabla
        self.assertEqual(tuple(tabla.columns), COLUMNAS_TABLA)
        self.assertEqual(tabla["fecha_objetivo"].tolist(), [date(2017, 6, 15), CORTE])
        pd.testing.assert_frame_equal(tabla, self.construir().tabla)

    def test_coincidencia_entrenamiento_inferencia(self):
        tabla = self.construir().tabla
        for i, fila in tabla.iterrows():
            objetivo = fila["fecha_objetivo"]
            futuro = construir_variables_futuras(observaciones(), 1, "A",
                                                 objetivo - timedelta(days=1), objetivo)
            pd.testing.assert_frame_equal(tabla.loc[[i], list(COLUMNAS_PREDICTORAS)].reset_index(drop=True),
                                          futuro)

    def test_predicciones_previas_explicitas_sin_ventas_reales_futuras(self):
        datos = observaciones()
        futuro = datos.iloc[[-1, -1, -1]].copy()
        futuro["date"] = ["2017-06-17", "2017-06-18", "2017-06-19"]
        futuro["sales"] = 1e12
        entrada = pd.concat([datos, futuro], ignore_index=True)
        previas = {date(2017, 6, 17): 100.25, date(2017, 6, 18): 200.5}
        x = construir_variables_futuras(entrada, 1, "A", CORTE, date(2017, 6, 19), previas)
        self.assertEqual(x.iloc[0]["lag_1"], 200.5)
        self.assertEqual(x.iloc[0]["lag_7"], 12)
        self.assertEqual(x.iloc[0]["lag_14"], 5)
        self.assertAlmostEqual(x.iloc[0]["media_7"], (12 + 13 + 14 + 15 + 16 + 100.25 + 200.5) / 7)
        pd.testing.assert_frame_equal(x, construir_variables_futuras(
            datos, 1, "A", CORTE, date(2017, 6, 19), previas))
        with self.assertRaises(HistorialInsuficiente):
            construir_variables_futuras(entrada, 1, "A", CORTE, date(2017, 6, 19))

    def test_error_inferencia_sin_historial_suficiente(self):
        with self.assertRaisesRegex(HistorialInsuficiente, "lag_14") as contexto:
            construir_variables_futuras(observaciones().iloc[-7:], 1, "A", CORTE, date(2017, 6, 17))
        self.assertEqual(contexto.exception.faltantes["lag_14"], (date(2017, 6, 3),))
        with self.assertRaises(HistorialInsuficiente):
            construir_variables_futuras(observaciones().iloc[:0], 1, "A", CORTE, date(2017, 6, 17))

    def test_lectura_y_procesamiento_diez_series_configuradas(self):
        claves = leer_configuracion()
        datos = pd.concat([observaciones(t, f) for t, f in claves], ignore_index=True)
        datos = datos.sort_values(["date", "store_nbr", "family"])
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            datos.to_csv(ruta, index=False)
            with ruta.open("a", encoding="utf-8") as archivo:
                t, f = claves[0]
                archivo.write(f"2017-06-17,{t},{f},NO_LEER,NO_LEER\n")
            historico = cargar_historico_csv(ruta, claves, CORTE, tamano_bloque=23)
            resultados = construir_tablas(historico, CORTE)
        self.assertEqual(len(resultados), 10)
        self.assertEqual(tuple((r.store_nbr, r.family) for r in resultados), claves)
        self.assertTrue(all(r.observaciones_historicas == 16 and len(r.tabla) == 2
                            and len(r.descartes) == 14 for r in resultados))

    def test_predicciones_previas_invalidas(self):
        for previas in ({CORTE: 1}, {date(2017, 6, 19): 1},
                        {date(2017, 6, 17): -1}, {date(2017, 6, 17): float("nan")},
                        {date(2017, 6, 17): float("inf")}):
            with self.subTest(previas=previas), self.assertRaises(ValueError):
                construir_variables_futuras(observaciones(), 1, "A", CORTE,
                                            date(2017, 6, 19), previas)

    def test_onpromotion_no_es_predictor(self):
        datos = observaciones()
        datos["onpromotion"] = 9999
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(datos).tabla)
        self.assertNotIn("onpromotion", COLUMNAS_TABLA)

    def test_historial_corto_descarta_sin_fabricar_filas(self):
        resultado = self.construir(observaciones().iloc[:3])
        self.assertTrue(resultado.tabla.empty)
        self.assertEqual(tuple(resultado.tabla.columns), COLUMNAS_TABLA)
        self.assertEqual(len(resultado.descartes), 3)


class VariablesAlternativasTests(unittest.TestCase):
    columnas = COLUMNAS_PREDICTORAS_ALTERNATIVAS
    esquema = ("store_nbr", "family", "fecha_objetivo", *columnas, "sales")

    def construir(self, datos=None, corte=CORTE):
        return construir_tabla_serie(observaciones() if datos is None else datos,
                                     1, "A", corte, columnas_predictoras=self.columnas)

    def futuras(self, datos=None, objetivo=CORTE + timedelta(days=1), previas=None):
        return construir_variables_futuras(
            observaciones() if datos is None else datos, 1, "A", CORTE, objetivo, previas,
            columnas_predictoras=self.columnas)

    def test_constantes_clasicas_y_extension_exacta(self):
        self.assertEqual(COLUMNAS_PREDICTORAS,
                         ("lag_1", "lag_7", "lag_14", "media_7", "dia_semana", "mes"))
        self.assertEqual(COLUMNAS_TABLA,
                         ("store_nbr", "family", "fecha_objetivo", *COLUMNAS_PREDICTORAS, "sales"))
        self.assertEqual(self.columnas, (*COLUMNAS_PREDICTORAS, "media_14"))
        clasico = construir_tabla_serie(observaciones(), 1, "A", CORTE)
        alternativo = self.construir()
        self.assertEqual(tuple(alternativo.tabla.columns), self.esquema)
        pd.testing.assert_frame_equal(alternativo.tabla.drop(columns="media_14"), clasico.tabla)
        self.assertEqual(tuple(self.futuras().columns), self.columnas)

    def test_clasico_explicito_igual_a_predeterminado_en_todas_las_entradas(self):
        datos = observaciones().drop(index=4)
        original = construir_tabla_serie(datos, 1, "A", CORTE)
        explicito = construir_tabla_serie(datos, 1, "A", CORTE,
                                          columnas_predictoras=COLUMNAS_PREDICTORAS)
        pd.testing.assert_frame_equal(original.tabla, explicito.tabla)
        self.assertEqual(original.descartes, explicito.descartes)
        self.assertEqual(original.descartes_por_variable, explicito.descartes_por_variable)
        self.assertEqual(original.observaciones_historicas, explicito.observaciones_historicas)
        a = construir_tablas(datos, CORTE, ((1, "A"),))
        b = construir_tablas(datos, CORTE, ((1, "A"),),
                            columnas_predictoras=COLUMNAS_PREDICTORAS)
        pd.testing.assert_frame_equal(a[0].tabla, b[0].tabla)
        objetivo = CORTE + timedelta(days=1)
        pd.testing.assert_frame_equal(
            construir_variables_futuras(observaciones(), 1, "A", CORTE, objetivo, None),
            construir_variables_futuras(observaciones(), 1, "A", CORTE, objetivo, None,
                                        columnas_predictoras=COLUMNAS_PREDICTORAS))

    def test_resultado_serie_conserva_constructor_posicional(self):
        r = construir_tabla_serie(observaciones(), 1, "A", CORTE)
        copia = ResultadoSerie(r.store_nbr, r.family, r.fecha_corte,
                               r.observaciones_historicas, r.tabla, r.descartes)
        self.assertEqual(copia.columnas_predictoras, COLUMNAS_PREDICTORAS)
        self.assertEqual(copia.descartes_por_variable, r.descartes_por_variable)

    def test_media_catorce_valores_conocidos(self):
        self.assertAlmostEqual(self.construir().tabla.iloc[0]["media_14"], 7.5)
        self.assertAlmostEqual(self.construir().tabla.iloc[1]["media_14"], 8.5)
        self.assertAlmostEqual(self.futuras().iloc[0]["media_14"], 9.5)

    def test_nucleo_consulta_exclusivamente_catorce_fechas_exactas(self):
        objetivo = CORTE + timedelta(days=1)
        esperadas = {objetivo - timedelta(days=k) for k in range(1, 15)}
        consultadas = set()

        class VentasVigiladas(dict):
            def __contains__(self, fecha):
                if fecha not in esperadas:
                    raise AssertionError(f"Fecha indebida: {fecha}")
                consultadas.add(fecha)
                return super().__contains__(fecha)

            def __getitem__(self, fecha):
                if fecha not in esperadas:
                    raise AssertionError(f"Fecha indebida: {fecha}")
                consultadas.add(fecha)
                return super().__getitem__(fecha)

        ventas = VentasVigiladas({d: float(d.day) for d in esperadas})
        x = _variables_por_fecha(ventas, objetivo, columnas_predictoras=self.columnas)
        self.assertEqual(consultadas, esperadas)
        self.assertAlmostEqual(x["media_14"], 9.5)

    def test_venta_objetivo_no_cambia_x_alternativa(self):
        datos = observaciones()
        datos.loc[14, "sales"] = 999999.25
        original = self.construir().tabla.iloc[[0]]
        cambiado = self.construir(datos).tabla.iloc[[0]]
        pd.testing.assert_frame_equal(original.loc[:, list(self.columnas)],
                                      cambiado.loc[:, list(self.columnas)])
        self.assertEqual(cambiado.iloc[0]["sales"], 999999.25)

    def test_ventas_posteriores_al_corte_no_cambian_x(self):
        datos = observaciones()
        futuro = datos.iloc[[-1]].copy()
        futuro["date"] = "2017-06-17"
        futuro["sales"] = "NO_USAR"
        entrada = pd.concat([datos, futuro], ignore_index=True)
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(entrada).tabla)
        pd.testing.assert_frame_equal(self.futuras(), self.futuras(entrada))

    def test_ausencia_t_menos_diez_solo_invalida_alternativo(self):
        datos = observaciones().drop(index=4)  # Para t=15 falta el día 5.
        clasico = construir_tabla_serie(datos, 1, "A", CORTE)
        self.assertIn(date(2017, 6, 15), clasico.tabla["fecha_objetivo"].tolist())
        r = self.construir(datos)
        d = next(d for d in r.descartes if d.fecha_objetivo == date(2017, 6, 15))
        self.assertEqual(d.faltantes, {"media_14": (date(2017, 6, 5),)})
        self.assertNotIn(date(2017, 6, 5), [d.fecha_objetivo for d in r.descartes])

    def test_ausencia_t_menos_catorce_conserva_ambos_motivos(self):
        r = self.construir(observaciones().drop(index=0))
        d = next(d for d in r.descartes if d.fecha_objetivo == date(2017, 6, 15))
        self.assertEqual(d.faltantes, {"lag_14": (date(2017, 6, 1),),
                                      "media_14": (date(2017, 6, 1),)})

    def test_ausencia_t_menos_siete_conserva_tres_motivos(self):
        r = self.construir(observaciones().drop(index=7))
        d = next(d for d in r.descartes if d.fecha_objetivo == date(2017, 6, 15))
        self.assertEqual(d.faltantes, {nombre: (date(2017, 6, 8),)
                                      for nombre in ("lag_7", "media_7", "media_14")})
        self.assertEqual(sum(d.fecha_objetivo == date(2017, 6, 15) for d in r.descartes), 1)

    def test_varias_ausencias_identificadas_completamente(self):
        datos = observaciones().drop(index=[0, 4, 7, 13])
        with self.assertRaises(HistorialInsuficiente) as contexto:
            construir_variables_futuras(datos, 1, "A", date(2017, 6, 14), date(2017, 6, 15),
                                        columnas_predictoras=self.columnas)
        self.assertEqual(contexto.exception.objetivo, date(2017, 6, 15))
        self.assertEqual(contexto.exception.faltantes, {
            "lag_1": (date(2017, 6, 14),), "lag_7": (date(2017, 6, 8),),
            "lag_14": (date(2017, 6, 1),),
            "media_7": (date(2017, 6, 14), date(2017, 6, 8)),
            "media_14": tuple(date(2017, 6, d) for d in (14, 8, 5, 1)),
        })

    def test_cero_explicito_valido_en_media_catorce(self):
        datos = observaciones()
        datos.loc[:13, "sales"] = 0
        r = self.construir(datos)
        self.assertEqual(len(r.tabla), 2)
        self.assertEqual(r.tabla.iloc[0]["media_14"], 0)
        self.assertEqual(r.tabla.iloc[0]["lag_1"], 0)

    def test_fracciones_conservadas_en_media_catorce_y_respuesta(self):
        r = self.construir(observaciones(escala=0.125))
        self.assertEqual(r.tabla.iloc[0]["media_14"], 0.9375)
        self.assertEqual(r.tabla.iloc[0]["sales"], 1.875)

    def test_media_catorce_evade_desbordamiento_de_suma_directa(self):
        datos = observaciones()
        datos["sales"] = 1e308
        x = self.futuras(datos)
        self.assertTrue(isfinite(x.iloc[0]["media_14"]))
        self.assertAlmostEqual(x.iloc[0]["media_14"] / 1e308, 1.0)

    def test_rechaza_ventas_y_predicciones_no_finitas(self):
        for valor in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(valor=valor):
                datos = observaciones()
                datos.loc[13, "sales"] = valor
                with self.assertRaises(ValueError):
                    self.construir(datos)
                with self.assertRaises(ValueError):
                    self.futuras(objetivo=CORTE + timedelta(days=2),
                                 previas={CORTE + timedelta(days=1): valor})

    def test_una_fila_sin_historial_conserva_esquema_e_identifica_fechas(self):
        r = self.construir(observaciones().iloc[:1])
        self.assertEqual(r.observaciones_historicas, 1)
        self.assertTrue(r.tabla.empty)
        self.assertEqual(tuple(r.tabla.columns), self.esquema)
        self.assertEqual(len(r.descartes), 1)
        self.assertEqual(r.descartes[0].faltantes["media_14"],
                         tuple(date(2017, 6, 1) - timedelta(days=d) for d in range(1, 15)))

    def test_tabla_sin_filas_aptas_conserva_siete_predictoras(self):
        r = self.construir(observaciones().iloc[:14])
        self.assertTrue(r.tabla.empty)
        self.assertEqual(tuple(r.tabla.columns), self.esquema)
        self.assertEqual(len(r.tabla.loc[:, list(self.columnas)].columns), 7)
        self.assertEqual(len(r.descartes), 14)

    def test_conteos_por_variable_y_contabilidad_sin_duplicacion(self):
        r = self.construir()
        self.assertEqual(r.descartes_por_variable, {
            "lag_1": 1, "lag_7": 7, "lag_14": 14, "media_7": 7,
            "dia_semana": 0, "mes": 0, "media_14": 14})
        self.assertEqual(r.observaciones_historicas, len(r.tabla) + len(r.descartes))
        self.assertEqual(len({d.fecha_objetivo for d in r.descartes}), len(r.descartes))
        self.assertGreater(sum(r.descartes_por_variable.values()), len(r.descartes))

    def test_entradas_y_predicciones_previas_no_se_mutan(self):
        datos = observaciones().iloc[::-1].copy(deep=True)
        copia = datos.copy(deep=True)
        previas = {date(2017, 6, 17): 100.25}
        copia_previas = dict(previas)
        self.construir(datos)
        construir_tablas(datos, CORTE, ((1, "A"),), columnas_predictoras=self.columnas)
        self.futuras(datos, date(2017, 6, 18), previas)
        pd.testing.assert_frame_equal(datos, copia)
        self.assertEqual(previas, copia_previas)

    def test_entrenamiento_e_inferencia_coinciden_en_siete_columnas(self):
        tabla = self.construir().tabla
        for i, fila in tabla.iterrows():
            objetivo = fila["fecha_objetivo"]
            x = construir_variables_futuras(
                observaciones(), 1, "A", objetivo - timedelta(days=1), objetivo,
                columnas_predictoras=self.columnas)
            pd.testing.assert_frame_equal(
                tabla.loc[[i], list(self.columnas)].reset_index(drop=True), x)

    def test_quince_pasos_con_predicciones_explicitas_y_sin_ventas_del_horizonte(self):
        futuro = pd.DataFrame({
            "date": [(CORTE + timedelta(days=p)).isoformat() for p in range(1, 16)],
            "store_nbr": 1, "family": "A", "sales": "NO_LEER", "onpromotion": "NO_LEER",
        })
        datos = pd.concat([observaciones(), futuro], ignore_index=True)
        previas = {}
        referencia = {date(2017, 6, d): float(d) for d in range(1, 17)}
        for paso in range(1, 16):
            objetivo = CORTE + timedelta(days=paso)
            x = self.futuras(datos, objetivo, previas)
            pd.testing.assert_frame_equal(x, self.futuras(objetivo=objetivo, previas=previas))
            fila = x.iloc[0]
            for k in (1, 7, 14):
                self.assertEqual(fila[f"lag_{k}"], referencia[objetivo - timedelta(days=k)])
            self.assertAlmostEqual(fila["media_14"], fsum(
                referencia[objetivo - timedelta(days=k)] / 14 for k in range(1, 15)))
            prediccion = paso * 100 + 0.25  # Valores suministrados; ningún estimador.
            previas[objetivo] = prediccion
            referencia[objetivo] = prediccion
        self.assertEqual(len(previas), 15)
        self.assertEqual(fila["lag_1"], 1400.25)
        self.assertEqual(fila["lag_7"], 800.25)
        self.assertEqual(fila["lag_14"], 100.25)
        self.assertAlmostEqual(fila["media_7"], 1100.25)
        self.assertAlmostEqual(fila["media_14"], 750.25)
        with self.assertRaises(HistorialInsuficiente) as contexto:
            self.futuras(datos, CORTE + timedelta(days=15))
        self.assertEqual(len(contexto.exception.faltantes["media_14"]), 14)

    def test_rechaza_predicciones_en_corte_objetivo_y_posteriores(self):
        objetivo = CORTE + timedelta(days=2)
        for fecha in (CORTE - timedelta(days=1), CORTE, objetivo, objetivo + timedelta(days=1)):
            with self.subTest(fecha=fecha), self.assertRaises(ValueError):
                self.futuras(objetivo=objetivo, previas={fecha: 100.25})
        for valor in (-1, True, "invalida"):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                self.futuras(objetivo=objetivo, previas={CORTE + timedelta(days=1): valor})

    def test_rechaza_grupos_no_canonicos_antes_de_leer_historico(self):
        invalidas = (None, 7, "media_14", set(self.columnas), {c: 1 for c in self.columnas},
                     (), COLUMNAS_PREDICTORAS[:-1], tuple(reversed(COLUMNAS_PREDICTORAS)),
                     tuple(reversed(self.columnas)), (*self.columnas, "media_14"),
                     (*COLUMNAS_PREDICTORAS, "onpromotion"),
                     (*COLUMNAS_PREDICTORAS, "lag_21"),
                     (*COLUMNAS_PREDICTORAS, "es_fin_de_semana"),
                     (*COLUMNAS_PREDICTORAS, "desconocida"), (*COLUMNAS_PREDICTORAS, 14))
        with patch("scripts.variables_predictoras._historico_serie") as historico, \
                patch("scripts.variables_predictoras.leer_configuracion") as configuracion:
            for columnas in invalidas:
                with self.subTest(columnas=columnas):
                    with self.assertRaisesRegex(ValueError, "columnas_predictoras"):
                        construir_tabla_serie(None, 1, "A", CORTE, columnas_predictoras=columnas)
                    with self.assertRaisesRegex(ValueError, "columnas_predictoras"):
                        construir_tablas(None, CORTE, columnas_predictoras=columnas)
                    with self.assertRaisesRegex(ValueError, "columnas_predictoras"):
                        construir_variables_futuras(None, 1, "A", CORTE, CORTE + timedelta(days=1),
                                                    columnas_predictoras=columnas)
                    with self.assertRaisesRegex(ValueError, "columnas_predictoras"):
                        _variables_por_fecha(None, CORTE, columnas_predictoras=columnas)
            historico.assert_not_called()
            configuracion.assert_not_called()

    def test_lista_canonica_se_copia_sin_estado_mutable(self):
        columnas = list(self.columnas)
        r = construir_tabla_serie(observaciones(), 1, "A", CORTE, columnas_predictoras=columnas)
        columnas.pop()
        self.assertEqual(r.columnas_predictoras, self.columnas)
        self.assertEqual(tuple(r.tabla.columns), self.esquema)
        self.assertEqual(construir_tabla_serie(observaciones(), 1, "A", CORTE).columnas_predictoras,
                         COLUMNAS_PREDICTORAS)

    def test_onpromotion_no_modifica_x_alternativa(self):
        datos = observaciones()
        datos["onpromotion"] = 9999
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(datos).tabla)
        pd.testing.assert_frame_equal(self.futuras(), self.futuras(datos))
        self.assertNotIn("onpromotion", self.columnas)

    def test_csv_sintetico_diez_series_orden_corte_y_aislamiento(self):
        claves = tuple((i, "SINTETICA") for i in range(1, 11))
        datos = pd.concat([observaciones(t, f, escala=t) for t, f in claves], ignore_index=True)
        datos = datos.sort_values(["date", "store_nbr", "family"])
        copia = datos.copy(deep=True)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "ventas_sinteticas.csv"
            datos.to_csv(ruta, index=False)
            with ruta.open("a", encoding="utf-8") as archivo:
                archivo.write("2017-06-17,1,SINTETICA,NO_LEER,NO_LEER\n")
            historico = cargar_historico_csv(ruta, claves, CORTE, tamano_bloque=23)
            resultados = construir_tablas(historico, CORTE, claves, columnas_predictoras=self.columnas)
        self.assertEqual(len(resultados), 10)
        self.assertEqual(tuple((r.store_nbr, r.family) for r in resultados), claves)
        for r in resultados:
            self.assertEqual(r.fecha_corte, CORTE)
            self.assertEqual(r.observaciones_historicas, 16)
            self.assertEqual((len(r.tabla), len(r.descartes)), (2, 14))
            self.assertEqual(tuple(r.tabla.columns), self.esquema)
            self.assertAlmostEqual(r.tabla.iloc[0]["media_14"], 7.5 * r.store_nbr)
            self.assertEqual(r.tabla["fecha_objetivo"].tolist(), [date(2017, 6, 15), CORTE])
        pd.testing.assert_frame_equal(datos, copia)

    def test_no_cambia_corte_ni_procesa_ventas_de_prueba_reservada(self):
        # Fechas sintéticas posteriores: sus respuestas son deliberadamente inválidas.
        from scripts.analizar_periodo_seleccion import VENTANAS_POSTERIORES
        reservada = VENTANAS_POSTERIORES[-1]
        futuro = pd.DataFrame({"date": [reservada.inicio.isoformat(), reservada.fin.isoformat()],
                               "store_nbr": 1, "family": "A", "sales": "NO_LEER",
                               "onpromotion": "NO_LEER"})
        datos = pd.concat([observaciones(), futuro], ignore_index=True)
        pd.testing.assert_frame_equal(self.construir().tabla, self.construir(datos).tabla)
        pd.testing.assert_frame_equal(self.futuras(), self.futuras(datos))
        self.assertEqual(self.construir(datos).fecha_corte, CORTE)

    def test_nucleo_sin_lectura_de_archivos_ni_sqlite(self):
        datos = observaciones()
        with patch("builtins.open", side_effect=AssertionError("Lectura indebida")), \
                patch("io.open", side_effect=AssertionError("Lectura indebida")), \
                patch("pandas.read_csv", side_effect=AssertionError("CSV indebido")), \
                patch("sqlite3.connect", side_effect=AssertionError("SQLite indebido")):
            self.construir(datos)
            self.futuras(datos)
            construir_tablas(datos, CORTE, ((1, "A"),), columnas_predictoras=self.columnas)

    def test_comparacion_directa_misma_serie_y_corte(self):
        datos = observaciones().drop(index=4)
        clasico = construir_tabla_serie(datos, 1, "A", CORTE)
        alternativo = self.construir(datos)
        self.assertEqual((clasico.observaciones_historicas, alternativo.observaciones_historicas),
                         (15, 15))
        self.assertEqual((len(clasico.tabla), len(alternativo.tabla)), (2, 0))
        self.assertEqual((len(clasico.descartes), len(alternativo.descartes)), (13, 15))
        motivos = {"lag_1": 2, "lag_7": 7, "lag_14": 13,
                   "media_7": 11, "dia_semana": 0, "mes": 0}
        self.assertEqual(clasico.descartes_por_variable, motivos)
        self.assertEqual(alternativo.descartes_por_variable, {**motivos, "media_14": 15})
        for nombre, r in (("clásico", clasico), ("alternativo", alternativo)):
            print(f"\nComparación sintética {nombre} | corte {r.fecha_corte}: "
                  f"históricas={r.observaciones_historicas}, aptas={len(r.tabla)}, "
                  f"descartadas={len(r.descartes)}, motivos={r.descartes_por_variable}")
        print("Los motivos pueden solaparse; más variables no demuestra mejor predicción.")


if __name__ == "__main__":
    unittest.main()
