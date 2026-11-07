"""Perfil G1 exclusivamente con CSV sintéticos temporales; sin datos reales."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import date
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backend.app.services.validacion_csv import validar_csv
from scripts import perfil_global_csv as perfil


FILAS = [
    ["2016-12-24", "2", "B", "0", "0"],
    ["2016-12-26", "1", "A", "2.5", "2"],
    ["2017-06-16", "1", "B", "1.25", "0"],
    ["2017-06-17", "2", "A", "3.25", "3"],
    ["2017-06-17", "1", "A", "4", "0"],
]


class PerfilGlobalCSVTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory(prefix="perfil_g1_")
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "sintetico.csv"
        self.escribir(FILAS)

    def escribir(self, filas, columnas=perfil.COLUMNAS):
        with self.ruta.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(columnas)
            escritor.writerows(filas)

    def analizar(self, tamano=2):
        return perfil.perfilar_csv_global(self.ruta, tamano)

    def test_archivo_valido_recuentos_y_ventas_exactos(self):
        reporte = self.analizar()
        self.assertEqual(reporte["volumen"], {
            "filas_examinadas": 5, "filas_validas": 5, "filas_invalidas": 0,
            "tiendas_distintas": 2, "familias_distintas": 2, "combinaciones_observadas": 4})
        self.assertEqual(reporte["ventas"], {
            "suma": 11.0, "promedio_por_registro": 2.2, "minimo": 0.0, "maximo": 4.0,
            "registros_cero": 1, "porcentaje_cero": 20.0, "registros_fraccionarios": 3,
            "porcentaje_fraccionarios": 60.0, "registros_positivos": 4})

    def test_columnas_extra_ignoradas_y_orden_independiente(self):
        original = self.analizar()
        self.escribir([["EXTRA NO NUMERICA", *reversed(f)] for f in FILAS], ["id", *reversed(perfil.COLUMNAS)])
        self.assertEqual(self.analizar(), original)

    def test_campos_sobrantes_rechazados_al_inicio_y_tras_cruzar_bloques(self):
        valida = ["2017-06-16", "1", "A", "2.5", "0"]
        casos = ([valida + ["SOBRA"]], [valida + ["SOBRA", "SOBRA_TAMBIEN"]],
                 [valida, valida + ["SOBRA"]], [valida] * 3 + [valida + ["SOBRA"]])
        for filas in casos:
            for tamano in (1, 2, 3, 100):
                with self.subTest(filas=len(filas), campos=len(filas[-1]), bloque=tamano):
                    self.escribir(filas)
                    contenido, mtime = self.ruta.read_bytes(), self.ruta.stat().st_mtime_ns
                    with patch.object(pd, "read_csv") as calculo:
                        with self.assertRaisesRegex(perfil.ErrorPerfilCSV, "más valores que cabeceras"):
                            self.analizar(tamano)
                    calculo.assert_not_called()  # no comienza el cálculo ni devuelve un perfil parcial
                    self.assertEqual(self.ruta.read_bytes(), contenido)
                    self.assertEqual(self.ruta.stat().st_mtime_ns, mtime)

    def test_id_declarado_no_permite_campos_fuera_de_la_cabecera(self):
        for tamano in (1, 2, 3, 100):
            with self.subTest(bloque=tamano):
                self.escribir([FILAS[0] + ["id_valido"], FILAS[1] + ["otro_id", "SOBRA"]],
                              (*perfil.COLUMNAS, "id"))
                contenido = self.ruta.read_bytes()
                with self.assertRaisesRegex(perfil.ErrorPerfilCSV, "más valores que cabeceras"):
                    self.analizar(tamano)
                self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_id_declarado_con_comas_y_saltos_citados_preserva_json_valido(self):
        esperado = json.dumps(self.analizar(), sort_keys=True, allow_nan=False)
        for columnas in ((*perfil.COLUMNAS, "id"), ("id", *reversed(perfil.COLUMNAS))):
            filas = [[*f, 'id, con "comillas"\ny salto'] if columnas[-1] == "id"
                     else ['id, con "comillas"\ny salto', *reversed(f)] for f in FILAS]
            self.escribir(filas, columnas)
            contenido, mtime = self.ruta.read_bytes(), self.ruta.stat().st_mtime_ns
            for tamano in (1, 2, 3, 100):
                with self.subTest(columnas=columnas, bloque=tamano):
                    self.assertEqual(json.dumps(self.analizar(tamano), sort_keys=True, allow_nan=False), esperado)
            self.assertEqual(self.ruta.read_bytes(), contenido)
            self.assertEqual(self.ruta.stat().st_mtime_ns, mtime)

    def test_cli_estructura_invalida_no_emite_json_parcial(self):
        self.escribir([FILAS[0], FILAS[1] + ["SOBRA"]])
        salida, resumen = io.StringIO(), io.StringIO()
        with redirect_stdout(salida), redirect_stderr(resumen):
            self.assertEqual(perfil.main([str(self.ruta), "--tamano-bloque", "1"]), 1)
        self.assertEqual(salida.getvalue(), "")
        self.assertIn("más valores que cabeceras", resumen.getvalue())

    def test_ausencias_globales_no_se_convierten_en_ventas_cero(self):
        self.escribir(FILAS[:2])
        reporte = self.analizar(1)
        temporal = reporte["temporalidad"]
        self.assertEqual((temporal["primera_fecha"], temporal["ultima_fecha"]), ("2016-12-24", "2016-12-26"))
        self.assertEqual((temporal["fechas_distintas"], temporal["dias_calendario_inclusivos"]), (2, 3))
        self.assertEqual(temporal["fechas_sin_registro_global"], ["2016-12-25"])
        self.assertEqual(temporal["filas_por_fecha"], [{"fecha": "2016-12-24", "filas": 1},
                                                     {"fecha": "2016-12-26", "filas": 1}])
        self.assertEqual(reporte["ventas"]["registros_cero"], 1)
        self.assertEqual(reporte["ventas"]["promedio_por_registro"], 1.25)
        self.assertEqual(reporte["volumen"]["filas_validas"], 2)

    def test_ausencia_individual_no_es_ausencia_global(self):
        filas = [FILAS[0], ["2016-12-25", "3", "C", "7", "0"], FILAS[1]]
        self.escribir(filas)
        self.assertEqual(self.analizar()["temporalidad"]["fechas_sin_registro_global"], [])

    def test_promociones_conteos_y_estadisticas(self):
        self.assertEqual(self.analizar()["promociones"], {
            "registros_cero": 3, "registros_positivos": 2, "porcentaje_con_promocion": 40.0,
            "suma_conteos_onpromotion": 5, "minimo": 0, "maximo": 3, "promedio_por_registro": 1.0})

    def test_agregados_por_anio_fecha_tienda_y_familia(self):
        reporte = self.analizar()
        self.assertEqual(reporte["temporalidad"]["filas_por_anio"], [{"anio": 2016, "filas": 2}, {"anio": 2017, "filas": 3}])
        self.assertEqual(reporte["temporalidad"]["filas_por_fecha"][-1], {"fecha": "2017-06-17", "filas": 2})
        self.assertEqual(reporte["distribuciones"]["por_tienda"], [{"store_nbr": 1, "filas": 3}, {"store_nbr": 2, "filas": 2}])
        self.assertEqual(reporte["distribuciones"]["por_familia"], [
            {"family": "A", "filas": 3, "ventas": 9.75}, {"family": "B", "filas": 2, "ventas": 1.25}])

    def test_empates_y_top_diez_deterministas_sin_seleccion_oficial(self):
        filas = [["2017-06-17", str(t), f, "2.5", "0"] for t in range(6, 0, -1) for f in ("B", "A")]
        self.escribir(filas)
        mayores = self.analizar(3)["distribuciones"]["combinaciones_mayor_volumen"]
        self.assertEqual([(g["store_nbr"], g["family"]) for g in mayores], [(t, f) for t in range(1, 6) for f in ("A", "B")])
        self.assertEqual(len(mayores), 10)
        self.assertEqual(self.analizar()["volumen"]["combinaciones_observadas"], 12)
        filas[0][3] = "100"
        self.escribir(filas)
        self.assertEqual(self.analizar()["distribuciones"]["combinaciones_mayor_volumen"][0]["store_nbr"], 6)

    def test_bloques_1_2_3_y_mayor_producen_mismo_reporte(self):
        esperado = self.analizar(1)
        for tamano in (2, 3, 100):
            with self.subTest(tamano=tamano):
                self.assertEqual(self.analizar(tamano), esperado)

    def test_desorden_cronologico_y_entre_bloques_permitido(self):
        esperado = self.analizar()
        self.escribir(list(reversed(FILAS)))
        self.assertEqual(self.analizar(1), esperado)

    def test_suma_compensada_con_fracciones_y_magnitudes_distintas(self):
        self.escribir([["2017-06-17", "1", "A", valor, "0"] for valor in ("1e16", "1", "1", "0.25", "0.25")])
        reportes = [self.analizar(b) for b in (1, 2, 3, 100)]
        self.assertTrue(all(r == reportes[0] for r in reportes))
        self.assertEqual(reportes[0]["ventas"]["suma"], float(10**16 + 2))
        self.escribir([["2017-06-17", "1", "A", "0.1", "0"] for _ in range(101)])
        # Tolerancia absoluta 1e-12: aritmética binaria de float64, sin redondeo del reporte.
        self.assertAlmostEqual(self.analizar()["ventas"]["suma"], 10.1, delta=1e-12)

    def test_archivo_vacio_sin_filas_o_solo_lineas_vacias(self):
        for contenido in ("", "\n\n", ",".join(perfil.COLUMNAS) + "\n"):
            with self.subTest(contenido=contenido):
                self.ruta.write_text(contenido, encoding="utf-8")
                with self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar()

    def test_columnas_obligatorias_ausentes_y_cabeceras_duplicadas(self):
        for columnas in (perfil.COLUMNAS[:-1], (*perfil.COLUMNAS, "sales")):
            self.escribir([], columnas)
            with self.assertRaises(perfil.ErrorPerfilCSV):
                self.analizar()

    def test_filas_truncadas_y_comillas_malformadas_rechazadas(self):
        for fila in ("2017-06-16,1,A,1", "2017-06-16,1", '"2017-06-16,1,A,1,0'):
            with self.subTest(fila=fila):
                self.ruta.write_text(",".join(perfil.COLUMNAS) + "\n" + fila + "\n", encoding="utf-8")
                with self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar(1)

    def test_fechas_invalidas_y_no_canonicas_rechazadas(self):
        for valor in ("2017-02-30", "2017-6-16", "2017/06/16", " 2017-06-16", "", "2017-06-16T00:00:00"):
            with self.subTest(valor=valor):
                self.escribir([[valor, "1", "A", "1", "0"]])
                with self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar()

    def test_ventas_negativas_no_finitas_y_no_representables(self):
        for valor in ("-1", "NaN", "Infinity", "-Infinity", "1e309", "1e-400", "x", "", "True"):
            with self.subTest(valor=valor):
                self.escribir([["2017-06-16", "1", "A", valor, "0"]])
                with self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar()

    def test_tipos_de_tienda_promocion_y_familia_invalidos(self):
        for indice, valores in ((1, ("0", "-1", "1.0", "True", "", str(2**63))),
                                (4, ("-1", "0.5", "NaN", "Infinity", "False", "", str(2**63))),
                                (2, ("", "  ", "A" * 513))):
            for valor in valores:
                with self.subTest(columna=indice, valor=valor):
                    fila = ["2017-06-16", "1", "A", "1", "0"]
                    fila[indice] = valor
                    self.escribir([fila])
                    with self.assertRaises(perfil.ErrorPerfilCSV):
                        self.analizar()

    def test_tamano_bloque_invalido_no_inicia_lectura(self):
        with patch.object(Path, "open", side_effect=AssertionError("No debe leer")):
            for valor in (0, -1, True, 1.5, "2", None, perfil.MAX_TAMANO_BLOQUE + 1):
                with self.subTest(valor=valor), self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar(valor)

    def test_archivo_ausente_ilegible_y_utf8_invalido(self):
        with self.assertRaises(perfil.ErrorPerfilCSV):
            perfil.perfilar_csv_global(self.ruta.with_name("ausente.csv"))
        with patch.object(Path, "open", side_effect=PermissionError("Privado")):
            with self.assertRaises(perfil.ErrorPerfilCSV):
                self.analizar()
        self.ruta.write_bytes(b"\xff\xfe\x80")
        with self.assertRaises(perfil.ErrorPerfilCSV):
            self.analizar()

    def test_no_escribe_csv_ni_crea_archivos_o_sqlite(self):
        contenido, mtime = self.ruta.read_bytes(), self.ruta.stat().st_mtime_ns
        archivos = set(self.ruta.parent.iterdir())
        with patch.object(sqlite3, "connect", side_effect=AssertionError("G1 no usa SQLite")):
            self.analizar()
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.ruta.stat().st_mtime_ns, mtime)
        self.assertEqual(set(self.ruta.parent.iterdir()), archivos)

    def test_importacion_no_lee_archivos_ni_invoca_analisis(self):
        nombre = "perfil_g1_importacion_aislada"
        spec = importlib.util.spec_from_file_location(nombre, Path(perfil.__file__))
        modulo = importlib.util.module_from_spec(spec)
        self.addCleanup(lambda: sys.modules.pop(nombre, None))
        sys.modules[nombre] = modulo
        with patch.object(Path, "open", side_effect=AssertionError("Importación lee")), \
                patch.object(pd, "read_csv", side_effect=AssertionError("Importación analiza")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("Importación conecta")):
            spec.loader.exec_module(modulo)
        self.assertTrue(callable(modulo.perfilar_csv_global))

    def test_reporte_json_sin_rutas_privadas_y_retrospectivo(self):
        reporte = self.analizar()
        self.assertEqual(set(reporte), {"fuente", "alcance", "interpretacion_temporal", "volumen", "temporalidad",
                                        "ventas", "promociones", "distribuciones", "advertencias"})
        texto = json.dumps(reporte, allow_nan=False)
        self.assertEqual(json.loads(texto), reporte)
        self.assertNotIn(str(self.ruta.parent), texto)
        self.assertEqual(reporte["fuente"]["nombre_archivo"], self.ruta.name)
        self.assertEqual(reporte["alcance"], "panorama_descriptivo_global")
        self.assertTrue(reporte["interpretacion_temporal"]["retrospectivo"])
        self.assertFalse(reporte["interpretacion_temporal"]["autorizado_para_seleccion_de_modelos"])
        self.assertGreater(date.fromisoformat(reporte["temporalidad"]["ultima_fecha"]), date(2017, 6, 16))

    def test_rechaza_suma_no_finita_entre_bloques_y_grupos(self):
        self.escribir([["2017-06-16", "1", "A", "1e308", "0"], ["2017-06-17", "2", "B", "1e308", "0"]])
        for tamano in (1, 2):
            with self.subTest(tamano=tamano), self.assertRaisesRegex(perfil.ErrorPerfilCSV, "Acumulado"):
                self.analizar(tamano)

    def test_cardinalidades_y_calendario_excesivos_rechazados(self):
        for constante in ("MAX_TIENDAS", "MAX_FAMILIAS", "MAX_COMBINACIONES", "MAX_FECHAS", "MAX_DIAS_CALENDARIO"):
            with self.subTest(limite=constante), patch.object(perfil, constante, 1):
                with self.assertRaises(perfil.ErrorPerfilCSV):
                    self.analizar(1)
        self.escribir([["0001-01-01", "1", "A", "1", "0"], ["9999-12-31", "1", "A", "1", "0"]])
        with self.assertRaisesRegex(perfil.ErrorPerfilCSV, "calendario"):
            self.analizar(1)

    def test_unicidad_no_verificada_se_declara_y_validador_independiente_detecta(self):
        self.escribir([FILAS[1], FILAS[1]])
        reporte = self.analizar()
        self.assertEqual(reporte["volumen"]["filas_validas"], 2)
        self.assertEqual(reporte["ventas"]["suma"], 5.0)
        self.assertIn("No se verifica unicidad", reporte["advertencias"][0])
        self.assertFalse(validar_csv(self.ruta).valido)

    def test_cero_negativo_y_familia_na_como_texto_no_ausente(self):
        self.escribir([["2017-06-16", "001", "NA", "-0.0", "00"]])
        reporte = self.analizar()
        self.assertEqual(reporte["ventas"]["suma"], 0.0)
        self.assertEqual(reporte["ventas"]["registros_cero"], 1)
        self.assertEqual(reporte["distribuciones"]["por_familia"][0]["family"], "NA")
        self.assertEqual(reporte["distribuciones"]["por_tienda"][0]["store_nbr"], 1)

    def test_error_tardio_no_devuelve_perfil_parcial(self):
        self.escribir([*FILAS, ["2017-06-18", "1", "A", "-1", "0"]])
        with self.assertRaisesRegex(perfil.ErrorPerfilCSV, "Registro 6"):
            self.analizar(1)

    def test_archivo_moderado_se_lee_por_bloques_sin_concatenar(self):
        cantidad, tamano = 20_003, 257
        self.escribir((["2017-06-17", str(i % 10 + 1), "A", "0.25", str(i % 3)] for i in range(cantidad)))
        leer = pd.read_csv
        largos = []

        @contextmanager
        def observar(*args, **kwargs):
            self.assertEqual(kwargs["chunksize"], tamano)
            self.assertEqual(set(kwargs["usecols"]), set(perfil.COLUMNAS))
            self.assertNotIn("nrows", kwargs)
            with leer(*args, **kwargs) as lector:
                def iterar():
                    for bloque in lector:
                        largos.append(len(bloque))
                        yield bloque
                yield iterar()

        with patch.object(pd, "read_csv", side_effect=observar) as llamada, \
                patch.object(pd, "concat", side_effect=AssertionError("No debe concatenar")):
            reporte = self.analizar(tamano)
        llamada.assert_called_once()
        self.assertEqual(sum(largos), cantidad)
        self.assertGreater(len(largos), 1)
        self.assertLessEqual(max(largos), tamano)
        self.assertEqual(reporte["volumen"]["filas_validas"], cantidad)
        self.assertEqual(reporte["ventas"]["suma"], cantidad / 4)

    def test_cli_exige_archivo_emite_json_y_no_guarda_resultados(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            perfil.main([])
        self.assertEqual(error.exception.code, 2)
        salida, resumen = io.StringIO(), io.StringIO()
        archivos = set(self.ruta.parent.iterdir())
        with redirect_stdout(salida), redirect_stderr(resumen):
            self.assertEqual(perfil.main([str(self.ruta), "--tamano-bloque", "2"]), 0)
        self.assertEqual(json.loads(salida.getvalue()), self.analizar(2))
        self.assertIn("5 registros", resumen.getvalue())
        self.assertEqual(set(self.ruta.parent.iterdir()), archivos)
        self.ruta.write_bytes(b"")
        salida, resumen = io.StringIO(), io.StringIO()
        with redirect_stdout(salida), redirect_stderr(resumen):
            self.assertEqual(perfil.main([str(self.ruta)]), 1)
        self.assertEqual(salida.getvalue(), "")
        self.assertIn("Perfil rechazado", resumen.getvalue())


if __name__ == "__main__":
    unittest.main()
