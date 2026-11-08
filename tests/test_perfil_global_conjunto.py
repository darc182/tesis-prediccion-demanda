"""G5-B1 con G1 y carga reales, exclusivamente CSV y SQLite temporales.

La fixture G4 aporta 88 filas/11 combinaciones/1.480 ventas globales frente a
80 filas/10 series/680 ventas importadas. No modifica fixtures anteriores.
Las corrupciones que evaden CHECK se inyectan solo en esta base temporal.
"""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

from backend.app.services import perfil_global_conjunto as servicio
from backend.app.api import datasets as api_datasets
from backend.app.storage.db import SCHEMA_PATH, conectar, inicializar_base
from scripts import perfil_global_csv as g1
from tests import test_integridad_exploracion as fixture


TABLAS_ANTERIORES = fixture.TABLAS


class PerfilGlobalConjuntoTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixture.IntegridadExploracionTests()
        self.addCleanup(self.datos.doCleanups)
        self.datos.setUp()
        self.ruta = self.datos.ruta
        # B2 carga un perfil obligatorio. Solo esta fixture lo retira para
        # representar deliberadamente un conjunto legado SIN perfil y probar
        # la inserción manual del servicio B1, sin desactivar G1 en producción.
        with closing(conectar(self.ruta)) as legado, legado:
            self.assertEqual(legado.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone(), (1,))
            legado.execute("DELETE FROM perfil_global_conjunto WHERE conjunto_id=?", (self.datos.conjunto,))
        copia = self.ruta.with_name("datos.csv")
        copia.write_bytes(self.datos.contenido)
        self.original = g1.perfilar_csv_global(copia, 3)
        self.nombre = self.datos.csv.name
        self.preparado = servicio.preparar_perfil_global(self.original, self.nombre)
        self.sha = hashlib.sha256(self.datos.contenido).hexdigest()
        self.conjunto = self.datos.conjunto
        self.c = conectar(self.ruta)
        self.addCleanup(self.c.close)
        self.c.execute("BEGIN")

    def guardar(self, **cambios):
        args = dict(conjunto_id=self.conjunto, preparado=self.preparado, sha256_csv=self.sha,
                    series_importadas=10, observaciones_importadas=80)
        args.update(cambios)
        servicio.guardar_perfil_global(self.c, **args)

    def recuperar(self):
        return servicio.recuperar_perfil_global(self.c, self.conjunto)

    def error(self, codigo, llamada):
        with self.assertRaises(servicio.ErrorPerfilGlobal) as captura:
            llamada()
        self.assertEqual(captura.exception.codigo, codigo)
        for secreto in (str(self.ruta), "SELECT", "INSERT", "Traceback", "datos.csv"):
            self.assertNotIn(secreto, str(captura.exception))

    def alterar_texto(self, texto, *, checksum=None):
        # Solo fixture: NaN/JSON malformado y exceso de tamaño vulneran CHECK.
        self.c.execute("PRAGMA ignore_check_constraints=ON")
        self.c.execute("UPDATE perfil_global_conjunto SET reporte_json=?,sha256_reporte=? WHERE conjunto_id=?",
                       (texto, checksum or hashlib.sha256(texto.encode("utf-8")).hexdigest(), self.conjunto))

    def test_esquema_tiene_nueve_tablas_y_nueva_fk_activa(self):
        tablas = {r[0] for r in self.c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tablas, set(TABLAS_ANTERIORES) | {"perfil_global_conjunto"})
        self.assertEqual(self.c.execute("PRAGMA foreign_keys").fetchone(), (1,))
        self.assertEqual(self.c.execute("PRAGMA foreign_key_list(perfil_global_conjunto)").fetchone()[2:5],
                         ("conjunto_datos", "conjunto_id", "id"))

    def test_identidad_pk_obligatoria_no_autogenerada(self):
        for clave in (None, 0, -1):
            with self.subTest(clave=clave), self.assertRaises(sqlite3.IntegrityError):
                self.c.execute("INSERT INTO perfil_global_conjunto VALUES (?,1,?,'{}',?,0,0)",
                               (clave, self.sha, "a" * 64))

    def test_referencia_inexistente_rechazada_por_sqlite(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.c.execute("INSERT INTO perfil_global_conjunto VALUES (999,1,?,'{}',?,0,0)", (self.sha, "a" * 64))

    def test_checks_hash_version_json_y_contadores(self):
        base = [self.conjunto, 1, self.sha, "{}", "a" * 64, 0, 0]
        for campo, valor in ((1, 0), (1, 1.5), (2, "a" * 63), (2, "z" * 64), (3, "{"),
                             (4, "b" * 65), (5, -1), (5, 0.5), (6, -1)):
            with self.subTest(campo=campo, valor=valor), self.assertRaises(sqlite3.IntegrityError):
                datos = base.copy()
                datos[campo] = valor
                self.c.execute("INSERT INTO perfil_global_conjunto VALUES (?,?,?,?,?,?,?)", datos)

    def test_solo_un_perfil_sin_reemplazar_registro_previo(self):
        self.guardar()
        antes = self.recuperar()
        self.error("perfil_invalido", self.guardar)
        self.assertEqual(self.recuperar(), antes)
        self.assertTrue(self.c.in_transaction)

    def test_dos_conjuntos_con_mismo_hash_tienen_perfiles_independientes(self):
        self.guardar()
        self.c.execute("INSERT INTO conjunto_datos VALUES (99,?,?,?,0)", (self.nombre, self.sha, "2026-01-01"))
        self.guardar(conjunto_id=99, series_importadas=0, observaciones_importadas=0)
        segundo = servicio.recuperar_perfil_global(self.c, 99)
        self.assertEqual(segundo["sha256_csv"], self.recuperar()["sha256_csv"])
        self.assertEqual(segundo["conjunto_id"], 99)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone(), (2,))

    def test_g1_real_recuperado_cifras_globales_y_subconjunto_distintos(self):
        self.guardar()
        r = self.recuperar()
        self.assertEqual((r["conjunto_id"], r["version_perfil"], r["sha256_csv"]), (self.conjunto, 1, self.sha))
        self.assertEqual((r["series_importadas"], r["observaciones_importadas"]), (10, 80))
        self.assertEqual((r["reporte"]["volumen"]["filas_examinadas"], r["reporte"]["ventas"]["suma"]), (88, 1480))
        self.assertEqual(self.c.execute("SELECT SUM(sales) FROM venta_diaria").fetchone(), (680,))
        esperado = deepcopy(self.original)
        esperado["fuente"]["nombre_archivo"] = self.nombre
        self.assertEqual(r["reporte"], esperado)
        json.dumps(r, allow_nan=False)

    def test_preparacion_no_muta_original_y_sustituye_nombre_temporal(self):
        antes = deepcopy(self.original)
        p = servicio.preparar_perfil_global(self.original, self.nombre)
        self.assertEqual(self.original, antes)
        self.assertEqual(self.original["fuente"]["nombre_archivo"], "datos.csv")
        self.assertEqual(json.loads(p.reporte_json)["fuente"]["nombre_archivo"], self.nombre)

    def test_nombres_inseguros_rechazados(self):
        for nombre in ("../ventas.csv", "C:\\datos\\ventas.csv", "/datos/a.csv", "a/b.csv", "CON.csv",
                       "COM1.extra.csv", "..csv", " .csv", "...csv", "ventas.csv.", "a\n.csv", "a.txt",
                       "ventas.csv ", "ventas\t.csv", "NUL.csv", "LPT9.extra.csv", "C:ventas.csv"):
            with self.subTest(nombre=nombre):
                self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(self.original, nombre))

    def test_nombres_base_aceptados_por_router_se_conservan_al_persistir(self):
        for nombre in ("ventas..csv", " ventas.csv", "ventas .csv", " .ventas..CSV", ".ventas.csv",
                       "ventas (copia).csv", "Ventas Perú.csv"):
            with self.subTest(nombre=nombre):
                self.assertEqual(api_datasets._nombre_seguro(nombre), nombre)
                antes = deepcopy(self.original)
                p = servicio.preparar_perfil_global(self.original, nombre)
                self.assertEqual(self.original, antes)
                self.assertEqual(json.loads(p.reporte_json)["fuente"]["nombre_archivo"], nombre)
                self.c.execute("SAVEPOINT nombre_publico")
                try:
                    self.c.execute("UPDATE conjunto_datos SET nombre_archivo=? WHERE id=?", (nombre, self.conjunto))
                    self.guardar(preparado=p)
                    self.assertEqual(self.recuperar()["reporte"]["fuente"]["nombre_archivo"], nombre)
                finally:
                    self.c.execute("ROLLBACK TO nombre_publico")
                    self.c.execute("RELEASE nombre_publico")

    def test_calendario_g1_de_102_anios_prepara_persiste_y_recupera(self):
        fechas = [date(2000, 12, 31)] + [date(anio, 1, 1) for anio in range(2001, 2102)]
        csv = self.ruta.with_name("datos.csv")
        contenido = ("date,store_nbr,family,sales,onpromotion\n" + "".join(
            f"{f},17,PREPARED FOODS,0.5,0\n" for f in fechas)).encode("utf-8")
        csv.write_bytes(contenido)
        reporte = g1.perfilar_csv_global(csv, 3)
        antes = deepcopy(reporte)
        self.assertEqual(reporte["temporalidad"]["dias_calendario_inclusivos"], 36526)
        self.assertEqual(len(reporte["temporalidad"]["filas_por_anio"]), 102)
        self.assertEqual(reporte["temporalidad"]["filas_por_anio"],
                         [{"anio": anio, "filas": 1} for anio in range(2000, 2102)])
        preparado = servicio.preparar_perfil_global(reporte, "centenario.csv")
        self.assertEqual(reporte, antes)
        sha = hashlib.sha256(contenido).hexdigest()
        self.c.execute("INSERT INTO conjunto_datos VALUES (99,'centenario.csv',?,'2026-01-01',0)", (sha,))
        serie = self.c.execute("INSERT INTO serie_temporal (conjunto_id,store_nbr,family,criterio_seleccion) "
                               "VALUES (99,17,'PREPARED FOODS','fixture centenaria sintética')").lastrowid
        self.c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,0.5,0)",
                           [(serie, f.isoformat()) for f in fechas])
        servicio.guardar_perfil_global(self.c, 99, preparado, sha, 1, 102)
        self.c.commit()
        with closing(sqlite3.connect(self.ruta.as_uri() + "?mode=ro", uri=True)) as lectura:
            lectura.execute("BEGIN")
            recuperado = servicio.recuperar_perfil_global(lectura, 99)
        esperado = deepcopy(reporte)
        esperado["fuente"]["nombre_archivo"] = "centenario.csv"
        self.assertEqual(recuperado["reporte"], esperado)
        self.assertEqual(recuperado["observaciones_importadas"], 102)
        self.assertEqual(recuperado["reporte"]["ventas"]["suma"], 51)
        self.assertEqual(self.c.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_limites_g1_y_coherencia_anual_siguen_rechazando_inconsistencias(self):
        copia = self.ruta.with_name("excesivo.csv")
        copia.write_text("date,store_nbr,family,sales,onpromotion\n"
                         "2000-01-01,17,PREPARED FOODS,0.5,0\n"
                         "2101-01-01,17,PREPARED FOODS,0.5,0\n", encoding="utf-8")
        with self.assertRaises(g1.ErrorPerfilCSV):
            g1.perfilar_csv_global(copia, 1)
        cambios = (
            lambda t: t.update(dias_calendario_inclusivos=36601),
            lambda t: t["filas_por_anio"].reverse(),
            lambda t: t["filas_por_anio"][1].update(anio=2016),
            lambda t: t["filas_por_anio"][0].update(filas=0),
            lambda t: t["filas_por_anio"][0].update(filas=23),
            lambda t: t["filas_por_fecha"][0].update(fecha="2016-12-23"),
        )
        for i, cambio in enumerate(cambios):
            with self.subTest(cambio=i):
                reporte = deepcopy(self.original)
                cambio(reporte["temporalidad"])
                self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(reporte, self.nombre))

    def test_nombre_distinto_del_conjunto_se_rechaza(self):
        otro = servicio.preparar_perfil_global(self.original, "otro.csv")
        self.error("perfil_invalido", lambda: self.guardar(preparado=otro))

    def test_hash_csv_invalido_o_distinto_del_conjunto(self):
        for sha in ("b" * 64, "A" * 64, "no-hash", True):
            with self.subTest(sha=sha):
                self.error("perfil_invalido", lambda: self.guardar(sha256_csv=sha))

    def test_serializacion_y_checksum_independientes_del_orden_de_claves(self):
        def invertir(v):
            if isinstance(v, dict):
                return {k: invertir(v[k]) for k in reversed(v)}
            return [invertir(x) for x in v] if isinstance(v, list) else v
        p = servicio.preparar_perfil_global(invertir(self.original), self.nombre)
        self.assertEqual(p, self.preparado)
        exacto = json.dumps(json.loads(p.reporte_json), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
        self.assertEqual(p.reporte_json, exacto)
        self.assertEqual(p.sha256_reporte, hashlib.sha256(exacto.encode("utf-8")).hexdigest())

    def test_modificacion_real_produce_checksum_distinto(self):
        copia = self.ruta.with_name("datos.csv")
        copia.write_bytes(self.datos.contenido.replace(b",1.5,0", b",1.75,0"))
        distinto = servicio.preparar_perfil_global(g1.perfilar_csv_global(copia, 3), self.nombre)
        self.assertNotEqual(distinto.sha256_reporte, self.preparado.sha256_reporte)
        self.assertEqual(json.loads(distinto.reporte_json)["ventas"]["suma"], 1482.5)

    def test_insercion_no_confirma_y_rollback_externo_elimina_perfil(self):
        self.guardar()
        self.assertTrue(self.c.in_transaction)
        with closing(conectar(self.ruta)) as otra:
            self.assertEqual(otra.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone(), (0,))
        self.c.rollback()
        self.c.execute("BEGIN")
        self.error("perfil_no_disponible", self.recuperar)

    def test_fallo_no_hace_rollback_de_otros_cambios_del_llamador(self):
        self.c.execute("INSERT INTO conjunto_datos VALUES (99,'previo.csv',?,'2026-01-01',0)", ("a" * 64,))
        self.error("perfil_invalido", lambda: self.guardar(observaciones_importadas=79))
        self.assertTrue(self.c.in_transaction)
        self.assertEqual(self.c.execute("SELECT id FROM conjunto_datos WHERE id=99").fetchone(), (99,))

    def test_commit_y_recuperacion_posterior_desde_conexion_ro(self):
        self.guardar()
        esperado = self.recuperar()
        self.c.commit()
        with closing(sqlite3.connect(self.ruta.as_uri() + "?mode=ro", uri=True)) as lectura:
            lectura.execute("BEGIN")
            self.assertEqual(servicio.recuperar_perfil_global(lectura, self.conjunto), esperado)

    def test_conexion_y_transaccion_son_obligatorias_sin_abrir_base(self):
        self.c.rollback()
        with patch.object(sqlite3, "connect", side_effect=AssertionError("abre base")):
            self.error("almacenamiento_no_disponible", self.guardar)
            self.error("almacenamiento_no_disponible", self.recuperar)
            self.error("almacenamiento_no_disponible", lambda: servicio.recuperar_perfil_global(None, 1))

    def test_insercion_exige_foreign_keys_activas(self):
        self.c.rollback()
        self.c.execute("PRAGMA foreign_keys=OFF")
        self.c.execute("BEGIN")
        self.error("almacenamiento_no_disponible", self.guardar)

    def test_lectura_sin_dml_ni_confirmacion_y_resultados_independientes(self):
        self.guardar()
        self.c.commit()
        antes = self.datos.estado()
        sentencias = []
        with closing(sqlite3.connect(self.ruta.as_uri() + "?mode=ro", uri=True)) as lectura:
            lectura.execute("BEGIN")
            lectura.set_trace_callback(sentencias.append)
            r = servicio.recuperar_perfil_global(lectura, self.conjunto)
            r["reporte"]["ventas"]["suma"] = 0
            self.assertEqual(servicio.recuperar_perfil_global(lectura, self.conjunto)["reporte"]["ventas"]["suma"], 1480)
            self.assertTrue(lectura.in_transaction)
        self.assertTrue(all(s.lstrip().upper().startswith("SELECT") for s in sentencias))
        self.assertEqual(self.datos.estado(), antes)

    def test_claves_duplicadas_incluso_iguales_en_cualquier_nivel(self):
        self.guardar()
        for buscar, reemplazo in (
            ('"alcance":', '"alcance":"panorama_descriptivo_global","alcance":'),
            ('"nombre_archivo":', '"nombre_archivo":"otro.csv","nombre_archivo":'),
            ('"suma":', '"suma":1480.0,"suma":'),
            ('"filas":', '"filas":999,"filas":'),
        ):
            with self.subTest(buscar=buscar):
                self.alterar_texto(self.preparado.reporte_json.replace(buscar, reemplazo, 1))
                self.error("perfil_corrupto", self.recuperar)

    def test_nan_infinity_y_desbordamiento_en_json_almacenado(self):
        self.guardar()
        for token in ("NaN", "Infinity", "-Infinity", "1e309"):
            with self.subTest(token=token):
                self.alterar_texto(self.preparado.reporte_json.replace('"suma":1480.0', '"suma":' + token))
                self.error("perfil_corrupto", self.recuperar)

    def test_no_finitos_y_tipos_inesperados_en_entrada(self):
        for valor in (float("nan"), float("inf"), float("-inf"), True, "1480", None):
            with self.subTest(valor=valor):
                r = deepcopy(self.original)
                r["ventas"]["suma"] = valor
                self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))
        r = deepcopy(self.original)
        r["advertencias"].append({"oculto": float("inf")})
        self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_enteros_fuera_de_rango_seguro_rechazados(self):
        r = deepcopy(self.original)
        r["volumen"]["filas_examinadas"] = 2**53
        self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))
        self.error("perfil_invalido", lambda: self.guardar(series_importadas=True))

    def test_version_desconocida_al_insertar_y_recuperar(self):
        self.error("version_no_soportada", lambda: self.guardar(preparado=replace(self.preparado, version_perfil=2)))
        self.guardar()
        self.c.execute("UPDATE perfil_global_conjunto SET version_perfil=2")
        self.error("version_no_soportada", self.recuperar)

    def test_checksum_corrupto_y_preparado_no_confiable(self):
        self.error("perfil_invalido", lambda: self.guardar(preparado=replace(self.preparado, sha256_reporte="b" * 64)))
        self.guardar()
        self.c.execute("UPDATE perfil_global_conjunto SET sha256_reporte=?", ("b" * 64,))
        self.error("perfil_corrupto", self.recuperar)

    def test_documento_truncado_con_checksum_actualizado(self):
        self.guardar()
        self.alterar_texto(self.preparado.reporte_json[:-3])
        self.error("perfil_corrupto", self.recuperar)

    def test_sobre_limite_de_bytes_sin_truncar_entrada(self):
        r = deepcopy(self.original)
        r["advertencias"] = ["x" * (servicio.MAX_BYTES_REPORTE + 1)]
        self.error("perfil_demasiado_grande", lambda: servicio.preparar_perfil_global(r, self.nombre))
        self.assertEqual(len(r["advertencias"][0]), servicio.MAX_BYTES_REPORTE + 1)

    def test_exactamente_16_mib_puede_guardarse_y_recuperarse(self):
        # Relleno sintético de una advertencia para probar el límite, no estadística real.
        r = deepcopy(self.original)
        r["advertencias"] = ["á"]
        p = servicio.preparar_perfil_global(r, self.nombre)
        faltan = servicio.MAX_BYTES_REPORTE - len(p.reporte_json.encode("utf-8"))
        r["advertencias"][0] += "x" * faltan
        p = servicio.preparar_perfil_global(r, self.nombre)
        self.assertEqual(len(p.reporte_json.encode("utf-8")), servicio.MAX_BYTES_REPORTE)
        self.guardar(preparado=p)
        self.assertEqual(self.recuperar()["reporte"]["advertencias"], r["advertencias"])
        r["advertencias"][0] += "x"
        self.error("perfil_demasiado_grande", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_lectura_controla_bytes_antes_de_interpretar(self):
        self.guardar()
        self.alterar_texto(" " * (servicio.MAX_BYTES_REPORTE + 1))
        with patch.object(servicio, "_leer_json", side_effect=AssertionError("interpreta documento enorme")):
            self.error("perfil_demasiado_grande", self.recuperar)

    def test_fechas_invalidas_y_ausencias_contradictorias(self):
        for fecha in ("2017-2-01", "2017-02-30", "0000-01-01"):
            r = deepcopy(self.original)
            r["temporalidad"]["primera_fecha"] = fecha
            self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))
        r = deepcopy(self.original)
        r["temporalidad"]["fechas_sin_registro_global"][0] = "2016-12-24"
        self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_contadores_porcentajes_y_promedios_inconsistentes(self):
        cambios = (("volumen", "filas_validas", 87), ("volumen", "filas_invalidas", 1),
                   ("ventas", "registros_positivos", 79), ("ventas", "porcentaje_cero", 20),
                   ("ventas", "promedio_por_registro", 100), ("promociones", "registros_cero", 0),
                   ("promociones", "suma_conteos_onpromotion", 1), ("promociones", "promedio_por_registro", 99))
        for seccion, clave, valor in cambios:
            with self.subTest(seccion=seccion, clave=clave):
                r = deepcopy(self.original)
                r[seccion][clave] = valor
                self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_distribuciones_contradictorias_en_todos_los_niveles(self):
        cambios = (("filas_por_fecha", "filas", 99), ("filas_por_anio", "filas", 99),
                   ("por_tienda", "filas", 99), ("por_familia", "filas", 99),
                   ("por_familia", "ventas", 9999), ("combinaciones_mayor_volumen", "ventas", 9999))
        for arreglo, clave, valor in cambios:
            with self.subTest(arreglo=arreglo):
                r = deepcopy(self.original)
                seccion = "temporalidad" if arreglo.startswith("filas_") else "distribuciones"
                r[seccion][arreglo][0][clave] = valor
                self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_redondeo_legitimo_admitido_sin_cambiar_valores(self):
        r = deepcopy(self.original)
        r["distribuciones"]["por_familia"][0]["ventas"] += 1e-13
        p = servicio.preparar_perfil_global(r, self.nombre)
        self.assertEqual(json.loads(p.reporte_json)["distribuciones"], r["distribuciones"])

    def test_g1_autentico_ceros_unico_fracciones_y_magnitudes_extremas(self):
        copia = self.ruta.with_name("casos.csv")
        for valores in ((0.0, 0.0), (0.25,), (1e-9, 2e-9), (1e307, 2e307), (0.1, 0.1, 0.1)):
            with self.subTest(valores=valores):
                texto = "date,store_nbr,family,sales,onpromotion\n" + "".join(
                    f"2017-06-{i+1:02d},1,A,{v},0\n" for i, v in enumerate(valores))
                copia.write_text(texto, encoding="utf-8")
                reporte = g1.perfilar_csv_global(copia, 1)
                p = servicio.preparar_perfil_global(reporte, self.nombre)
                self.assertEqual(json.loads(p.reporte_json)["ventas"], reporte["ventas"])
                self.assertEqual(json.loads(p.reporte_json)["promociones"]["suma_conteos_onpromotion"], 0)
                if valores[0] == 1e-9:
                    reporte["ventas"]["promedio_por_registro"] *= 1.01
                    self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(reporte, self.nombre))

    def test_top_completo_y_orden_de_listas_se_verifican(self):
        copia = self.ruta.with_name("solo_oficiales.csv")
        filas = self.datos.contenido.decode("utf-8").splitlines()
        copia.write_text("\n".join(f for f in filas if ",999," not in f) + "\n", encoding="utf-8")
        reporte = g1.perfilar_csv_global(copia, 3)
        servicio.validar_reporte_global(reporte)
        self.assertEqual(reporte["volumen"]["combinaciones_observadas"], 10)
        self.assertEqual(reporte["ventas"]["suma"], 680)
        reporte["distribuciones"]["combinaciones_mayor_volumen"].reverse()
        self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(reporte, self.nombre))

    def test_lectura_revalida_estadisticas_con_checksum_correcto(self):
        self.guardar()
        reporte = json.loads(self.preparado.reporte_json)
        reporte["distribuciones"]["por_familia"][0]["ventas"] += 100
        texto = json.dumps(reporte, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.alterar_texto(texto)
        self.error("perfil_corrupto", self.recuperar)

    def test_tamano_unicode_en_bytes_y_nombre_publico_unicode(self):
        p = servicio.preparar_perfil_global(self.original, "Ventas Perú.csv")
        self.assertIn("Perú", p.reporte_json)
        self.assertGreater(len(p.reporte_json.encode("utf-8")), len(p.reporte_json))
        self.c.execute("UPDATE conjunto_datos SET nombre_archivo='Ventas Perú.csv'")
        self.guardar(preparado=p)
        self.assertEqual(self.recuperar()["reporte"]["fuente"]["nombre_archivo"], "Ventas Perú.csv")

    def test_no_es_un_reporte_g1_o_campos_extra(self):
        for cambio in (lambda r: r.update(alcance="perfil_descriptivo_serie_hasta_corte"),
                       lambda r: r.update(ruta_modelo="C:/privado/modelo"),
                       lambda r: r["interpretacion_temporal"].update(autorizado_para_seleccion_de_modelos=True)):
            r = deepcopy(self.original)
            cambio(r)
            self.error("perfil_invalido", lambda: servicio.preparar_perfil_global(r, self.nombre))

    def test_nombre_hash_y_contadores_almacenados_se_reverifican(self):
        self.guardar()
        for sql, valores in (("UPDATE conjunto_datos SET nombre_archivo=?", ("otro.csv",)),
                             ("UPDATE conjunto_datos SET hash=?", ("a" * 64,)),
                             ("UPDATE perfil_global_conjunto SET series_importadas=?", (9,))):
            with self.subTest(sql=sql):
                self.c.execute("SAVEPOINT corrupto")
                self.c.execute(sql, valores)
                self.error("perfil_corrupto", self.recuperar)
                self.c.execute("ROLLBACK TO corrupto")
                self.c.execute("RELEASE corrupto")

    def test_perfil_de_conjunto_inactivo_conserva_identidad(self):
        self.guardar()
        antes = self.recuperar()
        self.c.execute("UPDATE conjunto_datos SET activo=0")
        self.c.execute("INSERT INTO conjunto_datos VALUES (99,'nuevo.csv',?,'2026-01-01',1)", ("a" * 64,))
        self.assertEqual(self.recuperar(), antes)

    def test_conjunto_sin_perfil_o_inexistente_no_inventa_reporte(self):
        self.error("perfil_no_disponible", self.recuperar)
        self.error("perfil_no_disponible", lambda: servicio.recuperar_perfil_global(self.c, 999))
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone(), (0,))

    def test_inicializacion_repetida_preserva_perfil_y_tablas_previas(self):
        self.guardar()
        self.c.commit()
        antes = self.datos.estado()
        inicializar_base(self.ruta)
        self.c.execute("BEGIN")
        self.assertEqual(self.recuperar()["sha256_csv"], self.sha)
        self.assertEqual(self.datos.estado(), antes)

    def test_esquema_anterior_ampliacion_explicita_sin_alterar_pronosticos(self):
        self.c.rollback()
        self.datos.crear_base()  # línea base real, solo sobre la SQLite sintética
        vieja = self.ruta.with_name("ocho_tablas.sqlite3")
        texto = SCHEMA_PATH.read_text(encoding="utf-8")
        with closing(conectar(vieja)) as c:
            c.executescript(texto.split("-- Identidad compartida")[0] + "COMMIT;")
            for tabla in TABLAS_ANTERIORES:
                filas = self.c.execute(f"SELECT * FROM {tabla}").fetchall()
                if filas:
                    c.executemany(f"INSERT INTO {tabla} VALUES ({','.join('?' for _ in filas[0])})", filas)
            c.commit()
            antes = tuple(tuple(c.execute(f"SELECT * FROM {t}")) for t in TABLAS_ANTERIORES)
            esquema = list(c.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"))
            self.assertEqual(len(esquema), 8)
            c.execute("BEGIN")
            self.error("perfil_no_disponible", lambda: servicio.recuperar_perfil_global(c, self.conjunto))
            c.rollback()
            inicializar_base(vieja)
            c.execute("BEGIN")
            self.assertEqual(tuple(tuple(c.execute(f"SELECT * FROM {t}")) for t in TABLAS_ANTERIORES), antes)
            actuales = dict(c.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"))
            self.assertEqual({k: actuales[k] for k, _ in esquema}, dict(esquema))
            self.assertEqual(len(actuales), 9)
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.error("perfil_no_disponible", lambda: servicio.recuperar_perfil_global(c, self.conjunto))

    def test_registro_manual_de_perfil_legado_no_modifica_pronosticos(self):
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM perfil_global_conjunto").fetchone(), (0,))
        self.c.rollback()
        self.datos.crear_base()
        antes = self.datos.estado()
        self.c.execute("BEGIN")
        self.guardar()
        self.recuperar()
        self.c.commit()
        self.assertEqual(self.datos.estado(), antes)

    def test_importacion_no_abre_csv_sqlite_ni_ejecuta_g1_o_modelos(self):
        nombre = "perfil_global_importacion_g5b1"
        spec = importlib.util.spec_from_file_location(nombre, servicio.__file__)
        modulo = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {nombre: modulo}), \
                patch.object(Path, "open", side_effect=AssertionError("abre CSV")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("abre SQLite")), \
                patch.object(g1, "perfilar_csv_global", side_effect=AssertionError("ejecuta G1")), \
                patch.object(fixture.modelos, "entrenar_modelo", side_effect=AssertionError("entrena")):
            spec.loader.exec_module(modulo)
        self.assertEqual(modulo.VERSION_PERFIL, 1)

    def test_errores_sqlite_se_convierten_en_mensajes_fijos(self):
        cerrada = conectar(self.ruta)
        cerrada.close()
        self.error("almacenamiento_no_disponible", lambda: servicio.recuperar_perfil_global(cerrada, 1))


if __name__ == "__main__":
    unittest.main()
