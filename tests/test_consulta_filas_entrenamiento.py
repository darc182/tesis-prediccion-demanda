"""Preparación real sobre SQLite temporal: disponibilidad, no entrenamiento."""

from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backend.app.services import consulta_filas_entrenamiento as consulta
from backend.app.storage.db import conectar, inicializar_base
from scripts.variables_predictoras import (
    COLUMNAS_PREDICTORAS, COLUMNAS_PREDICTORAS_ALTERNATIVAS, construir_tabla_serie,
)


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")


class DatosSinteticos:
    """Fixture compartida sin heredar ni duplicar casos de unittest."""

    def preparar_base(self):
        temporal = tempfile.TemporaryDirectory(prefix="tesis-filas-")
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "sintetica.sqlite3"
        inicializar_base(self.ruta)
        with closing(conectar(self.ruta)) as c, c:
            c.executemany(
                "INSERT INTO conjunto_datos (id,nombre_archivo,hash,fecha_carga,activo) VALUES (?,?,?,?,?)",
                [(2, "anterior_sintetico.csv", "h2", "2026-01-01", 0),
                 (7, "actual_sintetico.csv", "h7", "2026-01-02", 1)])
            c.executemany(
                "INSERT INTO serie_temporal (id,conjunto_id,store_nbr,family,criterio_seleccion) VALUES (?,?,?,?,?)",
                [(24, 2, 17, "PREPARED FOODS", "sintética"),
                 (83, 7, 17, "PREPARED FOODS", "sintética"),
                 (84, 7, 8, "BREAD/BAKERY", "sintética")])
            inicio = date(2017, 5, 1)
            c.executemany(
                "INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                [(83, (inicio + timedelta(days=i)).isoformat(), 0 if i % 9 == 0 else i / 8, i % 3)
                 for i in range(92) if inicio + timedelta(days=i) != date(2017, 6, 6)]
                + [(24, "2017-06-16", 9999.25, 0)])

    def ejecutar(self, sql, parametros=()):
        with closing(conectar(self.ruta)) as c, c:
            return c.execute(sql, parametros).fetchall()

    def directa(self, corte="2017-06-16", grupo="clasico"):
        # Las tres lecturas de referencia también comparten una instantánea.
        with closing(sqlite3.connect(f"{self.ruta.as_uri()}?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            conjunto = c.execute("SELECT id FROM conjunto_datos WHERE activo=1 ORDER BY id DESC LIMIT 1").fetchone()[0]
            tienda, familia = c.execute(
                "SELECT store_nbr,family FROM serie_temporal WHERE id=83 AND conjunto_id=?", (conjunto,)).fetchone()
            filas = c.execute("SELECT fecha,sales,onpromotion FROM venta_diaria WHERE serie_id=83 AND fecha<=? ORDER BY fecha", (corte,)).fetchall()
        datos = pd.DataFrame(filas, columns=["date", "sales", "onpromotion"])
        datos["store_nbr"], datos["family"] = tienda, familia
        columnas = COLUMNAS_PREDICTORAS if grupo == "clasico" else COLUMNAS_PREDICTORAS_ALTERNATIVAS
        return construir_tabla_serie(datos, tienda, familia, date.fromisoformat(corte), columnas_predictoras=columnas)

    def comprobar_conteos(self, respuesta, directo):
        self.assertEqual(respuesta["observaciones_historicas"], directo.observaciones_historicas)
        self.assertEqual(respuesta["filas_aptas"], len(directo.tabla))
        self.assertEqual(respuesta["filas_descartadas"], len(directo.descartes))
        self.assertEqual(respuesta["descartes_por_variable"], directo.descartes_por_variable)
        self.assertEqual(respuesta["filas_aptas"] + respuesta["filas_descartadas"], respuesta["observaciones_historicas"])


class ConsultaFilasEntrenamientoTests(DatosSinteticos, unittest.TestCase):
    def setUp(self):
        self.preparar_base()

    def consultar(self, serie=83, corte="2017-06-16", grupo="clasico"):
        return consulta.consultar_filas_entrenamiento(self.ruta, serie, corte, grupo)

    def error(self, codigo, **kwargs):
        with self.assertRaises(consulta.ErrorConsultaFilas) as contexto:
            self.consultar(**kwargs)
        self.assertEqual(contexto.exception.codigo, codigo)

    def test_ambos_grupos_tres_cortes_coinciden_con_preparacion_real(self):
        for grupo in ("clasico", "ampliado"):
            for corte in CORTES:
                with self.subTest(grupo=grupo, corte=corte):
                    r = self.consultar(corte=corte, grupo=grupo)
                    self.comprobar_conteos(r, self.directa(corte, grupo))
                    self.assertEqual(r["conjunto_id"], 7)
                    self.assertEqual(r["serie"], {"id": 83, "store_nbr": 17, "family": "PREPARED FOODS"})
                    self.assertEqual((r["fecha_corte"], r["grupo"]), (corte, grupo))
                    version = 1 if grupo == "clasico" else 2
                    self.assertEqual((r["version_contrato"], r["version_reglas"]), (version, version))
                    self.assertEqual(r["variables"], list(self.directa(corte, grupo).columnas_predictoras))

    def test_ausencia_t_menos_diez_afecta_media_14_sin_crear_candidatos(self):
        clasico, ampliado = self.directa(), self.directa(grupo="ampliado")
        self.assertIn(date(2017, 6, 16), set(clasico.tabla["fecha_objetivo"]))
        self.assertNotIn(date(2017, 6, 16), set(ampliado.tabla["fecha_objetivo"]))
        for grupo in ("clasico", "ampliado"):
            r = self.consultar(grupo=grupo)
            self.assertEqual(r["observaciones_historicas"], 46)  # 47 días, uno sin registro.
            self.comprobar_conteos(r, self.directa(grupo=grupo))
        self.assertLess(self.consultar(grupo="ampliado")["filas_aptas"], self.consultar()["filas_aptas"])

    def test_ceros_y_fracciones_llegan_intactos_a_la_preparacion(self):
        def observar(datos, *args, **kwargs):
            self.assertTrue(datos["sales"].eq(0).any())
            self.assertTrue(datos["sales"].mod(1).ne(0).any())
            self.assertFalse(datos["date"].eq("2017-06-06").any())
            copia = datos.copy(deep=True)
            r = construir_tabla_serie(datos, *args, **kwargs)
            pd.testing.assert_frame_equal(datos, copia)
            self.assertTrue(r.tabla["sales"].eq(0).any())
            self.assertTrue(r.tabla["sales"].mod(1).ne(0).any())
            return r
        with patch.object(consulta, "construir_tabla_serie", side_effect=observar) as preparar:
            self.consultar()
            self.consultar(grupo="ampliado")
        self.assertEqual(preparar.call_count, 2)

    def test_ventas_futuras_incluso_invalidas_no_se_leen_para_preparar(self):
        previos = {g: self.consultar(grupo=g) for g in ("clasico", "ampliado")}
        self.ejecutar("UPDATE venta_diaria SET sales='FUTURO_NO_UTILIZABLE' WHERE serie_id=83 AND fecha>'2017-06-16'")
        for grupo, previo in previos.items():
            self.assertEqual(self.consultar(grupo=grupo), previo)

    def test_recorte_sql_e_inclusion_del_corte_en_la_preparacion(self):
        def observar(datos, *args, **kwargs):
            self.assertEqual(datos["date"].max(), "2017-06-16")
            self.assertTrue(datos["date"].le("2017-06-16").all())
            return construir_tabla_serie(datos, *args, **kwargs)
        with patch.object(consulta, "construir_tabla_serie", side_effect=observar):
            self.consultar()

    def test_motivos_solapados_no_se_suman_para_descartes(self):
        for grupo in ("clasico", "ampliado"):
            r = self.consultar(grupo=grupo)
            self.assertGreater(sum(r["descartes_por_variable"].values()), r["filas_descartadas"])
            self.assertEqual(r["descartes_por_variable"]["dia_semana"], 0)
            self.assertEqual(r["descartes_por_variable"]["mes"], 0)
            self.comprobar_conteos(r, self.directa(grupo=grupo))

    def test_una_observacion_devuelve_cero_aptas_y_un_descarte(self):
        self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha<>'2017-06-16'")
        for grupo in ("clasico", "ampliado"):
            r = self.consultar(grupo=grupo)
            self.assertEqual((r["observaciones_historicas"], r["filas_aptas"], r["filas_descartadas"]), (1, 0, 1))
            self.comprobar_conteos(r, self.directa(grupo=grupo))

    def test_catorce_observaciones_son_historial_pero_no_filas_aptas(self):
        self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=83 AND (fecha<'2017-07-03' OR fecha>'2017-07-16')")
        for grupo in ("clasico", "ampliado"):
            r = self.consultar(corte="2017-07-16", grupo=grupo)
            self.assertEqual((r["observaciones_historicas"], r["filas_aptas"], r["filas_descartadas"]), (14, 0, 14))

    def test_sin_historial_y_solo_futuro_son_errores_controlados(self):
        self.error("sin_historial", serie=84)
        self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha<='2017-06-16'")
        self.error("sin_historial")

    def test_serie_inexistente_e_inactiva_no_se_consultan(self):
        for serie in (999, 24):
            with self.subTest(serie=serie):
                self.error("serie_inexistente", serie=serie)

    def test_sin_conjunto_activo(self):
        self.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error("sin_conjunto_activo")

    def test_no_asume_id_fijo_y_elige_activo_mas_reciente(self):
        self.ejecutar("UPDATE conjunto_datos SET activo=1 WHERE id=2")
        self.assertEqual(self.consultar()["conjunto_id"], 7)
        self.error("serie_inexistente", serie=24)
        self.ejecutar("UPDATE conjunto_datos SET activo=0 WHERE id=7")
        r = self.consultar(serie=24)
        self.assertEqual((r["conjunto_id"], r["observaciones_historicas"]), (2, 1))

    def test_identificadores_invalidos_antes_de_abrir_sqlite(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for valor in (0, -1, True, 1.5, "83", None, 2**63):
                self.error("serie_invalida", serie=valor)
            abrir.assert_not_called()

    def test_grupos_invalidos_antes_de_abrir_sqlite(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for valor in ("", "Clasico", "alternativo", "estacional_semanal_recursiva", "onpromotion", None, [], {}):
                self.error("grupo_invalido", grupo=valor)
            abrir.assert_not_called()

    def test_fechas_invalidas_antes_de_abrir_sqlite(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for valor in ("20170616", "2017-6-16", "2017-02-29", "2017-13-01", "2017-06-16T00:00:00", "", True, None, datetime(2017, 6, 16)):
                self.error("corte_invalido", corte=valor)
            abrir.assert_not_called()
        self.assertEqual(self.consultar(corte=date(2017, 6, 16)), self.consultar())

    def test_cortes_no_autorizados(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for corte in ("2017-06-15", "2017-07-02", "2017-07-15", "2017-08-16", "2018-06-16"):
                self.error("corte_no_autorizado", corte=corte)
            abrir.assert_not_called()

    def test_reserva_explicita_del_horizonte_final(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for corte in ("2017-07-17", "2017-07-31", "2017-08-01", "2017-08-15"):
                self.error("prueba_reservada", corte=corte)
            abrir.assert_not_called()

    def test_base_inexistente_no_se_crea(self):
        ruta = self.ruta.parent / "inexistente.sqlite3"
        with self.assertRaises(consulta.ErrorConsultaFilas) as error:
            consulta.consultar_filas_entrenamiento(ruta, 83, CORTES[0], "clasico")
        self.assertEqual(error.exception.codigo, "base_no_disponible")
        self.assertFalse(ruta.exists())

    def test_base_sin_esquema_no_se_inicializa(self):
        ruta = self.ruta.parent / "vacia.sqlite3"
        ruta.touch()
        with self.assertRaises(consulta.ErrorConsultaFilas) as error:
            consulta.consultar_filas_entrenamiento(ruta, 83, CORTES[0], "clasico")
        self.assertEqual(error.exception.codigo, "base_no_disponible")
        self.assertEqual(ruta.read_bytes(), b"")

    def test_observaciones_invalidas_no_se_imputan_ni_ocultan(self):
        for valor in (-1, float("inf"), "PRIVADO_NO_NUMERICO"):
            with self.subTest(valor=valor):
                self.ejecutar("UPDATE venta_diaria SET sales=? WHERE serie_id=83 AND fecha='2017-06-16'", (valor,))
                self.error("historial_invalido")

    def test_mode_ro_begin_filtro_sql_cierre_y_ausencia_total_de_escrituras(self):
        abrir_real = sqlite3.connect
        sentencias, conexiones, cambios = [], [], []
        escrituras = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                      sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE,
                      sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_ATTACH}
        def abrir(ruta_uri, **kwargs):
            self.assertTrue(ruta_uri.endswith("?mode=ro"))
            self.assertTrue(kwargs["uri"])
            c = abrir_real(ruta_uri, **kwargs)
            c.set_trace_callback(sentencias.append)
            def autorizar(accion, *args):
                if accion in escrituras:
                    cambios.append(accion)
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            c.set_authorizer(autorizar)
            conexiones.append(c)
            return c
        contenido = self.ruta.read_bytes()
        with patch.object(consulta.sqlite3, "connect", side_effect=abrir):
            for grupo in ("clasico", "ampliado"):
                for corte in CORTES:
                    self.consultar(corte=corte, grupo=grupo)
        self.assertEqual(cambios, [])
        self.assertEqual(sum(s == "BEGIN" for s in sentencias), 6)
        self.assertTrue(any("fecha <= '2017-06-16'" in s and "serie_id = 83" in s for s in sentencias))
        self.assertFalse(any(s.lstrip().upper().startswith(("PRAGMA", "INSERT", "UPDATE", "DELETE", "CREATE", "COMMIT", "BEGIN IMMEDIATE")) for s in sentencias))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                c.execute("SELECT 1")

    def test_instantanea_coherente_aunque_otro_escritor_cambie_activo_e_historico(self):
        # Solo el escritor de esta prueba modifica la base temporal, en WAL.
        with closing(sqlite3.connect(self.ruta)) as c:
            self.assertEqual(c.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        esperado = self.consultar()
        abrir_real = sqlite3.connect
        cambios = []
        def abrir(ruta_uri, **kwargs):
            c = abrir_real(ruta_uri, **kwargs)
            def observar(sql):
                if "FROM serie_temporal" in sql and not cambios:
                    cambios.append(True)
                    with closing(abrir_real(self.ruta)) as escritor, escritor:
                        escritor.execute("UPDATE conjunto_datos SET activo=0 WHERE id=7")
                        escritor.execute("UPDATE conjunto_datos SET activo=1 WHERE id=2")
                        escritor.execute("DELETE FROM venta_diaria WHERE serie_id=83")
            c.set_trace_callback(observar)
            return c
        with patch.object(consulta.sqlite3, "connect", side_effect=abrir):
            self.assertEqual(self.consultar(), esperado)
        self.assertEqual(cambios, [True])
        self.error("serie_inexistente")  # Una nueva consulta ve el nuevo activo.


if __name__ == "__main__":
    unittest.main()
