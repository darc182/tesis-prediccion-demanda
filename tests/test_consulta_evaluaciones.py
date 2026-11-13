"""HU005-A2: consultas reales a SQLite temporales con datos sintéticos."""

from contextlib import closing, contextmanager, ExitStack
from datetime import date, timedelta
import builtins
import io
import json
from math import sqrt
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import consulta_evaluaciones as servicio
from backend.app.services import consulta_pronosticos as pronosticos
from backend.app.services import metricas_evaluacion as metricas
from backend.app.storage.db import conectar, inicializar_base


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")
BASE = "estacional_semanal_recursiva"
ESTIMADAS = [8, 24, 30] * 5
REALES = [10, 20, 30] * 5


def fecha_objetivo(corte, paso):
    return (date.fromisoformat(corte) + timedelta(days=paso)).isoformat()


class ConsultaEvaluacionesTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        # También comprueba el escape correcto de caracteres al construir la URI.
        self.ruta = Path(temporal.name) / "sintética # evaluación.sqlite3"
        inicializar_base(self.ruta)
        with closing(conectar(self.ruta)) as c, c:
            c.executemany(
                "INSERT INTO conjunto_datos (id,nombre_archivo,hash,fecha_carga,activo) VALUES (?,?,?,?,?)",
                [(2, "anterior_sintetico.csv", "hash2", "2026-01-01", 0),
                 (7, "actual_sintetico.csv", "hash7", "2026-01-02", 1)])
            c.executemany(
                "INSERT INTO serie_temporal (id,conjunto_id,store_nbr,family,criterio_seleccion) VALUES (?,?,?,?,?)",
                [(24, 2, 17, "FAMILIA A", "sintética"), (83, 7, 17, "FAMILIA A", "sintética"),
                 (84, 7, 8, "FAMILIA B", "sintética")])
        self.crear_pronostico()
        self.insertar_reales()

    def ejecutar(self, sql, parametros=()):
        with closing(conectar(self.ruta)) as c, c:
            return c.execute(sql, parametros).fetchall()

    def corromper(self, sql, parametros=()):
        # Solo en la fixture temporal: permite valores que violan CHECK, sin
        # modificar el esquema ni sus restricciones de unicidad/NOT NULL.
        with closing(conectar(self.ruta)) as c, c:
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute(sql, parametros)

    def crear_pronostico(self, identificador=101, experimento=201, serie=83,
                         corte=CORTES[0], tecnica=BASE, tipo="validacion", estimadas=None):
        valores = ESTIMADAS if estimadas is None else estimadas
        with closing(conectar(self.ruta)) as c, c:
            c.execute(
                "INSERT INTO experimento (id,serie_id,tecnica,configuracion_json,"
                "inicio_entrenamiento,fin_entrenamiento,ruta_modelo) VALUES (?,?,?,?,?,?,?)",
                (experimento, serie, tecnica, '{"privado":"PRIVADO JSON"}',
                 "2017-01-01", corte, "PRIVADO C:/modelo"))
            c.execute(
                "INSERT INTO pronostico (id,experimento_id,fecha_corte,tipo,horizonte,fecha_generacion) "
                "VALUES (?,?,?,?,?,?)", (identificador, experimento, corte, tipo, 15, "2026-01-03T00:00:00"))
            c.executemany(
                "INSERT INTO detalle_pronostico (pronostico_id,paso,fecha_objetivo,venta_estimada) VALUES (?,?,?,?)",
                [(identificador, paso, fecha_objetivo(corte, paso), valor)
                 for paso, valor in enumerate(valores, 1)])

    def insertar_reales(self, serie=83, corte=CORTES[0], valores=None):
        valores = REALES if valores is None else valores
        with closing(conectar(self.ruta)) as c, c:
            # Inserción invertida; el servicio/A1 deben alinear por fecha.
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                          [(serie, fecha_objetivo(corte, paso), valor, 0)
                           for paso, valor in reversed(list(enumerate(valores, 1)))])

    def consultar(self, identificador=101):
        contenido = self.ruta.read_bytes()
        r = servicio.consultar_evaluacion(self.ruta, identificador)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        e = r["evaluacion"]
        self.assertEqual(e["observaciones_evaluadas"] + e["fechas_sin_registro"], 15)
        self.assertEqual((r["fecha_corte"], r["horizonte"]), (e["fecha_corte"], e["horizonte"]))
        self.assertEqual(len(e["detalles"]), 15)
        return r

    def error(self, codigo, identificador=101, ruta=None):
        ruta = self.ruta if ruta is None else ruta
        contenido = ruta.read_bytes() if isinstance(ruta, Path) and ruta.is_file() else None
        with self.assertRaises(servicio.ErrorConsultaEvaluacion) as error:
            servicio.consultar_evaluacion(ruta, identificador)
        self.assertEqual(error.exception.codigo, codigo)
        for privado in ("SELECT", "PRIVADO", "Traceback", str(self.ruta)):
            self.assertNotIn(privado, str(error.exception))
        if contenido is not None:
            self.assertEqual(ruta.read_bytes(), contenido)

    @contextmanager
    def lectura_vigilada(self, alterar_detalles=None):
        abrir_real = sqlite3.connect
        llamadas, conexiones, denegadas = [], [], []

        class Filas:
            def __init__(self, filas):
                self.filas = filas

            def fetchall(self):
                return self.filas

        class Lectura(sqlite3.Connection):
            def execute(conexion, sql, parametros=()):
                llamadas.append((sql, parametros, conexion.in_transaction))
                cursor = super().execute(sql, parametros)
                if alterar_detalles and "FROM detalle_pronostico" in sql:
                    return Filas(alterar_detalles([dict(f) for f in cursor.fetchall()]))
                return cursor

        def abrir(ruta_uri, **kwargs):
            self.assertEqual(ruta_uri, self.ruta.resolve().as_uri() + "?mode=ro")
            self.assertTrue(kwargs["uri"])
            c = abrir_real(ruta_uri, factory=Lectura, **kwargs)
            conexiones.append(c)

            def autorizar(accion, tabla, columna, *otros):
                # Cualquier intento de DML, DDL, ATTACH o PRAGMA se rechaza.
                permitidas = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                              sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION)
                if accion not in permitidas or (accion == sqlite3.SQLITE_READ
                                               and columna in ("configuracion_json", "ruta_modelo")):
                    denegadas.append((accion, tabla, columna))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            c.set_authorizer(autorizar)
            return c

        with ExitStack() as stack:
            stack.enter_context(patch.object(sqlite3, "connect", side_effect=abrir))
            for nombre in ("consultar_pronostico", "listar_pronosticos", "consultar_configuracion_pronostico"):
                stack.enter_context(patch.object(pronosticos, nombre, side_effect=AssertionError(nombre)))
            yield llamadas
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(denegadas, [])
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                sqlite3.Connection.execute(c, "SELECT 1")

    def test_pronostico_valido_y_procedencia_publica(self):
        r = self.consultar()
        self.assertEqual(set(r), {"conjunto_id", "serie", "pronostico_id", "experimento_id", "tecnica",
                                  "fecha_corte", "horizonte", "estado_cobertura", "alcance_metricas", "evaluacion"})
        self.assertEqual(r["serie"], {"id": 83, "store_nbr": 17, "family": "FAMILIA A"})
        self.assertEqual((r["conjunto_id"], r["pronostico_id"], r["experimento_id"]), (7, 101, 201))
        self.assertEqual((r["tecnica"], r["fecha_corte"], r["horizonte"]), (BASE, CORTES[0], 15))
        self.assertEqual(r["estado_cobertura"], "completa")
        self.assertEqual(r["evaluacion"]["cobertura"], 1)
        self.assertNotIn("PRIVADO", json.dumps(r))

    def test_metricas_exactas_de_ejemplo_conocido(self):
        e = self.consultar()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (2, 10, sqrt(20 / 3)))
        self.assertEqual([d["error_absoluto"] for d in e["detalles"]], [2, 4, 0] * 5)
        self.assertEqual([d["venta_real"] for d in e["detalles"]], REALES)

    def test_delega_una_vez_en_a1_y_conserva_resultado_integro(self):
        detalles = [{"paso": i, "fecha_objetivo": fecha_objetivo(CORTES[0], i), "venta_estimada": v}
                    for i, v in enumerate(ESTIMADAS, 1)]
        reales = [{"fecha": fecha_objetivo(CORTES[0], i), "sales": v} for i, v in enumerate(REALES, 1)]
        esperado = metricas.calcular_metricas(CORTES[0], detalles, reales)
        with patch.object(metricas, "calcular_metricas", wraps=metricas.calcular_metricas) as calcular:
            r = self.consultar()
        calcular.assert_called_once_with(CORTES[0], detalles, reales)
        self.assertEqual(r["evaluacion"], esperado)

    def test_cobertura_parcial_con_ausencias_interiores(self):
        faltan = (2, 8, 11)
        for paso in faltan:
            self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=? AND fecha=?",
                          (83, fecha_objetivo(CORTES[0], paso)))
        r = self.consultar()
        e = r["evaluacion"]
        self.assertEqual(r["estado_cobertura"], "parcial")
        self.assertEqual(r["alcance_metricas"], "Métricas sobre observaciones disponibles.")
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (12, 3, 12 / 15))
        self.assertEqual(e["mae"], 18 / 12)
        self.assertAlmostEqual(e["wape"], 100 * 18 / 240)
        self.assertAlmostEqual(e["rmse"], sqrt(52 / 12))
        self.assertEqual(e["fechas_ausentes"], [fecha_objetivo(CORTES[0], p) for p in faltan])
        for paso in faltan:
            self.assertIsNone(e["detalles"][paso - 1]["venta_real"])
            self.assertIsNone(e["detalles"][paso - 1]["error_absoluto"])

    def test_sin_observaciones_no_aparenta_evaluacion(self):
        self.ejecutar("DELETE FROM venta_diaria")
        r = self.consultar()
        e = r["evaluacion"]
        self.assertEqual(r["estado_cobertura"], "sin_observaciones")
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (None, None, None))
        self.assertEqual((e["observaciones_evaluadas"], e["fechas_sin_registro"], e["cobertura"]), (0, 15, 0))

    def test_ceros_reales_wape_indefinido_rmse_valido(self):
        self.ejecutar("UPDATE venta_diaria SET sales=0")
        self.ejecutar("UPDATE detalle_pronostico SET venta_estimada=2")
        r = self.consultar()
        self.assertEqual(r["estado_cobertura"], "completa")
        e = r["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (2, None, 2))
        self.assertEqual(e["cobertura"], 1)

    def test_ventas_fraccionarias(self):
        self.ejecutar("UPDATE venta_diaria SET sales=0.25")
        self.ejecutar("UPDATE detalle_pronostico SET venta_estimada=0.75")
        e = self.consultar()["evaluacion"]
        self.assertEqual((e["mae"], e["wape"], e["rmse"]), (0.5, 200, 0.5))

    def test_los_tres_cortes_autorizados(self):
        for i, corte in enumerate(CORTES):
            with self.subTest(corte=corte):
                if i:
                    self.crear_pronostico(101 + i, 201 + i, corte=corte)
                    self.insertar_reales(corte=corte)
                r = self.consultar(101 + i)
                self.assertEqual(r["fecha_corte"], corte)
                self.assertEqual((r["evaluacion"]["mae"], r["evaluacion"]["wape"]), (2, 10))
                self.assertEqual(r["evaluacion"]["detalles"][-1]["fecha_objetivo"], fecha_objetivo(corte, 15))

    def test_tecnicas_distintas_mismas_fechas_y_ventas(self):
        self.crear_pronostico(102, 202, tecnica="random_forest", estimadas=REALES)
        a, b = self.consultar(), self.consultar(102)
        self.assertNotEqual(a["tecnica"], b["tecnica"])
        self.assertEqual([(d["fecha_objetivo"], d["venta_real"]) for d in a["evaluacion"]["detalles"]],
                         [(d["fecha_objetivo"], d["venta_real"]) for d in b["evaluacion"]["detalles"]])
        self.assertEqual((b["evaluacion"]["mae"], b["evaluacion"]["wape"], b["evaluacion"]["rmse"]), (0, 0, 0))

    def test_dos_series_sin_mezclar_observaciones(self):
        self.crear_pronostico(102, 202, serie=84, estimadas=[100] * 15)
        self.insertar_reales(serie=84, valores=[100] * 15)
        a, b = self.consultar(), self.consultar(102)
        self.assertEqual(a["evaluacion"]["mae"], 2)
        self.assertEqual(b["serie"]["id"], 84)
        self.assertEqual(b["evaluacion"]["mae"], 0)
        self.assertEqual([d["venta_real"] for d in b["evaluacion"]["detalles"]], [100] * 15)

    def test_sin_conjunto_activo_rechazo_antes_de_ventas(self):
        self.ejecutar("UPDATE conjunto_datos SET activo=0")
        with self.lectura_vigilada() as llamadas:
            self.error("sin_conjunto_activo")
        self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_pronostico_del_conjunto_anterior_no_es_evaluable(self):
        self.crear_pronostico(102, 202, serie=24)
        with self.lectura_vigilada() as llamadas:
            self.error("pronostico_inactivo", 102)
        self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))
        self.ejecutar("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
        self.assertEqual(self.consultar(102)["conjunto_id"], 2)
        self.error("pronostico_inactivo", 101)

    def test_pronostico_inexistente(self):
        self.error("pronostico_inexistente", 999)

    def test_activo_mas_reciente_coincide_con_consulta_existente(self):
        self.ejecutar("UPDATE conjunto_datos SET activo=1 WHERE id=2")
        self.assertEqual(self.consultar()["conjunto_id"], 7)

    def test_identificadores_invalidos_no_abren_conexion(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")) as abrir:
            for valor in (None, True, False, 0, -1, 101.0, "101", "101 OR 1=1", [], {}, 2**63):
                with self.subTest(valor=valor):
                    self.error("pronostico_invalido", valor)
            abrir.assert_not_called()

    def test_detalles_omitidos(self):
        self.ejecutar("DELETE FROM detalle_pronostico WHERE pronostico_id=101 AND paso=8")
        with self.lectura_vigilada() as llamadas:
            self.error("datos_inconsistentes")
        self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_detalles_fechas_y_pasos_incorrectos(self):
        for sql, valor in (("UPDATE detalle_pronostico SET fecha_objetivo=? WHERE paso=8", "2017-06-16"),
                           ("UPDATE detalle_pronostico SET fecha_objetivo=? WHERE paso=8", "2017-06-25T00:00:00"),
                           ("UPDATE detalle_pronostico SET paso=? WHERE paso=8", 16)):
            with self.subTest(sql=sql, valor=valor):
                self.corromper(sql, (valor,))
                self.error("datos_inconsistentes")
                self.ejecutar("UPDATE detalle_pronostico SET paso=8,fecha_objetivo=? WHERE paso IN (8,16)",
                              (fecha_objetivo(CORTES[0], 8),))

    def test_esquema_impide_detalles_y_reales_duplicados(self):
        contenido = self.ruta.read_bytes()
        for sql, parametros in (
            ("INSERT INTO detalle_pronostico VALUES (?,?,?,?)", (101, 1, "2017-06-17", 10)),
            ("INSERT INTO detalle_pronostico VALUES (?,?,?,?)", (101, 16, "2017-06-17", 10)),
            ("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
             (83, "2017-06-17", 10, 0))):
            with self.subTest(sql=sql, parametros=parametros), self.assertRaises(sqlite3.IntegrityError):
                self.corromper(sql, parametros)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_lectura_de_detalles_duplicados_se_rechaza_defensivamente(self):
        # La unicidad física ya se comprueba arriba. Aquí se simula una lectura
        # corrupta de quince filas sin alterar tablas ni índices del esquema.
        for campo in ("paso", "fecha_objetivo"):
            def duplicar(filas):
                filas[1][campo] = filas[0][campo]
                return filas
            with self.subTest(campo=campo), self.lectura_vigilada(duplicar) as llamadas:
                self.error("datos_inconsistentes")
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_tipo_prueba_o_futuro_rechazado_antes_de_leer_ventas(self):
        for tipo in ("prueba", "futuro"):
            for corte in (CORTES[0], "2017-07-31"):
                self.ejecutar("UPDATE pronostico SET tipo=?,fecha_corte=?", (tipo, corte))
                with self.subTest(tipo=tipo, corte=corte), self.lectura_vigilada() as llamadas:
                    self.error("tipo_no_evaluable")
                self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_cortes_no_autorizados_incluida_reserva_no_leen_ventas(self):
        for corte in ("2017-06-17", "2017-07-17", "2017-07-31", "2017-08-01", "2017-08-15"):
            self.ejecutar("UPDATE pronostico SET fecha_corte=?", (corte,))
            with self.subTest(corte=corte), self.lectura_vigilada() as llamadas:
                self.error("corte_no_autorizado")
            self.assertFalse(any("FROM venta_diaria" in sql for sql, _, _ in llamadas))

    def test_metadatos_corruptos_no_aparentan_evaluacion(self):
        casos = (("UPDATE pronostico SET horizonte=?", "quince", "UPDATE pronostico SET horizonte=15"),
                 ("UPDATE pronostico SET fecha_corte=?", "20170616", "UPDATE pronostico SET fecha_corte='2017-06-16'"),
                 ("UPDATE pronostico SET fecha_generacion=?", "inválida", "UPDATE pronostico SET fecha_generacion='2026-01-03'"),
                 ("UPDATE pronostico SET duracion_segundos=?", float("inf"), "UPDATE pronostico SET duracion_segundos=NULL"),
                 ("UPDATE experimento SET fin_entrenamiento=?", "2017-07-01", "UPDATE experimento SET fin_entrenamiento='2017-06-16'"),
                 ("UPDATE serie_temporal SET store_nbr=? WHERE id=83", "tienda", "UPDATE serie_temporal SET store_nbr=17 WHERE id=83"),
                 ("UPDATE serie_temporal SET family=? WHERE id=83", b"familia", "UPDATE serie_temporal SET family='FAMILIA A' WHERE id=83"))
        for sql, valor, restaurar in casos:
            with self.subTest(sql=sql, valor=valor):
                self.corromper(sql, (valor,))
                self.error("datos_inconsistentes")
                self.ejecutar(restaurar)

    def test_referencias_rotas_controladas(self):
        for sql, restaurar in (("UPDATE pronostico SET experimento_id=999", "UPDATE pronostico SET experimento_id=201"),
                               ("UPDATE experimento SET serie_id=999", "UPDATE experimento SET serie_id=83")):
            # Únicamente la fixture ignora FK para reproducir una base dañada.
            with closing(sqlite3.connect(self.ruta)) as c, c:
                c.execute(sql)
            self.error("datos_inconsistentes")
            self.ejecutar(restaurar)

    def test_valores_invalidos_almacenados_en_reales_y_estimadas(self):
        for tabla, campo in (("venta_diaria", "sales"), ("detalle_pronostico", "venta_estimada")):
            for valor in (-1, float("inf"), -float("inf"), "PRIVADO NO_NUMERICO", b"1"):
                with self.subTest(tabla=tabla, valor=valor):
                    self.ejecutar(f"UPDATE {tabla} SET {campo}=?", (valor,))
                    self.error("datos_inconsistentes")
            self.ejecutar(f"UPDATE {tabla} SET {campo}=10")

    def test_nan_no_se_almacena_como_observacion_valida(self):
        # sqlite3 transforma NaN en NULL; el esquema actual lo impide.
        for tabla, campo in (("venta_diaria", "sales"), ("detalle_pronostico", "venta_estimada")):
            with self.subTest(tabla=tabla), self.assertRaises(sqlite3.IntegrityError):
                self.ejecutar(f"UPDATE {tabla} SET {campo}=?", (float("nan"),))

    def test_desbordamientos_de_a1_son_errores_controlados(self):
        self.ejecutar("UPDATE detalle_pronostico SET venta_estimada=?", (1e308,))
        self.error("datos_inconsistentes")

    def test_fecha_real_no_canonica_dentro_del_filtro_se_rechaza(self):
        self.ejecutar("UPDATE venta_diaria SET fecha=? WHERE fecha=?",
                      ("2017-06-20T00:00:00", "2017-06-20"))
        self.error("datos_inconsistentes")

    def test_base_ausente_corrupta_y_sin_esquema(self):
        ausente = self.ruta.parent / "PRIVADO_ausente.sqlite3"
        self.error("base_no_disponible", ruta=ausente)
        self.assertFalse(ausente.exists())
        for nombre, contenido in (("corrupta.sqlite3", b"Esto no es SQLite"), ("vacia.sqlite3", b"")):
            ruta = self.ruta.parent / nombre
            ruta.write_bytes(contenido)
            self.error("base_no_disponible", ruta=ruta)
            self.assertEqual(ruta.read_bytes(), contenido)

    def test_rutas_invalidas_controladas(self):
        for ruta in (None, 5, [], "", "PRIVADO\0ruta"):
            with self.subTest(ruta=ruta), self.assertRaises(servicio.ErrorConsultaEvaluacion) as error:
                servicio.consultar_evaluacion(ruta, 101)
            self.assertEqual(error.exception.codigo, "base_no_disponible")
            self.assertNotIn("PRIVADO", str(error.exception))

    def test_conexion_unica_ro_begin_filtro_parametrizado_y_cierre(self):
        # Valores inválidos fuera de la ventana no deben llegar a A1.
        self.ejecutar("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                      (83, "2017-06-16", "NO_USAR_HISTORIAL", 0))
        self.ejecutar("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                      (83, "2017-07-02", "NO_USAR_OTRA_VENTANA", 0))
        with self.lectura_vigilada() as llamadas:
            self.consultar()
        # La comprobación posterior de cierre falla antes de ejecutar SQL.
        consultas = llamadas
        self.assertEqual(consultas[0][:2], ("BEGIN", ()))
        self.assertEqual(len(consultas), 5)
        self.assertTrue(all(en_transaccion for _, _, en_transaccion in consultas[1:]))
        ventas = [(sql, p) for sql, p, _ in consultas if "FROM venta_diaria" in sql]
        self.assertEqual(len(ventas), 1)
        self.assertIn("serie_id = ? AND fecha > ? AND fecha <= ?", ventas[0][0])
        self.assertEqual(ventas[0][1], (83, "2017-06-16", "2017-07-01"))
        self.assertEqual(self.ejecutar("SELECT COUNT(*) FROM evaluacion"), [(0,)])

    def activar_wal(self):
        try:
            modo = self.ejecutar("PRAGMA journal_mode=WAL")[0][0]
        except sqlite3.OperationalError as error:
            self.skipTest(f"WAL no disponible en esta SQLite temporal: {error}")
        if modo != "wal":
            self.skipTest("El entorno no permite WAL en la SQLite temporal.")

    def test_snapshot_wal_cambio_de_activo_cabecera_detalles_y_reales(self):
        self.activar_wal()
        esperado = self.consultar()
        activo_real = pronosticos._activo

        def cambiar(conexion):
            activo = activo_real(conexion)  # Fija la instantánea antes del escritor.
            self.assertTrue(conexion.in_transaction)
            with closing(conectar(self.ruta)) as escritor, escritor:
                escritor.execute("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
                escritor.execute("UPDATE serie_temporal SET family='CAMBIADA' WHERE id=83")
                escritor.execute("UPDATE experimento SET tecnica='random_forest'")
                escritor.execute("UPDATE pronostico SET tipo='futuro'")
                escritor.execute("UPDATE detalle_pronostico SET venta_estimada=999")
                escritor.execute("UPDATE venta_diaria SET sales=999")
            return activo

        with patch.object(pronosticos, "_activo", side_effect=cambiar) as intercalado:
            self.assertEqual(servicio.consultar_evaluacion(self.ruta, 101), esperado)
        intercalado.assert_called_once()
        self.assertEqual(self.ejecutar("SELECT sales FROM venta_diaria LIMIT 1"), [(999.0,)])
        self.error("pronostico_inactivo")  # Una nueva instantánea ve el nuevo activo.

    def test_snapshot_wal_ventas_cambiadas_despues_de_leer_detalles(self):
        self.activar_wal()
        esperado = self.consultar()
        detalles_reales = pronosticos._detalles_validos

        def cambiar(conexion, *args):
            detalles = detalles_reales(conexion, *args)
            with closing(conectar(self.ruta)) as escritor, escritor:
                escritor.execute("UPDATE detalle_pronostico SET venta_estimada=99")
                escritor.execute("UPDATE venta_diaria SET sales=99")
                escritor.execute("DELETE FROM venta_diaria WHERE fecha='2017-06-24'")
            return detalles

        with patch.object(pronosticos, "_detalles_validos", side_effect=cambiar):
            self.assertEqual(servicio.consultar_evaluacion(self.ruta, 101), esperado)
        nuevo = self.consultar()
        self.assertEqual(nuevo["estado_cobertura"], "parcial")
        self.assertEqual(nuevo["evaluacion"]["observaciones_evaluadas"], 14)
        self.assertEqual(nuevo["evaluacion"]["mae"], 0)

    def test_importacion_y_consulta_sin_modelos_red_archivos_ni_subprocesos(self):
        codigo = compile(Path(servicio.__file__).read_text(encoding="utf-8"), servicio.__file__, "exec")
        aislado = ModuleType("consulta_evaluaciones_en_prueba")
        importar_real = builtins.__import__
        prohibidos = ("backend.app.services.registro_pronosticos", "scripts.modelos_prediccion",
                      "scripts.linea_base_prediccion", "sklearn", "fastapi", "uvicorn")

        def importar(nombre, globals=None, locals=None, fromlist=(), level=0):
            candidatos = (nombre,) + tuple(f"{nombre}.{item}" for item in (fromlist or ()))
            if any(c == p or c.startswith(p + ".") for c in candidatos for p in prohibidos):
                raise AssertionError(f"Dependencia de ejecución: {nombre}")
            return importar_real(nombre, globals, locals, fromlist, level)

        objetivos = ((builtins, "open"), (io, "open"), (Path, "open"), (os, "open"),
                     (socket, "socket"), (subprocess, "Popen"))
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            stack.enter_context(patch.object(builtins, "__import__", side_effect=importar))
            with patch.object(sqlite3, "connect", side_effect=AssertionError("Conexión al importar")):
                exec(codigo, aislado.__dict__)
            with self.lectura_vigilada():
                r = aislado.consultar_evaluacion(self.ruta, 101)
                self.assertEqual(r["evaluacion"]["mae"], 2)
            for doble in dobles:
                doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
