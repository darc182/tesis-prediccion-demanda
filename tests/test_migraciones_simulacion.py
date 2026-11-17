"""Preparación explícita sobre SQLite temporales; nunca bases del usuario."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from backend.app.storage import migraciones_simulacion as migracion
from backend.app.storage.db import inicializar_base


ORIGINALES = ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento",
              "pronostico", "detalle_pronostico", "evaluacion", "simulacion", "perfil_global_conjunto")


class MigracionesSimulacionTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "sintetica.sqlite3"
        inicializar_base(self.ruta)
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("INSERT INTO conjunto_datos VALUES (1,'sintetico','hash','2017-06-16',1)")
            c.execute("INSERT INTO serie_temporal VALUES (10,1,1,'FAMILIA','sintetico')")
            c.execute("INSERT INTO venta_diaria VALUES (1,10,'2017-06-01',0.5,0)")
            c.execute("INSERT INTO experimento VALUES (20,10,'random_forest','{}','2017-06-01','2017-06-16',NULL)")
            c.execute("INSERT INTO pronostico VALUES (30,20,'2017-06-16','validacion',15,'2017-06-16T00:00:00Z',0.1)")
            c.execute("INSERT INTO detalle_pronostico VALUES (30,1,'2017-06-17',0.1)")
            c.execute("INSERT INTO evaluacion VALUES (30,0.1,NULL,0.1)")
            c.execute("INSERT INTO simulacion VALUES (1,30,0.3,15,0.1,0.5,0.3,'legado','2017-06-16')")
        self.antes = self.originales()

    def sql(self, texto, valores=()):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA foreign_keys=ON")
            return c.execute(texto, valores).fetchall()

    def originales(self):
        with closing(sqlite3.connect(self.ruta)) as c:
            return [(nombre, c.execute("SELECT sql FROM sqlite_master WHERE name=?", (nombre,)).fetchone(),
                     c.execute(f'SELECT * FROM "{nombre}" ORDER BY 1').fetchall()) for nombre in ORIGINALES]

    def catalogo(self):
        return self.sql("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")

    def preparar(self):
        return migracion.preparar_esquema_simulaciones(self.ruta)

    def error(self, codigo, ruta=...):
        with self.assertRaises(migracion.ErrorPreparacionSimulaciones) as capturado:
            migracion.preparar_esquema_simulaciones(self.ruta if ruta is ... else ruta)
        self.assertEqual(capturado.exception.codigo, codigo)
        for privado in (str(self.ruta), "PRIVADO", "CREATE TABLE", "Traceback"):
            self.assertNotIn(privado, str(capturado.exception))

    def crear_alterado(self, transformar):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            for ddl in migracion._DDL:
                c.execute(transformar(ddl))
            c.execute("INSERT INTO version_esquema_simulaciones VALUES (1,1)")

    def ejecucion(self, **cambios):
        valores = dict(id=1, clave_idempotencia="operacion", modalidad="individual", numero_escenarios=1,
                       conjunto_id=1, serie_id=10, pronostico_id=30, fecha_registro="2026-11-16T12:00:00Z",
                       version_esquema=1, version_documento=1, version_regla="cobertura_horizonte_15_pendientes_v1",
                       identificador_implementacion="revision-sintetica", version_politica_numerica="binary64_v1",
                       solicitud_json="{}", sha256_solicitud="0" * 64,
                       origen_json="{}", sha256_origen="0" * 64,
                       contexto_json="{}", sha256_contexto="0" * 64,
                       advertencias_json="[]", sha256_advertencias="0" * 64)
        valores.update(cambios)
        return self.sql(f"INSERT INTO ejecucion_simulacion ({','.join(valores)}) VALUES ({','.join('?' for _ in valores)})",
                        tuple(valores.values()))

    def escenario(self, etiqueta="A", posicion=1, ejecucion=1, resultado="{}", huella="0" * 64):
        return self.sql("INSERT INTO escenario_simulacion VALUES (?,?,?,?,?)",
                        (ejecucion, etiqueta, posicion, resultado, huella))

    def test_preparacion_crea_tres_tablas_y_marcador(self):
        self.assertEqual(self.preparar(), {"version_esquema": 1, "estado": "preparado", "hubo_escritura": True})
        self.assertEqual(len(self.sql("SELECT name FROM sqlite_master WHERE type='table'")), 12)
        self.assertEqual(self.sql("SELECT * FROM version_esquema_simulaciones"), [(1, 1)])
        self.assertEqual(self.sql("SELECT * FROM ejecucion_simulacion"), [])
        self.assertEqual(self.sql("SELECT * FROM escenario_simulacion"), [])

    def test_preserva_nueve_tablas_datos_y_simulacion_heredada(self):
        self.preparar()
        self.assertEqual(self.originales(), self.antes)
        self.assertEqual(self.sql("SELECT regla_aplicada FROM simulacion"), [("legado",)])

    def test_segunda_ejecucion_verifica_sin_dml_ni_ddl(self):
        self.preparar()
        antes = self.catalogo()
        abrir = sqlite3.connect
        trazas = []

        def conectar(ruta, **opciones):
            c = abrir(ruta, **opciones)
            if ruta != ":memory:":
                c.set_trace_callback(trazas.append)
            return c

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            self.assertEqual(self.preparar(), {"version_esquema": 1, "estado": "verificado", "hubo_escritura": False})
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")) for s in trazas))
        self.assertEqual(self.catalogo(), antes)

    def test_mode_rw_fk_begin_y_cierre(self):
        abrir = sqlite3.connect
        conexiones, trazas = [], []

        def conectar(ruta, **opciones):
            c = abrir(ruta, **opciones)
            if ruta != ":memory:":
                self.assertEqual(ruta, self.ruta.resolve().as_uri() + "?mode=rw")
                self.assertEqual(opciones, {"uri": True})
                conexiones.append(c)
                c.set_trace_callback(trazas.append)
            return c

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            self.preparar()
        self.assertEqual(len(conexiones), 1)
        self.assertEqual(trazas[:3], ["PRAGMA foreign_keys=ON", "PRAGMA foreign_keys", "BEGIN IMMEDIATE"])
        self.assertEqual(trazas[-1], "COMMIT")
        with self.assertRaises(sqlite3.ProgrammingError):
            conexiones[0].execute("SELECT 1")

    def test_no_modifica_journal_mode_ni_version_global(self):
        self.sql("PRAGMA user_version=17")
        antes = self.sql("PRAGMA journal_mode")
        self.assertEqual(antes, [("delete",)])
        self.preparar()
        self.assertEqual(self.sql("PRAGMA journal_mode"), antes)
        self.assertEqual(self.sql("PRAGMA user_version"), [(17,)])

    def test_autorizador_impide_dml_sobre_tablas_originales(self):
        abrir = sqlite3.connect
        dml = []

        def autorizar(accion, tabla, columna, base, origen):
            if accion in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
                if tabla in ORIGINALES:
                    raise AssertionError("No escribir tablas originales")
                if tabla != "sqlite_master":
                    dml.append((accion, tabla))
            return sqlite3.SQLITE_OK

        def conectar(ruta, **opciones):
            c = abrir(ruta, **opciones)
            if ruta != ":memory:":
                c.set_authorizer(autorizar)
            return c

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            self.preparar()
        self.assertEqual(dml, [(sqlite3.SQLITE_INSERT, "version_esquema_simulaciones")])
        self.assertEqual(self.originales(), self.antes)

    def test_modalidades_validas(self):
        self.preparar()
        for identificador, (modalidad, n) in enumerate((("individual", 1), ("comparacion", 2), ("comparacion", 3)), 1):
            self.ejecucion(id=identificador, clave_idempotencia=str(identificador), modalidad=modalidad, numero_escenarios=n)
        self.assertEqual(len(self.sql("SELECT * FROM ejecucion_simulacion")), 3)

    def test_modalidades_y_conteos_invalidos(self):
        self.preparar()
        for modalidad, n in (("individual", 2), ("individual", 3), ("comparacion", 1), ("otra", 2), ("comparacion", 4), ("comparacion", 2.5)):
            with self.subTest(modalidad=modalidad, n=n), self.assertRaises(sqlite3.IntegrityError):
                self.ejecucion(modalidad=modalidad, numero_escenarios=n)

    def test_clave_idempotencia_unique(self):
        self.preparar()
        self.ejecucion()
        with self.assertRaises(sqlite3.IntegrityError):
            self.ejecucion(id=2)

    def test_etiquetas_y_posiciones_validas(self):
        self.preparar()
        self.ejecucion(modalidad="comparacion", numero_escenarios=3)
        for posicion, etiqueta in enumerate("ABC", 1):
            self.escenario(etiqueta, posicion)
        self.assertEqual(self.sql("SELECT etiqueta,posicion FROM escenario_simulacion ORDER BY posicion"), [("A", 1), ("B", 2), ("C", 3)])

    def test_etiquetas_y_posiciones_invalidas(self):
        self.preparar()
        self.ejecucion()
        for etiqueta, posicion in (("A", 2), ("B", 1), ("C", 2), ("D", 1), ("a", 1), ("", 1), ("A", 0), ("A", 1.5)):
            with self.subTest(etiqueta=etiqueta, posicion=posicion), self.assertRaises(sqlite3.IntegrityError):
                self.escenario(etiqueta, posicion)

    def test_pk_y_unique_hijos(self):
        self.preparar()
        self.ejecucion()
        self.escenario()
        with self.assertRaises(sqlite3.IntegrityError):
            self.escenario()
        indices = self.sql("PRAGMA index_list(escenario_simulacion)")
        self.assertEqual(sorted(f[3] for f in indices), ["pk", "u"])

    def test_fk_independientes_y_restrictivas(self):
        self.preparar()
        for campo in ("conjunto_id", "serie_id", "pronostico_id"):
            with self.subTest(campo=campo), self.assertRaises(sqlite3.IntegrityError):
                self.ejecucion(**{campo: 999})
        self.ejecucion()
        with self.assertRaises(sqlite3.IntegrityError):
            self.escenario(ejecucion=999)
        self.escenario()
        for tabla in ("pronostico", "ejecucion_simulacion"):
            with self.subTest(tabla=tabla), self.assertRaises(sqlite3.IntegrityError):
                self.sql(f"DELETE FROM {tabla}")
        for fila in self.sql("PRAGMA foreign_key_list(ejecucion_simulacion)"):
            self.assertEqual(fila[5:7], ("RESTRICT", "RESTRICT"))

    def test_desactivar_conjunto_preserva_historial(self):
        self.preparar()
        self.ejecucion()
        self.escenario()
        self.sql("UPDATE conjunto_datos SET activo=0")
        self.assertEqual(self.preparar()["estado"], "verificado")
        self.assertEqual(self.sql("SELECT etiqueta FROM escenario_simulacion"), [("A",)])

    def test_not_null_en_cabecera(self):
        self.preparar()
        for campo in ("clave_idempotencia", "modalidad", "numero_escenarios", "conjunto_id", "serie_id", "pronostico_id",
                      "fecha_registro", "version_esquema", "version_documento", "version_regla", "identificador_implementacion",
                      "version_politica_numerica", "solicitud_json", "sha256_solicitud", "origen_json", "sha256_origen",
                      "contexto_json", "sha256_contexto", "advertencias_json", "sha256_advertencias"):
            with self.subTest(campo=campo), self.assertRaises(sqlite3.IntegrityError):
                self.ejecucion(**{campo: None})

    def test_not_null_en_escenario(self):
        self.preparar()
        self.ejecucion()
        for valores in ((None, "A", 1, "{}", "0"*64), (1, None, 1, "{}", "0"*64), (1, "A", None, "{}", "0"*64),
                        (1, "A", 1, None, "0"*64), (1, "A", 1, "{}", None)):
            with self.subTest(valores=valores), self.assertRaises(sqlite3.IntegrityError):
                self.sql("INSERT INTO escenario_simulacion VALUES (?,?,?,?,?)", valores)

    def test_tipos_y_enteros_positivos(self):
        self.preparar()
        for campo in ("id", "conjunto_id", "serie_id", "pronostico_id", "version_esquema", "version_documento"):
            for valor in (0, -1, 1.5, "incorrecto", b"1"):
                with self.subTest(campo=campo, valor=valor), self.assertRaises(sqlite3.IntegrityError):
                    self.ejecucion(**{campo: valor})

    def test_limites_texto_en_bytes(self):
        self.preparar()
        self.ejecucion(clave_idempotencia="é"*64)
        for campo in ("clave_idempotencia", "version_regla", "identificador_implementacion", "version_politica_numerica"):
            for valor in ("é"*65, "", "  ", "a\0b", b"texto"):
                with self.subTest(campo=campo, valor=valor), self.assertRaises(sqlite3.IntegrityError):
                    self.ejecucion(id=2, **{campo: valor})

    def test_fecha_registro_utc_canonica(self):
        self.preparar()
        for fecha in ("2026-11-16", "2026-11-16T12:00:00+00:00", "2026-02-30T12:00:00Z", "2026-11-16T25:00:00Z", "PRIVADO"):
            with self.subTest(fecha=fecha), self.assertRaises(sqlite3.IntegrityError):
                self.ejecucion(fecha_registro=fecha)

    def test_huellas_formato_y_tipo(self):
        self.preparar()
        for campo in ("sha256_solicitud", "sha256_origen", "sha256_contexto", "sha256_advertencias"):
            for valor in ("a"*63, "a"*65, "A"*64, "g"*64, b"a"*64, "a"*64 + "\0"):
                with self.subTest(campo=campo, valor=valor), self.assertRaises(sqlite3.IntegrityError):
                    self.ejecucion(**{campo: valor})
        self.ejecucion()
        with self.assertRaises(sqlite3.IntegrityError):
            self.escenario(huella="invalida")

    def test_documentos_objetos_y_advertencias_array(self):
        self.preparar()
        for campo in ("solicitud_json", "origen_json", "contexto_json"):
            for valor in ("{", "[]", "null", '"texto"', "42", b"{}"):
                with self.subTest(campo=campo, valor=valor), self.assertRaises(sqlite3.IntegrityError):
                    self.ejecucion(**{campo: valor})
        with self.assertRaises(sqlite3.IntegrityError):
            self.ejecucion(advertencias_json="{}")
        self.ejecucion()
        for valor in ("{", "[]", "null", b"{}"):
            with self.subTest(valor=valor), self.assertRaises(sqlite3.IntegrityError):
                self.escenario(resultado=valor)

    def test_limite_documentos_exactamente_y_exceso(self):
        self.preparar()
        limite = migracion.MAX_BYTES_DOCUMENTO
        objeto = '{"a":"' + 'a'*(limite-8) + '"}'
        self.assertEqual(len(objeto.encode()), limite)
        self.ejecucion(origen_json=objeto)
        self.escenario(resultado=objeto)
        for campo in ("solicitud_json", "origen_json", "contexto_json"):
            with self.subTest(campo=campo), self.assertRaises(sqlite3.IntegrityError):
                self.ejecucion(id=2, clave_idempotencia="otra", **{campo: objeto + " "})
        with self.assertRaises(sqlite3.IntegrityError):
            self.escenario("B", 2, resultado=objeto + " ")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ejecucion(id=2, clave_idempotencia="otra", advertencias_json='["'+'a'*limite+'"]')

    def test_sql_no_acredita_semantica_futura_ni_cardinalidad(self):
        self.preparar()
        # Huella ficticia y duplicados JSON son deliberados: el preparador no
        # implementa ni promete el contrato de snapshots del próximo incremento.
        self.ejecucion(modalidad="comparacion", numero_escenarios=3, origen_json='{"a":1,"a":2}')
        self.assertEqual(self.preparar()["estado"], "verificado")
        self.assertEqual(self.sql("SELECT * FROM escenario_simulacion"), [])

    def test_rechaza_preparacion_parcial_sin_completar(self):
        self.sql(migracion._DDL[0])
        antes = self.catalogo()
        self.error("esquema_simulaciones_incompatible")
        self.assertEqual(self.catalogo(), antes)

    def test_rechaza_tabla_con_columnas_incorrectas(self):
        self.crear_alterado(lambda sql: sql.replace("version_documento", "documento_ajeno"))
        antes = self.catalogo()
        self.error("esquema_simulaciones_incompatible")
        self.assertEqual(self.catalogo(), antes)

    def test_rechaza_fk_adulterada(self):
        self.crear_alterado(lambda sql: sql.replace("REFERENCES pronostico(id)", "REFERENCES experimento(id)"))
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_unique_ausente(self):
        self.crear_alterado(lambda sql: sql.replace("UNIQUE (ejecucion_id,posicion),", ""))
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_check_alterado(self):
        self.crear_alterado(lambda sql: sql.replace("posicion BETWEEN 1 AND 3", "posicion BETWEEN 1 AND 9"))
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_indice_adicional_sobre_tabla_nueva(self):
        self.preparar()
        self.sql("CREATE UNIQUE INDEX condicion_ajena ON ejecucion_simulacion(pronostico_id)")
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_trigger_nuevo(self):
        self.preparar()
        self.sql("CREATE TRIGGER ajeno AFTER INSERT ON escenario_simulacion BEGIN SELECT 1; END")
        self.error("esquema_simulaciones_incompatible")

    def estado_con_datos(self):
        return self.catalogo(), [(tabla, self.sql(f'SELECT * FROM "{tabla}" ORDER BY 1,2'))
                                 for tabla in ORIGINALES + migracion.TABLAS_NUEVAS]

    def comprobar_variantes_trigger(self, tabla):
        self.preparar()
        self.ejecucion()
        self.escenario()
        mezclado = "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(tabla))
        for nombre in (tabla, tabla.upper(), mezclado):
            for citado in (False, True):
                for operacion in ("INSERT", "UPDATE", "DELETE"):
                    with self.subTest(tabla=nombre, citado=citado, operacion=operacion):
                        destino = f'"{nombre}"' if citado else nombre
                        self.sql(f"CREATE TRIGGER vigilado AFTER {operacion} ON {destino} BEGIN SELECT 1; END")
                        try:
                            antes = self.estado_con_datos()
                            self.error("esquema_simulaciones_incompatible")
                            self.assertEqual(self.estado_con_datos(), antes)
                        finally:
                            self.sql("DROP TRIGGER vigilado")

    def test_triggers_version_todas_las_variantes(self):
        self.comprobar_variantes_trigger("version_esquema_simulaciones")

    def test_triggers_ejecucion_todas_las_variantes(self):
        self.comprobar_variantes_trigger("ejecucion_simulacion")

    def test_triggers_escenario_todas_las_variantes(self):
        self.comprobar_variantes_trigger("escenario_simulacion")

    def test_trigger_mayusculas_borrado_rechazado_antes_de_utilizar_base(self):
        self.preparar()
        self.ejecucion()
        self.sql('CREATE TRIGGER borrado AFTER INSERT ON "ESCENARIO_SIMULACION" '
                 'BEGIN DELETE FROM escenario_simulacion '
                 'WHERE ejecucion_id=NEW.ejecucion_id AND etiqueta=NEW.etiqueta; END')
        self.assertEqual(self.sql("SELECT tbl_name FROM sqlite_master WHERE name='borrado'"),
                         [("ESCENARIO_SIMULACION",)])
        antes, abrir, trazas, conexiones = self.estado_con_datos(), sqlite3.connect, [], []

        def conectar(ruta, **opciones):
            c = abrir(ruta, **opciones)
            if ruta != ":memory:":
                c.set_trace_callback(trazas.append)
                conexiones.append(c)
            return c

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            self.error("esquema_simulaciones_incompatible")
        self.assertIn("ROLLBACK", trazas)
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")) for s in trazas))
        self.assertEqual(self.estado_con_datos(), antes)
        with self.assertRaises(sqlite3.ProgrammingError):
            conexiones[0].execute("SELECT 1")

    def test_trigger_tabla_ajena_no_rechazado(self):
        self.preparar()
        self.ejecucion()
        self.escenario()
        for operacion in ("INSERT", "UPDATE", "DELETE"):
            with self.subTest(operacion=operacion):
                self.sql(f'CREATE TRIGGER permitido AFTER {operacion} ON "SIMULACION" BEGIN SELECT 1; END')
                try:
                    antes = self.estado_con_datos()
                    self.assertEqual(self.preparar(), {"version_esquema": 1, "estado": "verificado", "hubo_escritura": False})
                    self.assertEqual(self.estado_con_datos(), antes)
                finally:
                    self.sql("DROP TRIGGER permitido")

    def test_rechaza_version_futura_sin_sobrescribir(self):
        self.preparar()
        self.sql("UPDATE version_esquema_simulaciones SET version=2")
        self.error("version_no_soportada")
        self.assertEqual(self.sql("SELECT version FROM version_esquema_simulaciones"), [(2,)])

    def test_rechaza_marcador_vacio(self):
        self.preparar()
        self.sql("DELETE FROM version_esquema_simulaciones")
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_marcador_inconsistente_con_checks_desactivados(self):
        self.preparar()
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute("UPDATE version_esquema_simulaciones SET version=0")
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_vista_en_lugar_de_tabla(self):
        self.sql("CREATE VIEW ejecucion_simulacion AS SELECT 1 AS id")
        antes = self.catalogo()
        self.error("esquema_simulaciones_incompatible")
        self.assertEqual(self.catalogo(), antes)

    def test_rechaza_pk_alterada(self):
        self.crear_alterado(lambda sql: sql.replace("PRIMARY KEY (ejecucion_id,etiqueta)", "PRIMARY KEY (ejecucion_id,posicion)"))
        self.error("esquema_simulaciones_incompatible")

    def test_rechaza_cascada_adulterada(self):
        self.crear_alterado(lambda sql: sql.replace("ON DELETE RESTRICT", "ON DELETE CASCADE"))
        self.error("esquema_simulaciones_incompatible")

    def test_marcador_singleton_y_version_positiva(self):
        self.preparar()
        for valores in ((2, 1), (1, 0), (1, -1), (1, "otra"), (1, None)):
            with self.subTest(valores=valores), self.assertRaises(sqlite3.IntegrityError):
                self.sql("INSERT OR REPLACE INTO version_esquema_simulaciones VALUES (?,?)", valores)
        self.assertEqual(self.sql("SELECT * FROM version_esquema_simulaciones"), [(1, 1)])

    def test_base_ausente_no_creada(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        self.error("base_no_disponible", ausente)
        self.assertFalse(ausente.exists())

    def test_rutas_invalidas(self):
        for ruta in (None, "", "  ", "a\0b", 1, False):
            with self.subTest(ruta=ruta):
                self.error("base_no_disponible", ruta)

    def test_base_danada(self):
        danada = self.ruta.with_name("danada.sqlite3")
        danada.write_bytes(b"archivo sintetico no SQLite")
        self.error("base_no_disponible", danada)

    def test_base_sin_esquema(self):
        vacia = self.ruta.with_name("vacia.sqlite3")
        with closing(sqlite3.connect(vacia)):
            pass
        self.error("esquema_original_incompatible", vacia)

    def test_esquema_original_incompleto(self):
        self.sql("DROP TABLE perfil_global_conjunto")
        self.error("esquema_original_incompatible")

    def test_esquema_original_con_restriccion_distinta(self):
        self.sql("CREATE UNIQUE INDEX no_publicado ON simulacion(pronostico_id)")
        self.error("esquema_original_incompatible")

    def test_fk_original_rota_rechaza_sin_crear(self):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("UPDATE serie_temporal SET conjunto_id=999")
        self.error("fallo_integridad")
        self.assertFalse(any(f[1] in migracion.TABLAS_NUEVAS for f in self.catalogo()))

    def fallo_inyectado(self, fallo, despues=False, clase_error=sqlite3.OperationalError):
        antes, abrir, conexiones, trazas = self.catalogo(), sqlite3.connect, [], []

        class Vigilada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                if fallo(sql) and not despues:
                    raise clase_error("PRIVADO fallo sintetico")
                resultado = super().execute(sql, parametros)
                if fallo(sql) and despues:
                    raise clase_error("PRIVADO fallo sintetico")
                return resultado

        def conectar(ruta, **opciones):
            if ruta == ":memory:":
                return abrir(ruta, **opciones)
            c = abrir(ruta, factory=Vigilada, **opciones)
            conexiones.append(c)
            c.set_trace_callback(trazas.append)
            return c

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            if clase_error is sqlite3.OperationalError:
                self.error("error_persistencia")
            else:
                with self.assertRaises(clase_error):
                    self.preparar()
        self.assertIn("ROLLBACK", trazas)
        self.assertEqual(self.catalogo(), antes)
        self.assertEqual(self.originales(), self.antes)
        with self.assertRaises(sqlite3.ProgrammingError):
            conexiones[0].execute("SELECT 1")

    def test_rollback_si_falla_creacion(self):
        self.fallo_inyectado(lambda s: s.startswith("CREATE TABLE escenario_simulacion"))

    def test_rollback_si_falla_registro_version(self):
        self.fallo_inyectado(lambda s: s.startswith("INSERT INTO version_esquema_simulaciones"))

    def test_rollback_tras_ddl_ejecutado(self):
        self.fallo_inyectado(lambda s: s.startswith("CREATE TABLE ejecucion_simulacion"), despues=True)

    def test_rollback_tras_insert_version(self):
        self.fallo_inyectado(lambda s: s.startswith("INSERT INTO version_esquema_simulaciones"), despues=True)

    def test_programacion_inesperada_revierte_y_se_propaga(self):
        self.fallo_inyectado(lambda s: s.startswith("CREATE TABLE ejecucion_simulacion"), despues=True, clase_error=RuntimeError)

    def test_verificacion_final_detecta_ddl_ignorado(self):
        abrir, antes = sqlite3.connect, self.catalogo()

        class Ignorada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                if sql.startswith("CREATE TABLE escenario_simulacion"):
                    return super().execute("SELECT 1")
                return super().execute(sql, parametros)

        def conectar(ruta, **opciones):
            return abrir(ruta, **opciones) if ruta == ":memory:" else abrir(ruta, factory=Ignorada, **opciones)

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar):
            self.error("fallo_integridad")
        self.assertEqual(self.catalogo(), antes)

    def test_importacion_sin_conexiones_ni_preparacion_automatica(self):
        spec = importlib.util.spec_from_file_location("migracion_aislada", migracion.__file__)
        modulo = importlib.util.module_from_spec(spec)
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir al importar")):
            spec.loader.exec_module(modulo)
        self.assertTrue(callable(modulo.preparar_esquema_simulaciones))
        for nombre in ("backend/app/storage/db.py", "backend/app/main.py"):
            self.assertNotIn("preparar_esquema_simulaciones", Path(nombre).read_text(encoding="utf-8"))

    def test_concurrencia_serializada_sin_wal(self):
        abrir = sqlite3.connect
        primero_dentro, segundo_intenta, liberar = Event(), Event(), Event()

        class Vigilada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                if sql == "BEGIN IMMEDIATE":
                    if primero_dentro.is_set():
                        segundo_intenta.set()
                    resultado = super().execute(sql, parametros)
                    if not primero_dentro.is_set():
                        primero_dentro.set()
                        if not liberar.wait(5):
                            raise AssertionError("Tiempo de coordinacion agotado")
                    return resultado
                return super().execute(sql, parametros)

        def conectar(ruta, **opciones):
            return abrir(ruta, **opciones) if ruta == ":memory:" else abrir(ruta, factory=Vigilada, **opciones)

        with patch.object(migracion.sqlite3, "connect", side_effect=conectar), ThreadPoolExecutor(max_workers=2) as pool:
            primero = pool.submit(self.preparar)
            try:
                self.assertTrue(primero_dentro.wait(5))
                segundo = pool.submit(self.preparar)
                self.assertTrue(segundo_intenta.wait(5))
            finally:
                liberar.set()
            resultados = [primero.result(timeout=10), segundo.result(timeout=10)]
        self.assertEqual([r["estado"] for r in resultados], ["preparado", "verificado"])
        self.assertEqual(self.sql("SELECT * FROM version_esquema_simulaciones"), [(1, 1)])
        self.assertEqual(self.sql("PRAGMA journal_mode"), [("delete",)])
        self.assertEqual(self.originales(), self.antes)

    def test_resultado_json_sin_rutas_ni_documentos_productivos(self):
        resultado = self.preparar()
        self.assertEqual(json.loads(json.dumps(resultado, allow_nan=False)), resultado)
        self.assertEqual(set(resultado), {"version_esquema", "estado", "hubo_escritura"})


if __name__ == "__main__":
    unittest.main()
