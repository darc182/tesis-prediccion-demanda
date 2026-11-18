"""Historial exclusivamente en SQLite temporales y con motores sintéticos puros."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
from threading import Event, get_ident
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import consulta_historial_simulaciones as servicio
from backend.app.services import registro_simulaciones as registro
from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.services import trayectoria_simulacion as trayectoria
from backend.app.storage import migraciones_simulacion as esquema
from tests import test_registro_simulaciones as fixtures


class ConsultaHistorialSimulacionesTests(unittest.TestCase):
    def setUp(self):
        # Composición de fixtures: no hereda ni ejecuta los tests de registro.
        self.datos = fixtures.RegistroSimulacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta

    def listar(self, **kwargs):
        return servicio.listar_ejecuciones_simulaciones(self.ruta, **kwargs)

    def detalle(self, identificador):
        return servicio.consultar_ejecucion_simulacion(self.ruta, identificador)

    def basicos(self):
        return [self.datos.individual(clave="privada-individual"), self.datos.comparar(clave="privada-ab", n=2),
                self.datos.comparar(clave="privada-abc", n=3)]

    def todos(self):
        r = self.basicos()
        self.datos.datos.crear_pronostico(102, 202, serie=84, tecnica="random_forest")
        r.append(self.datos.individual(clave="privada-otra-serie", identificador=102))
        self.datos.datos.crear_pronostico(103, 203, serie=24, tecnica="hist_gradient_boosting")
        self.sql("UPDATE conjunto_datos SET activo=CASE WHEN id=2 THEN 1 ELSE 0 END")
        r.append(self.datos.individual(clave="privada-otro-conjunto", identificador=103))
        self.sql("UPDATE conjunto_datos SET activo=CASE WHEN id=7 THEN 1 ELSE 0 END")
        return r

    def sql(self, texto, valores=()):
        return self.datos.datos.ejecutar(texto, valores)

    def error(self, codigo, funcion, *args, **kwargs):
        with self.assertRaises(servicio.ErrorConsultaHistorialSimulaciones) as capturado:
            funcion(*args, **kwargs)
        self.assertEqual(capturado.exception.codigo, codigo)
        for privado in (str(self.ruta), "PRIVADO", "SELECT", "Traceback", "privada-"):
            self.assertNotIn(privado, str(capturado.exception))

    @contextmanager
    def vigilar(self):
        abrir, hilo = sqlite3.connect, get_ident()
        conexiones, consultas, denegadas = [], [], []
        permitidas = {"sqlite_master", "sqlite_schema", *esquema.TABLAS_NUEVAS}
        escrituras = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                      sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TRIGGER,
                      sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX,
                      sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_ALTER_TABLE}
        def autorizar(accion, tabla, columna, base, origen):
            if accion in escrituras or (accion == sqlite3.SQLITE_READ and tabla not in permitidas):
                denegadas.append((accion, tabla, columna))
                return sqlite3.SQLITE_DENY
            if accion == sqlite3.SQLITE_PRAGMA and tabla in ("journal_mode", "user_version"):
                denegadas.append((accion, tabla, columna))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        class Vigilada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                consultas.append((sql, tuple(parametros)))
                return super().execute(sql, parametros)
        def conectar(ruta, **opciones):
            if get_ident() == hilo:
                self.assertEqual(ruta, self.ruta.resolve().as_uri() + "?mode=ro")
                self.assertEqual(opciones, {"uri": True})
                c = abrir(ruta, factory=Vigilada, **opciones)
                c.set_authorizer(autorizar)
                conexiones.append(c)
                return c
            return abrir(ruta, **opciones)
        with patch.object(sqlite3, "connect", side_effect=conectar):
            yield conexiones, consultas, denegadas
        self.assertEqual(len(conexiones), 1)
        self.assertEqual([s for s, _ in consultas if s.startswith("BEGIN")], ["BEGIN"])
        self.assertFalse(denegadas)
        for c in conexiones:
            with self.assertRaises(sqlite3.ProgrammingError):
                c.execute("SELECT 1")

    @contextmanager
    def sin_calculos_actuales(self):
        with ExitStack() as pila:
            for modulo, nombre in ((motor, "_numero"), (comparacion, "_seleccion"), (snapshot, "_parametros"),
                                   (trayectoria, "calcular_trayectoria"), (comparacion, "comparar_escenarios"),
                                   (snapshot, "preparar_snapshot"), (registro.lectura, "_leer_pronostico_simulable_en_conexion"),
                                   (esquema, "preparar_esquema_simulaciones")):
                pila.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("No invocar operaciones actuales")))
            pila.enter_context(patch.object(motor, "VERSION_REGLA", "regla_futura"))
            pila.enter_context(patch.object(motor, "TECNICAS", ("tecnica_futura",)))
            yield

    def test_historial_vacio(self):
        self.assertEqual(self.listar(), {"total": 0, "limite": 20, "desplazamiento": 0, "ejecuciones": []})

    def test_detalle_individual_identico_al_registro(self):
        r = self.datos.individual(clave="privada-individual")
        esperado = {k: r[k] for k in ("ejecucion_id", "fecha_registro", "modalidad", "numero_escenarios", "historico")}
        self.assertEqual(snapshot.serializar_documento(self.detalle(r["ejecucion_id"])), snapshot.serializar_documento(esperado))
        self.assertEqual(len(self.detalle(r["ejecucion_id"])["historico"]["origen"]["detalles"]), 15)

    def test_detalles_ab_y_abc_integrales(self):
        for n in (2, 3):
            r = self.datos.comparar(clave=str(n), n=n)
            d = self.detalle(r["ejecucion_id"])
            self.assertEqual(d["historico"], r["historico"])
            self.assertEqual([e["etiqueta"] for e in d["historico"]["resultados"]], list("ABC"[:n]))
            self.assertTrue(all(len(e["simulacion"]["trayectoria"]) == 15 for e in d["historico"]["resultados"]))

    def test_listado_orden_desc_y_campos_historicos(self):
        r = self.basicos()
        l = self.listar()
        self.assertEqual(l["total"], 3)
        self.assertEqual([e["ejecucion_id"] for e in l["ejecuciones"]], [x["ejecucion_id"] for x in reversed(r)])
        self.assertEqual([e["etiquetas"] for e in l["ejecuciones"]], [["A", "B", "C"], ["A", "B"], ["A"]])
        for resumen, guardado in zip(l["ejecuciones"], reversed(r)):
            self.assertEqual(resumen["fecha_registro"], guardado["fecha_registro"])
            origen = guardado["historico"]["origen"]
            for campo in ("conjunto_id", "serie", "pronostico_id", "experimento_id", "tecnica", "fecha_corte"):
                self.assertEqual(resumen[campo], origen[campo])

    def test_filtros_individuales(self):
        self.todos()
        for filtro, esperado in (({"pronostico_id": 101}, 3), ({"conjunto_id": 2}, 1), ({"serie_id": 84}, 1),
                                 ({"modalidad": "individual"}, 3), ({"modalidad": "comparacion"}, 2)):
            with self.subTest(filtro=filtro):
                self.assertEqual(self.listar(**filtro)["total"], esperado)

    def test_filtros_combinados_y_sin_coincidencias(self):
        self.todos()
        self.assertEqual(self.listar(conjunto_id=7, serie_id=83, pronostico_id=101, modalidad="comparacion")["total"], 2)
        self.assertEqual(self.listar(conjunto_id=2, serie_id=83)["ejecuciones"], [])
        self.assertEqual(self.listar(pronostico_id=999)["total"], 0)

    def test_paginacion_limite_offset_y_total(self):
        r = self.todos()
        pagina = self.listar(limite=2, desplazamiento=1)
        self.assertEqual((pagina["total"], pagina["limite"], pagina["desplazamiento"]), (5, 2, 1))
        self.assertEqual([e["ejecucion_id"] for e in pagina["ejecuciones"]], [r[3]["ejecucion_id"], r[2]["ejecucion_id"]])
        self.assertEqual(self.listar(desplazamiento=5)["ejecuciones"], [])
        self.assertEqual(self.listar(desplazamiento=2**63-1)["total"], 5)
        self.assertEqual(len(self.listar(limite=50)["ejecuciones"]), 5)

    def test_paginacion_filtrada(self):
        r = self.basicos()
        p = self.listar(modalidad="comparacion", limite=1, desplazamiento=1)
        self.assertEqual(p["total"], 2)
        self.assertEqual(p["ejecuciones"][0]["ejecucion_id"], r[1]["ejecucion_id"])

    def test_identificadores_invalidos_antes_de_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for identificador in (0, -1, True, "1", 1.0, None, 2**63):
                self.error("ejecucion_invalida", self.detalle, identificador)

    def test_filtros_invalidos_antes_de_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for campo in ("pronostico_id", "conjunto_id", "serie_id"):
                for valor in (0, -1, True, "1", 1.0, 2**63):
                    with self.subTest(campo=campo, valor=valor):
                        self.error("filtro_invalido", self.listar, **{campo: valor})
            for valor in ("futuro", "INDIVIDUAL", "individual' OR 1=1", True, 1, []):
                self.error("filtro_invalido", self.listar, modalidad=valor)

    def test_paginacion_invalida_antes_de_sqlite(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for valor in (0, -1, 51, True, "20", 20.0, None):
                self.error("paginacion_invalida", self.listar, limite=valor)
            for valor in (-1, True, "0", 0.0, None, 2**63):
                self.error("paginacion_invalida", self.listar, desplazamiento=valor)

    def test_ejecucion_inexistente(self):
        self.error("ejecucion_inexistente", self.detalle, 999)

    def test_conjunto_inactivo_y_origen_actual_modificado(self):
        r = self.basicos()
        self.sql("UPDATE conjunto_datos SET activo=0")
        self.sql("UPDATE detalle_pronostico SET venta_estimada=999")
        self.sql("UPDATE experimento SET tecnica='TECNICA_CAMBIADA'")
        self.sql("UPDATE serie_temporal SET family='OTRA'")
        with self.vigilar(), self.sin_calculos_actuales():
            d = self.detalle(r[2]["ejecucion_id"])
        self.assertEqual(d["historico"], r[2]["historico"])
        with self.vigilar(), self.sin_calculos_actuales():
            l = self.listar(conjunto_id=7)
        self.assertEqual(l["total"], 3)
        self.assertEqual(l["ejecuciones"][0]["serie"]["family"], "FAMILIA A")
        self.assertNotEqual(l["ejecuciones"][0]["tecnica"], "TECNICA_CAMBIADA")

    def test_ceros_int_float_y_menos_cero_preservados(self):
        for i, valor in enumerate((0, 0.0, -0.0)):
            r = self.datos.individual(clave=str(i), inventario=valor)
            d = self.detalle(r["ejecucion_id"])
            leido = d["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]
            self.assertIs(type(leido), type(valor))
            self.assertEqual(leido.hex() if type(leido) is float else leido, valor.hex() if type(valor) is float else valor)

    def test_subnormales_y_residuos(self):
        for i, (valores, inventario) in enumerate((([math.ulp(0.0)]+[0.0]*14, 0), ([0.1]*3+[0.0]*12, 0.3))):
            self.datos.datos.crear_pronostico(102+i, 202+i, estimadas=valores)
            r = self.datos.individual(identificador=102+i, inventario=inventario, reserva=0, clave=str(i))
            q = self.detalle(r["ejecucion_id"])["historico"]["resultados"][0]["simulacion"]["reposicion"]
            self.assertGreater(q, 0)
            self.assertEqual(q.hex(), r["historico"]["resultados"][0]["simulacion"]["reposicion"].hex())

    def test_unica_conexion_ro_begin_y_sql_parametrizado(self):
        r = self.datos.individual()
        with self.vigilar() as (_, consultas, _):
            self.listar(pronostico_id=101, conjunto_id=7, serie_id=83, modalidad="individual", limite=2, desplazamiento=0)
        conteos = [(s, v) for s, v in consultas if s.startswith("SELECT COUNT")]
        self.assertEqual(conteos[0][1], (101, 7, 83, "individual"))
        self.assertNotIn("'individual'", conteos[0][0])
        with self.vigilar():
            self.detalle(r["ejecucion_id"])

    def test_reutiliza_lector_publicado_en_la_misma_conexion(self):
        r = self.datos.individual()
        with patch.object(registro, "_recuperar_en_conexion", wraps=registro._recuperar_en_conexion) as recuperar, self.vigilar() as (conexiones, _, _):
            self.detalle(r["ejecucion_id"])
        recuperar.assert_called_once()
        self.assertIs(recuperar.call_args.args[0], conexiones[0])
        guardado = self.datos.paquete_guardado("accion")
        self.assertEqual(recuperar.call_args.args[1:], ("accion", guardado["solicitud_json"]))

    def test_no_expone_claves_ni_datos_privados(self):
        r = self.basicos()
        for respuesta in (self.listar(), self.detalle(r[0]["ejecucion_id"])):
            texto = json.dumps(respuesta, allow_nan=False)
            for privado in ("privada-", "clave_idempotencia", "configuracion_json", "ruta_modelo", "PRIVADO", str(self.ruta)):
                self.assertNotIn(privado, texto)

    def test_consultas_no_cambian_base_ni_tablas(self):
        r = self.basicos()
        antes = self.ruta.read_bytes(), self.datos.tablas(), self.datos.antiguas()
        with self.vigilar(): self.listar()
        with self.vigilar(): self.detalle(r[0]["ejecucion_id"])
        self.assertEqual((self.ruta.read_bytes(), self.datos.tablas(), self.datos.antiguas()), antes)

    def test_hash_modificado_rechaza_detalle_y_pagina(self):
        r = self.basicos()
        self.sql("UPDATE escenario_simulacion SET sha256_resultado=? WHERE ejecucion_id=? AND etiqueta='A'", ("0"*64, r[1]["ejecucion_id"]))
        self.error("snapshot_corrupto", self.detalle, r[1]["ejecucion_id"])
        self.error("snapshot_corrupto", self.listar)

    def test_documento_adulterado_con_hash_recalculado(self):
        r = self.datos.individual()
        texto = self.sql("SELECT origen_json FROM ejecucion_simulacion")[0][0]
        d = json.loads(texto)
        d["detalles"][0]["venta_estimada"] = 999
        texto = snapshot.serializar_documento(d)
        self.sql("UPDATE ejecucion_simulacion SET origen_json=?,sha256_origen=?", (texto, snapshot.calcular_sha256(texto)))
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_tipos_y_no_finitos_con_hash_recalculado(self):
        r = self.datos.individual()
        original = self.sql("SELECT resultado_json FROM escenario_simulacion")[0][0]
        for campo, valor in (("reposicion", 285), ("horizonte", 15.0), ("recepcion_dentro_horizonte", 1)):
            d = json.loads(original)
            d[campo] = valor
            texto = snapshot.serializar_documento(d)
            self.sql("UPDATE escenario_simulacion SET resultado_json=?,sha256_resultado=?", (texto, snapshot.calcular_sha256(texto)))
            with self.subTest(campo=campo):
                self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])
        texto = original.replace('"reposicion":285.0', '"reposicion":1e999')
        self.assertNotEqual(texto, original)
        self.sql("UPDATE escenario_simulacion SET resultado_json=?,sha256_resultado=?", (texto, snapshot.calcular_sha256(texto)))
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_regla_o_campo_privado_adulterados(self):
        r = self.datos.individual()
        original = json.loads(self.sql("SELECT resultado_json FROM escenario_simulacion")[0][0])
        for cambio in ({"version_regla": "regla_desconocida"}, {"ruta_modelo": "PRIVADO C:/modelo"}):
            d = deepcopy(original)
            d.update(cambio)
            texto = snapshot.serializar_documento(d)
            self.sql("UPDATE escenario_simulacion SET resultado_json=?,sha256_resultado=?", (texto, snapshot.calcular_sha256(texto)))
            self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_hijo_faltante(self):
        r = self.datos.comparar(n=3)
        self.sql("DELETE FROM escenario_simulacion WHERE etiqueta='C'")
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_posicion_y_etiqueta_adulteradas(self):
        r = self.datos.individual()
        self.datos.datos.corromper("UPDATE escenario_simulacion SET etiqueta='D'")
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])
        self.datos.datos.corromper("UPDATE escenario_simulacion SET etiqueta='A',posicion=2")
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_json_no_canonico_o_duplicado_aun_con_hash_nuevo(self):
        r = self.datos.individual()
        for texto in ('{ "a":1}', '{"a":1,"a":2}'):
            self.sql("UPDATE ejecucion_simulacion SET origen_json=?,sha256_origen=?", (texto, snapshot.calcular_sha256(texto)))
            self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_version_documento_futura(self):
        r = self.datos.individual()
        self.sql("UPDATE ejecucion_simulacion SET version_documento=2")
        self.error("snapshot_incompatible", self.detalle, r["ejecucion_id"])

    def test_cabecera_invalida(self):
        r = self.datos.individual()
        self.datos.datos.corromper("UPDATE ejecucion_simulacion SET fecha_registro='PRIVADO'")
        with self.vigilar():
            self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_clave_interna_corrupta_no_se_publica(self):
        r = self.datos.individual()
        self.datos.datos.corromper("UPDATE ejecucion_simulacion SET clave_idempotencia=' '")
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])

    def test_esquema_sin_preparar_y_parcial(self):
        for t in reversed(esquema.TABLAS_NUEVAS): self.sql("DROP TABLE " + t)
        self.error("esquema_no_preparado", self.listar)
        self.sql(esquema._DDL[0])
        self.error("esquema_incompatible", self.listar)

    def test_esquema_adulterado_y_version_futura(self):
        self.sql("UPDATE version_esquema_simulaciones SET version=2")
        self.error("version_no_soportada", self.listar)
        self.sql("UPDATE version_esquema_simulaciones SET version=1")
        self.sql("CREATE UNIQUE INDEX ajeno ON ejecucion_simulacion(pronostico_id)")
        self.error("esquema_incompatible", self.listar)

    def test_base_ausente_danada_y_vacia(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        self.error("base_no_disponible", servicio.listar_ejecuciones_simulaciones, ausente)
        self.assertFalse(ausente.exists())
        danada = self.ruta.with_name("danada.sqlite3")
        danada.write_bytes(b"contenido sintetico no sqlite")
        self.error("base_no_disponible", servicio.consultar_ejecucion_simulacion, danada, 1)
        vacia = self.ruta.with_name("vacia.sqlite3")
        with closing(sqlite3.connect(vacia)): pass
        self.error("esquema_no_preparado", servicio.listar_ejecuciones_simulaciones, vacia)

    def test_rutas_invalidas(self):
        for ruta in (None, "", "a\0b", False, 1):
            self.error("base_no_disponible", servicio.listar_ejecuciones_simulaciones, ruta)

    def test_corrupcion_fuera_pagina_no_se_oculta_dentro(self):
        r = self.basicos()
        self.sql("DELETE FROM escenario_simulacion WHERE ejecucion_id=?", (r[0]["ejecucion_id"],))
        self.assertEqual(self.listar(limite=1)["total"], 3)
        self.error("snapshot_corrupto", self.listar, limite=1, desplazamiento=2)

    def test_error_inesperado_se_propaga_y_cierra(self):
        r = self.datos.individual()
        with self.vigilar(), patch.object(registro, "_recuperar_en_conexion", side_effect=RuntimeError("programacion")):
            with self.assertRaises(RuntimeError): self.detalle(r["ejecucion_id"])

    def test_importacion_sin_efectos(self):
        codigo = Path(servicio.__file__).read_text(encoding="utf-8")
        modulo = ModuleType("consulta_historial_aislada")
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")), \
             patch.object(esquema, "preparar_esquema_simulaciones", side_effect=AssertionError("No preparar")):
            exec(compile(codigo, "consulta_historial_aislada", "exec"), modulo.__dict__)
        self.assertTrue(callable(modulo.listar_ejecuciones_simulaciones))

    def test_wal_insercion_entre_total_y_pagina(self):
        r = self.basicos()
        self.assertEqual(self.sql("PRAGMA journal_mode=WAL")[0][0], "wal")
        permiso, terminado = Event(), Event()
        abrir = sqlite3.connect
        def insertar():
            if not permiso.wait(20): raise TimeoutError("Coordinacion agotada")
            try: return self.datos.individual(clave="concurrente")
            finally: terminado.set()
        class Coordinada(sqlite3.Connection):
            def execute(c, sql, parametros=()):
                cursor = super().execute(sql, parametros)
                if sql.startswith("SELECT COUNT(*) FROM ejecucion_simulacion"):
                    permiso.set()
                    if not terminado.wait(20): raise TimeoutError("Escritura concurrente agotada")
                return cursor
        def conectar(ruta, **opciones):
            return abrir(ruta, factory=Coordinada, **opciones)
        with ThreadPoolExecutor(max_workers=1) as pool:
            futuro = pool.submit(insertar)
            try:
                with patch.object(sqlite3, "connect", side_effect=conectar):
                    pagina = self.listar(limite=2, desplazamiento=1)
            finally: permiso.set()
            nuevo = futuro.result(timeout=25)
        self.assertEqual(pagina["total"], 3)
        self.assertEqual([x["ejecucion_id"] for x in pagina["ejecuciones"]], [r[1]["ejecucion_id"], r[0]["ejecucion_id"]])
        siguiente = self.listar()
        self.assertEqual(siguiente["total"], 4)
        self.assertEqual(siguiente["ejecuciones"][0]["ejecucion_id"], nuevo["ejecucion_id"])

    def test_wal_cabecera_e_hijos_misma_instantanea(self):
        r = self.datos.comparar(n=3)
        self.assertEqual(self.sql("PRAGMA journal_mode=WAL")[0][0], "wal")
        permiso, terminado = Event(), Event()
        original = registro._recuperar_en_conexion
        def modificar():
            if not permiso.wait(20): raise TimeoutError("Coordinacion agotada")
            try: self.sql("DELETE FROM escenario_simulacion WHERE ejecucion_id=? AND etiqueta='C'", (r["ejecucion_id"],))
            finally: terminado.set()
        def recuperar(*args):
            permiso.set()
            if not terminado.wait(20): raise TimeoutError("Cambio concurrente agotado")
            return original(*args)
        with ThreadPoolExecutor(max_workers=1) as pool:
            futuro = pool.submit(modificar)
            try:
                with patch.object(registro, "_recuperar_en_conexion", side_effect=recuperar):
                    detalle = self.detalle(r["ejecucion_id"])
            finally: permiso.set()
            futuro.result(timeout=25)
        self.assertEqual(detalle["historico"], r["historico"])
        self.assertEqual(len(detalle["historico"]["resultados"]), 3)
        self.error("snapshot_corrupto", self.detalle, r["ejecucion_id"])


if __name__ == "__main__":
    unittest.main()
