"""S4-A5-02: POST explícitos solo con bases SQLite temporales sintéticas."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
from threading import Barrier
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import simulaciones as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import registro_simulaciones as registro
from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.services import trayectoria_simulacion as trayectoria
from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import simulacion_desde_pronostico as lectura
from backend.app.storage import migraciones_simulacion as esquema
from tests import test_registro_simulaciones as fixtures


BASE = "/api/simulaciones"


def individual(**cambios):
    return dict(pronostico_id=101, inventario_disponible=30, tiempo_abastecimiento=0,
                reserva=5, clave_idempotencia="accion") | cambios


def comparada(n=2, **cambios):
    return dict(pronostico_id=101, escenarios=fixtures.escenarios(n), clave_idempotencia="accion") | cambios


class ApiRegistroSimulacionesTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.RegistroSimulacionesTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        anteriores = dict(app.dependency_overrides)
        def restaurar():
            app.dependency_overrides.clear()
            app.dependency_overrides.update(anteriores)
        self.addCleanup(restaurar)
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))

    def post(self, datos=None, modalidad="individual", **opciones):
        if "content" not in opciones:
            opciones["content"] = json.dumps(individual() if datos is None else datos, ensure_ascii=True).encode("utf-8")
        opciones.setdefault("headers", {"Content-Type": "application/json"})
        respuesta = self.cliente.post(BASE + "/" + modalidad, **opciones)
        self.assertEqual(respuesta.headers.get("cache-control"), "no-store")
        return respuesta

    def error(self, respuesta, codigo):
        self.assertEqual(respuesta.status_code, api.ERRORES_REGISTRO[codigo][0], respuesta.text)
        self.assertEqual(respuesta.json(), {"detail": {"codigo": codigo, "mensaje": api.ERRORES_REGISTRO[codigo][1]}})
        for privado in (str(self.ruta), "PRIVADO", "Traceback", "SELECT", "ruta_modelo", "configuracion_json"):
            self.assertNotIn(privado, respuesta.text)

    def exacto(self, a, b):
        self.assertEqual(snapshot.serializar_documento(a), snapshot.serializar_documento(b))

    def counts(self, cabeceras, hijos):
        self.assertEqual([len(t) for t in self.datos.tablas()], [cabeceras, hijos])

    def test_individual_201_confirmado_y_manual(self):
        antiguas = self.datos.antiguas()
        r = self.post()
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual((r.json()["estado_registro"], r.json()["hubo_escritura"], r.json()["numero_escenarios"]), ("creado", True, 1))
        sim = r.json()["historico"]["resultados"][0]["simulacion"]
        self.assertEqual(sim["reposicion"], 285)
        self.assertEqual(sim["resumen"]["balance_neto_final"], 5)
        self.assertEqual(len(sim["trayectoria"]), 15)
        self.counts(1, 1)
        self.assertEqual(self.datos.antiguas(), antiguas)

    def test_individual_200_mismos_ids_fecha_documentos(self):
        primero = self.post().json()
        tablas = self.datos.tablas()
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        esperado = primero | {"estado_registro": "reutilizado", "hubo_escritura": False}
        self.exacto(r.json(), esperado)
        self.assertEqual(self.datos.tablas(), tablas)
        self.counts(1, 1)

    def test_ab_creacion_y_reutilizacion(self):
        self.comprobar_comparacion(2)

    def test_abc_creacion_y_reutilizacion(self):
        self.comprobar_comparacion(3)

    def comprobar_comparacion(self, n):
        datos = comparada(n)
        r = self.post(datos, "comparacion")
        self.assertEqual(r.status_code, 201, r.text)
        primero, tablas = r.json(), self.datos.tablas()
        self.assertEqual([s["etiqueta"] for s in primero["historico"]["resultados"]], list("ABC"[:n]))
        self.assertEqual(primero["modalidad"], "comparacion")
        self.assertTrue(all(len(s["simulacion"]["trayectoria"]) == 15 for s in primero["historico"]["resultados"]))
        r = self.post(datos, "comparacion")
        self.assertEqual(r.status_code, 200, r.text)
        self.exacto(r.json(), primero | {"estado_registro": "reutilizado", "hubo_escritura": False})
        self.assertEqual(self.datos.tablas(), tablas)
        self.counts(1, n)

    def test_conflictos_parametros_pronostico_y_modalidad(self):
        self.post()
        antes = self.datos.tablas()
        for datos in (individual(reserva=6), individual(pronostico_id=999)):
            self.error(self.post(datos), "conflicto_idempotencia")
        self.error(self.post(comparada(), "comparacion"), "conflicto_idempotencia")
        self.assertEqual(self.datos.tablas(), antes)

    def test_clave_distinta_crea_otra_ejecucion(self):
        a, b = self.post(), self.post(individual(clave_idempotencia="otra"))
        self.assertEqual((a.status_code, b.status_code), (201, 201))
        self.assertNotEqual(a.json()["ejecucion_id"], b.json()["ejecucion_id"])
        self.exacto(a.json()["historico"], b.json()["historico"])
        self.counts(2, 2)

    def test_delegacion_individual_unica_con_originales(self):
        with patch.object(api, "registrar_simulacion", wraps=registro.registrar_simulacion) as espia:
            r = self.post(individual(inventario_disponible=-0.0, reserva=0))
        self.assertEqual(r.status_code, 201, r.text)
        espia.assert_called_once_with(self.ruta, 101, -0.0, 0, 0, clave_idempotencia="accion")
        self.assertIs(type(espia.call_args.args[2]), float)
        self.assertEqual(espia.call_args.args[2].hex(), (-0.0).hex())
        self.assertIs(type(espia.call_args.args[4]), int)

    def test_delegacion_comparacion_unica_sin_transformar(self):
        datos = comparada(3)
        datos["escenarios"][0]["reserva"] = -0.0
        antes = deepcopy(datos)
        with patch.object(api, "registrar_comparacion_simulaciones", wraps=registro.registrar_comparacion_simulaciones) as espia:
            r = self.post(datos, "comparacion")
        self.assertEqual(r.status_code, 201, r.text)
        espia.assert_called_once_with(self.ruta, 101, datos["escenarios"], clave_idempotencia="accion")
        self.exacto(espia.call_args.args[2], antes["escenarios"])
        self.assertEqual(datos, antes)

    def test_reintento_sin_dml_calculo_origen_o_validadores_actuales(self):
        for modalidad, datos in (("individual", individual()), ("comparacion", comparada(3, clave_idempotencia="abc"))):
            primero = self.post(datos, modalidad).json()
            antes = self.datos.tablas()
            self.datos.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
            self.datos.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=999 WHERE pronostico_id=101")
            with ExitStack() as pila:
                for modulo, nombre in ((motor, "_numero"), (comparacion, "_seleccion"), (snapshot, "_parametros"),
                                       (snapshot, "preparar_snapshot"), (trayectoria, "calcular_trayectoria"),
                                       (comparacion, "comparar_escenarios"), (lectura, "_leer_pronostico_simulable_en_conexion")):
                    pila.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("No llamar")))
                pila.enter_context(patch.object(motor, "VERSION_REGLA", "futura"))
                pila.enter_context(patch.object(motor, "TECNICAS", ("futura",)))
                with self.datos.vigilar() as (conexiones, llamadas, denegadas):
                    r = self.post(datos, modalidad)
            self.assertEqual(r.status_code, 200, r.text)
            self.exacto(r.json(), primero | {"estado_registro": "reutilizado", "hubo_escritura": False})
            self.assertEqual(self.datos.tablas(), antes)
            self.assertFalse(denegadas)
            self.assertEqual(len(conexiones), 1)
            self.assertFalse(any(s.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")) for _, s in llamadas))
            self.datos.datos.ejecutar("UPDATE conjunto_datos SET activo=CASE WHEN id=7 THEN 1 ELSE 0 END")

    def test_nueva_solicitud_conserva_validadores_vigentes(self):
        with patch.object(motor, "_numero", side_effect=motor.ErrorSimulacionReposicion("valor_invalido")) as validar:
            self.error(self.post(), "parametros_invalidos")
        self.assertTrue(validar.called)
        self.counts(0, 0)

    def test_identidad_int_float_cero_y_menos_cero(self):
        self.post(individual(inventario_disponible=0))
        for v in (0.0, -0.0):
            self.error(self.post(individual(inventario_disponible=v)), "conflicto_idempotencia")
        for i, v in enumerate((0, 0.0, -0.0)):
            r = self.post(individual(clave_idempotencia=str(i), inventario_disponible=v))
            self.assertEqual(r.status_code, 201, r.text)
            valor = r.json()["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]
            self.assertIs(type(valor), type(v))
            if type(v) is float: self.assertEqual(valor.hex(), v.hex())
        self.counts(4, 4)

    def test_subnormales_residuos_y_grandes_finitos(self):
        casos = (([math.ulp(0.0)]+[0.0]*14, 0), ([0.1]*3+[0.0]*12, 0.3), ([1e308]+[0.0]*14, 1e308))
        for i, (estimadas, inventario) in enumerate(casos):
            self.datos.datos.crear_pronostico(102+i, 202+i, estimadas=estimadas)
            datos = individual(pronostico_id=102+i, inventario_disponible=inventario, reserva=0, clave_idempotencia=str(i))
            r = self.post(datos)
            self.assertEqual(r.status_code, 201, r.text)
            h = r.json()["historico"]
            if i < 2: self.assertGreater(h["resultados"][0]["simulacion"]["reposicion"], 0)
            self.exacto(self.post(datos).json()["historico"], h)
            if type(inventario) is float:
                self.assertEqual(h["solicitud"]["escenarios"][0]["inventario_disponible"].hex(), inventario.hex())

    def test_enteros_grandes_exactos_y_precision_no_representable(self):
        r = self.post(individual(inventario_disponible=2**53, reserva=0))
        self.assertEqual(r.status_code, 201, r.text)
        self.assertIs(type(r.json()["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]), int)
        self.error(self.post(individual(inventario_disponible=2**53+1, clave_idempotencia="perdida")), "precision_no_representable")
        self.counts(1, 1)

    def test_decimales_no_nulos_subdesbordados_se_rechazan(self):
        with patch.object(api, "registrar_simulacion") as espia:
            for token in ("1e-9999", "-1e-9999", "0.00001e-9999", "2e-324"):
                contenido = json.dumps(individual()).replace('"inventario_disponible": 30', '"inventario_disponible": '+token)
                self.error(self.post(content=contenido), "precision_no_representable")
        espia.assert_not_called()

    def test_ceros_con_exponente_y_subnormal_minimo(self):
        for i, token in enumerate(("0e-9999", "-0.0e-9999", "5e-324")):
            contenido = json.dumps(individual(clave_idempotencia=str(i))).replace('"inventario_disponible": 30', '"inventario_disponible": '+token)
            r = self.post(content=contenido)
            self.assertEqual(r.status_code, 201, r.text)
            valor = r.json()["historico"]["solicitud"]["escenarios"][0]["inventario_disponible"]
            self.assertIs(type(valor), float)
            self.assertEqual(valor.hex(), float(token).hex())

    def test_enteros_fuera_de_binary64_se_rechazan_sin_convertir(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            self.error(self.post(individual(inventario_disponible=10**400)), "parametros_invalidos")
            contenido = json.dumps(individual()).replace('"reserva": 5', '"reserva": '+'9'*5000)
            self.error(self.post(content=contenido), "parametros_invalidos")

    def test_identificadores_invalidos(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
            for pid in (0, -1, True, "101", 101.0, None, 2**63):
                self.error(self.post(individual(pronostico_id=pid)), "pronostico_invalido")

    def test_clave_invalida_y_limite_utf8(self):
        for clave in (None, 1, True, "", " ", "a\0b", "é"*65, "\ud800"):
            with self.subTest(clave=repr(clave)), patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir")):
                self.error(self.post(individual(clave_idempotencia=clave)), "clave_idempotencia_invalida")
        r = self.post(individual(clave_idempotencia="é"*64))
        self.assertEqual(r.status_code, 201, r.text)
        self.assertNotIn("clave_idempotencia", r.text)

    def test_cuerpo_ausente_invalido_o_raiz_no_objeto(self):
        with patch.object(api, "registrar_simulacion") as espia:
            for contenido in (b"", b" ", b"{", b'{"a":}', b'{} basura', b'\xff', b'\xef\xbb\xbf{}'):
                self.error(self.post(content=contenido), "entrada_invalida")
            for contenido in (b"[]", b'"texto"', b"42", b"true", b"null"):
                self.error(self.post(content=contenido), "solicitud_invalida")
        espia.assert_not_called()

    def test_content_type_estricto(self):
        with patch.object(api, "registrar_simulacion") as espia:
            for headers in ({}, {"Content-Type": "text/plain"}, {"Content-Type": "application/problem+json"},
                            {"Content-Type": "application/json; charset=latin-1"}, {"Content-Type": "application/json; foo=x"},
                            [("Content-Type", "application/json"), ("Content-Type", "application/json")]):
                self.error(self.post(headers=headers), "entrada_invalida")
        espia.assert_not_called()
        for i, tipo in enumerate(('application/json; charset=UTF-8', 'Application/JSON; charset="utf-8"')):
            self.assertEqual(self.post(individual(clave_idempotencia=str(i)), headers={"Content-Type": tipo}).status_code, 201)

    def test_duplicados_raiz_y_escenario(self):
        a = json.dumps(individual())
        b = json.dumps(comparada())
        casos = (("individual", a[:-1]+',"reserva":5}'),
                 ("individual", a[:-1]+',"reser\\u0076a":5}'),
                 ("comparacion", b.replace('"etiqueta": "A"', '"etiqueta": "A", "etiqueta": "A"')))
        for modalidad, contenido in casos:
            with patch.object(api, "registrar_simulacion") as uno, patch.object(api, "registrar_comparacion_simulaciones") as varios:
                self.error(self.post(content=contenido, modalidad=modalidad), "entrada_invalida")
            uno.assert_not_called()
            varios.assert_not_called()

    def test_constantes_no_finitas_y_exponentes_fuera_de_rango(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"):
            contenido = json.dumps(individual()).replace('"reserva": 5', '"reserva": '+token)
            self.error(self.post(content=contenido), "entrada_invalida" if "Infinity" in token or token == "NaN" else "parametros_invalidos")
        self.counts(0, 0)

    def test_limite_bytes_y_un_byte_de_exceso(self):
        base = json.dumps(individual()).encode("utf-8")
        completo = base + b" "*(api.MAX_BYTES_CUERPO-len(base))
        self.assertEqual(len(completo), 16384)
        self.assertEqual(self.post(content=completo).status_code, 201)
        with patch.object(api, "registrar_simulacion") as espia:
            self.error(self.post(content=completo+b" "), "entrada_invalida")
        espia.assert_not_called()

    def test_limite_stream_sin_content_length_y_unicode_multibyte(self):
        base = json.dumps(individual(clave_idempotencia="é"*64), ensure_ascii=False).encode("utf-8")
        def bloques(extra):
            yield base
            yield b" "*(api.MAX_BYTES_CUERPO-len(base)+extra)
        self.assertEqual(self.post(content=bloques(0)).status_code, 201)
        self.error(self.post(content=bloques(1)), "entrada_invalida")

    def test_profundidad_json_excesiva(self):
        self.error(self.post(content=b'{"reserva":'+b'['*1100+b'0'+b']'*1100+b'}'), "entrada_invalida")

    def test_campos_faltantes_y_adicionales(self):
        for campo in individual():
            datos = individual()
            del datos[campo]
            self.error(self.post(datos), "solicitud_invalida")
        self.error(self.post(individual(extra=1)), "solicitud_invalida")
        self.error(self.post(comparada(extra=1), "comparacion"), "solicitud_invalida")

    def test_parametros_strings_booleanos_objetos_y_plazos(self):
        for campo in ("inventario_disponible", "reserva", "tiempo_abastecimiento"):
            for valor in (True, "1", None, {}, [], -1):
                self.error(self.post(individual(**{campo: valor})), "parametros_invalidos")
        for plazo in (1.0, 0.0, 16):
            self.error(self.post(individual(tiempo_abastecimiento=plazo)), "parametros_invalidos")
        self.counts(0, 0)

    def test_escenarios_incompletos_extra_tipos_y_etiquetas(self):
        casos = ([], fixtures.escenarios(1), fixtures.escenarios(3)+[fixtures.escenarios()[0]],
                 fixtures.escenarios()[::-1], [fixtures.escenarios()[0]]*2,
                 [{"etiqueta": "A"}, fixtures.escenarios()[1]], "AB", None,
                 [fixtures.escenarios()[0] | {"extra": 1}, fixtures.escenarios()[1]],
                 [fixtures.escenarios()[0] | {"etiqueta": "a"}, fixtures.escenarios()[1]])
        for escenarios in casos:
            self.error(self.post(comparada(escenarios=escenarios), "comparacion"), "escenarios_invalidos")
        e = fixtures.escenarios()
        e[1]["reserva"] = True
        self.error(self.post(comparada(escenarios=e), "comparacion"), "parametros_invalidos")
        self.counts(0, 0)

    def test_query_rechazada_antes_de_servicio(self):
        for modalidad in ("individual", "comparacion"):
            self.error(self.post(comparada() if modalidad == "comparacion" else individual(), modalidad, params={"limite": 1}), "entrada_invalida")
        self.counts(0, 0)

    def test_pronostico_inexistente_inactivo_y_sin_activo(self):
        self.error(self.post(individual(pronostico_id=999)), "pronostico_inexistente")
        self.datos.datos.crear_pronostico(102, 202, serie=24)
        self.error(self.post(individual(pronostico_id=102)), "pronostico_inactivo")
        self.datos.datos.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.error(self.post(), "sin_conjunto_activo")

    def test_tipo_prueba_futuro_y_corte_no_autorizado(self):
        for i, tipo in enumerate(("prueba", "futuro")):
            self.datos.datos.crear_pronostico(102+i, 202+i, tipo=tipo)
            self.error(self.post(individual(pronostico_id=102+i)), "tipo_no_evaluable")
        self.datos.datos.crear_pronostico(104, 204, corte="2017-06-15")
        self.error(self.post(individual(pronostico_id=104)), "corte_no_autorizado")

    def test_pronostico_corrupto(self):
        self.datos.datos.ejecutar("DELETE FROM detalle_pronostico WHERE paso=15")
        self.error(self.post(), "pronostico_corrupto")
        self.counts(0, 0)

    def test_esquema_no_preparado_y_parcial(self):
        for t in reversed(esquema.TABLAS_NUEVAS): self.datos.datos.ejecutar("DROP TABLE "+t)
        self.error(self.post(), "esquema_no_preparado")
        self.datos.datos.ejecutar(esquema._DDL[0])
        self.error(self.post(), "esquema_incompatible")

    def test_base_ausente_y_danada(self):
        for nombre, contenido in (("ausente.sqlite3", None), ("danada.sqlite3", b"no sqlite sintetica")):
            ruta = self.ruta.with_name(nombre)
            if contenido is not None: ruta.write_bytes(contenido)
            app.dependency_overrides[obtener_ruta_base] = lambda: ruta
            self.error(self.post(), "base_no_disponible")
            if contenido is None: self.assertFalse(ruta.exists())

    def test_precision_y_desbordamiento_de_resultados(self):
        for i, (valores, codigo) in enumerate((([1e16,1.0]+[0.0]*13,"precision_no_representable"), ([1e308,1e308]+[0.0]*13,"resultado_no_finito"))):
            self.datos.datos.crear_pronostico(102+i, 202+i, estimadas=valores)
            self.error(self.post(individual(pronostico_id=102+i,inventario_disponible=0,reserva=0)), codigo)
        self.counts(0, 0)

    def test_origen_modificado_durante_calculo(self):
        original = trayectoria.calcular_trayectoria
        def cambiar(*args):
            self.datos.datos.ejecutar("UPDATE detalle_pronostico SET venta_estimada=11 WHERE paso=1")
            return original(*args)
        with patch.object(trayectoria, "calcular_trayectoria", side_effect=cambiar):
            self.error(self.post(), "origen_modificado")
        self.counts(0, 0)

    def comprobar_rollback(self, momento):
        self.post(individual(clave_idempotencia="conservar"))
        antes = self.datos.tablas()
        abrir = sqlite3.connect
        conexiones = []
        class Fallida(sqlite3.Connection):
            hijos = 0
            cerrada = False
            def execute(c, sql, parametros=()):
                cursor = super().execute(sql, parametros)
                if sql.startswith("INSERT INTO escenario_simulacion"):
                    c.hijos += 1
                    if momento == "hijo" and c.hijos == 1: raise sqlite3.OperationalError("PRIVADO SELECT ruta")
                return cursor
            def commit(c):
                if momento == "commit": raise sqlite3.OperationalError("PRIVADO")
                return super().commit()
            def close(c):
                c.cerrada = True
                return super().close()
        def conectar(ruta, **opciones):
            c = abrir(ruta, factory=Fallida, **opciones)
            conexiones.append(c)
            return c
        with patch.object(sqlite3,"connect",side_effect=conectar):
            self.error(self.post(comparada(3, clave_idempotencia="revertida"),"comparacion"),"error_persistencia")
        self.assertTrue(all(c.cerrada for c in conexiones))
        self.assertEqual(self.datos.tablas(), antes)
        r = self.post(comparada(3,clave_idempotencia="revertida"),"comparacion")
        self.assertEqual(r.status_code,201,r.text)
        self.counts(2,4)

    def test_rollback_despues_del_primer_hijo(self): self.comprobar_rollback("hijo")

    def test_rollback_antes_de_commit(self): self.comprobar_rollback("commit")

    def carrera(self, misma_clave=True, mismos_parametros=True, wal=False):
        if wal: self.assertEqual(self.datos.datos.ejecutar("PRAGMA journal_mode=WAL")[0][0], "wal")
        barrera = Barrier(2)
        original = trayectoria.calcular_trayectoria
        def calcular(*args):
            barrera.wait(timeout=30)
            return original(*args)
        def ejecutar(i):
            with TestClient(app) as cliente:
                datos = individual(clave_idempotencia="accion" if misma_clave else str(i), inventario_disponible=0 if mismos_parametros else i)
                r = cliente.post(BASE+"/individual", json=datos)
                self.assertEqual(r.headers.get("cache-control"),"no-store")
                return r
        with patch.object(trayectoria,"calcular_trayectoria",side_effect=calcular), ThreadPoolExecutor(max_workers=2) as pool:
            futuros=[pool.submit(ejecutar,i) for i in range(2)]
            resultados=[f.result(timeout=45) for f in futuros]
        return resultados

    def test_concurrencia_misma_clave_solicitud(self):
        r=self.carrera()
        self.assertEqual(sorted(x.status_code for x in r),[200,201])
        self.assertEqual(r[0].json()["ejecucion_id"],r[1].json()["ejecucion_id"])
        self.counts(1,1)

    def test_concurrencia_conflicto_misma_clave(self):
        r=self.carrera(mismos_parametros=False)
        self.assertEqual(sorted(x.status_code for x in r),[201,409])
        self.error(next(x for x in r if x.status_code==409),"conflicto_idempotencia")
        self.counts(1,1)

    def test_concurrencia_claves_distintas_en_wal(self):
        r=self.carrera(misma_clave=False,wal=True)
        self.assertEqual([x.status_code for x in r],[201,201])
        self.assertNotEqual(r[0].json()["ejecucion_id"],r[1].json()["ejecucion_id"])
        self.counts(2,2)

    def test_get_despues_de_post_es_integro_y_no_registra(self):
        r=self.post(comparada(3),"comparacion")
        antes=self.datos.tablas()
        with patch.object(api,"registrar_simulacion",side_effect=AssertionError("No registrar")), patch.object(api,"registrar_comparacion_simulaciones",side_effect=AssertionError("No registrar")):
            listado=self.cliente.get(BASE+"/historial")
            detalle=self.cliente.get(BASE+"/historial/"+str(r.json()["ejecucion_id"]))
            rechazado=self.cliente.request("GET",BASE+"/historial",content=b"{}")
        self.assertEqual((listado.status_code,detalle.status_code,rechazado.status_code),(200,200,422))
        self.exacto(detalle.json()["historico"],r.json()["historico"])
        self.assertEqual(self.datos.tablas(),antes)

    def test_catalogo_completo_y_sanitizacion(self):
        self.assertLessEqual(set(registro._MENSAJES),set(api.ERRORES_REGISTRO))
        for codigo in registro._MENSAJES:
            error=registro.ErrorRegistroSimulacion(codigo)
            error.args=("PRIVADO SELECT C:/ruta",)
            for funcion,modo,datos in (("registrar_simulacion","individual",individual()),("registrar_comparacion_simulaciones","comparacion",comparada())):
                with self.subTest(codigo=codigo,modo=modo),patch.object(api,funcion,side_effect=error):
                    self.error(self.post(datos,modo),codigo)

    def test_error_inesperado_y_codigo_desconocido(self):
        error=registro.ErrorRegistroSimulacion("parametros_invalidos")
        error.codigo="PRIVADO"
        for causa in (RuntimeError("PRIVADO"),error):
            with patch.object(api,"registrar_simulacion",side_effect=causa):
                self.error(self.post(),"error_interno")

    def test_respuesta_servicio_contradictoria_rechazada(self):
        r=self.datos.individual()
        r["hubo_escritura"]=False
        with patch.object(api,"registrar_simulacion",return_value=r):
            self.error(self.post(),"error_interno")

    def test_sin_ddl_preparacion_ni_lecturas_privadas(self):
        antiguas=self.datos.antiguas()
        with self.datos.vigilar() as (conexiones,llamadas,denegadas),patch.object(esquema,"preparar_esquema_simulaciones",side_effect=AssertionError("No preparar")):
            r=self.post()
        self.assertEqual(r.status_code,201,r.text)
        self.assertEqual([uri.rsplit("?",1)[1] for uri,_ in conexiones],["mode=ro","mode=rw"])
        self.assertEqual([s for _,s in llamadas if s.startswith("BEGIN")],["BEGIN","BEGIN IMMEDIATE"])
        self.assertFalse(any(s.startswith(("CREATE","ALTER","DROP")) for _,s in llamadas))
        self.assertFalse(denegadas)
        self.assertEqual(self.datos.antiguas(),antiguas)

    def test_importacion_y_openapi_sin_efectos(self):
        with patch.object(sqlite3,"connect",side_effect=AssertionError("No abrir")),patch.object(esquema,"preparar_esquema_simulaciones",side_effect=AssertionError("No preparar")),patch.object(trayectoria,"calcular_trayectoria",side_effect=AssertionError("No calcular")):
            codigo=Path(api.__file__).read_text(encoding="utf-8")
            modulo=ModuleType("router_aislado")
            exec(compile(codigo,"router_aislado","exec"),modulo.__dict__)
            doc=app.openapi()
        self.assertIn(BASE+"/individual",doc["paths"])

    def test_openapi_completo_y_get_conservados(self):
        doc=self.cliente.get("/openapi.json").json()
        for ruta,modelo in (("individual","SolicitudIndividual"),("comparacion","SolicitudComparacion")):
            op=doc["paths"][BASE+"/"+ruta]
            self.assertEqual(set(op),{"post"})
            op=op["post"]
            self.assertEqual(set(op["responses"]),{"200","201","422","404","409","503","500"})
            self.assertTrue(op["requestBody"]["required"])
            self.assertEqual(op["requestBody"]["content"]["application/json"]["schema"]["$ref"],"#/components/schemas/"+modelo)
            for respuesta in op["responses"].values(): self.assertIn("Cache-Control",respuesta["headers"])
        for nombre in ("SolicitudIndividual","SolicitudComparacion","EscenarioRegistro","RegistroSimulacion"):
            self.assertFalse(doc["components"]["schemas"][nombre]["additionalProperties"])
        for ruta in ("/historial","/historial/{ejecucion_id}"):
            self.assertEqual(set(doc["paths"][BASE+ruta]),{"get"})
        self.assertEqual(self.cliente.get("/health").json(),{"status":"ok"})


if __name__ == "__main__":
    unittest.main()
