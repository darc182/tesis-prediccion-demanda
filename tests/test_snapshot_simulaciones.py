"""Snapshots puros con objetos sintéticos; SQLite solo en memoria para contrato SQL."""

from contextlib import ExitStack, closing
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from backend.app.services import snapshot_simulaciones as snapshot
from backend.app.services import simulacion_reposicion as motor
from backend.app.services import trayectoria_simulacion as trayectoria
from backend.app.services import comparacion_simulaciones as comparacion
from backend.app.storage import migraciones_simulacion as esquema


def fixture(n=1, valores=None, inventario=30, reserva=5):
    p = {"id": 101, "experimento_id": 201, "tecnica": "random_forest", "fecha_corte": "2017-06-16", "horizonte": 15}
    d = [{"paso": i + 1, "fecha_objetivo": (date(2017, 6, 17) + timedelta(days=i)).isoformat(), "venta_estimada": v}
         for i, v in enumerate([10] * 15 if valores is None else valores)]
    procedencia = {"conjunto_id": 7, "serie": {"id": 83, "store_nbr": 17, "family": "SINTÉTICA"}}
    escenarios = [{"etiqueta": e, "inventario_disponible": inventario, "tiempo_abastecimiento": plazo, "reserva": reserva}
                  for e, plazo in zip("ABC"[:n], (0, 2, 15))]
    r = (trayectoria.calcular_trayectoria(p, d, inventario, 0, reserva) if n == 1
         else comparacion.comparar_escenarios(p, d, escenarios))
    return p, d, procedencia, escenarios, r


def paquete(n=1, **opciones):
    return snapshot.preparar_snapshot(*fixture(n, **opciones), modalidad="individual" if n == 1 else "comparacion")


class SnapshotSimulacionesTests(unittest.TestCase):
    def error(self, funcion, *args, codigo=None, **kwargs):
        with self.assertRaises(snapshot.ErrorSnapshotSimulacion) as capturado:
            funcion(*args, **kwargs)
        if codigo is not None:
            self.assertEqual(capturado.exception.codigo, codigo)
        for privado in ("PRIVADO", "SELECT", "Traceback", "ruta_modelo"):
            self.assertNotIn(privado, str(capturado.exception))

    def leer(self, texto, version=1, huella=...):
        return snapshot.leer_documento(texto, snapshot.calcular_sha256(texto) if huella is ... else huella,
                                       version_documento=version)

    def adulterar(self, p, documento, cambiar, indice=0):
        destino = p if documento != "resultado" else p["escenarios"][indice]
        valor = json.loads(destino[documento + "_json"])
        cambiar(valor)
        texto = snapshot.serializar_documento(valor)
        destino[documento + "_json"] = texto
        destino["sha256_" + documento] = snapshot.calcular_sha256(texto)

    def test_individual_completo_y_ejemplo_manual(self):
        p = paquete()
        documentos = snapshot.verificar_paquete(p)
        r = documentos["resultados"][0]["simulacion"]
        self.assertEqual((r["demanda_considerada"], r["reposicion"], r["resumen"]["balance_neto_final"]), (150, 125, 5))
        self.assertEqual(len(documentos["origen"]["detalles"]), 15)
        self.assertEqual(r["trayectoria"][0]["balance_neto"], 145)
        self.assertEqual(documentos["advertencias"], r["advertencias"])
        self.assertEqual(p["numero_escenarios"], 1)
        self.assertEqual(p["escenarios"][0]["etiqueta"], "A")

    def test_comparacion_ab_integra(self):
        datos = fixture(2)
        p = snapshot.preparar_snapshot(*datos, modalidad="comparacion")
        d = snapshot.verificar_paquete(p)
        self.assertEqual(d["resultados"], datos[-1]["resultados"])
        self.assertEqual(d["advertencias"], datos[-1]["advertencias_comparacion"])
        self.assertEqual([f["etiqueta"] for f in p["escenarios"]], ["A", "B"])

    def test_comparacion_abc_integra(self):
        datos = fixture(3)
        d = snapshot.verificar_paquete(snapshot.preparar_snapshot(*datos, modalidad="comparacion"))
        self.assertEqual(d["resultados"], datos[-1]["resultados"])
        self.assertFalse(d["resultados"][2]["simulacion"]["recepcion_dentro_horizonte"])

    def test_procedencia_integrada_a3(self):
        datos = list(fixture(2))
        datos[-1].update(deepcopy(datos[2]))
        snapshot.verificar_paquete(snapshot.preparar_snapshot(*datos, modalidad="comparacion"))
        datos[-1]["serie"]["id"] = 99
        self.error(snapshot.preparar_snapshot, *datos, modalidad="comparacion", codigo="integridad_invalida")

    def test_tres_tecnicas(self):
        for tecnica in motor.TECNICAS:
            with self.subTest(tecnica=tecnica):
                datos = list(fixture())
                datos[0]["tecnica"] = datos[-1]["tecnica"] = tecnica
                snapshot.verificar_paquete(snapshot.preparar_snapshot(*datos, modalidad="individual"))

    def test_fechas_date_y_secuencias_tuple_adaptadas(self):
        datos = list(fixture())
        datos[0]["fecha_corte"] = date(2017, 6, 16)
        for d in datos[1]:
            d["fecha_objetivo"] = date.fromisoformat(d["fecha_objetivo"])
        datos[1], datos[3] = tuple(datos[1]), tuple(datos[3])
        p = snapshot.preparar_snapshot(*datos, modalidad="individual")
        self.assertEqual(snapshot.verificar_paquete(p)["origen"]["fecha_corte"], "2017-06-16")

    def test_experimento_opcional_nulo(self):
        datos = list(fixture())
        del datos[0]["experimento_id"]
        datos[-1]["experimento_id"] = None
        p = snapshot.preparar_snapshot(*datos, modalidad="individual")
        self.assertIsNone(snapshot.verificar_paquete(p)["origen"]["experimento_id"])

    def test_bytes_canonicos_y_oraculo_hash_independiente(self):
        texto = snapshot.serializar_documento({"z": 0.1, "á": "ñ", "a": [1, 1.0, -0.0]})
        esperado = '{"a":[1,1.0,-0.0],"z":0.1,"á":"ñ"}'
        self.assertEqual(texto.encode("utf-8"), esperado.encode("utf-8"))
        self.assertEqual(snapshot.calcular_sha256(texto), hashlib.sha256(esperado.encode("utf-8")).hexdigest())
        self.assertEqual(snapshot.calcular_sha256("{}"), "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a")

    def test_repetibilidad_y_orden_claves(self):
        self.assertEqual(paquete(3), paquete(3))
        self.assertEqual(snapshot.serializar_documento({"b": 2, "a": 1}), snapshot.serializar_documento({"a": 1, "b": 2}))

    def test_ida_vuelta_binary64_extremos(self):
        valores = [0.1, 0.3, 3 * 0.1 - 0.3, math.ulp(0.0), -math.ulp(0.0), sys.float_info.max, -sys.float_info.max, 0.0, -0.0]
        leidos = self.leer(snapshot.serializar_documento({"valores": valores}))["valores"]
        self.assertEqual([v.hex() for v in leidos], [v.hex() for v in valores])

    def test_residuo_binario_real_del_motor(self):
        p = paquete(valores=[0.1] * 3 + [0] * 12, inventario=0.3, reserva=0)
        d = snapshot.verificar_paquete(p)
        q = d["resultados"][0]["simulacion"]["reposicion"]
        self.assertGreater(q, 0)
        self.assertEqual(Fraction.from_float(q), 3 * Fraction.from_float(0.1) - Fraction.from_float(0.3))

    def test_subnormal_real_del_motor(self):
        minimo = math.ulp(0.0)
        p = paquete(valores=[minimo] + [0.0] * 14, inventario=0, reserva=0)
        self.assertEqual(snapshot.verificar_paquete(p)["resultados"][0]["simulacion"]["reposicion"].hex(), minimo.hex())

    def test_magnitud_grande_real_del_motor(self):
        p = paquete(valores=[1e308] + [0.0] * 14, inventario=1e308, reserva=0)
        self.assertEqual(snapshot.verificar_paquete(p)["resultados"][0]["simulacion"]["demanda_considerada"].hex(), (1e308).hex())

    def test_balances_negativos_reales(self):
        d = snapshot.verificar_paquete(paquete(3, inventario=0, reserva=0))
        self.assertEqual(d["resultados"][2]["simulacion"]["resumen"]["balance_neto_final"], -150)

    def test_menos_cero_solicitud_y_origen_conservado(self):
        p = paquete(valores=[-0.0] * 15, inventario=-0.0, reserva=-0.0)
        d = snapshot.verificar_paquete(p)
        self.assertEqual(d["solicitud"]["escenarios"][0]["inventario_disponible"].hex(), (-0.0).hex())
        self.assertEqual(d["origen"]["detalles"][0]["venta_estimada"].hex(), (-0.0).hex())
        self.assertEqual(d["resultados"][0]["simulacion"]["parametros"]["inventario_disponible"].hex(), (0.0).hex())

    def test_identidad_solicitud_int_float_y_signo_cero(self):
        hashes = [paquete(inventario=v)["sha256_solicitud"] for v in (0, 0.0, -0.0)]
        self.assertEqual(len(set(hashes)), 3)

    def test_rechaza_tipos_no_json_sin_convertir(self):
        for v in (Decimal("0.1"), Fraction(1, 3), (1, 2), date(2017, 6, 16), {1: "dato"}):
            with self.subTest(valor=v):
                self.error(snapshot.serializar_documento, {"v": v}, codigo="estructura_invalida")

    def test_nan_infinito_en_cualquier_profundidad(self):
        for v in (math.nan, math.inf, -math.inf):
            with self.subTest(valor=v):
                self.error(snapshot.serializar_documento, {"v": [{"x": v}]}, codigo="representacion_invalida")

    def test_json_constantes_y_desbordamiento(self):
        for texto in ('{"v":NaN}', '{"v":Infinity}', '{"v":-Infinity}', '{"v":1e999}'):
            with self.subTest(texto=texto):
                self.error(self.leer, texto, codigo="representacion_invalida")

    def test_subdesbordamiento_textual_no_se_repara(self):
        self.error(self.leer, '{"v":1e-9999}', codigo="integridad_invalida")

    def test_duplicados_raiz_y_anidados(self):
        for texto in ('{"v":1,"v":2}', '{"v":[{"x":1,"x":2}]}', '{"a":1,"\\u0061":2}'):
            with self.subTest(texto=texto):
                self.error(self.leer, texto, codigo="integridad_invalida")

    def test_json_invalido_o_raiz_escalar(self):
        for texto in ("{", "[]x", "null", "42", '"dato"', "true"):
            with self.subTest(texto=texto):
                self.error(self.leer, texto, codigo="estructura_invalida")

    def test_no_canonicos_no_se_reparan(self):
        for texto in ('{ "a":1}', '{"b":2,"a":1}', '{"a":1e0}', '{"a":"\\u00f1"}', '{}\n'):
            with self.subTest(texto=texto):
                self.error(self.leer, texto, codigo="integridad_invalida")

    def test_hash_alterado_y_formato(self):
        for huella in ("0" * 64, "A" * 64, "a" * 63, "g" * 64, b"0" * 64, None):
            with self.subTest(huella=huella):
                self.error(self.leer, "{}", huella=huella, codigo="integridad_invalida")

    def test_version_desconocida_y_tipos(self):
        for version in (0, 2, True, 1.0, "1", None):
            with self.subTest(version=version):
                self.error(self.leer, "{}", version, codigo="version_no_soportada")

    def test_limite_exacto_y_un_byte_exceso(self):
        n = snapshot.MAX_BYTES_DOCUMENTO
        texto = snapshot.serializar_documento({"a": "a" * (n - 8)})
        self.assertEqual(len(texto.encode()), n)
        self.assertEqual(self.leer(texto)["a"], "a" * (n - 8))
        self.error(snapshot.serializar_documento, {"a": "a" * (n - 7)}, codigo="documento_demasiado_grande")
        self.error(self.leer, texto + " ", codigo="documento_demasiado_grande")

    def test_limite_unicode_multibyte(self):
        n = snapshot.MAX_BYTES_DOCUMENTO
        valor = "é" * ((n - 8) // 2)
        texto = snapshot.serializar_documento({"a": valor})
        self.assertEqual(len(texto.encode("utf-8")), n)
        self.assertEqual(self.leer(texto)["a"], valor)
        self.error(snapshot.serializar_documento, {"a": valor + "a"}, codigo="documento_demasiado_grande")

    def test_unicode_no_normalizado_y_surrogates(self):
        self.assertNotEqual(snapshot.calcular_sha256(snapshot.serializar_documento({"a": "é"})),
                            snapshot.calcular_sha256(snapshot.serializar_documento({"a": "e\u0301"})))
        self.error(snapshot.serializar_documento, {"a": "\ud800"}, codigo="representacion_invalida")

    def test_anidados_ciclos_y_profundidad(self):
        documento = {"a": [{"b": [0.1, -0.0]}]}
        self.assertEqual(self.leer(snapshot.serializar_documento(documento)), documento)
        ciclo = []
        ciclo.append(ciclo)
        self.error(snapshot.serializar_documento, ciclo, codigo="estructura_invalida")
        for _ in range(snapshot.MAX_PROFUNDIDAD + 1):
            documento = [documento]
        self.error(snapshot.serializar_documento, documento, codigo="estructura_invalida")

    def test_campos_paquete_faltantes_y_adicionales(self):
        p = paquete()
        for campo in p:
            alterado = deepcopy(p)
            del alterado[campo]
            with self.subTest(campo=campo):
                self.error(snapshot.verificar_paquete, alterado, codigo="estructura_invalida")
        p["ruta_modelo"] = "PRIVADO"
        self.error(snapshot.verificar_paquete, p, codigo="estructura_invalida")

    def test_campos_documentos_faltantes_y_adicionales(self):
        for nombre, campo in (("solicitud", "modalidad"), ("origen", "serie"), ("contexto", "version_regla"), ("resultado", "resumen")):
            for adicional in (False, True):
                p = paquete()
                def cambiar(d):
                    if adicional:
                        d["ruta_modelo"] = "PRIVADO"
                    else:
                        del d[campo]
                self.adulterar(p, nombre, cambiar)
                with self.subTest(nombre=nombre, adicional=adicional):
                    self.error(snapshot.verificar_paquete, p, codigo="estructura_invalida")

    def test_identificadores_adulterados_con_hash_nuevo(self):
        for nombre in ("solicitud", "contexto", "resultado"):
            p = paquete()
            self.adulterar(p, nombre, lambda d: d.update(pronostico_id=102))
            with self.subTest(nombre=nombre):
                self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_procedencia_y_metadata_cabecera_incompatibles(self):
        for campo, valor in (("conjunto_id", 8), ("serie_id", 84), ("numero_escenarios", True), ("pronostico_id", 101.0)):
            p = paquete()
            p[campo] = valor
            with self.subTest(campo=campo):
                self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_regla_implementacion_politica_incompatibles(self):
        for campo in ("version_regla", "identificador_implementacion", "version_politica_numerica"):
            p = paquete()
            p[campo] = "futura"
            with self.subTest(campo=campo):
                self.error(snapshot.verificar_paquete, p)

    def test_tipos_int_bool_float_resultados(self):
        for campo, valor in (("pronostico_id", True), ("horizonte", 15.0), ("reposicion", 125), ("recepcion_dentro_horizonte", 1)):
            p = paquete()
            self.adulterar(p, "resultado", lambda d: d.update({campo: valor}))
            with self.subTest(campo=campo):
                self.error(snapshot.verificar_paquete, p)

    def test_fechas_y_pasos_adulterados(self):
        cambios = [lambda d: d["detalles"][0].update(paso=True),
                   lambda d: d["detalles"][0].update(fecha_objetivo="2017-06-18"),
                   lambda d: d.update(fecha_corte="2017-6-16"),
                   lambda d: d.update(fecha_corte="2017-02-30")]
        for cambiar in cambios:
            p = paquete()
            self.adulterar(p, "origen", cambiar)
            self.error(snapshot.verificar_paquete, p)

    def test_trayectoria_incompleta_y_fechas_repetidas(self):
        for cambiar in (lambda d: d["trayectoria"].pop(),
                        lambda d: d["trayectoria"][1].update(fecha_objetivo="2017-06-17"),
                        lambda d: d["trayectoria"][0].update(paso=1.0)):
            p = paquete()
            self.adulterar(p, "resultado", cambiar)
            self.error(snapshot.verificar_paquete, p)

    def test_demanda_entre_origen_y_resultados(self):
        p = paquete(2)
        self.adulterar(p, "resultado", lambda d: d["trayectoria"][0].update(demanda_pronosticada=9.0), indice=1)
        self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_demanda_total_distinta_entre_escenarios(self):
        p = paquete(2)
        self.adulterar(p, "resultado", lambda d: d.update(demanda_considerada=149.0), indice=1)
        self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_parametros_y_recepcion_incompatibles(self):
        for cambiar in (lambda d: d["parametros"].update(reserva=6.0),
                        lambda d: d.update(fecha_recepcion="2017-06-18"),
                        lambda d: d["trayectoria"][0].update(recepcion_simulada=0.0)):
            p = paquete()
            self.adulterar(p, "resultado", cambiar)
            self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_resumen_y_proyecciones_incompatibles(self):
        for cambiar in (lambda d: d["resumen"].update(balance_neto_final=0.0),
                        lambda d: d["resumen"].update(dias_con_balance_negativo=True),
                        lambda d: d["trayectoria"][0].update(disponible_hipotetico=0.0)):
            p = paquete()
            self.adulterar(p, "resultado", cambiar)
            self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_etiquetas_cantidad_y_orden(self):
        for cambiar in (lambda p: p["escenarios"].reverse(),
                        lambda p: p["escenarios"].pop(),
                        lambda p: p["escenarios"][0].update(etiqueta="B"),
                        lambda p: p["escenarios"][0].update(posicion=True)):
            p = paquete(2)
            cambiar(p)
            self.error(snapshot.verificar_paquete, p)

    def test_estructura_escenarios_de_entrada(self):
        for escenarios in ([], [{"etiqueta": "B"}], "A", [True]):
            datos = list(fixture())
            datos[3] = escenarios
            self.error(snapshot.preparar_snapshot, *datos, modalidad="individual")

    def test_entradas_numericas_invalidas_sin_conversion(self):
        for valor in (True, "30", 2**53 + 1, Decimal("30"), math.inf, -1):
            datos = list(fixture())
            datos[3][0]["inventario_disponible"] = valor
            with self.subTest(valor=valor):
                self.error(snapshot.preparar_snapshot, *datos, modalidad="individual")

    def test_privados_y_tipos_extra_entrada_rechazados(self):
        for indice, campo in ((0, "configuracion_json"), (2, "ruta_modelo")):
            datos = list(fixture())
            datos[indice][campo] = "PRIVADO"
            self.error(snapshot.preparar_snapshot, *datos, modalidad="individual")

    def test_no_mutacion_y_sin_alias_mutables(self):
        datos = fixture(3)
        antes = deepcopy(datos)
        p = snapshot.preparar_snapshot(*datos, modalidad="comparacion")
        self.assertEqual(datos, antes)
        a, b = snapshot.verificar_paquete(p), snapshot.verificar_paquete(p)
        a["origen"]["detalles"][0]["venta_estimada"] = 999
        a["resultados"][0]["simulacion"]["supuestos"].append("cambio")
        self.assertEqual(b, snapshot.verificar_paquete(p))
        self.assertNotEqual(a["resultados"][0]["simulacion"]["supuestos"], a["resultados"][1]["simulacion"]["supuestos"])
        self.assertEqual(datos, antes)

    def test_pureza_preparacion_lectura_e_importacion(self):
        datos = fixture(3)
        codigo = Path(snapshot.__file__).read_text(encoding="utf-8")
        aislado = ModuleType("snapshot_aislado")
        with ExitStack() as pila:
            for objetivo in ("builtins.open", "io.open", "os.open", "sqlite3.connect", "socket.socket", "subprocess.Popen"):
                pila.enter_context(patch(objetivo, side_effect=AssertionError("E/S no permitida")))
            for modulo, nombre in ((motor, "calcular_simulacion"), (trayectoria, "calcular_trayectoria"), (comparacion, "comparar_escenarios")):
                pila.enter_context(patch.object(modulo, nombre, side_effect=AssertionError("No recalcular")))
            exec(compile(codigo, "snapshot_aislado", "exec"), aislado.__dict__)
            p = aislado.preparar_snapshot(*datos, modalidad="comparacion")
            self.assertEqual(len(aislado.verificar_paquete(p)["resultados"]), 3)

    def test_metadatos_opcionales_incorrectos(self):
        for extra in ({"tipo": "desconocido"}, {"inicio_entrenamiento": "2017-01-01"},
                      {"inicio_entrenamiento": "2017-01-01", "fin_entrenamiento": "2017-06-17"},
                      {"fecha_generacion": "incorrecta"}, {"duracion_segundos": True}):
            datos = list(fixture())
            datos[0].update(extra)
            with self.subTest(extra=extra):
                self.error(snapshot.preparar_snapshot, *datos, modalidad="individual")

    def test_tipos_externos_rechazados_antes_de_deepcopy(self):
        class Externo:
            def __deepcopy__(self, memo):
                raise AssertionError("No ejecutar protocolos externos")
        for indice in (2, 3, 4):
            datos = list(fixture())
            if indice == 2:
                datos[2]["serie"]["family"] = Externo()
            elif indice == 3:
                datos[3][0]["reserva"] = Externo()
            else:
                datos[4]["advertencias"].append(Externo())
            self.error(snapshot.preparar_snapshot, *datos, modalidad="individual", codigo="estructura_invalida")

    def test_metadatos_textuales_subclase_rechazados(self):
        class Texto(str):
            pass
        for campo in ("identificador_implementacion", "version_politica_numerica"):
            p = paquete()
            p[campo] = Texto(p[campo])
            self.error(snapshot.verificar_paquete, p, codigo="version_no_soportada")

    def test_resultado_no_finito_con_hash_recalculado(self):
        p = paquete()
        fila = p["escenarios"][0]
        texto = fila["resultado_json"].replace('"reposicion":125.0', '"reposicion":1e999')
        self.assertNotEqual(texto, fila["resultado_json"])
        fila["resultado_json"], fila["sha256_resultado"] = texto, snapshot.calcular_sha256(texto)
        self.error(snapshot.verificar_paquete, p, codigo="representacion_invalida")

    def test_fechas_origen_limite_de_calendario(self):
        datos = list(fixture())
        datos[0]["fecha_corte"] = "9999-12-31"
        self.error(snapshot.preparar_snapshot, *datos, modalidad="individual", codigo="estructura_invalida")

    def test_origen_identificadores_y_cantidades_invalidos(self):
        for cambiar in (lambda d: d.update(conjunto_id=True), lambda d: d["serie"].update(id=0),
                        lambda d: d.update(experimento_id=2**63),
                        lambda d: d["detalles"][0].update(venta_estimada=2**53+1)):
            p = paquete()
            self.adulterar(p, "origen", cambiar)
            self.error(snapshot.verificar_paquete, p)

    def test_publicado_float_a_int_y_cero_con_signo(self):
        p = paquete()
        self.adulterar(p, "resultado", lambda d: d["trayectoria"][0].update(demanda_pronosticada=10))
        self.error(snapshot.verificar_paquete, p, codigo="estructura_invalida")
        texto = snapshot.serializar_documento({"a": -0.0})
        self.assertEqual(texto, '{"a":-0.0}')
        self.assertNotEqual(snapshot.calcular_sha256(texto), snapshot.calcular_sha256('{"a":0.0}'))

    def test_compatibilidad_columnas_sql_en_memoria_sin_migrar(self):
        p = paquete(3)
        self.assertEqual(snapshot.MAX_BYTES_DOCUMENTO, esquema.MAX_BYTES_DOCUMENTO)
        with closing(sqlite3.connect(":memory:")) as c:
            c.execute("PRAGMA foreign_keys=ON")
            for tabla, identificador in (("conjunto_datos", 7), ("serie_temporal", 83), ("pronostico", 101)):
                c.execute(f"CREATE TABLE {tabla}(id INTEGER PRIMARY KEY)")
                c.execute(f"INSERT INTO {tabla} VALUES (?)", (identificador,))
            for ddl in esquema._DDL:
                c.execute(ddl)
            cabecera = {k: v for k, v in p.items() if k != "escenarios"}
            cabecera.update(id=1, clave_idempotencia="sintetica", fecha_registro="2026-11-16T00:00:00Z", version_esquema=1)
            c.execute(f"INSERT INTO ejecucion_simulacion ({','.join(cabecera)}) VALUES ({','.join('?' for _ in cabecera)})", tuple(cabecera.values()))
            for escenario in p["escenarios"]:
                c.execute("INSERT INTO escenario_simulacion VALUES (?,?,?,?,?)", (1, escenario["etiqueta"], escenario["posicion"],
                          escenario["resultado_json"], escenario["sha256_resultado"]))
            self.assertEqual(c.execute("SELECT resultado_json FROM escenario_simulacion ORDER BY posicion").fetchall(),
                             [(e["resultado_json"],) for e in p["escenarios"]])

    def lector_recargado(self):
        # Simula un proceso nuevo que importa snapshots después de actualizar
        # el motor. Solo lee código autorizado, nunca documentos o bases reales.
        codigo = Path(snapshot.__file__).read_text(encoding="utf-8")
        lector = ModuleType("lector_historico_recargado")
        exec(compile(codigo, "lector_historico_recargado", "exec"), lector.__dict__)
        return lector

    def comprobar_lectura_historica(self, p):
        antes = deepcopy(p)
        esperado = snapshot.serializar_documento(snapshot.verificar_paquete(p))
        lector = self.lector_recargado()
        leido = lector.verificar_paquete(p)
        self.assertEqual(lector.serializar_documento(leido), esperado)
        self.assertEqual(p, antes)
        return leido

    def test_historico_v1_tras_actualizacion_regla_motor(self):
        p = paquete()
        with patch.object(motor, "VERSION_REGLA", "regla_motor_v2"):
            self.assertEqual(self.comprobar_lectura_historica(p)["version_regla"], snapshot.VERSION_REGLA_V1)

    def test_historico_v1_tras_retirada_tecnica_motor(self):
        p = paquete()
        with patch.object(motor, "TECNICAS", ("estacional_semanal_recursiva", "hist_gradient_boosting")):
            self.assertEqual(self.comprobar_lectura_historica(p)["origen"]["tecnica"], "random_forest")

    def test_historico_individual_ab_abc_con_motor_futuro(self):
        historicos = [paquete(n, valores=[0.1]*3 + [0]*12, inventario=0.3, reserva=0) for n in (1, 2, 3)]
        with patch.object(motor, "VERSION_REGLA", "regla_motor_v2"), patch.object(motor, "TECNICAS", ("otra_tecnica",)):
            for p in historicos:
                with self.subTest(escenarios=p["numero_escenarios"]):
                    r = self.comprobar_lectura_historica(p)["resultados"][0]["simulacion"]
                    self.assertGreater(r["reposicion"], 0)
                    original = json.loads(p["escenarios"][0]["resultado_json"])["reposicion"]
                    self.assertEqual(r["reposicion"].hex(), original.hex())

    def test_historico_lector_sin_constantes_motor_actual(self):
        p = paquete()
        with patch.object(motor, "VERSION_REGLA"), patch.object(motor, "TECNICAS"):
            del motor.VERSION_REGLA
            del motor.TECNICAS
            self.comprobar_lectura_historica(p)

    def test_lector_v1_independiente_alias_escritor_vigente(self):
        p = paquete()
        esperado = snapshot.serializar_documento(snapshot.verificar_paquete(p))
        with patch.object(snapshot, "VERSION_DOCUMENTO", 2), \
             patch.object(snapshot, "IDENTIFICADOR_IMPLEMENTACION", "implementacion_futura"), \
             patch.object(snapshot, "VERSION_POLITICA_NUMERICA", "politica_futura"):
            self.assertEqual(snapshot.serializar_documento(snapshot.verificar_paquete(p)), esperado)

    def test_regla_desconocida_documento_v1_con_hash_nuevo(self):
        for nombre in ("contexto", "resultado"):
            p = paquete()
            self.adulterar(p, nombre, lambda d: d.update(version_regla="regla_desconocida"))
            with self.subTest(documento=nombre):
                self.error(snapshot.verificar_paquete, p, codigo="integridad_invalida")

    def test_paquete_formato_futuro_rechazado(self):
        p = paquete()
        p["version_documento"] = 2
        self.error(snapshot.verificar_paquete, p, codigo="version_no_soportada")

    def test_escritor_v1_rechaza_motor_regla_v2(self):
        datos_anteriores = fixture()
        with patch.object(motor, "VERSION_REGLA", "regla_motor_v2"):
            datos_nuevos = fixture()
            self.assertEqual(datos_nuevos[-1]["version_regla"], "regla_motor_v2")
            for datos in (datos_anteriores, datos_nuevos):
                self.error(snapshot.preparar_snapshot, *datos, modalidad="individual", codigo="version_no_soportada")

    def test_escritor_v1_rechaza_tecnica_retirada(self):
        datos = fixture()
        with patch.object(motor, "TECNICAS", ("estacional_semanal_recursiva",)):
            self.error(snapshot.preparar_snapshot, *datos, modalidad="individual", codigo="version_no_soportada")

    def test_escritor_v1_rechaza_resultado_incompatible(self):
        for cambio in (lambda r: r.update(version_regla="regla_desconocida"),
                       lambda r: r.update(identificador_implementacion="implementacion_desconocida")):
            datos = list(fixture())
            cambio(datos[-1])
            self.error(snapshot.preparar_snapshot, *datos, modalidad="individual")

    def test_escritor_v1_rechaza_versiones_de_escritor_incompatibles(self):
        datos = fixture()
        for campo, valor in (("VERSION_DOCUMENTO", 2), ("IDENTIFICADOR_IMPLEMENTACION", "implementacion_futura"),
                             ("VERSION_POLITICA_NUMERICA", "politica_futura")):
            with self.subTest(campo=campo), patch.object(snapshot, campo, valor):
                self.error(snapshot.preparar_snapshot, *datos, modalidad="individual", codigo="version_no_soportada")


if __name__ == "__main__":
    unittest.main()
