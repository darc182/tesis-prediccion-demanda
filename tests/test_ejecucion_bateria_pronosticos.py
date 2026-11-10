"""I1-B1: coordinación y persistencia parcial con SQLite sintética temporal."""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import ejecucion_bateria_pronosticos as ejecucion
from backend.app.services import plan_bateria_pronosticos as planificador
from backend.app.services import registro_pronosticos as registro
from backend.app.services.consulta_pronosticos import (
    consultar_configuracion_pronostico, consultar_pronostico,
)
from backend.app.storage.db import inicializar_base
from scripts import modelos_prediccion as modelos


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")


class EjecucionBateriaPronosticosTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory(prefix="tesis_i1b1_")
        self.addCleanup(temporal.cleanup)
        self.ruta = (Path(temporal.name) / "bateria_sintetica.sqlite3").resolve()
        inicializar_base(self.ruta)
        inicio = date(2017, 4, 1)
        self.filas = [((inicio + timedelta(days=i)).isoformat(),
                       0.0 if i % 11 == 0 else 10.125 + i % 7 / 4 + i // 7 / 8, i % 3)
                      for i in range(122) if i != 14]
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute("PRAGMA foreign_keys=ON")
            c.executemany("INSERT INTO conjunto_datos VALUES (?,?,?,?,?)", [
                (2, "anterior_sintetico.csv", "hash2", "2026-01-01", 0),
                (7, "actual_sintetico.csv", "hash7", "2026-01-02", 1)])
            c.executemany("INSERT INTO serie_temporal VALUES (?,?,?,?,?)", [
                (24, 2, 17, "PREPARED FOODS", "sintética"),
                (83, 7, 17, "PREPARED FOODS", "sintética"),
                (84, 7, 8, "BREAD/BAKERY", "sintética")])
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                          [(serie, fecha, venta + diferencia, promo)
                           for serie, diferencia in ((24, 80.5), (83, 0), (84, 30.25))
                           for fecha, venta, promo in self.filas])

    def leer(self, sql, parametros=()):
        with closing(sqlite3.connect(f"{self.ruta.as_uri()}?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            return c.execute(sql, parametros).fetchall()

    def cambiar(self, sql, parametros=()):
        with closing(sqlite3.connect(self.ruta)) as c, c:
            c.execute(sql, parametros)

    def conteos(self):
        return tuple(self.leer(f"SELECT COUNT(*) FROM {tabla}")[0][0]
                     for tabla in ("experimento", "pronostico", "detalle_pronostico"))

    def ejecutar(self, corte=CORTES[0], **opciones):
        return ejecucion.ejecutar_bateria(
            **{"ruta_base": self.ruta, "conjunto_id": 7, "serie_id": 83,
               "fecha_corte": corte, **opciones})

    def comprobar_resumen(self, respuesta, estados):
        self.assertEqual([e["estado"] for e in respuesta["resultados"]], estados)
        self.assertEqual(respuesta["resumen"], {
            s: estados.count(s) for s in ("creado", "reutilizado", "fallido", "no_ejecutado")})
        self.assertEqual(sum(respuesta["resumen"].values()), 5)
        for elemento in respuesta["resultados"]:
            if elemento["estado"] in {"creado", "reutilizado"}:
                self.assertGreater(elemento["experimento_id"], 0)
                self.assertGreater(elemento["pronostico_id"], 0)
            else:
                self.assertNotIn("experimento_id", elemento)
                self.assertNotIn("pronostico_id", elemento)
            if elemento["estado"] == "no_ejecutado":
                self.assertIsNone(elemento["duracion_segundos"])
            else:
                self.assertGreaterEqual(elemento["duracion_segundos"], 0)
        self.assertGreaterEqual(respuesta["duracion_segundos"], sum(
            e["duracion_segundos"] or 0 for e in respuesta["resultados"]))
        json.dumps(respuesta, allow_nan=False)

    def test_reales_tres_cortes_configuracion_quince_fechas_y_reutilizacion_calcula(self):
        for numero, corte in enumerate(CORTES, 1):
            with self.subTest(corte=corte):
                plan = planificador.construir_plan_bateria(7, 83, corte)
                with patch.object(registro, "_calcular", wraps=registro._calcular) as calcular, \
                        patch.object(modelos, "entrenar_modelo", wraps=modelos.entrenar_modelo) as entrenar:
                    primera = self.ejecutar(corte)
                self.assertEqual((calcular.call_count, entrenar.call_count), (5, 4))
                self.comprobar_resumen(primera, ["creado"] * 5)
                self.assertEqual(primera["estado"], "completa")
                self.assertTrue(primera["contexto_verificado"])
                self.assertIsNone(primera["interrupcion"])
                self.assertEqual(self.conteos(), (numero * 5, numero * 5, numero * 75))
                for orden, (elemento, previsto) in enumerate(zip(primera["resultados"], plan["elementos"]), 1):
                    self.assertEqual(elemento["orden"], orden)
                    for clave in previsto:
                        self.assertEqual(elemento[clave], previsto[clave])
                    self.assertEqual((elemento["conjunto_id"], elemento["serie_id"], elemento["fecha_corte"]),
                                     (7, 83, corte))
                    guardada = consultar_configuracion_pronostico(self.ruta, elemento["pronostico_id"])
                    self.assertEqual(guardada["configuracion"], previsto["configuracion"])
                    consulta = consultar_pronostico(self.ruta, elemento["pronostico_id"])
                    self.assertEqual([d["fecha_objetivo"] for d in consulta["detalles"]], [
                        (date.fromisoformat(corte) + timedelta(days=i)).isoformat() for i in range(1, 16)])
                contenido = self.ruta.read_bytes()
                sentencias = []
                abrir = registro._abrir_base

                def observar(*args, **kwargs):
                    c = abrir(*args, **kwargs)
                    c.set_trace_callback(sentencias.append)
                    return c

                with patch.object(registro, "_abrir_base", side_effect=observar), \
                        patch.object(registro, "_calcular", wraps=registro._calcular) as calcular, \
                        patch.object(modelos, "entrenar_modelo", wraps=modelos.entrenar_modelo) as entrenar:
                    segunda = self.ejecutar(corte)
                self.comprobar_resumen(segunda, ["reutilizado"] * 5)
                self.assertEqual((calcular.call_count, entrenar.call_count), (5, 4))
                self.assertEqual([(e["experimento_id"], e["pronostico_id"]) for e in primera["resultados"]],
                                 [(e["experimento_id"], e["pronostico_id"]) for e in segunda["resultados"]])
                self.assertEqual(self.ruta.read_bytes(), contenido)
                self.assertFalse(any(s.lstrip().upper().startswith(
                    ("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in sentencias))
        self.assertEqual(self.leer("PRAGMA foreign_key_check"), [])

    def test_error_calculo_primero_tercero_ultimo_persiste_exitos_y_continua(self):
        for indice, corte in zip((0, 2, 4), CORTES):
            with self.subTest(indice=indice):
                calcular = registro._calcular
                llamadas = []

                def fallar(*args, **kwargs):
                    posicion = len(llamadas)
                    llamadas.append(args[1])
                    if posicion == indice:
                        raise registro.ErrorRegistroPronostico("error_calculo", "SQL privado /ruta/secreta")
                    return calcular(*args, **kwargs)

                with patch.object(registro, "_calcular", side_effect=fallar):
                    r = self.ejecutar(corte)
                estados = ["creado"] * 5
                estados[indice] = "fallido"
                self.comprobar_resumen(r, estados)
                self.assertEqual(r["estado"], "parcial")
                self.assertTrue(r["contexto_verificado"])
                self.assertIsNone(r["interrupcion"])
                self.assertEqual(len(llamadas), 5)
                self.assertNotIn("privado", json.dumps(r))
                self.assertEqual(self.conteos(), (4 * (CORTES.index(corte) + 1),
                                                 4 * (CORTES.index(corte) + 1),
                                                 60 * (CORTES.index(corte) + 1)))

    def test_cambio_activo_despues_del_primero_detiene_sin_rollback_ficticio(self):
        registrar = registro.registrar_pronostico

        def cambiar_activo(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            self.cambiar("UPDATE conjunto_datos SET activo=CASE id WHEN 2 THEN 1 ELSE 0 END")
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=cambiar_activo) as llamada:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["creado"] + ["no_ejecutado"] * 4)
        self.assertEqual(llamada.call_count, 1)
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertFalse(r["contexto_verificado"])
        self.assertEqual(r["estado"], "interrumpida")
        self.assertEqual(self.conteos(), (1, 1, 15))

    def test_cambio_historial_tras_tercero_detiene_restantes(self):
        registrar = registro.registrar_pronostico
        llamadas = []

        def cambiar_historial(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            llamadas.append(resultado)
            if len(llamadas) == 3:
                self.cambiar("UPDATE venta_diaria SET sales=sales+0.125 WHERE serie_id=83 AND fecha=?",
                             (CORTES[0],))
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=cambiar_historial):
            r = self.ejecutar()
        self.comprobar_resumen(r, ["creado"] * 3 + ["no_ejecutado"] * 2)
        self.assertEqual(len(llamadas), 3)
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertEqual(self.conteos(), (3, 3, 45))

    def test_cambio_entre_prelectura_y_calculo_restaurado_se_detecta_en_registro(self):
        registrar = registro.registrar_pronostico
        original = self.leer("SELECT sales FROM venta_diaria WHERE serie_id=83 AND fecha=?", (CORTES[0],))[0][0]

        def cambiar_y_restaurar(*args, **kwargs):
            self.cambiar("UPDATE venta_diaria SET sales=100.125 WHERE serie_id=83 AND fecha=?", (CORTES[0],))
            resultado = registrar(*args, **kwargs)
            self.cambiar("UPDATE venta_diaria SET sales=? WHERE serie_id=83 AND fecha=?", (original, CORTES[0]))
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=cambiar_y_restaurar) as llamada:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["creado"] + ["no_ejecutado"] * 4)
        self.assertEqual(llamada.call_count, 1)
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertFalse(r["contexto_verificado"])
        self.assertEqual(self.conteos(), (1, 1, 15))

    def test_cambio_durante_calculo_registro_revierte_y_no_hay_otro_intento(self):
        calcular = registro._calcular

        def cambiar(*args, **kwargs):
            resultado = calcular(*args, **kwargs)
            self.cambiar("UPDATE venta_diaria SET onpromotion=onpromotion+1 WHERE serie_id=83 AND fecha=?",
                         (CORTES[0],))
            return resultado

        with patch.object(registro, "_calcular", side_effect=cambiar) as llamada:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["fallido"] + ["no_ejecutado"] * 4)
        self.assertEqual(llamada.call_count, 1)
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertEqual(self.conteos(), (0, 0, 0))

    def test_cambio_antes_del_primer_intento_no_inicia_modelos(self):
        comprobar = ejecucion._comprobar_contexto
        llamadas = []

        def cambiar(*args, **kwargs):
            llamadas.append(None)
            if len(llamadas) == 2:
                self.cambiar("UPDATE serie_temporal SET family='OTRA FAMILIA' WHERE id=83")
            return comprobar(*args, **kwargs)

        with patch.object(ejecucion, "_comprobar_contexto", side_effect=cambiar), \
                patch.object(registro, "registrar_pronostico") as registrar:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["no_ejecutado"] * 5)
        registrar.assert_not_called()
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")

    def test_error_infraestructura_detiene_y_mensajes_no_filtran_detalles(self):
        registrar = registro.registrar_pronostico
        for excepcion, codigo in ((sqlite3.OperationalError("SQL /privado"), "base_no_disponible"),
                                   (RuntimeError("traceback /privado"), "error_interno"),
                                   (registro.ErrorRegistroPronostico("desconocido", "/privado"), "error_interno")):
            with self.subTest(codigo=codigo):
                llamadas = []

                def fallar(*args, **kwargs):
                    llamadas.append(None)
                    if len(llamadas) == 2:
                        raise excepcion
                    return registrar(*args, **kwargs)

                with patch.object(registro, "registrar_pronostico", side_effect=fallar):
                    r = self.ejecutar()
                self.comprobar_resumen(r, ["creado" if codigo == "base_no_disponible" else "reutilizado",
                                          "fallido", "no_ejecutado", "no_ejecutado", "no_ejecutado"])
                self.assertEqual(len(llamadas), 2)
                self.assertEqual(r["interrupcion"]["codigo"], codigo)
                self.assertNotIn("privado", json.dumps(r))
        self.assertEqual(self.conteos(), (1, 1, 15))

    def test_persistencia_fallida_segundo_revierte_solo_ese_elemento(self):
        self.cambiar("CREATE TRIGGER fallo_sintetico BEFORE INSERT ON detalle_pronostico "
                     "WHEN NEW.pronostico_id>1 BEGIN SELECT RAISE(ABORT,'privado SQL'); END")
        with patch.object(registro, "registrar_pronostico", wraps=registro.registrar_pronostico) as registrar:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["creado", "fallido"] + ["no_ejecutado"] * 3)
        self.assertEqual(registrar.call_count, 2)
        self.assertEqual(r["interrupcion"]["codigo"], "error_persistencia")
        self.assertEqual(self.conteos(), (1, 1, 15))
        self.assertEqual(self.leer("PRAGMA foreign_key_check"), [])
        self.assertNotIn("privado", json.dumps(r))

    def test_conflicto_no_se_omite_con_busqueda_rapida(self):
        self.ejecutar()
        self.cambiar("UPDATE detalle_pronostico SET venta_estimada=999 WHERE pronostico_id=1 AND paso=1")
        contenido = self.ruta.read_bytes()
        with patch.object(registro, "_calcular", wraps=registro._calcular) as calcular:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["fallido"] + ["no_ejecutado"] * 4)
        self.assertEqual(calcular.call_count, 1)
        self.assertEqual(r["interrupcion"]["codigo"], "conflicto")
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_serie_ajena_conjunto_inactivo_y_ausencia_activo_rechazan_antes_de_generar(self):
        casos = (({"serie_id": 24}, "serie_inactiva"),
                 ({"serie_id": 999}, "serie_inexistente"),
                 ({"conjunto_id": 2}, "conjunto_inactivo"))
        with patch.object(registro, "registrar_pronostico") as registrar:
            for cambios, codigo in casos:
                with self.subTest(codigo=codigo), self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                    self.ejecutar(**cambios)
                self.assertEqual(error.exception.codigo, codigo)
            self.cambiar("UPDATE conjunto_datos SET activo=0")
            with self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                self.ejecutar()
            self.assertEqual(error.exception.codigo, "sin_conjunto_activo")
        registrar.assert_not_called()
        self.assertEqual(self.conteos(), (0, 0, 0))

    def test_entradas_invalidas_y_reserva_no_abren_sqlite(self):
        casos = (({"conjunto_id": True}, "conjunto_invalido"),
                 ({"serie_id": 0}, "serie_invalida"),
                 ({"fecha_corte": "2017-6-16"}, "corte_invalido"),
                 ({"fecha_corte": "2017-06-17"}, "corte_no_autorizado"),
                 ({"fecha_corte": "2017-07-17"}, "prueba_reservada"),
                 ({"fecha_corte": "2017-08-01"}, "prueba_reservada"),
                 ({"fecha_corte": "2017-08-15"}, "prueba_reservada"))
        with patch.object(ejecucion.sqlite3, "connect") as abrir:
            for cambios, codigo in casos:
                with self.subTest(codigo=codigo), self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                    self.ejecutar(**cambios)
                self.assertEqual(error.exception.codigo, codigo)
        abrir.assert_not_called()

    def test_base_ausente_o_sin_esquema_no_se_crea_ni_inicializa(self):
        ausente = self.ruta.parent / "no_existe.sqlite3"
        vacia = self.ruta.parent / "vacia.sqlite3"
        vacia.write_bytes(b"")
        with patch.object(registro, "registrar_pronostico") as registrar:
            for ruta in (ausente, vacia):
                with self.subTest(ruta=ruta.name), self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                    self.ejecutar(ruta_base=ruta)
                self.assertEqual(error.exception.codigo, "base_no_disponible")
        registrar.assert_not_called()
        self.assertFalse(ausente.exists())
        self.assertEqual(vacia.read_bytes(), b"")

    def test_sin_historial_e_historial_no_finito_rechazos_publicos(self):
        self.cambiar("DELETE FROM venta_diaria WHERE serie_id=83")
        with patch.object(registro, "registrar_pronostico") as registrar:
            with self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                self.ejecutar()
            self.assertEqual(error.exception.codigo, "sin_historial")
            self.cambiar("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (83,?, ?,0)",
                         (CORTES[0], float("inf")))
            with self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
                self.ejecutar()
            self.assertEqual(error.exception.codigo, "historial_invalido")
        registrar.assert_not_called()

    def test_infraestructura_falla_en_lectura_posterior_preserva_confirmado(self):
        comprobar = ejecucion._comprobar_contexto
        llamadas = []

        def fallar(*args, **kwargs):
            llamadas.append(None)
            if len(llamadas) == 3:
                raise ejecucion.ErrorEjecucionBateria("base_no_disponible")
            return comprobar(*args, **kwargs)

        with patch.object(ejecucion, "_comprobar_contexto", side_effect=fallar), \
                patch.object(registro, "registrar_pronostico", wraps=registro.registrar_pronostico) as registrar:
            r = self.ejecutar()
        self.comprobar_resumen(r, ["creado"] + ["no_ejecutado"] * 4)
        self.assertEqual(registrar.call_count, 1)
        self.assertEqual(r["interrupcion"]["codigo"], "base_no_disponible")
        self.assertFalse(r["contexto_verificado"])
        self.assertEqual(self.conteos(), (1, 1, 15))

    def test_ids_positivos_sin_registro_real_no_se_publican_como_exito(self):
        falso = registro.ResultadoRegistro(7, 83, 999, 999, registro.TECNICAS[0], CORTES[0], 15, 15, True)
        with patch.object(registro, "registrar_pronostico", return_value=falso):
            r = self.ejecutar()
        self.comprobar_resumen(r, ["fallido"] + ["no_ejecutado"] * 4)
        self.assertEqual(r["interrupcion"]["codigo"], "resultado_invalido")
        self.assertEqual(self.conteos(), (0, 0, 0))

    def comprobar_json_corrupto_tras_commit(self, transformar):
        registrar = registro.registrar_pronostico
        confirmados = []
        textos = []

        def corromper(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            confirmados.append(resultado)
            texto = self.leer("SELECT configuracion_json FROM experimento WHERE id=?",
                              (resultado.experimento_id,))[0][0]
            corrupto = transformar(texto)
            # El lector permisivo conserva identidad e historial: reproduce la
            # brecha, sin depender de que fallen esas comprobaciones posteriores.
            anterior, permisiva = json.loads(texto), json.loads(corrupto)
            self.assertEqual(registro._contenido_config(permisiva), registro._contenido_config(anterior))
            self.assertEqual(permisiva["historial"], anterior["historial"])
            with self.assertRaises((ValueError, RecursionError)):
                registro._leer_configuracion_estricta(corrupto)
            with closing(sqlite3.connect(self.ruta)) as c, c:
                # Solo esta conexión de inyección, en SQLite temporal. Permite
                # simular NaN/Infinity almacenados pese al CHECK json_valid;
                # no altera el esquema ni la configuración de otras conexiones.
                c.execute("PRAGMA ignore_check_constraints=ON")
                c.execute("UPDATE experimento SET configuracion_json=? WHERE id=?",
                          (corrupto, resultado.experimento_id))
            textos.append(corrupto)
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=corromper) as llamada, \
                patch.object(registro, "_leer_configuracion_estricta",
                             wraps=registro._leer_configuracion_estricta) as lector:
            respuesta = self.ejecutar()
        self.comprobar_resumen(respuesta, ["fallido"] + ["no_ejecutado"] * 4)
        self.assertEqual(llamada.call_count, 1)
        self.assertEqual(lector.call_count, 2)  # Paridad del test y lectura posterior real.
        self.assertEqual(respuesta["estado"], "interrumpida")
        self.assertFalse(respuesta["contexto_verificado"])
        esperado = {"codigo": "resultado_invalido",
                    "mensaje": "El registro no acreditó un resultado válido para la configuración solicitada."}
        self.assertEqual(respuesta["interrupcion"], esperado)
        self.assertEqual(respuesta["resultados"][0]["error"], esperado)
        for privado in ("privado", "SELECT", "UPDATE", str(self.ruta)):
            self.assertNotIn(privado, json.dumps(respuesta, ensure_ascii=False))
        self.assertEqual(len(confirmados), 1)
        self.assertTrue(confirmados[0].nuevo)
        self.assertEqual(self.conteos(), (1, 1, 15))
        # La escritura confirmada y la corrupción de prueba siguen presentes:
        # el coordinador no sanea, elimina ni simula un rollback global.
        self.assertEqual(self.leer("SELECT configuracion_json FROM experimento WHERE id=?",
                                  (confirmados[0].experimento_id,)), [(textos[0],)])
        self.assertEqual(self.leer("PRAGMA foreign_key_check"), [])

    def test_json_historial_duplicado_no_acredita_commit_ni_inicia_restantes(self):
        self.comprobar_json_corrupto_tras_commit(
            lambda texto: '{"historial":{"privado":"ruta privada SQL"},' + texto[1:])

    def test_json_claves_duplicadas_anidadas_se_rechazan(self):
        def duplicar(texto):
            config = json.loads(texto)
            config["auditoria"]["control_integridad"] = "MARCADOR_JSON"
            return json.dumps(config).replace(
                '"MARCADOR_JSON"',
                '{"nivel":{"dato":"privado","dato":"valor público"}}')

        self.comprobar_json_corrupto_tras_commit(duplicar)

    def test_json_numeros_no_finitos_en_auditoria_se_rechazan(self):
        # Una SQLite nueva por variante mantiene independientes las filas
        # corruptas anteriores, sin borrar ni reparar registros confirmados.
        for token in ("NaN", "Infinity", "-Infinity", "1e309"):
            with self.subTest(token=token):
                caso = EjecucionBateriaPronosticosTests()
                caso.setUp()
                try:
                    def no_finito(texto):
                        config = json.loads(texto)
                        config["auditoria"]["control_integridad"] = "MARCADOR_JSON"
                        return json.dumps(config).replace(
                            '"MARCADOR_JSON"', '{"privado":' + token + '}')

                    caso.comprobar_json_corrupto_tras_commit(no_finito)
                finally:
                    caso.doCleanups()

    def test_historia_reciente_ausente_error_inferencia_y_cero_observado_no_equivale(self):
        self.cambiar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha='2017-06-14'")
        with patch.object(modelos, "entrenar_modelo", wraps=modelos.entrenar_modelo) as entrenar:
            parcial = self.ejecutar()
        self.comprobar_resumen(parcial, ["creado"] + ["fallido"] * 4)
        self.assertEqual(entrenar.call_count, 4)  # Ajustes válidos; falla la inferencia reciente.
        self.assertEqual(parcial["estado"], "parcial")
        self.assertTrue(all(e["error"]["codigo"] == "error_calculo" for e in parcial["resultados"][1:]))
        self.assertEqual(self.conteos(), (1, 1, 15))
        # Otro contexto de corte, con cero observado y fracciones reales conservadas.
        self.cambiar("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (83,'2017-06-14',0,0)")
        completa = self.ejecutar(CORTES[1])
        self.comprobar_resumen(completa, ["creado"] * 5)
        self.assertGreater(completa["historial_inicial"]["cantidad"], parcial["historial_inicial"]["cantidad"])
        self.assertEqual(self.leer("SELECT COUNT(*) FROM venta_diaria WHERE serie_id=83 AND fecha='2017-04-15'"), [(0,)])
        self.assertTrue(any(v[1] != int(v[1]) for v in self.filas))

    def test_futuro_y_otra_serie_no_alteran_contexto_ni_predicciones(self):
        primera = self.ejecutar()
        detalles = [consultar_pronostico(self.ruta, e["pronostico_id"])["detalles"] for e in primera["resultados"]]
        registrar = registro.registrar_pronostico

        def cambiar_fuera(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            self.cambiar("UPDATE venta_diaria SET sales=sales+1000 WHERE serie_id=84 OR fecha>'2017-06-16'")
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=cambiar_fuera):
            segunda = self.ejecutar()
        self.comprobar_resumen(segunda, ["reutilizado"] * 5)
        self.assertTrue(segunda["contexto_verificado"])
        self.assertEqual(primera["historial_inicial"], segunda["historial_inicial"])
        self.assertEqual(detalles, [consultar_pronostico(self.ruta, e["pronostico_id"])["detalles"]
                                    for e in segunda["resultados"]])

    def test_get_reales_solo_lectura_no_ejecutan_bateria_ni_entrenan(self):
        r = self.ejecutar()
        contenido = self.ruta.read_bytes()
        anteriores = app.dependency_overrides.copy()
        try:
            app.dependency_overrides.pop(obtener_ruta_base, None)
            with patch.dict(os.environ, {"TESIS_DB_PATH": str(self.ruta)}), \
                    patch.object(ejecucion, "ejecutar_bateria") as bateria, \
                    patch.object(registro, "_calcular") as calcular, \
                    patch.object(modelos, "entrenar_modelo") as entrenar, TestClient(app) as cliente:
                self.assertEqual(obtener_ruta_base().resolve(), self.ruta)
                self.assertEqual(cliente.get("/api/pronosticos/active/series/83").status_code, 200)
                for elemento in r["resultados"]:
                    url = f"/api/pronosticos/active/{elemento['pronostico_id']}"
                    self.assertEqual(cliente.get(url).status_code, 200)
                    respuesta = cliente.get(url + "/configuracion")
                    self.assertEqual(respuesta.status_code, 200)
                    self.assertEqual(respuesta.json()["configuracion"], elemento["configuracion"])
                for doble in (bateria, calcular, entrenar):
                    doble.assert_not_called()
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(anteriores)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (5, 5, 75))

    def test_coordinador_lecturas_ro_begin_sin_escrituras_propias(self):
        conectar = sqlite3.connect
        registrar = registro.registrar_pronostico
        lecturas = []
        dentro_registro = []

        def registrar_sin_observar(*args, **kwargs):
            dentro_registro.append(True)
            try:
                return registrar(*args, **kwargs)
            finally:
                dentro_registro.pop()

        def observar(ruta, *args, **kwargs):
            c = conectar(ruta, *args, **kwargs)
            if not dentro_registro and isinstance(ruta, str) and ruta.endswith("?mode=ro"):
                sentencias = []
                c.set_trace_callback(sentencias.append)
                lecturas.append(sentencias)
            return c

        with patch.object(ejecucion.sqlite3, "connect", side_effect=observar), \
                patch.object(registro, "registrar_pronostico", side_effect=registrar_sin_observar):
            self.ejecutar()
        self.assertEqual(len(lecturas), 11)
        for sentencias in lecturas:
            self.assertEqual(sentencias[0], "BEGIN")
            self.assertTrue(all(s.lstrip().upper().startswith(("BEGIN", "SELECT")) for s in sentencias))

    def test_lectura_consistente_con_escritor_concurrente_en_wal(self):
        self.cambiar("PRAGMA journal_mode=WAL")
        plan = planificador.construir_plan_bateria(7, 83, CORTES[0])
        inicial = ejecucion._comprobar_contexto(self.ruta, plan)
        conectar = sqlite3.connect
        modificadas = []
        ruta = self.ruta

        class LecturaConcurrente(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                cursor = super().execute(sql, *args, **kwargs)
                if sql.startswith("SELECT id, hash") and not modificadas:
                    with closing(conectar(ruta)) as escritor, escritor:
                        escritor.execute("UPDATE conjunto_datos SET hash='otro' WHERE id=7")
                        escritor.execute("UPDATE venta_diaria SET sales=sales+1 WHERE serie_id=83")
                    modificadas.append(True)
                return cursor

        def observar(*args, **kwargs):
            return conectar(*args, **kwargs, factory=LecturaConcurrente)

        with patch.object(ejecucion.sqlite3, "connect", side_effect=observar):
            self.assertEqual(ejecucion._comprobar_contexto(self.ruta, plan), inicial)
        with self.assertRaises(ejecucion.ErrorEjecucionBateria) as error:
            ejecucion._comprobar_contexto(self.ruta, plan, inicial)
        self.assertEqual(error.exception.codigo, "historial_cambiado")

    def test_resultado_inconsistente_no_expone_identificadores_inventados(self):
        for cambio in ({"conjunto_id": 2}, {"experimento_id": True}, {"pronostico_id": -3},
                       {"cantidad_detalles": 14}, {"nuevo": 1}):
            with self.subTest(cambio=cambio):
                falso = replace(registro.ResultadoRegistro(7, 83, 999, 999, registro.TECNICAS[0],
                                                          CORTES[0], 15, 15, True), **cambio)
                with patch.object(registro, "registrar_pronostico", return_value=falso) as registrar:
                    r = self.ejecutar()
                self.comprobar_resumen(r, ["fallido"] + ["no_ejecutado"] * 4)
                self.assertEqual(registrar.call_count, 1)
                self.assertEqual(r["interrupcion"]["codigo"], "resultado_invalido")
        self.assertEqual(self.conteos(), (0, 0, 0))

    def test_planificador_fuente_unica_no_mutacion_y_ejecucion_secuencial(self):
        plan = planificador.construir_plan_bateria(7, 83, CORTES[0])
        original = deepcopy(plan)
        eventos = []
        registrar = registro.registrar_pronostico

        def observar(*args, **kwargs):
            eventos.append(("inicio", args[2]))
            resultado = registrar(*args, **kwargs)
            kwargs["configuracion_solicitada"]["parametros"].clear()
            eventos.append(("fin", args[2]))
            return resultado

        with patch.object(planificador, "construir_plan_bateria", return_value=plan) as construir, \
                patch.object(registro, "registrar_pronostico", side_effect=observar):
            r = self.ejecutar()
        construir.assert_called_once_with(7, 83, CORTES[0])
        self.assertEqual(plan, original)
        self.assertEqual(eventos, [(evento, e["tecnica"]) for e in plan["elementos"] for evento in ("inicio", "fin")])
        r["resultados"][1]["variables"].clear()
        self.assertEqual(plan, original)
        self.assertTrue(r["resultados"][2]["variables"])

    def test_importar_no_ejecuta_ni_abre_archivos_datos_red_o_servidores(self):
        codigo = compile(Path(ejecucion.__file__).read_text(encoding="utf-8"), ejecucion.__file__, "exec")
        modulo = ModuleType("prueba_importacion_i1b1")
        # dataclass necesita que el módulo figure en sys.modules al definir clases.
        import sys
        import socket
        import subprocess
        with patch.dict(sys.modules, {modulo.__name__: modulo}), \
                patch.object(registro, "registrar_pronostico") as registrar, \
                patch.object(planificador, "construir_plan_bateria") as planear, \
                patch.object(ejecucion.sqlite3, "connect") as abrir, \
                patch("builtins.open") as archivo, patch.object(socket, "socket") as red, \
                patch.object(subprocess, "Popen") as proceso:
            exec(codigo, modulo.__dict__)
            for doble in (registrar, planear, abrir, archivo, red, proceso):
                doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
