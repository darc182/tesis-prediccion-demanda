"""I1-B2: HTTP real con TestClient, modelos reales y SQLite sintética temporal."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, timedelta
import inspect
import json
import os
from pathlib import Path
import sqlite3
from threading import Barrier, Event
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import pronosticos as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import ejecucion_bateria_pronosticos as ejecucion
from backend.app.services import plan_bateria_pronosticos as planificador
from backend.app.services import registro_pronosticos as registro
from tests import test_ejecucion_bateria_pronosticos as fixtures


PREFIJO = "/api/pronosticos/active"
CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")


class ApiBateriaPronosticosTests(unittest.TestCase):
    def setUp(self):
        self.datos = fixtures.EjecucionBateriaPronosticosTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        anteriores = app.dependency_overrides.copy()
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides.pop(obtener_ruta_base, None)
        self.enterContext(patch.dict(os.environ, {"TESIS_DB_PATH": str(self.ruta)}))
        self.assertEqual(obtener_ruta_base().resolve(), self.ruta)
        self.cliente = self.enterContext(TestClient(app))

    def post(self, corte=CORTES[0], serie=83, conjunto=7):
        return self.cliente.post(f"{PREFIJO}/series/{serie}/bateria",
                                 json={"conjunto_id": conjunto, "fecha_corte": corte})

    def comprobar(self, respuesta, estado, estados):
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.headers.get("Cache-Control"), "no-store")
        r = respuesta.json()
        self.assertEqual(set(r), {"conjunto_id", "serie_id", "fecha_corte", "horizonte", "estado",
                                  "contexto_verificado", "resultados", "resumen", "interrupcion", "duracion_segundos"})
        self.assertEqual(r["estado"], estado)
        self.assertEqual([e["estado"] for e in r["resultados"]], estados)
        self.assertEqual(r["resumen"], {s: estados.count(s) for s in (
            "creado", "reutilizado", "fallido", "no_ejecutado")})
        plan = planificador.construir_plan_bateria(r["conjunto_id"], r["serie_id"], r["fecha_corte"])
        for orden, (elemento, previsto) in enumerate(zip(r["resultados"], plan["elementos"]), 1):
            self.assertEqual(elemento["orden"], orden)
            for clave in previsto:
                self.assertEqual(elemento[clave], previsto[clave])
            self.assertEqual((elemento["conjunto_id"], elemento["serie_id"], elemento["fecha_corte"]),
                             (r["conjunto_id"], r["serie_id"], r["fecha_corte"]))
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
        self.assertGreaterEqual(r["duracion_segundos"], 0)
        self.comprobar_sin_privados(r)
        return r

    def comprobar_sin_privados(self, contenido):
        texto = json.dumps(contenido, ensure_ascii=False, allow_nan=False)
        prohibidas = {"historial_inicial", "sha256_observaciones", "hash_conjunto", "historial", "ruta_modelo"}

        def recorrer(valor):
            if isinstance(valor, dict):
                self.assertFalse(prohibidas.intersection(valor))
                for item in valor.values():
                    recorrer(item)
            elif isinstance(valor, list):
                for item in valor:
                    recorrer(item)

        recorrer(contenido)
        for marca in ("PRIVADO", "SELECT", "Traceback"):
            self.assertNotIn(marca, texto)
        self.assertNotIn(json.dumps(str(self.ruta)), texto)

    def error(self, respuesta, estado, codigo):
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertEqual(respuesta.headers.get("Cache-Control"), "no-store")
        self.assertEqual(respuesta.json(), {"detail": {"codigo": codigo, "mensaje": api.ERRORES[codigo][1]}})
        self.comprobar_sin_privados(respuesta.json())

    def test_completas_reales_tres_cortes_y_reutilizaciones_sin_escrituras(self):
        for numero, corte in enumerate(CORTES, 1):
            with self.subTest(corte=corte):
                with patch.object(api, "ejecutar_bateria", wraps=ejecucion.ejecutar_bateria) as ejecutar, \
                        patch.object(registro, "_calcular", wraps=registro._calcular) as calcular:
                    primera = self.comprobar(self.post(corte), "completa", ["creado"] * 5)
                ejecutar.assert_called_once_with(self.ruta, 7, 83, corte)
                self.assertEqual(calcular.call_count, 5)
                self.assertEqual(self.datos.conteos(), (numero * 5, numero * 5, numero * 75))
                for elemento in primera["resultados"]:
                    url = f"{PREFIJO}/{elemento['pronostico_id']}"
                    detalle = self.cliente.get(url).json()
                    cfg = self.cliente.get(url + "/configuracion").json()
                    self.assertEqual(cfg["experimento_id"], elemento["experimento_id"])
                    self.assertEqual(cfg["configuracion"], elemento["configuracion"])
                    self.assertEqual([d["fecha_objetivo"] for d in detalle["detalles"]], [
                        (date.fromisoformat(corte) + timedelta(days=i)).isoformat() for i in range(1, 16)])
                contenido = self.ruta.read_bytes()
                sentencias = []
                abrir = registro._abrir_base

                def observar(*args, **kwargs):
                    c = abrir(*args, **kwargs)
                    c.set_trace_callback(sentencias.append)
                    return c

                with patch.object(registro, "_abrir_base", side_effect=observar), \
                        patch.object(registro, "_calcular", wraps=registro._calcular) as calcular:
                    segunda = self.comprobar(self.post(corte), "completa", ["reutilizado"] * 5)
                self.assertEqual(calcular.call_count, 5)
                self.assertEqual([(e["experimento_id"], e["pronostico_id"]) for e in primera["resultados"]],
                                 [(e["experimento_id"], e["pronostico_id"]) for e in segunda["resultados"]])
                self.assertEqual(self.ruta.read_bytes(), contenido)
                self.assertFalse(any(s.lstrip().upper().startswith(
                    ("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in sentencias))

    def test_parcial_error_calculo_en_tercero_conserva_cuatro_registros(self):
        calcular = registro._calcular
        llamadas = []

        def fallar(*args, **kwargs):
            llamadas.append(None)
            if len(llamadas) == 3:
                raise registro.ErrorRegistroPronostico("error_calculo", "PRIVADO SELECT /ruta")
            return calcular(*args, **kwargs)

        with patch.object(registro, "_calcular", side_effect=fallar):
            r = self.comprobar(self.post(), "parcial", ["creado", "creado", "fallido", "creado", "creado"])
        self.assertEqual(len(llamadas), 5)
        self.assertTrue(r["contexto_verificado"])
        self.assertIsNone(r["interrupcion"])
        self.assertEqual(r["resultados"][2]["error"]["codigo"], "error_calculo")
        self.assertEqual(self.datos.conteos(), (4, 4, 60))

    def test_historial_cambiado_tras_segundo_interrumpe_y_preserva_confirmados(self):
        registrar = registro.registrar_pronostico
        llamadas = []

        def cambiar(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            llamadas.append(resultado)
            if len(llamadas) == 2:
                self.datos.cambiar("UPDATE venta_diaria SET sales=sales+0.125 WHERE serie_id=83 AND fecha=?",
                                   (CORTES[0],))
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=cambiar):
            r = self.comprobar(self.post(), "interrumpida", ["creado"] * 2 + ["no_ejecutado"] * 3)
        self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertFalse(r["contexto_verificado"])
        self.assertEqual(len(llamadas), 2)
        self.assertEqual(self.datos.conteos(), (2, 2, 30))

    def test_infraestructura_tras_primero_es_200_interrumpida_sin_rollback_global(self):
        registrar = registro.registrar_pronostico
        llamadas = []

        def fallar(*args, **kwargs):
            llamadas.append(None)
            if len(llamadas) == 2:
                raise sqlite3.OperationalError("PRIVADO SELECT " + str(self.ruta))
            return registrar(*args, **kwargs)

        with patch.object(registro, "registrar_pronostico", side_effect=fallar):
            r = self.comprobar(self.post(), "interrumpida", ["creado", "fallido"] + ["no_ejecutado"] * 3)
        self.assertEqual(r["interrupcion"]["codigo"], "base_no_disponible")
        self.assertEqual(len(llamadas), 2)
        self.assertEqual(self.datos.conteos(), (1, 1, 15))

    def test_entradas_estrictas_no_delegan(self):
        validos = {"conjunto_id": 7, "fecha_corte": CORTES[0]}
        casos = [(None, "entrada_bateria_invalida"), ([], "entrada_bateria_invalida")]
        casos += [({**validos, "conjunto_id": v}, "conjunto_invalido")
                  for v in (None, True, False, "7", 7.0, 0, -1, 2**63)]
        casos += [({**validos, "fecha_corte": v}, "corte_invalido")
                  for v in (None, True, 20170616, "2017-6-16", "2017-02-30", "2017-06-16\n")]
        casos += [({k: v for k, v in validos.items() if k != campo}, codigo)
                  for campo, codigo in (("conjunto_id", "conjunto_invalido"), ("fecha_corte", "corte_invalido"))]
        casos += [({**validos, campo: valor}, "entrada_bateria_invalida")
                  for campo, valor in (("tecnica", "random_forest"), ("configuracion", {}),
                                       ("parametros", {}), ("cantidad", 1), ("elementos", []))]
        with patch.object(api, "ejecutar_bateria") as ejecutar:
            for cuerpo, codigo in casos:
                with self.subTest(cuerpo=cuerpo):
                    self.error(self.cliente.post(f"{PREFIJO}/series/83/bateria", json=cuerpo), 422, codigo)
            for serie in ("0", "-1", "abc", "83.5", "83.0", str(2**63)):
                with self.subTest(serie=serie):
                    self.error(self.post(serie=serie), 422, "serie_invalida")
            self.error(self.cliente.post(f"{PREFIJO}/series/83/bateria", content="{PRIVADO",
                                        headers={"Content-Type": "application/json"}), 422, "entrada_bateria_invalida")
        ejecutar.assert_not_called()
        self.assertEqual(self.datos.conteos(), (0, 0, 0))

    def test_cortes_no_permitidos_y_reserva_no_registran(self):
        with patch.object(registro, "registrar_pronostico") as registrar:
            for corte, codigo in (("2017-06-17", "corte_no_autorizado"),
                                  ("2017-07-17", "prueba_reservada"),
                                  ("2017-08-01", "prueba_reservada"), ("2017-08-15", "prueba_reservada")):
                with self.subTest(corte=corte):
                    self.error(self.post(corte), 422, codigo)
        registrar.assert_not_called()

    def test_rechazos_iniciales_serie_conjunto_historial_e_infraestructura(self):
        with patch.object(registro, "registrar_pronostico") as registrar:
            self.error(self.post(serie=24), 404, "serie_inactiva")
            self.error(self.post(serie=999), 404, "serie_inexistente")
            self.error(self.post(conjunto=2), 409, "conjunto_inactivo")
            self.datos.cambiar("DELETE FROM venta_diaria WHERE serie_id=83")
            self.error(self.post(), 422, "sin_historial")
            self.datos.cambiar("UPDATE conjunto_datos SET activo=0")
            self.error(self.post(), 409, "sin_conjunto_activo")
            ausente = self.ruta.parent / "inexistente.sqlite3"
            with patch.dict(os.environ, {"TESIS_DB_PATH": str(ausente)}):
                self.error(self.post(), 503, "base_no_disponible")
            self.assertFalse(ausente.exists())
        registrar.assert_not_called()

    def test_excepciones_previas_y_inesperadas_son_publicas_y_no_cacheables(self):
        for error, codigo, estado in (
                (ejecucion.ErrorEjecucionBateria("historial_cambiado"), "historial_cambiado", 409),
                (sqlite3.OperationalError("PRIVADO SELECT /ruta"), "base_no_disponible", 503),
                (RuntimeError("PRIVADO Traceback /ruta"), "error_interno", 500)):
            with self.subTest(codigo=codigo), patch.object(api, "ejecutar_bateria", side_effect=error) as ejecutar:
                self.error(self.post(), estado, codigo)
                ejecutar.assert_called_once()

    def test_proyeccion_explicita_no_muta_ni_expone_campos_privados(self):
        interno = ejecucion.ejecutar_bateria(self.ruta, 7, 83, CORTES[0])
        interno["PRIVADO"] = "SELECT /ruta"
        interno["resumen"]["PRIVADO"] = 99
        for elemento in interno["resultados"]:
            elemento["PRIVADO"] = "SELECT /ruta"
            elemento["configuracion"]["historial"] = {"PRIVADO": 1}
            elemento["configuracion"]["parametros"]["PRIVADO"] = {"ruta": "/secreto"}
        anterior = deepcopy(interno)
        with patch.object(api, "ejecutar_bateria", return_value=interno):
            self.comprobar(self.post(), "completa", ["creado"] * 5)
        self.assertEqual(interno, anterior)

    def test_errores_publicos_no_copian_mensajes_privados_ni_ids_de_fallos(self):
        with patch.object(registro, "_calcular", side_effect=registro.ErrorRegistroPronostico("error_calculo", "PRIVADO")):
            interno = ejecucion.ejecutar_bateria(self.ruta, 7, 83, CORTES[0])
        for elemento in interno["resultados"]:
            elemento.update({"experimento_id": 999, "pronostico_id": 999})
            elemento["error"]["mensaje"] = "PRIVADO SELECT /ruta"
        anterior = deepcopy(interno)
        with patch.object(api, "ejecutar_bateria", return_value=interno):
            r = self.comprobar(self.post(), "parcial", ["fallido"] * 5)
        self.assertEqual(interno, anterior)
        self.assertTrue(all(e["error"]["mensaje"] == str(ejecucion.ErrorEjecucionBateria("error_calculo"))
                            for e in r["resultados"]))

    def test_solicitudes_concurrentes_reales_sin_duplicados_ni_orden_de_hilos(self):
        barrera = Barrier(2)
        calcular = registro._calcular

        def sincronizar(*args, **kwargs):
            resultado = calcular(*args, **kwargs)
            barrera.wait(timeout=30)  # Límite del arnés para detectar un interbloqueo del test.
            return resultado

        with patch.object(registro, "_calcular", side_effect=sincronizar), ThreadPoolExecutor(max_workers=2) as pool:
            respuestas = [f.result(timeout=90) for f in [pool.submit(self.post), pool.submit(self.post)]]
        resultados = []
        for respuesta in respuestas:
            r = respuesta.json()
            resultados.append(self.comprobar(respuesta, "completa", [e["estado"] for e in r["resultados"]]))
            self.assertTrue(all(e["estado"] in {"creado", "reutilizado"} for e in r["resultados"]))
        for a, b in zip(*(r["resultados"] for r in resultados)):
            self.assertEqual(sorted([a["estado"], b["estado"]]), ["creado", "reutilizado"])
            self.assertEqual((a["experimento_id"], a["pronostico_id"]), (b["experimento_id"], b["pronostico_id"]))
        self.assertEqual(self.datos.conteos(), (5, 5, 75))
        self.assertEqual(self.datos.leer("PRAGMA foreign_key_check"), [])

    def test_concurrentes_cambio_contexto_conserva_un_registro_confirmado(self):
        def cambiar():
            self.datos.cambiar("UPDATE venta_diaria SET sales=sales+1 WHERE serie_id=83 AND fecha=?", (CORTES[0],))

        barrera = Barrier(2, action=cambiar)
        registrar = registro.registrar_pronostico

        def sincronizar(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            barrera.wait(timeout=30)
            return resultado

        with patch.object(registro, "registrar_pronostico", side_effect=sincronizar), ThreadPoolExecutor(max_workers=2) as pool:
            respuestas = [f.result(timeout=60) for f in [pool.submit(self.post), pool.submit(self.post)]]
        primeros = []
        for respuesta in respuestas:
            estado = respuesta.json()["resultados"][0]["estado"]
            r = self.comprobar(respuesta, "interrumpida", [estado] + ["no_ejecutado"] * 4)
            primeros.append(r["resultados"][0])
            self.assertEqual(r["interrupcion"]["codigo"], "historial_cambiado")
        self.assertEqual(sorted(e["estado"] for e in primeros), ["creado", "reutilizado"])
        self.assertEqual(primeros[0]["pronostico_id"], primeros[1]["pronostico_id"])
        self.assertEqual(self.datos.conteos(), (1, 1, 15))
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{primeros[0]['pronostico_id']}").status_code, 200)

    def test_post_individual_conserva_201_200_y_get_no_genera(self):
        url = f"{PREFIJO}/series/83"
        cuerpo = {"tecnica": registro.TECNICAS[0], "fecha_corte": CORTES[0]}
        individual = self.cliente.post(url, json=cuerpo)
        self.assertEqual(individual.status_code, 201)
        bateria = self.comprobar(self.post(), "completa", ["reutilizado"] + ["creado"] * 4)
        self.assertEqual(bateria["resultados"][0]["pronostico_id"], individual.json()["pronostico_id"])
        self.assertEqual(self.cliente.post(url, json=cuerpo).status_code, 200)
        contenido = self.ruta.read_bytes()
        with patch.object(api, "ejecutar_bateria") as ejecutar, patch.object(registro, "_calcular") as calcular:
            self.assertEqual(self.cliente.get(url).status_code, 200)
            for elemento in bateria["resultados"]:
                detalle = f"{PREFIJO}/{elemento['pronostico_id']}"
                self.assertEqual(self.cliente.get(detalle).status_code, 200)
                self.assertEqual(self.cliente.get(detalle + "/configuracion").status_code, 200)
            ejecutar.assert_not_called()
            calcular.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_arranque_openapi_y_documentacion_no_ejecutan_ni_abren_sqlite(self):
        namespace = {"__name__": "arranque_sintetico_i1b2"}
        codigo = compile(Path("backend/app/main.py").read_text(encoding="utf-8"), "main_sintetico.py", "exec")
        with patch.object(api, "ejecutar_bateria") as ejecutar, patch.object(registro, "_calcular") as calcular, \
                patch.object(sqlite3, "connect", side_effect=AssertionError("No abrir al arrancar")) as abrir:
            exec(codigo, namespace)
            with TestClient(namespace["app"]) as cliente:
                self.assertEqual(cliente.get("/health").status_code, 200)
                self.assertEqual(cliente.get("/docs").status_code, 200)
                esquema = cliente.get("/openapi.json").json()
            for doble in (ejecutar, calcular, abrir):
                doble.assert_not_called()
        ruta = esquema["paths"][f"{PREFIJO}/series/{{serie_id}}/bateria"]["post"]
        self.assertTrue(ruta["requestBody"]["required"])
        self.assertEqual(set(ruta["responses"]), {"200", "404", "409", "422", "500", "503"})
        modelos = esquema["components"]["schemas"]
        self.assertFalse(modelos["SolicitudBateria"]["additionalProperties"])
        self.assertEqual(set(modelos["SolicitudBateria"]["required"]), {"conjunto_id", "fecha_corte"})
        self.assertEqual(set(modelos["ResultadoBateria"]["properties"]["estado"]["enum"]),
                         {"completa", "parcial", "interrumpida"})
        self.assertIn("desconexión", ruta["description"])
        self.assertFalse(inspect.iscoroutinefunction(api.generar_bateria))

    def test_desconexion_simulada_asgi_no_garantiza_cancelacion_ni_resumen(self):
        primero, continuar, desconectado = Event(), Event(), Event()
        registrar = registro.registrar_pronostico

        def pausar(*args, **kwargs):
            resultado = registrar(*args, **kwargs)
            if not primero.is_set():
                primero.set()
                if not continuar.wait(timeout=30):
                    raise RuntimeError("Arnés no liberó la solicitud")
            return resultado

        async def simular():
            entregado = False
            cuerpo = json.dumps({"conjunto_id": 7, "fecha_corte": CORTES[0]}).encode()

            async def recibir():
                nonlocal entregado
                if not entregado:
                    entregado = True
                    return {"type": "http.request", "body": cuerpo, "more_body": False}
                return {"type": "http.disconnect"}

            async def enviar(mensaje):
                if desconectado.is_set():
                    raise OSError("Transporte sintético desconectado")

            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                     "http_version": "1.1", "method": "POST", "scheme": "http", "root_path": "",
                     "path": f"{PREFIJO}/series/83/bateria", "query_string": b"",
                     "headers": [(b"content-type", b"application/json")],
                     "client": ("127.0.0.1", 12345), "server": ("sintetico", 80)}
            tarea = asyncio.create_task(app(scope, recibir, enviar))
            try:
                self.assertTrue(await asyncio.to_thread(primero.wait, 30))
                desconectado.set()
            finally:
                continuar.set()
            with self.assertRaises(OSError):
                await tarea

        with patch.object(registro, "registrar_pronostico", side_effect=pausar):
            asyncio.run(simular())
        self.assertTrue(desconectado.is_set())
        self.assertEqual(self.datos.conteos(), (5, 5, 75))
        self.assertEqual(len(self.cliente.get(f"{PREFIJO}/series/83").json()["pronosticos"]), 5)


if __name__ == "__main__":
    unittest.main()
