"""API de pronósticos con TestClient y bases SQLite temporales por prueba."""

from contextlib import closing
from datetime import date, timedelta
import inspect
import json
import os
import sqlite3
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from backend.app.api import pronosticos as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import consulta_pronosticos as consulta
from backend.app.services import registro_pronosticos as registro
from backend.app.services import configuracion_experimentos as configuraciones
from backend.app.storage.db import conectar
from tests import test_registro_pronosticos as fixtures


BASE = "estacional_semanal_recursiva"
PREFIJO = "/api/pronosticos/active"
OMITIDA = object()


class ApiPronosticosTests(unittest.TestCase):
    def setUp(self):
        # Reutiliza únicamente la preparación de datos sintéticos de HU004-D1.
        self.datos = fixtures.RegistroPronosticosTests()
        self.datos.setUp()
        self.addCleanup(self.datos.doCleanups)
        self.ruta = self.datos.ruta
        anteriores = dict(app.dependency_overrides)
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))

    def post(self, tecnica=BASE, corte="2017-06-16", serie=83, *, configuracion=OMITIDA):
        cuerpo = {"tecnica": tecnica, "fecha_corte": corte}
        if configuracion is not OMITIDA:
            cuerpo["configuracion"] = configuracion
        return self.cliente.post(f"{PREFIJO}/series/{serie}", json=cuerpo)

    def crear(self, tecnica=BASE, corte="2017-06-16", serie=83, *, configuracion=OMITIDA):
        respuesta = self.post(tecnica, corte, serie, configuracion=configuracion)
        self.assertEqual(respuesta.status_code, 201, respuesta.text)
        return respuesta.json()

    def comprobar_respuesta_publica(self, respuesta):
        self.assertEqual(set(respuesta), {"conjunto_id", "serie_id", "experimento_id", "pronostico_id",
                                          "tecnica", "fecha_corte", "horizonte", "cantidad_detalles", "nuevo", "reutilizado"})
        self.assertEqual((respuesta["horizonte"], respuesta["cantidad_detalles"]), (15, 15))

    def crear_cuatro_http(self):
        casos = [
            ("random_forest", OMITIDA),
            ("random_forest", {"tecnica": "random_forest", "parametros": {
                "n_estimators": 200, "max_depth": 4, "random_state": 7}}),
            ("hist_gradient_boosting", OMITIDA),
            ("hist_gradient_boosting", {"tecnica": "hist_gradient_boosting", "parametros": {
                "max_iter": 150, "learning_rate": 0.1, "l2_regularization": 0.5, "random_state": 11}}),
        ]
        return [(t, c, self.crear(t, configuracion=c)) for t, c in casos]

    def comprobar_alternativo_real(self, tecnica, parametros):
        default = self.crear(tecnica)
        anterior = self.cliente.get(f"{PREFIJO}/{default['pronostico_id']}").json()
        solicitud = {"tecnica": tecnica, "parametros": parametros}
        modelos = []
        entrenar = fixtures.modelos.entrenar_modelo

        def observar(*args, **kwargs):
            m = entrenar(*args, **kwargs)  # fit real; no se sustituye ningún resultado
            modelos.append(m)
            return m

        with patch.object(api, "registrar_pronostico", wraps=registro.registrar_pronostico) as registrar, \
                patch.object(fixtures.modelos, "entrenar_modelo", side_effect=observar):
            alternativo = self.crear(tecnica, configuracion=solicitud)
        registrar.assert_called_once_with(self.ruta, 83, tecnica, "2017-06-16",
            configuracion_solicitada=configuraciones.normalizar_configuracion(solicitud))
        self.assertEqual(len(modelos), 1)
        estimador = modelos[0].estimador
        if tecnica == "random_forest":
            self.assertEqual(len(estimador.estimators_), parametros["n_estimators"])
        else:
            self.assertEqual(estimador.n_iter_, parametros["max_iter"])
        c = json.loads(self.datos.consultar("SELECT configuracion_json FROM experimento WHERE id=?",
                                           (alternativo["experimento_id"],))[0][0])
        self.assertEqual(json.dumps(c["parametros"], sort_keys=True),
                         json.dumps(estimador.get_params(deep=False), sort_keys=True))
        for k, v in parametros.items():
            self.assertEqual(c["parametros"][k], v)
        self.assertNotEqual(default["experimento_id"], alternativo["experimento_id"])
        self.assertNotEqual(default["pronostico_id"], alternativo["pronostico_id"])
        contenido = self.ruta.read_bytes()
        for config, previo in ((solicitud, alternativo), (OMITIDA, default)):
            r = self.post(tecnica, configuracion=config)
            self.assertEqual(r.status_code, 200, r.text)
            self.comprobar_respuesta_publica(r.json())
            self.assertTrue(r.json()["reutilizado"])
            self.assertEqual((r.json()["experimento_id"], r.json()["pronostico_id"]),
                             (previo["experimento_id"], previo["pronostico_id"]))
            self.assertEqual(len(self.cliente.get(f"{PREFIJO}/{previo['pronostico_id']}").json()["detalles"]), 15)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{default['pronostico_id']}").json(), anterior)
        self.assertEqual(self.datos.conteos(), (2, 2, 30, 0, 0))
        # Otro parámetro crea otro experimento; no se exige que las estimaciones difieran.
        nuevos = dict(parametros)
        nuevos["random_state"] = 43
        tercero = self.crear(tecnica, configuracion={"tecnica": tecnica, "parametros": nuevos})
        self.assertEqual(len({r["experimento_id"] for r in (default, alternativo, tercero)}), 3)
        self.comprobar_respuesta_publica(alternativo)

    def test_post_random_forest_configurado_entrena_200_arboles_y_reutiliza_independientemente(self):
        self.comprobar_alternativo_real("random_forest", {"n_estimators": 200, "max_depth": 4, "random_state": 7})

    def test_post_hist_gradient_boosting_configurado_entrena_150_iteraciones_y_reutiliza(self):
        self.comprobar_alternativo_real("hist_gradient_boosting", {"max_iter": 150, "learning_rate": 0.1,
                                                                "l2_regularization": 0.5, "random_state": 11})

    def test_cuatro_experimentos_http_conservan_formato_historico_y_reutilizan_sin_dml(self):
        historicos = self.datos.historicos()
        casos = self.crear_cuatro_http()
        self.assertEqual(len({r["experimento_id"] for _, _, r in casos}), 4)
        self.assertEqual(len({r["pronostico_id"] for _, _, r in casos}), 4)
        self.assertEqual(self.datos.conteos(), (4, 4, 60, 0, 0))
        contenido = self.ruta.read_bytes()
        sentencias = []
        abrir = registro._abrir_base

        def observar(*args, **kwargs):
            c = abrir(*args, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c

        with patch.object(registro, "_abrir_base", side_effect=observar):
            for t, c, previo in reversed(casos):
                respuesta = self.post(t, configuracion=c)
                self.assertEqual(respuesta.status_code, 200, respuesta.text)
                self.assertEqual(respuesta.json(), {**previo, "nuevo": False, "reutilizado": True})
                self.comprobar_respuesta_publica(respuesta.json())
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in sentencias))
        lista = self.cliente.get(f"{PREFIJO}/series/83")
        self.assertEqual({p["id"] for p in lista.json()["pronosticos"]}, {r["pronostico_id"] for _, _, r in casos})
        for _, _, r in casos:
            detalle = self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}")
            self.assertEqual(len(detalle.json()["detalles"]), 15)
            self.assertEqual(set(detalle.json()), {"conjunto_id", "serie", "pronostico", "detalles"})
            self.assertEqual(set(detalle.json()["pronostico"]), {
                "id", "experimento_id", "tecnica", "fecha_corte", "tipo", "horizonte", "fecha_generacion",
                "inicio_entrenamiento", "fin_entrenamiento", "duracion_segundos"})
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.datos.historicos(), historicos)
        self.assertEqual(self.datos.consultar("PRAGMA foreign_key_check"), [])

    def test_configuraciones_invalidas_son_422_sin_registro_entrenamiento_ni_conexiones(self):
        casos = [None, [], "PRIVADO C:/datos/secreto", 1, True, {},
                 {"parametros": {"n_estimators": 200}},
                 {"tecnica": "hist_gradient_boosting"}, {"tecnica": "otra"},
                 {"tecnica": "random_forest", "otro": "PRIVADO C:/datos/secreto"},
                 {"tecnica": "random_forest", "parametros": []}]
        for parametros in ({"otro": 1}, {"n_estimators": 0}, {"n_estimators": 501}, {"n_estimators": True},
                           {"n_estimators": 100.0}, {"n_estimators": "200"}, {"random_state": False},
                           {"max_depth": -1}, {"n_jobs": 2}, {"warm_start": True},
                           {"ccp_alpha": 0.1}, {"max_features": 7}, {"max_features": 0}):
            casos.append({"tecnica": "random_forest", "parametros": parametros})
        for variables in ([], list(configuraciones.VARIABLES[:-1]), [*configuraciones.VARIABLES, "onpromotion"],
                          list(reversed(configuraciones.VARIABLES)), [*configuraciones.VARIABLES, "lag_1"]):
            casos.append({"tecnica": "random_forest", "variables": variables})
        for campo, valor in (("horizonte", 30), ("horizonte", 15.0), ("version_contrato", 2),
                              ("version_contrato", True), ("version_reglas", 2)):
            casos.append({"tecnica": "random_forest", campo: valor})
        contenido = self.ruta.read_bytes()
        with patch.object(api, "registrar_pronostico") as registrar, \
                patch.object(fixtures.modelos, "entrenar_modelo") as entrenar, \
                patch.object(registro, "_abrir_base") as abrir:
            for c in casos:
                with self.subTest(configuracion=c):
                    respuesta = self.post("random_forest", configuracion=c)
                    self.assertEqual(respuesta.status_code, 422, respuesta.text)
                    self.assertEqual(respuesta.json(), {"detail": {"codigo": "configuracion_invalida",
                                                       "mensaje": api.ERRORES["configuracion_invalida"][1]}})
                    self.assertNotIn("PRIVADO", respuesta.text)
                    self.assertNotIn("C:/datos", respuesta.text)
            registrar.assert_not_called()
            entrenar.assert_not_called()
            abrir.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.datos.conteos(), (0, 0, 0, 0, 0))

    def test_hgb_y_linea_base_rechazan_cambios_fijos_y_rangos_de_f2(self):
        casos = [("hist_gradient_boosting", {"early_stopping": True}),
                 ("hist_gradient_boosting", {"max_iter": 501}),
                 ("hist_gradient_boosting", {"learning_rate": 0}),
                 ("hist_gradient_boosting", {"max_features": 1.1}),
                 ("hist_gradient_boosting", {"l2_regularization": -0.1}),
                 (BASE, {"periodo_estacional_dias": 14}), (BASE, {"random_state": 42})]
        with patch.object(api, "registrar_pronostico") as registrar:
            for t, p in casos:
                with self.subTest(tecnica=t, parametros=p):
                    r = self.post(t, configuracion={"tecnica": t, "parametros": p})
                    self.assertEqual(r.status_code, 422)
                    self.assertEqual(r.json()["detail"]["codigo"], "configuracion_invalida")
            registrar.assert_not_called()

    def test_numeros_no_finitos_recibidos_en_json_son_422(self):
        with patch.object(api, "registrar_pronostico") as registrar, \
                patch.object(fixtures.modelos, "entrenar_modelo") as entrenar:
            for numero in ("NaN", "Infinity", "-Infinity", "1e309"):
                cuerpo = ('{"tecnica":"hist_gradient_boosting","fecha_corte":"2017-06-16",'
                          '"configuracion":{"tecnica":"hist_gradient_boosting","parametros":{"learning_rate":'
                          + numero + '}}}')
                r = self.cliente.post(f"{PREFIJO}/series/83", content=cuerpo, headers={"Content-Type": "application/json"})
                self.assertEqual(r.status_code, 422, r.text)
                self.assertEqual(r.json()["detail"]["codigo"], "configuracion_invalida")
            registrar.assert_not_called()
            entrenar.assert_not_called()

    def test_error_de_validacion_no_expone_excepcion_interna(self):
        with patch.object(api.configuraciones, "normalizar_configuracion", side_effect=ValueError("PRIVADO C:/secreto")), \
                patch.object(api, "registrar_pronostico") as registrar:
            r = self.post("random_forest", configuracion={"tecnica": "random_forest"})
            self.assertEqual(r.status_code, 422)
            self.assertNotIn("PRIVADO", r.text)
            self.assertNotIn("C:/secreto", r.text)
            registrar.assert_not_called()

    def test_max_features_entero_y_fraccion_preservan_tipo_identidad_y_reutilizacion(self):
        filas = []
        for valor in (1, 1.0):
            c = {"tecnica": "random_forest", "parametros": {"n_estimators": 20, "max_features": valor}}
            r = self.crear("random_forest", configuracion=c)
            texto = self.datos.consultar("SELECT configuracion_json FROM experimento WHERE id=?", (r["experimento_id"],))[0][0]
            guardada = json.loads(texto)
            self.assertIs(type(guardada["parametros"]["max_features"]), type(valor))
            filas.append((r, guardada))
            repetido = self.post("random_forest", configuracion=c)
            self.assertEqual(repetido.status_code, 200)
            self.assertEqual(repetido.json()["pronostico_id"], r["pronostico_id"])
        self.assertNotEqual(filas[0][0]["experimento_id"], filas[1][0]["experimento_id"])
        self.assertNotEqual(registro._contenido_config(filas[0][1]), registro._contenido_config(filas[1][1]))

    def test_linea_base_configurada_explicita_reutiliza_el_default_original(self):
        r = self.crear()
        contenido = self.ruta.read_bytes()
        respuesta = self.post(configuracion={"tecnica": BASE})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json()["pronostico_id"], r["pronostico_id"])
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_defaults_explicitos_se_normalizan_y_reutilizan_el_post_antiguo(self):
        for t in ("random_forest", "hist_gradient_boosting"):
            r = self.crear(t)
            contenido = self.ruta.read_bytes()
            respuesta = self.post(t, configuracion={"tecnica": t})
            self.assertEqual(respuesta.status_code, 200)
            self.assertEqual(respuesta.json()["experimento_id"], r["experimento_id"])
            self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_post_antiguo_de_regresores_sigue_delegando_cuatro_argumentos_sin_kwargs(self):
        for t in ("random_forest", "hist_gradient_boosting"):
            with patch.object(api, "registrar_pronostico", wraps=registro.registrar_pronostico) as registrar:
                r = self.crear(t)
            registrar.assert_called_once_with(self.ruta, 83, t, "2017-06-16")
            self.comprobar_respuesta_publica(r)

    def test_configuracion_no_habilita_campos_externos_ni_tecnicas_invalidas(self):
        cuerpo = {"tecnica": "random_forest", "fecha_corte": "2017-06-16",
                  "configuracion": {"tecnica": "random_forest"}, "campo_desconocido": "PRIVADO"}
        with patch.object(api, "registrar_pronostico") as registrar:
            r = self.cliente.post(f"{PREFIJO}/series/83", json=cuerpo)
            self.assertEqual(r.status_code, 422)
            self.assertEqual(r.json()["detail"]["codigo"], "entrada_invalida")
            self.assertNotIn("PRIVADO", r.text)
            r = self.post("otra", configuracion={"tecnica": "random_forest"})
            self.assertEqual(r.status_code, 422)
            self.assertEqual(r.json()["detail"]["codigo"], "tecnica_invalida")
            registrar.assert_not_called()

    def test_configurada_conserva_cortes_autorizados_y_bloqueo_prueba_final(self):
        # Completar solo esta fixture temporal hasta el tercer corte: los regresores
        # exigen rezagos disponibles, a diferencia de los respaldos de la línea base.
        with closing(conectar(self.ruta)) as conexion, conexion:
            conexion.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                                 [(83, (date(2017, 7, 1) + timedelta(days=i)).isoformat(), i * 0.25, 0)
                                  for i in range(1, 16)])
        c = {"tecnica": "random_forest", "parametros": {"n_estimators": 20}}
        for corte in ("2017-06-16", "2017-07-01", "2017-07-16"):
            r = self.crear("random_forest", corte, configuracion=c)
            self.assertEqual(r["fecha_corte"], corte)
        for corte, codigo in (("2017-06-15", "corte_no_autorizado"), ("2017-07-02", "corte_no_autorizado"),
                               ("2017-07-31", "prueba_reservada")):
            with patch.object(fixtures.modelos, "entrenar_modelo") as entrenar:
                r = self.post("random_forest", corte, configuracion=c)
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()["detail"]["codigo"], codigo)
                entrenar.assert_not_called()
        self.assertEqual(self.datos.conteos(), (3, 3, 45, 0, 0))

    def test_registros_anteriores_se_consultan_y_reutilizan_desde_http_sin_migrar(self):
        casos = [("random_forest", 1), ("hist_gradient_boosting", 2)]
        for t, version in casos:
            r = self.crear(t)
            if version == 1:
                self.datos.convertir_a_formato_antiguo(r["experimento_id"])
            contenido = self.ruta.read_bytes()
            self.assertEqual(self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}").status_code, 200)
            respuesta = self.post(t, configuracion={"tecnica": t})
            self.assertEqual(respuesta.status_code, 200)
            self.assertEqual(respuesta.json()["pronostico_id"], r["pronostico_id"])
            self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_get_configuradas_es_solo_lectura_y_no_expone_campos_internos(self):
        r = self.crear("random_forest", configuracion={"tecnica": "random_forest", "parametros": {"n_estimators": 20}})
        contenido = self.ruta.read_bytes()
        with patch.object(api, "registrar_pronostico") as registrar, \
                patch.object(fixtures.modelos, "entrenar_modelo") as entrenar, \
                patch.object(registro, "_abrir_base") as abrir:
            for url in (f"{PREFIJO}/series/83", f"{PREFIJO}/{r['pronostico_id']}"):
                respuesta = self.cliente.get(url)
                self.assertEqual(respuesta.status_code, 200)
                for campo in ("configuracion", "parametros", "sha256", "historial", "auditoria", "versiones", "ruta_modelo"):
                    self.assertNotIn(campo, respuesta.text)
            registrar.assert_not_called()
            entrenar.assert_not_called()
            abrir.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_openapi_documenta_configuracion_opcional_objeto_no_nulo(self):
        esquema = self.cliente.get("/openapi.json").json()["components"]["schemas"]["SolicitudPronostico"]
        self.assertEqual(set(esquema["required"]), {"tecnica", "fecha_corte"})
        self.assertEqual(esquema["properties"]["configuracion"]["type"], "object")
        self.assertNotIn("anyOf", esquema["properties"]["configuracion"])
        self.assertIs(esquema["additionalProperties"], False)
        self.assertIn("contrato F2", esquema["properties"]["configuracion"]["description"])
        self.assertEqual(esquema["properties"]["configuracion"]["examples"][0]["parametros"]["n_estimators"], 200)

    def test_get_serie_activa_sin_pronosticos(self):
        respuesta = self.cliente.get(f"{PREFIJO}/series/83")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json(), {
            "conjunto_id": 7, "serie": {"id": 83, "store_nbr": 17, "family": "PREPARED FOODS"},
            "pronosticos": [],
        })

    def test_post_linea_base_crea_y_delega_solo_la_solicitud(self):
        with patch.object(api, "registrar_pronostico", wraps=registro.registrar_pronostico) as generar:
            r = self.crear()
        generar.assert_called_once_with(self.ruta, 83, BASE, "2017-06-16")
        self.assertEqual(set(r), {"conjunto_id", "serie_id", "experimento_id", "pronostico_id",
                                  "tecnica", "fecha_corte", "horizonte", "cantidad_detalles", "nuevo", "reutilizado"})
        self.assertEqual((r["conjunto_id"], r["serie_id"], r["horizonte"], r["cantidad_detalles"]), (7, 83, 15, 15))
        self.assertTrue(r["nuevo"])
        self.assertFalse(r["reutilizado"])
        self.assertEqual(self.datos.conteos(), (1, 1, 15, 0, 0))

    def test_post_random_forest_real(self):
        self.assertEqual(self.crear("random_forest")["cantidad_detalles"], 15)

    def test_post_hist_gradient_boosting_real(self):
        self.assertEqual(self.crear("hist_gradient_boosting")["cantidad_detalles"], 15)

    def test_post_identico_devuelve_200_reutiliza_sin_escribir(self):
        primero = self.crear()
        contenido = self.ruta.read_bytes()
        respuesta = self.post()
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        segundo = respuesta.json()
        self.assertTrue(segundo["reutilizado"])
        self.assertFalse(segundo["nuevo"])
        self.assertEqual((primero["experimento_id"], primero["pronostico_id"]),
                         (segundo["experimento_id"], segundo["pronostico_id"]))
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_get_lista_y_detalle_quince_fechas_y_pasos(self):
        creado = self.crear()
        listado = self.cliente.get(f"{PREFIJO}/series/83")
        self.assertEqual(listado.status_code, 200)
        metadatos = listado.json()["pronosticos"][0]
        for campo in ("id", "tecnica", "fecha_corte", "tipo", "horizonte", "fecha_generacion",
                      "inicio_entrenamiento", "fin_entrenamiento"):
            self.assertIn(campo, metadatos)
        detalle = self.cliente.get(f"{PREFIJO}/{creado['pronostico_id']}")
        self.assertEqual(detalle.status_code, 200)
        cuerpo = detalle.json()
        self.assertEqual(cuerpo["pronostico"], metadatos)
        self.assertEqual(cuerpo["conjunto_id"], 7)
        self.assertEqual(cuerpo["serie"]["id"], 83)
        filas = cuerpo["detalles"]
        self.assertEqual(len(filas), 15)
        self.assertEqual([f["paso"] for f in filas], list(range(1, 16)))
        self.assertEqual([f["fecha_objetivo"] for f in filas],
                         [(date(2017, 6, 16) + timedelta(days=d)).isoformat() for d in range(1, 16)])
        guardados = self.datos.consultar("SELECT paso,fecha_objetivo,venta_estimada FROM detalle_pronostico ORDER BY paso")
        self.assertEqual([(f["paso"], f["fecha_objetivo"], f["venta_estimada"]) for f in filas], guardados)

    def test_lista_con_orden_estable_por_corte_tecnica_id(self):
        self.crear()
        self.crear(corte="2017-07-01")
        # Una técnica adicional basta para verificar el desempate del listado.
        self.crear("random_forest")
        filas = self.cliente.get(f"{PREFIJO}/series/83").json()["pronosticos"]
        self.assertEqual([f["fecha_corte"] for f in filas], ["2017-07-01", "2017-06-16", "2017-06-16"])
        self.assertEqual([f["tecnica"] for f in filas[1:]], [BASE, "random_forest"])

    def test_get_no_genera_entrena_ni_escribe_y_usa_mode_ro(self):
        r = self.crear()
        contenido = self.ruta.read_bytes()
        historicos = self.datos.historicos()
        conectar_original = sqlite3.connect
        aperturas = []
        sentencias = []

        def observar(database, **kwargs):
            aperturas.append((database, kwargs))
            c = conectar_original(database, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c

        with patch.object(api, "registrar_pronostico", side_effect=AssertionError("GET no debe generar")), \
                patch.object(registro, "_calcular", side_effect=AssertionError("GET no debe calcular")), \
                patch.object(fixtures.modelos, "entrenar_modelo", side_effect=AssertionError("GET no debe entrenar")), \
                patch.object(consulta.sqlite3, "connect", side_effect=observar):
            self.assertEqual(self.cliente.get(f"{PREFIJO}/series/83").status_code, 200)
            self.assertEqual(self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}").status_code, 200)
        self.assertEqual(len(aperturas), 2)
        self.assertTrue(all(uri.endswith("?mode=ro") and args["uri"] for uri, args in aperturas))
        self.assertFalse(any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
                             for sql in sentencias))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.datos.historicos(), historicos)

    def test_serie_inactiva_get_y_post_404(self):
        self.assertEqual(self.cliente.get(f"{PREFIJO}/series/24").status_code, 404)
        self.assertEqual(self.post(serie=24).status_code, 404)
        self.assertEqual(self.datos.conteos(), (0, 0, 0, 0, 0))

    def test_serie_y_pronostico_inexistentes_404(self):
        self.assertEqual(self.post(serie=999).status_code, 404)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/series/999").status_code, 404)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/999").status_code, 404)

    def test_cambio_activo_aisla_listas_y_detalles(self):
        anterior = self.crear()
        with closing(conectar(self.ruta)) as c, c:
            c.execute("UPDATE conjunto_datos SET activo=0 WHERE id=7")
            c.execute("UPDATE conjunto_datos SET activo=1 WHERE id=2")
        self.assertEqual(self.cliente.get(f"{PREFIJO}/series/83").status_code, 404)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{anterior['pronostico_id']}").status_code, 404)
        self.assertEqual(self.post(serie=83).status_code, 404)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/series/24").json()["pronosticos"], [])
        nuevo = self.crear(serie=24)
        detalle = self.cliente.get(f"{PREFIJO}/{nuevo['pronostico_id']}").json()
        self.assertEqual((detalle["conjunto_id"], detalle["serie"]["id"]), (2, 24))
        self.assertEqual(self.datos.conteos(), (2, 2, 30, 0, 0))

    def test_tecnica_desconocida_422(self):
        r = self.post(tecnica="otra")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["codigo"], "tecnica_invalida")

    def test_fechas_invalidas_y_no_autorizadas_422(self):
        for fecha in ("2017-02-30", "2017-06-15", "20170616", "2017-06-16T00:00:00", "2017-07-02"):
            with self.subTest(fecha=fecha):
                self.assertEqual(self.post(corte=fecha).status_code, 422)
        self.assertEqual(self.datos.conteos(), (0, 0, 0, 0, 0))

    def test_prueba_final_reservada_422(self):
        for corte in ("2017-07-17", "2017-07-31", "2017-08-01"):
            with self.subTest(corte=corte):
                r = self.post(corte=corte)
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()["detail"]["codigo"], "prueba_reservada")

    def test_campos_obligatorios_tipos_y_extras(self):
        for cuerpo in ({}, {"tecnica": BASE}, {"fecha_corte": "2017-06-16"},
                       {"tecnica": None, "fecha_corte": "2017-06-16"},
                       {"tecnica": BASE, "fecha_corte": 1497571200},
                       {"tecnica": BASE, "fecha_corte": "2017-06-16", "horizonte": 30}):
            with self.subTest(cuerpo=cuerpo):
                r = self.cliente.post(f"{PREFIJO}/series/83", json=cuerpo)
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()["detail"]["codigo"], "entrada_invalida")
        self.assertEqual(self.cliente.post(f"{PREFIJO}/series/83").status_code, 422)
        for identificador in ("0", "-1", "abc"):
            self.assertEqual(self.cliente.get(f"{PREFIJO}/series/{identificador}").status_code, 422)

    def test_error_generacion_revierte_y_oculta_sqlite(self):
        self.datos.consultar("CREATE TRIGGER fallo BEFORE INSERT ON detalle_pronostico WHEN NEW.paso=8 "
                             "BEGIN SELECT RAISE(ABORT,'SQLITE PRIVADO C:/secreto/tesis.sqlite3'); END")
        historicos = self.datos.historicos()
        r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("PRIVADO", r.text)
        self.assertNotIn(str(self.ruta), r.text)
        self.assertEqual(self.datos.conteos(), (0, 0, 0, 0, 0))
        self.assertEqual(self.datos.historicos(), historicos)

    def test_conflicto_409_sin_sobrescribir(self):
        self.crear()
        self.datos.consultar("UPDATE detalle_pronostico SET venta_estimada=99 WHERE paso=1")
        contenido = self.ruta.read_bytes()
        self.assertEqual(self.post().status_code, 409)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_base_inexistente_sin_esquema_y_danada_503(self):
        rutas = [self.ruta.with_name(n) for n in ("ausente.sqlite3", "vacia.sqlite3", "danada.sqlite3")]
        rutas[1].touch()
        rutas[2].write_bytes(b"no es sqlite")
        for ruta in rutas:
            app.dependency_overrides[obtener_ruta_base] = lambda ruta=ruta: ruta
            for metodo, url in (("get", f"{PREFIJO}/series/83"), ("get", f"{PREFIJO}/1"),
                                ("post", f"{PREFIJO}/series/83")):
                with self.subTest(ruta=ruta.name, metodo=metodo):
                    r = getattr(self.cliente, metodo)(url, **({"json": {"tecnica": BASE, "fecha_corte": "2017-06-16"}}
                                                            if metodo == "post" else {}))
                    self.assertEqual(r.status_code, 503, r.text)
                    self.assertNotIn(str(ruta), r.text)
                    self.assertNotIn("no such table", r.text)
        self.assertFalse(rutas[0].exists())
        self.assertEqual(rutas[1].stat().st_size, 0)

    def test_get_detalle_incompleto_409_sin_inventar_datos(self):
        r = self.crear()
        self.datos.consultar("DELETE FROM detalle_pronostico WHERE paso=15")
        contenido = self.ruta.read_bytes()
        respuesta = self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}")
        self.assertEqual(respuesta.status_code, 409)
        self.assertEqual(respuesta.json()["detail"]["codigo"], "pronostico_corrupto")
        self.assertEqual(self.datos.conteos()[2], 14)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_get_detalle_fechas_y_valores_corruptos(self):
        r = self.crear()
        for columna, invalido in (("fecha_objetivo", "2017-06-99"), ("fecha_objetivo", "2018-01-01"),
                                   ("venta_estimada", -1), ("venta_estimada", float("inf")),
                                   ("venta_estimada", "NO_NUMERICO")):
            original = self.datos.consultar(f"SELECT {columna} FROM detalle_pronostico WHERE paso=1")[0][0]
            self.datos.consultar(f"UPDATE detalle_pronostico SET {columna}=? WHERE paso=1", (invalido,))
            self.assertEqual(self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}").status_code, 409)
            self.datos.consultar(f"UPDATE detalle_pronostico SET {columna}=? WHERE paso=1", (original,))

    def test_get_detecta_pasos_y_horizonte_corruptos(self):
        r = self.crear()
        with closing(conectar(self.ruta)) as c, c:
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute("UPDATE detalle_pronostico SET paso=16 WHERE paso=15")
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}").status_code, 409)
        with closing(conectar(self.ruta)) as c, c:
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute("UPDATE detalle_pronostico SET paso=15 WHERE paso=16")
            c.execute("UPDATE pronostico SET horizonte=14")
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{r['pronostico_id']}").status_code, 409)

    def test_sin_conjunto_activo_409(self):
        self.datos.consultar("UPDATE conjunto_datos SET activo=0")
        self.assertEqual(self.post().status_code, 409)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/series/83").status_code, 409)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/1").status_code, 409)

    def test_correspondencia_codigos_servicio_y_errores_sanitizados(self):
        casos = {"serie_invalida": 422, "tecnica_invalida": 422, "corte_invalido": 422,
                 "corte_no_autorizado": 422, "prueba_reservada": 422, "serie_inexistente": 404,
                 "serie_inactiva": 404, "sin_conjunto_activo": 409, "historial_cambiado": 409,
                 "conflicto": 409, "sin_historial": 422, "error_calculo": 422,
                 "base_no_disponible": 503, "error_persistencia": 503, "resultado_invalido": 500,
                 "desconocido": 500}
        for codigo, estado in casos.items():
            with self.subTest(codigo=codigo), patch.object(api, "registrar_pronostico", side_effect=
                    registro.ErrorRegistroPronostico(codigo, "traza privada C:/datos/tesis.sqlite3")):
                r = self.post()
                self.assertEqual(r.status_code, estado)
                self.assertNotIn("privada", r.text)
                self.assertNotIn("C:/datos", r.text)

    def test_errores_inesperados_500_sin_exponer_trazas(self):
        with patch.object(api, "registrar_pronostico", side_effect=RuntimeError("secreto SQL privado")):
            r = self.post()
            self.assertEqual(r.status_code, 500)
            self.assertNotIn("secreto", r.text)
        with patch.object(api, "listar_pronosticos", side_effect=RuntimeError("secreto de lectura")):
            self.assertEqual(self.cliente.get(f"{PREFIJO}/series/83").status_code, 500)

    def test_no_expone_ruta_modelo_ni_configuracion_interna(self):
        r = self.crear()
        self.datos.consultar("UPDATE experimento SET ruta_modelo='C:/ruta/privada/modelo.pkl'")
        for url in (f"{PREFIJO}/series/83", f"{PREFIJO}/{r['pronostico_id']}"):
            respuesta = self.cliente.get(url)
            self.assertEqual(respuesta.status_code, 200)
            self.assertNotIn("C:/ruta", respuesta.text)
            self.assertNotIn("configuracion_json", respuesta.text)

    def test_health_y_endpoints_sprint_uno_conservan_contratos(self):
        self.assertEqual(self.cliente.get("/health").json(), {"status": "ok"})
        series = self.cliente.get("/api/datasets/active/series")
        self.assertEqual(series.status_code, 200)
        self.assertEqual(set(series.json()), {"conjunto_id", "series"})
        historico = self.cliente.get("/api/datasets/active/series/83/sales")
        self.assertEqual(historico.status_code, 200)
        self.assertEqual(set(historico.json()), {"conjunto_id", "serie", "observaciones"})
        csv = b"date,store_nbr,family,sales,onpromotion\n2017-06-14,17,PREPARED FOODS,0,0\n2017-06-16,17,PREPARED FOODS,2.5,3\n"
        carga = self.cliente.post("/api/datasets/upload", files={"file": ("ventas.csv", csv, "text/csv")})
        self.assertEqual(carga.status_code, 200, carga.text)
        self.assertTrue(carga.json()["exito"])
        self.assertEqual(set(carga.json()), {"exito", "conjunto_id", "sha256", "series_importadas",
                                            "observaciones_importadas", "errores", "advertencias"})
        nuevas = self.cliente.get("/api/datasets/active/series").json()["series"]
        historico = self.cliente.get(f"/api/datasets/active/series/{nuevas[0]['id']}/sales").json()
        self.assertEqual([v["sales"] for v in historico["observaciones"]], [0, 2.5])

    def test_openapi_docs_y_post_sincrono(self):
        respuesta = self.cliente.get("/openapi.json")
        self.assertEqual(respuesta.status_code, 200)
        esquema = respuesta.json()
        paths = esquema["paths"]
        self.assertEqual(set(paths["/api/pronosticos/active/series/{serie_id}"]), {"get", "post"})
        self.assertIn("get", paths["/api/pronosticos/active/{pronostico_id}"])
        self.assertIn("201", paths["/api/pronosticos/active/series/{serie_id}"]["post"]["responses"])
        self.assertIn("200", paths["/api/pronosticos/active/series/{serie_id}"]["post"]["responses"])
        self.assertEqual(set(esquema["components"]["schemas"]["SolicitudPronostico"]["required"]),
                         {"tecnica", "fecha_corte"})
        for url, metodo in (("/health", "get"), ("/api/datasets/upload", "post"),
                            ("/api/datasets/active/series", "get"),
                            ("/api/datasets/active/series/{serie_id}/sales", "get")):
            self.assertIn(metodo, paths[url])
        self.assertEqual(self.cliente.get("/docs").status_code, 200)
        self.assertFalse(inspect.iscoroutinefunction(api.generar))

    def test_arranque_y_docs_no_ejecutan_operaciones(self):
        contenido = self.ruta.read_bytes()
        with patch.object(api, "registrar_pronostico", side_effect=AssertionError("generación en arranque")) as generar, \
                patch.object(registro, "_calcular", side_effect=AssertionError("cálculo en arranque")) as calcular, \
                patch.object(consulta, "_conexion_lectura", side_effect=AssertionError("lectura en arranque")) as leer:
            with TestClient(app) as cliente:
                self.assertEqual(cliente.get("/health").status_code, 200)
                self.assertEqual(cliente.get("/docs").status_code, 200)
                self.assertEqual(cliente.get("/openapi.json").status_code, 200)
            generar.assert_not_called()
            calcular.assert_not_called()
            leer.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_tesis_db_path_respetado_mediante_dependencia_existente(self):
        # El override sigue aislando la prueba; su ruta se resuelve mediante la
        # misma dependencia que usará la aplicación al recibir TESIS_DB_PATH.
        with patch.dict(os.environ, {"TESIS_DB_PATH": str(self.ruta)}):
            app.dependency_overrides[obtener_ruta_base] = lambda: obtener_ruta_base()
            self.assertEqual(self.cliente.get(f"{PREFIJO}/series/83").status_code, 200)


if __name__ == "__main__":
    unittest.main()
