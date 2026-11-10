"""Planificación I1-A: contrato puro, sin datos, modelos ni ejecución."""

import asyncio
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
import builtins
import io
import json
import multiprocessing
from pathlib import Path
import socket
import sqlite3
import subprocess
import threading
from types import ModuleType
import unittest
from unittest.mock import patch

import pandas as pd

from backend.app.services import configuracion_experimentos as contrato
from backend.app.services import plan_bateria_pronosticos as planificador
from backend.app.services.validacion_csv import MAX_ENTERO_SQLITE


CORTES = ("2017-06-16", "2017-07-01", "2017-07-16")
ORDEN = (contrato.BASE, "random_forest", "random_forest",
         "hist_gradient_boosting", "hist_gradient_boosting")


class PlanBateriaPronosticosTests(unittest.TestCase):
    def plan(self, **cambios):
        return planificador.construir_plan_bateria(
            **{"conjunto_id": 7, "serie_id": 83, "fecha_corte": CORTES[0], **cambios})

    def comprobar_error(self, codigo, **cambios):
        with self.assertRaises(planificador.ErrorPlanBateria) as error:
            self.plan(**cambios)
        self.assertEqual(error.exception.codigo, codigo)

    def test_orden_fijo_cinco_elementos_y_metadata_canonica(self):
        for corte in CORTES:
            with self.subTest(corte=corte):
                p = self.plan(fecha_corte=corte)
                self.assertEqual(set(p), {"conjunto_id", "serie_id", "fecha_corte", "elementos"})
                self.assertEqual((p["conjunto_id"], p["serie_id"], p["fecha_corte"]), (7, 83, corte))
                self.assertEqual(len(p["elementos"]), 5)
                self.assertEqual(tuple(e["tecnica"] for e in p["elementos"]), ORDEN)
                for i, e in enumerate(p["elementos"]):
                    self.assertEqual(set(e), {"tecnica", "configuracion", "variables", "version_contrato",
                                              "version_reglas", "horizonte", "identidad_predictiva"})
                    version = 2 if i in (2, 4) else 1
                    variables = ([] if i == 0 else list(contrato.VARIABLES_ALTERNATIVAS_V2)
                                 if version == 2 else list(contrato.VARIABLES))
                    self.assertEqual((e["version_contrato"], e["version_reglas"]), (version, version))
                    self.assertEqual(e["variables"], variables)
                    for campo in ("tecnica", "variables", "version_contrato", "version_reglas", "horizonte"):
                        self.assertEqual(e[campo], e["configuracion"][campo])

    def test_configuraciones_normalizadas_y_parametros_completos_del_contrato(self):
        for e in self.plan()["elementos"]:
            with self.subTest(tecnica=e["tecnica"], version=e["version_contrato"]):
                cfg = e["configuracion"]
                self.assertEqual(contrato.normalizar_configuracion(cfg), cfg)
                esperados = contrato.configuracion_predeterminada(e["tecnica"])["parametros"]
                self.assertEqual(json.dumps(cfg["parametros"], sort_keys=True, allow_nan=False),
                                 json.dumps(esperados, sort_keys=True, allow_nan=False))
                self.assertEqual(e["identidad_predictiva"], contrato.identidad_configuracion(cfg))
                self.assertRegex(e["identidad_predictiva"], r"^[0-9a-f]{64}$")

    def test_defaults_no_se_redefinen_en_el_planificador(self):
        original = contrato.configuracion_predeterminada
        def predeterminada(tecnica):
            cfg = original(tecnica)
            if tecnica == "random_forest":
                cfg["parametros"]["n_estimators"] += 1
            elif tecnica == "hist_gradient_boosting":
                cfg["parametros"]["max_iter"] += 1
            return cfg
        # Dependencia sustituida solo en este test: el servicio debe seguir F2.
        with patch.object(contrato, "configuracion_predeterminada", side_effect=predeterminada):
            for e in self.plan()["elementos"]:
                self.assertEqual(e["configuracion"]["parametros"], predeterminada(e["tecnica"])["parametros"])
        self.assertEqual(self.plan()["elementos"][1]["configuracion"]["parametros"], original("random_forest")["parametros"])

    def test_determinismo_fecha_objeto_y_json_serializable(self):
        primero = self.plan()
        for _ in range(3):
            self.assertEqual(self.plan(), primero)
            self.assertEqual(self.plan(fecha_corte=date(2017, 6, 16)), primero)
        self.assertEqual(json.loads(json.dumps(primero, allow_nan=False)), primero)

    def test_no_hay_duplicados_efectivos(self):
        elementos = self.plan()["elementos"]
        self.assertEqual(len({e["identidad_predictiva"] for e in elementos}), 5)
        self.assertEqual(len({contrato.json_canonico(e["configuracion"]) for e in elementos}), 5)
        for a, b in ((1, 2), (3, 4)):
            self.assertEqual(elementos[a]["configuracion"]["parametros"], elementos[b]["configuracion"]["parametros"])
            self.assertNotEqual(elementos[a]["identidad_predictiva"], elementos[b]["identidad_predictiva"])

    def test_identidad_predictiva_separada_de_contexto(self):
        referencia = self.plan()
        identidades = [e["identidad_predictiva"] for e in referencia["elementos"]]
        contextos = set()
        for conjunto, serie, corte in ((7, 83, CORTES[0]), (7, 84, CORTES[0]),
                                      (8, 83, CORTES[0]), (7, 83, CORTES[1]), (7, 83, CORTES[2])):
            p = self.plan(conjunto_id=conjunto, serie_id=serie, fecha_corte=corte)
            contextos.add((p["conjunto_id"], p["serie_id"], p["fecha_corte"]))
            self.assertEqual([e["identidad_predictiva"] for e in p["elementos"]], identidades)
            self.assertEqual(p["elementos"], referencia["elementos"])
        self.assertEqual(len(contextos), 5)

    def test_sin_referencias_mutables_compartidas(self):
        def referencias(objeto):
            if isinstance(objeto, (dict, list)):
                yield id(objeto)
                hijos = objeto.values() if isinstance(objeto, dict) else objeto
                for hijo in hijos:
                    yield from referencias(hijo)
        a, b = self.plan(), self.plan()
        ra, rb = list(referencias(a)), list(referencias(b))
        self.assertEqual(len(ra), len(set(ra)))  # También entre metadata y configuración.
        self.assertEqual(len(rb), len(set(rb)))
        self.assertTrue(set(ra).isdisjoint(rb))

    def test_mutar_un_resultado_no_altera_otros_elementos_llamadas_o_defaults(self):
        referencia = self.plan()
        defaults = {t: contrato.configuracion_predeterminada(t) for t in set(ORDEN)}
        for indice in range(5):
            with self.subTest(indice=indice):
                p = self.plan()
                e = p["elementos"][indice]
                e["configuracion"]["parametros"]["NO_ADMITIDO"] = [1, 2]
                e["configuracion"]["variables"].append("NO_ADMITIDA")
                self.assertEqual(e["variables"], referencia["elementos"][indice]["variables"])
                e["variables"].clear()
                e["horizonte"] = 1
                for otro in range(5):
                    if otro != indice:
                        self.assertEqual(p["elementos"][otro], referencia["elementos"][otro])
                self.assertEqual(self.plan(), referencia)
                self.assertEqual({t: contrato.configuracion_predeterminada(t) for t in set(ORDEN)}, defaults)

    def test_identificadores_invalidos_se_rechazan_antes_de_normalizar(self):
        class EnteroDerivado(int):
            pass
        with patch.object(contrato, "normalizar_configuracion") as normalizar:
            for campo, codigo in (("conjunto_id", "conjunto_invalido"), ("serie_id", "serie_invalida")):
                for valor in (0, -1, True, False, 1.0, "7", None, [], {}, MAX_ENTERO_SQLITE + 1, EnteroDerivado(7)):
                    with self.subTest(campo=campo, valor=valor):
                        self.comprobar_error(codigo, **{campo: valor})
            normalizar.assert_not_called()

    def test_limites_validos_no_simulan_pertenencia_al_activo(self):
        for identificador in (1, MAX_ENTERO_SQLITE):
            p = self.plan(conjunto_id=identificador, serie_id=identificador)
            self.assertEqual((p["conjunto_id"], p["serie_id"]), (identificador, identificador))
            self.assertEqual(len(p["elementos"]), 5)
            self.assertNotIn("activo", p)
            self.assertNotIn("pertenencia_verificada", p)

    def test_fechas_invalidas_se_rechazan_antes_de_normalizar(self):
        with patch.object(contrato, "normalizar_configuracion") as normalizar:
            for valor in ("", "20170616", "2017-6-16", "2017-02-29", "2017-13-01", " 2017-06-16",
                          "2017-06-16T00:00:00", b"2017-06-16", None, True, 20170616, [], {},
                          datetime(2017, 6, 16), datetime(2017, 6, 16, tzinfo=timezone.utc)):
                with self.subTest(valor=valor):
                    self.comprobar_error("corte_invalido", fecha_corte=valor)
            normalizar.assert_not_called()

    def test_cortes_no_autorizados(self):
        with patch.object(contrato, "normalizar_configuracion") as normalizar:
            for corte in ("2017-06-15", "2017-06-17", "2017-06-30", "2017-07-02", "2017-07-15",
                          "2017-08-16", "2017-08-31", "2016-06-16", "2018-06-16"):
                self.comprobar_error("corte_no_autorizado", fecha_corte=corte)
            normalizar.assert_not_called()

    def test_reserva_y_horizonte_de_quince_dias(self):
        for corte in CORTES:
            p = self.plan(fecha_corte=corte)
            for e in p["elementos"]:
                self.assertEqual(e["horizonte"], 15)
                self.assertLess(date.fromisoformat(corte) + timedelta(days=e["horizonte"]), date(2017, 8, 1))
        with patch.object(contrato, "normalizar_configuracion") as normalizar:
            inicio = date(2017, 7, 17)
            for i in range(30):
                self.comprobar_error("prueba_reservada", fecha_corte=inicio + timedelta(days=i))
            normalizar.assert_not_called()

    def test_no_admite_solicitudes_fuera_de_los_cinco_casos(self):
        for campo in ("tecnica", "tecnicas", "grupo", "variables", "parametros", "configuracion",
                      "version_contrato", "version_reglas", "horizonte", "elementos", "limite"):
            with self.subTest(campo=campo), self.assertRaises(TypeError):
                self.plan(**{campo: "NO_ADMITIDO"})
        with self.assertRaises(TypeError):
            planificador.construir_plan_bateria(7, 83, CORTES[0], [])

    def test_guardia_de_cantidad_y_duplicados_efectivos(self):
        casos = planificador._CASOS
        for alterados in (casos[:-1], casos + (casos[0],), (casos[0],) * 5):
            with patch.object(planificador, "_CASOS", alterados):
                self.comprobar_error("plan_invalido")
        self.assertEqual(len(self.plan()["elementos"]), 5)

    def test_importar_y_construir_no_hacen_e_s_entrenamiento_red_ni_servidores(self):
        codigo = compile(Path(planificador.__file__).read_text(encoding="utf-8"), planificador.__file__, "exec")
        aislado = ModuleType("planificador_puro_en_prueba")
        importar_real = builtins.__import__
        prohibidos = ("scripts.modelos_prediccion", "scripts.linea_base_prediccion",
                      "backend.app.services.registro_pronosticos", "backend.app.services.consulta_pronosticos",
                      "backend.app.storage", "backend.app.api", "sklearn", "fastapi", "uvicorn")

        def importar(nombre, *args, **kwargs):
            if any(nombre == p or nombre.startswith(p + ".") for p in prohibidos):
                raise AssertionError(f"Dependencia de ejecución indebida: {nombre}")
            return importar_real(nombre, *args, **kwargs)

        objetivos = [(builtins, "open"), (io, "open"), (Path, "open"), (pd, "read_csv"),
                     (sqlite3, "connect"), (socket, "socket"), (socket, "create_connection"),
                     (subprocess, "Popen"), (threading.Thread, "start"),
                     (multiprocessing.Process, "start"), (asyncio.BaseEventLoop, "create_server"),
                     (asyncio, "create_task")]
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(obj, nombre, side_effect=AssertionError(nombre)))
                      for obj, nombre in objetivos]
            stack.enter_context(patch.object(builtins, "__import__", side_effect=importar))
            exec(codigo, aislado.__dict__)
            for corte in CORTES:
                self.assertEqual(aislado.construir_plan_bateria(7, 83, corte), self.plan(fecha_corte=corte))
            for doble in dobles:
                doble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
