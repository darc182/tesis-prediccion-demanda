"""Auditoría H4: cruces entre disponibilidad, ajuste real y experimentos.

Solo SQLite temporal y TestClient, con TESIS_DB_PATH explícita. No se evalúa
precisión ni se exige igualdad entre técnicas o configuraciones diferentes.
Las pruebas individuales de fórmulas, reserva y transacciones siguen en sus
módulos originales; aquí se comprueban las conexiones entre esos contratos.
"""

from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import date, timedelta
import json
from math import fsum, isfinite
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import configuracion_experimentos as contrato
from backend.app.services import registro_pronosticos as registro
from backend.app.storage.db import conectar, inicializar_base
from scripts import modelos_prediccion as modelos
from scripts.variables_predictoras import construir_tabla_serie


CORTES = (date(2017, 6, 16), date(2017, 7, 1), date(2017, 7, 16))
PREFIJO = "/api/pronosticos/active"


def configuraciones(variantes=False):
    """Casos pequeños reproducibles; no cambian defaults productivos."""
    casos = [("base", {"tecnica": contrato.BASE})]
    for tecnica in modelos.TECNICAS:
        parametros = {"random_state": 23, "min_samples_leaf": 2}
        parametros.update({"n_estimators": 9} if tecnica == "random_forest"
                          else {"max_iter": 11, "learning_rate": 0.075})
        for version, variables in ((1, contrato.VARIABLES), (2, contrato.VARIABLES_ALTERNATIVAS_V2)):
            solicitud = {"tecnica": tecnica, "version_contrato": version,
                         "version_reglas": version, "variables": list(variables),
                         "parametros": dict(parametros)}
            casos.append((f"{tecnica}/{version}", solicitud))
            if variantes and version == 1:
                otra = deepcopy(solicitud)
                otra["parametros"]["max_depth"] = 3
                casos.append((f"{tecnica}/{version}/profundidad", otra))
    return casos


class IntegridadTemporalModeladoTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory(prefix="hu004_h4_")
        self.addCleanup(temporal.cleanup)
        self.ruta = (Path(temporal.name) / "auditoria_sintetica.sqlite3").resolve()
        self.assertEqual(self.ruta.parent, Path(temporal.name).resolve())
        inicializar_base(self.ruta)
        anteriores = app.dependency_overrides.copy()
        app.dependency_overrides.pop(obtener_ruta_base, None)
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        self.enterContext(patch.dict(os.environ, {"TESIS_DB_PATH": str(self.ruta)}))
        self.assertEqual(obtener_ruta_base().resolve(), self.ruta)
        self.cliente = self.enterContext(TestClient(app))
        inicio = date(2017, 4, 1)
        self.filas = [((inicio + timedelta(days=i)).isoformat(),
                       0.0 if i % 11 == 0 else 10.125 + (i % 7) / 4 + (i // 7) / 8, i % 3)
                      for i in range(122) if inicio + timedelta(days=i) != date(2017, 4, 15)]
        with closing(conectar(self.ruta)) as c, c:
            c.executemany(
                "INSERT INTO conjunto_datos (id,nombre_archivo,hash,fecha_carga,activo) VALUES (?,?,?,?,?)",
                [(2, "anterior_sintetico.csv", "h4-anterior", "2026-01-01", 0),
                 (7, "actual_sintetico.csv", "h4-actual", "2026-01-02", 1)])
            c.executemany(
                "INSERT INTO serie_temporal (id,conjunto_id,store_nbr,family,criterio_seleccion) VALUES (?,?,?,?,?)",
                [(24, 2, 17, "PREPARED FOODS", "sintética"),
                 (83, 7, 17, "PREPARED FOODS", "sintética"),
                 (84, 7, 8, "BREAD/BAKERY", "sintética")])
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                          [(serie, fecha, venta + desplazamiento, promo)
                           for serie, desplazamiento in ((24, 80.5), (83, 0), (84, 30.25))
                           for fecha, venta, promo in self.filas])

    def leer(self, sql, parametros=()):
        with closing(sqlite3.connect(f"{self.ruta.as_uri()}?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            return c.execute(sql, parametros).fetchall()

    def historial(self, corte, serie=83):
        with closing(sqlite3.connect(f"{self.ruta.as_uri()}?mode=ro", uri=True)) as c:
            c.execute("BEGIN")
            tienda, familia = c.execute(
                "SELECT store_nbr,family FROM serie_temporal WHERE id=?", (serie,)).fetchone()
            filas = c.execute("SELECT fecha,sales,onpromotion FROM venta_diaria "
                              "WHERE serie_id=? AND fecha<=? ORDER BY fecha",
                              (serie, corte.isoformat())).fetchall()
        datos = pd.DataFrame(filas, columns=["date", "sales", "onpromotion"])
        datos["store_nbr"], datos["family"] = tienda, familia
        return datos, tienda, familia

    def preparado(self, solicitud, corte, serie=83):
        datos, tienda, familia = self.historial(corte, serie)
        efectiva = contrato.normalizar_configuracion(solicitud)
        return construir_tabla_serie(datos, tienda, familia, corte,
                                     columnas_predictoras=efectiva["variables"])

    def disponibilidad(self, solicitud, corte, serie=83):
        grupo = "clasico" if solicitud["version_contrato"] == 1 else "ampliado"
        r = self.cliente.get(f"{PREFIJO}/series/{serie}/filas-entrenamiento",
                            params={"fecha_corte": corte.isoformat(), "grupo": grupo})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def post(self, solicitud, corte, serie=83):
        return self.cliente.post(f"{PREFIJO}/series/{serie}", json={
            "tecnica": solicitud["tecnica"], "fecha_corte": corte.isoformat(),
            "configuracion": solicitud})

    def resultado(self, respuesta, solicitud, corte, serie=83, conjunto=7):
        self.assertEqual(respuesta.status_code, 201, respuesta.text)
        registro_publico = respuesta.json()
        pid, eid = registro_publico["pronostico_id"], registro_publico["experimento_id"]
        detalle = self.cliente.get(f"{PREFIJO}/{pid}")
        config = self.cliente.get(f"{PREFIJO}/{pid}/configuracion")
        self.assertEqual((detalle.status_code, config.status_code), (200, 200))
        datos, configuracion = detalle.json(), config.json()
        efectiva = contrato.normalizar_configuracion(solicitud)
        self.assertEqual(configuracion["configuracion"], efectiva)
        self.assertEqual((datos["conjunto_id"], datos["serie"]["id"]), (conjunto, serie))
        self.assertEqual((configuracion["conjunto_id"], configuracion["serie"], configuracion["fecha_corte"]),
                         (conjunto, datos["serie"], corte.isoformat()))
        self.assertEqual((datos["pronostico"]["experimento_id"], datos["pronostico"]["tecnica"]),
                         (eid, solicitud["tecnica"]))
        self.assertEqual((datos["pronostico"]["fecha_corte"], datos["pronostico"]["horizonte"]),
                         (corte.isoformat(), 15))
        fechas = tuple(d["fecha_objetivo"] for d in datos["detalles"])
        self.assertEqual(fechas, tuple((corte + timedelta(days=i)).isoformat() for i in range(1, 16)))
        self.assertEqual([d["paso"] for d in datos["detalles"]], list(range(1, 16)))
        self.assertTrue(all(isfinite(d["venta_estimada"]) and d["venta_estimada"] >= 0 for d in datos["detalles"]))
        guardada = json.loads(self.leer("SELECT configuracion_json FROM experimento WHERE id=?", (eid,))[0][0])
        self.assertEqual(contrato.recuperar_configuracion(guardada), efectiva)
        return {"registro": registro_publico, "detalle": datos, "efectiva": efectiva,
                "guardada": guardada, "fechas": fechas,
                "identidad": contrato.identidad_configuracion(efectiva),
                "contexto": (conjunto, serie, corte.isoformat(), 15, fechas)}

    @contextmanager
    def observar_modelo(self, solicitud, corte, serie=83):
        """Espías que delegan fit/predict reales; no fabrican estimaciones."""
        efectiva = contrato.normalizar_configuracion(solicitud)
        preparada = self.preparado(solicitud, corte, serie)
        columnas = tuple(efectiva["variables"])
        clase = (modelos.RandomForestRegressor if efectiva["tecnica"] == "random_forest"
                 else modelos.HistGradientBoostingRegressor)
        fit_real, predict_real, entrenar_real = clase.fit, clase.predict, modelos.entrenar_modelo
        observado = {"preparada": preparada, "fit": 0, "xs": [], "modelo": None}

        def fit(estimador, x, y):
            pd.testing.assert_frame_equal(x, preparada.tabla.loc[:, list(columnas)])
            pd.testing.assert_series_equal(y, preparada.tabla["sales"])
            observado["fit"] += 1
            return fit_real(estimador, x, y)

        def predict(estimador, x):
            self.assertEqual(tuple(x.columns), columnas)
            self.assertEqual(tuple(estimador.feature_names_in_), columnas)
            self.assertEqual(estimador.n_features_in_, len(columnas))
            observado["xs"].append(x.copy(deep=True))
            return predict_real(estimador, x)

        def entrenar(datos, tienda, familia, tecnica, fecha_corte, **kwargs):
            self.assertTrue(datos["date"].le(pd.Timestamp(corte)).all())
            self.assertEqual((tienda, familia, tecnica, fecha_corte),
                             (preparada.store_nbr, preparada.family, efectiva["tecnica"], corte))
            modelo = entrenar_real(datos, tienda, familia, tecnica, fecha_corte, **kwargs)
            self.assertEqual(json.dumps(modelo.estimador.get_params(deep=False), sort_keys=True),
                             json.dumps(efectiva["parametros"], sort_keys=True))
            observado["modelo"] = modelo
            return modelo

        with patch.object(clase, "fit", new=fit), patch.object(clase, "predict", new=predict), \
                patch.object(modelos, "entrenar_modelo", side_effect=entrenar):
            yield observado

    def comprobar_filas(self, disponibilidad, observado):
        p, m = observado["preparada"], observado["modelo"]
        self.assertEqual(observado["fit"], 1)
        self.assertIsNotNone(m)
        self.assertEqual((disponibilidad["filas_aptas"], disponibilidad["filas_descartadas"]),
                         (len(p.tabla), len(p.descartes)))
        self.assertEqual((m.filas_validas, m.filas_descartadas),
                         (disponibilidad["filas_aptas"], disponibilidad["filas_descartadas"]))
        self.assertEqual(disponibilidad["observaciones_historicas"], p.observaciones_historicas)
        self.assertEqual(m.filas_validas + m.filas_descartadas, p.observaciones_historicas)
        self.assertEqual(disponibilidad["descartes_por_variable"], p.descartes_por_variable)
        self.assertEqual(disponibilidad["variables"], list(m.columnas_predictoras))

    def test_cohortes_tres_cortes_unen_get_preparacion_fit_recursion_y_registro(self):
        self.assertEqual(registro.CORTES_VALIDACION, CORTES)
        cohortes = []
        for corte in CORTES:
            resultados = []
            for nombre, solicitud in configuraciones(variantes=True):
                with self.subTest(corte=corte, caso=nombre):
                    if solicitud["tecnica"] == contrato.BASE:
                        r = self.resultado(self.post(solicitud, corte), solicitud, corte)
                        valores = [d["venta_estimada"] for d in r["detalle"]["detalles"]]
                        self.assertEqual(valores[7:], valores[:7] + valores[:1])
                    else:
                        conteos = self.disponibilidad(solicitud, corte)
                        with self.observar_modelo(solicitud, corte) as obs:
                            r = self.resultado(self.post(solicitud, corte), solicitud, corte)
                        self.comprobar_filas(conteos, obs)
                        self.assertEqual(len(obs["xs"]), 15)
                        entrenamiento = r["guardada"]["entrenamiento"]
                        self.assertEqual((entrenamiento["filas_validas"], entrenamiento["filas_descartadas"]),
                                         (conteos["filas_aptas"], conteos["filas_descartadas"]))
                        self.assertEqual(entrenamiento["ultima_respuesta_real"],
                                         obs["modelo"].ultima_fecha_entrenamiento.isoformat())
                        self.assertTrue(obs["preparada"].tabla["sales"].eq(0).any())
                        self.assertTrue(obs["preparada"].tabla["sales"].mod(1).ne(0).any())
                        self.assertNotIn(date(2017, 4, 15), set(obs["preparada"].tabla["fecha_objetivo"]))
                        valores = [d["venta_estimada"] for d in r["detalle"]["detalles"]]
                        for paso, x in enumerate(obs["xs"], 1):
                            if paso > 1:
                                self.assertEqual(x.iloc[0]["lag_1"], valores[paso - 2])
                            if paso > 7:
                                self.assertEqual(x.iloc[0]["lag_7"], valores[paso - 8])
                        self.assertEqual(obs["xs"][14].iloc[0]["lag_14"], valores[0])
                        if solicitud["version_contrato"] == 2:
                            self.assertAlmostEqual(obs["xs"][14].iloc[0]["media_14"],
                                                   fsum(v / 14 for v in valores[:14]))
                    resultados.append(r)
            # Mismas condiciones observacionales y temporales, diferentes configuraciones.
            esperado = (7, 83, corte.isoformat(), 15,
                        tuple((corte + timedelta(days=i)).isoformat() for i in range(1, 16)))
            self.assertEqual({r["contexto"] for r in resultados}, {esperado})
            self.assertEqual(len({r["identidad"] for r in resultados}), 7)
            self.assertEqual(len({r["guardada"]["historial"]["sha256_observaciones"] for r in resultados}), 1)
            cohortes.append(resultados)
        # Igual configuración entre cortes no vuelve comparables sus horizontes.
        self.assertEqual(len({r["identidad"] for c in cohortes for r in c}), 7)
        self.assertEqual(len({r["contexto"] for c in cohortes for r in c}), 3)
        self.assertEqual(len({r["registro"]["experimento_id"] for c in cohortes for r in c}), 21)
        self.assertEqual(self.leer("SELECT COUNT(*) FROM detalle_pronostico"), [(315,)])

    def test_futuro_perturbado_y_solicitud_equivalente_reproducen_cada_cohorte(self):
        for corte in CORTES:
            with self.subTest(corte=corte):
                with closing(conectar(self.ruta)) as c, c:
                    c.executemany("UPDATE venta_diaria SET sales=?,onpromotion=? WHERE serie_id=83 AND fecha=?",
                                  [(venta, promo, fecha) for fecha, venta, promo in self.filas])
                modelos_observados = []
                entrenar_real, calcular_real = modelos.entrenar_modelo, registro._calcular

                def entrenar(*args, **kwargs):
                    m = entrenar_real(*args, **kwargs)
                    modelos_observados.append(m)
                    return m

                def calcular(instantanea, tecnica, fecha_corte, **kwargs):
                    self.assertTrue(all(f[0] <= corte.isoformat() for f in instantanea.filas))
                    self.assertTrue(all(isinstance(f[1], (int, float)) for f in instantanea.filas))
                    return calcular_real(instantanea, tecnica, fecha_corte, **kwargs)

                with patch.object(modelos, "entrenar_modelo", side_effect=entrenar), \
                        patch.object(registro, "_calcular", side_effect=calcular) as calcular_observado:
                    previos = [(s, self.resultado(self.post(s, corte), s, corte)) for _, s in configuraciones()]
                    with closing(conectar(self.ruta)) as c, c:
                        c.execute("UPDATE venta_diaria SET sales='NO_USAR_FUTURO',onpromotion=999 "
                                  "WHERE serie_id=83 AND fecha>?", (corte.isoformat(),))
                    contenido = self.ruta.read_bytes()
                    for solicitud, previo in previos:
                        completa = contrato.normalizar_configuracion(solicitud)
                        repetida = self.post(completa, corte)
                        self.assertEqual(repetida.status_code, 200, repetida.text)
                        self.assertEqual((repetida.json()["experimento_id"], repetida.json()["pronostico_id"]),
                                         (previo["registro"]["experimento_id"], previo["registro"]["pronostico_id"]))
                        detalle = self.cliente.get(f"{PREFIJO}/{repetida.json()['pronostico_id']}")
                        self.assertEqual(detalle.json(), previo["detalle"])
                        self.assertEqual(contrato.identidad_configuracion(completa), previo["identidad"])
                    self.assertEqual(self.ruta.read_bytes(), contenido)
                self.assertEqual(calcular_observado.call_count, 10)
                self.assertEqual(len(modelos_observados), 8)  # Cuatro regresores, dos ajustes reales por caso.
                self.assertEqual(len({id(m.estimador) for m in modelos_observados}), 8)

    def test_filas_aptas_no_garantizan_inferencia_y_hueco_no_equivale_a_cero(self):
        corte = CORTES[0]
        ausente = (corte - timedelta(days=10)).isoformat()
        with closing(conectar(self.ruta)) as c, c:
            c.execute("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha=?", (ausente,))
        base = self.resultado(self.post({"tecnica": contrato.BASE}, corte), {"tecnica": contrato.BASE}, corte)
        self.assertEqual(len(base["detalle"]["detalles"]), 15)
        anteriores = {}
        for nombre, solicitud in configuraciones()[1:]:
            with self.subTest(caso=nombre, observacion="ausente"):
                conteos = self.disponibilidad(solicitud, corte)
                anteriores[nombre] = conteos
                self.assertGreater(conteos["filas_aptas"], 0)
                contenido = self.ruta.read_bytes()
                with self.observar_modelo(solicitud, corte) as obs:
                    r = self.post(solicitud, corte)
                self.assertEqual(r.status_code, 422, r.text)
                self.assertEqual(r.json()["detail"]["codigo"], "error_calculo")
                self.comprobar_filas(conteos, obs)
                self.assertEqual(len(obs["xs"]), 3 if solicitud["version_contrato"] == 1 else 0)
                self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.leer("SELECT COUNT(*) FROM experimento"), [(1,)])
        with closing(conectar(self.ruta)) as c, c:
            c.execute("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (83,?,0,0)", (ausente,))
        for nombre, solicitud in configuraciones()[1:]:
            with self.subTest(caso=nombre, observacion="cero real"):
                conteos = self.disponibilidad(solicitud, corte)
                self.assertEqual(conteos["observaciones_historicas"], anteriores[nombre]["observaciones_historicas"] + 1)
                with self.observar_modelo(solicitud, corte) as obs:
                    self.resultado(self.post(solicitud, corte), solicitud, corte)
                self.comprobar_filas(conteos, obs)
                fila = obs["preparada"].tabla.loc[obs["preparada"].tabla["fecha_objetivo"] == date.fromisoformat(ausente)]
                self.assertEqual(fila["sales"].tolist(), [0.0])
                self.assertEqual(len(obs["xs"]), 15)

    def test_identidad_predictiva_igual_no_confunde_series_ni_conjuntos(self):
        solicitud = next(s for _, s in configuraciones() if s["tecnica"] == "random_forest" and s["version_contrato"] == 2)
        corte = CORTES[0]
        resultados, estimadores = [], []
        for serie, conjunto in ((83, 7), (84, 7), (24, 2)):
            if conjunto == 2:
                with closing(conectar(self.ruta)) as c, c:
                    c.execute("UPDATE conjunto_datos SET activo=(id=2)")
            with self.subTest(serie=serie, conjunto=conjunto):
                conteos = self.disponibilidad(solicitud, corte, serie)
                with self.observar_modelo(solicitud, corte, serie) as obs:
                    r = self.resultado(self.post(solicitud, corte, serie), solicitud, corte, serie, conjunto)
                self.comprobar_filas(conteos, obs)
                resultados.append(r)
                estimadores.append(obs["modelo"].estimador)
                self.assertEqual(r["guardada"]["historial"]["conjunto_id"], conjunto)
        self.assertEqual(len({r["identidad"] for r in resultados}), 1)
        self.assertEqual(len({r["contexto"] for r in resultados}), 3)
        self.assertEqual(len({r["guardada"]["historial"]["sha256_observaciones"] for r in resultados}), 3)
        self.assertEqual(len({id(e) for e in estimadores}), 3)
        self.assertEqual(len({r["registro"]["experimento_id"] for r in resultados}), 3)
        for previo in resultados[:2]:
            r = self.cliente.get(f"{PREFIJO}/{previo['registro']['pronostico_id']}")
            self.assertEqual(r.status_code, 404)
        self.assertEqual(self.leer("SELECT COUNT(*) FROM detalle_pronostico"), [(45,)])


if __name__ == "__main__":
    unittest.main()
