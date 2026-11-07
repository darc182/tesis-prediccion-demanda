"""Auditoría F5: CSV sintético -> HTTP -> motor real -> SQLite temporal.

No se importa ninguna fixture que abra datos del usuario. Los registros v1 se
construyen aquí como legado SINTÉTICO, convirtiendo resultados actuales reales.
Ejecutar: .venv/Scripts/python -m unittest tests.test_integracion_motor_experimentos -v
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import csv
from datetime import date, timedelta
import io
import json
from math import isfinite
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier, get_ident
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import pronosticos as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import configuracion_experimentos as contrato
from backend.app.services import registro_pronosticos as registro
from backend.app.storage.db import conectar, inicializar_base
from scripts import modelos_prediccion as modelos
from scripts.variables_predictoras import construir_tabla_serie


RAIZ = Path(__file__).resolve().parents[1]
PREFIJO = "/api/pronosticos/active"
CORTE = "2017-06-16"
TABLAS = ("conjunto_datos", "serie_temporal", "venta_diaria", "experimento",
          "pronostico", "detalle_pronostico", "evaluacion", "simulacion")


def csv_sintetico(series):
    """Historial reproducible, con fracciones, cero real y un día ausente.

    Termina el 16/07: no fabrica observaciones de la prueba final de agosto.
    La serie adicional existe solo en el CSV y debe ser descartada al cargar.
    """
    salida = io.StringIO(newline="")
    escritor = csv.writer(salida)
    escritor.writerow(("date", "store_nbr", "family", "sales", "onpromotion"))
    inicio, fin = date(2016, 12, 1), date(2017, 7, 16)
    dias = 0
    for i in range((fin - inicio).days + 1):
        fecha = inicio + timedelta(days=i)
        if fecha == date(2016, 12, 25):
            continue
        dias += 1
        for indice, serie in enumerate(series):
            valor = 0.0 if i % 19 == 0 else 10.25 + indice + i % 7 + (i % 3) / 4
            escritor.writerow((fecha.isoformat(), serie["store_nbr"], serie["family"], valor, i % 4))
        escritor.writerow((fecha.isoformat(), 999, "NO HABILITADA F5", 9999, 0))
    return salida.getvalue().encode("utf-8"), dias


class IntegracionMotorExperimentosTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory(prefix="hu004_f5_")
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "motor.sqlite3"
        self.assertFalse(self.ruta.exists())
        inicializar_base(self.ruta)
        self.series = json.loads((RAIZ / "config/series_seleccionadas.json").read_text(
            encoding="utf-8"))["series"]
        self.csv, self.dias = csv_sintetico(self.series)
        anteriores = app.dependency_overrides.copy()
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        self.cliente = self.enterContext(TestClient(app))
        self.assertEqual(self.conteos(), (0,) * len(TABLAS))
        carga = self.subir()
        self.assertEqual(carga.status_code, 200, carga.text)
        self.carga = carga.json()
        self.conjunto = self.carga["conjunto_id"]
        activas = self.cliente.get("/api/datasets/active/series").json()
        self.serie = next(s for s in activas["series"] if
                          (s["store_nbr"], s["family"]) == (17, "PREPARED FOODS"))
        self.url = f"{PREFIJO}/series/{self.serie['id']}"

    def sql(self, texto, parametros=()):
        with closing(conectar(self.ruta)) as conexion, conexion:
            return conexion.execute(texto, parametros).fetchall()

    def estado(self, tablas=TABLAS):
        with closing(conectar(self.ruta)) as conexion:
            return tuple(tuple(conexion.execute(f"SELECT * FROM {tabla} ORDER BY 1, 2").fetchall())
                         for tabla in tablas)

    def conteos(self):
        return tuple(self.sql(f"SELECT COUNT(*) FROM {tabla}")[0][0] for tabla in TABLAS)

    def subir(self):
        return self.cliente.post("/api/datasets/upload", files={"file": ("sintetico_f5.csv", self.csv, "text/csv")})

    def post(self, tecnica=contrato.BASE, parametros=None, corte=CORTE, configuracion=None):
        solicitud = {"tecnica": tecnica, "fecha_corte": corte}
        if parametros is not None:
            solicitud["configuracion"] = {"tecnica": tecnica, "parametros": parametros}
        if configuracion is not None:
            solicitud["configuracion"] = configuracion
        return self.cliente.post(self.url, json=solicitud)

    def crear(self, *args, **kwargs):
        respuesta = self.post(*args, **kwargs)
        self.assertEqual(respuesta.status_code, 201, respuesta.text)
        return respuesta.json()

    def config_guardada(self, experimento):
        return json.loads(self.sql("SELECT configuracion_json FROM experimento WHERE id=?", (experimento,))[0][0])

    def actualizar_config(self, experimento, texto, *, permitir_invalido=False):
        with closing(conectar(self.ruta)) as conexion, conexion:
            if permitir_invalido:
                # Solo fixture SQLite temporal: permite sembrar NaN/Infinity o
                # JSON malformado que json_valid del esquema normalmente impide.
                conexion.execute("PRAGMA ignore_check_constraints=ON")
            conexion.execute("UPDATE experimento SET configuracion_json=? WHERE id=?", (texto, experimento))

    def error(self, respuesta, estado, codigo):
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertEqual(respuesta.json(), {"detail": {"codigo": codigo, "mensaje": api.ERRORES[codigo][1]}})
        self.assertNotIn(str(self.ruta), respuesta.text)

    def test_carga_desde_cero_diez_series_historico_sin_rellenar_y_arranque_inerte(self):
        self.assertEqual((self.carga["series_importadas"], self.carga["observaciones_importadas"]),
                         (10, 10 * self.dias))
        self.assertEqual(self.carga["errores"], [])
        self.assertEqual(self.carga["advertencias"], [])
        self.assertEqual(self.conteos(), (1, 10, 10 * self.dias, 0, 0, 0, 0, 0))
        self.assertEqual(set(self.sql("SELECT store_nbr,family FROM serie_temporal")),
                         {(s["store_nbr"], s["family"]) for s in self.series})
        self.assertEqual(self.sql("SELECT COUNT(*) FROM venta_diaria WHERE fecha>='2017-08-01'"), [(0,)])
        anterior = self.estado()
        respuesta = self.cliente.get(f"/api/datasets/active/series/{self.serie['id']}/sales",
                                     params={"desde": "2016-12-24", "hasta": "2016-12-26"})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual([o["fecha"] for o in respuesta.json()["observaciones"]], ["2016-12-24", "2016-12-26"])
        self.assertTrue(all(o["sales"] > 0 for o in respuesta.json()["observaciones"]))
        with patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("arranque entrena")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("arranque consulta")):
            with TestClient(app) as cliente:
                self.assertEqual(cliente.get("/health").json(), {"status": "ok"})
                self.assertEqual(cliente.get("/docs").status_code, 200)
                rutas = cliente.get("/openapi.json").json()["paths"]
                self.assertIn("/api/datasets/upload", rutas)
                self.assertIn("/api/pronosticos/active/{pronostico_id}/configuracion", rutas)
        self.assertEqual(self.estado(), anterior)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_cinco_experimentos_entrenamiento_real_consulta_y_reutilizacion(self):
        casos = [(contrato.BASE, None), ("random_forest", None),
                 ("random_forest", {"n_estimators": 200, "max_depth": 4, "random_state": 7}),
                 ("hist_gradient_boosting", None),
                 ("hist_gradient_boosting", {"max_iter": 150, "learning_rate": 0.1, "random_state": 7})]
        entrenar = modelos.entrenar_modelo
        observados = []

        def observar(*args, **kwargs):
            # Observación de fit REAL: además prueba que no hay bloqueo durante entrenamiento.
            with closing(conectar(self.ruta)) as c:
                c.execute("BEGIN IMMEDIATE")
                c.rollback()
            self.assertTrue((args[0]["date"].dt.date <= date.fromisoformat(CORTE)).all())
            modelo = entrenar(*args, **kwargs)
            self.assertEqual(tuple(modelo.estimador.feature_names_in_), contrato.VARIABLES)
            observados.append(modelo)
            return modelo

        historico = self.estado(TABLAS[:3])
        with patch.object(modelos, "entrenar_modelo", side_effect=observar):
            resultados = [self.crear(t, p) for t, p in casos]
        self.assertEqual(len(observados), 4)
        self.assertEqual(len({r["experimento_id"] for r in resultados}), 5)
        self.assertEqual(self.conteos()[3:6], (5, 5, 75))
        for indice, ((tecnica, parametros), resultado) in enumerate(zip(casos, resultados)):
            with self.subTest(tecnica=tecnica, parametros=parametros):
                efectiva = contrato.normalizar_configuracion({"tecnica": tecnica, "parametros": parametros or {}})
                pid, eid = resultado["pronostico_id"], resultado["experimento_id"]
                detalle = self.cliente.get(f"{PREFIJO}/{pid}")
                configuracion = self.cliente.get(f"{PREFIJO}/{pid}/configuracion")
                self.assertEqual((detalle.status_code, configuracion.status_code), (200, 200))
                datos = configuracion.json()
                self.assertEqual((datos["conjunto_id"], datos["serie"], datos["pronostico_id"], datos["experimento_id"]),
                                 (self.conjunto, self.serie, pid, eid))
                self.assertEqual(datos["configuracion"], efectiva)
                self.assertEqual(datos["es_predeterminada"], parametros is None)
                self.assertEqual(contrato.recuperar_configuracion(self.config_guardada(eid)), efectiva)
                if indice:
                    modelo = observados[indice - 1]
                    self.assertEqual(json.dumps(modelo.estimador.get_params(), sort_keys=True),
                                     json.dumps(efectiva["parametros"], sort_keys=True))
                    if tecnica == "random_forest":
                        self.assertEqual(len(modelo.estimador.estimators_), efectiva["parametros"]["n_estimators"])
                    else:
                        self.assertEqual(modelo.estimador.n_iter_, efectiva["parametros"]["max_iter"])
                detalles = detalle.json()["detalles"]
                self.assertEqual(len(detalles), 15)
                self.assertEqual([d["fecha_objetivo"] for d in detalles],
                                 [(date.fromisoformat(CORTE) + timedelta(days=i)).isoformat() for i in range(1, 16)])
                self.assertEqual([d["paso"] for d in detalles], list(range(1, 16)))
                self.assertTrue(all(isfinite(d["venta_estimada"]) and d["venta_estimada"] >= 0 for d in detalles))
                self.assertTrue(any(d["venta_estimada"] % 1 for d in detalles))
                anterior = self.estado()
                repetida = self.post(tecnica, parametros)
                self.assertEqual(repetida.status_code, 200, repetida.text)
                self.assertEqual((repetida.json()["experimento_id"], repetida.json()["pronostico_id"]), (eid, pid))
                self.assertEqual(self.estado(), anterior)
                if parametros is None:
                    self.assertEqual(self.post(tecnica, configuracion=efectiva).status_code, 200)
        self.assertEqual(self.estado(TABLAS[:3]), historico)
        self.assertEqual(len(self.cliente.get(self.url).json()["pronosticos"]), 5)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_gets_en_wal_solo_lectura_sin_entrenamiento_ni_dml(self):
        resultado = self.crear()
        self.sql("PRAGMA journal_mode=WAL")
        anterior = self.estado()
        sentencias, rutas = [], []
        abrir = sqlite3.connect

        def observar(ruta, *args, **kwargs):
            rutas.append(str(ruta))
            conexion = abrir(ruta, *args, **kwargs)
            conexion.set_trace_callback(sentencias.append)
            return conexion

        with patch.object(sqlite3, "connect", side_effect=observar), \
                patch.object(api, "registrar_pronostico", side_effect=AssertionError("GET registra")), \
                patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("GET entrena")):
            for url in ("/api/datasets/active/series", f"/api/datasets/active/series/{self.serie['id']}/sales",
                        self.url, f"{PREFIJO}/{resultado['pronostico_id']}",
                        f"{PREFIJO}/{resultado['pronostico_id']}/configuracion"):
                respuesta = self.cliente.get(url)
                self.assertEqual(respuesta.status_code, 200, respuesta.text)
                for privado in ("sha256_observaciones", "hash_conjunto", "ruta_modelo", "historial", str(self.ruta)):
                    self.assertNotIn(privado, respuesta.text)
        self.assertEqual(len(rutas), 5)
        self.assertTrue(all("mode=ro" in ruta for ruta in rutas))
        self.assertTrue(sentencias)
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE"))
                             for s in sentencias))
        self.assertEqual(self.estado(), anterior)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_legado_sintetico_v1_v2_auditoria_y_json_sin_migracion(self):
        resultado = self.crear()
        eid, pid = resultado["experimento_id"], resultado["pronostico_id"]
        actual = self.config_guardada(eid)
        heredada = deepcopy(actual)
        heredada["version_registro"] = 1
        heredada.pop("configuracion_predictiva")
        abrir = sqlite3.connect
        sentencias = []

        def observar(*args, **kwargs):
            conexion = abrir(*args, **kwargs)
            conexion.set_trace_callback(sentencias.append)
            return conexion

        def sin_dml():
            self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
                                 for s in sentencias), sentencias)

        for config in (heredada, actual):
            config = deepcopy(config)
            config["versiones"] = {"python": "AUDITORIA SINTETICA"}
            config["auditoria"]["duracion_sintetica"] = 123.5
            texto = json.dumps(config, ensure_ascii=False, indent=2)
            self.actualizar_config(eid, texto)
            anterior = self.estado()
            self.assertEqual(self.cliente.get(f"{PREFIJO}/{pid}").status_code, 200)
            recuperada = self.cliente.get(f"{PREFIJO}/{pid}/configuracion")
            self.assertEqual(recuperada.status_code, 200, recuperada.text)
            self.assertTrue(recuperada.json()["es_predeterminada"])
            sentencias.clear()
            with patch.object(sqlite3, "connect", side_effect=observar):
                repetida = self.post()
            sin_dml()
            self.assertEqual(repetida.status_code, 200, repetida.text)
            self.assertEqual((repetida.json()["experimento_id"], repetida.json()["pronostico_id"]), (eid, pid))
            self.assertEqual(self.estado(), anterior)
            self.assertEqual(self.sql("SELECT configuracion_json FROM experimento WHERE id=?", (eid,)), [(texto,)])
        incompleta = deepcopy(heredada)
        incompleta["parametros"] = {}
        desconocida = deepcopy(actual)
        desconocida["version_registro"] = 999
        # Dos valores contradictorios: un lector que elija silenciosamente el
        # último vería el default 7 e intentaría reutilizar el experimento.
        textos = {"legado_incompleto": json.dumps(incompleta), "version_desconocida": json.dumps(desconocida)}
        original = json.dumps(actual)
        for periodo in (8, 7):
            textos[f"parametros_repetidos_{periodo}"] = (
                f'{{"parametros": {{"periodo_estacional_dias": {periodo}}}, ' + original[1:])
        textos["parametro_interno_repetido"] = original.replace(
            '"periodo_estacional_dias": 7', '"periodo_estacional_dias": 7, "periodo_estacional_dias": 7')
        for campo in ("auditoria", "historial"):
            fragmento = json.dumps(actual[campo])
            textos[f"duplicado_en_{campo}"] = original.replace(
                f'"{campo}": {fragmento}', f'"{campo}": {{"duplicada": 1, "duplicada": 1, ' + fragmento[1:])
        textos["duplicado_en_objeto_dentro_de_lista"] = original.replace(
            '"auditoria": {', '"auditoria": {"lista": [{"x": 1, "x": 2}], ')
        for numero in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            textos[f"auditoria_{numero}"] = original.replace(
                '"auditoria": {', f'"auditoria": {{"lista": [{{"valor": {numero}}}], ')
        textos["json_malformado"] = original[:-1]
        for nombre, texto in textos.items():
            with self.subTest(configuracion_corrupta=nombre):
                self.actualizar_config(eid, texto, permitir_invalido=True)
                anterior = self.estado()
                sentencias.clear()
                with patch.object(sqlite3, "connect", side_effect=observar):
                    self.error(self.cliente.get(f"{PREFIJO}/{pid}/configuracion"), 409, "pronostico_corrupto")
                    self.error(self.post(), 409, "conflicto")
                sin_dml()
                self.assertEqual(self.estado(), anterior)
                self.assertEqual(self.sql("SELECT configuracion_json FROM experimento WHERE id=?", (eid,)), [(texto,)])

    def test_conflictos_incompletos_ambiguos_y_cambio_historico_sin_sobrescritura(self):
        resultado = self.crear()
        pid = resultado["pronostico_id"]
        self.sql("UPDATE detalle_pronostico SET venta_estimada=venta_estimada+1 WHERE pronostico_id=? AND paso=1", (pid,))
        anterior = self.estado()
        self.error(self.post(), 409, "conflicto")
        self.assertEqual(self.estado(), anterior)
        self.sql("DELETE FROM detalle_pronostico WHERE pronostico_id=? AND paso=15", (pid,))
        anterior = self.estado()
        self.error(self.post(), 409, "conflicto")
        self.error(self.cliente.get(f"{PREFIJO}/{pid}"), 409, "pronostico_corrupto")
        self.assertEqual(self.estado(), anterior)
        # Duplicado deliberado de la misma identidad, solo en la base temporal.
        self.sql("INSERT INTO pronostico (experimento_id,fecha_corte,tipo,horizonte,fecha_generacion,duracion_segundos) "
                 "SELECT experimento_id,fecha_corte,tipo,horizonte,fecha_generacion,duracion_segundos FROM pronostico WHERE id=?", (pid,))
        anterior = self.estado()
        self.error(self.post(), 409, "conflicto")
        self.assertEqual(self.estado(), anterior)
        # Otro corte completo permite aislar la comprobación de huella histórica.
        self.crear(corte="2017-07-01")
        self.sql("UPDATE venta_diaria SET sales=sales+0.25 WHERE serie_id=? AND fecha='2017-06-01'", (self.serie["id"],))
        anterior = self.estado()
        self.error(self.post(corte="2017-07-01"), 409, "conflicto")
        self.assertEqual(self.estado(), anterior)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_concurrencia_http_identica_busca_dentro_de_begin_immediate(self):
        calcular, buscar, abrir = registro._calcular, registro._buscar_identico, registro._abrir_base
        barrera = Barrier(2)
        eventos = {}

        def sincronizar(*args, **kwargs):
            resultado = calcular(*args, **kwargs)  # cálculo real de línea base
            barrera.wait(timeout=30)
            return resultado

        def observar(ruta, escritura=False):
            conexion = abrir(ruta, escritura)
            if escritura:
                conexion.set_trace_callback(lambda s: eventos.setdefault(get_ident(), []).append(s))
            return conexion

        def comprobar_busqueda(conexion, *args):
            self.assertTrue(conexion.in_transaction)
            self.assertIn("BEGIN IMMEDIATE", eventos[get_ident()])
            eventos[get_ident()].append("BUSQUEDA_DUPLICADOS")
            return buscar(conexion, *args)

        def solicitar():
            with TestClient(app) as cliente:
                return cliente.post(self.url, json={"tecnica": contrato.BASE, "fecha_corte": CORTE})

        with patch.object(registro, "_calcular", side_effect=sincronizar), \
                patch.object(registro, "_abrir_base", side_effect=observar), \
                patch.object(registro, "_buscar_identico", side_effect=comprobar_busqueda):
            with ThreadPoolExecutor(max_workers=2) as ejecutor:
                futuros = [ejecutor.submit(solicitar) for _ in range(2)]
                respuestas = [f.result(timeout=45) for f in futuros]
        self.assertEqual(sorted(r.status_code for r in respuestas), [200, 201])
        self.assertEqual(len({(r.json()["experimento_id"], r.json()["pronostico_id"]) for r in respuestas}), 1)
        self.assertEqual(self.conteos()[3:6], (1, 1, 15))
        self.assertEqual(len(eventos), 2)
        for traza in eventos.values():
            self.assertLess(traza.index("BEGIN IMMEDIATE"), traza.index("BUSQUEDA_DUPLICADOS"))
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_fallo_detalle_revierte_alternativa_real_y_preserva_experimento_previo(self):
        previo = self.crear()
        self.sql("CREATE TRIGGER fallo_f5 BEFORE INSERT ON detalle_pronostico "
                 "WHEN NEW.paso=8 BEGIN SELECT RAISE(ABORT, 'FALLO PRIVADO F5'); END")
        anterior = self.estado()
        respuesta = self.post("random_forest", {"n_estimators": 20})
        self.error(respuesta, 503, "error_persistencia")
        self.assertNotIn("FALLO PRIVADO", respuesta.text)
        self.assertEqual(self.estado(), anterior)
        self.assertEqual(self.cliente.get(f"{PREFIJO}/{previo['pronostico_id']}").status_code, 200)
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])

    def test_revalidacion_durante_calculo_rechaza_historico_y_nuevo_activo(self):
        calcular = registro._calcular
        for cambio in ("historico", "activo"):
            with self.subTest(cambio=cambio):
                esperado = []

                def intercalar(*args, **kwargs):
                    resultado = calcular(*args, **kwargs)
                    if cambio == "historico":
                        self.sql("UPDATE venta_diaria SET onpromotion=onpromotion+1 WHERE serie_id=? AND fecha=?",
                                 (self.serie["id"], CORTE))
                    else:
                        self.assertEqual(self.subir().status_code, 200)
                    esperado.append(self.estado())
                    return resultado

                with patch.object(registro, "_calcular", side_effect=intercalar):
                    respuesta = self.post()
                estado, codigo = (409, "historial_cambiado") if cambio == "historico" else (404, "serie_inactiva")
                self.error(respuesta, estado, codigo)
                self.assertEqual(self.estado(), esperado[0])
                self.assertEqual(self.conteos()[3:6], (0, 0, 0))

    def test_excepcion_entrenamiento_no_persiste_ni_expone_datos(self):
        self.crear()
        anterior = self.estado()
        with patch.object(modelos, "entrenar_modelo", side_effect=ValueError("SELECT PRIVADO C:/secreto")):
            self.error(self.post("random_forest", {"n_estimators": 20}), 422, "error_calculo")
        self.assertEqual(self.estado(), anterior)

    def test_temporalidad_tres_cortes_recursion_y_ventas_futuras_excluidas(self):
        for corte, fin in ((CORTE, "2017-07-01"), ("2017-07-01", "2017-07-16"), ("2017-07-16", "2017-07-31")):
            resultado = self.crear(corte=corte)
            detalles = self.cliente.get(f"{PREFIJO}/{resultado['pronostico_id']}").json()["detalles"]
            self.assertEqual(detalles[-1]["fecha_objetivo"], fin)
            self.assertEqual(detalles[7]["venta_estimada"], detalles[0]["venta_estimada"])
        entrenar = modelos.entrenar_modelo
        observados = []

        def observar(*args, **kwargs):
            datos = args[0]
            self.assertLessEqual(datos["date"].max().date(), date.fromisoformat(CORTE))
            tabla = construir_tabla_serie(datos, args[1], args[2], args[4])
            self.assertNotIn(date(2016, 12, 25), set(tabla.tabla["fecha_objetivo"]))
            descarte = next(d for d in tabla.descartes if d.fecha_objetivo == date(2016, 12, 26))
            self.assertIn(date(2016, 12, 25), descarte.faltantes["lag_1"])
            modelo = entrenar(*args, **kwargs)
            self.assertEqual(tuple(modelo.estimador.feature_names_in_), contrato.VARIABLES)
            self.assertNotIn("onpromotion", modelo.estimador.feature_names_in_)
            observados.append(modelo)
            return modelo

        parametros = {"n_estimators": 20, "max_depth": 3}
        with patch.object(modelos, "entrenar_modelo", side_effect=observar):
            primero = self.crear("random_forest", parametros)
            self.sql("UPDATE venta_diaria SET sales=999999.75,onpromotion=9999 WHERE serie_id=? AND fecha>?",
                     (self.serie["id"], CORTE))
            anterior = self.estado()
            repetida = self.post("random_forest", parametros)
        self.assertEqual(len(observados), 2)
        self.assertEqual(repetida.status_code, 200, repetida.text)
        self.assertEqual(repetida.json()["pronostico_id"], primero["pronostico_id"])
        self.assertEqual(self.estado(), anterior)
        for corte, codigo in (("2017-07-17", "prueba_reservada"), ("2017-07-31", "prueba_reservada"),
                              ("2017-08-01", "prueba_reservada"), ("2017-06-17", "corte_no_autorizado")):
            self.error(self.post(corte=corte), 422, codigo)
        self.assertEqual(self.estado(), anterior)

    def test_tipos_max_features_preservados_desde_http_hasta_estimador_y_configuracion(self):
        resultados = []
        for valor in (1, 1.0):
            resultado = self.crear("random_forest", {"n_estimators": 20, "max_features": valor})
            config = self.cliente.get(f"{PREFIJO}/{resultado['pronostico_id']}/configuracion").json()["configuracion"]
            self.assertIs(type(config["parametros"]["max_features"]), type(valor))
            self.assertIs(type(self.config_guardada(resultado["experimento_id"])["parametros"]["max_features"]), type(valor))
            resultados.append((resultado, contrato.identidad_configuracion(config)))
        self.assertNotEqual(resultados[0][0]["experimento_id"], resultados[1][0]["experimento_id"])
        self.assertNotEqual(resultados[0][1], resultados[1][1])

    def test_errores_http_no_entrenan_y_segunda_carga_aisla_conjunto_anterior(self):
        previo = self.crear()
        anterior = self.estado()
        with patch.object(modelos, "entrenar_modelo", side_effect=AssertionError("inválida entrena")), \
                patch.object(registro, "_abrir_base", side_effect=AssertionError("inválida abre base")):
            for config in ({"tecnica": "random_forest", "parametros": {"n_estimators": True}},
                           {"tecnica": "random_forest", "variables": list(reversed(contrato.VARIABLES))}):
                self.error(self.post("random_forest", configuracion=config), 422, "configuracion_invalida")
        self.assertEqual(self.estado(), anterior)
        self.error(self.cliente.get(f"{PREFIJO}/999999/configuracion"), 404, "pronostico_inexistente")
        self.assertEqual(self.subir().status_code, 200)
        for sufijo in ("", "/configuracion"):
            self.error(self.cliente.get(f"{PREFIJO}/{previo['pronostico_id']}{sufijo}"), 404, "pronostico_inexistente")
        self.error(self.cliente.get(self.url), 404, "serie_inexistente")
        self.error(self.post(), 404, "serie_inactiva")
        self.assertEqual(self.sql("SELECT COUNT(*) FROM conjunto_datos WHERE activo=1"), [(1,)])
        self.assertEqual(self.conteos()[3:6], (1, 1, 15))
        self.assertEqual(self.sql("PRAGMA foreign_key_check"), [])
        ausente = self.ruta.with_name("ausente.sqlite3")
        app.dependency_overrides[obtener_ruta_base] = lambda: ausente
        self.error(self.cliente.get(f"{PREFIJO}/{previo['pronostico_id']}/configuracion"), 503, "base_no_disponible")
        self.assertFalse(ausente.exists())


if __name__ == "__main__":
    unittest.main()
