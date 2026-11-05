"""Persistencia de pronósticos exclusivamente en bases temporales sintéticas."""

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import pandas as pd
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from backend.app.services import registro_pronosticos as registro
from backend.app.services import configuracion_experimentos as configuracion
from backend.app.services.consulta_pronosticos import consultar_pronostico, listar_pronosticos
from backend.app.storage.db import conectar, inicializar_base
from scripts import modelos_prediccion as modelos


CORTE = date(2017, 6, 16)
BASE = "estacional_semanal_recursiva"


class RegistroPronosticosTests(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.ruta = Path(temporal.name) / "prueba.sqlite3"
        inicializar_base(self.ruta)
        self.filas = [((CORTE - timedelta(days=59-i)).isoformat(), (i % 7) * 0.25, 2)
                      for i in range(60)]
        with closing(conectar(self.ruta)) as c, c:
            c.executemany("INSERT INTO conjunto_datos (id,nombre_archivo,hash,fecha_carga,activo) "
                          "VALUES (?,?,?,?,?)", [(2, "anterior.csv", "hash2", "2026-01-01", 0),
                                                (7, "actual.csv", "hash7", "2026-01-02", 1)])
            c.executemany("INSERT INTO serie_temporal (id,conjunto_id,store_nbr,family,criterio_seleccion) "
                          "VALUES (?,?,?,?,?)", [(24, 2, 17, "PREPARED FOODS", "sintética"),
                                                (83, 7, 17, "PREPARED FOODS", "sintética"),
                                                (84, 7, 8, "BREAD/BAKERY", "sintética")])
            c.executemany("INSERT INTO venta_diaria (serie_id,fecha,sales,onpromotion) VALUES (?,?,?,?)",
                          [(83, *fila) for fila in self.filas]
                          + [(83, (CORTE + timedelta(days=i)).isoformat(), 1e9, 3) for i in range(1, 16)]
                          + [(24, "2017-06-16", 300, 0), (84, "2017-06-16", 500, 0)])

    def consultar(self, sql, parametros=()):
        with closing(conectar(self.ruta)) as c, c:
            return c.execute(sql, parametros).fetchall()

    def registrar(self, tecnica=BASE, corte=CORTE, serie=83, *, configuracion_solicitada=None):
        argumentos = {} if configuracion_solicitada is None else {"configuracion_solicitada": configuracion_solicitada}
        return registro.registrar_pronostico(self.ruta, serie, tecnica, corte, **argumentos)

    def conteos(self):
        return tuple(self.consultar(f"SELECT COUNT(*) FROM {t}")[0][0]
                     for t in ("experimento", "pronostico", "detalle_pronostico", "evaluacion", "simulacion"))

    def historicos(self):
        return tuple(self.consultar(f"SELECT * FROM {t} ORDER BY id")
                     for t in ("conjunto_datos", "serie_temporal", "venta_diaria"))

    def resultado_config(self, nombre, valor=0):
        """Cálculo SINTÉTICO: parámetros válidos RF, sin afirmar que se entrenaron.

        Usa predicciones de línea base como datos de prueba y cambia sus metadatos.
        Los modelos reales continúan probándose sin mocks en sus propios tests.
        """
        with closing(conectar(self.ruta)) as c:
            instantanea = registro._leer_instantanea(c, 83, CORTE)
        predicciones, config, inicio, fin = registro._calcular(instantanea, BASE, CORTE)
        predictiva = configuracion.normalizar_configuracion({
            "tecnica": "random_forest", "parametros": {"n_estimators": {"A": 100, "B": 110, "C": 120}[nombre]}})
        config.update({"tecnica": "random_forest", "parametros": predictiva["parametros"],
                       "configuracion_predictiva": predictiva,
                       "reglas": configuracion.reglas_heredadas("random_forest")})
        predicciones = tuple(replace(p, tecnica="random_forest", venta_estimada=p.venta_estimada + valor)
                             for p in predicciones)
        return predicciones, config, inicio, fin

    def registrar_resultado(self, resultado):
        with patch.object(registro, "_calcular", return_value=resultado):
            return self.registrar(resultado[1]["tecnica"])

    def test_registro_linea_base_relaciones_y_metadatos(self):
        anterior = self.historicos()
        r = self.registrar()
        self.assertTrue(r.nuevo)
        self.assertFalse(r.reutilizado)
        self.assertEqual((r.conjunto_id, r.serie_id, r.horizonte, r.cantidad_detalles), (7, 83, 15, 15))
        self.assertEqual((r.tecnica, r.fecha_corte), (BASE, "2017-06-16"))
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))
        fila = self.consultar(
            "SELECT e.serie_id,p.experimento_id,p.tipo,p.horizonte,p.fecha_corte,e.ruta_modelo,"
            "p.duracion_segundos,p.fecha_generacion FROM experimento e "
            "JOIN pronostico p ON p.experimento_id=e.id WHERE p.id=?", (r.pronostico_id,))[0]
        self.assertEqual(fila[:6], (83, r.experimento_id, "validacion", 15, "2017-06-16", None))
        self.assertGreaterEqual(fila[6], 0)
        self.assertIsNotNone(datetime.fromisoformat(fila[7]).tzinfo)
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])
        self.assertEqual(self.historicos(), anterior)

    def comprobar_regresor(self, tecnica):
        r = self.registrar(tecnica)
        self.assertEqual(r.cantidad_detalles, 15)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))
        config = json.loads(self.consultar("SELECT configuracion_json FROM experimento")[0][0])
        self.assertEqual(config["tecnica"], tecnica)
        self.assertEqual(config["parametros"]["random_state"], 42)
        self.assertEqual(config["entrenamiento"]["filas_validas"], 46)
        self.assertEqual(config["entrenamiento"]["filas_descartadas"], 14)
        self.assertEqual(config["versiones"]["scikit_learn"], modelos.sklearn.__version__)
        self.assertFalse(config["reglas"]["onpromotion_predictor"])

    def test_registro_random_forest_real(self):
        self.comprobar_regresor("random_forest")

    def test_registro_hist_gradient_boosting_real(self):
        self.comprobar_regresor("hist_gradient_boosting")

    def registrar_cuatro_reales(self):
        """Integración F3: cuatro entrenamientos reales, sin dobles del cálculo."""
        casos = [
            ("random_forest", None),
            ("random_forest", {"tecnica": "random_forest", "parametros": {
                "n_estimators": 200, "max_depth": 4, "random_state": 7}}),
            ("hist_gradient_boosting", None),
            ("hist_gradient_boosting", {"tecnica": "hist_gradient_boosting", "parametros": {
                "max_iter": 150, "learning_rate": 0.1, "l2_regularization": 0.5, "random_state": 11}}),
        ]
        return [(t, c, self.registrar(t, configuracion_solicitada=c)) for t, c in casos]

    def test_cuatro_experimentos_reales_coexisten_se_consultan_y_reutilizan_sin_escribir(self):
        historicos = self.historicos()
        casos = self.registrar_cuatro_reales()
        self.assertTrue(all(r.nuevo for _, _, r in casos))
        self.assertEqual(len({r.experimento_id for _, _, r in casos}), 4)
        self.assertEqual(len({r.pronostico_id for _, _, r in casos}), 4)
        self.assertEqual(self.conteos(), (4, 4, 60, 0, 0))
        consultas = {}
        textos = {}
        for tecnica, solicitud, r in casos:
            consulta = consultar_pronostico(self.ruta, r.pronostico_id)
            consultas[r.pronostico_id] = consulta
            self.assertEqual(consulta["pronostico"]["experimento_id"], r.experimento_id)
            self.assertEqual(len(consulta["detalles"]), 15)
            texto = self.consultar("SELECT configuracion_json FROM experimento WHERE id=?", (r.experimento_id,))[0][0]
            textos[r.experimento_id] = texto
            c = json.loads(texto)
            efectiva = configuracion.normalizar_configuracion({"tecnica": tecnica} if solicitud is None else solicitud)
            self.assertEqual(c["version_registro"], 2)
            self.assertEqual(c["configuracion_predictiva"], efectiva)
            self.assertEqual(c["parametros"], efectiva["parametros"])
            # Segundo entrenamiento REAL e independiente: contrasta JSON y resultados almacenados.
            datos = pd.DataFrame(self.filas, columns=["date", "sales", "onpromotion"])
            datos["store_nbr"], datos["family"] = 17, "PREPARED FOODS"
            m = modelos.entrenar_modelo(datos, 17, "PREPARED FOODS", tecnica, CORTE,
                                        configuracion_solicitada=solicitud)
            self.assertEqual(json.dumps(c["parametros"], sort_keys=True),
                             json.dumps(m.estimador.get_params(deep=False), sort_keys=True))
            if tecnica == "random_forest":
                self.assertEqual(len(m.estimador.estimators_), c["parametros"]["n_estimators"])
            else:
                self.assertEqual(m.estimador.n_iter_, c["parametros"]["max_iter"])
            predicciones = modelos.pronosticar_recursivo(m, datos)
            self.assertEqual([d["venta_estimada"] for d in consulta["detalles"]],
                             [p.venta_estimada for p in predicciones])
        contenido = self.ruta.read_bytes()
        sentencias = []
        abrir = registro._abrir_base

        def observar(*args, **kwargs):
            c = abrir(*args, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c

        with patch.object(registro, "_abrir_base", side_effect=observar):
            for tecnica, solicitud, r in reversed(casos):
                repetido = self.registrar(tecnica, configuracion_solicitada=solicitud)
                self.assertTrue(repetido.reutilizado)
                self.assertEqual((repetido.experimento_id, repetido.pronostico_id), (r.experimento_id, r.pronostico_id))
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in sentencias))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual({p["id"] for p in listar_pronosticos(self.ruta, 83)["pronosticos"]}, set(consultas))
        for _, _, r in casos:
            self.assertEqual(consultar_pronostico(self.ruta, r.pronostico_id), consultas[r.pronostico_id])
            self.assertEqual(self.consultar("SELECT configuracion_json FROM experimento WHERE id=?", (r.experimento_id,))[0][0],
                             textos[r.experimento_id])
        self.assertEqual(self.historicos(), historicos)
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])

    def test_configuraciones_invalidas_no_calculan_ni_insertan(self):
        self.registrar()
        contenido = self.ruta.read_bytes()
        casos = [
            ("random_forest", {"tecnica": "hist_gradient_boosting"}),
            ("random_forest", {"tecnica": "random_forest", "parametros": {"n_estimators": True}}),
            ("random_forest", {"tecnica": "random_forest", "parametros": {"n_estimators": 501}}),
            ("random_forest", {"tecnica": "random_forest", "parametros": {"n_jobs": 2}}),
            ("random_forest", {"tecnica": "random_forest", "parametros": {"otro": 3}}),
            ("random_forest", {"tecnica": "random_forest", "variables": list(reversed(configuracion.VARIABLES))}),
            ("random_forest", {"tecnica": "random_forest", "variables": list(configuracion.VARIABLES[:-1])}),
            ("random_forest", {"tecnica": "random_forest", "variables": [*configuracion.VARIABLES, "onpromotion"]}),
            ("hist_gradient_boosting", {"tecnica": "hist_gradient_boosting", "parametros": {"early_stopping": True}}),
            ("hist_gradient_boosting", {"tecnica": "hist_gradient_boosting", "parametros": {"learning_rate": float("inf")}}),
            (BASE, {"tecnica": BASE, "parametros": {"periodo_estacional_dias": 14}}),
        ]
        with patch.object(registro, "_calcular") as calcular, patch.object(registro, "_abrir_base") as abrir:
            for tecnica, solicitud in casos:
                with self.subTest(solicitud=solicitud), self.assertRaises(registro.ErrorRegistroPronostico) as error:
                    self.registrar(tecnica, configuracion_solicitada=solicitud)
                self.assertEqual(error.exception.codigo, "error_calculo")
            calcular.assert_not_called()
            abrir.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_linea_base_opcional_explicita_sigue_siendo_la_original(self):
        anterior = self.registrar()
        contenido = self.ruta.read_bytes()
        repetido = self.registrar(configuracion_solicitada={"tecnica": BASE})
        self.assertTrue(repetido.reutilizado)
        self.assertEqual(repetido.pronostico_id, anterior.pronostico_id)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_alternativo_real_reutiliza_formato_antiguo_sin_migrarlo(self):
        t = "random_forest"
        solicitud = {"tecnica": t, "parametros": {"n_estimators": 30, "max_depth": 3}}
        r = self.registrar(t, configuracion_solicitada=solicitud)
        self.convertir_a_formato_antiguo(r.experimento_id)
        contenido = self.ruta.read_bytes()
        repetido = self.registrar(t, configuracion_solicitada=solicitud)
        self.assertTrue(repetido.reutilizado)
        self.assertEqual(repetido.pronostico_id, r.pronostico_id)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_fallo_persistencia_alternativa_real_preserva_default(self):
        t = "random_forest"
        r = self.registrar(t)
        consulta = consultar_pronostico(self.ruta, r.pronostico_id)
        historicos = self.historicos()
        self.consultar("CREATE TRIGGER fallo BEFORE INSERT ON detalle_pronostico WHEN NEW.paso=8 "
                       "BEGIN SELECT RAISE(ABORT,'fallo alternativa real'); END")
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "fallo alternativa real") as error:
            self.registrar(t, configuracion_solicitada={"tecnica": t, "parametros": {"n_estimators": 30}})
        self.assertEqual(error.exception.codigo, "error_persistencia")
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(consultar_pronostico(self.ruta, r.pronostico_id), consulta)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))
        self.assertEqual(self.historicos(), historicos)
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])

    def test_no_registra_alternativa_si_estimador_ignora_solicitud(self):
        t = "random_forest"
        self.registrar(t)
        crear = modelos.crear_estimador
        contenido = self.ruta.read_bytes()

        def ignorar(tecnica, **kwargs):
            return crear(tecnica)

        with patch.object(modelos, "crear_estimador", side_effect=ignorar):
            with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "parámetros reales"):
                self.registrar(t, configuracion_solicitada={"tecnica": t, "parametros": {"n_estimators": 30}})
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_concurrencia_de_alternativa_real_no_duplica(self):
        t = "random_forest"
        solicitud = {"tecnica": t, "parametros": {"n_estimators": 30, "random_state": 7}}
        barrera = Barrier(2)
        calcular = registro._calcular

        def sincronizar(*args, **kwargs):
            resultado = calcular(*args, **kwargs)  # fit real; solo sincronizamos la llegada al registro
            barrera.wait(timeout=20)
            return resultado

        with patch.object(registro, "_calcular", side_effect=sincronizar), ThreadPoolExecutor(max_workers=2) as ejecutor:
            futuros = [ejecutor.submit(self.registrar, t, configuracion_solicitada=solicitud) for _ in range(2)]
            resultados = [f.result(timeout=30) for f in futuros]
        self.assertEqual(sorted(r.nuevo for r in resultados), [False, True])
        self.assertEqual(len({(r.experimento_id, r.pronostico_id) for r in resultados}), 1)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_rechaza_serie_inactiva(self):
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conjunto activo"):
            self.registrar(serie=24)
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_rechaza_serie_inexistente(self):
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "no existe"):
            self.registrar(serie=999)

    def test_rechaza_tecnica_desconocida(self):
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "Técnica desconocida"):
            self.registrar("otra")

    def test_rechaza_base_ausente_sin_crearla_y_base_sin_esquema(self):
        ausente = self.ruta.with_name("ausente.sqlite3")
        with self.assertRaises(registro.ErrorRegistroPronostico):
            registro.registrar_pronostico(ausente, 83, BASE, CORTE)
        self.assertFalse(ausente.exists())
        vacia = self.ruta.with_name("vacia.sqlite3")
        vacia.touch()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "sin esquema"):
            registro.registrar_pronostico(vacia, 83, BASE, CORTE)
        self.assertEqual(vacia.stat().st_size, 0)

    def test_rechaza_cortes_no_autorizados_y_prueba_reservada(self):
        for corte in ("2017-06-15", "2017-06-17", "2017-07-02", "2017-07-15",
                      "2017-07-17", "2017-07-31", "2017-08-01", "2017-08-15",
                      "2017-02-30", "20170616"):
            with self.subTest(corte=corte), self.assertRaises(registro.ErrorRegistroPronostico):
                self.registrar(corte=corte)
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "prueba final reservada"):
            self.registrar(corte="2017-07-31")
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_admite_exclusivamente_tres_cortes_aprobados(self):
        self.assertEqual(registro.CORTES_VALIDACION,
                         (CORTE, date(2017, 7, 1), date(2017, 7, 16)))
        for corte in registro.CORTES_VALIDACION:
            self.assertEqual(self.registrar(corte=corte).cantidad_detalles, 15)
        self.assertEqual(self.conteos(), (3, 3, 45, 0, 0))
        self.assertEqual(self.consultar("SELECT MAX(fecha_objetivo) FROM detalle_pronostico")[0][0],
                         "2017-07-31")

    def test_conserva_ceros_fracciones_pasos_y_fechas(self):
        self.registrar()
        detalles = self.consultar("SELECT paso,fecha_objetivo,venta_estimada "
                                  "FROM detalle_pronostico ORDER BY paso")
        self.assertEqual([d[0] for d in detalles], list(range(1, 16)))
        self.assertEqual([d[1] for d in detalles],
                         [(CORTE + timedelta(days=i)).isoformat() for i in range(1, 16)])
        self.assertEqual([d[2] for d in detalles[:7]], [fila[1] for fila in self.filas[-7:]])
        self.assertIn(0.0, [d[2] for d in detalles])
        self.assertIn(0.25, [d[2] for d in detalles])

    def test_lectura_exclusiva_serie_y_fechas_hasta_corte(self):
        self.consultar("UPDATE venta_diaria SET sales='NO_LEER' WHERE serie_id=? AND fecha>?",
                       (83, CORTE.isoformat()))
        original = registro.linea_base.pronosticar_serie

        def verificar(datos, tienda, familia, corte):
            self.assertEqual(len(datos), 60)
            self.assertEqual(set(datos["store_nbr"]), {17})
            self.assertEqual(set(datos["family"]), {"PREPARED FOODS"})
            self.assertLessEqual(datos["date"].max().date(), corte)
            return original(datos, tienda, familia, corte)

        with patch.object(registro.linea_base, "pronosticar_serie", side_effect=verificar) as llamada:
            self.registrar()
        llamada.assert_called_once()

    def test_fechas_del_historial_real_no_del_calendario_esperado(self):
        self.consultar("DELETE FROM venta_diaria WHERE serie_id=? AND fecha=?", (83, CORTE.isoformat()))
        self.registrar()
        inicio, fin, texto = self.consultar(
            "SELECT inicio_entrenamiento,fin_entrenamiento,configuracion_json FROM experimento")[0]
        self.assertEqual(inicio, self.filas[0][0])
        self.assertEqual(fin, "2017-06-15")
        config = json.loads(texto)
        self.assertEqual(config["historial"]["cantidad"], 59)
        self.assertEqual(config["historial"]["fin_disponible"], fin)
        self.assertEqual(len(config["historial"]["sha256_observaciones"]), 64)

    def test_rollback_completo_si_falla_un_detalle(self):
        self.consultar("CREATE TRIGGER fallo BEFORE INSERT ON detalle_pronostico WHEN NEW.paso=8 "
                       "BEGIN SELECT RAISE(ABORT,'fallo detalle simulado'); END")
        anterior = self.historicos()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "Registro revertido.*fallo detalle"):
            self.registrar()
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))
        self.assertEqual(self.historicos(), anterior)

    def test_rollback_si_falla_cabecera_pronostico(self):
        self.consultar("CREATE TRIGGER fallo BEFORE INSERT ON pronostico "
                       "BEGIN SELECT RAISE(ABORT,'fallo cabecera simulado'); END")
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "Registro revertido"):
            self.registrar()
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_rollback_si_un_trigger_ignora_un_detalle(self):
        self.consultar("CREATE TRIGGER ignorar BEFORE INSERT ON detalle_pronostico WHEN NEW.paso=8 "
                       "BEGIN SELECT RAISE(IGNORE); END")
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "quince detalles esperados"):
            self.registrar()
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_reutilizacion_identica_sin_escrituras(self):
        primero = self.registrar()
        contenido = self.ruta.read_bytes()
        consultas = []
        abrir = registro._abrir_base

        def observar(ruta, escritura=False):
            c = abrir(ruta, escritura)
            c.set_trace_callback(consultas.append)
            return c

        with patch.object(registro, "_abrir_base", side_effect=observar):
            segundo = self.registrar()
        self.assertTrue(segundo.reutilizado)
        self.assertFalse(segundo.nuevo)
        self.assertEqual((primero.experimento_id, primero.pronostico_id),
                         (segundo.experimento_id, segundo.pronostico_id))
        self.assertFalse(any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
                             for sql in consultas))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def convertir_a_formato_antiguo(self, experimento_id):
        texto = self.consultar("SELECT configuracion_json FROM experimento WHERE id=?",
                               (experimento_id,))[0][0]
        config = json.loads(texto)
        config.pop("configuracion_predictiva")
        config["version_registro"] = 1
        texto = json.dumps(config, ensure_ascii=False, indent=2)
        self.consultar("UPDATE experimento SET configuracion_json=? WHERE id=?", (texto, experimento_id))
        return texto

    def test_reutiliza_defaults_antiguos_de_las_tres_tecnicas_sin_migrar(self):
        historicos = self.historicos()
        for tecnica in registro.TECNICAS:
            with self.subTest(tecnica=tecnica):
                anterior = self.registrar(tecnica)
                texto = self.convertir_a_formato_antiguo(anterior.experimento_id)
                contenido = self.ruta.read_bytes()
                nuevo = self.registrar(tecnica)
                self.assertTrue(nuevo.reutilizado)
                self.assertEqual((nuevo.experimento_id, nuevo.pronostico_id),
                                 (anterior.experimento_id, anterior.pronostico_id))
                self.assertEqual(self.ruta.read_bytes(), contenido)
                self.assertEqual(self.consultar("SELECT configuracion_json FROM experimento WHERE id=?",
                                                (anterior.experimento_id,)), [(texto,)])
        self.assertEqual(self.conteos(), (3, 3, 45, 0, 0))
        self.assertEqual(self.historicos(), historicos)

    def test_cambios_solo_en_auditoria_versiones_y_duracion_reutilizan_sin_escritura(self):
        for formato in (1, 2):
            with self.subTest(formato=formato):
                # Otra serie temporal sintética permite comprobar ambos formatos sin borrar.
                serie = 83 if formato == 1 else 84
                r = self.registrar(serie=serie)
                if formato == 1:
                    self.convertir_a_formato_antiguo(r.experimento_id)
                self.consultar("UPDATE pronostico SET duracion_segundos=999 WHERE id=?", (r.pronostico_id,))
                calcular = registro._calcular

                def cambiar_metadatos(*args):
                    p, c, inicio, fin = calcular(*args)
                    c["versiones"] = {"python": "otra", "pandas": "otra", "scikit_learn": "otra"}
                    c["auditoria"] = {"otro_dato_tecnico": 123}
                    c["entrenamiento"] = {"observaciones_disponibles": 999}
                    return p, c, inicio, fin

                contenido = self.ruta.read_bytes()
                with patch.object(registro, "_calcular", side_effect=cambiar_metadatos):
                    repetido = self.registrar(serie=serie)
                self.assertTrue(repetido.reutilizado)
                self.assertEqual(repetido.pronostico_id, r.pronostico_id)
                self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (2, 2, 30, 0, 0))

    def test_historial_cambiado_entre_ejecuciones_no_reutiliza_aunque_estimaciones_coincidan(self):
        r = self.registrar()
        self.convertir_a_formato_antiguo(r.experimento_id)
        originales = consultar_pronostico(self.ruta, r.pronostico_id)
        # Un dato antiguo no afecta la copia semanal, pero sí la huella del histórico.
        self.consultar("UPDATE venta_diaria SET sales=sales+10 WHERE serie_id=83 AND fecha=?",
                       (self.filas[0][0],))
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "histórico.*cambió") as error:
            self.registrar()
        self.assertEqual(error.exception.codigo, "conflicto")
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(consultar_pronostico(self.ruta, r.pronostico_id), originales)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_historial_no_verificable_no_se_reutiliza(self):
        r = self.registrar()
        texto = self.convertir_a_formato_antiguo(r.experimento_id)
        c = json.loads(texto)
        del c["historial"]
        self.consultar("UPDATE experimento SET configuracion_json=?", (json.dumps(c),))
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "histórico.*verificarse"):
            self.registrar()
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_ambiguedad_entre_formato_antiguo_y_nuevo_rechazada(self):
        a, b = self.resultado_config("A"), self.resultado_config("B")
        ra, rb = self.registrar_resultado(a), self.registrar_resultado(b)
        antiguo = deepcopy(a[1])
        antiguo.pop("configuracion_predictiva")
        antiguo["version_registro"] = 1
        self.consultar("UPDATE experimento SET configuracion_json=? WHERE id=?",
                       (json.dumps(antiguo), rb.experimento_id))
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "ambiguos"):
            self.registrar_resultado(a)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertNotEqual(ra.pronostico_id, rb.pronostico_id)

    def test_configuraciones_antiguas_abc_se_reutilizan_independientemente(self):
        resultados = {n: self.resultado_config(n, v) for n, v in (("A", 0), ("B", 10), ("C", 20))}
        registros = {n: self.registrar_resultado(r) for n, r in resultados.items()}
        for r in registros.values():
            self.convertir_a_formato_antiguo(r.experimento_id)
        contenido = self.ruta.read_bytes()
        for n in ("C", "A", "B"):
            r = self.registrar_resultado(resultados[n])
            self.assertTrue(r.reutilizado)
            self.assertEqual((r.experimento_id, r.pronostico_id),
                             (registros[n].experimento_id, registros[n].pronostico_id))
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (3, 3, 45, 0, 0))

    def test_series_cortes_y_tecnicas_mantienen_contextos_separados(self):
        registros = [self.registrar(), self.registrar(corte=date(2017, 7, 1)),
                     self.registrar(serie=84), self.registrar("random_forest")]
        self.assertEqual(len({r.pronostico_id for r in registros}), 4)
        for r in registros:
            repetido = self.registrar(r.tecnica, r.fecha_corte, r.serie_id)
            self.assertTrue(repetido.reutilizado)
            self.assertEqual(repetido.pronostico_id, r.pronostico_id)
        self.assertEqual(self.conteos(), (4, 4, 60, 0, 0))

    def test_defaults_del_contrato_no_reetiquetan_estimador_con_otro_parametro(self):
        crear = modelos.crear_estimador
        for cambio in ({"n_estimators": 101}, {"max_features": 1}):
            def cambiar(tecnica):
                return crear(tecnica).set_params(**cambio)

            contenido = self.ruta.read_bytes()
            with self.subTest(cambio=cambio), patch.object(modelos, "crear_estimador", side_effect=cambiar):
                with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "configuración solicitada"):
                    self.registrar("random_forest")
            self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_configuracion_no_se_asocia_a_otra_tecnica(self):
        resultado = self.resultado_config("A")
        contenido = self.ruta.read_bytes()
        # Predicciones y solicitud de base, pero configuración RF: rechazar antes de INSERT.
        predicciones = tuple(replace(p, tecnica=BASE) for p in resultado[0])
        with patch.object(registro, "_calcular", return_value=(predicciones, *resultado[1:])):
            with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "configuración.*técnica") as error:
                self.registrar()
        self.assertEqual(error.exception.codigo, "resultado_invalido")
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_rechaza_duplicado_con_valores_conflictivos_sin_sobrescribir(self):
        self.registrar()
        self.consultar("UPDATE detalle_pronostico SET venta_estimada=99 WHERE paso=1")
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conflictivo"):
            self.registrar()
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_rechaza_duplicado_incompleto(self):
        self.registrar()
        self.consultar("DELETE FROM detalle_pronostico WHERE paso=15")
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "integridad.*incompleto") as error:
            self.registrar()
        self.assertEqual(error.exception.codigo, "conflicto")
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 14, 0, 0))

    def test_configuracion_antigua_irreconocible_rechaza_sin_inventar_defaults(self):
        self.registrar()
        self.consultar("UPDATE experimento SET configuracion_json='{}'")
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "reconstruir con seguridad") as error:
            self.registrar()
        self.assertEqual(error.exception.codigo, "conflicto")
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_tres_configuraciones_coexisten_y_se_reutilizan_sin_escrituras(self):
        resultados = {n: self.resultado_config(n, v) for n, v in (("A", 0), ("B", 10), ("C", 20))}
        historicos = self.historicos()
        registros = {n: self.registrar_resultado(r) for n, r in resultados.items()}
        self.assertTrue(all(r.nuevo for r in registros.values()))
        self.assertEqual(len({r.experimento_id for r in registros.values()}), 3)
        self.assertEqual(len({r.pronostico_id for r in registros.values()}), 3)
        self.assertEqual(self.conteos(), (3, 3, 45, 0, 0))
        contenido = self.ruta.read_bytes()
        sentencias = []
        abrir = registro._abrir_base

        def observar(*args, **kwargs):
            c = abrir(*args, **kwargs)
            c.set_trace_callback(sentencias.append)
            return c

        with patch.object(registro, "_abrir_base", side_effect=observar):
            for nombre in ("B", "A", "C", "A", "B"):
                r = self.registrar_resultado(resultados[nombre])
                self.assertTrue(r.reutilizado)
                self.assertEqual((r.experimento_id, r.pronostico_id),
                                 (registros[nombre].experimento_id, registros[nombre].pronostico_id))
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
                             for s in sentencias))
        lista = listar_pronosticos(self.ruta, 83)["pronosticos"]
        self.assertEqual({p["id"] for p in lista}, {r.pronostico_id for r in registros.values()})
        for nombre, r in registros.items():
            consulta = consultar_pronostico(self.ruta, r.pronostico_id)
            self.assertEqual(consulta["pronostico"]["experimento_id"], r.experimento_id)
            self.assertEqual(len(consulta["detalles"]), 15)
            self.assertEqual([d["venta_estimada"] for d in consulta["detalles"]],
                             [p.venta_estimada for p in resultados[nombre][0]])
            texto, inicio, fin = self.consultar(
                "SELECT configuracion_json,inicio_entrenamiento,fin_entrenamiento "
                "FROM experimento WHERE id=?", (r.experimento_id,))[0]
            self.assertEqual(json.loads(texto), resultados[nombre][1])
            self.assertEqual((inicio, fin), resultados[nombre][2:])
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.historicos(), historicos)
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])

    def test_busqueda_independiente_del_orden_de_insercion(self):
        # Cada subcaso prepara su propia SQLite temporal, sin limpiar registros.
        for orden in (("B", "A", "C"), ("C", "B", "A"), ("A", "C", "B")):
            with self.subTest(orden=orden):
                datos = RegistroPronosticosTests()
                datos.setUp()
                try:
                    resultados = {n: datos.resultado_config(n) for n in orden}
                    registros = {n: datos.registrar_resultado(resultados[n]) for n in orden}
                    for n in reversed(orden):
                        r = datos.registrar_resultado(resultados[n])
                        self.assertTrue(r.reutilizado)
                        self.assertEqual(r.pronostico_id, registros[n].pronostico_id)
                    self.assertEqual(datos.conteos(), (3, 3, 45, 0, 0))
                finally:
                    datos.doCleanups()

    def test_compara_contenido_json_sin_depender_de_espacios_ni_orden_de_claves(self):
        resultado = self.resultado_config("A")
        primero = self.registrar_resultado(resultado)

        def invertir(objeto):
            if isinstance(objeto, dict):
                return {k: invertir(v) for k, v in reversed(list(objeto.items()))}
            if isinstance(objeto, list):
                return [invertir(v) for v in objeto]
            return objeto

        texto = json.dumps(invertir(resultado[1]), ensure_ascii=True, indent=4)
        self.consultar("UPDATE experimento SET configuracion_json=?", (texto,))
        contenido = self.ruta.read_bytes()
        segundo = self.registrar_resultado(resultado)
        self.assertTrue(segundo.reutilizado)
        self.assertEqual(segundo.pronostico_id, primero.pronostico_id)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_orden_de_variables_diferencia_configuraciones(self):
        a = self.resultado_config("A")
        b = self.resultado_config("A")
        b[1]["configuracion_predictiva"]["variables"].reverse()
        b[1]["reglas"]["predictoras"].reverse()
        ra, rb = self.registrar_resultado(a), self.registrar_resultado(b)
        self.assertNotEqual(ra.experimento_id, rb.experimento_id)
        self.assertEqual(self.registrar_resultado(a).pronostico_id, ra.pronostico_id)
        self.assertEqual(self.registrar_resultado(b).pronostico_id, rb.pronostico_id)
        self.assertEqual(self.conteos(), (2, 2, 30, 0, 0))

    def test_misma_configuracion_resultado_distinto_conflicto_aunque_exista_otra(self):
        a = self.resultado_config("A")
        self.registrar_resultado(a)
        self.registrar_resultado(self.resultado_config("B", 10))
        contenido = self.ruta.read_bytes()
        otra_auditoria = deepcopy(a[1])
        otra_auditoria["versiones"] = {"python": "otra", "scikit_learn": "otra"}
        alterado = (tuple(replace(p, venta_estimada=p.venta_estimada + 1) for p in a[0]),
                    otra_auditoria, *a[2:])
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conflictivo") as error:
            self.registrar_resultado(alterado)
        self.assertEqual(error.exception.codigo, "conflicto")
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_configuracion_incompleta_no_se_confunde_con_otra_completa(self):
        a, b = self.resultado_config("A"), self.resultado_config("B", 10)
        ra = self.registrar_resultado(a)
        rb = self.registrar_resultado(b)
        self.consultar("DELETE FROM detalle_pronostico WHERE pronostico_id=? AND paso=15",
                       (ra.pronostico_id,))
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "integridad.*incompleto"):
            self.registrar_resultado(a)
        self.assertEqual(self.registrar_resultado(b).pronostico_id, rb.pronostico_id)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_rechaza_duplicados_ambiguos_sin_elegir_un_resultado(self):
        a, b = self.resultado_config("A"), self.resultado_config("B")
        ra = self.registrar_resultado(a)
        rb = self.registrar_resultado(b)
        # Datos heredados ambiguos: convertir B en otra fila con configuración A.
        self.consultar("UPDATE experimento SET configuracion_json=? WHERE id=?",
                       (json.dumps(a[1]), rb.experimento_id))
        for modificar in (False, True):
            with self.subTest(resultados_distintos=modificar):
                if modificar:
                    self.consultar("UPDATE detalle_pronostico SET venta_estimada=99 WHERE pronostico_id=?",
                                   (ra.pronostico_id,))
                contenido = self.ruta.read_bytes()
                with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "ambiguos") as error:
                    self.registrar_resultado(a)
                self.assertEqual(error.exception.codigo, "conflicto")
                self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_rechaza_metadatos_distintos_de_la_misma_configuracion(self):
        self.registrar()
        for columna, valor in (("inicio_entrenamiento", "2017-04-01"),
                                ("fin_entrenamiento", "2017-06-15"), ("ruta_modelo", "modelo.pkl")):
            with self.subTest(columna=columna):
                anterior = self.consultar(f"SELECT {columna} FROM experimento")[0][0]
                self.consultar(f"UPDATE experimento SET {columna}=?", (valor,))
                contenido = self.ruta.read_bytes()
                with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conflictivo"):
                    self.registrar()
                self.assertEqual(self.ruta.read_bytes(), contenido)
                self.consultar(f"UPDATE experimento SET {columna}=?", (anterior,))

    def test_fallo_registro_b_revierte_sin_alterar_a(self):
        a, b = self.resultado_config("A"), self.resultado_config("B", 10)
        ra = self.registrar_resultado(a)
        consulta_a = consultar_pronostico(self.ruta, ra.pronostico_id)
        historicos = self.historicos()
        self.consultar("CREATE TRIGGER fallo BEFORE INSERT ON detalle_pronostico WHEN NEW.paso=8 "
                       "BEGIN SELECT RAISE(ABORT,'fallo B'); END")
        contenido = self.ruta.read_bytes()
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "Registro revertido.*fallo B"):
            self.registrar_resultado(b)
        self.assertEqual(self.ruta.read_bytes(), contenido)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))
        self.assertEqual(consultar_pronostico(self.ruta, ra.pronostico_id), consulta_a)
        self.assertEqual(self.historicos(), historicos)
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])
        self.assertTrue(self.registrar_resultado(a).reutilizado)

    def test_concurrencia_real_serializa_busqueda_y_evitar_duplicados_con_otra_configuracion(self):
        self.registrar_resultado(self.resultado_config("B", 10))
        a = self.resultado_config("A")
        barrera = Barrier(2)

        def calcular(*args):
            barrera.wait(timeout=10)
            return a

        with patch.object(registro, "_calcular", side_effect=calcular):
            with ThreadPoolExecutor(max_workers=2) as ejecutor:
                futuros = [ejecutor.submit(self.registrar, "random_forest") for _ in range(2)]
                resultados = [f.result(timeout=20) for f in futuros]
        self.assertEqual(sorted(r.nuevo for r in resultados), [False, True])
        self.assertEqual(len({(r.experimento_id, r.pronostico_id) for r in resultados}), 1)
        self.assertEqual(self.conteos(), (2, 2, 30, 0, 0))
        self.assertEqual(self.consultar("PRAGMA foreign_key_check"), [])

    def test_error_calculo_no_modifica_ningun_dato(self):
        contenido = self.ruta.read_bytes()
        with patch.object(registro, "_calcular", side_effect=ValueError("historial insuficiente")):
            with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "Serie 83.*historial insuficiente"):
                self.registrar()
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_entrenamiento_antes_de_transaccion_de_escritura(self):
        eventos = []
        entrenar = modelos.entrenar_modelo
        abrir = registro._abrir_base

        def observar(ruta, escritura=False):
            if escritura:
                eventos.append("abrir_escritura")
            c = abrir(ruta, escritura)
            self.assertEqual(c.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            return c

        def entrenar_sin_bloqueo(*args, **kwargs):
            eventos.append("entrenar")
            # Una conexión independiente puede reservar escritura durante fit.
            with closing(conectar(self.ruta)) as c:
                c.execute("BEGIN IMMEDIATE")
                c.rollback()
            return entrenar(*args, **kwargs)

        with patch.object(registro, "_abrir_base", side_effect=observar), \
                patch.object(modelos, "entrenar_modelo", side_effect=entrenar_sin_bloqueo):
            self.registrar("random_forest")
            self.registrar("random_forest", configuracion_solicitada={
                "tecnica": "random_forest", "parametros": {"n_estimators": 30}})
        self.assertEqual(eventos, ["entrenar", "abrir_escritura", "entrenar", "abrir_escritura"])

    def test_recomprueba_activo_justo_antes_de_guardar(self):
        calcular = registro._calcular

        def cambiar_activo(*args):
            resultado = calcular(*args)
            with closing(conectar(self.ruta)) as c, c:
                c.execute("UPDATE conjunto_datos SET activo=0 WHERE id=7")
                c.execute("UPDATE conjunto_datos SET activo=1 WHERE id=2")
            return resultado

        with patch.object(registro, "_calcular", side_effect=cambiar_activo):
            with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conjunto activo"):
                self.registrar()
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_rechaza_historial_modificado_durante_calculo(self):
        calcular = registro._calcular

        def cambiar_historial(*args):
            resultado = calcular(*args)
            self.consultar("UPDATE venta_diaria SET sales=22 WHERE serie_id=83 AND fecha=?", (self.filas[0][0],))
            return resultado

        with patch.object(registro, "_calcular", side_effect=cambiar_historial):
            with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "cambiaron durante el cálculo"):
                self.registrar()
        self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_intercalado_dos_solicitudes_no_duplica(self):
        calcular = registro._calcular
        interiores = []

        def intercalar(*args):
            resultado = calcular(*args)
            with patch.object(registro, "_calcular", side_effect=calcular):
                interiores.append(self.registrar())
            return resultado

        with patch.object(registro, "_calcular", side_effect=intercalar):
            exterior = self.registrar()
        self.assertTrue(interiores[0].nuevo)
        self.assertTrue(exterior.reutilizado)
        self.assertEqual(exterior.pronostico_id, interiores[0].pronostico_id)
        self.assertEqual(self.conteos(), (1, 1, 15, 0, 0))

    def test_rechaza_resultados_invalidos_antes_de_insertar(self):
        calcular = registro._calcular
        cambios = [{"store_nbr": 99}, {"family": "OTRA"}, {"tecnica": "otra"},
                   {"fecha_corte": date(2017, 7, 1)}, {"paso": 0},
                   {"fecha_objetivo": CORTE}, {"venta_estimada": -1},
                   {"venta_estimada": float("nan")}, {"venta_estimada": float("inf")},
                   {"venta_estimada": True}, {"venta_estimada": "2.5"}, "corto", "largo"]
        for cambio in cambios:
            def alterar(*args):
                p, config, inicio, fin = calcular(*args)
                if cambio == "corto":
                    p = p[:-1]
                elif cambio == "largo":
                    p = (*p, p[-1])
                else:
                    p = (replace(p[0], **cambio), *p[1:])
                return p, config, inicio, fin

            with self.subTest(cambio=cambio), patch.object(registro, "_calcular", side_effect=alterar):
                with self.assertRaises(registro.ErrorRegistroPronostico):
                    self.registrar()
            self.assertEqual(self.conteos(), (0, 0, 0, 0, 0))

    def test_sin_conjunto_activo(self):
        self.consultar("UPDATE conjunto_datos SET activo=0")
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "No existe un conjunto activo"):
            self.registrar()

    def test_sin_historial_hasta_corte(self):
        self.consultar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha<=?", (CORTE.isoformat(),))
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "no hay historial"):
            self.registrar()


if __name__ == "__main__":
    unittest.main()
