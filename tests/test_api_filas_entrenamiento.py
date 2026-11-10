"""GET de disponibilidad con TestClient y SQLite exclusivamente sintética."""

from contextlib import ExitStack
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from backend.app.api import pronosticos as api
from backend.app.api.datasets import obtener_ruta_base
from backend.app.main import app
from backend.app.services import consulta_filas_entrenamiento as consulta
from backend.app.services import registro_pronosticos as registro
from scripts import modelos_prediccion as modelos
from tests.test_consulta_filas_entrenamiento import CORTES, DatosSinteticos


PREFIJO = "/api/pronosticos/active/series"


class ApiFilasEntrenamientoTests(DatosSinteticos, unittest.TestCase):
    def setUp(self):
        self.preparar_base()
        anteriores = dict(app.dependency_overrides)
        self.addCleanup(lambda: (app.dependency_overrides.clear(), app.dependency_overrides.update(anteriores)))
        app.dependency_overrides[obtener_ruta_base] = lambda: self.ruta
        self.cliente = self.enterContext(TestClient(app))

    def consultar(self, serie=83, corte="2017-06-16", grupo="clasico"):
        return self.cliente.get(f"{PREFIJO}/{serie}/filas-entrenamiento",
                                params={"fecha_corte": corte, "grupo": grupo})

    def comprobar_error(self, respuesta, codigo, estado=422):
        self.assertEqual(respuesta.status_code, estado, respuesta.text)
        self.assertEqual(respuesta.json(), {"detail": {"codigo": codigo, "mensaje": api.ERRORES[codigo][1]}})
        self.assertEqual(respuesta.headers.get("cache-control"), "no-store")
        for privado in (str(self.ruta), "Traceback", "PRIVADO", "SELECT ", "sqlite3", "FUTURO_NO_UTILIZABLE"):
            self.assertNotIn(privado, respuesta.text)

    def test_get_ambos_grupos_tres_cortes_compara_con_tabla_real(self):
        for grupo in ("clasico", "ampliado"):
            for corte in CORTES:
                with self.subTest(grupo=grupo, corte=corte):
                    r = self.consultar(corte=corte, grupo=grupo)
                    self.assertEqual(r.status_code, 200, r.text)
                    self.assertEqual(r.headers["cache-control"], "no-store")
                    body = r.json()
                    self.assertEqual(set(body), {"conjunto_id", "serie", "fecha_corte", "grupo",
                        "version_contrato", "version_reglas", "variables", "observaciones_historicas",
                        "filas_aptas", "filas_descartadas", "descartes_por_variable"})
                    self.assertEqual(body["serie"], {"id": 83, "store_nbr": 17, "family": "PREPARED FOODS"})
                    self.assertEqual((body["conjunto_id"], body["fecha_corte"], body["grupo"]), (7, corte, grupo))
                    version = 1 if grupo == "clasico" else 2
                    self.assertEqual((body["version_contrato"], body["version_reglas"]), (version, version))
                    directo = self.directa(corte, grupo)
                    self.assertEqual(body["variables"], list(directo.columnas_predictoras))
                    self.comprobar_conteos(body, directo)

    def test_ausentes_ceros_y_fracciones_conservan_contabilidad(self):
        for grupo in ("clasico", "ampliado"):
            body = self.consultar(grupo=grupo).json()
            self.assertEqual(body["observaciones_historicas"], 46)
            self.comprobar_conteos(body, self.directa(grupo=grupo))
            self.assertGreater(sum(body["descartes_por_variable"].values()), body["filas_descartadas"])

    def test_historial_insuficiente_devuelve_200_cero_aptas(self):
        self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha<>'2017-06-16'")
        for grupo in ("clasico", "ampliado"):
            r = self.consultar(grupo=grupo)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual((r.json()["observaciones_historicas"], r.json()["filas_aptas"], r.json()["filas_descartadas"]), (1, 0, 1))
            self.comprobar_conteos(r.json(), self.directa(grupo=grupo))

    def test_serie_sin_historial_y_con_solo_ventas_futuras(self):
        self.comprobar_error(self.consultar(serie=84), "sin_historial")
        self.ejecutar("DELETE FROM venta_diaria WHERE serie_id=83 AND fecha<='2017-06-16'")
        self.comprobar_error(self.consultar(), "sin_historial")

    def test_serie_inactiva_e_inexistente(self):
        for serie in (24, 999):
            self.comprobar_error(self.consultar(serie=serie), "serie_inexistente", 404)

    def test_sin_conjunto_activo(self):
        self.ejecutar("UPDATE conjunto_datos SET activo=0")
        self.comprobar_error(self.consultar(), "sin_conjunto_activo", 409)

    def test_ids_invalidos_se_rechazan_antes_del_servicio(self):
        with patch.object(api, "consultar_filas_entrenamiento") as servicio:
            for serie in ("no-entero", "1.5", 0, -1):
                self.comprobar_error(self.consultar(serie=serie), "serie_invalida")
            servicio.assert_not_called()
        with patch.object(consulta.sqlite3, "connect") as abrir:
            self.comprobar_error(self.consultar(serie=2**63), "serie_invalida")
            abrir.assert_not_called()

    def test_grupos_desconocidos_y_linea_base_no_se_aceptan(self):
        with patch.object(api, "consultar_filas_entrenamiento") as servicio:
            for grupo in ("", "Clasico", "alternativo", "estacional_semanal_recursiva", "onpromotion"):
                self.comprobar_error(self.consultar(grupo=grupo), "grupo_invalido")
            servicio.assert_not_called()

    def test_fecha_invalida_y_corte_no_autorizado_no_abren_base(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for corte in ("20170616", "2017-6-16", "2017-02-29", "2017-06-16T00:00:00", "x' OR 1=1 --"):
                self.comprobar_error(self.consultar(corte=corte), "corte_invalido")
            for corte in ("2017-06-15", "2017-07-02", "2017-08-16"):
                self.comprobar_error(self.consultar(corte=corte), "corte_no_autorizado")
            abrir.assert_not_called()

    def test_proteccion_explicita_de_prueba_final(self):
        with patch.object(consulta.sqlite3, "connect") as abrir:
            for corte in ("2017-07-17", "2017-07-31", "2017-08-01", "2017-08-15"):
                self.comprobar_error(self.consultar(corte=corte), "prueba_reservada")
            abrir.assert_not_called()

    def test_parametros_obligatorios(self):
        url = f"{PREFIJO}/83/filas-entrenamiento"
        with patch.object(api, "consultar_filas_entrenamiento") as servicio:
            self.comprobar_error(self.cliente.get(url), "corte_invalido")
            self.comprobar_error(self.cliente.get(url, params={"grupo": "clasico"}), "corte_invalido")
            self.comprobar_error(self.cliente.get(url, params={"fecha_corte": CORTES[0]}), "grupo_invalido")
            servicio.assert_not_called()

    def test_no_acepta_tecnica_hiperparametros_ni_parametros_ambiguos(self):
        url = f"{PREFIJO}/83/filas-entrenamiento"
        comunes = [("fecha_corte", CORTES[0]), ("grupo", "clasico")]
        with patch.object(api, "consultar_filas_entrenamiento") as servicio:
            for extra in (("tecnica", "random_forest"), ("max_depth", "3"), ("grupo", "ampliado"), ("fecha_corte", CORTES[1])):
                self.comprobar_error(self.cliente.get(url, params=[*comunes, extra]), "entrada_filas_invalida")
            servicio.assert_not_called()

    def test_futuro_invalido_no_afecta_conteos_ni_se_expone(self):
        previos = {g: self.consultar(grupo=g).json() for g in ("clasico", "ampliado")}
        self.ejecutar("UPDATE venta_diaria SET sales='FUTURO_NO_UTILIZABLE' WHERE serie_id=83 AND fecha>'2017-06-16'")
        for grupo, previo in previos.items():
            r = self.consultar(grupo=grupo)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), previo)

    def test_historico_corrupto_error_controlado_sin_escrituras(self):
        self.ejecutar("UPDATE venta_diaria SET sales='PRIVADO_NO_NUMERICO' WHERE serie_id=83 AND fecha='2017-06-16'")
        contenido = self.ruta.read_bytes()
        self.comprobar_error(self.consultar(), "historial_invalido", 409)
        self.assertEqual(self.ruta.read_bytes(), contenido)

    def test_base_inexistente_o_sin_esquema_no_se_crea_ni_inicializa(self):
        otra = self.ruta.parent / "inexistente.sqlite3"
        app.dependency_overrides[obtener_ruta_base] = lambda: otra
        self.comprobar_error(self.consultar(), "base_no_disponible", 503)
        self.assertFalse(otra.exists())
        otra.touch()
        self.comprobar_error(self.consultar(), "base_no_disponible", 503)
        self.assertEqual(otra.read_bytes(), b"")

    def test_get_no_entrena_predice_registra_inicializa_ni_modifica_base(self):
        contenido = self.ruta.read_bytes()
        objetivos = [(api, "registrar_pronostico"), (registro, "registrar_pronostico"),
                     (registro, "_calcular"), (modelos, "entrenar_modelo"),
                     (modelos, "crear_estimador"), (modelos, "pronosticar_recursivo"),
                     (modelos.RandomForestRegressor, "fit"), (modelos.RandomForestRegressor, "predict"),
                     (modelos.HistGradientBoostingRegressor, "fit"), (modelos.HistGradientBoostingRegressor, "predict")]
        with ExitStack() as stack:
            dobles = [stack.enter_context(patch.object(objeto, nombre, side_effect=AssertionError(nombre)))
                      for objeto, nombre in objetivos]
            for grupo in ("clasico", "ampliado"):
                for corte in CORTES:
                    self.assertEqual(self.consultar(corte=corte, grupo=grupo).status_code, 200)
            for doble in dobles:
                doble.assert_not_called()
        self.assertEqual(self.ruta.read_bytes(), contenido)
        for tabla in ("experimento", "pronostico", "detalle_pronostico", "evaluacion", "simulacion"):
            self.assertEqual(self.ejecutar(f"SELECT COUNT(*) FROM {tabla}")[0][0], 0)

    def test_fallo_interno_sanitizado_y_sin_cache(self):
        with patch.object(api, "consultar_filas_entrenamiento", side_effect=RuntimeError("PRIVADO /ruta/interna")):
            self.comprobar_error(self.consultar(), "error_interno", 500)

    def test_endpoints_previos_conservan_respuesta_y_cache(self):
        r = self.cliente.get(f"{PREFIJO}/83")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"conjunto_id": 7,
            "serie": {"id": 83, "store_nbr": 17, "family": "PREPARED FOODS"}, "pronosticos": []})
        self.assertNotIn("cache-control", r.headers)
        r = self.cliente.post(f"{PREFIJO}/83", json={"tecnica": "invalida", "fecha_corte": CORTES[0]})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["codigo"], "tecnica_invalida")
        self.assertNotIn("cache-control", r.headers)

    def test_openapi_documenta_grupos_y_conteos_sin_tecnica(self):
        esquema = self.cliente.get("/openapi.json").json()
        ruta = esquema["paths"]["/api/pronosticos/active/series/{serie_id}/filas-entrenamiento"]["get"]
        self.assertEqual({p["name"] for p in ruta["parameters"]}, {"serie_id", "fecha_corte", "grupo"})
        grupo = next(p for p in ruta["parameters"] if p["name"] == "grupo")
        self.assertEqual(grupo["schema"]["enum"], ["clasico", "ampliado"])
        self.assertIn("solaparse", esquema["components"]["schemas"]["FilasEntrenamiento"]["properties"]["descartes_por_variable"]["description"])


if __name__ == "__main__":
    unittest.main()
