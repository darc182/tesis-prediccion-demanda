"""Persistencia de pronósticos exclusivamente en bases temporales sintéticas."""

from contextlib import closing
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.app.services import registro_pronosticos as registro
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

    def registrar(self, tecnica=BASE, corte=CORTE, serie=83):
        return registro.registrar_pronostico(self.ruta, serie, tecnica, corte)

    def conteos(self):
        return tuple(self.consultar(f"SELECT COUNT(*) FROM {t}")[0][0]
                     for t in ("experimento", "pronostico", "detalle_pronostico", "evaluacion", "simulacion"))

    def historicos(self):
        return tuple(self.consultar(f"SELECT * FROM {t} ORDER BY id")
                     for t in ("conjunto_datos", "serie_temporal", "venta_diaria"))

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
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "incompleto"):
            self.registrar()
        self.assertEqual(self.conteos(), (1, 1, 14, 0, 0))

    def test_rechaza_configuracion_conflictiva(self):
        self.registrar()
        self.consultar("UPDATE experimento SET configuracion_json='{}'")
        with self.assertRaisesRegex(registro.ErrorRegistroPronostico, "conflictivo"):
            self.registrar()
        self.assertEqual(self.consultar("SELECT configuracion_json FROM experimento"), [("{}",)])

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

        def entrenar_sin_bloqueo(*args):
            eventos.append("entrenar")
            # Una conexión independiente puede reservar escritura durante fit.
            with closing(conectar(self.ruta)) as c:
                c.execute("BEGIN IMMEDIATE")
                c.rollback()
            return entrenar(*args)

        with patch.object(registro, "_abrir_base", side_effect=observar), \
                patch.object(modelos, "entrenar_modelo", side_effect=entrenar_sin_bloqueo):
            self.registrar("random_forest")
        self.assertEqual(eventos, ["entrenar", "abrir_escritura"])

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
