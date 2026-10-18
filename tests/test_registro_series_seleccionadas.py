"""Integridad del registro aprobado y reproducción con el selector original."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

from scripts.analizar_periodo_seleccion import (
    FIN_SELECCION,
    HORIZONTE_DIAS,
    INICIO_SELECCION,
    TRAIN,
    calcular_caracteristicas,
    contar_prefijo_fechas,
    leer_bloques_hasta_corte,
)
from scripts.seleccionar_series import (
    CUOTAS,
    PERCENTILES,
    preparar_candidatos,
    seleccionar_series,
)


RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "config" / "series_seleccionadas.json"
APROBADAS = [
    ("A", 17, "PREPARED FOODS"),
    ("A", 8, "BREAD/BAKERY"),
    ("B", 30, "LINGERIE"),
    ("B", 37, "HOME AND KITCHEN II"),
    ("B", 18, "PERSONAL CARE"),
    ("C", 29, "BEAUTY"),
    ("C", 37, "PET SUPPLIES"),
    ("C", 29, "POULTRY"),
    ("D", 23, "HARDWARE"),
    ("D", 5, "SCHOOL AND OFFICE SUPPLIES"),
]
UMBRALES = {
    "A": (0, True, 5, True),
    "B": (5, False, 35, True),
    "C": (35, False, 70, True),
    "D": (70, False, 100, False),
}


class RegistroSeriesSeleccionadasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registro = json.loads(REGISTRO.read_text(encoding="utf-8"))

    def test_estructura_y_metadatos(self) -> None:
        registro = self.registro
        self.assertEqual(set(registro), {
            "dataset", "periodo_seleccion", "horizonte_prediccion_dias",
            "regla_seleccion", "series",
        })
        dataset = registro["dataset"]
        self.assertEqual(set(dataset), {"nombre", "contexto", "archivo", "sha256"})
        self.assertTrue(dataset["nombre"])
        self.assertIn("Corporación Favorita", dataset["contexto"])
        self.assertEqual(dataset["archivo"], "data/raw/train.csv")
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", dataset["sha256"]))
        self.assertEqual(registro["periodo_seleccion"], {
            "inicio": INICIO_SELECCION.isoformat(),
            "fin": FIN_SELECCION.isoformat(),
        })
        self.assertEqual(registro["horizonte_prediccion_dias"], HORIZONTE_DIAS)

        regla = registro["regla_seleccion"]
        self.assertEqual(set(regla), {
            "version", "granularidad", "excluir_volumen_total_cero",
            "denominador_porcentaje_ceros", "transformacion_volumen",
            "percentiles_por_cuota", "orden_grupos", "desempate",
            "maximo_series_por_tienda", "maximo_series_por_familia", "grupos",
        })
        self.assertEqual(regla["version"], "1.0")
        self.assertEqual(regla["granularidad"], ["store_nbr", "family"])
        self.assertIs(regla["excluir_volumen_total_cero"], True)
        self.assertEqual(regla["denominador_porcentaje_ceros"],
                         "filas existentes de la serie")
        self.assertEqual(regla["transformacion_volumen"], "log1p")
        self.assertEqual(regla["percentiles_por_cuota"], {
            str(cuota): [int(p * 100) for p in percentiles]
            for cuota, percentiles in PERCENTILES.items()
        })
        self.assertEqual(regla["orden_grupos"], list(CUOTAS))
        self.assertEqual(regla["desempate"],
                         ["store_nbr ascendente", "family alfabeticamente"])
        self.assertEqual(regla["maximo_series_por_tienda"], 2)
        self.assertEqual(regla["maximo_series_por_familia"], 2)
        self.assertEqual(set(regla["grupos"]), set(CUOTAS))
        for grupo, cuota in CUOTAS.items():
            contenido = regla["grupos"][grupo]
            self.assertEqual(set(contenido), {"porcentaje_ceros", "cuota"})
            self.assertEqual(contenido["cuota"], cuota)
            minimo, minimo_inclusivo, maximo, maximo_inclusivo = UMBRALES[grupo]
            self.assertEqual(contenido["porcentaje_ceros"], {
                "minimo": minimo,
                "minimo_inclusivo": minimo_inclusivo,
                "maximo": maximo,
                "maximo_inclusivo": maximo_inclusivo,
            })

    def test_diez_series_aprobadas_sin_duplicados_y_con_diversidad(self) -> None:
        series = self.registro["series"]
        self.assertIsInstance(series, list)
        self.assertEqual(len(series), 10)
        for serie in series:
            self.assertEqual(set(serie), {"grupo", "store_nbr", "family"})
            self.assertIn(serie["grupo"], CUOTAS)
            self.assertIs(type(serie["store_nbr"]), int)
            self.assertGreater(serie["store_nbr"], 0)
            self.assertIsInstance(serie["family"], str)
            self.assertTrue(serie["family"])
        claves = [(s["grupo"], s["store_nbr"], s["family"]) for s in series]
        pares = [(s["store_nbr"], s["family"]) for s in series]
        self.assertEqual(claves, APROBADAS)
        self.assertEqual(len(set(pares)), 10)
        self.assertEqual(Counter(s["grupo"] for s in series), Counter(CUOTAS))
        self.assertLessEqual(max(Counter(s["store_nbr"] for s in series).values()), 2)
        self.assertLessEqual(max(Counter(s["family"] for s in series).values()), 2)

    def test_hash_sha256_del_csv_por_bloques(self) -> None:
        huella = hashlib.sha256()
        with TRAIN.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                huella.update(bloque)
        self.assertEqual(self.registro["dataset"]["sha256"], huella.hexdigest())

    def test_coincide_con_selector_original(self) -> None:
        prefijo, _ = contar_prefijo_fechas(TRAIN)
        resultado = calcular_caracteristicas(
            leer_bloques_hasta_corte(TRAIN, prefijo)
        )
        elegidas = seleccionar_series(preparar_candidatos(resultado.series))
        generadas = [
            (fila.grupo, int(fila.store_nbr), fila.family)
            for fila in elegidas.itertuples(index=False)
        ]
        registradas = [
            (s["grupo"], s["store_nbr"], s["family"])
            for s in self.registro["series"]
        ]
        self.assertEqual(generadas, registradas)


if __name__ == "__main__":
    unittest.main()
