"""Pruebas sintéticas de reglas y corte temporal de la selección provisional."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analizar_periodo_seleccion import (
    calcular_caracteristicas,
    contar_prefijo_fechas,
    leer_bloques_hasta_corte,
)
from scripts.seleccionar_series import preparar_candidatos, seleccionar_series


def filas_serie(tienda: int, familia: str, ceros: int,
                volumen_unitario: float = 1.0) -> list[dict]:
    """Veinte filas existentes; el resto de días puede estar ausente."""
    return [
        {
            "date": f"2017-05-{dia + 1:02d}",
            "store_nbr": str(tienda),
            "family": familia,
            "sales": 0.0 if dia < ceros else volumen_unitario,
            "onpromotion": 1 if dia == 0 else 0,
        }
        for dia in range(20)
    ]


def candidatos_sinteticos() -> pd.DataFrame:
    especificaciones = [
        (1, "A1", 0), (1, "A2", 1),
        (1, "B1", 4), (1, "B2", 4), (1, "B3", 4),
        (2, "B4", 4), (3, "B5", 4), (4, "B6", 4),
        (5, "C1", 10), (6, "C2", 10), (7, "C3", 10),
        (8, "D1", 16), (9, "D2", 16),
    ]
    filas = [fila for tienda, familia, ceros in especificaciones
             for fila in filas_serie(tienda, familia, ceros)]
    return preparar_candidatos(calcular_caracteristicas([pd.DataFrame(filas)]).series)


class SeleccionarSeriesTests(unittest.TestCase):
    def test_corte_excluye_ventas_y_promociones_futuras(self) -> None:
        filas = [
            {"date": "2017-06-14", "store_nbr": "1", "family": "A",
             "sales": 0, "onpromotion": 1},
            {"date": "2017-06-16", "store_nbr": "1", "family": "A",
             "sales": 5, "onpromotion": 0},
            {"date": "2017-06-17", "store_nbr": "1", "family": "A",
             "sales": 999999, "onpromotion": 999},
        ]
        serie = preparar_candidatos(
            calcular_caracteristicas([pd.DataFrame(filas)]).series
        ).iloc[0]
        self.assertEqual(serie["observaciones"], 2)
        self.assertEqual(serie["volumen_total"], 5)
        self.assertEqual(serie["porcentaje_cero"], 50)
        self.assertEqual(serie["porcentaje_promocion"], 50)

        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "train.csv"
            ruta.write_text(
                "date,store_nbr,family,sales,onpromotion\n"
                "2017-06-16,1,A,5,0\n"
                "2017-06-17,1,A,NO_LEER,NO_LEER\n", encoding="utf-8"
            )
            prefijo, total = contar_prefijo_fechas(ruta, tamano_bloque=1)
            bloques = list(leer_bloques_hasta_corte(ruta, prefijo, tamano_bloque=1))
            self.assertEqual((prefijo, total), (1, 2))
            self.assertEqual(bloques[0]["sales"].tolist(), ["5"])

    def test_grupos_y_exclusion_de_series_siempre_cero(self) -> None:
        especificaciones = [(1, "A", 1), (2, "B", 7),
                            (3, "C", 14), (4, "D", 15),
                            (5, "SOLO_CERO", 20)]
        filas = [fila for tienda, familia, ceros in especificaciones
                 for fila in filas_serie(tienda, familia, ceros)]
        candidatos = preparar_candidatos(
            calcular_caracteristicas([pd.DataFrame(filas)]).series
        )
        self.assertEqual(
            dict(zip(candidatos["family"], candidatos["grupo"])),
            {"A": "A", "B": "B", "C": "C", "D": "D"},
        )
        self.assertEqual(
            dict(zip(candidatos["family"], candidatos["porcentaje_cero"])),
            {"A": 5, "B": 35, "C": 70, "D": 75},
        )

    def test_fechas_ausentes_no_son_ceros(self) -> None:
        filas = pd.DataFrame([
            {"date": "2017-06-14", "store_nbr": "1", "family": "A",
             "sales": 0, "onpromotion": 0},
            {"date": "2017-06-16", "store_nbr": "1", "family": "A",
             "sales": 4, "onpromotion": 0},
        ])
        serie = preparar_candidatos(calcular_caracteristicas([filas]).series).iloc[0]
        self.assertEqual(serie["observaciones"], 2)
        self.assertEqual(serie["porcentaje_cero"], 50)

    def test_cuotas_diversidad_y_determinismo(self) -> None:
        candidatos = candidatos_sinteticos()
        primero = seleccionar_series(candidatos)
        segundo = seleccionar_series(candidatos.sample(frac=1, random_state=37))
        self.assertEqual(len(primero), 10)
        self.assertEqual(primero["grupo"].value_counts().to_dict(),
                         {"A": 2, "B": 3, "C": 3, "D": 2})
        self.assertLessEqual(int(primero["store_nbr"].value_counts().max()), 2)
        self.assertLessEqual(int(primero["family"].value_counts().max()), 2)
        self.assertEqual(len(primero[["store_nbr", "family"]].drop_duplicates()), 10)
        self.assertEqual(
            primero[["store_nbr", "family"]].values.tolist(),
            segundo[["store_nbr", "family"]].values.tolist(),
        )
        self.assertEqual(set(primero.loc[primero["grupo"].eq("B"), "store_nbr"]),
                         {"2", "3", "4"})

    def test_faltan_candidatos_sin_relajar_cuotas(self) -> None:
        candidatos = candidatos_sinteticos()
        candidatos = candidatos.loc[~candidatos["grupo"].eq("D")]
        with self.assertRaisesRegex(ValueError, "Grupo D: 0 candidatos"):
            seleccionar_series(candidatos)

    def test_limite_de_dos_series_por_familia(self) -> None:
        candidatos = candidatos_sinteticos().copy()
        indices_a = candidatos.index[candidatos["grupo"].eq("A")]
        candidatos.loc[indices_a, "store_nbr"] = ["20", "21"]
        candidatos.loc[indices_a, "family"] = "F"
        indices_b = candidatos.index[candidatos["grupo"].eq("B")][:3]
        candidatos.loc[indices_b, "family"] = "F"
        seleccion = seleccionar_series(candidatos)
        self.assertEqual(int(seleccion["family"].eq("F").sum()), 2)
        self.assertTrue(seleccion.loc[seleccion["grupo"].eq("B"),
                                      "family"].ne("F").all())

    def test_desempate_tienda_numerica_y_familia_alfabetica(self) -> None:
        candidatos = candidatos_sinteticos()
        a = candidatos.loc[candidatos["grupo"].eq("A")].copy()
        a["store_nbr"] = ["10", "2"]
        a["family"] = ["Z", "A"]
        a["volumen_total"] = 10.0
        otros = candidatos.loc[~candidatos["grupo"].eq("A")]
        orden = seleccionar_series(pd.concat([a, otros]))
        self.assertEqual(orden.iloc[0]["store_nbr"], "2")

        a["store_nbr"] = "20"
        orden = seleccionar_series(pd.concat([a, otros]))
        self.assertEqual(orden.iloc[0]["family"], "A")


if __name__ == "__main__":
    unittest.main()
