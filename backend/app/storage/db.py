"""Conexión e inicialización explícita del esquema SQLite local."""

import sqlite3
from contextlib import closing
from pathlib import Path


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def conectar(ruta: str | Path) -> sqlite3.Connection:
    """Abre una conexión con las claves foráneas activadas."""
    conexion = sqlite3.connect(ruta)
    conexion.execute("PRAGMA foreign_keys = ON")
    return conexion


def inicializar_base(ruta: str | Path) -> None:
    """Crea el esquema en la ruta indicada, sin poblar tablas."""
    with closing(conectar(ruta)) as conexion:
        conexion.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
