-- Las fechas y marcas de tiempo se almacenan como texto ISO 8601.
BEGIN;

CREATE TABLE IF NOT EXISTS conjunto_datos (
    id INTEGER PRIMARY KEY,
    nombre_archivo TEXT NOT NULL CHECK (length(trim(nombre_archivo)) > 0),
    hash TEXT NOT NULL CHECK (length(trim(hash)) > 0),
    fecha_carga TEXT NOT NULL,
    activo INTEGER NOT NULL DEFAULT 1 CHECK (activo IN (0, 1))
);

CREATE TABLE IF NOT EXISTS serie_temporal (
    id INTEGER PRIMARY KEY,
    conjunto_id INTEGER NOT NULL REFERENCES conjunto_datos(id),
    store_nbr INTEGER NOT NULL CHECK (store_nbr > 0),
    family TEXT NOT NULL CHECK (length(trim(family)) > 0),
    criterio_seleccion TEXT NOT NULL CHECK (length(trim(criterio_seleccion)) > 0),
    UNIQUE (conjunto_id, store_nbr, family)
);

CREATE TABLE IF NOT EXISTS venta_diaria (
    id INTEGER PRIMARY KEY,
    serie_id INTEGER NOT NULL REFERENCES serie_temporal(id),
    fecha TEXT NOT NULL,
    sales REAL NOT NULL,
    onpromotion INTEGER NOT NULL CHECK (onpromotion >= 0),
    UNIQUE (serie_id, fecha)
);

CREATE TABLE IF NOT EXISTS experimento (
    id INTEGER PRIMARY KEY,
    serie_id INTEGER NOT NULL REFERENCES serie_temporal(id),
    tecnica TEXT NOT NULL CHECK (length(trim(tecnica)) > 0),
    configuracion_json TEXT NOT NULL CHECK (json_valid(configuracion_json)),
    inicio_entrenamiento TEXT NOT NULL,
    fin_entrenamiento TEXT NOT NULL,
    ruta_modelo TEXT,
    CHECK (inicio_entrenamiento <= fin_entrenamiento)
);

CREATE TABLE IF NOT EXISTS pronostico (
    id INTEGER PRIMARY KEY,
    experimento_id INTEGER NOT NULL REFERENCES experimento(id),
    fecha_corte TEXT NOT NULL,
    tipo TEXT NOT NULL CHECK (tipo IN ('validacion', 'prueba', 'futuro')),
    horizonte INTEGER NOT NULL DEFAULT 15 CHECK (horizonte = 15),
    fecha_generacion TEXT NOT NULL,
    duracion_segundos REAL CHECK (duracion_segundos >= 0)
);

CREATE TABLE IF NOT EXISTS detalle_pronostico (
    pronostico_id INTEGER NOT NULL REFERENCES pronostico(id),
    paso INTEGER NOT NULL CHECK (paso BETWEEN 1 AND 15),
    fecha_objetivo TEXT NOT NULL,
    venta_estimada REAL NOT NULL,
    PRIMARY KEY (pronostico_id, paso),
    UNIQUE (pronostico_id, fecha_objetivo)
);

CREATE TABLE IF NOT EXISTS evaluacion (
    pronostico_id INTEGER PRIMARY KEY REFERENCES pronostico(id),
    mae REAL NOT NULL CHECK (mae >= 0),
    wape REAL CHECK (wape >= 0),
    rmse REAL CHECK (rmse >= 0)
);

CREATE TABLE IF NOT EXISTS simulacion (
    id INTEGER PRIMARY KEY,
    pronostico_id INTEGER NOT NULL REFERENCES pronostico(id),
    inventario_disponible REAL NOT NULL CHECK (inventario_disponible >= 0),
    tiempo_abastecimiento INTEGER NOT NULL CHECK (
        typeof(tiempo_abastecimiento) = 'integer'
        AND tiempo_abastecimiento BETWEEN 0 AND 15
    ),
    reserva REAL NOT NULL CHECK (reserva >= 0),
    demanda_considerada REAL NOT NULL CHECK (demanda_considerada >= 0),
    reposicion REAL NOT NULL CHECK (reposicion >= 0),
    regla_aplicada TEXT NOT NULL CHECK (length(trim(regla_aplicada)) > 0),
    fecha TEXT NOT NULL
);

COMMIT;
