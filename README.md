# Predicción de demanda y simulación de reposición de inventario

Aplicación web en desarrollo como Trabajo de Integración Curricular de Tecnología Superior en Desarrollo de Software. Usa datos públicos del caso de estudio *Store Sales – Time Series Forecasting* de Corporación Favorita, Ecuador. No se conecta a sus sistemas internos ni utiliza datos privados.

## Qué funciona actualmente

- Carga y validación de un CSV de ventas, con selección de diez series a nivel tienda–familia y almacenamiento local en SQLite.
- Consulta de las series activas y visualización del histórico por tienda, familia e intervalo de fechas en una interfaz web.
- Preparación de series para modelado y una línea base semanal recursiva de 15 días como scripts independientes. La línea base aún no está integrada en la interfaz ni evaluada frente a las demás técnicas.

El trabajo continúa con la comparación temporal de técnicas de predicción y la integración de pronósticos y escenarios simulados de reposición. Los escenarios no constituyen recomendaciones operativas reales.

## Tecnologías y organización

- **Backend:** Python, FastAPI, SQLite y pruebas con `unittest`.
- **Frontend:** React, TypeScript y Vite.
- **Análisis:** Python y pandas; datos y criterios de selección descritos en `config/series_seleccionadas.json`.
- **Metodología:** CRISP-DM para organizar el análisis de datos y el modelado.

El código se organiza en `backend/` (API, validación y persistencia), `frontend/` (interfaz), `scripts/` (análisis, preparación y línea base), `config/` (selección de series) y `tests/` (pruebas de Python).

## Ejecutar la aplicación localmente

Requiere Python, Node.js y npm. Desde la raíz del repositorio, en PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

En otra terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Abra la dirección local que muestra Vite. La configuración de desarrollo dirige `/health` y `/api` al backend en `127.0.0.1:8000`.

La aplicación necesita un CSV con las columnas esperadas por la carga para mostrar series e históricos. El archivo de datos no se incluye en el repositorio; la selección se basa en el conjunto público mencionado arriba. Los scripts de análisis requieren además las dependencias de datos, como pandas, instaladas en el entorno.

## Pruebas

Desde la raíz, con el entorno de Python activo y las dependencias de prueba instaladas:

```powershell
python -m pip install -r backend/requirements-test.txt pandas
python -m unittest discover -s tests
```

Para verificar la interfaz:

```powershell
cd frontend
npm run lint
npm run build
```
