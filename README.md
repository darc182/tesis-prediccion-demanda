# Predicción de demanda y análisis de reposición de inventario

**Trabajo de Integración Curricular — Tecnología Superior en Desarrollo de Software · ESFOT / Escuela Politécnica Nacional**

Aplicación web académica para explorar ventas históricas, generar pronósticos de demanda y comparar su precisión mediante datos públicos de **Corporación Favorita (Ecuador)**.

> **¿Es la primera vez que abre este proyecto?** Empiece por [«Instalación paso a paso»](#3-instalación-paso-a-paso-en-windows) y después siga [«Primera demostración guiada»](#6-primera-demostración-guiada). No es necesario conocer aprendizaje automático para realizar el recorrido básico.

> **Importante:** este repositorio contiene el **código fuente**, no una aplicación publicada en Internet ni un ejecutable para abrir con doble clic. Para usarla en su computadora es necesario iniciar dos servicios locales: FastAPI (backend) y React/Vite (frontend).

## 1. ¿Qué hace este proyecto?

El sistema permite estudiar un caso concreto: **¿cuánto podría vender una tienda de una determinada familia de productos durante los próximos 15 días?**

El recorrido general es:

```text
Archivo de ventas (train.csv)
       ↓
1. Datos: cargar y validar el archivo
       ↓
2. Histórico: elegir tienda y familia; explorar las ventas
       ↓
3. Pronósticos: generar estimaciones para 15 días
       ↓
4. Comparación: comparar las estimaciones con ventas reales
       ↓
5. Simulación: escenarios hipotéticos de reposición [interfaz en desarrollo]
```

**Ejemplo:** seleccionar la **tienda 8** y la familia **BREAD/BAKERY** (panadería), generar pronósticos y verificar cuál presentó menor error en los períodos de validación.

La aplicación es un **prototipo académico**. No está conectada a los sistemas internos de Corporación Favorita, no maneja inventarios reales de esa empresa y no genera órdenes de compra.

### Tecnologías utilizadas

| Parte | Tecnología | Para qué se utiliza |
|---|---|---|
| Interfaz | React, TypeScript y Vite | Pantallas, formularios, gráficos y tablas |
| Servidor | Python y FastAPI | Validación, consultas, experimentos y evaluación |
| Almacenamiento | SQLite | Datos seleccionados, experimentos y pronósticos |
| Procesamiento y modelos | pandas, NumPy y scikit-learn | Preparación de datos y predicción |
| Método de trabajo | CRISP-DM y desarrollo iterativo | Organización del análisis y la implementación |

## 2. Antes de empezar

Esta guía está escrita para **Windows 10/11 con PowerShell**. Los comandos se ejecutan en su computadora, no en la página web de GitHub.

Necesita instalar:

1. **Python 3.12** (el entorno de desarrollo del proyecto utiliza Python 3.12.5).
2. **Node.js compatible con Vite 8**, por ejemplo Node.js 22.12 o posterior de la rama 22.
3. **Git**, si desea descargar el proyecto mediante `git clone`. También puede descargar un ZIP desde GitHub.
4. Conexión a Internet para descargar dependencias y el conjunto de datos la primera vez.

Puede verificar las instalaciones abriendo PowerShell y escribiendo:

```powershell
py -3.12 --version
node --version
npm --version
git --version
```

Si un comando indica que no se reconoce, instale esa herramienta antes de continuar. Si descargó el código como ZIP, no necesita Git para ejecutar la aplicación.

**El archivo `train.csv` NO viene incluido en el repositorio.** Descárguelo desde el conjunto de datos público de Kaggle:

**[Store Sales – Time Series Forecasting (Kaggle): archivos de datos](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)**

Es posible que Kaggle solicite iniciar sesión o aceptar las condiciones del conjunto. Necesita específicamente **`train.csv`**. No use `test.csv` ni `sample_submission.csv` para el recorrido descrito aquí.

## 3. Instalación paso a paso en Windows

> Realice esta sección **una sola vez** por computadora o después de una instalación nueva. No salte directamente al botón «Cargar datos» sin preparar primero la base SQLite.

### Paso 1 — Descargar el proyecto

En PowerShell:

```powershell
git clone https://github.com/darc182/tesis-prediccion-demanda.git
cd tesis-prediccion-demanda
```

**Alternativa sin Git:** en la página del repositorio pulse **Code → Download ZIP**, descomprima la carpeta y ábrala. En el Explorador de Windows, entre en esa carpeta, escriba `powershell` en la barra de dirección y presione **Enter**.

A partir de aquí, todos los comandos indicados para la raíz del proyecto deben ejecutarse desde la carpeta que contiene `backend`, `frontend` y `requirements.txt`.

### Paso 2 — Crear el entorno de Python

```powershell
py -3.12 -m venv .venv
```

El comando crea la carpeta `.venv`, donde se instalarán las bibliotecas del proyecto. **No necesita activar manualmente el entorno virtual**: los comandos siguientes usan directamente su Python.

### Paso 3 — Instalar las bibliotecas de Python

Ejecute los **tres comandos**, uno detrás de otro:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-modelado.txt
```

¿Por qué son tres? Porque el repositorio separa las dependencias generales, las del servidor FastAPI y las del modelado con scikit-learn. Instalar solamente el primer archivo puede dejar funciones sin sus bibliotecas.

Espere a que **cada instalación termine sin errores** antes de ejecutar la siguiente.

### Paso 4 — Instalar la interfaz web

Desde la raíz del proyecto:

```powershell
cd frontend
npm ci
cd ..
```

`npm ci` utiliza el archivo de versiones del proyecto para instalar las dependencias de React/Vite. La carpeta `node_modules` se genera en su computadora.

### Paso 5 — Preparar la base de datos local

**Este paso es obligatorio en una descarga nueva:** GitHub no incluye el archivo SQLite local.

Primero cree la carpeta necesaria:

```powershell
New-Item -ItemType Directory -Force -Path .\data\local | Out-Null
```

Después ejecute:

```powershell
if (-not (Test-Path .\data\local\tesis.sqlite3)) {
    .\.venv\Scripts\python.exe -c "from backend.app.storage.db import inicializar_base; inicializar_base('data/local/tesis.sqlite3'); print('Base de datos creada correctamente')"
} else {
    Write-Host 'La base ya existe. No se ha modificado.'
}
```

**Resultado esperado la primera vez:** `Base de datos creada correctamente`.

Esta operación crea la estructura inicial de SQLite **sin cargar ventas todavía**. Si el archivo ya existía, el comando no lo reemplaza ni lo modifica. Una base antigua podría requerir una actualización de esquema revisada: no la borre para intentar solucionar un error.

### Paso 6 — Preparar el CSV original

Después de descargar y descomprimir los archivos de Kaggle, coloque `train.csv` en:

```text
tesis-prediccion-demanda/
└── data/
    ├── local/
    │   └── tesis.sqlite3
    └── raw/
        └── train.csv
```

Si no existe `data/raw`, créela desde la raíz del proyecto:

```powershell
New-Item -ItemType Directory -Force -Path .\data\raw | Out-Null
```

El archivo CSV también puede seleccionarse desde otra carpeta cuando utilice el formulario de carga. Guardarlo en `data/raw` solo ayuda a mantener el proyecto organizado.

**No suba `train.csv` ni `tesis.sqlite3` a GitHub.** Ambos están excluidos por `.gitignore`.

## 4. Encender la aplicación (cada vez que quiera usarla)

Necesita **dos ventanas de PowerShell abiertas al mismo tiempo**.

### Ventana 1 — Iniciar FastAPI (servidor)

Abra PowerShell **en la raíz del proyecto** y ejecute:

```powershell
$env:TESIS_DB_PATH = (Resolve-Path .\data\local\tesis.sqlite3).Path
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Mantenga abierta esta ventana mientras utilice el sistema. La variable `TESIS_DB_PATH` le dice al servidor qué archivo SQLite debe usar; debe definirse **en la misma ventana donde inicia FastAPI**.

**Comprobación rápida:** abra esta dirección en su navegador:

**http://127.0.0.1:8000/health**

Debe aparecer:

```json
{"status":"ok"}
```

Si aparece esa respuesta, FastAPI está funcionando.

### Ventana 2 — Iniciar React (interfaz)

Abra **otra** ventana de PowerShell en la raíz del proyecto:

```powershell
cd frontend
npm run dev
```

La terminal mostrará una dirección local, normalmente **http://localhost:5173/**. Ábrala en el navegador.

**Todo está listo cuando la página muestra «FastAPI conectado».**

| Dirección | Para qué sirve |
|---|---|
| http://localhost:5173/ | **Aplicación web**: aquí trabaja el usuario |
| http://127.0.0.1:8000/health | Comprueba que el servidor responde |
| http://127.0.0.1:8000/docs | Documentación técnica interactiva de la API (opcional) |

**Atención:** `localhost` es su propia computadora. Esta dirección no permite entrar a la aplicación desde otro equipo por Internet.

### Para cerrar la aplicación

En **cada una** de las dos terminales presione `Ctrl + C`. Esto detiene los servidores; **no elimina la base ni los resultados guardados**.

Al siguiente uso basta repetir la sección 4; **no debe reinstalar todo ni cargar el archivo de nuevo**, salvo que quiera utilizar otro conjunto de datos.

## 5. ¿Qué significa cada pestaña?

| Pestaña | Pregunta que ayuda a responder | Qué debe hacer el usuario |
|---|---|---|
| **Datos** | ¿Qué contiene el archivo de ventas? | Elegir `train.csv`, pulsar **Cargar y validar datos** y revisar el resumen |
| **Histórico** | ¿Cómo se ha vendido este producto en esta tienda? | Elegir tienda y familia; revisar gráfico, tabla y perfil estadístico |
| **Pronósticos** | ¿Cuánto se estima vender durante los siguientes 15 días? | Elegir fecha de corte, técnica o batería y generar pronósticos |
| **Comparación** | ¿Qué pronóstico estuvo más cerca de las ventas reales? | Elegir pronósticos ya generados y revisar MAE, WAPE y RMSE |
| **Simulación** | ¿Cómo variarían posibles escenarios de reposición? | **Interfaz no disponible todavía** en el estado documentado de este README |

> Las opciones de análisis, parámetros y comparaciones avanzadas son **opcionales** para realizar la primera demostración. No hace falta dominarlas todas para verificar que el flujo básico funciona.

## 6. Primera demostración guiada

**Objetivo:** completar un recorrido sencillo con un ejemplo real: tienda **8**, familia **BREAD/BAKERY**. No es necesario escribir SQL ni modificar código.

### A — Cargar las ventas (Datos)

1. Abra **http://localhost:5173/**.
2. Entre en **Datos**.
3. Pulse el selector **Archivo CSV** y elija `train.csv`.
4. Pulse **Cargar y validar datos**.
5. Compruebe que aparezca **«Datos cargados correctamente»** y el resumen de series y observaciones importadas.

La validación de un CSV grande puede tomar varios minutos; revise la ventana del servidor si necesita conocer si continúa procesándose. La aplicación analiza el archivo de origen, pero en SQLite guarda el subconjunto de **diez series tienda–familia** definido en `config/series_seleccionadas.json`, no todos los registros originales.

**¿Qué es el SHA-256?** Es una huella digital del contenido del CSV: sirve para distinguir si el archivo cambió. No es el ID numérico del conjunto.

### B — Explorar las ventas (Histórico)

1. Entre en **Histórico**.
2. En el selector de tienda, elija **8**.
3. En familia, elija **BREAD/BAKERY**.
4. Revise el gráfico y la tabla de ventas diarias.
5. Si desea explorar estadísticas, seleccione un corte permitido y consulte el perfil de la serie.

**Importante:** una venta igual a `0` significa que existe un registro con venta cero. Una fecha **sin registro** significa que no hay una observación registrada para ese día. La aplicación no debe confundirlos.

### C — Generar predicciones (Pronósticos)

1. Entre en **Pronósticos** y conserve la misma tienda y familia.
2. Seleccione el corte **2017-06-16**.
3. Para la primera demostración, use **«Ejecutar batería de cinco configuraciones»** en lugar de editar manualmente los parámetros avanzados.
4. Revise el resumen de ejecución. La batería intenta producir una línea base y cuatro configuraciones de modelos; cada elemento informa si fue creado, reutilizado, falló o no se ejecutó.
5. Consulte en el historial los pronósticos generados, sus tablas y sus gráficos de 15 días.

La batería **no adivina automáticamente el mejor modelo**. Solo genera un conjunto acotado de alternativas para poder evaluarlas después. Si algún elemento falla, consulte su estado: no dé por hecho que se generaron los cinco.

### D — Comparar resultados (Comparación)

1. Abra **Comparación**.
2. Seleccione la misma tienda **8** y familia **BREAD/BAKERY**.
3. Mantenga la modalidad **«Por corte — 15 días»**.
4. Elija el corte **2017-06-16**.
5. Seleccione **dos o tres pronósticos distintos** de ese mismo corte y serie.
6. Pulse **«Comparar evaluaciones»**.
7. Mire primero **WAPE** y **MAE**: en los mismos días evaluados, valores menores indican menor error.

Esta es la demostración básica recomendada. Si los pronósticos fueron generados para el corte indicado, no necesita usar la consolidación de tres cortes.

### E — Interpretar los errores sin complicaciones

| Indicador | Explicación sencilla |
|---|---|
| **MAE** | Error absoluto promedio entre la venta pronosticada y la real |
| **WAPE** | Error absoluto acumulado, expresado como porcentaje del total de ventas reales |
| **RMSE** | Medida de error que penaliza más los fallos grandes |
| **Cobertura 15/15** | Se encontraron ventas reales para las 15 fechas evaluadas |
| **Cobertura parcial** | Faltan registros reales de uno o más días; no deben inventarse ventas cero |

**Ejemplo:** si un modelo tiene un WAPE de `9 %` y otro `12 %` sobre las **mismas observaciones**, el primero presenta menor error porcentual acumulado. No significa que tenga una «exactitud del 91 %» garantizada.

## 7. Comparación avanzada: ¿qué son los tres cortes y las 45 observaciones?

Esta modalidad **no es necesaria para empezar**. Está destinada a quien quiera evaluar la misma configuración predictiva en varios períodos.

| Fecha de corte | Días cuyas ventas se predicen | Máximo de observaciones reales |
|---|---|---:|
| **2017-06-16** | 17 de junio al 1 de julio de 2017 | 15 |
| **2017-07-01** | 2 al 16 de julio de 2017 | 15 |
| **2017-07-16** | 17 al 31 de julio de 2017 | 15 |
| **Total** | Tres períodos consecutivos | **45** |

**¿De dónde sale 45?** De `15 + 15 + 15`. Son **hasta 45 ventas diarias reales de una misma tienda y familia, por configuración evaluada**. No son 45 tiendas ni 45 modelos. Si faltan registros, la cobertura será inferior a 45/45.

**¿Qué diferencia hay entre las dos modalidades?**

- **Por corte:** compara pronósticos correspondientes a **un período de 15 días**.
- **Consolidada:** reúne los errores de **tres períodos de 15 días** para evaluar el desempeño a lo largo de más fechas.

**Para utilizar Consolidada:**

1. Primero, en **Pronósticos**, genere pronósticos para **los tres cortes**. Puede ejecutar la batería en cada fecha.
2. Regrese a **Comparación → Consolidada** y pulse **Actualizar listado de consolidación**.
3. Elija **dos grupos** para comenzar. Cada grupo representa **una misma configuración predictiva** aplicada en los tres cortes.
4. Rellene **tres pronósticos por grupo**: uno del 16 de junio, otro del 1 de julio y otro del 16 de julio. Se necesitan **seis IDs distintos** en total.
5. Pulse **Consolidar evaluaciones**.

**Si dice «no comparten configuración predictiva»:** dentro de un grupo, los tres pronósticos deben coincidir en la técnica, el grupo y orden de variables, los parámetros efectivos, el contrato y las reglas; el backend también valida versiones compatibles. **Compartir solamente el nombre Random Forest o HistGradientBoosting no es suficiente.** Una forma sencilla de disminuir errores es escoger la **misma alternativa de batería** en cada fecha: por ejemplo, Random Forest clásico para el primer grupo e HistGradientBoosting ampliado para el segundo. Si aun así hay incompatibilidad, inspeccione la configuración de los tres pronósticos y las versiones declaradas.

**Si dice «0 candidatos elegibles» en una fecha:** todavía no hay pronósticos de validación elegibles para ese corte; vuelva a Pronósticos y genere los que falten.

La consolidación **no entrena, no crea pronósticos y no selecciona automáticamente un ganador**. Calcula las métricas con los pronósticos ya disponibles y las ventas observadas.

## 8. Problemas frecuentes y cómo resolverlos

| Lo que aparece | Qué revisar o hacer |
|---|---|
| **«No se pudo conectar con FastAPI»** | Compruebe que la **Ventana 1** siga abierta y que `http://127.0.0.1:8000/health` muestre `{"status":"ok"}`. |
| **La página `localhost:5173` no abre** | Compruebe que ejecutó `npm run dev` dentro de `frontend` y abra **la dirección exacta que muestra Vite**. Si el puerto 5173 está ocupado, Vite puede elegir otro. |
| **`npm` o `node` no se reconoce** | Instale Node.js, cierre y vuelva a abrir PowerShell. |
| **`py -3.12` no se reconoce** | Instale Python 3.12 con su lanzador para Windows o compruebe qué intérprete tiene disponible. |
| **`No module named fastapi` / `sklearn`** | No se instalaron todas las bibliotecas en `.venv`: repita los tres comandos del **Paso 3** utilizando el Python del entorno. |
| **HTTP 503: SQLite no disponible o esquema incompleto** | Revise que exista `data/local/tesis.sqlite3`, que `TESIS_DB_PATH` apunte allí **en la ventana de FastAPI**, y que la base tenga el esquema actual. **No borre una base existente.** |
| **HTTP 404 al consultar un perfil global** | Puede faltar un perfil para un conjunto importado con una versión anterior de la aplicación. Un conjunto antiguo no adquiere automáticamente todos los metadatos nuevos. |
| **CSV rechazado** | Use el `train.csv` original, compruebe que no esté comprimido y que conserve las columnas obligatorias: `date`, `store_nbr`, `family`, `sales` y `onpromotion`. |
| **No aparecen tiendas o familias** | Primero cargue correctamente el CSV, luego vuelva a consultar las series. Solo aparecen las diez combinaciones configuradas para el estudio. |
| **No aparecen pronósticos para comparar** | Genere primero pronósticos en la pestaña **Pronósticos**, para la misma tienda, familia y fecha de corte. |
| **Consolidada muestra grupos incompletos** | Hace falta un pronóstico elegible por cada uno de los tres cortes en cada grupo. |
| **No veo funcionando Simulación** | En esta versión de la interfaz, **Simulación está pendiente**. No indica un problema de instalación. |

### Si recibe HTTP 503 y sospecha que su SQLite es antigua

Puede probar con una **base nueva de demostración, sin borrar la existente**. Primero detenga FastAPI con `Ctrl + C`. Luego, desde la raíz del proyecto, ejecute:

```powershell
New-Item -ItemType Directory -Force -Path .\data\demo | Out-Null
if (-not (Test-Path .\data\demo\tesis_revision.sqlite3)) {
    .\.venv\Scripts\python.exe -c "from backend.app.storage.db import inicializar_base; inicializar_base('data/demo/tesis_revision.sqlite3'); print('Base de revision creada')"
} else {
    Write-Host 'Ya existe una base de revision. No se ha reemplazado.'
}
$env:TESIS_DB_PATH = (Resolve-Path .\data\demo\tesis_revision.sqlite3).Path
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Vuelva a **Datos** y cargue `train.csv`. Es un entorno **independiente**, por lo que los experimentos de otra SQLite no aparecerán aquí. Si esta base de revisión también es antigua o está incompleta, no la sobrescriba: haga una nueva con otro nombre o solicite una revisión técnica del esquema.

### Diagnóstico seguro de la base SQLite

Desde la **raíz del proyecto** puede comprobar las tablas de su base de forma **solo lectura**:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('file:data/local/tesis.sqlite3?mode=ro', uri=True); print('Tablas:', ', '.join(f[1] for f in c.execute('PRAGMA table_list') if f[2] == 'table')); c.close()"
```

El esquema inicial actual incluye, entre otras, las tablas `conjunto_datos`, `serie_temporal`, `venta_diaria`, `experimento`, `pronostico`, `evaluacion` y `perfil_global_conjunto`.

**Advertencia:** no ejecute scripts de cambio de esquema, restauraciones, borrados o migraciones sobre una base con resultados importantes sin una copia de seguridad previa y sin comprobar la compatibilidad. Modificar `schema.sql` **no actualiza automáticamente** un archivo SQLite antiguo.

## 9. Estructura del proyecto

```text
tesis-prediccion-demanda/
├── backend/
│   ├── app/
│   │   ├── api/                 # Endpoints FastAPI
│   │   ├── services/            # Reglas de negocio, análisis y evaluación
│   │   ├── storage/             # SQLite, esquema y preparación de tablas
│   │   └── main.py              # Inicio de FastAPI
│   └── requirements.txt         # Dependencias del servidor
├── frontend/
│   ├── src/                     # Componentes React y estilos
│   ├── package.json
│   └── package-lock.json
├── scripts/                     # Procesamiento, variables y modelos
├── config/
│   └── series_seleccionadas.json
├── tests/                       # Pruebas automatizadas
├── data/                        # Archivos locales: pueden no existir al clonar
│   ├── raw/                     # train.csv (no incluido en Git)
│   └── local/                   # tesis.sqlite3 (no incluida en Git)
├── requirements.txt             # Dependencias generales de Python
└── requirements-modelado.txt    # Dependencias del modelado
```

## 10. Alcance y estado actual

**Estado documentado: 19 de noviembre de 2026.** El proyecto está en desarrollo; la información de esta sección debe actualizarse a medida que se publiquen nuevos módulos.

| Funcionalidad | Estado en la interfaz actual |
|---|---|
| Carga, validación CSV y perfil global | Disponible |
| Selección de series e histórico de ventas | Disponible |
| Perfil estadístico por serie | Disponible |
| Pronósticos individuales y batería de cinco configuraciones | Disponible |
| Consulta, historial y visualización de pronósticos | Disponible |
| Comparación por corte y consolidación de tres cortes | Disponible |
| Simulación de reposición y consulta visual de escenarios | **En desarrollo; pestaña todavía no disponible** |

Se están implementando componentes técnicos de simulación, pero **tener código de backend no implica que la función ya pueda utilizarse desde React**. El recorrido público de este README termina por ahora en Comparación.

### Restricciones importantes del estudio

- Se trabajan **diez series representativas** identificadas por tienda y familia; no se pretende entrenar con todas las combinaciones de la competencia.
- El horizonte de cada pronóstico es de **15 días naturales**.
- Las técnicas centrales son **línea base estacional semanal, Random Forest e HistGradientBoosting**.
- La comparación utiliza **MAE, WAPE y RMSE**, y respeta la cantidad de ventas reales disponibles.
- Los tres cortes de validación son **16 de junio, 1 de julio y 16 de julio de 2017**. El período del **1 al 15 de agosto de 2017** se mantiene reservado para la evaluación final, no para escoger configuraciones.
- La simulación de inventario, cuando se integre, utilizará **parámetros hipotéticos**; no representa existencias reales ni recomendaciones comerciales oficiales.

## 11. Notas para revisores y docentes

**Para comprobar que el proyecto se ejecuta, no es necesario explorar todos los botones.** El camino mínimo de revisión es:

1. Verificar `http://127.0.0.1:8000/health`.
2. Abrir `http://localhost:5173/` y comprobar **FastAPI conectado**.
3. Cargar `train.csv` en **Datos**.
4. Elegir **tienda 8 → BREAD/BAKERY** en **Histórico**.
5. Ejecutar la batería para **2017-06-16** en **Pronósticos**.
6. Comparar dos pronósticos del mismo corte en **Comparación → Por corte**.

Con esos seis pasos ya se verifica el flujo principal actualmente disponible. La modalidad de 45 observaciones es un análisis adicional, **no un requisito para completar esta demostración**.

**Los resultados de un modelo no son cifras de inventario disponible:** son estimaciones de demanda. La evaluación confronta esas estimaciones con observaciones históricas, sin utilizar información futura como dato conocido al momento de producir cada pronóstico.

### Referencias

- [Repositorio del proyecto](https://github.com/darc182/tesis-prediccion-demanda).
- [Conjunto de datos público de Favorita en Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
- [Documentación de FastAPI](https://fastapi.tiangolo.com/).
- [Documentación de Vite](https://vite.dev/guide/).

---

**Autor del desarrollo:** Darwin Cachimil.  
**Naturaleza:** trabajo académico, sin vinculación operativa con Corporación Favorita.
