# Plataforma multiagente de análisis de fondos de pensiones

## 1. Propósito del negocio

Esta solución automatiza el análisis de fondos de pensiones (por defecto, el fondo
`ROP`) a partir de información pública de la **Superintendencia de Pensiones
(SUPEN)**. Su objetivo de negocio es apoyar la supervisión y la toma de decisiones
con evidencia cuantitativa sobre:

- evolución de activos, rentabilidad, afiliados, cuentas, beneficios y portafolios;
- liquidez y concentración de las inversiones;
- desequilibrio financiero, anomalías y volatilidad;
- diversificación y optimización del portafolio;
- flujos y fuga de afiliados;
- sostenibilidad financiera y demográfica.

El flujo calcula indicadores, recomienda variables para los análisis de machine
learning, ejecuta clustering, detección de anomalías, forecasting, análisis de
causalidad y optimización de portafolios. La configuración de indicadores, series,
columnas objetivo y umbrales se mantiene fuera del código para poder ajustar el
análisis sin modificar los algoritmos.

## 2. Flujo funcional y arquitectura

El punto de entrada conceptual es [`orquestador.py`](./orquestador.py). Para cada
mes:

1. **Scraper** consulta los endpoints de SUPEN para los siete tipos de datos.
2. Guarda los cambios como `data/<fondo>/<tipo>/<AAAA>/<MM>/<DD>/data.parquet`.
3. **Procesamiento** lee los Parquet, calcula los indicadores registrados y los
   guarda en `data/indicadores/<fondo>/<AAAA>/<MM>/<DD>/indicadores.parquet`.
4. **EDA recomendador** combina series e indicadores usando
   `configuracion/estructura_subagentes.json`, filtra nulos, baja varianza,
   outliers y multicolinealidad, y genera recomendaciones CSV en
   `eda_resultados/`.
5. **Riesgo** ejecuta en paralelo cinco subagentes y devuelve sus resultados:
   liquidez, diversificación, desequilibrio financiero, flujos de fondos y
   sostenibilidad.

### Backend distribuido

Cada componente puede ejecutarse como una API FastAPI independiente:

| Componente | Módulo ASGI | Puerto | Endpoint |
|---|---|---:|---|
| Scraper SUPEN | `scraper_agente.api_scraper:app` | 8002 | `GET /fetch/{tipo}` |
| Procesamiento | `procesamiento_agente.api_procesamiento:app` | 8001 | `GET /procesar` |
| EDA | `EDArecomend_agente.api_eda:app` | 8003 | `GET /eda_recomendar` |
| Orquestador de riesgo | `riesgo_agente.api_riesgo:app` | 8004 | `POST /procesar` |
| Liquidez | `riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app` | 8500 | `POST /liquidez` |
| Diversificación | `riesgo_agente.agenteDiversificacion.api_diversificacion:app` | 8501 | `POST /diversificacion` |
| Desequilibrio | `riesgo_agente.agenteDesequilibrioFinanciero.api_desequilibrio:app` | 8502 | `POST /desequilibrio` |
| Flujos | `riesgo_agente.agenteFlujoFondos.api_flujofondos:app` | 8503 | `POST /flujosfondos` |
| Sostenibilidad | `riesgo_agente.agenteSostenibilidad.api_sostenibilidad:app` | 8504 | `POST /sostenibilidad` |

El orquestador distribuido usa `requests` con reintentos para scraper,
procesamiento, EDA y riesgo. El orquestador de riesgo usa `httpx.AsyncClient` y
`asyncio.gather` para invocar simultáneamente a los cinco subagentes.

### Ejecución local

`AgenteOrquestadorLocal` mantiene el mismo contrato funcional, pero llama
directamente a `module_scraper.py`, `module_procesamiento.py`, `module_EDA.py` y
`module_riesgo.py`, sin levantar todos los servidores HTTP. El bloque ejecutable
de [`orquestador.py`](./orquestador.py) está configurado actualmente con este modo
local y con `FONDO = "ROP"`.

> **Nota operativa:** en `AgenteOrquestadorLocal.ejecutar` existe una asignación
> temporal que fija el cursor en `2025-04-01`. Para procesar automáticamente desde
> la última fecha disponible debe eliminarse o sustituirse esa línea.

### Frontend

El repositorio no incluye una aplicación frontend (no hay React, Vue, Angular,
templates ni archivos HTML). El consumo actual es por API REST, documentación
interactiva de FastAPI (`/docs`) y archivos generados. Un frontend futuro debería
consumir principalmente `/fetch/{tipo}`, `/procesar`, `/eda_recomendar` y el
orquestador `/procesar` de riesgo.

## 3. Estructura de carpetas

```text
multiagente/
├── orquestador.py                 # Coordinación mensual del pipeline
├── config.py                      # URLs, fondo, tipos y fecha inicial
├── module_*.py                    # Adaptadores para ejecución local
├── configuracion/
│   ├── estructura_subagentes.json # Series, indicadores y objetivos por módulo
│   ├── umbrales_activacion.json   # Reglas operador/valor para activar alertas
│   ├── cargar_json.py
│   └── devuelveObjetivo.py
├── scraper_agente/                # Cliente SUPEN, transformación y API scraper
├── procesamiento_agente/          # Indicadores y API de procesamiento
│   ├── indicadores/
│   ├── logica_procesamiento.py
│   └── registro_indicadores.py
├── EDArecomend_agente/            # Selección de variables y reportes EDA
├── riesgo_agente/                 # Orquestación y APIs de riesgo
│   ├── agenteRiesgoLiquidez/
│   ├── agenteDiversificacion/
│   ├── agenteDesequilibrioFinanciero/
│   ├── agenteFlujoFondos/
│   ├── agenteSostenibilidad/
│   └── utils/
├── data/
│   ├── ROP/<tipo>/<AAAA>/<MM>/<DD>/data.parquet # Datos de origen descargados
│   ├── indicadores/<fondo>/.../indicadores.parquet
│   ├── temporal/<subagente>/<módulo>/*.parquet   # Series preparadas para riesgo
│   ├── resultados/<subagente>/<módulo>/         # Predicciones, gráficos y tablas
│   ├── modelos/<subagente>/<módulo>/*.pkl       # Modelos serializados
│   └── insumos/                                 # Insumos adicionales
├── eda_resultados/                 # Recomendaciones EDA en CSV
├── Scrapper.bat                   # Arranque rápido del scraper (8002)
└── procesing.bat                  # Arranque rápido de procesamiento (8001)
```

Los nombres extensos de subagentes y módulos son intencionales: deben coincidir
con las claves de `estructura_subagentes.json` y con las constantes de
`riesgo_agente/utils/config.py`.

## 4. Tecnologías utilizadas

- Python 3.10+ (el código probado del repositorio contiene bytecode de Python
  3.12).
- FastAPI y Uvicorn para APIs REST.
- `requests` y `httpx` para comunicación HTTP.
- `pandas`, `numpy`, `scipy` y `python-dateutil` para ingestión y análisis.
- PyArrow/FastParquet mediante `pandas.to_parquet` y `read_parquet`.
- Scikit-learn para clustering, métricas, modelos y preprocesamiento.
- TensorFlow/Keras para el detector de anomalías LSTM.
- Statsmodels y las librerías de series temporales usadas por los módulos de
  forecasting y causalidad.
- JSON para configuración; CSV, Parquet, PNG, TXT y PKL para resultados y
  modelos.

No se encontró un `requirements.txt`, `pyproject.toml` ni otro manifiesto de
dependencias en el repositorio. Por ello, las versiones deben fijarse antes de
un despliegue reproducible.

## 5. Persistencia y base de datos

Este proyecto **no utiliza una base de datos relacional o NoSQL**: no hay
SQLAlchemy, migraciones, credenciales, archivos SQLite ni configuración de
PostgreSQL/MySQL/MongoDB. La persistencia es basada en archivos:

- datos fuente y resultados tabulares: Parquet;
- recomendaciones EDA y algunos reportes: CSV/TXT;
- modelos entrenados: PKL;
- gráficos: PNG.

Por tanto, no hay un servidor de base de datos que levantar ni scripts de
migración. La “inicialización” consiste en crear las carpetas y disponer de
permisos de escritura:

```powershell
Set-Location D:\pythonenvs\TFM\multiagente
New-Item -ItemType Directory -Force data, data\indicadores, data\temporal, `
  data\resultados, data\modelos, eda_resultados | Out-Null
```

Si se incorpora una base de datos en el futuro, deberá definirse explícitamente
el motor, el esquema, las migraciones y el adaptador de lectura/escritura; esos
elementos no forman parte de la implementación actual.

## 6. Instalación y levantamiento del entorno

### 6.1 Requisitos

- Windows 10/11.
- Python 3.10 o superior.
- Acceso de red a `https://webapps.supen.fi.cr`.
- Puertos locales libres: `8001`, `8002`, `8003`, `8004` y `8500`–`8504`.

### 6.2 Crear el entorno virtual

```powershell
Set-Location D:\pythonenvs\TFM\multiagente
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 6.3 Instalar dependencias

Instalar el conjunto base y los paquetes científicos:

```powershell
python -m pip install fastapi "uvicorn[standard]" requests httpx `
  pandas numpy scipy python-dateutil pyarrow scikit-learn `
  statsmodels matplotlib seaborn openpyxl tensorflow
```

La instalación de TensorFlow puede requerir una versión compatible con la
versión de Python y los controladores disponibles. Si un módulo concreto usa
una dependencia adicional, debe agregarse al entorno y documentarse aquí.

### 6.4 Arrancar las APIs

Ejecutar cada comando en una terminal distinta, desde la raíz del repositorio:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn scraper_agente.api_scraper:app --reload --port 8002
python -m uvicorn procesamiento_agente.api_procesamiento:app --reload --port 8001
python -m uvicorn EDArecomend_agente.api_eda:app --reload --port 8003
python -m uvicorn riesgo_agente.api_riesgo:app --reload --port 8004
```

Después, arrancar también los cinco subagentes:

```powershell
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app --reload --port 8500
python -m uvicorn riesgo_agente.agenteDiversificacion.api_diversificacion:app --reload --port 8501
python -m uvicorn riesgo_agente.agenteDesequilibrioFinanciero.api_desequilibrio:app --reload --port 8502
python -m uvicorn riesgo_agente.agenteFlujoFondos.api_flujofondos:app --reload --port 8503
python -m uvicorn riesgo_agente.agenteSostenibilidad.api_sostenibilidad:app --reload --port 8504
```

Verificar cada servicio en `http://localhost:<puerto>/docs`. Los archivos
`Scrapper.bat` y `procesing.bat` son atajos para los servicios de scraper y
procesamiento; sus rutas `cd` y `PYTHONPATH` están fijadas a
`D:\pythonenvs\TFM\multiagente`, por lo que deben ajustarse si el repositorio se
clona en otra ubicación.

### 6.5 Ejecutar el orquestador distribuido

Con las APIs levantadas:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m orquestador
```

El orquestador toma por defecto el fondo `ROP`, recorre los tipos
`comision`, `rendimiento`, `afiliado`, `cuenta`, `portafolio`, `beneficio` y
`libre_transferencia`, y procesa desde la fecha inicial configurada o desde la
última fecha encontrada en los Parquet.

### 6.6 Ejecutar el modo local

El modo local se ejecuta con:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m orquestador
```

El bloque `__main__` actual instancia `AgenteOrquestadorLocal`, por lo que no
depende de los puertos HTTP para procesamiento, EDA y riesgo. Para usar el
modo distribuido real debe habilitarse el bloque de `AgenteOrquestador`
correspondiente y tener todas las APIs disponibles.

## 7. Ejemplos de consumo

```powershell
# Consultar datos de rendimiento de ROP
Invoke-RestMethod "http://localhost:8002/fetch/rendimiento?fondo=ROP&fecha_inicio=2025-01-01&fecha_final=2025-01-31&guardar_parquet=true"

# Calcular indicadores
Invoke-RestMethod "http://localhost:8001/procesar?fondo=ROP&fecha_inicio=2025-01-01&fecha_final=2025-01-31"

# Recomendar variables
Invoke-RestMethod "http://localhost:8003/eda_recomendar?fondo=ROP&fecha=2025-01-01&fecha_final=2025-01-31"

# Ejecutar riesgo (el cuerpo se reenvía a los subagentes)
$body = @{ fondo = "ROP"; fecha = "2025-01-31"; entidad = "ninguna" } | ConvertTo-Json
Invoke-RestMethod "http://localhost:8004/procesar" -Method Post -ContentType "application/json" -Body $body
```

## 8. Configuración y operación

- Editar `config.py` para cambiar fondo, fecha inicial, tipos o URLs de las APIs
  principales.
- Editar `configuracion/estructura_subagentes.json` para declarar series,
  filtros, agregaciones, indicadores y columnas objetivo.
- Editar `configuracion/umbrales_activacion.json` para cambiar operadores y
  valores de activación de señales.
- Editar `procesamiento_agente/config.json` para parámetros específicos de
  indicadores, como ventanas históricas.
- Mantener la convención de fecha `YYYY-MM-DD` y la estructura
  `AAAA/MM/DD` de los Parquet.
- No versionar credenciales ni datos sensibles; los datos descargados y los
  modelos pueden ser voluminosos.

## 9. Limitaciones conocidas

- No existe frontend ni base de datos gestionada.
- La ejecución depende de la disponibilidad y formato de las APIs de SUPEN.
- Los subagentes requieren Parquet temporales ya preparados en
  `data/temporal/`.
- Varias rutinas exigen historial mínimo (por ejemplo, más de 24 o 36 periodos).
- Debe eliminarse el cursor fijo de abril de 2025 antes de usar el orquestador
  local como proceso incremental.
=======
#### Paso 2: Levantar API del Scraper (en una terminal)
#### uvicorn scraper_agente.api_scraper:app --reload --port 8002

#### Paso 3: Levantar API del Procesamiento (en otra terminal)
#### uvicorn procesamiento_agente.api_procesamiento:app --reload --port 8001

#### Paso 4: Levantar API del EDA (en otra terminal)
#### uvicorn EDArecomend_agente.api_eda:app --reload --port 8003

#### Paso 5: Levantar API de Riesgo (en otra terminal)
#### uvicorn riesgo_agente.api_riesgo:app --reload --port 8004

#### Paso 6: Levantar otras API  (en otra terminal)
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app --reload --port 8500

python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_diversificacion:app --reload --port 8501

python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_desequilibrio:app --reload --port 8502

python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_flujofondos:app --reload --port 8503

python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_sostenibilidad:app --reload --port 8504

#### Paso 4: Ejecutar el orquestador (en otra terminal o programado en cron)
#### python main.py ejecutar_scraper=False ejecutar_procesamiento=False ejecutar_eda=False ejecutar_riesgo=True
