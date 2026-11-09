# Backend - Especificación inicial para SDD

## 1. Alcance

Este documento describe el backend de la plataforma multiagente de análisis de
fondos de pensiones. Sirve como línea base para preparar un entorno de
**Software Design Description (SDD)**.

El backend coordina la ingestión de datos públicos de SUPEN, el cálculo de
indicadores, la recomendación de variables para machine learning y la ejecución
de análisis de riesgo especializados.

El punto de entrada funcional es [`orquestador.py`](./orquestador.py).

## 2. Objetivos del backend

- Descargar y versionar datos históricos por fondo, tipo y fecha.
- Detectar cambios en los datos antes de activar las etapas posteriores.
- Calcular indicadores financieros, demográficos, de liquidez, riesgo y
  sostenibilidad.
- Preparar datasets temporales para los subagentes.
- Ejecutar análisis estadístico y de machine learning.
- Exponer los componentes mediante APIs HTTP independientes.
- Permitir ejecución distribuida por servicios o ejecución local directa.
- Mantener configuración, modelos y resultados separados del código.

## 3. Arquitectura lógica

```text
                    +----------------------+
                    |  Orquestador mensual |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
       +-------------+  +-------------+  +-------------+
       | Scraper     |  | Procesamiento|  | EDA         |
       | SUPEN       |  | indicadores  |  | recomendador|
       +------+------+  +------+------+  +------+------+
              |                |                |
              +----------------+----------------+
                               v
                    +----------------------+
                    | Orquestador de riesgo|
                    +----------+-----------+
                               |
       +----------+-------------+-------------+----------+
       v          v             v             v          v
  Liquidez  Diversificación  Desequilibrio  Flujos  Sostenibilidad
```

El flujo utiliza almacenamiento basado en archivos Parquet. Los servicios no
comparten una base de datos transaccional; comparten el árbol de directorios
`data/`.

## 4. Componentes backend

### 4.1 Orquestador

Archivo: [`orquestador.py`](./orquestador.py)

Responsabilidades:

- Determinar la fecha inicial a partir de los Parquet existentes.
- Recorrer los meses hasta la fecha actual.
- Consultar los siete tipos de datos configurados.
- Detectar si hubo cambios.
- Activar procesamiento, EDA y riesgo según las banderas de ejecución.
- Reintentar llamadas HTTP fallidas.
- Ejecutar subagentes de riesgo de forma concurrente en el modo local.

Tipos de datos:

```text
comision
rendimiento
afiliado
cuenta
portafolio
beneficio
libre_transferencia
```

Configuración base: [`config.py`](./config.py).

El repositorio contiene dos variantes:

- `AgenteOrquestador`: llama servicios FastAPI por HTTP.
- `AgenteOrquestadorLocal`: llama directamente a los módulos Python.

### 4.2 Scraper SUPEN

Código: [`scraper_agente/`](./scraper_agente/)

Responsabilidades:

- Construir las URLs de SUPEN según el tipo de recurso.
- Consultar la API externa.
- Convertir respuestas JSON a `pandas.DataFrame`.
- Transformar `libre_transferencia` a matrices y resúmenes por entidad.
- Detectar cambios.
- Guardar datos en formato Parquet.

Endpoints externos configurados:

| Tipo | Recurso SUPEN |
|---|---|
| `comision` | `/Estadisticas/API/comision` |
| `rendimiento` | `/Estadisticas/API/rendimiento` |
| `afiliado` | `/Estadisticas/API/AFILIADO` |
| `cuenta` | `/estadisticas/api/cuenta` |
| `portafolio` | `/estadisticas/api/Portafolio` |
| `beneficio` | `/Estadisticas/API/BENEFICIO` |
| `libre_transferencia` | `/Estadisticas/API/LT` |

API interna:

```text
GET /fetch/{tipo}
```

Parámetros:

| Parámetro | Tipo | Obligatorio | Valor por defecto |
|---|---|---:|---|
| `tipo` | `str` | Sí, en la ruta | - |
| `fondo` | `str` | No | `ROP` |
| `fecha_inicio` | `str` | Sí | - |
| `fecha_final` | `str` | Sí | - |
| `guardar_parquet` | `bool` | No | `false` |

Respuesta conceptual:

```json
{
  "registros": 0,
  "columnas": [],
  "datos": [],
  "archivo": null,
  "cambio_detectado": false
}
```

### 4.3 Procesamiento de indicadores

Código: [`procesamiento_agente/`](./procesamiento_agente/)

La clase `AgenteProcesamiento`:

1. Obtiene las clases activas de
   [`registro_indicadores.py`](./procesamiento_agente/registro_indicadores.py).
2. Determina las fuentes Parquet requeridas.
3. Calcula el rango histórico necesario por indicador.
4. Carga datos mensuales.
5. Agrupa por entidad.
6. Ejecuta cada indicador con su frecuencia configurada.
7. Persiste resultados en `indicadores.parquet`.

La implementación registra indicadores de:

- concentración y diversificación;
- aportación y captación;
- afiliados y pensionados;
- rentabilidad y volatilidad;
- activo neto;
- liquidez;
- riesgo;
- sostenibilidad;
- flujos de fondos;
- índices Herfindahl-Hirschman;
- alpha de Jensen, beta y Sharpe.

API interna:

```text
GET /procesar
```

Parámetros:

```text
fecha_inicio: YYYY-MM-DD
fecha_final: YYYY-MM-DD
fondo: ROP
```

Respuesta conceptual:

```json
{
  "fondo": "ROP",
  "periodo": "2025-01-01 a 2025-01-31",
  "indicadores": {}
}
```

### 4.4 EDA recomendador

Código: [`EDArecomend_agente/`](./EDArecomend_agente/)

Responsabilidades:

- Leer la estructura declarativa de subagentes.
- Localizar series Parquet e indicadores dentro de un rango de fechas.
- Aplicar filtros y alias declarados en JSON.
- Combinar columnas de distintas fuentes.
- Evaluar porcentaje de nulos.
- Filtrar columnas con baja varianza.
- Detectar normalidad y filtrar outliers mediante Z-score o IQR.
- Calcular correlaciones con una columna objetivo, cuando existe.
- Reducir multicolinealidad.
- Guardar datasets base y recomendaciones en CSV.

Configuración principal:

- [`configuracion/estructura_subagentes.json`](./configuracion/estructura_subagentes.json)
- [`configuracion/umbrales_activacion.json`](./configuracion/umbrales_activacion.json)

API interna:

```text
GET /eda_recomendar
```

Parámetros:

```text
fecha: YYYY-MM-DD
fecha_final: YYYY-MM-DD
fondo: ROP
```

Respuesta conceptual:

```json
{
  "fondo": "ROP",
  "fecha": "2025-01-01",
  "indicadores": []
}
```

### 4.5 Orquestador de riesgo

Código: [`riesgo_agente/api_riesgo.py`](./riesgo_agente/api_riesgo.py)

El orquestador recibe un payload común y lo reenvía a los cinco subagentes:

```json
{
  "fondo": "ROP",
  "fecha": "2025-01-31",
  "entidad": "ninguna"
}
```

API:

```text
POST /procesar
```

La versión distribuida usa `httpx.AsyncClient` y `asyncio.gather`. La versión
local usa `module_riesgo.py` y ejecuta las funciones internas directamente.

Respuesta conceptual:

```json
{
  "liquidez": {},
  "diversificacion": {},
  "desequilibrio": {},
  "flujos": {},
  "sostenibilidad": {}
}
```

### 4.6 Subagentes de riesgo

Cada subagente recibe `fondo`, `fecha` y `entidad`. Puede recibir también
`usar_normalizado`.

| Subagente | API | Funcionalidades |
|---|---|---|
| Liquidez | `POST /liquidez` | Clustering, anomalías LSTM y forecasting de activo neto |
| Diversificación | `POST /diversificacion` | Clustering y optimización de portafolio |
| Desequilibrio | `POST /desequilibrio` | Clustering, anomalías, forecasting de activo neto y valor cuota |
| Flujos | `POST /flujofondos` | Clustering, forecasting de fuga y causalidad |
| Sostenibilidad | `POST /sostenibilidad` | Clustering, forecasting financiero y demográfico |

Los módulos cargan datasets desde `data/temporal/`, emplean los objetivos
definidos en `estructura_subagentes.json` y guardan productos en
`data/resultados/` y modelos en `data/modelos/`.

## 5. Contratos y reglas comunes

### 5.1 Fechas

- Formato de entrada: `YYYY-MM-DD`.
- Los Parquet se organizan como `AAAA/MM/DD`.
- El procesamiento general trabaja principalmente con cierres mensuales.
- Algunos indicadores se calculan con frecuencia diaria.

### 5.2 Identificadores

- `fondo`: por defecto `ROP`.
- `entidad`: entidad administradora o valor especial `ninguna`.
- `tipo`: uno de los siete recursos del scraper.
- `subagente` y `modulo`: nombres exactos definidos en el JSON.

### 5.3 Errores

- Las APIs FastAPI convierten errores de ejecución en HTTP 500.
- El orquestador principal registra errores de llamadas y realiza hasta tres
  intentos con espera exponencial.
- El orquestador de riesgo devuelve un objeto con `"error"` para un subagente
  que no pudo responder.
- No existe todavía un esquema formal de errores, trazabilidad distribuida o
  correlación de solicitudes.

## 6. Persistencia

### 6.1 Datos fuente

```text
data/<fondo>/<tipo>/<AAAA>/<MM>/<DD>/data.parquet
```

### 6.2 Indicadores

```text
data/indicadores/<fondo>/<AAAA>/<MM>/<DD>/indicadores.parquet
```

Cada registro contiene conceptualmente:

```text
entidad, fondo, fecha, indicador, valor
```

### 6.3 Datasets temporales

```text
data/temporal/<subagente>/<modulo>/<fecha>.parquet
```

Los subagentes filtran entidades con un historial mínimo antes de ejecutar los
modelos. Según el módulo, se requieren más de 24, 36 o 720 periodos.

### 6.4 Resultados y modelos

```text
data/resultados/<subagente>/<modulo>/
data/modelos/<subagente>/<modulo>/*.pkl
eda_resultados/*.csv
```

No se encontró una base de datos SQL/NoSQL, un ORM, migraciones ni un archivo
SQLite. La persistencia actual es un data lake local basado en archivos.

## 7. APIs y puertos

| Servicio | Import ASGI | Puerto |
|---|---|---:|
| Scraper | `scraper_agente.api_scraper:app` | 8002 |
| Procesamiento | `procesamiento_agente.api_procesamiento:app` | 8001 |
| EDA | `EDArecomend_agente.api_eda:app` | 8003 |
| Riesgo | `riesgo_agente.api_riesgo:app` | 8004 |
| Riesgo de liquidez | `riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app` | 8500 |
| Diversificación | `riesgo_agente.agenteDiversificacion.api_diversificacion:app` | 8501 |
| Desequilibrio financiero | `riesgo_agente.agenteDesequilibrioFinanciero.api_desequilibrio:app` | 8502 |
| Flujos de fondos | `riesgo_agente.agenteFlujoFondos.api_flujofondos:app` | 8503 |
| Sostenibilidad | `riesgo_agente.agenteSostenibilidad.api_sostenibilidad:app` | 8504 |

## 8. Arranque técnico

Desde la raíz del repositorio:

```powershell
Set-Location D:\pythonenvs\TFM\multiagente
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install fastapi "uvicorn[standard]" requests httpx `
  pandas numpy scipy python-dateutil pyarrow scikit-learn `
  statsmodels matplotlib seaborn openpyxl tensorflow
```

Configurar el path de importación:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Levantar los servicios en terminales separadas:

```powershell
python -m uvicorn scraper_agente.api_scraper:app --reload --port 8002
python -m uvicorn procesamiento_agente.api_procesamiento:app --reload --port 8001
python -m uvicorn EDArecomend_agente.api_eda:app --reload --port 8003
python -m uvicorn riesgo_agente.api_riesgo:app --reload --port 8004
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app --reload --port 8500
python -m uvicorn riesgo_agente.agenteDiversificacion.api_diversificacion:app --reload --port 8501
python -m uvicorn riesgo_agente.agenteDesequilibrioFinanciero.api_desequilibrio:app --reload --port 8502
python -m uvicorn riesgo_agente.agenteFlujoFondos.api_flujofondos:app --reload --port 8503
python -m uvicorn riesgo_agente.agenteSostenibilidad.api_sostenibilidad:app --reload --port 8504
```

Validación básica:

```powershell
Invoke-WebRequest http://localhost:8002/docs
Invoke-WebRequest http://localhost:8001/docs
Invoke-WebRequest http://localhost:8003/docs
Invoke-WebRequest http://localhost:8004/docs
```

## 9. Requisitos no funcionales para el SDD

### 9.1 Disponibilidad

- El scraper depende de la disponibilidad de SUPEN.
- Las llamadas externas deben tener timeout y reintentos configurables.
- Los servicios deben poder reiniciarse sin perder datos ya guardados.

### 9.2 Integridad

- No sobrescribir datos históricos sin una política explícita.
- Deduplicar indicadores por entidad, fondo, indicador y fecha.
- Validar columnas mínimas: `entidad`, `fondo` y `fecha`.
- Mantener consistencia entre nombres JSON, carpetas y constantes Python.

### 9.3 Rendimiento

- Ejecutar subagentes independientes en paralelo.
- Evitar cargar en memoria más historial del necesario.
- Considerar particionado y un catálogo de datos si crece el volumen.

### 9.4 Observabilidad

Para una versión SDD productiva se recomienda definir:

- logging estructurado;
- identificador de correlación por ciclo mensual;
- métricas por servicio y subagente;
- duración, filas procesadas y archivos generados;
- trazabilidad de modelos y parámetros;
- endpoint `/health` y `/ready`.

### 9.5 Seguridad

- Validar parámetros y payloads con modelos Pydantic.
- Restringir CORS y exposición de puertos.
- No registrar credenciales ni respuestas sensibles.
- Incorporar autenticación si las APIs se exponen fuera de localhost.
- Validar archivos y rutas para evitar escritura fuera de `data/`.

## 10. Decisiones de diseño actuales

| Decisión | Estado |
|---|---|
| Comunicación entre servicios | HTTP REST local |
| Framework API | FastAPI |
| Servidor ASGI | Uvicorn |
| Persistencia | Parquet/CSV/TXT/PKL |
| Orquestación | Bucle mensual en Python |
| Paralelismo de riesgo | `asyncio.gather` |
| Configuración | Python y JSON |
| Base de datos | No implementada |
| Frontend | No implementado |
| Contenerización | No implementada |

## 11. Riesgos y deuda técnica

- No existe manifiesto de dependencias con versiones fijadas.
- Algunas rutas de los scripts `.bat` están hardcodeadas a
  `D:\pythonenvs\TFM\multiagente`.
- El modo local tiene un cursor temporal fijado en `2025-04-01`.
- Existen adaptadores locales con manejo de errores que devuelven `None`.
- No todos los endpoints usan el mismo método HTTP o el mismo modelo de
  respuesta.
- No hay validación centralizada de payloads.
- No existe catálogo formal de datos ni versionado de esquemas Parquet.
- El almacenamiento local no proporciona transacciones, bloqueo ni control de
  concurrencia.
- La disponibilidad de los Parquet temporales es un prerrequisito implícito
  para los subagentes de riesgo.

## 12. Próximos artefactos SDD sugeridos

1. Diagrama de despliegue con procesos, puertos y volúmenes.
2. Especificación OpenAPI consolidada.
3. Modelos Pydantic de entrada y salida.
4. Catálogo de esquemas de cada tipo Parquet.
5. Matriz de dependencias entre indicadores, series y subagentes.
6. Política de versionado de modelos y resultados.
7. Estrategia de observabilidad y manejo de errores.
8. ADR sobre persistencia basada en archivos frente a una base de datos.
9. Plan de contenerización y configuración por ambiente.
10. Pruebas de contrato, integración y ejecución end-to-end.
