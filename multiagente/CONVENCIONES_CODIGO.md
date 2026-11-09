# Convenciones de código y nombres

## 1. Propósito

Este documento define las convenciones para variables, funciones, clases,
módulos, APIs, archivos, configuraciones y datos del proyecto multiagente.

El estándar está orientado a Python y debe aplicarse al código nuevo. El código
existente puede contener nombres históricos o inconsistentes; no se deben hacer
renombrados masivos sin una tarea específica y una revisión de compatibilidad.

## 2. Principios generales

- Priorizar nombres explícitos sobre abreviaturas.
- Usar nombres consistentes en todo el flujo: scraper, procesamiento, EDA y
  riesgo.
- Mantener el idioma español para conceptos de negocio y el inglés para
  librerías, patrones técnicos y nombres ya establecidos por APIs externas.
- No cambiar nombres públicos sin actualizar referencias, configuración y
  documentación.
- Preferir funciones pequeñas con una responsabilidad clara.
- Evitar nombres que dependan de contexto implícito.
- No reutilizar una variable para representar conceptos distintos.
- Mantener compatibilidad con `estructura_subagentes.json` y los nombres de
  columnas Parquet.

## 3. Convenciones de Python

Se adopta una variante de PEP 8:

| Elemento | Convención | Ejemplo |
|---|---|---|
| Variables | `snake_case` | `fecha_inicio` |
| Constantes | `UPPER_SNAKE_CASE` | `DATA_ROOT` |
| Funciones | `snake_case` | `cargar_datos_parquet` |
| Métodos | `snake_case` | `calcular_indicador` |
| Clases | `PascalCase` | `AgenteProcesamiento` |
| Módulos | `snake_case` | `logica_procesamiento.py` |
| Paquetes | minúsculas, preferiblemente `snake_case` | `procesamiento_agente` |
| Excepciones | `PascalCase` terminada en `Error` | `ConfiguracionError` |
| Tipos | `PascalCase` | `ResultadoRiesgo` |
| Variables privadas | prefijo `_` | `_score_normalizado_global` |
| Variables internas no usadas | `_` | `for _ in range(3)` |

No usar espacios, tildes, guiones ni caracteres especiales en identificadores
Python. Las tildes sí pueden aparecer en textos, documentación y nombres
funcionales externos que deban coincidir con la configuración.

## 4. Variables

### 4.1 Nombres descriptivos

Preferir:

```python
fecha_inicio = "2025-01-01"
fecha_final = "2025-01-31"
datos_por_tipo = {}
columnas_objetivo = []
```

Evitar:

```python
fi = "2025-01-01"
ff = "2025-01-31"
d = {}
cols = []
```

Las abreviaturas solo se permiten cuando son ampliamente conocidas en el
dominio, por ejemplo `df` para un `DataFrame` local y de alcance corto.

### 4.2 Convenciones por dominio

| Concepto | Nombre recomendado |
|---|---|
| Fondo | `fondo` |
| Entidad | `entidad` |
| Tipo de fuente | `tipo` o `tipo_dato` |
| Fecha de inicio | `fecha_inicio` |
| Fecha final | `fecha_final` |
| Fecha de procesamiento | `fecha_proceso` |
| Fecha de cierre | `fecha_cierre` |
| DataFrame | `df`, seguido de su significado si hay varios |
| Ruta de archivo | `ruta_archivo` |
| Directorio | `directorio` |
| URL | `url` o `<servicio>_url` |
| Resultado | `resultado` o `<operacion>_resultado` |
| Configuración | `configuracion` |
| Payload HTTP | `payload` |
| Respuesta HTTP | `response` |
| Indicadores | `indicadores` |
| Modelos | `modelos` |

Cuando existan varios DataFrame, usar nombres diferenciados:

```python
df_rendimiento
df_indicadores
df_resultado
df_predicciones
```

### 4.3 Booleanos

Los booleanos deben expresar una condición o capacidad:

```python
guardar_parquet = True
ejecutar_riesgo = True
usar_normalizado = False
cambio_detectado = False
```

Preferir prefijos `es_`, `tiene_`, `puede_`, `debe_`, `usar_`, `ejecutar_` o
`guardar_`. Evitar nombres ambiguos como `flag`, `check` o `estado` cuando el
valor realmente es booleano.

### 4.4 Valores opcionales

Usar `None` para ausencia de valor y documentar el tipo:

```python
archivo_resultado: Optional[str] = None
```

No usar cadenas vacías, `0` o listas vacías como sustitutos de `None` si tienen
significados distintos.

## 5. Constantes y configuración

Las constantes globales deben usar `UPPER_SNAKE_CASE`:

```python
DATA_ROOT = "data"
FONDO = "ROP"
FECHA_INICIAL_DEFAULT = datetime(2010, 7, 1)
SCRAPER_API_URL = "http://localhost:8002/fetch/"
```

Las URLs deben seguir el patrón `<SERVICIO>_API_URL` o
`SUBAGENTE_<NOMBRE>_URL`.

Las constantes funcionales de subagentes deben conservar los nombres definidos
en [`riesgo_agente/utils/config.py`](./riesgo_agente/utils/config.py), salvo que
se realice una migración completa:

```python
AGENTE_LIQUIDEZ
LIQ_MODULO_CLUSTERING
DES_MODULO_FORECASTING
SOS_MODULO_FORECAST_DEMO
```

Las configuraciones que cambian sin modificar código deben ir en JSON. Sus
claves deben usar `snake_case`, excepto cuando deban coincidir con columnas o
nombres históricos ya publicados.

## 6. Funciones y métodos

### 6.1 Nombres

Usar verbo + objeto:

```python
cargar_datos_parquet()
guardar_indicadores()
seleccionar_columnas_objetivo()
detectar_cambios()
ejecutar_subagente_liquidez()
```

Para funciones que devuelven booleanos, preferir:

```python
existe_archivo()
es_fecha_valida()
detecto_cambios()
tiene_historial_suficiente()
```

Para conversiones, usar `convertir_`, `normalizar_`, `transformar_` o
`preparar_` según el efecto real:

```python
normalizar_entidad()
transformar_libre_transferencia()
preparar_datos_modelo()
```

### 6.2 Parámetros

Los parámetros deben reflejar el contrato de la función:

```python
def cargar_datos_parquet_rango(
    tipo: str,
    fondo: str,
    fecha_inicio: datetime,
    fecha_final: datetime,
) -> pd.DataFrame:
    ...
```

No usar `*args` o `**kwargs` para ocultar contratos estables. Usarlos solo
cuando la API realmente sea variádica y documentar su contenido.

### 6.3 Funciones asíncronas

Las corrutinas deben tener nombres iguales a sus equivalentes síncronas, sin
añadir obligatoriamente el sufijo `_async`:

```python
async def run_riesgo(...):
    ...
```

Usar `async` cuando exista espera no bloqueante o concurrencia real. Documentar
si una función llamada desde una corrutina sigue siendo síncrona y puede
bloquear.

### 6.4 Funciones públicas y privadas

- Funciones públicas: nombre sin prefijo.
- Helpers internos: prefijo `_` cuando no formen parte del contrato.
- No usar `__` salvo métodos especiales de Python.

Ejemplo:

```python
def ejecutar(self):
    return self._calcular_resultados()

def _calcular_resultados(self):
    ...
```

## 7. Clases

Las clases deben usar `PascalCase` y representar una entidad, servicio,
estrategia o resultado:

```python
AgenteOrquestador
AgenteProcesamiento
AgenteEDARecomendador
SupenDataFetcher
ForecastingLiquidez
ClusteringEvaluator
VisualizadorClusters
```

Convenciones recomendadas:

- clases coordinadoras: `Agente<NOMBRE>`;
- clientes externos: `<Sistema>Client` o `<Sistema>DataFetcher`;
- evaluadores: `<Objetivo>Evaluator`;
- detectores: `<Objetivo>Detector`;
- visualizadores: `Visualizador<Objetivo>`;
- estrategias de forecasting: `Forecasting<Objetivo>`;
- optimizadores: `Optimizador<Objetivo>`;
- errores: `<Contexto>Error`.

El constructor debe recibir dependencias y parámetros de negocio explícitos:

```python
class AgenteProcesamiento:
    def __init__(self, fondo: str, fecha_inicio: str, fecha_final: str):
        ...
```

## 8. Módulos y paquetes

Los archivos Python deben tener nombres en minúsculas con `snake_case`:

```text
logica_procesamiento.py
registro_indicadores.py
cargar_series.py
api_scraper.py
```

Convenciones por responsabilidad:

| Sufijo o nombre | Uso |
|---|---|
| `api_*.py` | Aplicación FastAPI y rutas |
| `logica_*.py` | Coordinación de reglas de negocio |
| `module_*.py` | Adaptador para ejecución local o compatibilidad |
| `utils.py` | Utilidades acotadas al paquete |
| `helpers.py` | Funciones auxiliares compartidas |
| `config.py` | Constantes y configuración Python |
| `registro_*.py` | Registro declarativo de clases o componentes |
| `indicadores_*.py` | Implementaciones de indicadores |

No crear módulos genéricos nuevos como `misc.py`, `common.py` o `helpers2.py`.
Si una utilidad es compartida por varios paquetes, ubicarla en un módulo
claramente nombrado y documentar sus consumidores.

## 9. APIs y nombres externos

### 9.1 Endpoints

Las rutas HTTP deben usar minúsculas, sustantivos o verbos funcionales
consistentes:

```text
GET  /fetch/{tipo}
GET  /procesar
GET  /eda_recomendar
POST /procesar
POST /liquidez
POST /diversificacion
POST /desequilibrio
POST /flujofondos
POST /sostenibilidad
```

Los endpoints existentes forman parte del contrato y no deben renombrarse sin
versionado o compatibilidad temporal.

### 9.2 Parámetros HTTP

Usar `snake_case`:

```text
fecha_inicio
fecha_final
guardar_parquet
```

Mantener nombres de columnas y claves JSON que ya estén publicados, aunque
contengan nombres históricos como `columnasObjetivo` o
`flujo_neto_afiliados_fondosFondos`. Una nueva configuración puede adoptar
`snake_case`, pero debe definir una migración.

### 9.3 Métodos HTTP

- `GET`: consultas sin efectos secundarios.
- `POST`: procesamiento, entrenamiento, transformación o escritura.
- `PUT`: reemplazo completo de un recurso, si se incorpora.
- `PATCH`: actualización parcial, si se incorpora.
- `DELETE`: eliminación explícita y auditada.

## 10. Nombres de datos y columnas

Las columnas nuevas deben usar `snake_case`, sin espacios ni tildes:

```text
entidad
fondo
fecha
monto_activo_neto
rentabilidad_real_anual
tasa_fuga_afiliados
indice_riesgo_liquidez
```

Columnas derivadas deben conservar el significado y la unidad. Para importes,
preferir nombres que indiquen la moneda o unidad cuando no sea obvia:

```text
monto_colones
porcentaje_rentabilidad
cantidad_afiliados
```

Columnas técnicas reservadas:

```text
__source__
```

Solo usar prefijo doble para metadatos internos que no sean parte del dominio.

### 10.1 Fechas

- Variables Python: `fecha_inicio`, `fecha_final`, `fecha_actual`.
- Columnas: `fecha`.
- Formato de intercambio: `YYYY-MM-DD`.
- Particiones de archivos: `AAAA/MM/DD`.

No mezclar `fecha_fin` y `fecha_final` en nuevos contratos. Mantener
`fecha_final` para compatibilidad con las APIs existentes.

### 10.2 Rutas

Usar `os.path.join` o `pathlib.Path`; no concatenar rutas con separadores
manuales:

```python
ruta = os.path.join(data_root, fondo, tipo, anio, mes, dia, "data.parquet")
```

Las rutas de datos deben documentar sus variables:

```text
data/<fondo>/<tipo>/<AAAA>/<MM>/<DD>/data.parquet
```

## 11. Nombres de modelos y resultados

Los archivos generados deben incluir agente, módulo, fecha u objetivo cuando
sea necesario para identificar su procedencia:

```text
indicadores.parquet
data.parquet
resultados.csv
serie_datos.csv
regresion_<fondo>_<variable>.txt
<modelo>_modelo.pkl
```

Reglas:

- usar minúsculas para nombres técnicos nuevos;
- evitar espacios en nombres de archivos nuevos;
- usar guiones bajos;
- no sobrescribir resultados históricos sin una política explícita;
- conservar nombres existentes cuando otros módulos los consuman;
- documentar el productor y consumidor de cada archivo.

## 12. Imports

Orden recomendado:

1. biblioteca estándar;
2. dependencias externas;
3. módulos del proyecto.

Ejemplo:

```python
import asyncio
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI

from riesgo_agente.utils.config import DATA_ROOT
```

Preferir imports absolutos en APIs y puntos de entrada. Los imports relativos
son aceptables dentro de un paquete cuando el módulo se ejecuta como parte del
paquete.

No usar imports comodín:

```python
from modulo import *
```

## 13. Tipado y contratos

Las funciones públicas deben tener anotaciones de parámetros y retorno:

```python
def guardar_indicadores(
    fondo: str,
    fecha: str,
    resultados: Dict[Tuple[str, str], float],
) -> None:
    ...
```

Preferir tipos concretos sobre `Any`. Usar `Any` solo cuando el contrato sea
dinámico y documentar su estructura.

Para nuevas APIs, preferir modelos Pydantic:

```python
class RiesgoRequest(BaseModel):
    fondo: str = "ROP"
    fecha: date
    entidad: str
```

No cambiar el tipo o la semántica de un parámetro público sin revisar todos sus
consumidores.

## 14. Logging y mensajes

Los mensajes deben identificar componente y operación:

```text
[ORQUESTADOR] Ejecutando ciclo
[SCRAPER] Consultando rendimiento
[EDA] Generando recomendaciones
[SUBAGENTE LIQUIDEZ] Ejecutando forecasting
```

Para código nuevo:

- preferir `logging` sobre `print`;
- usar niveles `DEBUG`, `INFO`, `WARNING`, `ERROR`;
- no incluir secretos, tokens ni payloads sensibles;
- incluir fondo, tipo, fecha y correlación cuando estén disponibles;
- no ocultar excepciones con mensajes genéricos.

## 15. Excepciones

Capturar excepciones específicas:

```python
try:
    df = pd.read_parquet(ruta)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"No se encontró el Parquet: {ruta}") from exc
```

Evitar:

```python
try:
    ...
except Exception:
    return None
```

Si una API debe traducir una excepción a HTTP, conservar la causa y devolver
una respuesta coherente con el contrato. Si un subagente falla, el resultado
debe distinguir claramente entre resultado válido y error.

## 16. Compatibilidad con el código actual

El repositorio contiene algunos nombres que no siguen completamente este
estándar, por ejemplo:

- `run_EDA` mezcla mayúsculas y minúsculas;
- `process_flujoFondos` mezcla estilos;
- `forecastDeficit` usa camelCase;
- algunos módulos contienen archivos con ` copy` en el nombre;
- existen claves JSON con camelCase.

Estos nombres deben tratarse como deuda técnica. Para código nuevo:

```python
run_eda
process_flujo_fondos
forecast_deficit
```

Un cambio de nombre requiere:

1. localizar referencias directas e indirectas;
2. actualizar imports y configuración;
3. mantener un alias temporal si existe consumo externo;
4. actualizar documentación y ejemplos;
5. ejecutar pruebas de compatibilidad;
6. registrar la migración si afecta una API.

## 17. Checklist de revisión

- [ ] Variables y funciones nuevas usan `snake_case`.
- [ ] Clases nuevas usan `PascalCase`.
- [ ] Constantes usan `UPPER_SNAKE_CASE`.
- [ ] Los booleanos tienen nombres afirmativos.
- [ ] Las funciones comienzan con un verbo apropiado.
- [ ] Los parámetros reflejan el contrato real.
- [ ] Las columnas nuevas no tienen espacios ni tildes.
- [ ] Las rutas se construyen con `os.path.join` o `pathlib`.
- [ ] Los endpoints respetan el contrato vigente.
- [ ] Los imports están ordenados y no usan `*`.
- [ ] Las funciones públicas tienen tipos.
- [ ] Los errores no se silencian.
- [ ] No se introducen abreviaturas innecesarias.
- [ ] Los renombrados tienen revisión de referencias.
- [ ] Se actualizaron documentación y ejemplos afectados.

## 18. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0 | 2026-09-03 | Versión inicial para el backend multiagente |
