# Estándar de documentación Markdown

## 1. Propósito

Este estándar define cómo documentar el proyecto multiagente de análisis de
fondos de pensiones. Su objetivo es que la documentación sea consistente,
trazable y útil para desarrollo, operación, arquitectura y preparación del SDD.

Aplica a:

- documentación general del proyecto;
- backend y APIs;
- agentes y subagentes;
- configuración;
- datos, modelos y resultados;
- decisiones arquitectónicas;
- instalación, operación y troubleshooting.

## 2. Principios

1. **Una fuente de verdad:** documentar el comportamiento implementado, no el
   comportamiento deseado.
2. **Trazabilidad:** cada componente debe enlazar su código, configuración,
   entradas, salidas y consumidores.
3. **Reproducibilidad:** los pasos deben poder ejecutarse desde la raíz del
   repositorio.
4. **Precisión:** separar hechos actuales, decisiones pendientes y deuda técnica.
5. **Mantenibilidad:** preferir tablas y secciones estables frente a texto
   duplicado.
6. **Idioma:** redactar en español; conservar en inglés nombres de librerías,
   clases, funciones, endpoints y variables.
7. **Seguridad:** nunca incluir secretos, tokens, credenciales ni datos
   personales reales.

## 3. Ubicación y nombres de archivos

La documentación debe ubicarse en la raíz o en un directorio `docs/`.

| Documento | Propósito |
|---|---|
| `README.md` | Visión general, propósito, instalación rápida y navegación |
| `backend.md` | Arquitectura, contratos y operación del backend |
| `frontend.md` | Arquitectura y operación del frontend, si existe |
| `ESTANDAR_DOCUMENTACION.md` | Reglas de este estándar |
| `CONTRIBUTING.md` | Flujo de contribución y revisión |
| `SECURITY.md` | Reporte y tratamiento de vulnerabilidades |
| `docs/adr/ADR-NNN-titulo.md` | Decisiones arquitectónicas |
| `docs/agentes/<agente>.md` | Documentación detallada de un agente |
| `docs/apis/<servicio>.md` | Contrato y operación de una API |
| `docs/datos/<tipo>.md` | Esquema y ciclo de vida de datos |

Reglas de nombres:

- usar MAYÚSCULAS solo para documentos normativos de raíz;
- usar `snake_case` o `kebab-case` de forma consistente dentro de `docs/`;
- incluir extensiones `.md`;
- no crear nombres ambiguos como `notas.md`, `temporal.md` o `final.md`;
- enlazar documentos con rutas relativas;
- no duplicar una sección completa en varios documentos: enlazarla.

## 4. Metadatos mínimos

Todo documento técnico mantenido en `docs/` debe comenzar con un bloque de
metadatos:

```markdown
---
titulo: "Documentación del agente de riesgo de liquidez"
version: "1.0"
estado: "vigente"
responsable: "Equipo de arquitectura"
ultima_actualizacion: "AAAA-MM-DD"
alcance: "Backend"
---
```

Valores permitidos para `estado`:

- `borrador`: propuesta aún no validada;
- `vigente`: descripción aprobada del sistema actual;
- `deprecated`: válida históricamente, no usar para nuevas decisiones;
- `reemplazado`: sustituida por otro documento.

Los documentos de raíz (`README.md`, `backend.md`) pueden usar metadatos si el
repositorio adopta front matter de manera uniforme; no se debe mezclar
convenciones arbitrariamente.

## 5. Estructura común de un documento

Salvo que exista una razón explícita, usar este orden:

```markdown
# Título

## 1. Propósito
## 2. Alcance
## 3. Contexto
## 4. Arquitectura o diseño
## 5. Interfaces
## 6. Datos y persistencia
## 7. Configuración
## 8. Instalación y operación
## 9. Errores y observabilidad
## 10. Seguridad
## 11. Limitaciones
## 12. Validación
## 13. Referencias
## 14. Historial de cambios
```

No es obligatorio incluir secciones irrelevantes, pero no se deben omitir
`Propósito`, `Alcance`, `Limitaciones` y `Referencias` en documentos técnicos.

## 6. Convenciones Markdown

- usar un único `#` para el título;
- no saltar niveles de encabezado (`##` debe preceder a `###`);
- mantener líneas de texto razonablemente cortas;
- usar listas para enumeraciones y tablas para información comparativa;
- usar bloques de código con lenguaje declarado;
- usar `text` para diagramas ASCII;
- usar `powershell` para comandos Windows del proyecto;
- usar `json`, `python` o `http` según corresponda;
- incluir unidades, formatos y valores por defecto;
- usar enlaces a archivos y símbolos reales;
- no usar capturas de pantalla como sustituto de contratos o diagramas;
- no incluir HTML salvo que Markdown no sea suficiente;
- escapar caracteres especiales cuando afecten el formato.

Ejemplo de enlace correcto:

```markdown
La configuración se define en
[`config.py`](../config.py) y los umbrales en
[`umbrales_activacion.json`](../configuracion/umbrales_activacion.json).
```

## 7. Estándar para README

El `README.md` debe ser la puerta de entrada y no una especificación exhaustiva.
Debe contener, como mínimo:

1. nombre y descripción breve;
2. propósito de negocio;
3. capacidades principales;
4. diagrama o flujo resumido;
5. estructura de carpetas;
6. requisitos;
7. instalación rápida;
8. ejecución rápida;
9. enlaces a documentación detallada;
10. limitaciones conocidas;
11. licencia o estado del proyecto, si aplica.

El README debe responder rápidamente:

- ¿qué problema resuelve?;
- ¿cómo se ejecuta?;
- ¿dónde están los datos?;
- ¿qué servicios existen?;
- ¿qué documento debo leer después?

## 8. Estándar para backend

`backend.md` debe incluir:

### 8.1 Contexto y alcance

- objetivo de negocio;
- límites del backend;
- actores y consumidores;
- relación con el orquestador.

### 8.2 Arquitectura

- componentes;
- comunicación síncrona o asíncrona;
- dependencias externas;
- diagrama de flujo;
- modo distribuido y modo local, si ambos existen.

### 8.3 Inventario de servicios

Cada servicio debe documentarse con esta tabla:

| Campo | Descripción |
|---|---|
| Nombre | Nombre funcional |
| Módulo ASGI | Ruta `paquete.modulo:app` |
| Puerto | Puerto local o de despliegue |
| Responsabilidad | Qué procesa |
| Dependencias | Servicios, archivos o APIs externas |
| Estado | Vigente, experimental o deprecated |

### 8.4 Persistencia

Documentar:

- motor o mecanismo de almacenamiento;
- rutas;
- formato;
- particionamiento;
- esquema mínimo;
- política de deduplicación;
- retención y respaldo;
- concurrencia y limitaciones.

En este proyecto, la persistencia actual es por archivos Parquet, CSV, TXT y
PKL. No se debe documentar una base de datos SQL/NoSQL como existente hasta que
haya código, configuración y migraciones verificables.

### 8.5 Requisitos no funcionales

Incluir objetivos o estado de:

- disponibilidad;
- rendimiento;
- escalabilidad;
- seguridad;
- observabilidad;
- recuperación;
- integridad de datos.

Cuando aún no existan métricas, escribir `Por definir` y registrar la decisión
pendiente, no inventar valores.

## 9. Estándar para APIs

Cada API debe documentar un endpoint por sección:

```markdown
### `POST /procesar`

**Propósito:** Ejecuta el procesamiento de riesgo.

**Parámetros de ruta:** Ninguno.

**Query parameters:**

| Nombre | Tipo | Obligatorio | Default | Descripción |
|---|---|---:|---|---|
| `fondo` | `string` | No | `ROP` | Fondo a procesar |

**Body:**

```json
{
  "fondo": "ROP",
  "fecha": "2025-01-31",
  "entidad": "ninguna"
}
```

**Respuesta 200:**

```json
{
  "liquidez": {},
  "diversificacion": {}
}
```

**Errores:** `400`, `404`, `422`, `500`, según la implementación real.

**Dependencias:** Servicios o archivos consumidos.
```

Reglas:

- indicar método HTTP y ruta exacta;
- distinguir path, query y body;
- documentar tipo, obligatoriedad y valor por defecto;
- usar ejemplos válidos y mínimos;
- documentar respuestas exitosas y errores observados;
- indicar si la operación es idempotente;
- mantener el contrato sincronizado con FastAPI/OpenAPI.

## 10. Estándar para agentes y subagentes

Cada agente debe tener una ficha con:

| Campo | Contenido |
|---|---|
| Nombre | Nombre funcional |
| Código | Carpeta o módulo principal |
| Entrada | Payload, Parquet o configuración |
| Salida | Resultado, archivo o modelo |
| Objetivo de negocio | Decisión que apoya |
| Algoritmos | Clustering, forecasting, anomalías, etc. |
| Indicadores | Lista o enlace al registro |
| Historial mínimo | Periodos requeridos |
| Configuración | JSON o constantes usadas |
| API | Endpoint y puerto, si existe |
| Errores conocidos | Condiciones de fallo |
| Validación | Pruebas o comandos ejecutados |

Debe explicarse el flujo interno en el orden real de ejecución:

```text
cargar datos -> validar -> filtrar historial -> transformar
-> entrenar/calcular -> guardar resultados -> devolver respuesta
```

No describir como concurrentas operaciones que el código ejecuta
secuencialmente. La concurrencia debe justificarse con la implementación
(`asyncio.gather`, tareas o procesos).

## 11. Estándar para configuración

Toda configuración debe documentarse con:

- archivo y ruta;
- propietario funcional;
- estructura;
- valores permitidos;
- valor por defecto;
- efecto del cambio;
- necesidad de reiniciar servicios;
- ejemplo mínimo.

Para `estructura_subagentes.json`, documentar como mínimo:

```text
subagente -> modulo -> series -> columnas -> filtros
          -> indicadores -> columnasObjetivo
```

Para umbrales:

```text
indicador -> operador -> valor -> significado de activación
```

No duplicar en Markdown listas que cambian frecuentemente si pueden enlazarse
directamente al JSON.

## 12. Estándar para datos, modelos y resultados

Cada conjunto de datos debe documentar:

| Campo | Descripción |
|---|---|
| Ruta patrón | Ejemplo con variables |
| Formato | Parquet, CSV, TXT, PKL o PNG |
| Particiones | Fondo, tipo, año, mes, día, etc. |
| Columnas clave | `entidad`, `fondo`, `fecha`, etc. |
| Fuente | SUPEN, indicador calculado o modelo |
| Productor | Agente o módulo |
| Consumidor | Agente o módulo |
| Calidad mínima | Columnas, tipos e historial |
| Retención | Permanente, temporal o por definir |

Los patrones deben usar variables descriptivas:

```text
data/<fondo>/<tipo>/<AAAA>/<MM>/<DD>/data.parquet
data/indicadores/<fondo>/<AAAA>/<MM>/<DD>/indicadores.parquet
data/temporal/<subagente>/<modulo>/<fecha>.parquet
```

Para modelos PKL documentar algoritmo, versión de Python, versión de librerías,
variables de entrada, fecha de entrenamiento y compatibilidad.

## 13. Instalación, operación y troubleshooting

Los comandos deben:

- comenzar desde la raíz del repositorio;
- indicar el sistema operativo;
- incluir creación/activación del entorno;
- indicar variables de entorno;
- declarar puertos;
- separar terminales para servicios independientes;
- incluir una validación posterior;
- indicar cómo detener o reiniciar el servicio.

Formato recomendado:

```markdown
### Paso N: descripción

```powershell
comando
```

**Resultado esperado:** descripción observable.
```

Los problemas deben usar esta estructura:

| Síntoma | Causa probable | Diagnóstico | Solución |
|---|---|---|---|
| API no responde | Servicio detenido | Consultar `/docs` | Levantar Uvicorn |

No documentar como solución cambiar permisos, borrar datos o desactivar
seguridad sin explicar el impacto.

## 14. Decisiones arquitectónicas (ADR)

Cada ADR debe seguir esta plantilla:

```markdown
# ADR-NNN: Título de la decisión

## Estado
Propuesta | Aceptada | Rechazada | Reemplazada

## Fecha
AAAA-MM-DD

## Contexto
Problema y restricciones.

## Decisión
Alternativa seleccionada.

## Alternativas consideradas
1. Alternativa A: ventajas y desventajas.
2. Alternativa B: ventajas y desventajas.

## Consecuencias
Impactos positivos, negativos y deuda técnica.

## Validación
Pruebas, métricas o evidencia.

## Referencias
Enlaces a código y documentos.
```

Las decisiones sobre persistencia Parquet, ejecución local/distribuida,
contratos REST y contenerización deben registrarse como ADR cuando se
formalice el SDD.

## 15. Trazabilidad y referencias

Toda afirmación técnica importante debe poder rastrearse a uno de estos
elementos:

- archivo o símbolo;
- configuración;
- endpoint;
- prueba;
- resultado generado;
- ADR;
- requisito.

Usar una matriz cuando exista relación entre requisitos y componentes:

| Requisito | Componente | Implementación | Validación |
|---|---|---|---|
| Descargar datos SUPEN | Scraper | `scraper_agente/` | Prueba de API |

## 16. Control de cambios

Al final de documentos de diseño relevantes incluir:

```markdown
## Historial de cambios

| Versión | Fecha | Autor | Cambio |
|---|---|---|---|
| 1.0 | AAAA-MM-DD | Equipo | Versión inicial |
```

Reglas:

- incrementar versión ante cambios de contrato o arquitectura;
- actualizar `ultima_actualizacion`;
- marcar documentos obsoletos como `deprecated`;
- enlazar el documento que los reemplaza;
- revisar referencias a rutas y puertos después de mover código.

## 17. Checklist de revisión

Antes de aprobar un documento Markdown:

- [ ] Está escrito en español y conserva nombres técnicos reales.
- [ ] Tiene propósito y alcance.
- [ ] Distingue estado actual de propuesta futura.
- [ ] Enlaza archivos y configuraciones existentes.
- [ ] Los comandos parten de la raíz del repositorio.
- [ ] Los puertos, rutas y métodos HTTP coinciden con el código.
- [ ] Los ejemplos usan formatos válidos.
- [ ] Documenta entradas, salidas y errores.
- [ ] Explica persistencia y rutas de datos.
- [ ] No afirma que existe una base de datos o frontend si no existe.
- [ ] No contiene secretos ni información sensible.
- [ ] Incluye limitaciones y riesgos conocidos.
- [ ] Incluye validación o declara qué queda sin validar.
- [ ] Tiene historial de cambios si es un documento de diseño.

## 18. Aplicación al repositorio actual

La documentación existente debe organizarse así:

- [`README.md`](./README.md): visión general y guía de inicio.
- [`backend.md`](./backend.md): arquitectura y contratos del backend.
- Este documento: reglas de documentación.
- Próximos documentos: fichas individuales de APIs, agentes, datos y ADR.

La ausencia actual de frontend, base de datos relacional, manifiesto de
dependencias y contenedores debe registrarse como estado actual o deuda técnica,
no ocultarse mediante documentación aspiracional.
