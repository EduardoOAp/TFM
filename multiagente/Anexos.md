# Anexos — Supervisión Previsional con Sistemas Multiagente

## Contenido

- [Anexo A. Definición y estructura de los API’s](#anexo-a-definición-y-estructura-de-los-apis) — Tablas 1–8
- [Anexo B. Definición y estructura de los indicadores](#anexo-b-definición-y-estructura-de-los-indicadores) — Tablas 9–40
- [Anexo C. Series de datos por agente y módulo](#anexo-c-series-de-datos-por-agente-y-módulo) — Tablas 41–55

## Anexo A. Definición y estructura de los API’s

Este anexo detalla la conformación de las diferentes APIs que proporcionan los conjuntos de datos para su procesamiento y transformación.

La Tabla 1 presenta los rendimientos obtenidos por los diferentes regímenes de pensión que servirán como insumos para los modelos.

### Tabla 1. *Descripción de las características del API de rentabilidades.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| Entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| Tipo | Tipo de Rentabilidad | Texto | Nominal | NOMINAL – REAL |
| periodicidad | Periodicidad de cálculo | decimal | Ordinal | Mensual, Anual, 3 años, 5 años, 10 años. |
| rentabilidad | Valor de rentabilidad | Número | Nominal | Valor de la rentabilidad |
| fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| regimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| codigofondo | Código del Fondo | Texto | Nominal | 07 |
| fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 2 detalla las características de las comisiones aplicadas a los diferentes regímenes de pensión, constituyendo un insumo fundamental para el desarrollo de los modelos de análisis, al permitir evaluar el impacto de los costos administrativos sobre la rentabilidad neta de los fondos de pensiones.

### Tabla 2. *Descripción de las características del API de comisiones.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| tipo | Tipo de comisión | Texto | Nominal | aporte, rendimiento y saldo. |
| comision | Valor de la comisión | Decimal | Nominal | Valores de la comisión |
| fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| regimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| codigofondo | Código del Fondo | Texto | Nominal | 07 |
| fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 3 presenta las características de los portafolios de inversión asociados a los diferentes regímenes de pensión. Esta información resulta esencial para los modelos de evaluación de riesgo, ya que permite analizar la diversificación, la exposición a distintos instrumentos financieros y la liquidez de los activos administrados por cada entidad.

### Tabla 3. *Descripción de las características del API del portafolio de inversión.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| codigoentidad | Códigos de las Operadora | Texto | Nominal | Ejemplo A02, A04, A06… |
| codigomoneda | Código de la moneda | Entero | Nominal | 1: colones, 2: dólares |
| moneda | Descripción de la moneda | Texto | Nominal | Colones, dólares. |
| codigomodalidadinversion | Código de modalidad de inversión | Texto | Nominal | DE, AC, DI, FA, … |
| modalidadinversion | Descripción de modalidad de inversión | Texto | Nominal | Deuda estandarizada, acciones, deuda individualizada, fondos de participación, entre otros. |
| codigoinstrumento | Código de instrumento | Texto | Nominal | cdp, acc, entre otros. |
| instrumento | Descripción de instrumento | Texto | Nominal | Descripción de instrumento |
| codigofondoinversion | Código de fondo de inversión | Texto | Nominal | M, D, F, N, entre otros. |
| fondoinversion | Descripción de fondo de inversión | Texto | Nominal | Mercado, Dinero, Financiero, entre otros. |
| codigoplazo | Código del plazo de vencimiento | Texto | Ordinal | 1-6 |
| plazo | Descripción de plazo de vencimiento | Texto | Ordinal | 0-30 días, 30-180 días, etc. |
| codigosector | Código de sector | Entero | Nominal | 1, 2, 3 |
| sector | Descripción de sector | Texto | Nominal | Público, privado, extranjero. |
| codigoemisor_gestor | Código de emisor | Texto | Nominal | Código de emisor, (G, BCCR, UST…) |
| emisor_gestor | Descripción del gestor o emisor | Texto | Nominal | Nombre del gestor o emisor de la emisión |
| montocolones | Monto en colones | Número | Razón | Monto de la inversión en colones |
| montodolares | Monto en dólares | Número | Razón | Monto de la inversión en dólares |
| fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| regimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| codigofondo | Código del Fondo | Texto | Nominal | 07 |
| fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |
| tipo | Clasificación del emisor | Texto | Nominal | Emisor/Gestor |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 4 detalla las características agregadas de los movimientos por concepto de libre transferencia entre fondos de pensión. Esta información permite identificar patrones de migración de afiliados, evaluar la retención de usuarios y alimentar modelos de *clustering* o *forecasting* que advierten sobre posibles fugas masivas o comportamientos atípicos en la distribución de afiliados.

### Tabla 4. *Descripción de las características del API de libre transferencia.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| codigomoneda | Código de la moneda | Entero | Nominal | 1: colones, 2: dólares |
| moneda | Descripción de la moneda | Texto | Nominal | Colones, dólares. |
| BN_VITAL_C | Cantidad de personas que se pasaron BN Vital | Número | Nominal |  |
| BN_VITAL_M | Monto transferido a BN Vital | Número | Nominal |  |
| INS_PENSIONES_C | Cantidad de personas que se pasaron al INS | Número | Nominal |  |
| INS_PENSIONES_M | Monto transferido al INS | Número | Nominal |  |
| POPULAR_C | Cantidad de personas que se pasaron a POPULAR | Número | Nominal |  |
| POPULAR_M | Monto transferido a POPULAR | Número | Nominal |  |
| VIDA_PLENA_C | Cantidad de personas que se pasaron a VIDA PLENA | Número | Nominal |  |
| VIDA_PLENA_M | Monto transferido a VIDA PLENA | Número | Nominal |  |
| IBP_PENSIONES_C | Cantidad de personas que se pasaron a IBP | Número | Nominal |  |
| IBP_PENSIONES_M | Monto transferido a IBP | Número | Nominal |  |
| BACSJ_PENSIONES_C | Cantidad de personas que se pasaron al BAC | Número | Nominal |  |
| BACSJ_PENSIONES_M | Monto transferido al BAC | Número | Nominal |  |
| BCR_PENSION_C | Cantidad de personas que se pasaron al BCR | Número | Nominal |  |
| BCR_PENSION_M | Monto transferido al BCR | Número | Nominal |  |
| CCSS_OPC_C | Cantidad de personas que se pasaron al CCSS | Número | Nominal |  |
| CCSS_OPC_M | Monto transferido al CCSS | Número | Nominal |  |
| fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| regimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| codigofondo | Código del Fondo | Texto | Nominal | 07 |
| fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

Las características asociadas a los afiliados y aportantes del sistema previsional se detallan en la Tabla 5. Estos datos corresponden a series agregadas por entidad, fondo y grupo etario; y permiten analizar la evolución demográfica, la tasa de aportación y la estructura etaria de los contribuyentes. Esta información resulta esencial como base para modelos de sostenibilidad, envejecimiento poblacional y predicción de captación o retención de afiliados.

### Tabla 5. *Descripción de las características del API de afiliados y aportantes.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| codigosexo | Código de sexo | Texto | Nominal | M: Masculino F: Femenino |
| sexo | Descripción | Texto | Nominal | Masculino, Femenino |
| codigorangoedad | Código de rango de edad | Entero | Ordinal | 1, 2, 3, 4, 5, entre otros. |
| rangoedad | Descripción de rangos de edad | Texto | Ordinal | 1: < 31, 2:31 A < 45, 3:45 A < 59, 4:59 A < 100 y 5: >= 100 |
| afiliados | Cantidad de afiliados | Entero | Nominal | Cantidad de afiliados |
| aportantes | Cantidad de aportantes | Entero | Nominal | Cantidad de aportantes |
| salario_o_aportepromedio_colones | Monto promedio en colones de los aportes | Número | Nominal | Monto en colones promedio de aportes |
| salario_o_aportepromedio_dolares | Monto promedio en dólares de los aportes | Número | Nominal | Monto en dólares promedio de aportes |
| fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| regimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| codigofondo | Código del Fondo | Texto | Nominal | 07 |
| fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 6 presenta características agregadas sobre los beneficiarios y pensionados del sistema previsional, incluyendo el número total de pensionados y los montos promedio de los beneficios otorgados. Es importante señalar que los datos no corresponden a registros individualizados, sino a series estadísticas consolidadas por entidad y fondo. Esta información es clave para proyectar egresos del sistema, analizar la evolución demográfica de la población pensionada y evaluar la sostenibilidad financiera a mediano y largo plazo.

### Tabla 6. *Descripción de las características del API de beneficios y pensionados.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| codigosexo | Código de sexo | Texto | Nominal | M: Masculino F: Femenino |
| sexo | Descripción | Texto | Nominal | Masculino, Femenino |
| codigorangoedad | Código de rango de edad | Entero | Ordinal | 1, 2, 3, 4, 5, entre otros |
| rangoedad | Descripción de rangos de edad | Texto | Ordinal | 1: < 31, 2:31 A < 45, 3:45 A < 59, 4:59 A < 100 y 5: >= 100 |
| codigotipobeneficio | Código del tipo de beneficio | Entero | Nominal | 1, 2, 3, 4, entre otros |
| tipobeneficio | Descripción del tipo de beneficio | Texto | Nominal | Renta permanente, retiro programado, renta temporal. |
| Beneficio | Cantidad de beneficios | Número | Nominal |  |
| beneficiopromediocolones | Monto en colones del beneficio promedio | Número | Nominal |  |
| beneficiopromediodolares | Monto en dólares del beneficio promedio | Número | Nominal |  |
| Fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| Codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| Régimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| Codigofondo | Código del Fondo | Texto | Nominal | 07 |
| Fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 7 describe las características de los saldos contables y valores cuota de los fondos de pensiones. Estos datos permiten construir indicadores como el activo neto, el índice de equilibrio financiero y la rentabilidad ajustada, sirviendo como base para modelos de forecasting, evaluación de liquidez y detección de desequilibrios estructurales.

### Tabla 7. *Descripción de las características del API de saldos contables y valor cuota.*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| Entidad | Operadoras de Pensiones | Texto | Nominal | BAC San José Pensiones, Vida Plena, entre otros. |
| Codigocuenta | Códigos de las cuentas | Entero | Nominal | 1100000000000, 1300000000000, 2100000000000, entre otros. |
| Cuenta | Nombres de las cuentas | Texto | Nominal | VALOR CUOTA, ACTIVO, PASIVOS,INVERSIONES, GASTOS, entre otros. |
| Montocolones | Monto en colones | Número | Nominal | Monto en colones |
| Montodolares | Monto en dólares | Número | Nominal | Monto en dólares |
| Fecha | Fecha del Registro | Fecha |  | Mes año desde el 2010 |
| Codigoregimen | Código del Régimen | Entero | Nominal | Ejemplo ROP. |
| Régimen | Nombre del Régimen | Texto | Nominal | Régimen Complementario Obligatorio |
| Codigofondo | Código del Fondo | Texto | Nominal | 07 |
| Fondo | Nombre del Fondo | Texto | Nominal | Fondo obligatorio de pensiones |

*Fuente: [https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html](https://webapps.supen.fi.cr/Estadisticas/API/documentacion/index.html)*

La Tabla 8 detalla las características de las curvas soberanas publicadas por el Banco Central de Costa Rica (BCCR), que se utilizaran como referencia para el cálculo de tasas libres de riesgo. Estas curvas, que contienen tasas por plazo y fecha, permiten incorporar una dimensión macroeconómica al análisis previsional. Su integración con los datos del sistema de pensiones es fundamental para la estimación de indicadores financieros como el Alfa de Jensen, que evalúa el rendimiento ajustado al riesgo.

### Tabla 8. *Descripción de las características del API de curva soberana (Tasa libre de Riesgo).*

| **Nombre de la Variable** | **Descripción** | **Tipo** | **Escala** | **Valor/Rango** |
| --- | --- | --- | --- | --- |
| COD_INDICADORINTERNO | Código del indicador de curva soberana anual | Entero | Nominal | 25639 |
| DES_FECHA | Fecha del Registro | Fecha |  |  |
| NUM_VALOR | Valor del indicador | Decimal | Continua |  |

*Fuente:[https://gee.bccr.fi.cr/Indicadores/Suscripciones/WS/wsindicadoreseconomicos.asmx/ObtenerIndicadoresEconomicosXML](https://gee.bccr.fi.cr/Indicadores/Suscripciones/WS/wsindicadoreseconomicos.asmx/ObtenerIndicadoresEconomicosXML)*

Para el cálculo de algunos indicadores financieros, se requiere una estimación de la tasa libre de riesgo. Por lo que se optó por utilizar la curva soberana anual, la cual representa la relación entre las tasas de rendimiento y los plazos al vencimiento de instrumentos de deuda emitidos por el Estado, caracterizados por un riesgo crediticio y de liquidez mínimo. Esta curva, construida a partir de datos de mercado en un periodo determinado, permite estimar el rendimiento anualizado de referencia para evaluar el desempeño relativo de un portafolio en el mismo horizonte temporal.

## Anexo B. Definición y estructura de los indicadores

Esta sección detalla la estructura de cada indicador, sus fórmulas, fuentes de datos, ventanas temporales consideradas y su interpretación dentro del sistema previsional. En las tablas siguientes podemos apreciar cada uno de los indicadores definidos. La Tabla 9 describe el indicador de concentración de activos, junto con sus principales características y su utilidad dentro del contexto de los fondos de capitalización individual.

### Tabla 9. *Indicador de concentración de activos*

| **Nombre** | Concentración de activos |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**Esto se calcula identificando el activo con mayor proporción dentro del portafolio, y dividiendo su monto entre el total de inversiones del fondo. |
| **Umbral sugerido** | > 0.50 (es decir, más del 50% del total invertido está concentrado en un solo activo o tipo de activo).<br>Se utiliza como señal de alerta para fondos que dependen excesivamente del desempeño de un solo instrumento financiero, aumentando así el riesgo no diversificable para los afiliados. |
| **Qué mide** | Mide el porcentaje del portafolio de inversión que está concentrado en el activo más significativo. Se utiliza para identificar cuán expuesto está un fondo a riesgos específicos de un activo individual.Entre más alto este porcentaje, menor será la diversificación, lo que puede aumentar la vulnerabilidad del fondo frente a pérdidas en ese activo. |
| **Justificación** | Una alta concentración en un solo activo implica que el fondo depende en demasía del desempeño de ese activo. Si este sufre una pérdida de valor, la caída afectará a todos los afiliados del fondo.<br>En un régimen de Fondo de Capitalización Individual la exposición a activos concentrados puede tener consecuencias graves para el patrimonio individual.<br>Por lo que se requiere de vigilancia activa de la composición del portafolio para asegurar una adecuada diversificación. |
| **Justificación normativa** | Las operadoras deben evaluar los niveles de concentración por emisor, grupo económico, sector económico y clase de activo. (SUPEN, 2018) |
| **Relevancia en FCI** | En los sistemas de capitalización individual:<br>• No existe subsidio cruzado entre afiliados: si un afiliado tiene pérdidas, estas no son compensadas por otros.<br>• La volatilidad del portafolio impacta directamente el saldo de la cuenta individual.<br>• Una concentración superior al 50% significa que más de la mitad del capital está expuesto a un solo evento de mercado, lo cual es riesgoso.<br>• Este indicador debe ser usado para activar alarmas o medidas de redistribución si se detecta una exposición excesiva. |

**Fórmulas originales de la Tabla 9**

**Fórmula 1:**

$$
Concentración\ de\ activos\  = \ \frac{\sum activo\ más\ representativo}{\sum portafolio}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 10 se detalla el indicador de distribución de afiliados, utilizado para evaluar la proporción de aportantes concentrados en una misma entidad. Niveles altos pueden implicar riesgos estructurales y limitar la competencia dentro del sistema previsional.

### Tabla 10. *Indicador de distribución de Afiliados*

| **Nombre** | Distribución de Afiliados por Entidad |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 0.30 (más del 30% de los aportantes concentrados en una sola Entidad) |
| **Qué mide** | Calcula la proporción de aportantes concentrados en una única entidad dentro del sistema, reflejando el grado de centralización poblacional del sistema previsional en torno a dicha entidad. |
| **Justificación** | Una alta concentración en una entidad puede representar un riesgo estructural. Si esa entidad presenta problemas de gestión, insolvencia o reputación, puede impactar una proporción significativa de afiliados. Además, limita la competencia saludable entre fondos y puede generar vulnerabilidad sistémica en caso de fallos operativos o financieros. |
| **Justificación normativa** | La trayectoria de inversión se debe definir según las características demográficas de cada cohorte de afiliados. (SUPEN, 2024) |
| **Relevancia en FCI** | En un FCI, cada afiliado asume el resultado de su decisión. Sin embargo, la concentración alta puede dificultar la gestión del sistema, reducir la capacidad de elección real, y amplificar el impacto sistémico si esa entidad falla. Es esencial para supervisar la equidad y estabilidad del sistema previsional. |

**Fórmulas originales de la Tabla 10**

**Fórmula 1:**

$$
Proporción\ por\ entidad\  = \ \frac{\sum Aportantes\ entidad}{\sum Aportantes}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador presentado en la Tabla 11 aplica el concepto de entropía para medir la dispersión de las inversiones dentro del portafolio. Valores bajos revelan una concentración significativa, lo que puede aumentar la vulnerabilidad del fondo ante eventos negativos.

### Tabla 11. *Índice de Diversificación de Shannon*

| **Nombre** | Índice de Diversificación de Shannon |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>*p<sub>i</sub>* es la proporción de inversión en el activo *i* con respecto al total invertido. |
| **Umbral sugerido** | < 1.0: Riesgo de concentración muy alto<br>Rango interpretativo:<br>0.0–1.0: Muy concentrado;<br>1.0–1.5: Diversificación baja;<br>1.5–2.0: Aceptable;<br>>2.0: Buena a excelente diversificación |
| **Qué mide** | Evalúa qué tan distribuida está la inversión entre distintos activos dentro del portafolio. Utiliza el principio de entropía para estimar la incertidumbre o dispersión de las inversiones. Un valor más alto implica una distribución más balanceada; un valor más bajo indica concentración en pocos activos. |
| **Justificación** | Es una herramienta clave en la gestión de riesgos. Un portafolio concentrado en pocos activos está más expuesto a pérdidas ante eventos adversos específicos de esos activos. Usar la entropía de Shannon permite detectar cuándo un fondo depende demasiado de un grupo reducido de instrumentos financieros, lo cual compromete su resiliencia. |
| **Justificación normativa** | Se deben emplear métricas de dispersión para validar la diversificación efectiva del portafolio. (SUPEN, 2018) |
| **Relevancia en FCI** | En sistemas de capitalización individual (FCI), donde cada afiliado asume el riesgo del desempeño de su portafolio, una diversificación adecuada es fundamental. Un fondo con baja diversificación puede presentar mayor volatilidad y pérdida de valor, afectando directamente los saldos individuales de los afiliados. Este indicador permite monitorear la robustez y sostenibilidad de las estrategias de inversión adoptadas por las entidades. |

**Fórmulas originales de la Tabla 11**

**Fórmula 1:**

$$
H = - \sum_{i = 1}^{n}p_{i}*\ \ln(p_{i})
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

De acuerdo con la Tabla 12, el indicador mide la diferencia entre entradas y salidas de afiliados por libre transferencia. Un flujo neto negativo persistente puede ser una señal de pérdida de confianza o bajo atractivo del fondo para los afiliados activos.

### Tabla 12. *Indicados de Flujo Neto de Afiliados*

| **Nombre** | Flujo Neto de Afiliados |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>Entrantes entidad: número de personas que ingresan a un fondo por libre transferencia<br>Salientes entidad: número de personas que salen del fondo en el mismo período. |
| **Umbral sugerido** | < 0.0: Indica que el fondo está perdiendo más afiliados de los que gana (flujo negativo). |
| **Qué mide** | Cuantifica el balance relativo entre las personas que ingresan a un fondo de pensión por libre transferencia y las que lo abandonan. El resultado permite identificar si el fondo está siendo preferido o rechazado por los afiliados. |
| **Justificación** | Un flujo neto negativo recurrente es una señal temprana de pérdida de confianza por parte de los afiliados. Puede estar asociado a factores como bajo rendimiento, comisiones altas o baja percepción de solidez. La evaluación de este indicador permite alertar sobre la competitividad del fondo y el atractivo percibido por los afiliados activos. |
| **Justificación normativa** | El traslado de los activos financieros debe reflejarse con movimientos contables codificados para evaluar el flujo neto entre fondos. (SUPEN, 2024) |
| **Relevancia en FCI** | En los Fondos de Capitalización Individual, la decisión de un afiliado de cambiarse de entidad tiene implicaciones directas sobre la administración del capital acumulado. Por ello, monitorear el flujo neto permite detectar tendencias migratorias que, de consolidarse, podrían afectar la sostenibilidad financiera, operativa y reputacional del fondo en el tiempo. |

**Fórmulas originales de la Tabla 12**

**Fórmula 1:**

$$
Flujo\ neto\ de\ afiliados\  = \ \frac{\sum Entrantes\ entidad}{\sum Salientes\ entidad}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

Según lo expuesto en la Tabla 13, el Índice de Carga Administrativa cuantifica el porcentaje del activo neto del fondo que se destina al pago de comisiones. Un valor elevado puede afectar la rentabilidad para los afiliados e indicar ineficiencias operativas en la gestión del fondo.

### Tabla 13. *Índice de Carga Administrativa*

| **Nombre** | Índice de Carga Administrativa |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 1.5% |
| **Qué mide** | Mide el porcentaje del activo neto del fondo que se destina al pago de comisiones de administración. Evalúa qué tan costoso es mantener la gestión del fondo en relación con su tamaño. |
| **Justificación** | Una carga administrativa elevada reduce la rentabilidad neta para los afiliados. En fondos de gran tamaño, se espera que los costos operativos sean relativamente menores debido a las economías de escala. Un valor alto puede indicar ineficiencias operativas, excesos administrativos o problemas de competitividad frente a otras entidades del sistema. |
| **Justificación normativa** | El beneficio será menor en función de la comisión aplicada por la operadora sobre los rendimientos o aportes. (SUPEN, 2010) |
| **Relevancia en FCI** | En los FCI, las comisiones tienen un impacto directo sobre el crecimiento del capital del afiliado. Este indicador es clave para evaluar si la entidad está proporcionando valor agregado frente al costo que representa su gestión. |

**Fórmulas originales de la Tabla 13**

**Fórmula 1:**

$$
Indice\ de\ Carga\ Admnistrativa\  = \ \frac{(comisiones\ *\ Activo\ neto)}{Activo\ neto}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 14 se muestra la proporción de afiliados activos cercanos a la edad de jubilación. Un valor alto puede anticipar presiones sobre la sostenibilidad del fondo por mayores egresos y menor entrada de aportes.

### Tabla 14. *Índice de Envejecimiento de los Aportantes*

| **Nombre** | Índice de Envejecimiento de los Aportantes |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 60% |
| **Qué mide** | Mide la proporción de aportantes activos que tienen 59 años o más. Indica qué tan cerca está la población afiliada del retiro. |
| **Justificación** | Una alta proporción de aportantes próximos a la edad de jubilación implica una presión futura sobre los fondos por la conversión de estos aportantes en pensionados. Esto puede generar retiros masivos, menor captación neta y riesgo de desbalance si no existe un ingreso de afiliados jóvenes que compense dicha salida. |
| **Justificación normativa** | Se debe estimar la edad promedio y evolución etaria de las poblaciones afiliadas por fondo generacional. (SUPEN, 2024) |
| **Relevancia en FCI** | En un FCI, cada cuenta es individual y no existe subsidio entre afiliados. Si la mayoría de los afiliados está próxima a pensionarse, aumentan los egresos y disminuyen los ingresos por aportes, comprometiendo la sostenibilidad del fondo. Este indicador ayuda a prever escenarios de estrés operativo y financiero. |

**Fórmulas originales de la Tabla 14**

**Fórmula 1:**

$$
Indice\ de\ envejecimiento\  = \ \frac{\sum Aportantes\ mayores\ a\ 59\ años}{\sum Aportantes}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

Como se presenta en la Tabla 15, el Índice de Liquidez del Portafolio mide la proporción de activos líquidos dentro de un fondo. Un valor bajo puede limitar la capacidad del fondo para responder rápidamente a necesidades operativas o situaciones financieras urgentes.

### Tabla 15. *Índice de Liquidez del Portafolio*

| **Nombre** | Índice de Liquidez del Portafolio |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | < 0.5 (50%) |
| **Qué mide** | Proporción del portafolio de inversión compuesto por activos líquidos, es decir, activos que pueden convertirse rápidamente en efectivo sin pérdida significativa de valor. Es fundamental para evaluar la capacidad operativa y resiliencia del fondo ante eventos de fuga o presión financiera. |
| **Justificación** | Un bajo porcentaje de activos líquidos puede indicar que el fondo tiene limitada capacidad de respuesta ante eventos de fuga de afiliados, crisis económicas o necesidades operativas urgentes. Una baja liquidez también reduce la agilidad del fondo para reajustar su estrategia de inversión ante cambios de mercado. Ideal para monitorear la agilidad financiera de los fondos de pensión de capitalización individual. |
| **Justificación normativa** | Deben establecerse métricas para asegurar la disponibilidad de recursos líquidos ante flujos netos negativos. (SUPEN, 2018) |
| **Relevancia en FCI** | En los FCI, donde los aportes y retiros están vinculados directamente al comportamiento de cada afiliado, la liquidez es crítica. Una baja liquidez puede comprometer la capacidad del fondo de pagar beneficios a tiempo o responder ante movimientos bruscos de traspasos. |

**Fórmulas originales de la Tabla 15**

**Fórmula 1:**

$$
Indice\ de\ Liquidez = \ \frac{\sum Inversiones\ en\ activos\ líquidos}{\sum Inversiones}*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El Índice de Sostenibilidad Financiera compara los ingresos por aportes con los egresos por beneficios. Un valor bajo sugiere un desequilibrio financiero que podría generar tensiones de liquidez o requerir ajustes en las políticas del fondo. Esto se aprecia en la Tabla 16.

### Tabla 16. *Índice de Sostenibilidad Financiera*

| **Nombre** | Índice de Sostenibilidad Financiera |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | < 0.70 |
| **Qué mide** | Este índice evalúa si los ingresos esperados por los aportes de los afiliados son suficientes para cubrir los pagos de beneficios que el fondo debe realizar. Sirve como una aproximación al equilibrio de flujos entre ingresos y egresos. |
| **Justificación** | Un valor inferior a 0.70 indica que por cada colón que se requiere para pagar beneficios, solo se están recaudando 70 céntimos en aportes, lo que sugiere un desequilibrio financiero latente. Esta brecha puede derivar en tensiones de liquidez o en la necesidad de financiamiento adicional, subsidios cruzados o ajustes en la política de comisiones o beneficios. Es un proxy de tensión de flujos, útil para anticipar desequilibrios, aunque no mide sostenibilidad en sentido estructural o de largo plazo. |
| **Justificación normativa** | Se debe estimar si las trayectorias de inversión permiten alcanzar la tasa de reemplazo objetivo definida. (SUPEN, 2024) |
| **Relevancia en FCI** | En un fondo de capitalización individual, la autosuficiencia financiera es crítica, ya que no hay subsidio cruzado. Un índice bajo refleja que la masa activa no es suficiente para sostener el pago a los pensionados, afectando la percepción de estabilidad y la sostenibilidad operativa del fondo. Es clave para monitorear la salud financiera de fondos con egresos crecientes, como los que se acercan al ciclo de madurez. Además, sirve para alertar sobre el envejecimiento de la población y prever presiones futuras sobre el sistema. |

**Fórmulas originales de la Tabla 16**

**Fórmula 1:**

$$
ISF = \ \frac{\sum Aportantes\ *\ \overline{Salario}}{\sum Beneficios\ *\ \overline{Monto\ Beneficio}}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

Como se describe en la Tabla 17, el Índice mide la relación entre aportantes activos y pensionados. Un valor inferior a 1.0 indica un desbalance en el flujo de recursos, lo que podría comprometer la estabilidad y sostenibilidad del fondo a largo plazo.

### Tabla 17. *Índice de Sostenibilidad de Pensiones*

| **Nombre** | Índice de Sostenibilidad de Pensiones |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | < 1.0 |
| **Qué mide** | Mide cuántos aportantes activos existen por cada pensionado. Evalúa si la base activa del sistema es suficiente para sostener las obligaciones pasivas en el tiempo. |
| **Justificación** | Un valor inferior a 1.0 indica que hay más pensionados que personas activamente aportando al sistema, lo que genera un desbalance estructural en el flujo de recursos. Esta situación puede llevar al agotamiento de los fondos disponibles, poniendo en riesgo la estabilidad operativa y la sostenibilidad del fondo. Es útil para evaluar si el fondo tiene masa crítica suficiente para sostener sus operaciones y proyecciones de escala. |
| **Justificación normativa** | Se debe considerar la densidad de cotización y la acumulación proyectada para garantizar beneficios sostenibles (SUPEN, 2024). El sistema busca garantizar una pensión conforme a los derechos adquiridos por los trabajadores, mediante la adecuada administración y protección de los recursos. (Asamblea Legislativa, 2000) |
| **Relevancia en FCI** | Aunque los FCI funcionan bajo cuentas personales, este indicador permite anticipar si el sistema en su conjunto enfrenta una tendencia de envejecimiento poblacional y retiro masivo de recursos. Ayuda a identificar riesgos sistémicos y apoya en el diseño de estrategias de retención de afiliados jóvenes o de captación de nuevos cotizantes. |

**Fórmulas originales de la Tabla 17**

**Fórmula 1:**

$$
ISP = \ \frac{\sum Aportantes\ }{\sum Beneficios\ }
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador a continuación mide la relación entre pensionados y afiliados activos. Un valor alto sugiere una presión creciente sobre los recursos del fondo, lo que podría requerir ajustes en la gestión y estrategias para captar nuevos aportantes, como se detalla en la

### Tabla 18. *Indicador de Proporción de Pensionados sobre Aportantes*

| **Nombre** | Proporción de Pensionados sobre Aportantes |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 0.35 |
| **Qué mide** | Mide la relación entre la cantidad total de personas que ya están pensionadas y el total de personas activas que están realizando aportes en el periodo evaluado. No implica que los pensionados hayan sido parte de los aportantes actuales, sino que compara la magnitud de la carga pasiva con la base contributiva activa. |
| **Justificación** | Una proporción alta sugiere que el número de pensionados representa una fracción considerable respecto a quienes están aportando actualmente. Esto puede indicar madurez del fondo, presión creciente sobre los recursos y necesidad de estrategias para captación de nuevos aportantes o ajustes en gestión financiera. |
| **Justificación normativa** | Se deben estimar indicadores de envejecimiento y presión financiera sobre los fondos por número de pensionados. (SUPEN, 2024) |
| **Relevancia en FCI** | En los fondos de capitalización individual, donde cada cuenta es gestionada individualmente y no hay redistribución, este indicador permite identificar si existe un desequilibrio estructural a nivel del sistema que pueda impactar la eficiencia operativa, liquidez o percepción pública del fondo. Una carga pasiva alta puede señalar mayor sensibilidad a retiros masivos o menor escala económica operativa. |

**Fórmulas originales de la Tabla 18**

**Fórmula 1:**

$$
PPA = \ \frac{\sum Beneficios\ }{\sum Aportantes\ }
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

Según se detalla en la Tabla 19, la Rentabilidad Acumulada del Mes calcula la variación porcentual del valor cuota del fondo entre el inicio y el final del mes. Una rentabilidad mensual negativa recurrente podría indicar bajo desempeño o problemas de gestión, lo que afectaría la percepción y confianza de los afiliados.

### Tabla 19. *Indicador de Rentabilidad Acumulada del Mes*

| **Nombre** | Rentabilidad Acumulada del Mes |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde: VC es equivalente al valor cuota. |
| **Umbral sugerido** | < -0.03 |
| **Qué mide** | La rentabilidad generada por el fondo en el mes actual, evaluando la variación porcentual entre el valor cuota al inicio y al final del mes. Refleja el rendimiento del portafolio en el corto plazo. |
| **Justificación** | Una rentabilidad acumulada mensual negativa sostenida puede reflejar bajo desempeño del portafolio, afectaciones del mercado o mala gestión. Umbrales negativos significativos (ej. menores a -3%) pueden ser señales de alerta sobre pérdidas recurrentes que impacten la confianza del afiliado o motiven traspasos. |
| **Justificación normativa** | La rentabilidad acumulada debe compararse periódicamente contra la trayectoria y objetivo declarado. (SUPEN, 2024) |
| **Relevancia en FCI** | En los FCI, donde cada afiliado asume su propio riesgo de inversión, una rentabilidad mensual negativa recurrente puede afectar el valor acumulado en su cuenta individual. Este indicador es clave para monitorear la percepción y seguridad financiera de los afiliados y para anticipar efectos sobre la confianza del afiliado, las decisiones de aporte o traspaso, y la sostenibilidad del portafolio en el corto plazo. |

**Fórmulas originales de la Tabla 19**

**Fórmula 1:**

$$
{Rentabilidad}_{mes} = \left( \ \frac{{VC}_{fin\ de\ mes} - {VC}_{inicio\ de\ mes}\ }{{VC}_{inicio\ de\ mes}} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

De acuerdo con la información presentada en la Tabla 20, el indicador evalúa la rentabilidad neta del fondo, descontando los costos administrativos. Un valor negativo o bajo sugiere que las comisiones están reduciendo significativamente las ganancias, lo cual podría indicar ineficiencias operativas o costos desproporcionados.

### Tabla 20. *Indicador de Rentabilidad Ajustada por Comisión*

| **Nombre** | Rentabilidad Ajustada por Comisión |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>Donde: *An* equivale al activo neto del fondo. |
| **Umbral sugerido** | < -0.02 (anual) |
| **Qué mide** | Mide la rentabilidad efectiva del fondo después de restar los costos por comisiones. Refleja si la rentabilidad obtenida es suficiente para compensar el costo administrativo y generar valor neto para el afiliado. |
| **Justificación** | Un valor negativo indica que las comisiones están erosionando completamente las ganancias del fondo, o que incluso están generando una pérdida neta. Esto es una señal de ineficiencia operativa o de costos administrativos desproporcionados frente al rendimiento financiero. |
| **Justificación normativa** | El cálculo de beneficios debe considerar el rendimiento neto luego de aplicar la comisión de administración (SUPEN, 2010). El cálculo de rendimientos debe considerar la deducción de comisiones por administración, según lo establecido en la Ley de Protección al Trabajador. (Asamblea Legislativa, 2000) |
| **Relevancia en FCI** | En un fondo de capitalización individual, donde cada afiliado asume su propio riesgo y los costos impactan su rendimiento personal, es fundamental asegurar que las comisiones no anulen el retorno. Este indicador es clave para la transparencia, eficiencia y confianza del afiliado, ya que un valor bajo o negativo puede ser señal de ineficiencia operativa o riesgo reputacional, incluso si la rentabilidad bruta parece aceptable. |

**Fórmulas originales de la Tabla 20**

**Fórmula 1:**

$$
Rentabilidad\ Ajustada = \left( \ \frac{({Rentabilidad}_{neta}*\ An) - \ (Comision\ *\ An)}{An} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador de Rentabilidad Diaria que se muestra en la Tabla 21 evalúa la estabilidad del valor cuota del fondo mediante su variación porcentual día a día. Una rentabilidad diaria persistentemente negativa o con alta volatilidad puede señalar riesgos operativos o estrategias de inversión inadecuadas.

### Tabla 21. *Indicador de Rentabilidad Diaria*

| **Nombre** | Rentabilidad Diaria |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde: VC*<sub>t</sub>* es equivalente al valor cuota del mes *t*. |
| **Umbral sugerido** | Media < -0.01 o varianza alta respecto a la desviación estándar histórica |
| **Qué mide** | La variación porcentual diaria del valor cuota del fondo de pensiones, en otras palabras, la estabilidad o volatilidad en el rendimiento financiero. |
| **Justificación** | Rentabilidades diarias negativas persistentes o con alta varianza pueden alertar sobre riesgos operativos, estrategias agresivas o eventos externos. |
| **Justificación normativa** | Debe monitorearse el comportamiento diario de rentabilidad para detectar desvíos relevantes. (SUPEN, 2024) |
| **Relevancia en FCI** | La estabilidad del valor cuota es clave para la percepción de seguridad por parte de los afiliados. Movimientos erráticos pueden generar fuga o pérdida de confianza, afectando directamente los flujos y la sostenibilidad del fondo. |

**Fórmulas originales de la Tabla 21**

**Fórmula 1:**

$$
{Rentabilidad\ Diaria}_{t} = \left( \ \frac{{VC}_{t} - {VC}_{t - 1}\ }{{VC}_{t - 1}} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 22 se presenta el indicador de Riesgo de Liquidez, el cual estima la proporción del portafolio invertida en activos ilíquidos. Un valor elevado advierte sobre posibles dificultades para atender retiros o pagos, especialmente en escenarios de tensión financiera.

### Tabla 22. *Indicador de Riesgo de Liquidez*

| **Nombre** | Riesgo de Liquidez |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 0.7 (o 70%) |
| **Qué mide** | El porcentaje de la cartera invertido en activos considerados ilíquidos, es decir, que no pueden venderse fácilmente sin pérdida de valor. |
| **Justificación** | Una alta proporción de activos ilíquidos implica que el fondo podría tener dificultades para responder rápidamente a salidas de efectivo. Esto se traduce en un mayor riesgo ante eventos de pánico, retiros masivos o situaciones de estrés financiero. |
| **Justificación normativa** | Se deben estimar métricas para anticipar posibles riesgos de iliquidez que afecten pagos o transferencias. (SUPEN, 2018) |
| **Relevancia en FCI** | En un fondo de capitalización individual, cada cuenta debe estar disponible en cualquier momento. Una alta exposición a activos ilíquidos puede limitar la capacidad de pago y deteriorar la confianza del afiliado. |

**Fórmulas originales de la Tabla 22**

**Fórmula 1:**

$$
RiesgoLiquidez = \ \left( \frac{\sum Activos\ Iliquidos\ }{\sum Inversiones\ } \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 23, se presenta el Indicador que mide el porcentaje de nuevos afiliados incorporados al fondo en relación con el total de aportantes activos. Una tasa baja podría reflejar desinterés o baja competitividad frente a otras opciones del sistema, afectando la viabilidad operativa y la sostenibilidad a largo plazo del fondo.

### Tabla 23. *Indicador de Tasa de Captación de Afiliados*

| **Nombre** | Tasa de Captación de Afiliados |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | < 0.5% (equivalente a 0.005 en proporción) |
| **Qué mide** | El porcentaje de nuevos afiliados incorporados al fondo durante el periodo, en relación con el total actual de aportantes activos. |
| **Justificación** | Una tasa alta refleja que el fondo es atractivo para nuevos afiliados, lo cual puede estar asociado a buena rentabilidad, reputación, o servicios ofrecidos. Mientras que una tasa baja podría indicar desinterés o baja competitividad frente a otras opciones del sistema. |
| **Justificación normativa** | Debe estimarse la tasa de captación bruta de afiliados en el tiempo, en relación con nuevos registros activos. (SUPEN, 2024) |
| **Relevancia en FCI** | En fondos de capitalización individual, el crecimiento sostenido de la base de afiliados es crucial para asegurar ingresos constantes y sostenibilidad a largo plazo. Una baja tasa de captación puede afectar la viabilidad operativa futura del fondo. |

**Fórmulas originales de la Tabla 23**

**Fórmula 1:**

$$
TasaCaptacion = \ \left( \frac{\sum Aportantes\ nuevos}{\sum Aportantes\ } \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El Indicador que se muestra en la Tabla 24 mide el cambio porcentual en el número de afiliados entre dos periodos consecutivos. Una tasa negativa indica pérdida de afiliados, lo que puede afectar la sostenibilidad del fondo.

### Tabla 24. *Indicador de Tasa de Crecimiento de Aportantes*

| **Nombre** | Tasa de Crecimiento de Aportantes |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>Donde: *t* es el mes en proceso. |
| **Umbral sugerido** | < 0 (crecimiento negativo) |
| **Qué mide** | El cambio porcentual en el número de aportantes entre dos periodos consecutivos, permitiendo identificar si el fondo está ganando o perdiendo afiliados. |
| **Justificación** | Una tasa de crecimiento negativa indica pérdida de aportantes, lo cual puede deberse a factores como baja rentabilidad, reputación deficiente, cambios demográficos o migración hacia otros fondos. |
| **Justificación normativa** | Debe estimarse la tasa de crecimiento de afiliados como variable clave de sostenibilidad del fondo. (SUPEN, 2024) |
| **Relevancia en FCI** | En los Fondos de Capitalización Individual, un decrecimiento sostenido en la base de aportantes puede comprometer los ingresos por aportes, afectando la liquidez, sostenibilidad y percepción de confianza del fondo, y permite detectar tendencias a mediano plazo y orientar estrategias de retención y captación. |

**Fórmulas originales de la Tabla 24**

**Fórmula 1:**

$$
TasaCrecimiento = \ \left( \frac{\sum{Aportantes}_{t}\  - \ \sum{Aportantes}_{t - 1}}{\ \sum{Aportantes}_{t - 1}} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 25 se detalla el indicador que mide el aumento porcentual en la cantidad de pensionados respecto al mes anterior. Un crecimiento superior al 3% mensual puede generar presión sobre los egresos del fondo y afectar su estabilidad financiera.

### Tabla 25. *Indicador de Tasa de Crecimiento de Pensionados*

| **Nombre** | Tasa de Crecimiento de Pensionados |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>Donde: *t* es el mes en proceso. |
| **Umbral sugerido** | > 0.03 (crecimiento superior al 3% mensual) |
| **Qué mide** | El aumento porcentual en la cantidad de personas que han pasado a condición de pensionadas, respecto al mes anterior. |
| **Justificación** | Un incremento sostenido en la cantidad de pensionados implica una presión creciente sobre los egresos del fondo, lo que puede poner en riesgo su liquidez y estabilidad si no se acompaña de un crecimiento proporcional en aportantes. |
| **Justificación normativa** | Las operadoras deben contar con estudios técnicos que consideren las características demográficas de cada fondo generacional. (SUPEN, 2024) |
| **Relevancia en FCI** | En los Fondos de Capitalización Individual, este indicador es esencial para anticipar la presión sobre la salida de activos, prever necesidades de liquidez y planificar políticas de inversión orientadas a mantener el equilibrio financiero. |

**Fórmulas originales de la Tabla 25**

**Fórmula 1:**

$$
TasaCrecimiento = \ \left( \frac{\sum{Pensionados}_{t}\  - \ \sum{Pensionados}_{t - 1}}{\ \sum{Pensionados}_{t - 1}} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

La Tasa de Fuga de Afiliados, detallada en la Tabla 26, mide el porcentaje de afiliados que abandonan un fondo en un periodo determinado en relación con el total de aportantes. Este indicador es esencial para monitorear la salida neta de afiliados, ya que en un sistema de FCI, la fuga constante debilita la base de ingresos y afecta la percepción de solidez institucional.

### Tabla 26. *Indicador de Tasa de Fuga de Afiliados*

| **Nombre** | Tasa de Fuga de Afiliados |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | > 0.02 mensual (más del 2% de los aportantes abandona el fondo en un mes) |
| **Qué mide** | El porcentaje de afiliados que han salido de un fondo en un periodo determinado respecto al total de aportantes. |
| **Justificación** | Una tasa alta puede señalar pérdida de confianza, menor competitividad del fondo o percepciones negativas sobre su rentabilidad o servicios. Es una señal de alerta para supervisión o ajuste de estrategias. |
| **Justificación normativa** | Debe monitorearse la salida neta de afiliados desde fondos generacionales o entre operadoras. (SUPEN, 2024) |
| **Relevancia en FCI** | En un sistema de cuentas individuales como el FCI, donde el capital es propiedad del afiliado, la fuga constante de cotizantes debilita la base de ingresos y afecta la percepción de solidez institucional. |

**Fórmulas originales de la Tabla 26**

**Fórmula 1:**

$$
TasaFugaAfiliados = \ \left( \frac{\sum Afiliados\ salientes}{\sum Aportantes\ } \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

La Volatilidad, descrita en la Tabla 27 mide la desviación estándar del valor cuota diario durante los últimos 30 días. Un valor superior al doble del promedio histórico puede indicar inestabilidad en los precios o estrategias arriesgadas. En los fondos de capitalización individual, afecta la percepción de riesgo de los afiliados.

### Tabla 27. *Indicador de Volatilidad del Valor Cuota en 30 días*

| **Nombre** | Volatilidad del Valor Cuota en 30 días |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>**[Fórmula 2; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 3; véase debajo de la tabla]**​: desviación estándar o volatilidad de 30 días<br>VC*t−i* ​: valor cuota del fondo en el día *t−i*<br>**[Fórmula 4; véase debajo de la tabla]**: media de los valores cuota en los últimos 30 días<br>n: 30 |
| **Umbral sugerido** | > 2×STD histórica |
| **Qué mide** | La desviación estándar del valor cuota diario durante los últimos 30 días, reflejando la volatilidad reciente del fondo. |
| **Justificación** | Una volatilidad superior al doble del promedio histórico puede señalar inestabilidad en los precios, cambios bruscos de mercado o estrategias arriesgadas de inversión. |
| **Justificación normativa** | La volatilidad reciente debe compararse con el objetivo de rendimiento proyectado por trayectoria. (SUPEN, 2024) |
| **Relevancia en FCI** | En los fondos de capitalización individual, la volatilidad del valor cuota impacta directamente la percepción de riesgo de los afiliados, influyendo en decisiones de permanencia o transferencia. |

**Fórmulas originales de la Tabla 27**

**Fórmula 1:**

$$
\sigma_{30d} = \sqrt{\frac{1}{n}\sum_{i = 1}^{n}\left( VC_{t - i} - \mu_{30d} \right)^{2}}
$$

**Fórmula 2:**

$$
\mu_{30d} = \frac{1}{n}\sum_{i = 1}^{n}{VC_{t - i}}
$$

**Fórmula 3:**

$$
\sigma_{30d}
$$

**Fórmula 4:**

$$
\mu_{30d}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

La Volatilidad reciente del Activo Neto, que se muestra en la Tabla 28, mide la desviación estándar del activo neto en los últimos seis periodos. Si esta volatilidad supera el doble del promedio histórico, puede indicar inestabilidad en los precios o estrategias arriesgadas. Esto puede afectar la percepción de riesgo de los afiliados y sus decisiones de permanencia.

### Tabla 28. *Indicador de Volatilidad reciente del Activo Neto*

| **Nombre** | Volatilidad reciente del Activo Neto |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]** — es la desviación estándar muestral del activo neto en los últimos 6 periodos<br>**[Fórmula 3; véase debajo de la tabla]** — es la media del activo neto en los últimos 6 periodos<br>AN*<sub>t</sub>*​: Activo Neto del mes *t* |
| **Umbral sugerido** | > 2×STD histórica |
| **Qué mide** | Mide la desviación estándar del valor cuota diario durante los últimos 30 días, reflejando la volatilidad reciente del fondo. |
| **Justificación** | Una volatilidad superior al doble del promedio histórico puede señalar inestabilidad en los precios, cambios bruscos de mercado o estrategias arriesgadas de inversión. |
| **Justificación normativa** | La variabilidad del activo neto puede motivar rebalanceos del portafolio si se desvía de la trayectoria. (SUPEN, 2024) |
| **Relevancia en FCI** | En los fondos de capitalización individual, la volatilidad del valor cuota impacta directamente la percepción de riesgo de los afiliados, influyendo en decisiones de permanencia o transferencia. |

**Fórmulas originales de la Tabla 28**

**Fórmula 1:**

$$
{Volatilidad}_{rec} = \frac{\sigma_{t}^{(6)}}{{\sigma\mu}_{t}^{(6)}}
$$

**Fórmula 2:**

$$
\sigma_{t}^{(6)} = \sqrt{\frac{1}{5}\sum_{i = 0}^{5}\left( AN_{t - i} - \mu_{t}^{(6)} \right)^{2}}
$$

**Fórmula 3:**

$$
\mu_{t}^{(6)} = \frac{1}{6}\sum_{i = 0}^{5}{AN_{t - i}}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 29 se detalla el Flujo Neto Monetario de Afiliados por Libre Transferencia, el cual muestra si un fondo está perdiendo o ganando activos en valor monetario debido a libres transferencias. Un valor negativo persistente puede indicar desconfianza de los afiliados o desventajas en rentabilidad y comisiones.

### Tabla 29. *Indicador de Flujo Neto Monetario de Afiliados por Libre Transferencia*

| **Nombre** | Flujo Neto Monetario de Afiliados por Libre Transferencia |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | <−0.05 |
| **Qué mide** | Si un fondo está perdiendo o ganando activos en valor monetario como resultado de las transferencias libres entre fondos. |
| **Justificación** | Un valor negativo persistente indica una disminución neta de capital, lo que podría reflejar desconfianza por parte de los afiliados o una desventaja comparativa en términos de rentabilidad, comisiones u otros factores. |
| **Justificación normativa** | El traslado de los activos financieros debe reflejarse con movimientos contables codificados para evaluar el flujo neto entre fondos. (SUPEN, 2024) |
| **Relevancia en FCI** | En los fondos de capitalización individual, este indicador es crucial para monitorear la fuga de capital, evaluar la competitividad del fondo y anticipar posibles desequilibrios financieros derivados de salidas masivas. |

**Fórmulas originales de la Tabla 29**

**Fórmula 1:**

$$
{FlujoNeto}_{monto} = \ \left( \frac{\sum{entrantes}_{monto} - \ \sum{salientes}_{monto}}{\sum montototal} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El Índice de Equilibrio Financiero, descrito en la Tabla 30, mide la capacidad del fondo para cubrir sus egresos operativos con sus ingresos. En un sistema de cuentas individuales, este índice es esencial para asegurar la viabilidad del fondo y evitar intervenciones externas.

### Tabla 30. *Índice de Equilibrio Financiero*

| **Nombre** | Índice de Equilibrio Financiero |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]** |
| **Umbral sugerido** | <0.95 |
| **Qué mide** | La capacidad del fondo para cubrir sus egresos operativos con sus ingresos. |
| **Justificación** | Un valor inferior a 1 indica que los egresos superan los ingresos, lo que representa un déficit operativo. Un umbral de 0.95 permite anticipar situaciones cercanas al desequilibrio. |
| **Justificación normativa** | Los estudios deben considerar la relación entre aportaciones, salidas y acumulación de recursos por cohorte. (SUPEN, 2024), El régimen debe ser supervisado para garantizar equilibrio entre rentabilidad, liquidez y seguridad de los fondos invertidos. (Asamblea Legislativa, 2000) |
| **Relevancia en FCI** | En un sistema de cuentas individuales, este índice es vital para anticipar desequilibrios entre ingresos por aportes y egresos por beneficios, asegurando la viabilidad del fondo sin necesidad de intervención externa. Una caída sostenida de este indicador es una señal de alerta y puede requerir medidas de ajuste o revisión de política de comisiones, inversiones o captación de afiliados. |

**Fórmulas originales de la Tabla 30**

**Fórmula 1:**

$$
EquilibrioFinaciero = \ \left( \frac{Ingresos}{Egresos} \right)
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 31 se detalla el *Ratio de Sharpe*, el cual mide la rentabilidad adicional por unidad de riesgo asumido, usando una tasa libre de riesgo como referencia. En los fondos de capitalización individual, este indicador verifica que el riesgo asumido esté compensado por los beneficios obtenidos.

### Tabla 31. *Indicador Ratio de Sharpe*

| **Nombre** | Ratio de Sharpe |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]**​: rentabilidad promedio del fondo en el periodo<br>**[Fórmula 3; véase debajo de la tabla]**​: rentabilidad de un activo libre de riesgo en el mismo periodo<br>**[Fórmula 4; véase debajo de la tabla]**​: desviación estándar de la rentabilidad del fondo en el mismo periodo |
| **Umbral sugerido** | < 1 |
| **Qué mide** | Cuánta rentabilidad adicional obtiene un fondo por cada unidad de riesgo asumido, considerando una tasa libre de riesgo como referencia. |
| **Justificación** | Un valor menor a 1 indica que el fondo está generando menos rentabilidad por unidad de riesgo que lo esperado, sugiriendo ineficiencia o exposición excesiva al riesgo. |
| **Justificación normativa** | Las operadoras deberán valorar la relación entre rentabilidad y riesgo, ajustando el retorno por volatilidad del portafolio. (SUPEN, 2018) |
| **Relevancia en FCI** | En los fondos de capitalización individual, donde el rendimiento impacta directamente la cuenta personal del afiliado, este indicador resulta clave para verificar que el nivel de riesgo asumido por el fondo esté adecuadamente compensado por los beneficios obtenidos. |

**Fórmulas originales de la Tabla 31**

**Fórmula 1:**

$$
Sharpe = \ \left( \frac{R_{fondo} - \ R_{libre\_ riesgo}}{\sigma_{fondo}} \right)*100
$$

**Fórmula 2:**

$$
R_{fondo}
$$

**Fórmula 3:**

$$
R_{libre\_ riesgo}
$$

**Fórmula 4:**

$$
\sigma_{fondo}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El índice detallado en la Tabla 32, mide la concentración de inversiones en emisores internacionales. En fondos de capitalización individual, una alta diversificación geográfica protege a los afiliados de choques externos.

### Tabla 32. *Índice de Herfindahl-Hirschman de concentración internacional*

| **Nombre** | Índice de *Herfindahl-Hirschman* de concentración internacional |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde: *s<sub>i</sub>*:​ proporción de inversión externa, es decir, si=**[Fórmula 2; véase debajo de la tabla]**<br>*n*: número total de emisores externos en las que está invertido el portafolio |
| **Umbral sugerido** | ≤ 0.1 |
| **Qué mide** | Cuantifica el nivel de concentración de las inversiones del portafolio en emisores internacionales. Valores bajos indican mayor diversificación internacional. |
| **Justificación** | Un HHI bajo, por ejemplo, menor a 0.1, refleja que las inversiones están bien distribuidas a nivel internacional, mitigando riesgos geopolíticos, regulatorios o económicos localizados. |
| **Justificación normativa** | Las entidades reguladas deberán demostrar mediante estudios técnicos que las decisiones de inversión en activos extranjeros no generan concentraciones inadecuadas. (SUPEN, 2024) |
| **Relevancia en FCI** | En fondos de capitalización individual, donde la seguridad y rendimiento sostenible son esenciales, una alta diversificación geográfica es clave para proteger a los afiliados de choques externos en mercados específicos. |

**Fórmulas originales de la Tabla 32**

**Fórmula 1:**

$$
{HHI}_{ext} = \sum_{i = 1}^{n}\left( s_{i} \right)^{2}
$$

**Fórmula 2:**

$$
\frac{Inversi{ón}_{i}}{\sum{inversion}_{i}}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 33 se detalla el Índice de Herfindahl-Hirschman de concentración nacional, el cual mide la concentración de inversiones en emisores nacionales. En fondos de capitalización individual, una alta diversificación nacional es crucial para proteger a los afiliados de crisis sectoriales.

### Tabla 33. *Índice de Herfindahl-Hirschman de concentración nacional*

| **Nombre** | Índice de *Herfindahl-Hirschman* de concentración nacional |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>*s<sub>i</sub>*:​ proporción de inversión externa, es decir, si=**[Fórmula 2; véase debajo de la tabla]**<br>*n*: número total de emisores nacionales en las que está invertido el portafolio |
| **Umbral sugerido** | ≤ 0.1 |
| **Qué mide** | Cuantifica el nivel de concentración de las inversiones del portafolio en emisores nacionales. Valores bajos indican mayor diversificación nacional. |
| **Justificación** | Una alta concentración, o sea un valor alto del HHI, indica que el portafolio depende excesivamente de unos pocos emisores, aumentando el riesgo ante shocks específicos del mercado. Un valor bajo del HHI sugiere una mejor dispersión del riesgo. |
| **Justificación normativa** | La operadora deberá estimar indicadores de concentración de cartera como el índice de Herfindahl-Hirschman (IHH). (SUPEN, 2018) |
| **Relevancia en FCI** | Para los fondos de capitalización individual, que no cuentan con esquemas de subsidio cruzado, una alta concentración puede hacerlos más vulnerables a crisis sectoriales, afectando la seguridad del afiliado. Este indicador permite prevenir esos riesgos desde el diseño del portafolio. |

**Fórmulas originales de la Tabla 33**

**Fórmula 1:**

$$
{HHI}_{nac} = \sum_{i = 1}^{n}\left( s_{i} \right)^{2}
$$

**Fórmula 2:**

$$
\frac{Inversi{ón}_{i}}{\sum{inversion}_{i}}
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador descrito en la Tabla 34, mide el cambio porcentual anual de la rentabilidad del fondo. Un crecimiento menor al 1% puede reflejar debilidad estructural o efectos negativos del entorno macroeconómico.

### Tabla 34. *Indicador Tasa de Crecimiento de la Rentabilidad*

| **Nombre** | Tasa de Crecimiento de la Rentabilidad |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>*R<sub>t</sub>*: rentabilidad de la entidad en el mes actual<br>*R<sub>t−1​</sub>*: rentabilidad del fondo en el mismo mes del año anterior |
| **Umbral sugerido** | <0.01 o <1% |
| **Qué mide** | El cambio porcentual anual de la rentabilidad del fondo. Permite evaluar si el desempeño está mejorando o deteriorándose. |
| **Justificación** | Una tasa de crecimiento anual de rentabilidad menor al 1% puede reflejar debilidad estructural o efectos negativos del entorno macroeconómico, afectando la competitividad del fondo frente a sus pares. |
| **Justificación normativa** | Debe evaluarse la tasa de crecimiento acumulada de rentabilidad para verificar cumplimiento de metas. (SUPEN, 2024) |
| **Relevancia en FCI** | En fondos de capitalización individual, el crecimiento sostenido de la rentabilidad es fundamental para maximizar los ahorros del afiliado. La disminución progresiva puede afectar la percepción de estabilidad y rentabilidad esperada a largo plazo. |

**Fórmulas originales de la Tabla 34**

**Fórmula 1:**

$$
{TCR}_{t} = \left( \frac{R_{t} - R_{t - 1}}{R_{t - 1}} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador detallado en la Tabla 35, mide la proporción de nuevos afiliados que se unen a un fondo mediante libre transferencia. Si esta tasa supera el umbral indicado puede sugerir que se está atrayendo afiliados de otros fondos, lo que refleja su competitividad y posibles mejoras en rentabilidad, costos o servicios.

### Tabla 35. *Indicador Tasa de Captación por Libre Transferencia*

| **Nombre** | Tasa de Captación por Libre Transferencia |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]**​: número de aportantes que ingresaron por libre transferencia<br>**[Fórmula 3; véase debajo de la tabla]**​: número total de aportantes. |
| **Umbral sugerido** | >0.001 o >0.1% |
| **Qué mide** | La proporción de nuevos afiliados que ingresan a un fondo mediante libre transferencia, respecto al total de aportantes. |
| **Justificación** | Una tasa superior al 0.1% indica que el fondo está atrayendo afiliados desde otros fondos, reflejando competitividad y posible mejora en percepción de rentabilidad, costos o servicios. |
| **Justificación normativa** | Debe considerarse la captación neta de afiliados en el largo plazo como insumo para definir la política de inversión. (SUPEN, 2024) |
| **Relevancia en FCI** | En un fondo de capitalización individual, este indicador es clave para monitorear la atracción efectiva de afiliados desde competidores. Un valor creciente puede estar asociado a ventajas comparativas sostenibles del fondo. |

**Fórmulas originales de la Tabla 35**

**Fórmula 1:**

$$
TCLT = \left( \frac{{Aportantes}_{entrantesLT}}{TotalAportantes} \right)*100
$$

**Fórmula 2:**

$$
{Aportantes}_{entrantesLT}
$$

**Fórmula 3:**

$$
TotalAportantes
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

La Tasa de Aportación Activa es un indicador esencial que mide el porcentaje de afiliados que realizan aportes en comparación con el total de afiliados registrados. Una tasa inferior al 85% puede indicar problemas en la continuidad de aportes y la sostenibilidad del sistema de capitalización individual (ver Tabla 67).

### Tabla 36. *Indicador Tasa de Aportación Activa*

| **Nombre** | Tasa de Aportación Activa |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]**: número de afiliados que están aportando activamente en la entidad.<br>**[Fórmula 3; véase debajo de la tabla]**​: total de aportantes registrados en el fondo |
| **Umbral sugerido** | < 0.85 o < 85% |
| **Qué mide** | El porcentaje de afiliados que efectivamente están realizando aportes, respecto al total de afiliados registrados. |
| **Justificación** | Una tasa por debajo del 85% puede indicar problemas en la continuidad de aportes, informalidad laboral o debilitamiento en los mecanismos de recaudación del fondo. También podría reflejar un envejecimiento de la base activa o una alta proporción de afiliados no activos económicamente. |
| **Justificación normativa** | El monto de los beneficios dependerá del saldo acumulado en la cuenta de capitalización individual. (SUPEN, 2010) |
| **Relevancia en FCI** | En un sistema de capitalización individual, la regularidad de los aportes es esencial para la acumulación de patrimonio. Una baja tasa de aportación puede afectar la sostenibilidad del sistema y disparar alertas sobre posibles brechas en la protección previsional futura. |

**Fórmulas originales de la Tabla 36**

**Fórmula 1:**

$$
TasaAportacion = \left( \frac{{Aportantes}_{entidad}}{TotalAportantes} \right)*100
$$

**Fórmula 2:**

$$
{Aportantes}_{entidad}
$$

**Fórmula 3:**

$$
TotalAportantes
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El indicador detallado en la Tabla 37, mide la diferencia entre el valor máximo y el valor mínimo del valor cuota durante los últimos 30 días. Este rango de fluctuación es crucial para identificar fondos con comportamiento riesgoso o inestable, ayudando a anticipar decisiones de transferencia o retiro motivadas por la percepción de riesgo.

### Tabla 37. *Indicador Rango Móvil del Valor Cuota*

| **Nombre** | Rango Móvil del Valor Cuota |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>*VC<sub>t</sub>* ​: valor cuota diario del fondo en el día t de los últimos 30 días |
| **Umbral sugerido** | >0.02 (equivalente a más del 2% de variación de los últimos 30 días) |
| **Qué mide** | La diferencia entre el valor máximo y el valor mínimo del valor cuota durante los últimos 30 días. Indica el rango de fluctuación del fondo. |
| **Justificación** | Un rango móvil amplio puede indicar una alta volatilidad en el valor de la cuota, lo que sugiere comportamientos erráticos o sensibles a condiciones de mercado. Ayuda a identificar fondos con comportamiento riesgoso o inestable. |
| **Justificación normativa** | Debe evaluarse el desvío del valor cuota respecto a la trayectoria declarada en el fondo (SUPEN, 2024). |
| **Relevancia en FCI** | En fondos de capitalización individual, una alta variabilidad en el valor cuota puede generar desconfianza entre los afiliados. Este indicador ayuda a anticipar decisiones de transferencia o retiro motivadas por percepción de riesgo. |

**Fórmulas originales de la Tabla 37**

**Fórmula 1:**

$$
{Rangol}_{VC} = max({VC}_{t}) - min({VC}_{t}),\ para\ t\  \in \ últimos\ 30\ días
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

En la Tabla 38 se detalla el indicador que evalúa el nivel de diversificación del portafolio, penalizando la concentración en un único activo.

### Tabla 38. *Indicador de Diversificación y Riesgo Relativo*

| **Nombre** | Diversificación y Riesgo Relativo |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>Donde:<br>Activo más grande: monto invertido en el emisor con mayor peso en el portafolio.<br>TotalActivos: suma de todas las inversiones del portafolio. |
| **Umbral sugerido** | >0.75 |
| **Qué mide** | Evalúa el nivel de diversificación del portafolio penalizando la concentración en un único activo. Un valor cercano a 1 implica buena diversificación. |
| **Justificación** | Si el portafolio depende fuertemente de un solo activo, está expuesto a mayor riesgo específico. Este indicador permite valorar qué tan repartida está la inversión. Es útil como complemento de métricas más complejas como la entropía de Shannon. |
| **Justificación normativa** | Deben evaluarse los efectos de la diversificación sobre la exposición total al riesgo. (SUPEN, 2018) |
| **Relevancia en FCI** | En fondos de capitalización individual, la diversificación es clave para mitigar riesgos. Este indicador ayuda a proteger los intereses de los afiliados ante pérdidas concentradas. |

**Fórmulas originales de la Tabla 38**

**Fórmula 1:**

$$
DiversificacionyRiesgo = \left( \frac{{\sum Activo\ mas\ grande}_{}}{\sum TotalActivos} \right)*100
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

La Tabla 39 detalla el indicador Beta que indica cuánto varía el fondo respecto a los cambios en el mercado general. Mantener la beta por debajo de un umbral determinado es crucial para controlar el riesgo sistemático del portafolio y proteger los intereses de los afiliados.

### Tabla 39. *Indicador Beta del Fondo (Sensibilidad al Mercado)*

| **Nombre** | Beta del Fondo (Sensibilidad al Mercado) |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]**: Rentabilidad del fondo.<br>**[Fórmula 3; véase debajo de la tabla]**: Rentabilidad del índice de referencia del mercado.<br>*Cov*: Covarianza.<br>*Var*: Varianza. |
| **Umbral sugerido** | <1.2 |
| **Qué mide** | La sensibilidad del fondo a los movimientos del mercado. Indica cuánto varía el fondo respecto a los cambios en el mercado general. |
| **Justificación** | Un valor de **[Fórmula 4; véase debajo de la tabla]** mayor a 1 implica mayor volatilidad respecto al mercado. Mantener el **[Fórmula 5; véase debajo de la tabla]** por debajo de 1.2 busca controlar el riesgo sistemático del portafolio. |
| **Justificación normativa** | Deben estimarse métricas que reflejen la sensibilidad del portafolio ante los movimientos del mercado de referencia. (SUPEN, 2018) |
| **Relevancia en FCI** | Fondos con un alta **[Fórmula 6; véase debajo de la tabla]** pueden sufrir pérdidas amplificadas en caídas del mercado. Para los FCI, es importante mantener volatilidades moderadas para proteger los intereses de los afiliados. |

**Fórmulas originales de la Tabla 39**

**Fórmula 1:**

$$
\beta = \frac{\text{Cov}\left( R_{\text{fondo}},R_{\text{mercado}} \right)}{\text{Var}\left( R_{\text{mercado}} \right)}
$$

**Fórmula 2:**

$$
R_{\text{fondo}}
$$

**Fórmula 3:**

$$
R_{\text{mercado}}
$$

**Fórmula 4:**

$$
\beta
$$

**Fórmula 5:**

$$
\beta
$$

**Fórmula 6:**

$$
\beta
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

El Alpha de Jensen mide si un fondo genera una rentabilidad superior o inferior a la esperada dada su exposición al riesgo de mercado, tal como lo detalla la Tabla 40.

### Tabla 40. *Indicador Alpha de Jensen*

| **Nombre** | Alpha de Jensen |
| --- | --- |
| **Fórmula** | **[Fórmula 1; véase debajo de la tabla]**<br>donde:<br>**[Fórmula 2; véase debajo de la tabla]**: Rentabilidad del fondo.<br>**[Fórmula 3; véase debajo de la tabla]**: Rentabilidad del índice de referencia del mercado.<br>**[Fórmula 4; véase debajo de la tabla]** :es la tasa libre de riesgo. |
| **Umbral sugerido** | < 0 |
| **Qué mide** | Si un fondo genera una rentabilidad superior o inferior a la esperada dada su exposición al riesgo de mercado. Un **[Fórmula 5; véase debajo de la tabla]** positivo sugiere que el fondo ha superado el rendimiento ajustado por riesgo esperado; un **[Fórmula 6; véase debajo de la tabla]** negativo sugiere un desempeño inferior. |
| **Justificación** | Un **[Fórmula 7; véase debajo de la tabla]** menor a 0 indica que el fondo no está compensando adecuadamente el riesgo asumido frente al mercado. Para supervisión previsional en régimen de FCI, un **[Fórmula 8; véase debajo de la tabla]** negativo prolongado puede ser signo de gestión ineficiente, riesgo de pérdida de confianza de afiliados, o necesidad de revisar estrategias de inversión. |
| **Justificación**<br>**normativa** | Las operadoras deben considerar modelos que permitan separar el rendimiento sistemático del específico para evaluar gestores de inversión. (SUPEN, 2018) |
| **Relevancia en FCI** | En fondos de capitalización individual, donde cada afiliado asume el desempeño de su cuenta, es crucial que el rendimiento esté alineado o supere el riesgo asumido. Un **[Fórmula 9; véase debajo de la tabla]** negativo persistente puede derivar en fugas de afiliados, presión sobre las reservas y deterioro de la reputación del fondo. |

**Fórmulas originales de la Tabla 40**

**Fórmula 1:**

$$
\alpha = R_{\text{fondo}} - \left( R_{\text{rf}} + \beta\left( R_{\text{mercado}} - R_{\text{rf}} \right) \right)
$$

**Fórmula 2:**

$$
R_{\text{fondo}}
$$

**Fórmula 3:**

$$
R_{\text{mercado}}
$$

**Fórmula 4:**

$$
R_{\text{rf}}
$$

**Fórmula 5:**

$$
\alpha
$$

**Fórmula 6:**

$$
\alpha
$$

**Fórmula 7:**

$$
\alpha
$$

**Fórmula 8:**

$$
\alpha
$$

**Fórmula 9:**

$$
\alpha
$$

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

## Anexo C. Series de datos por agente y módulo

En este anexo, se detallan los indicadores y series de datos que conforman los conjuntos de datos de cada uno de los módulos que componen los subagentes del agente de riesgo.

- **Subagente de Riesgo y liquidez**

Este agente utiliza indicadores diseñados para medir la capacidad de los fondos de cumplir con sus obligaciones de corto plazo, la exposición a activos no líquidos y la concentración en el portafolio. Los insumos incluyen medidas de liquidez, volatilidad y rentabilidad, así como flujos operativos, como se aprecia en las tablas Tabla **72**, Tabla **73** y Tabla **74**. Estos datos permiten alimentar modelos de clasificación, detección de anomalías y predicción del deterioro del activo neto.

### Tabla 41. *Datos para* *Clustering de riesgo de liquidez*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Clustering de riesgo de liquidez<br>Objetivo: Clasificar los fondos según su riesgo de liquidez: bajo, moderado o alto. | IndicadorRiesgoLiquidez |
| IndicadorIndiceLiquidezPortafolio |  |
| IndicadorDiversificacionShannon |  |
| IndicadorConcentracionActivos |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorHerfindahlHirschmanExterno |  |
| IndicadorVolatilidad30Dias |  |
| IndicadorVolatilidadRecienteActivoNeto |  |
| rentabilidad (REAL, MENSUAL) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 42. *Datos para* *Detección de anomalías de liquidez*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Detección de anomalías de liquidez.<br>Objetivo: Identificar fondos con comportamientos inusuales en términos de liquidez. | IndicadorRiesgoLiquidez |
| IndicadorIndiceLiquidezPortafolio |  |
| IndicadorDiversificacionShannon |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorConcentracionActivos |  |
| IndicadorVolatilidad30Dias |  |
| Montocolones(ingreso) |  |
| Montocolones(gastos) |  |
| Montocolones(activo_neto) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 43. *Datos para* *Forecasting de liquidez del activo neto*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting de liquidez del activo neto<br>Objetivo: Predecir si el activo neto disminuirá hasta niveles críticos en los próximos 12-24 meses. | IndicadorVariacionActivoNeto |
| IndicadorRentabilidadAcumuladaMes |  |
| IndicadorVolatilidadRecienteActivoNeto |  |
| IndicadorVolatilidad30Dias |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorBeta |  |
| IndicadorRiesgoLiquidez |  |
| IndicadorIndiceLiquidezPortafolio |  |
| IndicadorTasaFugaAfiliados |  |
| Montocolones(gastos) |  |
| Montocolones(activo_neto) |  |
| rentabilidad (REAL, MENSUAL) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

- **Subagente de Diversificación**

Los conjuntos de datos definidos para este agente incluyen métricas estructurales sobre la distribución del portafolio, tales como índices de concentración y diversificación, junto con indicadores de liquidez y desempeño ajustado por riesgo, como se aprecia en las tablasTabla **75** y Tabla **76**. Estos insumos permiten analizar la solidez del portafolio de inversión y explorar posibles mejoras en la distribución de activos.

### Tabla 44. *Datos para* *Análisis de diversificación de portafolio*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Análisis de diversificación de portafolio<br>Objetivo: Determinar si la diversificación del portafolio es suficiente para mitigar riesgos financieros, y evaluar exposición a activos de baja liquidez o alto riesgo. | IndicadorDiversificacionShannon |
| IndicadorConcentracionActivos |  |
| IndicadorSharpeRatio |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorHerfindahlHirschmanExterno |  |
| IndicadorHerfindahlHirschmanLocal |  |
| IndicadorIndiceLiquidezPortafolio |  |
| Montocolones(portafolio) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 45. *Datos para* *Optimización de portafolios líquidos*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Optimización de portafolios líquidos.<br>Objetivo: Recomendar ajustes en la composición del portafolio para mejorar la liquidez del fondo. | IndicadorIndiceLiquidezPortafolio |
| IndicadorRiesgoLiquidez |  |
| IndicadorConcentracionActivos |  |
| IndicadorBeta |  |
| IndicadorSharpeRatio |  |
| IndicadorDiversificacionShannon |  |
| Montocolones(portafolio) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

- **Subagente de Desequilibrio Financiero**

Este agente emplea datos que reflejan las condiciones de equilibrio operativo y financiero de los fondos, integrando variables como aportes, fugas, rentabilidad, variación del valor cuota y sostenibilidad. Las tablas Tabla **77**, Tabla **78**, Tabla **79** y Tabla **80** muestran los insumos que identifican patrones de riesgo, comportamientos atípicos y trayectorias que anticipen desequilibrios relevantes.

### Tabla 46. *Datos para* *Clustering de riesgo financiero*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Clustering de riesgo financiero<br>Objetivo: Agrupar fondos según patrones de riesgo (bajo, medio, alto). | IndicadorTasaFugaAfiliados |
| IndicadorTasaCaptacionAfiliados |  |
| IndicadorDistribucionAfiliadosEntidadFondo |  |
| IndicadorFlujoNetoAfiliados |  |
| IndicadorVolatilidad30Dias |  |
| IndicadorRangoMensualValorCuota |  |
| IndicadorSharpeRatio |  |
| IndicadorTasaAportación |  |
| IndicadorTasaCrecimientoRentabilidad |  |
| IndicadorIndiceEquilibrioFinanciero |  |
| IndicadorTasaFugaAfiliados |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 47. *Datos para* *Detección de Anomalías Financieras*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Detección de Anomalías Financieras.<br>Detectar fondos con comportamientos financieros atípicos. | IndicadorIndiceSostenibilidadFinanciera |
| IndicadorIndiceSostenibilidadPensiones |  |
| IndicadorIndiceEnvejecimiento |  |
| IndicadorProporcionPensionadosAfiliados |  |
| IndicadorRangoMensualValorCuota |  |
| IndicadorSharpeRatio |  |
| IndicadorTasaAportación |  |
| IndicadorTasaCrecimientoRentabilidad |  |
| IndicadorTasaCrecimientoPensionados |  |
| IndicadorTasaCrecimientoAfiliados |  |
| Montocolones(ingreso) |  |
| Montocolones(gastos) |  |
| Montocolones(activo_neto) |  |
| Saldo_o_aportecolones(afiliado) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 48. *Datos para* *Forecasting Financiero de activo neto*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting Financiero de activo neto.<br>Objetivo: Predecir si el activo neto aumentará o disminuirá en los próximos 6-12 meses | IndicadorVariacionActivoNeto |
| IndicadorRentabilidadAcumuladaMes |  |
| IndicadorVolatilidadRecienteActivoNeto |  |
| IndicadorVolatilidad30Dias |  |
| IndicadorRentabilidadDiaria |  |
| IndicadorIndiceEquilibrioFinanciero |  |
| IndicadorSharpeRatio |  |
| IndicadorRangoMensualValorCuota |  |
| IndicadorBeta |  |
| IndicadorAlphaJensen |  |
| IndicadorTasaAportacion |  |
| IndicadorTasaCrecimientoRentabilidad |  |
| IndicadorRentabilidadAjustadaComision |  |
| IndicadorRiesgoLiquidez |  |
| Montocolones(activo_neto) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 49. *Datos para* *Forecasting del valor cuota*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting valor cuota<br>Objetivo: Predecir el valor futuro de la cuota de un fondo específico, usando como variables explicativas indicadores técnicos derivados del comportamiento histórico diario del fondo. | IndicadorRangoMensualValorCuota |
| IndicadorRentabilidadDiaria |  |
| IndicadorVolatilidad30Dias |  |
| Montocolones(valor de la cuota) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

- **Subagente de flujos de fondos**

Para modelar las dinámicas de migración de afiliados, este agente utiliza indicadores demográficos y financieros, así como registros de transferencias entre fondos. Las tablas Tabla **81**, Tabla **82** y Tabla **83** detallan los insumos que permiten alimentar los modelos de clustering y forecasting, orientados a entender patrones de fuga, captación o estabilidad.

### Tabla 50. *Datos para* *Clustering de flujos de afiliados*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Clustering de flujos de afiliados<br>Objetivo: Identificar patrones de migración de afiliados entre fondos, clasificando según fuga, captación o estabilidad. | IndicadorTasaFugaAfiliados |
| IndicadorFlujoNetoAfiliados |  |
| IndicadorDistribucionAfiliadosEntidadFondo |  |
| IndicadorRiesgoLiquidez |  |
| IndicadorIndiceLiquidezPortafolio |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorFlujoNetoAfiliadosMonto |  |
| IndicadorTasaCaptacionAfiliadosLT |  |
| IndicadorConcentracionActivos |  |
| Afiliado(aportantes) |  |
| Libretransferencia(cantidad_saliente) |  |
| Libretransferencia(cantidad_entrante) |  |
| Libretransferencia(flujo_neto) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 51. *Datos para* *Forecasting de fuga de afiliados*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting de fuga de afiliados<br>Objetivo: Predecir si un fondo perderá o retendrá afiliados en los próximos meses. | IndicadorTasaFugaAfiliados |
| IndicadorTasaCaptacionAfiliados |  |
| IndicadorFlujoNetoAfiliados |  |
| IndicadorDistribucionAfiliadosEntidadFondo |  |
| IndicadorVolatilidad30Dias |  |
| IndicadorFlujoNetoAfiliadosMonto |  |
| IndicadorRangoMensualValorCuota |  |
| IndicadorTasaAportación |  |
| IndicadorTasaCaptacionAfiliadosLT |  |
| IndicadorTasaCrecimientoAfiliados |  |
| IndicadorProporcionPensionadosAfiliados |  |
| Afiliado(aportantes) |  |
| Afiliado(salario_o_aportecolones) |  |
| rentabilidad (REAL, MENSUAL) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 52. *Datos para la Relación entre rentabilidad y flujos*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Relación entre rentabilidad y flujos<br>Objetivo: Evaluar si la pérdida de afiliados se relaciona con la rentabilidad neta real del fondo. | IndicadorRentabilidadAjustadaComision |
| IndicadorRentabilidadAcumuladaMes |  |
| IndicadorFlujoNetoAfiliadosMonto |  |
| IndicadorRangoMensualValorCuota |  |
| IndicadorTasaCaptacionAfiliadosLT |  |
| IndicadorTasaCrecimientoRentabilidad |  |
| IndicadorSharpeRatio |  |
| IndicadorFlujoNetoAfiliados |  |
| rentabilidad (REAL, MENSUAL) |  |
| montocolones(valorcuota – promedio) |  |
| comisión(SALDO) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

- **Subagente de Sostenibilidad**

Las tablas Tabla **84**, Tabla **85** y Tabla **86** puntualizan como el agente de sostenibilidad integra variables demográficas, operativas y de desempeño financiero, con el objetivo de proyectar el envejecimiento de la base afiliada, estimar déficits futuros y agrupar fondos por su grado de sostenibilidad. Los insumos de datos seleccionados permiten construir modelos predictivos y segmentaciones estratégicas para la supervisión regulatoria.

### Tabla 53. *Datos para* *Forecasting demográfico*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting demográfico<br>Objetivo: Proyectar el envejecimiento de la base afiliada y anticipar tensiones por retiros o maduración del fondo. | IndicadorIndiceEnvejecimiento |
| IndicadorProporcionPensionadosAfiliados |  |
| IndicadorTasaCrecimientoAfiliados |  |
| IndicadorTasaCrecimientoPensionados |  |
| Afiliado(aportantes, > a 59 años) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 54. *Datos para* *Forecasting de déficit financiero*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Forecasting de déficit financiero<br>Objetivo: Proyectar si el fondo enfrentará presión en su equilibrio entre aportes y beneficios, desde una lógica operativa y de eficiencia en FCI. | IndicadorIndiceSostenibilidadFinanciera |
| IndicadorRentabilidadAjustadaComision |  |
| IndicadorTasaAportación |  |
| IndicadorVariacionActivoNeto |  |
| IndicadorRentabilidadAcumuladaMes |  |
| IndicadorTasaCrecimientoAfiliados |  |
| IndicadorTasaCrecimientoPensionados |  |
| IndicadorIndiceEquilibrioFinanciero |  |
| Afiliado(salario_o_aporte_colones) |  |
| Beneficios(beneficiocolones(promedio)) |  |
| Montocolones(Ingreso) |  |
| Montocolones(egreso) |  |
| Afiliado(aportantes) |  |
| Beneficio(beneficios) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*

### Tabla 55. *Datos para* *Clustering de sostenibilidad*

| **Módulo** | **Indicador/serie** |
| --- | --- |
| Clustering de sostenibilidad<br>Objetivo: Agrupar fondos según patrones de sostenibilidad operativa, demográfica y de eficiencia, útil para segmentar gestión y regulación. | IndicadorIndiceSostenibilidadFinanciera |
| IndicadorIndiceSostenibilidadFinanciera |  |
| IndicadorIndiceSostenibilidadPensiones |  |
| IndicadorDiversificacionShannon |  |
| IndicadorVolatilidadRecienteActivoNeto |  |
| IndicadorDiversificacionYRiesgo |  |
| IndicadorProporcionPensionadosAfiliados |  |
| IndicadorHerfindahlHirschmanExterno |  |
| IndicadorBeta |  |
| IndicadorIndiceCargaAdministrativa |  |
| rentabilidad (REAL, MENSUAL) |  |
| montocolones(activoneto) |  |
| afiliado(aportantes) |  |

*Fuente: Elaboración propia con base en recursos del sitio de la SUPEN*
