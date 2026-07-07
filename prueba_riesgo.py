from EDArecomend_agente.logica_EDA import AgenteEDARecomendador
from datetime import datetime

# Rango de fechas (inclusive)
fecha_inicio = datetime.strptime("2012-01-31", "%Y-%m-%d")
fecha_final = datetime.strptime("2024-12-31", "%Y-%m-%d")
fondo = "ROP"

# Paso 2: Crear el agente recomendador
agente_eda = AgenteEDARecomendador()

# Ejecutar mes a mes
fecha_actual = fecha_inicio
resultados = agente_eda.ejecutar_EDA(fecha_inicio, fecha_final,fondo)
print(resultados)


