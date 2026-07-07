from procesamiento_agente.logica_procesamiento import AgenteProcesamiento
from datetime import datetime
from dateutil.relativedelta import relativedelta  # asegúrate de tener instalada esta librería
from pandas.tseries.offsets import MonthEnd

# Rango de fechas (inclusive)
fecha_inicio = datetime.strptime("2009-12-31", "%Y-%m-%d")
fecha_final = datetime.strptime("2025-03-31", "%Y-%m-%d")
fondo = "ROP"

# Ejecutar mes a mes
fecha_actual = fecha_inicio
while fecha_actual <= fecha_final:
    fecha_str = fecha_actual.strftime("%Y-%m-%d")
    # Agrega esto para obtener el último día del mes siguiente
    fecha_siguiente = (fecha_actual + relativedelta(months=1)).replace(day=1) + MonthEnd(0)
    fecha_siguiente_str = fecha_siguiente.strftime("%Y-%m-%d")
    
    print(f"\n[INFO] Procesando para fecha: {fecha_str}")
    agente = AgenteProcesamiento(fondo=fondo, fecha_inicio=fecha_str, fecha_final=fecha_siguiente_str)
    agente.ejecutar()
    
    # Avanzar al último día del mes siguiente
    #fecha_actual = (fecha_actual + relativedelta(months=1)).replace(day=31)
    fecha_actual = (fecha_actual + relativedelta(months=1)).replace(day=1) + MonthEnd(0)

