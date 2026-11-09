from typing import Dict, Any, Optional
from .procesamiento_agente.logica_procesamiento import AgenteProcesamiento
from dateutil.relativedelta import relativedelta  # asegúrate de tener instalada esta librería
from pandas.tseries.offsets import MonthEnd

# Rango de fechas (inclusive)
#fecha_inicio = datetime.strptime("2025-02-28", "%Y-%m-%d")
#fecha_final = datetime.strptime("2025-06-30", "%Y-%m-%d")
#fondo = "ROP"

def run_procesamiento(
    fondo: str,
    fecha_inicio: str,
    fecha_final: str
) -> Optional[Dict[str, Any]]:
    
    # Ejecutar mes a mes
    print(f"    [LOCAL PROCESAMIENTO] Procesando datos para fondo '{fondo}' desde {fecha_inicio} hasta {fecha_final}")
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
    return {"status": "ok", "message": "Datos procesados localmente."}
