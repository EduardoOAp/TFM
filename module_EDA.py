from typing import Dict, Any, Optional
from .EDArecomend_agente.logica_EDA import AgenteEDARecomendador
from datetime import datetime
#import sys
import os
import re
#sys.path.append(os.path.dirname(__file__))
from .EDArecomend_agente.config import JSON_PATH
from .configuracion.cargar_json import cargar_config
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import calendar

def fin_de_mes(fecha: date) -> date:
    """Dado un objeto fecha, devuelve el último día del siguiente mes."""
    # Avanzar un mes
    siguiente_mes = fecha + relativedelta(months=1)
    # Obtener el último día de ese mes
    ultimo_dia = calendar.monthrange(siguiente_mes.year, siguiente_mes.month)[1]
    return date(siguiente_mes.year, siguiente_mes.month, ultimo_dia)

DATA_ROOT = "data"
TEMPORAL_ROOT = "temporal"
CONFIG_FILE_PATH = os.path.abspath(__file__)

BASE_DIR = os.path.dirname(os.path.dirname(CONFIG_FILE_PATH))
JSON_PATH = os.path.join(BASE_DIR, "multiagente", "configuracion", "estructura_subagentes.json")
estructura = cargar_config(JSON_PATH)
for entrada in estructura:
    agente = entrada.get("subagente")
    modulo = entrada.get("modulo")
    break

ruta_guardado = os.path.join("multiagente",
                    "data", "temporal", agente, modulo)
# Listar todos los archivos y carpetas
elementos = os.listdir(ruta_guardado)
patron_fecha = r"\d{4}-\d{2}-\d{2}"
archivos = [f for f in elementos if os.path.isfile(os.path.join(ruta_guardado, f))]
for archivo in archivos:
    match = re.search(patron_fecha, archivo)
    if match:
        fecha_str = match.group()
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()    


def run_EDA(
    fondo: str,
    fecha_inicio: str,
    fecha_final: str
) -> Optional[Dict[str, Any]]:

    # Paso 2: Crear el agente recomendador
    agente_eda = AgenteEDARecomendador()

    # Ejecutar mes a mes
    # fecha_actual = fecha_inicio
    resultados = agente_eda.ejecutar_EDA(fecha_inicio, fecha_final,fondo)
    print(resultados)
    return {"status": "ok", "message": "EDA generada localmente."}    


