from typing import List, Dict, Optional
import pandas as pd
from configuracion.cargar_json import cargar_config

def seleccionar_columnas_objetivo_desde_json(
    subagente: str,
    modulo: str,
    df_columnas: List[str],
    cantidad: int = 1
) -> List[str]:
    """
    Busca un subagente y un módulo en el JSON de configuración y retorna hasta `cantidad`
    columnas objetivo que existan en el DataFrame.

    Parámetros:
    - json_config: lista de agentes (estructura JSON cargada)
    - subagente: nombre exacto del subagente a buscar
    - modulo: nombre exacto del módulo a buscar
    - df_columnas: lista de columnas del DataFrame
    - cantidad: cuántas columnas objetivo válidas retornar como máximo

    Retorna:
    - Lista de columnas objetivo válidas encontradas
    """
    JSON_PATH = "estructura_subagentes.json"
    estructura = cargar_config(JSON_PATH)    
    seleccionadas=[]
    for entrada in estructura:
        if entrada.get("subagente") == subagente and entrada.get("modulo") == modulo:
            columnas_objetivo = entrada.get("columnasObjetivo", [])
            disponibles = [col for col in columnas_objetivo if col in df_columnas]
            seleccionadas = disponibles[:cantidad]
            if seleccionadas:
                print(f"✅ [{subagente} - {modulo}] Columnas objetivo disponibles: {seleccionadas}")
                return seleccionadas
                break
            else:
                print(f"❌ [{subagente} - {modulo}] No hay columnas objetivo válidas en el DataFrame")
            

    print(f"❌ Subagente '{subagente}' con módulo '{modulo}' no encontrado en el JSON.")
    return []
