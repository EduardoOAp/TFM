# Importación principal del agente de procesamiento
#from .logica_procesamiento import AgenteProcesamiento
from procesamiento_agente.registro_indicadores import REGISTRO_INDICADORES
print("[INFO] Indicadores activos:")
for config in REGISTRO_INDICADORES.values():
    if config.get("activo", True):
        print(f"  - {config['clase'].__name__}")