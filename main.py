# main.py
import asyncio
from multiagente import AgenteOrquestador
from multiagente import AgenteOrquestadorLocal

if __name__ == "__main__":
    print("[MAIN] Iniciando el sistema multiagente...")
    orquestador = AgenteOrquestadorLocal(fondo="ROP", 
                    ejecutar_scraper=False, ejecutar_procesamiento=False, 
                    ejecutar_eda=False,ejecutar_riesgo=True)
    asyncio.run(orquestador.ejecutar())
    print("[MAIN] Proceso completado.")
