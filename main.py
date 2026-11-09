# main.py

from multiagente import AgenteOrquestador

if __name__ == "__main__":
    print("[MAIN] Iniciando el sistema multiagente...")
    orquestador = AgenteOrquestador(fondo="ROP", 
                    ejecutar_scraper=False, ejecutar_procesamiento=False, 
                    ejecutar_eda=False,ejecutar_riesgo=True)
    orquestador.ejecutar()
    print("[MAIN] Proceso completado.")
