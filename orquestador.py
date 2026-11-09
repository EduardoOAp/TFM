import asyncio 
import requests
import os
import time
from typing import Optional
from datetime import datetime, timedelta
import glob
import pandas as pd
from .config import (
    DATA_ROOT, SCRAPER_API_URL, PROCESAMIENTO_API_URL,
    EDA_API_URL, RIESGO_API_URL, TIPOS, FONDO, FECHA_INICIAL_DEFAULT
)

# Importa tus módulos locales que simularán las APIs

from . import module_scraper as scraper_module
from . import module_procesamiento as procesamiento_module
from . import module_EDA as eda_module
from . import module_riesgo as riesgo_module

def obtener_ultima_fecha_cargada(tipo: str, fondo: str) -> Optional[datetime]:
    carpeta_base = os.path.join("multiagente",DATA_ROOT, fondo, tipo)
    patrones = os.path.join(carpeta_base, "*", "*", "*", "data.parquet")
    archivos = glob.glob(patrones)
    if not archivos:
        return None

    fechas = []
    for archivo in archivos:
        try:
            partes = archivo.split(os.sep)
            anio, mes, dia = map(int, partes[-4:-1])
            fechas.append(datetime(anio, mes, dia))
        except Exception as e:
            print(f"[WARN] No se pudo extraer fecha de {archivo}: {e}")
            continue
    return max(fechas) if fechas else None

def obtener_fecha_inicio_mas_reciente(fondo: str) -> datetime:
    fechas = []
    for tipo in TIPOS:
        fecha = obtener_ultima_fecha_cargada(tipo, fondo)
        if fecha:
            fechas.append(fecha)
    return max(fechas) + timedelta(days=1) if fechas else FECHA_INICIAL_DEFAULT

def llamada_con_reintentos(url: str, params: dict, max_reintentos: int = 3, timeout: int = 10):
    for intento in range(1, max_reintentos + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"[ERROR] Intento {intento} fallido al llamar {url}: {type(e).__name__}: {e}")
            if intento < max_reintentos:
                time.sleep(2 ** intento)
    return None

class AgenteOrquestador:
    def __init__(self, fondo=FONDO, ejecutar_scraper=True, ejecutar_procesamiento=True,
                 ejecutar_eda=True, ejecutar_riesgo=True):
        self.fondo = fondo
        self.ejecutar_scraper = ejecutar_scraper
        self.ejecutar_procesamiento = ejecutar_procesamiento
        self.ejecutar_eda = ejecutar_eda
        self.ejecutar_riesgo = ejecutar_riesgo
        self.fecha_inicio, self.fecha_final = self.calcular_fechas_automaticas()

    def calcular_fechas_automaticas(self):
        fechas = []
        for tipo in TIPOS:
            fecha = obtener_ultima_fecha_cargada(tipo, self.fondo)
            if fecha:
                fechas.append(fecha)
        fecha_inicio = min(fechas) + timedelta(days=1) if fechas else FECHA_INICIAL_DEFAULT
        fecha_final = datetime.today()
        return fecha_inicio.strftime("%Y-%m-%d"), fecha_final.strftime("%Y-%m-%d")

    def ejecutar(self):
        fecha_inicio = obtener_fecha_inicio_mas_reciente(self.fondo)
        fecha_actual = datetime.today()
        fecha_cursor = datetime(fecha_inicio.year, fecha_inicio.month, 1)

        while fecha_cursor <= fecha_actual:
            inicio_mes = fecha_cursor
            fin_mes = (inicio_mes + pd.DateOffset(months=1) - timedelta(days=1)).to_pydatetime()
            print(f"\n[ORQUESTADOR] Ejecutando ciclo: {inicio_mes.date()} -> {fin_mes.date()}")

            cambios_detectados = False
            if self.ejecutar_scraper:
                for tipo in TIPOS:
                    print(f"[ORQUESTADOR] Consultando tipo '{tipo}' desde API scraper...")
                    response = llamada_con_reintentos(
                        f"{SCRAPER_API_URL}{tipo}",
                        params={
                            "fondo": self.fondo,
                            "fecha_inicio": inicio_mes.strftime("%Y-%m-%d"),
                            "fecha_final": fin_mes.strftime("%Y-%m-%d"),
                            "guardar_parquet": True
                        }
                    )
                    if response:
                        json_data = response.json()
                        if json_data.get("cambio_detectado"):
                            cambios_detectados = True
                            print(f"[ORQUESTADOR] Cambio detectado para {tipo}.")
                        else:
                            print(f"[ORQUESTADOR] Sin cambios para {tipo}.")
                    else:
                        print(f"[ERROR] No se obtuvo respuesta válida del scraper para tipo {tipo}.")

            if cambios_detectados or self.ejecutar_procesamiento:
                print("[ORQUESTADOR] Ejecutando procesamiento.")
                self.invocar_api_procesamiento(inicio_mes, fin_mes)

            if cambios_detectados or self.ejecutar_eda:
                print("[ORQUESTADOR] Ejecutando EDA.")
                self.invocar_api_EDA(inicio_mes, fin_mes)

            if cambios_detectados or self.ejecutar_riesgo:
                print("[ORQUESTADOR] Ejecutando Riesgo.")
                self.invocar_api_Riesgo(inicio_mes)

            fecha_cursor = (fecha_cursor + pd.DateOffset(months=1)).to_pydatetime()

    def invocar_api_procesamiento(self, fecha_inicio, fecha_final):
        print("[ORQUESTADOR] Invocando API de procesamiento...")
        llamada_con_reintentos(PROCESAMIENTO_API_URL, {
            "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
            "fecha_final": fecha_final.strftime("%Y-%m-%d"),
            "fondo": self.fondo
        })

    def invocar_api_EDA(self, fecha_inicio, fecha_final):
        print("[ORQUESTADOR] Disparando EDA")
        llamada_con_reintentos(EDA_API_URL, {
            "fecha": fecha_inicio.strftime("%Y-%m-%d"),
            "fecha_final": fecha_final.strftime("%Y-%m-%d"),
            "fondo": self.fondo
        })

    def invocar_api_Riesgo(self, fecha: datetime):
        print("[ORQUESTADOR] Disparando Riesgo")
        llamada_con_reintentos(RIESGO_API_URL, {
            "fecha": fecha.strftime("%Y-%m-%d"),
            "entidad": "ninguna",
            "fondo": self.fondo
        })

# --- CLASE DE PRUEBA REAL: AgenteOrquestadorLocal ---
# Esta es la clase que invoca los módulos Python locales

class AgenteOrquestadorLocal(AgenteOrquestador):
    def __init__(self, fondo=FONDO, ejecutar_scraper=True, ejecutar_procesamiento=True,
                 ejecutar_eda=True, ejecutar_riesgo=True):
        super().__init__(fondo, ejecutar_scraper, ejecutar_procesamiento,
                         ejecutar_eda, ejecutar_riesgo)
        print("\n--- INICIANDO AgenteOrquestadorLocal (Modo de prueba local real) ---")
        # Ajusta DATA_ROOT si tus módulos locales necesitan un path diferente para guardar
        # scraper_module.DATA_ROOT_FOR_TEST = "otra_carpeta_de_prueba"

    # Sobreescribimos los métodos de invocación de API para llamar a los módulos locales
    def invocar_api_procesamiento(self, fecha_inicio: datetime, fecha_final: datetime):
        print("[ORQUESTADOR LOCAL] Invocando módulo de procesamiento LOCAL...")
        return procesamiento_module.run_procesamiento(
            fondo=self.fondo,
            fecha_inicio=fecha_inicio.strftime("%Y-%m-%d"),
            fecha_final=fecha_final.strftime("%Y-%m-%d")
        )

    def invocar_api_EDA(self, fecha_inicio: datetime, fecha_final: datetime):
        print("[ORQUESTADOR LOCAL] Disparando módulo EDA LOCAL...")
        return eda_module.run_eda(
            fondo=self.fondo,
            fecha=fecha_inicio.strftime("%Y-%m-%d"),
            fecha_final=fecha_final.strftime("%Y-%m-%d")
        )

    # Sobreescribimos el método de invocación de la API de Riesgo
    async def invocar_api_Riesgo(self, fecha: datetime): # ¡Ahora es async!
        print("[ORQUESTADOR LOCAL] Disparando módulo Riesgo LOCAL...")
        return await riesgo_module.run_riesgo( # ¡await para esperar la corrutina!
            fondo=self.fondo,
            fecha=fecha.strftime("%Y-%m-%d"),
            entidad="ninguna"
        )
            
    async def ejecutar(self):
        # La lógica principal del bucle es la misma, pero la llamada al scraper cambia
        
        fecha_inicio_para_bucle = obtener_fecha_inicio_mas_reciente(self.fondo)
        fecha_actual = datetime.today()
        fecha_cursor = datetime(fecha_inicio_para_bucle.year, fecha_inicio_para_bucle.month, 1)
        # Eliminar luego
        fecha_cursor = datetime(2025,4,1)
        # Eliminar Luego
        while fecha_cursor <= fecha_actual:
            inicio_mes = fecha_cursor
            fin_mes = (inicio_mes + pd.DateOffset(months=1) - timedelta(days=1)).to_pydatetime()
            
            if fin_mes > fecha_actual:
                fin_mes = fecha_actual
            
            print(f"\n[ORQUESTADOR LOCAL] Ejecutando ciclo: {inicio_mes.date()} -> {fin_mes.date()}")

            cambios_detectados = False
            if self.ejecutar_scraper:
                for tipo in TIPOS:
                    print(f"[ORQUESTADOR LOCAL] Consultando tipo '{tipo}' desde módulo scraper LOCAL...")
                    # ¡AQUÍ ES DONDE LLAMAMOS A TU scraper_module.run_scraper DIRECTAMENTE!
                    response_data = scraper_module.run_scraper(
                        tipo=tipo, # Pasamos el tipo explícitamente a la función local
                        fondo=self.fondo,
                        fecha_inicio=inicio_mes.strftime("%Y-%m-%d"),
                        fecha_final=fin_mes.strftime("%Y-%m-%d"),
                        guardar_parquet=True # Siempre guardamos en las pruebas si hay cambios
                    )
                    
                    if response_data: # Asumimos que el módulo local devuelve un dict
                        if response_data.get("cambio_detectado"):
                            cambios_detectados = True
                            print(f"[ORQUESTADOR LOCAL] Cambio detectado para {tipo}.")
                        else:
                            print(f"[ORQUESTADOR LOCAL] Sin cambios para {tipo}.")
                    else:
                        print(f"[ERROR LOCAL] No se obtuvo respuesta válida del scraper local para tipo {tipo}.")

            if cambios_detectados or self.ejecutar_procesamiento:
                print("[ORQUESTADOR LOCAL] Ejecutando procesamiento LOCAL.")
                self.invocar_api_procesamiento(inicio_mes, fin_mes)

            if cambios_detectados or self.ejecutar_eda:
                print("[ORQUESTADOR LOCAL] Ejecutando EDA LOCAL.")
                self.invocar_api_EDA(inicio_mes, fin_mes)

            # Donde llamas a invocar_api_Riesgo, ahora necesitas await
            if cambios_detectados or self.ejecutar_riesgo:
                print("[ORQUESTADOR LOCAL] Ejecutando Riesgo LOCAL.")
                await self.invocar_api_Riesgo(fin_mes) # ¡await aquí!

            fecha_cursor = (fecha_cursor + pd.DateOffset(months=1)).to_pydatetime()
            # Esta condición es crucial para salir del bucle cuando ya se procesó el mes actual
            if inicio_mes.month == fecha_actual.month and inicio_mes.year == fecha_actual.year:
                break

# --- Parte principal para ejecutar el agente de prueba ---
if __name__ == "__main__":
    # Asegúrate de que las carpetas de datos existan
    # ... (código para crear directorios) ...

    orquestador_prueba = AgenteOrquestadorLocal(
        fondo=FONDO,
        ejecutar_scraper=True,
        ejecutar_procesamiento=True,
        ejecutar_eda=True,
        ejecutar_riesgo=True
    )
    # ¡Importante! Si ejecutar() es async, debes usar asyncio.run()
    asyncio.run(orquestador_prueba.ejecutar())

    print("\n--- Agente Orquestador en modo local REAL (usando SupenDataFetcher y subagentes de riesgo) finalizado ---")

# Para real
"""
if __name__ == "__main__":
    orquestador = AgenteOrquestador(
        ejecutar_scraper=True,
        ejecutar_procesamiento=True,
        ejecutar_eda=True,
        ejecutar_riesgo=True
    )
    orquestador.ejecutar()
"""
