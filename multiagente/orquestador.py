import requests
import os
from typing import Optional
from datetime import datetime, timedelta
import glob
import pandas as pd
from .config import DATA_ROOT, SCRAPER_API_URL, PROCESAMIENTO_API_URL
from .config import EDA_API_URL, RIESGO_API_URL
from .config import TIPOS, FONDO, FECHA_INICIAL_DEFAULT

#INDICADORES_PATH = "indicadores"
#JSON_SUBAGENTES = "estructura_subagentes.json"
#UMBRAL_JSON_PATH = "umbrales_activacion.json"

def obtener_ultima_fecha_cargada(tipo: str, fondo: str) -> Optional[datetime]:
    carpeta_base = os.path.join(DATA_ROOT, fondo, tipo)
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
        except:
            continue
    return max(fechas) if fechas else None

def obtener_fecha_inicio_mas_reciente(fondo: str) -> datetime:
    fechas = [obtener_ultima_fecha_cargada(tipo, fondo) for tipo in TIPOS]
    fechas_validas = [f for f in fechas if f is not None]
    if fechas_validas:
        return max(fechas_validas) + timedelta(days=1)
    else:
        return FECHA_INICIAL_DEFAULT

class AgenteOrquestador:
    def __init__(self, fondo=FONDO, ejecutar_scraper=False, ejecutar_procesamiento=False, ejecutar_eda=False, ejecutar_riesgo=False):
        self.fondo = fondo
        self.ejecutar_scraper = ejecutar_scraper
        self.ejecutar_procesamiento = ejecutar_procesamiento
        self.ejecutar_eda = ejecutar_eda
        self.ejecutar_riesgo = ejecutar_riesgo
        self.fecha_inicio, self.fecha_final = self.calcular_fechas_automaticas()

    def calcular_fechas_automaticas(self):
        fechas = [obtener_ultima_fecha_cargada(tipo, self.fondo) for tipo in TIPOS if obtener_ultima_fecha_cargada(tipo, self.fondo)]
        if fechas:
            fecha_inicio = min(fechas) + timedelta(days=1)
        else:
            fecha_inicio = datetime.strptime("2020-01-01", "%Y-%m-%d")
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
                    try:
                        response = requests.get(
                            f"{SCRAPER_API_URL}{tipo}",
                            params={
                                "fondo": self.fondo,
                                "fecha_inicio": inicio_mes.strftime("%Y-%m-%d"),
                                "fecha_final": fin_mes.strftime("%Y-%m-%d"),
                                "guardar_parquet": True
                            }
                        )
                        response.raise_for_status()
                        json_data = response.json()
                        if json_data.get("cambio_detectado"):
                            cambios_detectados = True
                            print(f"[ORQUESTADOR] Cambio detectado para {tipo}.")
                        else:
                            print(f"[ORQUESTADOR] Sin cambios para {tipo}.")
                    except Exception as e:
                        print(f"[ERROR] Fallo al consultar el tipo {tipo}: {e}")

            if cambios_detectados or self.ejecutar_procesamiento:
                print("[ORQUESTADOR] Ejecución del procesamiento.")
                self.invocar_api_procesamiento(inicio_mes, fin_mes)
            else:
                print("[ORQUESTADOR] Cambios detectados, pero la ejecución del procesamiento está desactivada.")
            
            if cambios_detectados or self.ejecutar_eda:
                print("[ORQUESTADOR] Ejecución del EDA.")
                self.evaluar_umbral_e_invocar_eda(inicio_mes)
            else:
                print("[ORQUESTADOR] No se ejecuta EDA.")

            if self.ejecutar_riesgo:
                print("[ORQUESTADOR] Ejecución del Riesgo.")
                self.invocar_api_Riesgo(inicio_mes)

            fecha_cursor = (fecha_cursor + pd.DateOffset(months=1)).to_pydatetime()

    def invocar_api_procesamiento(self, fecha_inicio, fecha_final):
        print("[ORQUESTADOR] Invocando API de procesamiento...")
        try:
            response = requests.get(PROCESAMIENTO_API_URL, params={
                "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
                "fecha_final": fecha_final.strftime("%Y-%m-%d"),
                "fondo": self.fondo
            })
            print(f"[ORQUESTADOR] Procesamiento retornó {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Fallo al invocar el procesamiento: {e}")

    def invocar_api_EDA(self, fecha: datetime):
        print(f"[ORQUESTADOR] Disparando EDA")
        try:
            response = requests.get(EDA_API_URL, params={
                        "fecha": fecha.strftime("%Y-%m-%d"),
                        "fondo": self.fondo
                    })        
            print(f"[ORQUESTADOR] EDA retornó {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Fallo al invocar el EDA: {e}")

    def invocar_api_Riesgo(self, fecha: datetime):
        print(f"ORQUESTADOR] Disparando Riesgo")
        try:
            response = requests.get(RIESGO_API_URL, params={
                        "fecha": fecha.strftime("%Y-%m-%d"),
                        "fondo": self.fondo
                    })        
            print(f"[ORQUESTADOR] Riesgo retornó {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Fallo al invocar Riesgo: {e}")

if __name__ == "__main__":
    orquestador = AgenteOrquestador()
    orquestador.ejecutar()
