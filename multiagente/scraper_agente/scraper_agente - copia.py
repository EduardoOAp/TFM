import requests
import pandas as pd
import os
from datetime import datetime
from fastapi import HTTPException

API_SCRAPER_URL = "http://localhost:8000/fetch/"
ENDPOINTS = {
    "comision": "https://webapps.supen.fi.cr/Estadisticas/API/comision",
    "rendimiento": "https://webapps.supen.fi.cr/Estadisticas/API/rendimiento",
    "afiliado": "https://webapps.supen.fi.cr/Estadisticas/API/AFILIADO",
    "cuenta": "https://webapps.supen.fi.cr/estadisticas/api/cuenta",
    "portafolio": "https://webapps.supen.fi.cr/estadisticas/api/Portafolio",
    "beneficio": "https://webapps.supen.fi.cr/Estadisticas/API/BENEFICIO",
    "libre_transferencia": "https://webapps.supen.fi.cr/Estadisticas/API/LT"
}

class SupenScraper:
    def __init__(self, tipo: str, fondo: str, fecha_inicio: str, fecha_final: str):
        self.tipo = tipo
        self.fondo = fondo
        self.fecha_inicio = fecha_inicio
        self.fecha_final = fecha_final

    def obtener_datos(self) -> pd.DataFrame:
        params = {
            "fondo": self.fondo,
            "fecha_inicio": self.fecha_inicio,
            "fecha_final": self.fecha_final,
            "guardar_parquet": False
        }
        try:
            response = requests.get(f"{API_SCRAPER_URL}{self.tipo}", params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json()["datos"])
        except Exception as e:
            print(f"[ERROR] Fallo al consultar '{self.tipo}': {e}")
            return pd.DataFrame()

    def ruta_archivo(self) -> str:
        year, month, day = self.fecha_inicio[:4], self.fecha_inicio[5:7], self.fecha_inicio[8:10]
        directory = os.path.join("data", self.fondo, self.tipo, year, month, day)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "data.parquet")

class SupenDataFetcher:
    def __init__(self, tipo: str, fondo: str, fecha_inicio: str, fecha_final: str):
        self.tipo = tipo.lower()
        self.fondo = fondo
        self.fecha_inicio = fecha_inicio
        self.fecha_final = fecha_final
        self.url = self.build_url()

    def build_url(self) -> str:
        if self.tipo not in ENDPOINTS:
            raise ValueError("Tipo de recurso no válido")
        return f"{ENDPOINTS[self.tipo]}?Fondo={self.fondo}&FechaInicio={self.fecha_inicio}&FechaFinal={self.fecha_final}"

    def fetch_json(self) -> pd.DataFrame:
        try:
            print(f"[DEBUG] Llamando a URL SUPEN: {self.url}")
            response = requests.get(self.url)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error al acceder a la API de SUPEN: {str(e)}")
        except ValueError:
            raise HTTPException(status_code=500, detail="Error al interpretar la respuesta JSON")

    def save_to_parquet(self, df: pd.DataFrame) -> str:
        try:
            start_date = datetime.strptime(self.fecha_final, "%Y-%m-%d")
            year = str(start_date.year)
            month = f"{start_date.month:02d}"
            day = f"{start_date.day:02d}"
            directory = os.path.join("data", self.fondo, self.tipo, year, month, day)
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, "data.parquet")
            df.to_parquet(filepath, index=False)
            return filepath
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

    def get_parquet_path(self) -> str:
        # Usamos la fecha_final como referencia de carpeta
        start_date = datetime.strptime(self.fecha_final, "%Y-%m-%d")
        year = str(start_date.year)
        month = f"{start_date.month:02d}"
        day = f"{start_date.day:02d}"
        
        # Creamos el path: data/ROP/tipo/yyyy/mm/dd/data.parquet
        directory = os.path.join("data", self.fondo, self.tipo, year, month, day)
        os.makedirs(directory, exist_ok=True)  # Asegura que los directorios existan
        
        # Ruta completa del archivo parquet
        return os.path.join(directory, "data.parquet")
            
            
    def detectar_cambios_y_guardar(self, nuevo_df: pd.DataFrame) -> tuple[bool, str]:
        filepath = self.get_parquet_path()

        if os.path.exists(filepath):
            try:
                df_anterior = pd.read_parquet(filepath)
                if nuevo_df.sort_index(axis=1).equals(df_anterior.sort_index(axis=1)):
                    print(f"[DEBUG] No hubo cambios en {filepath}")
                    return False, filepath
            except Exception as e:
                print(f"[WARN] Error al comparar archivos anteriores: {e}")

        self.save_to_parquet(nuevo_df)
        print(f"[DEBUG] Guardado nuevo archivo en: {filepath}")
        return True, filepath
