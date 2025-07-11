import requests
import pandas as pd
import os
from datetime import datetime
from fastapi import HTTPException
import re
import unicodedata

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
    

def normalizar_entidad(nombre: str) -> str:
    nombre = nombre.upper()
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = nombre.encode("ascii", "ignore").decode("utf-8")
    nombre = re.sub(r"[\s\-_]", "", nombre)  # elimina espacios, guiones, guiones bajos
    return nombre   

def construir_matrices_transferencias(df: pd.DataFrame):
    columnas_c = [col for col in df.columns if col.endswith('_C')]
    columnas_m = [col for col in df.columns if col.endswith('_M')]

    registros = []

    for _, fila in df.iterrows():
        origen = normalizar_entidad(fila['entidadorigen'])

        for col_c in columnas_c:
            entidad_base = col_c.replace('_C', '')
            col_m = f"{entidad_base}_M"
            destino = normalizar_entidad(entidad_base)

            cantidad = fila.get(col_c, 0) or 0
            monto = fila.get(col_m, 0.0) or 0.0

            registros.append({
                "entidad_origen": origen,
                "entidad_destino": destino,
                "cantidad": cantidad,
                "monto": monto
            })

    df_transferencias = pd.DataFrame(registros) 
    matriz_cant = df_transferencias.pivot_table(
        index="entidad_origen", columns="entidad_destino", values="cantidad", aggfunc="sum", fill_value=0
    )

    matriz_monto = df_transferencias.pivot_table(
        index="entidad_origen", columns="entidad_destino", values="monto", aggfunc="sum", fill_value=0
    )

    return matriz_cant, matriz_monto   

def fusionar_con_datos_originales(df_original: pd.DataFrame, resumen: pd.DataFrame) -> pd.DataFrame:
    df_sin_c_m = df_original[[col for col in df_original.columns if not col.endswith(('_C', '_M'))]].copy()
    df_sin_c_m["entidad_norm"] = df_sin_c_m["entidadorigen"].apply(normalizar_entidad)

    resumen_reset = resumen.reset_index().rename(columns={"entidad": "entidad_norm"})
    df_final = pd.merge(df_sin_c_m, resumen_reset, on="entidad_norm", how="left").drop(columns=["entidad_norm"])
    return df_final


def resumen_transferencias_completo(matriz_cant: pd.DataFrame, matriz_monto: pd.DataFrame) -> pd.DataFrame:
    resumen = pd.DataFrame({
        "cantidad_saliente": matriz_cant.sum(axis=1),
        "cantidad_entrante": matriz_cant.sum(axis=0),
        "monto_saliente": matriz_monto.sum(axis=1),
        "monto_entrante": matriz_monto.sum(axis=0),
    })

    resumen = resumen.fillna(0)
    resumen.index.name = "entidad"
    return resumen
    
def transformar_libre_transferencia(df: pd.DataFrame) -> pd.DataFrame:
    matriz_cant, matriz_monto = construir_matrices_transferencias(df)

    # Paso 2: generar resumen por entidad
    resumen = resumen_transferencias_completo(matriz_cant, matriz_monto)

    # Paso 3: fusionar con datos originales (sin columnas *_C y *_M)
    df_resultado = fusionar_con_datos_originales(df, resumen)

    print(df_resultado)
    return df_resultado    

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
            if self.tipo == "libre_transferencia":
                #if "datos" not in data:
                #    raise ValueError("Respuesta JSON inválida: falta 'datos'")
                #df = pd.DataFrame(data["datos"])
                df = pd.DataFrame(data)  
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)                 
                print(df.head(10))
                df = transformar_libre_transferencia(df)  
            else:
                df = pd.DataFrame(data)
            return df
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
