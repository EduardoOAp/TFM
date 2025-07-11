from fastapi import FastAPI, Query, HTTPException
from typing import Optional
import pandas as pd
from scraper_agente.scraper_agente import SupenDataFetcher

app = FastAPI(title="API Scraper SUPEN", description="Devuelve datos desde SUPEN en formato JSON")

# Ruta base donde están guardados los datos descargados por tipo
DATA_ROOT = "data"

@app.get("/fetch/{tipo}")
def fetch_datos(
    tipo: str,
    fondo: str = Query("ROP"),
    fecha_inicio: str = Query(..., description="Formato YYYY-MM-DD"),
    fecha_final: str = Query(..., description="Formato YYYY-MM-DD"),
    guardar_parquet: bool = Query(False)
):
    try:
        fetcher = SupenDataFetcher(tipo, fondo, fecha_inicio, fecha_final)
        try:
            df = fetcher.fetch_json()
            df = df.convert_dtypes()
            # Reemplazo de valores no compatibles con JSON
            #df = df.replace({pd.NA: None, pd.NaT: None, float('inf'): None, float('-inf'): None})

            # Convertimos todos los NaN de tipo float a None sin usar fillna
            #df = df.applymap(lambda x: None if pd.isna(x) else x)
            #df = df.mask(df.isna(), other=None)
            print("[DEBUG] Limpieza de NaN exitosa.")
        except Exception as e:
            print(f"[ERROR] Fallo durante la limpieza del DataFrame: {e}")
            raise        
        
        archivo = None
        cambio = False
        if guardar_parquet:
            try:
                cambio, archivo = fetcher.detectar_cambios_y_guardar(df)
                archivo = fetcher.save_to_parquet(df)
                print(f"[DEBUG] Archivo guardado en: {archivo}")
            except Exception as e:
                print(f"[ERROR] Fallo al guardar el archivo: {e}")
                raise

        return {
            "registros": len(df),
            "columnas": list(df.columns),
            "datos": df.head(10).to_dict(orient="records"),
            "archivo": archivo,
            "cambio_detectado": cambio 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer datos: {str(e)}")
