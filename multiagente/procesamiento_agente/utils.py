from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import os
import json
from calendar import monthrange
import glob

# --------------------------- CONFIG ---------------------------
DATA_ROOT = "data"
INDICADOR_ROOT = "indicadores"

# --------------------------- UTILIDADES ---------------------------

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

def obtener_fecha_anterior(fecha: str, unidad: str = 'months', cantidad: int = 1) -> datetime:
    fecha_dt = pd.to_datetime(fecha)
    if unidad == 'years':
        return fecha_dt - pd.DateOffset(years=cantidad)
    return fecha_dt - pd.DateOffset(months=cantidad)

def cargar_datos_parquet(tipo: str, fondo: str, fecha: datetime) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_ROOT, fondo, tipo, f"{fecha.year}", f"{fecha.month:02d}", f"{fecha.day:02d}", "data.parquet")
    try:
        if not os.path.exists(path):
            print(f"[WARN] Archivo no encontrado para tipo '{tipo}' en fecha {fecha.strftime('%Y-%m-%d')}")
            return None

        df = pd.read_parquet(path)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"]) #.dt.strftime("%Y-%m-%d")

        return df    

    #try:
    #    return pd.read_parquet(path) if os.path.exists(path) else None
    except Exception as e:
        print(f"[ERROR] al leer {tipo}: {e}")
        return None
        
def cargar_datos_parquet_rango(tipo: str, fondo: str, fecha_inicio: datetime, fecha_final: datetime) -> pd.DataFrame:
    datos = []
    fecha_actual = fecha_inicio
    # Tipos de datos que se cargan diario (todos los días)
    #TIPOS_DIARIOS = {"cuenta"}
    
    #if tipo in TIPOS_DIARIOS:
    #    fechas = pd.date_range(fecha_inicio, fecha_final, freq='D')
    #else:
    fecha_final = fecha_final + pd.offsets.MonthEnd(0)
    fechas = pd.date_range(fecha_inicio, fecha_final, freq='ME')  # solo fin de mes

    for fecha in fechas:
        df = cargar_datos_parquet(tipo, fondo, fecha)
        if df is not None:
            datos.append(df)

    if datos:
        return pd.concat(datos, ignore_index=True)
    else:
        return pd.DataFrame()        

def guardar_indicadores(fondo: str, fecha: str, resultados: Dict[Tuple[str, str], float]) -> None:
    dt = datetime.strptime(fecha, "%Y-%m-%d")

    registros = []

    for (entidad, indicador), valor in resultados.items():
        registros.append({
            "entidad": entidad,
            "fondo": fondo,
            "fecha": fecha,
            "indicador": indicador,
            "valor": valor
        })
        #print(f"entidad {entidad} fondo {fondo} fecha {fecha} indicador {indicador} valor {valor}")

    df = pd.DataFrame(registros)
    ultimo_dia = monthrange(dt.year, dt.month)[1]
    directory = os.path.join(DATA_ROOT,INDICADOR_ROOT, fondo, f"{dt.year}", f"{dt.month:02d}", f"{ultimo_dia:02d}")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, "indicadores.parquet")
    if os.path.exists(filepath):
        df_existente = pd.read_parquet(filepath)
        df_final = pd.concat([df_existente, df], ignore_index=True)
    else:
        df_final = df    
    # Evitar duplicados por entidad-indicador-fecha
    df_final = df_final.drop_duplicates(subset=["entidad", "fondo", "indicador", "fecha"], keep="first")

    df_final.to_parquet(filepath, index=False)

def get_parametro_indicador(nombre: str, clave: str, valor_default=None):
    
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_config = os.path.join(ruta_actual, "config.json")

    try:
        with open(ruta_config, "r") as f:
            config = json.load(f)
        return config.get(nombre, {}).get(clave, valor_default)
    except FileNotFoundError:
        print(f"[WARN] No se encontró config.json en: {ruta_config}")
        return valor_default
    except Exception as e:
        print(f"[ERROR] al leer config.json: {e}")
        return valor_default
        