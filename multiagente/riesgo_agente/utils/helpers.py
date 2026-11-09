import os
import pandas as pd
from datetime import datetime
from riesgo_agente.utils.config import TEMPORAL_ROOT

def cargar_dataframe_temporal(agente: str, modulo: str, fecha: str, data_root: str = "data"):
    ruta = os.path.join(data_root, TEMPORAL_ROOT, agente, modulo, f"{fecha}.parquet")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    df = pd.read_parquet(ruta)
    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def filtrar_entidades_con_historia(df: pd.DataFrame, min_periodos: int = 24) -> pd.DataFrame:
    return df[df.groupby("entidad")["fecha"].transform("nunique") > min_periodos].copy()

def columnas_validas(df: pd.DataFrame) -> list:
    return df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

def completar_fechas_mensuales(df: pd.DataFrame, columna_fecha: str ="fecha", columna_entidad: str="entidad"):
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    entidades = df[columna_entidad].unique()
    
    df_completo = []

    for ent in entidades:
        df_ent = df[df[columna_entidad] == ent].copy()
        fechas_disponibles = df_ent[columna_fecha].dropna()
        
        if fechas_disponibles.empty:
            continue
        
        fecha_inicio = fechas_disponibles.min()
        fecha_fin = fechas_disponibles.max()
        fechas_completas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="ME")
        
        df_ent.set_index(columna_fecha, inplace=True)
        df_ent = df_ent.reindex(fechas_completas)
        
        df_ent[columna_entidad] = ent
        df_ent.index.name = columna_fecha
        
        df_completo.append(df_ent.reset_index())

    return pd.concat(df_completo, ignore_index=True)

def completar_fechas_diarias(df: pd.DataFrame, columna_fecha: str ="fecha", columna_entidad: str="entidad"):
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    entidades = df[columna_entidad].unique()
    
    df_completo = []

    for ent in entidades:
        df_ent = df[df[columna_entidad] == ent].copy()
        fechas_disponibles = df_ent[columna_fecha].dropna()
        
        if fechas_disponibles.empty:
            continue
        
        fecha_inicio = fechas_disponibles.min()
        fecha_fin = fechas_disponibles.max()
        fechas_completas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
        
        df_ent.set_index(columna_fecha, inplace=True)
        df_ent = df_ent.reindex(fechas_completas)
        
        df_ent[columna_entidad] = ent
        df_ent.index.name = columna_fecha
        
        df_completo.append(df_ent.reset_index())

    return pd.concat(df_completo, ignore_index=True)

def unir_serie_y_predicciones(path_serie: str, path_resultados: str, columna_objetivo: str, salida: str) -> pd.DataFrame:
    """
    Une la serie histórica con los resultados de predicción por modelo,
    creando una tabla comparativa lista para graficar.

    Parámetros:
    - path_serie: Ruta al CSV con la serie histórica.
    - path_resultados: Ruta al CSV con las predicciones.
    - columna_objetivo: Nombre de la columna con el valor real.
    - salida: Ruta para guardar el archivo combinado.

    Retorna:
    - DataFrame combinado y ordenado.
    """
    # Cargar datos
    serie_df = pd.read_csv(path_serie)
    resultados_df = pd.read_csv(path_resultados)

    # Asegurar formato de fecha
    serie_df['fecha'] = pd.to_datetime(serie_df['fecha'])
    resultados_df['fecha'] = pd.to_datetime(resultados_df['fecha'])

    # Filtrar columnas necesarias
    cols_serie = ['entidad', 'fondo', 'fecha', columna_objetivo]
    serie_filtrada = serie_df[cols_serie]

    # Pivotear predicciones
    resultados_pivot = resultados_df.pivot_table(
        index=['entidad', 'fecha'],
        columns='modelo',
        values='prediccion'
    ).reset_index()

    resultado_final = pd.merge(serie_filtrada, resultados_pivot, on=['entidad', 'fecha'], how='outer')
    # Ordenar
    resultado_final = resultado_final.sort_values(by=['entidad', 'fondo', 'fecha'])

    # Exportar
    resultado_final.to_csv(salida, index=False)

    return resultado_final

# Probar función con archivos actuales
#ruta_salida = "data\resultados\Agente de Sostenibilidad\Forecasting de déficit financiero/comparacion_equilibrio_financiero.csv"
#df_combinado = unir_serie_y_predicciones(
#    path_serie='data\resultados\Agente de Sostenibilidad\Forecasting de déficit financiero\serie_datos.csv',
#    path_resultados='data\resultados\Agente de Sostenibilidad\Forecasting de déficit financiero\resultados.csv',
#    columna_objetivo='indice_equilibrio_financiero',
#    salida=ruta_salida
#)




