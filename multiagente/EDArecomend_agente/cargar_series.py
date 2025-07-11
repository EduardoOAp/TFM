import os
import pandas as pd
import json
from glob import glob
from datetime import datetime
from .config import PATH_SERIES, PATH_INDICADORES, JSON_PATH, UMBRAL_JSON_PATH
from .utils import procesar_columna, evaluar_umbral

def cargar_series_desde_json(columnas_comunes=["entidad", "fondo", "fecha"], 
                             fecha_inicio=None, fecha_fin=None, agente=None, modulo=None, 
                             func_recomendar_columnas=None):
    json_path = os.path.join(os.path.dirname(__file__), JSON_PATH)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Archivo no encontrado: {JSON_PATH}")

    with open(json_path, "r", encoding="utf-8") as f:
        estructura = json.load(f)

    resultados = {}
    #rutas_fechas = glob(os.path.join(carpeta_indicadores, "*", "*", "*", "*", "indicadores.parquet"))

    fecha_inicio_dt = fecha_inicio #datetime.strptime(fecha_inicio, "%Y-%m-%d") if fecha_inicio else None
    fecha_fin_dt = fecha_fin #datetime.strptime(fecha_fin, "%Y-%m-%d") if fecha_fin else None
    #print(f"[INFO] Fecha inicio: {fecha_inicio_dt}, Fecha fin: {fecha_fin_dt}")
    #print(f"[INFO] Procesando fondo {fondo}, fecha: {fecha_actual.date()}")
    modulos_a_probar = [modulo]
    #modulos_a_probar = ["Forecasting de liquidez activo neto"]
    """
    modulos_a_probar = [
                        "Forecasting de liquidez activo neto",
                        "Detección de anomalías de liquidez",
                        "Clustering de riesgo de liquidez",
                        "Análisis de diversificación de portafolio",
                        "Clustering de Riesgo Financiero",
                        "Detección de Anomalías Financieras",
                        "Forecasting Financiero activo neto",
                        "Clustering de flujos de afiliados",
                        "Forecasting de fuga de afiliados",
                        "Relación entre rentabilidad y flujos",
                        "Forecasting de déficit financiero",
                        "Forecasting demográfico",
                        "Clustering de sostenibilidad",
                        "Forecasting Financiero valor cuota"
                        ]  # agrega más si quieres
    """                        
    for bloque in estructura:
        if bloque["modulo"] not in modulos_a_probar:
            continue            
        agente = bloque["subagente"]
        modulo = bloque["modulo"]
        series = bloque.get("series", [])
        indicadores_requeridos = bloque.get("indicadores", [])
        print(f"[INFO] Procesando módulo: {agente} - {modulo}")

        dfs = []

        for serie in series:
            parquet = serie["parquet"]
            columnas = serie.get("columnas", [])
            
            ruta_glob = os.path.join(PATH_SERIES, "*", parquet, "*", "*", "*", "data.parquet")
            #print(f"[DEBUG] Ruta glob: {ruta_glob}")
            for ruta_parquet in glob(ruta_glob):
                partes = ruta_parquet.split(os.sep)
                try:
                    rel_path = os.path.relpath(ruta_parquet, PATH_SERIES)
                    partes = rel_path.split(os.sep)
                    fondo, tipo, anio, mes, dia = partes[:5]
                    #print(f"[DEBUG] series fondo={fondo}, tipo={tipo}, fecha={anio}-{mes}-{dia}")
                    fecha_actual = datetime.strptime(f"{anio}-{mes}-{dia}", "%Y-%m-%d")
                    #print(f"[DEBUG] {ruta_parquet} → fecha: {fecha_actual}, dentro de rango: {not ((fecha_inicio_dt and fecha_actual < fecha_inicio_dt) or (fecha_fin_dt and fecha_actual > fecha_fin_dt))}")
                except Exception as e:
                    print(f"[ERROR DEBUG] {ruta_parquet} {e}")
                    continue

                if (fecha_inicio_dt and fecha_actual < fecha_inicio_dt) or  \
                    (fecha_fin_dt and fecha_actual > fecha_fin_dt):
                    #print(f"Fuera del rango: {ruta_parquet}")
                    continue

                #if not os.path.exists(ruta_parquet):
                    #print(f"Parquet no encontrado: {ruta_parquet}")
                #    continue
                try:
                    df_base = pd.read_parquet(ruta_parquet)
                    #pd.set_option("display.max_columns", None)
                    #pd.set_option("display.width", 1000)
                    #print(df_base.head(10))                 
                    if df_base.empty or len(df_base.columns) == 0:
                        print(f"[SKIP] Archivo vacío o sin columnas: {ruta_parquet}")
                        continue
                    #print(tipo)
                    if tipo == "Libre_transferencia":
                        df_base = df_base.rename(columns={"entidadorigen": "entidad"})

                    # Validación opcional: ¿contiene las columnas clave?
                    if not all(c in df_base.columns for c in columnas_comunes):
                        print(f"[SKIP] Archivo sin columnas necesarias: {ruta_parquet}")
                        continue                
                    if "fecha" in df_base.columns:
                        df_base["fecha"] = pd.to_datetime(df_base["fecha"]).dt.date
                    if "codigofondo" in df_base.columns:
                        df_base["fondo"] = df_base["codigofondo"]   

                    #print(f"[INFO] Cargando: {ruta_parquet}")
                    #print(f"[INFO] Columnas disponibles: {list(df_base.columns)}")
                    #pd.set_option("display.max_columns", None)
                    #pd.set_option("display.width", 1000)
                    #print(df_base.head(10))                

                    for col_def in columnas:
                        df_col = procesar_columna(df_base, columnas_comunes, parquet, col_def)
                        if df_col is not None:
                            df_col["__source__"] = f"{parquet}-{col_def.get('columna_valor', '')}"
                            dfs.append(df_col)
                except Exception as e:
                    print(f"[ERROR] Fallo al leer {ruta_parquet}: {e}")
        
        #print("Fechas series ",sorted(dfs["fecha"].unique()))
            # Procesar indicadores
        ruta_glob_ind = os.path.join(PATH_INDICADORES, "*", "*", "*", "*", "indicadores.parquet")
        indicadores_validos = []
        for path_indicador in glob(ruta_glob_ind):
            partes = path_indicador.split(os.sep)
            try:
                fondo, anio, mes, dia = partes[-5:-1]
                fecha_actual = datetime.strptime(f"{anio}-{mes}-{dia}", "%Y-%m-%d")
                #print(f"[DEBUG] indicadores fondo={fondo}, fecha={anio}-{mes}-{dia}")
                #print(f"[DEBUG] {path_indicador} → fecha: {fecha_actual}, dentro de rango: {not ((fecha_inicio_dt and fecha_actual < fecha_inicio_dt) or (fecha_fin_dt and fecha_actual > fecha_fin_dt))}")
            except Exception as e:
                print(f"[ERROR DEBUG] {partes} {e}")
                continue

            if (fecha_inicio_dt and fecha_actual < fecha_inicio_dt) or  (fecha_fin_dt and fecha_actual > fecha_fin_dt):
                continue
            #print(f"[DEBUG] leer {path_indicador}")
            try:

                df_ind = pd.read_parquet(path_indicador)
                if "fecha" in df_ind.columns:
                    df_ind["fecha"] = pd.to_datetime(df_ind["fecha"]).dt.date
                # Filtrar solo los indicadores relevantes
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", 1000)
                #print(df_ind.head(100))                  
                df_ind = df_ind[df_ind["indicador"].isin(indicadores_requeridos)]            
                indicadores_validos.append(df_ind)
                #print(indicadores_validos)
            except Exception as e:
                print(f"[ERROR] Fallo al leer indicadores: {path_indicador}: {e}")                    

        #print("Fechas ind ",sorted(df_ind["fecha"].unique()))   
        if indicadores_validos:
            # Pivotear: convertir 'indicador' en columnas
            df_ind_total = pd.concat(indicadores_validos, ignore_index=True)          
            df_ind_total = df_ind_total[df_ind_total["indicador"].isin(indicadores_requeridos)]
            # Agrupar para asegurar unicidad antes del pivot
            df_ind_total = (
                df_ind_total
                .groupby(columnas_comunes + ["indicador"], as_index=False)["valor"]
                .last()
            )            
            df_ind_total = df_ind_total.pivot(index=columnas_comunes, columns="indicador", values="valor").reset_index()
            #print("Fechas ind ",sorted(df_ind_total["fecha"].unique())) 
            #pd.set_option("display.max_columns", None)
            #pd.set_option("display.width", 1000)
            #print("df_ind_total")
            #print(df_ind_total.head(100))  

            dfs.append(df_ind_total)            
            #print(f"[MERGE2] DataFrame listo para merge - columnas: {df_ind.columns.tolist()}")
        #print(dfs)

        df_filtrado = None
        dfs = [df for df in dfs if not df.empty and not df.dropna(axis=1, how="all").empty]

        if dfs:
            # Concatenar todo en uno solo para evitar merges duplicados
            df_all = pd.concat(dfs, ignore_index=True)
            #df_all.head(100)

            # Eliminar columna auxiliar si existe
            if "__source__" in df_all.columns:
                df_all = df_all.drop(columns="__source__")            

            # Agrupar por claves y hacer promedio de columnas repetidas
            columnas_datos = [c for c in df_all.columns if c not in columnas_comunes]
            df_combinado = (
                df_all
                .groupby(columnas_comunes, as_index=False)[columnas_datos]
                .last()
            )
            #print("Fechas ind ",sorted(df_combinado["fecha"].unique())) 
            #df_combinado = dfs[0]
            # para corroborar
            #for i, df in enumerate(dfs):
            #    print(f"[DEBUG] DataFrame {i} columnas: {df.columns.tolist()}")                
            col_total = pd.Index([])
            for df in dfs:
                col_total = col_total.append(pd.Index(df.columns))

            #duplicadas = col_total[col_total.duplicated()].tolist()
            #if duplicadas:
            #    print(f"[ADVERTENCIA] Columnas duplicadas antes del merge: {duplicadas}")                    
            # fin de corroborar
            #for df in dfs[1:]:
            #    df_combinado = pd.merge(df_combinado, df, on=columnas_comunes, how='inner')
            #print(f"[FINAL] Combinación completada para: {subagente} - {modulo} ({fondo}-{anio}-{mes}-{dia})")
            #print(f"[FINAL] Combinación completada para: {subagente} - {modulo}")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            #print(df_combinado.head(10))
            #print(sorted(df_combinado["fecha"].unique()))
            df_filtrado = df_combinado.dropna()
            ruta_guardado = os.path.join(
                "data", "temporal", f"{agente}_{modulo}_{fecha_fin_dt:%Y-%m-%d}.csv"
            )
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            if os.path.exists(ruta_guardado):
                os.remove(ruta_guardado)                
            df_filtrado.to_csv(ruta_guardado, index=False)            
        #print("Fechas df_filtrado ",sorted(df_filtrado["fecha"].unique()))
        ruta_guardado = None

        if func_recomendar_columnas:                            
            columnas_ind = [col for col in df_filtrado.columns if col not in columnas_comunes + ["target"]]
            recomendadas = func_recomendar_columnas(df_filtrado, columnas_indicadores=columnas_ind, 
                                                    target_col="target", agente=agente, modulo=modulo)  
            print('recomendadas ', recomendadas)
            columnas_a_usar = columnas_comunes + recomendadas #+ ["target"]
            df_filtrado = df_filtrado[columnas_a_usar]

            resultados = evaluar_umbral(df_filtrado, fecha_fin_dt, UMBRAL_JSON_PATH)                              

        #if df_filtrado is not None and not df_filtrado.empty:
        #    columnas_ind = [col for col in df_filtrado.columns if col not in ["entidad", "fondo", "fecha", "target"]]

        #    recomendadas = AgenteEDARecomendador.recomendar_columnas(df_filtrado, columnas_indicadores=columnas_ind, target_col="target", modulo=modulo)
        #    print("Columnas recomendadas:", recomendadas)

        #    resultados = evaluar_umbral(df_filtrado, fecha_fin_dt)
            #print("Fechas df_filtrado ",sorted(df_filtrado["fecha"].unique()))
            #print(f"umbrales", resultados)

            umbrales_incumplidos = sum(
                1 for v in resultados.values() 
                if isinstance(v, dict) and v.get("umbrales_incumplidos", 0) > 0
                )            
            ruta_guardado = None
            if umbrales_incumplidos > 0:  
                ruta_guardado = os.path.join(
                    "data", "temporal", agente, modulo, f"{fecha_fin_dt:%Y-%m-%d}.parquet"
                )
                os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
                if os.path.exists(ruta_guardado):
                    os.remove(ruta_guardado)                
                df_filtrado.to_parquet(ruta_guardado, index=False)

        if ruta_guardado != None:
            # Asignar al diccionario
            resultados[(agente, modulo)] = ruta_guardado 

    return resultados
