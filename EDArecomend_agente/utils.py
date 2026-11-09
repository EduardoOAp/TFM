import pandas as pd
import re
import os
from operator import lt, le, gt, ge, eq, ne
from configuracion.cargar_json import cargar_config

OPERADORES = {"<": lt, "<=": le, ">": gt, ">=": ge, "==": eq, "!=": ne}

def procesar_columna(df_base, columnas_comunes, parquet, col_def):
    df = df_base.copy()
    col_valor = col_def["columna_valor"]
    alias = col_def.get("alias")
    filtros = col_def.get("filtros", [])
    funcion = col_def.get("funcion")
    modo = col_def.get("modo", "")

    # Aplicar filtros
    for filtro in filtros:
        col = filtro["columna"]
        cond = filtro["condicion"]
        val = filtro["valor"]
        #if col == "codigoinstrumento":
        #    print('Antes ',df.head(100))
        #    print(df.dtypes, type(val))
        if cond == "==":
            df = df[df[col].astype(str) == val]
        elif cond == "!=":
            df = df[df[col].astype(str) != val]            
        elif cond == "contiene":
            df = df[df[col].astype(str).str.contains(val, case=False)]
        #print(f"col {col} cond {cond} valor_filtro {val}")
        #if col == "codigoinstrumento":
        #    print(df.head(100))

    if df.empty:
        #print(f"[SKIP] DataFrame vacío después de aplicar el filtro: {col} {cond} {val}")
        df_vacio = df_base[columnas_comunes].drop_duplicates()
        # Paso 2: Añadir la columna 'monto' a este nuevo DataFrame y establecer su valor en 0
        df_vacio[alias] = 0        
        #print('Vacio ',pd.DataFrame(df_vacio).head())
        return pd.DataFrame(df_vacio)
        #None #pd.DataFrame(columns=columnas_comunes + [alias or f"{parquet}_{col_valor}"])
        
    # Modo regex: seleccionar múltiples columnas
    if modo == "regex_columnas":
        patron = col_valor.replace("*", ".*")
        columnas_match = [c for c in df.columns if re.match(patron, c)]
        # Si no hay columnas que coincidan, el resultado es vacío
        if not columnas_match:
            print("[SKIP] No se encontraron columnas que coincidan con el patrón regex.")
            return None #pd.DataFrame(columns=columnas_comunes)        
        columnas_finales = columnas_comunes + [c for c in columnas_match if c not in columnas_comunes]
        df = df[columnas_finales].copy()
        df = df.rename(columns={c: alias or f"{parquet}_{c}" for c in columnas_match if c not in columnas_comunes})
        return df

    # Validar columna
    if col_valor not in df.columns:
        print(f"[SKIP] Columna de valor no encontrada: {col_valor}")
        return None #pd.DataFrame(columns=columnas_comunes + [alias or f"{parquet}_{col_valor}"])

    nuevo_nombre = alias or f"{parquet}_{col_valor}"

    columnas_finales = [c for c in columnas_comunes if c in df.columns]
    if col_valor not in columnas_comunes:
        columnas_finales.append(col_valor)

    df = df[columnas_finales].copy()

    # 3. Después de seleccionar columnas, verificar si el DataFrame está vacío
    if df.empty:
        print(f"[SKIP] DataFrame vacío después de seleccionar columnas.")
        return None #pd.DataFrame(columns=columnas_finales)   
     

    # Agregación si aplica
    if funcion in ["promedio", "suma"]:
        agg_func = "mean" if funcion == "promedio" else "sum"
        df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
        
        # Opcionalmente, puedes reemplazar los NaN con 0 antes de la agregación
        df[col_valor] = df[col_valor].fillna(0)        
        df = df.groupby(columnas_comunes).agg({col_valor: agg_func}).reset_index()

    if df.empty:
        print(f"[SKIP] DataFrame vacío después de group by.")
        return None #pd.DataFrame(columns=columnas_finales)    
    
    # Renombrar si aplica
    if alias and col_valor in df.columns:
        df = df.rename(columns={col_valor: alias})
    elif col_valor not in columnas_comunes:
        df = df.rename(columns={col_valor: nuevo_nombre})

    #if alias is not None and 'monto_' in alias.lower():
        #pd.set_option("display.max_columns", None)
        #pd.set_option("display.width", 1000)
        #print(df.head(10))
        #print(f"[MERGE] DataFrame listo para merge - columnas: {df.columns.tolist()}")

    return df

def evaluar_umbral(df, fecha, umbral_path):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    fecha = pd.to_datetime(fecha).normalize()
    df = df[df["fecha"] == fecha]   

    if not os.path.exists(umbral_path):
        raise FileNotFoundError(f"Archivo no encontrado: {umbral_path}")        
    umbrales = cargar_config(umbral_path)    

    resultados = {}
    try:
        for columna in df.columns:
            if columna in umbrales:
                operador = umbrales[columna]['operador']
                umbral = umbrales[columna]['valor']
                op_func = OPERADORES.get(operador)
                if op_func:
                    umbrales_incumplidos = op_func(df[columna], umbral)
                    porcentaje_incumplidos = umbrales_incumplidos.mean() * 100
                    resultados[columna] = {
                        'umbral': f"{operador} {umbral}",
                        'porcentaje_incumplidos': round(porcentaje_incumplidos, 2),
                        'umbrales_incumplidos': int(umbrales_incumplidos.sum()),
                        'total': len(df)
                    }
    except Exception as e:
        print(f"[ERROR] Evaluación umbral falló: {e}") 

    return resultados