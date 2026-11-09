import pandas as pd
import re
import json
import os
from operator import lt, le, gt, ge, eq, ne

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
        if cond == "==":
            df = df[df[col] == val]
        elif cond == "!=":
            df = df[df[col] != val]
        elif cond == "contiene":
            df = df[df[col].astype(str).str.contains(val, case=False)]
            #df.head(100)

    # Modo regex: seleccionar múltiples columnas
    if modo == "regex_columnas":
        patron = col_valor.replace("*", ".*")
        columnas_match = [c for c in df.columns if re.match(patron, c)]
        columnas_finales = columnas_comunes + [c for c in columnas_match if c not in columnas_comunes]
        df = df[columnas_finales].copy()
        df = df.rename(columns={c: alias or f"{parquet}_{c}" for c in columnas_match if c not in columnas_comunes})
        return df

    # Validar columna
    if col_valor not in df.columns:
        print(f"[SKIP] Columna no encontrada: {col_valor}")
        return None

    nuevo_nombre = alias or f"{parquet}_{col_valor}"

    columnas_finales = [c for c in columnas_comunes if c in df.columns]
    if col_valor not in columnas_comunes:
        columnas_finales.append(col_valor)

    df = df[columnas_finales].copy()

    # Agregación si aplica
    if funcion in ["promedio", "suma"]:
        agg_func = "mean" if funcion == "promedio" else "sum"
        df = df.groupby(columnas_comunes).agg({col_valor: agg_func}).reset_index()

    # Renombrar si aplica
    if alias and col_valor in df.columns:
        df = df.rename(columns={col_valor: alias})
    elif col_valor not in columnas_comunes:
        df = df.rename(columns={col_valor: nuevo_nombre})

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
    #print(f"fecha {fecha}")    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    #print(f"unbral {df.tail(150)}")

    #df.loc[:,"fecha"] = pd.to_datetime(df["fecha"]).dt.date
    #if isinstance(fecha, pd.Timestamp):
    #    fecha = fecha.date()
    #df = df[df["fecha"] == fecha]   
 
    if not os.path.exists(umbral_path):
        raise FileNotFoundError(f"Archivo no encontrado: {umbral_path}")        

    with open(umbral_path, 'r') as f:
        umbrales = json.load(f)       

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