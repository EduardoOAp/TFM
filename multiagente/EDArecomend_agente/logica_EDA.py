import os
import pandas as pd
import numpy as np
from scipy.stats import normaltest
from .config import JSON_PATH
import json
from .cargar_series import cargar_series_desde_json

class AgenteEDARecomendador:
    def __init__(self, nombre="eda_pensiones", carpeta_salida="eda_resultados"):
        self.nombre = nombre
        self.carpeta_salida = carpeta_salida
        os.makedirs(carpeta_salida, exist_ok=True)

    def recomendar_columnas(
        self,
        df,
        columnas_indicadores,
        target_col=None,  
        agente="",
        modulo="",          
        umbral_corr=0.1,
        umbral_nulos=0.2,
        umbral_varianza=0.01,
        revisar_multicolinealidad=True,
        umbral_multicolinealidad=0.9,
        aplicar_filtro_outliers=True,
        alpha_normalidad=0.05,
        umbral_z=3,
        k_iqr=1.5        
    ):
        
        
        def es_normal(columna, alpha=0.05):
            if columna.dropna().shape[0] < 20:
                return False  # no aplicar test si hay pocos datos
            stat, p = normaltest(columna.dropna())
            return p > alpha  # True si no se rechaza H0 = distribución normal  
                
        def filtrar_outliers_adaptativo(df, columnas, alpha=0.05, umbral_z=3, k_iqr=1.5):
            mask_total = pd.Series([True] * len(df), index=df.index)

            for col in columnas:
                if es_normal(df[col], alpha):
                    z = ((df[col] - df[col].mean()) / df[col].std()).abs()
                    mask = z < umbral_z
                else:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = df[col].between(Q1 - k_iqr * IQR, Q3 + k_iqr * IQR)

                mask_total &= mask

            return df[mask_total]   

        def detectar_normalidad(df, columnas, alpha=0.05):
            distribuciones = {}
            for col in columnas:
                serie = df[col].dropna()
                if len(serie) < 20:
                    distribuciones[col] = "indefinido"  # muestra insuficiente
                else:
                    stat, p = normaltest(serie)
                    distribuciones[col] = "normal-zscore" if p > alpha else "no_normal-iqr"
            return distribuciones             
        
        reporte = {}

        for col in columnas_indicadores:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                except Exception as e:
                    print(f"No se pudo convertir la columna {col}: {e}")
        # Paso 1: Nulos
        nulos_pct = df[columnas_indicadores].isnull().mean()
        columnas_sin_muchos_nulos = nulos_pct[nulos_pct <= umbral_nulos].index.tolist()
        reporte["columnas_sin_muchos_nulos"] = columnas_sin_muchos_nulos

        # Paso 2: Varianza
        columnas_numericas = [col for col in columnas_sin_muchos_nulos if pd.api.types.is_numeric_dtype(df[col])]
        #print("numericas ", columnas_numericas)
        varianzas = df[columnas_numericas].var()
        #print(varianzas.sort_values())
        columnas_con_varianza = varianzas[varianzas > umbral_varianza].index.tolist()
        reporte["columnas_con_varianza"] = columnas_con_varianza

        # Paso 2.5: Outlier adaptativo
        if aplicar_filtro_outliers:
            df = filtrar_outliers_adaptativo(df, columnas_con_varianza)
            distribuciones = detectar_normalidad(df, columnas_con_varianza, alpha=alpha_normalidad)
            reporte["tipo_distribucion"] = list(distribuciones.values())
            reporte["columnas_con_outlier"] = list(distribuciones.keys())
            
        # Paso 3: Correlación con target (si existe)
        if target_col and target_col in df.columns and df[target_col].dtype in [np.float64, np.int64]:
            correlaciones = df[columnas_con_varianza + [target_col]].corr()[target_col].drop(target_col)
            columnas_recomendadas = correlaciones[abs(correlaciones) > umbral_corr].sort_values(ascending=False).index.tolist()
        else:
            columnas_recomendadas = columnas_con_varianza

        reporte["columnas_recomendadas_pre_colinealidad"] = columnas_recomendadas

        # Paso 4: Multicolinealidad
        if revisar_multicolinealidad and len(columnas_recomendadas) > 1:
            df_corr = df[columnas_recomendadas].corr().abs()
            upper_triangle = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

            columnas_a_eliminar = [
                col for col in upper_triangle.columns if any(upper_triangle[col] > umbral_multicolinealidad)
            ]

            columnas_finales = [col for col in columnas_recomendadas if col not in columnas_a_eliminar]
            reporte["columnas_eliminadas_por_colinealidad"] = columnas_a_eliminar
            reporte["columnas_recomendadas_final"] = columnas_finales
        else:
            reporte["columnas_eliminadas_por_colinealidad"] = []
            reporte["columnas_recomendadas_final"] = columnas_recomendadas

        # Exportar CSV
        df_reporte = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in reporte.items()]))
        nombre_archivo = f"eda_{agente.replace(' ', '_')}_{modulo.replace(' ', '_')}"
        archivo = os.path.join(self.carpeta_salida, f"{nombre_archivo}_recomendacion.csv")
        df_reporte.to_csv(archivo, index=False)

        return reporte["columnas_recomendadas_final"]
    
    def ejecutar_EDA(self, fecha_inicio, fecha_fin, fondo):
        resultados_globales = {}
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            estructura = json.load(f)
        for bloque in estructura:
            agente = bloque["subagente"]
            modulo = bloque["modulo"]
            resultados_globales = cargar_series_desde_json(
                columnas_comunes=["entidad", "fondo", "fecha"],
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin,
                agente=agente,
                modulo=modulo,
                func_recomendar_columnas=self.recomendar_columnas
            )
        return resultados_globales    