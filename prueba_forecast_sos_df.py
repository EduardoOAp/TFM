# clustering_financiero.py

import os
import sys
import pandas as pd
from riesgo_agente.utils.helpers import completar_fechas_mensuales, unir_serie_y_predicciones

# Asegurar que 'multiagente' esté en el sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from riesgo_agente.agenteSostenibilidad.forecasting_deficit import ForecastingDeficit  # Ajusta si el path real es diferente
#from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import VisualizadorClusters


if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente =  "Agente de Sostenibilidad"
    modulo = "Forecasting de déficit financiero"
    fecha = "2024-12-31"
    print("aqui")
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)
    print(df.info())

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    print(df.columns)
    # Conteo de fechas por entidad
    conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    print("Conteo de fechas por entidad:\n", conteo_fechas)

    # Verificar fechas faltantes por entidad
    entidades = df["entidad"].unique()

    columna_objetivo="indice_equilibrio_financiero"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    evaluator = ForecastingDeficit(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = evaluator.ejecutar()

    ruta_salida = r"data/resultados/Agente de Sostenibilidad/Forecasting de déficit financiero/comparacion_equilibrio_financiero.csv"

    df_combinado = unir_serie_y_predicciones(
        path_serie=r"data/resultados/Agente de Sostenibilidad/Forecasting de déficit financiero/serie_datos.csv",
        path_resultados=r"data/resultados/Agente de Sostenibilidad/Forecasting de déficit financiero/resultados.csv",
        columna_objetivo="indice_equilibrio_financiero",
        salida=ruta_salida
    )

    #resultados = modelo.entrenar_por_entidad(df)
    #graficar_forecasts(df, resultados, columna_objetivo)
    #print(resultados)

