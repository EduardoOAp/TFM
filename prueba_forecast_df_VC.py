# clustering_financiero.py

import os
import sys
import pandas as pd
from riesgo_agente.utils.helpers import completar_fechas_mensuales

# Asegurar que 'multiagente' esté en el sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from riesgo_agente.agenteDesequilibrioFinanciero.forecasting_valor_cuota import ForecastingValorCuota
#from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import VisualizadorClusters

if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente =  "Agente de Desequilibrio Financiero"
    modulo = "Forecasting Financiero valor cuota"
    fecha = "2024-12-31"

    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)
    print(df.info())

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 720].copy()

    # Conteo de fechas por entidad
    conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    print("Conteo de fechas por entidad:\n", conteo_fechas)

    # Verificar fechas faltantes por entidad
    entidades = df["entidad"].unique()

    columna_objetivo="valor_cuota"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    evaluator = ForecastingValorCuota(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = evaluator.ejecutar()
    #resultados = modelo.entrenar_por_entidad(df)
    #graficar_forecasts(df, resultados, columna_objetivo)
    #print(resultados)




