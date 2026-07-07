# clustering_financiero.py

import os
import sys
import pandas as pd
from riesgo_agente.utils.helpers import completar_fechas_mensuales

# Asegurar que 'multiagente' esté en el sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from riesgo_agente.agenteFlujoFondos.forecasting_flujofondos import ForecastingFlujoFondos
#from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import VisualizadorClusters

if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente = "Agente de Flujos de Fondos"
    modulo = "Forecasting de fuga de afiliados"
    fecha = "2024-12-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    print(df.columns)
    # Conteo de fechas por entidad
    conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    print("Conteo de fechas por entidad:\n", conteo_fechas)
    df.to_csv("fugadeafilaidos.csv")
    # Verificar fechas faltantes por entidad
    entidades = df["entidad"].unique()

    columna_objetivo="tasa_fuga_afiliados"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    evaluator = ForecastingFlujoFondos(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = evaluator.ejecutar()
    #resultados = modelo.entrenar_por_entidad(df)
    #graficar_forecasts(df, resultados, columna_objetivo)
    #print(resultados)




