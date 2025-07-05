from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_LIQUIDEZ, LIQ_MODULO_CLUSTERING, LIQ_MODULO_ANOMALIAS, LIQ_MODULO_FORECASTING
from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import ClusteringEvaluator
from riesgo_agente.agenteRiesgoLiquidez.deteccion_anomalias_lstm import LSTMAnomalyDetector
from riesgo_agente.agenteRiesgoLiquidez.forecasting_liquidez import ForecastingLiquidez

def ejecutar_subagente_liquidez(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_LIQUIDEZ, LIQ_MODULO_CLUSTERING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)
    columnas = columnas_validas(df)

    # 2. Ejecutar Clustering
    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                     usar_normalizado=usar_normalizado)
    clustering.ejecutar()

    # 3. Ejecutar Anomalías
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    anomalias = LSTMAnomalyDetector(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                    agente=AGENTE_LIQUIDEZ, modulo=LIQ_MODULO_ANOMALIAS)
    anomalias.ejecutar()

    # 4. Ejecutar Forecasting
    df = cargar_dataframe_temporal(AGENTE_LIQUIDEZ, LIQ_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)      
    forecasting = ForecastingLiquidez(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                       agente=AGENTE_LIQUIDEZ, modulo=LIQ_MODULO_FORECASTING)
    forecasting.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "anomalias": anomalias.resultado,
        "forecasting": forecasting.resultado
    }