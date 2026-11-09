from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_LIQUIDEZ, LIQ_MODULO_CLUSTERING, LIQ_MODULO_ANOMALIAS, LIQ_MODULO_FORECASTING
from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import ClusteringEvaluator
from riesgo_agente.agenteRiesgoLiquidez.clustering_riesgo import VisualizadorClusters
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

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas_cluster,
                                     usar_normalizado=usar_normalizado,
                                     mostrarGrafico=False)
    clustering.ejecutar()
    df_resultado = clustering.df.copy()
    df_resultado["cluster"] = clustering.mejor_modelo[1]["etiquetas"]
   
    # 3. Ejecutar Anomalías
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    anomalias = LSTMAnomalyDetector(
        df=df,
        secuencia=12, 
        tipo_modelo="autoencoder", 
        epochs=50, 
        batch_size=16, 
        umbral_factor=1.5
    )
    #anomalias = LSTMAnomalyDetector(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                agente=AGENTE_LIQUIDEZ, modulo=LIQ_MODULO_ANOMALIAS)
    anomalias.ejecutar()

    # 4. Ejecutar Forecasting
    df = cargar_dataframe_temporal(AGENTE_LIQUIDEZ, LIQ_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)   

    columna_objetivo="monto_activo_neto"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    forecasting = ForecastingLiquidez(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar, mostrarGrafico=False
    )
    forecasting.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "anomalias": anomalias.resultado,
        "forecasting": forecasting.resultado
    }