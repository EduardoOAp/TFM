from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_SOSTENIBILIDAD, SOS_MODULO_CLUSTERING, SOS_MODULO_FORECASTING, SOS_MODULO_FORECAST_DEMO
from riesgo_agente.agenteSostenibilidad.clustering_sostenibilidad import ClusteringEvaluator
from riesgo_agente.agenteSostenibilidad.forecasting_deficit import ForecastingDeficit
from riesgo_agente.agenteSostenibilidad.forecasting_demografico import ForecastingDemografico

def ejecutar_subagente_sostenibilidad(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_CLUSTERING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)
    columnas = columnas_validas(df)

    # 2. Ejecutar Clustering
    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                     usar_normalizado=usar_normalizado)
    clustering.ejecutar()

    # 3. Ejecutar Forecstating deficit
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    forecastDeficit = ForecastingDeficit(df, fondo=fondo, entidad=entidad, columnas=columnas)
    forecastDeficit.ejecutar()

    # 4. Ejecutar Forecasting demografico
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECAST_DEMO, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)      
    forecastDemog = ForecastingDemografico(df, fondo=fondo, entidad=entidad, columnas=columnas)
    forecastDemog.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "forecasting Deficit": forecastDeficit.resultado,
        "forecasting Demográfico": forecastDemog.resultado
    }