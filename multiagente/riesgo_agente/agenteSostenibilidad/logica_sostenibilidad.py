from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_SOSTENIBILIDAD, SOS_MODULO_CLUSTERING, SOS_MODULO_FORECASTING, SOS_MODULO_FORECAST_DEMO
from riesgo_agente.agenteSostenibilidad.clustering_sostenibilidad import ClusteringEvaluator
from riesgo_agente.agenteSostenibilidad.forecasting_deficit import ForecastingDeficit
from riesgo_agente.agenteSostenibilidad.forecasting_demografico import ForecastingDemografico
from configuracion.devuelveObjetivo import seleccionar_columnas_objetivo_desde_json

def ejecutar_subagente_sostenibilidad(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_CLUSTERING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df, min_periodos = 36) 
    #columnas = columnas_validas(df)

    # 2. Ejecutar Clustering
    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    clustering = ClusteringEvaluator(
        df=df,
        fondo=fondo, entidad=entidad,
        columnas=columnas_cluster,
        usar_normalizado=usar_normalizado, mostrarGrafico=False
    )    
    clustering.ejecutar()

    # 3. Ejecutar Forecstating deficit
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df, min_periodos = 36)    

    columna_objetivo = seleccionar_columnas_objetivo_desde_json(
        AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING, df.columns, 1)
    #columna_objetivo="indice_equilibrio_financiero"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha"]+columna_objetivo).tolist()
    
    #conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    #print("Conteo de fechas por entidad antes:\n", conteo_fechas)  

    forecastDeficit = ForecastingDeficit(df=df,
        columna_objetivo=columna_objetivo[0],
        columnas_exogenas=columnas_usar,mostrarGrafico=False
    )
    forecastDeficit.ejecutar()

    # 4. Ejecutar Forecasting demografico
    df = cargar_dataframe_temporal(AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECAST_DEMO, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)   

    columna_objetivo = seleccionar_columnas_objetivo_desde_json(
        AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECAST_DEMO, df.columns, 1)
    #columna_objetivo="aportantes_59_anios"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha"]+columna_objetivo).tolist()

    forecastDemog = ForecastingDemografico(df=df,
        columna_objetivo=columna_objetivo[0],
        columnas_exogenas=columnas_usar, mostrarGrafico=False
    )
    forecastDemog.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "forecasting Deficit": forecastDeficit.resultado,
        "forecasting Demográfico": forecastDemog.resultado
    }