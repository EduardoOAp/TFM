from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_FLUJOS, FLU_MODULO_CLUSTERING, FLU_MODULO_FORECASTING, FLU_MODULO_RELACION
from riesgo_agente.agenteFlujoFondos.clustering_flujofondos import ClusteringEvaluator
from riesgo_agente.agenteFlujoFondos.forecasting_flujofondos import ForecastingFlujoFondos
from riesgo_agente.agenteFlujoFondos.causalidad_flujofondos import CausalidadEvaluator

def ejecutar_subagente_flujofondos(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_FLUJOS, FLU_MODULO_CLUSTERING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)
    columnas = columnas_validas(df)

    # 2. Ejecutar Clustering
    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                     usar_normalizado=usar_normalizado)
   
    clustering.ejecutar()

    # 3. Ejecutar Forecasting
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_FLUJOS, FLU_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    
    columna_objetivo="tasa_fuga_afiliados"
    
    forecasting = ForecastingFlujoFondos(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas
    )
    forecasting.ejecutar()

    # 3. Ejecutar Causalidad
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_FLUJOS, FLU_MODULO_RELACION, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    
    
    causalidad = CausalidadEvaluator(
        df=df,
        fondo=fondo,
        columnas=columnas
    )    
    causalidad.ejecutar()    

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "forecasting": forecasting.resultado,
        "causalidad": causalidad.resultado
        }
