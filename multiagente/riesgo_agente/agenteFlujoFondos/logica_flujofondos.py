from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_FLUJOS, FLU_MODULO_CLUSTERING, FLU_MODULO_FORECASTING, FLU_MODULO_RELACION
from riesgo_agente.agenteFlujoFondos.clustering_flujofondos import ClusteringEvaluator
from riesgo_agente.agenteFlujoFondos.clustering_flujofondos import VisualizadorClusters
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

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    clustering = ClusteringEvaluator(
        df=df,
        fondo=fondo, 
        entidad=entidad,
        columnas=columnas_cluster,
        usar_normalizado=usar_normalizado
        )

    clustering.ejecutar()

    df_resultado = clustering.df.copy()
    df_resultado["cluster"] = clustering.mejor_modelo[1]["etiquetas"]

    visualizador = VisualizadorClusters(
        df_resultado, columnas_cluster, False
    )

    visualizador.graficar_dispersion(guardar=True)
    df_scores = clustering._score_normalizado_global()
    visualizador.graficar_comparacion_metricas(df_scores)
    visualizador.graficar_evolucion_por_entidad()

    # 3. Ejecutar Forecasting
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_FLUJOS, FLU_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    
    columna_objetivo="tasa_fuga_afiliados"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()    
    
    forecasting = ForecastingFlujoFondos(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar, mostrarGrafico=False
    )
    forecasting.ejecutar()

    # 3. Ejecutar Causalidad
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_FLUJOS, FLU_MODULO_RELACION, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)  

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()  
    
    causalidad = CausalidadEvaluator(
        df=df,
        fondo=fondo,
        columnas=columnas_cluster
    )    
    causalidad.ejecutar()    

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "forecasting": forecasting.resultado,
        "causalidad": causalidad.resultado
        }
