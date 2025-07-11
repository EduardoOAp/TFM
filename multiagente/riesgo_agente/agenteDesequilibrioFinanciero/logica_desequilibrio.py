from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING, DES_MODULO_ANOMALIAS, DES_MODULO_FORECASTING, DES_MODULO_FORECAST_VC
from riesgo_agente.agenteDesequilibrioFinanciero.clustering_financiero import ClusteringEvaluator
from riesgo_agente.agenteDesequilibrioFinanciero.clustering_financiero import VisualizadorClusters
from riesgo_agente.agenteDesequilibrioFinanciero.deteccion_anomalias import AnomaliaDetector
from riesgo_agente.agenteDesequilibrioFinanciero.deteccion_anomalias import VisualizadorAnomalias
from riesgo_agente.agenteDesequilibrioFinanciero.forecasting_activo_neto import ForecastingActivoNeto
from riesgo_agente.agenteDesequilibrioFinanciero.forecasting_valor_cuota import ForecastingValorCuota

def ejecutar_subagente_desequilibrio(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)
    columnas = columnas_validas(df)
    mostrarGrafico = False
    # 2. Ejecutar Clustering

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    clustering = ClusteringEvaluator(
        df=df,
        fondo=fondo,
        entidad=entidad,
        columnas=columnas_cluster,
        usar_normalizado=usar_normalizado
    )

    #clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                 usar_normalizado=usar_normalizado)
    clustering.ejecutar()

    df_resultado = clustering.df.copy()
    df_resultado["cluster"] = clustering.mejor_modelo[1]["etiquetas"]
    visualizador = VisualizadorClusters(
        df_resultado, columnas_cluster, mostrarGrafico
    )

    # Tablas de resumen
    visualizador.graficar_dispersion(guardar=True)
    df_scores = clustering._score_normalizado_global()
    visualizador.graficar_comparacion_metricas(df_scores)
    visualizador.graficar_evolucion_por_entidad()

    # 3. Ejecutar Anomalías
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_ANOMALIAS, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    rolling = True
    columnasAnomalias = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()
    columnasAnomalias = ["tasa_crecimiento_aportantes","tasa_crecimiento_pensionados","tasa_crecimiento_rentabilidad"]
    # Definir las columnas que deseas conservar
    columnas_deseadas = ["entidad", "fondo", "fecha", 
                        "tasa_crecimiento_aportantes", 
                        "tasa_crecimiento_pensionados", 
                        "tasa_crecimiento_rentabilidad"]

    # Filtrar el DataFrame original para conservar solo esas columnas
    df = df[columnas_deseadas].copy()
    df_resultado = None
    anomalias = AnomaliaDetector(
        df=df,
        df_resultado=df_resultado,
        rolling=rolling,
        columnas=columnasAnomalias,
        por_entidad=True
    )
    #anomalias = AnomaliaDetector(df, columnas=columnas,
    #                                agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_ANOMALIAS)
    anomalias.ejecutar()

    df_resultado = anomalias.df_resultado
    visualizador = VisualizadorAnomalias(
        df_resultado=df_resultado, columnas_valores=columnasAnomalias, por_entidad=True, mostrarGrafico=mostrarGrafico
    )
    visualizador.graficar_dispersion()
    visualizador.graficar_anomalias_temporales_todas(altura_por_entidad=2)
    #print(visualizador.resumen_por_entidad())
    visualizador.mapa_calor_variables()

    # 4. Ejecutar Forecasting AN
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)      
    
    columna_objetivo="variacion_activo_neto"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    forecasting = ForecastingActivoNeto(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = forecasting.ejecutar()
    #forecasting = ForecastingActivoNeto(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                   agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_FORECASTING)

    # 4. Ejecutar Forecasting VC
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)

    columna_objetivo="valor_cuota"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    forecast_vc = ForecastingValorCuota(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = forecast_vc.ejecutar()

    #forecast_vc = ForecastingValorCuota(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                   agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_FORECAST_VC)

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "anomalias": anomalias.resultado,
        "forecasting_AN": forecasting.resultado,
        "forecasting_VC": forecast_vc.resultado
    }
