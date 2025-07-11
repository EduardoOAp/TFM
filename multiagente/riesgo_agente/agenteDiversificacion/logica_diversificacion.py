from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS, DIV_MODULO_OPTIMIZACION
from riesgo_agente.agenteDiversificacion.clustering_diversificacion import ClusteringEvaluator
from riesgo_agente.agenteDiversificacion.clustering_diversificacion import VisualizadorClusters
from riesgo_agente.agenteDiversificacion.optimizacion_portafolios import OptimizadorPortafolioLiquido
from riesgo_agente.agenteDiversificacion.optimizacion_portafolios import VisualizadorOptimizacion


def ejecutar_subagente_diversificacion(payload: dict):
    fondo = payload["fondo"]
    entidad = payload["entidad"]
    fecha = payload["fecha"]
    usar_normalizado = payload.get("usar_normalizado", False)

    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS, fecha, data_root=DATA_ROOT)
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

    #clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                 usar_normalizado=usar_normalizado)
    clustering.ejecutar()
    
    df_resultado = clustering.df.copy()
    df_resultado["cluster"] = clustering.mejor_modelo[1]["etiquetas"]
    print("[INFO] Clustering completado. Ejemplo de resultado:")

    visualizador = VisualizadorClusters(
        df_resultado, columnas_cluster, False
    )

    visualizador.graficar_dispersion(guardar=True)
    df_scores = clustering._score_normalizado_global()
    visualizador.graficar_comparacion_metricas(df_scores)
    visualizador.graficar_evolucion_por_entidad()


    # 3. Ejecutar Optimizacións
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    columnas_optimizacion= df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    objetivos = ["sharpe_ratio", "indice_liquidez_portafolio"]
    resultados_globales = {}

    for objetivo in objetivos:
        optimizacion = OptimizadorPortafolioLiquido(
            df=df,
            columna_objetivo=objetivo,
            columnas_predictoras=columnas_optimizacion,
            horizonte=6
        )
        resultados = optimizacion.ejecutar()
        resultados_globales[objetivo] = {
            "resultados": resultados,
            "mejor_modelo": optimizacion.mejor_modelo
        }

        # Mostrar resumen
        #print(f"\n📌 Resultados para: {objetivo}")
        #for entidad, res in resultados.items():
        #    if "error" in res:
        #        print(f"❌ {entidad}: {res['error']}")
        #    else:
        #        print(f"✅ {entidad}: mse={res['mse']}, r2={res['r2']}, accuracy_10pct={res['accuracy_10pct']}")

        #print(f"\n🏆 Mejor modelo para {objetivo}: {optimizacion.mejor_modelo[0]}")
        #for k, v in optimizacion.mejor_modelo[1].items():
        #    if k != "modelo":
        #        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    #optimizacion = OptimizadorPortafolioLiquido(df, fondo=fondo, entidad=entidad, columnas=columnas,
    #                                agente=AGENTE_DIVERSIFICACION, modulo=DIV_MODULO_OPTIMIZACION)
    visualizador = VisualizadorOptimizacion(optimizacion.resultados, fecha, False)
    visualizador.graficar_metricas(guardar=True, ruta=f"{fecha}_metricas.png")

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "optimizacion": optimizacion.resultados
        }
