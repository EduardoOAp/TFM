from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS, DIV_MODULO_OPTIMIZACION
from riesgo_agente.agenteDiversificacion.clustering_diversificacion import ClusteringEvaluator
from riesgo_agente.agenteDiversificacion.optimizacion_portafolios import OptimizadorPortafolioLiquido


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
    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                     usar_normalizado=usar_normalizado)
    clustering.ejecutar()

    # 3. Ejecutar Anomalías
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    optimizacion = OptimizadorPortafolioLiquido(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                    agente=AGENTE_DIVERSIFICACION, modulo=DIV_MODULO_OPTIMIZACION)
    optimizacion.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "optimizacion": optimizacion.resultado
        }
