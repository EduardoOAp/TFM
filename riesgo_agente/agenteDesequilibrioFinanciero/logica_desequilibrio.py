from riesgo_agente.utils.helpers import cargar_dataframe_temporal, filtrar_entidades_con_historia, columnas_validas
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING, DES_MODULO_ANOMALIAS, DES_MODULO_FORECASTING, DES_MODULO_FORECAST_VC
from riesgo_agente.agenteDesequilibrioFinanciero.clustering_financiero import ClusteringEvaluator
from riesgo_agente.agenteDesequilibrioFinanciero.deteccion_anomalias import AnomaliaDetector
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

    # 2. Ejecutar Clustering
    clustering = ClusteringEvaluator(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                     usar_normalizado=usar_normalizado)
    clustering.ejecutar()

    # 3. Ejecutar Anomalías
    # 1. Cargar y preparar el DataFrame
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_ANOMALIAS, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)    

    anomalias = AnomaliaDetector(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                    agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_ANOMALIAS)
    anomalias.ejecutar()

    # 4. Ejecutar Forecasting AN
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_FORECASTING, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)      
    forecasting = ForecastingActivoNeto(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                       agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_FORECASTING)
    forecasting.ejecutar()

    # 4. Ejecutar Forecasting VC
    df = cargar_dataframe_temporal(AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC, fecha, data_root=DATA_ROOT)
    df = filtrar_entidades_con_historia(df)      
    forecast_vc = ForecastingValorCuota(df, fondo=fondo, entidad=entidad, columnas=columnas,
                                       agente=AGENTE_DESEQUILIBRIO, modulo=DES_MODULO_FORECAST_VC)
    forecast_vc.ejecutar()

    # 5. Armar resultado si se requiere
    return {
        "clustering": clustering.resultado,
        "anomalias": anomalias.resultado,
        "forecasting_AN": forecasting.resultado,
        "forecasting_VC": forecast_vc.resultado
    }
