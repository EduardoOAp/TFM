import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Asegurar que el path raíz esté incluido
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from riesgo_agente.agenteDiversificacion.optimizacion_portafolios import OptimizadorPortafolioLiquido
from riesgo_agente.agenteDiversificacion.optimizacion_portafolios import VisualizadorOptimizacion

def cargar_datos(ruta_parquet: str) -> pd.DataFrame:
    if not os.path.exists(ruta_parquet):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_parquet}")
    logging.info(f"Cargando datos desde: {ruta_parquet}")
    df = pd.read_parquet(ruta_parquet)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def main():
    DATA_ROOT = "data"
    agente = "Agente de Diversificación"
    modulo = "Optimización de portafolios líquidos"
    fecha = "2024-12-31"
    fondo = "ROP"

    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha + ".parquet")
    df = cargar_datos(ruta)

    # Filtrar entidades con más de 24 meses
    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    objetivos = ["sharpe_ratio", "indice_liquidez_portafolio"]
    resultados_globales = {}

    for objetivo in objetivos:
        logging.info(f"\n🔁 Procesando objetivo: {objetivo}")
        evaluator = OptimizadorPortafolioLiquido(
            df=df,
            columna_objetivo=objetivo,
            columnas_predictoras=columnas_cluster,
            horizonte=6
        )
        resultados = evaluator.ejecutar()
        resultados_globales[objetivo] = {
            "resultados": resultados,
            "mejor_modelo": evaluator.mejor_modelo
        }

        # Mostrar resumen
        print(f"\n📌 Resultados para: {objetivo}")
        for entidad, res in resultados.items():
            if "error" in res:
                print(f"❌ {entidad}: {res['error']}")
            else:
                print(f"✅ {entidad}: mse={res['mse']}, r2={res['r2']}, accuracy_10pct={res['accuracy_10pct']}")

        print(f"\n🏆 Mejor modelo para {objetivo}: {evaluator.mejor_modelo[0]}")
        for k, v in evaluator.mejor_modelo[1].items():
            if k != "modelo":
                print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # Resultados
    #mejor_entidad, resultado = evaluator.mejor_modelo
    #logging.info(f"Mejor modelo entrenado para entidad: {mejor_entidad}")
    #for k, v in resultado.items():
    #    logging.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    visualizador = VisualizadorOptimizacion(evaluator.resultados)
    visualizador.graficar_metricas(guardar=True, ruta=f"{fecha}_metricas.png")
    print(visualizador.mostrar_tabla())


if __name__ == "__main__":
    main()
