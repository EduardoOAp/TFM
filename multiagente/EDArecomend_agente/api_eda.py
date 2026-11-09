from fastapi import FastAPI, HTTPException, Query
from EDArecomend_agente.logica_EDA import AgenteEDARecomendador

app = FastAPI(
    title="API de EDA",
    description="Obtiene las columnas más adecuadas para su proceso de machime learning."
)

@app.get("/eda_recomendar")
def ejecutar_EDA(
    fecha: str = Query(..., description="Fecha de inicio en formato YYYY-MM-DD"),
    fecha_final: str = Query(..., description="Fecha final en formato YYYY-MM-DD"),
    fondo: str = Query("ROP", description="Fondo a analizar")
):
    try:
        agente = AgenteEDARecomendador(fecha, fecha_final, fondo)
        resultados = agente.ejecutar_EDA()
        # Se supone que el resultado es una lista con la ubicación de los parquet
        # que luego se debe pasar al agente de riesgo
        return {
            "fondo": fondo,
            "fecha": f"{fecha}",
            "indicadores": resultados
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))