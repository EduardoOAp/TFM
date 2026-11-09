from fastapi import FastAPI, HTTPException, Query
from procesamiento_agente.logica_procesamiento import AgenteProcesamiento

app = FastAPI(
    title="API de Procesamiento de Indicadores",
    description="Calcula indicadores financieros y demográficos desde datos recolectados por el scraper."
)

@app.get("/procesar")
def procesar_indicadores(
    fecha_inicio: str = Query(..., description="Fecha de inicio en formato YYYY-MM-DD"),
    fecha_final: str = Query(..., description="Fecha final en formato YYYY-MM-DD"),
    fondo: str = Query("ROP", description="Fondo a analizar")
):
    try:
        agente = AgenteProcesamiento(fondo, fecha_inicio, fecha_final)
        resultados = agente.ejecutar()

        return {
            "fondo": fondo,
            "periodo": f"{fecha_inicio} a {fecha_final}",
            "indicadores": resultados
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))