from fastapi import FastAPI, Request
from riesgo_agente.agenteDiversificacion.logica_diversificacion import ejecutar_subagente_diversificacion

app = FastAPI(
    title="API de Procesamiento de agente de Desequilibrio",
    description="Invoca modulos del agente de Desequilibrio."
)

print("API DE DESEQUILIBRIO CARGADA CORRECTAMENTE")

@app.post("/diversificacion")
async def ejecutar_diversificacion(request: Request):
    required_keys = ["fondo", "entidad", "fecha"]
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Falta el parámetro requerido: {key}")  
          
    payload = await request.json()
    try:
        resultado = ejecutar_subagente_diversificacion(payload)
        return resultado
    except Exception as e:
        return {"error": str(e)}