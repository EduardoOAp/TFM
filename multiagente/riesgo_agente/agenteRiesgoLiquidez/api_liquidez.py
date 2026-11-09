from fastapi import FastAPI, Request
from riesgo_agente.agenteRiesgoLiquidez.logica_liquidez import ejecutar_subagente_liquidez

app = FastAPI(
    title="API de Procesamiento de agente de liquidez",
    description="Invoca modulos del agente de liquidez ."
)

print("API DE LIQUIDEZ CARGADA CORRECTAMENTE")

@app.post("/liquidez")
async def ejecutar_liquidez(request: Request):
    required_keys = ["fondo", "entidad", "fecha"]
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Falta el parámetro requerido: {key}")  
          
    payload = await request.json()
    try:
        resultado = ejecutar_subagente_liquidez(payload)
        return resultado
    except Exception as e:
        return {"error": str(e)}