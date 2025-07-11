from fastapi import FastAPI, Request
from riesgo_agente.agenteSostenibilidad.logica_sostenibilidad import ejecutar_subagente_sostenibilidad

app = FastAPI(
    title="API de Procesamiento de agente de sostenibilidad",
    description="Invoca modulos del agente de sostenibilidad."
)

print("API DE LIQUIDEZ CARGADA CORRECTAMENTE")

@app.post("/sostenibilidad")
async def ejecutar_sostenibilidad(request: Request):
    required_keys = ["fondo", "entidad", "fecha"]
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Falta el parámetro requerido: {key}")  
          
    payload = await request.json()
    try:
        resultado = ejecutar_subagente_sostenibilidad(payload)
        return resultado
    except Exception as e:
        return {"error": str(e)}