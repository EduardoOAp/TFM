from fastapi import FastAPI, Request
from riesgo_agente.agenteFlujoFondos.logica_flujofondos import ejecutar_subagente_flujofondos

app = FastAPI(
    title="API de Procesamiento de Agente de Flujos de Fondos",
    description="Invoca modulos del Agente de Flujos de Fondos."
)

print("API DE FLUJO DE FONDOS CARGADA CORRECTAMENTE")

@app.post("/flujo_fondos")
async def ejecutar_diversificacion(request: Request):
    required_keys = ["fondo", "entidad", "fecha"]
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Falta el parámetro requerido: {key}")  
          
    payload = await request.json()
    try:
        resultado = ejecutar_subagente_flujofondos(payload)
        return resultado
    except Exception as e:
        return {"error": str(e)}