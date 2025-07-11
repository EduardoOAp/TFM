from fastapi import FastAPI, Request
from riesgo_agente.agenteDesequilibrioFinanciero.logica_desequilibrio import ejecutar_subagente_desequilibrio

app = FastAPI(
    title="API de Procesamiento de agente de Desequilibrio",
    description="Invoca modulos del agente de Desequilibrio."
)

print("API DE DESEQUILIBRIO CARGADA CORRECTAMENTE")

@app.post("/desequilibrio")
async def ejecutar_desequilibrio(request: Request):
    required_keys = ["fondo", "entidad", "fecha"]
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Falta el parámetro requerido: {key}")  
          
    payload = await request.json()
    try:
        resultado = ejecutar_subagente_desequilibrio(payload)
        return resultado
    except Exception as e:
        return {"error": str(e)}