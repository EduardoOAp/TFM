import requests
from fastapi import FastAPI, Request
from riesgo_agente.utils.config import SUBAGENTE_LIQUIDEZ_URL

app = FastAPI(
    title="API de Procesamiento de Riesgo",
    description="Invoca los subagentes."
)

@app.post("/procesar")
async def evaluar_riesgo(request: Request):
    payload = await request.json()

    try:
        print(f"[AGENTE RIESGO] Payload recibido: {payload}")
        print(f"[AGENTE RIESGO] Llamando a: {SUBAGENTE_LIQUIDEZ_URL}")
        response = requests.post(SUBAGENTE_LIQUIDEZ_URL, json=payload)
        response.raise_for_status()
        liquidez_resultado = response.json()
    except Exception as e:
        liquidez_resultado = {"error": str(e)}

    return {
        "liquidez": liquidez_resultado
    }
