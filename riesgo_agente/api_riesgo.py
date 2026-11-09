import asyncio
import httpx
from fastapi import FastAPI, Request
from riesgo_agente.utils.config import (
    SUBAGENTE_LIQUIDEZ_URL,
    SUBAGENTE_DIVERSIFICACION_URL,
    SUBAGENTE_DESEQUILIBRIO_URL,
    SUBAGENTE_FLUJOS_URL,
    SUBAGENTE_SOSTENIBILIDAD_URL
)

app = FastAPI(title="API de Procesamiento de Riesgo")

SUBAGENTES = {
    "liquidez": SUBAGENTE_LIQUIDEZ_URL,
    "diversificacion": SUBAGENTE_DIVERSIFICACION_URL,
    "desequilibrio": SUBAGENTE_DESEQUILIBRIO_URL,
    "flujos": SUBAGENTE_FLUJOS_URL,
    "sostenibilidad": SUBAGENTE_SOSTENIBILIDAD_URL
}

async def invocar_subagente(nombre: str, url: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return nombre, response.json()
    except Exception as e:
        return nombre, {"error": str(e)}

@app.post("/procesar")
async def evaluar_riesgo(request: Request):
    payload = await request.json()

    tareas = [
        invocar_subagente(nombre, url, payload)
        for nombre, url in SUBAGENTES.items()
    ]
    resultados = await asyncio.gather(*tareas)
    
    return {nombre: resultado for nombre, resultado in resultados}
