# src/riesgo_module.py

import asyncio
from typing import Dict, Any

# Importa las URLs de la configuración (aunque no las uses para invocar, las usas para definir la lista de subagentes)
from .riesgo_agente.utils.config import (
    SUBAGENTE_LIQUIDEZ_URL, SUBAGENTE_DIVERSIFICACION_URL,
    SUBAGENTE_DESEQUILIBRIO_URL, SUBAGENTE_FLUJOS_URL,
    SUBAGENTE_SOSTENIBILIDAD_URL
)

# Importa las funciones de procesamiento local para cada subagente
from .riesgo_agente import module_subagente_liquidez as liquidez_module
from .riesgo_agente import module_subagente_diversificacion as diversificacion_module
from .riesgo_agente import module_subagente_desequilibrio as desequilibrio_module
from .riesgo_agente import module_subagente_flujofondos as flujos_module
from .riesgo_agente import module_subagente_sostenibilidad as sostenibilidad_module

# Mapeo de nombres de subagentes a sus funciones locales
# Nota: La clave (nombre) debe coincidir con la clave en tu diccionario SUBAGENTES original
# y el valor debe ser la función que procesa la carga útil.
LOCAL_SUBAGENTES_FUNCTIONS = {
    "liquidez": liquidez_module.process_liquidez,
    "diversificacion": diversificacion_module.process_diversificacion,
    "desequilibrio": desequilibrio_module.process_desequilibrio, # Asumiendo que creas este módulo y función
    "flujos": flujos_module.process_flujoFondos, # Asumiendo que creas este módulo y función
    "sostenibilidad": sostenibilidad_module.process_sostenibilidad # Asumiendo que creas este módulo y función
}

# La función principal que tu orquestador llamará
async def run_riesgo(fondo: str, fecha: str, entidad: str) -> Dict[str, Any]:
    """
    Simula la ejecución del API de Riesgo localmente, invocando subagentes locales.
    """
    print(f"  [RIESGO_MODULE] Ejecutando run_riesgo localmente para fondo: {fondo}, fecha: {fecha}, entidad: {entidad}")

    # Construye el payload que se enviaría a los subagentes (similar al de la API FastAPI)
    payload_for_subagents = {
        "fondo": fondo,
        "fecha": fecha,
        "entidad": entidad,
        # Puedes añadir otros datos relevantes que los subagentes necesiten
    }

    # Prepara las tareas para invocar a cada subagente local
    tareas = []
    for nombre, func in LOCAL_SUBAGENTES_FUNCTIONS.items():
        # Cada tarea es una llamada a la función asíncrona del subagente
        tareas.append(invocar_subagente_local(nombre, func, payload_for_subagents))

    # Ejecuta todas las tareas de forma concurrente
    resultados = await asyncio.gather(*tareas)

    # Formatea los resultados igual que tu API FastAPI
    return {nombre: resultado for nombre, resultado in resultados}

# Función auxiliar para invocar un subagente local
async def invocar_subagente_local(nombre: str, func, payload: Dict[str, Any]):
    try:
        # Llama directamente a la función local del subagente
        result = await func(payload)
        return nombre, result
    except Exception as e:
        print(f"[ERROR SUBAGENTE LOCAL] Fallo en {nombre}: {str(e)}")
        return nombre, {"error": str(e)}

# Para hacer el módulo ejecutable directamente si lo deseas, o para pruebas rápidas
if __name__ == "__main__":
    async def test_run():
        test_payload = {
            "fondo": "ROP",
            "fecha": "2025-09-30",
            "entidad": "Ninguna"
        }
        results = await run_riesgo(
            fondo=test_payload["fondo"],
            fecha=test_payload["fecha"],
            entidad=test_payload["entidad"]
        )
        print("\nResultados de la ejecución de Riesgo Local:")
        print(results)

    asyncio.run(test_run())

