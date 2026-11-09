from .agenteRiesgoLiquidez.logica_liquidez import ejecutar_subagente_liquidez
from typing import Dict, Any
import asyncio

async def process_liquidez(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función que simula la API del subagente de Desequilibrio.
    Directamente invoca la lógica de negocio principal.
    """
    print(f"      [SUBAGENTE LIQUIDEZ] Invocando lógica interna para liquidez...")

    # Validar las claves requeridas del payload, similar a como lo hace tu API.
    # En este punto, el payload ya debería estar limpio por el orquestador principal.
    required_keys = ["fondo", "entidad", "fecha"]    
    for key in required_keys:
        if key not in payload:
            print(f"[ERROR SUBAGENTE LIQUIDEZ] Payload missing required key: {key}")
            # Puedes lanzar un error o devolver un dict de error.
            return {"error": f"Falta el parámetro requerido: {key}"}

    try:
        # Llamamos a tu función de lógica de negocio real
        # Si ejecutar_subagente_desequilibrio no es async, no necesitas await
        resultado = ejecutar_subagente_liquidez(payload)
        return resultado
    except Exception as e:
        print(f"[ERROR SUBAGENTE LIQUIDEZ] Error durante la ejecución: {str(e)}")
        return {"error": str(e)}
    
# Para pruebas directas de este módulo si lo necesitas
if __name__ == "__main__":
    async def test_subagente():
        sample_payload = {
            "fondo": "MiFondo",
            "entidad": "MiEntidad",
            "fecha": "2025-07-29"
        }
        result = await process_flujoFondos(sample_payload)
        print("\nResultado del subagente liquidez (directo):")
        print(result)

    asyncio.run(test_subagente())    