import json
import os

def cargar_config(nombre_archivo: str) -> dict:
    ruta = os.path.join(os.path.dirname(__file__), nombre_archivo)
    #print('ruta',ruta)
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)
