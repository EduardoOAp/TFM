import os

BASE_DIR = os.path.dirname(__file__)
PATH_SERIES = os.path.abspath(os.path.join(BASE_DIR, "../data"))
PATH_INDICADORES = os.path.join(PATH_SERIES, "indicadores")
BASE_DIR2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR2, "configuracion", "estructura_subagentes.json")
UMBRAL_JSON_PATH = os.path.join(BASE_DIR2, "configuracion", "umbrales_activacion.json")
#JSON_PATH = "estructura_subagentes.json"
#UMBRAL_JSON_PATH = "umbrales_activacion.json"
