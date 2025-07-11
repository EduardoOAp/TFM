import os

BASE_DIR = os.path.dirname(__file__)
PATH_SERIES = os.path.abspath(os.path.join(BASE_DIR, "../data"))
PATH_INDICADORES = os.path.join(PATH_SERIES, "indicadores")
JSON_PATH = os.path.join(BASE_DIR, "estructura_subagentes.json")
UMBRAL_JSON_PATH = os.path.join(BASE_DIR, "umbrales_activacion.json")