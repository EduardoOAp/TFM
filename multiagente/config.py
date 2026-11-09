from datetime import datetime

DATA_ROOT = "data"
SCRAPER_API_URL = "http://localhost:8002/fetch/"
PROCESAMIENTO_API_URL = "http://localhost:8001/procesar"
EDA_API_URL = "http://localhost:8003/eda_recomendar"
RIESGO_API_URL = "http://localhost:8004/procesar"
TIPOS = ["comision", "rendimiento", "afiliado", "cuenta", "portafolio", "beneficio", "libre_transferencia"]
FONDO = "ROP"
FECHA_INICIAL_DEFAULT = datetime(2010, 7, 1)