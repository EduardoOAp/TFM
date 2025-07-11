#### Paso 2: Levantar API del Scraper (en una terminal)
#### uvicorn scraper_agente.api_scraper:app --reload --port 8002

#### Paso 3: Levantar API del Procesamiento (en otra terminal)
#### uvicorn procesamiento_agente.api_procesamiento:app --reload --port 8001

#### Paso 4: Levantar API del EDA (en otra terminal)
#### uvicorn EDArecomend_agente.api_eda:app --reload --port 8003

#### Paso 5: Levantar API de Riesgo (en otra terminal)
#### uvicorn riesgo_agente.api_riesgo:app --reload --port 8004

#### Paso 6: Levantar otras API  (en otra terminal)
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_liquidez:app --reload --port 8500
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_diversificacion:app --reload --port 8501
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_desequilibrio:app --reload --port 8502
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_flujofondos:app --reload --port 8503
python -m uvicorn riesgo_agente.agenteRiesgoLiquidez.api_sostenibilidad:app --reload --port 8504

#### Paso 4: Ejecutar el orquestador (en otra terminal o programado en cron)
#### python main.py ejecutar_scraper=False ejecutar_procesamiento=False ejecutar_eda=False ejecutar_riesgo=True
