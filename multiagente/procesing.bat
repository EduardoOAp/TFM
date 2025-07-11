@echo off
REM === Script para levantar la API del agente de procesamiento ===

REM Ruta base del proyecto (ajusta si es necesario)
cd /d D:\pythonenvs\TFM\multiagente

REM Establecer PYTHONPATH para evitar problemas de importación relativa
set PYTHONPATH=D:\pythonenvs\TFM\multiagente

REM Ejecutar el servidor FastAPI usando uvicorn
uvicorn procesamiento_agente.api_procesamiento:app --reload --port 8001

pause
