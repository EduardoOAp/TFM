from typing import List, Dict, Optional
import pandas as pd
from pandas import DateOffset

class IndicadorBase:
    nombre: str = ""
    descripcion: str = ""
    formula: str = ""
    fuentes: List[str] = []
    ventana_historica = DateOffset(months=0)
    activo = True
    frecuencia = "Mensual"

    def __init__(self, fondo: str, fecha_inicio: str, fecha_final: str, datos: Dict[str, pd.DataFrame], resultados_previos: Optional[Dict] = None):
        self.fondo = fondo
        self.fecha_inicio = fecha_inicio
        self.fecha_final = fecha_final
        self.datos = datos
        self.resultados_previos = resultados_previos or {}

    def calcular(self, entidad: str, fondo: str, fecha: str):
        raise NotImplementedError("Este método debe ser implementado por cada indicador.")
    
# IMPORTS de indicadores concretos
# Dsequilibrio financiero
from .indicadores_financieros import (
    IndicadorRentabilidadAjustadaComision,
    IndicadorTasaCrecimientoRentabilidad,
    IndicadorVariacionActivoNeto,
    IndicadorVolatilidad30Dias,
    IndicadorRentabilidadDiaria,
    IndicadorSharpeRatio,
    IndicadorAlphaJensen,
    IndicadorBeta,
    IndicadorIndiceEquilibrioFinanciero,
    IndicadorRentabilidadAcumuladaMes,
)  

from .indicadores_riesgo import (
    IndicadorConcentracionActivos,
    IndicadorDiversificacionYRiesgo,
    IndicadorDiversificacionShannon,
    IndicadorHerfindahlHirschmanLocal,
    IndicadorHerfindahlHirschmanExterno,
    IndicadorDistribucionAfiliadosEntidadFondo,
)

## Flujo de fondos
from .indicadores_demograficos import (
    IndicadorTasaAportacion,
    IndicadorTasaFugaAfiliados,
    IndicadorFlujoNetoAfiliadosFondos,
    IndicadorFlujoNetoAfiliadosMontos,
    IndicadorTasaCaptacionAfiliados,
    IndicadorTasaCaptacionAfiliadosLT,
    IndicadorTasaCrecimientoAfiliados,
    IndicadorTasaCrecimientoPensionados,
)


from .indicadores_sostenibilidad import (
    IndicadorIndiceSostenibilidadPensiones,
    IndicadorIndiceSostenibilidadFinanciera,
    IndicadorIndiceEnvejecimiento,
    IndicadorProporcionPensionadosAfiliados,
    IndicadorIndiceCargaAdministrativa,
)

from .indicadores_liquidez import (
    IndicadorIndiceLiquidezPortafolio,
    IndicadorRiesgoLiquidez,
    IndicadorRangoMensualValorCuota,
    IndicadorVolatilidadRecienteActivoNeto,
)