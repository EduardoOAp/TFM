from procesamiento_agente.indicadores import (
    IndicadorConcentracionActivos,
    IndicadorTasaAportacion,
    IndicadorIndiceEnvejecimiento,
    IndicadorTasaFugaAfiliados,
    IndicadorIndiceCargaAdministrativa,
    IndicadorIndiceEquilibrioFinanciero,
    IndicadorDiversificacionYRiesgo,
    IndicadorRentabilidadAjustadaComision,
    IndicadorIndiceSostenibilidadPensiones,
    IndicadorProporcionPensionadosAfiliados,
    IndicadorIndiceSostenibilidadFinanciera,
    IndicadorTasaCaptacionAfiliadosLT,
    IndicadorTasaCrecimientoPensionados,
    IndicadorTasaCrecimientoRentabilidad,
    IndicadorVariacionActivoNeto,
    IndicadorTasaCrecimientoAfiliados,
    IndicadorTasaCaptacionAfiliados,
    IndicadorVolatilidadRecienteActivoNeto,
    IndicadorDistribucionAfiliadosEntidadFondo,
    IndicadorFlujoNetoAfiliadosFondos,
    IndicadorFlujoNetoAfiliadosMontos,
    IndicadorDiversificacionShannon,
    IndicadorIndiceLiquidezPortafolio,
    IndicadorRiesgoLiquidez,
    IndicadorRentabilidadDiaria,
    IndicadorVolatilidad30Dias,
    IndicadorRentabilidadAcumuladaMes,
    IndicadorRangoMensualValorCuota,
    IndicadorSharpeRatio,
    IndicadorAlphaJensen,
    IndicadorBeta,
    IndicadorHerfindahlHirschmanLocal,    
    IndicadorHerfindahlHirschmanExterno,
)

# Diccionario global: nombre del indicador -> clase del indicador
REGISTRO_INDICADORES = {
    clase.nombre: {
        "clase": clase,
        "activo": getattr(clase, "activo", True),  # default: True
        "frecuencia": getattr(clase, "frecuencia", "mensual")
    }
    for clase in [
        IndicadorConcentracionActivos,
        IndicadorTasaAportacion,
        IndicadorIndiceEnvejecimiento,
        IndicadorTasaFugaAfiliados,
        IndicadorIndiceCargaAdministrativa,
        IndicadorIndiceEquilibrioFinanciero,
        IndicadorDiversificacionYRiesgo,
        IndicadorRentabilidadAjustadaComision,
        IndicadorIndiceSostenibilidadPensiones,
        IndicadorProporcionPensionadosAfiliados,
        IndicadorIndiceSostenibilidadFinanciera,
        IndicadorTasaCaptacionAfiliadosLT,
        IndicadorTasaCrecimientoPensionados,
        IndicadorTasaCrecimientoRentabilidad,
        IndicadorVariacionActivoNeto,
        IndicadorTasaCrecimientoAfiliados,
        IndicadorTasaCaptacionAfiliados,
        IndicadorVolatilidadRecienteActivoNeto,
        IndicadorDistribucionAfiliadosEntidadFondo,
        IndicadorFlujoNetoAfiliadosFondos,
        IndicadorFlujoNetoAfiliadosMontos,
        IndicadorDiversificacionShannon,
        IndicadorIndiceLiquidezPortafolio,
        IndicadorRiesgoLiquidez,
        IndicadorRentabilidadDiaria,
        IndicadorVolatilidad30Dias,
        IndicadorRentabilidadAcumuladaMes,
        IndicadorRangoMensualValorCuota,
        IndicadorSharpeRatio,
        IndicadorAlphaJensen,
        IndicadorBeta,
        IndicadorHerfindahlHirschmanLocal,    
        IndicadorHerfindahlHirschmanExterno,
    ]
}

def obtener_fuentes_unicas():
    fuentes = set()
    for clase in REGISTRO_INDICADORES.values():
        fuentes.update(clase.fuentes)
    return sorted(fuentes)
