from . import IndicadorBase
import pandas as pd
from procesamiento_agente.utils import get_parametro_indicador
"""
IndicadorIndiceLiquidezPortafolio
IndicadorRiesgoLiquidez
IndicadorRangoMensualValorCuota
IndicadorVolatilidadRecienteActivoNeto
"""

# Clasificación de liquidez
def es_liquido(row):
    instrumento = (row["instrumento"] or "").strip().upper()
    tipo_fondo = (row["fondoinversion"] or "").strip().upper()
    plazo = row.get("codigoplazo", 99)  # por si viene vacío

    if instrumento == "DEUDA ESTANDARIZADA" and plazo <= 2:
        return True
    elif instrumento in {"PAPEL COMERCIAL", "RECOMPRAS", "ACCIONES", "DEUDA INDIVIDUAL"}:
        return True
    elif tipo_fondo in {"MERCADO DE DINERO", "ETF", "MUTUO"}:
        return True
    return False

class IndicadorRangoMensualValorCuota(IndicadorBase):
    nombre = "rango_mensual_valorcuota"
    descripcion = "Diferencia entre máximo y mínimo valor cuota del mes."
    formula = "max - min del valorcuota en el mes"
    fuentes = ["cuenta"]
    ventana_historica = pd.Timedelta(days=90)
    activo = True
    frecuencia = "diaria"
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            ventana_indicador = pd.Timedelta(days=30)
            fecha_dt = pd.to_datetime(fecha)
            inicio_ventana = fecha_dt - ventana_indicador
           
            df = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["cuenta"].fillna("").str.upper().str.contains("VALOR.*CUOTA", regex=True)) &
                (df["fecha"] >= inicio_ventana) &
                (df["fecha"] <= fecha_dt)
            ].copy()
            
            if df.empty:
                return None
            if len(df) < ventana_indicador.days:
                print(f"no hay suficientes días del dataset {self.nombre}")
                return None  # o retornar un valor como 0, dependiendo de cómo quieras manejar este caso                        

            df["fecha"] = pd.to_datetime(df["fecha"])
            df = df.set_index("fecha").sort_index()

            if pd.Series(df.index.date).nunique() < ventana_indicador.days:
                print(f"[INFO] {self.nombre}: solo {df.index.date.nunique()} días disponibles.")
                return None
            #print(round(df["montocolones"].max() - df["montocolones"].min(), 4))
            return round(df["montocolones"].max() - df["montocolones"].min(), 4)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorVolatilidadRecienteActivoNeto(IndicadorBase):
    nombre = "volatilidad_reciente_activo_neto"
    descripcion = "Desviación estándar del activo neto en ventana móvil."
    formula = "Rolling STD(Activo Neto, ventana=6 meses)"
    fuentes = ["cuenta"]
    ventana_historica = pd.DateOffset(years=1)
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            
            ventana = get_parametro_indicador(self.nombre, "ventana_meses", 6) 
            df_filtrado = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & 
                             df["cuenta"].str.contains("ACTIVO NETO")].copy()
            df_filtrado["fecha"] = pd.to_datetime(df_filtrado["fecha"])
            df_filtrado = df_filtrado.groupby("fecha")["montocolones"].sum().sort_index()

            if len(df_filtrado) < ventana:
                return None

            rolling_std = df_filtrado.rolling(window=ventana).std()
            rolling_mean = df_filtrado.rolling(window=ventana).mean()

            if fecha not in rolling_std.index or pd.isna(rolling_std.loc[fecha]) or pd.isna(rolling_mean.loc[fecha]):
                return None

            volatilidad_pct = (rolling_std.loc[fecha] / rolling_mean.loc[fecha]) * 100

            return round(volatilidad_pct, 2)  # devuelve porcentaje con 2 decimales
            #rolling_volatilidad = df_filtrado.rolling(window=ventana).std()
            #resultado = rolling_volatilidad.loc[fecha] if fecha in rolling_volatilidad.index else None

            #return round(resultado, 2) if resultado and not pd.isna(resultado) else None
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorIndiceLiquidezPortafolio(IndicadorBase):
    nombre = "indice_liquidez_portafolio"
    descripcion = "Porcentaje de activos líquidos sobre total invertido."
    formula = "(Inversión en Activos Líquidos / Total de Inversiones) * 100"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("portafolio")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)].copy()

            total = df["montocolones"].sum()
            if total == 0:
                return None            
            # Aplicar la clasificación
            df["es_liquido"] = df.apply(es_liquido, axis=1)
            liquidos = df[df["es_liquido"]]["montocolones"].sum()

            return round((liquidos / total) * 100, 2) if total > 0 else None
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorRiesgoLiquidez(IndicadorBase):
    nombre = "indice_riesgo_liquidez"
    descripcion = "Porcentaje de activos ilíquidos sobre total invertido."
    formula = "(Activos Ilíquidos / Total de Inversiones) * 100"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("portafolio")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)].copy()
            total = df["montocolones"].sum()
            if total == 0:
                return None            
            # Aplicar la clasificación
            df["es_liquido"] = df.apply(es_liquido, axis=1)
            iliquidos = df[~df["es_liquido"]]["montocolones"].sum()
            return round((iliquidos / total) * 100, 2) if total > 0 else None
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None
                    

