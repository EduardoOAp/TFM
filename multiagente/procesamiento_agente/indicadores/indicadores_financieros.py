import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd
from . import IndicadorBase

""" Indicadores de desquilibrio financiero
IndicadorRentabilidadAjustadaComision -
IndicadorTasaCrecimientoRentabilidad-
IndicadorVariacionActivoNeto -
IndicadorVolatilidad30Dias -
IndicadorRentabilidadDiaria -
IndicadorSharpeRatio -
IndicadorAlphaJensen -
IndicadorBeta -
IndicadorIndiceEquilibrioFinanciero - 
IndicadorRentabilidadAcumuladaMes
"""

class IndicadorRentabilidadAjustadaComision(IndicadorBase):
    nombre = "rentabilidad_ajustada_comision"
    descripcion = "Evalúa si la rentabilidad neta cubre las comisiones, detectando pérdidas potenciales."
    formula = "((Rentabilidad Neta * Activo Neto) - (Comisiones * Activo Neto)) / Activo Neto * 100"
    fuentes = ["rendimiento", "comision", "cuenta"]
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        pd.set_option("display.max_columns", None)      # muestra todas las filas
        df_ren = self.datos.get("rendimiento")
        df_com = self.datos.get("comision")
        df_cta = self.datos.get("cuenta")
        #print(df_com.head(10))
        #print(df_ren.head(10))
        #print(df_cta.head(10))
        if df_ren is None or df_com is None or df_cta is None:
            return None
        try:
            df_cta = df_cta[(df_cta["entidad"] == entidad) & (df_cta["codigofondo"] == fondo) &
                            (df_cta["cuenta"].str.contains("ACTIVO NETO")) & (df_cta["fecha"] == fecha)]
            
            try:
                total_activos = df_cta["montocolones"].sum()
            except TypeError:
                total_activos = None

            df_com = df_com[(df_com["entidad"] == entidad) & (df_com["codigofondo"] == fondo) &
                            (df_com["tipo"].str.lower() == "saldo") & (df_com["fecha"] == fecha)]
            df_ren = df_ren[(df_ren["entidad"] == entidad) & (df_ren["codigofondo"] == fondo) &
                            (df_ren["periodicidad"].str.lower() == "anual") &
                            (df_ren["tipo"].str.upper() == "REAL") & (df_ren["fecha"] == fecha)]

            total_activos = df_cta["montocolones"].sum()
            comision = df_com["comisión"].mean() * 10
            rentabilidad = df_ren["rentabilidad"].mean()

            if any(pd.isna(x) or x == 0 for x in [total_activos, comision, rentabilidad]):
                return None

            resultado = ((rentabilidad - comision) * total_activos) / total_activos
            return round(float(resultado), 2)
            
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorVariacionActivoNeto(IndicadorBase):
    nombre = "variacion_activo_neto"
    descripcion = "Evalúa la evolución de los activos netos del fondo."
    formula = "((Activo Neto Actual - Activo Neto Anterior) / Activo Neto Anterior) * 100"
    fuentes = ["cuenta"]
    ventana_historica = pd.DateOffset(months=2)
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            fecha_actual = pd.to_datetime(fecha)
            fecha_anterior = fecha_actual - MonthEnd(1) #pd.DateOffset(months=1)

            df_actual = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & 
                           (df["fecha"] == fecha) & (df["cuenta"].str.contains("ACTIVO NETO"))]
            df_anterior = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                             (df["fecha"] == fecha_anterior.strftime('%Y-%m-%d')) & (df["cuenta"].str.contains("ACTIVO NETO"))]

            activo_actual = df_actual["montocolones"].sum()
            activo_anterior = df_anterior["montocolones"].sum()
            if activo_anterior == 0:
                return None

            variacion = ((activo_actual - activo_anterior) / activo_anterior) * 100
            return round(variacion, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorRentabilidadDiaria(IndicadorBase):
    nombre = "rentabilidad_diaria"
    descripcion = "Variación porcentual diaria del valor cuota."
    formula = "((Valor Cuota Hoy - Valor Cuota Ayer) / Valor Cuota Ayer) * 100"
    fuentes = ["cuenta"]
    activo = True
    ventana_historica = pd.Timedelta(days=90)
    frecuencia = "diaria"

    def calcular(self, entidad: str, fondo: str, fecha: str):
        ventana_historica = pd.DateOffset(days=1)        
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None

        try:
            df = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["cuenta"].fillna("").str.upper().str.contains("VALOR.*CUOTA", regex=True))
            ].copy()

            if df.empty:
                return None
            
            df["fecha"] = pd.to_datetime(df["fecha"])
            df = df.sort_values("fecha").set_index("fecha")          
  
            fecha_dt = pd.to_datetime(fecha)
            fecha_ayer = fecha_dt - pd.DateOffset(days=1)

            if fecha_dt not in df.index or fecha_ayer not in df.index:
                return None
  
            cuota_hoy = df.loc[fecha_dt]["montocolones"].mean()
            cuota_ayer = df.loc[fecha_ayer]["montocolones"].mean()
              
            if pd.isna(cuota_hoy) or pd.isna(cuota_ayer) or cuota_ayer == 0:
                return None

            rentabilidad = ((cuota_hoy - cuota_ayer) / cuota_ayer) * 100
            #print(rentabilidad)
            return round(rentabilidad, 4)

        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None
            
class IndicadorVolatilidad30Dias(IndicadorBase):
    nombre = "volatilidad_valorcuota_30d"
    descripcion = "Desviación estándar del valor cuota en los últimos 30 días."
    formula = "STD(valorcuota últimos 30 días)"
    fuentes = ["cuenta"]
    activo = True
    ventana_historica = pd.Timedelta(days=90)
    frecuencia = "diaria"    
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        ventana_historica = pd.DateOffset(days=30)
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            df = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["cuenta"].fillna("").str.upper().str.contains("VALOR.*CUOTA", regex=True))
            ].copy()

            if df.empty:
                return None

            df["fecha"] = pd.to_datetime(df["fecha"])
            df = df.set_index("fecha").sort_index()

            fecha_dt = pd.to_datetime(fecha)
            fecha_min = fecha_dt - ventana_historica
            ventana = df.loc[fecha_min:fecha_dt]

            if ventana.empty:
                return None
            #print(round(ventana["montocolones"].std(), 4))
            std = ventana["montocolones"].std()
            return round(float(std), 4) if pd.notna(std) else None
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorSharpeRatio(IndicadorBase):
    nombre = "sharpe_ratio"
    descripcion = "Mide el exceso de rentabilidad por unidad de riesgo"
    formula = "(R_fondo - R_libre_riesgo) / Desviación estándar"
    fuentes = ["rendimiento"]
    activo = True
    ventana_historica = pd.DateOffset(months=12)
    
    def calcular(self, entidad, fondo, fecha):
        df = self.datos["rendimiento"]
        if df is None or df.empty:
            return None
        try:    

            # Leer tasa libre de riesgo desde parquet
            path = r"D:\pythonenvs\TFM\multiagente\data\insumos\curva_soberana.parquet"
            df_tasa = pd.read_parquet(path)
            df_tasa["fecha"] = pd.to_datetime(df_tasa["fecha"])
            fecha_inicio_mes = pd.to_datetime(fecha).replace(day=1)
            df_tasa["diff"] = (df_tasa["fecha"] - fecha_inicio_mes).abs()
            r_libre = df_tasa.loc[df_tasa["diff"].idxmin(), "valor"] 

            #print(f"aqui tasa {r_libre} fin {fecha_inicio_mes}")
            df_ren = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                #(df["fecha"] >= pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] == pd.to_datetime(fecha))
            ][["entidad","fecha", "rentabilidad"]].copy()
            df_ren["fecha"] = pd.to_datetime(df_ren["fecha"])
            #pd.set_option('display.max_columns', None)            
            #print(df_ren.head(20))
            # Rentabilidades del mismo fondo, pero de otras entidades (mercado)
            df_mk = df[
                (df["entidad"] == entidad) & 
                #(df["entidad"] !="TOTAL") &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                (df["fecha"] > pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] <= pd.to_datetime(fecha))
            ].copy()
            #print(df_mk.head(20))
            if df_ren.empty or df_mk.empty:
                return None
            
            # Agrupar por fecha y calcular desviación estándar de rentabilidad del mercado
            df_std = df_mk.groupby("entidad")["rentabilidad"].std().rename("r_std")
            #print("agrupado")
            #print(df_std.head(20))
            # Unir por entidad
            df_merged = pd.merge(df_ren, df_std, left_on="entidad", right_index=True)
            #print(df_merged.head(20))            
            #df_merged = df_merged[(df_merged["fecha"] == pd.to_datetime(fecha))]
            #print(df_merged.head(20))
            if df_merged.shape[0] < 1:
                return None  # Insuficientes datos para estimar 

            r_fondo = df_merged["rentabilidad"]
            r_std = df_merged["r_std"]

            # Sharpe Ratio por fecha, devolver promedio mensual (o último)
            sharpe_series = (r_fondo - r_libre) / r_std
            #print(sharpe_series.mean())
            return sharpe_series.mean()  

        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None    

class IndicadorAlphaJensen(IndicadorBase):
    nombre = "alpha_jensen"
    descripcion = "Rentabilidad obtenida con respecto a la esperado según su beta y el rendimiento del MK"
    formula = "R_fondo - R_rf - (Beta*(R_mercado - R_rf))"
    fuentes = ["rendimiento"]
    ventana_historica = pd.DateOffset(months=12)    
    activo = True
        
    def calcular(self, entidad, fondo, fecha):
        df = self.datos.get("rendimiento")
        
        # Leer tasa libre de riesgo desde parquet
        path = r"D:\pythonenvs\TFM\multiagente\data\insumos\curva_soberana.parquet"
        df_tasa = pd.read_parquet(path)
        df_tasa["fecha"] = pd.to_datetime(df_tasa["fecha"])    
        fecha_inicio_mes = pd.to_datetime(fecha).replace(day=1)
        df_tasa["diff"] = (df_tasa["fecha"] - fecha_inicio_mes).abs()
        r_libre = df_tasa.loc[df_tasa["diff"].idxmin(), "valor"]             
        
        if df is None or df.empty:
            return None
        try:        

            # Filtrar rentabilidad del fondo (entidad objetivo)
            df_ren = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                #(df["fecha"] >= pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] == pd.to_datetime(fecha))
            ].copy()            
            df_ren["fecha"] = pd.to_datetime(df_ren["fecha"])
            # Filtrar mercado (otras entidades)
            df_mk = df[
                (df["entidad"] != entidad) &
                (df["entidad"] != "TOTAL") &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                #(df["fecha"] >= pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] == pd.to_datetime(fecha))
            ].copy()
            df_mk["fecha"] = pd.to_datetime(df_mk["fecha"])
            if df_ren.empty or df_mk.empty:
                return None

            # Agrupar mercado
            df_mk_agg = df_mk.groupby("fecha")["rentabilidad"].mean().rename("r_mercado")
            df_ren = df_ren[["fecha", "rentabilidad"]].rename(columns={"rentabilidad": "r_fondo"})
            df_merged = pd.merge(df_ren, df_mk_agg, on="fecha")

            if df_merged.shape[0] < 1:
                return None
            
            beta_calculador = IndicadorBeta(self.fondo, self.fecha_inicio, self.fecha_final, self.datos)
            beta = beta_calculador.calcular(entidad, fondo, fecha)

            if beta is None:
                return None

            try:
                r_fondo = df_merged["r_fondo"].mean()
            except TypeError:
                r_fondo  = None 

            try:
                r_mercado = df_merged["r_mercado"].mean()
            except TypeError:
                r_mercado  = None 
            
            if r_fondo == None or r_mercado == None:
                return None

            alfa = r_fondo - r_libre - beta * (r_mercado - r_libre)
            return alfa
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None    

class IndicadorBeta(IndicadorBase):
    nombre = "beta"
    descripcion = "Sensibilidad del fondo al mercado"
    formula = "Covarianza(fondo, mercado) / Varianza(mercado)"
    fuentes = ["rendimiento"]
    activo = True
    ventana_historica = pd.DateOffset(months=12)    

    def calcular(self, entidad, fondo, fecha):
        df = self.datos.get("rendimiento")
        if df is None or df.empty:
            return None
        try:            
            # Filtrar rentabilidad del fondo (entidad objetivo)
            df_ren = df[
                (df["entidad"] == entidad) &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                (df["fecha"] > pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] <= pd.to_datetime(fecha))
            ].copy()            
            df_ren["fecha"] = pd.to_datetime(df_ren["fecha"])
            #pd.set_option('display.max_columns', None)            
            #print(df_ren.head(20))
            # Filtrar mercado (otras entidades)
            df_mk = df[
                (df["entidad"] != entidad) &
                (df["entidad"] != "TOTAL") &
                (df["codigofondo"] == fondo) &
                (df["periodicidad"].str.lower() == "anual") &
                (df["tipo"].str.upper() == "REAL") &
                (df["fecha"] > pd.to_datetime(fecha) - self.ventana_historica) &
                (df["fecha"] <= pd.to_datetime(fecha))
            ].copy()
            df_mk["fecha"] = pd.to_datetime(df_mk["fecha"])
            if df_ren.empty or df_mk.empty:
                return None
            
            df_ren = df_ren[["fecha", "rentabilidad"]].rename(columns={"rentabilidad": "r_fondo"})
            df_mk_agg = df_mk.groupby("fecha")["rentabilidad"].mean().rename("r_mercado")

            df_merged = pd.merge(df_ren, df_mk_agg, on="fecha")

            if df_merged.shape[0] < 3:
                return None

            cov = df_merged["r_fondo"].cov(df_merged["r_mercado"])
            var = df_merged["r_mercado"].var()

            beta = cov / var if var != 0 else None
            return beta
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None   
          
class IndicadorIndiceEquilibrioFinanciero(IndicadorBase):
    nombre = "indice_equilibrio_financiero"
    descripcion = "Mide la relación entre ingresos y gastos para detectar si el fondo es deficitario."
    formula = "Índice de Equilibrio = Ingresos / Gastos"
    fuentes = ["cuenta"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            ingresos = df[df["cuenta"].str.contains("INGRESO")]["montocolones"].sum()
            gastos = df[df["cuenta"].str.contains("GASTO")]["montocolones"].sum()
            if gastos == 0:
                return None
            equilibrio = ingresos / gastos
            return round(equilibrio, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaCrecimientoRentabilidad(IndicadorBase):
    nombre = "tasa_crecimiento_rentabilidad"
    descripcion = "Evalúa si la rentabilidad ha mejorado o empeorado."
    formula = "((Rentabilidad Actual - Rentabilidad Anterior) / Rentabilidad Anterior) * 100"
    fuentes = ["rendimiento"]
    ventana_historica = pd.DateOffset(years=1)
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("rendimiento")
        if df is None or df.empty:
            return None
        try:
            fecha_actual = pd.to_datetime(fecha)
            fecha_anterior = (fecha_actual - pd.DateOffset(years=1)).replace(day=1) + pd.offsets.MonthEnd(0)

            df_actual = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                           (df["periodicidad"].str.lower() == "anual") &
                           (df["tipo"].str.upper() == "REAL") &
                           (df["fecha"] == fecha)]

            df_anterior = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                             (df["periodicidad"].str.lower() == "anual") &
                             (df["tipo"].str.upper() == "REAL") &
                             (df["fecha"] == fecha_anterior.strftime('%Y-%m-%d'))]

            rent_actual = df_actual["rentabilidad"].mean()
            rent_anterior = df_anterior["rentabilidad"].mean()
            if pd.isna(rent_anterior) or pd.isna(rent_actual):
                return None

            if rent_anterior == 0:
                return None

            tasa = ((rent_actual - rent_anterior) / rent_anterior) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorRentabilidadAcumuladaMes(IndicadorBase):
    nombre = "rentabilidad_acumulada_mes"
    descripcion = "Rentabilidad acumulada en el mes actual."
    formula = "((valorcuota fin de mes - inicio) / inicio) * 100"
    fuentes = ["cuenta"]
    ventana_historica = pd.DateOffset(days=31)
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("cuenta")
        if df is None or df.empty:
            return None
        try:
            fecha_dt = pd.to_datetime(fecha)
            inicio_mes = fecha_dt.replace(day=1)

            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo)]
            df = df[df["cuenta"].str.upper().str.contains("VALOR DE LA CUOTA")].copy()

            df["fecha"] = pd.to_datetime(df["fecha"])
            df = df.set_index("fecha").sort_index()

            cuota_inicio = df.loc[inicio_mes, "montocolones"] if inicio_mes in df.index else None
            cuota_actual = df.loc[fecha_dt, "montocolones"] if fecha_dt in df.index else None

            if pd.isna(cuota_inicio) or pd.isna(cuota_actual):
                return None
            if  cuota_inicio == 0:
                return None

            return round(((cuota_actual - cuota_inicio) / cuota_inicio) * 100, 4)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None


