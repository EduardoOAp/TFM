from . import IndicadorBase
import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd

""" ## Flujo de fondos
IndicadorTasaAportacion
IndicadorTasaFugaAfiliados
IndicadorFlujoNetoAfiliadosFondos
IndicadorTasaCaptacionAfiliados
IndicadorTasaCaptacionAfiliadosLT
IndicadorTasaCrecimientoAfiliados
IndicadorTasaCrecimientoPensionados
"""
class IndicadorTasaAportacion(IndicadorBase):
    nombre = "tasa_aportacion"
    descripcion = "Porcentaje de afiliados que están contribuyendo activamente."
    formula = "(Número de Aportantes / Número Total de Afiliados) * 100"
    fuentes = ["afiliado"]
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("afiliado")
        if df is None or df.empty:
            return None
        try:
            df_todo = df[(df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            total_afiliados = df_todo["aportantes"].sum()
            total_aportantes = df["aportantes"].sum()
            if total_afiliados == 0:
                return None
            if pd.isna(total_afiliados) or total_afiliados == 0:
                return None            
            tasa = (total_aportantes / total_afiliados) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaFugaAfiliados(IndicadorBase):
    nombre = "tasa_fuga_afiliados"
    descripcion = "Evalúa si los afiliados están abandonando masivamente un fondo."
    formula = "Tasa de Fuga = (Afiliados Salientes / Total Aportantes) * 100"
    fuentes = ["libre_transferencia", "afiliado"]
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_lt = self.datos.get("libre_transferencia")
        if "entidadorigen" in df_lt.columns:
            df_lt = df_lt.rename(columns={"entidadorigen": "entidad"})          
        df_afi = self.datos.get("afiliado")
        if df_lt is None or df_afi is None:
            return None
        try:
            df_afi = df_afi[(df_afi["codigofondo"] == fondo) & (df_afi["fecha"] == fecha)]
            df_lt = df_lt[(df_lt["entidad"] == entidad) & (df_lt["fecha"] == fecha) & (df_lt["codigofondo"] == fondo)]            

            total_aportantes = df_afi["aportantes"].sum()
            salientes = df_lt["cantidad_saliente"].sum()
            #print(f"[DEBUG] apo {total_aportantes} salientes {salientes}")

            if total_aportantes == 0:
                return None
            
            if pd.isna(total_aportantes) or total_aportantes == 0:
                return None

            tasa = (salientes / total_aportantes) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaCrecimientoAfiliados(IndicadorBase):
    nombre = "tasa_crecimiento_aportantes"
    descripcion = "Muestra si el sistema de pensiones está ganando o perdiendo aportantes con el tiempo."
    formula = "((Aportantes Actuales - Aportantes Previos) / Aportantes Previos) * 100"
    fuentes = ["afiliado"]
    ventana_historica = pd.DateOffset(months=14)
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("afiliado")
        if df is None or df.empty:
            return None
        try:
            df["fecha"] = pd.to_datetime(df["fecha"])

            #conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
            #print(conteo_fechas)

            # Verificar fechas faltantes por entidad
            #entidades = df["entidad"].unique()
            #for ent in entidades:
            #    fechas_ent = df[df["entidad"] == ent]["fecha"]
            #    print(f"{ent}: fechas únicas = {fechas_ent.nunique()}, rango = {fechas_ent.min()} → {fechas_ent.max()}")    


            fecha_actual = pd.to_datetime(fecha)
            fecha_anterior = (fecha_actual - pd.DateOffset(years=1)).replace(day=1) + pd.offsets.MonthEnd(0)
            #print(f"[DEBUG] Fechas: {fecha_anterior.date()} ➡ {fecha}")
            df_actual = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            df_anterior = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha_anterior)]

            actual = df_actual["aportantes"].sum()
            anterior = df_anterior["aportantes"].sum()
            if anterior == 0 or df_anterior.empty:
                print(f"[ERROR] {self.nombre} → {entidad} El DataFrame anterior está vacío {fecha_anterior}")
                return None

            tasa = ((actual - anterior) / anterior) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaCrecimientoPensionados(IndicadorBase):
    nombre = "tasa_crecimiento_pensionados"
    descripcion = "Proyecta cuántas personas se están jubilando cada año."
    formula = "((Pensionados Actuales - Pensionados Previos) / Pensionados Previos) * 100"
    fuentes = ["beneficio"]
    ventana_historica = pd.DateOffset(months=14)
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        #print(f"[DEBUG] {self.nombre} → Iniciando cálculo para {entidad} en {fecha}")
        df = self.datos.get("beneficio")
        if df is None or df.empty:
            #print(f"[ERROR] {self.nombre} → No hay DataFrame para beneficio")
            return None
        try:
            fecha_actual = pd.to_datetime(fecha)
            fecha_anterior = (fecha_actual - pd.DateOffset(years=1)).replace(day=1) + pd.offsets.MonthEnd(0)
            df["fecha"] = pd.to_datetime(df["fecha"])
            df_actual = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            df_anterior = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha_anterior)]
            
            if df_actual.empty:
                #print(f"[ERROR] {self.nombre} → El DataFrame está vacío")
                return None

            if df_anterior.empty:
                print(f"[ERROR] {self.nombre} → El DataFrame anterior está vacío")
                return None

            total_actual = df_actual["beneficio"].sum()
            total_anterior = df_anterior["beneficio"].sum()
            if total_anterior == 0 or pd.isna(total_anterior):
                return None

            tasa = ((total_actual - total_anterior) / total_anterior) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaCaptacionAfiliados(IndicadorBase):
    nombre = "tasa_captacion_afiliados"
    descripcion = "Indica qué tan atractivo es un fondo para nuevos afiliados."
    formula = "Tasa de Captación = (Afiliados Nuevos / Total Aportantes) * 100"
    fuentes = ["afiliado"]
    ventana_historica = pd.DateOffset(months=1)
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("afiliado")
        if df is None or df.empty:
            return None
        try:
            fecha_actual = pd.to_datetime(fecha)
            fecha_anterior = fecha_actual - MonthEnd(1)

            df_actual = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            df_anterior = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha_anterior.strftime('%Y-%m-%d'))]

            afiliados_actual = df_actual["aportantes"].sum()
            afiliados_anteriores = df_anterior["aportantes"].sum()
            nuevos_afiliados = afiliados_actual - afiliados_anteriores
            total_aportantes = df_actual["aportantes"].sum()

            if total_aportantes == 0 :
                return None
            if total_aportantes == 0 or pd.isna(total_aportantes):
                return None

            tasa = (nuevos_afiliados / total_aportantes) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorTasaCaptacionAfiliadosLT(IndicadorBase):
    nombre = "tasa_captacion_afiliados_lt"
    descripcion = "Indica qué tan atractivo es un fondo para nuevos afiliados por libre transferencia."
    formula = "Tasa de Captación = (Afiliados Entrantes / Total Aportantes) * 100"
    fuentes = ["libre_transferencia", "afiliado"]
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_lt = self.datos.get("libre_transferencia")
        if "entidadorigen" in df_lt.columns:
            df_lt = df_lt.rename(columns={"entidadorigen": "entidad"})          
        df_afi = self.datos.get("afiliado")
        if df_lt is None or df_afi is None:
            return None
        try:
            df_afi = df_afi[(df_afi["codigofondo"] == fondo) & (df_afi["fecha"] == fecha)]
            df_lt = df_lt[(df_lt["entidad"] == entidad) & (df_lt["fecha"] == fecha) & (df_lt["codigofondo"] == fondo)]

            total_aportantes = df_afi["aportantes"].sum()
            entrantes = df_lt["cantidad_entrante"].sum()
            if total_aportantes == 0 or pd.isna(total_aportantes):
                return None

            tasa = (entrantes / total_aportantes) * 100
            return round(tasa, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorFlujoNetoAfiliadosFondos(IndicadorBase):
    nombre = "flujo_neto_afiliados_fondos"
    descripcion = "Determina si un fondo está perdiendo o ganando afiliados."
    formula = "Flujo Neto = Afiliados Entrantes - Afiliados Salientes / Total Movimientos"
    fuentes = ["libre_transferencia"]
    activo = True

    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("libre_transferencia")   
        if df is None or df.empty:
            return None
        if "entidadorigen" in df.columns:
            df = df.rename(columns={"entidadorigen": "entidad"})       
        try:
            df_lt = df[(df["entidad"] == entidad) & (df["fecha"] == fecha) & (df["codigofondo"] == fondo)]
            df_lt1 = df[(df["fecha"] == fecha) & (df["codigofondo"] == fondo)]
            salientes = df_lt["cantidad_saliente"].sum()*1.00
            entrantes = df_lt["cantidad_entrante"].sum()*1.00
            total_movimientos = df_lt1["cantidad_entrante"].sum()*1.00
            #print(f"[DEBUG] salientes {salientes} entrantes {entrantes} total {total_movimientos}")
            if pd.isna(total_movimientos) or total_movimientos == 0:
                return None
            flujo_neto = (entrantes - salientes) / total_movimientos
            return flujo_neto
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None
            
class IndicadorFlujoNetoAfiliadosMontos(IndicadorBase):
    nombre = "flujo_neto_afiliados_montos"
    descripcion = "Determina si un fondo está perdiendo o ganando activos."
    formula = "Flujo Neto = Montos Entrantes - Montos Salientes / Total Montos"
    fuentes = ["libre_transferencia"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("libre_transferencia")
        if df is None or df.empty:
            return None
        if "entidadorigen" in df.columns:
            df = df.rename(columns={"entidadorigen": "entidad"})            

        try:
            df_lt = df[(df["entidad"] == entidad) & (df["fecha"] == fecha) & (df["codigofondo"] == fondo)]
            df_lt1 = df[(df["fecha"] == fecha) & (df["codigofondo"] == fondo)]

            salientes = df_lt["monto_saliente"].sum()*1.00
            entrantes = df_lt["monto_entrante"].sum()*1.00
            total_movimientos = df_lt1["monto_entrante"].sum()*1.00
            if pd.isna(total_movimientos) or total_movimientos == 0:
                return None            
            flujo_neto = (entrantes - salientes) / total_movimientos
            return flujo_neto
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None
