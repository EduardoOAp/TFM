from . import IndicadorBase
import pandas as pd

""" 
IndicadorIndiceSostenibilidadPensiones
IndicadorIndiceSostenibilidadFinanciera
IndicadorIndiceEnvejecimiento
IndicadorProporcionPensionadosAfiliados
IndicadorIndiceCargaAdministrativa
"""

class IndicadorIndiceSostenibilidadPensiones(IndicadorBase):
    nombre = "indice_sostenibilidad_pensiones"
    descripcion = "Relaciona la cantidad de aportantes con la cantidad de pensionados."
    formula = "Total de Aportantes / Total de Pensionados"
    fuentes = ["afiliado", "beneficio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_afi = self.datos.get("afiliado")
        df_ben = self.datos.get("beneficio")
        if df_afi is None or df_ben is None:
            return None

        try:
            df_afi = df_afi[(df_afi["entidad"] == entidad) & (df_afi["codigofondo"] == fondo) & (df_afi["fecha"] == fecha)]
            df_ben = df_ben[(df_ben["entidad"] == entidad) & (df_ben["codigofondo"] == fondo) & (df_ben["fecha"] == fecha)]
            total_aportantes = df_afi["aportantes"].sum()
            total_pensionados = df_ben["beneficio"].sum()
            if total_pensionados == 0:
                return None
            return round(total_aportantes / total_pensionados, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorIndiceSostenibilidadFinanciera(IndicadorBase):
    nombre = "indice_sostenibilidad_financiera"
    descripcion = "Evalúa si los aportes cubren los pagos de pensiones."
    formula = "(Total Aportantes * Salario Promedio) / (Total Pensionados * Beneficio Promedio)"
    fuentes = ["afiliado", "beneficio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_afi = self.datos.get("afiliado")
        df_ben = self.datos.get("beneficio")
        if df_afi is None or df_ben is None:
            return None

        try:
            df_afi = df_afi[(df_afi["entidad"] == entidad) & (df_afi["codigofondo"] == fondo) & (df_afi["fecha"] == fecha)]
            df_ben = df_ben[(df_ben["entidad"] == entidad) & (df_ben["codigofondo"] == fondo) & (df_ben["fecha"] == fecha)]
            total_aportantes = df_afi["aportantes"].sum()
            salario_promedio = df_afi["salario_o_aporte_colones"].mean()
            total_pensionados = df_ben["beneficio"].sum()
            beneficio_promedio = df_ben["beneficiocolones"].mean()
            if total_pensionados == 0 or beneficio_promedio == 0:
                return None
            return round((total_aportantes * salario_promedio) / (total_pensionados * beneficio_promedio), 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorProporcionPensionadosAfiliados(IndicadorBase):
    nombre = "proporcion_pensionados_afiliados"
    descripcion = "Proporción de pensionados sobre afiliados."
    formula = "(Total de Pensionados / Total de Aportantes) * 100"
    fuentes = ["afiliado", "beneficio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_afi = self.datos.get("afiliado")
        df_ben = self.datos.get("beneficio")
        if df_afi is None or df_ben is None:
            return None

        try:
            df_afi = df_afi[(df_afi["entidad"] == entidad) & (df_afi["codigofondo"] == fondo) & (df_afi["fecha"] == fecha)]
            df_ben = df_ben[(df_ben["entidad"] == entidad) & (df_ben["codigofondo"] == fondo) & (df_ben["fecha"] == fecha)]
            total_aportantes = df_afi["aportantes"].sum()
            total_pensionados = df_ben["beneficio"].sum()
            if total_aportantes == 0:
                return None
            return round((total_pensionados / total_aportantes) * 100, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorIndiceEnvejecimiento(IndicadorBase):
    nombre = "indice_envejecimiento"
    descripcion = "Porcentaje de aportantes mayores de 59 años."
    formula = "(Aportantes >= 59 años / Total Aportantes) * 100"
    fuentes = ["afiliado"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("afiliado")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            total_aportantes = df["aportantes"].sum()
            mayores_59 = df[df["codigorangoedad"] >= 3]["aportantes"].sum()
            if total_aportantes == 0:
                return None
            indice = (mayores_59 / total_aportantes) * 100
            return round(indice, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None
            
class IndicadorIndiceCargaAdministrativa(IndicadorBase):
    nombre = "indice_carga_administrativa"
    descripcion = "(Comisiones Totales / Activos Netos) * 100"
    formula = "(comisión por saldo del día * activo neto del día) / activo neto del día * 100"
    fuentes = ["comision", "cuenta"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df_com = self.datos.get("comision")
        df_cta = self.datos.get("cuenta")
        #print(list(df_com.columns))
        #print(list(df_cta.columns))
        #print(df_com[['entidad', 'tipo', 'comisión', 'fecha', 'codigofondo']].head(10))
        #print(df_cta[['entidad', 'cuenta', 'montocolones', 'fecha', 'codigofondo']].head(10))
        #print(df_com.dtypes)
        #print(df_cta.dtypes)
        if df_com is None or df_cta is None:
            print(f"[ERROR] {self.nombre} → DataFrame vacio")
            return None
        try:
            df_cta = df_cta[(df_cta["cuenta"].str.contains("ACTIVO NETO")) &
                            (df_cta["entidad"] == entidad) & (df_cta["codigofondo"] == fondo) &
                            (df_cta["fecha"] == fecha)]
            
            df_com = df_com[(df_com["entidad"] == entidad) & (df_com["codigofondo"] == fondo) &
                            (df_com["tipo"].str.lower() == "saldo") & (df_com["fecha"] == fecha)]

            if df_cta.empty:
                print(f"[ERROR] {self.nombre} → DataFrame 'cuenta' vacío")
                return None

            if df_com.empty:
                print(f"[ERROR] {self.nombre} → DataFrame 'comision' vacío")
                return None

            total_activos = df_cta["montocolones"].sum()
            if pd.isna(total_activos) or total_activos == 0:
                return None

            comision_prom = df_com["comisión"].mean()
            if pd.isna(comision_prom):
                return None

            indice = comision_prom  # ya está en porcentaje si se quiere representar así
            return round(float(indice), 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None