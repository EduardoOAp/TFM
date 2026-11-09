from . import IndicadorBase
import pandas as pd
import numpy as np
from procesamiento_agente.utils import get_parametro_indicador

""" -- Diversificación
IndicadorConcentracionActivos -
IndicadorDiversificacionYRiesgo - 
IndicadorDiversificacionShannon
IndicadorHerfindahlHirschmanLocal
IndicadorHerfindahlHirschmanExterno
IndicadorDistribucionAfiliadosEntidadFondo
"""

class IndicadorConcentracionActivos(IndicadorBase):
    nombre = "concentracion_activos"
    descripcion = "Porcentaje de inversión en el activo más representativo."
    formula = "(Inversión en un Activo / Total de Inversiones) * 100"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("portafolio")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            #pd.set_option('display.max_columns', None)   
            #print(df.head(20))
            total_inversiones = df["montocolones"].sum()
            if total_inversiones == 0:
                return None
            max_inversion = df.groupby("emisor_gestor")["montocolones"].sum().max()
            #print(f"ent {entidad} fechaa {fecha} maz {max_inversion} tot {total_inversiones}")
            concentracion = (max_inversion / total_inversiones) * 100
            return round(concentracion, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorDiversificacionYRiesgo(IndicadorBase):
    nombre = "diversificacion_y_riesgo"
    descripcion = "Cercano a 1 indica alta diversificación; cercano a 0, alta concentración."
    formula = "1 - (Concentración del activo más grande / Total de inversiones)"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("portafolio")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            total_inversiones = df["montocolones"].sum()
            if total_inversiones == 0:
                return None
            max_inversion = df.groupby("emisor_gestor")["montocolones"].sum().max()
            diversificacion = 1 - (max_inversion / total_inversiones)
            return round(diversificacion, 4)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorDiversificacionShannon(IndicadorBase):
    nombre = "indice_diversificacion_shannon"
    descripcion = "Diversificación usando entropía de Shannon."
    formula = "H = - Σ (p_i * ln(p_i)), donde p_i es la proporción de cada tipo de activo"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("portafolio")
        if df is None or df.empty:
            return None
        try:
            df = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) & (df["fecha"] == fecha)]
            total_inversion = df["montocolones"].sum()
            if total_inversion == 0:
                return None
            
            proporciones = df.groupby("emisor_gestor")["montocolones"].sum() / total_inversion
            shannon = -np.sum(proporciones * np.log(proporciones))

            return round(shannon, 4)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None

class IndicadorHerfindahlHirschmanLocal(IndicadorBase):
    nombre = "herfindahl_hirschman_local"
    descripcion = "Mide el nivel de concentración del portafolio"
    formula = "Suma del cuadrado del porcentaje de participación por sector dentro de cada sector"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad, fondo, fecha):
        df = self.datos["portafolio"]
        if df is None or df.empty:
            return None
        try:   
            df_inv = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                            (df["fecha"] == fecha)]
            if df_inv.empty:
                return None                
            df_nac = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                            (df["sector"].str.lower() != "extranjero") &
                            (df["fecha"] == fecha)]            
            #if df_nac.empty:
            #    return None    
            total = (df_inv["montocolones"]).sum()            
            herfindahl = (df_nac["montocolones"]).sum()
            if total == 0:
                return None    
            return (herfindahl / total) ** 2
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None  

class IndicadorHerfindahlHirschmanExterno(IndicadorBase):
    nombre = "herfindahl_hirschman_externo"
    descripcion = "Mide el nivel de concentración del portafolio"
    formula = "Suma del cuadrado del porcentaje de participación por sector dentro de cada sector"
    fuentes = ["portafolio"]
    activo = True
    
    def calcular(self, entidad, fondo, fecha):
        df = self.datos["portafolio"]
        if df is None or df.empty:
            return None
        try:   
            df_inv = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                            (df["fecha"] == fecha)]
            if df_inv.empty:
                return None                
            df_ext = df[(df["entidad"] == entidad) & (df["codigofondo"] == fondo) &
                            (df["sector"].str.lower() == "extranjero") &
                            (df["fecha"] == fecha)]            
            #if df_ext.empty:
            #    return None    
            total = (df_inv["montocolones"]).sum()            
            herfindahl = (df_ext["montocolones"]).sum()
            if total == 0:
                return None    
            return (herfindahl / total) ** 2
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None  
            
class IndicadorDistribucionAfiliadosEntidadFondo(IndicadorBase):
    nombre = "distribucion_afiliados_entidad_fondo"
    descripcion = "Indica en qué Entidad-Fondo se concentra la mayor cantidad de afiliados."
    formula = "Proporción por EF = (Aportantes en EF X / Total Aportantes) * 100"
    fuentes = ["afiliado"]
    activo = True
    
    def calcular(self, entidad: str, fondo: str, fecha: str):
        df = self.datos.get("afiliado")
        if df is None or df.empty:
            return None

        try:
            df_fecha = df[df["fecha"] == fecha]
            total_aportantes = df_fecha["aportantes"].sum()
            if total_aportantes == 0:
                return None

            df_filtrado = df_fecha[(df_fecha["entidad"] == entidad) & (df_fecha["codigofondo"] == fondo)]
            aportantes_entidad_fondo = df_filtrado["aportantes"].sum()

            proporcion = (aportantes_entidad_fondo / total_aportantes) * 100
            return round(proporcion, 2)
        except Exception as e:
            print(f"[ERROR] {self.nombre}: {e}")
            return None            
        