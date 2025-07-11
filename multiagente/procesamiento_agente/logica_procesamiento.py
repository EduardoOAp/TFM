
import pandas as pd
from .utils import guardar_indicadores, cargar_datos_parquet_rango  # si los tienes separados
from procesamiento_agente.registro_indicadores import REGISTRO_INDICADORES

class AgenteProcesamiento:
    def __init__(self, fondo: str, fecha_inicio: str, fecha_final: str):
        self.fondo = fondo
        self.fecha_inicio = fecha_inicio
        self.fecha_final = fecha_final
        self.fecha_inicio_obj = pd.to_datetime(fecha_inicio)
        self.fecha_final_obj = pd.to_datetime(fecha_final)
        self.clases_indicadores = list(REGISTRO_INDICADORES.values())
        self.clases_indicadores = [config["clase"]
                                   for config in REGISTRO_INDICADORES.values()
                                       if config.get("activo", True)
                                  ]

    def ejecutar(self):
        tipos_datos = set()
        for clase in self.clases_indicadores:
            tipos_datos.update(clase.fuentes)

        fecha_mas_antigua = min(
            self.fecha_inicio_obj - clase.ventana_historica
            for clase in self.clases_indicadores
        )
        
        #print(f"[DEBUG] Rango de carga: {fecha_mas_antigua.date()} ➡ {self.fecha_final_obj.date()}")
         
        # Paso 3: cargar los datos para ese rango dinámico
        datos = {
            tipo: cargar_datos_parquet_rango(tipo, self.fondo, fecha_mas_antigua, self.fecha_final_obj)
            for tipo in tipos_datos
        }
        for clase in self.clases_indicadores:
            tipo_principal = clase.fuentes[0]
            df = datos.get(tipo_principal)

            if df is None or df.empty:
                print(f"[WARNING] El DataFrame para '{tipo_principal}' está vacío. No se podrá calcular el indicador '{clase.__name__}'.")
            else:
                if "fecha" not in df.columns:
                    print(f"[WARNING] El DataFrame para '{tipo_principal}' no contiene la columna 'fecha'.")
                    continue

                fechas_disponibles = pd.to_datetime(df["fecha"], errors="coerce")
                if fechas_disponibles.isnull().all():
                    print(f"[WARNING] El DataFrame para '{tipo_principal}' no contiene fechas válidas.")
                else:
                    fecha_min = fechas_disponibles.min().date()
                    fecha_max = fechas_disponibles.max().date()
                    #print(f"[INFO] {tipo_principal}: fechas disponibles desde {fecha_min} hasta {fecha_max}")
          
              
        resultados = {}
        for clase in self.clases_indicadores:
            print(f"[DEBUG] Ejecutando indicador: {clase.__name__}")
            print(f"[CHECK] Indicador {clase.__name__} requiere fuente: {clase.fuentes[0]}")
            print(f"[CHECK] ¿Fuente disponible?: {clase.fuentes[0] in datos} | Columnas: {list(datos.get(clase.fuentes[0], pd.DataFrame()).columns)}")
            
            frecuencia = clase.frecuencia
            df_fuente = datos.get(clase.fuentes[0], pd.DataFrame())
            if "entidadorigen" in df_fuente.columns:
                df_fuente = df_fuente.rename(columns={"entidadorigen": "entidad"})            
            entidades = (
                [e for e in df_fuente["entidad"].unique() if e != "TOTAL"]
                    if "entidad" in df_fuente.columns else []
                )
            
            fechas_a_procesar = (
                pd.date_range(self.fecha_inicio, self.fecha_final, freq='D')
                if frecuencia == "diaria"
                else [self.fecha_final_obj]
            )  

            for fecha in fechas_a_procesar:
                instancia = clase(self.fondo, self.fecha_inicio, self.fecha_final, datos)
                resultados.clear()     

                for entidad in entidades:
                    #print(f"[TRACE] Indicador: {instancia.nombre} | Entidad: {entidad} | Fecha: {self.fecha_final}")
                    try:
                        #valor = instancia.calcular(entidad, self.fondo, self.fecha_final)
                        valor = instancia.calcular(entidad, self.fondo, fecha)
                    except Exception as e:
                        print(f"[ERROR] Fallo en {instancia.nombre} con entidad {entidad}: {e}")
                        valor = None
                    """
                    if valor is not None:
                        print(f"[OK] Resultado: {valor}")
                    else:
                        print(f"[WARNING] Sin resultado para {instancia.nombre} - {entidad}")
                    """
                    if valor is not None:
                        resultados[(entidad, instancia.nombre)] = valor
                

                guardar_indicadores(self.fondo, fecha.strftime("%Y-%m-%d"), resultados)
