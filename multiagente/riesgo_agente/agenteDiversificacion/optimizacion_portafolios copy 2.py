import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from riesgo_agente.utils.config  import DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION
from riesgo_agente.utils.config  import TEMPORAL_ROOT, RESULTADOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_mensuales
# Se necesita un optimizador, se puede usar scipy
from scipy.optimize import minimize
# Ignorar advertencias para una salida más limpia
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class OptimizadorPortafolioLiquido:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str, columnas_predictoras: List[str], horizonte: int = 6):
        self.df = df.copy()
        self.columna_objetivo = columna_objetivo
        self.columnas_predictoras = columnas_predictoras
        self.columnas_instrumentos = [col for col in self.columnas_predictoras if col.startswith("monto_")]
        self.entidades = df['entidad'].unique()        
        self.horizonte = horizonte
        self.modelo = None
        self.modelos_entidades = {} # Nuevo diccionario para almacenar modelos y scalers por entidad
        self.scalers_entidades = {}  # Guarda los scalers por entidad
        self.mejor_modelo_global = None # Renombrado para mayor claridad        
        self.resultados = {}
        self.agente = AGENTE_DIVERSIFICACION
        self.modulo = DIV_MODULO_OPTIMIZACION
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION)
        os.makedirs(self.ruta_resultado, exist_ok=True)    

    
    def preparar_datos(self, df: pd.DataFrame):
        df = df.dropna(subset=self.columnas_predictoras + [self.columna_objetivo])
        df = df.sort_values("fecha")
        X = df[self.columnas_predictoras].values
        y = df[self.columna_objetivo].values
        scaler = StandardScaler()  
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler

    def entrenar_mlp(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            raise ValueError("DataFrame vacío para entrenamiento.")
        if len(df) < 2: # Necesitas al menos dos puntos para escalar o entrenar
             raise ValueError("No hay suficientes datos para entrenar el modelo.")
        
        for entidad in self.entidades:
            df_entidad = df[df['entidad'] == entidad].copy()
            if len(df_entidad.dropna()) > 10:
                try:
                    X, y, X_scaler = self.preparar_datos(df_entidad)
                    param_grid = {
                        'hidden_layer_sizes': [(32,), (64,), (64, 32)],
                        'activation': ['relu', 'tanh'],
                        'solver': ['adam'],
                        'alpha': [0.0001, 0.001],
                        'learning_rate': ['constant', 'adaptive']
                    }

                    tscv = TimeSeriesSplit(n_splits=3)
                    search = RandomizedSearchCV(
                        MLPRegressor(
                            max_iter=15000,
                            early_stopping=True,
                            validation_fraction=0.1,
                            random_state=42
                        ),
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )

                    search.fit(X, y)
                    self.modelo = search.best_estimator_

                    y_pred = self.modelo.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    accuracy_personalizada = self.calcular_accuracy_regresion(y, y_pred)

                    self.modelos_entidades[entidad] = {
                        'modelo': self.modelo,
                        'mse': round(mse, 4),
                        'r2': round(r2, 4),
                        'accuracy_10pct': accuracy_personalizada,
                        'mejores_parametros': search.best_params_   
                        }
                    self.scalers_entidades[entidad] = {
                      "scaler_X": X_scaler
                    }
                    #resultados = {
                    #    'modelo': self.modelo,
                    #    'mse': round(mse, 4),
                    #    'r2': round(r2, 4),
                    #    'accuracy_10pct':accuracy_personalizada ,
                    #    "scaler_X": X_scaler,            
                    #    'mejores_parametros': search.best_params_
                    #}  
                    #resultados_entidades[entidad] = {"mse": resultados["mse"], 
                    #                                 "r2": resultados["r2"],
                    #                                 "accuracy_10pct": resultados["accuracy_10pct"],
                    #                                "modelo": resultados["modelo"],
                    #                                 "scaler_X": resultados["scaler_X"]} # Solo guarda el MSE en resultados
                except Exception as e:
                    print(f"Error al entrenar para '{entidad}': {e}")
            else:
                print(f"No hay suficientes datos para entrenar para '{entidad}'.")

        return #resultados_entidades       

    def _funcion_objetivo_optimizacion(self, pesos, modelo, scaler, escenario_base, columnas_instrumentos):
        """
        Función objetivo para el optimizador. Intenta maximizar el sharpe_ratio
        predicho por el modelo.
        """
        # Crear un escenario futuro con los nuevos pesos
        escenario_modificado = escenario_base.copy()
        
        if 'monto_total_cartera' not in escenario_modificado.columns:
            # Si no existe, crearla sumando las columnas de instrumentos.
            # Asegúrate de que las columnas de instrumentos sean numéricas.
            try:
                monto_total = escenario_modificado[columnas_instrumentos].sum(axis=1).iloc[0]
                #escenario_modificado.loc[:, 'monto_total_cartera'] = monto_total
            except KeyError as e:
                # Manejar el error si alguna columna de instrumento no está presente.
                raise ValueError(f"Error al calcular monto_total_cartera: falta la columna {e}")

        for i, col in enumerate(columnas_instrumentos):
            # Asegurarse de que el peso total no exceda 1 y los pesos sean proporcionales
            # Aquí asumimos que los pesos representan la proporción del monto_total_cartera
            if 'monto_total_cartera' in escenario_modificado.columns:
                 monto_total = escenario_modificado['monto_total_cartera'].iloc[0]
                 escenario_modificado[col] = pesos[i] * monto_total
            else:
                 escenario_modificado[col] = pesos[i] # O manejar de otra manera

        # Escalar el nuevo escenario y predecir
        X_scaled = scaler.transform(escenario_modificado[self.columnas_predictoras].values)
        sharpe_ratio_predicho = modelo.predict(X_scaled)[0]
        
        # Minimizar el valor negativo del sharpe_ratio para maximizarlo
        return -sharpe_ratio_predicho
        
    def predecir_escenarios(self, df_futuro: pd.DataFrame, entidad: str) -> List[float]:
        if entidad not in self.modelos_entidades or "modelo" not in self.modelos_entidades[entidad]:
            raise ValueError(f"No se encontró un modelo entrenado para la entidad '{entidad}'.")
    
        modelo = self.modelos_entidades[entidad]["modelo"]
        scaler_X = self.modelos_entidades[entidad]["scaler_X"]
        #scaler_y = self.modelos_entidades[entidad]["scaler_y"] # Si usaste un scaler para 'y'
        #print('Columnas ',self.columnas_predictoras)
        X_futuro = df_futuro[self.columnas_predictoras].values
        X_futuro_scaled = scaler_X.transform(X_futuro)
        
        predicciones_scaled = modelo.predict(X_futuro_scaled)
        # Invertir la escala de las predicciones si la variable objetivo fue escalada
        #predicciones_original_scale = scaler_y.inverse_transform(predicciones_scaled.reshape(-1, 1)).flatten().tolist()

        return predicciones_scaled

    def _preparar_datos_para_optimizacion(self, df_ent: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        datos_prediccion = {}
        # Obtener los nombres de las columnas que representan tipos de instrumentos
        # Aquí asumo que las columnas predictoras son las mismas para cada instrumento
        #columnas_instrumentos = [col for col in self.columnas_predictoras if col.startswith("monto_")]
        
        # Generar un DataFrame futuro (ej. la última fila) para cada instrumento
        # Puedes adaptar esta lógica para generar escenarios más complejos si lo necesitas
        df_futuro_base = df_ent.tail(1).copy()
        #print('columnas ',columnas_instrumentos)
        for instrumento in self.columnas_instrumentos:
            # Puedes usar el mismo DataFrame base para la predicción,
            # o generar uno específico para cada instrumento
            datos_prediccion[instrumento] = df_futuro_base
        print('Montos ',datos_prediccion)    
        return datos_prediccion
          
    def _optimizar_composicion(self, df: pd.DataFrame):
        #, liquidez_predicha: Dict[str, float]) -> Dict[str, float]

        """
        Optimiza la composición del portafolio para cada entidad usando su modelo entrenado.
        """
        resultados_optimizacion = {}
        
        #print("\nIniciando la optimización de portafolios...")
        for entidad, datosModelo in self.modelos_entidades.items():
        #for entidad, modelo in self.modelos_entidades.items():
            #print(f"--- Optimizando portafolio para la entidad: {entidad} ---")
            
            df_entidad = df[df['entidad'] == entidad].copy()
            modelo = datosModelo['modelo']
            #mse = resultados['mse']
            #r2 = resultados['r2']
            escenario_base = df_entidad.tail(1)
            
            if escenario_base.empty:
                print(f"No hay datos base para la entidad '{entidad}'. Se omite la optimización.")
                continue

            scaler = self.scalers_entidades[entidad]['scaler_X']
            num_instrumentos = len(self.columnas_instrumentos)

            # Restricciones para el optimizador
            restricciones = [
                {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}, # Suma de pesos = 1
                {'type': 'ineq', 'fun': lambda pesos: pesos} # Pesos no negativos (pesos >= 0)
            ]
            
            pesos_iniciales = np.ones(num_instrumentos) / num_instrumentos
            
            # Ejecutar la optimización
            try:
                resultado = minimize(
                    fun=self._funcion_objetivo_optimizacion,
                    x0=pesos_iniciales,
                    args=(modelo, scaler, escenario_base, self.columnas_instrumentos),
                    method='SLSQP',
                    bounds=[(0, 1) for _ in range(num_instrumentos)],
                    constraints=restricciones
                )

                if resultado.success:
                    pesos_optimos = {col: peso for col, peso in zip(self.columnas_instrumentos, resultado.x)}
                    resultados_optimizacion[entidad] = pesos_optimos
                    #print("Optimización exitosa. Composición óptima:")
                    print(pesos_optimos)
                else:
                    resultados_optimizacion[entidad] = {"error": "Optimización fallida."}
                    print("Optimización fallida para la entidad.",entidad)
            except Exception as e:
                print(f"Error durante la optimización para '{entidad}': {e}")
                resultados_optimizacion[entidad] = {"error": str(e)}

        return resultados_optimizacion

    def predecir_ultimos_6_meses(self, entidad: str, columna_objetivo: str) -> pd.DataFrame:
        """
        Predice los últimos 6 meses de datos para una entidad específica
        y devuelve tanto las predicciones como los valores reales.

        Args:
            entidad (str): El nombre de la entidad para la cual se harán las predicciones.
            columna_objetivo (str): El nombre de la columna que contiene los valores reales.

        Returns:
            pd.DataFrame: Un DataFrame con las fechas, los valores reales y las predicciones.
        """
        # 1. Obtener los últimos 6 meses del DataFrame original de la entidad
        df_entidad = self.df[self.df["entidad"] == entidad].copy()
        if df_entidad.empty:
            raise ValueError(f"No se encontraron datos para la entidad '{entidad}'.")
        
        # Asegurarse de que el DataFrame esté ordenado por fecha
        df_entidad.sort_values("fecha", inplace=True)
        
        # Seleccionar las últimas 6 filas (los 6 meses más recientes)
        df_futuro_6_meses = df_entidad.tail(6)
        
        # Extraer los valores reales de la columna objetivo
        valores_reales = df_futuro_6_meses[columna_objetivo].tolist()
        
        # 2. Utilizar la función de predicción existente
        try:
            predicciones = self.predecir_escenarios(df_futuro_6_meses, entidad)
        except ValueError as e:
            print(f"Error al predecir para la entidad '{entidad}': {e}")
            return pd.DataFrame()
            
        # 3. Formatear y devolver los resultados
        # Se crea un nuevo DataFrame con las fechas, los valores reales y las predicciones
        df_resultados = pd.DataFrame({
            "fecha": df_futuro_6_meses["fecha"].reset_index(drop=True),
            "valor_real": valores_reales,
            "prediccion": predicciones
        })
        
        return df_resultados

    def guardar_modelo(self, ruta: str):
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        ruta_completa = os.path.join(self.ruta_modelo,ruta)
        with open(ruta_completa, "wb") as f:
            pickle.dump(self.modelo, f)

    def calcular_accuracy_regresion(self, y_true, y_pred, tolerancia: float = 0.10) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tolerancia_absoluta = tolerancia * np.abs(y_true)
        correctas = np.abs(y_true - y_pred) <= tolerancia_absoluta
        return round(np.mean(correctas), 4)            

    def ejecutar(self):
        df_completo = completar_fechas_mensuales(self.df)
        df_completo = df_completo.sort_values(["entidad", "fecha"])
        df_completo = df_completo.groupby("entidad", group_keys=False).apply(lambda g: g.ffill())
        df_completo = df_completo.reset_index(drop=True)

        #resultados_entidades = {}
        #modelos_entidades_temporales = {}

        self.entrenar_mlp(df_completo)
        resultados = self.modelos_entidades
        
        composicion_optima = self._optimizar_composicion(df_completo)
        #print('resultados ',composicion_optima.items())
        """
        # 2. Entrenar un modelo MLP para cada tipo de activo
        for entidad in df_completo["entidad"].unique():
            df_ent = df_completo[df_completo["entidad"] == entidad].copy()
            try:
                resultado = self.entrenar_mlp(df_ent)
                #resultados_entidades[entidad] = resultado
                resultados_entidades[entidad] = {"mse": resultado["mse"], 
                                                 "r2": resultado["r2"],
                                                 "accuracy_10pct": resultado["accuracy_10pct"],
                                                 "modelo": resultado["modelo"]} # Solo guarda el MSE en resultados
                modelos_entidades_temporales[entidad] = {
                    "modelo": resultado["modelo"],
                    "scaler_X": resultado["scaler_X"]
                }                
            except Exception as e:
                resultados_entidades[entidad] = {"error": str(e)}

        self.resultados = resultados_entidades

        # Identificar y guardar el mejor modelo global
        mejor_entidad_global = None
        mejor_score_global = None
        mejor_r2_global = None
        mejor_resultado_global = None

        for entidad, resultado in resultados_entidades.items():
            if "mse" in resultado and (mejor_score_global is None or resultado["mse"] < mejor_score_global):
                mejor_score_global = resultado["mse"]
                mejor_r2_global = resultado["r2"]
                mejor_entidad_global = entidad
                mejor_resultado_global = resultado

        if mejor_entidad_global:
            self.mejor_modelo_global = (mejor_entidad_global, mejor_resultado_global)
            # Guardar el modelo y scalers del mejor modelo global directamente en la propiedad de la clase
            self.modelos_entidades["mejor_global"] = modelos_entidades_temporales[mejor_entidad_global]
        else:
            self.mejor_modelo_global = ("Ninguno", {"error": "No se encontraron modelos válidos"})
            self.modelos_entidades["mejor_global"] = {"error": "No se encontraron modelos válidos"}

             # Guardar el mejor modelo (mlp y scaler) para cada entidad que se entrenó exitosamente
        for entidad, res in resultados_entidades.items():
            if "mse" in res and entidad in modelos_entidades_temporales:
                self.modelos_entidades[entidad] = modelos_entidades_temporales[entidad]

        # 3. Predecir el valor del indicador de liquidez para el próximo período
        liquidez_predicha = {}
        for entidad in df_completo["entidad"].unique():
            df_ent = df_completo[df_completo["entidad"] == entidad].copy()
            # Agrega esta verificación antes de tomar la última fila
            if not df_ent.empty:
                # Tomar la última fila para hacer la predicción
                df_futuro = df_ent.tail(1)
                try:
                    prediccion = self.predecir_escenarios(df_futuro, entidad)
                    # Se asume que el MLP predice el valor del indicador de liquidez para esa entidad
                    liquidez_predicha[entidad] = prediccion[0]
                except Exception as e:
                    # Captura errores de predicción, como el ValueError, y los maneja
                    print(f"Error al predecir para la entidad '{entidad}': {e}")
                    liquidez_predicha[entidad] = None # O un valor por defecto
            else:
                # Maneja el caso en que el DataFrame de la entidad está vacío
                print(f"No hay datos suficientes para la entidad '{entidad}' para hacer la predicción.")
                liquidez_predicha[entidad] = None # O un valor por defecto            
            # Tomar la última fila para hacer la predicción
            #df_futuro = df_ent.tail(1)
            #prediccion = self.predecir_escenarios(df_futuro, entidad)
            # Se asume que el MLP predice el valor del indicador de liquidez para esa entidad
            #liquidez_predicha[entidad] = prediccion[0]

        print("Predicciones de liquidez por tipo de activo para el próximo período:")
        print(liquidez_predicha)

        #----
        resultados_optimizacion_por_entidad = {}
        for entidad_a_optimizar in df_completo["entidad"].unique():
            print(f"\n--- Optimizando portafolio para la entidad: {entidad_a_optimizar} ---")

            # Paso 3a. Obtener las predicciones para cada tipo de instrumento del portafolio
            df_entidad_opt = df_completo[df_completo["entidad"] == entidad_a_optimizar].copy()
            
            # Verificar si hay datos antes de continuar
            if df_entidad_opt.empty:
                print(f"No hay datos para la entidad '{entidad_a_optimizar}'. Se omite la optimización.")
                continue
            
            # Puedes usar un DF futuro para cada instrumento o uno genérico
            df_futuro_instrumentos = self._preparar_datos_para_optimizacion(df_entidad_opt)

            liquidez_predicha_instrumentos = {}
            for instrumento, df_futuro in df_futuro_instrumentos.items():
                try:
                    prediccion_instrumento = self.predecir_escenarios(df_futuro, entidad_a_optimizar)
                    liquidez_predicha_instrumentos[instrumento] = prediccion_instrumento[0]
                except Exception as e:
                    print(f"Error al predecir para el instrumento '{instrumento}' en '{entidad_a_optimizar}': {e}")
                    liquidez_predicha_instrumentos[instrumento] = None
            
            print("Predicciones de liquidez por instrumento para el próximo período:")
            print(liquidez_predicha_instrumentos)

            # Paso 3b. Filtrar predicciones inválidas
            liquidez_predicha_filtrada = {k: v for k, v in liquidez_predicha_instrumentos.items() if v is not None}
            
            if liquidez_predicha_filtrada:
                # Paso 3c. Optimizar la composición del portafolio
                composicion_optima = self._optimizar_composicion(liquidez_predicha_filtrada)
                resultados_optimizacion_por_entidad[entidad_a_optimizar] = composicion_optima
                print("\nRecomendación de composición de portafolio optimizada:")
                print(composicion_optima)
            else:
                print("\nNo se pueden optimizar los portafolios, no hay predicciones válidas.")
                resultados_optimizacion_por_entidad[entidad_a_optimizar] = {"error": "No hay predicciones válidas"}
       #----

        # 4. Optimizar la composición del portafolio
        composicion_optima = self._optimizar_composicion(liquidez_predicha)       

        print("\nRecomendación de composición de portafolio optimizada:")
        print(composicion_optima)         

        for entidad in df_completo["entidad"].unique():
            # Llama a la nueva función
            try:
                predicciones_comparadas = self.predecir_ultimos_6_meses(entidad, self.columna_objetivo)
                print(f"Comparación de predicciones y valores reales para {entidad}:\n{predicciones_comparadas}")
            except ValueError as e:
                print(f"No se pudo completar la predicción: {e}")
        """
        
        # Dentro del método ejecutar
# ... (después de obtener los resultados_optimizacion) ...
        
        print("\n--- Comparación y Recomendaciones ---")
        for entidad, composicion in composicion_optima.items():
            print(f"\nRecomendaciones para la entidad: {entidad}")
            if 'error' in composicion:
                print(f"  No hay recomendaciones disponibles debido a un error: {composicion['error']}")
                continue # Salta a la siguiente entidad en el bucle principal            
            # Obtener los datos más recientes del portafolio
            df_entidad = df_completo[df_completo['entidad'] == entidad].copy()
            if not df_entidad.empty:
                datos_actuales = df_entidad.tail(1).copy()

                if 'monto_total_cartera' not in datos_actuales.columns:
                    # Si no existe, crearla sumando las columnas de instrumentos.
                    # Asegúrate de que las columnas de instrumentos sean numéricas.
                    try:
                        monto_total = datos_actuales[self.columnas_instrumentos].sum(axis=1).iloc[0]
                        datos_actuales.loc[:, 'monto_total_cartera'] = monto_total
                        #datos_actuales['monto_total_cartera'] = monto_total
                    except KeyError as e:
                        # Manejar el error si alguna columna de instrumento no está presente.
                        raise ValueError(f"Error al calcular monto_total_cartera: falta la columna {e}")                
                monto_total_actual = datos_actuales['monto_total_cartera'].iloc[0]

                for instrumento, peso_optimo in composicion.items():
                    monto_real_actual = datos_actuales[instrumento].iloc[0]
                    monto_optimo_calculado = monto_total_actual * peso_optimo
                    
                    diferencia = monto_optimo_calculado - monto_real_actual
                    
                    if abs(diferencia) > (monto_total_actual * 0.01): # Un umbral del 1% para ser significativo
                        accion = "Aumentar" if diferencia > 0 else "Reducir"
                        print(f"  - Instrumento '{instrumento}': Monto Actual {monto_real_actual:,.2f} óptimo {monto_optimo_calculado:,.2f}. {accion} inversión en {abs(diferencia):,.2f}")
                    else:
                        print(f"  - Instrumento '{instrumento}': Monto Actual {monto_real_actual:,.2f} óptimo {monto_optimo_calculado:,.2f}. Mantener inversión, la diferencia es mínima.")
            else:
                print(f"  No hay datos recientes para comparar.")
        
        return resultados
    
class VisualizadorOptimizacion:
    def __init__(self, resultados: dict, fecha: str, mostrarGrafico: bool = False):
        """
        resultados: diccionario con resultados por entidad
        """
        self.resultados = resultados
        self.df_resultados = self._construir_dataframe()
        self.fechaTexto = fecha.replace("-", "")
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, self.fechaTexto)
        self.mostrarGrafico = mostrarGrafico        

    def _construir_dataframe(self) -> pd.DataFrame:
        filas = []
        for entidad, res in self.resultados.items():
            if "error" not in res:
                filas.append({
                    "entidad": entidad,
                    "mse": res.get("mse", None),
                    "r2": res.get("r2", None),
                    "modelo": str(res.get("modelo", "").__class__.__name__)
                })
        return pd.DataFrame(filas)

    def graficar_metricas(self, guardar=False, ruta="resultados_metricas.png"):
        if self.df_resultados.empty:
            print("No hay resultados para visualizar.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(data=self.df_resultados, x="entidad", y="mse", ax=axes[0])
        axes[0].set_title("Error Cuadrático Medio (MSE)")
        axes[0].tick_params(axis='x', rotation=45)

        sns.barplot(data=self.df_resultados, x="entidad", y="r2", ax=axes[1])
        axes[1].set_title("Coeficiente de Determinación (R²)")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if guardar:
            os.makedirs(os.path.dirname(os.path.join(self.ruta_resultado,ruta)), exist_ok=True)
            plt.savefig(os.path.join(self.ruta_resultado,ruta), bbox_inches='tight')
        if not self.mostrarGrafico:
            plt.close()
        plt.show()

    def mostrar_tabla(self):
        if not self.resultados:
            return "No hay resultados para visualizar."

        registros = []
        for entidad, res in self.resultados.items():
            if "mse" in res:
                registros.append({
                    "entidad": entidad,
                    "mse": res["mse"],
                    "r2": res["r2"],
                    "accuracy_10pct": res.get("accuracy_10pct", None),
                    "modelo": type(res["modelo"]).__name__
                })

        if not registros:
            return "No hay resultados válidos para mostrar."

        df_resultados = pd.DataFrame(registros)
        self.df_resultados = df_resultados
        os.makedirs(os.path.dirname(self.ruta_resultado), exist_ok=True)
        df_resultados.to_csv(os.path.join(self.ruta_resultado, "resultados.csv"), index=False)

        return df_resultados.sort_values("mse")
  
