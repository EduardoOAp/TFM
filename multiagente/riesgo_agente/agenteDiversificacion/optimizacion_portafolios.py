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
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from riesgo_agente.utils.config  import DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION
from riesgo_agente.utils.config  import TEMPORAL_ROOT, RESULTADOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_mensuales
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

class OptimizadorPortafolioLiquido:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str, columnas_predictoras: List[str], horizonte: int = 6):
        self.df = df.copy()
        self.columna_objetivo = columna_objetivo
        self.columnas_predictoras = columnas_predictoras
        self.horizonte = horizonte
        self.modelo = None
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
        return X_scaled, y

    def entrenar_mlp(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, y = self.preparar_datos(df)

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
                max_iter=2000,
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

        resultados = {
            'modelo': self.modelo,
            'mse': round(mse, 4),
            'r2': round(r2, 4),
            'accuracy_10pct':accuracy_personalizada ,
            'mejores_parametros': search.best_params_
        }
        return resultados       
    
    def predecir_escenarios(self, df_futuro: pd.DataFrame) -> List[float]:
        X_futuro = df_futuro[self.columnas_predictoras].values
        X_futuro_scaled = self.scaler.transform(X_futuro)
        return self.modelo.predict(X_futuro_scaled).tolist()

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

        resultados_entidades = {}

        for entidad in df_completo["entidad"].unique():
            df_ent = df_completo[df_completo["entidad"] == entidad].copy()
            try:
                resultado = self.entrenar_mlp(df_ent)
                resultados_entidades[entidad] = resultado
            except Exception as e:
                resultados_entidades[entidad] = {"error": str(e)}

        self.resultados = resultados_entidades

        # Evaluar y guardar el mejor modelo (según menor mse)
        mejor_entidad = None
        mejor_score = None
        mejor_resultado = None

        for entidad, resultado in resultados_entidades.items():
            if "mse" in resultado and (mejor_score is None or resultado["mse"] < mejor_score):
                mejor_score = resultado["mse"]
                mejor_entidad = entidad
                mejor_resultado = resultado

    # Buscar el mejor modelo por MSE (solo si hay alguno válido)
        modelos_validos = {ent: res for ent, res in resultados_entidades.items() if "mse" in res}

        if modelos_validos:
            mejor_entidad = min(modelos_validos, key=lambda e: modelos_validos[e]["mse"])
            self.mejor_modelo = (mejor_entidad, modelos_validos[mejor_entidad])
        else:
            self.mejor_modelo = ("Ninguno", {"error": "No se encontraron modelos válidos"})

        return resultados_entidades
    
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
  
