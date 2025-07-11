import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

## Quitar esta dos lineas despues de pruebas
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
###

from riesgo_agente.utils.config import AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING 
from riesgo_agente.utils.config import DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_mensuales, unir_serie_y_predicciones

#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ForecastingDeficit:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str, columnas_exogenas: List[str], 
                 horizonte: int = 6, mostrarGrafico : bool = False):
        self.df = df.dropna().copy()        
        self.columna_objetivo = columna_objetivo
        self.columnas_exogenas = columnas_exogenas
        self.modelo_holt = None
        self.modelo_multivariado = None
        self.modelo_lstm = None
        self.horizonte = horizonte
        self.scaler = None
        self.resultados = {}
        self.agente = AGENTE_SOSTENIBILIDAD
        self.modulo = SOS_MODULO_FORECASTING
        self.resultado = []
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')        
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_SOSTENIBILIDAD, SOS_MODULO_FORECASTING)
        os.makedirs(self.ruta_resultado, exist_ok=True)
        self.mostrarGrafico = mostrarGrafico        

    def entrenar_holt_winters(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df = df.sort_values("fecha").dropna(subset=["fecha"])  # asegura orden y fechas válidas
        # Asignar índice temporal y frecuencia fija (asumiendo datos mensuales)
        df = df.set_index("fecha")
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq="ME")  # o freq="MS" si es inicio de mes
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("El índice no es DatetimeIndex")

        serie = df[self.columna_objetivo].dropna()
        tscv = TimeSeriesSplit(n_splits=3)
        errores = []

        for train_idx, test_idx in tscv.split(serie):
            train, test = serie.iloc[train_idx], serie.iloc[test_idx]
            if len(train) < 2:
                continue
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                modelo = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
                for warn in w:
                    if issubclass(warn.category, ConvergenceWarning):
                        print(f"[ConvergenceWarning] Fold con fechas {train.index[0]} - {train.index[-1]}")

            #modelo = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
            pred = modelo.forecast(steps=len(test))
            errores.append(mean_squared_error(test, pred))

        modelo_final = ExponentialSmoothing(serie, trend="add", seasonal=None).fit()
        pred_final = modelo_final.forecast(steps=self.horizonte)
        fechas = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=self.horizonte, freq="ME")
        pred_final.index = fechas

        self.modelo_holt = modelo_final
        resultado = {
            "modelo": "Holt-Winters",
            "predicciones": pred_final.tolist(),
            "inicio": str(pred_final.index[0].date()),
            "fin": str(pred_final.index[-1].date()),
            "mse": round(np.mean(errores), 4),
            "splits": 3
        }
        self.resultados['holt_winters'] = resultado
        return resultado

    def entrenar_multivariado(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df = df.dropna(subset=self.columnas_exogenas + [self.columna_objetivo])
        df = df.sort_values("fecha")

        # Variables para validación cruzada
        X = df[self.columnas_exogenas].values
        y = df[self.columna_objetivo].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        predicciones_cv = []
        reales_cv = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            modelo_cv = GradientBoostingRegressor(random_state=42)
            modelo_cv.fit(X_train, y_train)
            y_pred = modelo_cv.predict(X_test)

            predicciones_cv.extend(y_pred)
            reales_cv.extend(y_test)

        mse = mean_squared_error(reales_cv, predicciones_cv)
        r2 = r2_score(reales_cv, predicciones_cv)

        # ENTRENAMIENTO FINAL PARA PREDICCIÓN FUTURA
        modelo_final = GradientBoostingRegressor(random_state=42)
        modelo_final.fit(X_scaled, y)
        self.modelo_multivariado = modelo_final

        # Predicción futura
        ultimos = df.tail(self.horizonte)
        if len(ultimos) < self.horizonte:
            raise ValueError("No hay suficientes datos futuros para predicción multivariada.")

        #X_future = ultimos[self.columnas_exogenas].values
        X_future = scaler.transform(ultimos[self.columnas_exogenas].values)
        y_pred_future = modelo_final.predict(X_future)
        #fechas_future = ultimos["fecha"].values
        fechas_future = pd.date_range(start=df["fecha"].max() + pd.DateOffset(months=1),
                                    periods=self.horizonte, freq="ME")        

        resultado = {
            "modelo": "GradientBoostingRegressor",
            "predicciones": y_pred_future.tolist(),
            "inicio": str(pd.to_datetime(fechas_future[0]).date()),
            "fin": str(pd.to_datetime(fechas_future[-1]).date()),
            "mse": round(mse, 4),
            "r2": round(r2, 4)
        }

        self.resultados['multivariado'] = resultado
        return resultado

    @staticmethod
    def forecast_con_teacher_forcing(model, serie_scaled, ventana: int, horizonte: int, reales_future_scaled: np.ndarray = None):
        """
        Realiza forecast con teacher forcing: usa los valores reales como entrada en cada paso.
        Si reales_future_scaled no se provee, se detiene en la longitud del set real.
        """
        entradas = serie_scaled[-ventana:].reshape(1, ventana, 1)
        predicciones = []

        for i in range(horizonte):
            pred = model.predict(entradas, verbose=0)
            predicciones.append(pred[0, 0])
            
            if reales_future_scaled is not None and i < len(reales_future_scaled):
                nuevo_valor = reales_future_scaled[i]
            else:
                nuevo_valor = pred[0, 0]  # vuelve al modo autoregresivo si no hay más reales
            
            entradas = np.append(entradas[:, 1:, :], np.array([[[nuevo_valor]]]), axis=1)

        return np.array(predicciones)    

    def entrenar_lstm(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df = df.sort_values("fecha").set_index("fecha").asfreq("ME")
        df = df[[self.columna_objetivo]].dropna()
        if len(df) < 30:
            self.resultados["lstm"] = {"error": "Datos insuficientes para entrenar LSTM"}
            return

        scaler = MinMaxScaler()
        serie_scaled = scaler.fit_transform(df)

        ventana = self.horizonte
        X, y = [], []
        for i in range(ventana, len(serie_scaled)):
            X.append(serie_scaled[i - ventana:i, 0])
            y.append(serie_scaled[i, 0])

        X = np.array(X).reshape(-1, ventana, 1)
        y = np.array(y)

        tscv = TimeSeriesSplit(n_splits=3)
        predicciones = []
        reales = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Sequential()
            model.add(Input(shape=(ventana, 1)))
            model.add(LSTM(50, activation="relu", return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])

            y_pred = model.predict(X_test, verbose=0).flatten()
            predicciones.extend(y_pred)
            reales.extend(y_test)

        reales_orig = scaler.inverse_transform(np.array(reales).reshape(-1, 1)).flatten()
        predicciones_orig = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()

        mse = mean_squared_error(reales_orig, predicciones_orig)
        r2 = r2_score(reales_orig, predicciones_orig)   

        ultimos = serie_scaled[-ventana:].reshape(1, ventana, 1)
        pred_scaled = []
        for _ in range(self.horizonte):
            pred = model.predict(ultimos, verbose=0)
            pred_scaled.append(pred[0, 0])
            ultimos = np.append(ultimos[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
        fechas = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=self.horizonte, freq="ME")

        self.modelo_lstm = model
        resultado = {
            "modelo": "LSTM",
            "predicciones": pred.tolist(),
            "inicio": str(fechas[0].date()),
            "fin": str(fechas[-1].date()),
            "mse": round(mse, 4),
            "r2": round(r2, 4)
        }
        self.resultados['lstm'] = resultado
        return resultado
        #temporalmente
        """
        self.graficar_forecast_lstm(
            fechas_reales=fechas_reales,
            reales_orig=reales_orig,
            predicciones_orig=predicciones_orig,
            fechas_forecast=fechas,
            predicciones_forecast=pred,
            fechas_teacher=fechas,
            predicciones_teacher=pred_teacher,
            mostrar_teacher=True
        )  
        """  

    def guardar_modelos(self, entidad: str, fondo: str):
        os.makedirs(self.ruta_modelo, exist_ok=True)
        #with open(f"{ruta_modelo}/{self.fondo}_modelo.pkl", "wb") as f:        
        if self.modelo_holt:
            with open(f"{self.ruta_modelo}/{fondo}_{entidad}_modelo.pkl", "wb") as f:
                pickle.dump(self.modelo_holt, f)
        if self.modelo_multivariado:
            with open(f"{self.ruta_modelo}/{fondo}_{entidad}_modelo.pkl", "wb") as f:
                pickle.dump(self.modelo_multivariado, f)
        if self.modelo_lstm:
            with open(f"{self.ruta_modelo}/{fondo}_{entidad}_modelo.pkl", "wb") as f:
                pickle.dump(self.modelo_lstm, f)

    def entrenar_por_entidad(self, df: pd.DataFrame, campo_entidad: str = "entidad", fondo: str = "ROP") -> Dict[str, Any]:
        resultados_por_entidad = {}
        entidades = df[campo_entidad].unique()

        for entidad in entidades:
            df_ent = df[df[campo_entidad] == entidad].copy()
            if len(df_ent) < 10:
                continue

            try:
                resultado_multi = self.entrenar_multivariado(df_ent)
                resultado_lstm = self.entrenar_lstm(df_ent)
                resultado_holt = self.entrenar_holt_winters(df_ent)                

                self.guardar_modelos(entidad, fondo)

                resultados_por_entidad[entidad] = {
                    "holt_winters": resultado_holt,
                    "multivariado": resultado_multi,
                    "lstm": resultado_lstm
                }
            except Exception as e:
                resultados_por_entidad[entidad] = {"error": str(e)}
        #borrar abajo        
        rows = []
        from datetime import datetime

        for entidad, modelos in resultados_por_entidad.items():
            for modelo_nombre, info in modelos.items():
                print(f"info {info}")
                fecha_inicio = datetime.strptime(info["inicio"], "%Y-%m-%d")
                fechas_fin_mes = pd.date_range(start=fecha_inicio, periods=len(info["predicciones"]), freq="M")
                for i, (pred, fecha) in enumerate(zip(info["predicciones"], fechas_fin_mes)):                
                #for i, pred in enumerate(info["predicciones"]):
                    #fecha = fecha_inicio + pd.Timedelta(days=i)
                    rows.append({
                        "entidad": entidad,
                        "modelo": info["modelo"],
                        "fecha": fecha,
                        "prediccion": pred,
                        "mae": info.get("mae"),                    
                        "r2": info.get("r2")
                    })

        df_resultado = pd.DataFrame(rows)
        ruta = os.path.join(self.ruta_resultado,"resultados.csv")
        os.makedirs(self.ruta_resultado, exist_ok=True)
        df_resultado.to_csv(ruta)
        ### Borra luego arriba
        return resultados_por_entidad
    
    def ejecutar(self):
        df_completo = completar_fechas_mensuales(self.df)    
        # Rellenar hacia adelante
        df_completo = df_completo.sort_values(["entidad", "fecha"])
        df_completo = df_completo.groupby("entidad", group_keys=False).apply(lambda g: g.ffill())
        df_completo = df_completo.reset_index(drop=True) 
        df = df_completo.copy()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")  
        # Verificar fechas faltantes por entidad
        entidades = df["entidad"].unique()

        for ent in entidades:
            fechas_ent = df[df["entidad"] == ent]["fecha"].dropna().sort_values()
            rango_completo = pd.date_range(start=fechas_ent.min(), end=fechas_ent.max(), freq="ME")
            fechas_faltantes = sorted(set(rango_completo) - set(fechas_ent))

            print(f"\n{ent}:")
            print(f"  Rango: {fechas_ent.min().date()} → {fechas_ent.max().date()}")
            print(f"  Fechas únicas registradas: {fechas_ent.nunique()}")
            print(f"  Fechas faltantes ({len(fechas_faltantes)}): {[f.date() for f in fechas_faltantes]}")

        ruta = os.path.join(self.ruta_resultado,"serie_datos.csv")
        df.to_csv(ruta)

        resultados = self.entrenar_por_entidad(df)
        self.graficar_forecasts(df, resultados)
        #print(resultados)
        #self.resultado = self.resultados
        ruta_salida = os.path.join(self.ruta_resultado,"comparacion_historia.csv")
        ruta_serie = ruta
        ruta_resultado = os.path.join(self.ruta_resultado,"resultados.csv")
        unir_serie_y_predicciones(
            path_serie=ruta_serie,
            path_resultados=ruta_resultado,
            columna_objetivo=self.columna_objetivo,
            salida=ruta_salida
        )        
        self.resultado = ["Se ejecutó forecast de liquidez"]
    
    @staticmethod
    def graficar_forecast_lstm(fechas_reales, reales_orig, predicciones_orig, fechas_forecast, predicciones_forecast, entidad="Entidad",
                               predicciones_teacher=None, fechas_teacher=None, mostrar_teacher=True):
        """
        Grafica el comportamiento del modelo LSTM:
        - Comparación entre y_test real vs predicho
        - Forecast futuro usando autoregresión
        """
        print(f"fechas_reales: {len(fechas_reales)}, reales_orig: {len(reales_orig)}, predicciones_orig: {len(predicciones_orig)}")
        plt.figure(figsize=(14, 6))
        
        # Serie de validación
        plt.plot(fechas_reales, reales_orig, label="Valores reales (validación)", color="black", linewidth=2)
        plt.plot(fechas_reales, predicciones_orig, label="Predicción LSTM (validación)", linestyle="--", color="orange")
        if mostrar_teacher and predicciones_teacher is not None:
            plt.plot(fechas_teacher, predicciones_teacher, label="Forecast con Teacher Forcing", color="green", linestyle=":")        
        # Forecast futuro
        plt.plot(fechas_forecast, predicciones_forecast, label="Forecast autoregresivo", color="red")

        plt.title(f"Predicción LSTM y Forecast futuro - {entidad}")
        plt.xlabel("Fecha")
        plt.ylabel("valor_cuota")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if not self.mostrarGrafico:
            plt.close()  
        plt.show()        

    def graficar_forecasts(self, df_historico, resultados, guardar_csv=True):
        import csv
        plt.figure(figsize=(14, 8))
        metricas = []

        entidades = resultados.keys()
        colores = plt.cm.tab10.colors  # Hasta 10 colores

        for i, entidad in enumerate(entidades):
            df_ent = df_historico[df_historico["entidad"] == entidad].copy()
            df_ent["fecha"] = pd.to_datetime(df_ent["fecha"])
            df_ent = df_ent.sort_values("fecha")

            # Color único por entidad
            color = colores[i % len(colores)]

            # Serie histórica
            plt.plot(df_ent["fecha"], df_ent[self.columna_objetivo], label=f"{entidad} - Hist", color=color, linewidth=1.2)

            # Holt-Winters
            hw = resultados[entidad].get("holt_winters")
            if hw:
                fechas_hw = pd.date_range(start=hw["inicio"], periods=len(hw["predicciones"]), freq="ME")
                plt.plot(fechas_hw, hw["predicciones"], linestyle="--", color=color, alpha=0.6, label=f"{entidad} - HW")
                metricas.append([entidad, "Holt-Winters", hw.get("mse", None), None, hw.get("splits", None)])

            # LSTM
            lstm = resultados[entidad].get("lstm")
            if lstm:
                fechas_lstm = pd.date_range(start=lstm["inicio"], periods=len(lstm["predicciones"]), freq="ME")
                plt.plot(fechas_lstm, lstm["predicciones"], linestyle="dotted", color=color, alpha=0.6, label=f"{entidad} - LSTM")
                metricas.append([entidad, "LSTM", lstm.get("mse", None), lstm.get("r2", None), lstm.get("splits", None)])

            # Multivariado
            mv = resultados[entidad].get("multivariado")
            if mv:
                fechas_mv = pd.date_range(start=mv["inicio"], periods=len(mv["predicciones"]), freq="ME")
                metricas.append([entidad, "GradientBoosting", mv.get("mse", None), mv.get("r2", None), mv.get("splits", None)])
                plt.plot(fechas_mv, mv["predicciones"], linestyle="dashdot", color=color, alpha=0.6, label=f"{entidad} - GB")

        plt.title(f"Forecast de {self.columna_objetivo} para todas las entidades")
        plt.xlabel("Fecha")
        plt.ylabel(self.columna_objetivo)
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        ruta = os.path.join(self.ruta_resultado,"comparacion_forecast.png")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)          
        plt.savefig(ruta, bbox_inches="tight")  
        if not self.mostrarGrafico:
            plt.close()                
        plt.show()

        # Guardar métricas
        if guardar_csv:   
            ruta = os.path.join(self.ruta_resultado,"metricas_forecast_deficit.csv")        
            with open(ruta, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["entidad", "modelo", "mse", "r2", "splits"])
                writer.writerows(metricas)
"""
if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente =  "Agente de Desequilibrio Financiero"
    modulo = "Forecasting Financiero valor cuota"
    fecha = "2024-12-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)
    columna_objetivo = 'valor_cuota'

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24*30].copy()
    
    df_completo = completar_fechas_diarias(df)    
    # Rellenar hacia adelante
    df_completo = df_completo.sort_values(["entidad", "fecha"])
    df_completo = df_completo.groupby("entidad", group_keys=False).apply(lambda g: g.ffill())
    df_completo = df_completo.reset_index(drop=True) 
    df = df_completo.copy()
    
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Conteo de fechas por entidad
    conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    print("Conteo de fechas por entidad:\n", conteo_fechas)

    # Verificar fechas faltantes por entidad
    entidades = df["entidad"].unique()

    for ent in entidades:
        fechas_ent = df[df["entidad"] == ent]["fecha"].dropna().sort_values()
        rango_completo = pd.date_range(start=fechas_ent.min(), end=fechas_ent.max(), freq="D")
        fechas_faltantes = sorted(set(rango_completo) - set(fechas_ent))

        print(f"\n{ent}:")
        print(f"  Rango: {fechas_ent.min().date()} → {fechas_ent.max().date()}")
        print(f"  Fechas únicas registradas: {fechas_ent.nunique()}")
        print(f"  Fechas faltantes ({len(fechas_faltantes)}): {[f.date() for f in fechas_faltantes]}")

    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()    
    #df["variacion_activo_neto"] = pd.to_numeric(df["variacion_activo_neto"], errors="coerce")
    #print(f"nulo {df["variacion_activo_neto"].isna().sum()}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)    
    #print(df.head(10))
    #print(df.dtypes)
    # Definir columnas excluyendo entidad, fondo y fecha
    columna_objetivo='valor_cuota'
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha",columna_objetivo]).tolist()

    modelo = ForecastingValorCuota(df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar
    )
    resultados = modelo.ejecutar()
    #modelo.graficar_forecasts(df, resultados, columna_objetivo)
    #print(resultados)

    plt.figure(figsize=(14, 8))
    metricas = []

    #entidades = resultados.keys()
    colores = plt.cm.tab10.colors  # Hasta 10 colores

    for i, entidad in enumerate(entidades):
        df_ent = df[df["entidad"] == entidad].copy()
        df_ent["fecha"] = pd.to_datetime(df_ent["fecha"])
        df_ent = df_ent.sort_values("fecha")

        # Color único por entidad
        color = colores[i % len(colores)]

        # Serie histórica
        plt.plot(df_ent["fecha"], df_ent[columna_objetivo], label=f"{entidad} - Hist", color=color, linewidth=1.2)


    plt.title(f"Forecast de {columna_objetivo} para todas las entidades")
    plt.xlabel("Fecha")
    plt.ylabel(columna_objetivo)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

"""