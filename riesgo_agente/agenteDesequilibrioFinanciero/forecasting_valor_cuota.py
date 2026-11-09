import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import tempfile
import random
import logging
import itertools
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

from riesgo_agente.utils.config import AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC
from riesgo_agente.utils.config import DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_diarias, unir_serie_y_predicciones
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ForecastingValorCuota:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str, columnas_exogenas: List[str], horizonte: int = 60, ventana_historica: int = 730, max_candidatos_arima: int = 12, max_candidatos_sarima: int = 16):
        self.df = df.copy()        
        self.columna_objetivo = columna_objetivo
        self.columnas_exogenas = columnas_exogenas
        self.modelo_holt = None
        self.modelo_multivariado = None
        self.modelo_lstm = None
        self.horizonte = horizonte
        self.ventana_historica = ventana_historica
        self.modelos_entidad = {}
        self.max_candidatos_arima = max_candidatos_arima
        self.max_candidatos_sarima = max_candidatos_sarima
        self.busquedas_parametros = []
        self.diagnosticos_residuos = []
        self.scaler = None
        self.resultados = {}
        self.agente = AGENTE_DESEQUILIBRIO
        self.modulo = DES_MODULO_FORECAST_VC
        self.resultado = []
        fecha_maxima = pd.to_datetime(df["fecha"], errors="coerce").max()
        if pd.isna(fecha_maxima):
            raise ValueError("No hay fechas válidas en los datos")
        self.fechaTexto = fecha_maxima.strftime("%Y%m%d")
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC, self.fechaTexto)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_DESEQUILIBRIO, DES_MODULO_FORECAST_VC)        
        os.makedirs(self.ruta_resultado, exist_ok=True)
        os.makedirs(self.ruta_modelo, exist_ok=True)
        self.scaler_lstm = None
        self.modelos_lstm_por_entidad = {}

    def preparar_y_diagnosticar(self, df: pd.DataFrame, campo_entidad: str = "entidad") -> pd.DataFrame:
        """Valida datos y genera diagnósticos por entidad sin usar observaciones futuras.

        Los huecos interiores se completan exclusivamente con el último dato conocido
        (ffill). Se informa la cantidad imputada; no se aplica bfill ni interpolación
        bidireccional para evitar anticipar información en el backtesting.
        """
        requeridas = {campo_entidad, "fecha", self.columna_objetivo}
        faltantes = requeridas.difference(df.columns)
        if faltantes:
            raise ValueError(f"Columnas requeridas ausentes: {sorted(faltantes)}")
        base = df.copy()
        base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
        base[self.columna_objetivo] = pd.to_numeric(base[self.columna_objetivo], errors="coerce")
        reportes, preparados = [], []
        for entidad, grupo in base.groupby(campo_entidad, dropna=False):
            n_original = len(grupo)
            fechas_invalidas = int(grupo["fecha"].isna().sum())
            valores_nulos = int(grupo[self.columna_objetivo].isna().sum())
            grupo = grupo.dropna(subset=["fecha"]).sort_values("fecha")
            duplicados = int(grupo.duplicated("fecha").sum())
            grupo = grupo.drop_duplicates("fecha", keep="last")
            if grupo.empty:
                reportes.append({"entidad": entidad, "registros": n_original,
                                 "fechas_invalidas": fechas_invalidas, "valores_nulos": valores_nulos,
                                 "duplicados": duplicados, "error": "Sin fechas válidas"})
                continue
            grupo = grupo.set_index("fecha").asfreq("D")
            faltantes_frecuencia = len(grupo) - (n_original - fechas_invalidas - duplicados)
            grupo[campo_entidad] = entidad
            grupo["valor_imputado"] = grupo[self.columna_objetivo].isna()
            grupo[self.columna_objetivo] = grupo[self.columna_objetivo].ffill()
            grupo = grupo.dropna(subset=[self.columna_objetivo])
            serie = grupo[self.columna_objetivo].astype(float)
            reporte = {"entidad": entidad, "registros": n_original,
                       "fechas_invalidas": fechas_invalidas, "valores_nulos": valores_nulos,
                       "duplicados": duplicados, "fechas_faltantes": max(0, faltantes_frecuencia),
                       "imputados": int(grupo["valor_imputado"].sum()),
                       "observaciones_finales": len(serie), "inicio": serie.index.min(),
                       "fin": serie.index.max(), "error": ""}
            if len(serie) >= 20 and serie.nunique() > 1:
                try:
                    estadistico, pvalor, rezagos, nobs, *_ = adfuller(serie, autolag="AIC")
                    reporte.update(adf_estadistico=float(estadistico), adf_pvalor=float(pvalor),
                                   adf_rezagos=int(rezagos), adf_observaciones=int(nobs),
                                   estacionaria_adf_5pct=bool(pvalor < 0.05))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    reporte["adf_error"] = str(exc)
                try:
                    descomposicion = seasonal_decompose(serie, model="additive", period=7)
                    residuo = descomposicion.resid.dropna()
                    estacional = descomposicion.seasonal.loc[residuo.index]
                    var_res = float(np.var(residuo + estacional))
                    reporte["fuerza_estacional_semanal"] = (max(0.0, 1.0 - float(np.var(residuo)) / var_res)
                                                               if var_res > 0 else 0.0)
                    reporte["varianza_residual_descomposicion"] = float(np.var(residuo))
                except ValueError as exc:
                    reporte["descomposicion_error"] = str(exc)
                try:
                    rezagos = min(14, len(serie) // 2 - 1)
                    reporte["acf_rezago_7"] = float(acf(serie, nlags=rezagos)[7]) if rezagos >= 7 else np.nan
                    reporte["pacf_rezago_1"] = float(pacf(serie, nlags=rezagos, method="ywm")[1])
                except ValueError as exc:
                    reporte["acf_pacf_error"] = str(exc)
            if len(serie) >= 8:
                q1, q3 = serie.quantile([0.25, 0.75])
                iqr = q3 - q1
                reporte["atipicos_iqr"] = int(((serie < q1 - 1.5 * iqr) |
                                                 (serie > q3 + 1.5 * iqr)).sum())
            reportes.append(reporte)
            preparados.append(grupo.reset_index())
        self.diagnosticos = pd.DataFrame(reportes)
        self.diagnosticos.to_csv(os.path.join(self.ruta_resultado, "diagnostico_series.csv"), index=False)
        if not preparados:
            raise ValueError("No hay series válidas después de la preparación")
        return pd.concat(preparados, ignore_index=True)
    
    def entrenar_holt_winters(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df = df.sort_values("fecha").dropna(subset=["fecha"])  # asegura orden y fechas válidas
        # Asignar índice temporal y frecuencia fija (asumiendo datos mensuales)
        df = df.set_index("fecha")
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].asfreq("D")
        df[self.columna_objetivo] = df[self.columna_objetivo].ffill()
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
        #fechas = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=self.horizonte, freq="D")
        fechas = pd.date_range(start=serie.index[-1] + pd.Timedelta(days=1),
                              periods=self.horizonte, freq="D")                

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
        #print('paso 1 mv ')    
        # Validar columnas necesarias
        #print('columnas_requeridas')      
        #print(self.columnas_exogenas)  
        #print(self.columna_objetivo)  
        columnas_requeridas = set(self.columnas_exogenas + [self.columna_objetivo, "fecha"])

        #print(columnas_requeridas)

        columnas_faltantes = columnas_requeridas - set(df.columns)
        #print('columnas_faltantes')
        #print(columnas_faltantes)
        if columnas_faltantes:
            print(f"Faltan columnas: {columnas_faltantes}")
            raise ValueError(f"Faltan columnas: {columnas_faltantes}")
        #print('paso 2 mv ')
        df = df.dropna(subset=self.columnas_exogenas + [self.columna_objetivo])
        df = df.sort_values("fecha")

        # Variables para validación cruzada
        X = df[self.columnas_exogenas].values
        y = df[self.columna_objetivo].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)  # únicamente para el ajuste final; CV usa pipeline

        tscv = TimeSeriesSplit(n_splits=5)
        predicciones_cv, reales_cv = [], []
        #print('paso 3 mv ')
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < 10 or len(test_idx) < 3:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            modelo_cv = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(random_state=42))
            modelo_cv.fit(X_train, y_train)
            y_pred = modelo_cv.predict(X_test)

            predicciones_cv.extend(y_pred)
            reales_cv.extend(y_test)

        # Métricas de validación cruzada
        mse = mean_squared_error(reales_cv, predicciones_cv)
        mae = mean_absolute_error(reales_cv, predicciones_cv)
        rmse = np.sqrt(mse)
        r2 = r2_score(reales_cv, predicciones_cv)

        # ENTRENAMIENTO FINAL PARA PREDICCIÓN FUTURA
        modelo_final = GradientBoostingRegressor(random_state=42)
        modelo_final.fit(X_scaled, y)
        self.modelo_multivariado = modelo_final

        # Predicción futura
        ultimos = df.tail(self.horizonte)
        if len(ultimos) < self.horizonte:
            raise ValueError("No hay suficientes datos futuros para predicción multivariada.")

        # Predicción futura
        #if df_future is None:
        df_future = df.tail(self.horizonte)
        
        if len(df_future) < self.horizonte:
            raise ValueError("No hay suficientes datos futuros para predicción multivariada.")
            
        df_future = df_future.sort_values("fecha")
        X_future = scaler.transform(df_future[self.columnas_exogenas].values)
        y_pred_future = modelo_final.predict(X_future)

        fechas_pred = pd.date_range(
            start=pd.to_datetime(df["fecha"].max()) + pd.Timedelta(days=1),
            periods=self.horizonte, freq="D"
        )            

        resultado = {
            "modelo": "GradientBoostingRegressor",
            "predicciones": y_pred_future.tolist(),
            "inicio": str(fechas_pred[0].date()),
            "fin": str(fechas_pred[-1].date()),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),            
            "r2": round(r2, 4)
        }
        #print('resultado mv ')
        #print(resultado)
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
        import tensorflow as tf
        tf.random.set_seed(42)
        df = df.copy()
        df = df.sort_values("fecha").set_index("fecha").asfreq("D")
        df = df[[self.columna_objetivo]].dropna()
        if len(df) < max(30, self.horizonte + 12):
            self.resultados["lstm"] = {"error": "Datos insuficientes para entrenar LSTM"}
            return self.resultados["lstm"]

        scaler = MinMaxScaler()
        serie_scaled = scaler.fit_transform(df)  # ajuste final; métricas CV deben usar escalador por fold

        ventana = self.horizonte    
        X, y = [], []
        for i in range(ventana, len(serie_scaled)):
            X.append(serie_scaled[i - ventana:i, 0])
            y.append(serie_scaled[i, 0])

        X = np.array(X).reshape(-1, ventana, 1)
        y = np.array(y)

        tscv = TimeSeriesSplit(n_splits=3)
        reales, predicciones = [], []
        fechas_reales = []
        mejor_modelo, mejor_mse = None, float("inf")

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Sequential()
            model.add(Input(shape=(ventana, 1)))
            model.add(LSTM(50, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

            y_pred = model.predict(X_test, verbose=0).flatten()
            predicciones.extend(y_pred)
            reales.extend(y_test)
            fechas_test = df.index[ventana:][test_idx]  # match con X_test
            fechas_reales.extend(fechas_test)

            mse_fold = mean_squared_error(y_test, y_pred)
            if mse_fold < mejor_mse:
                mejor_mse = mse_fold
                mejor_modelo = model
                pass  # el modelo LSTM nunca debe sobrescribir Holt
            """
            pred_teacher_scaled = self.forecast_con_teacher_forcing(
                model=model,
                serie_scaled=serie_scaled,
                ventana=self.horizonte,
                horizonte=self.horizonte,
                reales_future_scaled=y_test  # o cualquier otro tramo real
            )
            pred_teacher = scaler.inverse_transform(pred_teacher_scaled.reshape(-1, 1)).flatten()            
            """
        reales_orig = scaler.inverse_transform(np.array(reales).reshape(-1, 1)).flatten()
        predicciones_orig = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()

        mse = mean_squared_error(reales_orig, predicciones_orig)
        r2 = r2_score(reales_orig, predicciones_orig)
        mae = mean_absolute_error(reales_orig, predicciones_orig)
        rmse = np.sqrt(mse)

        # Entrenamiento final
        X_full = np.array([serie_scaled[i - ventana:i, 0] for i in range(ventana, len(serie_scaled))]).reshape(-1, ventana, 1)
        y_full = np.array([serie_scaled[i, 0] for i in range(ventana, len(serie_scaled))])

        modelo_final = Sequential([
            Input(shape=(ventana, 1)),
            LSTM(50, activation="relu"),
            Dropout(0.2),
            Dense(1)
        ])
        modelo_final.compile(optimizer="adam", loss="mse")
        modelo_final.fit(
            X_full, y_full,
            epochs=100,
            batch_size=8,
            verbose=0,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )  

        # Guardar modelo
        #modelo_final.save("modelo_lstm.keras")
        self.modelo_lstm = modelo_final

        ultimos = serie_scaled[-ventana:].reshape(1, ventana, 1)
        pred_scaled = []
        for _ in range(self.horizonte):
            pred = modelo_final.predict(ultimos, verbose=0)
            pred_scaled.append(pred[0, 0])
            ultimos = np.append(ultimos[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
        fechas_forecast = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=self.horizonte, freq="D")

        self.modelo_lstm = modelo_final
        self.scaler_lstm = scaler
        resultado = {
            "modelo": "LSTM",
            "predicciones": pred.tolist(),
            "inicio": str(fechas_forecast[0].date()),
            "fin": str(fechas_forecast[-1].date()),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),            
            "r2": round(r2, 4)
        }
        self.resultados['lstm'] = resultado
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
        # luego quitar    
        return resultado
        
    @staticmethod
    def calcular_metricas(real, pred):
        real, pred = np.asarray(real, dtype=float), np.asarray(pred, dtype=float)
        mse = mean_squared_error(real, pred)
        validos = real != 0
        return {"mse": float(mse), "mae": float(mean_absolute_error(real, pred)),
                "rmse": float(np.sqrt(mse)),
                "mape": float(np.mean(np.abs((real[validos] - pred[validos]) / real[validos])) * 100) if validos.any() else np.nan,
                "r2": float(r2_score(real, pred)) if len(real) > 1 else np.nan}

    @staticmethod
    def _ajustar_arima_sarima(serie, nombre, orden, estacional=(0, 0, 0, 0)):
        if nombre == "arima":
            return ARIMA(serie, order=orden).fit()
        return SARIMAX(serie, order=orden, seasonal_order=estacional,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=100)

    @staticmethod
    def _diagnostico_residuos(modelo, estacional=0):
        residuos = pd.Series(np.asarray(modelo.resid, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
        # Los primeros residuos pueden estar dominados por inicialización/diferenciación.
        descarte = max(1, int(estacional))
        residuos = residuos.iloc[descarte:]
        if len(residuos) < 12:
            return {"ljung_box_pvalor": np.nan, "residuos_media": np.nan,
                    "residuos_desviacion": np.nan, "residuos_ruido_blanco_5pct": False}
        lag = min(10, len(residuos) // 4)
        prueba = acorr_ljungbox(residuos, lags=[lag], return_df=True)
        pvalor = float(prueba["lb_pvalue"].iloc[-1])
        return {"ljung_box_pvalor": pvalor,
                "residuos_media": float(residuos.mean()),
                "residuos_desviacion": float(residuos.std()),
                "residuos_ruido_blanco_5pct": bool(pvalor >= 0.05)}

    def _buscar_parametros(self, train, validacion, nombre, entidad):
        """Selecciona parámetros con validación temporal interna, sin mirar el test final."""
        if len(train) < 35:
            raise ValueError("Entrenamiento insuficiente para búsqueda de parámetros")
        # ADF guía la diferenciación, pero se consideran ambas opciones para robustez.
        try:
            adf_p = float(adfuller(train, autolag="AIC")[1])
        except (ValueError, np.linalg.LinAlgError):
            adf_p = np.nan
        ds = (0, 1) if np.isfinite(adf_p) and adf_p < 0.05 else (1, 0)
        try:
            fuerza = seasonal_decompose(train, period=7, model="additive")
            resid = fuerza.resid.dropna()
            var = float(np.var((fuerza.seasonal + fuerza.resid).dropna()))
            fuerza_sem = max(0.0, 1.0 - float(np.var(resid)) / var) if var > 0 else 0.0
        except ValueError:
            fuerza_sem = 0.0
        if nombre == "arima":
            combinaciones = [(p, d, q, (0, 0, 0, 0)) for d in ds
                             for p, q in itertools.product((0, 1, 2), repeat=2)]
            limite = self.max_candidatos_arima
        else:
            # m=7 es hipótesis candidata; se incluye alternativa sin estacionalidad.
            estaciones = [(0, 0, 0, 0)]
            if len(train) >= 42:
                estaciones += [(P, D, Q, 7) for D in ((0, 1) if fuerza_sem >= 0.2 else (0,))
                               for P, Q in ((1, 0), (0, 1), (1, 1))]
            combinaciones = [(p, d, q, est) for est in estaciones for d in ds
                             for p, q in ((0, 1), (1, 0), (1, 1), (0, 0))]
            # Priorizar estacionalidad cuando el diagnóstico la sugiere.
            if fuerza_sem >= 0.2:
                combinaciones.sort(key=lambda item: item[3][3] != 7)
            limite = self.max_candidatos_sarima
        if limite < 1:
            raise ValueError("El máximo de candidatos debe ser al menos 1")
        mejor = None
        for p, d, q, est in combinaciones[:limite]:
            fila = {"entidad": entidad, "modelo": nombre, "p": p, "d": d, "q": q,
                    "P": est[0], "D": est[1], "Q": est[2], "m": est[3],
                    "adf_pvalor_entrenamiento": adf_p, "fuerza_estacional_semanal": fuerza_sem}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    modelo = self._ajustar_arima_sarima(train, nombre, (p, d, q), est)
                pred = np.asarray(modelo.forecast(len(validacion)), dtype=float)
                if not np.isfinite(pred).all():
                    raise ValueError("Predicciones no finitas")
                met = self.calcular_metricas(validacion, pred)
                fila.update({"rmse_validacion": met["rmse"], "aic": float(modelo.aic),
                             "bic": float(modelo.bic), "convergio": bool(modelo.mle_retvals.get("converged", True))
                             if hasattr(modelo, "mle_retvals") else True, "error": ""})
                fila.update(self._diagnostico_residuos(modelo, est[3] if est[3] else d))
                if mejor is None or met["rmse"] < mejor[0]:
                    mejor = (met["rmse"], (p, d, q), est, fila.copy())
            except (ValueError, TypeError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                fila["error"] = str(exc)
            self.busquedas_parametros.append(fila)
        if mejor is None:
            raise ValueError(f"No convergió ningún candidato válido de {nombre} para {entidad}")
        return mejor[1], mejor[2], mejor[3]

    def _evaluar_univariado(self, serie, nombre, entidad=""):
        n_test = min(max(30, int(np.ceil(len(serie) * .20))), len(serie) - 30)
        if n_test < 7:
            raise ValueError("Histórico insuficiente para prueba")
        train, test = serie.iloc[:-n_test], serie.iloc[-n_test:]
        parametros = {}
        if nombre in ("arima", "sarima"):
            n_validacion = min(max(14, self.horizonte), max(7, len(train) // 5))
            if len(train) - n_validacion < 35:
                n_validacion = len(train) - 35
            if n_validacion < 7:
                raise ValueError("Histórico insuficiente para validación interna")
            subtrain, validacion = train.iloc[:-n_validacion], train.iloc[-n_validacion:]
            orden, estacional, fila = self._buscar_parametros(subtrain, validacion, nombre, entidad)
            parametros = {"orden": orden, "orden_estacional": estacional,
                          "rmse_validacion": fila["rmse_validacion"]}

        def ajustar(datos):
            if nombre == "naive":
                return None, lambda n: np.repeat(datos.iloc[-1], n)
            if nombre == "promedio":
                return None, lambda n: np.repeat(datos.mean(), n)
            if nombre == "holt_winters":
                m = ExponentialSmoothing(datos, trend="add", seasonal="add" if len(datos) >= 21 else None,
                                         seasonal_periods=7 if len(datos) >= 21 else None).fit()
            elif nombre in ("arima", "sarima"):
                m = self._ajustar_arima_sarima(datos, nombre, orden, estacional)
            else:
                raise ValueError(nombre)
            return m, lambda n: m.forecast(n)

        modelo_test, pronosticar = ajustar(train)
        metricas = self.calcular_metricas(test, pronosticar(len(test)))
        modelo_final, pronosticar_final = ajustar(serie)
        pred = np.asarray(pronosticar_final(self.horizonte), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError("El pronóstico contiene valores no finitos")
        fechas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=self.horizonte, freq="D")
        info = {"modelo": nombre, "predicciones": pred.tolist(), "inicio": str(fechas[0].date()),
                "fin": str(fechas[-1].date()), **metricas, **parametros}
        if modelo_test is not None:
            info["aic"] = float(getattr(modelo_test, "aic", np.nan))
            info["bic"] = float(getattr(modelo_test, "bic", np.nan))
            if nombre in ("arima", "sarima"):
                diagnostico = self._diagnostico_residuos(modelo_test, estacional[3] if estacional[3] else orden[1])
                info.update(diagnostico)
                self.diagnosticos_residuos.append({"entidad": entidad, "modelo": nombre,
                                                   "particion": "prueba", **parametros, **diagnostico})
        return info, modelo_final
    
    @staticmethod
    def _guardar_pickle_atomico(objeto, ruta):
        """Serializa sin truncar el archivo previo; os.replace puede fallar si Windows lo bloquea."""
        directorio = os.path.dirname(os.path.abspath(ruta))
        os.makedirs(directorio, exist_ok=True)
        temporal = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=directorio, prefix="modelo_", suffix=".tmp", delete=False) as f:
                temporal = f.name
                pickle.dump(objeto, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporal, ruta)
        finally:
            if temporal and os.path.exists(temporal):
                os.unlink(temporal)

    def guardar_modelos(self, entidad: str, fondo: str):
        os.makedirs(self.ruta_modelo, exist_ok=True)
        identificador = f"{fondo}_{entidad}".replace("/", "_").replace("\\", "_")
        for nombre, modelo in self.modelos_entidad.items():
            if modelo is None:
                continue
            ruta_base = os.path.join(self.ruta_modelo, f"{identificador}_{nombre}")
            try:
                if nombre == "lstm":
                    # El formato Keras se guarda por separado de su escalador.
                    ruta_final = ruta_base + ".keras"
                    ruta_temporal = ruta_base + ".temporal.keras"
                    try:
                        modelo.save(ruta_temporal)
                        os.replace(ruta_temporal, ruta_final)
                    finally:
                        if os.path.exists(ruta_temporal):
                            os.unlink(ruta_temporal)
                    self._guardar_pickle_atomico(self.scaler_lstm, ruta_base + "_scaler.pkl")
                else:
                    self._guardar_pickle_atomico(modelo, ruta_base + ".pkl")
            except (OSError, pickle.PickleError, ValueError) as exc:
                # Un fallo al guardar un candidato no impide emitir CSV y gráficos.
                logging.exception("No se pudo guardar %s / %s: %s", entidad, nombre, exc)

    def _evaluar_gradient_boosting(self, datos: pd.DataFrame, serie: pd.Series):
        """GB multivariable: rezagos del objetivo, calendario y exógenas rezagadas.

        En evaluación y futuro, las exógenas se mantienen constantes al último
        valor observado al origen del pronóstico (escenario de persistencia).
        No se usan valores reales del período de prueba como predictores.
        """
        h = self.horizonte
        rezagos = (1, 2, 3, 7, 14, 30)
        train = serie.iloc[:-h]
        exogenas = [c for c in self.columnas_exogenas if c in datos.columns and c != self.columna_objetivo]
        historico = datos.sort_values("fecha").drop_duplicates("fecha", keep="last").set_index("fecha")
        historico = historico.reindex(serie.index)
        exo = pd.DataFrame(index=serie.index)
        for col in exogenas:
            valores = pd.to_numeric(historico[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
            if valores.loc[train.index].notna().sum() >= 60:
                exo[col] = valores
        # Variables explicativas conocidas en el momento de predecir t.
        def caracteristicas(hist_y, fecha, exo_ultimo):
            valores = [float(hist_y[-lag]) for lag in rezagos]
            valores += [float(np.mean(hist_y[-7:])), float(np.mean(hist_y[-30:])),
                        float(fecha.dayofweek), float(fecha.month), float(fecha.day)]
            valores += [float(exo_ultimo[col]) for col in exo.columns]
            return valores

        def ajustar(y_train):
            filas, objetivos = [], []
            for i in range(30, len(y_train)):
                # Para entrenar t, la exógena disponible es la de t-1.
                x_anteriores = exo.iloc[i-1].to_dict() if len(exo.columns) else {}
                if any(not np.isfinite(v) for v in x_anteriores.values()):
                    continue
                filas.append(caracteristicas(y_train.iloc[:i].to_numpy(), y_train.index[i], x_anteriores))
                objetivos.append(float(y_train.iloc[i]))
            if len(filas) < 60:
                raise ValueError("Datos insuficientes para Gradient Boosting")
            modelo = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
            modelo.fit(np.asarray(filas), np.asarray(objetivos))
            return modelo

        def pronosticar(modelo, historia, fechas, origen):
            valores = list(map(float, historia.to_numpy()))
            exo_ultimo = exo.loc[origen].to_dict() if len(exo.columns) else {}
            if any(not np.isfinite(v) for v in exo_ultimo.values()):
                raise ValueError("Exógenas no disponibles al origen del pronóstico")
            salida = []
            for fecha in fechas:
                x = np.asarray(caracteristicas(valores, fecha, exo_ultimo), dtype=float).reshape(1, -1)
                pred = float(modelo.predict(x)[0])
                salida.append(pred)
                valores.append(pred)
            return np.asarray(salida)

        modelo_test = ajustar(train)
        pred_test = pronosticar(modelo_test, train, serie.index[-h:], train.index[-1])
        metricas = self.calcular_metricas(serie.iloc[-h:], pred_test)
        modelo_final = ajustar(serie)
        fechas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
        pred = pronosticar(modelo_final, serie, fechas, serie.index[-1])
        info = {"modelo": "GradientBoostingRegressor", "predicciones": pred.tolist(),
                "inicio": str(fechas[0].date()), "fin": str(fechas[-1].date()),
                "escenario_exogenas": "Último valor observado constante", "exogenas_usadas": list(exo.columns),
                **metricas}
        return info, {"modelo": modelo_final, "exogenas": list(exo.columns),
                      "rezagos": rezagos, "escenario": "persistencia"}

    def _evaluar_lstm(self, serie: pd.Series):
        """LSTM univariada con evaluación recursiva de 60 pasos y escalado sin fuga."""
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        h = self.horizonte
        ventana = 30
        train, test = serie.iloc[:-h], serie.iloc[-h:]
        if len(train) < 120:
            raise ValueError("Datos insuficientes para LSTM")

        def ajustar(y_train):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
            X = np.array([scaled[i-ventana:i] for i in range(ventana, len(scaled))]).reshape(-1, ventana, 1)
            y = np.array([scaled[i] for i in range(ventana, len(scaled))])
            tf.keras.backend.clear_session()
            model = Sequential([Input(shape=(ventana, 1)), LSTM(32), Dense(1)])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=25, batch_size=32, verbose=0, shuffle=False,
                      validation_split=0.15,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])
            return model, scaler

        def pronosticar(modelo, scaler, historia):
            scaled = scaler.transform(historia.to_numpy().reshape(-1, 1)).ravel()
            ventana_actual = scaled[-ventana:].reshape(1, ventana, 1)
            pronosticos = []
            for _ in range(h):
                valor = float(modelo.predict(ventana_actual, verbose=0)[0, 0])
                pronosticos.append(valor)
                ventana_actual = np.concatenate((ventana_actual[:, 1:, :],
                                                 np.array([[[valor]]])), axis=1)
            return scaler.inverse_transform(np.asarray(pronosticos).reshape(-1, 1)).ravel()

        modelo_test, scaler_test = ajustar(train)
        pred_test = pronosticar(modelo_test, scaler_test, train)
        metricas = self.calcular_metricas(test, pred_test)
        modelo_final, scaler_final = ajustar(serie)
        pred = pronosticar(modelo_final, scaler_final, serie)
        fechas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
        self.scaler_lstm = scaler_final
        return ({"modelo": "LSTM", "predicciones": pred.tolist(),
                 "inicio": str(fechas[0].date()), "fin": str(fechas[-1].date()), **metricas}, modelo_final)

    def entrenar_por_entidad(self, df: pd.DataFrame, campo_entidad: str = "entidad", fondo: str = "ROP") -> Dict[str, Any]:
        resultados = {}
        self.busquedas_parametros = []
        self.diagnosticos_residuos = []
        filas, seleccion = [], []
        for entidad, datos in df.groupby(campo_entidad):
            datos = datos.sort_values("fecha").tail(self.ventana_historica).copy()
            serie = datos.set_index("fecha")[self.columna_objetivo]
            serie = pd.to_numeric(serie, errors="coerce").dropna()
            serie = serie[~serie.index.duplicated(keep="last")].asfreq("D").dropna()
            resultados[entidad] = {}
            self.modelos_entidad = {}
            if len(serie) < 45:
                resultados[entidad]["error"] = "Histórico insuficiente (mínimo 45 observaciones)"
                continue
            for nombre in ("naive", "promedio", "holt_winters", "arima", "sarima"):
                try:
                    info, modelo = self._evaluar_univariado(serie, nombre, str(entidad))
                    resultados[entidad][nombre] = info
                    self.modelos_entidad[nombre] = modelo
                except Exception as exc:
                    logging.exception("Error %s / %s", entidad, nombre)
                    resultados[entidad][nombre] = {"error": str(exc)}
            for nombre, evaluar in (("multivariado", lambda: self._evaluar_gradient_boosting(datos, serie)),
                                     ("lstm", lambda: self._evaluar_lstm(serie))):
                try:
                    logging.info("Entrenando %s para %s", nombre, entidad)
                    print(f"Entrenando {nombre} para {entidad}...")
                    info, modelo = evaluar()
                    resultados[entidad][nombre] = info
                    self.modelos_entidad[nombre] = modelo
                except Exception as exc:
                    logging.exception("Error %s / %s", entidad, nombre)
                    resultados[entidad][nombre] = {"error": str(exc)}
            candidatos = {k: v for k, v in resultados[entidad].items()
                          if "predicciones" in v and np.isfinite(v.get("rmse", np.nan))}
            if candidatos:
                mejor = min(candidatos, key=lambda k: candidatos[k]["rmse"])
                resultados[entidad]["mejor_modelo"] = mejor
                seleccion.append({"entidad": entidad, "modelo": mejor,
                                  "rmse": candidatos[mejor]["rmse"], "mae": candidatos[mejor]["mae"]})
            self.guardar_modelos(str(entidad), fondo)
            for nombre, info in candidatos.items():
                fechas = pd.date_range(info["inicio"], periods=len(info["predicciones"]), freq="D")
                for fecha, pred in zip(fechas, info["predicciones"]):
                    filas.append({"entidad": entidad, "modelo": nombre, "fecha": fecha,
                                  "prediccion": pred, "mse": info["mse"], "mae": info["mae"],
                                  "rmse": info["rmse"], "mape": info["mape"], "r2": info["r2"],
                                  "seleccionado": nombre == resultados[entidad].get("mejor_modelo")})
        pd.DataFrame(filas, columns=["entidad", "modelo", "fecha", "prediccion", "mse", "mae", "rmse", "mape", "r2", "seleccionado"]).to_csv(
            os.path.join(self.ruta_resultado, "resultados.csv"), index=False)
        pd.DataFrame(seleccion, columns=["entidad", "modelo", "rmse", "mae"]).to_csv(
            os.path.join(self.ruta_resultado, "modelos_seleccionados.csv"), index=False)
        pd.DataFrame(self.busquedas_parametros).to_csv(
            os.path.join(self.ruta_resultado, "parametros_modelos.csv"), index=False)
        pd.DataFrame(self.diagnosticos_residuos).to_csv(
            os.path.join(self.ruta_resultado, "diagnostico_residuos.csv"), index=False)
        return resultados
            
    def ejecutar(self):
        df = self.preparar_y_diagnosticar(self.df)
        ruta_serie = os.path.join(self.ruta_resultado, "serie_datos.csv")
        df.to_csv(ruta_serie, index=False)
        resultados = self.entrenar_por_entidad(df)
        for entidad, modelos in resultados.items():
            print(f"\nEntidad: {entidad}")
            print("Mejor modelo:", modelos.get("mejor_modelo"))

            for nombre, info in modelos.items():
                if not isinstance(info, dict):
                    continue

                if "error" in info:
                    print(f"  {nombre}: ERROR - {info['error']}")

                elif "predicciones" in info:
                    print(
                        f"  {nombre}: "
                        f"{len(info['predicciones'])} predicciones, "
                        f"RMSE={info.get('rmse')}"
                    )        
        self.graficar_forecasts(df, resultados)
        unir_serie_y_predicciones(
            path_serie=ruta_serie,
            path_resultados=os.path.join(self.ruta_resultado, "resultados.csv"),
            columna_objetivo=self.columna_objetivo,
            salida=os.path.join(self.ruta_resultado, "comparacion_historia.csv"))
        self.resultado = ["Se ejecutó forecast de desequilibrio valor cuota"]
        return resultados

    @staticmethod
    def graficar_forecast_lstm(fechas_reales, reales_orig, predicciones_orig, fechas_forecast, predicciones_forecast, entidad="Entidad",
                               predicciones_teacher=None, fechas_teacher=None, mostrar_teacher=True):
        """
        Grafica el comportamiento del modelo LSTM:
        - Comparación entre y_test real vs predicho
        - Forecast futuro usando autoregresión
        """
        #print(f"fechas_reales: {len(fechas_reales)}, reales_orig: {len(reales_orig)}, predicciones_orig: {len(predicciones_orig)}")
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
        plt.close()
        plt.show()        

    def graficar_forecasts(self, df_historico, resultados, guardar_csv=True):
        metricas = []
        fig, ax = plt.subplots(figsize=(14, 8))
        for entidad, modelos in resultados.items():
            historico = df_historico[df_historico["entidad"] == entidad].sort_values("fecha")
            ax.plot(historico["fecha"], historico[self.columna_objetivo], label=f"{entidad} histórico")
            for nombre, info in modelos.items():
                if not isinstance(info, dict) or "predicciones" not in info:
                    continue
                metricas.append({"entidad": entidad, "modelo": nombre,
                                 **{k: info.get(k) for k in ("mse", "mae", "rmse", "mape", "r2", "aic", "bic")}})
                if nombre == modelos.get("mejor_modelo"):
                    fechas = pd.date_range(info["inicio"], periods=len(info["predicciones"]), freq="D")
                    ax.plot(fechas, info["predicciones"], linestyle="--", label=f"{entidad} - {nombre}")
        ax.set(title=f"Forecast de {self.columna_objetivo} por entidad", xlabel="Fecha", ylabel=self.columna_objetivo)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.ruta_resultado, "comparacion_forecast.png"), bbox_inches="tight")
        plt.close(fig)
        if guardar_csv:
            pd.DataFrame(metricas).to_csv(os.path.join(self.ruta_resultado, "metricas_forecast.csv"), index=False)

"""
if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente =  "Agente de Desequilibrio Financiero"
    modulo = "Forecasting Financiero valor cuota"
    fecha = "2026-08-31"
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