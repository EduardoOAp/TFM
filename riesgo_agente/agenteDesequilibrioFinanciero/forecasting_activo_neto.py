"""Pronóstico mensual de activo neto por entidad.

Conserva la interfaz pública principal de ForecastingActivoNeto y los tres
algoritmos originales. La validación simula pronósticos de horizonte completo
sobre orígenes históricos comunes. No utiliza variables exógenas futuras no
conocidas; el modelo Gradient Boosting utiliza rezagos del activo neto y
variables de calendario conocidas en el futuro.
"""
import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import csv
import itertools
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib.ticker import FuncFormatter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
## Quitar esta dos lineas despues de pruebas
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
###

from riesgo_agente.utils.config import (
    AGENTE_DESEQUILIBRIO, DES_MODULO_FORECASTING,
    DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT,
)

from riesgo_agente.utils.helpers import unir_serie_y_predicciones
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ForecastingActivoNeto:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str,
                 columnas_exogenas: List[str], horizonte: int = 6,
                 divisor_visualizacion: float = 1_000_000.0, meses_grafico: int = 24,
                 max_candidatos_arima: int = 12, max_candidatos_sarima: int = 16):
        if horizonte < 1:
            raise ValueError('El horizonte debe ser positivo.')
        for col in ('fecha', 'entidad', columna_objetivo):
            if col not in df.columns:
                raise ValueError(f'Falta la columna obligatoria: {col}')
        self.df = df.copy()
        self.columna_objetivo = columna_objetivo
        self.columnas_exogenas = list(columnas_exogenas)
        self.horizonte = horizonte
        if not np.isfinite(divisor_visualizacion) or divisor_visualizacion <= 0:
            raise ValueError('divisor_visualizacion debe ser positivo y finito.')
        if meses_grafico < 1:
            raise ValueError('meses_grafico debe ser positivo.')
        self.divisor_visualizacion = float(divisor_visualizacion)
        self.meses_grafico = int(meses_grafico)
        if max_candidatos_arima < 1 or max_candidatos_sarima < 1:
            raise ValueError('Los límites de candidatos deben ser positivos.')
        self.max_candidatos_arima = int(max_candidatos_arima)
        self.max_candidatos_sarima = int(max_candidatos_sarima)
        self.busquedas_parametros = []
        self.diagnosticos_residuos = []
        self._entidad_actual = '' 
        self.modelo_naive = None
        self.modelo_promedio = None
        self.modelo_arima = None
        self.modelo_sarima = None
        self.modelo_holt = None
        self.modelo_multivariado = None
        self.modelo_lstm = None
        self.resultados = {}
        self.resultado = []
        self.agente = AGENTE_DESEQUILIBRIO
        self.modulo = DES_MODULO_FORECASTING
        fechas = pd.to_datetime(df['fecha'], errors='coerce').dropna()
        if fechas.empty:
            raise ValueError('No hay fechas válidas en el conjunto de datos.')
        self.fechaTexto = fechas.max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, self.agente,
                                           self.modulo, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, self.agente, self.modulo)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, self.agente, self.modulo)
        os.makedirs(self.ruta_resultado, exist_ok=True)
        self._modelos_entidad = {}

    def _mensual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Toma el último saldo diario observado de cada mes y entidad.

        Los meses sin observación permanecen como NaN: no se inventan saldos.
        El mes final incompleto se excluye para evitar entrenar sobre un cierre
        que aún no ocurrió (excepto cuando el último dato es fin de mes).
        """
        data = df.copy()
        data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
        data[self.columna_objetivo] = pd.to_numeric(data[self.columna_objetivo], errors='coerce')
        data = data.dropna(subset=['fecha', 'entidad', self.columna_objetivo])
        data = data.sort_values(['entidad', 'fecha'])
        if data.empty:
            raise ValueError('No existen observaciones válidas de activo neto.')
        grupos = []
        for entidad, parte in data.groupby('entidad', sort=False):
            ultimo_dia = parte['fecha'].max().normalize()
            # Solo meses completos para evitar proyectar desde un saldo parcial.
            if not ultimo_dia.is_month_end:
                parte = parte[parte['fecha'].dt.to_period('M') < ultimo_dia.to_period('M')]
            if parte.empty:
                continue
            ultimo = (parte.set_index('fecha').resample('ME').last())
            ultimo['entidad'] = entidad
            ultimo['fecha'] = ultimo.index
            grupos.append(ultimo.reset_index(drop=True))
        if not grupos:
            raise ValueError('No hay meses cerrados para pronosticar.')
        return pd.concat(grupos, ignore_index=True)

    def _serie(self, df: pd.DataFrame) -> pd.Series:
        data = df.copy()
        data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
        data = data.dropna(subset=['fecha', self.columna_objetivo]).sort_values('fecha')
        serie = data.set_index('fecha')[self.columna_objetivo].astype(float)
        serie = serie.groupby(serie.index.to_period('M')).last()
        serie.index = serie.index.to_timestamp(how='end').normalize()
        serie = serie.asfreq('ME')
        if serie.isna().any():
            faltantes = ', '.join(serie.index[serie.isna()].strftime('%Y-%m').tolist())
            raise ValueError(f'Faltan cierres mensuales ({faltantes}); revisar datos antes de entrenar.')
        if len(serie) < max(12, self.horizonte * 2 + 2):
            raise ValueError('Histórico mensual insuficiente para el horizonte solicitado.')
        return serie

    def _origenes(self, n: int):
        # Máximo tres pruebas independientes de horizonte completo, sin solapamiento.
        minimo = max(12, self.horizonte * 2 + 2)
        posibles = list(range(minimo, n - self.horizonte + 1, self.horizonte))
        return posibles[-3:]

    @staticmethod
    def _metricas(reales, predichos):
        a, b = np.asarray(reales, float), np.asarray(predichos, float)
        if not len(a) or len(a) != len(b):
            raise ValueError('No hay suficientes pares reales/predichos.')
        mse = float(mean_squared_error(a, b))
        return {
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(a, b)),
            'r2': float(r2_score(a, b)) if len(a) > 1 else None,
            'wape': float(100 * np.abs(a - b).sum() / np.abs(a).sum()) if np.abs(a).sum() else None,
        }

    @staticmethod
    def _holt_fit(serie):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            return ExponentialSmoothing(serie, trend='add', seasonal=None,
                                        initialization_method='estimated').fit(optimized=True)

    def _validar(self, serie, funcion):
        origenes = self._origenes(len(serie))
        if not origenes:
            raise ValueError('Se necesitan más meses para validación temporal de seis pasos.')
        reales, estimados = [], []
        for origen in origenes:
            pred = np.asarray(funcion(serie.iloc[:origen], self.horizonte), float).reshape(-1)
            if len(pred) != self.horizonte or not np.isfinite(pred).all():
                raise ValueError('Pronóstico de validación incompleto o no finito.')
            reales.extend(serie.iloc[origen:origen + self.horizonte].to_numpy())
            estimados.extend(pred)
        return {**self._metricas(reales, estimados), 'splits': len(origenes)}
    
    def entrenar_holt_winters(self, df: pd.DataFrame) -> Dict:
        serie = self._serie(df)
        metricas = self._validar(serie, lambda s, h: self._holt_fit(s).forecast(h))
        modelo = self._holt_fit(serie)
        self.modelo_holt = modelo
        pred = modelo.forecast(self.horizonte)
        resultado = self._resultado('Holt-Winters', serie, pred, metricas)
        self.resultados['holt_winters'] = resultado
        return resultado

    # Referencias simples y modelos estadísticos, todos validados en los
    # mismos orígenes y con el horizonte completo (6 meses por defecto).
    @staticmethod
    def _naive_predecir(serie, horizonte):
        return np.repeat(float(serie.iloc[-1]), horizonte)

    @staticmethod
    def _promedio_predecir(serie, horizonte):
        return np.repeat(float(serie.mean()), horizonte)

    @staticmethod
    def _ajustar_arima_sarima(serie, nombre, orden, estacional=(0, 0, 0, 0)):
        """Ajuste tomado del módulo de valor cuota; periodicidad mensual."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            if nombre == 'arima':
                return ARIMA(serie, order=orden).fit()
            return SARIMAX(serie, order=orden, seasonal_order=estacional,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False, maxiter=100)

    @staticmethod
    def _diagnostico_residuos(modelo, estacional=0):
        residuos = pd.Series(np.asarray(modelo.resid, dtype=float))
        residuos = residuos.replace([np.inf, -np.inf], np.nan).dropna()
        residuos = residuos.iloc[max(1, int(estacional)):]
        if len(residuos) < 12:
            return {'ljung_box_pvalor': np.nan, 'residuos_media': np.nan,
                    'residuos_desviacion': np.nan, 'residuos_ruido_blanco_5pct': None}
        lag = min(10, max(1, len(residuos) // 4))
        prueba = acorr_ljungbox(residuos, lags=[lag], return_df=True)
        pvalor = float(prueba['lb_pvalue'].iloc[-1])
        return {'ljung_box_pvalor': pvalor,
                'residuos_media': float(residuos.mean()),
                'residuos_desviacion': float(residuos.std()),
                'residuos_ruido_blanco_5pct': bool(pvalor >= 0.05)}

    def _buscar_parametros(self, train, validacion, nombre, entidad='', origen='final'):
        """Busca órdenes usando SOLO subentrenamiento y validación interna.

        Se reutiliza la metodología del valor cuota (ADF, candidatos, AIC/BIC,
        Ljung-Box y RMSE), con estacionalidad mensual m=12.
        """
        if len(train) < 14 or len(validacion) < 1:
            raise ValueError('Histórico insuficiente para búsqueda de parámetros.')
        try:
            adf_p = float(adfuller(train, autolag='AIC')[1])
        except (ValueError, np.linalg.LinAlgError):
            adf_p = np.nan
        ds = (0, 1) if np.isfinite(adf_p) and adf_p < 0.05 else (1, 0)
        if nombre == 'arima':
            candidatos = [(p, d, q, (0, 0, 0, 0)) for d in ds
                          for p, q in itertools.product((0, 1, 2), repeat=2)]
            limite = self.max_candidatos_arima
        else:
            # La estacionalidad anual requiere suficiente histórico.
            estaciones = [(0, 0, 0, 0)]
            if len(train) >= 36:
                estaciones += [(P, D, Q, 12) for D in (0, 1)
                               for P, Q in ((1, 0), (0, 1), (1, 1))]
            candidatos = [(p, d, q, est) for est in estaciones for d in ds
                          for p, q in ((0, 1), (1, 0), (1, 1), (0, 0))]
            # Garantiza probar candidatos estacionales cuando hay datos.
            if len(train) >= 36:
                estacionales = [c for c in candidatos if c[3][3] == 12]
                no_estacionales = [c for c in candidatos if c[3][3] == 0]
                candidatos = [v for pareja in itertools.zip_longest(no_estacionales, estacionales)
                              for v in pareja if v is not None]
            limite = self.max_candidatos_sarima
        mejor = None
        for p, d, q, est in candidatos[:limite]:
            fila = {'entidad': entidad, 'origen': origen, 'modelo': nombre,
                    'p': p, 'd': d, 'q': q, 'P': est[0], 'D': est[1],
                    'Q': est[2], 'm': est[3], 'adf_pvalor_entrenamiento': adf_p,
                    'n_entrenamiento': len(train), 'n_validacion': len(validacion)}
            try:
                ajustado = self._ajustar_arima_sarima(train, nombre, (p, d, q), est)
                pred = np.asarray(ajustado.forecast(len(validacion)), dtype=float)
                if len(pred) != len(validacion) or not np.isfinite(pred).all():
                    raise ValueError('Pronóstico no finito o incompleto')
                convergio = bool(ajustado.mle_retvals.get('converged', True)) if hasattr(ajustado, 'mle_retvals') else True
                mse = float(mean_squared_error(validacion, pred))
                fila.update({'rmse_validacion': float(np.sqrt(mse)),
                             'aic': float(ajustado.aic), 'bic': float(ajustado.bic),
                             'convergio': convergio, 'error': ''})
                fila.update(self._diagnostico_residuos(ajustado, est[3] or d))
                if not convergio or not np.isfinite(mse):
                    fila['error'] = 'No convergió o métrica no finita'
                else:
                    # Ruido blanco es un diagnóstico complementario, no veto rígido.
                    puntaje = (np.sqrt(mse), float(ajustado.aic))
                    if mejor is None or puntaje < mejor[0]:
                        mejor = (puntaje, (p, d, q), est, fila.copy())
            except (ValueError, TypeError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                fila['error'] = str(exc)
            self.busquedas_parametros.append(fila)
        if mejor is None:
            raise ValueError(f'No hubo candidatos convergentes para {nombre} ({entidad})')
        return mejor[1], mejor[2], mejor[3]

    def _pronostico_orden_automatico(self, serie, horizonte, nombre, origen):
        # Separamos validación INTERNA del origen de backtesting (sin fuga).
        n_val = min(self.horizonte, max(2, len(serie) // 5))
        if len(serie) - n_val < 14:
            raise ValueError('No hay histórico suficiente para búsqueda interna.')
        subtrain, validacion = serie.iloc[:-n_val], serie.iloc[-n_val:]
        orden, estacional, seleccion = self._buscar_parametros(
            subtrain, validacion, nombre, self._entidad_actual, origen)
        modelo = self._ajustar_arima_sarima(serie, nombre, orden, estacional)
        if hasattr(modelo, 'mle_retvals') and not modelo.mle_retvals.get('converged', True):
            raise ValueError(f'El ajuste final {nombre} no convergió')
        pred = np.asarray(modelo.forecast(horizonte), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError('Predicciones no finitas')
        return pred, modelo, orden, estacional, seleccion

    def _entrenar_arima_sarima(self, df, nombre):
        serie = self._serie(df)
        self._entidad_actual = str(df['entidad'].iloc[0]) if 'entidad' in df.columns else ''
        origenes = self._origenes(len(serie))
        if not origenes:
            raise ValueError('Histórico insuficiente para validación temporal.')
        reales, estimados = [], []
        for origen in origenes:
            pronostico, _, _, _, _ = self._pronostico_orden_automatico(
                serie.iloc[:origen], self.horizonte, nombre, f'backtest_{origen}')
            reales.extend(serie.iloc[origen:origen + self.horizonte].to_numpy())
            estimados.extend(pronostico)
        metricas = {**self._metricas(reales, estimados), 'splits': len(origenes)}
        pred, modelo, orden, estacional, seleccion = self._pronostico_orden_automatico(
            serie, self.horizonte, nombre, 'final')
        if nombre == 'arima':
            self.modelo_arima = modelo
        else:
            self.modelo_sarima = modelo
        resultado = self._resultado(nombre.upper(), serie, pred, metricas)
        resultado.update({'orden': orden, 'orden_estacional': estacional,
                          'aic': float(modelo.aic), 'bic': float(modelo.bic),
                          'rmse_validacion_parametros': seleccion['rmse_validacion']})
        diagnostico = self._diagnostico_residuos(modelo, estacional[3] or orden[1])
        self.diagnosticos_residuos.append({
            'entidad': self._entidad_actual, 'modelo': nombre,
            'p': orden[0], 'd': orden[1], 'q': orden[2],
            'P': estacional[0], 'D': estacional[1], 'Q': estacional[2], 'm': estacional[3],
            'aic': float(modelo.aic), 'bic': float(modelo.bic), **diagnostico})
        self.resultados[nombre] = resultado
        return resultado

    def entrenar_naive(self, df: pd.DataFrame) -> Dict:
        serie = self._serie(df)
        metricas = self._validar(serie, self._naive_predecir)
        self.modelo_naive = {'ultimo_valor': float(serie.iloc[-1])}
        resultado = self._resultado('Naive', serie,
                                    self._naive_predecir(serie, self.horizonte), metricas)
        self.resultados['naive'] = resultado
        return resultado

    def entrenar_promedio(self, df: pd.DataFrame) -> Dict:
        serie = self._serie(df)
        metricas = self._validar(serie, self._promedio_predecir)
        self.modelo_promedio = {'media_historica': float(serie.mean())}
        resultado = self._resultado('Promedio Histórico', serie,
                                    self._promedio_predecir(serie, self.horizonte), metricas)
        self.resultados['promedio'] = resultado
        return resultado

    def entrenar_arima(self, df: pd.DataFrame) -> Dict:
        return self._entrenar_arima_sarima(df, 'arima')

    def entrenar_sarima(self, df: pd.DataFrame) -> Dict:
        return self._entrenar_arima_sarima(df, 'sarima')
            
    @staticmethod
    def _features(serie, lags):
        df = pd.DataFrame({'objetivo': serie})
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = serie.shift(i)
        df['mes_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        return df.dropna()

    def _gb_entrenar(self, serie):
        lags = min(6, max(2, len(serie) // 4))
        matriz = self._features(serie, lags)
        if len(matriz) < 8:
            raise ValueError('Datos insuficientes para Gradient Boosting.')
        X = matriz.drop(columns='objetivo')
        y = matriz['objetivo']
        modelo = GradientBoostingRegressor(random_state=42, n_estimators=100,
                                           max_depth=2, learning_rate=0.05)
        modelo.fit(X, y)
        return modelo, lags

    def _gb_predecir(self, serie, horizonte, modelo=None, lags=None):
        if modelo is None:
            modelo, lags = self._gb_entrenar(serie)
        valores = list(serie.to_numpy(dtype=float))
        salida = []
        for fecha in pd.date_range(serie.index[-1] + pd.offsets.MonthEnd(1),
                                   periods=horizonte, freq='ME'):
            features = {f'lag_{i}': valores[-i] for i in range(1, lags + 1)}
            features['mes_sin'] = np.sin(2 * np.pi * fecha.month / 12)
            features['mes_cos'] = np.cos(2 * np.pi * fecha.month / 12)
            X = pd.DataFrame([features])[modelo.feature_names_in_]
            estimado = float(modelo.predict(X)[0])
            salida.append(estimado)
            valores.append(estimado)
        return salida
    
    def entrenar_multivariado(self, df: pd.DataFrame) -> Dict:
        serie = self._serie(df)
        # Sin escenarios exógenos futuros, no es válido utilizar valores reales
        # de variables explicativas de los últimos meses como meses futuros.
        metricas = self._validar(serie, self._gb_predecir)
        modelo, lags = self._gb_entrenar(serie)
        self.modelo_multivariado = {'modelo': modelo, 'lags': lags,
                                   'columnas': list(modelo.feature_names_in_)}
        pred = self._gb_predecir(serie, self.horizonte, modelo, lags)
        resultado = self._resultado('GradientBoostingRegressor', serie, pred, metricas)
        resultado['nota'] = ('Utiliza rezagos del activo neto y calendario; '
                             'no utiliza exógenas futuras desconocidas.')
        self.resultados['multivariado'] = resultado
        return resultado

    @staticmethod
    def _lstm_dependencias():
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        return Sequential, LSTM, Dense, Input, EarlyStopping, MinMaxScaler, tf

    def _lstm_entrenar(self, serie):
        Sequential, LSTM, Dense, Input, EarlyStopping, MinMaxScaler, tf = self._lstm_dependencias()
        tf.keras.utils.set_random_seed(42)
        ventana = min(12, max(3, self.horizonte), len(serie) // 3)
        scaler = MinMaxScaler()
        valores = scaler.fit_transform(serie.to_numpy().reshape(-1, 1)).ravel()
        X, y = [], []
        for i in range(ventana, len(valores)):
            X.append(valores[i - ventana:i])
            y.append(valores[i])
        X = np.asarray(X).reshape(-1, ventana, 1)
        y = np.asarray(y)
        if len(X) < 12:
            raise ValueError('Datos insuficientes para entrenar LSTM.')
        corte = max(1, int(len(X) * 0.8))
        if corte >= len(X):
            raise ValueError('No hay datos suficientes para validación interna LSTM.')
        modelo = Sequential([Input(shape=(ventana, 1)), LSTM(24), Dense(1)])
        modelo.compile(optimizer='adam', loss='mse')
        modelo.fit(X[:corte], y[:corte], validation_data=(X[corte:], y[corte:]),
                   epochs=100, batch_size=8, shuffle=False, verbose=0,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                            restore_best_weights=True)])
        # Ajuste final con todos los datos del origen de entrenamiento.
        epocas = max(10, min(100, len(modelo.history.history.get('loss', []))))
        final = Sequential([Input(shape=(ventana, 1)), LSTM(24), Dense(1)])
        final.compile(optimizer='adam', loss='mse')
        final.fit(X, y, epochs=epocas, batch_size=8, shuffle=False, verbose=0)
        return final, scaler, ventana

    def _lstm_predecir(self, serie, horizonte, entrenado=None):
        modelo, scaler, ventana = entrenado or self._lstm_entrenar(serie)
        valores = scaler.transform(serie.to_numpy().reshape(-1, 1)).ravel().tolist()
        salida = []
        for _ in range(horizonte):
            X = np.asarray(valores[-ventana:]).reshape(1, ventana, 1)
            siguiente = float(modelo.predict(X, verbose=0)[0, 0])
            valores.append(siguiente)
            salida.append(siguiente)
        return scaler.inverse_transform(np.asarray(salida).reshape(-1, 1)).ravel().tolist()
    
    def entrenar_lstm(self, df: pd.DataFrame) -> Dict:
        serie = self._serie(df)
        metricas = self._validar(serie, self._lstm_predecir)
        entrenado = self._lstm_entrenar(serie)
        self.modelo_lstm = entrenado
        pred = self._lstm_predecir(serie, self.horizonte, entrenado)
        resultado = self._resultado('LSTM', serie, pred, metricas)
        self.resultados['lstm'] = resultado
        return resultado

    def _resultado(self, nombre, serie, pred, metricas):
        fechas = pd.date_range(serie.index[-1] + pd.offsets.MonthEnd(1),
                               periods=self.horizonte, freq='ME')
        return {'modelo': nombre, 'predicciones': [float(v) for v in pred],
                'inicio': str(fechas[0].date()), 'fin': str(fechas[-1].date()),
                **metricas}
    
    def guardar_modelos(self, ruta_base: str, entidad: str, fondo: str):
        ruta = Path(ruta_base) / self.agente / self.modulo
        ruta.mkdir(parents=True, exist_ok=True)
        etiqueta = f'{fondo}_{entidad}'.replace('/', '_').replace('\\', '_')
        modelos = {'naive': self.modelo_naive,
                   'promedio': self.modelo_promedio,
                   'holt_winters': self.modelo_holt,
                   'arima': self.modelo_arima,
                   'sarima': self.modelo_sarima,
                   'multivariado': self.modelo_multivariado}
        for nombre, modelo in modelos.items():
            if modelo is not None:
                with open(ruta / f'{etiqueta}_{nombre}.pkl', 'wb') as archivo:
                    pickle.dump(modelo, archivo)
        if self.modelo_lstm is not None:
            modelo, scaler, ventana = self.modelo_lstm
            ruta_lstm = ruta / f'{etiqueta}_lstm.keras'
            modelo.save(str(ruta_lstm))
            with open(ruta / f'{etiqueta}_lstm_scaler.pkl', 'wb') as archivo:
                pickle.dump({'scaler': scaler, 'ventana': ventana}, archivo)

    def entrenar_por_entidad(self, df: pd.DataFrame, campo_entidad: str = 'entidad',
                             fondo: str = 'ROP', ruta_modelos: str = None) -> Dict[str, Any]:
        ruta_modelos = ruta_modelos or os.path.join(DATA_ROOT, MODELOS_ROOT)
        resultados = {}
        filas = []
        resumen = []
        for entidad, df_ent in df.groupby(campo_entidad, sort=False):
            self.modelo_holt = self.modelo_multivariado = self.modelo_lstm = None
            self.modelo_naive = self.modelo_promedio = None
            self.modelo_arima = self.modelo_sarima = None
            self.resultados = {}
            predicciones = {}
            errores = {}
            for clave, metodo in [('naive', self.entrenar_naive),
                                  ('promedio', self.entrenar_promedio),
                                  ('holt_winters', self.entrenar_holt_winters),
                                  ('arima', self.entrenar_arima),
                                  ('sarima', self.entrenar_sarima),
                                  ('multivariado', self.entrenar_multivariado),
                                  ('lstm', self.entrenar_lstm)]:
                try:
                    predicciones[clave] = metodo(df_ent)
                except Exception as exc:
                    errores[clave] = str(exc)
                    print(f'[{entidad}] {clave}: {exc}')
            if not predicciones:
                resultados[entidad] = {'error': 'Ningún modelo pudo entrenarse',
                                       'errores_modelos': errores}
                continue
            mejor = min(predicciones, key=lambda m: predicciones[m]['rmse'])
            resultados[entidad] = {**predicciones, 'mejor_modelo': mejor,
                                   'errores_modelos': errores}
            self._modelos_entidad[entidad] = resultados[entidad]
            try:
                self.guardar_modelos(ruta_modelos, str(entidad), fondo)
            except Exception as exc:
                resultados[entidad]['error_guardado'] = str(exc)
            for clave, info in predicciones.items():
                resumen.append({'entidad': entidad, 'modelo': clave,
                                'mejor_modelo': clave == mejor,
                                **{k: info.get(k) for k in ('mse', 'rmse', 'mae', 'r2', 'wape', 'splits')}})
                fechas = pd.date_range(start=info['inicio'], periods=len(info['predicciones']), freq='ME')
                for fecha, valor in zip(fechas, info['predicciones']):
                    filas.append({'entidad': entidad, 'modelo': clave, 'fecha': fecha,
                                  'prediccion': valor, 'seleccionado': clave == mejor,
                                  **{k: info.get(k) for k in ('mse', 'rmse', 'mae', 'r2', 'wape')}})
        pd.DataFrame(filas, columns=['entidad', 'modelo', 'fecha', 'prediccion',
                                     'seleccionado', 'mse', 'rmse', 'mae', 'r2', 'wape']).to_csv(
            os.path.join(self.ruta_resultado, 'resultados.csv'), index=False)
        pd.DataFrame(resumen, columns=['entidad', 'modelo', 'mejor_modelo',
                                       'mse', 'rmse', 'mae', 'r2', 'wape', 'splits']).to_csv(
            os.path.join(self.ruta_resultado, 'metricas_forecast.csv'), index=False)
        pd.DataFrame(self.busquedas_parametros).to_csv(
            os.path.join(self.ruta_resultado, 'parametros_modelos.csv'), index=False)
        pd.DataFrame(self.diagnosticos_residuos).to_csv(
            os.path.join(self.ruta_resultado, 'diagnostico_residuos.csv'), index=False)
        self.resultados_por_entidad = resultados
        return resultados
    
    def graficar_forecasts(self, df_historico, resultados, guardar_csv=True):
        """Gráficos por OPC, histórico reciente y siete pronósticos.

        El divisor afecta solamente el eje vertical: no altera los CSV,
        predicciones ni las métricas calculadas en unidades originales.
        """
        escala = (self.divisor_visualizacion
                  if 'monto' in self.columna_objetivo.lower() else 1.0)
        etiqueta_escala = (f' / {escala:,.0f}' if escala != 1 else '')
        for entidad, info in resultados.items():
            if 'error' in info:
                continue
            hist = df_historico[df_historico['entidad'] == entidad].copy()
            hist['fecha'] = pd.to_datetime(hist['fecha'])
            hist = hist.sort_values('fecha').tail(self.meses_grafico)
            if hist.empty:
                continue
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.plot(hist['fecha'], hist[self.columna_objetivo].astype(float) / escala,
                    label='Histórico (últimos meses)', color='#172554', linewidth=2.5)
            colores = {'naive': '#64748b', 'promedio': '#a16207',
                       'holt_winters': '#16a34a', 'arima': '#2563eb',
                       'sarima': '#7c3aed', 'multivariado': '#ea580c',
                       'lstm': '#dc2626'}
            for clave in ('naive', 'promedio', 'holt_winters', 'arima',
                          'sarima', 'multivariado', 'lstm'):
                datos = info.get(clave)
                if not isinstance(datos, dict) or 'predicciones' not in datos:
                    continue
                fechas = pd.date_range(datos['inicio'],
                                       periods=len(datos['predicciones']), freq='ME')
                ganador = clave == info.get('mejor_modelo')
                etiqueta = f"{datos['modelo']} (RMSE={datos['rmse']:,.2f})"
                if ganador:
                    etiqueta = '★ ' + etiqueta
                ax.plot(fechas, np.asarray(datos['predicciones'], float) / escala,
                        marker='o', markersize=4, linestyle='--',
                        color=colores[clave], linewidth=2.7 if ganador else 1.5,
                        alpha=1 if ganador else .78, label=etiqueta)
            ax.axvline(hist['fecha'].max(), color='gray', linestyle=':', alpha=.8)
            ax.set_title(f'{self.columna_objetivo} — {entidad} | Mejor: {info["mejor_modelo"]}')
            ax.set_xlabel('Fecha')
            ax.set_ylabel(f'{self.columna_objetivo}{etiqueta_escala}')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:,.2f}'))
            ax.grid(alpha=.22)
            ax.legend(loc='best', fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            nombre = str(entidad).replace('/', '_').replace('\\', '_')
            fig.savefig(os.path.join(self.ruta_resultado, f'forecast_{nombre}.png'),
                        dpi=160, bbox_inches='tight')
            plt.close(fig)

    def ejecutar(self):
        mensual = self._mensual(self.df)
        mensual = mensual.sort_values(['entidad', 'fecha']).reset_index(drop=True)
        ruta_serie = os.path.join(self.ruta_resultado, 'serie_datos.csv')
        mensual.to_csv(ruta_serie, index=False)
        resultados = self.entrenar_por_entidad(mensual)
        self.graficar_forecasts(mensual, resultados)
        ruta_resultados = os.path.join(self.ruta_resultado, 'resultados.csv')
        if os.path.getsize(ruta_resultados) > 0 and pd.read_csv(ruta_resultados).shape[0] > 0:
            try:
                unir_serie_y_predicciones(
                    path_serie=ruta_serie, path_resultados=ruta_resultados,
                    columna_objetivo=self.columna_objetivo,
                    salida=os.path.join(self.ruta_resultado, 'comparacion_historia.csv'))
            except Exception as exc:
                warnings.warn(f'No se pudo generar comparacion_historia.csv: {exc}')
        self.resultado = ['Se ejecutó forecast mensual de activo neto']
        return resultados


"""
if __name__ == "__main__":
    from pathlib import Path
    import traceback    
    DATA_ROOT = "data"     
    agente = "Agente de Desequilibrio Financiero"
    modulo = "Forecasting Financiero activo neto"
    fecha = "2026-08-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()

    # Rellenar hacia adelante
    df_completo = df.sort_values(["entidad", "fecha"])
    columnas_rellenar = df_completo.columns.difference(["entidad"])

    df_completo[columnas_rellenar] = (
    df_completo.groupby("entidad")[columnas_rellenar].ffill()
    )
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
        rango_completo = pd.date_range(start=fechas_ent.min(), end=fechas_ent.max(), freq="ME")
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
    columna_objetivo="variacion_activo_neto"
    columnas_usar = df.columns.difference(["entidad", "fondo", "fecha","variacion_activo_neto"]).tolist()

    modelo = ForecastingActivoNeto(
        df=df,
        columna_objetivo=columna_objetivo,
        columnas_exogenas=columnas_usar,
        horizonte=6,
    )
    
    # Ejecutar el proceso completo
    try:
        resultados = modelo.ejecutar()

        ruta = Path(modelo.ruta_resultado).resolve()

        print("\n" + "=" * 70)
        print("DIAGNÓSTICO DE GRÁFICOS")
        print("=" * 70)

        print(f"Ruta de resultados: {ruta}")
        print(f"Directorio existe: {ruta.exists()}")

        graficos = list(ruta.glob("forecast_*.png"))

        print(f"Gráficos encontrados: {len(graficos)}")

        for grafico in graficos:
            print(f"  {grafico.name}")

        print("\nResultados por entidad:")

        for entidad, info in resultados.items():
            if "error" in info:
                print(f"  {entidad}: ERROR - {info['error']}")
            else:
                print(
                    f"  {entidad}: "
                    f"Mejor modelo = {info.get('mejor_modelo')}"
                )

        if not graficos:
            print("\nIntentando generar nuevamente los gráficos...")

            df_mensual = modelo._mensual(modelo.df)

            modelo.graficar_forecasts(
                df_mensual,
                resultados
            )

            nuevos = list(ruta.glob("forecast_*.png"))
            print(f"Gráficos después del reintento: {len(nuevos)}")

    except Exception:
        print("\nERROR DURANTE LA EJECUCIÓN:")
        traceback.print_exc()

    # Mostrar resultados
    print("\nResultados del forecasting:")
    print(modelo.resultados)

"""