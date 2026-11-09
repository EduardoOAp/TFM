"""Detección temporal de anomalías de liquidez por fondo y entidad.

Los umbrales se calibran con muestras distintas de las evaluadas. Los modelos
Keras se guardan en .keras, mientras escaladores y metadatos se guardan en .pkl.
Requiere TensorFlow para entrenar o inferir; `--help` no lo requiere.
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import pickle
import re
import traceback
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from riesgo_agente.utils.config import (AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS,
        DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT)
except ImportError:
    AGENTE_LIQUIDEZ = 'Agente de Riesgo de Liquidez'
    LIQ_MODULO_ANOMALIAS = 'Detección de anomalías de liquidez'
    DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT = 'data', 'temporal', 'resultados', 'modelos'


def _seguro(valor):
    return re.sub(r'[^\w.\-]+', '_', str(valor), flags=re.UNICODE).strip('_') or 'sin_nombre'


def _keras():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('Falta TensorFlow. Instálelo en su entorno: pip install tensorflow') from exc
    return tf


def _ventanas(matriz, fechas, longitud, tipo):
    """Ventanas sin cruce entre entidades; etiqueta para autoencoder = fin de ventana."""
    X, y, f = [], [], []
    if tipo == 'autoencoder':
        for fin in range(longitud, len(matriz) + 1):
            X.append(matriz[fin-longitud:fin]); y.append(matriz[fin-longitud:fin]); f.append(fechas[fin-1])
    else:
        for fin in range(longitud, len(matriz)):
            X.append(matriz[fin-longitud:fin]); y.append(matriz[fin]); f.append(fechas[fin])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.Series(f).reset_index(drop=True)


class LSTMAnomalyDetector:
    """Entrena detectores separados para las entidades de un fondo.

    Parámetros anteriores permanecen; `fondo`, `columnas`, `entidad`, `graficos`
    y `min_registros` son opcionales. `entidad` es referencia, NO filtro.
    """
    def __init__(self, df: pd.DataFrame, secuencia=12, tipo_modelo='autoencoder',
                 epochs=50, batch_size=16, umbral_factor=1.5, fondo='ROP',
                 columnas=None, entidad=None, graficos=False, min_registros=None,
                 semilla=42):
        self.df = df.copy()
        self.secuencia = int(secuencia)
        self.tipo_modelo = str(tipo_modelo).lower()
        self.epochs, self.batch_size = int(epochs), int(batch_size)
        self.umbral_factor = float(umbral_factor)
        self.dropout, self.l2_reg = .3, .001
        self.fondo = str(fondo)
        self.entidad = entidad
        self.graficos = bool(graficos)
        self.semilla = int(semilla)
        self.min_registros = int(min_registros or max(30, 3*self.secuencia+6))
        self.agente, self.modulo = AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS
        if self.tipo_modelo not in ('autoencoder', 'predictor'):
            raise ValueError('tipo_modelo debe ser autoencoder o predictor')
        if self.secuencia < 2 or self.epochs < 1 or self.batch_size < 1 or self.min_registros < 1:
            raise ValueError('secuencia >= 2, epochs >= 1, batch_size >= 1 y min_registros >= 1')
        requeridas = {'fecha', 'entidad', 'fondo'}
        if faltantes := (requeridas - set(df.columns)):
            raise ValueError(f'Faltan columnas obligatorias: {sorted(faltantes)}')
        if columnas is None:
            self.columnas_numericas = [c for c in df.select_dtypes(include='number').columns
                                      if c not in {'cluster', 'anomalia', 'score_anomalia'}
                                      and not pd.api.types.is_bool_dtype(df[c])]
        else:
            self.columnas_numericas = list(columnas)
        if (not self.columnas_numericas or len(set(self.columnas_numericas)) != len(self.columnas_numericas)
                or set(self.columnas_numericas) - set(df.columns)):
            raise ValueError('Indique indicadores numéricos existentes y sin repetir mediante columnas=...')
        self.columnas = self.columnas_numericas
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], errors='coerce')
        self.df = self.df.loc[self.df['fondo'].astype(str).str.strip().str.casefold()
                              == self.fondo.strip().casefold()].copy()
        if self.df.empty or self.df['fecha'].notna().sum() == 0:
            raise ValueError(f'No hay registros con fechas válidas del fondo {self.fondo!r}')
        self.fechaTexto = self.df['fecha'].max().strftime('%Y%m%d')
        base = Path(DATA_ROOT)
        self.ruta_resultado = str(base / RESULTADOS_ROOT / self.agente / self.modulo / self.fechaTexto)
        self.ruta_modelo = str(base / MODELOS_ROOT / self.agente / self.modulo)
        self.ruta_temporal = str(base / TEMPORAL_ROOT / self.agente / self.modulo)
        Path(self.ruta_resultado).mkdir(parents=True, exist_ok=True)
        self.resultado = []
        self.errores_por_entidad = {}
        self.modelos_por_entidad = {}
        self.rutas_modelos_por_entidad = {}
        self.rutas_resultados_por_entidad = {}
        self.historial_hiperparametros = pd.DataFrame()
        self.df_resultado_global = pd.DataFrame()
        self.ruta_resultado_global = None
        self.scaler = None
        self.mejor_modelo = None

    def _preparar_entidad(self, datos):
        datos = datos.copy().sort_values('fecha', kind='stable')
        for c in self.columnas:
            datos[c] = pd.to_numeric(datos[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        datos = datos.dropna(subset=['fecha'] + self.columnas).reset_index(drop=True)
        if len(datos) < self.min_registros:
            raise ValueError(f'{len(datos)} registros válidos: mínimo requerido {self.min_registros}')
        if datos['fecha'].nunique() < self.min_registros:
            raise ValueError('Hay fechas repetidas o insuficientes. Consolidar a la frecuencia correcta antes de entrenar.')
        if datos['fecha'].duplicated().any():
            raise ValueError('Hay fechas duplicadas por entidad; se requiere una observación por fecha.')
        constantes = [c for c in self.columnas if datos[c].nunique() < 2]
        if constantes:
            raise ValueError(f'Indicadores constantes: {constantes}')
        return datos

    def preparar_datos(self, df, columnas):
        """Compatibilidad: transformación exploratoria; no usar para validación temporal."""
        datos = df.copy().sort_values('fecha')
        self.scaler = MinMaxScaler().fit(datos[columnas])
        return _ventanas(self.scaler.transform(datos[columnas]),
                         pd.to_datetime(datos['fecha']).tolist(), self.secuencia, self.tipo_modelo)

    def crear_modelo(self, input_shape):
        tf = _keras()
        tf.keras.utils.set_random_seed(self.semilla)
        reg = tf.keras.regularizers.l2(self.l2_reg)
        layers = tf.keras.layers
        if self.tipo_modelo == 'autoencoder':
            modelo = tf.keras.Sequential([
                layers.Input(shape=input_shape),
                layers.LSTM(32, kernel_regularizer=reg), layers.Dropout(self.dropout),
                layers.RepeatVector(input_shape[0]),
                layers.LSTM(32, return_sequences=True, kernel_regularizer=reg),
                layers.TimeDistributed(layers.Dense(input_shape[1]))])
        else:
            modelo = tf.keras.Sequential([
                layers.Input(shape=input_shape),
                layers.LSTM(32, kernel_regularizer=reg), layers.Dropout(self.dropout),
                layers.Dense(input_shape[1])])
        modelo.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return modelo

    def _entrenar(self, X, y, X_val, y_val):
        tf = _keras()
        tf.keras.backend.clear_session()
        m = self.crear_modelo((X.shape[1], X.shape[2]))
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                       restore_best_weights=True)]
        history = m.fit(X, y, validation_data=(X_val, y_val), batch_size=self.batch_size,
                        epochs=self.epochs, callbacks=callbacks, verbose=0, shuffle=False)
        return m, history

    @staticmethod
    def _puntuaciones(modelo, X, y):
        pred = modelo.predict(X, verbose=0)
        if pred.shape != y.shape:
            raise ValueError(f'Predicción {pred.shape} no coincide con objetivo {y.shape}')
        err = np.square(np.asarray(y)-np.asarray(pred))
        return err.mean(axis=tuple(range(1, err.ndim))), (err.mean(axis=1) if err.ndim == 3 else err)

    def _bloques(self, n):
        """Reparte cronológicamente observaciones crudas en 60/20/20."""
        i = max(self.secuencia+3, int(n*.6))
        j = max(i+3, int(n*.8))
        if n-j < 3 or j-i < 3:
            raise ValueError('La entidad no tiene suficientes registros para train/calibración/prueba.')
        return i, j

    def _crear_segmento(self, df, inicio_objetivos, fin_objetivos, scaler):
        """Transforma contexto histórico, sin introducir objetivos de otros segmentos."""
        inicio = max(0, inicio_objetivos-self.secuencia)
        matriz = scaler.transform(df.iloc[inicio:fin_objetivos][self.columnas])
        X, y, fechas = _ventanas(matriz, df['fecha'].iloc[inicio:fin_objetivos].tolist(),
                                 self.secuencia, self.tipo_modelo)
        # Un autoencoder etiqueta el final de cada ventana; excluir ventanas cuyo fin
        # pertenece al segmento anterior. El predictor tiene objetivo inmediatamente posterior.
        fechas_min = df['fecha'].iloc[inicio_objetivos]
        mask = fechas >= fechas_min
        return X[mask.to_numpy()], y[mask.to_numpy()], fechas.loc[mask].reset_index(drop=True)

    def _guardar_modelo(self, entidad, modelo, scaler, umbral, config=None):
        base = Path(self.ruta_modelo)
        base.mkdir(parents=True, exist_ok=True)
        id_ = f'{_seguro(self.fondo)}_{_seguro(entidad)}_{self.tipo_modelo}'
        keras_path, pkl_path = base / f'{id_}.keras', base / f'{id_}_metadata.pkl'
        modelo.save(keras_path)
        metadata = {'version': 3, 'archivo_modelo': keras_path.name, 'scaler': scaler,
                    'umbral': float(umbral), 'columnas': list(self.columnas),
                    'fondo': self.fondo, 'entidad': str(entidad), 'tipo_modelo': self.tipo_modelo,
                    'secuencia': self.secuencia, 'config': config or {}}
        with pkl_path.open('wb') as fh:
            pickle.dump(metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return str(keras_path), str(pkl_path)

    def _entrenar_entidad(self, entidad, datos):
        i, j = self._bloques(len(datos))
        # Entrenamiento interno/early stopping separado de calibración y prueba.
        corte = max(self.secuencia+2, int(i*.8))
        if i-corte < 2:
            raise ValueError('Muestra insuficiente para validación interna.')
        scaler = MinMaxScaler().fit(datos.iloc[:corte][self.columnas])
        Xtr, ytr, _ = self._crear_segmento(datos, self.secuencia if self.tipo_modelo=='predictor' else self.secuencia-1, corte, scaler)
        Xval, yval, _ = self._crear_segmento(datos, corte, i, scaler)
        if not len(Xtr) or not len(Xval):
            raise ValueError('No se pudieron construir ventanas para train/validación.')
        modelo, history = self._entrenar(Xtr, ytr, Xval, yval)
        # Calibración y prueba separados: scaler aprende solamente de train.
        Xcal, ycal, fechas_cal = self._crear_segmento(datos, i, j, scaler)
        Xtest, ytest, fechas_test = self._crear_segmento(datos, j, len(datos), scaler)
        if not len(Xcal) or not len(Xtest):
            raise ValueError('No se pudieron construir ventanas de calibración/prueba.')
        scores_cal, _ = self._puntuaciones(modelo, Xcal, ycal)
        # Robusto a eventos extremos en calibración.
        umbral = float(np.median(scores_cal) + self.umbral_factor * 1.4826 * np.median(np.abs(scores_cal-np.median(scores_cal))))
        umbral = max(umbral, float(np.finfo(float).eps))
        scores_test, errores_var = self._puntuaciones(modelo, Xtest, ytest)
        es_anomalia = scores_test > umbral
        drivers, indices = [], []
        for flag, fila in zip(es_anomalia, errores_var):
            if flag:
                limite = np.percentile(fila, 75)
                idx = np.flatnonzero(fila >= limite).tolist()
            else:
                idx = []
            indices.append(idx)
            drivers.append([self.columnas[k] for k in idx])
        salida = pd.DataFrame({'fecha': fechas_test, 'entidad': entidad, 'fondo': self.fondo,
                               'score_anomalia': scores_test, 'umbral': umbral,
                               'anomalia': es_anomalia, 'drivers_idx': indices,
                               'drivers': drivers, 'segmento': 'prueba_fuera_muestra'})
        long = pd.DataFrame(errores_var, columns=self.columnas)
        long.insert(0, 'fecha', fechas_test)
        long.insert(1, 'entidad', entidad)
        long.insert(2, 'fondo', self.fondo)
        long['anomalia'] = es_anomalia
        long['score_anomalia'] = scores_test
        long = long.melt(id_vars=['fecha', 'entidad','fondo','anomalia','score_anomalia'],
                         var_name='variable', value_name='error_variable')
        por_fecha = {pd.Timestamp(f): set(d) for f, d in zip(fechas_test, drivers)}
        long['es_driver'] = [bool(r.variable in por_fecha.get(pd.Timestamp(r.fecha), set()))
                             for r in long.itertuples(index=False)]
        rpath = Path(self.ruta_resultado)
        ident = f'{_seguro(self.fondo)}_{_seguro(entidad)}_{self.tipo_modelo}'
        csv = rpath / f'anomalias_{ident}.csv'
        salida.to_csv(csv, index=False)
        long.to_csv(rpath / f'errors_long_{ident}.csv', index=False)
        self.calcular_drivers_aggregados(long).to_csv(rpath / f'drivers_entidad_{ident}.csv', index=False)
        config = {'epochs': self.epochs, 'batch_size': self.batch_size,
                  'dropout': self.dropout, 'l2': self.l2_reg, 'umbral_factor': self.umbral_factor}
        keras_path, meta_path = self._guardar_modelo(entidad, modelo, scaler, umbral, config)
        if self.graficos:
            self._graficar(salida, entidad)
        self.modelos_por_entidad[entidad] = modelo
        self.rutas_modelos_por_entidad[entidad] = meta_path
        self.rutas_resultados_por_entidad[entidad] = str(csv)
        historia = {'entidad': entidad, 'fondo': self.fondo, 'tipo_modelo': self.tipo_modelo,
                    'n_entrenamiento': len(Xtr), 'n_validacion': len(Xval),
                    'n_calibracion': len(Xcal), 'n_prueba': len(Xtest),
                    'mse_calibracion': float(scores_cal.mean()),
                    'mse_prueba': float(scores_test.mean()), 'umbral': umbral,
                    'n_anomalias': int(es_anomalia.sum()), 'epochs_efectivos': len(history.history['loss']),
                    'modelo': keras_path, 'metadata': meta_path}
        return salida, long, historia

    def _graficar(self, salida, entidad):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(salida['fecha'], salida['score_anomalia'], label='Error normalizado')
        anom = salida.loc[salida['anomalia']]
        ax.scatter(anom['fecha'], anom['score_anomalia'], label='Anomalía', zorder=3)
        ax.axhline(float(salida['umbral'].iloc[0]), linestyle='--', label='Umbral calibrado')
        ax.set_title(f'Anomalías fuera de muestra: {entidad}')
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(Path(self.ruta_resultado)/f'anomalias_{_seguro(self.fondo)}_{_seguro(entidad)}_{self.tipo_modelo}.png', dpi=150)
        plt.close(fig)

    @staticmethod
    def calcular_drivers_aggregados(df_long, top_n=10):
        base = df_long.loc[df_long['anomalia'].astype(bool) & df_long['es_driver'].astype(bool)].copy()
        cols = ['variable', 'count_anomalias', 'mean_error', 'median_error', 'sum_error', 'freq_relativa']
        if base.empty:
            return pd.DataFrame(columns=cols)
        base['_evento'] = base['entidad'].astype(str) + '|' + pd.to_datetime(base['fecha']).astype(str)
        eventos = df_long.loc[df_long['anomalia'].astype(bool), ['entidad', 'fecha']].drop_duplicates()
        denominador = max(1, len(eventos))
        tabla = base.groupby('variable').agg(count_anomalias=('_evento', 'nunique'),
                  mean_error=('error_variable', 'mean'), median_error=('error_variable', 'median'),
                  sum_error=('error_variable', 'sum')).reset_index()
        tabla['freq_relativa'] = tabla['count_anomalias'] / denominador
        return tabla.sort_values(['count_anomalias', 'mean_error'], ascending=False).head(top_n)

    def drivers_anomalia(self, grupo_df, resultado_df, columnas=None, top_n=None):
        """Comparación descriptiva alineada por entidad y fecha, no causal."""
        columnas = list(columnas or self.columnas)
        base = grupo_df[['entidad', 'fecha']+columnas].copy()
        base['fecha'] = pd.to_datetime(base['fecha'])
        res = resultado_df[['entidad', 'fecha','anomalia']].copy()
        res['fecha'] = pd.to_datetime(res['fecha'])
        cruza = res.merge(base, on=['entidad','fecha'], how='left', validate='one_to_one')
        filas = []
        for col in columnas:
            normal = cruza.loc[~cruza['anomalia'].astype(bool), col].mean()
            anom = cruza.loc[cruza['anomalia'].astype(bool), col].mean()
            filas.append({'variable': col, 'media_normal': normal, 'media_anomalo': anom,
                          'diferencia': anom-normal, 'abs_diferencia': abs(anom-normal)})
        tabla = pd.DataFrame(filas).sort_values('abs_diferencia', ascending=False, na_position='last')
        return tabla.head(top_n) if top_n else tabla

    def ejecutar(self):
        resultados, longs, historiales = [], [], []
        self.errores_por_entidad = {}
        self.modelos_por_entidad = {}
        self.rutas_modelos_por_entidad = {}
        self.rutas_resultados_por_entidad = {}
        for entidad in sorted(self.df['entidad'].dropna().astype(str).unique()):
            try:
                datos = self._preparar_entidad(self.df.loc[self.df['entidad'].astype(str) == entidad])
                salida, long, historia = self._entrenar_entidad(entidad, datos)
                resultados.append(salida); longs.append(long); historiales.append(historia)
            except Exception as exc:
                self.errores_por_entidad[entidad] = str(exc)
                warnings.warn(f'Entidad {entidad}: {exc}', stacklevel=2)
        if not resultados:
            raise ValueError(f'Ninguna entidad pudo procesarse: {self.errores_por_entidad}')
        rpath = Path(self.ruta_resultado)
        global_df = pd.concat(resultados, ignore_index=True).sort_values(['entidad','fecha'])
        self.df_resultado_global = global_df.reset_index(drop=True)
        self.ruta_resultado_global = str(rpath / f'historial_anomalias_{self.tipo_modelo}_{_seguro(self.fondo)}.csv')
        global_df.to_csv(self.ruta_resultado_global, index=False)
        df_long = pd.concat(longs, ignore_index=True)
        df_long.to_csv(rpath / f'errors_long_global_{self.tipo_modelo}_{_seguro(self.fondo)}.csv', index=False)
        self.calcular_drivers_aggregados(df_long, top_n=20).to_csv(
            rpath / f'drivers_agrupados_{self.tipo_modelo}_{_seguro(self.fondo)}.csv', index=False)
        self.historial_hiperparametros = pd.DataFrame(historiales)
        self.historial_hiperparametros.to_csv(
            rpath / f'historial_hiperparametros_{self.tipo_modelo}_{_seguro(self.fondo)}.csv', index=False)
        self.df_drivers_global = self.drivers_anomalia(
            grupo_df=self.df,
            resultado_df=self.df_resultado_global,
            columnas=self.columnas_numericas,
            top_n=10
        )

        self.ruta_drivers_global = str(
            Path(self.ruta_resultado) / "drivers_entidad_global.csv"
        )

        self.df_drivers_global.to_csv(
            self.ruta_drivers_global,
            index=False
        )
        pd.DataFrame([{'entidad': e, 'error': err} for e, err in self.errores_por_entidad.items()],
                     columns=['entidad','error']).to_csv(rpath / f'errores_entidades_{_seguro(self.fondo)}.csv', index=False)
        self.resultado = [f'Anomalías LSTM: {len(resultados)} entidades del fondo {self.fondo}']
        return self.df_resultado_global


def predecir_anomalias(df_nuevo, ruta_metadata):
    """Inferencia sin reentrenamiento sobre una serie de UNA entidad ordenada por fecha.

    Incluya al menos `secuencia` puntos anteriores al primer objetivo de interés.
    """
    tf = _keras()
    ruta = Path(ruta_metadata)
    with ruta.open('rb') as f:
        meta = pickle.load(f)  # Cargar únicamente archivos confiables
    if 'fondo' in df_nuevo and not df_nuevo['fondo'].astype(str).eq(meta['fondo']).all():
        raise ValueError('Fondo distinto del modelo guardado')
    if 'entidad' in df_nuevo and not df_nuevo['entidad'].astype(str).eq(meta['entidad']).all():
        raise ValueError('Entidad distinta del modelo guardado')
    columnas = meta['columnas']
    if falta := (set(columnas) - set(df_nuevo.columns)):
        raise ValueError(f'Faltan indicadores: {sorted(falta)}')
    datos = df_nuevo.copy()
    datos['fecha'] = pd.to_datetime(datos['fecha'], errors='coerce')
    datos = datos.sort_values('fecha').dropna(subset=['fecha'])
    for col in columnas:
        datos[col] = pd.to_numeric(datos[col], errors='coerce').replace([np.inf,-np.inf], np.nan)
    if datos[columnas].isna().any().any():
        raise ValueError('Los datos nuevos contienen indicadores nulos/no numéricos; no se imputan implícitamente')
    if datos['fecha'].duplicated().any():
        raise ValueError('Fechas duplicadas en la entidad')
    matriz = meta['scaler'].transform(datos[columnas])
    X, y, fechas = _ventanas(matriz, datos['fecha'].tolist(), meta['secuencia'], meta['tipo_modelo'])
    if not len(X):
        raise ValueError('No existen suficientes registros para construir una ventana')
    model = tf.keras.models.load_model(ruta.with_name(meta['archivo_modelo']))
    scores, errs = LSTMAnomalyDetector._puntuaciones(model, X, y)
    return pd.DataFrame({'fecha': fechas, 'entidad': meta['entidad'], 'fondo': meta['fondo'],
                         'score_anomalia': scores, 'umbral': meta['umbral'],
                         'anomalia': scores > meta['umbral']})


def main(argv=None):
    p = argparse.ArgumentParser(description='Detector LSTM de anomalías de liquidez por entidad.')
    #p.add_argument('--archivo', help='Ruta .parquet o .csv; si se omite usa --fecha')
    p.add_argument('--fecha', default='2026-08-31', help='Fecha del archivo temporal, AAAA-MM-DD')
    p.add_argument('--fondo', default='ROP')
    p.add_argument('--entidad', default=None, help='Entidad de referencia; se procesan todas las del fondo')
    p.add_argument('--columnas', nargs='+', help='Indicadores numéricos exactos del archivo')
    p.add_argument('--tipo-modelo', choices=['autoencoder','predictor'], default='autoencoder')
    p.add_argument('--secuencia', type=int, default=12)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--umbral-factor', type=float, default=1.5)
    p.add_argument('--min-registros', type=int, default=None)
    p.add_argument('--graficos', action='store_true')
    args = p.parse_args(argv)
    #if not args.archivo and not args.fecha:
    #    p.error('Indique --archivo o --fecha')
    archivo = Path(DATA_ROOT)/TEMPORAL_ROOT/AGENTE_LIQUIDEZ/LIQ_MODULO_ANOMALIAS/f'{args.fecha}.parquet'
    if not archivo.is_file():
        p.error(f'No existe el archivo: {archivo.resolve()}')
    try:
        if archivo.suffix.lower() == '.parquet':
            df = pd.read_parquet(archivo)
        elif archivo.suffix.lower() == '.csv':
            df = pd.read_csv(archivo)
        else:
            p.error('Formato de entrada soportado: .parquet o .csv')
        detector = LSTMAnomalyDetector(df, secuencia=args.secuencia, tipo_modelo=args.tipo_modelo,
            epochs=args.epochs, batch_size=args.batch_size, umbral_factor=args.umbral_factor,
            fondo=args.fondo, entidad=args.entidad, columnas=args.columnas,
            graficos=args.graficos, min_registros=args.min_registros)
        print('Indicadores:', detector.columnas_numericas)
        out = detector.ejecutar()
        print('Registros evaluados fuera de muestra:', len(out))
        print('Entidades procesadas:', len(detector.rutas_resultados_por_entidad))
        print('CSV consolidado:', Path(detector.ruta_resultado_global).resolve())
        for ent, ruta in detector.rutas_resultados_por_entidad.items():
            print(f'  {ent}: {ruta}')
        for ent, error in detector.errores_por_entidad.items():
            print(f'  ERROR {ent}: {error}')
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
