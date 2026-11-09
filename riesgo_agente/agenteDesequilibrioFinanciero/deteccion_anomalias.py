"""Detección de anomalías de desequilibrio financiero mediante IsolationForest.

Por defecto procesa independientemente cada entidad de un fondo. Los resultados
históricos de modo rolling se puntúan fuera de muestra (ventana anterior).
Los primeros `window_size` registros quedan sin evaluar, nunca se consideran normales.
Los scores más bajos indican mayor anomalía y las etiquetas son 0 normal, 1 anómala,
NA no evaluada. Los umbrales son cuantiles del entrenamiento y NO alertas regulatorias.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

try:
    from riesgo_agente.utils.config import (
        AGENTE_DESEQUILIBRIO, DES_MODULO_ANOMALIAS,
        DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT,
    )
except ImportError:
    AGENTE_DESEQUILIBRIO = 'Agente de Desequilibrio Financiero'
    DES_MODULO_ANOMALIAS = 'Detección de Anomalías Financieras'
    DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT = 'data', 'temporal', 'resultados', 'modelos'


def _seguro(v):
    return re.sub(r'[^\w.\-]+', '_', str(v), flags=re.UNICODE).strip('_') or 'sin_nombre'


def _guardar_pickle(obj, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=ruta.parent, suffix='.tmp', delete=False) as f:
            temporal = Path(f.name)
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporal, ruta)
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()


class AnomaliaDetector:
    def __init__(self, df: pd.DataFrame, df_resultado: pd.DataFrame | None = None,
                 rolling: bool = True, columnas: list[str] | None = None,
                 window_size: int = 24, step_size: int = 6,
                 contamination: float = 0.05, random_state: int = 42,
                 por_entidad: bool = True, fondo: str = 'ROP',
                 min_registros: int = 12, n_estimators: int = 100):
        if not por_entidad:
            raise ValueError('Esta versión exige por_entidad=True para evitar mezclar entidades.')
        if not 0 < contamination < 0.5:
            raise ValueError('contamination debe estar entre 0 y 0.5 (exclusivo).')
        if window_size < 8 or step_size < 1 or min_registros < 8:
            raise ValueError('window_size y min_registros >= 8, step_size >= 1.')
        self.columnas = list(columnas or [])
        if not self.columnas or len(set(self.columnas)) != len(self.columnas):
            raise ValueError('Indique una lista de columnas numéricas sin duplicados.')
        requeridas = set(self.columnas) | {'fecha', 'entidad', 'fondo'}
        faltantes = sorted(requeridas - set(df.columns))
        if faltantes:
            raise ValueError(f'Columnas faltantes: {faltantes}')
        self.fondo = str(fondo)
        self.df = df.loc[df['fondo'].astype(str).str.strip().str.casefold() == self.fondo.strip().casefold()].copy()
        if self.df.empty:
            raise ValueError(f'No existen observaciones del fondo {fondo!r}.')
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], errors='coerce')
        self.df = self.df.loc[self.df['entidad'].notna()].copy()
        if self.df['fecha'].notna().sum() == 0:
            raise ValueError('No existen fechas válidas.')
        self.fechaTexto = self.df['fecha'].max().strftime('%Y%m%d')
        self.df_resultado = df_resultado
        self.rolling = bool(rolling)
        self.columnas_modelo = self.columnas
        self.window_size, self.step_size = int(window_size), int(step_size)
        self.min_registros, self.n_estimators = int(min_registros), int(n_estimators)
        self.contamination, self.random_state = float(contamination), int(random_state)
        self.por_entidad = True
        self.agente, self.modulo = AGENTE_DESEQUILIBRIO, DES_MODULO_ANOMALIAS
        base = Path(DATA_ROOT)
        self.ruta_resultado = str(base / RESULTADOS_ROOT / self.agente / self.modulo / self.fechaTexto)
        self.ruta_modelo = str(base / MODELOS_ROOT / self.agente / self.modulo)
        self.ruta_temporal = str(base / TEMPORAL_ROOT / self.agente / self.modulo)
        Path(self.ruta_resultado).mkdir(parents=True, exist_ok=True)
        self.entidades_disponibles = sorted(self.df['entidad'].astype(str).unique())
        self.modelos_por_entidad = {}  # diccionarios: modelo, scaler, umbral, columnas, etc.
        self.modelos_por_ventana = {}
        self.rutas_modelos_por_entidad = {}
        self.rutas_resultados_por_entidad = {}
        self.errores_por_entidad = {}
        self.df_validacion = pd.DataFrame()
        self.df_resultado_global = pd.DataFrame()
        self.resultado = None  # DataFrame real, no lista de mensajes
        self.modelo, self.scaler, self.parametros, self.score = None, None, None, None

    def _preparar_entidad(self, df):
        sub = df.copy()
        sub['fecha'] = pd.to_datetime(sub['fecha'], errors='coerce')
        for c in self.columnas:
            sub[c] = pd.to_numeric(sub[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(subset=['fecha'] + self.columnas).sort_values('fecha').reset_index(drop=True)
        if len(sub) < self.min_registros:
            raise ValueError(f'Se requieren {self.min_registros} registros completos; hay {len(sub)}.')
        if sub['fecha'].nunique() != len(sub):
            raise ValueError('Existen fechas duplicadas para esta entidad; agregue los datos antes del modelado.')
        constantes = [c for c in self.columnas if sub[c].nunique() < 2]
        if constantes:
            raise ValueError(f'Indicadores constantes: {constantes}')
        return sub

    def _ajustar(self, train):
        scaler = RobustScaler().fit(train[self.columnas])
        x = scaler.transform(train[self.columnas])
        modelo = IsolationForest(n_estimators=self.n_estimators, contamination='auto',
                                 random_state=self.random_state, n_jobs=-1)
        modelo.fit(x)
        scores = modelo.score_samples(x)
        umbral = float(np.quantile(scores, self.contamination))
        return {'modelo': modelo, 'scaler': scaler, 'umbral': umbral,
                'columnas': self.columnas, 'fondo': self.fondo,
                'fecha_entrenamiento': str(train['fecha'].max().date()),
                'contamination': self.contamination}

    def _aplicar(self, df, bundle):
        scores = bundle['modelo'].score_samples(bundle['scaler'].transform(df[self.columnas]))
        return scores, (scores < bundle['umbral']).astype(int)

    def _evaluacion_entidad(self, sub, entidad, n_splits=5):
        # Ventanas cronológicas train/test; nunca se ajusta sobre test.
        n = len(sub)
        tam_train = self.window_size if self.rolling else max(self.min_registros, int(n * .6))
        if n <= tam_train:
            return []
        step = max(1, int(np.ceil((n - tam_train) / max(1, n_splits))))
        filas = []
        for fin_train in range(tam_train, n, step):
            train = sub.iloc[max(0, fin_train - self.window_size) if self.rolling else 0:fin_train]
            test = sub.iloc[fin_train:min(n, fin_train + step)]
            if test.empty:
                continue
            b = self._ajustar(train)
            s_train = b['modelo'].score_samples(b['scaler'].transform(train[self.columnas]))
            s_test, pred = self._aplicar(test, b)
            ks, pv = ks_2samp(s_train, s_test)
            filas.append({'entidad': entidad, 'split': len(filas)+1,
                          'fecha_fin_train': train['fecha'].max(),
                          'fecha_inicio_test': test['fecha'].min(),
                          'observaciones_train': len(train), 'observaciones_test': len(test),
                          'anomalias_detectadas': int(pred.sum()),
                          'porcentaje_anomalias': float(pred.mean()*100),
                          'score_train_mean': float(s_train.mean()),
                          'score_test_mean': float(s_test.mean()),
                          'ks_stat': float(ks), 'ks_pvalue': float(pv),
                          'umbral': b['umbral']})
        return filas

    def validacion_cruzada_rolling(self, df=None, n_splits=5):
        return self._validacion(df, n_splits=n_splits, rolling=True)

    def validacion_cruzada(self, df=None, fecha_col='fecha', n_splits=5):
        if fecha_col != 'fecha':
            raise ValueError('La columna temporal requerida es fecha.')
        return self._validacion(df, n_splits=n_splits, rolling=False)

    def _validacion(self, df=None, n_splits=5, rolling=None):
        original = self.rolling
        self.rolling = original if rolling is None else rolling
        filas = []
        try:
            base = self.df if df is None else df
            for entidad, grupo in base.groupby('entidad'):
                try:
                    filas.extend(self._evaluacion_entidad(self._preparar_entidad(grupo), str(entidad), n_splits))
                except ValueError:
                    pass
        finally:
            self.rolling = original
        self.df_validacion = pd.DataFrame(filas)
        return self.df_validacion

    def _detectar_entidad(self, sub, entidad):
        salida = sub.copy()
        salida['score_anomalia'] = np.nan
        salida['anomalia'] = pd.Series(pd.NA, index=salida.index, dtype='Int64')
        salida['evaluado'] = 0
        bundles = []
        if self.rolling:
            if len(sub) <= self.window_size:
                raise ValueError(f'Rolling requiere más de {self.window_size} observaciones.')
            # Una predicción temporal fuera de muestra por cada bloque, sin reentrenar en test.
            for inicio in range(self.window_size, len(sub), self.step_size):
                train = sub.iloc[inicio-self.window_size:inicio]
                test = sub.iloc[inicio:min(len(sub), inicio+self.step_size)]
                b = self._ajustar(train)
                sc, pr = self._aplicar(test, b)
                salida.loc[test.index, 'score_anomalia'] = sc
                salida.loc[test.index, 'anomalia'] = pr
                salida.loc[test.index, 'evaluado'] = 1
                bundles.append(b)
        else:
            # Retrospectiva del conjunto completo: in-sample, no equivalente a backtesting.
            b = self._ajustar(sub)
            sc, pr = self._aplicar(sub, b)
            salida['score_anomalia'] = sc
            salida['anomalia'] = pd.Series(pr, dtype='Int64')
            salida['evaluado'] = 1
            bundles.append(b)
        # Modelo de despliegue: se ajusta con el histórico completo (no para evaluar backtest).
        final = self._ajustar(sub)
        final['entidad'] = entidad
        final['rolling'] = self.rolling
        final['window_size'] = self.window_size
        final['step_size'] = self.step_size
        self.modelos_por_entidad[entidad] = final
        self.modelos_por_ventana[entidad] = bundles
        return salida

    def entrenar_y_detectar(self, df=None):
        resultados = []
        base = self.df if df is None else df
        self.errores_por_entidad = {}
        for entidad, grupo in base.groupby('entidad'):
            entidad = str(entidad)
            try:
                sub = self._preparar_entidad(grupo)
                salida = self._detectar_entidad(sub, entidad)
                resultados.append(salida)
            except Exception as exc:
                self.errores_por_entidad[entidad] = str(exc)
                warnings.warn(f'Entidad {entidad}: {exc}')
        if not resultados:
            raise ValueError(f'Ninguna entidad pudo procesarse: {self.errores_por_entidad}')
        self.df_resultado_global = pd.concat(resultados, ignore_index=True).sort_values(['entidad', 'fecha']).reset_index(drop=True)
        self.resultado = self.df_resultado_global
        return self.resultado

    def ejecutar(self, graficos=False, n_splits=5):
        self._validacion(n_splits=n_splits)
        self.df_validacion.to_csv(Path(self.ruta_resultado) / 'historial_validacion_anomalias.csv', index=False)
        salida = self.entrenar_y_detectar()
        resumenes_drivers = []
        for entidad, grupo in salida.groupby('entidad'):
            sufijo = f'{_seguro(self.fondo)}_{_seguro(entidad)}'
            csv = Path(self.ruta_resultado) / f'resultados_{sufijo}.csv'
            grupo.to_csv(csv, index=False)
            self.rutas_resultados_por_entidad[str(entidad)] = str(csv)
            modelo = Path(self.ruta_modelo) / f'{sufijo}_modelo.pkl'
            _guardar_pickle(self.modelos_por_entidad[str(entidad)], modelo)
            self.rutas_modelos_por_entidad[str(entidad)] = str(modelo)
            dr = self.drivers_anomalia(df=grupo)
            dr.insert(0, 'entidad', entidad)
            dr.to_csv(Path(self.ruta_resultado) / f'drivers_{sufijo}.csv', index=False)
            resumenes_drivers.append(dr)
            if graficos:
                vis = VisualizadorAnomalias(grupo, self.columnas)
                vis.ruta_resultado = str(Path(self.ruta_resultado) / f'graficos_{_seguro(entidad)}')
                vis.graficar_anomalias_temporales(entidad)
                if len(self.columnas) >= 2:
                    vis.graficar_dispersion()
                vis.graficar_top_drivers(dr)
        ruta_global = Path(self.ruta_resultado) / f'resultados_{_seguro(self.fondo)}.csv'
        salida.to_csv(ruta_global, index=False)
        self.ruta_resultado_global = str(ruta_global)
        pd.concat(resumenes_drivers, ignore_index=True).to_csv(
            Path(self.ruta_resultado) / f'drivers_{_seguro(self.fondo)}.csv', index=False)
        # Compatibilidad con el reporte histórico de drivers del subagente.
        # Se calcula sobre el consolidado, pero únicamente con indicadores del modelo.
        nombre_drivers = 'drivers_anomalias_rolling.csv' if self.rolling else 'drivers_anomalias_df.csv'
        df_drivers = self.drivers_anomalia(df=salida)
        df_drivers.to_csv(Path(self.ruta_resultado) / nombre_drivers, index=False)
        self.ruta_drivers_anomalias = str(Path(self.ruta_resultado) / nombre_drivers)        
        pd.DataFrame([{'entidad': k, 'error': v} for k, v in self.errores_por_entidad.items()],
                     columns=['entidad', 'error']).to_csv(Path(self.ruta_resultado) / 'errores_por_entidad.csv', index=False)
        return salida

    def guardar_modelo(self, ruta):
        """Exporta conjunto de bundles por entidad (nunca reutiliza un scaler global)."""
        _guardar_pickle({'version': 2, 'fondo': self.fondo, 'columnas': self.columnas,
                         'modelos_por_entidad': self.modelos_por_entidad}, ruta)

    def cargar_modelo(self, ruta):
        with open(ruta, 'rb') as f:
            d = pickle.load(f)  # Solo archivos confiables.
        if 'modelos_por_entidad' not in d:
            raise ValueError('Formato de modelo incompatible.')
        self.modelos_por_entidad = d['modelos_por_entidad']
        self.columnas = list(d['columnas'])
        self.columnas_modelo = self.columnas
        self.fondo = d['fondo']
        return self

    def predecir_anomalias(self, df_nuevo):
        obligatorias = set(self.columnas) | {'entidad', 'fondo'}
        faltan = obligatorias - set(df_nuevo.columns)
        if faltan:
            raise ValueError(f'Faltan columnas: {sorted(faltan)}')
        salida = df_nuevo.copy()
        salida['anomalia'] = pd.Series(pd.NA, index=salida.index, dtype='Int64')
        salida['score_anomalia'] = np.nan
        salida['evaluado'] = 0
        for ent, indexes in salida.groupby('entidad').groups.items():
            bundle = self.modelos_por_entidad.get(str(ent))
            if bundle is None:
                continue
            sub = salida.loc[indexes]
            if (sub['fondo'].astype(str) != self.fondo).any():
                raise ValueError(f'Fondo distinto del modelo para {ent}.')
            X = sub[self.columnas].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
            validos = X.notna().all(axis=1)
            if not validos.any():
                continue
            ok = sub.loc[validos].copy()
            ok[self.columnas] = X.loc[validos]
            sc, pr = self._aplicar(ok, bundle)
            salida.loc[ok.index, 'score_anomalia'] = sc
            salida.loc[ok.index, 'anomalia'] = pr
            salida.loc[ok.index, 'evaluado'] = 1
        return salida

    def drivers_anomalia(self, columnas=None, top_n=None, df=None):
        """Diferencia descriptiva de medias, no atribución causal."""
        fuente = self.resultado if df is None else df
        if fuente is None:
            raise ValueError('Ejecute el detector primero.')
        columnas = list(self.columnas if columnas is None else columnas)
        normales = fuente.loc[(fuente['evaluado'] == 1) & (fuente['anomalia'] == 0)]
        anomalos = fuente.loc[(fuente['evaluado'] == 1) & (fuente['anomalia'] == 1)]
        filas = []
        for c in columnas:
            mn = normales[c].mean() if len(normales) else np.nan
            ma = anomalos[c].mean() if len(anomalos) else np.nan
            filas.append({'variable': c, 'media_normal': mn, 'media_anomalo': ma,
                          'diferencia': ma-mn, 'abs_diferencia': abs(ma-mn)})
        return pd.DataFrame(filas).sort_values('abs_diferencia', ascending=False,
                                               na_position='last').head(top_n).reset_index(drop=True)

    def resumen_por_entidad(self):
        if self.resultado is None:
            return pd.DataFrame(columns=['entidad','anomalia','normal','total','%_anomalias'])
        d = self.resultado[self.resultado['evaluado'] == 1]
        return d.groupby('entidad').agg(anomalia=('anomalia','sum'), total=('anomalia','size')).assign(
            normal=lambda x: x.total-x.anomalia,
            **{'%_anomalias': lambda x: 100*x.anomalia/x.total}).reset_index()

    def resumen_general(self):
        if self.resultado is None:
            return {'total_observaciones': 0, 'evaluadas': 0, 'anomalias_detectadas': 0}
        d = self.resultado
        return {'total_observaciones': len(d), 'evaluadas': int(d['evaluado'].sum()),
                'anomalias_detectadas': int((d['anomalia'] == 1).sum())}


class VisualizadorAnomalias:
    def __init__(self, df_resultado, columnas_valores, por_entidad=True, mostrarGrafico=False):
        self.df = df_resultado.copy()
        self.columnas_valores = list(columnas_valores)
        self.por_entidad, self.mostrarGrafico = por_entidad, mostrarGrafico
        fecha = pd.to_datetime(self.df['fecha']).max().strftime('%Y%m%d')
        self.ruta_resultado = str(Path(DATA_ROOT) / RESULTADOS_ROOT /
                                  AGENTE_DESEQUILIBRIO / DES_MODULO_ANOMALIAS / fecha)

    def _fin(self, nombre):
        import matplotlib.pyplot as plt
        Path(self.ruta_resultado).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(Path(self.ruta_resultado)/nombre, bbox_inches='tight', dpi=140)
        if self.mostrarGrafico:
            plt.show()
        plt.close()

    def graficar_dispersion(self):
        import matplotlib.pyplot as plt
        if len(self.columnas_valores) < 2:
            return
        x, y = self.columnas_valores[:2]
        plt.figure(figsize=(9, 5))
        for a, sub in self.df[self.df['evaluado'] == 1].groupby('anomalia'):
            plt.scatter(sub[x], sub[y], s=22, label='Anómalo' if a == 1 else 'Normal')
        plt.xlabel(x); plt.ylabel(y); plt.legend(); plt.title('Dispersión de anomalías')
        self._fin('dispersion.png')

    def graficar_anomalias_temporales(self, entidad):
        import matplotlib.pyplot as plt
        sub = self.df[self.df['entidad'].astype(str) == str(entidad)].sort_values('fecha')
        plt.figure(figsize=(10,4))
        plt.plot(sub['fecha'], sub['score_anomalia'], label='Puntuación')
        anomalos = sub[sub['anomalia'] == 1]
        plt.scatter(anomalos['fecha'], anomalos['score_anomalia'], label='Anomalía', color='red')
        plt.title(f'Anomalías: {entidad}'); plt.xlabel('Fecha'); plt.ylabel('score_samples (menor = anómalo)')
        plt.legend(); self._fin('anomalias_temporales.png')

    def graficar_anomalias_temporales_todas(self, altura_por_entidad=2):
        for ent in self.df['entidad'].dropna().unique():
            self.graficar_anomalias_temporales(ent)

    def resumen_por_entidad(self):
        return self.df[self.df['evaluado'] == 1].groupby('entidad')['anomalia'].agg(['sum','count'])

    def mapa_calor_variables(self):
        # La anomalía es multivariante; un conteo por variable no identifica drivers.
        import matplotlib.pyplot as plt
        import seaborn as sns
        d = self.df[self.df['anomalia'] == 1]
        if d.empty:
            return
        # Medias de indicadores originales por entidad, sin atribución causal.
        tabla = d.groupby('entidad')[self.columnas_valores].mean()
        plt.figure(figsize=(9,4)); sns.heatmap(tabla, cmap='viridis', annot=True)
        self._fin('mapa_variables_anomalas.png')

    def graficar_top_drivers(self, df_drivers, top_n=10):
        import matplotlib.pyplot as plt
        d = df_drivers.head(top_n).dropna(subset=['abs_diferencia'])
        if d.empty:
            return
        plt.figure(figsize=(9,4)); plt.barh(d['variable'], d['abs_diferencia'])
        plt.xlabel('Diferencia absoluta de medias (descriptiva)')
        self._fin('top_drivers.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Detección de anomalías por entidad con IsolationForest.')
    #parser.add_argument('--archivo', required=True, help='CSV o Parquet de entrada')
    parser.add_argument('--fecha', default='2026-08-31', help='Fecha del archivo temporal, AAAA-MM-DD')    
    parser.add_argument('--fondo', default='ROP')
    parser.add_argument('--entidad', default=None, help='Referencia opcional; se procesan todas las entidades del fondo')
    parser.add_argument('--columnas', nargs='+', default=None, help='Indicadores explícitos; recomendados')
    parser.add_argument('--rolling', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--window-size', type=int, default=24)
    parser.add_argument('--step-size', type=int, default=6)
    parser.add_argument('--contamination', type=float, default=.05)
    parser.add_argument('--min-registros', type=int, default=12)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--graficos', action='store_true')
    args = parser.parse_args(argv)

    archivo = Path(DATA_ROOT)/TEMPORAL_ROOT/AGENTE_DESEQUILIBRIO/DES_MODULO_ANOMALIAS/f'{args.fecha}.parquet'
    if not archivo.is_file():
        parser.error(f'No se encontró: {archivo}')
    try:
        df = pd.read_parquet(archivo) if archivo.suffix.lower() == '.parquet' else pd.read_csv(archivo)
        if args.columnas is None:
            excluir = {'entidad','fondo','fecha','cluster','anomalia','score_anomalia','evaluado'}
            columnas = [c for c in df.select_dtypes(include='number').columns if c not in excluir]
            print('Columnas numéricas detectadas:', columnas)
            print('Compruebe que ninguna es identificador; use --columnas para control explícito.')
        else:
            columnas = args.columnas
        detector = AnomaliaDetector(df, rolling=args.rolling, columnas=columnas,
                                   fondo=args.fondo, window_size=args.window_size,
                                   step_size=args.step_size, contamination=args.contamination,
                                   min_registros=args.min_registros, n_estimators=args.n_estimators)
        resultado = detector.ejecutar(graficos=args.graficos)
        print(f'Registros: {len(resultado)}; evaluados: {int(resultado.evaluado.sum())}; '
              f'entidades: {len(detector.rutas_resultados_por_entidad)}')
        print('Consolidado:', Path(detector.ruta_resultado_global).resolve())
        for ent, ruta in detector.rutas_resultados_por_entidad.items():
            print(f'  {ent}: {ruta}')
        for ent, err in detector.errores_por_entidad.items():
            print(f'  ERROR {ent}: {err}')
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
