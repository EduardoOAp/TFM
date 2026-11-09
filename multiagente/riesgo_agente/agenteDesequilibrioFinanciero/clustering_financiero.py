"""Clustering financiero: comparación, estabilidad temporal y aplicación reproducible.

La clasificación futura se admite únicamente para KMeans y GaussianMixture.
DBSCAN y AgglomerativeClustering se comparan con fines exploratorios.
No se atribuyen categorías regulatorias de riesgo a clústeres sin validación experta.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from riesgo_agente.utils.config import (
        AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING,
        DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT,
    )
except ImportError:
    # Permite ejecutar pruebas fuera del paquete sin alterar rutas en la aplicación.
    AGENTE_DESEQUILIBRIO = 'Agente de Desequilibrio Financiero'
    DES_MODULO_CLUSTERING = 'Clustering de Riesgo Financiero'
    DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT = 'data', 'temporal', 'resultados', 'modelos'


def _seguro(valor):
    return re.sub(r'[^\w.\-]+', '_', str(valor), flags=re.UNICODE).strip('_') or 'sin_nombre'


def _guardar_pickle(obj, destino):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.tmp', dir=destino.parent,
                                         delete=False) as archivo:
            temporal = Path(archivo.name)
            pickle.dump(obj, archivo, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporal, destino)
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()


class ClusteringEvaluator:
    """Compara clústeres en registros de una entidad y fondo específicos.

    `evaluacion_temporal`: cortes consecutivos; ajusta scaler/modelo solo en train.
    Para modelos inductivos evalúa asignaciones en test. Para DBSCAN y jerárquico
    mide estabilidad de partición al incorporar test, no predicción futura.
    """

    def __init__(self, df: pd.DataFrame, fondo: str, entidad: str, columnas: list,
                 usar_normalizado: bool = False, min_cluster_size: int = 3):
        # La entidad recibida se conserva como referencia,
        # pero NO se utiliza para filtrar el procesamiento.
        self.fondo = fondo
        self.entidad_referencia = str(entidad)
        self.columnas = list(columnas)
        self.usar_normalizado = usar_normalizado
        self.min_cluster_size = max(2, int(min_cluster_size))

        if not self.columnas or len(set(self.columnas)) != len(self.columnas):
            raise ValueError('Debe proporcionar columnas numéricas únicas para el clustering.')
        requeridas = set(self.columnas) | {'fecha', 'fondo', 'entidad'}
        ausentes = sorted(requeridas - set(df.columns))
        if ausentes:
            raise ValueError(f'Faltan columnas requeridas: {ausentes}')

        self.df_fondo = df.loc[df["fondo"].astype(str) == str(fondo)].copy()
        if self.df_fondo.empty:
            raise ValueError(
                f"No existen registros para el fondo {fondo!r}.")
        disponibles = (self.df_fondo["entidad"].dropna().astype(str).unique().tolist())

        if not disponibles:
            raise ValueError(
                f"No existen entidades válidas "
                f"para el fondo {fondo!r}."
            )

        self.entidades_disponibles = sorted(disponibles)

        # La entidad recibida se conserva como referencia.
        # NO limita las entidades procesadas por ejecutar().
        self.entidad = (self.entidad_referencia
            if self.entidad_referencia in disponibles
            else self.entidades_disponibles[0])
        datos = self.df_fondo.loc[self.df_fondo["entidad"].astype(str)
                                  == self.entidad].copy()        
        if datos.empty:
            raise ValueError(f'No existen filas para fondo={fondo!r}, entidad={entidad!r}.')
        datos['fecha'] = pd.to_datetime(datos['fecha'], errors='coerce')
        for col in self.columnas:
            datos[col] = pd.to_numeric(datos[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        self.diagnostico_datos = {
            'filas_filtradas': len(datos),
            'fechas_invalidas': int(datos['fecha'].isna().sum()),
            'filas_con_indicadores_nulos': int(datos[self.columnas].isna().any(axis=1).sum()),
        }
        datos = datos.dropna(subset=['fecha'] + self.columnas).sort_values('fecha').reset_index(drop=True)
        if len(datos) < 12:
            raise ValueError('Se requieren al menos 12 registros válidos para evaluar clusters.')
        constantes = [c for c in self.columnas if datos[c].nunique() < 2]
        if constantes:
            raise ValueError(f'Variables constantes, retírelas del clustering: {constantes}')
        self.df = datos
        self.scaler_final = StandardScaler().fit(datos[self.columnas])
        self.X = self.scaler_final.transform(datos[self.columnas])

        self.resultados = {}
        self.mejor_modelo = None
        self.modelo_final = None
        self.historial_scores = []
        self.df_validacion = pd.DataFrame()
        # Resultados de procesamiento multientidad.
        self.evaluadores_por_entidad = {}
        self.modelos_por_entidad = {}
        self.rutas_modelos_por_entidad = {}
        self.rutas_resultados_por_entidad = {}
        self.errores_por_entidad = {}

        self.df_resultado_global = pd.DataFrame() 

        self.agente, self.modulo = AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING
        self.fechaTexto = datos['fecha'].max().strftime('%Y%m%d')
        base = Path(DATA_ROOT)
        self.ruta_resultado = str(base / RESULTADOS_ROOT / self.agente / self.modulo / self.fechaTexto)
        self.ruta_modelo = str(base / MODELOS_ROOT / self.agente / self.modulo)
        self.ruta_temporal = str(base / TEMPORAL_ROOT / self.agente / self.modulo)
        Path(self.ruta_resultado).mkdir(parents=True, exist_ok=True)
        self.identificador = f'{_seguro(fondo)}_{_seguro(entidad)}'
        self.ruta_modelo_final = str(Path(self.ruta_modelo) / f'{self.identificador}_modelo.pkl')
        self.ruta_resultado_csv = str(Path(self.ruta_resultado) / f'resultados_{self.identificador}.csv')
        self.resultado = []

    @staticmethod
    def _candidatos(n):
        max_k = min(5, n // 3, n - 1)
        for k in range(2, max_k + 1):
            yield 'KMeans', k, None
            yield 'GMM', k, None
            yield 'Jerarquico', k, None
        for eps in (0.5, 1.0, 1.5):
            yield 'DBSCAN', None, eps

    @staticmethod
    def _crear(nombre, k, eps):
        if nombre == 'KMeans':
            return KMeans(n_clusters=k, random_state=42, n_init=10)
        if nombre == 'GMM':
            return GaussianMixture(n_components=k, random_state=42, n_init=3,
                                   reg_covar=1e-5)
        if nombre == 'Jerarquico':
            return AgglomerativeClustering(n_clusters=k)
        return DBSCAN(eps=eps, min_samples=5)

    @staticmethod
    def _etiquetar(modelo, X):
        if hasattr(modelo, 'fit_predict'):
            return np.asarray(modelo.fit_predict(X), dtype=int)
        modelo.fit(X)
        return np.asarray(modelo.predict(X), dtype=int)

    def _metricas(self, X, etiquetas):
        etiquetas = np.asarray(etiquetas)
        validos = etiquetas != -1
        grupos = np.unique(etiquetas[validos])
        ruido = float(np.mean(~validos))
        tamanos = np.unique(etiquetas[validos], return_counts=True)[1]
        datos = dict(n_clusters=len(grupos), ruido_pct=ruido * 100,
                     tamano_minimo=int(tamanos.min()) if len(tamanos) else 0,
                     silhouette=np.nan, calinski_harabasz=np.nan,
                     davies_bouldin=np.nan)
        # La etiqueta -1 es ruido: excluirla evita tratar todo el ruido como un clúster.
        if len(grupos) < 2 or validos.sum() <= len(grupos):
            return datos
        try:
            datos.update(silhouette=float(silhouette_score(X[validos], etiquetas[validos])),
                         calinski_harabasz=float(calinski_harabasz_score(X[validos], etiquetas[validos])),
                         davies_bouldin=float(davies_bouldin_score(X[validos], etiquetas[validos])))
        except ValueError:
            pass
        return datos

    def validacion_cruzada_temporal(self, n_splits=5):
        """Expanding-window temporal por fechas. Nunca escala test con su propio futuro."""
        fechas = np.array(sorted(self.df['fecha'].unique()))
        if len(fechas) < 6:
            self.df_validacion = pd.DataFrame()
            return self.df_validacion
        n_splits = max(1, min(int(n_splits), len(fechas) // 3))
        bloques = np.array_split(fechas, n_splits + 1)
        filas = []
        for i in range(1, len(bloques)):
            fechas_train = np.concatenate(bloques[:i])
            train = self.df[self.df['fecha'].isin(fechas_train)]
            test = self.df[self.df['fecha'].isin(bloques[i])]
            if len(train) < 10 or len(test) < 3:
                continue
            escalador = StandardScaler().fit(train[self.columnas])
            Xtrain = escalador.transform(train[self.columnas])
            Xtest = escalador.transform(test[self.columnas])
            Xfull = np.vstack([Xtrain, Xtest])
            for nombre, k, eps in self._candidatos(len(train)):
                fila = dict(modelo=nombre, k=k, eps=eps, split=i,
                            n_train=len(train), n_test=len(test),
                            fin_train=str(train['fecha'].max().date()),
                            inicio_test=str(test['fecha'].min().date()),
                            inductivo=nombre in ('KMeans', 'GMM'))
                try:
                    m = self._crear(nombre, k, eps)
                    etiquetas_train = self._etiquetar(m, Xtrain)
                    fila.update({f'train_{c}': v for c, v in self._metricas(Xtrain, etiquetas_train).items()})
                    if nombre in ('KMeans', 'GMM'):
                        etiquetas_test = np.asarray(m.predict(Xtest), dtype=int)
                        fila.update({f'test_{c}': v for c, v in self._metricas(Xtest, etiquetas_test).items()})
                        # Comparar la asignación de train al reentrenar incluyendo test.
                        mf = self._crear(nombre, k, eps)
                        self._etiquetar(mf, Xfull)
                        etiquetas_reentreno = np.asarray(mf.predict(Xtrain), dtype=int)
                    else:
                        # Evaluación transductiva, no equivale a predict() sobre test.
                        mf = self._crear(nombre, k, eps)
                        etiquetas_reentreno = self._etiquetar(mf, Xfull)[:len(Xtrain)]
                    fila['estabilidad_ari'] = float(adjusted_rand_score(etiquetas_train, etiquetas_reentreno))
                    fila['error'] = ''
                except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    fila['error'] = str(exc)
                filas.append(fila)
        self.df_validacion = pd.DataFrame(filas)
        self.df_validacion.to_csv(Path(self.ruta_resultado) /
                                  f'validacion_cruzada_{self.identificador}.csv', index=False)
        return self.df_validacion

    def _evaluar_modelo(self, nombre, etiquetas, extra_info, modelo=None):
        datos = self._metricas(self.X, etiquetas)
        self.resultados[nombre] = {
            **extra_info, **datos, 'etiquetas': list(map(int, etiquetas)),
            'modelo_entrenado': modelo,
        }

    def _obtener_modelo_desde_resultado(self, nombre_modelo):
        return self.resultados[nombre_modelo]['modelo_entrenado']

    def _score_normalizado_global(self, ruta_salida=None):
        """Ranking reproducible en los candidatos actuales; no score absoluto de riesgo."""
        ruta_salida = ruta_salida or str(Path(self.ruta_resultado) /
                                         f'score_normalizado_global_{self.identificador}.csv')
        filas = []
        for clave, info in self.resultados.items():
            filas.append({k: v for k, v in info.items()
                          if k not in ('etiquetas', 'modelo_entrenado')} | {'modelo': clave})
        df = pd.DataFrame(filas)
        if df.empty:
            raise ValueError('Ningún algoritmo produjo candidatos evaluables.')
        for col in ('silhouette', 'calinski_harabasz', 'davies_bouldin'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        valido = df['silhouette'].notna() & df['calinski_harabasz'].notna() & df['davies_bouldin'].notna()
        df['elegible'] = valido & (df['tamano_minimo'] >= self.min_cluster_size) & (df['ruido_pct'] <= 30)
        # No se seleccionan modelos sin predict() como clasificadores reutilizables.
        df['desplegable'] = df['algoritmo'].isin(['KMeans', 'GMM'])
        cv = self.df_validacion
        if not cv.empty and 'estabilidad_ari' in cv:
            if 'test_silhouette' not in cv.columns:
                cv = cv.copy()
                cv['test_silhouette'] = np.nan
            agrega = cv.groupby(['modelo', 'k', 'eps'], dropna=False).agg(
                estabilidad_ari=('estabilidad_ari', 'mean'),
                n_folds_validos=('estabilidad_ari', 'count'),
                silhouette_test=('test_silhouette', 'mean'),
            ).reset_index()
            # Se unen con nombres de configuración para impedir confundir k=2 y k=3.
            agrega['clave'] = agrega.apply(lambda r: self._clave(r['modelo'], r['k'], r['eps']), axis=1)
            df = df.merge(agrega[['clave', 'estabilidad_ari', 'n_folds_validos', 'silhouette_test']],
                          left_on='modelo', right_on='clave', how='left').drop(columns=['clave'])
        else:
            df['estabilidad_ari'] = np.nan
            df['n_folds_validos'] = 0
            df['silhouette_test'] = np.nan
        for c in ('silhouette', 'calinski_harabasz', 'davies_bouldin'):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        def normalizar(valores, invertir=False):
            x = pd.to_numeric(valores, errors='coerce')
            valid = x.notna()
            res = pd.Series(0.0, index=x.index)
            if valid.any():
                lo, hi = x[valid].min(), x[valid].max()
                res.loc[valid] = 0.5 if hi == lo else (x[valid] - lo) / (hi - lo)
                if invertir:
                    res.loc[valid] = 1 - res.loc[valid]
            return res
        df['sil_norm'] = normalizar(df['silhouette'])
        df['ch_norm'] = normalizar(df['calinski_harabasz'])
        df['db_norm'] = normalizar(df['davies_bouldin'], invertir=True)
        df['estabilidad_norm'] = ((pd.to_numeric(df['estabilidad_ari'], errors='coerce') + 1) / 2).clip(0, 1).fillna(0)
        df['score_combinado'] = (0.35 * df['sil_norm'] + 0.15 * df['ch_norm'] +
                                 0.15 * df['db_norm'] + 0.35 * df['estabilidad_norm'])
        # Penalizar cobertura deficiente de DBSCAN y grupos desbalanceados.
        df['score_combinado'] *= (1 - df['ruido_pct'] / 100)
        df.loc[~df['elegible'], 'score_combinado'] = np.nan
        df = df.sort_values(['score_combinado', 'modelo'], ascending=[False, True], na_position='last')
        df.to_csv(ruta_salida, index=False)
        return df

    @staticmethod
    def _clave(nombre, k, eps):
        if nombre == 'DBSCAN':
            return f'DBSCAN_eps={float(eps):g}'
        return f'{nombre}_k={int(k)}'

    def _score_combinado(self, sil, ch, db):
        return 0.5 * sil + 0.3 * (ch / 1000) + 0.2 / (db + 1e-6)

    def _score_normalizado(self, sil, ch, db, historial):
        filas = list(historial) + [dict(silhouette=sil, calinski=ch, davies=db)]
        matriz = pd.DataFrame(filas)[['silhouette', 'calinski', 'davies']].astype(float)
        matriz['davies'] = -matriz['davies']
        valores = MinMaxScaler().fit_transform(matriz)
        return float(valores[-1] @ np.array([0.5, 0.3, 0.2]))

    def ejecutar_evaluacion(self):
        self.resultados = {}
        for nombre, k, eps in self._candidatos(len(self.df)):
            clave = self._clave(nombre, k, eps)
            try:
                modelo = self._crear(nombre, k, eps)
                etiquetas = self._etiquetar(modelo, self.X)
                self._evaluar_modelo(clave, etiquetas,
                                     {'algoritmo': nombre, 'k': k, 'eps': eps}, modelo)
            except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                warnings.warn(f'No se pudo evaluar {clave}: {exc}')
        scores = self._score_normalizado_global()
        elegibles = scores[scores['elegible'] & scores['desplegable'] & scores['score_combinado'].notna()]
        if elegibles.empty:
            raise ValueError('No hay modelo inductivo elegible. Revise muestras, variables y criterios.')
        ganador = str(elegibles.iloc[0]['modelo'])
        self.mejor_modelo = (ganador, self.resultados[ganador])
        self.modelo_final = self.resultados[ganador]['modelo_entrenado']
        self._guardar_modelo()
        return scores

    def _guardar_modelo(self):
        if self.mejor_modelo is None:
            raise ValueError('Debe seleccionar un modelo antes de guardar.')
        _guardar_pickle({
            'version': 2,
            'mejor_modelo': (self.mejor_modelo[0], {k: v for k, v in self.mejor_modelo[1].items()
                                                 if k != 'modelo_entrenado'}),
            'modelo_entrenado': self.modelo_final, 'scaler': self.scaler_final,
            'columnas': self.columnas, 'fondo': self.fondo, 'entidad': self.entidad,
            'fecha_entrenamiento': self.fechaTexto,
            'metodo': self.mejor_modelo[1]['algoritmo'],
        }, self.ruta_modelo_final)

    def _perfiles(self, resultado):
        tabla = resultado.groupby('cluster')[self.columnas].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        tabla.columns = ['_'.join(t) for t in tabla.columns]
        tabla.reset_index().to_csv(Path(self.ruta_resultado) /
                                    f'perfiles_clusters_{self.identificador}.csv', index=False)
        dispersion = resultado.groupby('cluster').size().rename('registros').reset_index()
        dispersion.to_csv(Path(self.ruta_resultado) /
                          f'distribucion_clusters_{self.identificador}.csv', index=False)
        # Evolución histórica de pertenencia por período.
        resultado[['fecha', 'entidad', 'fondo', 'cluster']].to_csv(
            Path(self.ruta_resultado) / f'evolucion_clusters_{self.identificador}.csv', index=False)

    def _ejecutar_individual(self):
        """
        Ejecuta el clustering para una sola entidad.

        Genera:
        - Validación temporal.
        - Evaluación de modelos.
        - Selección del mejor modelo.
        - Archivo CSV individual.
        - Perfiles y distribución de clusters.
        """

        self.validacion_cruzada_temporal(n_splits=5)

        self.ejecutar_evaluacion()

        salida = self.df.copy()

        salida["cluster"] = (self.mejor_modelo[1]["etiquetas"])

        salida.to_csv(self.ruta_resultado_csv,index=False)

        self._perfiles(salida)

        return salida

    def ejecutar(self):
        """
        Procesa todas las entidades del fondo indicado.

        Genera:
        - Un CSV de resultados por entidad.
        - Un modelo independiente por entidad.
        - Un CSV global con todas las entidades.
        """

        resultados = []
        errores = {}

        self.evaluadores_por_entidad = {}
        self.modelos_por_entidad = {}
        self.rutas_modelos_por_entidad = {}
        self.rutas_resultados_por_entidad = {}

        # Archivo global.
        ruta_global = (Path(self.ruta_resultado) / f"resultados_{_seguro(self.fondo)}.csv")

        # Evitar conservar resultados globales anteriores.
        ruta_global.unlink(missing_ok=True)

        # Procesar todas las entidades.
        for entidad in self.entidades_disponibles:

            try:
                individual = ClusteringEvaluator(
                    df=self.df_fondo.loc[self.df_fondo["entidad"].astype(str)== entidad],
                            fondo=self.fondo,
                            entidad=entidad,
                            columnas=self.columnas,
                            usar_normalizado=self.usar_normalizado,
                            min_cluster_size=self.min_cluster_size
                        )

                # Compartir carpeta de resultados.
                individual.ruta_resultado = (self.ruta_resultado)

                # Archivo individual.
                individual.ruta_resultado_csv = str(
                    Path(self.ruta_resultado) / (
                        f"resultados_"
                        f"{individual.identificador}.csv"
                    )
                )

                # Ejecutar clustering de esta entidad.
                resultado = (individual._ejecutar_individual())

                resultados.append(resultado)

                # Guardar referencias individuales.
                self.evaluadores_por_entidad[entidad] = individual

                self.modelos_por_entidad[entidad] = individual.mejor_modelo

                self.rutas_modelos_por_entidad[entidad] = individual.ruta_modelo_final

                self.rutas_resultados_por_entidad[entidad] = individual.ruta_resultado_csv

            except Exception as exc:
                errores[entidad] = str(exc)

                warnings.warn(f"No fue posible procesar " f"{entidad}: {exc}")

        self.errores_por_entidad = errores

        if not resultados:
            raise ValueError(
                f"Ninguna entidad del fondo "
                f"{self.fondo} pudo procesarse: "
                f"{errores}"
            )

        # Consolidar resultados.
        global_df = pd.concat(resultados,ignore_index=True)

        global_df = (global_df.sort_values(["entidad", "fecha"]).reset_index(drop=True))

        # Guardar archivo global.
        global_df.to_csv(ruta_global,index=False)

        self.ruta_resultado_global = str(ruta_global)

        self.df_resultado_global = global_df

        # Conservar compatibilidad con los atributos
        # que anteriormente correspondían a una entidad.
        referencia = (
            self.entidad_referencia
            if self.entidad_referencia
            in self.evaluadores_por_entidad
            else next(iter(
                self.evaluadores_por_entidad
            ))
        )

        elegido = (self.evaluadores_por_entidad[referencia])

        self.mejor_modelo = elegido.mejor_modelo
        self.modelo_final = elegido.modelo_final
        self.scaler_final = elegido.scaler_final
        self.ruta_modelo_final = (elegido.ruta_modelo_final)

        self.resultados = elegido.resultados
        self.df_validacion = elegido.df_validacion

        self.resultado = [
            f"Se ejecutó clustering para "
            f"{len(resultados)} entidades "
            f"del fondo {self.fondo}"
        ]

        return global_df

class VisualizadorClusters:
    def __init__(self, df_resultado, columnas_cluster, MostrarGrafico=True):
        self.df_resultado = df_resultado.copy()
        self.columnas_cluster = list(columnas_cluster)
        self.mostrarGrafico = MostrarGrafico
        self.fechaTexto = pd.to_datetime(df_resultado['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DESEQUILIBRIO,
                                           DES_MODULO_CLUSTERING, self.fechaTexto)
        os.makedirs(self.ruta_resultado, exist_ok=True)

    def _finalizar(self, nombre, guardar=True):
        if guardar:
            plt.savefig(os.path.join(self.ruta_resultado, nombre), bbox_inches='tight', dpi=150)
        if self.mostrarGrafico:
            plt.show()
        plt.close()

    def graficar_dispersion(self, guardar=False, ruta=None):
        if len(self.columnas_cluster) < 2:
            raise ValueError('La dispersión requiere dos indicadores.')
        x, y = self.columnas_cluster[:2]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df_resultado, x=x, y=y, hue='cluster', palette='tab10', s=60)
        plt.title('Clustering: dispersión')
        plt.tight_layout()
        if ruta and guardar:
            os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
            plt.savefig(ruta, bbox_inches='tight', dpi=150)
            self._finalizar('dispersion.png', guardar=False)
        else:
            self._finalizar('dispersion.png', guardar=guardar)

    def graficar_comparacion_metricas(self, df_metricas):
        validos = df_metricas.dropna(subset=['score_combinado']).sort_values('score_combinado', ascending=False)
        if validos.empty:
            return
        plt.figure(figsize=(12, 6))
        sns.barplot(data=validos, x='modelo', y='score_combinado', hue='modelo', palette='viridis', legend=False)
        plt.xticks(rotation=45, ha='right')
        plt.title('Comparación de modelos de clustering')
        plt.tight_layout()
        self._finalizar('comparacion_metricas.png')
        validos.to_csv(os.path.join(self.ruta_resultado, 'metricas_modelos.csv'), index=False)

    def graficar_boxplots_por_cluster(self):
        for col in self.columnas_cluster:
            self.df_resultado.boxplot(column=col, by='cluster', figsize=(7, 4))
            plt.suptitle('')
            plt.tight_layout()
            self._finalizar(f'boxplot_{_seguro(col)}.png')

    def graficar_evolucion_por_entidad(self):
        df = self.df_resultado.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        entidades = df['entidad'].dropna().unique()
        if not len(entidades):
            return
        fig, axs = plt.subplots(len(entidades), 1, figsize=(11, max(3, 2 * len(entidades))),
                                sharex=True, squeeze=False)
        for i, entidad in enumerate(entidades):
            sub = df[df['entidad'] == entidad].sort_values('fecha')
            axs[i, 0].step(sub['fecha'], sub['cluster'], where='post')
            axs[i, 0].set_title(str(entidad), loc='left', fontsize=9)
            axs[i, 0].set_ylabel('Cluster')
        fig.tight_layout()
        self._finalizar('evolucion_clusters.png')

    def resumen_clusters(self):
        resumen = self.df_resultado.groupby('cluster').agg({
            'entidad': lambda s: s.mode().iloc[0] if not s.mode().empty else None,
            'fondo': lambda s: s.mode().iloc[0] if not s.mode().empty else None,
            'fecha': ['min', 'max', 'count'],
            **{col: 'mean' for col in self.columnas_cluster},
        })
        resumen.columns = ['_'.join(c).strip('_') for c in resumen.columns.values]
        return resumen.reset_index()

    def distribucion_entidades(self):
        return self.df_resultado.groupby(['cluster', 'entidad']).size().unstack(fill_value=0)

    def resumen_entidad_cluster(self):
        return self.df_resultado.groupby(['entidad', 'cluster']).agg(
            registros=('fecha', 'count'),
            **{col: (col, 'mean') for col in self.columnas_cluster}).reset_index()

    def generar_tabla_comportamiento_reciente(self):
        df = self.df_resultado.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        ultimo = df.sort_values('fecha').groupby('entidad').tail(1)[['entidad', 'cluster']]
        merged = df.merge(ultimo.rename(columns={'cluster': 'cluster_actual'}), on='entidad')
        actual = merged[merged['cluster'] == merged['cluster_actual']]
        return actual.groupby('entidad').agg(
            **{f'{col}_{m}': (col, m) for col in self.columnas_cluster for m in ('mean', 'std', 'min', 'max')},
            registros=('fecha', 'count')).reset_index()

    def interpretar_cambio_cluster(self, fecha_final):
        df = self.df_resultado.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        fecha_final = pd.Timestamp(fecha_final)
        perfiles = df.groupby('cluster')[self.columnas_cluster].mean()
        filas = []
        for entidad, grupo in df.groupby('entidad'):
            historial = grupo[grupo['fecha'] <= fecha_final].sort_values('fecha')
            if len(historial) < 2 or historial['fecha'].iloc[-1] != fecha_final:
                continue
            anterior, actual = historial.iloc[-2], historial.iloc[-1]
            comparacion = '; '.join(f'{col}: {actual[col]:.2f} vs perfil anterior {perfiles.loc[anterior["cluster"], col]:.2f}'
                                   for col in self.columnas_cluster)
            filas.append({'entidad': entidad, 'de_cluster': anterior['cluster'],
                          'a_cluster': actual['cluster'], 'cambios_clave': comparacion})
        return pd.DataFrame(filas, columns=['entidad', 'de_cluster', 'a_cluster', 'cambios_clave'])

    def interpretar_clusters(self):
        partes = []
        for entidad, grupo in self.df_resultado.groupby('entidad'):
            partes.append(f'Entidad: {entidad}')
            for cluster, sub in grupo.groupby('cluster'):
                perfil = sub[self.columnas_cluster].mean()
                partes.append(f'  Cluster {cluster} ({len(sub)} registros): ' +
                              ', '.join(f'{c}={v:.2f}' for c, v in perfil.items()))
        return '\n'.join(partes)

    def interpretar_clusters_multiclase(self):
        perfiles = self.df_resultado.groupby('cluster')[self.columnas_cluster].mean()
        general = self.df_resultado[self.columnas_cluster].mean()
        texto = ['Perfiles relativos de los clústeres (no equivalen a categorías de riesgo):']
        for clave, fila in perfiles.iterrows():
            texto.append(f'Cluster {clave}: ' + '; '.join(
                f'{c} {"por encima" if v > general[c] else "por debajo o igual"} del promedio global ({v:.2f})'
                for c, v in fila.items()))
        return '\n'.join(texto)


def predecir_cluster(df_nuevo, ruta_modelo):
    """Aplica modelo inductivo guardado, preservando filas y valores nulos."""
    with open(ruta_modelo, 'rb') as f:
        datos = pickle.load(f)  # solo cargar artefactos confiables
    modelo, scaler, columnas = datos['modelo_entrenado'], datos['scaler'], datos['columnas']
    if not hasattr(modelo, 'predict'):
        raise ValueError('El modelo guardado no admite inferencia inductiva sobre filas nuevas.')
    faltantes = sorted(set(columnas) - set(df_nuevo.columns))
    if faltantes:
        raise ValueError(f'Faltan indicadores: {faltantes}')
    for campo in ('fondo', 'entidad'):
        if campo in df_nuevo.columns and campo in datos:
            diferentes = df_nuevo[campo].notna() & (df_nuevo[campo].astype(str) != str(datos[campo]))
            if diferentes.any():
                raise ValueError(f'Los datos contienen {campo} diferente al del modelo guardado.')
    salida = df_nuevo.copy()
    X = salida[columnas].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    validas = X.notna().all(axis=1)
    salida['cluster'] = pd.Series(pd.NA, index=salida.index, dtype='Int64')
    if validas.any():
        salida.loc[validas, 'cluster'] = modelo.predict(scaler.transform(X.loc[validas])).astype(int)
    salida['observacion_clustering'] = np.where(validas, '', 'Indicadores faltantes o no numéricos')
    return salida

# Ejemplo de uso

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if __name__ == "__main__":
    from pathlib import Path
    import traceback   
    DATA_ROOT = "data"     
    agente = "Agente de Desequilibrio Financiero"
    modulo = "Clustering de Riesgo Financiero"
    fecha = "2026-08-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    modelo = ClusteringEvaluator(
    df=df,
    fondo="ROP",
    entidad="Ninguna",
    columnas=columnas_cluster,
    usar_normalizado=True
    )

    #df_resultado = evaluator.ejecutar()

    #print("Mejor modelo:", evaluator.mejor_modelo[0])
    #print("Modelo guardado:", evaluator.ruta_modelo_final)

    # Ejecutar el proceso completo
    try:
        resultados = modelo.ejecutar()

        #ruta = Path(modelo.ruta_resultado).resolve()

        print("\n" + "=" * 70)
        print("DIAGNÓSTICO DE GRÁFICOS")
        print("=" * 70)

        print(f"Ruta de resultados: {ruta}")
        print(f"Directorio existe: {ruta.exists()}")

        graficos = list(ruta.glob("cluster_*.png"))

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
    print("\nResultados del clustering:")
    print(modelo.resultados)
"""
    # Crear instancia del visualizador
    visualizador = VisualizadorClusters(df_resultado, columnas_cluster)

    # Tablas de resumen
    resumen = visualizador.resumen_entidad_cluster()
    distribucion = visualizador.distribucion_entidades()

    # Mostrar en consola
    print("\n Resumen por cluster:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    #resumen.head(150)
    #print(resumen)

    print("\n Distribución de entidades por cluster:")
    print(distribucion)    
    ruta = os.path.join(DATA_ROOT, 'resultados', AGENTE_DESEQUILIBRIO, DES_MODULO_CLUSTERING,"dispersion.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)    
    visualizador.graficar_dispersion(guardar=True, ruta=ruta)
    df_scores = evaluator._score_normalizado_global()
    visualizador.graficar_comparacion_metricas(df_scores)
    
    #visualizador.graficar_boxplots_por_cluster()
    visualizador.graficar_evolucion_por_entidad()
    # Asegúrate de que fecha sea datetime
    df_resultado["fecha"] = pd.to_datetime(df_resultado["fecha"])

    # Ordenar
    df_resultado = df_resultado.sort_values(["entidad", "fecha"])

    from datetime import datetime


    #resumen = generar_tabla_comportamiento_reciente(df_resultado, columnas_cluster)
    resumen = visualizador.interpretar_cambio_cluster(datetime.strptime(fecha, "%Y-%m-%d"))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    #resumen.head(150)
    print(resumen)
    resumen = visualizador.interpretar_clusters()
    print(resumen)
    resumen = visualizador.interpretar_clusters_multiclase()
    print(resumen)
    #print(df_resultado.columns.tolist())
    print("Mejor modelo:", evaluator.mejor_modelo[0])
    print("Métricas:")
    for k, v in evaluator.mejor_modelo[1].items():
        if k != "etiquetas":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    # (Opcional) Boxplots si lo usas
    #plt.figure()
    #visualizador.graficar_boxplots_por_cluster()
    # plt.savefig(os.path.join(ruta_graficos, "boxplots_por_cluster.png"), bbox_inches="tight")
    #plt.close()
"""


