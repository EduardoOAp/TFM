import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
## Quitar esta dos lineas despues de pruebas
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
###
from riesgo_agente.utils.config import AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS 
from riesgo_agente.utils.config import DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_mensuales, unir_serie_y_predicciones


class ClusteringEvaluator:
    def __init__(self, df: pd.DataFrame, fondo: str, entidad: str, columnas: list, usar_normalizado: bool = False):
        self.df = df.dropna(subset=columnas).copy()
        self.fondo = fondo
        self.entidad = entidad
        self.columnas = columnas
        self.scaler_final = StandardScaler().fit(self.df[columnas])
        self.X = StandardScaler().fit_transform(self.df[columnas])
        self.resultados = {}
        self.mejor_modelo = None
        self.modelo_final = None
        self.agente = AGENTE_DIVERSIFICACION
        self.modulo = DIV_MODULO_ANALISIS
        self.usar_normalizado = usar_normalizado
        self.historial_scores = []  # Para acumulación si se usa score normalizado
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS)
        os.makedirs(self.ruta_resultado, exist_ok=True)

# Evaluación de múltiples métodos de clustering con puntuación combinada

    def validacion_cruzada_temporal(self, n_splits=5):
        if not hasattr(self, 'resultados'):
            self.resultados = {}
        self.df["fecha"] = pd.to_datetime(self.df["fecha"])
        fechas_ordenadas = sorted(self.df["fecha"].unique())
        cortes = np.array_split(fechas_ordenadas, n_splits)
        #resultados_cv = []

        for i in range(n_splits):
            fecha_train = pd.Series(np.concatenate(cortes[:i+1]))
            df_train = self.df[self.df["fecha"].isin(fecha_train)].copy()

            if len(df_train) < 10:
                continue

            X_train = df_train[self.columnas].dropna()
            if X_train.shape[0] < 10:
                continue
            X_train = StandardScaler().fit_transform(X_train)
            
            for k in range(2, 6):
                try:
                    modelo_kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
                    modelo_gmm = GaussianMixture(n_components=k, random_state=42).fit(X_train)
                    modelo_hier = AgglomerativeClustering(n_clusters=k).fit(X_train)
                    etiquetas_kmeans = modelo_kmeans.labels_
                    etiquetas_gmm = modelo_gmm.predict(X_train)
                    etiquetas_hier = modelo_hier.labels_

                    for nombre, etiquetas in zip(
                        ["KMeans", "GMM", "Jerarquico"],
                        [etiquetas_kmeans, etiquetas_gmm, etiquetas_hier]
                    ):
                        sil = silhouette_score(X_train, etiquetas)
                        ch = calinski_harabasz_score(X_train, etiquetas)
                        db = davies_bouldin_score(X_train, etiquetas)
                        #score = self._score_combinado(sil, ch, db)

                        clave = f"modelo_{nombre}__k_{k}_split_{i+1}"
                        self.resultados[clave] = {
                            "split":i+1,
                            "eps": None,
                            "k": k,
                            "silhouette": sil,
                            "calinski_harabasz": ch,
                            "davies_bouldin": db,
                            "etiquetas": etiquetas.tolist(),
                            "modelo_entrenado": nombre
                        }
                        #resultados_cv.append({"split": i+1, "modelo": nombre, "k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}) #, "score": score})
                except Exception:
                    continue

            for eps in [0.5, 1.0, 1.5]:
                try:
                    modelo_dbscan = DBSCAN(eps=eps, min_samples=5).fit(X_train)
                    if len(set(modelo_dbscan.labels_)) > 1:# and -1 not in set(modelo_dbscan.labels_):
                        etiquetas = modelo_dbscan.labels_
                        sil = silhouette_score(X_train, etiquetas)
                        ch = calinski_harabasz_score(X_train, etiquetas)
                        db = davies_bouldin_score(X_train, etiquetas)
                        #score = self._score_combinado(sil, ch, db)
                        clave = f"modelo_DBSCAN_eps_{eps}_split_{i+1}"
                        self.resultados[clave] = {
                            "split":i+1,
                            "eps": eps,
                            "k": None,
                            "silhouette": sil,
                            "calinski_harabasz": ch,
                            "davies_bouldin": db,
                            "etiquetas": etiquetas.tolist(),
                            "modelo_entrenado": "DBSCAN"
                        }                                           
                        #resultados_cv.append({"split": i+1, "modelo": "DBSCAN", "eps": eps, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}) #, "score": score})
                except Exception:
                    continue
            #print(f"resultados {self.resultados}")
        ruta_salida = os.path.join(self.ruta_resultado,"score_normalizado_global_vc.csv")
        self._score_normalizado_global(ruta_salida=ruta_salida)
        return pd.DataFrame(self.resultados) #resultados_cv)

    def _score_combinado(self, sil, ch, db):
        return 0.5 * sil + 0.3 * (ch / 1000) + 0.2 * (1 / (db + 1e-6))
    
    def _score_normalizado(self, sil, ch, db, historial):
        """
        Score combinado normalizado dinámicamente en función del historial de modelos evaluados.
        
        :param sil: silhouette actual
        :param ch: calinski-harabasz actual
        :param db: davies-bouldin actual
        :param historial: lista de dicts con 'silhouette', 'calinski', 'davies'
        :return: float - score normalizado ponderado
        """
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        sils = [h["silhouette"] for h in historial] + [sil]
        chs = [h["calinski"] for h in historial] + [ch]
        dbs_inv = [1 / (h["davies"] + 1e-6) for h in historial] + [1 / (db + 1e-6)]

        sil_norm = MinMaxScaler().fit_transform(np.array(sils).reshape(-1, 1))[-1][0]
        ch_norm = MinMaxScaler().fit_transform(np.array(chs).reshape(-1, 1))[-1][0]
        db_norm = MinMaxScaler().fit_transform(np.array(dbs_inv).reshape(-1, 1))[-1][0]

        return 0.5 * sil_norm + 0.3 * ch_norm + 0.2 * db_norm
    
    def _score_normalizado_global(self, ruta_salida="data/resultados/score_normalizado_global.csv"):
        """
        Recalcula los scores combinados normalizados y guarda el ranking como CSV.

        :param ruta_salida: Ruta donde se guardará el archivo CSV con los resultados.
        :return: DataFrame ordenado por score_combinado (descendente).
        """
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        import numpy as np
        import os

        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

        # Convertir resultados a DataFrame
        df = pd.DataFrame([
            {"modelo": nombre, **datos}
            for nombre, datos in self.resultados.items()
        ])
        #print(df.head(10))
        # Invertir Davies-Bouldin
        df["davies_inv"] = 1 / (df["davies_bouldin"] + 1e-6)

        # Normalización con MinMax
        scaler = MinMaxScaler()
        df[["sil_norm", "ch_norm", "db_norm"]] = scaler.fit_transform(
            df[["silhouette", "calinski_harabasz", "davies_inv"]]
        )

        # Calcular score ponderado
        df["score_combinado"] = (
            0.5 * df["sil_norm"] +
            0.3 * df["ch_norm"] +
            0.2 * df["db_norm"]
        )

        df_ordenado = df.sort_values("score_combinado", ascending=False).reset_index(drop=True)

        # Guardar CSV
        df_ordenado.to_csv(ruta_salida, index=False)

        #print(f"Score normalizado guardado en: {ruta_salida}")
        return df_ordenado

    def _evaluar_modelo(self, nombre, etiquetas, extra_info, modelo=None):
        sil = silhouette_score(self.X, etiquetas)
        ch = calinski_harabasz_score(self.X, etiquetas)
        db = davies_bouldin_score(self.X, etiquetas)

        #usar_normalizado = self.usar_normalizado and len(self.historial_scores) >= 3

        #if usar_normalizado:
        #    score = self._score_normalizado(sil, ch, db, self.historial_scores)
        #else:
        #    score = self._score_combinado(sil, ch, db)

        self.resultados[nombre] = {
            **extra_info,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            #"score_combinado": score,
            "etiquetas": etiquetas.tolist(),
            "modelo_entrenado": modelo
        }

    def _obtener_modelo_desde_resultado(self, nombre_modelo):
        config = self.resultados[nombre_modelo]
        if config["algoritmo"] == "KMeans":
            return KMeans(n_clusters=config["k"], random_state=42).fit(self.X)
        elif config["algoritmo"] == "GMM":
            modelo = GaussianMixture(n_components=config["k"], random_state=42).fit(self.X)
            return modelo
        elif config["algoritmo"] == "Jerarquico":
            return AgglomerativeClustering(n_clusters=config["k"]).fit(self.X)
        elif config["algoritmo"] == "DBSCAN":
            return DBSCAN(eps=config["eps"], min_samples=5).fit(self.X)
        else:
            raise ValueError("Algoritmo no soportado")        

    def ejecutar_evaluacion(self):
        self.resultados = {}
        for k in range(2, 6):
            modelo = KMeans(n_clusters=k, random_state=42).fit(self.X)
            self._evaluar_modelo(f"KMeans_k={k}", modelo.labels_, {"algoritmo": "KMeans", "k": k}, modelo)

        for eps in [0.5, 1.0, 1.5]:
            modelo = DBSCAN(eps=eps, min_samples=5).fit(self.X)
            if len(set(modelo.labels_)) > 1:# and -1 not in set(modelo.labels_):
                self._evaluar_modelo(f"DBSCAN_eps={eps}", modelo.labels_, {"algoritmo": "DBSCAN", "eps": eps}, modelo)

        for k in range(2, 5):
            modelo = AgglomerativeClustering(n_clusters=k).fit(self.X)
            self._evaluar_modelo(f"Jerarquico_k={k}", modelo.labels_, {"algoritmo": "Jerarquico", "k": k}, modelo)

        for k in range(2, 5):
            modelo = GaussianMixture(n_components=k, random_state=42).fit(self.X)
            etiquetas = modelo.predict(self.X)
            self._evaluar_modelo(f"GMM_k={k}", etiquetas, {"algoritmo": "GMM", "k": k}, modelo)
        print(f'resultados {self.resultados}')
        df_scores = self._score_normalizado_global()
        nombre_mejor = df_scores.iloc[0]["modelo"]
        self.mejor_modelo = (nombre_mejor, self.resultados[nombre_mejor])
        self.modelo_final = self.resultados[nombre_mejor]["modelo_entrenado"]        
        #self.mejor_modelo = sorted(self.resultados.items(), key=lambda x: x[1]["score_combinado"], reverse=True)[0]
        #nombre_mejor = self.mejor_modelo[0]
        #self.modelo_final = self.resultados[nombre_mejor]["modelo_entrenado"]
        self._guardar_modelo()   

    def _guardar_modelo(self):
        os.makedirs(self.ruta_modelo, exist_ok=True)
        with open(f"{self.ruta_modelo}/{self.fondo}_modelo.pkl", "wb") as f:
            pickle.dump({
                "mejor_modelo": self.mejor_modelo,
                "modelo_entrenado": self.modelo_final,
                "scaler": self.scaler_final,
                "columnas": self.columnas
            }, f)

    def ejecutar(self):
        # Validación cruzada temporal
        df_cv = self.validacion_cruzada_temporal(n_splits=5)
        self.df_validacion = df_cv
        self.ejecutar_evaluacion()

        # Guardar resultado (opcional)
        df_cv.to_csv(os.path.join(self.ruta_resultado, f"validacion_cruzada_{self.fondo.replace(' ', '_')}.csv"), index=False)
        df_resultado = self.df.copy()
        df_resultado["cluster"] = self.mejor_modelo[1]["etiquetas"]
        df_resultado.to_csv(os.path.join(self.ruta_resultado, f"resultados_{self.fondo.replace(' ', '_')}.csv"), index=False)
        #self.resultado = df_resultado.to_dict(orient="records")
        self.resultado = ["Se ejecutó clustering de desequilibrio financiero"]

class VisualizadorClusters:
    def __init__(self, df_resultado, columnas_cluster, MostrarGrafico: bool = True):
        self.df_resultado = df_resultado.copy()
        self.columnas_cluster = columnas_cluster
        self.fechaTexto = pd.to_datetime(df_resultado['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_ANALISIS, self.fechaTexto)
        self.mostrarGrafico = MostrarGrafico

    def graficar_dispersion(self, guardar=False):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df_resultado, x=self.columnas_cluster[1], y=self.columnas_cluster[2], hue="cluster", palette="tab10", s=60)
        plt.title("Clustering: Dispersión")
        plt.tight_layout()
        ruta = os.path.join(self.ruta_resultado,"dispersion.png")
        if guardar and ruta:
            os.makedirs(os.path.dirname(ruta), exist_ok=True) 
            plt.savefig(ruta, bbox_inches="tight") 
        if not self.mostrarGrafico:
            plt.close()
        plt.show()

    def graficar_comparacion_metricas(self, df_metricas):
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_metricas = df_metricas.sort_values("score_combinado", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_metricas, x="modelo", y="score_combinado", hue="modelo", palette="viridis", legend=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Score Combinado (Normalizado)")
        plt.xlabel("Modelo")
        plt.title("Comparación de modelos de clustering")
        plt.tight_layout()
        ruta = os.path.join(self.ruta_resultado,"comparacion_metricas.png")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)          
        plt.savefig(ruta, bbox_inches="tight")
        if not self.mostrarGrafico:
            plt.close()
        plt.show()

        df_metricas = df_metricas.sort_values("score_combinado", ascending=False)
        ruta = os.path.join(self.ruta_resultado,"metricas_modelos.csv")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_metricas.to_csv(ruta, index=False)        
        #print(df_metricas)
        ranking_df = df_metricas.copy()
        ranking_df["silhouette_Rank"] = ranking_df["silhouette"].rank(ascending=False).astype(int)
        ranking_df["calinski_Rank"] = ranking_df["calinski_harabasz"].rank(ascending=False).astype(int)
        ranking_df["davies_Rank"] = ranking_df["davies_bouldin"].rank(ascending=True).astype(int)
        final_ranking = ranking_df[['modelo',"silhouette_Rank","calinski_Rank","davies_Rank"]]
        if self.mostrarGrafico:
            print(final_ranking)

    def graficar_boxplots_por_cluster(self):
        self.df_resultado.head(10)

        for col in self.columnas_cluster:
            self.df_resultado.boxplot(column=col, by="cluster", figsize=(3,3))
            plt.suptitle("")
            plt.xlabel('cluster')
            plt.ylabel(col)
            if not self.mostrarGrafico:
                plt.close()
            plt.show()

    def graficar_evolucion_por_entidad(self):
        self.df_resultado["fecha"] = pd.to_datetime(self.df_resultado["fecha"])
        self.df_resultado = self.df_resultado.sort_values(["entidad", "fecha"])
        entidades = self.df_resultado["entidad"].unique()
        n = len(entidades)

        fig, axs = plt.subplots(n, 1, figsize=(10, 1.5 * n), sharex=True, sharey=True)
        palette_entidades = sns.color_palette("tab10", n_colors=len(entidades))
        color_dict = dict(zip(entidades, palette_entidades))

        for i, entidad in enumerate(entidades):
            ax = axs[i]
            subdf = self.df_resultado[self.df_resultado["entidad"] == entidad]
            ax.plot(subdf["fecha"], subdf["cluster"], marker='o', linestyle='-', label=entidad, color=color_dict[entidad])
            ax.grid(axis='y', visible=False)
            ax.set_title(f"{entidad}", fontsize=8, loc="left")
            ax.set_ylabel("Cluster")
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.suptitle("Evolución de Clusters por Entidad", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        ruta = os.path.join(self.ruta_resultado,"evolucionClusters.png")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)          
        plt.savefig(ruta, bbox_inches="tight")        
        if not self.mostrarGrafico:
            plt.close()
        plt.show()

    def resumen_clusters(self):
        resumen = self.df_resultado.groupby("cluster").agg({
            "entidad": lambda x: x.mode()[0],
            "fondo": lambda x: x.mode()[0],
            "fecha": ["min", "max", "count"],
            **{col: "mean" for col in self.columnas_cluster}
        })
        resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]
        return resumen.reset_index()

    def distribucion_entidades(self): #df_resultado):
        return self.df_resultado.groupby(["cluster", "entidad"]).size().unstack(fill_value=0)   

    def resumen_entidad_cluster(self): #df_resultado, columnas_cluster):
        return (
            self.df_resultado
            .groupby(["entidad", "cluster"])
            .agg({
                "fecha": "count",
                **{col: "mean" for col in self.columnas_cluster}
            })
            .rename(columns={"fecha": "registros"})
            .reset_index()
        )         

    def generar_tabla_comportamiento_reciente(self): #df_resultado, columnas_cluster):
        # Asegurarse que fecha es datetime
        df_resultado = self.df_resultado
        df_resultado["fecha"] = pd.to_datetime(df_resultado["fecha"])

        # Obtener la última fecha por entidad
        ultima_fecha = df_resultado.groupby("entidad")["fecha"].max().reset_index()
        df_ultimo = pd.merge(df_resultado, ultima_fecha, on=["entidad", "fecha"], how="inner")

        # Obtener el cluster más reciente por entidad
        entidad_cluster_actual = df_ultimo[["entidad", "cluster"]].rename(columns={"cluster": "cluster_actual"})

        # Unir para filtrar histórico de cada entidad con su cluster actual
        df_join = pd.merge(df_resultado, entidad_cluster_actual, on="entidad", how="inner")
        df_historico_cluster_actual = df_join[df_join["cluster"] == df_join["cluster_actual"]]

        # Generar resumen estadístico por entidad
        resumen = df_historico_cluster_actual.groupby("entidad").agg({
            "fecha": ["min", "max", "count"],
            **{col: ["mean", "std", "min", "max"] for col in self.columnas_cluster}
        })

        resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]
        resumen = resumen.reset_index()

        #import ace_tools as tools; tools.display_dataframe_to_user(name="Resumen histórico por entidad y cluster actual", dataframe=resumen)
        return resumen

    def interpretar_cambio_cluster(self, fecha_final): #df_resultado, fecha_final, columnas_cluster):
        entidad_cluster_actual = self.df_resultado[ self.df_resultado["fecha"] == fecha_final][["entidad", "cluster"]]
        promedio_por_cluster = self.df_resultado.groupby("cluster")[ self.columnas_cluster].mean().round(2)

        interpretaciones = []
        # Cluster anterior más frecuente (excluyendo el actual)
        for _, fila in entidad_cluster_actual.iterrows():
            entidad = fila["entidad"]
            cluster_actual = fila["cluster"]
            historico =  self.df_resultado[self.df_resultado["entidad"] == entidad]

            # Cluster anterior más frecuente (excluyendo el actual)
            cluster_hist = historico[historico["fecha"] < fecha_final]["cluster"].mode()
            if not cluster_hist.empty:
                cluster_anterior = cluster_hist.iloc[0]
                if cluster_anterior != cluster_actual:
                    #print(f"cluster {cluster_anterior} {cluster_actual} {entidad}")                
                    # Interpretar cambio
                    cambios = []
                    for col in self.columnas_cluster:
                        valor_actual = historico[historico["fecha"] == fecha_final][col].values[0]
                        promedio_anterior = promedio_por_cluster.loc[cluster_anterior, col]
                        promedio_actual = promedio_por_cluster.loc[cluster_actual, col]
                        direccion = "↑" if promedio_actual > promedio_anterior else "↓"
                        cambios.append(f"{col}: {direccion} ({valor_actual:.2f} vs {promedio_anterior:.2f})")

                    interpretaciones.append({
                        "entidad": entidad,
                        "de_cluster": cluster_anterior,
                        "a_cluster": cluster_actual,
                        "cambios_clave": "; ".join(cambios)
                    })
                else:
                    cluster_anterior = cluster_hist.iloc[0]
                    cambios = []
                    for col in self.columnas_cluster:
                        valor_actual = historico[historico["fecha"] == fecha_final][col].values[0]
                        promedio_anterior = promedio_por_cluster.loc[cluster_anterior, col]
                        promedio_actual = promedio_por_cluster.loc[cluster_actual, col]
                        direccion = "=" 
                        cambios.append(f"{col}: {direccion} ({valor_actual:.2f} vs {promedio_actual:.2f})")                

                    interpretaciones.append({
                        "entidad": entidad,
                        "de_cluster": cluster_anterior,
                        "a_cluster": cluster_actual,
                        "cambios_clave": "; ".join(cambios)
                    })

        return pd.DataFrame(interpretaciones)

    def interpretar_clusters(self):
        df = self.df_resultado.copy()
        interpretaciones = []

        for entidad in df['entidad'].unique():
            df_ent = df[df['entidad'] == entidad]

            # Agrupación por cluster para obtener promedios por entidad
            df_ent_grouped = df_ent.groupby("cluster").mean(numeric_only=True)

            clusters = sorted(df_ent_grouped.index.unique())
            texto = f"Entidad: {entidad}\n"

            for cluster_id in clusters:
                desc = df_ent_grouped.loc[cluster_id]
                texto += f"- Cluster {cluster_id}:\n"
                
                # Iterar sobre las columnas numéricas del cluster
                for col, val in desc.items():
                    texto += f"    - {col.replace('_', ' ').capitalize()}: {val:.2f}\n"

            interpretaciones.append(texto)

        return "\n\n".join(interpretaciones)

    def interpretar_clusters_multiclase(self):
        df = self.df_resultado.copy()
        interpretaciones = []

        resumen = df.groupby("cluster").mean(numeric_only=True)
        resumen = resumen[self.columnas_cluster].round(2)
        promedio_general = resumen.mean()

        # Diccionario opcional para nombres amigables y frases para comparaciones
        frases_comparacion = {
            "indice_riesgo_liquidez": ("riesgo de liquidez", "mayor riesgo de liquidez", "menor riesgo de liquidez"),
            "indice_liquidez_portafolio": ("liquidez del portafolio", "liquidez superior", "liquidez inferior"),
            "rendimiento_rentabilidad": ("rendimiento", "rendimiento mayor", "rendimiento menor"),
            "volatilidad_valorcuota_30d": ("volatilidad valor cuota", "mayor volatilidad en valor cuota", "volatilidad más estable en valor cuota"),
            "volatilidad_reciente_activo_neto": ("variabilidad activo neto", "mayor variabilidad en activo neto", "menor variabilidad en activo neto"),
            "concentracion_activos": ("concentración de activos", "mayor concentración en activos", "menor concentración en activos"),
            "herfindahl_hirschman_local": ("HH local", "portafolio diversificado", "portafolio menos diversificado"),
            "indice_diversificacion_shannon":("indice diversificación shannon", "activos distribuidos de forma equilibrada", "menos variedad en la distribución")
            # Agrega más si quieres
        }

        # Interpretación individual por cluster (valores)
        for cluster_id, fila in resumen.iterrows():
            texto = f"**Cluster {cluster_id}**:\n"
            for col in resumen.columns:
                valor = fila[col]
                nombre_amigable = frases_comparacion.get(col, (col.replace('_', ' '), None, None))[0]
                texto += f"- {nombre_amigable.capitalize()}: {valor}\n"
            interpretaciones.append(texto)

        # Comparaciones entre clusters (relativas al promedio general)
        comparaciones = "**Comparaciones entre clusters:**\n\n"
        for cluster_id, fila in resumen.iterrows():
            comparaciones += f"- El *Cluster {cluster_id}* se distingue por tener:\n"
            for col in resumen.columns:
                valor_cluster = fila[col]
                valor_promedio = promedio_general[col]
                nombre_amigable, frase_alta, frase_baja = frases_comparacion.get(
                    col, (col.replace('_', ' '), "más alto", "más bajo")
                )

                if valor_cluster > valor_promedio:
                    comparaciones += f"  - **{frase_alta}** en la variable {nombre_amigable}.\n"
                else:
                    comparaciones += f"  - **{frase_baja}** en la variable {nombre_amigable}.\n"
            comparaciones += "\n"

        return "\n".join(interpretaciones) + "\n" + comparaciones



def predecir_cluster(df_nuevo, ruta_modelo):
    with open(ruta_modelo, "rb") as f:
        datos = pickle.load(f)

    modelo = datos["modelo_entrenado"]
    scaler = datos["scaler"]
    columnas = datos["columnas"]

    df_proc = df_nuevo.dropna(subset=columnas).copy()
    X_nuevo = scaler.transform(df_proc[columnas])

    if hasattr(modelo, "predict"):
        etiquetas = modelo.predict(X_nuevo)
    elif hasattr(modelo, "fit_predict"):  # ej. AgglomerativeClustering no tiene .predict
        etiquetas = modelo.fit_predict(X_nuevo)
    else:
        raise ValueError("El modelo no soporta predicción directa")

    df_proc["cluster"] = etiquetas
    return df_proc

# Ejemplo de uso
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente = "Agente de Desequilibrio Financiero"
    modulo = "Clustering de Riesgo Financiero"
    fecha = "2024-12-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    df = pd.read_parquet(ruta)
    print(df.dtypes)

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

    evaluator = ClusteringEvaluator(df, fondo="ROP", entidad="VIDA PLENA", columnas=columnas_cluster, usar_normalizado=True, agente=agente, modulo=modulo)
    #df_cv = evaluator.validacion_cruzada_temporal(n_splits=5)
    #df_cv.to_csv("data/resultados/validacion_cruzada_temporal.csv", index=False)    
    #evaluator.validacion_cruzada_temporal(n_splits=5)
    evaluator.ejecutar()
    # Añadir etiquetas al dataframe original
    df_resultado = evaluator.df.copy()
    df_resultado["cluster"] = evaluator.mejor_modelo[1]["etiquetas"]

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
"""
## Ejemplo de uso para predición con el modelo
"""
#ruta_modelo = "data/modelos/ROP_VIDA PLENA_clustering.pkl"
#df_nuevo = pd.read_parquet("data/nuevos_datos.parquet")
#df_con_clusters = predecir_cluster(df_nuevo, ruta_modelo)
#print(df_con_clusters.head())
