import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid
from typing import List, Dict
from scipy.stats import ks_2samp 
import matplotlib.pyplot as plt
import seaborn as sns

class AnomaliaDetector:
#    def __init__(self, columnas: List[str]):
    def __init__(self, columnas: List[str], window_size: int = 24, step_size: int = 6, contamination: float = 0.05, random_state: int = 42):
        self.columnas = columnas
        self.columnas_modelo = columnas
        self.window_size = window_size
        self.step_size = step_size
        self.contamination = contamination
        self.random_state = random_state
        self.modelo = None
        self.modelos_por_entidad = {} 
        self.modelos_por_ventana: Dict[str, List[IsolationForest]] = {}       
        self.scaler = RobustScaler()
        self.resultado = None
        self.parametros = None
        self.score = None
        self.por_entidad = True

    def validacion_cruzada_rolling(self, df: pd.DataFrame, n_splits: int = 5):
        
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha")

        resultados = []
        entidades = df["entidad"].unique() if self.por_entidad else [None]

        for entidad in entidades:
            df_sub = df[df["entidad"] == entidad] if entidad else df
            df_sub = df_sub.dropna(subset=self.columnas_modelo + ["fecha"]).copy()
            df_sub = df_sub.sort_values("fecha").reset_index(drop=True)

            if len(df_sub) < self.window_size + 1:
                continue

            splits = 0
            for start in range(0, len(df_sub) - 2 * self.window_size, self.step_size):
                
                #train_end = start + self.window_size
                #test_end = train_end + self.window_size

                train_end = start + self.window_size
                test_end = train_end + int(0.3 * self.window_size)
                print(f"{start}, {train_end}, {self.window_size}")
                print(f"{train_end}, {test_end}")

                df_train = df_sub.iloc[start:train_end]
                df_test = df_sub.iloc[train_end:test_end]

                #if len(df_test) < self.window_size:
                #    break           

                if len(df_train) < self.window_size:
                    break           

                X_train = self.scaler.fit_transform(df_train[self.columnas_modelo])
                X_test = self.scaler.transform(df_test[self.columnas_modelo])

                modelo = IsolationForest(n_estimators=50, max_samples=0.6, bootstrap=True, contamination="auto", random_state=self.random_state)
                modelo.fit(X_train)

                pred = modelo.predict(X_test)
                score = modelo.decision_function(X_test)
                #---
                score_train = modelo.decision_function(X_train)

                # Métricas de contraste
                media_train = score_train.mean()
                media_test = score.mean()
                std_train = score_train.std()
                std_test = score.std()
                ks_stat, ks_p = ks_2samp(score_train, score)  # Test de Kolmogorov-Smirnov

                print(f"score_train_mean {round(media_train, 4)} score_test_mean: {round(media_test, 4)} score_train_std: {round(std_train, 4)} score_test_std {round(std_test, 4)}  ks_stat: {round(ks_stat, 4)} ks_pvalue: {round(ks_p, 4)}")

                #---
 
                # Ajuste: Umbral dinámico de score (percentil bajo → anómalo)
                threshold = np.percentile(score, 100 * self.contamination)  # p.ej. 5%
                pred_score_based = (score < threshold).astype(int)                
                #print('pred ',pred)
                print(f'procesando vc {entidad} size {start} end {train_end} score {score.mean()}') 
                #print(f"Split {splits+1}, score min: {score.min():.4f}, max: {score.max():.4f}, mean: {score.mean():.4f}")

                #n_anomalias = (pred == -1).sum()
                n_anomalias = pred_score_based.sum()
                resultados.append({
                    "entidad": entidad if entidad else "GLOBAL",
                    "split": splits + 1,
                    #"observacion_test": df_sub.iloc[test_idx]["fecha"],
                    "observaciones_test": len(X_test),                    
                    "anomalias_detectadas": n_anomalias,  
                    "porcentaje_anomalias": round(100 * n_anomalias / len(X_test), 2),                    
                    "score_test": round(score.mean(), 4),
                    "score_min": round(score.min(), 4),  
                    "score_max": round(score.max(), 4),
                    "es_anomalia": int(pred[0] == -1)
                })
                splits += 1

        return pd.DataFrame(resultados).sort_values(by=["entidad", "split"]).reset_index(drop=True)

    def validacion_cruzada(self, df: pd.DataFrame, fecha_col: str = "fecha", n_splits: int = 5):
        df = df.copy()
        df[fecha_col] = pd.to_datetime(df[fecha_col])
        df = df.sort_values(fecha_col)

        resultados = []
        entidades = df["entidad"].unique() if self.por_entidad else [None]

        for entidad in entidades:
            df_sub = df[df["entidad"] == entidad] if entidad else df
            fechas = df_sub[fecha_col].dropna().unique()
            if len(fechas) < n_splits + 2:
                continue
            bloques = np.array_split(fechas, n_splits + 2)

            for i in range(1, n_splits):
                fechas_train = np.concatenate(bloques[:i])
                fechas_val = bloques[i]
                fechas_test = bloques[i + 1]

                df_train = df_sub[df_sub[fecha_col].isin(fechas_train)].dropna(subset=self.columnas_modelo)
                df_val = df_sub[df_sub[fecha_col].isin(fechas_val)].dropna(subset=self.columnas_modelo)
                df_test = df_sub[df_sub[fecha_col].isin(fechas_test)].dropna(subset=self.columnas_modelo)

                try:
                    X_train = self.scaler.fit_transform(df_train[self.columnas_modelo])
                    X_val = self.scaler.transform(df_val[self.columnas_modelo])
                    X_test = self.scaler.transform(df_test[self.columnas_modelo])

                    modelo = IsolationForest(n_estimators=100, contamination=self.contamination, random_state=self.random_state)
                    modelo.fit(X_train)

                    pred_val = modelo.predict(X_val)
                    pred_test = modelo.predict(X_test)
                    score_test = modelo.decision_function(X_test)

                    n_anomalias = (pred_test == -1).sum()
                    resultados.append({
                        "entidad": entidad if entidad else "GLOBAL",
                        "split": i,
                        "observaciones_test": len(X_test),
                        "anomalias_detectadas": n_anomalias,
                        "porcentaje_anomalias": round(n_anomalias / len(X_test) * 100, 2),
                        "score_promedio": round(score_test.mean(), 4)
                    })
                except Exception as e:
                    print(f"Error en entidad {entidad} split {i}: {e}")

        return pd.DataFrame(resultados).sort_values(by=["entidad", "split"]).reset_index(drop=True)

    #def _buscar_mejor_modelo(self, X_train: np.ndarray, X_val: np.ndarray):
    def _buscar_mejor_modelo_rolling(self, X_window: np.ndarray):
        param_grid = {
            'n_estimators': [100, 50],
            'contamination': [0.1, 0.03, 0.05],
            'max_features': [1.0, 0.6]
        }
        mejores = []
        for params in ParameterGrid(param_grid):
            try:
                modelo = IsolationForest(**params, random_state=42).fit(X_window)
                scores = modelo.decision_function(X_window)
                score_medio = scores.mean()
                mejores.append((score_medio, params, modelo))
            except Exception:
                continue

        if not mejores:
            raise ValueError("No se encontró ningún modelo válido.")
        mejores.sort(reverse=True, key=lambda x: x[0])
        return mejores[0][2], mejores[0][1], mejores[0][0]

    def entrenar_y_detectar(self, df: pd.DataFrame):
    #def entrenar_con_rolling_window(self, df: pd.DataFrame):
        resultados = []
        entidades = df["entidad"].unique() if self.por_entidad else [None]

        for entidad in entidades:
            df_sub = df[df["entidad"] == entidad] if entidad else df
            df_sub = df_sub.dropna(subset=self.columnas_modelo + ["fecha"]).copy()
            df_sub["fecha"] = pd.to_datetime(df_sub["fecha"])
            df_sub = df_sub.sort_values("fecha").reset_index(drop=True)

            if len(df_sub) < self.window_size:
                continue
            modelos = []
            df_sub["anomalia"] = 0
            df_sub["score_anomalia"] = np.nan

            for start in range(0, len(df_sub) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                              
                df_window = df_sub.iloc[start:end].copy()
                X = self.scaler.fit_transform(df_window[self.columnas_modelo])

                modelo = IsolationForest(n_estimators=50, contamination=self.contamination, random_state=self.random_state)
                modelo.fit(X)
                etiquetas = modelo.predict(X)
                scores = modelo.decision_function(X)
                 
                df_sub.loc[start:end-1, "anomalia"] = np.maximum(
                    df_sub.loc[start:end-1, "anomalia"].values,
                    (etiquetas == -1).astype(int)
                )

                df_sub.loc[start:end - 1, "score_anomalia"] = scores
                df_sub.loc[start:end - 1, "evaluado"] = 1
                total_procesados = df_sub["evaluado"].sum()
                total_anomalias = ((df_sub["anomalia"] == 1) & (df_sub["evaluado"] == 1)).sum()
                porcentaje_anomalias = round(total_anomalias / total_procesados * 100, 2) if total_procesados > 0 else 0

                print(f"[{entidad}] Registros evaluados: {total_procesados}, Anomalías: {total_anomalias} ({porcentaje_anomalias}%) score {scores.mean()}")

                #print(f'procesando {entidad} size {start} end {end} score {scores.mean()}') 
                modelos.append(modelo)

            resultados.append(df_sub)
            if self.por_entidad:
                self.modelos_por_entidad[entidad] = modelos[-1]
                self.modelos_por_ventana[entidad] = modelos            


        df_final = pd.concat(resultados).sort_index().reset_index(drop=True)
        self.resultado = df_final
        return df_final

    def guardar_modelo(self, ruta: str):
        with open(ruta, "wb") as f:
            pickle.dump({
                "por_entidad": self.por_entidad,
                "modelo": self.modelo,
                "modelos_por_entidad": self.modelos_por_entidad,
                "scaler": self.scaler,
                "parametros": self.parametros,
                "columnas": self.columnas
            }, f)

    def cargar_modelo(self, ruta: str):
        with open(ruta, "rb") as f:
            data = pickle.load(f)
            self.por_entidad = data.get("por_entidad", False)
            self.modelo = data.get("modelo")
            self.modelos_por_entidad = data.get("modelos_por_entidad", {})
            self.scaler = data["scaler"]
            self.parametros = data.get("parametros", {})
            # Soporte flexible para columna usada
            columnas = data.get("columnas")
            if hasattr(self, "columnas_modelo"):
                self.columnas_modelo = columnas
            else:
                self.columnas = columnas      

    def predecir_anomalias(self, df_nuevo: pd.DataFrame):
    #def predecir_rolling_window(self, df_nuevo: pd.DataFrame):
        """
        Aplica detección de anomalías con ventana móvil sobre un conjunto nuevo de datos.

        Requiere que:
            - 'fecha' esté presente y ordenada cronológicamente.
            - Las columnas del modelo estén completas (sin NaN).
            - El objeto tenga definidos: window_size, step_size, columnas_modelo, por_entidad.

        Devuelve:
            - DataFrame con columnas adicionales: 'anomalia' y 'score_anomalia'.
        """
        resultados = []
        columnas = self.columnas_modelo if hasattr(self, "columnas_modelo") else self.columnas
        entidades = df_nuevo["entidad"].unique() if self.por_entidad else [None]

        for entidad in entidades:
            df_sub = df_nuevo[df_nuevo["entidad"] == entidad] if entidad else df_nuevo
            df_sub = df_sub.dropna(subset=columnas + ["fecha"]).copy()
            df_sub["fecha"] = pd.to_datetime(df_sub["fecha"])
            df_sub = df_sub.sort_values("fecha").reset_index(drop=True)

            if len(df_sub) < self.window_size:
                continue

            df_sub["anomalia"] = 0
            df_sub["score_anomalia"] = np.nan

            for start in range(0, len(df_sub) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                df_window = df_sub.iloc[start:end].copy()
                X = self.scaler.fit_transform(df_window[columnas])

                # Entrenar modelo por ventana
                modelo = IsolationForest(
                    n_estimators=100,
                    contamination=self.contamination,
                    random_state=self.random_state
                )
                modelo.fit(X)
                etiquetas = modelo.predict(X)
                scores = modelo.decision_function(X)

                df_sub.loc[start:end-1, "anomalia"] = np.maximum(
                    df_sub.loc[start:end-1, "anomalia"].values,
                    (etiquetas == -1).astype(int)
                )
                df_sub.loc[start:end-1, "score_anomalia"] = scores

            resultados.append(df_sub)

        if resultados:
            return pd.concat(resultados).sort_index().reset_index(drop=True)
        else:
            return pd.DataFrame(columns=df_nuevo.columns.tolist() + ["anomalia", "score_anomalia"])

    def resumen_general(self) -> Dict:
        if self.resultado is None or self.resultado.empty:
            return {
                "mejores_parametros": None,
                "score_promedio": None,
                "total_observaciones": 0,
                "anomalias_detectadas": 0,
                "porcentaje_anomalias": 0.0,
            }

        df = self.resultado
        anom_alias = df[df["anomalia"].isin([-1, 1])]  # Soporte para clásico y rolling

        if self.por_entidad:
            mejores_parametros = {
                entidad: modelo_info.get("parametros", {})
                for entidad, modelo_info in self.modelos_por_entidad.items()
            }
            score_promedio = "por entidad"
        else:
            mejores_parametros = getattr(self, "parametros", "N/A")
            score_promedio = round(self.score, 4) if hasattr(self, "score") else "N/A"

        return {
            "mejores_parametros": mejores_parametros,
            "score_promedio": score_promedio,
            "total_observaciones": len(df),
            "anomalias_detectadas": len(anom_alias),
            "porcentaje_anomalias": round(len(anom_alias) / len(df) * 100, 2),
        }

    def resumen_por_entidad(self) -> pd.DataFrame:
        if self.resultado is None or self.resultado.empty:
            return pd.DataFrame(columns=["anomalia", "normal", "total", "%_anomalias"])
        
        df = self.resultado.copy()

        # Asegurarse de que exista columna 'entidad'
        if "entidad" not in df.columns:
            df["entidad"] = "GLOBAL"

        # Anomalías pueden ser -1 (clasificación clásica) o 1 (rolling)
        df["anomalia_binaria"] = df["anomalia"].apply(lambda x: 1 if x in [-1, 1] else 0)
        df["normal"] = df["anomalia"].apply(lambda x: 0 if x in [-1, 1] else 1)

        resumen = df.groupby("entidad")[["anomalia_binaria", "normal"]].sum()
        resumen = resumen.rename(columns={"anomalia_binaria": "anomalia"})
        resumen["total"] = resumen["anomalia"] + resumen["normal"]
        resumen["%_anomalias"] = round(resumen["anomalia"] / resumen["total"] * 100, 2)
        resumen = resumen.sort_values("%_anomalias", ascending=False)

        resumen.loc["TOTAL"] = resumen.sum(numeric_only=True)
        resumen.loc["TOTAL", "%_anomalias"] = round(
            resumen.loc["TOTAL", "anomalia"] / resumen.loc["TOTAL", "total"] * 100, 2
        )

        return resumen

    def drivers_anomalia(self, columnas: List[str] = None, top_n: int = None) -> pd.DataFrame:
        """
        Identifica variables con mayores diferencias de media entre normales y anómalas.
        Soporta anomalías marcadas como -1 (clásico) o 1 (rolling window).
        """
        if self.resultado is None or self.resultado.empty:
            raise ValueError("No hay resultados disponibles. Ejecuta primero 'entrenar_y_detectar' o 'entrenar_con_rolling'.")

        df = self.resultado.copy()

        if columnas is None:
            columnas = df.select_dtypes(include="number").columns.difference(["anomalia", "score_anomalia"])

        # Asegurar compatibilidad con ambos esquemas de etiquetas
        normales = df[df["anomalia"].isin([0, 1])] if df["anomalia"].isin([0, 1]).any() else df[df["anomalia"] == 1]
        anomalias = df[df["anomalia"].isin([-1, 1])]

        resultados = []
        for col in columnas:
            media_normal = normales[col].mean()
            media_anomalo = anomalias[col].mean()
            diferencia = media_anomalo - media_normal
            resultados.append({
                "variable": col,
                "media_normal": media_normal,
                "media_anomalo": media_anomalo,
                "diferencia": diferencia,
                "abs_diferencia": abs(diferencia)
            })

        df_resultado = pd.DataFrame(resultados).sort_values("abs_diferencia", ascending=False).reset_index(drop=True)
        return df_resultado if not top_n else df_resultado.head(top_n)

class VisualizadorAnomalias:
    def __init__(self, df_resultado, columnas_valores):
        self.df = df_resultado.copy()
        self.columnas_valores = columnas_valores
        self.resultado = None
        self.por_entidad = True     
    """
    def ajusta_dataset(self, por_entidad = True):
        df1 = self.df
        if not por_entidad:
            entidades_codificadas = [col for col in df1.columns if col.startswith("entidad_")]
            df["entidad"] = "ENTIDAD_BASE"  
            for col in entidades_codificadas:
                df1.loc[df1[col] == 1, "entidad"] = col.replace("entidad_", "").replace("_", " ")
            
            df1 = df1.drop(columns=[col for col in df1.columns if col.startswith("entidad_")])

        return df1
    """    
    def graficar_dispersion(self):
        if "anomalia" not in self.df.columns or "score_anomalia" not in self.df.columns:
            raise ValueError("El DataFrame debe tener columnas 'anomalia' y 'score_anomalia'")

        df_ent = self.df
        print('por entidad ',self.por_entidad)
        if not self.por_entidad:
            entidades_codificadas = [col for col in df_ent.columns if col.startswith("entidad_")]
            df_ent["entidad"] = "ENTIDAD_BASE"  
            for col in entidades_codificadas:
                df_ent.loc[df_ent[col] == 1, "entidad"] = col.replace("entidad_", "").replace("_", " ")
            
            df_ent = df_ent.drop(columns=[col for col in df_ent.columns if col.startswith("entidad_")])
        columnas_usar = df_ent.columns.difference(["entidad", "fondo", "fecha"]).tolist()
        x_col=columnas_usar[3] 
        y_col=columnas_usar[4]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_ent, x=x_col, y=y_col,
            hue="anomalia", palette={1: "blue", -1: "red", 0: "green"}, style="anomalia", s=60
        )
        plt.title(f"Anomalías detectadas en {y_col} vs {x_col}")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, labels=["Normal", "Anómalo"], title="Anomalía")        
        plt.tight_layout()
        plt.show()

    def graficar_anomalias_temporales(self, entidad):

        df_ent = self.df
        if not self.por_entidad:
            entidades_codificadas = [col for col in df_ent.columns if col.startswith("entidad_")]
            df_ent["entidad"] = "ENTIDAD_BASE"  
            for col in entidades_codificadas:
                df_ent.loc[df_ent[col] == 1, "entidad"] = col.replace("entidad_", "").replace("_", " ")
            
            df_ent = df_ent.drop(columns=[col for col in df_ent.columns if col.startswith("entidad_")])

        df_ent = self.df[self.df["entidad"] == entidad].copy()
        plt.figure(figsize=(10, 4))
        df_anomalias = df_ent[df_ent["anomalia"].isin([-1, 1])]

        plt.scatter(
            df_anomalias["fecha"],
            df_anomalias["score_anomalia"],
            color="red", label="Anomalía"
        )
        plt.title(f"Anomalías detectadas en la entidad {entidad}")
        plt.xlabel("Fecha")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graficar_anomalias_temporales_todas(self, altura_por_entidad=2):
        df = self.df

        if not por_entidad:
            entidades_codificadas = [col for col in df.columns if col.startswith("entidad_")]
            df["entidad"] = "ENTIDAD_BASE"  
            for col in entidades_codificadas:
                df.loc[df[col] == 1, "entidad"] = col.replace("entidad_", "").replace("_", " ")
            
            df = df.drop(columns=[col for col in df.columns if col.startswith("entidad_")])        
        #print(df.dtypes)

        entidades = df["entidad"].unique()
        n = len(entidades)
        fig, axs = plt.subplots(n, 1, figsize=(12, altura_por_entidad * n), sharex=True)
        palette_entidades = sns.color_palette("tab10", n_colors=len(entidades))
        color_dict = dict(zip(entidades, palette_entidades)) 

        if n == 1:
            axs = [axs]

        for i, entidad in enumerate(entidades):
            ax = axs[i]
            subdf = df[df["entidad"] == entidad].copy()
            ax.plot(subdf["fecha"], subdf["score_anomalia"], label=None,  color=color_dict[entidad])
            subdf_anom = subdf[subdf["anomalia"].isin([-1, 1])]
            ax.scatter(
                subdf_anom["fecha"],
                subdf_anom["score_anomalia"],
                color="red", label="Anomalía", s=20
            )            
            #ax.scatter(
            #    subdf[subdf["anomalia"] == -1]["fecha"],
            #    subdf[subdf["anomalia"] == -1]["score_anomalia"],
            #    color="red", label="Anomalía", s=20
            #)
            ax.set_title(f"{entidad}", fontsize=8, loc="left")
            ax.tick_params(axis='x', labelrotation=45)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_ylabel("Score", fontsize=8)
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            for spine in ax.spines.values():  # ← Oculta todos los bordes del gráfico
                spine.set_visible(False)                      

        plt.xlabel("Fecha")
        plt.tight_layout()
        plt.show()

    def resumen_por_entidad(self):
        df = self.df
        resumen = (
            df.groupby("entidad")["anomalia"]
            .apply(lambda x: x.isin([-1, 1]).sum())
            .reset_index(name="total_anomalias")
        )
        resumen["porcentaje"] = round(resumen["total_anomalias"] / len(self.df) * 100, 2)
        return resumen

    def mapa_calor_variables(self):
        df = self.df
        df_anomalias = df[df["anomalia"].isin([-1, 1])]
        columnas_validas = [col for col in self.columnas_valores if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        df_melt = df_anomalias.melt(id_vars=["entidad"], value_vars=columnas_validas)        
        tabla = df_melt.groupby(["entidad", "variable"]).size().unstack(fill_value=0)
        tabla = tabla.loc[tabla.sum(axis=1).sort_values(ascending=False).index,
                        tabla.sum().sort_values(ascending=False).index]        

        plt.figure(figsize=(10, 6))
        sns.heatmap(tabla, annot=True, fmt="d", cmap="Reds")
        plt.title("Frecuencia de anomalías por variable y entidad")
        plt.tight_layout()
        plt.show()

    def graficar_top_drivers(self, df_drivers: pd.DataFrame, top_n: int = 10):
        df_top = df_drivers.head(top_n)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_top, x="abs_diferencia", y="variable", palette="coolwarm")
        plt.title(f"Top {top_n} variables con mayor diferencia entre normales y anómalos")
        plt.xlabel("Diferencia Absoluta de Media")
        plt.ylabel("Variable")
        plt.tight_layout()
        plt.show()        

# Ejemplo de uso:
if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente = "Agente de Riesgo de Liquidez"
    modulo = "Detección de anomalías de liquidez"
    fecha = "2024-12-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    ruta_modelo = os.path.join(DATA_ROOT, "modelos", "anomalias_general.pkl")    
    df = pd.read_parquet(ruta)
    #df.to_csv('data/resultados/anomalias.csv')

    #print(df.dtypes)
    # Definir columnas excluyendo entidad, fondo y fecha
    #Conteo de fechas por entidad
    conteo_fechas = df.groupby("entidad")["fecha"].nunique().sort_values()
    # Filtrar entidades con más de 24 fechas únicas
    entidades_validas = conteo_fechas[conteo_fechas > 24].index.tolist()    

    # Aplicar filtro al DataFrame
    df = df[df["entidad"].isin(entidades_validas)].copy()    
    #print(conteo_fechas)
    
    por_entidad = True
    # Verificar fechas faltantes por entidad
    #entidades = df["entidad"].unique()
    #for ent in entidades:
    #    fechas_ent = df[df["entidad"] == ent]["fecha"]
    #    print(f"{ent}: fechas únicas = {fechas_ent.nunique()}, rango = {fechas_ent.min()} → {fechas_ent.max()}")    

    # Codificación one-hot de 'entidad'
    if not por_entidad:
        df = pd.get_dummies(df, columns=["entidad"], drop_first=True)
        print(df.head(10))
        #columnas_cluster = df.columns.difference(["fondo", "fecha"]).tolist()    
        columnas_usar = df.columns.difference(["fondo", "fecha"]).tolist()
    else:
        #columnas_cluster = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()    
        columnas_usar = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()
    print(columnas_usar)
    #df.to_csv("data/baseanomalias.csv", index=False) 

    # Inicializar detector con enfoque rolling
    detector = AnomaliaDetector(
        columnas=columnas_usar,
        window_size=36,
        step_size=12,
        contamination=0.03
    )    
    validacion_df = detector.validacion_cruzada(df)
    print("vc ",validacion_df)    
  
    df_anotado = detector.entrenar_y_detectar(df)

    detector.guardar_modelo(ruta_modelo)

    #detector.cargar_modelo(ruta_modelo)
    #predicciones_nuevo = detector.predecir_anomalias(df_anotado)    

    #print("Resumen general:", detector.resumen_general())
    print("Resumen por entidad:")
    print(detector.resumen_por_entidad())
    df_drivers = detector.drivers_anomalia()
    print(df_drivers.head(10))

    # Crear instancia del visualizador
    visualizador = VisualizadorAnomalias(df_resultado=df_anotado, columnas_valores=columnas_usar)
    #visualizador.ajusta_dataset(por_entidad)
    # Mostrar gráfico de dispersión entre dos variables
    visualizador.graficar_dispersion()

    # Mostrar evolución temporal de anomalías para una entidad específica
    #visualizador.graficar_anomalias_temporales(entidad="VIDA PLENA")
    visualizador.graficar_anomalias_temporales_todas(altura_por_entidad=2)

    # Mostrar resumen de anomalías por entidad
    print(visualizador.resumen_por_entidad())

    # Mostrar mapa de calor de frecuencia de anomalías por variable y entidad
    visualizador.mapa_calor_variables()

    drivers = detector.drivers_anomalia()
    visualizador.graficar_top_drivers(drivers)


