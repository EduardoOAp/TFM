import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
import statsmodels.api as sm
from riesgo_agente.utils.config import DATA_ROOT, AGENTE_FLUJOS, FLU_MODULO_RELACION
from riesgo_agente.utils.config  import TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT

class CausalidadEvaluator:

    def __init__(self, df: pd.DataFrame, fondo: str, columnas: list, max_lag: int = 3):
        self.df_original = df.copy()  # guardás el dataset completo
        self.fondo = fondo
        self.columnas = columnas
        self.max_lag = max_lag
        self.agente = AGENTE_FLUJOS
        self.modulo = FLU_MODULO_RELACION
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_FLUJOS, FLU_MODULO_RELACION, self.fechaTexto)
        os.makedirs(self.ruta_resultado, exist_ok=True)
        self.resultado = []

    def _analizar_entidad(self, df_entidad: pd.DataFrame, entidad: str):
        df_entidad = df_entidad.dropna(subset=self.columnas).copy()
        if df_entidad.empty or df_entidad.shape[0] < 30:
            print(f"Entidad {entidad} no tiene suficientes datos. Se omite.")
            return

        # Correlaciones
        resultados_correlacion = {}
        for i in range(len(self.columnas)):
            for j in range(i + 1, len(self.columnas)):
                col1, col2 = self.columnas[i], self.columnas[j]
                corr, p = pearsonr(df_entidad[col1], df_entidad[col2])
                resultados_correlacion[f"{col1} vs {col2}"] = {"correlacion": round(corr, 4), "p_valor": round(p, 4)}

        df_corr = pd.DataFrame.from_dict(resultados_correlacion, orient="index")
        df_corr.index.name = "par"
        df_corr.to_csv(os.path.join(self.ruta_resultado, f"correlacion_{self.fondo}_{entidad}.csv"))

        # Granger
        resultados_granger = {}
        for i in range(len(self.columnas)):
            for j in range(len(self.columnas)):
                if i != j:
                    col1, col2 = self.columnas[i], self.columnas[j]
                    try:
                        df_gc = df_entidad[[col1, col2]].dropna()
                        resultado = grangercausalitytests(df_gc, maxlag=self.max_lag)
                        resultados_granger[f"{col1} → {col2}"] = {
                            f"lag_{lag}": round(resultado[lag][0]["ssr_ftest"][1], 4)
                            for lag in range(1, self.max_lag + 1)
                        }
                    except Exception as e:
                        resultados_granger[f"{col1} → {col2}"] = {"error": str(e)}

        df_gc = pd.DataFrame.from_dict(resultados_granger, orient="index").reset_index()
        df_gc.rename(columns={"index": "relacion"}, inplace=True)
        df_gc.to_csv(os.path.join(self.ruta_resultado, f"granger_{self.fondo}_{entidad}.csv"), index=False)

        # Regresiones
        resumen_reg = []
        for y in self.columnas:
            X_cols = [col for col in self.columnas if col != y]
            df_model = df_entidad.dropna(subset=[y] + X_cols).copy()
            if df_model.shape[0] < 30:
                continue
            try:
                X = sm.add_constant(df_model[X_cols].astype(float))
                y_vals = df_model[y].astype(float)
                modelo = sm.OLS(y_vals, X).fit()
                resumen_reg.append({
                    "variable_objetivo": y,
                    "r2_ajustado": round(modelo.rsquared_adj, 4),
                    "variables_significativas": sum(p < 0.05 for p in modelo.pvalues[1:]),
                    "AIC": round(modelo.aic, 2),
                    "BIC": round(modelo.bic, 2),
                })
                with open(os.path.join(self.ruta_resultado, f"regresion_{self.fondo}_{entidad}_y_{y}.txt"), "w") as f:
                    f.write(modelo.summary().as_text())
            except Exception as e:
                print(f"Error en regresión {entidad} → {y}: {e}")
            
            from riesgo_agente.agenteFlujoFondos.causalidad_flujofondos import VisualizadorCausalidad
            vis = VisualizadorCausalidad(resultados_correlacion, resultados_granger, self.fechaTexto, entidad=entidad, mostrarGrafico=False)
            vis.graficar_heatmap_correlacion()
            vis.graficar_granger_mas_significativas()
            vis.tabla_rankings()    

        if resumen_reg:
            df_resumen = pd.DataFrame(resumen_reg).sort_values("r2_ajustado", ascending=False)
            df_resumen.to_csv(os.path.join(self.ruta_resultado, f"ranking_regresiones_{self.fondo}_{entidad}.csv"), index=False)

    def ejecutar_regresion_multivariante(self, variable_objetivo: str, variables_explicativas: list):
        df_modelo = self.df.dropna(subset=[variable_objetivo] + variables_explicativas).copy()
        X = df_modelo[variables_explicativas]
        y = df_modelo[variable_objetivo]
        X = sm.add_constant(X)

        modelo = sm.OLS(y, X).fit()
        self.resultados_regresion = modelo.summary().as_text()

        # Guardar resultados en TXT
        ruta_txt = os.path.join(self.ruta_resultado, f"regresion_{self.fondo}_{self.entidad}.txt")
        with open(ruta_txt, "w") as f:
            f.write(self.resultados_regresion)

        return modelo
    
    def ejecutar_regresiones_multiples(self):
        """
        Ejecuta regresión multivariante para cada variable como objetivo,
        usando las demás como explicativas. Guarda resultados individuales y ranking.
        """
        from collections import defaultdict

        resumen_general = []

        for variable_objetivo in self.columnas:
            explicativas = [col for col in self.columnas if col != variable_objetivo]
            df_modelo = self.df.dropna(subset=[variable_objetivo] + explicativas).copy()

            if len(df_modelo) < 30:
                continue  # evitar sobreajuste por muestra muy pequeña

            X = sm.add_constant(df_modelo[explicativas].astype(float))
            y = df_modelo[variable_objetivo].astype(float)

            modelo = sm.OLS(y, X).fit()
            ruta_txt = os.path.join(
                self.ruta_resultado, f"regresion_{self.fondo}_{self.entidad}_y_{variable_objetivo}.txt"
            )
            with open(ruta_txt, "w") as f:
                f.write(modelo.summary().as_text())

            resumen_general.append({
                "variable_objetivo": variable_objetivo,
                "r2_ajustado": round(modelo.rsquared_adj, 4),
                "variables_significativas": sum(p < 0.05 for p in modelo.pvalues[1:]),  # excluye constante
                "AIC": round(modelo.aic, 2),
                "BIC": round(modelo.bic, 2),
            })

        df_resumen = pd.DataFrame(resumen_general).sort_values("r2_ajustado", ascending=False)
        df_resumen.to_csv(os.path.join(self.ruta_resultado, f"ranking_regresiones_{self.fondo}_{self.entidad}.csv"), index=False)

        return df_resumen    

    def calcular_correlaciones(self):
        for i in range(len(self.columnas)):
            for j in range(i + 1, len(self.columnas)):
                col1, col2 = self.columnas[i], self.columnas[j]
                corr, p = pearsonr(self.df[col1], self.df[col2])
                self.resultados_correlacion[f"{col1} vs {col2}"] = {
                    "correlacion": round(corr, 4),
                    "p_valor": round(p, 4)
                }
    
    def calcular_causalidad_granger(self):
        for i in range(len(self.columnas)):
            for j in range(len(self.columnas)):
                if i != j:
                    col1, col2 = self.columnas[i], self.columnas[j]
                    try:
                        df_gc = self.df[[col1, col2]].dropna()
                        resultado = grangercausalitytests(df_gc, maxlag=self.max_lag)
                        self.resultados_granger[f"{col1} → {col2}"] = {
                            f"lag_{lag}": round(resultado[lag][0]["ssr_ftest"][1], 4)
                            for lag in range(1, self.max_lag + 1)
                        }
                    except Exception as e:
                        self.resultados_granger[f"{col1} → {col2}"] = {"error": str(e)}

    def guardar_resultados(self):
        # Correlaciones
        df_corr = pd.DataFrame.from_dict(self.resultados_correlacion, orient="index")
        df_corr.index.name = "par"
        df_corr.to_csv(os.path.join(self.ruta_resultado, f"correlacion_{self.fondo}_{self.entidad}.csv"))

        # Granger
        df_gc = pd.DataFrame.from_dict(self.resultados_granger, orient="index").reset_index()
        df_gc.rename(columns={"index": "relacion"}, inplace=True)
        df_gc.to_csv(os.path.join(self.ruta_resultado, f"granger_{self.fondo}_{self.entidad}.csv"), index=False)

    def ejecutar(self):
        entidades = list(self.df_original["entidad"].dropna().unique())
        entidades.append("global")  # para análisis completo

        for entidad in entidades:
            if entidad == "global":
                df_entidad = self.df_original
            else:
                df_entidad = self.df_original[self.df_original["entidad"] == entidad]

            print(f"\n Procesando entidad: {entidad}")
            self._analizar_entidad(df_entidad, entidad=entidad)
        self.resultado = ["Se ejecutó causalidad de desequilibrio financiero"]

class VisualizadorCausalidad:
    def __init__(self, resultados_correlacion: dict, resultados_granger: dict, fecha: str, entidad: str = "global", mostrarGrafico: bool = False):
        self.corr = resultados_correlacion
        self.granger = resultados_granger
        self.entidad = entidad
        self.fecha = fecha.replace("-", "")
        self.ruta_salida = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_FLUJOS, FLU_MODULO_RELACION, fecha, entidad)
        os.makedirs(self.ruta_salida, exist_ok=True)    
        self.mostrarGrafico = mostrarGrafico

    def _formato_df_correlacion(self):
        return pd.DataFrame.from_dict(self.corr, orient="index").reset_index().rename(
            columns={"index": "variables"}
        )

    def _formato_df_granger_minimo(self):
        filas = []
        for k, v in self.granger.items():
            if isinstance(v, dict) and all(k.startswith("lag_") for k in v.keys()):
                min_lag = min(v, key=v.get)
                filas.append({
                    "relacion": k,
                    "lag_min_p": min_lag,
                    "p_valor": v[min_lag],
                })
        return pd.DataFrame(filas)

    def graficar_heatmap_correlacion(self):
        df = self._formato_df_correlacion()
        df_matriz = df[["variables", "correlacion"]].copy()
        df_matriz[["var1", "var2"]] = df_matriz["variables"].str.split(" vs ", expand=True)
        matriz = df_matriz.pivot(index="var1", columns="var2", values="correlacion")
        matriz = matriz.fillna(matriz.T)  # simetría

        # Create a mask to hide the lower triangle.
        upper_triangle_mask = np.triu(np.ones_like(matriz, dtype=bool))   # np.tril

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matriz,
            mask=upper_triangle_mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
        )
        #sns.heatmap(matriz, annot=True, cmap="coolwarm", center=0)
        plt.title(f"Mapa de calor de correlaciones – {self.entidad}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.ruta_salida, "correlacion_heatmap.png"))
        if not self.mostrarGrafico:
            plt.close()
        plt.close()

    def graficar_granger_mas_significativas(self, umbral=0.05):
        df_granger = self._formato_df_granger_minimo()
        df_filtrado = df_granger[df_granger["p_valor"] < umbral].sort_values("p_valor")

        if df_filtrado.empty:
            print("No hay relaciones significativas por Granger con p <", umbral)
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_filtrado, x="p_valor", y="relacion", hue="relacion", palette="crest", legend=False)
        plt.xlabel("P-valor (Granger)")
        plt.ylabel("Relación (X → Y)")
        plt.title(f"Relaciones causales significativas – {self.entidad} (p < {umbral})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.ruta_salida, "granger_significativas.png"))
        if not self.mostrarGrafico:
            plt.close()
        plt.close()

    def tabla_rankings(self, top_n=10):
        df_corr = self._formato_df_correlacion().copy()
        df_corr["abs_correlacion"] = df_corr["correlacion"].abs()
        print("\nTop correlaciones absolutas:")
        print(df_corr.sort_values("abs_correlacion", ascending=False).head(top_n)[["variables", "correlacion", "p_valor"]])

        df_granger = self._formato_df_granger_minimo()
        print("\nRelaciones causales más significativas (Granger):")
        print(df_granger.sort_values("p_valor").head(top_n))    