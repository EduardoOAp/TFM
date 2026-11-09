import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from itertools import product
## Quitar esta dos lineas despues de pruebas
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
###
from riesgo_agente.utils.config import AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS 
from riesgo_agente.utils.config import DATA_ROOT, TEMPORAL_ROOT, RESULTADOS_ROOT, MODELOS_ROOT

class LSTMAnomalyDetector:
    def __init__(self, df: pd.DataFrame, secuencia=12, tipo_modelo="autoencoder", 
                 epochs=50, batch_size=16, umbral_factor=1.5):
        self.df = df.dropna().copy()        
        self.secuencia = secuencia
        self.tipo_modelo = tipo_modelo
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = 0.3
        self.l2_reg = 0.001
        self.umbral_factor = umbral_factor
        self.scaler = MinMaxScaler()
        self.resultado = []
        self.agente = AGENTE_LIQUIDEZ
        self.modulo = LIQ_MODULO_ANOMALIAS
        self.columnas_numericas = self.df.columns.difference(["entidad", "fondo", "fecha"]).tolist()
        self.mejor_modelo = None
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_LIQUIDEZ, LIQ_MODULO_ANOMALIAS)
        os.makedirs(self.ruta_resultado, exist_ok=True)

    def preparar_datos(self, df, columnas):
        datos_escalados = self.scaler.fit_transform(df[columnas])
        X, y = [], []
        for i in range(len(datos_escalados) - self.secuencia):
            X.append(datos_escalados[i:i + self.secuencia])
            y.append(datos_escalados[i + self.secuencia])
        return np.array(X), np.array(y), df["fecha"].iloc[self.secuencia:].reset_index(drop=True)

    def crear_modelo(self, input_shape):
        # Uso de metrics más apropiado para regresión/reconstrucción
        if self.tipo_modelo == "autoencoder":
            model = Sequential([
                Input(shape=input_shape),
                LSTM(64, activation='relu', return_sequences=False, kernel_regularizer=l2(self.l2_reg)),
                Dropout(self.dropout),
                RepeatVector(input_shape[0]),
                LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
                Dropout(self.dropout),
                TimeDistributed(Dense(input_shape[1], kernel_regularizer=l2(self.l2_reg)))
            ])
        else:
            # Predictor: devuelve vector de features (no secuencia)
            model = Sequential([
                Input(shape=input_shape),
                LSTM(64, activation='relu', kernel_regularizer=l2(self.l2_reg)),
                Dropout(self.dropout),
                Dense(input_shape[1], kernel_regularizer=l2(self.l2_reg))
            ])
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

    def _guardar_modelo(self):
        os.makedirs(self.ruta_modelo, exist_ok=True)
        with open(f"{self.ruta_modelo}/{self.fondo}_modelo.pkl", "wb") as f:
            pickle.dump({
                "mejor_modelo": self.mejor_modelo,
                "modelo_entrenado": self.modelo_final,
                "scaler": self.scaler_final,
                "columnas": self.columnas
            }, f)

    def validacion_cruzada(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        resultados = []
        histories = []
        labels = []        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X[train_idx], (X if self.tipo_modelo == "autoencoder" else y)[train_idx]
            X_test, y_test = X[test_idx], (X if self.tipo_modelo == "autoencoder" else y)[test_idx]

            model = self.crear_modelo((X.shape[1], X.shape[2]))

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=0
            )            
            histories.append(history)
            labels.append(f"Fold {i+1}")

            pred = model.predict(X_test)
            error = mean_squared_error(y_test.reshape(-1, X.shape[2]), pred.reshape(-1, X.shape[2]))
            resultados.append(error)
        return resultados, histories, labels
    
    def graficar_comparacion_varias(self, histories, labels, resultados, titulo, guardar: bool = True):  
        epochs_list = [range(1, len(h.history['accuracy']) + 1) for h in histories]

        plt.figure(figsize=(12, 4))
        plt.suptitle(titulo, fontsize=14)  # Título general

        plt.subplot(1, 3, 1)
        for history, label, ep in zip(histories, labels, epochs_list):
            plt.plot(ep, history.history['val_accuracy'], label=f'{label} - val')
            plt.plot(ep, history.history['accuracy'], linestyle='--', label=f'{label} - train')
        plt.title('Precisión comparativa')
        plt.xlabel('Épocas', fontsize=8)
        plt.ylabel('Accuracy', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=8)

        # Loss
        plt.subplot(1, 3, 2)
        for history, label, ep in zip(histories, labels, epochs_list):
            plt.plot(ep, history.history['val_loss'], label=f'{label} - val')
            plt.plot(ep, history.history['loss'], linestyle='--', label=f'{label} - train')
        plt.title('Pérdida comparativa')
        plt.xlabel('Épocas', fontsize=8)
        plt.ylabel('Loss', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=8)

        # Errores
        ep = list(range(1, len(resultados) + 1))
        plt.subplot(1, 3, 3)
        plt.plot(ep, resultados, marker='o', linestyle='-', label='Error por split')
        plt.title("Error por Fold de VC")
        plt.xlabel("Fold")
        plt.ylabel("Error (MSE)")
        #plt.xticks(range(1, len(errores_por_fold) + 1))
        plt.grid(True)
        plt.tight_layout()
        if guardar:
            os.makedirs(self.ruta_resultado, exist_ok=True)
            plt.savefig(os.path.join(self.ruta_resultado, "grafico_comparacion.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 

    def entrenar_completo(self, X, y):
        model = self.crear_modelo((X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        model.fit(
            X,
            X if self.tipo_modelo == "autoencoder" else y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.25,
            callbacks=[early_stop],
            verbose=0
        )        
        #model.fit(X, X if self.tipo_modelo == "autoencoder" else y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.25, verbose=0)
        return model
    
    def ajustar_hiperparametros(self, X, y, secuencias=[12], epochs_list=[50], batch_sizes=[16],
                                dropouts=[0.3], l2s=[0.001], entidad: str = "", graficar = False):
        mejor_config = None
        mejor_score = float("inf")
        historial_resultados = []  # Lista para guardar todos los resultados        
        for s, e, b, d, r in product(secuencias, epochs_list, batch_sizes, dropouts, l2s):
            self.secuencia = s
            self.epochs = e
            self.batch_size = b
            self.dropout = d
            self.l2_reg = r
            titulo = f"{entidad} secuencia={s}, epochs={e}, batch_size={b} dropout {d} reg {r}"
            #print(f"Evaluando: secuencia={s}, epochs={e}, batch_size={b} dropout {d} reg {r}")
            resultados, histories, labels = self.validacion_cruzada(X, y)
            if graficar:
                self.graficar_comparacion_varias(histories, labels, resultados, titulo, True)
            promedio = np.mean(resultados)

            # Obtener métricas por split
            epochs_efectivos = [len(h.history["loss"]) for h in histories]
            val_mse_final = [h.history["val_loss"][-1] for h in histories]            
            # Guardar cada resultado como un diccionario
            resultado_dict = {
                "entidad": entidad,
                "secuencia": s,
                "epochs": e,
                "batch_size": b,
                "dropout": d,
                "l2": r,
                "mse": promedio,
                "epochs_efectivos_prom": np.mean(epochs_efectivos),
                "val_mse_final_prom": np.mean(val_mse_final),
                "epochs_efectivos_splits": epochs_efectivos,
                "val_mse_final_splits": val_mse_final,  
            }
            historial_resultados.append(resultado_dict)
            if promedio < mejor_score:
                mejor_score = promedio
                mejor_config = (s, e, b, d, r)
            #print(f"Mejor configuración: secuencia={mejor_config[0]}, epochs={mejor_config[1]}, "
            #    f"batch_size={mejor_config[2]}, dropout={mejor_config[3]}, l2={mejor_config[4]} "
            #    f"con MSE={mejor_score:.6f}")
        return mejor_config, mejor_score, historial_resultados    

    def detectar_anomalias(self, model, X, y, fechas, entidad):
        """
        Devuelve:
            resultado_df: DataFrame con columnas ['fecha','score_anomalia','anomalia','entidad','umbral','drivers_idx']
            df_error_vars_long: DataFrame con filas por (fecha, entidad, variable, error, anomalia, score_anomalia)
        """
        pred = model.predict(X)
        # Desescalar X y pred
        n_samples, seq_len, n_vars = X.shape

        # Aplanar para poder aplicar inverse_transform
        X_flat = X.reshape(-1, n_vars)
        pred_flat = pred.reshape(-1, n_vars)

        X_real_flat = self.scaler.inverse_transform(X_flat)
        pred_real_flat = self.scaler.inverse_transform(pred_flat)

        # Volver a forma original
        X_real = X_real_flat.reshape(n_samples, seq_len, n_vars)
        pred_real = pred_real_flat.reshape(n_samples, seq_len, n_vars)


        # Calcular errores por muestra (score global) y por variable
        if self.tipo_modelo == "autoencoder":
            # X shape: (n_samples, seq_len, n_vars) ; pred same shape
            # Error por muestra: promedio por tiempo y variable
            errores_muestra = np.mean(np.square(X_real - pred_real), axis=(1,2))
            errores_variables = np.mean(np.square(X_real - pred_real), axis=1)
        else:
            # Para predictor: X_real no aplica (usa y)
            y_real = self.scaler.inverse_transform(y)
            pred_real = self.scaler.inverse_transform(pred)            
            # Predictor: y shape (n_samples, n_vars), pred shape (n_samples, n_vars)
            errores_muestra = np.mean(np.square(y_real - pred_real), axis=1)
            errores_variables = np.square(y_real - pred_real)

        umbral = errores_muestra.mean() + self.umbral_factor * errores_muestra.std()
        anomalías = errores_muestra > umbral

        # Para cada muestra anómala, identificar drivers por percentil (p.ej >= 75)
        drivers_idx = []
        drivers_names = []
        for i in range(len(errores_muestra)):
            ev = errores_variables[i]
            if anomalías[i]:
                umbral_driver = np.percentile(ev, 75)  # ajustable
                idxs = list(np.where(ev >= umbral_driver)[0])
                drivers_idx.append(idxs)
                drivers_names.append([self.columnas_numericas[j] for j in idxs])
            else:
                drivers_idx.append([])
                drivers_names.append([])

        resultado = pd.DataFrame({
            "fecha": fechas.reset_index(drop=True),
            "score_anomalia": errores_muestra,
            "anomalia": anomalías,
            "entidad": entidad,
            "umbral": umbral,
            "drivers_idx": drivers_idx,
            "drivers": drivers_names
        })

        # Crear DataFrame "wide" de errores por variable, luego "long"
        df_vars = pd.DataFrame(errores_variables, columns=self.columnas_numericas)
        df_vars.insert(0, "fecha", fechas.reset_index(drop=True))
        df_vars.insert(1, "entidad", entidad)
        df_vars["score_anomalia"] = errores_muestra
        df_vars["anomalia"] = anomalías

        # melt -> filas por fecha/entidad/variable
        df_long = df_vars.melt(id_vars=["fecha", "entidad", "score_anomalia", "anomalia"],
                               var_name="variable", value_name="error_variable")
        # Añadir una columna boolean si esa variable fue marcada driver por el método percentil
        # (opcional)
        # Para eso, construir un set por fila
        df_long["es_driver"] = df_long.apply(
            lambda r: r["variable"] in resultado.loc[resultado["fecha"] == r["fecha"], "drivers"].values[0],
            axis=1
        )

        return resultado, df_long
    
    def calcular_drivers_aggregados(self, df_long, top_n=10):
        """
        df_long: DataFrame con columnas ['fecha','entidad','score_anomalia','anomalia','variable','error_variable','es_driver']
        Devuelve DataFrame agregado por variable con métricas útiles.
        """
        # 1) considerar solo filas que son parte de muestras anómalas
        df_anom_vars = df_long[df_long["anomalia"] == True].copy()
        if df_anom_vars.empty:
            return pd.DataFrame(columns=["variable", "count_anomalias", "mean_error", "freq_relativa"])

        total_anom = df_anom_vars["fecha"].nunique()  # o len(df_anom_vars)/n_vars si prefieres otro denominador

        agg = df_anom_vars.groupby("variable").agg(
            count_anomalias = ("fecha", "nunique"),  # cuántas fechas anómalas en las que aparece la variable
            mean_error = ("error_variable", "mean"),
            median_error = ("error_variable", "median"),
            sum_error = ("error_variable", "sum")
        ).reset_index()

        # frecuencia relativa respecto al total de fechas anómalas
        agg["freq_relativa"] = agg["count_anomalias"] / total_anom
        agg = agg.sort_values("count_anomalias", ascending=False).reset_index(drop=True)
        return agg.head(top_n)
    
    def drivers_anomalia(self, grupo_df, resultado_df, columnas=None, top_n=None):
        """
        Calcula los drivers de anomalía usando los valores reales del dataset original.

        grupo_df: dataframe ORIGINAL SIN ESCALAR de la entidad (incluye fecha)
        resultado_df: dataframe de detectar_anomalias()
        columnas: columnas numéricas a evaluar
        top_n: número de variables a devolver
        """

        if resultado_df is None or resultado_df.empty:
            raise ValueError("No hay resultados disponibles para analizar drivers.")

        # Si no se proveen columnas, usar todas las numéricas
        if columnas is None:
            columnas = self.columnas_numericas

        # ---------------------------------------------
        # 1. Alinear grupo_df con la ventana de secuencia
        # ---------------------------------------------
        grupo_df_recortado = grupo_df.iloc[self.secuencia:].reset_index(drop=True)

        # ---------------------------------------------
        # 2. Separar índices de muestras normales y anómalas
        # ---------------------------------------------
        normales_idx = resultado_df.index[resultado_df["anomalia"] == False]
        anomalo_idx = resultado_df.index[resultado_df["anomalia"] == True]

        resultados = []

        # ---------------------------------------------
        # 3. Calcular medias usando valores reales (NO escalados)
        # ---------------------------------------------
        for col in columnas:
            media_normal = grupo_df_recortado.loc[normales_idx, col].mean() if len(normales_idx) > 0 else np.nan
            media_anomalo = grupo_df_recortado.loc[anomalo_idx, col].mean() if len(anomalo_idx) > 0 else np.nan

            diferencia = media_anomalo - media_normal

            resultados.append({
                "variable": col,
                "media_normal": media_normal,
                "media_anomalo": media_anomalo,
                "diferencia": diferencia,
                "abs_diferencia": abs(diferencia)
            })

        df_resultado = pd.DataFrame(resultados).sort_values(
            "abs_diferencia", ascending=False
        ).reset_index(drop=True)

        return df_resultado.head(top_n) if top_n else df_resultado

    def ejecutar(self):
        resumen_anomalias = []
        historial_global = []

        grupo_df_global = []
        resultado_df_global = []

        for entidad in self.df["entidad"].unique():

            # -----------------------------------------------------------
            # 1. Filtrar datos reales por entidad (NO ESCALADOS)
            # -----------------------------------------------------------
            grupo_df = self.df[
                (self.df["entidad"] == entidad) &
                (self.df["fondo"] == "ROP")
            ].copy().reset_index(drop=True)

            grupo_df_global.append(grupo_df.reset_index(drop=True))

            # -----------------------------------------------------------
            # 2. Preparar datos para el modelo (ESCALADOS)
            # -----------------------------------------------------------
            X, y, fechas = self.preparar_datos(grupo_df, self.columnas_numericas)

            # -----------------------------------------------------------
            # 3. Ajuste de hiperparámetros
            # -----------------------------------------------------------
            mejor_config, mejor_score, historial_resultados = self.ajustar_hiperparametros(
                X, y,
                secuencias=[12],
                epochs_list=[50],
                batch_sizes=[16],
                dropouts=[0.2],
                l2s=[0.0001],
                entidad=entidad
            )

            historial_global.extend(historial_resultados)

            # -----------------------------------------------------------
            # 4. Aplicar mejores hiperparámetros
            # -----------------------------------------------------------
            (self.secuencia,
            self.epochs,
            self.batch_size,
            self.dropout,
            self.l2_reg) = mejor_config

            # Volver a preparar X,y por si secuencia cambió
            X, y, fechas = self.preparar_datos(grupo_df, self.columnas_numericas)

            # -----------------------------------------------------------
            # 5. Entrenar modelo final
            # -----------------------------------------------------------
            modelo_entrenado = self.entrenar_completo(X, y)

            # -----------------------------------------------------------
            # 6. Detección de anomalías (incluye errores reales)
            # -----------------------------------------------------------
            df_ent_resultado, df_long_errors = self.detectar_anomalias(
                modelo_entrenado, X, y, fechas, entidad
            )
            
            resultado_df_global.append(df_ent_resultado.reset_index(drop=True))
            resumen_anomalias.append(df_ent_resultado)

            # Guardar errores por variable (reales)
            ruta_long = os.path.join(self.ruta_resultado, f"errors_long_{entidad}.csv")
            df_long_errors.to_csv(ruta_long, index=False)

            # -----------------------------------------------------------
            # 7. Drivers por entidad usando valores REALES
            # -----------------------------------------------------------
            df_drivers_entidad = self.drivers_anomalia(
                grupo_df=grupo_df,
                resultado_df=df_ent_resultado,
                columnas=self.columnas_numericas,
                top_n=10
            )

            df_drivers_entidad.to_csv(
                os.path.join(self.ruta_resultado, f"drivers_entidad_{entidad}.csv"),
                index=False
            )

        grupo_df_global = pd.concat(grupo_df_global, ignore_index=True)
        resultado_df_global = pd.concat(resultado_df_global, ignore_index=True)        
        # -----------------------------------------------------------
        # 7. Drivers por entidad usando valores REALES
        # -----------------------------------------------------------
        df_drivers_global = self.drivers_anomalia(
            grupo_df=grupo_df_global,
            resultado_df=resultado_df_global,
            columnas=self.columnas_numericas,
            top_n=10
        )

        df_drivers_global.to_csv(
            os.path.join(self.ruta_resultado, f"drivers_entidad_global.csv"),
            index=False
        )
        # -----------------------------------------------------------
        # 8. Drivers globales agregados
        # -----------------------------------------------------------
        df_todas = pd.concat(resumen_anomalias, ignore_index=True)

        df_long_global = pd.concat([
            pd.read_csv(os.path.join(self.ruta_resultado, f"errors_long_{ent}.csv"))
            for ent in self.df["entidad"].unique()
        ], ignore_index=True)

        df_drivers_agg = self.calcular_drivers_aggregados(df_long_global, top_n=20)
        df_drivers_agg.to_csv(os.path.join(self.ruta_resultado, "drivers_agrupados.csv"), index=False)

        # -----------------------------------------------------------
        # 9. Historial de hiperparámetros
        # -----------------------------------------------------------
        df_historial = pd.DataFrame(historial_global)
        ruta = os.path.join(self.ruta_resultado, f"historial_hiperparametros_{self.tipo_modelo}.csv")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_historial.to_csv(ruta, index=False)

        # -----------------------------------------------------------
        # 10. Guardar historial de anomalías
        # -----------------------------------------------------------
        df_todas.to_csv(
            os.path.join(self.ruta_resultado, f"historial_anomalias_{self.tipo_modelo}.csv"),
            index=False
        )

        print("[INFO] Historial de hiperparámetros guardado.")
        print("[INFO] Proceso completado correctamente.")

        self.resultado = ["Se ejecutó anomalias de liquidez"]
         

"""
if __name__ == "__main__":
    DATA_ROOT = "data"     
    agente = "Agente de Riesgo de Liquidez"
    modulo = "Detección de anomalías de liquidez"
    fecha = "2024-12-31"
    ruta = os.path.join(DATA_ROOT, "temporal", agente, modulo, fecha+".parquet")
    ruta_modelo = os.path.join(DATA_ROOT, "modelos", "anomalias_general.pkl")    
    df = pd.read_parquet(ruta)

    df = df[df.groupby("entidad")["fecha"].transform("nunique") > 24].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["entidad", "fondo", "fecha"])

    columnas_numericas = df.columns.difference(["entidad", "fondo", "fecha"]).tolist()
    output_dir = os.path.join(DATA_ROOT, "graficos_anomalias")
    resumen_anomalias = []
    historial_global = []
    detector = LSTMAnomalyDetector(tipo_modelo="predictor")  # "autoencoder") #   

    for entidad in df["entidad"].unique():
        grupo_df = df[(df["entidad"] == entidad) & (df["fondo"] == "ROP")].copy().reset_index(drop=True)

        X, y, fechas = detector.preparar_datos(grupo_df, columnas_numericas)

        print(f"[INFO] Buscando mejores hiperparámetros para {entidad}...")
        mejor_config, mejor_score, historial_resultados  = detector.ajustar_hiperparametros(X, y, 
                                                                     secuencias=[12], 
                                                                     epochs_list=[30, 50], 
                                                                     batch_sizes=[8, 16],
                                                                     dropouts=[0.2, 0.4],
                                                                     l2s=[0.0001, 0.001],
                                                                     entidad=entidad                                                                    
                                                                     )

        historial_global.extend(historial_resultados)
        (detector.secuencia, 
        detector.epochs, 
        detector.batch_size, 
        detector.dropout, 
        detector.l2_reg) = mejor_config        
        X, y, fechas = detector.preparar_datos(grupo_df, columnas_numericas)
        modelo_entrenado = detector.entrenar_completo(X, y)
        df_anomalias = detector.detectar_anomalias(modelo_entrenado, X, y, fechas, entidad, output_dir)

        resumen_anomalias.append(df_anomalias)

    df_historial = pd.DataFrame(historial_global)
    df_historial.to_csv("data\\temporal\\historial_hiperparametros_predictor.csv", index=False)
    print("[INFO] Historial de hiperparámetros guardado.")
    df_todas = pd.concat(resumen_anomalias, ignore_index=True)
    altura_por_entidad = 2
    entidades = df_todas["entidad"].unique()
    n = len(entidades)
    fig, axs = plt.subplots(n, 1, figsize=(12, altura_por_entidad * n), sharex=True)
    palette_entidades = sns.color_palette("tab10", n_colors=len(entidades))
    color_dict = dict(zip(entidades, palette_entidades)) 

    for i, entidad in enumerate(entidades):
        ax = axs[i]
        subdf = df_todas[df_todas["entidad"] == entidad].copy()
        ax.plot(subdf["fecha"], subdf["score_anomalia"], label=None,  color=color_dict[entidad])
        subdf_anom = subdf[subdf["anomalia"].isin([-1, 1])]
        ax.scatter(
            subdf_anom["fecha"],
            subdf_anom["score_anomalia"],
            color="red", label="Anomalía", s=20
        )         
        umbral_valor = subdf["umbral"].iloc[0]  # Tomar un único valor de umbral
        ax.axhline(umbral_valor, color='green', linestyle='--', label=f"Umbral = {umbral_valor:.4f}")
        ax.set_title(f"{entidad}", fontsize=8, loc="left")
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        for spine in ax.spines.values():  # ← Oculta todos los bordes del gráfico
            spine.set_visible(False)   

    plt.title("Presencia de anomalías por entidad a lo largo del tiempo")
    plt.xlabel("Fecha")
    plt.tight_layout()
    #grafico_tiempo = os.path.join(output_dir, "anomalias_por_tiempo.png")
    #plt.savefig(grafico_tiempo)
    #plt.close()
    #print(f"[✓] Gráfico temporal de anomalías guardado: {grafico_tiempo}")
    plt.show()

"""