import os
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
from riesgo_agente.utils.helpers import completar_fechas_mensuales, unir_serie_y_predicciones


class LSTMAnomalyDetector:
    def __init__(self, df: pd.DataFrame, secuencia=12, tipo_modelo="autoencoder", epochs=50, batch_size=16, umbral_factor=1.5):
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
            model = Sequential([
                Input(shape=input_shape),
                LSTM(64, activation='relu', kernel_regularizer=l2(self.l2_reg)),
                Dropout(self.dropout),
                Dense(input_shape[1], kernel_regularizer=l2(self.l2_reg))
            ])   
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
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
            print(f"Evaluando: secuencia={s}, epochs={e}, batch_size={b} dropout {d} reg {r}")
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
                "val_mse_final_splits": val_mse_final                
            }
            historial_resultados.append(resultado_dict)
            if promedio < mejor_score:
                mejor_score = promedio
                mejor_config = (s, e, b, d, r)
            print(f"Mejor configuración: secuencia={mejor_config[0]}, epochs={mejor_config[1]}, "
                f"batch_size={mejor_config[2]}, dropout={mejor_config[3]}, l2={mejor_config[4]} "
                f"con MSE={mejor_score:.6f}")
        return mejor_config, mejor_score, historial_resultados    

    def detectar_anomalias(self, model, X, y, fechas, entidad):
        pred = model.predict(X)
        errores = np.mean(np.square((X if self.tipo_modelo == "autoencoder" else y) - pred), axis=(1, 2) if self.tipo_modelo == "autoencoder" else 1)
        umbral = errores.mean() + self.umbral_factor * errores.std()
        anomalías = errores > umbral
        return pd.DataFrame({"fecha": fechas, "score_anomalia": errores, "anomalia": anomalías, "entidad": entidad, "umbral": umbral})

    def ejecutar(self):
        resumen_anomalias = []
        historial_global = []
        #print(f"{DATA_ROOT}, {TEMPORAL_ROOT}, {self.agente}, {self.modulo}, graficos_anomalias")
        #output_dir = os.path.join(self.ruta_modelo, "graficos_anomalias")
        #detector = LSTMAnomalyDetector(tipo_modelo="predictor")  # "autoencoder") #   
        #self.columnas_numericas = self.df.columns.difference(["entidad", "fondo", "fecha"]).tolist()

        for entidad in self.df["entidad"].unique():
            grupo_df = self.df[(self.df["entidad"] == entidad) & (self.df["fondo"] == "ROP")].copy().reset_index(drop=True)

            X, y, fechas = self.preparar_datos(grupo_df, self.columnas_numericas)

            print(f"[INFO] Buscando mejores hiperparámetros para {entidad}...")
            mejor_config, mejor_score, historial_resultados  = self.ajustar_hiperparametros(X, y, 
                                                                        secuencias=[12], 
                                                                        epochs_list=[30, 50], 
                                                                        batch_sizes=[8, 16],
                                                                        dropouts=[0.2, 0.4],
                                                                        l2s=[0.0001, 0.001],
                                                                        entidad=entidad                                                                    
                                                                        )

            historial_global.extend(historial_resultados)
            (self.secuencia, 
            self.epochs, 
            self.batch_size, 
            self.dropout, 
            self.l2_reg) = mejor_config        
            X, y, fechas = self.preparar_datos(grupo_df, self.columnas_numericas)
            modelo_entrenado = self.entrenar_completo(X, y)
            df_anomalias = self.detectar_anomalias(modelo_entrenado, X, y, fechas, entidad)
            resumen_anomalias.append(df_anomalias)
    
        df_historial = pd.DataFrame(historial_global)
        ruta = os.path.join(self.ruta_resultado, f"historial_hiperparametros_{self.tipo_modelo}.csv")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_historial.to_csv(ruta, index=False)
        print("[INFO] Historial de hiperparámetros guardado.")   
        df_todas = pd.concat(resumen_anomalias, ignore_index=True)
        ruta = os.path.join(self.ruta_resultado, f"historial_anomalias_{self.tipo_modelo}.csv")
        df_todas.to_csv(ruta, index=False)
        
        #self.resultado = df_historial.to_dict(orient="records")
        
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