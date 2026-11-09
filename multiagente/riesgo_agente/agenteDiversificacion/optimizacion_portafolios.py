import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from riesgo_agente.utils.config  import DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION
from riesgo_agente.utils.config  import TEMPORAL_ROOT, RESULTADOS_ROOT
from riesgo_agente.utils.helpers import completar_fechas_mensuales
# Se necesita un optimizador, se puede usar scipy
from scipy.optimize import minimize
# Ignorar advertencias para una salida más limpia
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class OptimizadorPortafolioLiquido:
    def __init__(self, df: pd.DataFrame, columna_objetivo: str, columnas_predictoras: List[str], horizonte: int = 6):
        self.df = df.copy()
        self.columna_objetivo = columna_objetivo
        self.columnas_predictoras = columnas_predictoras
        self.columnas_instrumentos = [col for col in self.columnas_predictoras if col.startswith("monto_")]
        self.entidades = df['entidad'].unique()        
        self.horizonte = horizonte
        #self.modelo = None
        self.modelos_entidades = {} # Nuevo diccionario para almacenar modelos y scalers por entidad
        self.scalers_entidades = {}  # Guarda los scalers por entidad
        self.mejor_modelo_global = None # Renombrado para mayor claridad        
        self.resultados = {}
        self.agente = AGENTE_DIVERSIFICACION
        self.modulo = DIV_MODULO_OPTIMIZACION
        self.fechaTexto = pd.to_datetime(df['fecha'], errors='coerce').max().strftime('%Y%m%d')
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, self.fechaTexto)
        self.ruta_modelo = os.path.join(DATA_ROOT, MODELOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION)
        self.ruta_temporal = os.path.join(DATA_ROOT, TEMPORAL_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION)
        os.makedirs(self.ruta_resultado, exist_ok=True)    

    def preparar_datos(self, df: pd.DataFrame):
        # Verifica que todas las predictoras existan
        faltan = [c for c in self.columnas_predictoras if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan columnas predictoras en df: {faltan}")

        # Limpieza y orden
        df = df.dropna(subset=self.columnas_predictoras + [self.columna_objetivo]).sort_values("fecha")

        # Asegura numérico y sin NaN en X, y
        X = df[self.columnas_predictoras].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        y = df[self.columna_objetivo].astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    
    def _agregar_columnas_peso(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Asegura numérico en montos y sin NaN
        df.loc[:, self.columnas_instrumentos] = (
            df[self.columnas_instrumentos]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
        )
        total = df[self.columnas_instrumentos].sum(axis=1)
        total = total.replace(0, np.nan)  # evita división por 0

        # Construye columnas peso_*
        self.columnas_peso = []
        for col in self.columnas_instrumentos:
            suf = col.replace("monto_", "")
            pcol = f"peso_{suf}"
            df[pcol] = (df[col] / total).fillna(0.0)
            self.columnas_peso.append(pcol)
        print('col 2', df.columns)
        return df
    

    def entrenar_mlp(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            raise ValueError("DataFrame vacío para entrenamiento.")
        if len(df) < 2: # Necesitas al menos dos puntos para escalar o entrenar
             raise ValueError("No hay suficientes datos para entrenar el modelo.")
        
        for entidad in self.entidades:
            df_entidad = df[df['entidad'] == entidad].copy()
            if len(df_entidad.dropna()) > 10:
                try:
                    X, y, X_scaler = self.preparar_datos(df_entidad)
                    param_grid = {
                        'hidden_layer_sizes': [(32,), (64,), (64, 32)],
                        'activation': ['relu', 'tanh'],
                        'solver': ['adam'],
                        'alpha': [0.0001, 0.001],
                        'learning_rate': ['constant', 'adaptive']
                    }

                    tscv = TimeSeriesSplit(n_splits=3)
                    search = RandomizedSearchCV(
                        MLPRegressor(
                            max_iter=15000,
                            early_stopping=True,
                            validation_fraction=0.1,
                            random_state=42
                        ),
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )

                    search.fit(X, y)
                    modelo_entrenado = search.best_estimator_
                    #self.modelo = search.best_estimator_

                    y_pred = modelo_entrenado.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    accuracy_personalizada = self.calcular_accuracy_regresion(y, y_pred)

                    self.modelos_entidades[entidad] = {
                        'modelo': modelo_entrenado,
                        'mse': round(mse, 4),
                        'r2': round(r2, 4),
                        'accuracy_10pct': accuracy_personalizada,
                        'mejores_parametros': search.best_params_   
                        }
                    self.scalers_entidades[entidad] = {
                      "scaler_X": X_scaler
                    }
                except Exception as e:
                    print(f"Error al entrenar para '{entidad}': {e}")
            else:
                print(f"No hay suficientes datos para entrenar para '{entidad}'.")

        return #resultados_entidades       
    
    def _funcion_objetivo_optimizacion(self, w, modelo, scaler, escenario_base, columnas_instrumentos):
        # Trabaja directamente con pesos (proporciones)
        esc = escenario_base.copy()

        w = np.asarray(w, dtype=float)
        if not np.isfinite(w).all():
            return 1e12
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return 1e12
        w = w / s  # normalización robusta

        # Asegura que existan columnas peso_*
        if not hasattr(self, "columnas_peso"):
            esc = self._agregar_columnas_peso(esc)  # crea peso_* si hiciera falta

        # Escribe los pesos w en las columnas peso_*
        for i, col_monto in enumerate(columnas_instrumentos):
            suf = col_monto.replace("monto_", "")
            col_peso = f"peso_{suf}"
            if col_peso not in esc.columns:
                esc[col_peso] = 0.0
            esc.loc[:, col_peso] = w[i]

        # (Opcional) mantener consistencia de montos si luego los usas fuera
        esc.loc[:, columnas_instrumentos] = (
            esc[columnas_instrumentos].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        )
        total_actual = float(esc[columnas_instrumentos].sum(axis=1).iloc[0])
        if np.isfinite(total_actual) and total_actual > 0:
            for i, col_monto in enumerate(columnas_instrumentos):
                esc.loc[:, col_monto] = w[i] * total_actual

        # Predicción
        try:
            X = esc[self.columnas_predictoras].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            X_scaled = scaler.transform(X)
            y_hat = float(np.asarray(modelo.predict(X_scaled)).squeeze())
        except Exception:
            return 1e12

        if not np.isfinite(y_hat):
            return 1e12
        return -y_hat  # maximizar y_hat
    
    """
    def predecir_escenarios(self, df_futuro: pd.DataFrame, entidad: str) -> List[float]:
        if entidad not in self.modelos_entidades or "modelo" not in self.modelos_entidades[entidad]:
            raise ValueError(f"No se encontró un modelo entrenado para la entidad '{entidad}'.")
    
        modelo = self.modelos_entidades[entidad]["modelo"]
        scaler_X = self.scalers_entidades[entidad]["scaler_X"]
        #scaler_y = self.modelos_entidades[entidad]["scaler_y"] # Si usaste un scaler para 'y'
        #print('Columnas ',self.columnas_predictoras)
        X_futuro = df_futuro[self.columnas_predictoras].values
        X_futuro_scaled = scaler_X.transform(X_futuro)
        
        predicciones_scaled = modelo.predict(X_futuro_scaled)
        # Invertir la escala de las predicciones si la variable objetivo fue escalada
        #predicciones_original_scale = scaler_y.inverse_transform(predicciones_scaled.reshape(-1, 1)).flatten().tolist()

        return predicciones_scaled
    """    
    """    
    def _preparar_datos_para_optimizacion(self, df_ent: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        datos_prediccion = {}
        # Obtener los nombres de las columnas que representan tipos de instrumentos
        # Aquí asumo que las columnas predictoras son las mismas para cada instrumento
        #columnas_instrumentos = [col for col in self.columnas_predictoras if col.startswith("monto_")]
        
        # Generar un DataFrame futuro (ej. la última fila) para cada instrumento
        # Puedes adaptar esta lógica para generar escenarios más complejos si lo necesitas
        df_futuro_base = df_ent.tail(1).copy()
        #print('columnas ',columnas_instrumentos)
        for instrumento in self.columnas_instrumentos:
            # Puedes usar el mismo DataFrame base para la predicción,
            # o generar uno específico para cada instrumento
            datos_prediccion[instrumento] = df_futuro_base
        print('Montos ',datos_prediccion)    
        return datos_prediccion
    """    
    def _optimizar_composicion(self, df: pd.DataFrame, carteraPorEntidad: bool = False ):
        """
        Optimiza la composición del portafolio para cada entidad usando su modelo entrenado,
        incluyendo:
        - Penalización por turnover (suave por defecto; duro opcional)
        - Penalización L2 hacia un ancla (cartera actual o política)
        - Penalización de entropía (diversificación)
        - Desglose de componentes del objetivo
        - Multi-start para robustez
        """
        resultados_optimizacion = {}

        # =========================
        # CONFIG (ajusta a tu gusto)
        # =========================
        # Penalizaciones
        lam_l2       = 0.25     # fuerza hacia ancla (w0 o política)
        tau_entropy  = 0.12     # empuja a pesos balanceados (w_i log w_i)
        gamma_tx     = 0.15     # costo de turnover (suave)
        eps_sm       = 1e-6     # suavizado para L1: sqrt((w-w0)^2 + eps_sm)

        # Turnover duro (opc.): si True, impone ∑|w - w0| <= T_max con variables auxiliares
        use_hard_turnover = False
        T_max             = 0.25  # máximo turnover permitido (25% del portafolio)

        # Multi-start
        K_starts = 6
        seed_ms  = 1234

        # Grupos e índices (ajusta nombres si difieren en tu data)
        col2idx = {c: i for i, c in enumerate(self.columnas_instrumentos)}

        try:
            idx_acc    = col2idx['monto_acciones']
            idx_vc     = col2idx['monto_capital_de_riesgo']
            idx_deuda  = col2idx['monto_deuda_estandarizada']
            idx_pc     = col2idx['monto_papel_comercial']
            idx_part   = col2idx['monto_participaciones']
            idx_recomp = col2idx['monto_recompras']
        except KeyError as e:
            raise KeyError(f"Columna faltante en columnas_instrumentos: {e}")

        grupos = {
            'deuda':     [idx_deuda, idx_pc],
            'variable':  [idx_acc, idx_part],
            'alternos':  [idx_vc, idx_recomp],
        }

        # RANGOS de política por grupo (intersecará con factibilidad por instrumento)
        # Ajusta aquí tu política estratégica (puede ser por entidad si lo deseas)
        politica = {
            'deuda':    (0.40, 0.50),
            'variable': (0.30,  0.4),
            'alternos': (0.05,  0.15),
        }

        # Cotas por instrumento (si quieres pisos/techos por instrumento, ponlos aquí)
        # Si no, deja 0..1 y lo controlan los rangos por grupo.
        piso_inst_default = 0.0
        techo_inst_default = 1.0

        rng = np.random.default_rng(seed_ms)

        def proyectar_simplex(w):
            # Proyección simple al simplex (sum=1, w>=0)
            w = np.maximum(w, 0.0)
            s = w.sum()
            return w / s if s > 0 else np.ones_like(w) / len(w)

        col_fecha = 'fecha'  # <- ajusta si tu columna se llama distinto
        ultima_fecha = df[col_fecha].max()
        ### Este escenario usa la estructura de toda la cartera del ROP
        escenario_base_global = (
            df.loc[df[col_fecha] == ultima_fecha, self.columnas_instrumentos]
            .sum()                     # suma sobre todas las entidades
            .to_frame().T              # deja una fila
        )
        #print('Base global ',escenario_base_global,)
        for entidad, datosModelo in self.modelos_entidades.items():
            ### En este caso se basa en la estructura de la entidad
            if carteraPorEntidad:
                df_entidad = df[df['entidad'] == entidad].copy()
                escenario_base_global = (
                df_entidad.loc[df_entidad[col_fecha] == ultima_fecha, self.columnas_instrumentos]
                .sum()                     # suma sobre todas las entidades
                .to_frame().T              # deja una fila
                    )            
                #print('B   ase entidad ',escenario_base_global,)
            ###
            modelo = datosModelo['modelo']
            escenario_base = escenario_base_global #df_entidad.tail(1)

            if escenario_base.empty:
                print(f"No hay datos base para la entidad '{entidad}'. Se omite la optimización.")
                continue

            scaler = self.scalers_entidades[entidad]['scaler_X']
            cols   = self.columnas_instrumentos
            n      = len(cols)

            # ====== obtener cartera base y pesos actuales w0 ======
            # Suma base (evita FutureWarning: usa iloc[0])
            base_inv = float(escenario_base[cols].sum(axis=1).iloc[0])
            if base_inv <= 0:
                # si no hay inversión, usa ancla uniforme
                w0 = np.ones(n) / n
            else:
                w0 = escenario_base[cols].iloc[0].to_numpy(dtype=float) / base_inv
                # limpia pequeñas negativas por ruido numérico
                w0 = np.clip(w0, 0.0, 1.0)
                w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n

            # ====== cotas por instrumento ======
            lb_inst = np.full(n, piso_inst_default, dtype=float)
            ub_inst = np.full(n, techo_inst_default, dtype=float)

            # ====== factibilidad por grupo (intersección con política) ======
            # suma de lb/ub por grupo (factible por instrumento)
            sum_lb_deuda = float(lb_inst[grupos['deuda']].sum())
            sum_ub_deuda = float(ub_inst[grupos['deuda']].sum())
            sum_lb_var   = float(lb_inst[grupos['variable']].sum())
            sum_ub_var   = float(ub_inst[grupos['variable']].sum())
            sum_lb_alt   = float(lb_inst[grupos['alternos']].sum())
            sum_ub_alt   = float(ub_inst[grupos['alternos']].sum())

            #print(f"[{entidad}] sum lb={sum(lb_inst):.4f}, sum ub={sum(ub_inst):.4f}")
            def interseccion(rpol, lo_feas, hi_feas, tag):
                lo_eff = max(rpol[0], lo_feas - 1e-9)  # pequeño relax por numérica
                hi_eff = min(rpol[1], hi_feas + 1e-9)
                #print(f"[{entidad}] {tag} feasible [{lo_feas:.3f}, {hi_feas:.3f}] -> rango ({lo_eff}, {hi_eff})")
                if lo_eff > hi_eff + 1e-12:
                    raise ValueError(f"[{entidad}] Rango inviable para grupo {tag}: ({lo_eff:.4f} > {hi_eff:.4f})")
                return (lo_eff, hi_eff)

            rango_deuda = interseccion(politica['deuda'],    sum_lb_deuda, sum_ub_deuda, 'deuda')
            rango_var   = interseccion(politica['variable'], sum_lb_var,   sum_ub_var,   'variable')
            rango_alt   = interseccion(politica['alternos'], sum_lb_alt,   sum_ub_alt,   'alternos')

            # ====== restricciones ======
            def cons_sum_w(w):
                return np.sum(w) - 1.0

            def cons_grupo_min(w, idxs, lo):
                return np.sum(w[idxs]) - lo

            def cons_grupo_max(w, idxs, hi):
                return hi - np.sum(w[idxs])

            # Si usamos turnover duro, el vector de decisión es [w (n), t (n)].
            # Si no, solo w (n).
            if use_hard_turnover:
                dim = 2 * n
                # índices
                def split(z):
                    return z[:n], z[n:]

                # restricciones SLSQP con variables auxiliares t_i >= |w_i - w0_i|
                def cons_sum_w_aug(z):
                    w, t = split(z)
                    return np.sum(w) - 1.0

                ineqs = []
                # grupo mínimos/máximos
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['deuda'],    lo=rango_deuda[0]:   np.sum(split(z)[0][idxs]) - lo}]
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['deuda'],    hi=rango_deuda[1]:   hi - np.sum(split(z)[0][idxs])}]
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['variable'], lo=rango_var[0]:     np.sum(split(z)[0][idxs]) - lo}]
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['variable'], hi=rango_var[1]:     hi - np.sum(split(z)[0][idxs])}]
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['alternos'], lo=rango_alt[0]:     np.sum(split(z)[0][idxs]) - lo}]
                ineqs += [{'type': 'ineq', 'fun': lambda z, idxs=grupos['alternos'], hi=rango_alt[1]:     hi - np.sum(split(z)[0][idxs])}]
                # turnover: t_i >= w_i - w0_i  y  t_i >= -(w_i - w0_i)
                for i in range(n):
                    ineqs.append({'type': 'ineq', 'fun': lambda z, i=i: split(z)[1][i] - (split(z)[0][i] - w0[i])})
                    ineqs.append({'type': 'ineq', 'fun': lambda z, i=i: split(z)[1][i] + (split(z)[0][i] - w0[i])})
                # sum(t) <= T_max
                ineqs.append({'type': 'ineq', 'fun': lambda z: T_max - np.sum(split(z)[1])})

                cons = [{'type': 'eq', 'fun': cons_sum_w_aug}] + ineqs

                # bounds: w in [lb, ub], t in [0, 1]
                lb = lb_inst.copy()
                ub = ub_inst.copy()
                bounds = [(lb[i], ub[i]) for i in range(n)] + [(0.0, 1.0) for _ in range(n)]

            else:
                # vector de decisión: solo w
                cons = [
                    {'type': 'eq',  'fun': cons_sum_w},
                    {'type': 'ineq','fun': lambda w: cons_grupo_min(w, grupos['deuda'],    rango_deuda[0])},
                    {'type': 'ineq','fun': lambda w: cons_grupo_max(w, grupos['deuda'],    rango_deuda[1])},
                    {'type': 'ineq','fun': lambda w: cons_grupo_min(w, grupos['variable'], rango_var[0])},
                    {'type': 'ineq','fun': lambda w: cons_grupo_max(w, grupos['variable'], rango_var[1])},
                    {'type': 'ineq','fun': lambda w: cons_grupo_min(w, grupos['alternos'], rango_alt[0])},
                    {'type': 'ineq','fun': lambda w: cons_grupo_max(w, grupos['alternos'], rango_alt[1])},
                ]
                bounds = [(lb_inst[i], ub_inst[i]) for i in range(n)]

            # ====== componentes del objetivo ======
            def comp_obj(w_only):
                # base: tu función actual (devuelve escalar; minimizamos)
                base = float(np.asarray(
                    self._funcion_objetivo_optimizacion(w_only, modelo, scaler, escenario_base, cols)
                ).squeeze())

                # L2 hacia w0 (o cambia w0 por w_policy si quieres anclar a una política)
                l2 = lam_l2 * float(np.sum((w_only - w0) ** 2))

                # entropía (empieza a +inf si algún w=0 -> usa clip)
                w_clip = np.clip(w_only, 1e-12, 1.0)
                ent = tau_entropy * float(np.sum(w_clip * np.log(w_clip)))

                # costo de turnover (suave)
                tx = gamma_tx * float(np.sum(np.sqrt((w_only - w0) ** 2 + eps_sm)))

                return base, l2, ent, tx

            def objetivo_total(z):
                if use_hard_turnover:
                    w, t = z[:n], z[n:]
                    base, l2, ent, tx_soft = comp_obj(w)
                    # con turnover duro, no añadimos costo de turnover (o pon un gamma_tx muy bajo si quieres ambos)
                    total = base + l2 + ent
                    return total if np.isfinite(total) else 1e12
                else:
                    w = z
                    base, l2, ent, tx = comp_obj(w)
                    total = base + l2 + ent + tx
                    return total if np.isfinite(total) else 1e12

            # ====== inicializaciones (multi-start) ======
            def factible_x0():
                # arranque desde w0 (ajustado a bounds)
                w_init = np.clip(w0, [b[0] for b in bounds[:n]], [b[1] for b in bounds[:n]])
                w_init = proyectar_simplex(w_init)

                # empuja a cumplir rangos de grupo suavemente (pequeño procedimiento greedy)
                def empuja_grupo(w, idxs, lo, hi):
                    s = w[idxs].sum()
                    if s < lo:
                        falta = lo - s
                        # redistribuye desde fuera del grupo hacia el grupo
                        fuera = np.setdiff1d(np.arange(n), idxs)
                        # quita proporcional fuera y añade dentro proporcional
                        if w[fuera].sum() > 0:
                            w[fuera] -= falta * (w[fuera] / w[fuera].sum())
                            w[idxs]  += falta * (w[idxs]  / max(w[idxs].sum(), 1e-12))
                    elif s > hi:
                        sobra = s - hi
                        fuera = np.setdiff1d(np.arange(n), idxs)
                        if w[idxs].sum() > 0:
                            w[idxs]  -= sobra * (w[idxs]  / w[idxs].sum())
                            w[fuera] += sobra * (w[fuera] / max(w[fuera].sum(), 1e-12))
                    w = np.clip(w, [b[0] for b in bounds[:n]], [b[1] for b in bounds[:n]])
                    w = proyectar_simplex(w)
                    return w

                w_init = empuja_grupo(w_init, grupos['deuda'],    *rango_deuda)
                w_init = empuja_grupo(w_init, grupos['variable'], *rango_var)
                w_init = empuja_grupo(w_init, grupos['alternos'], *rango_alt)

                if use_hard_turnover:
                    t_init = np.abs(w_init - w0)
                    # ajusta t a que sum(t)<=T_max
                    s = t_init.sum()
                    if s > T_max:
                        t_init *= (T_max / s)
                    return np.concatenate([w_init, t_init])
                else:
                    return w_init

            x0_base = factible_x0()

            def solve_once(x0_local):
                return minimize(
                    fun=objetivo_total,
                    x0=x0_local,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True, 'eps': 1e-8}
                )

            # multi-start: mejor de K arranques (incluye x0_base + aleatorios)
            best_res = None
            for k in range(K_starts):
                if k == 0:
                    x0_k = x0_base
                else:
                    # muestrea Dirichlet y proyecta
                    z = rng.dirichlet(np.ones(n))
                    wk = np.clip(z, [b[0] for b in bounds[:n]], [b[1] for b in bounds[:n]])
                    wk = proyectar_simplex(wk)
                    if use_hard_turnover:
                        tk = np.abs(wk - w0)
                        s = tk.sum()
                        if s > T_max:
                            tk *= (T_max / s)
                        x0_k = np.concatenate([wk, tk])
                    else:
                        x0_k = wk

                res_k = solve_once(x0_k)
                if best_res is None or (res_k.success and res_k.fun < best_res.fun):
                    best_res = res_k

            res = best_res
            if res is None:
                resultados_optimizacion[entidad] = {"error": "No se pudo iniciar la optimización."}
                continue

            print(res.success, res.status)
            print(res.message)
            #print("nit:", res.nit, "fun:", res.fun)

            if res.success:
                if use_hard_turnover:
                    w_opt = res.x[:n]
                else:
                    w_opt = res.x

                # desglose objetivo en solución
                base_, l2_, ent_, tx_ = comp_obj(w_opt)
                #print(f"[{entidad}] objetivo -> base:{base_:.6f}  L2:{l2_:.6f}  entropy:{ent_:.6f}  turnover_cost:{tx_:.6f}")
                # métricas de auditoría
                turnover = float(np.sum(np.abs(w_opt - w0)))
                pct_borde = float(np.mean(
                    (np.isclose(w_opt, [b[0] for b in bounds[:n]], atol=1e-8)) |
                    (np.isclose(w_opt, [b[1] for b in bounds[:n]], atol=1e-8))
                ))
                N_eff = float(1.0 / np.sum(w_opt ** 2))
                s_deuda = float(w_opt[grupos['deuda']].sum())
                s_var   = float(w_opt[grupos['variable']].sum())
                s_alt   = float(w_opt[grupos['alternos']].sum())
                #print(f"[{entidad}] turnover:{turnover:.4f}  N_efectivo:{N_eff:.2f}  grupos -> deuda:{s_deuda:.4f} variable:{s_var:.4f} alternos:{s_alt:.4f}  %en_borde:{pct_borde:.2%}")

                pesos_optimos = {col: float(p) for col, p in zip(cols, w_opt)}
                resultados_optimizacion[entidad] = pesos_optimos
                #print("pesos optimos ", pesos_optimos)
            else:
                resultados_optimizacion[entidad] = {"error": "Optimización fallida."}
                print("Optimización fallida para la entidad.", entidad)

        return resultados_optimizacion
    """ 
    def predecir_ultimos_6_meses(self, entidad: str, columna_objetivo: str) -> pd.DataFrame:
        ###
        ###Predice los últimos 6 meses de datos para una entidad específica
        ###y devuelve tanto las predicciones como los valores reales.

        ###Args:
        ###    entidad (str): El nombre de la entidad para la cual se harán las predicciones.
        ###    columna_objetivo (str): El nombre de la columna que contiene los valores reales.

        ###Returns:
        ###    pd.DataFrame: Un DataFrame con las fechas, los valores reales y las predicciones.
        
        # 1. Obtener los últimos 6 meses del DataFrame original de la entidad
        df_entidad = self.df[self.df["entidad"] == entidad].copy()
        if df_entidad.empty:
            raise ValueError(f"No se encontraron datos para la entidad '{entidad}'.")
        
        # Asegurarse de que el DataFrame esté ordenado por fecha
        df_entidad.sort_values("fecha", inplace=True)
        
        # Seleccionar las últimas 6 filas (los 6 meses más recientes)
        df_futuro_6_meses = df_entidad.tail(6)
        
        # Extraer los valores reales de la columna objetivo
        valores_reales = df_futuro_6_meses[columna_objetivo].tolist()
        
        # 2. Utilizar la función de predicción existente
        try:
            predicciones = self.predecir_escenarios(df_futuro_6_meses, entidad)
        except ValueError as e:
            print(f"Error al predecir para la entidad '{entidad}': {e}")
            return pd.DataFrame()
            
        # 3. Formatear y devolver los resultados
        # Se crea un nuevo DataFrame con las fechas, los valores reales y las predicciones
        df_resultados = pd.DataFrame({
            "fecha": df_futuro_6_meses["fecha"].reset_index(drop=True),
            "valor_real": valores_reales,
            "prediccion": predicciones
        })
        
        return df_resultados
    """    
    def guardar_modelo(self, ruta: str):
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        ruta_completa = os.path.join(self.ruta_modelo,ruta)
        with open(ruta_completa, "wb") as f:
            pickle.dump(self.modelo, f)
    
    def calcular_accuracy_regresion(self, y_true, y_pred, tolerancia: float = 0.10) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tolerancia_absoluta = tolerancia * np.abs(y_true)
        correctas = np.abs(y_true - y_pred) <= tolerancia_absoluta
        return round(np.mean(correctas), 4)            

    def ejecutar(self):
        df_completo = completar_fechas_mensuales(self.df)
        df_completo = df_completo.sort_values(["entidad", "fecha"])
        df_completo = df_completo.groupby("entidad", group_keys=False).apply(lambda g: g.ffill())
        df_completo = df_completo.reset_index(drop=True)

        # ➜ crea columnas peso_* y actualiza las predictoras para usar pesos
        df_completo = self._agregar_columnas_peso(df_completo)
        print('col 1',df_completo.columns)
        otras_cols = [c for c in self.columnas_predictoras if not c.startswith("monto_")]
        self.columnas_predictoras = otras_cols + getattr(self, "columnas_peso", [])        
        #resultados_entidades = {}
        #modelos_entidades_temporales = {}

        self.entrenar_mlp(df_completo)
        
        resultados = self.modelos_entidades
        #print("Parametros por entidad 1")
        #print(resultados)

        filas = []
        for entidad, res in resultados.items():
            filas.append({"entidad": entidad,
                        "mse": res.get("mse", None),
                        "r2": res.get("r2", None),
                        'accuracy_10pct': res.get("accuracy_10pct", None),
                        "modelo": str(res.get("modelo", "").__class__.__name__)
                    })
        d_res =  pd.DataFrame(filas)                                   
        print("Parametros por entidad")
        print(d_res.head(20))

        nombre_archivo = "Metricas_mlp.csv" 
        ruta = os.path.join(self.ruta_resultado, nombre_archivo)
        d_res.to_csv(ruta, index=False)        

        composicion_optima = self._optimizar_composicion(df_completo, False) #False cartera total
        print('resultados ',composicion_optima.items())
        
        # Dentro del método ejecutar
# ... (después de obtener los resultados_optimizacion) ...
        filas = []        
        #print("\n--- Comparación y Recomendaciones ---")
        for entidad, composicion in composicion_optima.items():
            #print(f"\nRecomendaciones para la entidad: {entidad}")
            if 'error' in composicion:
                print(f"  No hay recomendaciones disponibles debido a un error: {composicion['error']}")
                continue # Salta a la siguiente entidad en el bucle principal            
            # Obtener los datos más recientes del portafolio
            df_entidad = df_completo[df_completo['entidad'] == entidad].copy()
            if not df_entidad.empty:
                datos_actuales = df_entidad.tail(1).copy()

                if 'monto_total_cartera' not in datos_actuales.columns:
                    # Si no existe, crearla sumando las columnas de instrumentos.
                    # Asegúrate de que las columnas de instrumentos sean numéricas.
                    try:
                        monto_total = datos_actuales[self.columnas_instrumentos].sum(axis=1).iloc[0]
                        datos_actuales.loc[:, 'monto_total_cartera'] = monto_total
                        #datos_actuales['monto_total_cartera'] = monto_total
                    except KeyError as e:
                        # Manejar el error si alguna columna de instrumento no está presente.
                        raise ValueError(f"Error al calcular monto_total_cartera: falta la columna {e}")                
                monto_total_actual = datos_actuales['monto_total_cartera'].iloc[0]

                for instrumento, peso_optimo in composicion.items():
                    monto_real_actual = datos_actuales[instrumento].iloc[0]
                    monto_optimo_calculado = monto_total_actual * peso_optimo

                    filas.append({"entidad": entidad,
                                "instrumento": instrumento,
                                "monto_real": monto_real_actual,
                                "monto_optimo": monto_optimo_calculado,
                                'peso_optimo': peso_optimo
                            })                    
                    #diferencia = monto_optimo_calculado - monto_real_actual
                    #print(f"'{instrumento}';{monto_real_actual};{monto_optimo_calculado}")
                    #if abs(diferencia) > (monto_total_actual * 0.01): # Un umbral del 1% para ser significativo
                    #    accion = "Aumentar" if diferencia > 0 else "Reducir"
                    #    print(f"  - Instrumento '{instrumento}': Monto Actual {monto_real_actual:,.2f} óptimo {monto_optimo_calculado:,.2f}. {accion} inversión en {abs(diferencia):,.2f}")
                    #else:
                    #    print(f"  - Instrumento '{instrumento}': Monto Actual {monto_real_actual:,.2f} óptimo {monto_optimo_calculado:,.2f}. Mantener inversión, la diferencia es mínima.")
                d_optimo =  pd.DataFrame(filas)                     
            else:
                print(f"  No hay datos recientes para comparar.")
                           
            #print("Optimos por entidad")
            #print(d_optimo.head(20))   
            nombre_archivo = "OptimosPorEntidad.csv" 
            ruta = os.path.join(self.ruta_resultado, nombre_archivo)
            d_optimo.to_csv(ruta, index=False)        

        self.resultado = ["Se ejecutó optimización de desequilibrio financiero"]
        return resultados
    
class VisualizadorOptimizacion:
    def __init__(self, resultados: dict, fecha: str, mostrarGrafico: bool = False):
        """
        resultados: diccionario con resultados por entidad
        """
        self.resultados = resultados
        self.df_resultados = self._construir_dataframe()
        self.fechaTexto = fecha.replace("-", "")
        self.ruta_resultado = os.path.join(DATA_ROOT, RESULTADOS_ROOT, AGENTE_DIVERSIFICACION, DIV_MODULO_OPTIMIZACION, self.fechaTexto)
        self.mostrarGrafico = mostrarGrafico        

    def _construir_dataframe(self) -> pd.DataFrame:
        filas = []
        for entidad, res in self.resultados.items():
            if "error" not in res:
                filas.append({
                    "entidad": entidad,
                    "mse": res.get("mse", None),
                    "r2": res.get("r2", None),
                    "modelo": str(res.get("modelo", "").__class__.__name__)
                })
        return pd.DataFrame(filas)

    def graficar_metricas(self, guardar=False, ruta="resultados_metricas.png"):
        if self.df_resultados.empty:
            print("No hay resultados para visualizar.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(data=self.df_resultados, x="entidad", y="mse", ax=axes[0])
        axes[0].set_title("Error Cuadrático Medio (MSE)")
        axes[0].tick_params(axis='x', rotation=45)

        sns.barplot(data=self.df_resultados, x="entidad", y="r2", ax=axes[1])
        axes[1].set_title("Coeficiente de Determinación (R²)")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if guardar:
            os.makedirs(os.path.dirname(os.path.join(self.ruta_resultado,ruta)), exist_ok=True)
            plt.savefig(os.path.join(self.ruta_resultado,ruta), bbox_inches='tight')
        if not self.mostrarGrafico:
            plt.close()
        plt.show()

    def mostrar_tabla(self):
        if not self.resultados:
            return "No hay resultados para visualizar."

        registros = []
        for entidad, res in self.resultados.items():
            if "mse" in res:
                registros.append({
                    "entidad": entidad,
                    "mse": res["mse"],
                    "r2": res["r2"],
                    "accuracy_10pct": res.get("accuracy_10pct", None),
                    "modelo": type(res["modelo"]).__name__
                })

        if not registros:
            return "No hay resultados válidos para mostrar."

        df_resultados = pd.DataFrame(registros)
        self.df_resultados = df_resultados
        os.makedirs(os.path.dirname(self.ruta_resultado), exist_ok=True)
        df_resultados.to_csv(os.path.join(self.ruta_resultado, "resultados.csv"), index=False)

        return df_resultados.sort_values("mse")
  
