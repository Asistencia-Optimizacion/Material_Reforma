import random
import pandas as pd
import numpy as np


# ============================================================================
# Generador sintético de instancias de planeación de inventarios estacionales
# ============================================================================
def generar_datos(num_periodos: int = 12,
                                 rango_c_barato=(4, 6),
                                 rango_c_medio=(6, 8),
                                 rango_c_caro=(8, 12),
                                 rango_h_barato=(1, 3),
                                 rango_h_medio=(3, 5),
                                 rango_h_caro=(5, 8),
                                 demanda_base=(4, 6),
                                 demanda_ruido=1,
                                 prob_mes_caro=0.25,
                                 prob_mes_barato=0.35,
                                 start_inventory_max=3,
                                 ensure_windows=True,
                                 seed=None):
    """
    ============================================================================
    Generador sintético de datos para el problema de inventario multi-período
    ─────────────────────────────────────────────────────────────────────────────
    Crea una instancia artificial con estacionalidad en costos de producción y
    almacenamiento, útil para probar algoritmos de planeación, DP, RL o MIP.

    La lógica imita patrones típicos como los de tu ejemplo:
      • Meses con producción barata agrupados (p.ej. cosecha / capacidad ociosa).
      • Meses con producción cara dispersos (picos de escasez / costos elevados).
      • Holding (almacenamiento) a veces barato justo antes de un mes caro
        → incentiva la construcción de inventario.
      • Demanda relativamente estable (rango acotado) con pequeñas variaciones.

    Entradas
    --------
    num_periodos : int (default=12)
        Número de períodos (meses) en el horizonte de planeación.
    rango_c_barato, rango_c_medio, rango_c_caro : tuple(float,float)
        Rangos (min,max) para muestrear costos de producción por nivel.
    rango_h_barato, rango_h_medio, rango_h_caro : tuple(float,float)
        Rangos (min,max) para costos de almacenamiento por nivel.
    demanda_base : tuple(int,int)
        Rango uniforme base para la demanda (sin ruido).
    demanda_ruido : int
        Amplitud de ruido entero uniforme agregado a la demanda base.
    prob_mes_caro : float
        Probabilidad de marcar un período como “costo de producción caro”.
    prob_mes_barato : float
        Probabilidad de marcar un período como “costo de producción barato”.
        (Los restantes quedan como “medio”; se renormaliza si suma >1.)
    start_inventory_max : int
        Inventario inicial I0 ~ U{0..start_inventory_max}.
    ensure_windows : bool
        Si True, fuerza al menos un mes caro rodeado por uno o dos meses baratos
        para crear oportunidades de preproducción (útil para pruebas).
    seed : int or None
        Semilla para reproducibilidad.

    Salidas
    -------
    c : dict[int -> float]
        Costos de producción c_t (indexados 1..num_periodos).
    h : dict[int -> float]
        Costos de almacenamiento h_t (indexados 1..num_periodos).
    d : dict[int -> int]
        Demandas d_t (enteras, indexadas 1..num_periodos).
    I0 : int
        Inventario inicial (0..start_inventory_max).
    M : range
        Conjunto de períodos (range(1, num_periodos+1)).
    df : pd.DataFrame
        Tabla resumen con columnas:
            ['t','Mes','c_t','h_t','d_t','Clasif_c','Clasif_h','Demanda_rel']

    Notas
    -----
    • Las clasificaciones Barato/Medio/Caro se basan en los rangos definidos,
      no en cuantiles de la muestra (para consistencia entre instancias).
    • Si `ensure_windows=True`, se re-marca al menos un mes caro y meses
      adyacentes baratos (si existen) para inducir decisiones interesantes.
    • Las unidades (M COP/ton, ton) son relativas; escala libre.
    • Si quieres correlacionar holding con producción (p.ej. hold barato justo
      antes de prod cara), puedes ajustar el bloque que sigue al muestreo base.

    Ejemplo
    -------
    >>> c, h, d, I0, M, df = generar_instancia_inventario(seed=42)
    >>> df
    ============================================================================
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = num_periodos
    M = range(1, T + 1)

    # ----------------------------------------------------------------------
    # 1. Clasificaciones iniciales de costo de producción por período
    # ----------------------------------------------------------------------
    # Convert probabilities to weights that sum to 1
    p_caro = prob_mes_caro
    p_barato = prob_mes_barato
    p_medio = max(0.0, 1.0 - p_caro - p_barato)
    weights = np.array([p_barato, p_medio, p_caro])
    weights = weights / weights.sum()

    # Sample category for each period: 0=barato,1=medio,2=caro
    cat_c = np.random.choice([0,1,2], size=T, p=weights)

    # ----------------------------------------------------------------------
    # 2. ensure_windows: forzar al menos un pico caro con vecinos baratos
    # ----------------------------------------------------------------------
    if ensure_windows:
        # elegir un índice para "mes caro" si no existe uno ya
        if 2 not in cat_c:
            idx = np.random.randint(T)
            cat_c[idx] = 2
        # elige algunos vecinos a marcar como baratos si están medio
        idx_caro = int(np.where(cat_c == 2)[0][0])
        if idx_caro - 1 >= 0 and cat_c[idx_caro - 1] == 1:
            cat_c[idx_caro - 1] = 0
        if idx_caro + 1 < T and cat_c[idx_caro + 1] == 1:
            cat_c[idx_caro + 1] = 0

    # ----------------------------------------------------------------------
    # 3. Muestrear costos de producción según categoría
    # ----------------------------------------------------------------------
    c_vals = []
    for cat in cat_c:
        if cat == 0:   # barato
            c_vals.append(round(random.uniform(*rango_c_barato), 2))
        elif cat == 1: # medio
            c_vals.append(round(random.uniform(*rango_c_medio), 2))
        else:          # caro
            c_vals.append(round(random.uniform(*rango_c_caro), 2))

    # ----------------------------------------------------------------------
    # 4. Holding: correlacionar con producción (opcional)
    # ----------------------------------------------------------------------
    # Idea: antes de un mes caro de producción, haz el holding más barato
    # para incentivar anticipar producción. Después de mes caro, subir algo.
    h_vals = []
    for t in range(T):
        # por defecto: medio
        base = round(random.uniform(*rango_h_medio), 2)

        # si el próximo mes es caro en producción → bajar holding actual
        if t+1 < T and cat_c[t+1] == 2:
            base = round(random.uniform(*rango_h_barato), 2)

        # si el mes actual es caro en producción → holding un poco caro
        if cat_c[t] == 2:
            base = round(random.uniform(*rango_h_caro), 2)

        h_vals.append(base)

    # ----------------------------------------------------------------------
    # 5. Demanda: base + ruido
    # ----------------------------------------------------------------------
    # base uniforme entero
    d_low, d_high = demanda_base
    d_vals = np.random.randint(d_low, d_high+1, size=T)

    if demanda_ruido > 0:
        ruido = np.random.randint(-demanda_ruido, demanda_ruido+1, size=T)
        d_vals = np.clip(d_vals + ruido, 0, None)

    d_vals = d_vals.astype(int).tolist()

    # ----------------------------------------------------------------------
    # 6. Inventario inicial
    # ----------------------------------------------------------------------
    I0 = random.randint(0, start_inventory_max)

    # ----------------------------------------------------------------------
    # 7. Empaquetar en dicts 1..T
    # ----------------------------------------------------------------------
    c = {t+1: c_vals[t] for t in range(T)}
    h = {t+1: h_vals[t] for t in range(T)}
    d = {t+1: d_vals[t] for t in range(T)}

    # ----------------------------------------------------------------------
    # 8. DataFrame resumen con clasificación
    # ----------------------------------------------------------------------
    
    # etiquetas mes (ENE..DIC) si T<=12; si >12, usar M1..MT
    meses_12 = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']

    if T <= 12:
        mes_labels = meses_12[:T]
    else:
        mes_labels = [f"M{t+1}" for t in range(T)]

    def clasif_c(val):
        if rango_c_barato[0] <= val <= rango_c_barato[1]: return "Barato"
        if rango_c_caro[0]   <= val <= rango_c_caro[1]:   return "Caro"
        return "Medio"

    def clasif_h(val):
        if rango_h_barato[0] <= val <= rango_h_barato[1]: return "Barato"
        if rango_h_caro[0]   <= val <= rango_h_caro[1]:   return "Caro"
        return "Medio"

    d_arr = np.array(d_vals)
    d_med = np.median(d_arr)
    def clasif_d(val):
        if val > d_med: return "Alta"
        if val < d_med: return "Baja"
        return "Media"

    registros = []
    for t in range(T):
        registros.append({
            "t"       : t+1,
            "Mes"     : mes_labels[t],
            "c_t"     : c_vals[t],
            "h_t"     : h_vals[t],
            "d_t"     : d_vals[t],
            "Clasif_c": clasif_c(c_vals[t]),
            "Clasif_h": clasif_h(h_vals[t]),
            "Demanda_rel": clasif_d(d_vals[t]),
        })

    df = pd.DataFrame(registros)
    df.set_index(df['t'], inplace=True)

    return df
