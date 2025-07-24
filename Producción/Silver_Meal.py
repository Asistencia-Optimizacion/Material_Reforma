"""
Silver-Meal (planificación en bloques)

ENTRADAS ------------------------------------------------------------------
  M        ← lista ordenada de períodos                   (t₁, t₂, …, tₙ)
  d[t]     ← demanda del período t
  h[t]     ← costo de mantener 1 unidad al final de t
  K        ← costo fijo (setup) por lote
  I₀       ← inventario inicial antes del primer período

SALIDAS -------------------------------------------------------------------
  prod[t]  ← cantidad producida en el período t
  I_ini[t] ← inventario al inicio de t
  I_fin[t] ← inventario al final de t
  costo    ← (se calcula externamente usando prod e inventarios)

ALGORITMO -----------------------------------------------------------------

# 0. Inicialización
para t∈M:          prod[t] ← 0
I ← I₀                                  # inventario disponible
idx ← 1                                  # posición actual en la lista M

# 1. Repetir hasta agotar horizonte
mientras idx ≤ |M|:

    # 1.1 Definir inicio de bloque
    i ← idx                              # primer período sin cubrir
    t_ini ← M[i]                         # etiqueta del período i
    I_ini[t_ini] ← I                     # registrar inventario de arranque

    # 1.2 Explorar longitudes de bloque con regla Silver‑Meal
    mejor_len ← 1
    mejor_avg ← +∞
    demanda_acum ← 0

    para j desde i hasta |M|:
          t_j ← M[j]
          demanda_acum ← demanda_acum + d[t_j]

          # Producción neta necesaria para cubrir i..j
          q_net ← max( 0 , demanda_acum − I )
          
          # Inventario proyectado y costo de holding dentro del bloque
          inv_tmp ← I + q_net
          hold_cost ← 0
          para r = i … j:
                inv_tmp ← inv_tmp − d[ M[r] ]
                inv_tmp ← max(inv_tmp , 0)            # sin faltantes en proyección
                hold_cost ← hold_cost + h[ M[r] ] * inv_tmp
          
          total_cost ← K + hold_cost
          avg_cost   ← total_cost / ( j − i + 1 )

          si debug: imprimir(i, j, total_cost, avg_cost)

          # Criterio Silver‑Meal: detener cuando el promedio sube
          si avg_cost ≤ mejor_avg:
                mejor_len ← j − i + 1
                mejor_avg ← avg_cost
          si avg_cost > mejor_avg:
                romper bucle for‑j

    # 1.3 Bloque definitivo = i … i+mejor_len−1
    len_bloque ← mejor_len
    lot_demand ← Σ_{r=i}^{i+len_bloque−1}  d[ M[r] ]
    q_prod     ← max( 0 , lot_demand − I )            # producción en t_ini

    prod[t_ini] ← prod[t_ini] + q_prod
    I ← I + q_prod                                    # inventario tras producir

    si debug: imprimir lote definitivo y q_prod

    # 1.4 Consumir demanda dentro del bloque y registrar inventarios
    para r = i … i+len_bloque−1:
          t_r ← M[r]
          
          si r > i:                     # períodos intermedios
                I_ini[t_r] ← I
          
          I ← I − d[t_r]                # consumo real
          
          si I < 0:                     # corrección de faltante en ejecución
                falta ← −I
                prod[t_ini] ← prod[t_ini] + falta
                I ← 0
                si debug: imprimir ajuste por faltante
          
          I_fin[t_r] ← I

    # 1.5 Avanzar al siguiente bloque
    idx ← idx + len_bloque

# 2. Completar claves faltantes con ceros
para t∈M:
      I_ini.setdefault(t, 0)
      I_fin.setdefault(t, 0)
      prod.setdefault(t, 0)

# 3. Retornar resultados para cálculo posterior de costos
return prod , I_ini , I_fin

"""

def heuristica_silver_meal(M, c, h, d, I_0, debug=False):
    """
    Heurística Silver-Meal (lot-sizing aproximado).
    Extiende lote desde período i mientras el costo promedio (setup+holding)/período
    no aumente. Ajusta producción para evitar faltantes.

    Parámetros:
      M   : lista ordenada de períodos.
      c,h : dict[t] costos de producción y holding.
      d   : dict[t] demanda por período.
      I_0 : inventario inicial antes del primer período.
      debug : trazas opcionales.

    Retorna:
      produccion[t], inventario_ini[t], inventario_fin[t].
    """
    # Setup aproximado (no provisto en datos): múltiplo del costo medio p/ período
    K = (sum(c[t] for t in M) / len(M)) * 5.0

    n = len(M)
    produccion     = {t: 0 for t in M}
    inventario_ini = {}
    inventario_fin = {}

    inv = I_0   # inventario disponible al inicio del período actual
    idx = 0

    while idx < n:
        i      = idx
        t_ini  = M[i]
        inv_start = inv

        # inventario inicial en el período que inicia el lote
        inventario_ini[t_ini] = inv_start

        # explorar extensión del lote j = i..n-1
        prev_avg = float("inf")
        best_len = 1
        cum_dem  = 0

        for j in range(i, n):
            t_j = M[j]
            cum_dem += d[t_j]  # demanda acumulada i..j

            # prod neta requerida si cubro i..j
            net_prod = max(0, cum_dem - inv_start)

            # proyectar inventario período a período para costo holding
            inv_tmp   = inv_start + net_prod
            hold_cost = 0
            for r in range(i, j + 1):
                inv_tmp -= d[M[r]]
                if inv_tmp < 0:
                    inv_tmp = 0  # sin backorders en proyección
                hold_cost += inv_tmp * h[M[r]]

            total_cost = K + hold_cost
            avg_cost   = total_cost / (j - i + 1)

            if debug:
                print(f"Eval lote {t_ini}..{t_j}: total={total_cost:.2f}, avg={avg_cost:.2f}")

            # regla Silver-Meal: parar cuando aumenta costo promedio
            if avg_cost <= prev_avg:
                best_len = j - i + 1
                prev_avg = avg_cost
            else:
                break

        # lote elegido: i .. i+best_len-1
        lote_len = best_len
        lot_dem  = sum(d[M[r]] for r in range(i, i + lote_len))
        q        = max(0, lot_dem - inv_start)

        # producir en período inicial
        produccion[t_ini] += q
        inv += q

        if debug:
            print(f"\n[Lote definitivo {t_ini} cubre {lote_len} períodos] prod={q} inv_start={inv_start}")

        # consumir dentro del lote y registrar inventarios
        for r in range(i, i + lote_len):
            t_r = M[r]
            if r > i:
                inventario_ini[t_r] = inv  # arrastre

            inv -= d[t_r]
            if inv < 0:
                # ajuste anti-faltantes (retroproduce en t_ini)
                faltante = -inv
                produccion[t_ini] += faltante
                inv = 0
                if debug:
                    print(f"  Ajuste faltante en {t_r}: +{faltante} prod en {t_ini}.")

            inventario_fin[t_r] = inv

        # siguiente bloque
        idx += lote_len

    # completar claves faltantes (por seguridad)
    for t in M:
        inventario_ini.setdefault(t, 0)
        inventario_fin.setdefault(t, 0)
        produccion.setdefault(t, 0)

    return produccion, inventario_ini, inventario_fin
