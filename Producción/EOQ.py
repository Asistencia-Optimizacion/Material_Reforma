"""
Heurística “EOQ‑por‑bloques”  

ENTRADAS ------------------------------------------------------------------
  M                ← lista ordenada de períodos  (por ejemplo 1..N)
  d[t]             ← demanda en el período t
  I0               ← inventario inicial antes de empezar
  Q_target         ← tamaño objetivo de lote (p.ej. EOQ o valor dado)
  allow_backorder  ← FALSE   # si TRUE se permiten faltantes (opcional)
  MAX_HORIZON      ← |M|     # facultativo si se simulan horizontes rodantes

SALIDAS -------------------------------------------------------------------
  prod[t]          ← unidades producidas en el período t
  I_ini[t]         ← inventario al inicio de t  (antes de producir y servir d[t])
  I_fin[t]         ← inventario al final de t   (después de producir y servir d[t])
  costo_total      ← calcular externamente con K, h, etc.  (no se computa aquí)

---------------------------------------------------------------------------
ALGORITMO ------------------------------------------------------------------

# 0. Inicialización global
para t ∈ M:
      prod[t] ← 0
I ← I0                       # inventario “on‑hand”
idx ← 1                      # posición en M  (asumiendo M = 1…N)

# 1. Procesar períodos hasta agotar el horizonte
mientras idx ≤ MAX_HORIZON:

    # 1.1 Iniciar bloque nuevo en t_ini
    t_ini ← idx
    I_ini[t_ini] ← I

    # 1.2 Acumular demanda de períodos sucesivos
    demanda_acum ← 0
    k ← 0
    mientras idx + k ≤ MAX_HORIZON  y  demanda_acum < Q_target:
          demanda_acum ← demanda_acum + d[idx + k]
          k ← k + 1
    # al salir, bloque = {idx , … , idx + k − 1}

    # 1.3 Decidir producción en t_ini para cubrir el bloque
    faltante_bloque ← max(0 , demanda_acum − I)
    prod[t_ini]    ← prod[t_ini] + faltante_bloque
    I              ← I + faltante_bloque        # inventario después de producir

    # 1.4 Satisfacer demanda dentro del bloque
    para j = 0 … k−1:
          t_cur ← idx + j

          si j > 0:          # solo t_ini tiene producción; otros inician con inventario disponible
                I_ini[t_cur] ← I

          I ← I − d[t_cur]   # servir demanda del período

          si I < 0 y no allow_backorder:
                # Ajuste retroactivo: aumentar producción en t_ini
                prod[t_ini] ← prod[t_ini] − I      # −I es el faltante
                I            ← 0                   # quedamos sin faltantes

          I_fin[t_cur] ← I

    # 1.5 Avanzar al siguiente bloque
    idx ← idx + k

# 2. Si faltan claves en tablas de salida, completar con ceros
para t ∈ M:
      I_ini.setdefault(t, 0)
      I_fin.setdefault(t, 0)
      prod.setdefault(t, 0)

# 3. Retornar resultados primarios (costo se calcula aparte)
return prod , I_ini , I_fin


"""

import math

def heuristica_eoq(M, d, I_0, EOQ, debug=False):
    """
    Heurística tipo EOQ / lot-sizing por agrupación de períodos.
    Produce en el período t_ini lo necesario para cubrir demanda acumulada
    de un bloque de períodos hasta alcanzar EOQ (o fin de horizonte), usando
    inventario disponible. Sin faltantes (se ajusta producción si ocurre).

    Parámetros
      M     : lista ordenada de períodos (índices).
      d     : dict[t] demanda en t.
      I_0   : inventario inicial antes del primer período.
      EOQ   : tamaño económico de lote deseado (umbral de demanda acumulada).
      debug : imprime trazas si True.

    Devuelve
      produccion     : dict[t] unidades producidas en t (solo al inicio de lote).
      inventario_ini : dict[t] inventario al inicio de t (antes de producir/consumir en t).
      inventario_fin : dict[t] inventario al final de t (después de consumo).
    """

    n = len(M)
    produccion     = {t: 0 for t in M}
    inventario_ini = {}   # inv al inicio de t (antes de producir en lote que arranca en t)
    inventario_fin = {}   # inv al final de t (para costos de holding)

    inv = I_0  # inventario disponible al inicio del período actual
    idx = 0

    while idx < n:
        t_ini = M[idx]

        # inventario inicial del período donde inicia el lote
        inventario_ini[t_ini] = inv

        # agrupar períodos hasta alcanzar EOQ o fin del horizonte
        demanda_acum = 0
        k = 0
        while idx + k < n and demanda_acum < EOQ:
            demanda_acum += d[M[idx + k]]
            k += 1
        # al salir: bloque = períodos idx .. idx+k-1; demanda_acum cubre ese bloque

        # producir en t_ini lo faltante para cubrir el bloque dado el inventario actual
        q = max(0, demanda_acum - inv)
        produccion[t_ini] += q
        inv += q  # producción disponible de inmediato

        if debug:
            print(f"\n[Lote inicia en {t_ini}] inv_ini={inventario_ini[t_ini]}  demanda_lote={demanda_acum}  prod={q}")

        # consumir demanda período a período dentro del bloque
        for j in range(k):
            t_cur = M[idx + j]

            # registrar inventario inicial en períodos intermedios del bloque (no se produce)
            if j > 0:
                inventario_ini[t_cur] = inv

            inv -= d[t_cur]

            # no se permiten faltantes: si ocurren, aumentar producción retroactivamente en t_ini
            if inv < 0:
                faltante = -inv
                if debug:
                    print(f"  Ajuste: faltante {faltante} detectado en {t_cur}; aumentando prod en {t_ini}.")
                produccion[t_ini] += faltante
                inv = 0  # después del ajuste quedamos en cero (cubre exactamente)

            inventario_fin[t_cur] = inv
            if debug:
                print(f"    Fin {t_cur}: inv_fin={inv}")

        # avanzar al siguiente bloque
        idx += k

    # asegurar claves completas en las estructuras de salida
    for t in M:
        inventario_ini.setdefault(t, 0)
        inventario_fin.setdefault(t, 0)
        produccion.setdefault(t, 0)

    return produccion, inventario_ini, inventario_fin
