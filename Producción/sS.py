"""
(s, S) por períodos sin faltantes  

ENTRADAS ------------------------------------------------------------------
  M        ← lista ordenada de períodos                                (1…N)
  d[t]     ← demanda del período t
  I₀       ← inventario inicial antes del primer período
  s, S     ← parámetros de la política            (punto de pedido y nivel S)

SALIDAS -------------------------------------------------------------------
  prod[t]  ← cantidad producida en t
  I_ini[t] ← inventario al inicio de t  (antes de producir)
  I_fin[t] ← inventario al final de t   (después de demanda)
  costo    ← no se calcula aquí → debe evaluarse con cₜ •prod + hₜ •I_fin

ALGORITMO -----------------------------------------------------------------

1. Inicializar estructuras
      para t∈M:  prod[t] ← 0
      I ← I₀

2. Recorrer períodos t = 1 … N
      I_ini[t] ← I                       # registrar inventario inicial

      # 2.1 Regla (s, S) básica
      si I <  s:
          q ← S − I                      # reponer hasta S
      si I ≥ s:
          q ← 0                          # no reponer

      # 2.2 Garantizar que la demanda del período quede cubierta
      si I + q < d[t]:
          q ← q + ( d[t] − (I + q) )     # aumentar pedido para evitar faltantes

      prod[t] ← q
      I ← I + q                          # inventario tras recibir producción

      # 2.3 Satisfacer demanda
      I ← I − d[t]

      # 2.4 Protección contra inventario negativo (no se permiten faltantes)
      si I < 0:
          faltante ← −I
          prod[t] ← prod[t] + faltante   # producir de emergencia
          I ← 0
          si debug: imprimir mensaje de ajuste

      I_fin[t] ← I                       # registrar inventario final
      si debug: imprimir t, prod[t], I_ini[t], I_fin[t]

3. Completar claves faltantes (si alguna estructura quedó sin valor)
      para t∈M:
          I_ini.setdefault(t, 0)
          I_fin.setdefault(t, 0)
          prod.setdefault(t, 0)

4. Retornar resultados
      return prod , I_ini , I_fin        # (costo total se calcula aparte)

"""




def heuristica_ss(M, d, I_0, s, S, debug=False):
    """
    Heurística (s, S) con cobertura de demanda del período.
    Si inventario < s, reponer hasta S; si no, no producir.
    Luego asegurar que la producción cubra d[t] (sin faltantes).
    
    Params:
      M : lista ordenada de períodos.
      d : dict[t] demanda en t.
      I_0 : inventario inicial.
      s, S : parámetros de política (s<S).
      debug : trazas opcionales.
    
    Returns:
      produccion[t],
      inventario_ini[t],
      inventario_fin[t].
    """
    produccion     = {t: 0 for t in M}
    inventario_ini = {}
    inventario_fin = {}

    inv = I_0  # inventario al inicio del período actual

    for t in M:
        # inventario inicial (para graficar / reporte)
        inventario_ini[t] = inv

        # regla (s,S): ¿reponer?
        if inv < s:
            q = round(S - inv)
        else:
            q = 0

        # garantizar cobertura de demanda del período (no faltantes)
        if inv + q < d[t]:
            q += d[t] - (inv + q)

        produccion[t] = q
        inv += q  # producción disponible antes de satisfacer demanda

        # consumir demanda
        inv -= d[t]

        # protección numérica: no negativos
        if inv < 0:
            if debug:
                print(f"Ajuste tardío en {t}: faltante {-inv}; incrementando producción.")
            produccion[t] += -inv
            inv = 0

        inventario_fin[t] = inv

        if debug:
            print(f"t={t} inv_ini={inventario_ini[t]} prod={produccion[t]} demanda={d[t]} inv_fin={inventario_fin[t]}")

    return produccion, inventario_ini, inventario_fin
