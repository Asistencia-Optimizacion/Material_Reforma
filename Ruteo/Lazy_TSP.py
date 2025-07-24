import pulp as lp
import time
import sys
import os

sys.path.append(os.path.abspath("../Instances"))
from InsTSP import visualizar        # Visualiza el recorrido en coordenadas geográficas

# ============================================================================
# Modelo TSP sin eliminación anticipada de subtours ("Lazy Start")
# ============================================================================
# Formula un modelo de TSP clásico (basado en flujo binario), pero **sin**
# incluir restricciones de subtour desde el inicio.
#
# Se resuelve el problema con solo las restricciones de grado (entrada/salida),
# lo cual permite analizar soluciones iniciales o usarlo como base para
# callbacks o lazy constraints (en solvers más avanzados).
# ============================================================================
def optimizar_tcl_lazy_start(N: list, A: dict) -> tuple[lp.LpProblem, dict]:
    """
    ============================================================================
    Optimización del TSP con restricciones mínimas (modelo relajado)
    ─────────────────────────────────────────────────────────────────────────────
    Este modelo busca un ciclo mínimo sin incluir restricciones de subtour.
    Útil como punto de partida para técnicas iterativas o detección de subtours.

    Parámetros:
      • N : list
          Lista de nodos (IDs únicos, incluyendo la bodega)
      • A : dict
          Diccionario con distancias: A[(i, j)] = distancia_km

    Salida:
      • model : objeto PuLP con el modelo formulado
      • x     : diccionario de variables binarias de decisión: x[i, j] ∈ {0, 1}
    ============================================================================
    """

    # ------------------------------------------------------------------------
    # 1. DEFINICIÓN DEL MODELO
    # ------------------------------------------------------------------------
    model = lp.LpProblem("TSP_Lazy", lp.LpMinimize)

    # ------------------------------------------------------------------------
    # 2. VARIABLES DE DECISIÓN
    # ------------------------------------------------------------------------
    # x[i,j] = 1 si se viaja del nodo i al nodo j (para i ≠ j)
    x = {
        (i, j): lp.LpVariable(f"x_{i}_{j}", cat=lp.LpBinary)
        for i in N for j in N if i != j
    }

    # ------------------------------------------------------------------------
    # 3. FUNCIÓN OBJETIVO: minimizar la distancia total recorrida
    # ------------------------------------------------------------------------
    model += lp.lpSum(x[i, j] * A[i, j] for i, j in x), "Costo_total"

    # ------------------------------------------------------------------------
    # 4. RESTRICCIONES DE GRADO (flujo)
    # ------------------------------------------------------------------------

    # ▸ Una única salida por nodo
    for i in N:
        model += lp.lpSum(x[i, j] for j in N if i != j) == 1, f"Salida_{i}"

    # ▸ Una única entrada por nodo
    for i in N:
        model += lp.lpSum(x[j, i] for j in N if i != j) == 1, f"Entrada_{i}"

    # ------------------------------------------------------------------------
    # 5. RESOLVER MODELO
    # ------------------------------------------------------------------------
    solver = lp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    # (No se imprime el estado aquí para mantenerlo modular)

    return model, x

# ============================================================================
# Identificación de subtours en soluciones parciales de TSP
# ============================================================================
# A partir del diccionario de sucesores (succ), se recorren los nodos para
# identificar ciclos cerrados (subtours) que violan el requisito del ciclo único.
# ============================================================================

def identificar_subtours(succ: dict, N: list) -> list[list[str]]:
    """
    ============================================================================
    Identifica subtours en una solución de TSP basada en sucesores directos.
    ─────────────────────────────────────────────────────────────────────────────
    Entrada:
      • succ : dict
          Diccionario con asignaciones de sucesor → succ[i] = j, si se va de i a j
      • N    : list
          Lista de todos los nodos en la instancia

    Salida:
      • subtours : list[list[str]]
          Lista de ciclos encontrados. Cada subtour es una lista de nodos en orden.

    Notas:
      • Esta función no asume que hay un solo tour: retorna todos los ciclos cerrados.
      • No incluye tours incompletos (sin sucesor), pero sí ciclos de tamaño ≥ 1.
    ============================================================================
    """
    unvisited = set(N)   # Nodos que aún no han sido asignados a ningún tour
    subtours = []        # Lista de subtours identificados

    # Recorremos todos los nodos hasta agruparlos en ciclos
    while unvisited:
        
        start = unvisited.pop()  # Nodo inicial del subtour
        tour = [start]
        cur = start

        while True:
            nxt = succ.get(cur)

            # Finaliza el subtour si no hay sucesor o se cierra el ciclo
            if nxt is None or nxt == start:
                break

            tour.append(nxt)
            if nxt in unvisited:
                unvisited.remove(nxt)
            cur = nxt

        subtours.append(tour)

    return subtours

import pulp as lp

# ============================================================================
# Actualización iterativa del modelo TSP con eliminación de subtours
# ============================================================================
# Esta función toma un modelo de TSP parcialmente resuelto (sin restricciones
# de subtour), identifica los ciclos indeseados y los elimina agregando nuevas
# restricciones al modelo antes de resolverlo nuevamente.
# ============================================================================

def optimizar_tcl_lazy_update(model: lp.LpProblem,
                              x: dict,
                              subtours: list[list[str]]) -> tuple[lp.LpProblem, dict]:
    """
    ============================================================================
    Añade restricciones de subtour al modelo y resuelve nuevamente
    ─────────────────────────────────────────────────────────────────────────────
    Entrada:
      • model    : objeto PuLP del modelo actual (sin subtour o con pocos cortes)
      • x        : diccionario de variables binarias x[i,j] ∈ {0,1}
      • subtours : lista de subtours (cada uno como lista de nodos)

    Comportamiento:
      • Por cada subtour S, agrega la restricción:
            ∑_{i,j ∈ S, i≠j} x[i,j] ≤ |S| - 1
        lo cual fuerza a romper el ciclo.

    Salida:
      • model : modelo resuelto después de agregar las nuevas restricciones
      • x     : diccionario de variables (referencia sin cambios)
    ============================================================================
    """
    # ------------------------------------------------------------------------
    # 1. AGREGAR CORTES DE SUBTOUR AL MODELO
    # ------------------------------------------------------------------------
    for S in subtours:
        if len(S) <= 1:
            continue  # No se agrega restricción si el "subtour" es trivial
        model += lp.lpSum(
            x[(i, j)] for i in S for j in S if i != j and (i, j) in x
        ) <= len(S) - 1, f"Subtour_elim_{'_'.join(S)}"

    # ------------------------------------------------------------------------
    # 2. RESOLVER MODELO CON NUEVAS RESTRICCIONES
    # ------------------------------------------------------------------------
    solver = lp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    return model, x

# ============================================================================
# Resolución iterativa del TSP con eliminación progresiva de subtours
# ============================================================================
# Esta función implementa una estrategia tipo "cutting plane", en la que:
#   1. Se resuelve un modelo inicial sin restricciones de subtour
#   2. Se identifican ciclos inválidos (subtours)
#   3. Se agregan restricciones para eliminarlos
#   4. Se reoptimiza el modelo
# El proceso se repite hasta obtener un único tour válido o agotar iteraciones.
# ============================================================================

def optimizar_tcl_lazy(clientes, N: list, A: dict):
    """
    ============================================================================
    Modelo exacto de TSP con eliminación iterativa de subtours (Lazy Constraints)
    ─────────────────────────────────────────────────────────────────────────────
    Ejecuta un proceso iterativo de resolución del problema TSP donde se agregan
    restricciones de subtour solo cuando se detectan en la solución actual.

    Parámetros:
      • N : list
          Lista de nodos (identificadores únicos)
      • A : dict
          Diccionario con distancias: A[(i,j)] = distancia entre nodos

    Salida:
      • model : objeto PuLP final (puede incluir cortes de subtour)
      • x     : diccionario de variables binarias: x[i,j] = 1 si se toma ese arco
    ============================================================================
    """

    sols = []
    times = []
    valido = []

    # ------------------------------------------------------------------------
    # 1. Resolver modelo base (sin cortes de subtour)
    # ------------------------------------------------------------------------

    t_init_temp = time.perf_counter()
    model, x = optimizar_tcl_lazy_start(N, A)
    elapsed_temp = time.perf_counter() - t_init_temp

    obj = round(lp.value(model.objective), 2)
    sols.append(obj)
    times.append(elapsed_temp)
    valido.append(False)

    if lp.LpStatus[model.status] != "Optimal":
        print("No se encontró una solución inicial óptima.")
        return model, x

    # ------------------------------------------------------------------------
    # 2. Proceso iterativo de eliminación de subtours
    # ------------------------------------------------------------------------
    for iteracion in range(1, 1000):

        # ▸ Extraer arcos activos de la solución
        succ = {i: j for (i, j) in x if lp.value(x[(i, j)]) == 1}

        # ▸ Vamos a visualizar
        visualizar(clientes, succ, pplot = False, nombre_archivo=f"MIP/{iteracion}")

        # ▸ Detectar subtours presentes en la solución actual
        subtours = identificar_subtours(succ, N)

        # ▸ Si hay un solo ciclo que cubre todos los nodos, se ha terminado
        if len(subtours) == 1 and len(subtours[0]) == len(N):
            print(f"Tour válido encontrado en iteración {iteracion}.\n")
            break

        # ▸ Agregar cortes y resolver nuevamente
        t_init_temp = time.perf_counter()
        model, x = optimizar_tcl_lazy_update(model, x, subtours)
        elapsed_temp = time.perf_counter() - t_init_temp

        # ▸ Verificar si el modelo sigue siendo factible
        if lp.LpStatus[model.status] != "Optimal":
            print("Modelo se volvió infactible tras agregar cortes.")
            break

        print(f"Se invalidaron {len(subtours)} subtours --> Time solver: {round(elapsed_temp, 3)} seg.")
        obj = round(lp.value(model.objective), 2)
        sols.append(obj)
        times.append(times[-1] + elapsed_temp)
        valido.append(False)

    else:
        print("Límite de iteraciones alcanzado sin tour único (no convergió).")

    valido[-1] = True
    return model, x, sols, times, valido
