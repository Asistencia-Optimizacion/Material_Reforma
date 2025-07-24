import time
import sys
import os

sys.path.append(os.path.abspath("../Instances"))
from InsTSP import visualizar        # Visualiza el recorrido en coordenadas geográficas

# ============================================================================
# Cálculo del costo total de un tour dado (TSP)
# ============================================================================
# Recorre la lista ordenada de nodos en el tour y suma el costo de cada arco.
# Asume que el diccionario A contiene los costos entre pares (i, j).
# ============================================================================

def tour_cost(tour: list, A: dict) -> float:
    """
    ============================================================================
    Calcula el costo total asociado a un tour específico
    ─────────────────────────────────────────────────────────────────────────────
    Parámetros:
      • tour : list
          Lista ordenada de nodos visitados, que representa un ciclo (TSP)
      • A    : dict
          Diccionario con distancias o costos: A[(i, j)] = costo entre nodos

    Salida:
      • total : float
          Suma total del recorrido, incluyendo el regreso al nodo inicial

    Notas:
      • Si el tour está cerrado (inicio = fin), el ciclo se considera completo.
      • Si no lo está, esta función lo cierra automáticamente.
    ============================================================================
    """
    total = 0.0
    for k in range(len(tour)):
        i = tour[k]
        j = tour[(k + 1) % len(tour)]  # garantiza el cierre del ciclo
        if i != j:
            total += A[(i, j)]
    return total


# ============================================================================
# Identificación del nodo de inicio (depósito)
# ============================================================================
# Se establece el nodo de partida del tour. Por convención, si el nodo
# "Bodega" existe, se usa como depósito. En su defecto, se usa el primer nodo.
# ============================================================================

# ============================================================================
# Heurística Nearest Neighbor (ATSP)
# ============================================================================
# Construye un tour inicial partiendo desde un nodo dado, eligiendo en cada paso
# el nodo no visitado más cercano. Admite grafos dirigidos y asimétricos (A ≠ Aᵗ).
# ============================================================================

def tsp_nn(clientes, N: list, A: dict, start: str):

    """
    ============================================================================
    Heurística de vecino más cercano para TSP (asimétrico)
    ─────────────────────────────────────────────────────────────────────────────
    Parámetros:
      • N     : list
          Lista de nodos (IDs únicos)
      • A     : dict
          Diccionario de costos/distancias: A[(i,j)] = costo de ir de i a j
      • start : str
          Nodo inicial del recorrido (por ejemplo, "Bodega")

    Salida:
      • tour : list
          Lista ordenada de nodos visitados en el tour (sin cerrar)

    Notas:
      • El tour no se cierra automáticamente con el nodo inicial.
        Para cerrar el ciclo, puedes usar: tour + [tour[0]]
    ============================================================================
    """
    t_init_temp = time.perf_counter()
    unvisited = set(N)
    unvisited.remove(start)
    tour = [start]
    current = start
    elapsed_temp = time.perf_counter() - t_init_temp

    sols = [0]
    times = [elapsed_temp]
    valido = [False]
    cnt = 0

    while unvisited:
        t_init_temp = time.perf_counter()

        # Seleccionar el nodo no visitado más cercano desde el actual
        nxt = min(unvisited, key=lambda j: A[(current, j)])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

        # Guardar la información relevante
        elapsed_temp = time.perf_counter() - t_init_temp
        times.append(times[-1] + elapsed_temp)
        obj = round(tour_cost(tour, A), 2)
        sols.append(obj)
        valido.append(False)

        # Graficar
        visualizar(clientes, tour, pplot = False, nombre_archivo=f"NN/{cnt}")
        cnt += 1

    valido[-1] = True
    tour.append(start)
    return tour, sols, times, valido

# ============================================================================
# Heurística Cheapest Insertion (ATSP)
# ============================================================================
# Construye un tour de manera incremental. Parte desde un ciclo de 2 nodos
# y luego inserta el nodo no visitado que cause el menor incremento en el costo.
# Funciona sobre grafos dirigidos (A ≠ Aᵗ) y no requiere matriz simétrica.
# ============================================================================

def tsp_cheapest_insertion(clientes, N: list, A: dict, start):
    """
    Heurística de inserción más barata para TSP (asimétrico).
    Devuelve: tour, sols, times, valido
    """

    t_init_temp = time.perf_counter()
    otros = [i for i in N if i != start]

    # Caso trivial
    if not otros:
        return [start, start], [0], [time.perf_counter() - t_init_temp], [True]

    # Nodo inicial: el más barato desde start
    k = min(otros, key=lambda j: A[(start, j)])
    tour = [start, k]
    uninserted = [i for i in otros if i != k]

    elapsed_temp = time.perf_counter() - t_init_temp
    sols = [0]                           # como en tsp_nn
    times = [elapsed_temp]
    valido = [False]
    cnt = 0

    # Frame inicial
    visualizar(clientes, tour, pplot=False, nombre_archivo=f"CI/{cnt}")
    cnt += 1

    # Inserciones
    while uninserted:
        t_iter = time.perf_counter()

        best_delta = float("inf")
        best_pos = None
        best_node = None

        for node in uninserted:
            for idx in range(len(tour)):
                i = tour[idx]
                j = tour[0] if idx == len(tour) - 1 else tour[idx + 1]
                delta = A[(i, node)] + A[(node, j)] - A[(i, j)]
                if delta < best_delta:
                    best_delta = delta
                    best_pos = idx + 1
                    best_node = node

        tour.insert(best_pos, best_node)
        uninserted.remove(best_node)

        # métricas
        elapsed = time.perf_counter() - t_iter
        times.append(times[-1] + elapsed)
        sols.append(round(tour_cost(tour, A), 2))
        valido.append(False)

        # frame
        visualizar(clientes, tour, pplot=False, nombre_archivo=f"CI/{cnt}")
        cnt += 1

    # Cerrar tour
    tour.append(start)
    valido[-1] = True

    return tour, sols, times, valido



# ============================================================================
# Mejora local 2-opt para TSP dirigido (ATSP)
# ============================================================================
# Esta versión de 2-opt está adaptada para instancias dirigidas. A diferencia
# del TSP simétrico, aquí la inversión de un segmento implica reencadenar
# arcos respetando la dirección (A[i,j] ≠ A[j,i]).
# Se evalúan reemplazos de la forma:
#   Romper:    (i → a)  y  (b → j)
#   Reemplazar con: (i → b)  y  (a → j)
# Esto equivale a revertir el subcamino [a..b] manteniendo sentido.
# ============================================================================

def tsp_2opt_atsp(clientes, tour: list, A: dict, name):
    """
    Mejora local 2-opt para ATSP.
    Devuelve: tour_mejor, sols, times, valido
    """
    t0 = time.perf_counter()
    best = tour[:]
    best_cost = tour_cost(best, A)

    sols   = [round(best_cost, 2)]
    times  = [time.perf_counter() - t0]
    valido = [False]
    cnt = 0

    # Frame inicial
    visualizar(clientes, best, pplot=False, nombre_archivo=f"2OPT_{name}/{cnt}")
    cnt += 1

    improved = True
    while improved:
        improved = False
        n = len(best)

        for i in range(n - 1):
            for k in range(i + 1, n):
                candidate = best[:i + 1] + best[i + 1:k + 1][::-1] + best[k + 1:]
                try:
                    cand_cost = tour_cost(candidate, A)
                except KeyError:
                    continue

                if cand_cost < best_cost:
                    t_iter = time.perf_counter()

                    best = candidate
                    best_cost = cand_cost
                    improved = True

                    times.append(times[-1] + (time.perf_counter() - t_iter))
                    sols.append(round(best_cost, 2))
                    valido.append(False)

                    visualizar(clientes, best, pplot=False, nombre_archivo=f"2OPT_{name}/{cnt}")
                    cnt += 1
                    break
            if improved:
                break

    valido[-1] = True
    return best, sols, times, valido

