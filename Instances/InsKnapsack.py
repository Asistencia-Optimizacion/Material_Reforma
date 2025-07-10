import random
from typing import List, Tuple

def generate_knapsack_instance(
        n: int,
        *,
        w_min: int = 1,
        w_max: int = 15,
        v_min: int = 1,
        v_max: int = 10,
        capacity_ratio: float = 0.3,
        seed: int | None = None
) -> Tuple[List[int], List[int], int]:
    """
    Genera una instancia aleatoria del problema de la mochila (0-1).

    Parámetros
    ----------
    n : int
        Número de objetos.
    w_min, w_max : int
        Rango inclusivo para los pesos (w_i ∈ [w_min, w_max]).
    v_min, v_max : int
        Rango inclusivo para los valores (v_i ∈ [v_min, v_max]).
    capacity_ratio : float
        Capacidad = capacity_ratio · ∑_i w_i  (0 < ratio ≤ 1).
        Si el ratio es bajo, la instancia será más “apretada”.
    seed : int | None
        Fija la semilla del generador aleatorio para reproducibilidad.

    Devuelve
    --------
    weights : list[int]
    values  : list[int]
    capacity: int
    """
    if seed is not None:
        random.seed(seed)

    weights = [random.randint(w_min, w_max) for _ in range(n)]
    values  = [random.randint(v_min, v_max) for _ in range(n)]

    total_w = sum(weights)
    capacity = max(1, int(round(capacity_ratio * total_w)))

    return weights, values, capacity