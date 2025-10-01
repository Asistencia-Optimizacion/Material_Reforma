
from plot import plot_

# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada evaluar_indice que reciba como parámetro una tupla con una coordenada (x,y) 
y retorne el valor del índice ϕ para dicha coordenada. 
"""

def evaluar_indice(coords: tuple) -> float:
    """
    Calcula el valor del índice para una coordenada (x, y).

    Fórmula:
    --------
    f(x, y) = 100 * (x² + y - 11)² + 100 * (x + y² - 7)²

    Parámetros
    ----------
    coords : tuple
        Tupla con la coordenada (x, y).

    Retorna
    -------
    float
        Valor del índice en la coordenada dada.
    """
    x, y = coords
    return 100 * (x**2 + y - 11)**2 + 100 * (x + y**2 - 7)**2


# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada evaluar_factibilidad que reciba como parámetro una tupla representando 
una coordenada (x,y) y retorne True en caso de que dicha coordenada se encuentre dentro de la región de estudio 
(cumpla todas las restricciones) y False de lo contrario.
"""

def evaluar_factibilidad(coor: tuple) -> bool:
    """
    Evalúa si una coordenada (x, y) es factible según restricciones.

    Restricciones:
    --------------
    -6 ≤ x ≤ 6  
    -6 ≤ y ≤ 6  
    x + y ≥ -6

    Parámetros
    ----------
    coor : tuple
        Tupla con la coordenada (x, y).

    Retorna
    -------
    bool
        True si cumple todas las restricciones (factible).
        False en caso contrario.
    """
    x, y = coor

    return (
        x >= -6 and
        x <= 6 and
        y >= -6 and
        y <= 6 and
        x + y >= -6
    )


# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada grad_evaluar_phi que reciba como parámetro un diccionario donde 
las llaves son coordenadas (x,y)  y retorne el gradiente en dicho punto. Realicé las derivadas parciales 
de manera manual y adecuada.

Derivada parcial respecto de x
    ∂/∂x [100·(x² + y − 11)²] 
        = 100·2·(x² + y − 11)·∂/∂x(x² + y − 11) 
        = 200·(x² + y − 11)·(2x) 
        = 400x·(x² + y − 11).

    ∂/∂x [100·(x + y² − 7)²] 
        = 100·2·(x + y² − 7)·∂/∂x(x + y² − 7)
        = 200·(x + y² − 7)·(1) 
        = 200·(x + y² − 7).

    ∂ϕ/∂x = 400x·(x² + y − 11) + 200·(x + y² − 7).
    ∂ϕ/∂x = 400x³ + 400x·y − 4200x + 200y² − 1400.

Derivada parcial respecto de y
    ∂/∂y [100·(x² + y − 11)²] 
        = 100·2·(x² + y − 11)·∂/∂y(x² + y − 11)
        = 200·(x² + y − 11)·(1) 
        = 200·(x² + y − 11).

    ∂/∂y [100·(x + y² − 7)²] 
        = 100·2·(x + y² − 7)·∂/∂y(x + y² − 7)
        = 200·(x + y² − 7)·(2y) 
        = 400y·(x + y² − 7).

    ∂ϕ/∂y = 200·(x² + y − 11) + 400y·(x + y² − 7).
    ∂ϕ/∂y = 200x² + 400x·y + 400y³ − 2600y − 2200.

Gradiente
∇ϕ(x, y) = ( ∂ϕ/∂x , ∂ϕ/∂y )
         = ( 400x³ + 400x·y − 4200x + 200y² − 1400 , 200x² + 400x·y + 400y³ − 2600y − 2200 ).

"""

import numpy as np

def grad_phi(coords: tuple) -> np.ndarray:
    """
    Calcula el gradiente de la función φ(x, y) en una coordenada (x, y).

    Fórmula:
    --------
    ∇φ(x, y) = ( ∂φ/∂x , ∂φ/∂y )

    Donde:
    ∂φ/∂x = 400x³ + 400x·y − 4200x + 200y² − 1400  
    ∂φ/∂y = 200x² + 400x·y + 400y³ − 2600y − 2200

    Parámetros
    ----------
    coords : tuple
        Tupla con la coordenada (x, y).

    Retorna
    -------
    np.ndarray
        Vector gradiente [∂φ/∂x, ∂φ/∂y] evaluado en la coordenada dada.
    """

    x, y = coords


# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente en Python la función gradient_descent, que reciba como parámetros una tupla inicial (x, y), 
la función objetivo, la función que calcula el gradiente, la tasa de aprendizaje (alpha), el número máximo 
de iteraciones y un valor de tolerancia (epsilon); asegúrese de que cada nuevo punto permanezca dentro de 
la región factible y retorne el recorrido completo de puntos generados durante el proceso, mostrando la 
trayectoria seguida hasta aproximarse al mínimo de la función.
"""

def gradient_descent(x0: np.ndarray, f, grad, alpha: float, iters: int, eps: float = 1e-6):
    """
    Ejecuta el algoritmo de descenso por gradiente para minimizar una función.

    Fórmula de actualización:
    -------------------------
    x_{k+1} = x_k − α ∇f(x_k)

    Donde:
    - x_k es el punto actual.
    - α (alpha) es la tasa de aprendizaje.
    - ∇f(x_k) es el gradiente de la función en el punto actual.

    Parámetros
    ----------
    x0 : np.ndarray
        Punto inicial del algoritmo en forma de vector (x, y).
    f : callable
        Función objetivo que se desea minimizar.
    grad : callable
        Función que calcula el gradiente de f en un punto dado.
    alpha : float
        Tasa de aprendizaje (define el tamaño del paso).
    iters : int
        Número máximo de iteraciones permitidas.
    eps : float, opcional
        Tolerancia usada como criterio de convergencia, por defecto 1e-6.

    Retorna
    -------
    list[tuple[np.ndarray, float]]
        Lista con la trayectoria completa del algoritmo.
        Cada elemento es una tupla (x, f(x)), donde:
        - x es la coordenada actual en forma de np.ndarray.
        - f(x) es el valor de la función objetivo en ese punto.
    """
    pass



# ----------------------------------------------------------------------------------------------------------------------

# --- Configuración inicial ---
# Coordenadas iniciales (x⁰, y⁰)
coor_inicial = np.array([0.5, -0.25])

# Longitud de paso λ
lbd = 0.00001

# --- Ejecución de la búsqueda local ---
path_g2  = gradient_descent(coor_inicial, evaluar_indice, grad_phi, alpha=lbd, iters=100)

print("--- Mejor Solución ---")

print(f"Numero de pasos: {len(path_g2)}")
print(f"X: {path_g2[-1][0][0]}, Y: {path_g2[-1][0][1]}")
print(f"Valor de la función objetivo: {path_g2[-1][1]}")

# --- Visualización de resultados ---
plot_(evaluar_indice, path_g2)
