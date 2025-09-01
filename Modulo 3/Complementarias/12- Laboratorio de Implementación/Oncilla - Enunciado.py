
from plot import plot_

# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada evaluar_indice que reciba como parámetro una tupla con una coordenada (x,y) 
y retorne el valor del índice ϕ para dicha coordenada. 
"""

import numpy as np

def evaluar_indice(coords):
    """
    Calcula el valor del índice para (x, y) en escalares o arrays.

    Fórmula:
    --------
    ϕ(x, y) = [ -100 * (x² + y - 60)² - 100 * (x + y² - 25)² ] / 100

    Parámetros
    ----------
    coords : (x, y)
        Puede ser una tupla de escalares o una lista/array [X, Y].

    Retorna
    -------
    float | np.ndarray
        Valor del índice en la coordenada o matriz de valores.
    """
    x, y = coords




# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada generar_radar que reciba dos parámetros: una tupla con las 
coordenadas actuales (x^t, y^t) y la longitud λ. La función debe retornar las 8 coordenadas vecinas en una lista. 
"""

def generar_radar(coor: tuple, lbd: float) -> list[tuple]:
    """
    Genera las 8 coordenadas vecinas de un punto (x, y) en forma de cuadrado.

    Parámetros
    ----------
    coor : tuple
        Tupla con la coordenada actual (x, y).
    lbd : float
        Longitud de paso λ.

    Retorna
    -------
    list[tuple]
        Lista con las 8 coordenadas vecinas (4 cardinales y 4 diagonales).
    """
    x, y = coor



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
    y ≥ -7
    x ≥ -4
    y ≤ 5
    x + 0.5y ≤ 7
    x - 0.3y ≤ 7

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


# ----------------------------------------------------------------------------------------------------------------------

"""
Implemente una función en Python llamada encontrar_mejor_coor que reciba como parámetro un diccionario donde 
las llaves son coordenadas y los valores el índice ϕ correspondiente. Esta función debe retornar una tupla 
con la coordenada de menor índice y, además, el valor de dicho índice. En esta función, debe evaluar si el 
diccionario que entra por parámetro está vacío, en cuyo caso deberá retornar False. 

El Algoritmo 1 resume esta función.
"""

def encontrar_mejor_coor(my_dict: dict):
    """
    Encuentra la coordenada con el menor índice en un diccionario.

    Parámetros
    ----------
    my_dict : dict
        Diccionario donde:
        - Llaves: coordenadas (x, y).
        - Valores: índice ϕ asociado a cada coordenada.

    Retorna
    -------
    tuple | bool
        - (coordenada, índice) de la mejor ubicación (mayor índice).
        - False si el diccionario está vacío.
    """

    

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

"""
Función que implementa el método de búsqueda local.

Este método utiliza las funciones:
    - evaluar_indice
    - generar_radar
    - evaluar_factibilidad
    - encontrar_mejor_coor
para realizar un proceso de optimización local.
"""


def busqueda_local(coord_ini: tuple, lbd: float) -> list:
    """
    Realiza una búsqueda local para encontrar la mejor coordenada factible.

    Parámetros
    ----------
    coord_ini : tuple
        Coordenada inicial (x, y) desde donde comienza la búsqueda.
    lbd : float
        Longitud de paso (lambda) utilizada para generar el radar.

    Retorna
    -------
    list[tuple]
        Lista de pasos [(coordenada, índice)], incluyendo la inicial
        y cada mejora encontrada durante la búsqueda local.
    """

    # --- Inicialización ---
    ubicacion_actual = coord_ini
    objetivo_actual = evaluar_indice(ubicacion_actual)
    hay_mejora = True

    steps = [(ubicacion_actual, objetivo_actual)]
    print(f"Iniciamos en: {coord_ini} con F.O. = {objetivo_actual}")

    # --- Proceso iterativo ---
    while hay_mejora:
        mejor_objetivo = objetivo_actual

        # Generar las ubicaciones del radar
        radar = generar_radar(ubicacion_actual, lbd)
        my_dict = {}

        # Guardar las ubicaciones factibles con sus índices
        for coor in radar:
            if evaluar_factibilidad(coor):
                nuevo_objetivo = evaluar_indice(coor)
                my_dict[coor] = nuevo_objetivo

        # Buscar la mejor ubicación del radar y actualizar posición
        res = encontrar_mejor_coor(my_dict)
        if res is not False and res[1] < mejor_objetivo:
            ubicacion_actual = res[0]
            objetivo_actual = res[1]
            steps.append((ubicacion_actual, objetivo_actual))
        else:
            # Criterio de parada: no hay mejora
            hay_mejora = False

    # --- Resumen del proceso ---
    print('\n-------- Proceso iterativo de búsqueda local finalizado --------\n')
    print(f'Para la coordenada inicial {coord_ini} y un lambda de {lbd}:\n')
    print(f'Número de iteraciones: {len(steps)}')
    print(f'La mejor ubicación encontrada fue: '
          f'({round(ubicacion_actual[0], 3)}, {round(ubicacion_actual[1], 3)})')
    print(f'El mejor índice encontrado fue: {round(objetivo_actual, 5)}\n')

    return steps

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

# --- Configuración inicial ---
# Coordenadas iniciales (x⁰, y⁰)
coor_inicial = np.array([0.0, 0.0])

# Longitud de paso λ
lbd = 0.01

# --- Ejecución de la búsqueda local ---
steps = busqueda_local(coor_inicial, lbd) # type: ignore

# --- Visualización de resultados ---
plot_(evaluar_indice, steps, nombre_archivo="BL")


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


"""
Implemente una función en Python llamada grad_evaluar_phi que reciba como parámetro un diccionario donde 
las llaves son coordenadas (x,y)  y retorne el gradiente en dicho punto. Realicé las derivadas parciales 
de manera manual y adecuada.

Función
φ(x, y) = [ -100·(x² + y − 60)² − 100·(x + y² − 25)² ] / 100

Simplificando:
φ(x, y) = −(x² + y − 60)² − (x + y² − 25)²

Derivada parcial respecto de x

∂/∂x [ −(x² + y − 60)² ]
    = −2·(x² + y − 60) · ∂/∂x(x² + y − 60)
    = −2·(x² + y − 60) · (2x)
    = −4x·(x² + y − 60)

∂/∂x [ −(x + y² − 25)² ]
    = −2·(x + y² − 25) · ∂/∂x(x + y² − 25)
    = −2·(x + y² − 25) · (1)
    = −2·(x + y² − 25)

∂φ/∂x = −4x·(x² + y − 60) − 2·(x + y² − 25)

Derivada parcial respecto de y

∂/∂y [ −(x² + y − 60)² ]
    = −2·(x² + y − 60) · ∂/∂y(x² + y − 60)
    = −2·(x² + y − 60) · (1)
    = −2·(x² + y − 60)

∂/∂y [ −(x + y² − 25)² ]
    = −2·(x + y² − 25) · ∂/∂y(x + y² − 25)
    = −2·(x + y² − 25) · (2y)
    = −4y·(x + y² − 25)

∂φ/∂y = −2·(x² + y − 60) − 4y·(x + y² − 25)

Gradiente

∇φ(x, y) = ( ∂φ/∂x , ∂φ/∂y )
    = ( −4x·(x² + y − 60) − 2·(x + y² − 25) , −2·(x² + y − 60) − 4y·(x + y² − 25) )
"""

import numpy as np

def grad_phi(coords: tuple) -> np.ndarray:
    """
    Calcula el gradiente de la función φ(x, y) en un punto dado.

    Función:
    --------
    φ(x, y) = −(x² + y − 60)² − (x + y² − 25)²

    Derivadas parciales:
    ∂φ/∂x = −4x·(x² + y − 60) − 2·(x + y² − 25)
    ∂φ/∂y = −2·(x² + y − 60) − 4y·(x + y² − 25)

    Parámetros
    ----------
    coords : tuple
        Tupla (x, y) con la coordenada donde evaluar el gradiente.

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
    x_{k+1} = x_k + α ∇f(x_k)

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
    xs = [(x0.copy(), f(x0))]
    x = x0.copy()
    for _ in range(iters):
        g = grad(x)
        if np.linalg.norm(g) < eps:         # criterio de parada por gradiente pequeño
            break
        x -= alpha * g                      # actualización del punto
        if not evaluar_factibilidad(x):     # verificación de la región factible
            break
        xs.append((x.copy(), f(x)))
    return xs



# ----------------------------------------------------------------------------------------------------------------------

# --- Configuración inicial ---
# Coordenadas iniciales (x⁰, y⁰)
coor_inicial = np.array([0.0, 0.0])

# Longitud de paso λ
lbd = 0.00025

# --- Ejecución de la búsqueda local ---
path_g2  = gradient_descent(coor_inicial, evaluar_indice, grad_phi, alpha=lbd, iters=100)

print("--- Mejor Solución ---")

print(f"Numero de pasos: {len(path_g2)}")
print(f"X: {path_g2[-1][0][0]}, Y: {path_g2[-1][0][1]}")
print(f"Valor de la función objetivo: {path_g2[-1][1]}")

# --- Visualización de resultados ---
plot_(evaluar_indice, path_g2, nombre_archivo="DG")