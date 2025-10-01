
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
Implemente una función en Python llamada generar_radar que reciba dos parámetros: una tupla con las 
coordenadas actuales (x^t, y^t) y la longitud λ. La función debe retornar las 8 coordenadas vecinas en una lista. 
"""

from math import cos, pi, sin

def generar_radar(coor: tuple, lbd: float) -> list[tuple]:
    """
    Genera las 8 coordenadas vecinas de un punto (x, y) en un radio λ.

    Parámetros
    ----------
    coor : tuple
        Tupla con la coordenada actual (x, y).
    lbd : float
        Longitud de paso λ.

    Retorna
    -------
    list[tuple]
        Lista con las 8 coordenadas vecinas:
        - 4 en las direcciones cardinales (arriba, abajo, izquierda, derecha).
        - 4 en las diagonales (45°, 135°, 225°, 315°).
    """
    x, y = coor

    # --- Coordenadas cardinales ---
    coor1, coor2, coor3, coor4 = (
        (x + lbd, y),     # derecha
        (x - lbd, y),     # izquierda
        (x, y + lbd),     # arriba
        (x, y - lbd)      # abajo
    )

    # --- Coordenadas diagonales ---
    coor5, coor6, coor7, coor8 = (
        (x + lbd * cos(pi / 4), y + lbd * sin(pi / 4)),   # diagonal superior derecha
        (x + lbd * cos(pi / 4), y - lbd * sin(pi / 4)),   # diagonal inferior derecha
        (x - lbd * cos(pi / 4), y + lbd * sin(pi / 4)),   # diagonal superior izquierda
        (x - lbd * cos(pi / 4), y - lbd * sin(pi / 4))    # diagonal inferior izquierda
    )

    return [coor1, coor2, coor3, coor4, coor5, coor6, coor7, coor8]


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
        - (coordenada, índice) de la mejor ubicación (menor índice).
        - False si el diccionario está vacío.
    """
    if len(my_dict) != 0:
        mejor_ubicacion = (0, 0)
        mejor_objetivo = 10e12  # Valor inicial muy grande para comparación

        # Recorre todas las coordenadas buscando el menor índice
        for coordenada, indice in my_dict.items():
            if indice < mejor_objetivo:
                mejor_ubicacion = coordenada
                mejor_objetivo = indice

        return (mejor_ubicacion, mejor_objetivo)
    else:
        # Caso: diccionario vacío
        return False
    

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
coor_inicial = np.array([0.5, -0.25])

# Longitud de paso λ
lbd = 0.01

# --- Ejecución de la búsqueda local ---
steps = busqueda_local(coor_inicial, lbd) # type: ignore

# --- Visualización de resultados ---
plot_(evaluar_indice, steps)
