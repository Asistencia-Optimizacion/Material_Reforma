"""
Módulo para graficar superficies 3D y caminos sobre ellas usando Plotly.

Funciones:
----------
- plot_(evaluar_indice, steps_list=None, rango=(-4, 4), resolucion=500):
    Genera una gráfica 3D de una superficie definida por `evaluar_indice` y
    opcionalmente añade trayectorias (steps_list) sobre la superficie.
"""

import numpy as np
import plotly.graph_objects as go


def plot_(evaluar_indice, steps_list=None, rango=(-4, 4), resolucion=500):
    """
    Grafica una superficie en 3D junto con rutas opcionales.

    Parámetros
    ----------
    evaluar_indice : callable
        Función que recibe un arreglo [X, Y] y devuelve valores Z.
    steps_list : list | tuple, opcional
        Lista de trayectorias. Cada trayectoria es una lista de puntos
        en el formato (xy, z), donde xy = (x, y) o np.array([x, y]).
    rango : tuple, opcional
        Límite mínimo y máximo de los ejes x e y. Por defecto (-4, 4).
    resolucion : int, opcional
        Número de puntos por eje en la malla. Por defecto 500.
    """

    # --- Malla para la superficie ---
    x_vals = np.linspace(rango[0], rango[1], resolucion)
    y_vals = np.linspace(rango[0], rango[1], resolucion)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = evaluar_indice([X, Y])

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdGy',
        reversescale=True,
        opacity=0.9
    ))

    # --- Procesar caminos ---
    paths = []
    if steps_list is None:
        paths = []
    else:
        if (
            isinstance(steps_list, (list, tuple))
            and len(steps_list) > 0
            and isinstance(steps_list[0], tuple)
            and len(steps_list[0]) == 2
        ):
            # Caso: un solo camino
            paths = [steps_list]
        else:
            # Caso: múltiples caminos
            paths = list(steps_list)

    def _xy_from_point(p):
        """Convierte un punto en array/lista/tupla a coordenadas (x, y)."""
        if isinstance(p, np.ndarray):
            return float(p[0]), float(p[1])
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            return float(p[0]), float(p[1])
        else:
            raise ValueError(
                "Formato de punto no reconocido. "
                "Esperaba array([x,y]) o (x,y)."
            )

    # --- Añadir caminos a la figura ---
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, steps in enumerate(paths):
        xs, ys, zs = [], [], []
        for pt in steps:
            xy, z = pt[0], pt[1]
            x, y = _xy_from_point(xy)
            xs.append(x)
            ys.append(y)
            zs.append(float(z))

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines+markers',
            line=dict(color=colores[i % len(colores)], width=5),
            marker=dict(size=4, color=colores[i % len(colores)]),
            name=f'Ruta {i+1}'
        ))

    # --- Ajustes de layout ---
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Valor'
        ),
        width=900,
        height=700
    )

    # --- Mostrar figura ---
    fig.write_html("10 - Decenso por Gradiente/DG.html")
    print("Gráfica guardada como '10 - Decenso por Gradiente/DG.html'")
    #fig.show()
