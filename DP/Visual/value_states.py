import matplotlib.pyplot as plt
import numpy as np

def value_states_visual(env, V):
    """
    Dibuja un mapa de calor anotado de la función de valor V(s),
    asumiendo que los estados son pares (eje_y, eje_x).

    Esta versión es genérica y se adapta automáticamente a la estructura
    del espacio de estados del entorno.

    Parámetros
    ----------
    env : objeto de entorno con métodos:
        - state_space()
        - is_terminal(state)
        - atributos legibles para eje_x y eje_y (opcional)
    V : dict[state, float]
        Valor estimado para cada estado.
    """

    # Extraemos el espacio de estados y descartamos los terminales
    all_states = [s for s in env.state_space() if not env.is_terminal(s)]

    # Identificamos automáticamente la forma de los estados (deben ser 2D)
    sample_state = all_states[0]
    if len(sample_state) != 2:
        raise ValueError("La visualización solo admite estados con dos dimensiones.")

    # Asignación de ejes: eje_y = s[0], eje_x = s[1]
    eje_y_vals = sorted(set(s[0] for s in all_states))
    eje_x_vals = sorted(set(s[1] for s in all_states))

    # Mapeo de índices para matriz
    y_map = {val: i for i, val in enumerate(eje_y_vals)}
    x_map = {val: i for i, val in enumerate(eje_x_vals)}

    # Matriz de valores con NaNs por defecto
    value_matrix = np.full((len(eje_y_vals), len(eje_x_vals)), np.nan)

    for s in all_states:
        y, x = s
        if s in V:
            value_matrix[y_map[y], x_map[x]] = V[s]

    # Gráfico
    fig, ax = plt.subplots(figsize=(20, 16))
    cax = ax.imshow(value_matrix, cmap="YlGn", origin="upper", aspect="auto")

    # Anotaciones dentro de la matriz
    for y in range(len(eje_y_vals)):
        for x in range(len(eje_x_vals)):
            val = value_matrix[y, x]
            if not np.isnan(val):
                ax.text(x, y, f"{val:.1f}", ha='center', va='center', color='black', fontsize=9)

    # Etiquetas y ticks
    ax.set_title("Mapa de calor de la función de valor $V(s)$")
    ax.set_xlabel("Segundo componente del estado")
    ax.set_ylabel("Primer componente del estado")
    ax.set_xticks(np.arange(len(eje_x_vals)))
    ax.set_yticks(np.arange(len(eje_y_vals)))
    ax.set_xticklabels(eje_x_vals)
    ax.set_yticklabels(eje_y_vals)

    # Barra de color
    fig.colorbar(cax, ax=ax, label="$V(s)$")

    plt.tight_layout()
    plt.show()
