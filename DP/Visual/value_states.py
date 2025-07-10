import matplotlib.pyplot as plt
import numpy as np

def value_states_visual(env, V):
    """
    ============================================================================
    Visualización: Mapa de calor de la función de valor V(s)
    ─────────────────────────────────────────────────────────────────────────────
    Dibuja una matriz anotada que representa gráficamente los valores estimados
    V(s) sobre un entorno con estados bidimensionales (tipo (i, capacidad)).

    Entradas
      • env : entorno discreto que implementa
                 ▸ state_space()
                 ▸ is_terminal(state)
      • V   : dict[state, float]
                 función de valor V(s) para cada estado no terminal

    Requisitos
      ▸ Los estados deben tener exactamente dos componentes: (eje_y, eje_x)
    ============================================================================
    """

    # -------------------------------------------------------------------------
    # 1. EXTRAER ESTADOS NO TERMINALES Y VERIFICAR FORMATO
    # -------------------------------------------------------------------------
    all_states = [s for s in env.state_space() if not env.is_terminal(s)]

    # ▸ Verificamos que los estados sean bidimensionales
    sample_state = all_states[0]
    if len(sample_state) != 2:
        raise ValueError("La visualización solo admite estados con dos dimensiones.")

    # -------------------------------------------------------------------------
    # 2. CONSTRUIR MATRIZ DE VALORES PARA EL MAPA DE CALOR
    # -------------------------------------------------------------------------
    # ▸ Asignación: eje_y = s[0], eje_x = s[1]
    eje_y_vals = sorted(set(s[0] for s in all_states))  # ej: índice del ítem
    eje_x_vals = sorted(set(s[1] for s in all_states))  # ej: capacidad restante

    # ▸ Mapas de índice para acceder en la matriz
    y_map = {val: i for i, val in enumerate(eje_y_vals)}
    x_map = {val: i for i, val in enumerate(eje_x_vals)}

    # ▸ Matriz de valores (rellena con NaNs por defecto)
    value_matrix = np.full((len(eje_y_vals), len(eje_x_vals)), np.nan)

    for s in all_states:
        y, x = s
        if s in V:
            value_matrix[y_map[y], x_map[x]] = V[s]

    # -------------------------------------------------------------------------
    # 3. GRAFICAR MAPA DE CALOR CON ANOTACIONES
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20, 16))
    cax = ax.imshow(value_matrix, cmap="YlGn", origin="upper", aspect="auto")

    # ▸ Anotar valores dentro de cada celda
    for y in range(len(eje_y_vals)):
        for x in range(len(eje_x_vals)):
            val = value_matrix[y, x]
            if not np.isnan(val):
                ax.text(x, y, f"{val:.1f}", ha='center', va='center',
                        color='black', fontsize=9)

    # ▸ Etiquetas de ejes y ticks
    ax.set_title("Mapa de calor de la función de valor $V(s)$")
    ax.set_xlabel("Segundo componente del estado")
    ax.set_ylabel("Primer componente del estado")
    ax.set_xticks(np.arange(len(eje_x_vals)))
    ax.set_yticks(np.arange(len(eje_y_vals)))
    ax.set_xticklabels(eje_x_vals)
    ax.set_yticklabels(eje_y_vals)

    # ▸ Barra de color a la derecha
    fig.colorbar(cax, ax=ax, label="$V(s)$")

    plt.tight_layout()
    plt.show()
