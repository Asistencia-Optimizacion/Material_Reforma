import numpy as np
import matplotlib.pyplot as plt


def value_states_visual(env, V, policy=None, trajectory=None,
                        annotate=True, cmap="YlGn",
                        marker_size=1500, line_width=2.0, line_style=':'):
    """
    ============================================================================
    Visualización: Mapa de calor de la función de valor V(s) + trayectoria
    ─────────────────────────────────────────────────────────────────────────────
    Dibuja un mapa de calor anotado de la función de valor V(s) sobre un entorno
    con **estados bidimensionales discretos** (p.ej. (t, S) en inventarios,
    (i, c) en mochila). Opcionalmente superpone la **trayectoria seguida** por
    una política determinista o una secuencia de estados proporcionada.

    Características principales
    ---------------------------
    • Detección automática de la rejilla de estados: filas = primer componente,
      columnas = segundo componente.
    • Se ignoran estados terminales (se grafican solo los no terminales).
    • Se anotan valores numéricos V(s) dentro de cada celda (opcional).
    • Se puede superponer una trayectoria:
        ▸ Si proporcionas `trajectory`, se usa tal cual.
        ▸ Si proporcionas `policy` y *no* `trajectory`, la función simula
          desde el estado inicial de `env` hasta llegar a un estado terminal,
          usando `env.sim_step()` (no altera el estado final del entorno).
    • La trayectoria se dibuja como **línea roja punteada** con **círculos
      rojos huecos grandes** para no tapar las etiquetas del heatmap.

    Requisitos del entorno `env`
    ----------------------------
    Debe implementar los siguientes métodos / propiedades:
        state_space()  -> iterable de estados
        is_terminal(s) -> bool
        actions(s)     -> iterable de acciones legales (solo si se simula policy)
        sim_step(s,a)  -> (s_next, reward)   (no muta el entorno)
        reset()        -> estado inicial     (usado para simular policy)

    Parámetros
    ----------
    env : objeto entorno compatible
        Entorno discreto con estados bidimensionales.
    V : dict[state, float]
        Valor estimado de cada estado (solo se usa si el estado aparece en V;
        celdas sin dato se dejan como NaN).
    policy : dict o callable, opcional
        Política determinista. Si es dict, se indexa por `state`. Si es función,
        se llama como `policy(state) -> action`. Usada solo si `trajectory` es None.
    trajectory : list[state], opcional
        Secuencia explícita de estados a resaltar. Si se pasa, no se simula policy.
    annotate : bool, por defecto True
        Si True, escribe el valor V(s) dentro de cada celda disponible.
    cmap : str, por defecto "YlGn"
        Mapa de color para el heatmap.
    marker_size : float, por defecto 1500
        Tamaño de los círculos huecos que marcan la trayectoria.
    line_width : float, por defecto 2.0
        Grosor de la línea de trayectoria.
    line_style : str, por defecto ':'
        Estilo de la línea (p.ej. '-', '--', ':').

    Comportamiento respecto a política
    ----------------------------------
    • Si `trajectory` se proporciona, `policy` se ignora.
    • Si `trajectory` es None y `policy` no es None:
        1. Se llama `env.reset()` para obtener el estado inicial.
        2. Se aplica la política hasta llegar a un estado terminal:
              s, _ = env.sim_step(s, a)
        3. Se registra la secuencia de estados (incluyendo inicial y último
           estado antes del terminal).
        4. Se vuelve a llamar `env.reset()` para restaurar el entorno.
    • Si `policy` es un dict pero el estado no aparece exactamente (p.ej. S float),
      se intenta con `(t, int(round(S)))` como fallback.

    Devoluciones
    ------------
    fig, ax : objetos matplotlib
        Figura y ejes del gráfico generado.
    value_matrix : np.ndarray (shape = [#valores_y, #valores_x])
        Matriz de valores V(s) usada para el heatmap (NaN donde falta V).
    eje_y_vals : list
        Valores ordenados usados en el eje Y (primer componente del estado).
    eje_x_vals : list
        Valores ordenados usados en el eje X (segundo componente del estado).
    path : list[state]
        Trayectoria que se dibujó (vacía si no hubo ni policy ni trajectory).

    Notas
    -----
    • Se grafican solo estados **no terminales**.
    • Si el último estado de la trayectoria es terminal y no aparece en
      `state_space()` como no terminal, simplemente no se dibuja ese punto.
    • Para evitar tapar las anotaciones V(s), los marcadores de trayectoria
      son círculos huecos (sin relleno).
    • Si quieres dibujar acciones entre estados, puedes modificar la sección
      de “Superponer trayectoria”.

    Ejemplo mínimo
    --------------
    >>> V = policy_evaluation(env, policy)
    >>> value_states_visual(env, V, policy=policy)
    ============================================================================
    """
    # -------------------------------------------------------------------------
    # 0. Estados no terminales
    # -------------------------------------------------------------------------
    all_states = [s for s in env.state_space() if not env.is_terminal(s)]
    if not all_states:
        raise ValueError("No hay estados no terminales en el entorno.")

    sample_state = all_states[0]
    if len(sample_state) != 2:
        raise ValueError("La visualización solo admite estados con dos dimensiones.")

    # -------------------------------------------------------------------------
    # 1. Rejilla
    # -------------------------------------------------------------------------
    eje_y_vals = sorted(set(s[0] for s in all_states))   # filas
    eje_x_vals = sorted(set(s[1] for s in all_states))   # columnas

    y_map = {val: i for i, val in enumerate(eje_y_vals)}
    x_map = {val: i for i, val in enumerate(eje_x_vals)}

    value_matrix = np.full((len(eje_y_vals), len(eje_x_vals)), np.nan)
    for s in all_states:
        if s in V:
            y, x = s
            value_matrix[y_map[y], x_map[x]] = V[s]

    # -------------------------------------------------------------------------
    # 2. Trayectoria
    # -------------------------------------------------------------------------
    path = []

    if trajectory is not None:
        path = list(trajectory)

    elif policy is not None:
        start_state = env.reset()
        s = start_state
        path.append(s)

        def _get_action(st):
            if callable(policy):
                return policy(st)
            if st in policy:
                return policy[st]
            # fallback por si S flotante
            alt = (st[0], int(round(st[1])))
            if alt in policy:
                return policy[alt]
            raise KeyError(f"La política no define acción para el estado {st}.")

        while not env.is_terminal(s):
            a = _get_action(s)
            s, _ = env.sim_step(s, a)
            path.append(s)

        env.reset()  # restaurar entorno

    # -------------------------------------------------------------------------
    # 3. Heatmap
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20, 16))
    cax = ax.imshow(value_matrix, cmap=cmap, origin="upper", aspect="auto")

    if annotate:
        for yy in range(len(eje_y_vals)):
            for xx in range(len(eje_x_vals)):
                val = value_matrix[yy, xx]
                if not np.isnan(val):
                    ax.text(xx, yy, f"{val:.1f}", ha='center', va='center',
                            color='black', fontsize=9)

    ax.set_title("Mapa de calor de la función de valor $V(s)$")
    ax.set_xlabel("Segundo componente del estado")
    ax.set_ylabel("Primer componente del estado")
    ax.set_xticks(np.arange(len(eje_x_vals)))
    ax.set_yticks(np.arange(len(eje_y_vals)))
    ax.set_xticklabels(eje_x_vals)
    ax.set_yticklabels(eje_y_vals)

    fig.colorbar(cax, ax=ax, label="$V(s)$")

    # -------------------------------------------------------------------------
    # 4. Superponer trayectoria: círculos rojos huecos + línea punteada
    # -------------------------------------------------------------------------
    if path:
        xs = []
        ys = []
        for (yy, xx) in path:
            if yy in y_map and xx in x_map:
                ys.append(y_map[yy])
                xs.append(x_map[xx])

        if xs and ys:
            # línea punteada (debajo de los círculos)
            ax.plot(xs, ys,
                    color='red',
                    linestyle=line_style,
                    linewidth=line_width,
                    zorder=4)

            # círculos huecos
            ax.scatter(xs, ys,
                       facecolors='none',
                       edgecolors='red',
                       s=marker_size,
                       linewidths=2,
                       zorder=5,
                       label='Trayectoria')

    plt.tight_layout()
    plt.show()
