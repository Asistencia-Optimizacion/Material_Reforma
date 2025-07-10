import matplotlib.pyplot as plt
import networkx as nx

def draw_policy_dag(env, policy, initial_state, *, pastel=True, node_size=350):
    """
    Dibuja el DAG de estados de un entorno discreto con una política dada.

    Parámetros
    ----------
    env : objeto de entorno con:
        - state_space(), actions(s), sim_step(s, a), is_terminal(s)
    policy : dict[state, action]
        Política que determina la acción óptima desde cada estado
    initial_state : tupla
        Estado inicial desde donde seguir la política (e.g., (0, capacity))
    pastel : bool
        Colores suaves si True, intensos si False
    node_size : int
        Tamaño de los nodos en el grafo
    """

    # ---- 1) Construcción del grafo
    G = nx.DiGraph()
    for s in env.state_space():
        if env.is_terminal(s):
            continue
        for a in env.actions(s):
            ns, r = env.sim_step(s, a)
            G.add_edge(s, ns, action=a, reward=r)

    # ---- 2) Trayectoria inducida por la política
    optimal_edges = []
    state = initial_state
    while not env.is_terminal(state):
        a = policy.get(state)
        if a is None:
            break  # Evitamos errores si la política no está completa
        ns, _ = env.sim_step(state, a)
        optimal_edges.append((state, ns))
        state = ns

    # ---- 3) Layout generalizado (automático)
    # Detectamos dimensiones del estado
    states = list(G.nodes)
    if not states:
        raise ValueError("No hay nodos en el grafo")
    dim0_vals = sorted(set(s[0] for s in states))
    dim1_vals = sorted(set(s[1] for s in states))
    dx, dy = 1.6, 1.0
    pos = {s: (dim0_vals.index(s[0]) * dx,
               len(dim1_vals) - 1 - dim1_vals.index(s[1]) * dy)
           for s in states}

    # ---- 4) Colores para nodos según acción en política
    palette = dict(
        pastel=dict(primary="#b2e2b2", alt="#b3cde3", other="#d9d9d9"),
        vivid=dict(primary="#2ca02c", alt="#1f77b4", other="#7f7f7f")
    )
    colors = palette["pastel" if pastel else "vivid"]

    # Etiquetamos acción para colorear nodos según política
    decision_at = {}
    for u, v in optimal_edges:
        decision_at[u] = G.edges[(u, v)]["action"]

    # Coloreamos según tipo de acción (agrupamos por tipo si es string)
    def color_for_action(a):
        if isinstance(a, str):
            if a.lower() in ["take", "order", "comprar"]:
                return colors["primary"]
            elif a.lower() in ["skip", "omitir", "esperar"]:
                return colors["alt"]
        elif isinstance(a, int):
            return colors["primary"] if a > 0 else colors["alt"]
        return colors["other"]

    node_colors = [
        color_for_action(decision_at.get(n)) if n in decision_at else colors["other"]
        for n in G.nodes
    ]

    # ---- 5) Grosor de aristas
    edge_widths = [3 if e in optimal_edges else 1 for e in G.edges]

    # ---- 6) Dibujar el grafo
    plt.subplots(figsize=(20, 16))
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           edgecolors="black", linewidths=0.4)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.title("DAG de estados con trayectoria según la política")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
