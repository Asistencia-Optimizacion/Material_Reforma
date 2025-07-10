import matplotlib.pyplot as plt
import networkx as nx

def draw_policy_dag(env, policy, initial_state, *, pastel=True, node_size=350):
    """
    ============================================================================
    Visualización del DAG de estados inducido por una política dada
    ─────────────────────────────────────────────────────────────────────────────
    Dibuja el grafo dirigido de estados del entorno y resalta la trayectoria
    seguida por una política determinista a partir de un estado inicial.

    Entradas
      • env           : entorno discreto con métodos:
                           ▸ state_space(), actions(s), sim_step(s, a), is_terminal(s)
      • policy        : dict[state → action]
                           política determinista a visualizar
      • initial_state : tupla
                           estado inicial desde el cual seguir la política
      • pastel        : bool
                           si True, usa colores suaves (por defecto)
      • node_size     : int
                           tamaño de los nodos en el grafo

    Resultado
      • Visualización gráfica del DAG con:
          - nodos coloreados por tipo de acción ('take' / 'skip')
          - trayectoria seguida por la política destacada
    ============================================================================
    """

    # -------------------------------------------------------------------------
    # 1. CONSTRUCCIÓN DEL GRAFO COMPLETO DE ESTADOS Y TRANSICIONES
    # -------------------------------------------------------------------------
    G = nx.DiGraph()
    for s in env.state_space():
        if env.is_terminal(s):
            continue
        for a in env.actions(s):
            ns, r = env.sim_step(s, a)
            G.add_edge(s, ns, action=a, reward=r)

    # -------------------------------------------------------------------------
    # 2. TRAZAR LA TRAYECTORIA INDUCIDA POR LA POLÍTICA
    # -------------------------------------------------------------------------
    optimal_edges = []
    state = initial_state
    while not env.is_terminal(state):
        a = policy.get(state)
        if a is None:
            break  # Política incompleta o estado huérfano
        ns, _ = env.sim_step(state, a)
        optimal_edges.append((state, ns))
        state = ns

    # -------------------------------------------------------------------------
    # 3. LAYOUT DE NODOS BASADO EN COORDENADAS DISCRETAS (i, capacidad)
    # -------------------------------------------------------------------------
    states = list(G.nodes)
    if not states:
        raise ValueError("No hay nodos en el grafo")

    dim0_vals = sorted(set(s[0] for s in states))  # dimensión índice de ítem
    dim1_vals = sorted(set(s[1] for s in states))  # dimensión capacidad restante
    dx, dy = 1.6, 1.0
    pos = {s: (dim0_vals.index(s[0]) * dx,
               len(dim1_vals) - 1 - dim1_vals.index(s[1]) * dy)
           for s in states}

    # -------------------------------------------------------------------------
    # 4. DEFINIR COLORES SEGÚN ACCIÓN TOMADA EN LA POLÍTICA
    # -------------------------------------------------------------------------
    palette = dict(
        pastel=dict(primary="#b2e2b2", alt="#b3cde3", other="#d9d9d9"),
        vivid=dict(primary="#2ca02c", alt="#1f77b4", other="#7f7f7f")
    )
    colors = palette["pastel" if pastel else "vivid"]

    # ▸ Asignar acción tomada desde cada nodo (si está en la trayectoria óptima)
    decision_at = {u: G.edges[(u, v)]["action"] for u, v in optimal_edges}

    # ▸ Función auxiliar para asignar color según tipo de acción
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

    # -------------------------------------------------------------------------
    # 5. DESTACAR LA POLÍTICA COMO ARISTAS DE MAYOR PESO
    # -------------------------------------------------------------------------
    edge_widths = [3 if e in optimal_edges else 1 for e in G.edges]

    # -------------------------------------------------------------------------
    # 6. DIBUJAR EL GRAFO FINAL
    # -------------------------------------------------------------------------
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
