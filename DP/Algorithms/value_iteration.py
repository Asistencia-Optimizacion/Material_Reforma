def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    ============================================================================
    Algoritmo : Iteración de Valores (control óptimo, versión determinista)
    ─────────────────────────────────────────────────────────────────────────────
    Entradas
      • env    : entorno que implementa
                   ▸ state_space()         → Iterable[State]
                   ▸ is_terminal(s)        → bool
                   ▸ sim_step(s, a)        → (s', r)
                   ▸ actions(s)            → Iterable[Action]
      • gamma  : factor de descuento γ ∈ [0,1]
      • theta  : umbral de convergencia θ > 0

    Salida
      • policy : política óptima π*
      • V      : valores óptimos v* asociados a π*
    ============================================================================

    Descripción resumida
    ---------------------------------------------------------------------------
      1.  Inicializa V(s) ← 0 para todo s ∈ 𝒮
      2.  Repite hasta convergencia:
            ▸ V(s) ← max_a [ r + γ·V(s′) ]
            ▸ Δ ← max_s |v_antiguo - V(s)|
      3.  Deriva π*(s) ← argmax_a [ r + γ·V(s′) ]
      4.  Devuelve (π*, V)
    """

    # -------------------------------------------------------------------------
    # 1. INICIALIZACIÓN: V(s) ← 0 para todos los estados
    # -------------------------------------------------------------------------
    V = {s: 0.0 for s in env.state_space()}

    # -------------------------------------------------------------------------
    # 2. ITERACIÓN DE VALORES: actualizar V hasta convergencia (Δ < θ)
    # -------------------------------------------------------------------------
    while True:
        delta = 0.0  # Δ ← 0 (máximo cambio en esta iteración)

        for s in env.state_space():
            if env.is_terminal(s):         # omitimos terminales
                continue

            # ── 2.a Almacenar valor previo
            v = V[s]

            # ── 2.b Calcular el mejor retorno esperado
            best_q = max(
                (env.sim_step(s, a)[1] + gamma * V[env.sim_step(s, a)[0]])
                for a in env.actions(s)
            )

            # ── 2.c Actualizar V(s) ← best_q
            V[s] = best_q

            # ── 2.d Actualizar el máximo cambio observado Δ
            delta = max(delta, abs(v - best_q))

        # ── 2.e Criterio de parada
        if delta < theta:
            break

    # -------------------------------------------------------------------------
    # 3. DERIVAR POLÍTICA ÓPTIMA π*(s) ← argmax_a Q(s,a)
    # -------------------------------------------------------------------------
    policy = {}

    for s in env.state_space():
        if env.is_terminal(s):
            continue

        best_a, best_q = None, -float('inf')

        for a in env.actions(s):
            next_s, r = env.sim_step(s, a)
            q_sa = r + gamma * V[next_s]

            if q_sa > best_q:
                best_q, best_a = q_sa, a

        policy[s] = best_a

    # -------------------------------------------------------------------------
    # 4. SALIDA: política y función de valor óptimas
    # -------------------------------------------------------------------------
    return policy, V
