from policy_evaluation import policy_evaluation

def policy_iteration(env, policy, gamma: float = 1.0, theta: float = 1e-8, report: bool = False):
    """
    ============================================================================
    Algoritmo : Iteración de Políticas (versión determinista, in-place)
    ─────────────────────────────────────────────────────────────────────────────
    Entradas
      • env      : entorno que implementa
                     ▸ state_space()            → Iterable[State]
                     ▸ is_terminal(s)           → bool
                     ▸ sim_step(s, a)           → (s', r)
                     ▸ actions(s)               → Iterable[Action]
      • policy   : dict[State, Action] — política inicial (determinista)
      • gamma    : factor de descuento γ ∈ (0,1]
      • theta    : precisión usada en policy_evaluation
      • report   : pasa la señal a policy_evaluation para trazas

    Salida
      • policy   : política óptima π*
      • V        : valor óptimo V* asociado a π*
    ============================================================================

    Descripción resumida
    ---------------------------------------------------------------------------
      1.  Evalúa la política actual        →  V ← v_π          (policy_evaluation)
      2.  Mejora la política usando V      →  π ← greedy(V)
      3.  Si la política no cambió en ningún estado ⇒ convergencia
          de lo contrario, repetir desde 1
    """

    # -------------------------------------------------------------------------
    #  Bucle principal: repetir hasta que la política se vuelva estable
    # -------------------------------------------------------------------------
    while True:

        # 1. EVALUACIÓN DE LA POLÍTICA  (paso de predicción)
        #    Calcula V ← v_π hasta |Δ|<θ
        V = policy_evaluation(env, policy, gamma, theta, report)

        # 2. MEJORA DE LA POLÍTICA  (paso de control)
        policy_stable = True

        for s in env.state_space():

            if env.is_terminal(s):          # omitimos estados terminales
                continue

            old_a = policy[s]               # acción previa de la política

            # ── 2.a Buscar la acción greedy respecto a V(s)
            best_q = -float('inf')
            best_a = None

            for a in env.actions(s):
                next_s, r = env.sim_step(s, a)          # transición determinista
                q_sa = r + gamma * V[next_s]            # Qπ(s,a) = r + γ V(s')

                if q_sa > best_q:                       # maximizar Qπ(s,a)
                    best_q, best_a = q_sa, a

            # ── 2.b Actualizar π(s) ← argmax_a Qπ(s,a)
            policy[s] = best_a

            # ── 2.c Comprobar si cambió la acción
            if best_a != old_a:
                policy_stable = False

        # 3. CRITERIO DE PARADA: π no cambió en ningún estado
        if policy_stable:
            return policy, V
