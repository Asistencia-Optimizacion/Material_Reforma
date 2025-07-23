from policy_evaluation import policy_evaluation

"""
# Alterna evaluación y mejora hasta que la política se estabiliza
# Entradas:
#   S, A(s), p(s',r|s,a), γ, θ
# Salidas:
#   π*  : política óptima
#   V*  : función de valor óptima
# -------------------------------------------

# 1. Inicialización
π(s) ← acción cualquiera  para todo s ∈ S
V(s) ← 0

bucle:
    # 2. Evaluar la política actual π  (igual que el bloque anterior)
    repetir
        Δ ← 0
        para cada s ∈ S:
            v ← V(s)
            a ← π(s)
            V(s) ← Σ_{s',r} p(s',r | s,a) · [ r + γ · V(s') ]
            Δ ← max(Δ, |v − V(s)|)
    hasta que Δ < θ

    # 3. Mejorar la política
    policy_stable ← true
    para cada s ∈ S:
        old ← π(s)
        π(s) ← argmax_a Σ_{s',r} p(s',r | s,a) · [ r + γ · V(s') ]
        si old ≠ π(s):
            policy_stable ← false

    si policy_stable:
        break

return π, V
"""


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
    # 1. BUCLE PRINCIPAL: iterar hasta estabilizar la política
    # -------------------------------------------------------------------------
    while True:

        # ── 1.a Evaluación de la política actual: V ← v_π
        V = policy_evaluation(env, policy, gamma, theta, report)

        # ── 1.b Mejora de la política: π ← greedy(V)
        policy_stable = True  # bandera para verificar cambios en π

        for s in env.state_space():

            if env.is_terminal(s):  # omitimos terminales
                continue

            old_a = policy[s]       # acción actual según la política

            # ── Buscar la mejor acción (greedy respecto a V)
            best_q = -float('inf')
            best_a = None

            for a in env.actions(s):
                next_s, r = env.sim_step(s, a)     # transición determinista s ─a→ s'
                q_sa = r + gamma * V[next_s]       # Qπ(s,a) = r + γ·V(s')

                if q_sa > best_q:                  # maximizar Qπ(s,a)
                    best_q, best_a = q_sa, a

            # ── Actualizar π(s) ← argmax_a Qπ(s,a)
            policy[s] = best_a

            # ── Si la acción cambió, marcar política como inestable
            if best_a != old_a:
                policy_stable = False

        # ── 1.c Verificar convergencia: si π no cambió en ningún estado
        if policy_stable:
            return policy, V
