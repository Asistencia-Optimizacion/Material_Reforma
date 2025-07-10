def policy_evaluation(env, policy, gamma: float = 1.0, theta: float = 1e-6, report: bool = False):
    """
    ============================================================================
    Evaluación de una política  (determinista, entorno determinista)
    ─────────────────────────────────────────────────────────────────────────────
    Entradas
      • env      : entorno que implementa
                     ▸ state_space()         → Iterable[State]
                     ▸ is_terminal(s)        → bool
                     ▸ sim_step(s, a)        → (s', r)
      • policy   : dict[State, Action] — política determinista (π(a|s)=𝟙{a=policy[s]})
      • gamma    : factor de descuento γ ∈ (0,1]
      • theta    : umbral de convergencia
      • report   : si es True, muestra trazas de cada iteración

    Salida
      • policy   : se devuelve la misma política recibida (conveniencia)
      • V        : dict[State, float] — aproximación de v_π
    ============================================================================

    Notas
    -----
    * La función **no** modifica la política; simplemente la re-expone para que
      quien la llame pueda encadenar `policy, V = policy_evaluation(...)`
      sin perder la referencia.
    """
    # 1. Inicializar V(s)=0 para todos los estados
    V   = {s: 0.0 for s in env.state_space()}
    cnt = 1  # contador de iteraciones (solo para ‘report’)

    # 2. Iterar hasta la convergencia (Δ < θ)
    while True:

        if report:
            print(f"\nIteración {cnt}")

        delta = 0.0  # Δ ← 0

        # 2.a Recorremos cada estado s ∈ 𝒮
        for s in env.state_space():
            if env.is_terminal(s):          # omitimos terminales
                continue

            v      = V[s]                   # valor anterior
            a      = policy[s]              # acción dictada por π
            next_s, r = env.sim_step(s, a)  # transición única (determinista)

            # Bellman: v(s) ← r + γ·V(s')
            new_v = r + gamma * V[next_s]
            V[s]  = new_v

            # Actualizamos el máximo cambio Δ
            delta = max(delta, abs(v - new_v))

            if report and v != new_v:
                print(f"  s:{s} ─a:{a}→ s':{next_s} | "
                      f"r={r:+.3f}, γV(s')={gamma*V[next_s]:+.3f} → V(s)={new_v:+.3f}")

        if delta < theta:                   # criterio de parada
            break

        cnt += 1

    # 3. Devolver el valor estimado
    return V
