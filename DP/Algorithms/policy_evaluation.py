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

    * El valor v_π se estima iterativamente usando la ecuación de Bellman
      para políticas fijas:
            v(s) ← r + γ·v(s')
      en entornos deterministas.
    """

    # -------------------------------------------------------------------------
    # 1. INICIALIZACIÓN: V(s) ← 0 para todos los estados
    # -------------------------------------------------------------------------
    V   = {s: 0.0 for s in env.state_space()}   # función valor inicial
    cnt = 1                                     # contador de iteraciones (solo para trazas)

    # -------------------------------------------------------------------------
    # 2. ITERACIÓN PRINCIPAL: aplicar Bellman hasta convergencia (|Δ| < θ)
    # -------------------------------------------------------------------------
    while True:

        if report:
            print(f"\nIteración {cnt}")

        delta = 0.0  # Δ ← 0

        # ── 2.a Recorremos todos los estados del entorno
        for s in env.state_space():

            if env.is_terminal(s):        # omitimos estados terminales
                continue

            v = V[s]                      # valor anterior
            a = policy[s]                 # acción según política π
            next_s, r = env.sim_step(s, a)   # transición determinista: s ─a→ s'

            # ── 2.b Actualizar valor de estado: Bellman para políticas fijas
            new_v = r + gamma * V[next_s]
            V[s]  = new_v

            # ── 2.c Actualizar máximo cambio observado
            delta = max(delta, abs(v - new_v))

            # ── 2.d (Opcional) Mostrar trazas si hubo cambio
            if report and v != new_v:
                print(f"  s:{s} ─a:{a}→ s':{next_s} | "
                      f"r={r:+.3f}, γV(s')={gamma*V[next_s]:+.3f} → V(s)={new_v:+.3f}")

        # ── 2.e Criterio de parada: convergencia si Δ < θ
        if delta < theta:
            break

        cnt += 1

    # -------------------------------------------------------------------------
    # 3. SALIDA: devolver V (y re-exponer la política para conveniencia)
    # -------------------------------------------------------------------------
    return V
