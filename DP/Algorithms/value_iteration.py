def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    Iteración de valores para control óptimo (algoritmo de la figura “Value Iteration”).
    
    1. **Inicializar** V(s) arbitrariamente (aquí 0) para todos los estados.  
    2. **Repetir**  
         Δ ← 0  
         Para cada s ∈ S:  
             v ← V(s)  
             V(s) ← max_a Σ_{s′,r} p(s′,r | s,a) [ r + γ V(s′) ]  
             Δ ← max(Δ, |v − V(s)|)  
       **Hasta que** Δ < θ.  
    3. **Derivar** la política determinista óptima  
         π(s) ← arg max_a Σ_{s′,r} p(s′,r | s,a) [ r + γ V(s′) ]  
    4. **Devolver** (π*, V).
    
    Parámetros
    ----------
    env : objeto del entorno
        Implementa `state_space()`, `actions(s)`, `sim_step(s, a)` e `is_terminal(s)`.
    gamma : float
        Factor de descuento γ ∈ [0,1].
    theta : float
        Umbral de convergencia θ (termina cuando el mayor cambio en V es < θ).
    
    Retorna
    -------
    policy : dict {estado: acción}
        Política determinista óptima π*.
    V : dict {estado: valor}
        Valor óptimo v*(s) para cada estado.
    """

    # --- 1. Inicialización: V(s) = 0 para todos los estados -----------------
    V = {s: 0.0 for s in env.state_space()}

    # --- 2. Bucle principal de iteración de valores -------------------------
    while True:
        delta = 0.0           # Δ ← 0  (máximo cambio observado esta pasada)

        # Para cada s ∈ S
        for s in env.state_space():
            if env.is_terminal(s):
                continue      # los estados terminales ya tienen V(s) fijo

            # --- v ← V(s) (almacenar valor anterior) ------------------------
            # max_a [ r + γ V(s′) ]  ←  Q óptimo de (s,a) en un paso
            best_q = max(
                (env.sim_step(s, a)[1] + gamma * V[env.sim_step(s, a)[0]]
                 for a in env.actions(s))
            )

            # Actualizar Δ ← max(Δ, |v − V(s)|)
            delta = max(delta, abs(best_q - V[s]))

            # V(s) ← best_q  (mejor retorno esperado al actuar óptimamente)
            V[s] = best_q

        # Condición de parada: Δ < θ  → convergió
        if delta < theta:
            break

    # --- 3. Derivar una política determinista óptima π* ---------------------
    policy = {}
    for s in env.state_space():
        if env.is_terminal(s):
            continue

        best_a, best_q = None, -float('inf')

        # π(s) ← argmax_a Q(s,a) donde Q(s,a) = r + γ V(s′)
        for a in env.actions(s):
            next_s, r = env.sim_step(s, a)
            q_sa = r + gamma * V[next_s]
            if q_sa > best_q:
                best_q, best_a = q_sa, a

        policy[s] = best_a

    # --- 4. Salida: política y función de valores óptimas -------------------
    return policy, V