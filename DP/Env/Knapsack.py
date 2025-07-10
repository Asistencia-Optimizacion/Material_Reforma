class KnapsackEnv:
    """
    ============================================================================
    Entorno del Problema de la Mochila 0-1 (Knapsack)
    ─────────────────────────────────────────────────────────────────────────────
    Simula un entorno tipo MDP donde se toman decisiones secuenciales sobre 
    incluir o no objetos en una mochila de capacidad limitada.

    • Estado     : tupla (i, c)
                   ▸ i → índice del objeto actual (0 … n)
                   ▸ c → capacidad restante (0 … W)

    • Acciones   : "take", "skip"
                   ▸ "take" solo es legal si el objeto cabe (peso[i] ≤ c)

    • Recompensa : valor del objeto si se toma, 0 si se omite

    • Transición : determinista — avanza al siguiente objeto siempre (i ← i+1)

    • Episodio   : duración fija de n pasos (uno por cada objeto)

    Uso:
      - Compatible con programación dinámica (VI, PI)
      - Permite análisis y visualización de políticas
    ============================================================================
    """

    # =========================================================================
    # 1. INICIALIZACIÓN DEL ENTORNO
    # =========================================================================
    def __init__(self, weights, values, capacity):
        """
        Constructor del entorno.
        
        Parámetros:
        -----------
        weights  : lista de pesos de cada objeto
        values   : lista de valores (empleos) de cada objeto
        capacity : capacidad total de la mochila
        """
        assert len(weights) == len(values), "Todo objeto debe tener peso y valor."

        # --- 1.1 Parámetros del problema
        self.weights  = list(weights)             # Pesos de los objetos
        self.values   = list(values)              # Valores o beneficios (empleos generados)
        self.capacity = int(capacity)             # Capacidad máxima disponible
        self.n        = len(weights)              # Número total de objetos

        # --- 1.2 Estado actual (se actualiza con reset() o step())
        self.i = None                             # Índice del objeto actual (0 … n)
        self.c = None                             # Capacidad restante
        self.total_reward = None                  # Suma de recompensas acumuladas

        # --- 1.3 Espacio de estados (pares válidos (i, c))
        self._states = [(i, c) for i in range(self.n + 1)
                             for c in range(self.capacity + 1)]

    @property
    def state(self):
        """
        Estado actual del entorno, expresado como tupla (i, c)
        """
        return (self.i, self.c)

    def __repr__(self):
        """
        Representación legible del entorno
        """
        return (f"KnapsackEnv(#_Objetos = {self.n}, "
                f"Capacidad = {self.capacity}, "
                f"#_Estados = {len(self._states)})")

    # =========================================================================
    # 2. MODELO DEL ENTORNO: ACCIONES Y TRANSICIONES
    # =========================================================================

    def actions(self, state=None):
        """
        Devuelve las acciones válidas en un estado dado.

        Parámetros:
        -----------
        state : tupla (i, c)
            Estado del entorno. Si es None, usa el estado actual.

        Retorna:
        --------
        List[str] con acciones legales ("skip", "take")
        """
        if state is None:
            state = self.state

        i, c = state

        # --- Si ya no hay objetos por considerar, no hay acciones posibles
        if i >= self.n:
            return []

        # Siempre se puede omitir el objeto
        acts = ["skip"]

        # Solo se puede tomar si cabe dentro del presupuesto
        if self.weights[i] <= c:
            acts.append("take")

        return acts

    def step(self, action):
        """
        Ejecuta una acción sobre el estado actual del entorno.

        Parámetro:
        ----------
        action : str
            Acción a ejecutar: "take" o "skip"

        Retorna:
        --------
        state_next : tupla
            Nuevo estado tras la acción
        reward : float
            Recompensa recibida
        done : bool
            True si el episodio ha terminado (i == n)
        """
        # Validar que la acción sea legal
        if action not in self.actions():
            raise ValueError(f"Acción ilegal {action!r} en el estado {self.state}")

        i, c = self.i, self.c

        # Aplicar efecto de la acción
        if action == "take":
            reward = self.values[i]
            c -= self.weights[i]
        else:
            reward = 0

        # Avanzar al siguiente objeto
        i += 1
        self.i, self.c = i, c

        # Acumular recompensa y chequear finalización
        self.total_reward += reward
        done = (i == self.n)

        return self.state, reward, done

    def sim_step(self, state, action):
        """
        Simula una acción en un estado dado (sin afectar el entorno actual).

        Parámetros:
        -----------
        state : tupla (i, c)
            Estado desde el cual simular

        action : str
            Acción a simular ("take" o "skip")

        Retorna:
        --------
        next_state : tupla (i+1, c’)
        reward : float
        """
        i, c = state

        if action == "take":
            reward = self.values[i]
            c -= self.weights[i]
        else:
            reward = 0

        return (i + 1, c), reward

    # =========================================================================
    # 3. FUNCIONES AUXILIARES
    # =========================================================================

    def reset(self):
        """
        Reinicia el entorno al estado inicial.

        Retorna:
        --------
        state : tupla (0, capacidad)
        """
        self.i = 0
        self.c = self.capacity
        self.total_reward = 0
        return self.state

    def is_terminal(self, state=None):
        """
        Determina si un estado es terminal (i == n).

        Parámetro:
        ----------
        state : tupla, opcional
            Si es None, usa el estado actual.

        Retorna:
        --------
        bool
        """
        if state is None:
            state = self.state
        return state[0] >= self.n

    def state_space(self):
        """
        Devuelve todos los estados posibles (i, c) del entorno.

        Retorna:
        --------
        List[tuple]
        """
        return self._states

    def report_from_policy(self, policy):
        """
        Ejecuta un episodio completo usando una política dada
        y reporta el desempeño (valor, objetos tomados, uso del presupuesto).

        Parámetros:
        -----------
        policy : dict[state → action]
            Política determinista aplicada

        Retorna:
        --------
        valor_total      : suma de valores recolectados
        objetos_tomados  : lista de índices tomados
        peso_total       : presupuesto efectivamente utilizado
        """

        # --- 1. Inicializar entorno y acumuladores
        state = self.reset()
        objetos_tomados = []
        peso_total = 0
        valor_total = 0

        # --- 2. Ejecutar episodio según la política
        while not self.is_terminal(state):
            action = policy[state]
            if action == "take":
                idx = state[0]
                objetos_tomados.append(idx)
                peso_total += self.weights[idx]
                valor_total += self.values[idx]
            state, _, _ = self.step(action)

        # --- 3. Imprimir resultados (útil para depuración o validación)
        print("Objetos seleccionados:")
        for idx in objetos_tomados:
            w, v = self.weights[idx], self.values[idx]
            print(f"  • Obj {idx:>2}: peso={w}, valor={v}")

        print(f"FO (valor total):    {valor_total}")
        print(f"Presupuesto usado:   {peso_total}/{self.capacity}")

        return valor_total, objetos_tomados, peso_total
