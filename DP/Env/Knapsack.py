class KnapsackEnv:

    """
    Entorno del problema de la mochila 0-1.

    Estado  : tupla (i, c)
              i - índice del siguiente objeto a considerar   (0 … n).
              c - capacidad restante                         (0 … W).

    Acciones: "tomar", "omitir"
              "tomar" es legal solo si pesos[i] <= c.

    Recompensa: valor del objeto cuando se toma, 0 en caso contrario.

    Episodio: exactamente n decisiones → termina cuando i == n.
    """


    # ---------------------------------------------------------------------
    # Inicializar.
    def __init__(self, weights, values, capacity):
        
        assert len(weights) == len(values), "Todo objeto debe tener peso y valor."

        # 1. --- Información del problema:
        self.weights  = list(weights)           # Pesos del los objetos
        self.values   = list(values)            # Valores de los objetos
        self.capacity = int(capacity)           # Capacidad de la mochila
        self.n        = len(self.weights)       # Número de objetos que hay.
        
        # 2. --- Estado actual del problema:
        self.i = None               # Indice del elemento a evaluar
        self.c = None               # Capacidad disponible
        self.total_reward = None    # Retorno total
        
        # 3. --- Todos los posibles estados. 
        self._states = [(i, c) for i in range(self.n + 1) for c in range(self.capacity + 1)]

    # 4. --- El estado se define siempre como pareja: elemento y capacidad restante
    @property
    def state(self):
        return (self.i, self.c)

    
    # ---------------------------------------------------------------------
    # Imprimir.
    def __repr__(self):
        return (f"KnapsackEnv(#_Objetos = {self.n}, Capacidad = {self.capacity}, "f"#_Estados = {len(self._states)})")


    # ---------------------------------------------------------------------
    # Acciones sobre el ambiente. 

    # 1. --- Dado un estado, que acciones puedo hcaer (legales).
    def actions(self, state=None):

        if state is None: state = self.state            # Si no me pasaron un estado, tomo el del ambiente por defecto.
        i, c = state                                    # Sino, toma el objeto:capacidad_restante del estado.

        if i >= self.n: return []                       # Si ya acabe de revisar todos los objetos, no hay acciones.
        
        acts = ["skip"]                                 # Si quedan acciones por revisar, siempre lo puedo saltar.
        if self.weights[i] <= c: acts.append("take")    # Si cabe el la maleta (peso menor a la capacidad restante) añado la acción tomar.
        return acts
    
    # 2. --- Aplicar una acción sobre el estado actual (Función de transisición).
    def step(self, action):

        # Revisamos que no haya trampa con la acción a hacer. (Esto es factibilidad del problema).
        if action not in self.actions():
            raise ValueError(f"Illegal action {action!r} in state {self.state}")
        
        i, c = self.i, self.c               # Si la acción es legal, recuperamos la información del problema.
        
        # Si la acción es tomar: I. La recompensa es su valor, y II. se debe descontar el peso.
        if action == "take":
            reward = self.values[i]
            c -= self.weights[i]
        
        # Si la acción es omitir: I. No hay recompensa y II. no pasa nada con la capacidad.
        else:
            reward = 0
        
        # Siempre avanzo al siguiente objeto, con la capacidad del morral actualizada (capacidad restante).
        i += 1
        self.i, self.c = i, c
        
        # Calculo la utilidad total y reviso si ya acabe.
        self.total_reward += reward
        done = (i == self.n)

        return self.state, reward, done
    
    # 3. --- Simular una acción sobre el un estado dado (Función de transisición).
    def sim_step(self, state, action):

        # Recuperamos la información del problema.
        i, c = state

        # Si la acción es tomar: I. La recompensa es su valor, y II. se debe descontar el peso.
        if action == "take":
            reward = self.values[i]
            c -= self.weights[i]

        # Si la acción es omitir: I. No hay recompensa y II. no pasa nada con la capacidad.
        else:
            reward = 0
        
        # Siempre avanzo al siguiente objeto, con la capacidad del morral actualizada (capacidad restante).
        return (i + 1, c), reward


    # ---------------------------------------------------------------------
    # Reiniciar
    def reset(self):

        self.i = 0                  # Empezamos en el objeto 0
        self.c = self.capacity      # Reiniciamos la capacidad al máximo.
        self.total_reward = 0       # Y el total de la recompensa es 0.
        
        return self.state

    
    # ---------------------------------------------------------------------
    # Información

    # 1. -- Si estamos en un estado terminal / final.
    def is_terminal(self, state=None):
        if state is None: state = self.state
        return state[0] >= self.n
    
    # 2. -- Devuelve el espacio de busqueda.
    def state_space(self):
        return self._states