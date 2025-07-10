import matplotlib.pyplot as plt

class InventoryEnv:
    """
    ============================================================================
    Entorno de control de inventario con horizonte finito y demanda determinística
    ----------------------------------------------------------------------------
    Modelo clásico de decisión secuencial en logística. El objetivo es minimizar
    los costos de inventario y faltantes, modelados como función de recompensa.

    Estado   : (t, S) donde
                t → período actual (0 … n)
                S → inventario disponible al inicio del período (0 … capacity)

    Acciones : x ∈ {0, 1, ..., capacity - S}
                → unidades a pedir al proveedor en t

    Recompensa : -costo_total
                 donde
                     costo_total = costo_pedido + penalización
                     costo_pedido = 10 × x
                     penalización = 
                        • -10 × inventario_sobrante (costo de mantener)
                        • +20 × (faltante)^5       (costo severo por quiebre)

    Terminal : t == n
    ============================================================================
    """

    def __init__(self, demand, capacity, start_inventory=0):
        """
        Inicializa el entorno.

        ▸ demand          : lista con demanda determinista en cada período t
        ▸ capacity        : capacidad máxima del almacén
        ▸ start_inventory : inventario inicial disponible al inicio (en t=0)
        """
        assert 0 <= start_inventory <= capacity, "Inventario inicial fuera de rango."
        
        # --- Parámetros del entorno
        self.D         = list(demand)          # Demanda por período
        self.n         = len(self.D)           # Horizonte de planificación
        self.capacity  = int(capacity)         # Capacidad máxima de inventario
        self.start_inv = int(start_inventory)  # Inventario inicial

        # --- Estado interno (se actualiza en cada paso)
        self.t = None                 # Período actual
        self.S = None                 # Inventario actual (inicio de t)
        self.total_reward = None     # Acumulador de recompensas

        # --- Espacio total de estados (t, S)
        self._states = [(t, s)
                        for t in range(self.n + 1)
                        for s in range(self.capacity + 1)]

    @property
    def state(self):
        """ Devuelve el estado actual del entorno """
        return (self.t, self.S)

    def __repr__(self):
        """ Representación resumida del entorno """
        return (f"InventoryEnv(Horizonte = {self.n}, Capacidad = {self.capacity}, "
                f"#_Estados = {len(self._states)})")

    def actions(self, state=None):
        """
        Devuelve las acciones legales en un estado dado.

        ▸ Acción = unidades a pedir, máximo hasta llenar capacidad
        """
        if state is None:
            state = self.state
        t, S = state

        if t >= self.n:
            return []  # Estado terminal → no hay decisiones

        max_order = self.capacity - S
        return list(range(max_order + 1))

    def step(self, x):
        """
        Ejecuta acción x y avanza el entorno.

        ▸ x: cantidad pedida al proveedor
        Retorna: (nuevo estado, recompensa, done)
        """
        if x not in self.actions():
            raise ValueError(f"Illegal action {x} in state {self.state}")

        # --- Estado actual y parámetros del período
        t, S = self.t, self.S
        d = self.D[t]                             # Demanda del período t

        # --- Dinámica de inventario
        S_next = max(0, S + x - d)                # Inventario luego de satisfacer demanda
        S_next = min(self.capacity, S_next)       # Aplicar límite superior

        # --- Costos y penalizaciones
        order_cost = x * 10                       # Costo por pedido
        overstock = max(0, S + x - d)             # Exceso de inventario
        shortage  = max(0, d - (S + x))           # Faltante

        k_h = -10                                 # Holding cost (negativo: castigo)
        k_s = 20                                  # Penalización por faltante (severo)
        cost = order_cost + (k_h * overstock) + k_s * shortage**5

        reward = -cost                            # ↔ minimizar costo ≡ maximizar reward

        # --- Avanzar estado
        t_next = t + 1
        self.t, self.S = t_next, S_next
        self.total_reward += reward
        done = (t_next == self.n)

        return self.state, reward, done

    def sim_step(self, state, x):
        """
        Simula una transición desde un estado dado con acción x.
        No modifica el entorno interno.

        Retorna: (siguiente estado, reward)
        """
        t, S = state
        if x not in self.actions(state):
            raise ValueError(f"Illegal action {x} in state {state}")

        d = self.D[t]
        S_next = min(self.capacity, max(0, S + x - d))

        order_cost = x * 10
        overstock  = max(0, S + x - d)
        shortage   = max(0, d - (S + x))
        k_h = -10
        k_s = 20
        cost = order_cost + (k_h * overstock) + k_s * shortage**5
        reward = -cost

        return (t + 1, S_next), reward

    def reset(self):
        """ Reinicia el entorno al estado inicial """
        self.t = 0
        self.S = self.start_inv
        self.total_reward = 0
        return self.state

    def is_terminal(self, state=None):
        """ Verifica si un estado es terminal """
        if state is None:
            state = self.state
        return state[0] >= self.n

    def state_space(self):
        """ Devuelve el espacio total de estados posibles """
        return self._states

    def report_from_policy(self, policy):
        """
        Simula un episodio aplicando la política dada, imprime resumen
        y visualiza la evolución del inventario.

        ▸ policy: dict[(t, S)] → acción óptima x
        """
        # --- Inicializar entorno
        state = self.reset()
        t, S = state

        # --- Traza por período
        ts, demand_t, orders_t, invent_t, shortage_t = [], [], [], [], []

        # --- Ejecutar episodio completo
        while t < self.n:
            ts.append(t)
            invent_t.append(S)
            d = self.D[t]
            demand_t.append(d)

            a = policy.get((t, S), 0)
            orders_t.append(a)
            shortage_t.append(max(0, d - (S + a)))

            (t, S), _, _ = self.step(a)

        total_reward = self.total_reward

        # --- Mostrar resumen tabular
        print("Resumen por período:")
        print(f"{'t':>3} {'Inv.':>6} {'Dem.':>6} {'Ord.':>6} {'Falt.':>6}")
        for i in range(len(ts)):
            print(f"{ts[i]:>3} {invent_t[i]:>6} {demand_t[i]:>6} {orders_t[i]:>6} {shortage_t[i]:>6}")
        print(f"FO (recompensa total): {total_reward:.2f}")

        # --- Visualizar resultado
        plt.figure()
        plt.plot(ts, demand_t, marker='o', label='Demanda D[t]')
        plt.plot(ts, orders_t, marker='s', label='Pedido óptimo aₜ')
        plt.plot(ts, invent_t, marker='^', label='Inventario S (inicio)')
        plt.plot(ts, shortage_t, marker='d', label='Faltante post–pedido')

        plt.title("Demanda, pedido, inventario y faltante – política óptima")
        plt.xlabel("Período t")
        plt.ylabel("Unidades")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
