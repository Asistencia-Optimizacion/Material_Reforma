import matplotlib.pyplot as plt

class InventoryEnv:
    """
    Entorno de control de inventario con horizonte finito
    y demanda **determinística**.

    Estado   : tupla (t, S)
               t - período actual                    (0 … n)
               S - inventario disponible al inicio   (0 … capacity)

    Acciones : x ∈ {0, 1, …, capacity - S}
               (unidades a ordenar al proveedor).   No puede exceder el
               espacio libre en almacén.

    Recompensa : - costo_inmediato
                 donde  costo = order_cost + holding/shortage_penalty
                 order_cost  = x
                holding/shortage_penalty = (S + x - D[t])²
                 Se devuelve el negativo para que **maximizar** la recompensa sea equivalente a **minimizar** el costo.

    Episodio : n períodos → termina cuando t == n
    """

    # ------------------------------------------------------------------
    # Inicializar
    def __init__(self, demand, capacity, start_inventory=0):
        """
        demand          : lista con la demanda determinística por período
        capacity        : capacidad máxima del almacén (entero ≥ 0)
        start_inventory : stock inicial (0 … capacity)
        """
        assert 0 <= start_inventory <= capacity, "Inventario inicial fuera de rango."
        self.D          = list(demand)               # Demanda D[t]
        self.n          = len(self.D)                # Horizonte finito
        self.capacity   = int(capacity)              # Capacidad máxima
        self.start_inv  = int(start_inventory)       # Inventario inicial

        # ------ Estado actual del entorno
        self.t = None                                # Período actual
        self.S = None                                # Inventario actual
        self.total_reward = None                     # Recompensa acumulada

        # ------ Espacio de estados (t, S)   (incluye estado terminal t == n)
        self._states = [(t, s)
                        for t in range(self.n + 1)
                        for s in range(self.capacity + 1)]

    # ------------------------------------------------------------------
    # Propiedad de solo lectura para exponer el estado actual
    @property
    def state(self):
        return (self.t, self.S)

    # ------------------------------------------------------------------
    # Representación legible
    def __repr__(self):
        return (f"InventoryEnv(Horizonte = {self.n}, Capacidad = {self.capacity}, "
                f"#_Estados = {len(self._states)})")

    # ------------------------------------------------------------------
    # Conjunto de acciones legales en un estado dado
    def actions(self, state=None):
        if state is None:
            state = self.state
        t, S = state

        # Si estamos en el estado terminal no hay acciones
        if t >= self.n:
            return []

        max_order = self.capacity - S
        return list(range(max_order + 1))            # 0 … espacio libre

    # ------------------------------------------------------------------
    # Dinámica del entorno — aplicar acción real
    def step(self, x):
        """
        Ejecuta la orden x en el estado actual, actualiza el entorno
        y devuelve (nuevo_estado, reward, done).
        """
        # Comprobación de factibilidad
        if x not in self.actions():
            raise ValueError(f"Illegal action {x} in state {self.state}")

        # Desempaquetar estado actual
        t, S = self.t, self.S
        d    = self.D[t]                             # Demanda de este período

        # ------------ Inventario posterior a la demanda -------------
        S_next = max(0, S + x - d)                   # inventario neto
        S_next = min(self.capacity, S_next)          # límite de capacidad

        # ------------ Costos y recompensa ---------------------------
        order_cost   = x * 10
        overstock   = max(0, S + x - d)
        shortage    = max(0, d - (S + x))
        k_h = -10
        k_s = 20
        cost = order_cost + (k_h * overstock) + k_s * shortage**5
        reward       = -cost                         # maximizar recompensa ↔ minimizar costo

        # ------------ Avanzar período -------------------------------
        t_next = t + 1

        # ------------ Actualizar entorno ----------------------------
        self.t, self.S = t_next, S_next
        self.total_reward += reward
        done = (t_next == self.n)

        return self.state, reward, done

    # ------------------------------------------------------------------
    # Dinámica “offline” — simular sin modificar el entorno
    def sim_step(self, state, x):
        """
        Devuelve (estado_siguiente, reward) desde `state` al aplicar `x`,
        sin alterar el estado interno del entorno.
        """
        t, S = state
        if x not in self.actions(state):
            raise ValueError(f"Illegal action {x} in state {state}")

        d      = self.D[t]
        S_next = min(self.capacity, max(0, S + x - d))
        order_cost   = x * 10
        overstock   = max(0, S + x - d)
        shortage    = max(0, d - (S + x))
        k_h = -10
        k_s = 20
        cost = order_cost + (k_h * overstock) + k_s * shortage**5
        reward       = -cost
        return (t + 1, S_next), reward

    # ------------------------------------------------------------------
    # Reiniciar el episodio
    def reset(self):
        self.t = 0
        self.S = self.start_inv
        self.total_reward = 0
        return self.state

    # ------------------------------------------------------------------
    # Estado terminal
    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        return state[0] >= self.n

    # ------------------------------------------------------------------
    # Espacio de búsqueda
    def state_space(self):
        return self._states

    # ---------------------------------------------------------------------
    # Reportar Resultados
    def report_from_policy(self, policy):
        """
        Simula un episodio con la política dada y muestra:
        - Tabla de decisiones y estados
        - Recompensa total
        - Gráfico con demanda, pedido, inventario y faltantes
        """
        # --- 1. Reiniciar entorno
        state = self.reset()
        t, S = state

        # --- 2. Acumuladores
        ts, demand_t, orders_t, invent_t, shortage_t = [], [], [], [], []

        # --- 3. Ejecutar episodio
        while t < self.n:
            ts.append(t)
            invent_t.append(S)
            d = self.D[t]
            demand_t.append(d)

            a = policy.get((t, S), 0)
            orders_t.append(a)
            shortage_t.append(max(0, d - (S + a)))
            (t, S), reward, _ = self.step(a)

        total_reward = self.total_reward

        # --- 4. Imprimir resumen tabular
        print("Resumen por período:")
        print(f"{'t':>3} {'Inv.':>6} {'Dem.':>6} {'Ord.':>6} {'Falt.':>6}")
        for i in range(len(ts)):
            print(f"{ts[i]:>3} {invent_t[i]:>6} {demand_t[i]:>6} {orders_t[i]:>6} {shortage_t[i]:>6}")

        print(f"FO (recompensa total): {total_reward:.2f}")

        # --- 5. Gráfico de la trayectoria
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