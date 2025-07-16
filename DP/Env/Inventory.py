import matplotlib.pyplot as plt


class InventoryEnv:
    """
    ============================================================================
    Entorno de Planeación de Inventario con Demanda Determinística (MIP-equivalente)
    ─────────────────────────────────────────────────────────────────────────────
    Modelo secuencial (tipo MDP determinista) que implementa el clásico problema
    de lotes de producción / almacenamiento con horizonte finito.

    Formulación:
        min  Σ_t (c_t * x_t + h_t * I_t)
        s.t. I_1 = I0 + x_1 − d_1
             I_t = I_{t−1} + x_t − d_t          t = 2 … n
             0 ≤ I_t ≤ Capacidad
             x_t ≥ 0                             (producir/pedir)
    
    Implementación en el entorno:
    ------------------------------
    • Estado     : tupla (t, S)
                   ▸ t → período actual (0 … n)
                   ▸ S → inventario disponible al inicio de t (≡ I_{t-1})
    
    • Acciones   : x ∈ {min_pedido … max_pedido}
                   ▸ min_pedido = max(0, demanda[t] − S)   (evita faltantes)
                   ▸ max_pedido = capacidad − S            (respeta límite superior)
                   Si min_pedido > max_pedido ⇒ problema infactible en ese estado.
    
    • Recompensa : −(costo_producción[t] * x + costo_almacenamiento[t] * I_t)
                   donde I_t = S + x − demanda[t]
    
    • Transición : determinista. Avanza a (t+1, I_t).
    
    • Episodio   : duración fija de n pasos (uno por período t = 0 … n−1).
    ============================================================================
    """

    # =========================================================================
    # 1. INICIALIZACIÓN DEL ENTORNO
    # =========================================================================
    def __init__(self, demand, production_costs, holding_costs, capacity, start_inventory=0):
        """
        Constructor del entorno.

        Parámetros:
        -----------
        demand            : lista de demandas d_t por período (longitud n)
        production_costs  : lista de costos de producción c_t por período (longitud n)
        holding_costs     : lista de costos de almacenamiento h_t por período (longitud n)
        capacity          : capacidad máxima del inventario (entero ≥ 0)
        start_inventory   : inventario inicial I0 (0 … capacity)
        """
        # --- 1.1 Validaciones de tamaño
        self.demand           = list(map(int, demand))
        self.production_costs = list(map(float, production_costs))
        self.holding_costs    = list(map(float, holding_costs))
        self.n                = len(self.demand)

        assert len(self.production_costs) == self.n,  \
            "`production_costs` debe tener la misma longitud que `demand`."
        assert len(self.holding_costs) == self.n,     \
            "`holding_costs` debe tener la misma longitud que `demand`."

        # --- 1.2 Parámetros del problema
        self.capacity        = int(capacity)
        assert self.capacity >= 0, "La capacidad debe ser no negativa."
        assert 0 <= start_inventory <= self.capacity, \
            "Inventario inicial fuera de rango [0, capacity]."
        self.start_inventory = int(start_inventory)

        # --- 1.3 Estado actual (se actualiza con reset() / step())
        self.t = None                 # Período actual (0 … n)
        self.S = None                 # Inventario al inicio de t
        self.total_reward = None      # Acumulador de recompensas

        # --- 1.4 Espacio de estados (pares válidos (t, S))
        self._states = [(t, s)
                        for t in range(self.n + 1)
                        for s in range(self.capacity + 1)]

    @property
    def state(self):
        """
        Estado actual del entorno, expresado como tupla (t, S).
        """
        return (self.t, self.S)

    def __repr__(self):
        """
        Representación legible del entorno.
        """
        return (f"InventoryEnv(Horizonte = {self.n}, "
                f"Capacidad = {self.capacity}, "
                f"#_Estados = {len(self._states)})")

    # =========================================================================
    # 2. MODELO DEL ENTORNO: ACCIONES Y TRANSICIONES
    # =========================================================================
    def actions(self, state=None):
        """
        Devuelve las acciones válidas (pedidos x) en un estado dado.

        Parámetros:
        -----------
        state : tupla (t, S), opcional
            Estado del entorno. Si es None, usa el estado actual.

        Retorna:
        --------
        List[int] con cantidades legales a pedir en el período t.
        Lista vacía en estado terminal (t == n).
        """
        if state is None:
            state = self.state

        t, S = state

        # --- Estado terminal: no hay acciones
        if t >= self.n:
            return []

        d_t = self.demand[t]

        # Pedir lo suficiente para no quedar en faltante
        min_order = max(0, d_t - S)

        # No rebasar capacidad
        max_order = self.capacity - S

        if min_order > max_order:
            raise ValueError(
                f"Infeasible: demanda {d_t} excede capacidad {self.capacity} "
                f"desde inventario {S} en período t={t}."
            )

        return list(range(min_order, max_order + 1))

    def step(self, x):
        """
        Ejecuta una acción (pedido / producción) `x` sobre el estado actual.

        Parámetros:
        -----------
        x : int
            Cantidad a producir/pedir en el período actual t.

        Retorna:
        --------
        state_next : tupla (t+1, S’)
            Nuevo estado tras la acción.
        reward : float
            Recompensa recibida = −(c_t * x + h_t * I_t).
        done : bool
            True si el episodio ha terminado (t+1 == n).
        """
        # Validar acción
        if x not in self.actions():
            raise ValueError(f"Acción ilegal {x} en el estado {self.state}.")

        t, S = self.t, self.S
        d_t  = self.demand[t]

        # Dinámica de inventario
        I_t = S + x - d_t              # Inventario al final del período t
        # Por construcción de acciones(), I_t ∈ [0, capacity]

        # Costo y recompensa
        c_t = self.production_costs[t]
        h_t = self.holding_costs[t]
        cost = c_t * x + h_t * I_t
        reward = -cost

        # Avanzar estado
        t_next = t + 1
        self.t = t_next
        self.S = I_t                   # Inventario final pasa a inventario inicial siguiente
        self.total_reward += reward
        done = (t_next == self.n)

        return self.state, reward, done

    def sim_step(self, state, x):
        """
        Simula una transición desde un estado dado sin afectar el entorno.

        Parámetros:
        -----------
        state : tupla (t, S)
            Estado desde el cual simular.
        x : int
            Cantidad a producir/pedir.

        Retorna:
        --------
        next_state : tupla (t+1, S’)
        reward : float
        """
        t, S = state
        if x not in self.actions(state):
            raise ValueError(f"Acción ilegal {x} en el estado {state}.")

        d_t = self.demand[t]
        I_t = S + x - d_t

        c_t = self.production_costs[t]
        h_t = self.holding_costs[t]
        cost = c_t * x + h_t * I_t
        reward = -cost

        return (t + 1, I_t), reward

    # =========================================================================
    # 3. FUNCIONES AUXILIARES
    # =========================================================================
    def reset(self):
        """
        Reinicia el entorno al estado inicial.

        Retorna:
        --------
        state : tupla (0, start_inventory)
        """
        self.t = 0
        self.S = self.start_inventory
        self.total_reward = 0
        return self.state

    def is_terminal(self, state=None):
        """
        Determina si un estado es terminal (t == n).

        Parámetro:
        ----------
        state : tupla (t, S), opcional
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
        Devuelve todos los estados posibles (t, S) del entorno.

        Retorna:
        --------
        List[tuple]
        """
        return self._states

    def report_from_policy(self, policy, month_labels=None):
        """
        Ejecuta un episodio completo usando una política dada, imprime métricas
        agregadas (FO, total producción, total inventario) y genera el gráfico
        solicitado (demanda área + barras apiladas inventario/producción).

        Parámetros
        ----------
        policy : dict[(t, S) -> x]
            Política determinista. Debe definir acción válida para cada estado
            que pueda alcanzarse al simular desde el estado inicial.
        month_labels : list[str], opcional
            Etiquetas a usar en el eje X. Si None, se usan meses abreviados
            ['ENE','FEB',...]; se truncan o se generan M1..Mn si n>12.

        Retorna
        -------
        obj_lp : float
            Valor de la función objetivo (costo total) Σ_t (c_t x_t + h_t I_t).
        produccion : list[float]
            Lista de cantidades pedidas/producidas por período.
        inventario : list[float]
            Lista del inventario *inicial* de cada período (estado S).
            (Cambia a inv_final si quieres reportar I_t post-demanda.)
        """
        # ------------------------------------------------------------------
        # 1. Preparar etiquetas de meses
        # ------------------------------------------------------------------
        default_months = ['ENE','FEB','MAR','ABR','MAY','JUN',
                          'JUL','AGO','SEP','OCT','NOV','DIC']
        if month_labels is None:
            if self.n <= 12:
                month_labels = default_months[:self.n]
            else:
                month_labels = [f"M{t+1}" for t in range(self.n)]
        else:
            # si el usuario pasó más etiquetas de las que se necesitan, truncar
            if len(month_labels) < self.n:
                raise ValueError("month_labels insuficientes para el horizonte n.")
            month_labels = list(month_labels)[:self.n]

        # ------------------------------------------------------------------
        # 2. Simular episodio completo según la política
        # ------------------------------------------------------------------
        state = self.reset()
        t, S = state

        ts = []
        inv_start = []   # inventario al inicio (estado S)
        orders = []      # acción x_t
        demands = []     # demanda d_t
        inv_end = []     # inventario final I_t

        while not self.is_terminal((t, S)):
            ts.append(t)
            inv_start.append(S)

            if (t, S) not in policy:
                raise KeyError(f"La política no define acción para el estado {(t, S)}.")

            x = policy[(t, S)]
            orders.append(x)

            d_t = self.demand[t]
            demands.append(d_t)

            (t, S), reward, _ = self.step(x)
            inv_end.append(self.S)  # después de step(), self.S == I_t

        # ------------------------------------------------------------------
        # 3. Calcular costo total (FO) y métricas agregadas
        # ------------------------------------------------------------------
        # Recalcular explícitamente para claridad
        costos_prod = []
        costos_inv= []
        for k in range(len(ts)):
            t = ts[k]
            x_t = orders[k]
            I_t = inv_end[k]  # inventario final del período t
            c_t = self.production_costs[t]
            h_t = self.holding_costs[t]

            costos_prod.append(c_t * x_t )
            costos_inv.append(h_t * I_t)

        total_costos_prod = round(sum(costos_prod) , 3)
        total_costos_inv = round(sum(costos_inv) , 3)
        obj_lp = round(total_costos_prod + total_costos_inv, 3)
        produccion = orders
        inventario = inv_start  # <- usa inventario inicial; cambia a inv_end si prefieres I_t

        # ------------------------------------------------------------------
        # 4. Imprimir métricas con el formato solicitado
        print(f'FO (valor total): {total_costos_prod} (Producción) + {total_costos_inv} (Inventario) = {obj_lp}')
        print(f'Cantidad de toneladas pedidas (valor total): {sum(produccion)}.')
        print(f'Cantidad de toneladas en inventario (valor total): {sum(inventario)}.')

        # ------------------------------------------------------------------
        # 5. Gráfico: Demanda (área) + Barras apiladas Inventario + Producción
        # ------------------------------------------------------------------
        import numpy as np
        import matplotlib.pyplot as plt

        # Series numéricas para graficar con fill_between
        x_pos = np.arange(self.n)
        demanda_list = demands
        produccion_list = produccion
        inventario_list = inventario

        plt.figure(figsize=(16, 10))

        # Área de la demanda
        plt.fill_between(x_pos, demanda_list, color='lightgray', label='Demanda [Ton]', alpha=0.6)

        # Barras apiladas: inventario inicial como base, producción encima
        plt.bar(x_pos, inventario_list, label='Inventario inicial [Ton]', color='darkorange', alpha=0.9)
        plt.bar(x_pos, produccion_list, bottom=inventario_list, label='Producción [Ton]', color='royalblue')

        plt.title('Plan de producción [Ton]')
        plt.ylabel('Toneladas')
        plt.xticks(x_pos, month_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # 6. Estructura de salida útil
        # ------------------------------------------------------------------
        trayecto = []
        for k in range(len(ts)):
            trayecto.append(dict(
                t=ts[k],
                mes=month_labels[k],
                inv_ini=inv_start[k],
                pedido=orders[k],
                demanda=demands[k],
                inv_fin=inv_end[k],
                costo=costos_prod[k]+costos_inv[k],
            ))

        return obj_lp, costos_prod, costos_inv, produccion, inventario, trayecto

