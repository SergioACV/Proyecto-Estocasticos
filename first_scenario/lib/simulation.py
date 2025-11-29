class Simulator:
    """Simulación dinámica con pasos discretos."""
    
    def __init__(self, model, initial_state, dt=1.0, total_time=168.0):
        self.model = model
        
        # Estado poblacional (no por nodo)
        self.S, self.I = initial_state
        self.N = self.S + self.I
        
        self.dt = dt
        self.total_time = total_time
        self.time = 0

        self.initialize_statistics()

    def initialize_statistics(self):
        self.t_values = [0.0]
        self.S_values = [self.S]
        self.I_values = [self.I]
        self.history = [(0.0, self.S, self.I)]
        self.total_disinfections = 0

    def step(self):
        """Avanza la simulación un paso en el tiempo."""
        
        dS = self.model.dS_dt(self.S, self.I, self.N)
        dI = self.model.dI_dt(self.S, self.I, self.N)
        
        # Actualizamos estado con método de Euler
        self.S += dS * self.dt 
        self.I += dI * self.dt

        # Asegurar que los valores no sean negativos (estabilidad numérica)
        self.S = max(0.0, min(self.N, self.S))
        self.I = max(0.0, min(self.N, self.I))
        
        # Normalizar para mantener S + I = N
        total = self.S + self.I
        if total > 0:
            self.S = (self.S / total) * self.N
            self.I = (self.I / total) * self.N

        # Count number of disinfections
        disinf = self.model.disinfections_per_dt(self.I) * self.dt
        self.total_disinfections += disinf

        self.time += self.dt

        self.register_history()
 
    def register_history(self):
        self.t_values.append(self.time)
        self.S_values.append(self.S)
        self.I_values.append(self.I)
        self.history.append((self.time, self.S, self.I))

    def compute_gain(self, P):
        """
        Calcula G = (1/T) * ∫ P(t) dt usando regla del trapecio.
        P es una lista de valores P(t).
        """
        total_area = 0.0

        # Regla del trapecio sobre todos los intervalos
        for i in range(1, len(self.t_values)):
            t_prev = self.t_values[i-1]
            t_curr = self.t_values[i]

            P_prev = P[i-1]
            P_curr = P[i]

            dt = t_curr - t_prev
            area = 0.5 * (P_prev + P_curr) * dt
            total_area += area

        gain = total_area / self.total_time
        
        # Protección contra NaN/Inf
        if not (abs(gain) < 1e10):  # Detectar NaN o valores muy grandes
            return 0.0
        
        return gain
    
    def compute_gain_attacker(self):
        """
        El atacante controla I(t)/N.
        """
        P_att = [I / self.N for I in self.I_values]
        return self.compute_gain(P_att)
    
    def compute_gain_defender(self):
        """
        El defensor controla S(t)/N.
        (en el modelo simple SI)
        """
        P_def = [S / self.N for S in self.S_values]
        return self.compute_gain(P_def)

    def compute_defender_cost(self):
        """
        Costo del defensor: inversión en mantener tasa de recuperación r.
        No depende de cuántas desinfecciones ocurran, sino del nivel de defensa.
        """
        return self.model.cost_defender
    
    def compute_attacker_cost(self):
        """
        Costo del atacante: inversión en desarrollar malware con tasa beta.
        """
        return self.model.cost_attacker

    def compute_defender_payoff(self):
        """
        Payoff del defensor = Ganancia - Costo
        """
        payoff = self.gain_defender - self.cost_defender
        # Protección contra valores inválidos
        if not (abs(payoff) < 1e10):
            return -1e6  # Penalización grande pero finita
        return payoff

    def compute_attacker_payoff(self):
        """
        Payoff del atacante = Ganancia - Costo
        """
        payoff = self.gain_attacker - self.cost_attacker
        # Protección contra valores inválidos
        if not (abs(payoff) < 1e10):
            return -1e6  # Penalización grande pero finita
        return payoff

    def run(self):
        """Corre la simulación completa."""
        while self.time < self.total_time:
            self.step()
        
        self.gain_attacker = self.compute_gain_attacker()
        self.gain_defender = self.compute_gain_defender()

        self.cost_defender = self.compute_defender_cost()
        self.cost_attacker = self.compute_attacker_cost()

        self.payoff_attacker = self.compute_attacker_payoff()
        self.payoff_defender = self.compute_defender_payoff()

