class EpidemicModel:
    """Base general para modelos tipo SIR/SIS/etc."""
    
    def __init__(self, beta, r):
        self.beta = beta
        self.r = r
        
        # AJUSTE DE COSTOS:
        # Aumentamos un poco el peso del costo para que no siempre gane la estrategia máxima.
        # En el paper usan coeficientes específicos, aquí aproximamos para balancear el juego.
        self.cost_attacker_coeff = 0.05 
        self.cost_defender_coeff = 0.05
        
        self.cost_attacker = self.get_cost_attacker()
        self.cost_defender = self.get_cost_defender()
    
    def disinfections_per_dt(self, I):
        """Número de nodos que se recuperan por unidad de tiempo"""
        return self.r * I

    def get_cost_attacker(self):
        # Costo lineal respecto a la tasa de infección
        return self.cost_attacker_coeff * self.beta

    def get_cost_defender(self):
        # Costo lineal respecto a la tasa de recuperación
        return self.cost_defender_coeff * self.r

    def dS_dt(self, S, I, N):
        """Ecuación diferencial para S (Modelo SIS)"""
        # dS/dt = -beta*(I/N)*S + r*I
        return -self.beta * (I / N) * S + self.r * I

    def dI_dt(self, S, I, N):
        """Ecuación diferencial para I (Modelo SIS)"""
        # dI/dt = beta*(I/N)*S - r*I
        return self.beta * (I / N) * S - self.r * I