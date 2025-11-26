class EpidemicModel:
    """Base general para modelos tipo SIR/SIS/etc."""
    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def dS_dt(self, S, I, R):
        """Ecuación diferencial para S."""
        # Luego ponemos la fórmula correcta según el modelo
        return 0

    def dI_dt(self, S, I, R):
        """Ecuación diferencial para I."""
        return 0

    def dR_dt(self, S, I, R):
        """Ecuación diferencial para R (puede ser cero)."""
        return 0
