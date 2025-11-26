class EpidemicModel:
    """Base general para modelos tipo SIR/SIS/etc."""
    def __init__(self, beta, erre):
        self.beta = beta
        self.erre = erre

    def dS_dt(self, S, I):
        """Ecuación diferencial para S."""
        # Luego ponemos la fórmula correcta según el modelo
        return -self.beta * I * S + self.erre + I

    def dI_dt(self, S, I):
        """Ecuación diferencial para I."""
        return self.beta * I * S - self.erre * I

