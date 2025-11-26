class EpidemicModel:
    """Base general para modelos tipo SIR/SIS/etc."""
    def __init__(self, beta, r):
        self.beta = beta
        self.r = r

    def dS_dt(self, S, I, N):
        """Ecuación diferencial para S (versión desnormalizada)."""
        return -self.beta * (I / N) * S + self.r * I

    def dI_dt(self, S, I, N):
        """Ecuación diferencial para I (versión desnormalizada)."""
        return self.beta * (I / N) * S - self.r * I


