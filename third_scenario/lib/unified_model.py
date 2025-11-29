# lib/unified_model.py

class UnifiedEpidemicModel:
    """
    Modelo Unificado del paper (Caso 4).
    Tres compartimentos: S, I, R.
    
    Parámetros:
        beta  - tasa de infección
        gamma - tasa de inmunización de susceptibles
        r     - tasa de desinfección (SIS)
        lambda_ - tasa de desinfección + inmunización
        k0, k1 - coeficientes de costo del defensor y atacante
    """

    def __init__(self, beta, gamma, r, lambda_, k0=0.01, k1=0.01):
        self.beta = beta
        self.gamma = gamma
        self.r = r
        self.lambda_ = lambda_
        self.k0 = k0
        self.k1 = k1

    # Ecuaciones dinámicas del paper
    def dS_dt(self, S, I, R, N):
        return self.r * I - self.beta * S * I - self.gamma * S

    def dI_dt(self, S, I, R, N):
        return self.beta * S * I - self.lambda_ * I - self.r * I

    def dR_dt(self, S, I, R, N):
        return self.gamma * S + self.lambda_ * I

    # Costos
    @property
    def cost_defender(self):
        return self.k0 * (self.gamma + self.r + self.lambda_)

    @property
    def cost_attacker(self):
        return self.k1 * self.beta
