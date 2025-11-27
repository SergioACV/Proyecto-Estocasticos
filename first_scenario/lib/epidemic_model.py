COST_ATTACKER = {
    "Exploit": 2,
    "Generate": 1,
    "Launch": 1
}  # Costos del atacante: NOT USED

COST_DEFENDER = {
    "Detection": 2,
    "Reboot": 1
}  # Costos del defensor


class EpidemicModel:
    """Base general para modelos tipo SIR/SIS/etc."""
    
    def __init__(self, beta, r):
        self.beta = beta
        self.r = r

        self.cost_attacker = self.get_cost_attacker()
        self.cost_defender = self.get_cost_defender()
    
    def disinfections_per_dt(self, I):
        return self.r * I

    def get_cost_attacker(self):
        return self.beta * 1000

    def get_cost_defender(self):
        cost = 0
        for key, value in COST_DEFENDER.items():
            cost += value
        return cost

    def dS_dt(self, S, I, N):
        """Ecuación diferencial para S (desnormalizada)."""
        return -self.beta * (I / N) * S + self.r * I

    def dI_dt(self, S, I, N):
        """Ecuación diferencial para I (desnormalizada)."""
        return self.beta * (I / N) * S - self.r * I
