import math

class EpidemicModel:
    """
    Unified Patch-and-Removal epidemic model (equations 19–21 of the paper):

      dS/dt = r*I − β*(I/N)*S − γ*S
      dI/dt = β*(I/N)*S − λ*I − r*I
      dR/dt = λ*I + γ*S

    This class also exposes the event rates for:
      - disinfection only (I → S)
      - immunisation only (S → R)
      - disinfection + immunisation (I → R)
    """

    def __init__(self, beta, r, gamma, lambda_, N,
                 cost_disinfection=10, cost_immunisation=100):

        self.beta = beta
        self.r = r
        self.gamma = gamma
        self.lambda_ = lambda_
        self.N = N

        # Defender costs defined in the paper
        self.cost_disinfection = cost_disinfection           # k0,1 = 10
        self.cost_immunisation = cost_immunisation           # k0,2 = 100
        self.cost_combined = cost_disinfection + cost_immunisation  # k0,3 = 110

        # Attacker cost (paper: beta * 1000)
        self.cost_attacker = self.beta * 1000.0

    # ================== Differential equations ==================
    def dS_dt(self, S, I):
        return self.r * I - self.beta * (I / self.N) * S - self.gamma * S

    def dI_dt(self, S, I):
        return self.beta * (I / self.N) * S - self.lambda_ * I - self.r * I

    def dR_dt(self, S, I):
        return self.lambda_ * I + self.gamma * S

    # ================== Event rate functions ==================
    def disinfections_only_per_dt(self, I):
        # Disinfection-only: I → S
        return self.r * I

    def immunisations_from_S_per_dt(self, S):
        # Immunisation-only: S → R
        return self.gamma * S

    def disinfection_and_immunisation_per_dt(self, I):
        # Disinfection + immunisation: I → R
        return self.lambda_ * I
