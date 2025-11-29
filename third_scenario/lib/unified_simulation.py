# lib/unified_simulation.py
import math

class UnifiedSimulator:
    """
    Simulador para el Modelo Unificado (Caso 4).
    Tres compartimentos: S, I, R.
    """

    def __init__(self, model, initial_state, dt=1.0, total_time=168.0):
        self.model = model

        self.S, self.I, self.R = initial_state
        self.N = self.S + self.I + self.R

        self.dt = dt
        self.total_time = total_time
        self.time = 0.0

        self._init_statistics()

    def _init_statistics(self):
        self.t_values = [0.0]
        self.S_values = [self.S]
        self.I_values = [self.I]
        self.R_values = [self.R]

    def step(self):
        dS = self.model.dS_dt(self.S, self.I, self.R, self.N)
        dI = self.model.dI_dt(self.S, self.I, self.R, self.N)
        dR = self.model.dR_dt(self.S, self.I, self.R, self.N)

        self.S += dS * self.dt
        self.I += dI * self.dt
        self.R += dR * self.dt

        # Corrección numérica
        self.S = max(0.0, self.S)
        self.I = max(0.0, self.I)
        self.R = max(0.0, self.R)

        total = self.S + self.I + self.R
        if total > 0:
            self.S = (self.S / total) * self.N
            self.I = (self.I / total) * self.N
            self.R = (self.R / total) * self.N

        self.time += self.dt

        self.t_values.append(self.time)
        self.S_values.append(self.S)
        self.I_values.append(self.I)
        self.R_values.append(self.R)

    def run(self):
        while self.time < self.total_time:
            self.step()

        self.gain_attacker = self.average(self.I_values)
        self.gain_defender = self.average(self.S_values + self.R_values)
        self.cost_attacker = self.model.cost_attacker
        self.cost_defender = self.model.cost_defender

        self.payoff_attacker = self.gain_attacker - self.cost_attacker
        self.payoff_defender = self.gain_defender - self.cost_defender

    def average(self, lst):
        return sum(lst) / len(lst)
