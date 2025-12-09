import pandas as pd

class Simulator:
    """
    Discrete-time (Euler) simulator.
    It records:
      - Attacker and defender gains
      - Costs
      - Payoffs
      - Event counts (disinfection, immunisation, combined actions)
    """

    def __init__(self, model, initial_state, dt=1.0, total_time=168.0):
        self.model = model
        self.dt = dt
        self.total_time = total_time

        # Initial state (S, I, R)
        self.S, self.I, self.R = initial_state
        self.N = self.S + self.I + self.R
        self.time = 0.0

        self.initialize_statistics()

    def initialize_statistics(self):
        self.t_values = [0.0]
        self.S_values = [self.S]
        self.I_values = [self.I]
        self.R_values = [self.R]

        self.history = [(0.0, self.S, self.I, self.R)]

        # Continuous counters for event rates
        self.total_disinfections_only = 0.0
        self.total_immunisations_from_S = 0.0
        self.total_disinf_and_imm = 0.0

    def step(self):
        S, I, R = self.S, self.I, self.R
        m = self.model

        # ========= Euler update step =========
        dS = m.dS_dt(S, I)
        dI = m.dI_dt(S, I)
        dR = m.dR_dt(S, I)

        self.S += dS * self.dt
        self.I += dI * self.dt
        self.R += dR * self.dt

        # Ensure non-negativity for numerical stability
        self.S = max(self.S, 0)
        self.I = max(self.I, 0)
        self.R = max(self.R, 0)

        # ========= Event counters =========
        self.total_disinfections_only += m.disinfections_only_per_dt(I) * self.dt
        self.total_immunisations_from_S += m.immunisations_from_S_per_dt(S) * self.dt
        self.total_disinf_and_imm += m.disinfection_and_immunisation_per_dt(I) * self.dt

        # Update time
        self.time += self.dt

        self.t_values.append(self.time)
        self.S_values.append(self.S)
        self.I_values.append(self.I)
        self.R_values.append(self.R)

        self.history.append((self.time, self.S, self.I, self.R))

    # ========== Gain and cost calculations ==========
    def compute_time_average(self, P_list):
        total_area = 0
        for i in range(1, len(self.t_values)):
            dt = self.t_values[i] - self.t_values[i-1]
            total_area += 0.5 * (P_list[i] + P_list[i-1]) * dt
        return total_area / self.total_time

    def compute_gain_attacker(self):
        P = [I / self.N for I in self.I_values]
        return self.compute_time_average(P)

    def compute_gain_defender(self):
        P = [(S + R) / self.N for S, R in zip(self.S_values, self.R_values)]
        return self.compute_time_average(P)

    def compute_defender_cost(self):
        m = self.model
        return (
            self.total_disinfections_only * m.cost_disinfection +
            self.total_immunisations_from_S * m.cost_immunisation +
            self.total_disinf_and_imm * m.cost_combined
        )

    def compute_attacker_cost(self):
        return self.model.cost_attacker

    # ========== Main simulation loop ==========
    def run(self):
        while self.time < self.total_time:
            self.step()

        # Gains
        self.gain_attacker = self.compute_gain_attacker()
        self.gain_defender = self.compute_gain_defender()

        # Costs
        self.cost_attacker = self.compute_attacker_cost()
        self.cost_defender = self.compute_defender_cost()

        # Payoffs
        self.payoff_attacker = self.gain_attacker - self.cost_attacker
        self.payoff_defender = self.gain_defender - self.cost_defender

        # History DataFrame
        df = pd.DataFrame({
            "time": self.t_values,
            "S": self.S_values,
            "I": self.I_values,
            "R": self.R_values
        })

        return {
            "history": df,
            "gain_attacker": self.gain_attacker,
            "gain_defender": self.gain_defender,
            "cost_attacker": self.cost_attacker,
            "cost_defender": self.cost_defender,
            "payoff_attacker": self.payoff_attacker,
            "payoff_defender": self.payoff_defender,
            "total_disinfections_only": self.total_disinfections_only,
            "total_immunisations_from_S": self.total_immunisations_from_S,
            "total_disinf_and_imm": self.total_disinf_and_imm,
        }
