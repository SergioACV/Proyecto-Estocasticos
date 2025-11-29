from lib.unified_model import UnifiedEpidemicModel
from lib.unified_simulation import UnifiedSimulator

# Parámetros del Caso 4
N = 10000
I0 = 15
S0 = N - I0
R0 = 0

beta = 1.62
gamma = 5.0
r = 5.0
lambda_ = 10.0

initial_state = [S0, I0, R0]

model = UnifiedEpidemicModel(beta, gamma, r, lambda_)
sim = UnifiedSimulator(model, initial_state, dt=1.0, total_time=168.0)

sim.run()

print("Payoff atacante:", sim.payoff_attacker)
print("Payoff defensor:", sim.payoff_defender)
