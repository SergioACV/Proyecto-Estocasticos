
from lib.epidemic_model import EpidemicModel
from lib.simulation import Simulator

# Simulation parameters
TOTAL_TIME = 168
dt = 1.0
N = 10000
I0 = 15
S0 = N - I0
R0 = 0

# Parameters
beta = 1.62
r = 2
gamma = 1
lam = 15      

model = EpidemicModel(beta, r, gamma, lam, N)

sim = Simulator(model, initial_state=(S0, I0, R0), dt=dt, total_time=TOTAL_TIME)

result = sim.run()

print("======= RESULTS =======")
print("Attacker gain:", result["gain_attacker"])
print("Defender gain:", result["gain_defender"])
print("Attacker cost:", result["cost_attacker"])
print("Defender cost:", result["cost_defender"])
print("Attacker payoff:", result["payoff_attacker"])
print("Defender payoff:", result["payoff_defender"])
print("Disinfections:", result["total_disinfections_only"])
print("Immunisations from S:", result["total_immunisations_from_S"])
print("Disinfection + Immunisation:", result["total_disinf_and_imm"])

history = result["history"]
history.to_csv("history_patch_removal.csv", index=False)
print("\nFile 'history_patch_removal.csv' generated.")