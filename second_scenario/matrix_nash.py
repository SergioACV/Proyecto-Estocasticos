# game_matrix_patch_removal.py

import numpy as np
from lib.epidemic_model import EpidemicModel
from lib.simulation import Simulator


beta_values = [0.5, 1.0, 1.62, 2.0]       # attacker strategies
lambda_values = [1, 5, 10, 15, 20]        # defender strategies

# Simulation constants
TOTAL_TIME = 168
dt = 1
N = 10000
I0 = 15
S0 = N - I0
R0 = 0
r = 2
gamma = 1

# Matrices for payoffs
attacker_matrix = np.zeros((len(beta_values), len(lambda_values)))
defender_matrix = np.zeros((len(beta_values), len(lambda_values)))


for i, beta in enumerate(beta_values):
    for j, lam in enumerate(lambda_values):

        model = EpidemicModel(beta, r, gamma, lam, N)
        sim = Simulator(model, initial_state=(S0, I0, R0),
                               dt=dt, total_time=TOTAL_TIME)
        result = sim.run()

        attacker_matrix[i][j] = result["payoff_attacker"]
        defender_matrix[i][j] = result["payoff_defender"]

print("\n===== Attacker Payoff Matrix =====")
print(attacker_matrix)

print("\n===== Defender Payoff Matrix =====")
print(defender_matrix)

def find_nash_equilibrium(A, D):
    """
    Nash equilibrium in pure strategies.
    A: payoff matrix attacker
    D: payoff matrix defender
    """
    n_rows, n_cols = A.shape

    nash_points = []

    for i in range(n_rows):
        for j in range(n_cols):
            # Check attacker best response for column j
            attacker_best = np.max(A[:, j]) == A[i, j]

            # Check defender best response for row i
            defender_best = np.max(D[i, :]) == D[i, j]

            if attacker_best and defender_best:
                nash_points.append((i, j))

    return nash_points


nash = find_nash_equilibrium(attacker_matrix, defender_matrix)

print("\n===== Nash Equilibrium Points (indexes) =====")
print(nash)

if nash:
    for (i, j) in nash:
        print(f"\nNASH at β={beta_values[i]}, λ={lambda_values[j]}")
else:
    print("\nNo pure strategy Nash equilibrium found.")
