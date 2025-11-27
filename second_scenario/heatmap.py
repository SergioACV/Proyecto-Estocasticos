import numpy as np
import matplotlib.pyplot as plt

infection_rates = np.linspace(0.1, 1.0, 10)  # β
recover_rates = np.linspace(0.5, 2.0, 10)   # r


def defender_payoff(beta, r, gamma=1, lambd=2):
    return -1000 * beta / r


payoff_matrix = np.array([[defender_payoff(b, r) for r in recover_rates] for b in infection_rates])

plt.figure(figsize=(8,6))
plt.imshow(payoff_matrix, origin='lower', aspect='auto', 
           extent=[recover_rates[0], recover_rates[-1], infection_rates[0], infection_rates[-1]],
           cmap='magma')
plt.colorbar(label="Defender's payoff")
plt.xlabel("Recover/Disinfection rate (r)")
plt.ylabel("Infection rate (β)")
plt.title("Defender's payoff depending on infection and recovery rates")
plt.show()
