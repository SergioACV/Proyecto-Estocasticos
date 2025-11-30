# analyze_case4.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from lib.unified_model import UnifiedEpidemicModel
from lib.unified_simulation import UnifiedSimulator
from lib.nash import solve_nash

# ---------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Condiciones iniciales
N = 10000
I0 = 15
S0 = N - I0
R0 = 0

TOTAL_TIME = 168.0
DT = 1.0

# ---------------------------------------------------------------
# 1. SIMULACIÓN Y GRÁFICAS TEMPORALES
# ---------------------------------------------------------------

def run_and_plot_case4(beta, gamma, r, lambda_, title, filename):
    initial_state = [S0, I0, R0]
    model = UnifiedEpidemicModel(beta, gamma, r, lambda_)
    sim = UnifiedSimulator(model, initial_state, DT, TOTAL_TIME)
    sim.run()

    t = sim.t_values
    S = sim.S_values
    I = sim.I_values
    R = sim.R_values

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, "b--", linewidth=2, label="Susceptible")
    plt.plot(t, I, "r-", linewidth=2, label="Infectado")
    plt.plot(t, R, "g-.", linewidth=2, label="Recuperado/Inmune")

    plt.title(f"{title}\nβ={beta}, γ={gamma}, r={r}, λ={lambda_}")
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("Nodos")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

    print(f"Generada: {filename}")


# Figuras similares a Fig 5–7 del paper, pero ahora con (β, γ, r, λ)
run_and_plot_case4(1.5, 3, 2, 2, "Caso 4: Propagación Alta", "PropagaciónAlta.png")
run_and_plot_case4(1.0, 5, 5, 5, "Caso 4: Control Moderado", "ControlModerado.png")
run_and_plot_case4(0.6, 8, 8, 12, "Caso 4: Defensa muy fuerte", "DefensaMuyFuerte.png")


# ---------------------------------------------------------------
# 2. MATRICES DE PAYOFF (EXTENSIÓN DE TABLA 3)
# ---------------------------------------------------------------

print("\nSimulando Matrices de Payoff para Caso 4...")

attacker_betas = [0.5, 1.0, 1.5, 2.0]  
defender_gammas = [2, 4, 6]
defender_rs = [2, 5, 10]
defender_lambdas = [2, 5, 10]

# Para reducir dimensionalidad de la matriz 4D → combinamos en una sola lista de estrategias
defender_strategies = []
for g in defender_gammas:
    for r in defender_rs:
        for l in defender_lambdas:
            defender_strategies.append((g, r, l))

# MATRICES
payoff_matrix_A = np.zeros((len(attacker_betas), len(defender_strategies)))
payoff_matrix_D = np.zeros((len(attacker_betas), len(defender_strategies)))

for i, beta in enumerate(attacker_betas):
    for j, (gamma, r, lambda_) in enumerate(defender_strategies):

        model = UnifiedEpidemicModel(beta, gamma, r, lambda_)
        sim = UnifiedSimulator(model, [S0, I0, R0], DT, TOTAL_TIME)
        sim.run()

        payoff_matrix_A[i, j] = sim.payoff_attacker
        payoff_matrix_D[i, j] = sim.payoff_defender

# Guardar CSV
df_A = pd.DataFrame(
    payoff_matrix_A,
    index=[f"β={b}" for b in attacker_betas],
    columns=[f"γ={g}, r={r}, λ={l}" for (g, r, l) in defender_strategies]
)
df_D = pd.DataFrame(
    payoff_matrix_D,
    index=[f"β={b}" for b in attacker_betas],
    columns=[f"γ={g}, r={r}, λ={l}" for (g, r, l) in defender_strategies]
)

df_A.to_csv(os.path.join(OUTPUT_DIR, "payoff_atacante_caso4.csv"))
df_D.to_csv(os.path.join(OUTPUT_DIR, "payoff_defensor_caso4.csv"))

print("Matrices guardadas en CSV.")

# Heatmaps para el atacante y el defensor
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.heatmap(df_A, annot=False, cmap="Reds")
plt.title("Payoff del Atacante (Caso 4)")

plt.subplot(1, 2, 2)
sns.heatmap(df_D, annot=False, cmap="Greens")
plt.title("Payoff del Defensor (Caso 4)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmaps_caso4.png"))
plt.close()


# ---------------------------------------------------------------
# 3. EQUILIBRIO DE NASH
# ---------------------------------------------------------------

print("\nBuscando Equilibrio de Nash (Caso 4)...")

equilibria = solve_nash(payoff_matrix_A, payoff_matrix_D)

if not equilibria:
    print("No se encontró equilibrio para el Caso 4.")
else:
    eq_p, eq_q = equilibria[0]
    print("\nEquilibrio de Nash encontrado:")
    print("Probabilidades Atacante:", eq_p)
    print("Probabilidades Defensor:", eq_q)

    expected_A = np.dot(np.dot(eq_p, payoff_matrix_A), eq_q)
    expected_D = np.dot(np.dot(eq_p, payoff_matrix_D), eq_q)

    print(f"Payoff esperado Atacante: {expected_A:.4f}")
    print(f"Payoff esperado Defensor: {expected_D:.4f}")

    # Gráfica de estrategias mixtas
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar([str(b) for b in attacker_betas], eq_p, color="red")
    plt.title("Estrategia Mixta Atacante (Caso 4)")
    plt.xlabel("Beta")
    plt.ylabel("Probabilidad")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(defender_strategies)), eq_q, color="green")
    plt.title("Estrategia Mixta Defensor (Caso 4)")
    plt.xlabel("Estrategias (γ,r,λ)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nash_mixto_caso4.png"))
    plt.close()


# ---------------------------------------------------------------
# 4. GRÁFICAS EXTRA: PAYOFFS VS PARÁMETROS
# ---------------------------------------------------------------

print("\nGenerando gráficas extra (Payoff vs parámetros)...")

# ---------------------------
# Payoff del Defensor vs r
# ---------------------------
r_values = np.linspace(0.5, 15, 20)
payoffs_def_vs_r = []

beta_fixed = 1.5     # fija beta
gamma_fixed = 5      # fija gamma
lambda_fixed = 8     # fija lambda

for r_val in r_values:
    model = UnifiedEpidemicModel(beta_fixed, gamma_fixed, r_val, lambda_fixed)
    sim = UnifiedSimulator(model, [S0, I0, R0], DT, TOTAL_TIME)
    sim.run()
    payoffs_def_vs_r.append(sim.payoff_defender)

plt.figure(figsize=(10, 6))
plt.plot(r_values, payoffs_def_vs_r, "g-o", linewidth=2)
plt.title(f"Payoff del Defensor vs Tasa de Recuperación r\n(β={beta_fixed}, γ={gamma_fixed}, λ={lambda_fixed})")
plt.xlabel("Tasa de recuperación (r)")
plt.ylabel("Payoff del Defensor")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "payoff_def_vs_r.png"))
plt.close()

print("Generada: payoff_def_vs_r.png")

# ---------------------------
# Payoff del Atacante vs β
# ---------------------------
beta_values = np.linspace(0.5, 3.0, 20)
payoffs_att_vs_beta = []

gamma_fixed = 5      # fija gamma
r_fixed = 5          # fija r
lambda_fixed = 8     # fija lambda

for beta_val in beta_values:
    model = UnifiedEpidemicModel(beta_val, gamma_fixed, r_fixed, lambda_fixed)
    sim = UnifiedSimulator(model, [S0, I0, R0], DT, TOTAL_TIME)
    sim.run()
    payoffs_att_vs_beta.append(sim.payoff_attacker)

plt.figure(figsize=(10, 6))
plt.plot(beta_values, payoffs_att_vs_beta, "r-o", linewidth=2)
plt.title(f"Payoff del Atacante vs Tasa de Infección β\n(γ={gamma_fixed}, r={r_fixed}, λ={lambda_fixed})")
plt.xlabel("Tasa de infección (β)")
plt.ylabel("Payoff del Atacante")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "payoff_att_vs_beta.png"))
plt.close()

print("Generada: payoff_att_vs_beta.png")


print("\nAnálisis del Caso 4 completado.")
