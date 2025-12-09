import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.simulation import Simulator
from lib.epidemic_model import EpidemicModel
from lib.nash import solve_nash
import os

# Configuración
OUTPUT_DIR = "resultados"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Parámetros Globales
N = 10000
I0 = 15
S0 = N - I0
TOTAL_TIME = 168.0 # 1 semana en horas
DT = 1.0

def run_and_plot_scenario(beta, r, title, filename):
    initial_state = [S0, I0]
    model = EpidemicModel(beta, r)
    sim = Simulator(model, initial_state, DT, TOTAL_TIME)
    sim.run()
    
    t = sim.t_values
    S = sim.S_values
    I = sim.I_values
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b--', label='Susceptible', linewidth=2)
    plt.plot(t, I, 'r:', label='Infected', linewidth=2)
    # Nota: En modelo SIS no hay "Recovered" inmune permanente, 
    # pero podemos graficar N - S - I si hubiera latencia, aqui S+I=N.
    
    plt.title(f"{title}\n(beta={beta}, r={r})")
    plt.xlabel("Time (hours)")
    plt.ylabel("Nodes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Generada: {filename}")

run_and_plot_scenario(beta=1.62, r=0.5, title="Figura 5: Alta Propagación", filename="figura_5_high_spread.png")

run_and_plot_scenario(beta=1.0, r=2.0, title="Figura 6: Control Moderado", filename="figura_6_moderate.png")

run_and_plot_scenario(beta=0.5, r=5.0, title="Figura 7: Defensa Fuerte", filename="figura_7_strong_defense.png")


print("\nSimulando Matriz de Payoffs...")

# Definimos estrategias discretas
attacker_betas = [0.5, 1.0, 1.5, 2.0, 2.5] # Estrategias del Atacante (filas)
defender_rs = [0.5, 1.0, 2.0, 3.0, 5.0]    # Estrategias del Defensor (columnas)

payoff_matrix_A = np.zeros((len(attacker_betas), len(defender_rs)))
payoff_matrix_D = np.zeros((len(attacker_betas), len(defender_rs)))

for i, beta in enumerate(attacker_betas):
    for j, r in enumerate(defender_rs):
        model = EpidemicModel(beta, r)
        sim = Simulator(model, [S0, I0], DT, TOTAL_TIME)
        sim.run()
        
        payoff_matrix_A[i, j] = sim.payoff_attacker
        payoff_matrix_D[i, j] = sim.payoff_defender

# Guardar en CSV
df_A = pd.DataFrame(payoff_matrix_A, index=attacker_betas, columns=defender_rs)
df_D = pd.DataFrame(payoff_matrix_D, index=attacker_betas, columns=defender_rs)

df_A.to_csv(os.path.join(OUTPUT_DIR, "tabla_payoff_atacante.csv"))
df_D.to_csv(os.path.join(OUTPUT_DIR, "tabla_payoff_defensor.csv"))
print("Tablas de payoff guardadas.")

# Graficar Heatmaps 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(df_A, annot=True, cmap="Reds", fmt=".2f", ax=ax1)
ax1.set_title("Payoff Atacante (Infección - Costo)")
ax1.set_ylabel("Beta (Atacante)")
ax1.set_xlabel("R (Defensor)")

sns.heatmap(df_D, annot=True, cmap="Greens", fmt=".2f", ax=ax2)
ax2.set_title("Payoff Defensor (Salud - Costo)")
ax2.set_ylabel("Beta (Atacante)")
ax2.set_xlabel("R (Defensor)")

plt.savefig(os.path.join(OUTPUT_DIR, "tabla_3_heatmaps.png"))
plt.close()


#Equilibrio de Nash
print("\nCalculando Equilibrio de Nash...")
equilibria = solve_nash(payoff_matrix_A, payoff_matrix_D)

if not equilibria:
    print("No se encontró equilibrio puro o mixto simple.")
else:
    # Tomamos el primer equilibrio encontrado
    eq_p, eq_q = equilibria[0]
    
    print(f"Equilibrio encontrado:")
    print(f"Probabilidades Atacante (Beta): {eq_p}")
    print(f"Probabilidades Defensor (R): {eq_q}")

    # Calcular valor esperado (Payoff en equilibrio)
    expected_payoff_A = np.dot(np.dot(eq_p, payoff_matrix_A), eq_q)
    expected_payoff_D = np.dot(np.dot(eq_p, payoff_matrix_D), eq_q)
    print(f"Payoff Esperado Atacante: {expected_payoff_A:.4f}")
    print(f"Payoff Esperado Defensor: {expected_payoff_D:.4f}")

    payoffs_def_vs_r = np.dot(eq_p, payoff_matrix_D) # Promedio ponderado por estrategia de atacante
    
    plt.figure(figsize=(10, 6))
    plt.plot(defender_rs, payoffs_def_vs_r, 'g-o', linewidth=2)
    plt.axhline(y=expected_payoff_D, color='k', linestyle='--', label='Payoff Equilibrio')
    plt.title("Figura 8: Payoff del Defensor vs Tasa de Recuperación (R)\n(Ante estrategia de equilibrio del Atacante)")
    plt.xlabel("Tasa de Recuperación (r)")
    plt.ylabel("Payoff Esperado")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "figura_8_nash_defensor.png"))
    plt.close()

    # Eje X: Beta (Atacante), Eje Y: Payoff Atacante
    payoffs_att_vs_beta = np.dot(payoff_matrix_A, eq_q)
    
    plt.figure(figsize=(10, 6))
    plt.plot(attacker_betas, payoffs_att_vs_beta, 'r-o', linewidth=2)
    plt.axhline(y=expected_payoff_A, color='k', linestyle='--', label='Payoff Equilibrio')
    plt.title("Figura 9: Payoff del Atacante vs Tasa de Infección (Beta)\n(Ante estrategia de equilibrio del Defensor)")
    plt.xlabel("Tasa de Infección (Beta)")
    plt.ylabel("Payoff Esperado")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "figura_9_nash_atacante.png"))
    plt.close()

    # Gráfica Extra: Visualización de las estrategias mixtas
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar([str(b) for b in attacker_betas], eq_p, color='red', alpha=0.7)
    plt.title("Estrategia Mixta Atacante (Nash)")
    plt.xlabel("Beta")
    plt.ylabel("Probabilidad")
    
    plt.subplot(1, 2, 2)
    plt.bar([str(r) for r in defender_rs], eq_q, color='green', alpha=0.7)
    plt.title("Estrategia Mixta Defensor (Nash)")
    plt.xlabel("R (Recuperación)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nash_estrategias_mixtas.png"))
    plt.close()

print("\nAnálisis completo finalizado. Revisa la carpeta 'resultados'.")