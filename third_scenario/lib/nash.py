import numpy as np
from itertools import combinations

def solve_nash(A, B):
    """
    Encuentra el Equilibrio de Nash en estrategias mixtas para un juego bimatricial (A, B)
    usando el método de Enumeración de Soportes.
    
    Args:
        A: Matriz de payoffs del Jugador 1 (Atacante)
        B: Matriz de payoffs del Jugador 2 (Defensor)
        
    Returns:
        Lista de tuplas (p, q), donde p es la estrategia del J1 y q del J2.
    """
    m, n = A.shape
    equilibria = []

    # Iterar sobre todos los tamaños de soporte posibles k
    for k in range(1, min(m, n) + 1):
        # Iterar sobre todas las combinaciones de k estrategias para cada jugador
        for support_row in combinations(range(m), k):
            for support_col in combinations(range(n), k):
                
                try:
                    beta_mat = B[np.ix_(support_row, support_col)].T
                    # Agregamos restricción de suma = 1
                    left_side_p = np.vstack([beta_mat, np.ones((1, k))])
                    right_side_p = np.append(np.zeros(k), 1.0)
                    
                    # Como el sistema puede ser sobredeterminado con la restricción de suma,
                    # usamos mínimos cuadrados si k < n, o solución directa si cuadra.
                    # Simplificación para soporte k: Resolver B_sub.T * p = const
                    
                    # Método robusto:
                    M_p = np.zeros((k+1, k+1))
                    M_p[:-1, :-1] = beta_mat
                    M_p[:-1, -1] = -1 # constante v
                    M_p[-1, :-1] = 1  # suma p = 1
                    M_p[-1, -1] = 0
                    
                    rhs_p = np.zeros(k+1)
                    rhs_p[-1] = 1
                    
                    sol_p = np.linalg.solve(M_p, rhs_p)
                    p_sub = sol_p[:-1]
                    val_v = sol_p[-1]
                    
                    # Resolver q (Estrategia del Defensor) usando la matriz A (Atacante)
                    M_q = np.zeros((k+1, k+1))
                    M_q[:-1, :-1] = A[np.ix_(support_row, support_col)]
                    M_q[:-1, -1] = -1 # constante u
                    M_q[-1, :-1] = 1  # suma q = 1
                    M_q[-1, -1] = 0
                    
                    rhs_q = np.zeros(k+1)
                    rhs_q[-1] = 1
                    
                    sol_q = np.linalg.solve(M_q, rhs_q)
                    q_sub = sol_q[:-1]
                    val_u = sol_q[-1]
                    
                    # Verificar si las probabilidades son válidas (>= 0)
                    if np.all(p_sub >= -1e-10) and np.all(q_sub >= -1e-10):
                        # Construir vectores completos
                        p = np.zeros(m)
                        p[list(support_row)] = p_sub
                        p = np.maximum(p, 0) # Limpiar ruido numérico negativo
                        p /= p.sum() # Renormalizar
                        
                        q = np.zeros(n)
                        q[list(support_col)] = q_sub
                        q = np.maximum(q, 0)
                        q /= q.sum()
                        
                        # VERIFICAR CONDICIÓN DE MEJOR RESPUESTA (NASH)
                        # Nadie debe tener incentivo de cambiar a una estrategia fuera del soporte
                        
                        # Payoff esperado del Atacante si juega contra q
                        payoff_A = np.dot(A, q)
                        # El payoff en el soporte debe ser >= payoff fuera del soporte
                        max_payoff_A = np.max(payoff_A)
                        current_payoff_A = np.dot(p, payoff_A) # Debería ser val_u aproximadamente
                        
                        # Payoff esperado del Defensor si J1 juega p
                        payoff_B = np.dot(p, B)
                        max_payoff_B = np.max(payoff_B)
                        current_payoff_B = np.dot(payoff_B, q)
                        
                        tol = 1e-6
                        if (abs(max_payoff_A - current_payoff_A) < tol) and \
                           (abs(max_payoff_B - current_payoff_B) < tol):
                            
                            # Evitar duplicados
                            is_new = True
                            for ep, eq in equilibria:
                                if np.allclose(p, ep) and np.allclose(q, eq):
                                    is_new = False
                                    break
                            if is_new:
                                equilibria.append((p, q))
                                
                except np.linalg.LinAlgError:
                    continue

    return equilibria