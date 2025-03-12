# Here's the complete, ready-to-run Python script with parameter sweep, clearly commented and structured.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Constants and scaling factors
hbar = 1.0
GHz_to_MHz = 1000
total_time = 50
time_points = np.linspace(0, total_time, 5000)
n_max = 10

# Real-world scaled parameters (GHz/MHz)
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 50 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz

# Pauli and cavity operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# Initial Hamiltonians (without interaction)
H_qubit_real = 0.5 * omega_qubit_real * np.kron(sigma_z, np.eye(n_max))
H_cavity_real = omega_cavity_real * np.kron(np.eye(2), a_dagger @ a)
H_drive_real = drive_strength_real * np.kron(sigma_x, np.eye(n_max))

# Lindblad operators
L_qubit_real = np.sqrt(gamma_spont_real) * np.kron(sigma_minus, np.eye(n_max))
L_cavity_real = np.sqrt(kappa_real) * np.kron(np.eye(2), a)
L_list_real = [L_qubit_real, L_cavity_real]

# Identity operator
I = np.eye(2 * n_max, dtype=complex)

# Initial state setup
rho_qubit_init = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
rho_cavity_init = np.zeros((n_max, n_max), dtype=complex)
rho_cavity_init[0, 0] = 1.0
rho_init = np.kron(rho_qubit_init, rho_cavity_init)
rho_init += 0.01 * np.kron(np.eye(2), np.ones((n_max, n_max)) / n_max**2)
rho_init /= np.trace(rho_init)
rho_init_flat = rho_init.flatten()

# Coherence measurement function
def coherence_measure(rho):
    return np.sum(np.abs(rho - np.diag(np.diag(rho)))) / 2

# Parameter sweep setup
feedback_strengths = np.linspace(0, 3.0, 5)
coupling_strengths = np.linspace(50, 200, 4) / GHz_to_MHz

results = []

# Main simulation loop (parameter sweep)
for beta_max in feedback_strengths:
    for g_sweep in coupling_strengths:
        H_interaction_sweep = g_sweep * (np.kron(sigma_plus, a) + np.kron(sigma_minus, a_dagger))
        H_total_sweep = H_qubit_real + H_cavity_real + H_interaction_sweep + H_drive_real

        # Define RHS for this iteration
        def lindblad_rhs_sweep(t, rho_flat):
            rho = rho_flat.reshape(2 * n_max, 2 * n_max)
            drho_dt = -1j / hbar * (H_total_sweep @ rho - rho @ H_total_sweep)
            for L in L_list_real:
                rate = gamma_spont_real if L is L_qubit_real else kappa_real
                lindblad_term = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
                drho_dt += rate * lindblad_term
            rho_qubit = sum(rho[n*2:(n+1)*2, n*2:(n+1)*2] for n in range(n_max))
            coherence = np.abs(rho_qubit[0, 1]) + np.abs(rho_qubit[1, 0])
            F_rho = np.exp(-2 * (1 - coherence))
            beta = min(beta_max * (1 - coherence), beta_max)
            feedback = beta * F_rho * (I - rho) @ L_qubit_real @ rho @ L_qubit_real.conj().T @ (I - rho)
            drho_dt += feedback
            return drho_dt.flatten()

        # Solve ODE
        sol = solve_ivp(lindblad_rhs_sweep, [0, total_time], rho_init_flat, t_eval=time_points, method='RK45')

        # Calculate and store final metrics
        final_rho = sol.y[:, -1].reshape(2 * n_max, 2 * n_max)
        final_purity = np.real(np.trace(final_rho @ final_rho))
        final_coherence = coherence_measure(final_rho)

        results.append({
            'Feedback Strength (beta)': beta_max,
            'Coupling Strength (MHz)': g_sweep * GHz_to_MHz,
            'Final Purity': final_purity,
            'Final Coherence': final_coherence
        })

# Convert results to DataFrame and display
df_results = pd.DataFrame(results)
print(df_results)

# Optional visualization
fig, ax = plt.subplots(figsize=(10, 6))
for beta in feedback_strengths:
    subset = df_results[df_results['Feedback Strength (beta)'] == beta]
    ax.plot(subset['Coupling Strength (MHz)'], subset['Final Coherence'], marker='o', label=f'β={beta:.2f}')

ax.set_xlabel('Coupling Strength (MHz)')
ax.set_ylabel('Final Coherence')
ax.set_title('Final Coherence vs Coupling Strength for Various Feedback Strengths')
ax.legend()
ax.grid(True)
plt.show()
