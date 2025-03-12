import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Scaling factors (1 simulation unit = 1 GHz)
GHz_to_MHz = 1000

# Constants
hbar = 1.0
total_time = 50
time_points = np.linspace(0, total_time, 5000)

# Updated real-world parameters (GHz/MHz)
omega_qubit_real = 5.0             # 5 GHz qubit frequency
omega_cavity_real = 5.0            # 5 GHz cavity frequency
g_real = 100 / GHz_to_MHz          # 100 MHz coupling strength
drive_strength_real = 50 / GHz_to_MHz # 50 MHz external drive
gamma_spont_real = 1 / GHz_to_MHz  # 1 MHz spontaneous emission
kappa_real = 0.1 / GHz_to_MHz      # 0.1 MHz cavity decay
n_max = 10

# Operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# Convert to simulation units (GHz)
H_qubit_real = 0.5 * omega_qubit_real * np.kron(sigma_z, np.eye(n_max))
H_cavity_real = omega_cavity_real * np.kron(np.eye(2), a_dagger @ a)
H_interaction_real = g_real * (np.kron(sigma_plus, a) + np.kron(sigma_minus, a_dagger))
H_drive_real = drive_strength_real * np.kron(sigma_x, np.eye(n_max))
H_total_real = H_qubit_real + H_cavity_real + H_interaction_real + H_drive_real

# Lindblad operators (scaled)
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

# Updated RHS for real-world scaling
def lindblad_rhs_real(t, rho_flat):
    rho = rho_flat.reshape(2 * n_max, 2 * n_max)
    drho_dt = -1j / hbar * (H_total_real @ rho - rho @ H_total_real)

    for L in L_list_real:
        rate = gamma_spont_real if L is L_qubit_real else kappa_real
        lindblad_term = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        drho_dt += rate * lindblad_term

    # SER feedback
    rho_qubit = sum(rho[n*2:(n+1)*2, n*2:(n+1)*2] for n in range(n_max))
    coherence = np.abs(rho_qubit[0, 1]) + np.abs(rho_qubit[1, 0])
    F_rho = np.exp(-2 * (1 - coherence))
    beta = min(1.5 * (1 - coherence), 1.5)
    feedback = beta * F_rho * (I - rho) @ L_qubit_real @ rho @ L_qubit_real.conj().T @ (I - rho)
    drho_dt += feedback

    return drho_dt.flatten()

# Solve scaled system using adaptive RK45 integrator
solution_real = solve_ivp(lindblad_rhs_real, [0, total_time], rho_init_flat, t_eval=time_points, method='RK45')

# Calculate purity and coherence for scaled system
purity_real = np.zeros(len(solution_real.t))
coherence_real = np.zeros(len(solution_real.t))

def coherence_measure(rho):
    return np.sum(np.abs(rho - np.diag(np.diag(rho)))) / 2

for idx, rho_flat in enumerate(solution_real.y.T):
    rho_matrix = rho_flat.reshape(2 * n_max, 2 * n_max)
    purity_real[idx] = np.real(np.trace(rho_matrix @ rho_matrix))
    coherence_real[idx] = coherence_measure(rho_matrix)

# Plotting scaled results
plt.figure(figsize=(12, 6))
plt.plot(solution_real.t, purity_real, label='Purity (Scaled)')
plt.plot(solution_real.t, coherence_real, label='Coherence (Scaled)')
plt.xlabel('Time (ns)')
plt.ylabel('Value')
plt.title('SER Model Scaled to Realistic Superconducting Qubit-Cavity System')
plt.legend()
plt.grid(True)
plt.show()
