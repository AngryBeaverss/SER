import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0
dt = 0.001
total_time = 50
steps = int(total_time / dt)
time = np.linspace(0, total_time, steps)

# System parameters
omega_qubit = 1.0    # Qubit frequency
omega_cavity = 1.0   # Cavity frequency
g = 0.1              # Qubit-cavity coupling strength
drive_strength = 0.1 # External drive strength
gamma_spont = 0.01   # Qubit spontaneous emission rate
kappa = 0.001         # Cavity decay rate
n_max = 10           # Truncate cavity to 10 Fock states

# Pauli and cavity operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T

# Create cavity annihilation operator (a) in Fock basis
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# Construct full Hamiltonian for qubit-cavity system
H_qubit = 0.5 * omega_qubit * np.kron(sigma_z, np.eye(n_max))
H_cavity = omega_cavity * np.kron(np.eye(2), a_dagger @ a)
H_interaction = g * (np.kron(sigma_plus, a) + np.kron(sigma_minus, a_dagger))
H_drive = drive_strength * np.kron(sigma_x, np.eye(n_max))
H = H_qubit + H_cavity + H_interaction + H_drive

# Lindblad operators
L_qubit = np.sqrt(gamma_spont) * np.kron(sigma_minus, np.eye(n_max))
L_cavity = np.sqrt(kappa) * np.kron(np.eye(2), a)

# Identity operator for full system
I = np.eye(2 * n_max, dtype=complex)

# Initial state (mixed state with qubit in superposition and cavity in vacuum)
rho_qubit_init = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
rho_cavity_init = np.zeros((n_max, n_max), dtype=complex)
rho_cavity_init[0, 0] = 1.0  # Cavity starts in vacuum state
rho_init = np.kron(rho_qubit_init, rho_cavity_init)
for i in range(1, n_max):
    rho_init = rho_init + 0.01 * np.kron(np.eye(2), np.outer(np.ones(n_max) / n_max, np.ones(n_max) / n_max))
rho_init = rho_init / np.trace(rho_init)  # Normalize

rho = rho_init.copy()

# Helper functions
# Updated lindblad_ser function
def lindblad_ser(rho, H, L_list, I, dt):
    drho_dt = -1j / hbar * (H @ rho - rho @ H)  # Commutator
    for L in L_list:
        rate = gamma_spont if L is L_qubit else kappa
        lindblad = (L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))
        drho_dt += rate * lindblad
    # SER feedback with qubit coherence
    rho_qubit = np.zeros((2, 2), dtype=complex)
    for n in range(n_max):
        block = rho[n * 2:(n + 1) * 2, n * 2:(n + 1) * 2]
        rho_qubit += block
    # No division by n_max here!
    coherence = np.abs(rho_qubit[0, 1]) + np.abs(rho_qubit[1, 0])
    F_rho = np.exp(-2 * (1 - coherence))
    beta = min(1.5 * (1 - coherence), 1.5)  # Slightly increased feedback strength
    feedback = beta * F_rho * (I - rho) @ L_qubit @ rho @ L_qubit.conj().T @ (I - rho)
    drho_dt += feedback
    rho_next = rho + drho_dt * dt
    # Positivity enforcement
    vals, vecs = np.linalg.eigh(rho_next)
    vals_clamped = np.clip(vals, 0, None)
    rho_pos = vecs @ np.diag(vals_clamped) @ vecs.conj().T
    return rho_pos / np.trace(rho_pos)

def coherence_measure(rho):
    return np.sum(np.abs(rho - np.diag(np.diag(rho)))) / 2  # Sum of off-diagonal absolute values

# Storage
purity = np.zeros(steps)
coherence = np.zeros(steps)

# Main simulation loop
L_list = [L_qubit, L_cavity]
for step in range(steps):
    rho = lindblad_ser(rho, H, L_list, I, dt)
    purity[step] = np.real(np.trace(rho @ rho))  # Purity
    coherence[step] = coherence_measure(rho)      # Coherence

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, purity, label='Purity')
plt.plot(time, coherence, label='Coherence')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.title('SER Model in Realistic Jaynes-Cummings Scenario')
plt.show()