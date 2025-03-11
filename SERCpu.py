import numpy as np
from matplotlib import pyplot as plt

##############################################################################
# 1) SIMULATION PARAMETERS
##############################################################################
dt = .001           # Smaller time-step => fewer numerical leaps
total_time = 100.0
num_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time, num_steps + 1)

##############################################################################
# 2) SYSTEM SIZE SELECTION
##############################################################################
DSIZE = int(input("Enter matrix size (2 for 2x2, 4 for 4x4): ").strip())

##############################################################################
# 3) SYSTEM MATRICES (Dynamic DSIZE)
##############################################################################
A = np.random.randn(DSIZE, DSIZE) + 1j * np.random.randn(DSIZE, DSIZE)
rho_init = A @ A.conj().T  # Make it positive semidefinite
rho_init /= np.trace(rho_init)  # Normalize to trace = 1

# Hamiltonian and Lindblad operators - simple example
H_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
H_init = (H_init + H_init.conj().T) / 2  # Hermitian

L_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
L_init = (L_init + L_init.conj().T) / 2  # Hermitian

I_init = np.eye(DSIZE, dtype=np.complex128)

##############################################################################
# 4) ARRAYS TO STORE PURITY, ENTROPY, COHERENCE, TRACE
##############################################################################
purity_vals = np.zeros(num_steps + 1)
entropy_vals = np.zeros(num_steps + 1)
coherence_vals = np.zeros(num_steps + 1)
trace_vals = np.zeros(num_steps + 1)

##############################################################################
# 5) HELPER FUNCTIONS
##############################################################################
def compute_quantities(rho_m):
    """Compute purity, entropy, coherence, trace from a DSIZE x DSIZE density matrix."""
    rho2 = rho_m @ rho_m
    purity = np.real(np.trace(rho2))

    w = np.linalg.eigvalsh(rho_m)
    w_pos = w[w > 1e-12]  # Ignore tiny negative drift
    entropy = -np.sum(w_pos * np.log2(w_pos)) if len(w_pos) > 0 else 0.0

    coherence = np.sum(np.abs(rho_m)) - np.trace(np.abs(rho_m))
    trace = np.real(np.trace(rho_m))

    return purity, entropy, coherence, trace

def enforce_positivity(rho_m):
    """Project rho_m onto the positive semidefinite cone."""
    w, v = np.linalg.eigh(rho_m)
    w_clamped = np.clip(w, 0, None)
    rho_corrected = v @ np.diag(w_clamped) @ v.conj().T
    return rho_corrected / np.trace(rho_corrected)

def ser_lindblad(rho, H, L, I, hbar, gamma, beta, dt):
    """Compute the time derivative of rho using the Lindblad equation with SER feedback."""
    d = DSIZE

    # 1) Commutator term: (-i/hbar) [H, rho]
    H_rho = H @ rho
    rho_H = rho @ H
    commutator = -1j / hbar * (H_rho - rho_H)

    # 2) Lindblad dissipator: gamma * [L rho L^\dagger - 0.5 {L^\dagger L, rho}]
    L_rho_Ldag = L @ rho @ L.conj().T
    Ldag_L = L.conj().T @ L
    Ldag_L_rho = Ldag_L @ rho
    rho_Ldag_L = rho @ Ldag_L
    lindblad_term = gamma * (L_rho_Ldag - 0.5 * (Ldag_L_rho + rho_Ldag_L))

    # 3) SER feedback term
    coherence_squared = np.sum(np.abs(rho) ** 2) - np.sum(np.abs(np.diag(rho)) ** 2)
    F_rho = np.exp(-coherence_squared)
    I_minus_rho = I - rho
    L_rho = L @ rho
    L_rho_Ldag_new = L_rho @ L.conj().T
    temp = I_minus_rho @ L_rho_Ldag_new
    SER_feedback = beta * F_rho * temp @ I_minus_rho

    # Combine all terms
    drho_dt = commutator + lindblad_term + SER_feedback
    return rho + dt * drho_dt

##############################################################################
# 6) INITIALIZATION (step = 0)
##############################################################################
rho_m = rho_init.copy()
rho_m = enforce_positivity(rho_m)

p0, e0, c0, t0 = compute_quantities(rho_m)
purity_vals[0] = p0
entropy_vals[0] = e0
coherence_vals[0] = np.real(c0)
trace_vals[0] = t0

print(f"Step 0, Purity={p0:.4f}, Entropy={e0:.4f}, Coherence={c0:.4f}, Trace={t0:.4f}")

##############################################################################
# 7) MAIN LOOP
##############################################################################
hbar = 1.0

for step in range(1, num_steps + 1):
    coherence_prev = coherence_vals[step - 1]
    gamma = 0.3 * np.exp(-2.0 * (1.0 - coherence_prev))

    norm_entropy = entropy_vals[step - 1] / np.log2(DSIZE) if step > 1 else 0.0
    beta_0 = 0.5  # Baseline beta value
    beta = beta_0 * (1 - np.exp(-coherence_prev))
    beta = min(beta, 2.0)

    # Update rho using the CPU-based function
    rho_m = ser_lindblad(rho_m, H_init, L_init, I_init, hbar, gamma, beta, dt)
    rho_m = enforce_positivity(rho_m)

    p, e, c, t = compute_quantities(rho_m)
    purity_vals[step] = p
    entropy_vals[step] = e
    coherence_vals[step] = np.real(c)
    trace_vals[step] = t

    if step % 10000 == 0:
        print(f"Step {step}, Purity={p:.4f}, Entropy={e:.4f}, "
              f"Coherence={c:.4f}, Trace={t:.4f}, gamma={gamma:.4f}, beta={beta:.6f}")

##############################################################################
# 8) FINAL OUTPUT AND PLOTTING
##############################################################################
print("-- Final Values ---")
print(f"Purity = {purity_vals[-1]:.6f}")
print(f"Entropy = {entropy_vals[-1]:.6f}")
print(f"Coherence = {coherence_vals[-1]:.6f}")
print(f"Trace = {trace_vals[-1]:.6f}")

# Save simulation data
np.save("rho_cpu_final.npy", rho_m)
np.savetxt("rho_cpu_final.csv", rho_m.view(float), delimiter=",")

# Save evolution results
results = np.column_stack((time_axis, purity_vals, entropy_vals, coherence_vals))
np.savetxt("SER_cpu_results.csv", results, delimiter=",", header="Time,Purity,Entropy,Coherence", comments="")

# Generate and save plots
plt.figure(figsize=(10, 6))
plt.plot(time_axis, purity_vals, label='Purity')
plt.plot(time_axis, entropy_vals, label='Entropy')
plt.plot(time_axis, coherence_vals, label='Coherence')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.title('Evolution of Purity, Entropy, Coherence')
plt.grid(True)
plt.savefig("SER_plot.png", dpi=300)
plt.savefig("SER_plot.pdf")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(time_axis, purity_vals, label='Purity')
plt.plot(time_axis, entropy_vals, label='Entropy')
plt.plot(time_axis, coherence_vals, label='Coherence')
plt.plot(time_axis, trace_vals, label='Trace', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.title('Evolution of Purity, Entropy, Coherence, and Trace')
plt.grid(True)
plt.show()

trace_var = np.max(trace_vals) - np.min(trace_vals)
print(f"Trace variation: {trace_var:.1e}")