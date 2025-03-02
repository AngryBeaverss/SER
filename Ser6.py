import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

##############################################################################
# 1) OPENCL SETUP
##############################################################################
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

with open('ser_lindblad.cl', 'r') as f:
    kernel_code = f.read()
program = cl.Program(context, kernel_code).build()

##############################################################################
# 2) SIMULATION PARAMETERS
##############################################################################
dt = 1e-4           # Smaller time-step => fewer numerical leaps
total_time = 5.0
num_steps = int(total_time / dt)

# We'll keep your dynamic approach from earlier:
#   gamma = 0.3 * exp(-2*(1 - coherence)),
#   beta = 0.0001 + 0.0001 * (entropy_change).
# but we won't clamp gamma or beta, so they can grow or shrink naturally.
#
# The key difference: we do a positivity projection each time step.

##############################################################################
# 3) SYSTEM MATRICES (2x2)
##############################################################################
rho_init = np.array([1.0 + 0j, 0.0 + 0j,
                     0.0 + 0j, 0.0 + 0j],
                    dtype=np.complex128)  # pure state |0>

# Example Hamiltonian, Lindblad operator, identity
H_init = np.array([0.0 + 0j, 1.0 + 0j,
                   1.0 + 0j, 0.0 + 0j],
                  dtype=np.complex128)
L_init = np.array([0.0 + 0j, 0.0 + 0j,
                   0.0 + 0j, 1.0 + 0j],
                  dtype=np.complex128)
I_init = np.array([1.0 + 0j, 0.0 + 0j,
                   0.0 + 0j, 1.0 + 0j],
                  dtype=np.complex128)

##############################################################################
# 4) CREATE OPENCL BUFFERS
##############################################################################
mf = cl.mem_flags
rho_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_init)
H_buf   = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=H_init)
L_buf   = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=L_init)
I_buf   = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=I_init)

##############################################################################
# 5) ARRAYS TO STORE PURITY, ENTROPY, COHERENCE, TRACE
##############################################################################
purity_vals     = np.zeros(num_steps + 1)
entropy_vals    = np.zeros(num_steps + 1)
coherence_vals  = np.zeros(num_steps + 1)
trace_vals      = np.zeros(num_steps + 1)

##############################################################################
# 6) HELPER FUNCTIONS
##############################################################################
def compute_quantities(rho_1d):
    """Compute purity, entropy, coherence, trace from a 2x2 flattened array."""
    rho_m = rho_1d.reshape(2, 2)
    rho2  = rho_m @ rho_m
    purity = np.real(np.trace(rho2))

    # Eigenvalues for entropy
    w = np.linalg.eigvalsh(rho_m)
    w_pos = w[w > 1e-12]  # ignore tiny negative drift
    if len(w_pos) > 0:
        entropy = -np.sum(w_pos * np.log2(w_pos))
    else:
        entropy = 0.0

    coherence = np.abs(rho_m[0, 1])
    trace = np.real(np.trace(rho_m))
    return purity, entropy, coherence, trace

def enforce_positivity(rho_m):
    """
    Project rho_m onto the positive semidefinite cone.
    1) Diagonalize
    2) Clamp negative eigenvalues to 0
    3) Reconstruct
    4) Normalize trace to 1
    """
    w, v = np.linalg.eigh(rho_m)
    w_clamped = np.clip(w, 0, None)
    rho_corrected = v @ np.diag(w_clamped) @ v.conj().T
    tr = np.trace(rho_corrected)
    if np.abs(tr) < 1e-15:
        # fallback if trace is near zero
        return np.eye(2, dtype=np.complex128) / 2
    return rho_corrected / tr

##############################################################################
# 7) INITIALIZATION (step = 0)
##############################################################################
# Read initial rho from GPU, positivity-project
rho_host = np.empty_like(rho_init)
cl.enqueue_copy(queue, rho_host, rho_buf).wait()

rho_m = rho_host.reshape(2,2)
rho_m = enforce_positivity(rho_m)
rho_host = rho_m.flatten()

# Compute initial quantities
p0, e0, c0, t0 = compute_quantities(rho_host)
purity_vals[0]    = p0
entropy_vals[0]   = e0
coherence_vals[0] = c0
trace_vals[0]     = t0

# Write the positivity-corrected ρ back
cl.enqueue_copy(queue, rho_buf, rho_host).wait()

print(f"Step 0, Purity={p0:.4f}, Entropy={e0:.4f}, Coherence={c0:.4f}, Trace={t0:.4f}")

##############################################################################
# 8) MAIN LOOP
##############################################################################
hbar = 1.0

for step in range(1, num_steps + 1):

    # -- Define gamma, beta from the *previous* step's data (step-1) --
    coherence_prev = coherence_vals[step-1]
    # gamma = 0.3 e^{-2(1-coherence)}
    gamma = 0.3 * np.exp(-2.0 * (1.0 - coherence_prev))

    if step > 1:
        # measure change in entropy from step-1 to step-2
        ent_prev     = entropy_vals[step-1]
        ent_prevprev = entropy_vals[step-2]
        ent_change   = abs(ent_prev - ent_prevprev)
    else:
        ent_change   = 0.0

    # beta = 0.0001 + 0.0001 * entropy_change
    beta = 0.0001 + 0.0001 * ent_change

    # -- Evolve one time-step via the kernel --
    program.ser_lindblad(
        queue, (4,), None,
        rho_buf, H_buf, L_buf, I_buf,
        np.float64(hbar), np.float64(gamma), np.float64(beta), np.float64(dt)
    )

    # -- Read the new ρ --
    cl.enqueue_copy(queue, rho_host, rho_buf).wait()
    rho_m = rho_host.reshape(2,2)

    # -- Positivity projection --
    rho_m = enforce_positivity(rho_m)
    rho_host = rho_m.flatten()

    # -- Compute and store updated quantities (this is step # step) --
    p, e, c, t = compute_quantities(rho_host)
    purity_vals[step]    = p
    entropy_vals[step]   = e
    coherence_vals[step] = c
    trace_vals[step]     = t

    # -- Write corrected ρ back to GPU for next iteration --
    cl.enqueue_copy(queue, rho_buf, rho_host).wait()

    # -- Print occasionally --
    if step % 10000 == 0:
        print(f"Step {step}, Purity={p:.4f}, Entropy={e:.4f}, "
              f"Coherence={c:.4f}, Trace={t:.4f}, gamma={gamma:.4f}, beta={beta:.6f}")

##############################################################################
# 9) FINAL OUTPUT AND PLOTTING
##############################################################################
final_p = purity_vals[-1]
final_e = entropy_vals[-1]
final_c = coherence_vals[-1]
final_t = trace_vals[-1]

print("\n--- Final Values ---")
print(f"Purity = {final_p:.6f}")
print(f"Entropy = {final_e:.6f}")
print(f"Coherence = {final_c:.6f}")
print(f"Trace = {final_t:.6f}")

time_axis = np.linspace(0, total_time, num_steps + 1)
plt.figure(figsize=(10, 6))
plt.plot(time_axis, purity_vals,    label='Purity')
plt.plot(time_axis, entropy_vals,   label='Entropy')
plt.plot(time_axis, coherence_vals, label='Coherence')
plt.plot(time_axis, trace_vals,     label='Trace', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.title('Evolution of Purity, Entropy, Coherence, and Trace (SER with Positivity Projection)')
plt.grid(True)
plt.show()

trace_var = np.max(trace_vals) - np.min(trace_vals)
print(f"Trace variation: {trace_var:.1e}")
