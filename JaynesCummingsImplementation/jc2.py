import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

# OpenCL Setup
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Simulation parameters
dt = 0.01
total_time = 50
num_steps = int(total_time / dt)
hbar = 1.0
beta_base = 0.2  # Base feedback strength

# Define different initial states
initial_states = {
    "Ground |0⟩": np.array([[1, 0], [0, 0]], dtype=np.complex128),
    "Excited |1⟩": np.array([[0, 0], [0, 1]], dtype=np.complex128),
    "Maximally Mixed": np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128),
    "Superposition (|0⟩+|1⟩)/√2": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
}

# Jaynes-Cummings Hamiltonian
g = 0.5
omega = 1.0
H_init = np.array([[0, g], [g, omega]], dtype=np.complex128)

# Collapse operator (σ_x for decoherence)
L_init = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# OpenCL Buffers
mf = cl.mem_flags
H_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H_init.view(np.float64))
L_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=L_init.view(np.float64))

# OpenCL Kernel
kernel_code = """
typedef double2 cplx_t;
cplx_t cplx_add(cplx_t a, cplx_t b) { return (cplx_t)(a.x + b.x, a.y + b.y); }
cplx_t cplx_sub(cplx_t a, cplx_t b) { return (cplx_t)(a.x - b.x, a.y - b.y); }
cplx_t cplx_mul(cplx_t a, cplx_t b) { return (cplx_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

__kernel void jc_ser(
    __global cplx_t* rho, __global cplx_t* d_rho, __global cplx_t* H, __global cplx_t* L,
    double gamma, double beta, int dim, double hbar) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= dim || j >= dim) return;

    cplx_t d_rho_ij = (cplx_t)(0.0, 0.0);

    // Hamiltonian term
    cplx_t H_rho = (cplx_t)(0.0, 0.0);
    cplx_t rho_H = (cplx_t)(0.0, 0.0);
    for (int k = 0; k < dim; k++) {
        H_rho = cplx_add(H_rho, cplx_mul(H[i * dim + k], rho[k * dim + j]));
        rho_H = cplx_add(rho_H, cplx_mul(rho[i * dim + k], H[k * dim + j]));
    }
    cplx_t comm = cplx_sub(H_rho, rho_H);
    d_rho_ij = cplx_mul((cplx_t)(-1.0 / hbar, 0.0), comm);

    // Lindblad term
    cplx_t L_rho_L = cplx_mul(cplx_mul(L[i * dim + j], rho[i * dim + j]), L[j * dim + i]);
    cplx_t gamma_term = cplx_mul((cplx_t)(gamma, 0.0), L_rho_L);
    d_rho_ij = cplx_add(d_rho_ij, gamma_term);

    // SER Feedback Term
    cplx_t ser_term = cplx_mul((cplx_t)(beta, 0.0), L_rho_L);
    d_rho_ij = cplx_add(d_rho_ij, ser_term);
    d_rho[i * dim + j] = d_rho_ij;
}
"""

prg = cl.Program(ctx, kernel_code).build()

# --- Run Simulation for Each Initial State ---
results = {}

for label, rho_init in initial_states.items():
    rho_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_init.view(np.float64))
    d_rho_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rho_init.nbytes)

    rho = rho_init.copy()

    # Storage lists
    rho_11_vals, entropy_vals, purity_vals, coherence_vals, beta_vals = [], [], [], [], []

    for step in range(num_steps):
        # Copy rho to buffer before running kernel
        cl.enqueue_copy(queue, rho_buf, rho.view(np.float64).flatten()).wait()

        # Set gamma and dynamically adjust beta
        gamma = 0.1
        coherence = np.abs(rho[0, 1])
        dynamic_beta = beta_base * np.exp(-2 * (1 - coherence))

        # Store beta values for analysis
        beta_vals.append(dynamic_beta)

        # Run OpenCL kernel
        prg.jc_ser(queue, (2, 2), None, rho_buf, d_rho_buf, H_buf, L_buf,
                   np.float64(gamma), np.float64(dynamic_beta), np.int32(2), np.float64(hbar))

        # Copy updated d_rho from device
        d_rho_host = np.empty_like(rho)
        cl.enqueue_copy(queue, d_rho_host, d_rho_buf).wait()

        # Euler update
        rho += dt * d_rho_host

        # Enforce positivity
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 1e-10)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho /= np.trace(rho)

        # Store observables
        rho_11_vals.append(rho[1, 1].real)
        entropy_vals.append(-np.sum(eigvals * np.log2(eigvals)))
        purity_vals.append(np.trace(np.dot(rho, rho)).real)
        coherence_vals.append(coherence)

    results[label] = {
        'rho_11': rho_11_vals,
        'entropy': entropy_vals,
        'purity': purity_vals,
        'coherence': coherence_vals,
        'beta_vals': beta_vals
    }

# --- Plot Results ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for label, data in results.items():
    time_axis = np.linspace(0, num_steps * dt, num_steps)
    axs[0, 0].plot(time_axis, data['rho_11'], label=label)
    axs[0, 1].plot(time_axis, data['entropy'], label=label)
    axs[1, 0].plot(time_axis, data['purity'], label=label)
    axs[1, 1].plot(time_axis, data['coherence'], label=label)

for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.show()
