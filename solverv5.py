import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags

########################################################
#                 1. CONFIGURATION                    #
########################################################

# Simulation constants
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
m = 9.10938356e-31    # Electron mass (kg)
L = 1e-9              # System size in meters (1 nm)
N = 200               # Spatial resolution
dx = L / N            # Spatial step size
x = np.linspace(0, L, N)

# Dissipation and measurement parameters
gamma = 0.01               # Dissipation strength
beta_SER = 0.0025          # Structured energy return factor (for SER runs)
measurement_strength = 0.03 # Measurement noise amplitude

# Time parameters
n_time_steps = 1000
time = np.linspace(0, 5e-14, n_time_steps)  # seconds

# Ensemble parameters
n_seeds = 5  # Number of seeds for ensemble averaging

########################################################
#                2. HAMILTONIAN SETUP                  #
########################################################

def hamiltonian_sparse(N, dx):
    """Construct the 1D kinetic-energy Hamiltonian using a sparse matrix."""
    coeff = -hbar**2 / (2 * m * dx**2)
    main_diag = np.full(N, -2 * coeff)
    off_diag = np.full(N - 1, coeff)
    # Using CSR format for efficient linear operations
    return diags([main_diag, off_diag, off_diag], [0, -1, 1], format="csr")

H = hamiltonian_sparse(N, dx)

########################################################
#                3. INITIAL CONDITIONS                 #
########################################################

def initial_wavefunction(x, k0=5*np.pi/L):
    """Defines a Gaussian wave packet centered in the middle of the box."""
    width = L/10
    psi = np.exp(-(x - L/2)**2 / (2 * width**2)) * np.exp(1j * k0 * x)
    # Normalize
    norm_factor = np.linalg.norm(psi)
    if norm_factor != 0:
        psi /= norm_factor
    return psi

psi0 = initial_wavefunction(x)

########################################################
#                4. HELPER FUNCTIONS                   #
########################################################


def compute_purity(psi):
    """Compute purity = sum(|psi|^4)."""
    return np.sum(np.abs(psi)**4)


def compute_entropy(psi):
    """Compute a simplified von Neumann-like entropy based on the diagonal of the density matrix.
    We assume a pure state for this 1D wavefunction, so the typical definition would be S = -Tr(rho log rho).
    But as a proxy, we can treat |psi|^2 as probabilities and compute:
      S = - sum(p_i log p_i)
    where p_i = |psi_i|^2.
    """
    p = np.abs(psi)**2
    p = np.clip(p, 1e-15, 1.0)  # avoid log(0)
    return -np.sum(p * np.log(p))


def structured_energy_return(psi, beta):
    """Implements a smooth saturated SER function: beta * tanh(|psi|^2) * psi"""
    E = np.abs(psi)**2
    return beta * np.tanh(E) * psi


def schrodinger_equation(t, y, H, gamma, beta, measurement_strength):
    """Differential equation for the wavefunction, with or without SER."""
    psi_real = y[:N]
    psi_imag = y[N:]
    psi = psi_real + 1j * psi_imag

    # Probability for dynamic noise scaling
    prob = np.sum(np.abs(psi)**2)

    # Stochastic measurement noise
    noise_term = measurement_strength * np.random.randn(N) * np.sqrt(max(prob, 1e-6))

    # SER term if beta != 0
    if beta != 0:
        ser_term = structured_energy_return(psi, beta)
    else:
        ser_term = 0.0

    # Hamiltonian operation
    H_psi = H.dot(psi)

    # dpsi_dt: standard Schrodinger + dissipation + SER + measurement noise
    dpsi_dt = (-1j / hbar) * H_psi - gamma * psi + ser_term - noise_term * psi

    return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

########################################################
#         5. SINGLE RUN SIMULATION FUNCTION            #
########################################################

def run_simulation(
    beta=0.0,
    seed=None,
    time_array=time,
    psi_init=psi0,
    solver_method='DOP853',
    rtol=1e-10,
    atol=1e-12,
    max_step=np.inf
):
    """Run a single simulation with given parameters, returning wavefunction array."""
    if seed is not None:
        np.random.seed(seed)

    y0 = np.concatenate([np.real(psi_init), np.imag(psi_init)])

    sol = solve_ivp(
        schrodinger_equation,
        [time_array[0], time_array[-1]],
        y0,
        t_eval=time_array,
        args=(H, gamma, beta, measurement_strength),
        method=solver_method,
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )

    psi_evol = sol.y[:N] + 1j * sol.y[N:]
    return psi_evol, sol.t

########################################################
#         6. MULTI-SEED (ENSEMBLE) SIMULATION          #
########################################################

def ensemble_simulation(
    beta=0.0,
    n_seeds=3,
    time_array=time,
    psi_init=psi0
):
    """Runs multiple seeds and returns averaged purity, entropy, and probability over time."""

    # Storage arrays
    all_purity = []
    all_entropy = []
    all_probability = []

    for seed in range(n_seeds):
        psi_evol, t_out = run_simulation(beta=beta, seed=seed, time_array=time_array, psi_init=psi_init)
        prob_density = np.abs(psi_evol)**2

        # Compute metrics
        purity_vals = np.array([compute_purity(psi_evol[:, i]) for i in range(len(t_out))])
        entropy_vals = np.array([compute_entropy(psi_evol[:, i]) for i in range(len(t_out))])
        total_prob = np.sum(prob_density, axis=0)

        all_purity.append(purity_vals)
        all_entropy.append(entropy_vals)
        all_probability.append(total_prob)

    # Convert to numpy and average
    all_purity = np.array(all_purity)
    all_entropy = np.array(all_entropy)
    all_probability = np.array(all_probability)

    # Mean across seeds
    avg_purity = np.mean(all_purity, axis=0)
    avg_entropy = np.mean(all_entropy, axis=0)
    avg_probability = np.mean(all_probability, axis=0)

    return t_out, avg_purity, avg_entropy, avg_probability

########################################################
#         7. RUN BOTH STANDARD QM AND SER MODEL        #
########################################################

if __name__ == "__main__":
    # Standard QM (beta=0)
    t_std, purity_std, entropy_std, prob_std = ensemble_simulation(beta=0.0, n_seeds=n_seeds)

    # Structured Energy Return (beta=beta_SER)
    t_ser, purity_ser, entropy_ser, prob_ser = ensemble_simulation(beta=beta_SER, n_seeds=n_seeds)

    # Convert time to femtoseconds
    t_fs_std = t_std * 1e15
    t_fs_ser = t_ser * 1e15

    ########################################################
    # 8. PLOT COMPARISONS
    ########################################################
    plt.figure(figsize=(8, 5))
    plt.plot(t_fs_std, prob_std, label="Total Probability (QM)", color='blue', linestyle='--')
    plt.plot(t_fs_ser, prob_ser, label="Total Probability (SER)", color='orange')
    plt.xlabel("Time (fs)")
    plt.ylabel("Average Probability")
    plt.title("Ensemble-Averaged Total Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t_fs_std, purity_std, label="Purity (QM)", color='blue', linestyle='--')
    plt.plot(t_fs_ser, purity_ser, label="Purity (SER)", color='orange')
    plt.xlabel("Time (fs)")
    plt.ylabel("Average Purity")
    plt.title("Ensemble-Averaged Purity")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t_fs_std, entropy_std, label="Entropy (QM)", color='blue', linestyle='--')
    plt.plot(t_fs_ser, entropy_ser, label="Entropy (SER)", color='orange')
    plt.xlabel("Time (fs)")
    plt.ylabel("Average Entropy")
    plt.title("Ensemble-Averaged Entropy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final values
    print("\n========= FINAL VALUES (Ensemble-Averaged) =========")
    print(f"QM Final Probability  : {prob_std[-1]:.4e}")
    print(f"SER Final Probability : {prob_ser[-1]:.4e}")
    print(f"QM Final Purity       : {purity_std[-1]:.4e}")
    print(f"SER Final Purity      : {purity_ser[-1]:.4e}")
    print(f"QM Final Entropy      : {entropy_std[-1]:.4e}")
    print(f"SER Final Entropy     : {entropy_ser[-1]:.4e}\n")
