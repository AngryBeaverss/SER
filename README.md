## Overview  
This repository provides Python implementations and analysis scripts for the **Structured Energy Return (SER)** model—a nonlinear feedback mechanism designed to combat decoherence in open quantum systems. The SER model actively reshapes coherence loss, enabling partial or near-complete coherence restoration in quantum-optical scenarios.

The repository contains:

- `rkSER.py`: Implementation of the SER model using an adaptive Runge-Kutta (RK45) integrator for robust and stable numerical solutions.
- `SERsweep.py`: Parameter sweep script exploring the effects of varying feedback and coupling strengths, facilitating a comprehensive understanding of the SER model's robustness and scale-invariance.
- `SER_Parameter_Sweep_Extension.pdf`: Extended documentation detailing theoretical foundations, numerical analyses, and practical implications of the SER model.

## What is Structured Energy Return (SER)?
SER introduces a state-dependent nonlinear feedback term into the standard Lindblad master equation:

dρ/dt = - (i/ℏ) [H, ρ] + γ (LρL† - ½{L†L, ρ}) + β F(ρ) (I - ρ) LρL† (I - ρ)


- **Feedback Strength (β):** Controls the intensity of coherence restoration.
- **SER mechanism** measures how far the density matrix ρ is from a pure state and acts to reduce mixedness.

## Features and Results

### Numerical Highlights:
- **Positivity Enforcement:** Ensures physical validity by projecting negative eigenvalues back to zero, crucial for stability.
- **Dimension Variability:** Successfully tested on:
  - 2×2 systems: Re-purifies the system to near-pure states.
  - 4×4 Systems: Achieves stable states with moderate purity and significant coherence.
- **Realistic Quantum Systems (Jaynes–Cummings Model):** Demonstrates coherence preservation and sustained Rabi oscillations.

### Parameter Sweep Analysis:
- Systematically explores coupling strengths (50–200 MHz) and feedback strengths (β = 0–3.0).
- Demonstrates robust scale-invariance and clear trends toward optimized coherence.

## Included Files:

| File                            | Description |
|---------------------------------|--------------------------------------------|
| `rkSER.py`                      | Core Python implementation using adaptive RK45 integrator for stability and accuracy. |
| `SERsweep.py`                   | Performs systematic parameter sweeps to analyze the SER model’s robustness. Includes plotting routines for visual analysis of coherence vs. coupling and feedback strengths. |
| `SER_Parameter_Sweep_Extension.pdf` | Comprehensive report detailing the evolution, implementation, numerical stability considerations, and experimental implications of the SER model. |

## Usage Instructions

1. **Dependencies:**
   - Python 3.x
   - NumPy, SciPy (`solve_ivp` integrator), Matplotlib

   Install via pip:
   ```bash
   pip install numpy scipy matplotlib
   python rkSER.py
   python SERsweep.py
