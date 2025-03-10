Overview

This repository contains the Python and OpenCL code for simulating the Structured Energy Return (SER) model, a feedback-based modification to the Lindblad 
equation aimed at reshaping decoherence in quantum systems. The model introduces a structured feedback mechanism that can slow, redistribute, or partially 
reverse coherence loss in open quantum systems.

This implementation allows for 2×2 and 4×4 density matrix simulations, showcasing how SER modifies standard quantum evolution.

Features

Implements the SER-modified Lindblad equation to study decoherence and feedback effects.
Supports both 2×2 and 4×4 quantum systems (selected at runtime).
Tracks key quantities:
Purity (Tr(ρ²))
Entropy (von Neumann entropy)
Coherence (off-diagonal elements)
Trace conservation check
Uses OpenCL for acceleration, enabling efficient simulations on compatible hardware.
Outputs results in CSV format and generates plots of system evolution.

Requirements

This simulation requires:

Python 3.x
NumPy
Matplotlib
PyOpenCL (for GPU acceleration)

To install dependencies, run:

pip install numpy matplotlib pyopencl

Usage Instructions

Clone the repository and navigate to the directory:


git clone AngryBeaverss/SER

cd SER

Ensure OpenCL is installed and configured correctly for your system. You can check OpenCL compatibility by running:


import pyopencl as cl
print(cl.get_platforms())
Run the simulation script:

python Ser6_dynamic.py

Select the system size when prompted:

Enter 2 for a 2×2 density matrix simulation
Enter 4 for a 4×4 density matrix simulation
The simulation will run for a specified time duration, displaying progress every 10,000 steps.

Output Files Generated:

rho_final.npy: Final density matrix (NumPy binary format).
rho_final.csv: Final density matrix in human-readable CSV format.
SER_results.csv: Evolution of purity, entropy, and coherence over time.
SER_plot.png & SER_plot.pdf: Graphical plots of the simulation results.
To visualize the results, open SER_plot.png or re-run the plotting section in Python.

Expected Behavior

2×2 systems: The system may re-purify, approaching a nearly pure state under SER feedback.
4×4 systems: The system typically stabilizes at a partially mixed state with nonzero coherence, rather than fully decohering.
Entropy Evolution: Unlike standard Lindblad decay, entropy may peak and then decrease under strong SER feedback.

Future Work

Optimize OpenCL kernel for larger systems (beyond 4×4).
Test in real quantum hardware (e.g., cavity QED, superconducting qubits).
Improve numerical stability in high-feedback scenarios.
Expand to entanglement-preserving models.

License & Credits

Author: Ryan Wallace

This project is shared for research and verification purposes. Contributions and discussions are welcome.

If using this model in research, please cite appropriately.
