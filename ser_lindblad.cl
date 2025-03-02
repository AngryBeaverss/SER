__kernel void ser_lindblad(
    __global double2 *rho,
    __global double2 *H,
    __global double2 *L,
    __global double2 *I,
    double hbar,
    double gamma,
    double beta,
    double dt)
{
    int gid = get_global_id(0);
    if (gid >= 4) return;  // Ensure gid is within bounds for a 2x2 system

    // Map gid to matrix indices
    int i = gid / 2;
    int j = gid % 2;

    // Load matrices into private memory
    double2 rho_private[4];
    double2 H_private[4];
    double2 L_private[4];
    double2 I_private[4];
    for (int k = 0; k < 4; k++) {
        rho_private[k] = rho[k];
        H_private[k]   = H[k];
        L_private[k]   = L[k];
        I_private[k]   = I[k];
    }

    // Compute commutator [H, rho]
    double2 H_rho_ij = (double2)(0.0, 0.0);
    double2 rho_H_ij = (double2)(0.0, 0.0);
    for (int k = 0; k < 2; k++) {
        int idx_ik = i * 2 + k;
        int idx_kj = k * 2 + j;
        H_rho_ij += H_private[idx_ik] * rho_private[idx_kj];
        rho_H_ij += rho_private[idx_ik] * H_private[idx_kj];
    }
    double2 commutator = H_rho_ij - rho_H_ij;

    // Compute Lindblad dissipator
    double2 L_rho_Ldag = (double2)(0.0, 0.0);
    double2 Ldag_L_rho = (double2)(0.0, 0.0);
    double2 rho_Ldag_L = (double2)(0.0, 0.0);
    for (int k = 0; k < 2; k++) {
        int idx_ik = i * 2 + k;
        int idx_kj = k * 2 + j;
        L_rho_Ldag += L_private[idx_ik] * rho_private[idx_kj] * L_private[idx_kj];
        Ldag_L_rho += L_private[idx_ik] * L_private[idx_kj] * rho_private[idx_kj];
        rho_Ldag_L += rho_private[idx_ik] * L_private[idx_kj] * L_private[idx_kj];
    }
    double2 lindblad_term = gamma * (L_rho_Ldag - 0.5 * (Ldag_L_rho + rho_Ldag_L));

    // Compute feedback function F(rho) = exp(-coherence_squared)
    double coherence_squared = rho_private[1].x * rho_private[1].x + rho_private[1].y * rho_private[1].y;
    double F_rho = exp(-coherence_squared);

    // Compute (I - rho)
    double2 I_minus_rho[4];
    for (int k = 0; k < 4; k++) {
        I_minus_rho[k] = I_private[k] - rho_private[k];
    }

    // Compute L * rho
    double2 L_rho[4];
    for (int k = 0; k < 4; k++) {
        L_rho[k] = (double2)(0.0, 0.0);
        for (int m = 0; m < 2; m++) {
            int idx_km = k / 2 * 2 + m;
            int idx_mj = m * 2 + k % 2;
            L_rho[k] += L_private[idx_km] * rho_private[idx_mj];
        }
    }

    // Compute L * rho * Ldag
    double2 L_rho_Ldag_new[4];
    for (int k = 0; k < 4; k++) {
        L_rho_Ldag_new[k] = (double2)(0.0, 0.0);
        for (int m = 0; m < 2; m++) {
            int idx_km = k / 2 * 2 + m;
            int idx_mj = m * 2 + k % 2;
            L_rho_Ldag_new[k] += L_rho[idx_km] * L_private[idx_mj];
        }
    }

    // Compute SER feedback term
    double2 SER_feedback[4];
    for (int k = 0; k < 4; k++) {
        SER_feedback[k] = (double2)(0.0, 0.0);
        for (int m = 0; m < 2; m++) {
            int idx_km = k / 2 * 2 + m;
            int idx_mj = m * 2 + k % 2;
            SER_feedback[k] += I_minus_rho[idx_km] * L_rho_Ldag_new[idx_mj] * I_minus_rho[k];
        }
        SER_feedback[k] *= beta * F_rho;
    }

    // Compute total derivative of rho
    double2 drho_dt = (-1.0 / hbar) * commutator + lindblad_term + SER_feedback[gid];

    // Update density matrix
    rho[gid] += dt * drho_dt;
}
