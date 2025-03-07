// ==============================================
// You can define DSIZE at compile time:
// e.g. clBuildProgram with "-D DSIZE=12"
// ==============================================

#ifndef DSIZE
#define DSIZE 4
#endif

// Simple inline functions for complex arithmetic with double2
inline double2 cAdd(double2 a, double2 b) {
    return (double2)(a.x + b.x, a.y + b.y);
}
inline double2 cSub(double2 a, double2 b) {
    return (double2)(a.x - b.x, a.y - b.y);
}
inline double2 cMul(double2 a, double2 b) {
    // Complex multiply: (ax + i ay)*(bx + i by) = (ax*bx - ay*by) + i(ax*by + ay*bx)
    return (double2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline double2 cScale(double2 a, double s) {
    return (double2)(a.x*s, a.y*s);
}
// Multiply by +i: i*z = i*(x + i y) = -y + i x
inline double2 cMulI(double2 a) {
    return (double2)(-a.y, a.x);
}

// ==============================================
// KERNEL
// ==============================================
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
    // Dimension from compile-time define
    int d = DSIZE;

    // Global id in [0, d*d)
    int gid = get_global_id(0);
    if (gid >= d*d) return;

    // Map gid to matrix indices (row, col)
    int i = gid / d;
    int j = gid % d;

    // We'll copy all input matrices into private arrays of size d*d.
    // Because we used DSIZE, we can safely allocate up to DSIZE*DSIZE here.
    // If DSIZE=12, that's 144 elements, which is usually safe on GPU for local or private arrays.
    __private double2 rho_private[DSIZE*DSIZE];
    __private double2 H_private[DSIZE*DSIZE];
    __private double2 L_private[DSIZE*DSIZE];
    __private double2 I_private[DSIZE*DSIZE];

    // Load global memory into private arrays
    for (int k = 0; k < d*d; k++) {
        rho_private[k] = rho[k];
        H_private[k]   = H[k];
        L_private[k]   = L[k];
        I_private[k]   = I[k];
    }

    // -------------------------------------------------
    // 1) Compute commutator term:  (-i/hbar) [H, rho]
    // -------------------------------------------------
    double2 H_rho_ij = (double2)(0.0, 0.0);
    double2 rho_H_ij = (double2)(0.0, 0.0);
    for (int k = 0; k < d; k++) {
        int idx_ik = i*d + k;
        int idx_kj = k*d + j;
        // H*rho
        H_rho_ij = cAdd(H_rho_ij, cMul(H_private[idx_ik], rho_private[idx_kj]));
        // rho*H
        rho_H_ij = cAdd(rho_H_ij, cMul(rho_private[idx_ik], H_private[idx_kj]));
    }

    // comm_real = (H_rho_ij - rho_H_ij)/hbar
    double2 comm_real = cScale(cSub(H_rho_ij, rho_H_ij), 1.0/hbar);
    double2 commutator = cMulI(comm_real);
    commutator = cScale(commutator, -1.0);

    // -------------------------------------------------
    // 2) Lindblad dissipator: gamma * [ L rho L^\dagger - 0.5{L^\dagger L, rho} ]
    // -------------------------------------------------
    // We'll compute L_rho_Ldag, Ldag_L_rho, rho_Ldag_L just like your original code
    double2 L_rho_Ldag_ij = (double2)(0.0, 0.0);
    double2 Ldag_L_rho_ij = (double2)(0.0, 0.0);
    double2 rho_Ldag_L_ij = (double2)(0.0, 0.0);

    for (int k = 0; k < d; k++) {
        int idx_ik = i*d + k;
        int idx_kj = k*d + j;

        // Conjugate of L[k*d + j]
        double2 L_conj_kj = (double2)( L_private[idx_kj].x, -L_private[idx_kj].y );
        double2 L_conj_ik = (double2)( L_private[idx_ik].x, -L_private[idx_ik].y );

        // L_rho_Ldag
        // add L[i,k]*rho[k,j]*L^\dagger[j,k] => L[i,k]*rho[k,j]*conjugate(L[k,j])
        L_rho_Ldag_ij = cAdd(
            L_rho_Ldag_ij,
            cMul( cMul(L_private[idx_ik], rho_private[idx_kj]), L_conj_kj )
        );

        // Ldag_L_rho => L^\dagger[i,k]*L[k,j]*rho[j,j]? Wait, we want (L^\dagger L) * rho
        // But to keep consistent with your original code, we do something akin to:
        // L^\dagger[i,k]*L[k,j]*rho[j, j???]
        // Actually let's keep the same pattern you had:
        Ldag_L_rho_ij = cAdd(
            Ldag_L_rho_ij,
            cMul( cMul(L_conj_ik, L_private[idx_kj]), rho_private[idx_kj] )
        );

        // rho_Ldag_L => rho[i,k]*L^\dagger[k,j]*L[j,j]?
        // We'll do the same pattern:
        rho_Ldag_L_ij = cAdd(
            rho_Ldag_L_ij,
            cMul( cMul(rho_private[idx_ik], L_conj_kj), L_private[idx_kj] )
        );
    }

    // Combine them for the Lindblad term
    // lindblad_term = gamma * ( L_rho_Ldag - 0.5*( Ldag_L_rho + rho_Ldag_L ) )
    double2 ld_sum = cSub(L_rho_Ldag_ij, cScale(cAdd(Ldag_L_rho_ij, rho_Ldag_L_ij), 0.5));
    double2 lindblad_term = cScale(ld_sum, gamma);

    // -------------------------------------------------
    // 3) SER "feedback" term
    //    We'll replicate your logic with coherence_squared, F_rho = exp(-coherence^2), etc.
    // -------------------------------------------------
    double coherence_squared = 0.0;
    // compute sum of |rho[i,j]|^2 for i != j
    for (int ii = 0; ii < d; ii++) {
        for (int jj = 0; jj < d; jj++) {
            if (ii != jj) {
                int idx2 = ii*d + jj;
                double rx = rho_private[idx2].x;
                double ry = rho_private[idx2].y;
                coherence_squared += (rx*rx + ry*ry);
            }
        }
    }
    double F_rho = exp(-coherence_squared);

    // I_minus_rho = I - rho
    __private double2 I_minus_rho[DSIZE*DSIZE];
    for (int k = 0; k < d*d; k++) {
        I_minus_rho[k] = cSub(I_private[k], rho_private[k]);
    }

    // Next we need L_rho => L*rho
    __private double2 L_rho_full[DSIZE*DSIZE];
    for (int k = 0; k < d*d; k++) {
        L_rho_full[k] = (double2)(0.0, 0.0);
    }
    // multiply L*rho
    for (int row = 0; row < d; row++) {
        for (int col = 0; col < d; col++) {
            double2 sumVal = (double2)(0.0, 0.0);
            for (int m = 0; m < d; m++) {
                int idx_rm = row*d + m;
                int idx_mc = m*d + col;
                sumVal = cAdd(sumVal, cMul(L_private[idx_rm], rho_private[idx_mc]));
            }
            L_rho_full[row*d + col] = sumVal;
        }
    }

    // Then L_rho_Ldag_new => (L*rho)*L^\dagger
    __private double2 L_rho_Ldag_new[DSIZE*DSIZE];
    for (int k = 0; k < d*d; k++) {
        L_rho_Ldag_new[k] = (double2)(0.0, 0.0);
    }
    for (int row = 0; row < d; row++) {
        for (int col = 0; col < d; col++) {
            double2 sumVal = (double2)(0.0, 0.0);
            for (int m = 0; m < d; m++) {
                int idx_rm = row*d + m;
                int idx_mc = m*d + col;
                // conjugate of L[m*d + col]
                double2 L_conj = (double2)( L_private[idx_mc].x, -L_private[idx_mc].y );
                sumVal = cAdd(sumVal, cMul(L_rho_full[idx_rm], L_conj));
            }
            L_rho_Ldag_new[row*d + col] = sumVal;
        }
    }

    // Now compute SER_feedback = beta * F_rho * (I - rho)* (L_rho_Ldag_new)* (I - rho)
    __private double2 tempA[DSIZE*DSIZE];
    for (int k = 0; k < d*d; k++) {
        tempA[k] = (double2)(0.0, 0.0);
    }
    // temp = (I-rho)*(L_rho_Ldag_new)
    for (int row = 0; row < d; row++) {
        for (int col = 0; col < d; col++) {
            double2 sumVal = (double2)(0.0, 0.0);
            for (int m = 0; m < d; m++) {
                int idx_rm = row*d + m;
                int idx_mc = m*d + col;
                sumVal = cAdd(sumVal, cMul(I_minus_rho[idx_rm], L_rho_Ldag_new[idx_mc]));
            }
            tempA[row*d + col] = sumVal;
        }
    }

    // SER_feedback_final[row*d + col] = beta * F_rho * tempA * (I-rho)
    __private double2 SER_feedback_full[DSIZE*DSIZE];
    for (int k = 0; k < d*d; k++) {
        SER_feedback_full[k] = (double2)(0.0, 0.0);
    }
    for (int row = 0; row < d; row++) {
        for (int col = 0; col < d; col++) {
            double2 sumVal = (double2)(0.0, 0.0);
            for (int m = 0; m < d; m++) {
                int idx_rm = row*d + m;
                int idx_mc = m*d + col;
                sumVal = cAdd(sumVal, cMul(tempA[idx_rm], I_minus_rho[idx_mc]));
            }
            sumVal = cScale(sumVal, beta * F_rho);
            SER_feedback_full[row*d + col] = sumVal;
        }
    }

    // Combine all pieces for d(rho)/dt at (i,j)
    // drho_dt_ij = commutator + lindblad_term + SER_feedback
    // (Note: commutator + lindblad_term + SER_feedback) must be added at index (i*d + j).
    double2 drho_dt_ij = (double2)(0.0, 0.0);
    // add commutator
    drho_dt_ij = cAdd(drho_dt_ij, commutator);
    // add Lindblad
    drho_dt_ij = cAdd(drho_dt_ij, lindblad_term);
    // add SER feedback
    drho_dt_ij = cAdd(drho_dt_ij, SER_feedback_full[gid]);

    // Update rho
    // rho[i,j] += dt * drho_dt_ij
    double2 updatedVal = cAdd(rho_private[gid], cScale(drho_dt_ij, dt));

    // Write it back to global memory
    rho[gid] = updatedVal;
}
