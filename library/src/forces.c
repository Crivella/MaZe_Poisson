#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "mpi_base.h"
#include "linalg.h"
#include "omp_base.h"

static inline void pbc_displacement(
    const double *pos, long int ia, long int ib, double L, double *dx, double *dy, double *dz, double *dr2
) {
    double x = pos[ib * 3]     - pos[ia * 3];
    double y = pos[ib * 3 + 1] - pos[ia * 3 + 1];
    double z = pos[ib * 3 + 2] - pos[ia * 3 + 2];
    x -= L * nearbyint(x / L);
    y -= L * nearbyint(y / L);
    z -= L * nearbyint(z / L);
    *dx = x;
    *dy = y;
    *dz = z;
    *dr2 = x * x + y * y + z * z;
}

double compute_forces_harmonic_bond(long int n_p, const double *pos, double *forces, double L, double k, double r0_val) {
    double energy = 0.0;
    const double eps = 1e-15;

    long int n3 = n_p * 3;
    long int n_triplets = n_p / 3; // Assuming water-like molecules with 3 sites per molecule (O, H1, H2).
    double *fcs = forces;

    memset(fcs, 0, n3 * sizeof(double));

    #pragma omp parallel for reduction(+:energy)
    for (long int m = 0; m < n_triplets; m++) {
        long int iO = m * 3;
        long int iH1 = iO + 1;
        long int iH2 = iO + 2;

        double dx, dy, dz, dr2, dr, fab;

        // O - H1
        pbc_displacement(pos, iO, iH1, L, &dx, &dy, &dz, &dr2);
        dr = sqrt(dr2);
        if (dr2 > eps) {
            fab = -k * (dr - r0_val) / dr;
        } else {
            fab = 0.0;
        }
        fcs[iH1 * 3    ] += dx * fab;
        fcs[iH1 * 3 + 1] += dy * fab;
        fcs[iH1 * 3 + 2] += dz * fab;
        fcs[iO  * 3    ] -= dx * fab;
        fcs[iO  * 3 + 1] -= dy * fab;
        fcs[iO  * 3 + 2] -= dz * fab;
        energy += 0.5 * k * (dr - r0_val) * (dr - r0_val);

        // O - H2
        pbc_displacement(pos, iO, iH2, L, &dx, &dy, &dz, &dr2);
        dr = sqrt(dr2);
        if (dr2 > eps) {
            fab = -k * (dr - r0_val) / dr;
        } else {
            fab = 0.0;
        }
        fcs[iH2 * 3    ] += dx * fab;
        fcs[iH2 * 3 + 1] += dy * fab;
        fcs[iH2 * 3 + 2] += dz * fab;
        fcs[iO  * 3    ] -= dx * fab;
        fcs[iO  * 3 + 1] -= dy * fab;
        fcs[iO  * 3 + 2] -= dz * fab;
        energy += 0.5 * k * (dr - r0_val) * (dr - r0_val);
    }

    return energy;
}

double compute_forces_harmonic_angle(long int n_p, const double *pos, double *forces, double L, double k, double theta0_val) {
    double energy = 0.0;
    const double eps = 1e-15;

    long int n3 = n_p * 3;
    long int n_triplets = n_p / 3; // Assuming water-like molecules with 3 sites per molecule (O, H1, H2).
    double *fcs = forces;

    memset(fcs, 0, n3 * sizeof(double));

    #pragma omp parallel for reduction(+:energy)
    for (long int m = 0; m < n_triplets; m++) {
        long int iO = m * 3;
        long int iH1 = iO + 1;
        long int iH2 = iO + 2;

        double dxab, dyab, dzab, drab2, drab;
        double dxac, dyac, dzac, drac2, drac;
        double dxbc, dybc, dzbc, drbc2, drbc;

        pbc_displacement(pos, iO, iH1, L, &dxab, &dyab, &dzab, &drab2);
        pbc_displacement(pos, iO, iH2, L, &dxac, &dyac, &dzac, &drac2);
        pbc_displacement(pos, iH1, iH2, L, &dxbc, &dybc, &dzbc, &drbc2);

        drab = sqrt(drab2);
        drac = sqrt(drac2);
        drbc = sqrt(drbc2);
        if (drab2 < eps || drac2 < eps || drbc2 < eps) {
            continue;
        }

        double cos_theta = (drab2 + drac2 - drbc2) / (2.0 * drab * drac);
        if (cos_theta > 1.0) {
            cos_theta = 1.0;
        } else if (cos_theta < -1.0) {
            cos_theta = -1.0;
        }
        double theta = acos(cos_theta);

        double dudtheta = k * (theta - theta0_val);
        double dacosdz;
        if (cos_theta * cos_theta == 1.0) {
            dacosdz = 0.0;
        } else {
            dacosdz = -1.0 / sqrt(1.0 - cos_theta * cos_theta);
        }

        double dzdab = (drab2 - drac2 + drbc2) / (2.0 * drab2 * drac);
        double dzdac = (drac2 - drab2 + drbc2) / (2.0 * drac2 * drab);
        double dzdbc = -drbc / (drac * drab);

        double fab = -(dudtheta * dacosdz * dzdab) / drab;
        double fac = -(dudtheta * dacosdz * dzdac) / drac;
        double fbc = -(dudtheta * dacosdz * dzdbc) / drbc;

        // ab contribution (O-H1)
        fcs[iH1 * 3    ] += dxab * fab;
        fcs[iH1 * 3 + 1] += dyab * fab;
        fcs[iH1 * 3 + 2] += dzab * fab;
        fcs[iO  * 3    ] -= dxab * fab;
        fcs[iO  * 3 + 1] -= dyab * fab;
        fcs[iO  * 3 + 2] -= dzab * fab;

        // ac contribution (O-H2)
        fcs[iH2 * 3    ] += dxac * fac;
        fcs[iH2 * 3 + 1] += dyac * fac;
        fcs[iH2 * 3 + 2] += dzac * fac;
        fcs[iO  * 3    ] -= dxac * fac;
        fcs[iO  * 3 + 1] -= dyac * fac;
        fcs[iO  * 3 + 2] -= dzac * fac;

        // bc contribution (H1-H2)
        fcs[iH2 * 3    ] += dxbc * fbc;
        fcs[iH2 * 3 + 1] += dybc * fbc;
        fcs[iH2 * 3 + 2] += dzbc * fbc;
        fcs[iH1 * 3    ] -= dxbc * fbc;
        fcs[iH1 * 3 + 1] -= dybc * fbc;
        fcs[iH1 * 3 + 2] -= dzbc * fbc;

        energy += 0.5 * k * (theta - theta0_val) * (theta - theta0_val);
    }

    return energy;
}

double compute_force_short_range(
    int n_p,
    double *pos,
    double *charges,
    double *forces, // Output forces on each particle (n_p, 3)
    double R_c,
    double sigma_gauss,
    double L
) {
    double shift_potential, potential = 0.0;

    double R_c2 = R_c * R_c;
    double inv_rc = 1.0 / R_c;
    double inv_r2c = inv_rc * inv_rc;
    double inv_r3c = inv_r2c * inv_rc;
    double xc = R_c / (sqrt(2.0) * sigma_gauss);
    double erf_term_c = 1.0 - erf(xc);
    double exp_term_c = exp(-xc*xc);

    for (int ip = 0; ip < n_p; ip++) {
        // mpi_fprintf(stderr,"Computing short-range forces for particle %d...\n", ip);
        double px = pos[3*ip];
        double py = pos[3*ip + 1];
        double pz = pos[3*ip + 2];
        double qi = charges[ip];

        for (int jp = ip + 1; jp < n_p; jp++) {
            double dx = px - pos[3*jp];
            double dy = py - pos[3*jp + 1];
            double dz = pz - pos[3*jp + 2];

            // PBC
            dx -= L * round(dx / L);
            dy -= L * round(dy / L);
            dz -= L * round(dz / L);

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > R_c2 || r2 == 0.0) continue;
    
            double r = sqrt(r2);

            double qj = charges[jp];

            double inv_r = 1.0 / r;
            double inv_r2 = inv_r * inv_r;
            double inv_r3 = inv_r2 * inv_r;

            double x = r / (sqrt(2.0) * sigma_gauss);

            double erf_term = 1.0 - erf(x);
            double exp_term = exp(-x*x);


            double factor_c =
                qi * qj *
                (
                    erf_term_c * inv_r3c +
                    (sqrt(2.0) / (sqrt(M_PI) * sigma_gauss)) * exp_term_c * inv_r2c
                );

            double factor =
                qi * qj *
                (
                    erf_term * inv_r3 +
                    (sqrt(2.0) / (sqrt(M_PI) * sigma_gauss)) * exp_term * inv_r2
                );

            //Apply shifted of the forces to ensure that the forces go to zero at the cutoff distance
            double shift = factor - factor_c;

            double fx = shift * dx;
            double fy = shift * dy;
            double fz = shift * dz;

            forces[3*ip]     += fx;
            forces[3*ip + 1] += fy;
            forces[3*ip + 2] += fz;

            // Use symmetry to update the force on particle jp
            forces[3*jp]     -= fx;
            forces[3*jp + 1] -= fy;
            forces[3*jp + 2] -= fz;

            // potentiel
            shift_potential = qi * qj * erf_term_c * inv_rc;
            potential += qi * qj * erf_term / r - shift_potential;

            // mpi_fprintf(stderr,"Short-range force between particles %d and %d: fx = %f, fy = %f, fz = %f\n", ip, jp, fx, fy, fz);
        }
    }
    return potential; 
}

// /*
// Compute the forces on each particle by computing the field from the potential using finite differences.
// New version computes the field only where the particles are located.

// @param n_grid: the number of grid points in each dimension
// @param n_p: the number of particles
// @param h: the grid spacing
// @param num_neigh: the number of neighbors for each particle
// @param phi: the potential field of size n_grid * n_grid * n_grid
// @param neighbors: Array (x,y,z) of neighbors indexes for each particle (n_p x 8 x 3)
// @param charges: the charges on each particle of size n_p
// @param pos: the positions of the particles of size n_p * 3
// @param forces: the output forces on each particle of size n_p * 3
// @param g: the function to compute the charge assignment

// @return the sum of the charges on the neighbors
// */
double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double)
) {
    int nn3 = num_neigh * 3;
    long int n = n_grid;
    long int n2 = n * n;

    long int i, j, k, jn, in2;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;
    double E, qc;
    
    int n_loc = get_n_loc();
    int n_start = get_n_start();

    double const h2 = 2.0 * h;
    double const L = n * h;
    double px, py, pz, chg;
    
    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(phi, n_loc, n);

    double sum_q = 0.0;
    #pragma omp parallel for private(i, j, k, i0, i1, i2, in2, j0, j1, j2, jn, k0, k1, k2, E, qc, px, py, pz, chg) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip * nn3;
        j0 = ip*3;
        forces[j0] = 0.0;
        forces[j0+1] = 0.0;
        forces[j0+2] = 0.0;
        px = pos[j0];
        py = pos[j0 + 1];
        pz = pos[j0 + 2];
        chg = charges[ip];

        // printf("ip: %d, chg: %f, px: %f, py: %f, pz: %f L: %f, h: %f\n", ip, chg, px, py, pz, L, h);
        for (int in = 0; in < nn3; in += 3) {
            i1 = i0 + in;
            i = neighbors[i1] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            in2 = i * n2;
            jn = j * n;

            qc = chg * g(px - (i+n_start)*h, L, h) * g(py - j*h, L, h) * g(pz - k*h, L, h);
            sum_q += qc;
            // X
            i1 = (i+1) * n2;
            i2 = (i-1) * n2;
            E = (phi[i2 + jn + k] - phi[i1 + jn + k]) / h2;
            forces[j0] += qc * E;
            // Y
            j1 = ((j+1) % n) * n;
            j2 = ((j-1 + n) % n) * n;
            E = (phi[in2 + j2 + k] - phi[in2 + j1 + k]) / h2;
            forces[j0 + 1] += qc * E;
            // Z
            k1 = ((k+1) % n);
            k2 = ((k-1 + n) % n);
            E = (phi[in2 + jn + k2] - phi[in2 + jn + k1]) / h2;
            forces[j0 + 2] += qc * E;
        }
    }
  
    allreduce_sum(&sum_q, 1);
    allreduce_sum(forces, 3 * n_p);
    
    return 0.0;
}

/*
Compute the particle-particle forces using the tabulated Tosi-Fumi potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [A, B, C, D, sigma, alpha, beta] (7, n_p, n_p)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_tf_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
    int ip, jp;
    int n_p2 = 2 * n_p;
    long int n_p_pow2 = n_p * n_p;
    long int idx1, idx2;

    double *A = params;
    double *B = A + n_p_pow2;
    double *C = B + n_p_pow2;
    double *D = C + n_p_pow2;
    double *sigma_TF = D + n_p_pow2;
    double *alpha = sigma_TF + n_p_pow2;
    double *beta = alpha + n_p_pow2;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double potential_energy = 0.0;
    double a, b, c, d, sigma, al, be;

    #pragma omp parallel for private(app, ip, jp, r_diff, r_mag, f_mag, V_mag, a, b, c, d, sigma, al, be, idx1, idx2) reduction(+:potential_energy)
    for (int i = 0; i < n_p; i++) {
        r_mag = 0.0;
        ip = i * 3;
        idx1 = i * n_p;
        forces[ip] = 0.0;
        forces[ip + 1] = 0.0;
        forces[ip + 2] = 0.0;
        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            jp = 3 * j;
            app = pos[ip] - pos[jp];
            app -= L * round(app / L);
            r_mag = app * app;
            r_diff[0] = app;
            app = pos[ip + 1] - pos[jp + 1];
            app -= L * round(app / L);
            r_diff[1] = app;
            r_mag += app * app;
            app = pos[ip + 2] - pos[jp + 2];
            app -= L * round(app / L);
            r_diff[2] = app;
            r_mag += app * app;
            r_mag = sqrt(r_mag);
            if (!isfinite(r_mag) || r_mag <= 1e-12) {
                mpi_fprintf(
                    stderr,
                    "Error: TF r_mag non-finite or too small (i=%d j=%d r=%e). "
                    "pos_i=(%e %e %e) pos_j=(%e %e %e)\n",
                    i, j, r_mag,
                    pos[ip], pos[ip + 1], pos[ip + 2],
                    pos[jp], pos[jp + 1], pos[jp + 2]
                );
                exit(1);
            }
            if (r_mag > r_cut) {
                continue;
            }
            r_diff[0] /= r_mag;
            r_diff[1] /= r_mag;
            r_diff[2] /= r_mag;
                
            idx2 = idx1 + j;
            a = A[idx2];
            b = B[idx2];
            c = C[idx2];
            d = D[idx2];
            sigma = sigma_TF[idx2];
            al = alpha[idx2];
            be = beta[idx2];

            f_mag = b * a * exp(b * (sigma - r_mag)) - 6 * c / pow(r_mag, 7) - 8 * d / pow(r_mag, 9) - al;
            V_mag = a * exp(b * (sigma - r_mag)) - c / pow(r_mag, 6) - d / pow(r_mag, 8) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    return potential_energy / 2;
}

/*
Compute the particle-particle forces using the tabulated Lennard-Jones potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [sigma, epsilon] (4, n_p, n_p)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
static double compute_lj_tail_correction(int n_p, double L, double *params, double r_cut) {
    long int n_p_pow2 = n_p * n_p;
    double *sigma_lj = params;
    double *epsilon_lj = sigma_lj + n_p_pow2;
    double volume = L * L * L;
    double tail = 0.0;

    if (r_cut <= 0.0 || volume <= 0.0) {
        return 0.0;
    }

    for (int i = 0; i < n_p; i++) {
        long int idx1 = i * n_p;
        for (int j = 0; j < n_p; j++) {
            long int idx = idx1 + j;
            double sigma = sigma_lj[idx];
            double epsilon = epsilon_lj[idx];
            if (epsilon == 0.0) {
                continue;
            }

            double sigma2 = sigma * sigma;
            double sigma3 = sigma2 * sigma;
            double sigma6 = sigma3 * sigma3;
            double sigma12 = sigma6 * sigma6;
            double rc3 = r_cut * r_cut * r_cut;
            double rc9 = rc3 * rc3 * rc3;
            tail += epsilon * (sigma12 / (9.0 * rc9) - sigma6 / (3.0 * rc3));
        }
    }

    return 8.0 * M_PI * tail / volume;
}

double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces, int lj_force_shift) {
    int ip, jp;
    int n_p2 = 2 * n_p;
    long int n_p_pow2 = n_p * n_p;
    long int idx1, idx2;

    double *sigma_lj = params;
    double *epsilon_lj = sigma_lj + n_p_pow2;
    double *alpha = epsilon_lj + n_p_pow2;
    double *beta = alpha + n_p_pow2;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double potential_energy = 0.0;
    double epsilon, sigma, al, be;

    #pragma omp parallel for private(app, ip, jp, r_diff, r_mag, f_mag, V_mag, epsilon, sigma, al, be, idx1, idx2) reduction(+:potential_energy)
    for (int i = 0; i < n_p; i++) {
        r_mag = 0.0;
        ip = i * 3;
        idx1 = i * n_p;
        forces[ip] = 0.0;
        forces[ip + 1] = 0.0;
        forces[ip + 2] = 0.0;
        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            jp = 3 * j;
            app = pos[ip] - pos[jp];
            app -= L * round(app / L);
            r_mag = app * app;
            r_diff[0] = app;
            app = pos[ip + 1] - pos[jp + 1];
            app -= L * round(app / L);
            r_diff[1] = app;
            r_mag += app * app;
            app = pos[ip + 2] - pos[jp + 2];
            app -= L * round(app / L);
            r_diff[2] = app;
            r_mag += app * app;
            r_mag = sqrt(r_mag);
            if (!isfinite(r_mag) || r_mag <= 1e-12) {
                // TODO: The first one is redundnat with the next check
                //  The second one should never happen?
                continue;
            }
            if (r_mag > r_cut) {
                continue;
            }
            r_diff[0] /= r_mag;
            r_diff[1] /= r_mag;
            r_diff[2] /= r_mag;
                
            idx2 = idx1 + j;
            sigma = sigma_lj[idx2];
            epsilon = epsilon_lj[idx2];
            al = alpha[idx2];
            be = beta[idx2];
            if (!isfinite(sigma) || !isfinite(epsilon) || !isfinite(al) || !isfinite(be)) {
                mpi_fprintf(
                    stderr,
                    "Error: LJ params non-finite (i=%d j=%d sigma=%e epsilon=%e alpha=%e beta=%e)\n",
                    i, j, sigma, epsilon, al, be
                );
                exit(1);
            }

            //write f_mag and V_mag for lennard-jones potential
            f_mag = 4 * epsilon * (12 * pow(sigma / r_mag, 12) - 6 * pow(sigma / r_mag, 6)) / r_mag - al;
            V_mag = 4 * epsilon * (pow(sigma / r_mag, 12) - pow(sigma / r_mag, 6)) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    potential_energy /= 2.0;
    if (!lj_force_shift) {
        potential_energy += compute_lj_tail_correction(n_p, L, params, r_cut);
    }
    return potential_energy;
}


/*
Compute the particle-particle forces using the SC repulsive potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [nu, d, B] (3)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_sc_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
    int i, j, k, ip, jp;
    double nu, d, B_nu, alpha, beta;
    double potential_energy = 0.0;

    int size = n_p * 3;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double d_over_r_pow;
    double f_k;

    nu    = params[0];
    d     = params[1];
    B_nu  = params[2];
    alpha = params[3];
    beta  = params[4];

    memset(forces, 0, size * sizeof(double));

    #pragma \
        omp parallel private(i, j, k, ip, jp, r_diff, r_mag, f_mag, f_k, V_mag, d_over_r_pow) \
        reduction(+:potential_energy, forces[:size])
    for (i = 0; i < n_p; i++) {
        ip = 3 * i;
        for (j = i + 1; j < n_p; j++) {
            jp = 3 * j;

            r_mag = 0.0;
            for (k = 0; k < 3; k++) {
                app = pos[ip + k] - pos[jp + k];
                app -= L * round(app / L);
                r_mag += app * app;
                r_diff[k] = app;
            }

            r_mag = sqrt(r_mag);
            if (!isfinite(r_mag) || r_mag <= 1e-12) {
                // TODO: The first one is redundnat with the next check
                //  The second one should never happen?
                continue;
            }
            if (r_mag > r_cut) {
                continue;
            }

            d_over_r_pow = pow(d / r_mag, nu);
            V_mag = B_nu * d_over_r_pow + alpha * r_mag + beta;
            f_mag = B_nu * nu * d_over_r_pow / r_mag - alpha;

            for (k = 0; k < 3; k++) {
                f_k = f_mag * r_diff[k] / r_mag;
                forces[ip + k] += f_k;
                forces[jp + k] -= f_k;
            }

            potential_energy += V_mag;
        }
    }
    return potential_energy;
}

double compute_lj_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces) {
    double r2 = vx * vx + vy * vy + vz * vz;
    double r = sqrt(r2);
    if (r > r_cut || r < 1e-15) {
        return 0.0;
    }
    long int ia3 = ia * 3;
    long int ib3 = ib * 3;
    long int idx = ia * np + ib;
    double sigma = params[idx];
    double epsilon = params[idx + np * np];
    double alpha = params[idx + 2 * np * np];
    double beta = params[idx + 3 * np * np];

    double inv_r = 1.0 / r;
    double sr = sigma * inv_r;
    double sr2 = sr * sr;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;

    double f_mag = 4 * epsilon * (12 * sr12 - 6 * sr6) * inv_r - alpha;
    double V_mag = 4 * epsilon * (sr12 - sr6) + alpha * r + beta;

    double fx = -f_mag * vx * inv_r;
    double fy = -f_mag * vy * inv_r;
    double fz = -f_mag * vz * inv_r;

    forces[ia3    ] += fx;
    forces[ia3 + 1] += fy;
    forces[ia3 + 2] += fz;

    forces[ib3    ] -= fx;
    forces[ib3 + 1] -= fy;
    forces[ib3 + 2] -= fz;

    return V_mag;
}

double compute_tf_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces) {
    double r2 = vx * vx + vy * vy + vz * vz;
    double r = sqrt(r2);
    if (r > r_cut || r < 1e-15) {
        return 0.0;
    }
    long int ia3 = ia * 3;
    long int ib3 = ib * 3;
    long int idx = ia * np + ib;
    double *A = params;
    double *B = A + np * np;
    double *C = B + np * np;
    double *D = C + np * np;
    double *sigma = D + np * np;
    double *alpha = sigma + np * np;
    double *beta = alpha + np * np;

    double a = A[idx];
    double b = B[idx];
    double c = C[idx];
    double d = D[idx];
    double sig = sigma[idx];
    double al = alpha[idx];
    double be = beta[idx];

    double exp_term = exp(b * (sig - r));
    double r6 = r2 * r2 * r2;
    double r7 = r6 * r;
    double r8 = r7 * r;
    double r9 = r8 * r;

    double f_mag = b * a * exp_term - 6.0 * c / r7 - 8.0 * d / r9 - al;
    double V_mag = a * exp_term - c / r6 - d / r8 + al * r + be;

    double inv_r = 1.0 / r;
    double fx = -f_mag * vx * inv_r;
    double fy = -f_mag * vy * inv_r;
    double fz = -f_mag * vz * inv_r;

    forces[ia3    ] += fx;
    forces[ia3 + 1] += fy;
    forces[ia3 + 2] += fz;

    forces[ib3    ] -= fx;
    forces[ib3 + 1] -= fy;
    forces[ib3 + 2] -= fz;

    return V_mag;
}

double compute_sc_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces) {
    double r2 = vx * vx + vy * vy + vz * vz;
    double r = sqrt(r2);
    if (r > r_cut || r < 1e-15) {
        return 0.0;
    }
    long int ia3 = ia * 3;
    long int ib3 = ib * 3;

    double nu    = params[0];
    double d     = params[1];
    double B_nu  = params[2];
    double alpha = params[3];
    double beta  = params[4];

    double d_over_r_pow = pow(d / r, nu);
    double V_mag = B_nu * d_over_r_pow + alpha * r + beta;
    double f_mag = B_nu * nu * d_over_r_pow / r - alpha;

    double inv_r = 1.0 / r;
    double fx = -f_mag * vx * inv_r;
    double fy = -f_mag * vy * inv_r;
    double fz = -f_mag * vz * inv_r;

    forces[ia3    ] += fx;
    forces[ia3 + 1] += fy;
    forces[ia3 + 2] += fz;

    forces[ib3    ] -= fx;
    forces[ib3 + 1] -= fy;
    forces[ib3 + 2] -= fz;

    return V_mag;
}
