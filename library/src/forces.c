#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi_base.h"
#include "linalg.h"

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
    double *phi, long int *neighbors, long int *charges, double *pos, double *forces,
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

    return sum_q;
}

/*
Compute the particle-particle forces using the tabulated Tosi-Fumi potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param B: the parameter B of the potential
@param params: the parameters of the potential [A, C, D, sigma, alpha, beta] (6, n_p, n_p)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces) {
    int ip, jp;
    int n_p2 = 2 * n_p;
    long int n_p_pow2 = n_p * n_p;
    long int idx1, idx2;

    double *A = params;
    double *C = A + n_p_pow2;
    double *D = C + n_p_pow2;
    double *sigma_TF = D + n_p_pow2;
    double *alpha = sigma_TF + n_p_pow2;
    double *beta = alpha + n_p_pow2;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double potential_energy = 0.0;
    double a, c, d, sigma, al, be;

    #pragma omp parallel for private(app, ip, jp, r_diff, r_mag, f_mag, V_mag, a, c, d, sigma, al, be, idx1, idx2) reduction(+:potential_energy)
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
            if (r_mag > r_cut) {
                continue;
            }
            r_diff[0] /= r_mag;
            r_diff[1] /= r_mag;
            r_diff[2] /= r_mag;
                
            idx2 = idx1 + j;
            a = A[idx2];
            c = C[idx2];
            d = D[idx2];
            sigma = sigma_TF[idx2];
            al = alpha[idx2];
            be = beta[idx2];

            f_mag = B * a * exp(B * (sigma - r_mag)) - 6 * c / pow(r_mag, 7) - 8 * d / pow(r_mag, 9) - al;
            V_mag = a * exp(B * (sigma - r_mag)) - c / pow(r_mag, 6) - d / pow(r_mag, 8) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    return potential_energy / 2;
}

/*
Compute the reaction force using the finite difference method.

@param n_grid: the number of grid points in each dimension
@param n_p: the number of particles
@param h: the grid spacing
@param num_neigh: the number of neighbors for each particle
@param phi_v: the potential field of size n_grid * n_grid * n_grid (elect pot in vacuum)
@param phi_s: the potential field of size n_grid * n_grid * n_grid (elect pot in solvent)
*/
double compute_forces_reaction_field(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi_v, double *phi_s, long int *neighbors, long int *charges, double *pos,
    double *forces_rf, double (*g)(double, double, double)
) {
    int n_loc = get_n_loc();

    double *delta_phi = mpi_grid_allocate(n_loc, n_grid);

    long int n3 = n_loc * n_grid * n_grid;

    vec_copy(phi_s, delta_phi, n3);
    daxpy(phi_v, delta_phi, -1.0, n3);  // delta_phi = phi_s - phi_v

    compute_force_fd(
        n_grid, n_p, h, num_neigh,
        delta_phi, neighbors, charges, pos, forces_rf, g
    );

    mpi_grid_free(delta_phi, n_grid);
}
