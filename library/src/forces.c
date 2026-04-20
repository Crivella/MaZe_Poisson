#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sphere_intersect.h"
#include "mpi_base.h"
#include "linalg.h"

static int pbc_grid_index(int idx, int n) {
    // Same periodic index wrap used elsewhere as (idx + n) % n, generalized for larger offsets.
    idx %= n;
    if (idx < 0) idx += n;
    return idx;
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

    return sum_q;
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
double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
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

            //write f_mag and V_mag for lennard-jones potential
            f_mag = 4 * epsilon * (12 * pow(sigma / r_mag, 12) - 6 * pow(sigma / r_mag, 6)) / r_mag - al;
            V_mag = 4 * epsilon * (pow(sigma / r_mag, 12) - pow(sigma / r_mag, 6)) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    return potential_energy / 2;
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


/*
Compute the stress tensor forces on particles

@param g: the grid structure containing the grid parameters
@param p: the particles structure containing the particle parameters
@param phi: the potential field of size n_grid * n_grid * n_grid
@param out_forces: the output forces on each particle of size n_p * 3
*/
void compute_stress_tensor_forces(
    // grid *g,
    // particles *p,
    // const double *phi,
    // double *out_forces
    int n, double eps_s, int n_p, double L, double h, const double *phi, double *pos, double *solv_radii, double *out_forces, int use_pbc
)
{
    // const int n = g->n;
    // const double h = g->h;
    // const double eps_s = g->eps_s;
    // const double eps_m = 1.0;
    const double stress_prefactor = 1.0 / (4.0 * M_PI);
    // const unsigned int *region = g->region;
    // const int use_pbc = (g->bc_type == BC_TYPE_PBC);
    
    double h2 = h * h;
    for (int i = 0; i < n_p * 3; i++) out_forces[i] = 0.0;

    int ip, jp, kp, num_points_min = 0;

    double Ex, Ey, Ez, fl_dir;
    for (int p_idx = 0; p_idx < n_p; p_idx++) {
        // ip = floor(p->pos[p_idx * 3 + 0] / h);
        // jp = floor(p->pos[p_idx * 3 + 1] / h);
        // kp = floor(p->pos[p_idx * 3 + 2] / h);

        ip = round(pos[p_idx * 3 + 0] / h);
        jp = round(pos[p_idx * 3 + 1] / h);
        kp = round(pos[p_idx * 3 + 2] / h);

        num_points_min = ceil(solv_radii[p_idx] / h) + 1;
        // num_points_min = ceil(solv_radii[p_idx] / h) + 2; // Extra margin, to be tested.
        // build the list of points to consider for force calculation, must be a cube of side 2*num_points_min centered on the particle
        long idx_a, idx_b;
        // Negative x face: i = ip - num_points_min, j in [jp - num_points_min, jp + num_points_min], k in [kp - num_points_min, kp + num_points_min]
        for (int i = ip - num_points_min; i <= ip + num_points_min; i += 2 * num_points_min) {
            fl_dir = (i < ip) ? -1.0 : 1.0;
            for (int j = jp - num_points_min; j <= jp + num_points_min; j++) {
                for (int k = kp - num_points_min; k <= kp + num_points_min; k++) {
                    int ib_raw = i + (fl_dir > 0.0 ? 1 : -1);
                    if (!use_pbc && (i < 0 || i >= n || ib_raw < 0 || ib_raw >= n || j <= 0 || j >= n - 1 || k <= 0 || k >= n - 1)) {
                        continue;
                    }
                    int ia = pbc_grid_index(i, n);
                    int ja = pbc_grid_index(j, n);
                    int ka = pbc_grid_index(k, n);
                    int ib = pbc_grid_index(ia + (fl_dir > 0.0 ? 1 : -1), n);

                    idx_a = ka + ja * n + ia * n * n;
                    idx_b = ka + ja * n + ib * n * n;
                    

                    long idx_yp, idx_ym, idx_zp, idx_zm;
                    long idx_b_yp, idx_b_ym, idx_b_zp, idx_b_zm;

                    int jp = pbc_grid_index(ja + 1, n);
                    int jm = pbc_grid_index(ja - 1, n);
                    int kp = pbc_grid_index(ka + 1, n);
                    int km = pbc_grid_index(ka - 1, n);

                    idx_yp = ka + jp * n + ia * n * n;
                    idx_ym = ka + jm * n + ia * n * n;
                    idx_zp = kp + ja * n + ia * n * n;
                    idx_zm = km + ja * n + ia * n * n;

                    idx_b_yp = ka + jp * n + ib * n * n;
                    idx_b_ym = ka + jm * n + ib * n * n;
                    idx_b_zp = kp + ja * n + ib * n * n;
                    idx_b_zm = km + ja * n + ib * n * n;
                    
                    Ex = - fl_dir * (phi[idx_b] - phi[idx_a]) / h;
                    Ey = -((phi[idx_yp] - phi[idx_ym]) + // (i,j+1,k) - (i,j-1,k), y direction at (i,j,k)
                        (phi[idx_b_yp] - phi[idx_b_ym]) // (i+1,j+1,k) - (i+1,j-1,k), y direction at (i+1,j,k)
                    ) / (4.0 * h);
                    Ez = -(
                        (phi[idx_zp] - phi[idx_zm]) + // (i,j,k) - (i,j,k-1)
                        (phi[idx_b_zp] - phi[idx_b_zm]) // (i+1,j,k) - (i+1,j,k-1)
                    ) / (4.0 * h);
                    
                    out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * fl_dir * eps_s * (Ex * Ex - 0.5 * (Ex * Ex + Ey * Ey + Ez * Ez));
                    out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * fl_dir * eps_s * Ex * Ey;
                    out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * fl_dir * eps_s * Ex * Ez;
                }
            }
        }

        // Negative y face: j = jp - num_points_min, i in [ip - num_points_min, ip + num_points_min], k in [kp - num_points_min, kp + num_points_min]
        for (int j = jp - num_points_min; j <= jp + num_points_min; j += 2 * num_points_min) {
            fl_dir = (j < jp) ? -1.0 : 1.0;
            for (int i = ip - num_points_min; i <= ip + num_points_min; i++) {
                for (int k = kp - num_points_min; k <= kp + num_points_min; k++) {
                    int jb_raw = j + (fl_dir > 0.0 ? 1 : -1);
                    if (!use_pbc && (j < 0 || j >= n || jb_raw < 0 || jb_raw >= n || i <= 0 || i >= n - 1 || k <= 0 || k >= n - 1)) {
                        continue;
                    }
                    int ia = pbc_grid_index(i, n);
                    int ja = pbc_grid_index(j, n);
                    int ka = pbc_grid_index(k, n);
                    int jb = pbc_grid_index(ja + (fl_dir > 0.0 ? 1 : -1), n);

                    idx_a = ka + ja * n + ia * n * n;
                    idx_b = ka + jb * n + ia * n * n;

                    long idx_xp, idx_xm, idx_zp, idx_zm;
                    long idx_b_xp, idx_b_xm, idx_b_zp, idx_b_zm;
                    int ip1 = pbc_grid_index(ia + 1, n);
                    int im1 = pbc_grid_index(ia - 1, n);
                    int kp = pbc_grid_index(ka + 1, n);
                    int km = pbc_grid_index(ka - 1, n);
                    idx_xp = ka + ja * n + ip1 * n * n;
                    idx_xm = ka + ja * n + im1 * n * n;
                    idx_zp = kp + ja * n + ia * n * n;
                    idx_zm = km + ja * n + ia * n * n;
                    idx_b_xp = ka + jb * n + ip1 * n * n;
                    idx_b_xm = ka + jb * n + im1 * n * n;
                    idx_b_zp = kp + jb * n + ia * n * n;
                    idx_b_zm = km + jb * n + ia * n * n;
                    Ex = -((phi[idx_xp] - phi[idx_xm]) + // (i,j,k) - (i-1,j,k)
                        (phi[idx_b_xp]- phi[idx_b_xm]) // (i+1,j,k) - (i,j,k)
                    ) / (4.0 * h);
                    Ey = - fl_dir * (phi[idx_b] - phi[idx_a]) / h;
                    Ez = -(
                        (phi[idx_zp] - phi[idx_zm]) + // (i,j,k) - (i,j,k-1)
                        (phi[idx_b_zp] - phi[idx_b_zm]) // (i+1,j,k) - (i+1,j,k-1)
                    ) / (4.0 * h); 
                      
                    out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * fl_dir * eps_s * Ey * Ex;
                    out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * fl_dir * eps_s * (Ey * Ey - 0.5 * (Ex * Ex + Ey * Ey + Ez * Ez));
                    out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * fl_dir * eps_s * Ey * Ez;
                }
            }
        }

        // z faces
        for (int k = kp - num_points_min; k <= kp + num_points_min; k += 2 * num_points_min) {
            fl_dir = (k < kp) ? -1.0 : 1.0;
            for (int i = ip - num_points_min; i <= ip + num_points_min; i++) {
                for (int j = jp - num_points_min; j <= jp + num_points_min; j++) {
                    int kb_raw = k + (fl_dir > 0.0 ? 1 : -1);
                    if (!use_pbc && (k < 0 || k >= n || kb_raw < 0 || kb_raw >= n || i <= 0 || i >= n - 1 || j <= 0 || j >= n - 1)) {
                        continue;
                    }
                    int ia = pbc_grid_index(i, n);
                    int ja = pbc_grid_index(j, n);
                    int ka = pbc_grid_index(k, n);
                    int kb = pbc_grid_index(ka + (fl_dir > 0.0 ? 1 : -1), n);

                    idx_a = ka + ja * n + ia * n * n;
                    idx_b = kb + ja * n + ia * n * n;

                    long idx_xp, idx_xm, idx_yp, idx_ym;
                    long idx_b_xp, idx_b_xm, idx_b_yp, idx_b_ym;
                    int ip1 = pbc_grid_index(ia + 1, n);
                    int im1 = pbc_grid_index(ia - 1, n);
                    int jp = pbc_grid_index(ja + 1, n);
                    int jm = pbc_grid_index(ja - 1, n);
                    idx_xp = ka + ja * n + ip1 * n * n;
                    idx_xm = ka + ja * n + im1 * n * n;
                    idx_yp = ka + jp * n + ia * n * n;
                    idx_ym = ka + jm * n + ia * n * n;
                    idx_b_xp = kb + ja * n + ip1 * n * n;
                    idx_b_xm = kb + ja * n + im1 * n * n;
                    idx_b_yp = kb + jp * n + ia * n * n;
                    idx_b_ym = kb + jm * n + ia * n * n;

                    Ex = -((phi[idx_xp] - phi[idx_xm]) + // (i,j,k) - (i-1,j,k)
                        (phi[idx_b_xp]- phi[idx_b_xm]) // (i+1,j,k) - (i,j,k)
                    ) / (4.0 * h);
                    Ey = -((phi[idx_yp] - phi[idx_ym]) + // (i,j+1,k) - (i,j-1,k), y direction at (i,j,k)
                        (phi[idx_b_yp] - phi[idx_b_ym]) // (i+1,j+1,k) - (i+1,j-1,k), y direction at (i+1,j,k)
                    ) / (4.0 * h);
                    Ez = - fl_dir * (phi[idx_b] - phi[idx_a]) / h;

                    out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * fl_dir * eps_s * Ez * Ex;
                    out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * fl_dir * eps_s * Ez * Ey;
                    out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * fl_dir * eps_s * (Ez * Ez - 0.5 * (Ex * Ex + Ey * Ey + Ez * Ez));
                }
            }
        }
    }
}
