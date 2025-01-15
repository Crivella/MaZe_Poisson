#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/*
Compute the forces on each particle by computing the field from the potential using finite differences
@param n_grid: the number of grid points in each dimension
@param n_p: the number of particles
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param q: the charge on a grid of size n_grid * n_grid * n_grid
@param neighbors: Array (x,y,z) of neighbors indexes for each particle (n_p x 8 x 3)
@param forces: the output forces on each particle of size n_p * 3

@return the sum of the charges on the neighbors
*/
double compute_force_fd_old(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces) {
    long int n = n_grid;
    long int n2 = n * n;
    long int n3 = n * n * n;

    int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;

    h *= 2.0;

    double *E = (double *)malloc(n3 * sizeof(double));

    double sum_q = 0.0;

    /////////////////////////////////////////////////////////////////////////////////
    // X component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 0; i < n; i++) {
        i0 = i * n2;
        i1 = ((i+1) % n) * n2;
        i2 = ((i-1 + n) % n) * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            j1 = i1 + j*n;
            j2 = i2 + j*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j1 + k;
                k2 = j2 + k;
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 0] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0];
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            // printf("i: %d, j: %d, k: %d,  E: %f,  q: %f\n", i, j, k, E[i2], q[i2]);
            forces[j0 + 0] += q[i2] * E[i2];
            sum_q += q[i2];
        }
    }


    /////////////////////////////////////////////////////////////////////////////////
    // Y component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 0; i < n; i++) {
        i0 = i * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            j1 = i0 + ((j+1) % n)*n;
            j2 = i0 + ((j-1 + n) % n)*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j1 + k;
                k2 = j2 + k;
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 1] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0];
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            forces[j0 + 1] += q[i2] * E[i2];
        }
    }


    /////////////////////////////////////////////////////////////////////////////////
    // Z component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 0; i < n; i++) {
        i0 = i * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j0 + ((k+1) % n);
                k2 = j0 + ((k-1 + n) % n);
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 2] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0];
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            forces[j0 + 2] += q[i2] * E[i2];
        }
    }

    free(E);

    return sum_q;
}


/*
Compute the forces on each particle by computing the field from the potential using finite differences.
New version computes the field only where the particles are located.

@param n_grid: the number of grid points in each dimension
@param n_p: the number of particles
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param q: the charge on a grid of size n_grid * n_grid * n_grid
@param neighbors: Array (x,y,z) of neighbors indexes for each particle (n_p x 8 x 3)
@param forces: the output forces on each particle of size n_p * 3

@return the sum of the charges on the neighbors
*/
double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces) {
    long int n = n_grid;
    long int n2 = n * n;

    int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;

    h *= 2.0;

    double E, qc;

    double sum_q = 0.0;
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2, E, qc) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0] = 0.0;
        forces[j0+1] = 0.0;
        forces[j0+2] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1];
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            qc = q[i * n2 + j * n + k];
            sum_q += qc;
            // X
            i1 = ((i+1) % n) * n2;
            i2 = ((i-1 + n) % n) * n2;
            E = (phi[i2 + j*n + k] - phi[i1 + j*n + k]) / h;
            forces[j0] += qc * E;
            // Y
            i1 = i * n2;
            j1 = ((j+1) % n) * n;
            j2 = ((j-1 + n) % n) * n;
            E = (phi[i1 + j2 + k] - phi[i1 + j1 + k]) / h;
            forces[j0 + 1] += qc * E;
            // Z
            j1 = j * n;
            k1 = ((k+1) % n);
            k2 = ((k-1 + n) % n);
            E = (phi[i1 + j1 + k2] - phi[i1 + j1 + k1]) / h;
            forces[j0 + 2] += qc * E;
        }
    }

    return sum_q;
}

double compute_force_fd_mpi_old(
        int n_grid, int n_loc, int n_start, int n_p, double h,
        double *phi, double *bot, double *top, double *q, long int *neighbors,
        double *forces
    ) {
    long int n = n_grid;
    long int n2 = n * n;
    long int n3 = n_loc * n2;

    int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;

    h *= 2.0;

    double *E = (double *)malloc(n3 * sizeof(double));

    double sum_q = 0.0;

    /////////////////////////////////////////////////////////////////////////////////
    // X component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 1; i < n_loc - 1; i++) {
        i0 = i * n2;
        i1 = (i+1) * n2;
        i2 = (i-1) * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            j1 = i1 + j*n;
            j2 = i2 + j*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j1 + k;
                k2 = j2 + k;
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // i0 = 0;  // Ignored because 0
    i1 = n2;  // 1 * n2
    // i2 = n_loc - 1; // Ignored in favor of bot
    #pragma omp parallel for private(j, k, j0, j1, j2)
    for (j = 0; j < n; j++) {
        j0 = j * n;
        j1 = i1 + j * n;
        for (k = 0; k < n; k++) {
            k0 = j0 + k;
            k1 = i1 + j0 + k;
            // k2 = j1 + k;
            E[k0] = (bot[k0] - phi[k1]) / h;  // - from swapped indices
        }
    }

    i0 = (n_loc - 1) * n2;  // Ignored in favor of top
    i1 = 0;  // Ignored in favor of top
    i2 = (n_loc - 2) * n2;
    #pragma omp parallel for private(j, k, j0, j1, j2)
    for (j = 0; j < n; j++) {
        j0 = i0 + j * n;
        j2 = i2 + j * n;
        for (k = 0; k < n; k++) {
            k0 = j0 + k;
            k2 = j2 + k;
            E[k0] = (phi[k2] - top[k0 - i0]) / h;  // - from swapped indices
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 0] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            // printf("i: %d, j: %d, k: %d,  E: %f,  q: %f\n", i, j, k, E[i2], q[i2]);
            forces[j0 + 0] += q[i2] * E[i2];
            sum_q += q[i2];
        }
    }


    /////////////////////////////////////////////////////////////////////////////////
    // Y component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 0; i < n_loc; i++) {
        i0 = i * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            j1 = i0 + ((j+1) % n)*n;
            j2 = i0 + ((j-1 + n) % n)*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j1 + k;
                k2 = j2 + k;
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 1] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            forces[j0 + 1] += q[i2] * E[i2];
        }
    }


    /////////////////////////////////////////////////////////////////////////////////
    // Z component
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k0, k1, k2)
    for (i = 0; i < n_loc; i++) {
        i0 = i * n2;
        for (j = 0; j < n; j++) {
            j0 = i0 + j*n;
            for (k = 0; k < n; k++) {
                k0 = j0 + k;
                k1 = j0 + ((k+1) % n);
                k2 = j0 + ((k-1 + n) % n);
                E[k0] = (phi[k2] - phi[k1]) / h;  // - from swapped indices
            }
        }
    }
    // Compute the forces
    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0 + 2] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1 + 0] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            i2 = i * n2 + j * n + k;
            forces[j0 + 2] += q[i2] * E[i2];
        }
    }

    free(E);

    return sum_q;
}

/*
Compute the forces on each particle by computing the field from the potential using finite differences.
New version computes the field only where the particles are located.
MPI aware version that keeps into account that every process owns a slab of the grid.

@param n_grid: the number of grid points in each dimension
@param n_loc: the number of slices in the slab owned by the process
@param n_start: the index of the first slice owned by the process
@param n_p: the number of particles
@param h: the grid spacing
@param phi: the potential field of size (n_loc, n_grid, n_grid)
@param bot: the bottom slice of the grid of size (n_grid, n_grid)
@param top: the top slice of the grid of size (n_grid, n_grid)
@param q: the charge on a grid of size (n_loc, n_grid, n_grid)
@param neighbors: Array (x,y,z) of 8 neighbors indexes for each particle (n_p, 8, 3)
@param forces: the output forces on each particle of size (n_p, 3)
*/
double compute_force_fd_mpi(
        int n_grid, int n_loc, int n_start, int n_p, double h,
        double *phi, double *bot, double *top, double *q, long int *neighbors,
        double *forces
    ) {
    long int n = n_grid;
    long int n2 = n * n;

    int i, j, k, jn, in2;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;

    h *= 2.0;

    double E, qc;
    int n_loc1 = n_loc - 1;
    double *ptr1, *ptr2;
    double *ptr_p1 = phi + n2;  // Pointer to the first +1 slice
    double *ptr_m1 = phi + (n_loc - 2) * n2; // Pointer to the last -1 slice

    double sum_q = 0.0;
    #pragma omp parallel for private(i, j, k, i0, i1, i2, in2, j0, j1, j2, jn, k0, k1, k2, E, qc) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip*24;
        j0 = ip*3;
        forces[j0] = 0.0;
        forces[j0+1] = 0.0;
        forces[j0+2] = 0.0;
        for (int in = 0; in < 8; in++) {
            i1 = i0 + in*3;
            i = neighbors[i1] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            in2 = i * n2;
            jn = j * n;

            qc = q[in2 + jn + k];
            sum_q += qc;
            // X
            if (i == 0) {
                ptr1 = ptr_p1;
                ptr2 = bot;
            } else if (i == n_loc1) {
                ptr1 = top;
                ptr2 = ptr_m1;
            } else {
                ptr1 = phi + in2 + n2;
                ptr2 = phi + in2 - n2;
            }
            E = (ptr2[jn + k] - ptr1[jn + k]) / h;
            forces[j0] += qc * E;
            // Y
            i1 = in2;
            j1 = ((j+1) % n) * n;
            j2 = ((j-1 + n) % n) * n;
            E = (phi[i1 + j2 + k] - phi[i1 + j1 + k]) / h;
            forces[j0 + 1] += qc * E;
            // Z
            j1 = i1 + jn;
            k1 = ((k+1) % n);
            k2 = ((k-1 + n) % n);
            E = (phi[j1 + k2] - phi[j1 + k1]) / h;
            forces[j0 + 2] += qc * E;
        }
    }

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

