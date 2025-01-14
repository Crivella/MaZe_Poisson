#include <stdlib.h>
#include <stdio.h>


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
double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces) {
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

double compute_force_fd_mpi(
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