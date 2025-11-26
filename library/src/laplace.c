#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "mpi_base.h"
#include "linalg.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

/*
Apply a 3-D Laplace filter to a 3-D array with cyclic boundary conditions
The code uses an input array of shape (n+2, n, n) and output array of shape (n, n, n)
The +2 is used to either use memcpy to swap the top and bottom slices (skipping the % in the first loop)
or uses MPI to exchange the top and bottom slices between processes
@param u: the input array
@param u_new: the output array
@param n: the size of the array in each dimension
*/
void laplace_filter(double *u, double *u_new, int size1, int size2) {
    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = size2 * size2;

    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter: u and u_new are the same array (in-place operation not supported)\n");
        exit(1);
    }

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[size2];
    int jnext[size2];
    int kprev[size2];
    int knext[size2];
    for (int t = 0; t < size2; ++t) {
        kprev[t] = ((t - 1 + size2 ) % size2);
        knext[t] = ((t + 1      ) % size2);
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(u, size1, size2);

    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k1, k2)
    for (i = 0; i < size1; i++) {
        i0 = i * n2;
        i1 = i0 + n2;
        i2 = i0 - n2;
        for (j = 0; j < size2; j++) {
            j0 = j * size2;
            j1 = jnext[j];
            j2 = jprev[j];
            for (k = 0; k < size2; k++) {
                k1 = knext[k];
                k2 = kprev[k];
                u_new[i0 + j0 + k] = (
                    u[i1 + j0 + k] +
                    u[i2 + j0 + k] +
                    u[i0 + j1 + k] +
                    u[i0 + j2 + k] +
                    u[i0 + j0 + k1] +
                    u[i0 + j0 + k2] -
                    u[i0 + j0 + k] * 6.0
                    );
            }
        }
    }
}

EXTERN_C void laplace_filter_pb(
    double *u, double *u_new, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = size2 * size2;

    long int idx0, idx_x, idx_y, idx_z;

    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter_pb: u and u_new are the same array (in-place operation not supported)\n");
        exit(1);
    }

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[size2];
    int jnext[size2];
    int kprev[size2];
    int knext[size2];
    for (int t = 0; t < size2; ++t) {
        kprev[t] = ((t - 1 + size2 ) % size2);
        knext[t] = ((t + 1      ) % size2);
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(u, size1, size2);
    mpi_grid_exchange_bot_top(eps_x, size1, size2);
    mpi_grid_exchange_bot_top(eps_y, size1, size2);
    mpi_grid_exchange_bot_top(eps_z, size1, size2);

    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k1, k2, idx0, idx_x, idx_y, idx_z)
    for (i = 0; i < size1; i++) {
        i0 = i * n2;
        i1 = i0 + n2;
        i2 = i0 - n2;
        for (j = 0; j < size2; j++) {
            j0 = j * size2;
            j1 = jnext[j];
            j2 = jprev[j];
            for (k = 0; k < size2; k++) {
                k1 = knext[k];
                k2 = kprev[k];
                idx0 = i0 + j0 + k;
                idx_x = i2 + j0 + k;
                idx_y = i0 + j2 + k;
                idx_z = i0 + j0 + k2;
                u_new[idx0] = (
                    u[i1 + j0 + k]  * eps_x[idx0] +
                    u[idx_x]        * eps_x[idx_x] +
                    u[i0 + j1 + k]  * eps_y[idx0] +
                    u[idx_y]        * eps_y[idx_y] +
                    u[i0 + j0 + k1] * eps_z[idx0] +
                    u[idx_z]        * eps_z[idx_z] - 
                    u[idx0] * ( 
                        eps_x[idx0] + eps_x[idx_x] +
                        eps_y[idx0] + eps_y[idx_y] +
                        eps_z[idx0] + eps_z[idx_z] +
                        k2_screen[idx0]
                    )
                );
            }
        }
    }
}
