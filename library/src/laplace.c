#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "constants.h"
#include "mpi_base.h"
#include "linalg.h"
#include "laplace.h"

static electrostatic_discretization_type g_discretization =
    ELECTROSTATIC_DISCRETIZATION_STANDARD;

static int is_mehrstellen4(void) {
    return g_discretization == ELECTROSTATIC_DISCRETIZATION_MEHRSTELLEN4;
}

void laplace_set_discretization(electrostatic_discretization_type discretization) {
    g_discretization = discretization;
}

electrostatic_discretization_type laplace_get_discretization(void) {
    return g_discretization;
}

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
static void laplace_filter_impl(double *u, double *u_new, int size1, int size2, int mehrstellen4) {
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
                double filtered;
                if (mehrstellen4) {
                    const double faces =
                        u[i1 + j0 + k] + u[i2 + j0 + k] +
                        u[i0 + j1 + k] + u[i0 + j2 + k] +
                        u[i0 + j0 + k1] + u[i0 + j0 + k2];
                    const double edges =
                        u[i1 + j1 + k] + u[i1 + j2 + k] +
                        u[i2 + j1 + k] + u[i2 + j2 + k] +
                        u[i1 + j0 + k1] + u[i1 + j0 + k2] +
                        u[i2 + j0 + k1] + u[i2 + j0 + k2] +
                        u[i0 + j1 + k1] + u[i0 + j1 + k2] +
                        u[i0 + j2 + k1] + u[i0 + j2 + k2];
                    filtered =
                        -4.0 * u[i0 + j0 + k] + faces / 3.0 + edges / 6.0;
                } else {
                    filtered =
                        u[i1 + j0 + k] + u[i2 + j0 + k] +
                        u[i0 + j1 + k] + u[i0 + j2 + k] +
                        u[i0 + j0 + k1] + u[i0 + j0 + k2] -
                        6.0 * u[i0 + j0 + k];
                }
                u_new[i0 + j0 + k] = filtered;
            }
        }
    }
}

void laplace_filter(double *u, double *u_new, int size1, int size2) {
    laplace_filter_impl(u, u_new, size1, size2, is_mehrstellen4());
}

void laplace_filter_standard(double *u, double *u_new, int size1, int size2) {
    laplace_filter_impl(u, u_new, size1, size2, 0);
}

void laplace_filter_mehrstellen4(double *u, double *u_new, int size1, int size2) {
    laplace_filter_impl(u, u_new, size1, size2, 1);
}

void laplace_jacobi_step(
    double *rhs, double *u, double *u_new,
    int size1, int size2, double omega
) {
    const long int n3 = (long int)size1 * size2 * size2;
    laplace_filter_standard(u, u_new, size1, size2);
    #pragma omp parallel for
    for (long int idx = 0; idx < n3; ++idx) {
        u_new[idx] = u[idx] + omega * (rhs[idx] - u_new[idx]);
    }
}

/* Mehrstellen right-hand side B_h f = (I + Delta_h/12) f. */
void laplace_filter_rhs(double *u, double *u_new, int size1, int size2) {
    const long int n2 = (long int)size2 * size2;
    const long int n3 = (long int)size1 * n2;
    if (!is_mehrstellen4()) {
        if (u != u_new) memcpy(u_new, u, n3 * sizeof(double));
        return;
    }
    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter_rhs: in-place operation not supported\n");
        exit(1);
    }

    mpi_grid_exchange_bot_top(u, size1, size2);
    #pragma omp parallel for
    for (int i = 0; i < size1; ++i) {
        const long int i0 = (long int)i * n2;
        const long int im = i0 - n2;
        const long int ip = i0 + n2;
        for (int j = 0; j < size2; ++j) {
            const long int j0 = (long int)j * size2;
            const long int jm = (long int)((j - 1 + size2) % size2) * size2;
            const long int jp = (long int)((j + 1) % size2) * size2;
            for (int k = 0; k < size2; ++k) {
                const int km = (k - 1 + size2) % size2;
                const int kp = (k + 1) % size2;
                const long int idx = i0 + j0 + k;
                const double faces =
                    u[im + j0 + k] + u[ip + j0 + k] +
                    u[i0 + jm + k] + u[i0 + jp + k] +
                    u[i0 + j0 + km] + u[i0 + j0 + kp];
                u_new[idx] = 0.5 * u[idx] + faces / 12.0;
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
