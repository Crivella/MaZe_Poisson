// Implementation of the SSOR (Symmetric Successive Over Relaxation) preconditioner
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"

void solve_diag(double *b, int n_loc, int n, int n_start) {
    dscal(b, -6.0, n_loc * n * n);
}

void solve_uppder_branched(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;
    long int i, j, k, k1, k2;

    long int n2 = n * n;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    long int idx0, idx1;
    double app;



    // #pragma omp parallel for private(i,j,k, i0, j0, i1, j1, k1, i2, j2, k2, app, idx0, idx1)
    // Cant really OpenMP this easily as one result depends on the previous ones
    for (i = nm1; i >= 0; i--) {
        i0 = i * n2;
        i1 = ((i + 1) % n) * n2;
        i2 = ((i - 1 + n) % n) * n2;
        for (j = nm1; j >= 0; j--) {
            j0 = j * n;
            j1 = ((j + 1) % n) * n;
            j2 = ((j - 1 + n) % n) * n;
            for (k = nm1; k >= 0; k--) {
                k1 = ((k + 1) % n);
                k2 = ((k - 1 + n) % n);

                idx0 = i0 + j0 + k;
                app = b[idx0];

                idx1 = i1 + j0 + k;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i2 + j0 + k;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j1 + k;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j2 + k;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j0 + k1;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j0 + k2;
                if ( idx1 > idx0 ) {
                    app -= b[idx1];
                }
                app /= -6.0;
                b[idx0] = app;
            }
        }
    }
}

void solve_lower_branched(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;
    long int i, j, k, k1, k2;

    long int n2 = n * n;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    long int idx0, idx1;
    double app;

    // #pragma omp parallel for private(i,j,k, i0, j0, i1, j1, k1, i2, j2, k2, app, idx0, idx1)
    for (i = 0; i < n; i++) {
        i0 = i * n2;
        i1 = ((i + 1) % n) * n2;
        i2 = ((i - 1 + n) % n) * n2;
        for (j = 0; j < n; j++) {
            j0 = j * n;
            j1 = ((j + 1) % n) * n;
            j2 = ((j - 1 + n) % n) * n;
            for (k = 0; k < n; k++) {
                k1 = ((k + 1) % n);
                k2 = ((k - 1 + n) % n);

                idx0 = i0 + j0 + k;
                app = b[idx0];

                idx1 = i1 + j0 + k;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i2 + j0 + k;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j1 + k;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j2 + k;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j0 + k1;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                idx1 = i0 + j0 + k2;
                if ( idx1 < idx0 ) {
                    app -= b[idx1];
                }
                app /= -6.0;
                b[idx0] = app;
            }
        }
    }
}

void solve_upper(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;
    int nm2 = n - 2;
    int i, j, k, k1, k2;

    long int n2 = n * n;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    // mpi_grid_exchange_bot_top(b, n_loc, n);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Edge case for i = n - 1
    i0 = nm1 * n2;
        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
            /////////////////////////////////////////////
            // Edge case for k =  n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        // #pragma omp parallel for private(j, k, j0, j1, k1)
        for (j = nm2; j > 0; j--) {
            j0 = j * n;
            j1 = j0 + n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
        j1 = n;
        j2 = nm1 * n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////

    // #pragma omp parallel for private(i,j,k, i0, j0, i1, j1, k1, i2, j2, k2)
    for (i = nm2; i > 0; i--) {
        i0 = i * n2;
        i1 = i0 + n2;

        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        for (j = nm2; j > 0; j--) {
            j0 = j * n;
            j1 = j0 + n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i0 + j1 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
        j1 = n;
        j2 = nm1 * n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Edge case for i = 0
        i0 = 0;
        i1 = n2;
        i2 = nm1 * n2;

        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        // #pragma omp parallel for private(j, k, j0, j1, k1)
        for (j = nm2; j > 0; j--) {
            j0 = j * n;
            j1 = j0 + n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
        j1 = n;
        j2 = nm1 * n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k1]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
}

void solve_lower(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;
    long int i, j, k, k1, k2;

    long int n2 = n * n;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    // mpi_grid_exchange_bot_top(b, n_loc, n);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Edge case for i = 0
    i0 = 0;
        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////


        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                // k1 = k + 1;
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
        j1 = 0;
        j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////

    // #pragma omp parallel for private(i,j,k, i0, j0, i1, j1, k1, i2, j2, k2)
    for (i = 1; i < nm1; i++) {
        i0 = i * n2;
        i2 = i0 - n2;

        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
        j1 = 0;
        j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Edge case for i = n - 1
        i0 = nm1 * n2;
        i1 = 0;
        i2 = i0 - n2;

        /////////////////////////////////////////////
        // Edge case for j = 0
        j0 = 0;
            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////


        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                // k1 = k + 1;
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        }

        /////////////////////////////////////////////
        // Edge case for j = n - 1
        j0 = nm1 * n;
        j1 = 0;
        j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k]
            ) / -6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] -
                    b[i1 + j0 + k] -
                    b[i2 + j0 + k] -
                    b[i0 + j1 + k] -
                    b[i0 + j2 + k] -
                    b[i0 + j0 + k2]
                ) / -6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] -
                b[i1 + j0 + k] -
                b[i2 + j0 + k] -
                b[i0 + j1 + k] -
                b[i0 + j2 + k] -
                b[i0 + j0 + k1] -
                b[i0 + j0 + k2]
            ) / -6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
}

/*
Apply the SSOR preconditioner by decomposing the matrix of the problem A.x = b into
A = D + L + L^T  where D is the diagonal, L is the lower triangular part and L^T is the upper triangular part
The preconditioner in this case is:

P = (D + L) . D^-1 . (D + L)^T
P . v = r
v = P^-1 . r

M1 = (D + L)     LOWER
M2 = D^-1        DIAG
M3 = (D + L)^T   UPPER

M1 . M2 . M3 . v = r

M3 . v = y
M2 . y = z

z = solve_M1 (b) = TRIANG_SOLVE_M1 (b)
y = solve_M2 (z) = D . Z
v = solve_M3 (y) = TRIANG_SOLVE_M3 (y)

The function is built to work both with separate input/output arrays or in-place
@param in: the input array
@param out: the output array (can be the same as in)
@param s1: the size of the first dimension
@param s2: the size of the second/third dimension
@param n_start1: the starting index of the first dimension (used for MPI)
*/
void precond_ssor_apply(double *in, double *out, int s1, int s2, int n_start) {
    if ( in != out ) {
        memcpy(out, in, s1 * s2 * s2 * sizeof(double));  // Copy the input to output
    }
    solve_lower(out, s1, s2, n_start);  // z = M1^-1 . b
    solve_diag(out, s1, s2, n_start);  // y = M2^-1 . z
    solve_upper(out, s1, s2, n_start);  // v = M3^-1 . y
}

