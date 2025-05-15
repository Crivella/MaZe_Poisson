// Implementation of the SSOR (Symmetric Successive Over Relaxation) preconditioner
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"

/*
CUTOFF = 5 works but make it extremely slower (44s/it vs 0.5s/it)
CUTOFF = 3 works but make it extremely slower (6s/it vs 0.5s/it)
CUTOFF = 1 
*/
int CUTOFF = 3;
double ELEM_CUTOFF = 1.0e-10;
long int *index_map_lower = NULL;
long int *index_map_upper = NULL;

void solve_diag(double *b, int n_loc, int n, int n_start) {
    dscal(b, -6.0, n_loc * n * n);
}

void solve_upper_branched(double *b, int n_loc, int n, int n_start) {
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
            ) / 6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i0 + j1 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        for (j = nm2; j > 0; j--) {
            j0 = j * n;
            j1 = j0 + n;

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i0 + j1 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            // #pragma omp parallel for private(k, k1)
            for (k = nm2; k > 0; k--) {
                k1 = k + 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k1]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            k1 = 1;
            k2 = nm1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////


        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                // k1 = k + 1;
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i2 + j0 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////

        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////


        for (j = 1; j < nm1; j++) {
            j0 = j * n;
            j2 = j0 - n;

            /////////////////////////////////////////////
            // Edge case for k = 0
            k = 0;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                // k1 = k + 1;
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
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
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k]
            ) / 6.0;
            /////////////////////////////////////////////

            for (k = 1; k < nm1; k++) {
                k2 = k - 1;
                b[i0 + j0 + k] = (
                    b[i0 + j0 + k] +
                    b[i1 + j0 + k] +
                    b[i2 + j0 + k] +
                    b[i0 + j1 + k] +
                    b[i0 + j2 + k] +
                    b[i0 + j0 + k2]
                ) / 6.0;
            }

            /////////////////////////////////////////////
            // Edge case for k = n - 1
            k = nm1;
            k1 = 0;
            k2 = k - 1;
            b[i0 + j0 + k] = (
                b[i0 + j0 + k] +
                b[i1 + j0 + k] +
                b[i2 + j0 + k] +
                b[i0 + j1 + k] +
                b[i0 + j2 + k] +
                b[i0 + j0 + k1] +
                b[i0 + j0 + k2]
            ) / 6.0;
            /////////////////////////////////////////////
        /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
}

void solve_upper_branched_2(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;

    long int n2 = n * n;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    long int n3 = n * n2;  // 3D array flattened to 1D

    double app;
    double *out = (double *)calloc(n3, sizeof(double));

    long int idx;
    long int ts = (long int)(pow(6, CUTOFF) + 1.5);
    // mpi_printf("SSOR: ts = %ld\n", ts);
    long int *todo1 = (long int *)malloc(ts * sizeof(long int));
    long int *todo2 = (long int *)malloc(ts * sizeof(long int));
    long int *ptr;

    for ( long int i = 0; i < n3; i++ ) {
        app = b[i] / 6.0;
        out[i] += app;
        for (int a=0; a < 7; a++) {
            todo1[a] = index_map_upper[7*i + a];
        }

        for (int a=0; a < CUTOFF; a++) {
            app /= 6.0;
            if (fabs(app) < ELEM_CUTOFF) {
                break;
            }
            int b = 0, c = 0, d = 0;
            while (todo1[b] != -1) {
                idx = todo1[b++];
                c = 0;
                while (index_map_upper[7*idx + c] != -1) {
                    todo2[d++] = index_map_upper[7*idx + c++];
                }

                out[idx] += app;
            }
            todo2[d] = -1;

            ptr = todo1;
            todo1 = todo2;
            todo2 = ptr;
        }
    }

    memcpy(b, out, n3 * sizeof(double));

    free(out);
    free(todo1);
    free(todo2);
}

void solve_lower_branched_2(double *b, int n_loc, int n, int n_start) {
    int nm1 = n - 1;

    long int n2 = n * n;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    long int n3 = n * n2;  // 3D array flattened to 1D

    double app;
    double *out = (double *)calloc(n3, sizeof(double));

    long int idx;
    long int ts = (long int)(pow(6, CUTOFF) + 1.5);
    long int *todo1 = (long int *)malloc(ts * sizeof(long int));
    long int *todo2 = (long int *)malloc(ts * sizeof(long int));
    long int *ptr;

    for ( long int i = 0; i < n3; i++ ) {
        app = b[i] / 6.0;
        out[i] += app;
        for (int a=0; a < 7; a++) {
            todo1[a] = index_map_lower[7*i + a];
        }

        for (int a=0; a < CUTOFF; a++) {
            app /= 6.0;
            if (fabs(app) < ELEM_CUTOFF) {
                break;
            }
            int b = 0, c = 0, d = 0;
            while (todo1[b] != -1) {
                idx = todo1[b++];
                c = 0;
                while (index_map_lower[7*idx + c] != -1) {
                    todo2[d++] = index_map_lower[7*idx + c++];
                }

                out[idx] += app;
            }
            todo2[d] = -1;

            ptr = todo1;
            todo1 = todo2;
            todo2 = ptr;
        }
    }

    memcpy(b, out, n3 * sizeof(double));

    free(out);
    free(todo1);
    free(todo2);
}

void init_index_map_upper_2(int size1, int size2) {
    int n = size2;
    long int n2 = n * n;

    long int map_size = 7 * size1 * n2;  // Need 7 to have the -1 terminator also on the last one
    index_map_upper = (long int *)malloc(map_size * sizeof(long int));

    int i, j, k, k1, k2;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    int cnt;
    long int idx0, idx1;

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
                cnt = 0;

                idx1 = i1 + j0 + k;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }
                idx1 = i2 + j0 + k;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j1 + k;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j2 + k;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j0 + k1;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j0 + k2;
                if ( idx1 < idx0 ) {
                    index_map_upper[7*idx0 + cnt++] = idx1;
                }

                index_map_upper[7*idx0 + cnt] = -1;  // Add the -1 terminator
            }
        }
    }
}

void init_index_map_lower_2(int size1, int size2) {
    int n = size2;
    long int n2 = n * n;

    long int map_size = 7 * size1 * n2;  // Need 7 to have the -1 terminator also on the last one
    index_map_lower = (long int *)malloc(map_size * sizeof(long int));

    int i, j, k, k1, k2;
    long int i0, j0, i1, j1, i2, j2;

    #ifdef __MPI
        mpi_printf("SSOR with MPI not yet implemented\n");
        exit(1);
    #endif

    int cnt;
    long int idx0, idx1;

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
                cnt = 0;

                idx1 = i1 + j0 + k;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }
                idx1 = i2 + j0 + k;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j1 + k;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j2 + k;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j0 + k1;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }
                idx1 = i0 + j0 + k2;
                if ( idx1 > idx0 ) {
                    index_map_lower[7*idx0 + cnt++] = idx1;
                }

                index_map_lower[7*idx0 + cnt] = -1;  // Add the -1 terminator
            }
        }
    }
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


void precond_ssor_apply2(double *in, double *out, int s1, int s2, int n_start) {
    if ( index_map_lower == NULL) {
        init_index_map_lower_2(s1, s2);
        init_index_map_upper_2(s1, s2);
    }
    if ( in != out ) {
        memcpy(out, in, s1 * s2 * s2 * sizeof(double));  // Copy the input to output
    }
    solve_lower_branched_2(out, s1, s2, n_start);  // z = M1^-1 . b
    solve_diag(out, s1, s2, n_start);  // y = M2^-1 . z
    solve_upper_branched_2(out, s1, s2, n_start);  // v = M3^-1 . y
}