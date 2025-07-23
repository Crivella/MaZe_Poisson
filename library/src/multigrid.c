#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "multigrid.h"


/*
Apply the multigrid method to solve the Poisson equation.  A.out = in

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param n_start1: starting index for the first dimension (used for restriction)
*/
void multigrid_apply(double *in, double *out, int s1, int s2, int n_start1) {
    int n1 = s2;
    int n2 = n1 / 2;
    int n3 = n2 / 2;

    int n_loc1 = s1;

    int n_loc2 = (n_loc1 + 1 - (n_start1 % 2)) / 2;
    int n_start2 = (n_start1 + 1) / 2;

    int n_loc3 = (n_loc2 + 1 - (n_start2 % 2)) / 2;
    int n_start3 = (n_start2 + 1) / 2;

    if (n_loc3 == 0) {
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        mpi_fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        mpi_fprintf(stderr, "This case is not yet implemented, please use MG preconditioner with atleast 4 slices\n");
        mpi_fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        exit(1);
    }

    long int size1 = n_loc1 * n1 * n1;
    long int size2 = n_loc2 * n2 * n2;
    long int size3 = n_loc3 * n3 * n3;

    double *r1 = mpi_grid_allocate(n_loc1, n1);
    double *r2 = mpi_grid_allocate(n_loc2, n2);
    double *r3 = mpi_grid_allocate(n_loc3, n3);

    double *e2 = mpi_grid_allocate(n_loc2, n2);
    double *e3 = mpi_grid_allocate(n_loc3, n3);

    double *tmp2 = mpi_grid_allocate(n_loc2, n2);

    // #pragma omp parallel for
    // for (int i = 0; i < size1; i++) {
    //     out[i] = 0;  // tmp1 = in
    // }
    smooth(in, out, n_loc1, n1, 5);  // out = smooth(in, out)  ~solve(A . out = in)
    // r1  =  in - A . out
    laplace_filter(out, r1, n_loc1, n1);
    // #pragma omp parallel for
    // for (long int i = 0; i < size1; i++) {
    //     r1[i] = in[i] - r1[i];  // r1 = in - A . out
    // }
    dscal(r1, -1.0, size1);
    daxpy(in, r1, 1.0, size1);  // r1 = in + r1
    restriction(r1, r2, n_loc1, n1, n_start1);  // r2 = restriction(r1)

    // #pragma omp parallel for
    // for (long int i = 0; i < size2; i++) {
    //     e2[i] = 0;  // tmp2 = r2
    // }
    memset(e2, 0, size2 * sizeof(double));  // e2 = 0
    smooth(r2, e2, n_loc2, n2, 5);  // e2 = smooth(r2)  ~solve(A . e2 = r2)
    // tmp2  =  r2 - A . e2
    laplace_filter(e2, tmp2, n_loc2, n2);
    // #pragma omp parallel for
    // for (long int i = 0; i < size2; i++) {
    //     tmp2[i] = r2[i] - tmp2[i];  // tmp2 = r2 - A . e2
    // }
    dscal(tmp2, -1.0, size2);
    daxpy(e2, tmp2, 1.0, size2);  // e2 = e2 + tmp2
    restriction(tmp2, r3, n_loc2, n2, n_start2);  // r3 = restriction(r2 - A . e2)

    // #pragma omp parallel for
    // for (long int i = 0; i < size3; i++) {
    //     e3[i] = 0;
    // }
    memset(e3, 0, size3 * sizeof(double));  // e3 = 0
    smooth(r3, e3, n_loc3, n3, 15);  // e3 = smooth(r3)  ~solve(A . e3 = r3)

    prolong(e3, r2, n_loc3, n3, n_loc2, n2, n_start2);
    daxpy(r2, e2, 1.0, size2);  // e2 = e2 + prolong(r3)
    prolong(e2, r1, n_loc2, n2, n_loc1, n1, n_start1);
    daxpy(r1, out, 1.0, size1);  // out = out + prolong(e2)
    smooth(in, out, n_loc1, n1, 5);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r1, n1);
    mpi_grid_free(r2, n2);
    mpi_grid_free(r3, n3);
    mpi_grid_free(e2, n2);
    mpi_grid_free(e3, n3);
    mpi_grid_free(tmp2, n2);
}

void prolong(double *in, double *out, int s1, int s2, int target_s1, int target_s2, int target_n_start) {
    int a, b;
    long int i, j, k;
    long int i0, j0, k0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    long int target_n2 = target_s2 * target_s2;

    double app;

    int d = target_n_start % 2;

    #pragma omp parallel for private(i, j, k, i0, j0, k0, i1, j1, k1, a, b, app)
    for (i = 0; i < s1; i++) {
        a = i * n2;
        i0 = (i * 2 + d) * target_n2;
        i1 = i0 + target_n2;
        for (j = 0; j < s2; j++) {
            b = j * s2;
            j0 = j * 2 * target_s2;
            j1 = j0 + target_s2;
            for (k = 0; k < s2; k++) {
                k0 = k * 2;
                k1 = k0 + 1;
                app = in[a + b + k] * 0.125;
                out[i0 + j0 + k0] = app;
                out[i0 + j0 + k1] = app;
                out[i0 + j1 + k0] = app;
                out[i0 + j1 + k1] = app;
                out[i1 + j0 + k0] = app;
                out[i1 + j0 + k1] = app;
                out[i1 + j1 + k0] = app;
                out[i1 + j1 + k1] = app;
            }
        }
    }

    // In case of odd number of slices, we need to wrap around the top slice from the proc below
    // as the 1st bottom slice of the current proc
    mpi_grid_exchange_bot_top(out, target_s1, target_s2);
    if (d) {
        memcpy(out, out - target_n2, target_n2 * sizeof(double));
    }
}

void restriction(double *in, double *out, int s1, int s2, int n_start) {
    int a, b;
    long int i, j, k;
    long int i0, j0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    int s3 = s2 / 2;
    long int n3 = s3 * s3;

    // If the number of slices in the first dimension is odd, we need to wrap around
    // the bottom slice above to apply the averaging with PBCs
    mpi_grid_exchange_bot_top(in, s1, s2);

    #pragma omp parallel for private(i, j, k, i0, i1, j0, j1, k1, a, b)
    for (i = n_start % 2; i < s1; i+=2) {
        i0 = i * n2;
        i1 = (i+1) * n2;
        a = i / 2 * n3;
        for (j = 0; j < s2; j+=2) {
            j0 = j * s2;
            j1 = ((j+1) % s2) * s2;
            b = j / 2 * s3;
            for (k = 0; k < s2; k+=2) {
                k1 = (k+1) % s2;
                out[a + b + k / 2] = (
                    in[i0 + j0 + k] +
                    in[i0 + j0 + k1] +
                    in[i0 + j1 + k] +
                    in[i0 + j1 + k1] +
                    in[i1 + j0 + k] +
                    in[i1 + j0 + k1] +
                    in[i1 + j1 + k] +
                    in[i1 + j1 + k1]
                ) * 0.125;
            }
        }
    }
}

void smooth_jacobi(double *in, double *out, int s1, int s2, double tol) {
    long int n3 = s1 * s2 * s2;

    double omega = 0.66 / -6.0;

    double *tmp = (double *)malloc(n3 * sizeof(double));

    for (int iter=0; iter < tol; iter++) { 
        laplace_filter(out, tmp, s1, s2);  // res += A . out

        #pragma omp parallel for
        for (long int i = 0; i < n3; i++) {
            out[i] += omega * (in[i] - tmp[i]);  // out += omega * (in - res)
        }
    }

    free(tmp);
}

void smooth_lcg(double *in, double *out, int s1, int s2, double tol) {
    conj_grad(in, out, out, 1E-1, s1, s2);  // out = ~solve(A . out = in)
}

void smooth_diag(double *in, double *out, int s1, int s2, double tol) {
    long int n3 = s1 * s2 * s2;
    for (int i = 0; i < n3; i++) {
        out[i] = in[i] / -6.0;
    }
}

void smooth(double *in, double *out, int s1, int s2, double tol) {
    // smooth_lcg(in, out, s1, s2, tol);
    smooth_jacobi(in, out, s1, s2, tol);
    // smooth_diag(in, out, s1, s2, tol);
}

