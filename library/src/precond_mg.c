#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "charges.h"
#include "linalg.h"
#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"

void precond_mg_init(precond *p) {
    // p->tol = 1E-9;
    p->tol = 5;

    p->n1 = p->n;
    p->n2 = p->n / 2;
    p->n3 = p->n2 / 2;

    p->n_loc1 = get_n_loc();
    p->n_start1 = get_n_start();

    p->n_loc2 = (p->n_loc1 + 1 - (p->n_start1 % 2)) / 2;
    p->n_start2 = (p->n_start1 + 1) / 2;

    p->n_loc3 = (p->n_loc2 + 1 - (p->n_start2 % 2)) / 2;
    p->n_start3 = (p->n_start2 + 1) / 2;

    if (p->n_loc3 == 0) {
        fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        fprintf(stderr, "This case is not yet implemented, please use MG preconditioner with atleast 4 slices\n");
        fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        exit(1);
    }

    // int coars_size1 = n;
    p->grid1 = mpi_grid_allocate(p->n_loc1, p->n1);
    p->grid2 = mpi_grid_allocate(p->n_loc2, p->n2);
    p->grid3 = mpi_grid_allocate(p->n_loc3, p->n3);

    p->apply = precond_mg_apply;
    p->prolong = precond_mg_prolong;
    p->restriction = precond_mg_restriction; 
    p->smooth = precond_mg_smooth;
}

void precond_mg_cleanup(precond *p) {
    if ( p->grid1 != NULL) {
        mpi_grid_free(p->grid1, p->n1);
        p->grid1 = NULL;
    }
    if (p->grid2 != NULL) {
        mpi_grid_free(p->grid2, p->n2);
        p->grid2 = NULL;
    }
    if (p->grid3 != NULL) {
        mpi_grid_free(p->grid3, p->n3);
        p->grid3 = NULL;
    }
}

void precond_mg_apply(precond *p, double *in, double *out) {
    long int size1 = p->n_loc1 * p->n1 * p->n1;
    long int size2 = p->n_loc2 * p->n2 * p->n2;
    long int size3 = p->n_loc3 * p->n3 * p->n3;

    double *tmp1 = mpi_grid_allocate(p->n_loc1, p->n1);
    double *tmp2_1 = mpi_grid_allocate(p->n_loc2, p->n2);
    double *tmp2_2 = mpi_grid_allocate(p->n_loc2, p->n2);
    double *tmp3 = mpi_grid_allocate(p->n_loc3, p->n3);

    #pragma omp parallel for
    for (int i = 0; i < size1; i++) {
        tmp1[i] = 0;  // tmp1 = in
        // p->grid1[i] = 0;  // grid1 = 0
    }

    p->smooth(in, tmp1, p->n_loc1, p->n1, 5);  // out = smooth(in, out)  ~solve(A . out = in)
    laplace_filter(tmp1, p->grid1, p->n_loc1, p->n1);
    #pragma omp parallel for
    for (int i = 0; i < size1; i++) {
        p->grid1[i] = in[i] - p->grid1[i];  // r1 = in - A . out
    }
    p->restriction(p->grid1, tmp2_1, p->n_loc1, p->n1);  // r2 = restriction(r1)

    #pragma omp parallel for
    for (int i = 0; i < size2; i++) {
        p->grid2[i] = 0;  // tmp2 = r2
    }
    p->smooth(tmp2_1, p->grid2, p->n_loc2, p->n2, 5);  // e2 = smooth(r2)  ~solve(A . e2 = r2)

    laplace_filter(p->grid2, tmp2_2, p->n_loc2, p->n2);
    #pragma omp parallel for
    for (int i = 0; i < size2; i++) {
        tmp2_2[i] = tmp2_1[i] - tmp2_1[i];  // tmp2 = r2 - A . e2
    }
    p->restriction(tmp2_2, tmp3, p->n_loc2, p->n2);  // r3 = restriction(r2 - A . e2)

    #pragma omp parallel for
    for (int i = 0; i < size3; i++) {
        p->grid3[i] = 0;
    }
    p->smooth(tmp3, p->grid3, p->n_loc3, p->n3, 20);  // e3 = smooth(r3)  ~solve(A . e3 = r3)

    p->prolong(p->grid3, tmp2_1, p->n_loc3, p->n3, p->n_loc2, p->n2);
    daxpy(tmp2_1, p->grid2, 1.0, size2);  // e2 = e2 + prolong(r3)
    p->prolong(p->grid2, p->grid1, p->n_loc2, p->n2, p->n_loc1, p->n1);
    daxpy(p->grid1, tmp1, 1.0, size1);  // out = out + prolong(e2)
    p->smooth(in, tmp1, p->n_loc1, p->n1, 5);  // out = smooth(in, out)  ~solve(A . out = in)

    #pragma omp parallel for
    for (int i = 0; i < size1; i++) {
        out[i] = tmp1[i];
    }

    mpi_grid_free(tmp1, p->n1);
    mpi_grid_free(tmp2_1, p->n2);
    mpi_grid_free(tmp2_2, p->n2);
    mpi_grid_free(tmp3, p->n3);
}

// TODO !!!!!!!!!!!!!!!!!!!
// Account for restriction and prolong MPI aware when number of procs is odd

void precond_mg_prolong(double *in, double *out, int s1, int s2, int ts1, int target_s2) {
    int a, b;
    long int i, j, k;
    long int i0, j0, k0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    long int target_n2 = target_s2 * target_s2;

    double app;

    #pragma omp parallel for private(i, j, k, i0, j0, k0, i1, j1, k1, a, b, app)
    for (i = 0; i < s1; i++) {
        a = i * n2;
        i0 = i * 2 * target_n2;
        i1 = i0 + target_n2;
        for (j = 0; j < s2; j++) {
            b = j * s2;
            j0 = j * 2 * target_s2;
            j1 = j0 + target_s2;
            for (k = 0; k < s2; k++) {
                k0 = k * 2;
                k1 = k0 + 1;
                app = in[a + b + k] * 0.125;  // 1/8
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
}

void precond_mg_restriction(double *in, double *out, int s1, int s2) {
    // fprintf(stderr, "precond_mg_restriction %d  %d\n", s1, s2);
    int a, b;
    long int i, j, k;
    long int i0, j0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    int s3 = s2 / 2;
    long int n3 = s3 * s3;

    // If the number of slices in the first dimension is odd, we need to wrap around
    // the bottom slice above to apply the averaging with PBCs
    // if (s1 % 2 == 1) {
    mpi_grid_exchange_bot_top(in, s1, s2);
    // }

    // fprintf(stderr, "___grid_exchange_bot_top done\n");

    #pragma omp parallel for private(i, j, k, i0, i1, j0, j1, k1, a, b)
    for (i = 0; i < s1; i+=2) {
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
                ) * 0.125;  // 1/8
            }
        }
    }
    // fprintf(stderr, "___restriction done\n");
}

void precond_mg_smooth_jacobi(double *in, double *out, int s1, int s2, double tol) {
    // fprintf(stderr, "Running Jacobi smoother %d %d\n", s1, s2);
    long int n3 = s1 * s2 * s2;

    double omega = 0.66 / -6.0;

    double *tmp = (double *)malloc(n3 * sizeof(double));

    for (int iter=0; iter < tol; iter++) { 
        // fprintf(stderr, "___Jacobi smoother iter %d\n", iter);
        laplace_filter(out, tmp, s1, s2);  // res += A . out

        #pragma omp parallel for
        for (long int i = 0; i < n3; i++) {
            out[i] += omega * (in[i] - tmp[i]);  // out += omega * (in - res)
        }
    }

    free(tmp);
    // fprintf(stderr, "___DONE\n");
}

void precond_mg_smooth_lcg(double *in, double *out, int s1, int s2, double tol) {
    conj_grad(in, out, out, 1E-9, s1, s2);  // out = ~solve(A . out = in)
}

void precond_mg_smooth_diag(double *in, double *out, int s1, int s2, double tol) {
    long int n3 = s1 * s2 * s2;
    for (int i = 0; i < n3; i++) {
        out[i] = in[i] / -6.0;
    }
}

void precond_mg_smooth(double *in, double *out, int s1, int s2, double tol) {
    // precond_mg_smooth_lcg(in, out, s1, s2, tol);
    precond_mg_smooth_jacobi(in, out, s1, s2, tol);
    // precond_mg_smooth_diag(in, out, s1, s2, tol);
}

