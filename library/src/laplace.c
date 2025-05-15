#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "mpi_base.h"
#include "linalg.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
    long int n2 = size2 * size2;

    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter: u and u_new are the same array (in-place operation not supported)\n");
        exit(1);
    }

    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(u, size1, size2);

    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2)
    for (i = 0; i < size1; i++) {
        i0 = i * n2;
        i1 = i0 + n2;
        i2 = i0 - n2;
        for (j = 0; j < size2; j++) {
            j0 = j * size2;
            j1 = ((j+1) % size2) * size2;
            j2 = ((j-1 + size2) % size2) * size2;
            for (k = 0; k < size2; k++) {
                u_new[i0 + j0 + k] = (
                    u[i1 + j0 + k] +
                    u[i2 + j0 + k] +
                    u[i0 + j1 + k] +
                    u[i0 + j2 + k] +
                    u[i0 + j0 + ((k+1) % size2)] +
                    u[i0 + j0 + ((k-1 + size2) % size2)] -
                    u[i0 + j0 + k] * 6.0
                    );
            }
        }
    }
}

/*
Solve the system of linear equations Ax = b using the conjugate gradient method where A is the Laplace filter
Allows in-place computation by having either:
- x == b
- x == x0
@param b: the right-hand side of the system of equations
@param x0: the initial guess for the solution
@param x: the solution to the system of equations
@param tol: the tolerance for the solution
@param n: the size of the arrays (n_tot = n * n * n)
*/
int conj_grad(double *b, double *x0, double *x, double tol, int size1, int size2) {
    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;
    long int limit = n2;
    long int iter = 0, res = -1;

    // printf("Running conjugate gradient with %d elements\n", n3);

    double *r = (double *)malloc(n3 * sizeof(double));
    double *Ap = (double *)malloc(n3 * sizeof(double));
    double *p = mpi_grid_allocate(size1, size2);
    double alpha, beta, r_dot_v, rn_dot_rn, rn_dot_vn;

    // Allow for inplace computation by having b == x
    laplace_filter(x0, r, size1, size2);  // r = A . x
    daxpy(b, r, -1.0, n3);  // r = A . x - b
    if (x != x0)
    {
        #pragma omp parallel for
        for (i = 0; i < n3; i++) {
            x[i] = x0[i];
        }
    }

    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        p[i] = r[i] / 6.0;  // p = -v = -(P^-1 . r) = - ( -r / 6.0 ) = r / 6.0
    }

    // Since v = P^-1 . r = -r / 6.0 we do not need to ever compute v
    // We can also remove 2 dot products per iteration by computing
    //   r_dot_v and rn_dot_vn directly from the previous values
    r_dot_v = - ddot(r, p, n3);  // <r, v>

    while(iter < limit) {
        laplace_filter(p, Ap, size1, size2);

        alpha = r_dot_v / ddot(p, Ap, n3);  // alpha = <r, v> / <p | A | p>
        daxpy(p, x, alpha, n3);  // x_new = x + alpha * p
        daxpy(Ap, r, alpha, n3);  // r_new = r + alpha * Ap

        rn_dot_rn = ddot(r, r, n3);  // <r_new, r_new>
        if (sqrt(rn_dot_rn) <= tol) {
            res = iter;
            break;
        }

        rn_dot_vn = - rn_dot_rn / 6.0;  // <r_new, v_new>
        beta = rn_dot_vn / r_dot_v;  // beta = <r_new, v_new> / <r, v>
        r_dot_v = rn_dot_vn;  // <r, v> = <r_new, v_new>

        #pragma omp parallel for
        for (i = 0; i < n3; i++) {
            p[i] = beta * p[i] + r[i] / 6.0;  // p = -v + beta * p
        }        

        iter++;
    }

    free(r);
    free(Ap);
    mpi_grid_free(p, size2);

    return res;
}

/*
Solve the system of linear equations Ax = b using the conjugate gradient method where A is the Laplace filter
Allows in-place computation by having either:
- x == b
- x == x0
@param b: the right-hand side of the system of equations
@param x0: the initial guess for the solution
@param x: the solution to the system of equations
@param tol: the tolerance for the solution
@param n: the size of the arrays (n_tot = n * n * n)
@param apply: apply the preconditioner
*/
int conj_grad_precond(
    double *b, double *x0, double *x, double tol, int size1, int size2,
    void (*apply)(double *, double *, int, int, int)
) {
    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;
    long int limit = n2;
    long int iter = 0, res = -1;

    // printf("Running conjugate gradient with %d elements\n", n3);

    double *r = (double *)malloc(n3 * sizeof(double));
    double *Ap = (double *)malloc(n3 * sizeof(double));
    // double *v = (double *)malloc(n3 * sizeof(double));
    double *v = mpi_grid_allocate(size1, size2);
    double *p = mpi_grid_allocate(size1, size2);
    double alpha, beta, r_dot_v, rn_dot_rn, rn_dot_vn;

    // Allow for inplace computation by having b == x
    laplace_filter(x0, r, size1, size2);  // r = A . x
    daxpy(b, r, -1.0, n3);  // r = A . x - b
    if (x != x0)
    {
        #pragma omp parallel for
        for (i = 0; i < n3; i++) {
            x[i] = x0[i];
        }
    }

    apply(r, v, size1, size2, get_n_start());  // v = P^-1 . r
    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        p[i] = -v[i];
    }
    r_dot_v = ddot(r, v, n3);  // <r, v>

    while(iter < limit) {
        laplace_filter(p, Ap, size1, size2);

        alpha = r_dot_v / ddot(p, Ap, n3);  // alpha = <r, v> / <p | A | p>
        daxpy(p, x, alpha, n3);  // x_new = x + alpha * p
        daxpy(Ap, r, alpha, n3);  // r_new = r + alpha * Ap

        rn_dot_rn = ddot(r, r, n3);  // <r_new, r_new>
        if (sqrt(rn_dot_rn) <= tol) {
            res = iter;
            break;
        }

        apply(r, v, size1, size2, get_n_start());  // v = P^-1 . r
        rn_dot_vn = ddot(r, v, n3);  // <r_new, v_new>
        beta = rn_dot_vn / r_dot_v;  // beta = <r_new, v_new> / <r, v>
        r_dot_v = rn_dot_vn;  // <r, v> = <r_new, v_new>

        #pragma omp parallel for
        for (i = 0; i < n3; i++) {
            p[i] = beta * p[i] - v[i];  // p = -v + beta * p
        }        

        iter++;
    }

    free(r);
    // free(v);
    free(Ap);
    mpi_grid_free(v, size2);
    mpi_grid_free(p, size2);

    return res;
}

/*
Apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension
@param precond: the preconditioner function

@return the number of iterations for convergence of the LCG
*/
EXTERN_C int verlet_poisson(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    void (*precond)(double *, double *, int, int, int)
) {
    int iter_conv;

    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    // Compute the constraint with the provisional value of the field phi
    laplace_filter(phi, tmp, size1, size2);
    daxpy(q, tmp, (4 * M_PI) / h, n3);  // sigma_p = A . phi + 4 * pi * rho

    // Apply LCG
    if (precond == NULL) {
        iter_conv = conj_grad(tmp, y, y, tol, size1, size2);  // Inplace y <- y0
    } else {
        iter_conv = conj_grad_precond(tmp, y, y, tol, size1, size2, precond);  // Inplace y <- y0
    }

    // Scale the field with the constrained 'force' term
    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        phi[i] -= y[i];
    }

    // Free temporary arrays
    free(tmp);

    return iter_conv;
}
