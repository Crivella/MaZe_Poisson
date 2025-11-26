#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plcg.h"
#include "constants.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "linalg.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

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
    double alpha, beta, r_dot_v, rn_dot_rn, rn_dot_vn;

    // Allow for inplace computation by having b == x
    laplace_filter(x0, r, size1, size2);  // r = A . x
    daxpy(b, r, -1.0, n3);  // r = A . x - b

    if (x != x0)
    {
        vec_copy(x0, x, n3);  // Inplace copy
    }

    double *Ap = (double *)malloc(n3 * sizeof(double));
    double *p = mpi_grid_allocate(size1, size2);
    // p = -v = -(P^-1 . r) = - ( -r / 6.0 ) = r / 6.0
    vec_copy(r, p, n3);  // p = r
    dscal(p, 1.0 / 6.0, n3);  // p = r / 6.0

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
        // if (sqrt(rn_dot_rn) <= tol) {
        if (norm_inf(r, n3) <= tol) {
            // printf("iter = %d - res = %lf\n", iter, norm_inf(r, n3));
            res = iter;
            break;
        }
    
        rn_dot_vn = - rn_dot_rn / 6.0;  // <r_new, v_new>
        beta = rn_dot_vn / r_dot_v;  // beta = <r_new, v_new> / <r, v>
        r_dot_v = rn_dot_vn;  // <r, v> = <r_new, v_new>

        dscal(p, beta, n3);  // p = beta * p
        daxpy(r, p, 1.0 / 6.0, n3);  // p = p - v = p + r / 6.0

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
    double *v = mpi_grid_allocate(size1, size2);
    double *p = mpi_grid_allocate(size1, size2);
    double alpha, beta, r_dot_v, rn_dot_rn, rn_dot_vn;

    // Allow for inplace computation by having b == x
    laplace_filter(x0, r, size1, size2);  // r = A . x
    daxpy(b, r, -1.0, n3);  // r = A . x - b
    if (x != x0)
    {
        vec_copy(x0, x, n3);  // Inplace copy
    }

    apply(r, v, size1, size2, get_n_start());  // v = P^-1 . r
    vec_copy(v, p, n3);  // p = v
    dscal(p, -1.0, n3);  // p = -v = - (P^-1 . r)
    r_dot_v = ddot(r, v, n3);  // <r, v>

    while(iter < limit) {
        laplace_filter(p, Ap, size1, size2);

        alpha = r_dot_v / ddot(p, Ap, n3);  // alpha = <r, v> / <p | A | p>
        daxpy(p, x, alpha, n3);  // x_new = x + alpha * p
        daxpy(Ap, r, alpha, n3);  // r_new = r + alpha * Ap

        rn_dot_rn = ddot(r, r, n3);  // <r_new, r_new>
        // if (sqrt(rn_dot_rn) <= tol) {
        if (norm_inf(r, n3) <= tol) {
            res = iter;
            break;
        }

        apply(r, v, size1, size2, get_n_start());  // v = P^-1 . r
        rn_dot_vn = ddot(r, v, n3);  // <r_new, v_new>
        beta = rn_dot_vn / r_dot_v;  // beta = <r_new, v_new> / <r, v>
        r_dot_v = rn_dot_vn;  // <r, v> = <r_new, v_new>

        dscal(p, beta, n3);  // p = beta * p
        daxpy(v, p, -1.0, n3);  // p = p - v

        iter++;
    }

    free(r);
    free(Ap);
    mpi_grid_free(v, size2);
    mpi_grid_free(p, size2);

    return res;
}


EXTERN_C int conj_grad_pb(
    double *b, double *x0, double *x, double tol, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
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
    laplace_filter_pb(
        x0, r, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );  // r = A . x
    daxpy(b, r, -1.0, n3);  // r = A . x - b
    if (x != x0)
    {
        vec_copy(x0, x, n3);  // Inplace copy
    }

    // p = -v = -(P^-1 . r) = - ( -r / 6.0 ) = r / 6.0
    vec_copy(r, p, n3);  // p = r
    dscal(p, 1.0 / 6.0, n3);  // p = r / 6.0

    // Since v = P^-1 . r = -r / 6.0 we do not need to ever compute v
    // We can also remove 2 dot products per iteration by computing
    //   r_dot_v and rn_dot_vn directly from the previous values
    r_dot_v = - ddot(r, p, n3);  // <r, v>

    while(iter < limit) {
        laplace_filter_pb(
            p, Ap, size1, size2,
            eps_x, eps_y, eps_z, k2_screen
        );

        alpha = r_dot_v / ddot(p, Ap, n3);  // alpha = <r, v> / <p | A | p>
        daxpy(p, x, alpha, n3);  // x_new = x + alpha * p
        daxpy(Ap, r, alpha, n3);  // r_new = r + alpha * Ap

        rn_dot_rn = ddot(r, r, n3);  // <r_new, r_new>
        // if (sqrt(rn_dot_rn) <= tol) {
        if (norm_inf(r, n3) <= tol) {
            res = iter;
            break;
        }
        
        rn_dot_vn = - rn_dot_rn / 6.0;  // <r_new, v_new>
        beta = rn_dot_vn / r_dot_v;  // beta = <r_new, v_new> / <r, v>
        r_dot_v = rn_dot_vn;  // <r, v> = <r_new, v_new>

        dscal(p, beta, n3);  // p = beta * p
        daxpy(r, p, 1.0 / 6.0, n3);  // p = p - v = p + r / 6.0     

        iter++;
    }

    free(r);
    free(Ap);
    mpi_grid_free(p, size2);

    return res;
}
