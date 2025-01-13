#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

/*
Print information about the OpenMP number of threads
*/
EXTERN_C int get_omp_info(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/*
Apply a 3-D Laplace filter to a 3-D array with cyclic boundary conditions
@param u: the input array
@param u_new: the output array
@param n: the size of the array in each dimension
*/
EXTERN_C void laplace_filter(double *u, double *u_new, int n) {
    int i, j, k;
    int i_prev, i_next, j_prev, j_next, k_prev, k_next;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int n2 = n * n;
    #pragma omp parallel for private(i, j, k, i_prev, i_next, j_prev, j_next, k_prev, k_next, i0, i1, i2, j0, j1, j2)
    for (i = 0; i < n; i++) {
        i_prev = (i - 1 + n) % n;
        i_next = (i + 1) % n;
        i0 = i * n2;
        i1 = i_prev * n2;
        i2 = i_next * n2;
        for (j = 0; j < n; j++) {
            j_prev = (j - 1 + n) % n;
            j_next = (j + 1) % n;
            j0 = j * n;
            j1 = j_prev * n;
            j2 = j_next * n;
            for (k = 0; k < n; k++) {
                k_prev = (k - 1 + n) % n;
                k_next = (k + 1) % n;
                u_new[i0 + j0 + k] = (
                    u[i1 + j0 + k] +
                    u[i2 + j0 + k] +
                    u[i0 + j1 + k] +
                    u[i0 + j2 + k] +
                    u[i0 + j0 + k_prev] +
                    u[i0 + j0 + k_next] -
                    u[i0 + j0 + k] * 6.0
                    );
            }
        }
    }
}

/*
Compute the dot product of two vectors
@param u: the first vector
@param v: the second vector
@param n: the size of the vectors
@return the dot product of the two vectors
*/
EXTERN_C double ddot(double *u, double *v, long int n) {
    long int i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += u[i] * v[i];
    }
    return result;
}

/*
Compute the sum of two vectors scaled by a constant (res = u + alpha * v)
and store the result in a third vector
@param v: the first vector
@param u: the second vector
@param result: the vector to store the result
@param alpha: the scaling constant
@param n: the size of the vectors
*/
EXTERN_C void daxpy(double *v, double *u, double *result, double alpha, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        result[i] = u[i] + alpha * v[i];
    }
}

/*
Compute the sum of two vectors scaled by a constant (u += alpha * v)
and store the result in the second vector
@param v: the first vector
@param u: the second vector
@param alpha: the scaling constant
@param n: the size of the vectors
*/
EXTERN_C void daxpy2(double *v, double *u, double alpha, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        u[i] += alpha * v[i];
    }
}

/*
Compute the Euclidean norm of a vector
@param u: the vector
@param n: the size of the vector
@return the Euclidean norm of the vector
*/
EXTERN_C double norm(double *u, long int n) {
    return sqrt(ddot(u, u, n));
}

/*
Solve the system of linear equations Ax = b using the conjugate gradient method where A is the Laplace filter
@param b: the right-hand side of the system of equations
@param x0: the initial guess for the solution
@param x: the solution to the system of equations
@param tol: the tolerance for the solution
@param n: the size of the arrays (n_tot = n * n * n)
*/
EXTERN_C int conj_grad(double *b, double *x0, double *x, double tol, int n) {
    long int i;
    long int n3 = n * n * n;
    long int limit = n * n;
    long int iter = 0, res = -1;

    // printf("Running conjugate gradient with %d elements\n", n3);

    double *r = (double *)malloc(n3 * sizeof(double));
    double *p = (double *)malloc(n3 * sizeof(double));
    double *Ap = (double *)malloc(n3 * sizeof(double));
    double alpha, beta, r_dot_v, rn_dot_rn, rn_dot_vn;

    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        x[i] = x0[i];
    }
    laplace_filter(x, r, n);  // r = A . x
    daxpy2(b, r, -1.0, n3);  // r = A . x - b

    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        p[i] = r[i] / 6.0;  // p = -v = -(P^-1 . r) = - ( -r / 6.0 ) = r / 6.0
    }

    // Since v = P^-1 . r = -r / 6.0 we do not need to ever compute v
    // We can also remove 2 dot products per iteration by computing
    //   r_dot_v and rn_dot_vn directly from the previous values
    r_dot_v = - ddot(r, r, n3) / 6.0;  // <r, v>

    while(iter < limit) {
        laplace_filter(p, Ap, n);

        alpha = r_dot_v / ddot(p, Ap, n3);  // alpha = <r, v> / <p | A | p>
        daxpy2(p, x, alpha, n3);  // x_new = x + alpha * p
        daxpy2(Ap, r, alpha, n3);  // r_new = r + alpha * Ap

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
        // if (iter % 100 == 0) {
        //     printf("Iteration %ld: %16.8ff %16.8f\n", iter, norm(r, n3), tol);
        // }
    }

    free(r);
    free(p);
    free(Ap);

    return res;
}
