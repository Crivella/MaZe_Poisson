#include <math.h>

#include "mpi_base.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

#ifdef __LAPACK ///////////////////////////////////////////////////////////////////////////

#include <cblas.h>

EXTERN_C double ddot(double *u, double *v, long int n) {
    double result = 0.0;
    result = cblas_ddot(n, u, 1, v, 1);
    allreduce_sum(&result, 1);
    return result;
}

EXTERN_C void daxpy(double *v, double *u, double alpha, long int n) {
    cblas_daxpy(n, alpha, v, 1, u, 1);
}

EXTERN_C double norm(double *u, long int n) {
    return cblas_dnrm2(n, u, 1);
}

/*
Scale a vector by a constant x = alpha * x
@param x: the vector to be scaled
@param alpha: the scaling constant
@param n: the size of the vector
*/
EXTERN_C void dscal(double *x, double alpha, long int n) {
    cblas_dscal(n, alpha, x, 1);
}

/*
Copy a vector from in to out
@param in: the input vector
@param out: the output vector
*/
EXTERN_C void vec_copy(double *in, double *out, long int n) {
    cblas_dcopy(n, in, 1, out, 1);
}

#else // __LAPACK ///////////////////////////////////////////////////////////////////////////



/*
Copy a vector from in to out
@param in: the input vector
@param out: the output vector
*/
EXTERN_C void vec_copy(double *in, double *out, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        out[i] = in[i];
    }
}

/*
Scale a vector by a constant x = alpha * x
@param x: the vector to be scaled
@param alpha: the scaling constant
@param n: the size of the vector
*/
EXTERN_C void dscal(double *x, double alpha, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        x[i] *= alpha;
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
    allreduce_sum(&result, 1);
    return result;
}

/*
Compute the sum of two vectors scaled by a constant (u += alpha * v)
and store the result in the second vector
@param v: the first vector
@param u: the second vector
@param alpha: the scaling constant
@param n: the size of the vectors
*/
EXTERN_C void daxpy(double *v, double *u, double alpha, long int n) {
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

#endif // __LAPACK ///////////////////////////////////////////////////////////////////////////