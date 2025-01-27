#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "mpi_base.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

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