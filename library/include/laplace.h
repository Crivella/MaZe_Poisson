#ifndef __LAPLACE_H
#define __LAPLACE_H

#include "mp_structs.h"


void laplace_filter(double *u, double *u_new, int size1, int size2);
int conj_grad(double *b, double *x0, double *x, double tol, int size1, int size2);
int conj_grad_precond(
    double *b, double *x0, double *x, double tol, int size1, int size2,
    void (*apply)(double *, double *, int, int, int)
);
int verlet_poisson(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y, int size1, int size2,
    void (*precond)(double *, double *, int, int, int)
);
int verlet_poisson_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y, int size1, int size2
);

int conj_grad_pb(
    double *b, double *x0, double *x, double tol, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);

int verlet_poisson_pb(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);

#endif