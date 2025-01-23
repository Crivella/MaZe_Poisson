#ifndef __LAPLACE_H
#define __LAPLACE_H


void laplace_filter(double *u, double *u_new, int n);
int conj_grad(double *b, double *x0, double *x, double tol, int n);
int verlet_poisson(double tol, double h, double* phi, double* phi_prev, double* q, double* y, int n_grid);

#endif