#ifndef __LAPLACE_H
#define __LAPLACE_H


void laplace_filter(double *u, double *u_new, int n);
int conj_grad(double *b, double *x0, double *x, double tol, int n);

void laplace_filter_mpi(double *u, double *u_new, double *top, double *bot, int n_loc, int n);
double conj_grad_mpi_iter1(double *Ap, double *p, int n_loc, int n);
double conj_grad_mpi_iter2(double *Ap, double *p, double *r, double *x, double alpha, int n_loc, int n);
void conj_grad_mpi_iter3(double *r, double *p, double *r_dot, int n_loc, int n);

#endif