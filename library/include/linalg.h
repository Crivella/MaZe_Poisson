#ifndef __LINALG_H
#define __LINALG_H

void copy(double *x, double *y, long int n);
void dscal(double *x, double alpha, long int n);
double ddot(double *u, double *v, long int n);
void daxpy(double *v, double *u, double alpha, long int n);
double norm(double *u, long int n);

#endif