#ifndef __LINALG_H
#define __LINALG_H

double ddot(double *u, double *v, long int n);
void daxpy(double *v, double *u, double alpha, long int n);
double norm(double *u, long int n);

#endif