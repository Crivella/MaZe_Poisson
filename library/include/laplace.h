#ifndef __LAPLACE_H
#define __LAPLACE_H

#include "enums.h"

void laplace_filter(double *u, double *u_new, int size1, int size2);
void laplace_filter_standard(double *u, double *u_new, int size1, int size2);
void laplace_filter_mehrstellen4(double *u, double *u_new, int size1, int size2);
void laplace_jacobi_step(
    double *rhs, double *u, double *u_new,
    int size1, int size2, double omega
);
void laplace_filter_rhs(double *u, double *u_new, int size1, int size2);
void laplace_set_discretization(electrostatic_discretization_type discretization);
electrostatic_discretization_type laplace_get_discretization(void);
void laplace_filter_pb(
    double *u, double *u_new, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);

#endif // __LAPLACE_H
