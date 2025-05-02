#ifndef __MP_CHARGES_H
#define __MP_CHARGES_H

double g(double x, double L, double h);
double update_charges_cic(int n_grid, int n_p, double h, double *pos, long int *neighbors, long int *charges, double *q);
double update_charges_splquadr(int n_grid, int n_p, double h, double *pos, long int *neighbors, long int *charges, double *q);
double update_charges_splcubic(int n_grid, int n_p, double h, double *pos, long int *neighbors, long int *charges, double *q);

#endif // __MP_CHARGES_H