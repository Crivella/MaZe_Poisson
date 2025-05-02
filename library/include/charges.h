#ifndef __MP_CHARGES_H
#define __MP_CHARGES_H

double spread_cic(double x, double L, double h);
double spread_spline_quadr(double x, double L, double h);
double spread_spline_cubic(double x, double L, double h);

double update_charges(
    int n_grid, int n_p, double h, int num_neigh,
    double *pos, long int *neighbors, long int *charges, double *q,
    double (*g)(double, double, double)
);

#endif // __MP_CHARGES_H