#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, long int *charges, double *pos, double *forces,
    double (*g)(double, double, double)
);
double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces);

#endif