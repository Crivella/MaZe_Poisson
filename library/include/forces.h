#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces);
double compute_force_fd_mpi(
        int n_grid, int n_loc, int n_start, int n_p, double h,
        double *phi, double *bot, double *top, double *q, long int *neighbors,
        double *forces
    );
double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces);

#endif