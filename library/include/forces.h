#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, long int *charges, double *pos, double *forces,
    double (*g)(double, double, double)
);
double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces);

void rescale_forces(int n_grid, int n_p, double *forces);

double compute_forces_reaction_field(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi_v, double *phi_s, long int *neighbors, long int *charges, double *pos,
    double *forces_rf, double (*g)(double, double, double)
);

double compute_forces_dielec_boundary(
    int n_grid, int n_p, double h, int num_neigh,
    double *k2, double *H_ratio[], double *H_mask[], double *r_hat[],
    double *forces_db
);

double compute_forces_ionic_boundary(
    int n_grid, int n_p, double h, int num_neigh,
    double *k2,double *H_ratio[], double *H_mask[], double *r_hat[],
    double *forces_ib
);

#endif