#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double)
);
double compute_tf_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_sc_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);

// Pairwise nonbonded contribution for intramolecular exclusions (applies opposite sign)
double compute_lj_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces);
double compute_tf_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces);
double compute_sc_pair_force_excl(long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np, double *params, double *forces);

// Intramolecular harmonic bond and angle forces (water-like molecules: 3 sites per molecule).
double compute_forces_harmonic_bond(long int n_p, const double *pos, double *forces, double L, double k, double r0_val, int rank, int size);
double compute_forces_harmonic_angle(long int n_p, const double *pos, double *forces, double L, double k, double theta0_val, int rank, int size);

#endif