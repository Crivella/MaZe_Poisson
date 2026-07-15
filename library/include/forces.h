#ifndef __FORCES_H
#define __FORCES_H

#include "mp_structs.h"

static inline void pbc_displacement(
    const double *pos, long int ia, long int ib, double L, double *dx, double *dy, double *dz, double *dr2
) {
    double x = pos[ib * 3]     - pos[ia * 3];
    double y = pos[ib * 3 + 1] - pos[ia * 3 + 1];
    double z = pos[ib * 3 + 2] - pos[ia * 3 + 2];
    x -= L * nearbyint(x / L);
    y -= L * nearbyint(y / L);
    z -= L * nearbyint(z / L);
    *dx = x;
    *dy = y;
    *dz = z;
    *dr2 = x * x + y * y + z * z;
}

double compute_force_fd(
    int n_grid, int n_p, int n_loc, int n_start, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double)
);
double compute_force_short_range(
    int n_p,
    double *pos,
    double *charges,
    double *forces,
    double R_c,
    double sigma_gauss,
    double L,
    neighbor **neighbors,
    int np_local, int np_start
);
// Short-range analytic correction for the Wendland C2 screening density. Its compact support
// (r <= sigma) means the correction is exactly zero (value and slope) at r = sigma, so no
// truncation shift is needed, unlike the Gaussian case above.
double compute_force_short_range_wendland_c2(
    int n_p,
    double *pos,
    double *charges,
    double *forces,
    double sigma,
    double L,
    neighbor **neighbors,
    int np_local, int np_start
);
// Same as above, for the Wendland C4 screening density.
double compute_force_short_range_wendland_c4(
    int n_p,
    double *pos,
    double *charges,
    double *forces,
    double sigma,
    double L,
    neighbor **neighbors,
    int np_local, int np_start
);
double compute_tf_forces(
    int n_p, int n_typ, double L, int *types, double *pos, double *params,
    double r_cut, neighbor **neighbors, int np_local, int np_start,
    double *forces
);
double compute_sc_forces(
    int n_p, double L, double *pos, double *params,
    double r_cut, neighbor **neighbors, int np_local, int np_start,
    double *forces
);
double compute_lj_forces(
    int n_p, int n_typ, double L, int *types, double *pos, double *params,
    double r_cut, neighbor **neighbors, int np_local, int np_start,
    double *forces, int lj_force_shift
);

// Pairwise nonbonded contribution for intramolecular exclusions (applies opposite sign)
double compute_lj_pair_force_excl(
    long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np,
    double *params, double *forces
);
double compute_tf_pair_force_excl(
    long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np,
    double *params, double *forces
);
double compute_sc_pair_force_excl(
    long int ia, long int ib, double vx, double vy, double vz, double r_cut, long int np,
    double *params, double *forces
);

// Intramolecular harmonic bond and angle forces (water-like molecules: 3 sites per molecule).
double compute_forces_harmonic_bond(
    long int n_p, const double *pos, double *forces, double L, double k, double r0_val
);
double compute_forces_harmonic_angle(
    long int n_p, const double *pos, double *forces, double L, double k, double theta0_val
);

#endif
