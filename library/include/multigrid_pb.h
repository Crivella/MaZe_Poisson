#ifndef __MP_MULTIGRID_PB_H
#define __MP_MULTIGRID_PB_H

#define MG_ITER_LIMIT_PB 1000

#define MG_SOLVE_SM_PB 4
#define MG_RECURSION_FACTOR_PB 2

void restriction_eps(double *eps_in, double *eps_out, int s1, int s2, int axis);
void smooth_pb(double *in, double *out, int s1, int s2, double tol, double *eps_x, double *eps_y, double *eps_z, double *k2_screen);

int multigrid_pb_apply(
    double *in, double *out, int s1, int s2, int n_start1, int sm, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);


#endif // __MP_MULTIGRID_PB_H