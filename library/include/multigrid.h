#ifndef __MP_MULTIGRID_H
#define __MP_MULTIGRID_H

#define MG_ITER_LIMIT 1000

#define MG_SOLVE_SM 3
#define MG_RECURSION_FACTOR 1.2

void prolong(double *in, double *out, int s1, int s2, int ts1, int ts2, int tns);
void restriction(double *in, double *out, int s1, int s2, int n_start);
void smooth(double *in, double *out, int s1, int s2, double tol);

int multigrid_apply(
    double *in, double *out, int s1, int s2, int n_start1, int sm
);


#endif // __MP_MULTIGRID_H