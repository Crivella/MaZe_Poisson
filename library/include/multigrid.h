#ifndef __MP_MULTIGRID_H
#define __MP_MULTIGRID_H

#define MG_ITER_LIMIT 1000

#define MG_SOLVE_SM1 2
#define MG_SOLVE_SM2 4
#define MG_SOLVE_SM3 6
#define MG_SOLVE_SM4 2

#define MG_PRECD_SM1 5
#define MG_PRECD_SM2 5
#define MG_PRECD_SM3 15
#define MG_PRECD_SM4 5

void prolong(double *in, double *out, int s1, int s2, int ts1, int ts2, int tns);
void restriction(double *in, double *out, int s1, int s2, int n_start);
void smooth(double *in, double *out, int s1, int s2, double tol);

int multigrid_apply(
    double *in, double *out, int s1, int s2, int n_start1,
    int sm1, int sm2, int sm3, int sm4
);


#endif // __MP_MULTIGRID_H