#ifndef __MP_MULTIGRID_H
#define __MP_MULTIGRID_H

void prolong(double *in, double *out, int s1, int s2, int ts1, int ts2, int tns);
void restriction(double *in, double *out, int s1, int s2, int n_start);
void smooth(double *in, double *out, int s1, int s2, double tol);

void multigrid_apply(double *in, double *out, int s1, int s2, int n_start1);


#endif // __MP_MULTIGRID_H