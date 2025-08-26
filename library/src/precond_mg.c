#include "multigrid.h"

void precond_mg_apply(double *in, double *out, int s1, int s2, int n_start) {
    multigrid_apply_recursive(in, out, s1, s2, n_start, 5, 5, 15, 5);  // out = MG . in
}
