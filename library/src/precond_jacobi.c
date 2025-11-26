#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mp_structs.h"

void precond_jacobi_apply(double *in, double *out, int s1, int s2, int n_start) {
    long int n3 = s1 * s2 * s2;
    #pragma omp parallel for
    for (long int i = 0; i < n3; i++) {
        out[i] = -in[i] / 6.0;  // out = -in / 6
    }
}
