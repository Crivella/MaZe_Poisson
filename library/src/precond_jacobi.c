#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mp_structs.h"
#include "mpi_base.h"

void precond_jacobi_init(precond *p) {
    p->n1 = p->n;

    p->n_loc1 = get_n_loc();

    p->apply = precond_jacobi_apply;
}

void precond_jacobi_cleanup(precond *p) {
    // No cleanup needed for Jacobi preconditioner
}

void precond_jacobi_apply(precond *p, double *in, double *out) {
    long int n3 = p->n_loc1 * p->n1 * p->n1;
    #pragma omp parallel for
    for (long int i = 0; i < n3; i++) {
        out[i] = -in[i] / 6.0;  // out = -in / 6
    }
}
