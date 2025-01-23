#include <string.h>
#include <math.h>

#include "laplace.h"
#include "charges.h"
#include "mp_structs.h"

void * lcg_grid_init_field(grid *grid) {
    long int i;
    long int n3 = grid->n_local * grid->n * grid->n;

    double const constant = -4 * M_PI / grid->h;

    memcpy(grid->phi_p, grid->phi_n, n3 * sizeof(double));

    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        grid->y[i] = 0.0;
        grid->phi_n[i] = constant * grid->q[i];
    }
    conj_grad(grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n);
}

int lcg_grid_update_field(grid *grid) {
    return verlet_poisson(grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y, grid->n);
}   

double lcg_grid_update_charges(grid *grid, particles *p) {
    return update_charges(grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q);
}
