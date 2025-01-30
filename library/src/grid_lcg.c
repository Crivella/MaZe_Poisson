#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"

void lcg_grid_init(grid * grid) {
    int n = grid->n;

    grid->n_local = get_n_loc();

    long int size = grid->n_local * n * n;
    grid->size = size;

    grid->y = (double *)malloc(size * sizeof(double));
    grid->phi_p = (double *)malloc(size * sizeof(double));
    grid->q = (double *)malloc(size * sizeof(double));
    grid->phi_n = (double *)malloc(size * sizeof(double));

    grid->init_field = lcg_grid_init_field;
    grid->update_field = lcg_grid_update_field;
}

void lcg_grid_cleanup(grid * grid) {
}

void * lcg_grid_init_field(grid *grid) {
    long int i;

    double const constant = -4 * M_PI / grid->h;

    memcpy(grid->phi_p, grid->phi_n, grid->size * sizeof(double));

    #pragma omp parallel for
    for (i = 0; i < grid->size; i++) {
        grid->y[i] = 0.0;
        grid->phi_n[i] = constant * grid->q[i];
    }
    conj_grad(grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n);
}

int lcg_grid_update_field(grid *grid) {
    return verlet_poisson(grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y, grid->n);
}   
