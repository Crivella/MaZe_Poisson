#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "laplace.h"
#include "mympi.h"
#include "charges.h"
#include "mp_structs.h"

lcg_grid * lcg_grid_init(int n, double L, double h, double tol) {
    lcg_grid *grid = (lcg_grid *)malloc(sizeof(lcg_grid));
    grid->n = n;
    grid->L = L;
    grid->h = h;

    grid->n_local = get_n_loc();

    grid->y = (double *)malloc(grid->n_local * n * n * sizeof(double));
    grid->q = (double *)malloc(grid->n_local * n * n * sizeof(double));
    grid->phi_p = (double *)malloc(grid->n_local * n * n * sizeof(double));
    grid->phi_n = (double *)malloc(grid->n_local * n * n * sizeof(double));

    grid->tol = tol;
    grid->n_iters = 0;

    grid->free = lcg_grid_free;
    grid->init_field = lcg_grid_init_field;
    grid->update_field = lcg_grid_update_field;
    grid->update_charges = lcg_grid_update_charges;

    return grid;
}

void * lcg_grid_free(lcg_grid *grid) {
    free(grid->y);
    free(grid->q);
    free(grid->phi_p);
    free(grid->phi_n);
    free(grid);

    return NULL;
}

void * lcg_grid_init_field(lcg_grid *grid) {
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

    return NULL;
}

int lcg_grid_update_field(lcg_grid *grid) {
    return verlet_poisson(grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y, grid->n);
}   

double lcg_grid_update_charges(lcg_grid *grid, particles *p) {
    return update_charges(grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q);
}
