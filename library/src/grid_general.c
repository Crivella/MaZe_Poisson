#include <stdlib.h>

#include "mympi.h"
#include "mp_structs.h"

grid * grid_init(int n, double L, double h, double tol, int type) {
    grid *new = (grid *)malloc(sizeof(grid));
    new->type = type;
    new->n = n;
    new->L = L;
    new->h = h;

    new->n_local = get_n_loc();

    new->y = NULL;
    // new->q = NULL;
    new->phi_p = NULL;
    // new->phi_n = NULL;
    new->ig2 = NULL;

    switch (type) {
        case GRID_TYPE_LCG:
            new->y = (double *)malloc(new->n_local * n * n * sizeof(double));
            new->phi_p = (double *)malloc(new->n_local * n * n * sizeof(double));

            new->init_field = lcg_grid_init_field;
            new->update_field = lcg_grid_update_field;
            new->update_charges = lcg_grid_update_charges;
            break;
        case GRID_TYPE_FFT:
            new->ig2 = (double *)malloc(n * n * n * sizeof(double));

            new->init_field = fft_grid_init_field;
            new->update_field = fft_grid_update_field;
            new->update_charges = fft_grid_update_charges;
            break;
        default:
            break;
    }
    
    new->q = (double *)malloc(new->n_local * n * n * sizeof(double));
    new->phi_n = (double *)malloc(new->n_local * n * n * sizeof(double));

    new->tol = tol;
    new->n_iters = 0;

    new->free = grid_free;

    return new;
}

void * grid_free(grid *grid) {
    if (grid->ig2 != NULL) {
        free(grid->ig2);
    }
    if (grid->phi_p != NULL) {
        free(grid->phi_p);
    }
    if (grid->y != NULL) {
        free(grid->y);
    }
    if (grid->q != NULL) {
        free(grid->q);
    }
    if (grid->phi_n != NULL) {
        free(grid->phi_n);
    }
    free(grid);

    return NULL;
}
