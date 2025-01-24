#include <stdlib.h>
#include <math.h>

#include "mpi_base.h"
#include "charges.h"
#include "mp_structs.h"
#include "fftw_wrap.h"


grid * grid_init(int n, double L, double h, double tol, int type) {
    grid *new = (grid *)malloc(sizeof(grid));
    new->type = type;
    new->n = n;
    new->L = L;
    new->h = h;

    new->n_local = get_n_loc();

    new->y = NULL;
    new->q = NULL;
    new->phi_p = NULL;
    new->phi_n = NULL;
    new->ig2 = NULL;

    new->update_charges = grid_update_charges;

    long int size;

    switch (type) {
        case GRID_TYPE_LCG:
            lcg_grid_init(new);
            break;
        case GRID_TYPE_FFT:
            fft_grid_init(new);
            break;
        default:
            break;
    }

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
}

double grid_update_charges(grid *grid, particles *p) {
    return update_charges(grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q);
}
