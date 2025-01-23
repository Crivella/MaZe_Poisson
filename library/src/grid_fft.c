#include "charges.h"
#include "mp_structs.h"
#include "fftw_wrap.h"

void * fft_grid_init_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);
}

int fft_grid_update_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);

    return 0;
}

double fft_grid_update_charges(grid *grid, particles *p) {
    return update_charges(grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q);
}
