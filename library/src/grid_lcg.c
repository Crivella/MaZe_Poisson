#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "charges.h"
#include "laplace.h"
#include "mp_structs.h"
#include "mpi_base.h"

#ifdef __MPI
void lcg_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();

    int n = grid->n;
    int rank = mpid->rank;
    int size = mpid->size;
    long int buffer_size = n * n;

    int div, mod;
    int n_loc, n_start;

    div = n / size;
    mod = n % size;
    for (int i=0; i<size; i++) {
        if (i < mod) {
            n_loc = div + 1;
            n_start = i * n_loc;
        } else {
            n_loc = div;
            n_start = i * n_loc + mod;
        }
        mpid->n_loc_list[i] = n_loc;
        mpid->n_start_list[i] = n_start;
    }

    grid->n_local = mpid->n_loc_list[rank];
    grid->n_start = mpid->n_start_list[rank];
    mpid->n_loc = grid->n_local;
    mpid->n_start = grid->n_start;
    mpid->buffer_size = buffer_size;
    if (size > 1) {
        mpid->bot = (double *)malloc(buffer_size * sizeof(double));
        mpid->top = (double *)malloc(buffer_size * sizeof(double));
    }
}

#else  // __MPI

void lcg_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void lcg_grid_init(grid * grid) {
    int n = grid->n;

    lcg_grid_init_mpi(grid);

    long int size = grid->n_local * n * n;
    grid->size = size;

    grid->y = (double *)malloc(size * sizeof(double));
    grid->phi_p = (double *)malloc(size * sizeof(double));
    grid->q = (double *)malloc(size * sizeof(double));
    grid->phi_n = (double *)malloc(size * sizeof(double));

    grid->init_field = lcg_grid_init_field;
    grid->update_field = lcg_grid_update_field;
    grid->update_charges = lcg_grid_update_charges;
}

void lcg_grid_cleanup(grid * grid) {
}

void lcg_grid_init_field(grid *grid) {
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

double lcg_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
