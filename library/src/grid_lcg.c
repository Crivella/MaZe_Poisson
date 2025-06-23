#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "constants.h"
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
}

#else  // __MPI

void lcg_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void lcg_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;

    long int n2 = n * n;

    lcg_grid_init_mpi(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    grid->init_field = lcg_grid_init_field;
    grid->update_field = lcg_grid_update_field;
    grid->update_charges = lcg_grid_update_charges;

    switch (grid->precond_type) {
        case PRECOND_TYPE_BLOCKJACOBI:
            precond_blockjacobi_init();
            break;
        default:
            break;
    }
}

void lcg_grid_cleanup(grid * grid) {
    free(grid->q);

    switch (grid->precond_type) {
        case PRECOND_TYPE_BLOCKJACOBI:
            precond_blockjacobi_cleanup();
            break;
        default:
            break;
    }

    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
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
    conj_grad(grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n);

    if (grid->pb_enabled) {
        memcpy(grid->phi_s_prev, grid->phi_s, grid->size * sizeof(double));

        #pragma omp parallel for
        for (i = 0; i < grid->size; i++) {
            grid->y_s[i] = 0.0;
            grid->phi_s[i] = constant * grid->q[i];
        }

        conj_grad_pb(
            grid->phi_s, grid->y_s, grid->phi_s, grid->tol, grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    }
}

int lcg_grid_update_field(grid *grid) {
    void (*precond)(double *, double *, int, int, int);

    switch (grid->precond_type) {
        case PRECOND_TYPE_NONE:
            precond = NULL;
            break;
        case PRECOND_TYPE_JACOBI:
            precond = precond_jacobi_apply;
            break;
        case PRECOND_TYPE_MG:
            precond = precond_mg_apply;
            break;
        case PRECOND_TYPE_SSOR:
            precond = precond_ssor_apply;
            break;
        case PRECOND_TYPE_BLOCKJACOBI:
            precond = precond_blockjacobi_apply;
            break;
        default:
            break;
    }

    if (grid->pb_enabled) {
        verlet_poisson_pb(
            grid->tol, grid->h, grid->phi_s, grid->phi_s_prev, grid->q, grid->y_s,
            grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    }
    return verlet_poisson(
        grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
        grid->n_local, grid->n,
        precond
    );
}   

double lcg_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
