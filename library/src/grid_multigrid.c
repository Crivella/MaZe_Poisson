#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "constants.h"
#include "charges.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"

#ifdef __MPI
void multigrid_grid_init_mpi(grid *grid) {
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

void multigrid_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void multigrid_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;

    long int n2 = n * n;

    multigrid_grid_init_mpi(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    memset(grid->phi_p, 0, size * sizeof(double));  // phi_p = 0
    memset(grid->phi_n, 0, size * sizeof(double));  // phi_n = 0

    grid->init_field = multigrid_grid_init_field;
    grid->update_field = multigrid_grid_update_field;
    grid->update_charges = multigrid_grid_update_charges;
}

void multigrid_grid_cleanup(grid * grid) {
    free(grid->q);

    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
}

void multigrid_grid_init_field(grid *grid) {
    long int i;

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    memcpy(grid->phi_p, grid->phi_n, grid->size * sizeof(double));  // phi_prev = phi_n
    // phi_n = constant * q
    memcpy(grid->phi_n, grid->q, grid->size * sizeof(double));
    dscal(grid->phi_n, constant, grid->size);

    if (grid->pb_enabled) {
        conj_grad_pb(
            grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        conj_grad(grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n);
    }
}

int multigrid_grid_update_field(grid *grid) {
    int precond = 1;
    long int n2 = grid->n * grid->n;
    long int n3 = grid->n_local * n2;
    
    double tol = grid->tol;
    double app;
    int iter_conv, limit;
    limit = 1000;

    double *tmp = mpi_grid_allocate(grid->n_local, grid->n);
    double *tmp2 = mpi_grid_allocate(grid->n_local, grid->n);

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    // memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    // vec_copy(grid->phi_n, grid->phi_p, grid->size);  // phi_prev = phi_n
    
    // Compute provisional update for the field phi
    int i;

    // #pragma omp parallel for private(app)
    // for (i = 0; i < n3; i++) {
    //     app = grid->phi_n[i];
    //     grid->phi_n[i] = 2 * app - grid->phi_p[i];
    //     grid->phi_p[i] = app;
    // }
    
    // phi_n = constant * q
    vec_copy(grid->q, tmp, grid->size);
    dscal(tmp, constant, grid->size);

    switch (grid->precond_type) {
        case PRECOND_TYPE_NONE:
            precond = 0;
            break;
        default:
            break;
    }

    if (grid->pb_enabled) {
        fprintf(stderr, "Multigrid Poisson-Boltzmann not implemented yet.\n");
        exit(1);
    } 
    if (precond) {
        fprintf(stderr, "Multigrid with preconditioner not implemented yet.\n");
        exit(1);
    }

    iter_conv = 0;
    app = tol + 1.0;  // Initialize app to a value greater than tol

    while(iter_conv < limit) {
        multigrid_apply(
            tmp, grid->phi_n, grid->n_local, grid->n, grid->n_start,
            MG_SOLVE_SM1, MG_SOLVE_SM2, MG_SOLVE_SM3, MG_SOLVE_SM4
        );

         // Compute the residual
        laplace_filter(grid->phi_n, tmp2, grid->n_local, grid->n);  // tmp2 = A . phi
        daxpy(tmp, tmp2, -1.0, n3);  // tmp2 = A . phi - (- 4pi/h q)
        
        // app = sqrt(ddot(tmp2, tmp2, n3));  // Compute the norm of the residual
        app = norm_inf(tmp2, n3);   // Compute norm_inf of residual
        
        // if (iter_conv > 1000) {
        if (app <= tol){
            // printf("iter = %d - res = %lf\n", iter_conv, app);
            break;
        }
        
        // memset(tmp, 0, n3 * sizeof(double));
        // vec_copy(grid->q, tmp, grid->size);
        // dscal(tmp, constant, grid->size);
        
        iter_conv++;
        // printf("iter = %d - res = %.9lf\n", iter_conv, app);
    }

    // multigrid_apply(
    //     grid->phi_p, grid->phi_n, grid->n_local, grid->n, grid->n_start,
    //     MG_SOLVE_SM1, MG_SOLVE_SM2, MG_SOLVE_SM3, MG_SOLVE_SM4
    // );

    return iter_conv;
}   

double multigrid_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
