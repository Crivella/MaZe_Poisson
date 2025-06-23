#include <stdio.h>
#include <stdlib.h>

#include "mpi_base.h"
#include "mp_structs.h"

char grid_type_str[2][16] = {"LCG", "FFT"};
int get_grid_type_num() {
    return GRID_TYPE_NUM;
}
char *get_grid_type_str(int n) {
    return grid_type_str[n];
}

char precond_type_str[PRECOND_TYPE_NUM][16] = {"NONE", "JACOBI", "MG", "SSOR", "BLOCKJACOBI"};
int get_precond_type_num() {
    return PRECOND_TYPE_NUM;
}
char *get_precond_type_str(int n) {
    return precond_type_str[n];
}




grid * grid_init(int n, double L, double h, double tol, int grid_type, int precond_type) {
    void   (*init_func)(grid *);
    switch (grid_type) {
        case GRID_TYPE_LCG:
            init_func = lcg_grid_init;
            break;
        case GRID_TYPE_FFT:
            init_func = fft_grid_init;
            break;
        default:
            break;
    }

    grid *new = (grid *)malloc(sizeof(grid));
    new->type = grid_type;
    new->precond_type = precond_type;
    new->n = n;
    new->L = L;
    new->h = h;

    new->n_local = n;
    new->n_start = 0;

    new->y = NULL;
    new->q = NULL;
    new->phi_p = NULL;
    new->phi_n = NULL;
    new->ig2 = NULL;

    init_func(new);

    new->tol = tol;
    new->n_iters = 0;

    new->free = grid_free;

    return new;
}

void grid_pb_init(grid *grid, double eps_s, double I, double w, double kbar2) {
    // Initialize the grid for Poisson-Boltzmann simulations
    grid->eps_s = eps_s;
    grid->I = I;
    grid->w = w;
    grid->kbar2 = kbar2;

    // Initialize the solvent potential and dielectric constant arrays
    int n = grid->n;
    int n_local = grid->n_local;
    grid->phi_s_prev = mpi_grid_allocate(n_local, n);
    grid->phi_s = mpi_grid_allocate(n_local, n);

    // long int eps_size = grid->size * 3;
    grid->eps_x = (double *)malloc(grid->size * sizeof(double));
    grid->eps_y = (double *)malloc(grid->size * sizeof(double));
    grid->eps_z = (double *)malloc(grid->size * sizeof(double));
    #pragma omp parallel for
    for (long int i = 0; i < grid->size; i++) {
        grid->eps_x[i] = eps_s;  // Initialize the dielectric constant in x direction
        grid->eps_y[i] = eps_s;  // Initialize the dielectric constant in y direction
        grid->eps_z[i] = eps_s;  // Initialize the dielectric constant in z direction
    }
    grid->eps[0] = grid->eps_x;  // Assign x dielectric constant
    grid->eps[1] = grid->eps_y;  // Assign y dielectric constant
    grid->eps[2] = grid->eps_z;  // Assign z dielectric constant

    grid->k2 = (double *)malloc(grid->size * sizeof(double));
    #pragma omp parallel for
    for (long int i = 0; i < grid->size; i++) {
        grid->k2[i] = kbar2;  // Initialize the screening factor
    }
}

void grid_pb_free(grid *grid) {
    if (grid->phi_s != NULL) {
        mpi_grid_free(grid->phi_s_prev, grid->n);
        mpi_grid_free(grid->phi_s, grid->n);

        free(grid->eps_x);
        free(grid->eps_y);
        free(grid->eps_z);
    }
}

void grid_free(grid *grid) {
    switch (grid->type) {
        case GRID_TYPE_LCG:
            lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_FFT:
            fft_grid_cleanup(grid);
            break;
        default:
            break;
    }

    grid_pb_free(grid);

    free(grid);
}

void update_eps_and_k2_transition(grid *grid) {
    // Update the dielectric constant and screening factor based on the grid's transition state
    double eps_s = grid->eps_s;
    double kbar2 = grid->kbar2;

    int n = grid->n;
    int n_local = grid->n_local;

    #pragma omp parallel for
    for (long int i = 0; i < grid->size; i++) {
        grid->eps_x[i] = 1.0 + (eps_s - 1.0) * grid->H[0];  // Update x dielectric constant
        grid->eps_y[i] = 1.0 + (eps_s - 1.0) * grid->H[1];  // Update y dielectric constant
        grid->eps_z[i] = 1.0 + (eps_s - 1.0) * grid->H[2];  // Update z dielectric constant
        grid->k2[i] = kbar2 * grid->H[5];  // Update screening factor
    }
}    

void grid_compute_h(grid *grid, particles *particles) {
    // Compute the H function and its derivatives for the transition region
    int n = grid->n;
    double h = grid->h;
    double L = grid->L;
    double w = grid->w;

    double *pos = particles->pos;
    double *radius = particles->solv_radii;

    mpi_fprintf(stderr, "Computing H function not implemented yet\n");
    exit(1);
}