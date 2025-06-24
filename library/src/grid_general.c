#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    new->pb_enabled = 0;  // Poisson-Boltzmann not enabled by default
    new->eps_s = 0.0;  // Dielectric constant of the solvent
    new->w = 0.0;  // Ionic boundary width
    new->kbar2 = 0.0;  // Screening factor
    new->y_s = NULL;  // Intermediate field constraint for solvent
    new->k2 = NULL;  // Screening factor
    new->phi_s_prev = NULL;  // Solvent potential
    new->phi_s = NULL;  // Solvent potential
    new->eps_x = NULL;  // Dielectric constant in x direction
    new->eps_y = NULL;  // Dielectric constant in y direction
    new->eps_z = NULL;  // Dielectric constant in z direction
    

    init_func(new);

    new->tol = tol;
    new->n_iters = 0;

    new->free = grid_free;

    return new;
}

void grid_pb_init(grid *grid, double eps_s, double w, double kbar2) {
    // Initialize the grid for Poisson-Boltzmann simulations
    grid->pb_enabled = 1;  // Enable Poisson-Boltzmann
    grid->eps_s = eps_s;
    grid->w = w;
    grid->kbar2 = kbar2;

    // Initialize the solvent potential and dielectric constant arrays
    int n = grid->n;
    int n_local = grid->n_local;
    grid->y_s = mpi_grid_allocate(n_local, n);
    grid->phi_s_prev = mpi_grid_allocate(n_local, n);
    grid->phi_s = mpi_grid_allocate(n_local, n);

    // long int eps_size = grid->size * 3;
    grid->eps_x = mpi_grid_allocate(n_local, n);
    grid->eps_y = mpi_grid_allocate(n_local, n);
    grid->eps_z = mpi_grid_allocate(n_local, n);
    #pragma omp parallel for
    for (long int i = 0; i < grid->size; i++) {
        grid->eps_x[i] = eps_s;  // Initialize the dielectric constant in x direction
        grid->eps_y[i] = eps_s;  // Initialize the dielectric constant in y direction
        grid->eps_z[i] = eps_s;  // Initialize the dielectric constant in z direction
    }

    grid->k2 = (double *)malloc(grid->size * sizeof(double));
    #pragma omp parallel for
    for (long int i = 0; i < grid->size; i++) {
        grid->k2[i] = kbar2;  // Initialize the screening factor
    }
}

void grid_pb_free(grid *grid) {
    if (grid->pb_enabled) {
        mpi_grid_free(grid->y_s, grid->n);
        mpi_grid_free(grid->phi_s_prev, grid->n);
        mpi_grid_free(grid->phi_s, grid->n);

        mpi_grid_free(grid->eps_x, grid->n);
        mpi_grid_free(grid->eps_y, grid->n);
        mpi_grid_free(grid->eps_z, grid->n);
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

void grid_compute_h(grid *grid, particles *particles) {
    // Compute the H function and its derivatives for the transition region
    int n = grid->n;
    double h = grid->h;
    double L = grid->L;
    double w = grid->w;

    double *pos = particles->pos;
    double *radii = particles->solv_radii;

    mpi_fprintf(stderr, "Computing H function not implemented yet\n");
    exit(1);
}

void grid_update_eps_and_k2(grid *g, particles *p) {
    // Update the dielectric constant and screening factor based on the grid's transition regions
    // grid_compute_h(g, p);  // Compute the H function and its derivatives
    int n = g->n;
    int n_local = g->n_local;
    int n_start = g->n_start;

    double h = g->h;
    double L = g->L;
    double w = g->w;

    double eps_s = g->eps_s;
    double kbar2 = g->kbar2;
    double r_solv;

    long int n2 = n * n;

    double px, py, pz;
    int idx_x, idx_y, idx_z;

    double w2 = w * w;  // Square of the ionic boundary width
    double w3 = w2 * w;  // Cube of the ionic boundary width
    double hd2 = h / 2.0;  // Half the grid spacing

    #pragma omp parallel for
    for (long int i = 0; i < g->size; i++) {
        g->eps_x[i] = (eps_s - 1.0);
        g->eps_y[i] = (eps_s - 1.0);
        g->eps_z[i] = (eps_s - 1.0);
        g->k2[i] = kbar2;  // Update screening factor
    }

    #pragma omp parallel for private(r_solv, px, py, pz, idx_x, idx_y, idx_z)
    for (int np = 0; np < p->n_p; np++) {
        r_solv = p->solv_radii[np];
        px = p->pos[np * 3];
        py = p->pos[np * 3 + 1];
        pz = p->pos[np * 3 + 2];


        double r;
        double r_solv_p2 = pow(r_solv + w, 2);
        double r_solv_m2 = pow(r_solv - w, 2);

        int idx_range = (int)floor((r_solv + w) / h) + 1;

        idx_x = (int)floor(px / h);
        idx_y = (int)floor(py / h);
        idx_z = (int)floor(pz / h);

        // mpi_printf("Doing particle %d with radius %f at position (%f, %f, %f) w=%f, h=%f\n", np, r_solv, px, py, pz, w, h);
        // mpi_printf(" - idx_x=%d, idx_y=%d, idx_z=%d, idx_range=%d\n", idx_x, idx_y, idx_z, idx_range);
        int start_x = idx_x - idx_range;
        int end_x = idx_x + idx_range;
        start_x -= n_start;  // Adjust for local grid start
        end_x -= n_start;  // Adjust for local grid start
        if (start_x >= n_local || end_x < 0) continue;  // Skip if the particle is outside the local grid
        if (start_x < 0) start_x = 0;
        if (end_x >= n_local) end_x = n_local - 1;

        int start_y = idx_y - idx_range;
        int end_y = idx_y + idx_range;
        if (start_y < 0) start_y = 0;
        if (end_y >= n) end_y = n - 1;

        int start_z = idx_z - idx_range;
        int end_z = idx_z + idx_range;
        if (start_z < 0) start_z = 0;
        if (end_z >= n) end_z = n - 1;

        double dx, dy, dz;
        double dx2, dy2, dz2;
        double app1, app2;

        // mpi_printf(" - start_x=%d, end_x=%d\n", start_x, end_x);
        // mpi_printf(" - start_y=%d, end_y=%d\n", start_y, end_y);
        // mpi_printf(" - start_z=%d, end_z=%d\n", start_z, end_z);
        
        for (int i = start_x; i <= end_x; i++) {
            dx = px - (i + n_start) * h;  // Calculate the distance in x direction
            dx2 = dx * dx;
            for (int j = start_y; j <= end_y; j++) {
                dy = py - j * h;  // Calculate the distance in y direction
                dy2 = dy * dy;
                // mpi_fprintf(stderr, "   - i=%d, j=%d, dx=%f, dy=%f, r2=%f r_solv_P2=%f\n", i, j, dx, dy, r2, r_solv_p2);
                for (int k = start_z; k <= end_z; k++) {
                    dz = pz - k * h;  // Calculate the distance in z direction
                    dz2 = dz * dz;
                    r = dx2 + dy2 + dz2;
                    long int idx = i * n2 + j * n + k;  // Calculate the index in the grid

                    if (r > r_solv_p2) {
                        // Outside the radius, skip this point
                        // continue;  // Skip if outside the radius
                    } else if (r > r_solv_m2) {
                        // Inside the transition region, set dielectric constant to a fraction
                        app2 = sqrt(r) - r_solv + w;  // Calculate the distance in the transition region
                        g->k2[idx] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (0): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        g->k2[idx] = 0.0;  // Set screening factor to zero
                    }

                    app1 = dx + hd2;  // Adjust for half the grid spacing
                    app2 = app1 * app1 + dy2 + dz2;
                    if (app2 > r_solv_p2) {
                        // Do nothihng
                    } else if (app2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(app2) - r_solv + w;
                        g->eps_x[idx] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (X): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        g->eps_x[idx] = 0.0;
                    }

                    app1 = dy + hd2;  // Adjust for half the grid spacing
                    app2 = dx2 + app1 * app1 + dz2;
                    if (app2 > r_solv_p2) {
                        // Do nothihng
                    } else if (app2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(app2) - r_solv + w;
                        g->eps_y[idx] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (Y): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        g->eps_y[idx] = 0.0;
                    }

                    app1 = dz + hd2;  // Adjust for half the grid spacing
                    app2 = dx2 + dy2 + app1 * app1;
                    if (app2 > r_solv_p2) {
                        // Do nothihng
                    } else if (app2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(app2) - r_solv + w;
                        g->eps_z[idx] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (Z): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        g->eps_z[idx] = 0.0;
                    }
                }
            }
        }
    }
    for (long int i = 0; i < g->size; i++) {
        g->eps_x[i] += 1.0;  // Update x dielectric constant
        g->eps_y[i] += 1.0;  // Update y dielectric constant
        g->eps_z[i] += 1.0;  // Update z dielectric constant
        // mpi_printf("%.1f  ", g->eps_x[i]);
    }
    // for (long int i = 0; i < grid->size; i++) {
    //     grid->eps_x[i] = 1.0 + (eps_s - 1.0) * grid->H[0][i];  // Update x dielectric constant
    //     grid->eps_y[i] = 1.0 + (eps_s - 1.0) * grid->H[1][i];  // Update y dielectric constant
    //     grid->eps_z[i] = 1.0 + (eps_s - 1.0) * grid->H[2][i];  // Update z dielectric constant
    //     grid->k2[i] = kbar2 * grid->H[5][i];  // Update screening factor
    // }
}    
