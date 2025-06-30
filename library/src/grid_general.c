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
    grid->k2 = (double *)malloc(grid->size * sizeof(double));
}

void grid_pb_free(grid *grid) {
    if (grid->pb_enabled) {
        mpi_grid_free(grid->y_s, grid->n);
        mpi_grid_free(grid->phi_s_prev, grid->n);
        mpi_grid_free(grid->phi_s, grid->n);

        mpi_grid_free(grid->eps_x, grid->n);
        mpi_grid_free(grid->eps_y, grid->n);
        mpi_grid_free(grid->eps_z, grid->n);

        free(grid->k2);
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

void grid_update_eps_and_k2(grid *g, particles *p) {
    // Update the dielectric constant and screening factor based on the grid's transition regions
    // int size = get_size();
    // if (size > 1) {
    //     mpi_fprintf(stderr, "Poisson-Boltzmann update_eps_and_k2 not supported in parallel yet.\n");
    //     exit(1);
    // }
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

    long int size = g->size;
    double *k2 = g->k2;
    double *eps_x = g->eps_x;
    double *eps_y = g->eps_y;
    double *eps_z = g->eps_z;

    #pragma omp parallel for
    for (long int i = 0; i < size; i++) {
        eps_x[i] = (eps_s - 1.0);
        eps_y[i] = (eps_s - 1.0);
        eps_z[i] = (eps_s - 1.0);
        k2[i] = kbar2;  // Update screening factor
    }

    // #pragma \
    //     omp parallel for private(r_solv, px, py, pz, idx_x, idx_y, idx_z) \
    //     reduction(*:k2[:size], eps_x[:size], eps_y[:size], eps_z[:size])
    for (int np = 0; np < p->n_p; np++) {
        r_solv = p->solv_radii[np];
        px = p->pos[np * 3];
        py = p->pos[np * 3 + 1];
        pz = p->pos[np * 3 + 2];


        double r2;
        double r_solv_p2 = pow(r_solv + w, 2);
        double r_solv_m2 = pow(r_solv - w, 2);

        int idx_range = (int)floor((r_solv + w) / h) + 1;

        idx_x = (int)floor(px / h);
        idx_y = (int)floor(py / h);
        idx_z = (int)floor(pz / h);

        double dx, dy, dz;
        double dx2, dy2, dz2;
        double app1, app2;

        int i0, j0, k0;
        long int idx_cen;
        
        for (int di = -idx_range; di <= idx_range; di++) {
            i0 = idx_x + di;
            dx = px - i0 * h;  // Calculate the distance in x direction
            dx2 = dx * dx;
            i0 = (i0 + n) % n;  // Wrap around for periodic boundary conditions
            i0 -= n_start;  // Adjust for local grid start
            if (i0 < 0 || i0 >= n_local) continue;  // Skip if the point is outside the local grid
            i0 *= n2;  // Convert to linear index
            for (int dj = -idx_range; dj <= idx_range; dj++) {
                j0 = idx_y + dj;
                dy = py - j0 * h;  // Calculate the distance in y direction
                dy2 = dy * dy;
                j0 = (j0 + n) % n;  // Wrap around for periodic boundary conditions
                j0 *= n;
                for (int dk = -idx_range; dk <= idx_range; dk++) {
                    k0 = idx_z + dk;
                    dz = pz - k0 * h;  // Calculate the distance in z direction
                    dz2 = dz * dz;
                    k0 = (k0 + n) % n;  // Wrap around for periodic boundary conditions

                    r2 = dx2 + dy2 + dz2;

                    idx_cen = i0 + j0 + k0;  // Calculate the index in the grid

                    if (r2 >= r_solv_p2) {
                        // Outside the radius, skip this point
                        // continue;  // Skip if outside the radius
                    } else if (r2 > r_solv_m2) {
                        // Inside the transition region, set dielectric constant to a fraction
                        app2 = sqrt(r2) - r_solv + w;  // Calculate the distance in the transition region
                        k2[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (0): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        k2[idx_cen] = 0.0;  // Set screening factor to zero
                    }

                    // *************** X + h/2 ***************
                    app1 = dx - hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_x[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (X): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_x[idx_cen] = 0.0;
                    }

                    // *************** Y + h/2 ***************
                    app1 = dy - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_y[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (Y): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_y[idx_cen] = 0.0;
                    }

                    // *************** Z + h/2 ***************
                    app1 = dz - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_z[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                        // mpi_printf("Transition region (Z): i=%d, j=%d, k=%d\n", i, j, k);
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_z[idx_cen] = 0.0;
                    }
                }
            }
        }
    }
    for (long int i = 0; i < g->size; i++) {
        eps_x[i] += 1.0;  // Update x dielectric constant
        eps_y[i] += 1.0;  // Update y dielectric constant
        eps_z[i] += 1.0;  // Update z dielectric constant
        // mpi_printf("%.1f  ", g->eps_x[i]);
    }
    // for (long int i = 0; i < grid->size; i++) {
    //     grid->eps_x[i] = 1.0 + (eps_s - 1.0) * grid->H[0][i];  // Update x dielectric constant
    //     grid->eps_y[i] = 1.0 + (eps_s - 1.0) * grid->H[1][i];  // Update y dielectric constant
    //     grid->eps_z[i] = 1.0 + (eps_s - 1.0) * grid->H[2][i];  // Update z dielectric constant
    //     grid->k2[i] = kbar2 * grid->H[5][i];  // Update screening factor
    // }
}    

double grid_get_pb_delta_energy_elec(grid *g){
    if (! g->pb_enabled) {
        // Poisson-Boltzmann is not enabled, return 0
        return 0.0;
    }

    double delta_energy = 0.0;

    #pragma omp parallel for reduction(+:delta_energy)
    for (long int i = 0; i < g->size; i++) {
        // Calculate the change in energy due to the Poisson-Boltzmann potential
        delta_energy += 0.5 * (g->phi_s[i] - g->phi_n[i]);
    }

    allreduce_sum(&delta_energy, 1);

    return delta_energy;
}