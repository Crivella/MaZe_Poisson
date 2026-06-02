#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"

grid * grid_init(int n, double L, double h, double tol, double eps, double eps_int, int grid_type, int precond_type) {
    void   (*init_func)(grid *);
    switch (grid_type) {
        case GRID_TYPE_LCG:
            init_func = lcg_grid_init;
            break;
        case GRID_TYPE_FFT:
            init_func = fft_grid_init;
            break;
        case GRID_TYPE_MGRID:
            init_func = multigrid_grid_init;  // Assuming multigrid_init is defined elsewhere
            break;
        case GRID_TYPE_MAZE_LCG:
            init_func = maze_lcg_grid_init;  
            break;
        case GRID_TYPE_MAZE_MGRID:
            init_func = maze_multigrid_grid_init;  
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
    new->eps_s = eps;  // Dielectric constant of the solvent
    new->eps_int = eps_int;  // Dielectric constant inside the solute

    new->n_local = n;
    new->n_start = 0;

    new->y = NULL;
    new->q = NULL;
    new->phi_p = NULL;
    new->phi_n = NULL;
    new->ig2 = NULL;


    new->pb_enabled = 0;  // Poisson-Boltzmann not enabled by default
    new->nonpolar_enabled = 0; //nonpolar forces not enabled by default
    new->w = 0.0;  // Ionic boundary width
    new->kbar2 = 0.0;  // Screening factor

    new->k2 = NULL;  // Screening factor
    new->eps_x = NULL;  // Dielectric constant in x direction
    new->eps_y = NULL;  // Dielectric constant in y direction
    new->eps_z = NULL;  // Dielectric constant in z direction
    
    init_func(new);

    new->tol = tol;
    new->n_iters = 0;

    new->free = grid_free;

    return new;
}

void grid_pb_init(grid *grid, double w, double kbar2, int nonpolar_enabled) {
    // Initialize the grid for Poisson-Boltzmann simulations
    grid->pb_enabled = 1;  // Enable Poisson-Boltzmann
    grid->nonpolar_enabled = nonpolar_enabled; //nonpolar forces ON/OFF
    grid->w = w;
    grid->kbar2 = kbar2;

    // Initialize the solvent potential and dielectric constant arrays
    int n = grid->n;
    int n_local = grid->n_local;

    grid->eps_x = mpi_grid_allocate(n_local, n);
    grid->eps_y = mpi_grid_allocate(n_local, n);
    grid->eps_z = mpi_grid_allocate(n_local, n);
    grid->k2 = (double *)malloc(grid->size * sizeof(double));
}

void grid_pb_free(grid *grid) {
    if (grid->pb_enabled) {
        mpi_grid_free(grid->eps_x, grid->n);
        mpi_grid_free(grid->eps_y, grid->n);
        mpi_grid_free(grid->eps_z, grid->n);

        free(grid->k2);
    }
}

void smooth_charges_none(grid *grid, particles *p) {
    // No smoothing, return the original charges
}

void smooth_charges_diffusion(grid *grid, particles *p) {
    int n = grid->n;
    int n_loc = grid->n_local;
    int n_start = grid->n_start;

    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = n * n;
    long int size = grid->size;

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[n];
    int jnext[n];
    int kprev[n];
    int knext[n];
    for (int t = 0; t < n; ++t) {
        kprev[t] = ((t - 1 + n ) % n);
        knext[t] = ((t + 1      ) % n);
        jprev[t] = kprev[t] * n;
        jnext[t] = knext[t] * n;
    }

    double D = 1 / 6.2;  // Diffusion coefficient for a simple 3D diffusion process on a grid
    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units
    int num_steps = ceil(sigma * sigma / (2.0 * D)) + 1;

    double *u = grid->q;  // Input charge distribution
    double *u_new = (double *)malloc(size * sizeof(double));  // Temporary array for the new charge distribution
    vec_copy(u, u_new, size);  // Initialize the new charge distribution with the current values

    for (int step = 0; step < num_steps; step++) {
        // Exchange the top and bottom slices
        mpi_grid_exchange_bot_top(grid->q, n_loc, n);

        #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k1, k2)
        for (i = 0; i < n_loc; i++) {
            i0 = i * n2;
            i1 = i0 + n2;
            i2 = i0 - n2;
            for (j = 0; j < n; j++) {
                j0 = j * n;
                j1 = jnext[j];
                j2 = jprev[j];
                for (k = 0; k < n; k++) {
                    k1 = knext[k];
                    k2 = kprev[k];
                    u_new[i0 + j0 + k] += D * (
                        u[i1 + j0 + k] +
                        u[i2 + j0 + k] +
                        u[i0 + j1 + k] +
                        u[i0 + j2 + k] +
                        u[i0 + j0 + k1] +
                        u[i0 + j0 + k2] -
                        u[i0 + j0 + k] * 6.0
                    );
                }
            }
        }

        vec_copy(u_new, u, size);  // Copy the new charge distribution back to the original array
    }

    free(u_new);
}

void grid_smoothing_init(grid *grid, int method, double r_cut, double sigma) {
    grid->smoothing = method;
    grid->smoothing_rcut = r_cut;
    grid->smoothing_sigma = sigma;

    switch (grid->smoothing) {
        case SMOOTHING_TYPE_NONE:
            grid->smooth_charges = smooth_charges_none;
            break;
        case SMOOTHING_TYPE_GAUSS:
            // For now performed outside in theh python code
            grid->smooth_charges = smooth_charges_none;
            if (grid->smoothing_rcut <= 0.0 || grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for Gaussian smoothing:\n");
                mpi_fprintf(stderr, "r_cut: %f, sigma: %f\n", grid->smoothing_rcut, grid->smoothing_sigma);
                exit(1);
            }
            break;
        case SMOOTHING_TYPE_DIFFUSION:
            grid->smooth_charges = smooth_charges_diffusion;
            if (
                grid->smoothing_rcut <= 0.0 || grid->smoothing_sigma <= 0.0
            ) {
                mpi_fprintf(stderr, "Invalid parameters for diffusion-based smoothing:\n");
                mpi_fprintf(stderr, "r_cut: %f, sigma: %f\n", grid->smoothing_rcut, grid->smoothing_sigma);
                exit(1);
            }
            break;
        // Additional smoothing methods can be added here
        default:
            break;
    }

    // Additional initialization for smoothing can be added here if needed
}

void grid_smoothing_free(grid *grid) {
    // No dynamic memory allocated for smoothing, so nothing to free
}

void grid_free(grid *grid) {
    switch (grid->type) {
        case GRID_TYPE_LCG:
            lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_FFT:
            fft_grid_cleanup(grid);
            break;
        case GRID_TYPE_MGRID:
            multigrid_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_LCG:
            maze_lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_MGRID:
            maze_multigrid_grid_cleanup(grid);
            break;
        default:
            break;
    }

    grid_pb_free(grid);
    grid_smoothing_free(grid);

    free(grid);
}

void grid_update_eps_and_k2(grid *g, particles *p) {
    // Update the dielectric constant and screening factor based on the grid's transition regions
    int n = g->n;
    int n_local = g->n_local;
    int n_start = g->n_start;

    double h = g->h;
    double L = g->L;
    double w = g->w;

    double eps_s = g->eps_s;
    double eps_int = g->eps_int;
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
        eps_x[i] = (eps_s - eps_int);
        eps_y[i] = (eps_s - eps_int);
        eps_z[i] = (eps_s - eps_int);
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
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_z[idx_cen] = 0.0;
                    }
                }
            }
        }
    }
    for (long int i = 0; i < size; i++) {
        eps_x[i] += eps_int;  // Update x dielectric constant
        eps_y[i] += eps_int;  // Update y dielectric constant
        eps_z[i] += eps_int;  // Update z dielectric constant
    }
}    

/*Important, when called for IO must be called by all procs*/
double grid_get_energy_elec(grid *g){
    double energy = 0.0;

    #pragma omp parallel for reduction(+:energy)
    for (long int i = 0; i < g->size; i++) {
        // Calculate the change in energy due to the Poisson-Boltzmann potential
        energy += 0.5 * g->q[i] * g->phi_n[i];
    }

    allreduce_sum(&energy, 1);

    return energy;
}



