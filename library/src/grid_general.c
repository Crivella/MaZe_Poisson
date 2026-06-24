#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"
#include "fftw_wrap.h"

grid * grid_init(
    int n, double L, double h, double tol, double eps, double eps_int,
    grid_type grid_type, precond_type precond_type
) {
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
        case GRID_TYPE_SCAFACOS:
            init_func = scafacos_grid_init;  
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

#ifdef __MPI

void grid_init_mpi(grid *grid) {
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

void grid_init_mpi_fft(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    
    init_rfft(grid->n, &grid->n_local, &grid->n_start);

    int rank = mpid->rank;
    int size = mpid->size;
    int n_loc, n_start;

    mpid->n_loc = grid->n_local;
    mpid->n_start = grid->n_start;
    for (int i=0; i<size; i++) {
        n_loc = mpid->n_loc;
        n_start = mpid->n_start;
        MPI_Bcast(&n_loc, 1, MPI_INT, i, MPI_COMM_WORLD);
        MPI_Bcast(&n_start, 1, MPI_INT, i, MPI_COMM_WORLD);
        mpid->n_loc_list[i] = n_loc;
        mpid->n_start_list[i] = n_start;
        // printf("FFT MPI(%d %d): n_local = %d, n_start = %d\n", rank, i, n_loc, n_start);
    }
    // Check that if some processors have no local grid points they should be skipped
    // from the loop communication
    if (rank < size-1) {
        if (mpid->n_loc_list[rank+1] == 0) {
            mpid->next_rank = 0;
        } 
    }
    if (rank == 0) {
        if (mpid->n_loc_list[size-1] == 0) {
            for (int i=size-1; i>=0; i--) {
                if (mpid->n_loc_list[i] > 0) {
                    mpid->prev_rank = i;
                    break;
                }
            }
        }
    }
}
#else  // __MPI

void grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}

void grid_init_mpi_fft(grid *grid) {
    mpi_data *mpid = get_mpi_data();

    init_rfft(grid->n, &grid->n_local, &grid->n_start);

    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}

#endif  // __MPI

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

void smooth_charges_none(grid *grid) {
    // No smoothing, return the original charges
}

/*
Allocate and initialize the Gaussian smoothing kernel in Fourier space.
The kernel is generated in real space as a 3D Gaussian function, normalized, and then transformed
to Fourier space using the forward FFT.
The resulting Fourier-space kernel is stored in the grid structure for later use in smoothing the charge distribution.
*/
void smooth_charges_gauss_init(grid *grid) {
    int n = grid->n;
    int nh = n / 2 + 1;
    long int n2 = n * n;
    long int c_size = grid->n_local * nh * n;  // Size of the complex-space grid for the local portion
    long int r_size = grid->n_local * n2;  // Size of the real-space grid for the local portion

    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units
    double sigma2 = sigma * sigma;

    double *gaussian_kernel = (double *)calloc(r_size, sizeof(double));

    // Generate the Gaussian kernel in Real space
    int i, j, k;
    int di, dj, dk;
    double ri, rj, r2;
    for (int i_loc = 0; i_loc < grid->n_local; i_loc++) {
        i = grid->n_start + i_loc;
        di = i > n / 2 ? i - n : i;  // Wrap around for periodicity
        ri = di * di;
        for (j = 0; j < n; j++) {
            dj = j > n / 2 ? j - n : j;  // Wrap around for periodicity
            rj = ri + dj * dj;
            for (k = 0; k < n; k++) {
                dk = k > n / 2 ? k - n : k;  // Wrap around for periodicity
                r2 = rj + dk * dk;
                gaussian_kernel[i_loc * n2 + j * n + k] = exp(-(double)r2 / (2 * sigma2));
            }
        }
    }

    // Normalize the kernel
    double sum = 0.0;
    for (i = 0; i < r_size; i++) {
        sum += gaussian_kernel[i];
    }
    allreduce_sum(&sum, 1);
    dscal(gaussian_kernel, 1.0 / sum, r_size);  // Normalize so that the sum of the kernel is 1
    dscal(gaussian_kernel, 1.0 / pow(n, 3), r_size);  // FFT normalization factor

    // Convert the kernel in Fourier space
    grid->smoothing_kernel = malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, grid->n_local, gaussian_kernel, (fftw_complex *)grid->smoothing_kernel);

    free(gaussian_kernel);
}

void smooth_charges_gauss(grid *grid) {
    // Apply Gaussian smoothing to the charge distribution using convolution in Fourier space
    int n = grid->n;
    int nh = n / 2 + 1;
    int n_loc = grid->n_local;
    int n_start = grid->n_start;
    long int n2 = n * n;
    long int c_size = n_loc * n * nh;  // Size of the complex-space grid for the local portion

    double *q = grid->q;  // Original charge distribution

    // Perform forward FFT on the original charge distribution
    fftw_complex *q_fft = (fftw_complex *)malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, n_loc, q, q_fft);

    // Convolve in Fourier space (element-wise multiplication)
    fftw_complex *kernel_fft = (fftw_complex *)grid->smoothing_kernel;
    #pragma omp parallel for
    for (long int i = 0; i < c_size; i++) {
        q_fft[i] *= kernel_fft[i];
    }

    // Perform inverse FFT to get the smoothed charge distribution
    irfft_3d(n, n_loc, q_fft, q);

    free(q_fft);
}

void smooth_charges_diffusion(grid *grid) {
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
    grid->smoothing_kernel = NULL;  // Initialize the smoothing kernel to NULL

    switch (grid->smoothing) {
        case SMOOTHING_TYPE_NONE:
            grid->smooth_charges = smooth_charges_none;
            break;
        case SMOOTHING_TYPE_GAUSS:
            // For now performed outside in theh python code
            smooth_charges_gauss_init(grid);  // Initialize the Gaussian smoothing kernel if needed
            grid->smooth_charges = smooth_charges_gauss;
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
    if (grid->smoothing_kernel != NULL) {
        free(grid->smoothing_kernel);
        grid->smoothing_kernel = NULL;
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
        case GRID_TYPE_MGRID:
            multigrid_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_LCG:
            maze_lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_MGRID:
            maze_multigrid_grid_cleanup(grid);
            break;
        case GRID_TYPE_SCAFACOS:
            scafacos_grid_cleanup(grid);
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

    if (g->type == GRID_TYPE_SCAFACOS) {
        // mpi_fprintf(stderr, "Warning: grid_get_energy_elec needs testing with ScaFacos.\n");
        return 0.0;
        for (long int i = 0; i < g->n_p; i++) {
            energy += g->fcs_potential[i];
            // energy += 0.5 * g->fcs_potential[i];
            // energy += 0.5 * g->phi_n[i] * g->fcs_charges[i];
            // g->fcs_potential[i] * g->fcs_charges[i]
        }

        allreduce_sum(&energy, 1);

        return energy;
    }

    #pragma omp parallel for reduction(+:energy)
    for (long int i = 0; i < g->size; i++) {
        // Calculate the change in energy due to the Poisson-Boltzmann potential
        energy += 0.5 * g->q[i] * g->phi_n[i];
    }

    allreduce_sum(&energy, 1);

    return energy;
}



