#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define KB 3.1668 * 1e-6 // E_h/K
#define a0 0.529177210903 // Example constant, replace with actual value

// Constants and utilities
#define a0 0.529177 // Example constant, replace with actual value
typedef struct Grid {
    int N_p;
    float tol;
    bool debug;
    int N;
    double L;
    double h;
    double *phi_prev;
    double *phi;
    int *q;
    double temperature;
    double q_tot;
    Particles* particles;
    bool potential_notelec;
} Grid;

typedef struct MD_Variables {
    char* potential;
} MD_Variables;

// Particle structure
typedef struct {
    Grid* grid;
    int N_p;
    double* masses;
    double* positions;  // Shape: (N_p, 3)
    double* velocities; // Shape: (N_p, 3)
    double* charges;
    double* forces;     // Shape: (N_p, 3)
    double* forces_notelec;
    char* potential_info;
    double r_cutoff;
    double B;
    double sigma;
    double epsilon;
    double* neighbors;  // Shape: (N_p, 8, 3)
    // Additional TF parameters
    double* A;
    double* C;
    double* D;
    double* sigma_TF;
    double alpha;
    double beta;
    void (*ComputeForceNotElec)(void* self); // Function pointer
} Particles;

void RescaleVelocities(Grid* grid) {
    // Initialize velocity accumulators
    double init_vel_Na[3] = {0.0, 0.0, 0.0};
    double new_vel_Na[3] = {0.0, 0.0, 0.0};
    double init_vel_Cl[3] = {0.0, 0.0, 0.0};
    double new_vel_Cl[3] = {0.0, 0.0, 0.0};

    // Sum initial velocities for Na and Cl
    for (int i = 0; i < grid->N_p; i++) {
        if (grid->particles->charges[i] == 1.0) {
            for (int j = 0; j < 3; j++) {
                init_vel_Na[j] += grid->particles->velocities[i * 3 + j];
            }
        } else if (grid->particles->charges[i] == -1.0) {
            for (int j = 0; j < 3; j++) {
                init_vel_Cl[j] += grid->particles->velocities[i * 3 + j];
            }
        }
    }

    // Compute initial temperature
    double mi_vi2_sum = 0.0;
    for (int i = 0; i < grid->N_p; i++) {
        double vi2 = 0.0;
        for (int j = 0; j < 3; j++) {
            vi2 += grid->particles->velocities[i * 3 + j] * grid->particles->velocities[i * 3 + j];
        }
        mi_vi2_sum += grid->particles->masses[i] * vi2;
    }
    grid->temperature = mi_vi2_sum / (3.0 * grid->N_p * KB);

    printf("Total initial velocities:\nNa = [%f, %f, %f] \nCl = [%f, %f, %f]\nOld T = %f\n",
           init_vel_Na[0], init_vel_Na[1], init_vel_Na[2],
           init_vel_Cl[0], init_vel_Cl[1], init_vel_Cl[2],
           grid->temperature);

    // Rescale velocities
    for (int i = 0; i < grid->N_p; i++) {
        if (grid->particles->charges[i] == 1.0) {
            for (int j = 0; j < 3; j++) {
                grid->particles->velocities[i * 3 + j] -= 2.0 * init_vel_Na[j] / grid->N_p;
            }
        } else if (grid->particles->charges[i] == -1.0) {
            for (int j = 0; j < 3; j++) {
                grid->particles->velocities[i * 3 + j] -= 2.0 * init_vel_Cl[j] / grid->N_p;
            }
        }
    }

    // Sum rescaled velocities for Na and Cl
    for (int i = 0; i < grid->N_p; i++) {
        if (grid->particles->charges[i] == 1.0) {
            for (int j = 0; j < 3; j++) {
                new_vel_Na[j] += grid->particles->velocities[i * 3 + j];
            }
        } else if (grid->particles->charges[i] == -1.0) {
            for (int j = 0; j < 3; j++) {
                new_vel_Cl[j] += grid->particles->velocities[i * 3 + j];
            }
        }
    }

    // Compute new temperature
    mi_vi2_sum = 0.0;
    for (int i = 0; i < grid->N_p; i++) {
        double vi2 = 0.0;
        for (int j = 0; j < 3; j++) {
            vi2 += grid->particles->velocities[i * 3 + j] * grid->particles->velocities[i * 3 + j];
        }
        mi_vi2_sum += grid->particles->masses[i] * vi2;
    }
    grid->temperature = mi_vi2_sum / (3.0 * grid->N_p * KB);

    printf("Total scaled velocities:\nNa = [%f, %f, %f] \nCl = [%f, %f, %f]\nNew T = %f\n",
           new_vel_Na[0], new_vel_Na[1], new_vel_Na[2],
           new_vel_Cl[0], new_vel_Cl[1], new_vel_Cl[2],
           grid->temperature);
}

// Function to compute kinetic energy
void Energy(Grid* grid, int iter, int print_energy, FILE* file_output_energy) {
    double kinetic = 0.0;

    // Compute kinetic energy
    for (int i = 0; i < grid->N_p; i++) {
        double vi2 = 0.0;
        for (int j = 0; j < 3; j++) {
            vi2 += grid->particles->velocities[i * 3 + j] * grid->particles->velocities[i * 3 + j];
        }
        kinetic += 0.5 * grid->particles->masses[i] * vi2;
    }

    // Print or save energy values if required
    if (print_energy) {
        fprintf(file_output_energy, "%d,%f,%f\n", iter, kinetic, grid->potential_notelec);
    }
}

// Function to compute temperature
void Temperature(Grid* grid, int iter, int print_temperature, FILE* file_output_temperature) {
    double mi_vi2_sum = 0.0;

    // Compute sum of m * v^2
    for (int i = 0; i < grid->N_p; i++) {
        double vi2 = 0.0;
        for (int j = 0; j < 3; j++) {
            vi2 += grid->particles->velocities[i * 3 + j] * grid->particles->velocities[i * 3 + j];
        }
        mi_vi2_sum += grid->particles->masses[i] * vi2;
    }

    // Compute temperature
    grid->temperature = mi_vi2_sum / (3.0 * grid->N_p * KB);

    // Print or save temperature values if required
    if (print_temperature) {
        fprintf(file_output_temperature, "%d,%f\n", iter, grid->temperature);
    }
}


// Function to set charges
void SetCharges(Particles *particles) {
    Grid *grid = particles->grid;
    double L = grid->L;
    double h = grid->h;

    // Allocate memory for q in the grid and initialize to zero
    size_t q_size = grid->N * grid->N * grid->N;
    grid->q = (double *)calloc(q_size, sizeof(double));
    if (!grid->q) {
        perror("Memory allocation failed for grid->q");
        exit(EXIT_FAILURE);
    }

    // Compute updates and indices
    for (int i = 0; i < particles->N_p; ++i) {
        double *pos = &particles->positions[i * 3];
        for (int j = 0; j < 8; ++j) { // Assumes 8 neighbors per particle
            double *neighbor = &particles->neighbors[(i * 8 + j) * 3];
            double diff[3], product = 1.0;

            for (int k = 0; k < 3; ++k) {
                diff[k] = pos[k] - neighbor[k] * h;
                product *= g(diff[k], L, h);
            }

            int x_idx = (int)(neighbor[0]);
            int y_idx = (int)(neighbor[1]);
            int z_idx = (int)(neighbor[2]);
            int index = x_idx * grid->N * grid->N +
                        y_idx * grid->N + z_idx;

            grid->q[index] += particles->charges[i] * product;
        }
    }

    // Validate charge preservation
    double q_tot_expected = 0.0, q_tot = 0.0;
    for (int i = 0; i < particles->N_p; ++i) {
        q_tot_expected += particles->charges[i];
    }
    for (size_t i = 0; i < q_size; ++i) {
        q_tot += grid->q[i];
    }

    if (q_tot + 1e-6 < q_tot_expected) {
        fprintf(stderr, "Error: charge is not preserved. q_tot = %f\n", q_tot);
        free(grid->q);
        exit(EXIT_FAILURE);
    }
}

// ComputeForce_FD function and its associated other functions
void ComputeForce_FD(Particles *self, bool prev) {
    double h = self->grid->h;

    // Choose the appropriate potential
    double ***phi_v = prev ? self->grid->phi_prev : self->grid->phi;

    int nx = self->grid->N;
    int ny = self->grid->N;
    int nz = self->grid->N;
    int n_particles = self->N_p;

    // Allocate memory for electric field components
    double ***E_x = Allocate3DArray(nx, ny, nz);
    double ***E_y = Allocate3DArray(nx, ny, nz);
    double ***E_z = Allocate3DArray(nx, ny, nz);

    // Compute electric field components (central difference approximation)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                E_x[i][j][k] = (phi_v[i + 1][j][k] - phi_v[i - 1][j][k]) / (2 * h);
                E_y[i][j][k] = (phi_v[i][j + 1][k] - phi_v[i][j - 1][k]) / (2 * h);
                E_z[i][j][k] = (phi_v[i][j][k + 1] - phi_v[i][j][k - 1]) / (2 * h);
            }
        }
    }

    // Allocate memory for forces
    self->forces = (double *)calloc(n_particles * 3, sizeof(double)); // Linearized forces array

    // Compute forces for each particle based on neighbors
    for (int p = 0; p < n_particles; ++p) {
        for (int n = 0; n < 8; ++n) { // Loop over 8 neighboring grid points
            int x = (int)self->neighbors[p * 8 + n * 3 + 0];
            int y = (int)self->neighbors[p * 8 + n * 3 + 1];
            int z = (int)self->neighbors[p * 8 + n * 3 + 2];

            if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) { // Ensure indices are within bounds
                double q_neighbor = self->grid->q[x * ny * nz + y * nz + z];

                self->forces[p * 3 + 0] -= q_neighbor * E_x[x][y][z];
                self->forces[p * 3 + 1] -= q_neighbor * E_y[x][y][z];
                self->forces[p * 3 + 2] -= q_neighbor * E_z[x][y][z];
            }
        }
    }

    // Deallocate electric field arrays
    Free3DArray(E_x, nx, ny);
    Free3DArray(E_y, nx, ny);
    Free3DArray(E_z, nx, ny);
}

double ***Allocate3DArray(int nx, int ny, int nz) {
    double ***array = (double ***)malloc(nx * sizeof(double **));
    for (int i = 0; i < nx; ++i) {
        array[i] = (double **)malloc(ny * sizeof(double *));
        for (int j = 0; j < ny; ++j) {
            array[i][j] = (double *)malloc(nz * sizeof(double));
        }
    }
    return array;
}

void Free3DArray(double ***array, int nx, int ny) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

// VerletPoisson function
void VerletPoisson(Grid* grid, double* y, int size, double* y_new, int* iter_conv) {
    double tol = grid->tol;
    double h = grid->h;
    double* phi_temp = (double*)malloc(size * sizeof(double));
    double* sigma_p = (double*)malloc(size * sizeof(double));
    double* matrix_mult_result = (double*)malloc(size * sizeof(double));
    
    // Compute provisional update for the field phi
    for (int i = 0; i < size; i++) {
        phi_temp[i] = grid->phi[i];
        grid->phi[i] = 2 * grid->phi[i] - grid->phi_prev[i];
        grid->phi_prev[i] = phi_temp[i];
    }

    // Compute the constraint with the provisional value of the field phi
    MatrixVectorProduct(grid->phi, matrix_mult_result, size);
    for (int i = 0; i < size; i++) {
        sigma_p[i] = grid->q[i] / h + matrix_mult_result[i] / (4 * M_PI);
    }

    // Apply LCG
    *iter_conv = PrecondLinearConjGradPoisson(sigma_p, y_new, size, tol);

    // Scale the field with the constrained 'force' term
    for (int i = 0; i < size; i++) {
        grid->phi[i] -= y_new[i] * (4 * M_PI);
    }

    if (grid->debug) {
        double* debug_matrix_mult_result = (double*)malloc(size * sizeof(double));
        MatrixVectorProduct(y_new, debug_matrix_mult_result, size);
        
        double max_precision = 0.0;
        for (int i = 0; i < size; i++) {
            double diff = fabs(debug_matrix_mult_result[i] - sigma_p[i]);
            if (diff > max_precision) {
                max_precision = diff;
            }
        }
        printf("LCG precision     : %f\n", max_precision);

        MatrixVectorProduct(grid->phi, debug_matrix_mult_result, size);
        double max_constraint = 0.0;
        for (int i = 0; i < size; i++) {
            double constraint = fabs(grid->q[i] / h + debug_matrix_mult_result[i] / (4 * M_PI));
            if (constraint > max_constraint) {
                max_constraint = constraint;
            }
        }
        printf("max of constraint: %f\n", max_constraint);

        free(debug_matrix_mult_result);
    }

    // Free temporary arrays
    free(phi_temp);
    free(sigma_p);
    free(matrix_mult_result);
}
