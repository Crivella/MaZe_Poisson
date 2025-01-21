#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

// Constants (Define your constants here, e.g., a0, conv_mass, kB)
#define MAX_LINE_LENGTH 1024  // Maximum length of a line in the CSV file
#define amu_to_kg  1.66054 * 1e-27 // conversion 
#define m_e  9.1093837 * 1e-31 // kg
#define conv_mass amu_to_kg / m_e
#define KB 3.1668 * 1e-6 // E_h/K
#define a0 0.529177210903 // Example constant, replace with actual value

// Struct to hold parameters for TF
typedef struct {
    double A;
    double C;
    double D;
    double sigma_TF;
} Parameters;

// Struct to represent a dictionary-like structure
typedef struct {
    int key;
    Parameters* values;
} DictTFEntry;

// Function to get the dictionary equivalent in C
DictTFEntry* GetDictTF() {
    // Allocate memory for the dictionary (3 entries)
    int size = 3;
    DictTFEntry* dictTF = (DictTFEntry*)malloc(size * sizeof(DictTFEntry));
    if (dictTF == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("memory successfully assigned\n");
    }

    // Charge keys
    int charge_totNaCl = 0;
    int charge_totNaNa = 2;
    int charge_totClCl = -2;

    // NaCl parameters
    dictTF[0].key = charge_totNaCl;
    dictTF[0].values = (Parameters*)malloc(sizeof(Parameters));
    dictTF[0].values->A = 7.7527 * 1e-3;
    dictTF[0].values->C = 0.2569 / (a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[0].values->D = 0.3188 / (a0 * a0 * a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[0].values->sigma_TF = 2.755 / a0;

    // NaNa parameters
    dictTF[1].key = charge_totNaNa;
    dictTF[1].values = (Parameters*)malloc(sizeof(Parameters));
    dictTF[1].values->A = 9.6909 * 1e-3;
    dictTF[1].values->C = 0.0385 / (a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[1].values->D = 0.0183 / (a0 * a0 * a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[1].values->sigma_TF = 2.340 / a0;

    // ClCl parameters
    dictTF[2].key = charge_totClCl;
    dictTF[2].values = (Parameters*)malloc(sizeof(Parameters));
    dictTF[2].values->A = 5.8145 * 1e-3;
    dictTF[2].values->C = 2.6607 / (a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[2].values->D = 5.3443 / (a0 * a0 * a0 * a0 * a0 * a0 * a0 * a0);
    dictTF[2].values->sigma_TF = 3.170 / a0;

    return dictTF;
}

// Function to free the dictionary
void FreeDictTF(DictTFEntry* dictTF) {
    for (int i = 0; i < 3; i++) {
        free(dictTF[i].values);
    }
    free(dictTF);
}

struct Particles;
// Struct for Grid
typedef struct Grid {
    int N;
    int N_tot;
    int N_p;
    double h;
    double q_tot;
    double L;
    double dt;
    double elec;
    double not_elec;
    double T;
    double kBT;
    double* offset_update; // 7x3 matrix as flat array
    double* q;             // Charge grid
    double* phi;           // Electrostatic field grid
    double* phi_prev;      // Previous electrostatic field grid
    double temperature;
    double potential_notelec;
    struct Particles* particles;  // Pointer to Particles struct
} Grid;

// Initialization function for Grid
Grid* initGrid(int N, int N_p, double L, double dt, double elec, double not_elec, double T) {
    Grid* grid = (Grid*)malloc(sizeof(Grid));
    grid->N = N;
    grid->N_tot = N*N*N;
    grid->N_p = N_p;
    grid->h = L/N;
    grid->L = L;
    grid->dt = dt;
    grid->elec = elec;
    grid->not_elec = not_elec;
    grid->T = T;
    grid->kBT = KB * grid->T;

    // Allocate memory for arrays
    grid->q = (double*)calloc(N * N * N, sizeof(double));
    grid->phi = (double*)calloc(N * N * N, sizeof(double));
    grid->phi_prev = (double*)calloc(N * N * N, sizeof(double));
    grid->offset_update = (double*)calloc(7 * 3, sizeof(double));

    return grid;
}

// Function to compute the weight, as described in Im et al. (1998) - eqn 24
double g(double x, double L, double h) {
    x = x - L * round(x / L);
    x = fabs(x);
    return (x < h) ? 1.0 - x / h : 0.0;
}

// Returns the smallest distance between two neighboring particles, enforcing PBC
double BoxScale(double diff, double L) {
    return diff - L * round(diff / L);
}

// Free memory allocated for Grid
void freeGrid(Grid* grid) {
    free(grid->q);
    free(grid->phi);
    free(grid->phi_prev);
    free(grid->offset_update);
    free(grid);
}

// // NEEDS TO BE SEND WHEN GRID IS CALLED
// typedef struct MD_Variables {
//     char* potential;
// } MD_Variables;

// Particle structure
typedef struct Particles {
    Grid* grid; // pointer to grid structure
    DictTFEntry* tf_params;
    int N_p;
    double* masses;
    double** positions;  // Shape: (N_p, 3)
    double* velocities; // Shape: (N_p, 3)
    double* charges;
    double* forces;     // Shape: (N_p, 3)
    double* forces_notelec;
    char potential_info;
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
    void (*ComputeForceNotElec)(struct Particles *); // Function pointer
} Particles;

void ComputeTFForces(Particles *self) {
    int N_p = self->N_p;
    double L = self->grid->L;
    double r_cutoff = self->r_cutoff;
    double B = self->B;

    // Temporary arrays to hold intermediate calculations
    double *r_diff = (double *)malloc(N_p * N_p * 3 * sizeof(double));
    double *r_mag = (double *)malloc(N_p * N_p * sizeof(double));
    double *f_mag = (double *)calloc(N_p * N_p, sizeof(double));
    double *V_mag = (double *)calloc(N_p * N_p, sizeof(double));
    double (*r_cap)[3] = (double (*)[3])malloc(N_p * N_p * 3 * sizeof(double));
    double (*pairwise_forces)[3] = (double (*)[3])malloc(N_p * N_p * 3 * sizeof(double));
    double *net_forces = (double *)calloc(N_p * 3, sizeof(double));

    if (!r_diff || !r_mag || !f_mag || !V_mag || !r_cap || !pairwise_forces || !net_forces) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Compute all pairwise differences and apply BoxScale
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            for (int k = 0; k < 3; ++k) {
                double diff = self->positions[i * 3 + k] - self->positions[j * 3 + k];
                r_diff[(i * N_p + j) * 3 + k] = BoxScale(diff, L);
            }
        }
    }

    // Compute pairwise distances and unit vectors
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            double sum_sq = 0.0;
            for (int k = 0; k < 3; ++k) {
                double val = r_diff[(i * N_p + j) * 3 + k];
                sum_sq += val * val;
            }
            if (i == j) {
                r_mag[i * N_p + j] = INFINITY;
            } else {
                r_mag[i * N_p + j] = sqrt(sum_sq);
                for (int k = 0; k < 3; ++k) {
                    r_cap[i * N_p + j][k] = r_diff[(i * N_p + j) * 3 + k] / r_mag[i * N_p + j];
                }
            }
        }
    }

    // Apply cutoff mask and compute forces and potentials
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            double r = r_mag[i * N_p + j];
            if (r <= r_cutoff) {
                double exp_term = exp(B * (self->sigma_TF[i] - r));
                f_mag[i * N_p + j] = B * self->A[i] * exp_term
                                     - 6.0 * self->C[i] / pow(r, 7)
                                     - 8.0 * self->D[i] / pow(r, 9)
                                     - self->alpha;
                V_mag[i * N_p + j] = self->A[i] * exp_term
                                     - self->C[i] / pow(r, 6)
                                     - self->D[i] / pow(r, 8)
                                     + self->alpha * r
                                     + self->beta;
            }
        }
    }

    // Compute pairwise forces
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            for (int k = 0; k < 3; ++k) {
                pairwise_forces[i * N_p + j][k] = f_mag[i * N_p + j] * r_cap[i * N_p + j][k];
            }
        }
    }

    // Sum forces to get net forces
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            for (int k = 0; k < 3; ++k) {
                net_forces[i * 3 + k] += pairwise_forces[i * N_p + j][k];
            }
        }
    }

    // Store net forces and potential energy
    memcpy(self->forces_notelec, net_forces, N_p * 3 * sizeof(double));

    double potential_energy = 0.0;
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            potential_energy += V_mag[i * N_p + j];
        }
    }
    self->grid->potential_notelec = potential_energy / 2.0;

    // Free temporary arrays
    free(r_diff);
    free(r_mag);
    free(f_mag);
    free(V_mag);
    free(r_cap);
    free(pairwise_forces);
    free(net_forces);
}


void ComputePairwiseTFParameters(Particles *self, DictTFEntry *tf_params, int tf_params_size) {
    int N_p = self->N_p;
    double r_cutoff = self->r_cutoff;

    // Allocate memory for pairwise TF parameters
    self->A = (double *)malloc(N_p * N_p * sizeof(double));
    self->C = (double *)malloc(N_p * N_p * sizeof(double));
    self->D = (double *)malloc(N_p * N_p * sizeof(double));
    self->sigma_TF = (double *)malloc(N_p * N_p * sizeof(double));

    if (!self->A || !self->C || !self->D || !self->sigma_TF) {
        perror("Memory allocation failed for TF parameters");
        exit(EXIT_FAILURE);
    }

    // Initialize beta
    self->beta = 0.0;

    // Compute pairwise parameters
    for (int i = 0; i < N_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            double charge_sum = self->charges[i] + self->charges[j];

            // Lookup TF parameters based on charge sum
            Parameters *params = NULL;
            for (int k = 0; k < tf_params_size; ++k) {
                if (tf_params[k].key == (int)charge_sum) {
                    params = tf_params[k].values;
                    break;
                }
            }

            if (params == NULL) {
                fprintf(stderr, "Error: No TF parameters found for charge sum %.2f\n", charge_sum);
                exit(EXIT_FAILURE);
            }

            double A = params->A;
            double C = params->C;
            double D = params->D;
            double sigma_TF = params->sigma_TF;

            self->A[i * N_p + j] = A;
            self->C[i * N_p + j] = C;
            self->D[i * N_p + j] = D;
            self->sigma_TF[i * N_p + j] = sigma_TF;

            // Compute V_shift and alpha
            double V_shift = A * exp(self->B * (sigma_TF - r_cutoff)) 
                             - C / pow(r_cutoff, 6) 
                             - D / pow(r_cutoff, 8);
            double alpha = A * self->B * exp(self->B * (sigma_TF - r_cutoff)) 
                           - 6 * C / pow(r_cutoff, 7) 
                           - 8 * D / pow(r_cutoff, 9);

            // Update beta (average across all pair interactions)
            self->beta += -V_shift - alpha * r_cutoff;
        }
    }

    // Normalize beta by the number of interactions
    self->beta /= (N_p * N_p);

    free(self->A);
    free(self->C);
    free(self->D);
    free(self->sigma_TF);
}

// Constructor for Particles
Particles* initParticles(Grid* grid, DictTFEntry* tf_params, double* charges, double* masses, double** positions, char potential) {
    Particles* p = (Particles*)malloc(sizeof(Particles));
    p->grid = grid;
    p->N_p = grid->N_p;

    // Allocate and initialize arrays
    p->masses = (double*)malloc(p->N_p * sizeof(double));
    p->positions = (double**)malloc(p->N_p * 3 * sizeof(double));
    p->velocities = (double*)calloc(p->N_p * 3, sizeof(double));
    p->charges = (double*)malloc(p->N_p * sizeof(double));
    p->forces = (double*)calloc(p->N_p * 3, sizeof(double));
    p->forces_notelec = (double*)calloc(p->N_p * 3, sizeof(double));
    p->potential_info = potential; // should be gotten from MD_variables as an input

    // Copy provided data
    for (int i = 0; i < p->N_p; i++) {
        p->masses[i] = masses[i];
        p->charges[i] = charges[i];
        for (int j = 0; j < 3; j++) {
            p->positions[i][j] = positions[i][j];
        }
    }

    // Potential-specific initialization
    if (p->potential_info == 'T') {
        p->r_cutoff = 0.5 * grid->L;
        p->B = 3.1546 * a0;
        // Allocating memory for variables
        p->A = (double*)malloc(p->N_p * p->N_p * sizeof(double));
        p->C = (double*)malloc(p->N_p * p->N_p * sizeof(double));
        p->D = (double*)malloc(p->N_p * p->N_p * sizeof(double));
        p->sigma_TF = (double*)malloc(p->N_p * p->N_p * sizeof(double));
        p->tf_params = tf_params; 
        p->ComputeForceNotElec = ComputeTFForces; // Assigning the function

    } else if (p->potential_info== 'L') {
        p->sigma = 3.00512 * 2 / a0;
        p->epsilon = 5.48 * 1e-4;
        p->r_cutoff = 2.5 * p->sigma;
        //p->ComputeForceNotElec = NULL; // Assign actual function later
    }

    p->neighbors = (double*)malloc(p->N_p * 8 * 3); // allocating memory for neighbours
    ComputePairwiseTFParameters(p, tf_params, 3); // computing A, C, D params from particles

    return p;
}

// Destructor for Particles
void FreeParticles(Particles* p) {
    free(p->masses);
    free(p->positions);
    free(p->velocities);
    free(p->charges);
    free(p->forces);
    free(p->forces_notelec);
    free(p->neighbors);
    if (p->A) free(p->A);
    if (p->C) free(p->C);
    if (p->D) free(p->D);
    if (p->sigma_TF) free(p->sigma_TF);
    free(p);
}

char *trim_whitespace(char *str) {
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str)) str++;

    // Trim trailing space
    if (*str == 0) return str; // All spaces

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end + 1) = '\0';

    return str;
}

double *read_column_from_csv(const char *filename, int column_index, int num_rows) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    char *token;
    int current_column;
    int row_count = 0;
    int capacity = num_rows > 0 ? num_rows : 100; // Initial capacity

    // Allocate memory for the array
    double *column_data = (double *)malloc(capacity * sizeof(double));
    if (!column_data) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    // Skip the header row
    if (fgets(line, sizeof(line), file) == NULL) {
        perror("Error reading header row");
        free(column_data);
        fclose(file);
        return NULL;
    }

    // Read CSV data
    while (fgets(line, sizeof(line), file)) {
        current_column = 0;
        token = strtok(line, ",");
        while (token) {
            if (current_column == column_index) {
                if (row_count >= capacity) {
                    capacity *= 2;
                    double *new_data = (double *)realloc(column_data, capacity * sizeof(double));
                    if (!new_data) {
                        perror("Error reallocating memory");
                        free(column_data);
                        fclose(file);
                        return NULL;
                    }
                    column_data = new_data;
                }

                // Trim whitespace and parse the value
                token = trim_whitespace(token);
                char *endptr;
                double value = strtod(token, &endptr);
                if (*endptr != '\0') { // Invalid number
                    fprintf(stderr, "Invalid number in CSV: '%s'\n", token);
                    continue;
                }
                column_data[row_count++] = value;
                break;
            }
            token = strtok(NULL, ",");
            current_column++;
        }
    }

    fclose(file);

    // Adjust size if necessary
    if (row_count < capacity) {
        double *shrinked_data = (double *)realloc(column_data, row_count * sizeof(double));
        if (shrinked_data) {
            column_data = shrinked_data;
        }
    }

    return column_data; // The caller is responsible for managing memory
}

int main() {
    printf("Starting int main\n");

    // Initialize the dictionary
    DictTFEntry* tf_params = GetDictTF();
    printf("initialzed tf_params");

    // Grid and variables
    int N_p = 128;
    Grid* grid = initGrid(80, N_p, 16.51, 0.25, 1.0, 1.0, 1500); // 1.0 is True
    int rows = N_p + 1;
    const char *filename = "examples/input_files/input_coord128.csv";
    char potential = 'T';

    // Read CSV columns
    double* charges = read_column_from_csv(filename, 0, rows);
    double* masses = read_column_from_csv(filename, 1, rows);
    double* positions_x = read_column_from_csv(filename, 3, rows);
    double* positions_y = read_column_from_csv(filename, 4, rows);
    double* positions_z = read_column_from_csv(filename, 5, rows);

    if (!charges || !masses || !positions_x || !positions_y || !positions_z) {
        fprintf(stderr, "Error: Failed to read CSV data.\n");
        return EXIT_FAILURE;
    }

    // Allocate memory for positions
    double **positions = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        positions[i] = (double *)malloc(3 * sizeof(double));
        positions[i][0] = positions_x[i] / a0;
        positions[i][1] = positions_y[i] / a0;
        positions[i][2] = positions_z[i] / a0;
    }

    // Initialize Particles
    Particles* particles = initParticles(grid, tf_params, charges, masses, positions, potential);
    grid->particles = particles;

    // Clean up memory
    for (int i = 0; i < rows; i++) {
        free(positions[i]);
    }
    free(positions);
    free(charges);
    free(masses);
    free(positions_x);
    free(positions_y);
    free(positions_z);
    freeGrid(grid);
    FreeDictTF(tf_params);
    FreeParticles(particles);

    return 0;
}
