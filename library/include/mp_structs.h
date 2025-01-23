#ifndef __MP_STRUCTS_H
#define __MP_STRUCTS_H

#define GRID_TYPE_LCG 0
#define GRID_TYPE_FFT 1

// Struct typedefs
typedef struct lcg_grid lcg_grid;
typedef struct fft_grid fft_grid;

typedef struct particles particles;

// Struct function definitions
lcg_grid * lcg_grid_init(int n, double L, double h, double tol);
fft_grid * fft_grid_init(int n, double L, double h);
particles * particles_init(int n, int n_p, double L, double h, int type);

void * lcg_grid_free(lcg_grid *grid);
void * fft_grid_free(fft_grid *grid);
void * particles_free(particles *p);

void * lcg_grid_init_field(lcg_grid *grid);
void * fft_grid_init_field(fft_grid *grid);

int lcg_grid_update_field(lcg_grid *grid);
int fft_grid_update_field(fft_grid *grid);

double lcg_grid_update_charges(lcg_grid *grid, particles *p);
double fft_grid_update_charges(fft_grid *grid, particles *p);

void * particles_init_potential(particles *p, int pot_type);
void * particles_init_potential_tf(particles *p);
void * particles_init_potential_ld(particles *p);
void * particles_update_nearest_neighbors(particles *p);
double particles_compute_forces_field(particles *p, void *grid);
double particles_compute_forces_tf(particles *p);
double particles_compute_forces_ld(particles *p);
void * particles_compute_forces(particles *p, void *grid);
double particles_get_temperature(particles *p);
double particles_get_kinetic_energy(particles *p);
void * particles_get_momentum(particles *p, double *out);
void * particles_rescale_velocities(particles *p);

// Struct definitions
struct lcg_grid {
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing

    int n_local; // X - Number of grid points per dimension (MPI aware)

    double *y;  // Intermediate field constraint
    double *q;  // Charge density
    double *phi_p;  // Previous potential
    double *phi_n;  // Last potential

    double tol;  // Tolerance for the LCG
    long int n_iters;  // Number of iterations for convergence of the LCG

    void *      (*free)( lcg_grid *);
    void *      (*init_field)( lcg_grid *);
    int         (*update_field)( lcg_grid *);
    double      (*update_charges)( lcg_grid *, particles *);
};

struct fft_grid {
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing

    // int n_local; // X - Number of grid points per dimension (MPI aware)

    double *q;  // Charge density
    double *phi;  // Last potential
    double *ig2;  // Inverse of the laplacian

    void *     (*free)( fft_grid *);
    void *     (*init_field)( fft_grid *);
    int        (*update_field)( fft_grid *);
    double     (*update_charges)( fft_grid *, particles *);
};

struct particles {
    int n;  // Number of grid points per dimension
    int n_p;  // Number of particles
    double L;  // Length of the grid
    double h;  // Grid spacing
    int grid_type;  // Type of the grid

    double *pos;  // Particle positions (n_p x 3)
    double *vel;  // Particle velocities (n_p x 3)
    double *fcs_elec;  // Particle electric forces (n_p x 3)
    double *fcs_noel;  // Particle non-electric forces (n_p x 3)
    double *fcs_tot;  // Particle total forces (n_p x 3)
    double *mass;  // Particle masses (n_p)
    long int *charges;  // Particle charges (n_p)
    long int *neighbors;  // Particle neighbors (n_p x 8 x 3)

    double r_cut;
    double sigma;
    double epsilon;
    double *tf_params;  // Parameters for the TF potential (6 x n_p x n_p)

    void *      (*free)( particles *);

    void *      (*init_potential)( particles *, int pot_type);
    void *      (*init_potential_tf)( particles *);
    void *      (*init_potential_ld)( particles *);
    void *      (*update_nearest_neighbors)( particles *);
    double      (*compute_forces_field)( particles *, void *);
    double      (*compute_forces_noel)( particles *);
    void *      (*compute_forces)( particles *, void *);
    double      (*get_temperature)( particles *);
    double      (*get_kinetic_energy)( particles *);
    void *      (*get_momentum)( particles *, double *);

    void *      (*rescale_velocities)( particles *);
};



#endif // __MP_STRUCTS_H