#ifndef __MP_STRUCTS_H
#define __MP_STRUCTS_H

#define GRID_TYPE_NUM 2
#define GRID_TYPE_LCG 0
#define GRID_TYPE_FFT 1

#define PARTICLE_POTENTIAL_TYPE_NUM 2
#define PARTICLE_POTENTIAL_TYPE_TF 0
#define PARTICLE_POTENTIAL_TYPE_LD 1

#define CHARGE_ASS_SCHEME_TYPE_NUM 3
#define CHARGE_ASS_SCHEME_TYPE_CIC 0
#define CHARGE_ASS_SCHEME_TYPE_SPLQUAD 1
#define CHARGE_ASS_SCHEME_TYPE_SPLCUB 2

#define INTEGRATOR_TYPE_NUM 2
#define INTEGRATOR_TYPE_OVRVO 0
#define INTEGRATOR_TYPE_VERLET 1

#define INTEGRATOR_ENABLED 1
#define INTEGRATOR_DISABLED 0

#define PRECOND_TYPE_NUM 2
#define PRECOND_TYPE_JACOBI 0
#define PRECOND_TYPE_MG 1

// Struct typedefs
typedef struct grid grid;
typedef struct particles particles;
typedef struct integrator integrator;
typedef struct precond precond;

// Struct function definitions
grid * grid_init(int n, double L, double h, double tol, int type, int precond_type);
particles * particles_init(int n, int n_p, double L, double h, int cas_type);
integrator * integrator_init(int n_p, double dt, int type);
precond * precond_init(int n, double L, double h, int type);

void grid_free(grid *grid);
void particles_free(particles *p);
void integrator_free(integrator *integrator);

void lcg_grid_init(grid * grid);
void lcg_grid_cleanup(grid * grid);
void lcg_grid_init_field(grid *grid);
int lcg_grid_update_field(grid *grid);
double lcg_grid_update_charges(grid *grid, particles *p);

void fft_grid_init(grid * grid);
void fft_grid_cleanup(grid * grid);
void fft_grid_init_field(grid *grid);
int fft_grid_update_field(grid *grid);
double fft_grid_update_charges(grid *grid, particles *p);

void particles_init_potential(particles *p, int pot_type);
void particles_init_potential_tf(particles *p);
void particles_init_potential_ld(particles *p);
void particles_update_nearest_neighbors_cic(particles *p);
void particles_update_nearest_neighbors_spline(particles *p);
double particles_compute_forces_field(particles *p, grid *grid);
double particles_compute_forces_tf(particles *p);
double particles_compute_forces_ld(particles *p);
void particles_compute_forces_tot(particles *p);
double particles_get_temperature(particles *p);
double particles_get_kinetic_energy(particles *p);
void particles_get_momentum(particles *p, double *out);
void particles_rescale_velocities(particles *p);

void ovrvo_integrator_init(integrator *integrator);
void ovrvo_integrator_part1(integrator *integrator, particles *p);
void ovrvo_integrator_part2(integrator *integrator, particles *p);
void ovrvo_integrator_init_thermostat(integrator *integrator, double *params);
void ovrvo_integrator_stop_thermostat(integrator *integrator);

void verlet_integrator_init(integrator *integrator);
void verlet_integrator_part1(integrator *integrator, particles *p);
void verlet_integrator_part2(integrator *integrator, particles *p);
void verlet_integrator_init_thermostat(integrator *integrator, double *params);
void verlet_integrator_stop_thermostat(integrator *integrator);

// Preconditioner function definitions
void precond_cleanup(precond *p);
void precond_apply(precond *p, double *in, double *out);
void precond_prolong(double *in, double *out, int s1, int s2, int ts1, int ts2);
void precond_restriction(double *in, double *out, int s1, int s2);
void precond_smooth(double *in, double *out, int s1, int s2, double tol);

// void precond_jacobi_init(precond *p);
// void precond_jacobi_cleanup(precond *p);
// void precond_jacobi_prolong(precond *p, double *in, double *out);
// void precond_jacobi_restriction(precond *p, double *in, double *out);
// void precond_jacobi_smooth(precond *p, double *in, double *out);

void precond_mg_init(precond *p);
void precond_mg_cleanup(precond *p);
void precond_mg_apply(precond *p, double *in, double *out);
void precond_mg_prolong(double *in, double *out, int s1, int s2, int ts1, int ts2);
void precond_mg_restriction(double *in, double *out, int s1, int s2);
void precond_mg_smooth(double *in, double *out, int s1, int s2, double tol);

// Struct definitions
struct precond {
    int type;  // Type of the preconditioner
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing

    // Multigrid parameters
    double tol; // Tolerance for the smoothing
    int n1;
    int n_loc1;  // Number of grid points per dimension (MPI aware)
    int n_start1; // Start index of the grid in the global array (MPI aware)
    double *grid1;  // Intermediate grid for the preconditioner

    int n2;
    int n_loc2;  // Number of grid points per dimension (MPI aware)
    int n_start2; // Start index of the grid in the global array (MPI aware)
    double *grid2;  // Intermediate grid for the preconditioner

    int n3;
    int n_loc3;  // Number of grid points per dimension (MPI aware)
    int n_start3; // Start index of the grid in the global array (MPI aware)
    double *grid3;  // Intermediate grid for the preconditioner

    void (*apply)( precond *, double *, double *); // Apply function
    void (*prolong)( double *, double *, int, int, int, int); // Prolongation function
    void (*restriction)( double *, double *, int, int); // Restriction function
    void (*smooth)( double *, double *, int, int, double); // Smoothing function


    void (*free)( precond *);
};

struct grid {
    int type;  // Type of the grid
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing

    long int size;  // Total number of grid points
    int n_local; // X - Number of grid points per dimension (MPI aware)
    int n_start; // Start index of the grid in the global array (MPI aware)

    double *y;  // Intermediate field constraint
    double *q;  // Charge density
    double *phi_p;  // Previous potential (could be NULL if not needed by the method)
    double *phi_n;  // Last potential
    double *ig2;  // Inverse of the laplacian

    int precond_type;  // Type of the preconditioner
    precond *precond;  // Preconditioner

    double tol;  // Tolerance for the LCG
    long int n_iters;  // Number of iterations for convergence of the LCG

    void    (*free)( grid *);
    void    (*init_field)( grid *);
    int     (*update_field)( grid *);
    double  (*update_charges)( grid *, particles *);
};

struct particles {
    int n;  // Number of grid points per dimension
    int n_p;  // Number of particles
    double L;  // Length of the grid
    double h;  // Grid spacing

    int num_neighbors;  // Number of neighbors per particle

    int pot_type;  // Type of the potential
    int cas_type;  // Type of the charge assignment scheme

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

    void    (*free)( particles *);

    void    (*init_potential)( particles *, int pot_type);
    void    (*init_potential_tf)( particles *);
    void    (*init_potential_ld)( particles *);

    void    (*update_nearest_neighbors)( particles *);
    double  (*charges_spread_func)( double, double, double);

    double  (*compute_forces_field)( particles *, grid *);
    double  (*compute_forces_noel)( particles *);
    void    (*compute_forces_tot)( particles *);

    double  (*get_temperature)( particles *);
    double  (*get_kinetic_energy)( particles *);
    void    (*get_momentum)( particles *, double *);

    void    (*rescale_velocities)( particles *);
};

struct integrator {
    int type;  // Type of the integrator
    int n_p;  // Number of particles
    double dt;  // Time step
    double T;  // Temperature

    int enabled;  // Thermostat enabled
    double c1;  // Thermostat parameter
    double c2;  // Thermostat parameter

    void    (*part1)( integrator *, particles *);
    void    (*part2)( integrator *, particles *);
    void    (*init_thermostat)( integrator *, double *);
    void    (*stop_thermostat)( integrator *);
    void    (*free)( integrator *);
};

#endif // __MP_STRUCTS_H