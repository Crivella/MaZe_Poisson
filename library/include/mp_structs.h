#ifndef __MP_STRUCTS_H
#define __MP_STRUCTS_H

#define GRID_TYPE_NUM 3
#define GRID_TYPE_LCG 0
#define GRID_TYPE_FFT 1
#define GRID_TYPE_MGRID 2

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

#define PRECOND_TYPE_NUM 5
#define PRECOND_TYPE_NONE 0
#define PRECOND_TYPE_JACOBI 1
#define PRECOND_TYPE_MG 2
#define PRECOND_TYPE_SSOR 3
#define PRECOND_TYPE_BLOCKJACOBI 4

// Struct typedefs
typedef struct grid grid;
typedef struct particles particles;
typedef struct integrator integrator;

// Struct function definitions
grid * grid_init(int n, double L, double h, double tol, double eps, int type, int precond_type);
particles * particles_init(int n, int n_p, int n_typ, double L, double h, int cas_type);
integrator * integrator_init(int n_p, double dt, int type);

void grid_free(grid *grid);
void particles_free(particles *p);
void integrator_free(integrator *integrator);

void grid_pb_init(grid *grid, double w, double kbar2);
void grid_pb_free(grid *grid);
void grid_update_eps_and_k2(grid *grid, particles *particles);
double grid_get_energy_elec(grid *grid);

void lcg_grid_init(grid * grid);
void lcg_grid_cleanup(grid * grid);
void lcg_grid_init_field(grid *grid);
int lcg_grid_update_field(grid *grid);
double lcg_grid_update_charges(grid *grid, particles *p);

void multigrid_grid_init(grid * grid);
void multigrid_grid_cleanup(grid * grid);
void multigrid_grid_init_field(grid *grid);
int multigrid_grid_update_field(grid *grid);
double multigrid_grid_update_charges(grid *grid, particles *p);

void fft_grid_init(grid * grid);
void fft_grid_cleanup(grid * grid);
void fft_grid_init_field(grid *grid);
int fft_grid_update_field(grid *grid);
double fft_grid_update_charges(grid *grid, particles *p);

void particles_pb_init(particles *p, double gamma_np, double beta_np, double *solv_radii);
void particles_pb_free(particles *p);

void particles_init_potential(particles *p, int pot_type, double *pot_params);
void particles_init_potential_tf(particles *p, double *pot_params);
void particles_init_potential_ld(particles *p, double *pot_params);
void particles_update_nearest_neighbors_cic(particles *p);
void particles_update_nearest_neighbors_spline(particles *p);

double particles_compute_forces_field(particles *p, grid *grid);
double particles_compute_forces_tf(particles *p);
double particles_compute_forces_ld(particles *p);
double particles_compute_forces_rf(particles *p, grid *grid);
double particles_compute_forces_pb(particles *p, grid *grid);
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
void precond_jacobi_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_mg_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_ssor_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_blockjacobi_apply(double *in, double *out, int s1, int s2, int n_start);

void precond_blockjacobi_init();
void precond_blockjacobi_cleanup();

#define H_ARR_SIZE 4

// Struct definitions
struct grid {
    int type;  // Type of the grid
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing
    double eps_s;  // Dielectric constant of the solvent

    long int size;  // Total number of grid points
    int n_local; // X - Number of grid points per dimension (MPI aware)
    int n_start; // Start index of the grid in the global array (MPI aware)

    double *y;  // Intermediate field constraint
    double *q;  // Charge density
    double *phi_p;  // Previous potential (could be NULL if not needed by the method)
    double *phi_n;  // Last potential
    double *ig2;  // Inverse of the laplacian

    int precond_type;  // Type of the preconditioner

    // Poisson-Boltzmann specific
    int pb_enabled;  // Poisson-Boltzmann enabled
    double w;  // Ionic boundary width
    double kbar2;  // Screening factor

    double *k2;  // Screening factor
    double *eps_x;  // Dielectric constant
    double *eps_y;  // Dielectric constant
    double *eps_z;  // Dielectric constant

    double tol;  // Tolerance for the LCG
    long int n_iters;  // Number of iterations for convergence of the LCG

    void    (*free)( grid *);
    void    (*init_field)( grid *);
    void    (*apply_precond)( double *, double *, int, int, int);
    int     (*update_field)( grid *);
    double  (*update_charges)( grid *, particles *);
};

struct particles {
    int n;  // Number of grid points per dimension
    int n_p;  // Number of particles
    int n_typ;  // Number of particle types (charge, masses, ... definitions)
    double L;  // Length of the grid
    double h;  // Grid spacing

    int num_neighbors;  // Number of neighbors per particle

    int pot_type;  // Type of the potential
    int cas_type;  // Type of the charge assignment scheme

    int *types;  // Particle types (n_p)
    double *pos;  // Particle positions (n_p x 3)
    double *vel;  // Particle velocities (n_p x 3)
    double *fcs_elec;  // Particle electric forces (n_p x 3)
    double *fcs_noel;  // Particle non-electric forces (n_p x 3)
    double *fcs_tot;  // Particle total forces (n_p x 3)
    double *mass;  // Particle masses (n_p)
    double *charges;  // Particle charges (n_p)
    long int *neighbors;  // Particle neighbors (n_p x 8 x 3)

    double r_cut;
    double sigma;
    double epsilon;
    double *tf_params;  // Parameters for the TF potential (7 x n_p x n_p)

    // Poisson-Boltzmann specific
    int pb_enabled;  // Poisson-Boltzmann enabled
    double gamma_np;
    double beta_np;
    // double *fcs_rf; // Particle reaction field forces (n_p x 3)
    double *fcs_db; // Dielectric boundary forces (n_p x 3)
    double *fcs_ib; // Ionic boundary forces (n_p x 3)
    double *fcs_np; // Non-polar forces (n_p x 3)
    double *solv_radii; // Solvation radii for each particle (n_p)

    void    (*free)( particles *);

    void    (*init_potential)( particles *, int, double *);
    void    (*init_potential_tf)( particles *, double *);
    void    (*init_potential_ld)( particles *, double *);

    void    (*update_nearest_neighbors)( particles *);
    double  (*charges_spread_func)( double, double, double);

    double  (*compute_forces_field)( particles *, grid *);
    double  (*compute_forces_noel)( particles *);
    void    (*compute_forces_tot)( particles *);
    double  (*compute_forces_pb)( particles *, grid *);

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