#include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <math.h>

#include "linalg.h"
#include "charges.h"
#include "constants.h"
#include "forces.h"
#include "mp_structs.h"
#include "mpi_base.h"

#define NUM_NEIGH_CIC 8
#define NUM_NEIGH_SPLINE 64

// Potential types
char potential_type_str[2][16] = {"TF", "LD"};

int get_potential_type_num() {
    return PARTICLE_POTENTIAL_TYPE_NUM;
}

char *get_potential_type_str(int n) {
    return potential_type_str[n];
}

// Charge assignment scheme types
char ca_scheme_type_str[3][16] = {"CIC", "SPL_QUADR", "SPL_CUBIC"};

int get_ca_scheme_type_num() {
    return CHARGE_ASS_SCHEME_TYPE_NUM;
}

char *get_ca_scheme_type_str(int n) {
    return ca_scheme_type_str[n];
}

void particle_charges_init(particles *p, int cas_type) {
    int n_p = p->n_p;

    p->cas_type = cas_type;
    switch (cas_type) {
        case CHARGE_ASS_SCHEME_TYPE_CIC:
            p->num_neighbors = NUM_NEIGH_CIC;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_CIC * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_cic;
            p->charges_spread_func = spread_cic;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLQUAD:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_quadr;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLCUB:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_cubic;
            break;
        default:
            mpi_fprintf(stderr, "Invalid charge assignment scheme type %d\n", cas_type);
            exit(1);
            break;
    }   
}

particles * particles_init(int n, int n_p, double L, double h, int cas_type) {
    particles *p = (particles *)malloc(sizeof(particles));
    p->n = n;
    p->n_p = n_p;
    p->L = L;
    p->h = h;

    p->pos = (double *)malloc(n_p * 3 * sizeof(double));
    p->vel = (double *)malloc(n_p * 3 * sizeof(double));
    p->fcs_elec = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_noel = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_tot = (double *)calloc(n_p * 3, sizeof(double));
    p->mass = (double *)malloc(n_p * sizeof(double));
    p->charges = (long int *)malloc(n_p * sizeof(long int));

    p->fcs_db = NULL;
    p->fcs_ib = NULL;
    p->solv_radii = NULL;

    // p->neighbors = (long int *)malloc(n_p * 24 * sizeof(long int));
    particle_charges_init(p, cas_type);

    p->tf_params = NULL;

    p->free = particles_free;
    p->init_potential = particles_init_potential;
    p->init_potential_tf = particles_init_potential_tf;
    p->init_potential_ld = particles_init_potential_ld;
    
    p->compute_forces_field = particles_compute_forces_field;
    p->compute_forces_noel = NULL;
    p->compute_forces_tot = particles_compute_forces_tot;
    p->get_temperature = particles_get_temperature;
    p->get_kinetic_energy = particles_get_kinetic_energy;
    p->get_momentum = particles_get_momentum;
    p->rescale_velocities = particles_rescale_velocities;

    return p;
}

void particles_pb_init(particles *p, double gamma_np, double beta_np, double probe_radius) {
    p->non_polar = 1;
    p->gamma_np = gamma_np;
    p->beta_np = beta_np;
    p->probe_radius = probe_radius;

    p->compute_forces_field = particles_compute_forces_pb;
    p->compute_forces_dielec_boundary = particles_compute_forces_dielec_boundary;
    p->compute_forces_ionic_boundary = particles_compute_forces_ionic_boundary;

    p->solv_radii = (double *)malloc(p->n_p * sizeof(double));
    // TODO: Initialize the solvation radii for each particle

    // Allocate forces for dielectric and ionic boundary conditions
    // p->fcs_rf = (double *)calloc(p->n_p * 3, sizeof(double));  // Reaction field forces
    p->fcs_db = (double *)calloc(p->n_p * 3, sizeof(double));  // Dielectric boundary forces
    p->fcs_ib = (double *)calloc(p->n_p * 3, sizeof(double));  // Ionic boundary forces
    
    // // Initialize the potential for Poisson-Boltzmann
    // p->init_potential(p, PARTICLE_POTENTIAL_TYPE_LD);
}

void particles_pb_free(particles *p) {
    if (p->solv_radii != NULL) {
        free(p->solv_radii);
        free(p->fcs_db);
        free(p->fcs_ib);
    }
}

void particles_free(particles *p) {
    free(p->pos);
    free(p->vel);
    free(p->fcs_elec);
    free(p->fcs_noel);
    free(p->fcs_tot);
    free(p->mass);
    free(p->charges);
    free(p->neighbors);
    if (p->tf_params != NULL) {
        free(p->tf_params);
    }

    particles_pb_free(p);

    free(p);
}

void particles_init_potential(particles *p, int pot_type) {
    p->pot_type = pot_type;
    switch (pot_type) 
    {
    case PARTICLE_POTENTIAL_TYPE_TF:
        p->init_potential_tf(p);
        break;
    case PARTICLE_POTENTIAL_TYPE_LD:
        p->init_potential_ld(p);
        break;
    default:
        mpi_fprintf(stderr, "Invalid potential type %d\n", pot_type);
        exit(1);
        break;
    }
}

// This really needs to be generalized
#define A_ClCl 5.8145 * 0.001  // Hartree = 15.2661 kJ/mol
#define C_ClCl 2.6607 / a0_6  // = 6985.6841 kj/mol * Ang^6
#define D_ClCl 5.3443 / a0_8  // = 14,031.5897 kj/mol * Ang^8
#define sigma_TF_ClCl 3.170 / a0  // 3.170 angs

#define A_NaNa 9.6909 * 0.001  // Hartree = 25.4435 kJ/mol
#define C_NaNa 0.0385 / a0_6  // = 101.1719 kj/mol * Ang^6
#define D_NaNa 0.0183 / a0_8  // = 48.1771 kj/mol * Ang^8
#define sigma_TF_NaNa 2.340 / a0  // 2.340 angs

#define A_NaCl 7.7527 * 0.001  // Hartree = 20.3548 kJ/mol
#define C_NaCl 0.2569 / a0_6  // = 674.4798 kj/mol * Ang^6
#define D_NaCl 0.3188 / a0_8  // = 837.0777 kj/mol * Ang^8
#define sigma_TF_NaCl 2.755 / a0  // 2.755 angs

#define charge_min -2.0

#define B 3.1546 * a0


double tf_dict[5][4] = {
    {A_ClCl, C_ClCl, D_ClCl, sigma_TF_ClCl},
    {0.0, 0.0, 0.0, 0.0},
    {A_NaCl, C_NaCl, D_NaCl, sigma_TF_NaCl},
    {0.0, 0.0, 0.0, 0.0},
    {A_NaNa, C_NaNa, D_NaNa, sigma_TF_NaNa}
};

void particles_init_potential_tf(particles *p) {
    int np = p->n_p;
    long int np2 = np * np;

    p->tf_params = (double *)malloc(6 * np2 * sizeof(double));

    double r_cut = p->L / 2.0;
    p->r_cut = r_cut;
    double r_cut_6 = pow(r_cut, 6);
    double r_cut_7 = r_cut_6 * r_cut;
    double r_cut_8 = r_cut_7 * r_cut;
    double r_cut_9 = r_cut_8 * r_cut;

    long int in, inj;
    long int charge_sum;
    double A,C,D,sigma,v_shift,alpha,beta;
    for (int i = 0; i < np; i++) {
        in = i * np;
        for (int j = 0; j < np; j++) {
            inj = in + j;
            charge_sum =  p->charges[i] + p->charges[j] - charge_min;

            A = tf_dict[charge_sum][0];
            C = tf_dict[charge_sum][1];
            D = tf_dict[charge_sum][2];
            sigma = tf_dict[charge_sum][3];

            v_shift = A * exp(B * (sigma - r_cut)) - C / r_cut_6 - D / r_cut_8;
            alpha = A * B * exp(B * (sigma - r_cut)) - 6 * C / r_cut_7 - 8 * D / r_cut_9;
            beta = - v_shift - alpha * r_cut;

            p->tf_params[        inj] = A;
            p->tf_params[  np2 + inj] = C;
            p->tf_params[2*np2 + inj] = D;
            p->tf_params[3*np2 + inj] = sigma;
            p->tf_params[4*np2 + inj] = alpha;
            p->tf_params[5*np2 + inj] = beta;
        }
    }

    p->compute_forces_noel = particles_compute_forces_tf;
}

void particles_init_potential_ld(particles *p) {
    p->sigma = 3.00512 * 2 / a0;
    p->epsilon = 5.48 * 1e-4;
    p->r_cut = 2.5 * p->sigma;

    p->compute_forces_noel = particles_compute_forces_ld;
}

void particles_update_nearest_neighbors_cic(particles *p) {
    int np = p->n_p;
    int n = p->n;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->neighbors;
    double *pos = p->pos;

    int i, j;
    long int i0, i1;
    int ni, nj, nk, nip, njp, nkp;

    #pragma omp parallel for private(i, j, i0, i1, ni, nj, nk, nip, njp, nkp)
    for (i = 0; i < np; i++) {
        i0 = i * 3;
        i1 = i * NUM_NEIGH_CIC * 3;

        ni = (int)floor(pos[i0] / h);
        nj = (int)floor(pos[i0 + 1] / h);
        nk = (int)floor(pos[i0 + 2] / h);

        nip = (ni + 1) % n;
        njp = (nj + 1) % n;
        nkp = (nk + 1) % n;

        for (int j = 0; j < 24; j += 3) {
            neighbors[i1 + j + 0] = ni;
            neighbors[i1 + j + 1] = nj;
            neighbors[i1 + j + 2] = nk;
        }

        neighbors[i1 +  3 + 0] = nip;  // 1,0,0
        neighbors[i1 +  6 + 1] = njp;  // 0,1,0
        neighbors[i1 +  9 + 2] = nkp;  // 0,0,1

        neighbors[i1 + 12 + 0] = nip;  // 1,1,0
        neighbors[i1 + 12 + 1] = njp;

        neighbors[i1 + 15 + 0] = nip;  // 1,0,1
        neighbors[i1 + 15 + 2] = nkp;

        neighbors[i1 + 18 + 1] = njp;  // 0,1,1
        neighbors[i1 + 18 + 2] = nkp;

        neighbors[i1 + 21 + 0] = nip;  // 1,1,1
        neighbors[i1 + 21 + 1] = njp;
        neighbors[i1 + 21 + 2] = nkp;
    }
}

void particles_update_nearest_neighbors_spline(particles *p) {    
    int np = p->n_p;
    int n = p->n;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->neighbors;
    double *pos = p->pos;

    int i;
    long int i0, i1;
    int ni, nj, nk, nip, njp, nkp;

    #pragma omp parallel for private(i, i0, i1, ni, nj, nk, nip, njp, nkp)
    for (i = 0; i < np; i++) {
        i0 = i * 3;
        i1 = i * NUM_NEIGH_SPLINE * 3;

        ni = (int)floor(pos[i0] / h);
        nj = (int)floor(pos[i0 + 1] / h);
        nk = (int)floor(pos[i0 + 2] / h);

        for (int a=-1; a <= 2; a++) {
            nip = (ni + a + n) % n;
            for (int b=-1; b <= 2; b++) {
                njp = (nj + b + n) % n;
                for (int c=-1; c <= 2; c++) {
                    nkp = (nk + c + n) % n;

                    neighbors[i1] = nip;
                    neighbors[i1 + 1] = njp;
                    neighbors[i1 + 2] = nkp;
                    i1 += 3;
                }
            }
        }
    }    
}

double particles_compute_forces_field(particles *p, grid *grid) {
    return compute_force_fd(
        p->n, p->n_p, p->h, p->num_neighbors,
        grid->phi_n, p->neighbors, p->charges, p->pos, p->fcs_elec,
        p->charges_spread_func
    );
}

double particles_compute_forces_tf(particles *p) {
    return compute_tf_forces(p->n_p, p->L, p->pos, B, p->tf_params, p->r_cut, p->fcs_noel);
}

double particles_compute_forces_ld(particles *p) {
    return 0.0;
}

double particles_compute_forces_pb(particles *p, grid *grid) {
    compute_forces_reaction_field(
        p->n, p->n_p, p->h, p->num_neighbors,
        grid->phi_n, grid->phi_s, p->neighbors, p->charges, p->pos,
        p->fcs_elec, p->charges_spread_func
    );
}

double particles_compute_forces_dielec_boundary(particles *p, grid *grid) {
    if (grid->w == 0.0) {
        // If the dielectric boundary is not set, we do not compute forces
        return 0.0;
    }
    if (p->fcs_db != NULL) {
        compute_forces_dielec_boundary(
            p->n, p->n_p, p->h, p->num_neighbors,
            grid->k2, grid->H_ratio, grid->H_mask, grid->r_hat,
            p->fcs_db
        );
    }
    return 0.0;
}

double particles_compute_forces_ionic_boundary(particles *p, grid *grid) {
    if (grid->w == 0.0) {
        // If the dielectric boundary is not set, we do not compute forces
        return 0.0;
    }
    if (p->fcs_ib != NULL) {
        return compute_forces_ionic_boundary(
            p->n, p->n_p, p->h, p->num_neighbors,
            grid->k2, grid->H_ratio, grid->H_mask, grid->r_hat,
            p->fcs_ib
        );
    }
    return 0.0;
}

void particles_compute_forces_tot(particles *p) {
    int size = p->n_p * 3;
    vec_copy(p->fcs_elec, p->fcs_tot, size);
    if (p->fcs_noel != NULL) {
        daxpy(p->fcs_noel, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_db != NULL) {
        daxpy(p->fcs_db, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_ib != NULL) {
        daxpy(p->fcs_ib, p->fcs_tot, 1.0, size);
    }
}


double particles_get_temperature(particles *p) {
    return 2 * particles_get_kinetic_energy(p) / (3 * p->n_p * kB);
}

double particles_get_kinetic_energy(particles *p) {
    long int ni;
    double kin = 0.0;
    double app;

    #pragma omp parallel for private(ni, app) reduction(+:kin)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        app = 0.0;
        // printf("vel: %e, %e, %e\n", p->vel[ni], p->vel[ni + 1], p->vel[ni + 2]);
        for (int j = 0; j < 3; j++) {
            app += pow(p->vel[ni + j], 2);
        }
        // printf("app: %e, mass: %f\n", app, p->mass[i]);
        kin += p->mass[i] * app;
    }

    return 0.5 * kin;
}

void particles_get_momentum(particles *p, double *out) {
    int ni;
    double mass;
    double px = 0.0, py = 0.0, pz = 0.0;

    #pragma omp parallel for private(ni, mass) reduction(+:px, py, pz)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        mass = p->mass[i];
        px += mass * p->vel[ni];
        py += mass * p->vel[ni + 1];
        pz += mass * p->vel[ni + 2];
    }

    out[0] = px;
    out[1] = py;
    out[2] = pz;
}

// This also needs to be generalized
void particles_rescale_velocities(particles *p) {
    long int min_charge = -1;
    double init_vel[3][3] = {
        {0.0, 0.0, 0.0},  // Cl
        {0.0, 0.0, 0.0},  // 0
        {0.0, 0.0, 0.0}   // Na
    };

    for (int i = 0; i < p->n_p; i++) {
        for (int j = 0; j < 3; j++) {
            init_vel[p->charges[i] - min_charge][j] += p->vel[i * 3 + j];
        }
    }

    for (int i = 0; i < p->n_p; i++) {
        for (int j = 0; j < 3; j++) {
            p->vel[i * 3 + j] -= 2 * init_vel[p->charges[i] - min_charge][j] / p->n_p;
        }
    }
}

