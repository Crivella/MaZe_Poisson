#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "charges.h"
#include "constants.h"
#include "forces.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "omp_base.h"

#define NUM_NEIGH_CIC 8
#define NUM_NEIGH_SPLINE 64

neighbor *neighbor_init() {
    neighbor *res = (neighbor *)malloc(sizeof(neighbor));
    res->valid=0;
    res->next = NULL;

    return res;
}

void neighbor_free(neighbor *curr) {
    neighbor *next;

    while (curr != NULL) {
        next = curr->next;
        free(curr);
        curr = next;
    }
}

void particle_charges_init(particles *p, int cas_type) {
    int n_p = p->n_p;

    p->cas_type = cas_type;
    switch (cas_type) {
        case CHARGE_ASS_SCHEME_TYPE_CIC:
            p->num_neighbors = NUM_NEIGH_CIC;
            p->grid_neighbors = (long int *)malloc(n_p * NUM_NEIGH_CIC * 3 * sizeof(long int));
            p->update_grid_nearest_neighbors = particles_update_grid_nearest_neighbors_cic;
            p->charges_spread_func = spread_cic;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLQUAD:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->grid_neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_grid_nearest_neighbors = particles_update_grid_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_quadr;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLCUB:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->grid_neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_grid_nearest_neighbors = particles_update_grid_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_cubic;
            break;
        default:
            mpi_fprintf(stderr, "Invalid charge assignment scheme type %d\n", cas_type);
            exit(1);
            break;
    }   
}

void particle_pneigh_sphere(particles *p) {
    int i;
    double dx, dy, dz, dr2;
    double rc2 = p->r_cut * p->r_cut;

    neighbor *curr = NULL;

    #pragma omp parallel private(i, curr, dx, dy, dz, dr2)
    for (int i_loc = 0; i_loc < p->np_local; i_loc++) {
        i = p->np_start + i_loc;

        curr = p->particle_neighbors[i_loc];

        for (int j = 0; j < p->n_p; j++) {
            if (i == j) {
                continue;
            }
            pbc_displacement(p->pos, j, i, p->L, &dx, &dy, &dz, &dr2);
            if (dr2 < rc2) {
                curr->valid = 1;
                curr->idx = j;
                curr->dx = dx;
                curr->dy = dy;
                curr->dz = dz;
                curr->dist = sqrt(dr2);

                if (curr->next == NULL) {
                    curr->next = neighbor_init();
                }
                curr = curr->next;
            }
        }

        curr->valid = 0; // Truncate the neighbor list after processing this cell
    }
}

void particle_pneigh_cell_list(particles *p) {
    long int i, j;
    double *pos = p->pos;
    double dx, dy, dz, dr2;
    double rc2 = p->r_cut * p->r_cut;

    int s1 = p->cell_list_size;
    long int s1_2 = s1 * s1;
    long int s1_3 = s1 * s1 * s1;
    int cell_idx;
    int cx, cy, cz;
    double cell_length = p->cell_list_length;

    int ncx, ncy, ncz;

    neighbor *curr = NULL;

    int *particle_cell_x = (int *)malloc(p->n_p * sizeof(int));
    int *particle_cell_y = (int *)malloc(p->n_p * sizeof(int));
    int *particle_cell_z = (int *)malloc(p->n_p * sizeof(int));
    // long int *particle_cell_index = (long int *)malloc(p->n_p * sizeof(long int));

    // Build cell list
    for (i = 0; i < s1_3; i++) {
        p->cell_list_head[i] = -1;
    }
    for (i = 0; i < p->n_p; i++) {
        cx = (int)(pos[3*i + 0] / cell_length);
        cy = (int)(pos[3*i + 1] / cell_length);
        cz = (int)(pos[3*i + 2] / cell_length);
        cell_idx = cx * s1_2 + cy * s1 + cz;
        p->cell_list_next[i] = p->cell_list_head[cell_idx];
        p->cell_list_head[cell_idx] = i;

        particle_cell_x[i] = cx;
        particle_cell_y[i] = cy;
        particle_cell_z[i] = cz;
        // particle_cell_index[i] = cell_idx; 
    }

    // Find neighbors using cell list
    #pragma omp parallel private( \
        i, j, curr, cx, cy, cz, ncx, ncy, ncz, cell_idx, \
        dx, dy, dz, dr2 \
    )
    for (int i_loc = 0; i_loc < p->np_local; i_loc++) {
        i = p->np_start + i_loc;
        cx = particle_cell_x[i];
        cy = particle_cell_y[i];
        cz = particle_cell_z[i];

        curr = p->particle_neighbors[i_loc];

        // This has to be local if OpenMP has to be enabled
        // Ideally r_cut should be such that each cells contains on average multiple particles so it is easier
        // to check for the cell_idx
        int *done = (int *)calloc(s1_3, sizeof(int));

        for (int dx_cell = -1; dx_cell <= 1; dx_cell++) {
            ncx = (cx + dx_cell + s1) % s1;
            for (int dy_cell = -1; dy_cell <= 1; dy_cell++) {
                ncy = (cy + dy_cell + s1) % s1;
                for (int dz_cell = -1; dz_cell <= 1; dz_cell++) {
                    ncz = (cz + dz_cell + s1) % s1;
                    cell_idx = ncx * s1_2 + ncy * s1 + ncz;

                    if (done[cell_idx]) {
                        continue; // Skip this cell if it's already done
                    }
                    done[cell_idx] = 1; // Mark this cell as done

                    j = p->cell_list_head[cell_idx];
                    while (j != -1) {
                        if (i != j) { // Only process if not done
                            pbc_displacement(pos, j, i, p->L, &dx, &dy, &dz, &dr2);
                            if (dr2 < rc2) {
                                curr->valid = 1;
                                curr->idx = j;
                                curr->dx = dx;
                                curr->dy = dy;
                                curr->dz = dz;
                                curr->dist = sqrt(dr2);

                                if (curr->next == NULL) {
                                    curr->next = neighbor_init();
                                }
                                curr = curr->next;
                            }
                        }
                        j = p->cell_list_next[j];
                    }
                }
            }
        }

        curr->valid = 0; // Truncate the neighbor list after processing this cell
        free(done);
    }

    free(particle_cell_x);
    free(particle_cell_y);
    free(particle_cell_z);
}

void particle_pneigh_init(particles *p, int method, double r_cut) {
    // Use local number of particles for MPI compatibility
    p->particle_neighbor_method = method;
    p->particle_neighbors = (neighbor **)malloc(p->np_local * sizeof(neighbor *));
    for (int i = 0; i < p->np_local; i++) {
        p->particle_neighbors[i] = neighbor_init();
    }

    switch (method)
    {
    case PARTICLE_NEIGHBOR_TYPE_SPHERE:
        p->update_particle_neighbors = particle_pneigh_sphere;
        break;
    case PARTICLE_NEIGHBOR_TYPE_CELL_LIST:
        if (r_cut <= 0.0) {
            mpi_fprintf(stderr, "Invalid cutoff radius %f for cell list neighbor method\n", r_cut);
            exit(1);
        }
        // mpi_fprintf(stderr, "Initializing cell list with cutoff radius %f and box size %f\n", r_cut, p->L);
        int s1 = floor(p->L / r_cut);
        // mpi_fprintf(stderr, "Number of cells per dimension: %d\n", s1);
        p->cell_list_size = s1;  // Number of cells per dimension
        p->cell_list_length = p->L / s1;  // Adjust cell size to fit an integer number of cells in the box
        // mpi_printf("Cell side length: %f\n", p->cell_list_size);
        p->cell_list_head = (long int *)malloc(s1 * s1 * s1 * sizeof(long int));
        p->cell_list_next = (long int *)malloc(p->n_p * sizeof(long int));
        p->update_particle_neighbors = particle_pneigh_cell_list;
        break;
    
    default:
        break;
    }
}

void particle_pneigh_free(particles *p) {
    if (p->particle_neighbors != NULL) {
        for (int i = 0; i < p->np_local; i++) {
            neighbor_free(p->particle_neighbors[i]);
        }
        free(p->particle_neighbors);
        p->particle_neighbors = NULL;
    }
    if (p->cell_list_head != NULL) {
        free(p->cell_list_head);
        p->cell_list_head = NULL;
    }
    if (p->cell_list_next != NULL) {
        free(p->cell_list_next);
        p->cell_list_next = NULL;
    }

}


#ifdef __MPI
void particle_init_mpi(particles *p) {
    mpi_data *mpid = get_mpi_data();

    int np = p->n_p;
    int rank = mpid->rank;
    int size = mpid->size;

    int div, mod;
    int np_loc, np_start;

    div = np / size;
    mod = np % size;
    for (int i=0; i<size; i++) {
        if (i < mod) {
            np_loc = div + 1;
            np_start = i * np_loc;
        } else {
            np_loc = div;
            np_start = i * np_loc + mod;
        }
        mpid->np_loc_list[i] = np_loc;
        mpid->np_start_list[i] = np_start;
    }

    p->np_local = mpid->np_loc_list[rank];
    p->np_start = mpid->np_start_list[rank];
    mpid->np_loc = p->np_local;
    mpid->np_start = p->np_start;
}

#else  // __MPI

void particle_init_mpi(particles *p) {
    mpi_data *mpid = get_mpi_data();
    p->np_local = p->n_p;
    p->np_start = 0;
    mpid->np_loc = p->n_p;
    mpid->np_start = 0;
}  // Do nothing

#endif  // __MPI


particles * particles_init(int n_p, int n_typ, double L, double h, ca_scheme_type cas_type) {
    particles *p = (particles *)malloc(sizeof(particles));
    p->n_p = n_p;
    p->n_typ = n_typ;
    p->L = L;
    p->h = h;

    p->types = (int *)malloc(n_p * sizeof(int));
    p->pos = (double *)malloc(n_p * 3 * sizeof(double));
    p->vel = (double *)malloc(n_p * 3 * sizeof(double));
    p->fcs_elec = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_noel = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_tot = (double *)calloc(n_p * 3, sizeof(double));
    p->mass = (double *)malloc(n_p * sizeof(double));
    p->charges = (double *)malloc(n_p * sizeof(double));

    p->particle_neighbor_method = PARTICLE_NEIGHBOR_TYPE_CELL_LIST;
    p->particle_neighbors = NULL;
    p->cell_list_head = NULL;
    p->cell_list_next = NULL;

    p->is_water = 0;
    p->fcs_intra = NULL; // Intramolecular forces total
    p->fcs_corr = NULL; // Electrostatic correction forces
    p->energy_intra = 0.0; // Intramolecular energy bond-angle
    p->energy_corr = 0.0; // Intramolecular exclusion correction energy
    p->compute_intramolecular_forces = NULL;
    p->compute_forces_electrostatic_correction = NULL;
    p->lj_force_shift = 1;

    p->pb_enabled = 0;  // Poisson-Boltzmann not enabled by default
    p->fcs_db = NULL;
    p->fcs_ib = NULL;
    p->fcs_np = NULL;
    p->solv_radii = NULL;

    particle_init_mpi(p);

    // p->neighbors = (long int *)malloc(n_p * 24 * sizeof(long int));
    particle_charges_init(p, cas_type);

    p->tf_params = NULL;
    p->sc_params = NULL;

    p->free = particles_free;
    p->init_potential = particles_init_potential;
    
    p->compute_forces_field = particles_compute_forces_field;
    p->compute_forces_noel = NULL;
    p->compute_forces_tot = particles_compute_forces_tot;
    p->get_temperature = particles_get_temperature;
    p->get_kinetic_energy = particles_get_kinetic_energy;
    p->get_momentum = particles_get_momentum;
    p->rescale_velocities = particles_rescale_velocities;
    // p->rescale_velocities = particles_zero_linear;
    p->rescale_momenta = particles_rescale_momenta;
    // p->rescale_momenta = particles_rescale_momenta_water;

    return p;
}

void particles_pb_init(particles *p, double gamma_np, double beta_np, double *solv_radii) {
    p->pb_enabled = 1;  // Enable Poisson-Boltzmann
    p->gamma_np = gamma_np;
    p->beta_np = beta_np;

    p->compute_forces_pb = particles_compute_forces_pb;

    p->solv_radii = (double *)malloc(p->n_p * sizeof(double));
    memcpy(p->solv_radii, solv_radii, p->n_p * sizeof(double));

    // Allocate forces for dielectric and ionic boundary conditions
    p->fcs_np = (double *)calloc(p->n_p * 3, sizeof(double));  // Non-polar forces
    p->fcs_db = (double *)calloc(p->n_p * 3, sizeof(double));  // Dielectric boundary forces
    p->fcs_ib = (double *)calloc(p->n_p * 3, sizeof(double));  // Ionic boundary forces
    
    // // Initialize the potential for Poisson-Boltzmann
    // p->init_potential(p, PARTICLE_POTENTIAL_TYPE_LD);
}

void particles_pb_free(particles *p) {
    if (p->pb_enabled) {
        free(p->solv_radii);
        free(p->fcs_db);
        free(p->fcs_ib);
        free(p->fcs_np);
    }
}

void particles_water_init(particles *p, int is_water, water_electrostatic_type corr_type) {
    if (!is_water) {
        return;
    }
    p->is_water = 1;
    p->corr_type = corr_type;
    
    p->fcs_intra = (double *)calloc(p->n_p * 3, sizeof(double)); // Intramolecular forces total
    p->fcs_corr = (double *)calloc(p->n_p * 3, sizeof(double)); // Electrostatic correction forces

    p->compute_intramolecular_forces = particles_compute_intramolecular_forces;
    switch (corr_type) {
        case WATER_ELECTROSTATIC_CORR_TYPE_SPREAD:
            p->compute_forces_electrostatic_correction = particles_compute_forces_electrostatic_correction_spread;
            break;
        case WATER_ELECTROSTATIC_CORR_TYPE_SR:
            p->compute_forces_electrostatic_correction = particles_compute_forces_electrostatic_correction_sr;
            break;
        default:
            mpi_fprintf(stderr, "Invalid electrostatic correction type %d\n", corr_type);
            exit(1);
    }
}

void particles_water_free(particles *p) {
    if (p->is_water) {
        free(p->fcs_intra);
        free(p->fcs_corr);
    }
}

void particles_free(particles *p) {
    free(p->types);
    free(p->pos);
    free(p->vel);
    free(p->fcs_elec);
    free(p->fcs_noel);
    free(p->fcs_tot);
    free(p->mass);
    free(p->charges);
    free(p->grid_neighbors);
    if (p->tf_params != NULL) {
        free(p->tf_params);
    }
    if (p->sc_params != NULL) {
        free(p->sc_params);
    }

    particle_pneigh_free(p);
    particles_pb_free(p);
    particles_water_free(p);

    free(p);
}

void particles_init_potential(particles *p, int pot_type, double *pot_params) {
    p->pot_type = pot_type;
    switch (pot_type) 
    {
    case PARTICLE_POTENTIAL_TYPE_TF:
        particles_init_potential_tf(p, pot_params);
        break;
    case PARTICLE_POTENTIAL_TYPE_LJ:
        particles_init_potential_lj(p, pot_params);
        break;
    case PARTICLE_POTENTIAL_TYPE_SC:
        particles_init_potential_sc(p, pot_params);
        break;
    default:
        mpi_fprintf(stderr, "Invalid potential type %d\n", pot_type);
        exit(1);
        break;
    }
}

void particles_init_potential_tf(particles *p, double *pot_params) {
    int typ1, typ2;
    int n_p = p->n_p;
    int n_typ = p->n_typ;
    int n_typ2 = n_typ * n_typ;
    long int np2 = n_p * n_p;

    p->tf_params = (double *)malloc(7 * n_typ2 * sizeof(double));

    double r_cut = (p->r_cut > 0.0) ? p->r_cut : p->L / 2.0;
    p->r_cut = r_cut;
    double r_cut_6 = pow(r_cut, 6);
    double r_cut_7 = r_cut_6 * r_cut;
    double r_cut_8 = r_cut_7 * r_cut;
    double r_cut_9 = r_cut_8 * r_cut;

    long int in, inj, idx0, idx1, idx2;
    double A, B, C, D, sigma, v_shift, alpha, beta;
    for (typ1 = 0; typ1 < n_typ; typ1++) {
        in = typ1 * n_typ;
        for (typ2 = 0; typ2 < n_typ; typ2++) {
            inj = in + typ2;

            // Assuming pot_params is structured as [A, B, C, D, sigma] for each type pair
            idx0 = typ1 * n_typ + typ2;
            idx1 = idx0 * 5;
            idx2 = idx0 * 7;  // Index for tf_params [A, B, C, D, sigma, alpha, beta]

            A = pot_params[idx1 + 0];
            B = pot_params[idx1 + 1];
            C = pot_params[idx1 + 2];
            D = pot_params[idx1 + 3];
            sigma = pot_params[idx1 + 4];

            v_shift = A * exp(B * (sigma - r_cut)) - C / r_cut_6 - D / r_cut_8;
            alpha = A * B * exp(B * (sigma - r_cut)) - 6 * C / r_cut_7 - 8 * D / r_cut_9;
            beta = - v_shift - alpha * r_cut;

            p->tf_params[0*n_typ2 + inj] = A;
            p->tf_params[1*n_typ2 + inj] = B;
            p->tf_params[2*n_typ2 + inj] = C;
            p->tf_params[3*n_typ2 + inj] = D;
            p->tf_params[4*n_typ2 + inj] = sigma;
            p->tf_params[5*n_typ2 + inj] = alpha;
            p->tf_params[6*n_typ2 + inj] = beta;
        }
    }

    p->compute_forces_noel = particles_compute_forces_tf;
}

void particles_init_potential_lj(particles *p, double *pot_params) {
    int typ1, typ2;
    int n_p = p->n_p;
    int n_typ = p->n_typ;
    long int np2 = n_p * n_p;
    int n_typ2 = n_typ * n_typ;

    p->lj_params = (double *)malloc(4 * n_typ2 * sizeof(double));

    double r_cut = (p->r_cut > 0.0) ? p->r_cut : p->L / 2.0;
    p->r_cut = r_cut;
    
    long int in, inj;
    double sigma, epsilon, v_shift, alpha, beta;
    for (typ1 = 0; typ1 < n_typ; typ1++) {
        in = typ1 * n_typ;
        for (typ2 = 0; typ2 < n_typ; typ2++) {
            inj = in + typ2;

            sigma = pot_params[2*inj + 0];
            epsilon = pot_params[2*inj + 1];

            if (p->lj_force_shift) {
                v_shift = 4 * epsilon * (pow(sigma / r_cut, 12) - pow(sigma / r_cut, 6));
                alpha = 4 * (12 * epsilon * pow(sigma / r_cut, 12) - 6 * epsilon * pow(sigma / r_cut, 6)) / r_cut;
                beta = - v_shift - alpha * r_cut;
            } else {
                alpha = 0.0;
                beta = 0.0;
            }

            if (!isfinite(sigma) || !isfinite(epsilon) || !isfinite(alpha) || !isfinite(beta)) {
                mpi_fprintf(
                    stderr,
                    "Error: LJ params non-finite (typ1=%d typ2=%d sigma=%e epsilon=%e alpha=%e beta=%e)\n",
                    typ1, typ2, sigma, epsilon, alpha, beta
                );
                exit(1);
            }

            p->lj_params[0*n_typ2 + inj] = sigma;
            p->lj_params[1*n_typ2 + inj] = epsilon;
            p->lj_params[2*n_typ2 + inj] = alpha;
            p->lj_params[3*n_typ2 + inj] = beta;
        }
    }

    p->compute_forces_noel = particles_compute_forces_lj;
}

void particles_init_potential_sc(particles *p, double *pot_params) {
    double alpha, beta;
    double nu, d, B_nu, r_cut;
    double d_over_r_cut, d_over_r_cut_pow;

    p->sc_params = (double *)malloc(5 * sizeof(double));

    nu = pot_params[0];
    d = pot_params[1];
    B_nu = pot_params[2];

    p->r_cut = (p->r_cut > 0.0) ? p->r_cut : 0.5 * p->L;
    r_cut = p->r_cut;

    d_over_r_cut = d / r_cut;
    d_over_r_cut_pow = pow(d_over_r_cut, nu);

    alpha = B_nu * nu * d_over_r_cut_pow / r_cut; 
    beta = - B_nu * d_over_r_cut_pow - alpha * r_cut;

    p->sc_params[0] = pot_params[0];
    p->sc_params[1] = pot_params[1];
    p->sc_params[2] = pot_params[2];
    p->sc_params[3] = alpha;
    p->sc_params[4] = beta;
    p->compute_forces_noel = particles_compute_forces_sc;
}

void particles_update_grid_nearest_neighbors_cic(particles *p, grid *g) {
    int n = g->n;
    int np = p->n_p;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->grid_neighbors;
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

void particles_update_grid_nearest_neighbors_spline(particles *p, grid *g) {    
    int n = g->n;
    int np = p->n_p;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->grid_neighbors;
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

// TODO: spezzare questo in modo che quando si ha PB_FORCE_TYPE_STRESS_TENSOR invece di puntare a questa
// funzione ne usiamo una che non fa niente
// In piu se si fa PB_FORCE_TYPE_STRESS_TENSOR va skippata la parte di short range correction?
double particles_compute_forces_field(particles *p, grid *grid) {
    if (grid->pb_enabled && grid->pb_force_type == PB_FORCE_TYPE_STRESS_TENSOR) {
        // The tensor forces are already resetting this 2 zero, not sure which is the best place to do this
        // depends on who we want to be responsible for what
        // memset(p->fcs_elec, 0, p->n_p * 3 * sizeof(double));
        return 0.0;
    }
    double res = compute_force_fd(
        grid->n, p->n_p, grid->n_local, grid->n_start, p->h, p->num_neighbors,
        grid->phi_n, p->grid_neighbors, p->charges, p->pos, p->fcs_elec,
        p->charges_spread_func
    );
    if (
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C2 ||
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C2_NOFFT
    ) {
        double *fcs_tmp = (double *)malloc(p->n_p * 3 * sizeof(double));
        res += compute_force_short_range_wendland_c2(
            p->n_p, p->pos, p->charges, fcs_tmp, grid->smoothing_sigma, p->L,
            p->particle_neighbors, p->np_local, p->np_start
        );
        daxpy(fcs_tmp, p->fcs_elec, 1.0, p->n_p * 3);
        free(fcs_tmp);
    } else if (
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C4 ||
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C4_NOFFT
    ) {
        double *fcs_tmp = (double *)malloc(p->n_p * 3 * sizeof(double));
        res += compute_force_short_range_wendland_c4(
            p->n_p, p->pos, p->charges, fcs_tmp, grid->smoothing_sigma, p->L,
            p->particle_neighbors, p->np_local, p->np_start
        );
        daxpy(fcs_tmp, p->fcs_elec, 1.0, p->n_p * 3);
        free(fcs_tmp);
    } else if (
        grid->smoothing == SMOOTHING_TYPE_GAUSS ||
        grid->smoothing == SMOOTHING_TYPE_DIFFUSION
    ) {
        double *fcs_tmp = (double *)malloc(p->n_p * 3 * sizeof(double));
        res += compute_force_short_range(
            p->n_p, p->pos, p->charges, fcs_tmp, p->r_cut, grid->smoothing_sigma, p->L,
            p->particle_neighbors, p->np_local, p->np_start
        );
        daxpy(fcs_tmp, p->fcs_elec, 1.0, p->n_p * 3);
        free(fcs_tmp);
    } else if (grid->smoothing == SMOOTHING_TYPE_NONE) {
        // No short-range correction is needed.
    } else {
        mpi_fprintf(stderr, "Invalid smoothing type %d\n", grid->smoothing);
        exit(1);
    }
    return res;
}

double particles_compute_forces_tf(particles *p) {
    return compute_tf_forces(
        p->n_p, p->n_typ, p->L, p->types, p->pos, p->tf_params,
        p->r_cut, p->particle_neighbors, p->np_local, p->np_start,
        p->fcs_noel
    );
}

double particles_compute_forces_sc(particles *p) {
    return compute_sc_forces(
        p->n_p, p->L, p->pos, p->sc_params,
        p->r_cut, p->particle_neighbors, p->np_local, p->np_start,
        p->fcs_noel
    );
}

double particles_compute_forces_lj(particles *p) { 
    return compute_lj_forces(
        p->n_p, p->n_typ, p->L, p->types, p->pos, p->lj_params,
        p->r_cut, p->particle_neighbors, p->np_local, p->np_start,
        p->fcs_noel, p->lj_force_shift
    );
}

double particles_compute_intramolecular_forces(particles *p) {
    if (!p->is_water) {
        return 0.0;
    }

    long int np = p->n_p;
    long int n3 = np * 3;

    double *fcs = p->fcs_intra;
    double *fcs_tmp = (double *)calloc(n3, sizeof(double));

    memset(fcs, 0, np * 3 * sizeof(double));
    double energy_bond = compute_forces_harmonic_bond(np, p->pos, fcs_tmp, p->L, kb, r0);
    daxpy(fcs_tmp, fcs, 1.0, n3);  // fcs += fcs_tmp
    double energy_angle = compute_forces_harmonic_angle(np, p->pos, fcs_tmp, p->L, ka, theta0);
    daxpy(fcs_tmp, fcs, 1.0, n3);  // fcs += fcs_tmp

    free(fcs_tmp);

    p->energy_intra = energy_bond + energy_angle;

    return p->energy_intra;
}

double particles_compute_forces_electrostatic_correction_spread(particles *p, grid *g) {
    long int size = p->n_p * 3;

    // int rank = get_rank();
    // int size_mpi = get_size();


    long int n_p = p->n_p;
    long int n_triplets = n_p / 3;
    double *pos = p->pos;
    double *charges = p->charges;
    double *fcs = p->fcs_corr;
    double L = p->L;
    double h = g->h;
    double energy_corr = 0.0;
    double dx, dy, dz, r2, r, force_mag;

    memset(fcs, 0, size * sizeof(double));

    // Use the number of neighbors based on the spreading function.
    int num_neighbors = p->num_neighbors;

    // Intramolecular correction from spread charges:
    // loop over unique atom pairs in each molecule (O-H1, O-H2, H1-H2).
    #pragma omp parallel for reduction(+:energy_corr, fcs[:size])
    for (long int m = 0; m < n_triplets; m++) {
        long int idx = m * 3;

        for (int a = 0; a < 3; a++) {
            long int ia = idx + a;
            long int fa = ia * 3;
            long int na0 = ia * num_neighbors * 3;
            double pxa = pos[fa];
            double pya = pos[fa + 1];
            double pza = pos[fa + 2];
            double qa = charges[ia];

            for (int b = a + 1; b < 3; b++) {
                long int ib = idx + b;
                long int fb = ib * 3;
                long int nb0 = ib * num_neighbors * 3;
                double pxb = pos[fb];
                double pyb = pos[fb + 1];
                double pzb = pos[fb + 2];
                double qb = charges[ib];

                for (int j1 = 0; j1 < num_neighbors; j1++) {
                    long int i1 = na0 + j1 * 3;
                    long int ni = p->grid_neighbors[i1];
                    long int nj = p->grid_neighbors[i1 + 1];
                    long int nk = p->grid_neighbors[i1 + 2];

                    double wx1 = p->charges_spread_func(pxa - ni * h, L, h);
                    double wy1 = p->charges_spread_func(pya - nj * h, L, h);
                    double wz1 = p->charges_spread_func(pza - nk * h, L, h);
                    double qg1 = qa * wx1 * wy1 * wz1;
                    if (qg1 == 0.0) {
                        continue;
                    }

                    for (int j2 = 0; j2 < num_neighbors; j2++) {
                        long int i2 = nb0 + j2 * 3;
                        long int ni2 = p->grid_neighbors[i2];
                        long int nj2 = p->grid_neighbors[i2 + 1];
                        long int nk2 = p->grid_neighbors[i2 + 2];

                        if (ni == ni2 && nj == nj2 && nk == nk2) {
                            continue;
                        }

                        double wx2 = p->charges_spread_func(pxb - ni2 * h, L, h);
                        double wy2 = p->charges_spread_func(pyb - nj2 * h, L, h);
                        double wz2 = p->charges_spread_func(pzb - nk2 * h, L, h);
                        double qg2 = qb * wx2 * wy2 * wz2;
                        if (qg2 == 0.0) {
                            continue;
                        }

                        dx = (ni - ni2) * h;
                        dy = (nj - nj2) * h;
                        dz = (nk - nk2) * h;

                        // Minimum image convention in real space.
                        dx -= L * nearbyint(dx / L);
                        dy -= L * nearbyint(dy / L);
                        dz -= L * nearbyint(dz / L);

                        r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 == 0.0) {
                            continue;
                        }
                        r = sqrt(r2);

                        force_mag = -(qg1 * qg2) / (r2 * r);
                        energy_corr -= (qg1 * qg2) / r;

                        fcs[fa    ] += force_mag * dx;
                        fcs[fa + 1] += force_mag * dy;
                        fcs[fa + 2] += force_mag * dz;
                        fcs[fb    ] -= force_mag * dx;
                        fcs[fb + 1] -= force_mag * dy;
                        fcs[fb + 2] -= force_mag * dz;
                    }
                }
            }
        }
    }

    p->energy_corr = energy_corr;
    return p->energy_corr;
}

// TODO: remove and leave only the non-self
double particles_compute_forces_electrostatic_correction_spread_self(particles *p, grid *g) {
    // Start from intramolecular spread correction (O-H1, O-H2, H1-H2).
    particles_compute_forces_electrostatic_correction_spread(p, g);

    long int n_p = p->n_p;
    long int n3 = n_p * 3;
    double *pos = p->pos;
    double *charges = p->charges;
    double *fcs = p->fcs_corr;
    double L = p->L;
    double h = g->h;
    double energy_corr = p->energy_corr;

    int num_neighbors = p->num_neighbors;

    // Add same-atom chargelet-chargelet contributions (j1 != j2).
    #pragma omp parallel for reduction(+:energy_corr, fcs[:n3])
    for (long int ia = 0; ia < n_p; ia++) {
        long int fa = ia * 3;
        long int na0 = ia * num_neighbors * 3;
        double pxa = pos[fa];
        double pya = pos[fa + 1];
        double pza = pos[fa + 2];
        double qa = charges[ia];

        for (int j1 = 0; j1 < num_neighbors; j1++) {
            long int i1 = na0 + j1 * 3;
            long int ni = p->grid_neighbors[i1];
            long int nj = p->grid_neighbors[i1 + 1];
            long int nk = p->grid_neighbors[i1 + 2];

            double wx1 = p->charges_spread_func(pxa - ni * h, L, h);
            double wy1 = p->charges_spread_func(pya - nj * h, L, h);
            double wz1 = p->charges_spread_func(pza - nk * h, L, h);
            double qg1 = qa * wx1 * wy1 * wz1;
            if (qg1 == 0.0) {
                continue;
            }

            for (int j2 = j1 + 1; j2 < num_neighbors; j2++) {
                long int i2 = na0 + j2 * 3;
                long int ni2 = p->grid_neighbors[i2];
                long int nj2 = p->grid_neighbors[i2 + 1];
                long int nk2 = p->grid_neighbors[i2 + 2];

                double wx2 = p->charges_spread_func(pxa - ni2 * h, L, h);
                double wy2 = p->charges_spread_func(pya - nj2 * h, L, h);
                double wz2 = p->charges_spread_func(pza - nk2 * h, L, h);
                double qg2 = qa * wx2 * wy2 * wz2;
                if (qg2 == 0.0) {
                    continue;
                }

                double dx = (ni - ni2) * h;
                double dy = (nj - nj2) * h;
                double dz = (nk - nk2) * h;
                dx -= L * nearbyint(dx / L);
                dy -= L * nearbyint(dy / L);
                dz -= L * nearbyint(dz / L);

                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 == 0.0) {
                    continue;
                }
                double r = sqrt(r2);
                double force_mag = -(qg1 * qg2) / (r2 * r);

                fcs[fa    ] += force_mag * dx;
                fcs[fa + 1] += force_mag * dy;
                fcs[fa + 2] += force_mag * dz;
                energy_corr -= (qg1 * qg2) / r;
            }
        }
    }

    p->energy_corr = energy_corr;
    return p->energy_corr;
}

double particles_compute_forces_electrostatic_correction_sr(particles *p, grid *g) {
    if (!p->is_water) {
        p->energy_corr = 0.0;
        return p->energy_corr;
    }

    long int n_p = p->n_p;
    long int n3 = n_p * 3;
    long int n_triplets = n_p / 3;
    double *pos = p->pos;
    double *fcs = p->fcs_corr;
    double L = p->L;
    double energy_corr = 0.0;

    memset(fcs, 0, n3 * sizeof(double));

    // Utility for minimum image
    #define MIN_IMG(d) (d -= L * nearbyint(d / L))

    #pragma omp parallel for reduction(+:energy_corr, fcs[:n3])
    for (long int m = 0; m < n_p / 3; m++) {
        long int iO = m * 3;
        double qO = p->charges[iO];
        double qH1 = p->charges[iO + 1];
        double qH2 = p->charges[iO + 2];

        long int o3 = iO * 3;
        long int h13 = (iO + 1) * 3;
        long int h23 = (iO + 2) * 3;

        // Vectors O->H1 and O->H2 with PBC
        double r1x = pos[h13    ] - pos[o3    ];
        double r1y = pos[h13 + 1] - pos[o3 + 1];
        double r1z = pos[h13 + 2] - pos[o3 + 2];
        MIN_IMG(r1x); MIN_IMG(r1y); MIN_IMG(r1z);

        double r2x = pos[h23    ] - pos[o3    ];
        double r2y = pos[h23 + 1] - pos[o3 + 1];
        double r2z = pos[h23 + 2] - pos[o3 + 2];
        MIN_IMG(r2x); MIN_IMG(r2y); MIN_IMG(r2z);

        double r1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + 1e-15;
        double r2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + 1e-15;

        double hhx = r2x - r1x;
        double hhy = r2y - r1y;
        double hhz = r2z - r1z;
        MIN_IMG(hhx); MIN_IMG(hhy); MIN_IMG(hhz);

        double inv_r, inv_r3, fac;

        // O-H1
        inv_r = 1.0 / r1;
        inv_r3 = inv_r * inv_r * inv_r;
        fac = -qO * qH1 * inv_r3;
        fcs[o3    ] -= fac * r1x;
        fcs[o3 + 1] -= fac * r1y;
        fcs[o3 + 2] -= fac * r1z;
        fcs[h13    ] += fac * r1x;
        fcs[h13 + 1] += fac * r1y;
        fcs[h13 + 2] += fac * r1z;
        energy_corr -= qO * qH1 * inv_r;

        // O-H2
        inv_r = 1.0 / r2;
        inv_r3 = inv_r * inv_r * inv_r;
        fac = -qO * qH2 * inv_r3;
        fcs[o3    ] -= fac * r2x;
        fcs[o3 + 1] -= fac * r2y;
        fcs[o3 + 2] -= fac * r2z;
        fcs[h23    ] += fac * r2x;
        fcs[h23 + 1] += fac * r2y;
        fcs[h23 + 2] += fac * r2z;
        energy_corr -= qO * qH2 * inv_r;

        // H1-H2
        inv_r = 1.0 / (sqrt(hhx * hhx + hhy * hhy + hhz * hhz) + 1e-15);
        inv_r3 = inv_r * inv_r * inv_r;
        fac = -qH1 * qH2 * inv_r3;
        fcs[h13    ] -= fac * hhx;
        fcs[h13 + 1] -= fac * hhy;
        fcs[h13 + 2] -= fac * hhz;
        fcs[h23    ] += fac * hhx;
        fcs[h23 + 1] += fac * hhy;
        fcs[h23 + 2] += fac * hhz;
        energy_corr -= qH1 * qH2 * inv_r;
    }

    #undef MIN_IMG

    p->energy_corr = energy_corr;
    return p->energy_corr;
}

// double particles_compute_forces_electrostatic_correction_maze(particles *p, grid *g) {
//     if (!p->is_water) {
//         p->energy_corr = 0.0;
//         return p->energy_corr;
//     }

//     int rank = 0;
//     int size_mpi = 1;
// #ifdef __MPI
//     size_mpi = get_size();
//     if (size_mpi < 1) {
//         size_mpi = 1;
//     }
//     rank = get_rank();
// #endif

//     long int size = p->n_p * 3;
//     if (p->fcs_corr == NULL) {
//         p->fcs_corr = (double *)calloc(size, sizeof(double));
//     } else {
//         memset(p->fcs_corr, 0, size * sizeof(double));
//     }

//     long int np = p->n_p;
//     double energy_corr = 0.0;

//     double *forces_tmp = (double *)calloc(size, sizeof(double));
//     double *charges_tmp = (double *)calloc(np, sizeof(double));
//     if (forces_tmp == NULL || charges_tmp == NULL) {
//         mpi_fprintf(stderr, "Error: Failed to allocate temporary buffers for electrostatic correction.\n");
//         free(forces_tmp);
//         free(charges_tmp);
//         return 0.0;
//     }

//     long int grid_size = g->size;
//     double *q_save = (double *)malloc(grid_size * sizeof(double));
//     double *phi_n_save = (double *)malloc(grid_size * sizeof(double));
//     double *phi_p_save = NULL;
//     if (g->phi_p != NULL) {
//         phi_p_save = (double *)malloc(grid_size * sizeof(double));
//     }
//     if (q_save == NULL || phi_n_save == NULL || (g->phi_p != NULL && phi_p_save == NULL)) {
//         mpi_fprintf(stderr, "Error: Failed to allocate grid save buffers for electrostatic correction.\n");
//         free(forces_tmp);
//         free(charges_tmp);
//         free(q_save);
//         free(phi_n_save);
//         free(phi_p_save);
//         return 0.0;
//     }

//     // Save global grid state; it will be restored after per-molecule correction.
//     memcpy(q_save, g->q, grid_size * sizeof(double));
//     memcpy(phi_n_save, g->phi_n, grid_size * sizeof(double));
//     if (g->phi_p != NULL) {
//         memcpy(phi_p_save, g->phi_p, grid_size * sizeof(double));
//     }

//     // Ensure neighbors are up to date before looping.
//     p->update_grid_nearest_neighbors(p);

//     double *charges_orig = p->charges;
//     for (long int m = 0; m < np / 3; m++) {
//         if (size_mpi > 1 && (m % size_mpi) != rank) {
//             continue;
//         }
//         long int iO = m * 3;

//         memset(charges_tmp, 0, np * sizeof(double));
//         charges_tmp[iO] = charges_orig[iO];
//         charges_tmp[iO + 1] = charges_orig[iO + 1];
//         charges_tmp[iO + 2] = charges_orig[iO + 2];

//         p->charges = charges_tmp;

//         g->update_charges(g, p);
//         // Use plain multigrid field update for correction (no MAZE state dependency).
//         multigrid_grid_init_field(g);
//         int corr_field_res = multigrid_grid_update_field(g);
//         if (corr_field_res == -1) {
//             int rank = get_rank();
//             if (rank == 0) {
//                 FILE *fp = solver_open_not_converged_log();
//                 if (fp != NULL) {
//                     fprintf(fp,
//                             "electrostatic correction did not converge for molecule %ld/%ld; correction skipped\n",
//                             m, np / 3);
//                     fclose(fp);
//                 }
//             }
//             continue;
//         }

//         compute_force_fd(
//             g->n, p->n_p, g->h, p->num_neighbors,
//             g->phi_n, p->neighbors, p->charges, p->pos, forces_tmp,
//             p->charges_spread_func
//         );

//         // Subtractive correction: self-field contribution.
//         daxpy(forces_tmp, p->fcs_corr, -1.0, size);
//     }

//     p->charges = charges_orig;

//     // Restore global grid state to avoid altering the main simulation fields.
//     memcpy(g->q, q_save, grid_size * sizeof(double));
//     memcpy(g->phi_n, phi_n_save, grid_size * sizeof(double));
//     if (g->phi_p != NULL) {
//         memcpy(g->phi_p, phi_p_save, grid_size * sizeof(double));
//     }

//     free(q_save);
//     free(phi_n_save);
//     free(phi_p_save);
//     free(forces_tmp);
//     free(charges_tmp);

//     #ifdef __MPI
//     if (size_mpi > 1) {
//         allreduce_sum(p->fcs_corr, np * 3);
//         allreduce_sum(&energy_corr, 1);
//     }
//     #endif

//     p->energy_corr = energy_corr;
//     return p->energy_corr;
// }


double calc_h_ratio(double rad, double w2, double w3) {
    return (
        (
            -(3 / (4 * w3)) * pow(rad, 2) +
             (3 / (2 * w2)) * rad
        ) /
        (
            -(1 / (4 * w3)) * pow(rad, 3) +
             (3 / (4 * w2)) * pow(rad, 2)
        ) 
    );
}

double particles_compute_forces_pb_roux(particles *p, grid *g) {
    if (! g->pb_enabled) {
        // Poisson-Boltzmann is not enabled
        mpi_fprintf(stderr, "Poisson-Boltzmann forces are not enabled in the grid.\n");
        exit(1);
    }
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
    double h3 = h * h * h;  // Cube of the grid spacing

    // double S = 0.0;  // Sum of the derivatives of the dielectric constant
    double non_polar_energy = 0.0;

    long int size = p->n_p * 3;
    double *fcs_db = p->fcs_db;
    double *fcs_ib = p->fcs_ib;
    double *fcs_np = p->fcs_np;

    memset(fcs_db, 0, size * sizeof(double));  // Initialize forces to zero
    memset(fcs_ib, 0, size * sizeof(double));  // Initialize forces to zero
    memset(fcs_np, 0, size * sizeof(double));  // Initialize non-polar forces to zero

    mpi_grid_exchange_bot_top(g->phi_n, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_x, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_y, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_z, n_local, n);

    #pragma \
        omp parallel for private(r_solv, px, py, pz, idx_x, idx_y, idx_z) \
        reduction(+:non_polar_energy, fcs_db[:size], fcs_ib[:size], fcs_np[:size])
    for (int np = 0; np < p->n_p; np++) {
        int np3 = np * 3;

        r_solv = p->solv_radii[np];
        px = p->pos[np3];
        py = p->pos[np3 + 1];
        pz = p->pos[np3 + 2];

        double r1, r2;
        double r_solv_p2 = pow(r_solv + w, 2);
        double r_solv_m2 = pow(r_solv - w, 2);

        int idx_range = (int)floor((r_solv + w) / h) + 2;

        idx_x = (int)floor(px / h);
        idx_y = (int)floor(py / h);
        idx_z = (int)floor(pz / h);

        double dx, dy, dz;
        double dx2, dy2, dz2;
        double rx, ry, rz;  // Relative coordinates
        double app1, app2;

        int i0, j0, k0, i1, i2, j1, j2, k1, k2;
        long int idx_cen;
        long int idx_bwd_x, idx_bwd_y, idx_bwd_z;
        long int idx_fwd_x, idx_fwd_y, idx_fwd_z;
        
        double phi_center, phi_bwd, phi_fwd, delta_phi;
        double h_ratio;

        double d_eps_x, d_eps_y, d_eps_z, d_eps_norm, inv_grad;

        double eps_x_cen, eps_y_cen, eps_z_cen;
        double eps_x_bwd, eps_y_bwd, eps_z_bwd;

        for (int di = -idx_range; di <= idx_range; di++) {
            i0 = idx_x + di;
            dx = px - i0 * h;  // Calculate the distance in x direction
            dx2 = dx * dx;
            i0 = ((i0 + n) % n);  // Wrap around for periodic boundary conditions
            i0 -= n_start;  // Adjust for local grid start
            if (i0 < 0 || i0 >= n_local) { 
                continue;  // Skip if the point is outside the local grid
            }
            i0 *= n2;  // Convert to linear index
            i1 = i0 + n2;
            i2 = i0 - n2;
            for (int dj = -idx_range; dj <= idx_range; dj++) {
                j0 = idx_y + dj;
                dy = py - j0 * h;  // Calculate the distance in y direction
                dy2 = dy * dy;
                j0 = (j0 + n) % n;  // Wrap around for periodic boundary conditions
                j1 = ((j0 + 1) % n) * n;
                j2 = ((j0 - 1 + n) % n) * n;  // Wrap around for periodic boundary conditions
                j0 *= n;
                for (int dk = -idx_range; dk <= idx_range; dk++) {
                    k0 = idx_z + dk;
                    dz = pz - k0 * h;  // Calculate the distance in z direction
                    dz2 = dz * dz;
                    k0 = (k0 + n) % n;  // Wrap around for periodic boundary conditions
                    k1 = (k0 + 1) % n;
                    k2 = (k0 - 1 + n) % n;  // Wrap around for periodic boundary conditions

                    idx_cen   = i0 + j0 + k0;  // Calculate the index in the grid
                    idx_fwd_x = i1 + j0 + k0;
                    idx_fwd_y = i0 + j1 + k0;
                    idx_fwd_z = i0 + j0 + k1;
                    idx_bwd_x = i2 + j0 + k0;
                    idx_bwd_y = i0 + j2 + k0;
                    idx_bwd_z = i0 + j0 + k2;

                    eps_x_cen = g->eps_x[idx_cen];
                    eps_y_cen = g->eps_y[idx_cen];
                    eps_z_cen = g->eps_z[idx_cen];
                    eps_x_bwd = g->eps_x[idx_bwd_x];
                    eps_y_bwd = g->eps_y[idx_bwd_y];
                    eps_z_bwd = g->eps_z[idx_bwd_z];

                    d_eps_x = eps_x_cen - eps_x_bwd;
                    d_eps_y = eps_y_cen - eps_y_bwd;
                    d_eps_z = eps_z_cen - eps_z_bwd;
                    d_eps_norm = sqrt(d_eps_x * d_eps_x + d_eps_y * d_eps_y + d_eps_z * d_eps_z) / h;
                    // S += d_eps_norm;
                    inv_grad = (d_eps_norm > 0.0) ? 1.0 / d_eps_norm : 0.0;

                    phi_center = g->phi_n[idx_cen];

                    r2 = dx2 + dy2 + dz2;
                    // *************** CENTER***************
                    if (r2 >= r_solv_p2) {
                        // Outside the radius, do nothing
                    } else if (r2 > r_solv_m2) {
                        // Inside the transition region, set dielectric constant to a fraction
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        // *********************** Ionic boundary forces ***********************
                        app1 = g->k2[idx_cen] * phi_center * phi_center * h_ratio;
                        fcs_ib[np3    ] += app1 * rx;
                        fcs_ib[np3 + 1] += app1 * ry;
                        fcs_ib[np3 + 2] += app1 * rz;

                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** X - h/2 ***************
                    app1 = dx + hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    app2 = inv_grad * d_eps_x;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = app1 / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_x_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_x] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** X + h/2 ***************
                    app1 = dx - hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = app1 / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_x_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_x] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Y - h/2 ***************
                    app1 = dy + hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    app2 = inv_grad * d_eps_y;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = app1 / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_y_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_y] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Y + h/2 ***************
                    app1 = dy - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = app1 / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_y_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_y] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Z - h/2 ***************
                    app1 = dz + hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    app2 = inv_grad * d_eps_z;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = app1 / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_z_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_z] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Z + h/2 ***************
                    app1 = dz - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = app1 / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_z_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_z] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }
                }
            }
        }
    }

    allreduce_sum(fcs_db, size);
    allreduce_sum(fcs_ib, size);
    allreduce_sum(fcs_np, size);

    dscal(fcs_db, h / (8.0 * M_PI), size);
    dscal(fcs_ib, h / (8.0 * M_PI), size);

    if (g->nonpolar_enabled && fabs(p->gamma_np) > 0.0) {
        double eps_delta = eps_s - eps_int;
        if (fabs(eps_delta) < 1e-12) {
            mpi_fprintf(
                stderr,
                "Invalid Poisson-Boltzmann setup: nonpolar forces require eps_s != eps_int.\n"
            );
            exit(1);
        }
        dscal(fcs_np, -p->gamma_np * h / eps_delta, size);
    }

    return non_polar_energy;
}

double particles_compute_forces_pb_stress_tensor(particles *p, grid *g) {
    long int size = p->n_p * 3;
    double non_polar_energy = 0.0;

    if (g->nonpolar_enabled) {
        // Call the Roux method directly to avoid infinite recursion:
        // particles_compute_forces_pb would re-dispatch to this function.
        non_polar_energy = particles_compute_forces_pb_roux(p, g);
    } else {
        memset(p->fcs_np, 0, size * sizeof(double));
    }

    compute_stress_tensor_forces(
        g->n, g->eps_s, p->n_p, g->L, g->h,
        g->phi_n, g->region, p->pos, p->solv_radii, p->fcs_elec,
        g->stress_tensor_bc_type == STRESS_TENSOR_BC_TYPE_PBC
    );

    memset(p->fcs_db, 0, size * sizeof(double));
    memset(p->fcs_ib, 0, size * sizeof(double));
    return non_polar_energy;
}

double particles_compute_forces_pb(particles *p, grid *g) {
    switch (g->pb_force_type)
    {
        case MAP_NOT_INITIALIZED:
            mpi_fprintf(stderr, "Poisson-Boltzmann force type not initialized. Please call `grid_pb_init` before running Poisson-Boltzmann related functions.\n");
            exit(1);
        case PB_FORCE_TYPE_PB_ROUX:
            return particles_compute_forces_pb_roux(p, g);
        case PB_FORCE_TYPE_STRESS_TENSOR:
            return particles_compute_forces_pb_stress_tensor(p, g);
        default:
            mpi_fprintf(stderr, "Invalid Poisson-Boltzmann force type specified.\n");
            exit(1);
    }
}

void particles_compute_forces_tot(particles *p) {
    int size = p->n_p * 3;
    memset(p->fcs_tot, 0, size * sizeof(double));  // Initialize total forces to zero
    if (p->fcs_elec != NULL) {
        daxpy(p->fcs_elec, p->fcs_tot, 1.0, size);  // Add the electric forces
    }
    if (p->fcs_noel != NULL) {
        daxpy(p->fcs_noel, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_db != NULL) {
        daxpy(p->fcs_db, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_ib != NULL) {
        daxpy(p->fcs_ib, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_np != NULL) {
        daxpy(p->fcs_np, p->fcs_tot, 1.0, size);
    }
    if (p->is_water && p->fcs_intra != NULL && p->fcs_corr != NULL) {
        daxpy(p->fcs_intra, p->fcs_tot, 1.0, size);
        // fcs_corr is stored with negative sign; add it to subtract the correction.
        daxpy(p->fcs_corr, p->fcs_tot, 1.0, size);
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

// works only for 2 species with same population, below is a more general version that does not assume that
// void particles_rescale_velocities(particles *p) {
//     double *init_vel = (double *)calloc(p->n_typ * 3, sizeof(double));

//     for (int i = 0; i < p->n_p; i++) {
//         for (int j = 0; j < 3; j++) {
//             init_vel[p->types[i] * 3 + j] += p->vel[i * 3 + j];
//         }
//     }

//     for (int i = 0; i < p->n_p; i++) {
//         for (int j = 0; j < 3; j++) {
//             p->vel[i * 3 + j] -= 2 * init_vel[p->types[i] * 3 + j] / p->n_p;
//         }
//     }

//     free(init_vel);
// }

// removes average velocity per species
void particles_rescale_velocities(particles *p) {
    double *init_vel = (double *)calloc(p->n_typ * 3, sizeof(double));
    int *type_counts = (int *)calloc(p->n_typ, sizeof(int));

    if (init_vel == NULL || type_counts == NULL) {
        mpi_fprintf(stderr, "Failed to allocate buffers for velocity rescaling\n");
        exit(1);
    }

    for (int i = 0; i < p->n_p; i++) {
        int type = p->types[i];
        if (type < 0 || type >= p->n_typ) {
            mpi_fprintf(stderr, "Invalid particle type %d for particle %d\n", type, i);
            exit(1);
        }

        type_counts[type]++;
        for (int j = 0; j < 3; j++) {
            init_vel[type * 3 + j] += p->vel[i * 3 + j];
        }
    }
    
    for (int i = 0; i < p->n_p; i++) {
        int type = p->types[i];
        for (int j = 0; j < 3; j++) {
            p->vel[i * 3 + j] -= init_vel[type * 3 + j] / type_counts[type];
        }
    }

    free(init_vel);
    free(type_counts);
}

// removes averge velocity of the center of mass - works for any number of species and populations
void particles_rescale_momenta(particles *p) {
    long int ni;
    double px = 0.0, py = 0.0, pz = 0.0;
    double m_tot = 0.0;

    #pragma omp parallel for private(ni) reduction(+:px, py, pz, m_tot)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        px += p->mass[i] * p->vel[ni];
        py += p->mass[i] * p->vel[ni + 1];
        pz += p->mass[i] * p->vel[ni + 2];
        m_tot += p->mass[i];
    }

    if (m_tot <= 0.0) {
        return;
    }

    double vx_cm = px / m_tot;
    double vy_cm = py / m_tot;
    double vz_cm = pz / m_tot;

    #pragma omp parallel for private(ni)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        p->vel[ni]     -= vx_cm;
        p->vel[ni + 1] -= vy_cm;
        p->vel[ni + 2] -= vz_cm;
    }

    px = 0.0;
    py = 0.0;
    pz = 0.0;
    #pragma omp parallel for private(ni) reduction(+:px, py, pz)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        px += p->mass[i] * p->vel[ni];
        py += p->mass[i] * p->vel[ni + 1];
        pz += p->mass[i] * p->vel[ni + 2];
    }
    mpi_printf("total momentum after rescale: %e %e %e\n", px, py, pz);
}

void particles_zero_linear(particles *p) {
    double px = 0.0, py = 0.0, pz = 0.0;
    double m_tot = 0.0;

    // somma momento totale e massa totale
    #pragma omp parallel for reduction(+:px,py,pz,m_tot)
    for (int i = 0; i < p->n_p; i++) {
        long int vi = i * 3;
        double m = p->mass[i];
        px += m * p->vel[vi];
        py += m * p->vel[vi + 1];
        pz += m * p->vel[vi + 2];
        m_tot += m;
    }

    if (m_tot <= 0.0) {
        return;
    }

    double vx_cm = px / m_tot;
    double vy_cm = py / m_tot;
    double vz_cm = pz / m_tot;

    // sottrai la velocità COM a tutti gli atomi
    #pragma omp parallel for
    for (int i = 0; i < p->n_p; i++) {
        long int vi = i * 3;
        p->vel[vi]     -= vx_cm;
        p->vel[vi + 1] -= vy_cm;
        p->vel[vi + 2] -= vz_cm;
    }

    // opzionale: stampa momento totale dopo la correzione
    double px2 = 0.0, py2 = 0.0, pz2 = 0.0;
    #pragma omp parallel for reduction(+:px2,py2,pz2)
    for (int i = 0; i < p->n_p; i++) {
        long int vi = i * 3;
        double m = p->mass[i];
        px2 += m * p->vel[vi];
        py2 += m * p->vel[vi + 1];
        pz2 += m * p->vel[vi + 2];
    }
    mpi_printf("total momentum after zero linear: %e %e %e\n", px2, py2, pz2);
}
