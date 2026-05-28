#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mp_structs.h"
#include "mpi_base.h"
#include "omp_base.h"
#include <stdbool.h>

#define MAX_ITG_PARAMS 10

particles *g_particles = NULL;
integrator *g_integrator = NULL;
grid *g_grid = NULL;

static double solver_total_charge_from_particles(void) {
    double q_local = 0.0;
    int n_p = g_particles->n_p;
    for (int i = 0; i < n_p; i++) {
        q_local += g_particles->charges[i];
    }

    return q_local;
}

void solver_initialize() {
    int size = init_mpi();
    int rank = get_rank();

    int n_threads = get_omp_max_threads();

    mpi_printf("******************************************************\n");
    mpi_printf("* MAZE_POISSON\n");
    if (size > 0) {
        mpi_printf("*   MPI     enabled  running on %d processes\n", size);
    } else {
        mpi_printf("*   MPI     not available\n");
    }
    if (n_threads > 0) {
        mpi_printf("*   OpenMP  enabled  running on %d threads\n", n_threads);
    } else {
        mpi_printf("*   OpenMP  not available\n");
    }
    mpi_printf("******************************************************\n");
}

void solver_initialize_grid(
    int n_grid, double L, double h, double tol, double eps, double eps_int, int grid_type, int precond_type
) {
    g_grid = grid_init(n_grid, L, h, tol, eps, eps_int, grid_type, precond_type);
}

void solver_initialize_grid_pois_boltz(double w, double kbar2, int nonpolar_enabled) {
    // Initialize the solvent potential and dielectric constant arrays
    grid_pb_init(g_grid, w, kbar2, nonpolar_enabled);
}

void solver_initialize_grid_smoothing(int method, double r_cut, double sigma) {
    // Initialize the charge smoothing to perform after charge assignment
    grid_smoothing_init(g_grid, method, r_cut, sigma);
}

void solver_initialize_particles(
    int n, int n_typ, double L, double h, int n_p, int pot_type, int cas_type,
    int *types, double *pos, double *vel, double *mass, double *charges,
    double *pot_params, double r_cut, int lj_force_shift
) {
    // TODO: Should R_c and r_cut be the same? If not the variable name should be more clearer/descriptive
    g_particles = particles_init(n, n_p, n_typ, L, h, cas_type);

    g_particles->r_cut = r_cut;
    g_particles->lj_force_shift = lj_force_shift;

    memcpy(g_particles->types, types, n_p * sizeof(int));
    memcpy(g_particles->pos, pos, n_p * 3 * sizeof(double));
    memcpy(g_particles->vel, vel, n_p * 3 * sizeof(double));
    memcpy(g_particles->mass, mass, n_p * sizeof(double));
    memcpy(g_particles->charges, charges, n_p * sizeof(double));

    g_particles->init_potential(g_particles, pot_type, pot_params);
}

void solver_initialize_particles_pois_boltz(double gamma_np, double beta_np, double *solv_radii) {
    particles_pb_init(g_particles, gamma_np, beta_np, solv_radii);
}

void solver_initialize_particles_water(int is_water, int corr_type) {
    particles_water_init(g_particles, is_water, corr_type);
}

void solver_initialize_integrator(int n_p, double dt, double T, double gamma, int itg_type, int itg_enabled) {
    g_integrator = integrator_init(n_p, dt, itg_type);

    double itg_params[MAX_ITG_PARAMS];
    itg_params[0] = T;
    switch (itg_type) {
        case INTEGRATOR_TYPE_OVRVO:
            itg_params[1] = gamma;
            break;
        case INTEGRATOR_TYPE_VERLET:
            break;
        default:
            break;
    }

    if (itg_enabled == 1) {
        g_integrator->init_thermostat(g_integrator, itg_params);
    }
}

int solver_update_charges() {
    int res = 0;
    double q_tot_loc;
    double q_ref;
    
    g_particles->update_nearest_neighbors(g_particles);
    q_tot_loc = g_grid->update_charges(g_grid, g_particles);

    q_ref = solver_total_charge_from_particles();

    double diff = fabs(q_ref - q_tot_loc);
    if (diff > 1e-4) {
        if (diff > 1e-2) {
            res = 1;
            printf("Charge conservation error: q_ref = %.6f, q_tot_loc = %.6f\n", q_ref, q_tot_loc);
            exit(1);
        } else {
            mpi_printf("Charge conservation warning: q_ref = %.6f, q_tot_loc = %.6f\n", q_ref, q_tot_loc);
        }
    }

    return res;
}

void solver_smoothing() {
    g_grid->smooth_charges(g_grid, g_particles);
}

void solver_init_field() {
    g_grid->init_field(g_grid);
}

void solver_set_field(double *phi) {
    mpi_grid_distribute_buffer(g_grid->phi_n, phi, g_grid->n);
}

void solver_set_field_prev(double *phi) {
    if (g_grid->phi_p != NULL) {
        mpi_grid_distribute_buffer(g_grid->phi_p, phi, g_grid->n);
    }
}

int solver_update_field() {
    return g_grid->update_field(g_grid);
}

void solver_update_eps_k2() {
    // Update the dielectric constant and screening factor based on the grid's transition state
    grid_update_eps_and_k2(g_grid, g_particles);
}

double solver_compute_forces_elec() {
    return g_particles->compute_forces_field(g_particles, g_grid);
}

double solver_compute_forces_noel() {
    return g_particles->compute_forces_noel(g_particles);
}

double solver_compute_forces_pb() {
    return g_particles->compute_forces_pb(g_particles, g_grid);
}

double solver_compute_intramolecular_forces() {
    return g_particles->compute_intramolecular_forces(g_particles);
}

double solver_compute_forces_electrostatic_correction() {
    return g_particles->compute_forces_electrostatic_correction(g_particles, g_grid);
}

void solver_compute_forces_tot() {
    g_particles->compute_forces_tot(g_particles);
}

double get_energy_intra() {
    return g_particles->energy_intra;
}

double get_energy_corr() {
    return g_particles->energy_corr;
}

// void solver_compute_forces() {
//     solver_compute_forces_elec();
//     solver_compute_forces_noel();
//     solver_compute_forces_tot();
// }

void integrator_part_1() {
    g_integrator->part1(g_integrator, g_particles);
}

void integrator_part_2() {
    g_integrator->part2(g_integrator, g_particles);
}

void solver_rescale_velocities() {
    g_particles->rescale_velocities(g_particles);
    // g_particles->rescale_momenta(g_particles);
}

// int solver_initialize_md(int preconditioning, int vel_rescale) {
//     int res = 0;

//     #pragma omp parallel for reduction(+:q_tot)
//     for (int i = 0; i < g_particles->n_p; i++) {
//         q_tot += g_particles->charges[i];
//     }

//     // Step 0 Verlet
//     res |= solver_update_charges();
//     if (preconditioning == 1)
//         solver_init_field();
//     solver_compute_forces();

//     // Step 1 Verlet
//     integrator_part_1();
//     res |= solver_update_charges();
//     if (preconditioning == 1) 
//         solver_init_field();
//     solver_compute_forces();
//     integrator_part_2();

//     if (vel_rescale == 1) 
//         solver_rescale_velocities();

//     return res;
// }

// void solver_md_loop_iter() {
//     integrator_part_1();
//     solver_update_charges();
//     solver_update_field();
//     solver_compute_forces();
//     integrator_part_2();
// }

int solver_check_thermostat() {
    int res = 0;
    double temp;
    if (g_integrator->enabled == INTEGRATOR_ENABLED) {
        temp = g_particles->get_temperature(g_particles);
        if (fabs(temp - g_integrator->T) < 100) {
            res = 1;
            g_integrator->stop_thermostat(g_integrator);
        }
    }

    return res;
}

// void solver_run_n_steps(int n_steps) {
//     for (int i = 0; i < n_steps; i++) {
//         solver_md_loop_iter();
//     }
// }

void solver_finalize() {
    if (g_particles != NULL) {
        g_particles->free(g_particles);
        g_particles = NULL;
    }
    if (g_grid != NULL) {
        g_grid->free(g_grid);
        g_grid = NULL;
    }
    if (g_integrator != NULL) {
        g_integrator->free(g_integrator);
        g_integrator = NULL;
    }

    cleanup_mpi();
}

void get_pos(double *recv) {
    memcpy(recv, g_particles->pos, g_particles->n_p * 3 * sizeof(double));
}

void get_vel(double *recv) {
    memcpy(recv, g_particles->vel, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_elec(double *recv) {
    memcpy(recv, g_particles->fcs_elec, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_noel(double *recv) {
    memcpy(recv, g_particles->fcs_noel, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_intra(double *recv) {
    if (g_particles->fcs_intra != NULL) {
        memcpy(recv, g_particles->fcs_intra, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_corr(double *recv) {
    if (g_particles->fcs_corr != NULL) {
        memcpy(recv, g_particles->fcs_corr, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_db(double *recv) {
    if (g_particles->fcs_db != NULL) {
        memcpy(recv, g_particles->fcs_db, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_ib(double *recv) {
    if (g_particles->fcs_ib != NULL) {
        memcpy(recv, g_particles->fcs_ib, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_np(double *recv) {
    if (g_particles->fcs_np != NULL) {
        memcpy(recv, g_particles->fcs_np, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_tot(double *recv) {
    memcpy(recv, g_particles->fcs_tot, g_particles->n_p * 3 * sizeof(double));
}

void get_types(int *recv) {
    memcpy(recv, g_particles->types, g_particles->n_p * sizeof(int));
}

void get_charges(double *recv) {
    memcpy(recv, g_particles->charges, g_particles->n_p * sizeof(double));
}

void get_masses(double *recv) {
    memcpy(recv, g_particles->mass, g_particles->n_p * sizeof(double));
}

void get_radii(double *recv) {
    if (g_particles->solv_radii != NULL) {
        memcpy(recv, g_particles->solv_radii, g_particles->n_p * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * sizeof(double));
    }
}

void get_field(double *recv) {
    mpi_grid_collect_buffer(g_grid->phi_n, recv, g_grid->n);
}

void get_field_prev(double *recv) {
    double *ptr = g_grid->phi_p !=  NULL ? g_grid->phi_p : g_grid->phi_n;
    mpi_grid_collect_buffer(ptr, recv, g_grid->n);
}

// void get_field_s(double *recv) {
//     if (g_grid->phi_s != NULL) {
//         mpi_grid_collect_buffer(g_grid->phi_s, recv, g_grid->n);
//     } else {
//         memset(recv, 0, g_grid->n * sizeof(double));
//     }
// }

// void get_phi_s_prev(double *recv) {
//     if (g_grid->phi_s_prev != NULL) {
//         mpi_grid_collect_buffer(g_grid->phi_s_prev, recv, g_grid->n);
//     } else {
//         memset(recv, 0, g_grid->n * sizeof(double));
//     }
// }

void get_q(double *recv) {
    mpi_grid_collect_buffer(g_grid->q, recv, g_grid->n);
}

void set_q(double *q_new) {
    mpi_grid_distribute_buffer(g_grid->q, q_new, g_grid->n);
}

double get_kinetic_energy() {
    return g_particles->get_kinetic_energy(g_particles);
}

double get_energy_elec() {
    return grid_get_energy_elec(g_grid);
}

void get_momentum(double *recv) {
    g_particles->get_momentum(g_particles, recv);
}

double get_temperature() {
    return g_particles->get_temperature(g_particles);
}
