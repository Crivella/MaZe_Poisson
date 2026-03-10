#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "debug_log.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "omp_base.h"

#define MAX_ITG_PARAMS 10

particles *g_particles = NULL;
integrator *g_integrator = NULL;
grid *g_grid = NULL;

double q_tot = 0.0;


static double solver_total_charge_from_particles(void) {
    double q_local = 0.0;
    int n_p = g_particles->n_p;
    for (int i = 0; i < n_p; i++) {
        q_local += g_particles->charges[i];
    }

// #ifdef __MPI
//     int size = get_size();
//     if (size > 1) {
//         double q_sum = q_local;
//         allreduce_sum(&q_sum, 1);

//         double q_max = q_local;
//         allreduce_max(&q_max, 1);

//         double q_min_neg = -q_local;
//         allreduce_max(&q_min_neg, 1);
//         double q_min = -q_min_neg;

//         if (fabs(q_max - q_min) < 1e-12) {
//             return q_max; // Particles replicated on each rank.
//         }
//         return q_sum; // Particles distributed across ranks.
//     }
// #endif

    return q_local;
}

static int using_cubic_spline_assignment(void) {
    return (g_particles != NULL && g_particles->cas_type == CHARGE_ASS_SCHEME_TYPE_SPLCUB);
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

void solver_initialize_particles(
    int n, int n_typ, double L, double h, int n_p, int pot_type, int cas_type,
    int *types, double *pos, double *vel, double *mass, double *charges,
    double *pot_params
) {
    // if (is_water) {
    //     if (n_p % 3 != 0) {
    //         mpi_fprintf(stderr, "Error: iswater=True but n_p=%d is not a multiple of 3.\n", n_p);
    //         exit(1);
    //     }
    //     int t0 = types[0];
    //     int t1 = types[1];
    //     int t2 = types[2];
    //     for (int m = 1; m < n_p / 3; m++) {
    //         int i = m * 3;
    //         if (types[i] != t0 || types[i + 1] != t1 || types[i + 2] != t2) {
    //             mpi_fprintf(
    //                 stderr,
    //                 "Error: iswater=True but atom ordering is inconsistent at molecule %d. "
    //                 "Expected repeating triplets [%d,%d,%d].\n",
    //                 m, t0, t1, t2
    //             );
    //             exit(1);
    //         }
    //     }
    // }

    g_particles = particles_init(n, n_p, n_typ, L, h, cas_type);
    // g_particles->is_water = is_water;
    // if (g_particles->is_water) {
    //     g_particles->fcs_intra = (double *)calloc(n_p * 3, sizeof(double));
    //     g_particles->fcs_corr = (double *)calloc(n_p * 3, sizeof(double));
    // }

    memcpy(g_particles->types, types, n_p * sizeof(int));
    memcpy(g_particles->pos, pos, n_p * 3 * sizeof(double));
    memcpy(g_particles->vel, vel, n_p * 3 * sizeof(double));
    memcpy(g_particles->mass, mass, n_p * sizeof(double));
    memcpy(g_particles->charges, charges, n_p * sizeof(double));

// #ifdef __MPI
//     int size = get_size();
//     if (size > 1) {
//         bcast_double(g_particles->pos, n_p * 3, 0);
//         bcast_double(g_particles->vel, n_p * 3, 0);
//         bcast_double(g_particles->mass, n_p, 0);
//         bcast_double(g_particles->charges, n_p, 0);
//     }
// #endif
    
    g_particles->init_potential(g_particles, pot_type, pot_params);
}

// void solver_set_electrostatic_correction(int corr_type) {
//     if (g_particles == NULL) {
//         return;
//     }
//     g_corr_type = corr_type;
//     switch (corr_type) {
//         case 0:
//             g_particles->compute_forces_electrostatic_correction =
//                 particles_compute_forces_electrostatic_correction_spread;
//             break;
//         case 1:
//             g_particles->compute_forces_electrostatic_correction =
//                 particles_compute_forces_electrostatic_correction_sr;
//             break;
//         default:
//             mpi_fprintf(stderr, "Invalid electrostatic correction type %d\n", corr_type);
//             exit(1);
//     }
// }

// static const char *corr_type_name(void) {
//     switch (g_corr_type) {
//         case 0:
//             return "SPREAD";
//         case 1:
//             return "SR";
//         default:
//             return "UNKNOWN";
//     }
// }

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
    charge_update_calls += 1;
    
    g_particles->update_nearest_neighbors(g_particles);
    q_tot_loc = g_grid->update_charges(g_grid, g_particles);

    q_ref = solver_total_charge_from_particles();
    q_tot = q_ref;

    double diff = fabs(q_ref - q_tot_loc);
    if (diff > 1e-4) {
        if (diff > 1e-2) {
            res = 1;
            printf("Charge conservation error: q_ref = %.6f, q_tot_loc = %.6f\n", q_ref, q_tot_loc);
            dump_charge_assignment_diagnostics(g_particles, g_grid, q_ref, q_tot_loc);
            dump_mpi_particle_consistency(g_particles);
            exit(1);
        } else {
            mpi_printf("Charge conservation warning: q_ref = %.6f, q_tot_loc = %.6f\n", q_ref, q_tot_loc);
            dump_mpi_particle_consistency(g_particles);
        }
    }

    if (using_cubic_spline_assignment()) {
        if (ensure_q_neighbor_history_buffers(g_particles) == 0) {
            if (!qn_hist_initialized) {
                capture_q_neighbor_snapshot(g_particles, qn_hist_curr, ng_hist_curr);
                memcpy(qn_hist_prev, qn_hist_curr, qn_hist_size * sizeof(double));
                memcpy(ng_hist_prev, ng_hist_curr, qn_hist_size * 3 * sizeof(long int));
                qn_hist_initialized = 1;
            } else {
                memcpy(qn_hist_prev, qn_hist_curr, qn_hist_size * sizeof(double));
                memcpy(ng_hist_prev, ng_hist_curr, qn_hist_size * 3 * sizeof(long int));
                capture_q_neighbor_snapshot(g_particles, qn_hist_curr, ng_hist_curr);
            }

            if (q_fail_pending) {
                dump_q_neighbor_triplet_csv_from_rank0(
                    qn_fail_prev, ng_fail_prev, qn_fail_curr, ng_fail_curr, qn_hist_curr, ng_hist_curr,
                    q_fail_step, charge_update_calls, g_particles->n_p, g_particles->num_neighbors
                );
                q_fail_pending = 0;
                q_fail_step = -1;
            }
        }
    }

    return res;
}

void solver_init_field() {
    g_grid->init_field(g_grid);
}

void solver_set_field(double *phi) {
    int n = g_grid->n;
    long int n2 = n * n;
    int n_start = get_n_start();

    memcpy(g_grid->phi_n, phi + n_start * n2, g_grid->size * sizeof(double));
}

void solver_set_field_prev(double *phi) {
    if (g_grid->phi_p != NULL) {
        int n = g_grid->n;
        long int n2 = n * n;
        int n_start = get_n_start();

        memcpy(g_grid->phi_p, phi + n_start * n2, g_grid->size * sizeof(double));
    }
}

int solver_update_field() {
    field_update_calls += 1;
    int res = g_grid->update_field(g_grid);
    if (res == -1) {
        mpi_printf("update_field failed at step=%ld: q_tot=%e\n", field_update_calls, q_tot);
        field_not_converged = 1;
    }
    return res;
}

void solver_update_eps_k2() {
    // Update the dielectric constant and screening factor based on the grid's transition state
    grid_update_eps_and_k2(g_grid, g_particles);
}

void solver_compute_forces_elec() {
    g_particles->compute_forces_field(g_particles, g_grid);
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
    // if (g_particles->is_water) {
    //     particles_compute_intramolecular_forces(g_particles);
    //     particles_compute_forces_electrostatic_correction_sr(g_particles, g_grid);
    // }
    g_particles->compute_forces_tot(g_particles);
    if (field_not_converged) {
        double p[3] = {0.0, 0.0, 0.0};
        double f_tot[3] = {0.0, 0.0, 0.0};
        double f_elec[3] = {0.0, 0.0, 0.0};
        double f_noel[3] = {0.0, 0.0, 0.0};
        double f_intra[3] = {0.0, 0.0, 0.0};
        double f_corr[3] = {0.0, 0.0, 0.0};
        long int n3 = g_particles->n_p * 3;

        for (long int i = 0; i < n3; i += 3) {
            if (g_particles->fcs_tot != NULL) {
                f_tot[0] += g_particles->fcs_tot[i];
                f_tot[1] += g_particles->fcs_tot[i + 1];
                f_tot[2] += g_particles->fcs_tot[i + 2];
            }
            if (g_particles->fcs_elec != NULL) {
                f_elec[0] += g_particles->fcs_elec[i];
                f_elec[1] += g_particles->fcs_elec[i + 1];
                f_elec[2] += g_particles->fcs_elec[i + 2];
            }
            if (g_particles->fcs_noel != NULL) {
                f_noel[0] += g_particles->fcs_noel[i];
                f_noel[1] += g_particles->fcs_noel[i + 1];
                f_noel[2] += g_particles->fcs_noel[i + 2];
            }
            if (g_particles->fcs_intra != NULL) {
                f_intra[0] += g_particles->fcs_intra[i];
                f_intra[1] += g_particles->fcs_intra[i + 1];
                f_intra[2] += g_particles->fcs_intra[i + 2];
            }
            if (g_particles->fcs_corr != NULL) {
                f_corr[0] += g_particles->fcs_corr[i];
                f_corr[1] += g_particles->fcs_corr[i + 1];
                f_corr[2] += g_particles->fcs_corr[i + 2];
            }
        }
        allreduce_sum(f_tot, 3);
        allreduce_sum(f_elec, 3);
        allreduce_sum(f_noel, 3);
        allreduce_sum(f_intra, 3);
        allreduce_sum(f_corr, 3);
        g_particles->get_momentum(g_particles, p);
        allreduce_sum(p, 3);

        if (using_cubic_spline_assignment()) {
            if (ensure_q_neighbor_history_buffers(g_particles) == 0 && qn_hist_initialized) {
                memcpy(qn_fail_prev, qn_hist_prev, qn_hist_size * sizeof(double));
                memcpy(qn_fail_curr, qn_hist_curr, qn_hist_size * sizeof(double));
                memcpy(ng_fail_prev, ng_hist_prev, qn_hist_size * 3 * sizeof(long int));
                memcpy(ng_fail_curr, ng_hist_curr, qn_hist_size * 3 * sizeof(long int));
                q_fail_pending = 1;
                q_fail_step = field_update_calls;
            }
        }

        int rank = get_rank();
        if (rank == 0) {
            FILE *fp = solver_open_not_converged_log();
            if (fp != NULL) {
                fprintf(fp, "not converged: step=%ld q_tot=%e n_p=%d\n",
                        field_update_calls, q_tot, g_particles->n_p);
                fprintf(fp, "corr_type=%s energy_corr=%e\n",
                        get_water_electrostatic_type_str(g_particles->corr_type), g_particles->energy_corr);
                fprintf(fp, "Ptot %e %e %e\n", p[0], p[1], p[2]);
                fprintf(fp, "Ftot %e %e %e\n", f_tot[0], f_tot[1], f_tot[2]);
                fprintf(fp, "Felec %e %e %e\n", f_elec[0], f_elec[1], f_elec[2]);
                fprintf(fp, "Fnoel %e %e %e\n", f_noel[0], f_noel[1], f_noel[2]);
                fprintf(fp, "Fintra %e %e %e\n", f_intra[0], f_intra[1], f_intra[2]);
                fprintf(fp, "Fcorr %e %e %e\n", f_corr[0], f_corr[1], f_corr[2]);
                double *f_elec = g_particles->fcs_elec;
                double *f_intra = g_particles->fcs_intra;
                double *f_corr = g_particles->fcs_corr;
                for (int i = 0; i < g_particles->n_p; i++) {
                    long int ni = i * 3;
                    long int pi = i * 3;
                    double iex = f_intra ? f_intra[ni] : 0.0;
                    double iey = f_intra ? f_intra[ni + 1] : 0.0;
                    double iez = f_intra ? f_intra[ni + 2] : 0.0;
                    double cex = f_corr ? f_corr[ni] : 0.0;
                    double cey = f_corr ? f_corr[ni + 1] : 0.0;
                    double cez = f_corr ? f_corr[ni + 2] : 0.0;
                    fprintf(fp, "particle %d elec %e %e %e\n",
                            i, f_elec[ni], f_elec[ni + 1], f_elec[ni + 2]);
                    fprintf(fp, "particle %d intra %e %e %e\n",
                            i, iex, iey, iez);
                    fprintf(fp, "particle %d corr %e %e %e\n",
                            i, cex, cey, cez);
                    fprintf(fp, "particle %d pos %e %e %e vel %e %e %e\n",
                            i,
                            g_particles->pos[pi], g_particles->pos[pi + 1], g_particles->pos[pi + 2],
                            g_particles->vel[pi], g_particles->vel[pi + 1], g_particles->vel[pi + 2]);
                }
                fprintf(fp, "\n");
                fclose(fp);
            }
        }
        field_not_converged = 0;
    }
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
    // g_particles->rescale_velocities(g_particles);
    g_particles->rescale_momenta(g_particles);
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

    finalize_debug_log();
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
