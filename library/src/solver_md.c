#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "mp_structs.h"
#include "mpi_base.h"
#include "omp_base.h"

#define MAX_ITG_PARAMS 10

particles *g_particles = NULL;
integrator *g_integrator = NULL;
grid *g_grid = NULL;

double q_tot = 0.0;
static int field_not_converged = 0;
static long int field_update_calls = 0;
static long int charge_update_calls = 0;
static int g_corr_type = -1;
static char g_output_dir[4096] = ".";
static double *qn_hist_prev = NULL;
static double *qn_hist_curr = NULL;
static double *qn_fail_prev = NULL;
static double *qn_fail_curr = NULL;
static long int *ng_hist_prev = NULL;
static long int *ng_hist_curr = NULL;
static long int *ng_fail_prev = NULL;
static long int *ng_fail_curr = NULL;
static long int qn_hist_size = 0;
static int qn_hist_initialized = 0;
static int q_fail_pending = 0;
static long int q_fail_step = -1;

static double solver_total_charge_from_particles(void) {
    double q_local = 0.0;
    int n_p = g_particles->n_p;
    for (int i = 0; i < n_p; i++) {
        q_local += g_particles->charges[i];
    }

#ifdef __MPI
    int size = get_size();
    if (size > 1) {
        double q_sum = q_local;
        allreduce_sum(&q_sum, 1);

        double q_max = q_local;
        allreduce_max(&q_max, 1);

        double q_min_neg = -q_local;
        allreduce_max(&q_min_neg, 1);
        double q_min = -q_min_neg;

        if (fabs(q_max - q_min) < 1e-12) {
            return q_max; // Particles replicated on each rank.
        }
        return q_sum; // Particles distributed across ranks.
    }
#endif

    return q_local;
}

static void dump_charge_assignment_diagnostics(double q_ref, double q_tot_loc) {
    if (g_particles == NULL || g_grid == NULL) {
        return;
    }

    int n_p = g_particles->n_p;
    int nn3 = g_particles->num_neighbors * 3;
    double *pos = g_particles->pos;
    double *charges = g_particles->charges;
    long int *neighbors = g_particles->neighbors;
    double (*g)(double, double, double) = g_particles->charges_spread_func;
    double h = g_grid->h;
    double L = g_grid->n * h;

    double min_sum = DBL_MAX;
    double max_sum = -DBL_MAX;
    int bad = 0;
    int nan_pos = 0;

    mpi_printf("Charge assignment diagnostics: q_ref=%.6f q_tot_loc=%.6f n_p=%d nn=%d\n",
               q_ref, q_tot_loc, n_p, g_particles->num_neighbors);

    int reported = 0;
    for (int i = 0; i < n_p; i++) {
        int i1 = i * 3;
        int i2 = i * nn3;

        double px = pos[i1 + 0];
        double py = pos[i1 + 1];
        double pz = pos[i1 + 2];

        if (isnan(px) || isnan(py) || isnan(pz)) {
            nan_pos += 1;
            continue;
        }

        double sum_w = 0.0;
        for (int j = 0; j < nn3; j += 3) {
            long int ni = neighbors[i2 + j + 0];
            long int nj = neighbors[i2 + j + 1];
            long int nk = neighbors[i2 + j + 2];
            double app = g(px - ni * h, L, h) * g(py - nj * h, L, h) * g(pz - nk * h, L, h);
            sum_w += app;
        }

        if (sum_w < min_sum) {
            min_sum = sum_w;
        }
        if (sum_w > max_sum) {
            max_sum = sum_w;
        }

        if (fabs(sum_w - 1.0) > 1e-3) {
            bad += 1;
            if (reported < 8) {
                mpi_printf(
                    "  bad particle i=%d q=%.6f sum_w=%.6f pos=(%.6f %.6f %.6f)\n",
                    i, charges[i], sum_w, px, py, pz
                );
                reported += 1;
            }
        }
    }

    mpi_printf("  sum_w range: [%.6f, %.6f], bad=%d, nan_pos=%d\n",
               min_sum, max_sum, bad, nan_pos);
}

void solver_set_output_path(const char *path) {
    if (path == NULL || path[0] == '\0') {
        g_output_dir[0] = '.';
        g_output_dir[1] = '\0';
        return;
    }
    snprintf(g_output_dir, sizeof(g_output_dir), "%s", path);
}

const char *solver_get_output_path(void) {
    return g_output_dir;
}

long int solver_get_field_step(void) {
    return field_update_calls;
}

FILE *solver_open_not_converged_log(void) {
    char filename[8192];
    snprintf(filename, sizeof(filename), "%s/not_converged.txt", g_output_dir);
    return fopen(filename, "a");
}

static int using_cubic_spline_assignment(void) {
    return (g_particles != NULL && g_particles->cas_type == CHARGE_ASS_SCHEME_TYPE_SPLCUB);
}

static int ensure_q_neighbor_history_buffers(particles *p) {
    if (p == NULL) {
        return 1;
    }
    long int size = (long int)p->n_p * p->num_neighbors;
    long int idx_size = size * 3;
    if (qn_hist_size == size && qn_hist_prev != NULL && qn_hist_curr != NULL &&
        qn_fail_prev != NULL && qn_fail_curr != NULL &&
        ng_hist_prev != NULL && ng_hist_curr != NULL &&
        ng_fail_prev != NULL && ng_fail_curr != NULL) {
        return 0;
    }

    free(qn_hist_prev);
    free(qn_hist_curr);
    free(qn_fail_prev);
    free(qn_fail_curr);
    free(ng_hist_prev);
    free(ng_hist_curr);
    free(ng_fail_prev);
    free(ng_fail_curr);
    qn_hist_prev = NULL;
    qn_hist_curr = NULL;
    qn_fail_prev = NULL;
    qn_fail_curr = NULL;
    ng_hist_prev = NULL;
    ng_hist_curr = NULL;
    ng_fail_prev = NULL;
    ng_fail_curr = NULL;
    qn_hist_size = 0;
    qn_hist_initialized = 0;

    qn_hist_prev = (double *)malloc(size * sizeof(double));
    qn_hist_curr = (double *)malloc(size * sizeof(double));
    qn_fail_prev = (double *)malloc(size * sizeof(double));
    qn_fail_curr = (double *)malloc(size * sizeof(double));
    ng_hist_prev = (long int *)malloc(idx_size * sizeof(long int));
    ng_hist_curr = (long int *)malloc(idx_size * sizeof(long int));
    ng_fail_prev = (long int *)malloc(idx_size * sizeof(long int));
    ng_fail_curr = (long int *)malloc(idx_size * sizeof(long int));
    if (qn_hist_prev == NULL || qn_hist_curr == NULL || qn_fail_prev == NULL || qn_fail_curr == NULL ||
        ng_hist_prev == NULL || ng_hist_curr == NULL || ng_fail_prev == NULL || ng_fail_curr == NULL) {
        free(qn_hist_prev);
        free(qn_hist_curr);
        free(qn_fail_prev);
        free(qn_fail_curr);
        free(ng_hist_prev);
        free(ng_hist_curr);
        free(ng_fail_prev);
        free(ng_fail_curr);
        qn_hist_prev = NULL;
        qn_hist_curr = NULL;
        qn_fail_prev = NULL;
        qn_fail_curr = NULL;
        ng_hist_prev = NULL;
        ng_hist_curr = NULL;
        ng_fail_prev = NULL;
        ng_fail_curr = NULL;
        return 1;
    }
    qn_hist_size = size;
    return 0;
}

static void capture_q_neighbor_snapshot(double *q_out, long int *idx_out) {
    int n_p = g_particles->n_p;
    int num_neighbors = g_particles->num_neighbors;
    double L = g_particles->L;
    double h = g_particles->h;
    double *pos = g_particles->pos;
    double *charges = g_particles->charges;
    long int *neighbors = g_particles->neighbors;
    double (*g)(double, double, double) = g_particles->charges_spread_func;

    for (int ip = 0; ip < n_p; ip++) {
        long int p3 = ip * 3;
        long int n0 = (long int)ip * num_neighbors * 3;
        long int o0 = (long int)ip * num_neighbors;
        double px = pos[p3];
        double py = pos[p3 + 1];
        double pz = pos[p3 + 2];
        double q = charges[ip];
        for (int j = 0; j < num_neighbors; j++) {
            long int idx3 = n0 + j * 3;
            long int ni = neighbors[idx3];
            long int nj = neighbors[idx3 + 1];
            long int nk = neighbors[idx3 + 2];
            long int out = o0 + j;
            idx_out[out * 3] = ni;
            idx_out[out * 3 + 1] = nj;
            idx_out[out * 3 + 2] = nk;
            q_out[out] = q * g(px - ni * h, L, h) * g(py - nj * h, L, h) * g(pz - nk * h, L, h);
        }
    }
}

static void dump_q_neighbor_triplet_csv(
    const double *q_prev, const long int *ng_prev,
    const double *q_curr, const long int *ng_curr,
    const double *q_next, const long int *ng_next,
    long int nonconv_step, long int next_step, int n_p, int num_neighbors
) {
    int rank = get_rank();
    if (rank != 0) {
        return;
    }

    char filename[8192];
    snprintf(filename, sizeof(filename), "%s/q_cubic_context_step_%07ld.csv", g_output_dir, nonconv_step);
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        return;
    }
    fprintf(fp, "nonconv_step,phase,grid_step,particle,neighbor,i,j,k,q_assigned,count\n");

    long int n_entries = (long int)n_p * num_neighbors;
    long int *ui = (long int *)malloc(n_entries * sizeof(long int));
    long int *uj = (long int *)malloc(n_entries * sizeof(long int));
    long int *uk = (long int *)malloc(n_entries * sizeof(long int));
    double *uq = (double *)malloc(n_entries * sizeof(double));
    int *ucnt = (int *)malloc(n_entries * sizeof(int));
    int *up = (int *)malloc(n_entries * sizeof(int));
    int *un = (int *)malloc(n_entries * sizeof(int));
    if (ui == NULL || uj == NULL || uk == NULL || uq == NULL || ucnt == NULL || up == NULL || un == NULL) {
        free(ui);
        free(uj);
        free(uk);
        free(uq);
        free(ucnt);
        free(up);
        free(un);
        fclose(fp);
        return;
    }

    const char *phase_names[3] = {"prev", "curr", "next"};
    const long int phase_steps[3] = {nonconv_step - 1, nonconv_step, next_step};
    const double *phase_q[3] = {q_prev, q_curr, q_next};
    const long int *phase_ng[3] = {ng_prev, ng_curr, ng_next};

    for (int ph = 0; ph < 3; ph++) {
        long int nuniq = 0;
        for (int ip = 0; ip < n_p; ip++) {
            long int o0 = (long int)ip * num_neighbors;
            for (int j = 0; j < num_neighbors; j++) {
                long int idx = o0 + j;
                long int ii = phase_ng[ph][idx * 3];
                long int jj = phase_ng[ph][idx * 3 + 1];
                long int kk = phase_ng[ph][idx * 3 + 2];
                long int found = -1;
                for (long int u = 0; u < nuniq; u++) {
                    if (ui[u] == ii && uj[u] == jj && uk[u] == kk) {
                        found = u;
                        break;
                    }
                }
                if (found >= 0) {
                    uq[found] += phase_q[ph][idx];
                    ucnt[found] += 1;
                } else {
                    ui[nuniq] = ii;
                    uj[nuniq] = jj;
                    uk[nuniq] = kk;
                    uq[nuniq] = phase_q[ph][idx];
                    ucnt[nuniq] = 1;
                    up[nuniq] = ip;
                    un[nuniq] = j;
                    nuniq += 1;
                }
            }
        }

        for (long int u = 0; u < nuniq; u++) {
            fprintf(fp, "%ld,%s,%ld,%d,%d,%ld,%ld,%ld,%.17e,%d\n",
                    nonconv_step, phase_names[ph], phase_steps[ph], up[u], un[u],
                    ui[u], uj[u], uk[u], uq[u], ucnt[u]);
        }
    }

    free(ui);
    free(uj);
    free(uk);
    free(uq);
    free(ucnt);
    free(up);
    free(un);
    fclose(fp);
}

static void dump_q_neighbor_triplet_csv_from_rank0(
    const double *q_prev_local, const long int *ng_prev_local,
    const double *q_curr_local, const long int *ng_curr_local,
    const double *q_next_local, const long int *ng_next_local,
    long int nonconv_step, long int next_step, int n_p, int num_neighbors
) {
    int rank = get_rank();
    int size = get_size();
    long int nvec = (long int)n_p * num_neighbors;
    long int nidx = nvec * 3;

    if (size <= 1) {
        dump_q_neighbor_triplet_csv(
            q_prev_local, ng_prev_local, q_curr_local, ng_curr_local, q_next_local, ng_next_local,
            nonconv_step, next_step, n_p, num_neighbors
        );
        return;
    }

    // Current code path is primarily intended for serial debugging.
    // For MPI, only rank 0 local portion will be written.
    if (rank == 0) {
        dump_q_neighbor_triplet_csv(
            q_prev_local, ng_prev_local, q_curr_local, ng_curr_local, q_next_local, ng_next_local,
            nonconv_step, next_step, n_p, num_neighbors
        );
    } else {
        (void)nidx;
        (void)nvec;
        (void)ng_prev_local;
        (void)ng_curr_local;
        (void)ng_next_local;
        (void)q_prev_local;
        (void)q_curr_local;
        (void)q_next_local;
        (void)nonconv_step;
        (void)next_step;
        (void)n_p;
        (void)num_neighbors;
    }
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
    int n, int n_typ, double L, double h, int n_p, int pot_type, int cas_type, int is_water,
    int *types, double *pos, double *vel, double *mass, double *charges,
    double *pot_params
) {
    g_particles = particles_init(n, n_p, n_typ, L, h, cas_type);
    g_particles->is_water = is_water;
    if (g_particles->is_water) {
        g_particles->fcs_intra = (double *)calloc(n_p * 3, sizeof(double));
        g_particles->fcs_corr = (double *)calloc(n_p * 3, sizeof(double));
    }

    memcpy(g_particles->types, types, n_p * sizeof(int));
    memcpy(g_particles->pos, pos, n_p * 3 * sizeof(double));
    memcpy(g_particles->vel, vel, n_p * 3 * sizeof(double));
    memcpy(g_particles->mass, mass, n_p * sizeof(double));
    memcpy(g_particles->charges, charges, n_p * sizeof(double));

#ifdef __MPI
    int size = get_size();
    if (size > 1) {
        bcast_double(g_particles->pos, n_p * 3, 0);
        bcast_double(g_particles->vel, n_p * 3, 0);
        bcast_double(g_particles->mass, n_p, 0);
        bcast_double(g_particles->charges, n_p, 0);
    }
#endif
    
    g_particles->init_potential(g_particles, pot_type, pot_params);
}

void solver_set_electrostatic_correction(int corr_type) {
    if (g_particles == NULL) {
        return;
    }
    g_corr_type = corr_type;
    switch (corr_type) {
        case 0:
            g_particles->compute_forces_electrostatic_correction =
                particles_compute_forces_electrostatic_correction_spread;
            break;
        case 1:
            g_particles->compute_forces_electrostatic_correction =
                particles_compute_forces_electrostatic_correction_sr;
            break;
        default:
            mpi_fprintf(stderr, "Invalid electrostatic correction type %d\n", corr_type);
            exit(1);
    }
}

static const char *corr_type_name(void) {
    switch (g_corr_type) {
        case 0:
            return "SPREAD";
        case 1:
            return "SR";
        default:
            return "UNKNOWN";
    }
}

void solver_initialize_particles_pois_boltz(double gamma_np, double beta_np, double *solv_radii) {
    particles_pb_init(g_particles, gamma_np, beta_np, solv_radii);
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

static void dump_mpi_particle_consistency(void) {
#ifdef __MPI
    if (g_particles == NULL) {
        return;
    }
    int size = get_size();
    if (size <= 1) {
        return;
    }

    int n_p = g_particles->n_p;
    double *pos = g_particles->pos;
    double *charges = g_particles->charges;

    double sum_pos = 0.0;
    double sum_pos2 = 0.0;
    double sum_q = 0.0;
    for (int i = 0; i < n_p; i++) {
        int i1 = i * 3;
        double x = pos[i1 + 0];
        double y = pos[i1 + 1];
        double z = pos[i1 + 2];
        sum_pos += x + y + z;
        sum_pos2 += x * x + y * y + z * z;
        sum_q += charges[i];
    }

    double max_sum_pos = sum_pos;
    double max_sum_pos2 = sum_pos2;
    double max_sum_q = sum_q;
    allreduce_max(&max_sum_pos, 1);
    allreduce_max(&max_sum_pos2, 1);
    allreduce_max(&max_sum_q, 1);

    double min_sum_pos_neg = -sum_pos;
    double min_sum_pos2_neg = -sum_pos2;
    double min_sum_q_neg = -sum_q;
    allreduce_max(&min_sum_pos_neg, 1);
    allreduce_max(&min_sum_pos2_neg, 1);
    allreduce_max(&min_sum_q_neg, 1);

    double min_sum_pos = -min_sum_pos_neg;
    double min_sum_pos2 = -min_sum_pos2_neg;
    double min_sum_q = -min_sum_q_neg;

    if (fabs(max_sum_pos - min_sum_pos) > 1e-8 ||
        fabs(max_sum_pos2 - min_sum_pos2) > 1e-8 ||
        fabs(max_sum_q - min_sum_q) > 1e-12) {
        mpi_printf("MPI particle mismatch: sum_pos[min,max]=[%e,%e] sum_pos2[min,max]=[%e,%e] sum_q[min,max]=[%e,%e]\n",
                   min_sum_pos, max_sum_pos, min_sum_pos2, max_sum_pos2, min_sum_q, max_sum_q);
    }
#endif
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
            dump_charge_assignment_diagnostics(q_ref, q_tot_loc);
            dump_mpi_particle_consistency();
            exit(1);
        } else {
            mpi_printf("Charge conservation warning: q_ref = %.6f, q_tot_loc = %.6f\n", q_ref, q_tot_loc);
            dump_mpi_particle_consistency();
        }
    }

    if (using_cubic_spline_assignment()) {
        if (ensure_q_neighbor_history_buffers(g_particles) == 0) {
            if (!qn_hist_initialized) {
                capture_q_neighbor_snapshot(qn_hist_curr, ng_hist_curr);
                memcpy(qn_hist_prev, qn_hist_curr, qn_hist_size * sizeof(double));
                memcpy(ng_hist_prev, ng_hist_curr, qn_hist_size * 3 * sizeof(long int));
                qn_hist_initialized = 1;
            } else {
                memcpy(qn_hist_prev, qn_hist_curr, qn_hist_size * sizeof(double));
                memcpy(ng_hist_prev, ng_hist_curr, qn_hist_size * 3 * sizeof(long int));
                capture_q_neighbor_snapshot(qn_hist_curr, ng_hist_curr);
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
                        corr_type_name(), g_particles->energy_corr);
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
    free(qn_hist_prev);
    free(qn_hist_curr);
    free(qn_fail_prev);
    free(qn_fail_curr);
    free(ng_hist_prev);
    free(ng_hist_curr);
    free(ng_fail_prev);
    free(ng_fail_curr);
    qn_hist_prev = NULL;
    qn_hist_curr = NULL;
    qn_fail_prev = NULL;
    qn_fail_curr = NULL;
    ng_hist_prev = NULL;
    ng_hist_curr = NULL;
    ng_fail_prev = NULL;
    ng_fail_curr = NULL;
    qn_hist_size = 0;
    qn_hist_initialized = 0;
    q_fail_pending = 0;
    q_fail_step = -1;

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
