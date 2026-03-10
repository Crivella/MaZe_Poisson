#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "mpi_base.h"
#include "mp_structs.h"

int field_not_converged = 0;
long int field_update_calls = 0;
long int charge_update_calls = 0;
int g_corr_type = -1;
char g_output_dir[4096] = ".";
double *qn_hist_prev = NULL;
double *qn_hist_curr = NULL;
double *qn_fail_prev = NULL;
double *qn_fail_curr = NULL;
long int *ng_hist_prev = NULL;
long int *ng_hist_curr = NULL;
long int *ng_fail_prev = NULL;
long int *ng_fail_curr = NULL;
long int qn_hist_size = 0;
int qn_hist_initialized = 0;
int q_fail_pending = 0;
long int q_fail_step = -1;

FILE *g_not_converged_log_fp = NULL;


void solver_set_output_path(const char *path) {
    if (path == NULL || path[0] == '\0') {
        g_output_dir[0] = '.';
        g_output_dir[1] = '\0';
        return;
    }
    snprintf(g_output_dir, sizeof(g_output_dir), "%s", path);
}

// const char *solver_get_output_path(void) {
//     return g_output_dir;
// }

// long int solver_get_field_step(void) {
//     return field_update_calls;
// }

FILE *solver_open_not_converged_log(void) {
    if (g_not_converged_log_fp != NULL) {
        return g_not_converged_log_fp;
    }

    char filename[8192];
    snprintf(filename, sizeof(filename), "%s/not_converged.txt", g_output_dir);

    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open not_converged log file at %s\n", filename);
        return NULL;
    }

    g_not_converged_log_fp = fp;

    return fp;
}

int ensure_q_neighbor_history_buffers(particles *p) {
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

void capture_q_neighbor_snapshot(particles *g_particles, double *q_out, long int *idx_out) {
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

void finalize_debug_log() {
    if (g_not_converged_log_fp != NULL) {
        fclose(g_not_converged_log_fp);
        g_not_converged_log_fp = NULL;
    }
    if (qn_hist_prev != NULL) {
        free(qn_hist_prev);
        qn_hist_prev = NULL;
    }
    if (qn_hist_curr != NULL) {
        free(qn_hist_curr);
        qn_hist_curr = NULL;
    }
    if (qn_fail_prev != NULL) {
        free(qn_fail_prev);
        qn_fail_prev = NULL;
    }
    if (qn_fail_curr != NULL) {
        free(qn_fail_curr);
        qn_fail_curr = NULL;
    }
    if (ng_hist_prev != NULL) {
        free(ng_hist_prev);
        ng_hist_prev = NULL;
    }
    if (ng_hist_curr != NULL) {
        free(ng_hist_curr);
        ng_hist_curr = NULL;
    }
    if (ng_fail_prev != NULL) {
        free(ng_fail_prev);
        ng_fail_prev = NULL;
    }
    if (ng_fail_curr != NULL) {
        free(ng_fail_curr);
        ng_fail_curr = NULL;
    }
    qn_hist_size = 0;
    qn_hist_initialized = 0;
    q_fail_pending = 0;
    q_fail_step = -1;
}

void dump_charge_assignment_diagnostics(
    particles *g_particles, grid *g_grid,
    double q_ref, double q_tot_loc
) {
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

void dump_q_neighbor_triplet_csv(
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

void dump_q_neighbor_triplet_csv_from_rank0(
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

void dump_mpi_particle_consistency(particles *g_particles) {
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
}
