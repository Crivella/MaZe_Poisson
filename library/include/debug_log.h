#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#include <stdio.h>

#include "mp_structs.h"

extern int field_not_converged;
extern long int field_update_calls;
extern long int charge_update_calls;
extern int g_corr_type;
extern char g_output_dir[4096];
extern double *qn_hist_prev;
extern double *qn_hist_curr;
extern double *qn_fail_prev;
extern double *qn_fail_curr;
extern long int *ng_hist_prev;
extern long int *ng_hist_curr;
extern long int *ng_fail_prev;
extern long int *ng_fail_curr;
extern long int qn_hist_size;
extern int qn_hist_initialized;
extern int q_fail_pending;
extern long int q_fail_step;

extern FILE *g_not_converged_log_fp;

FILE *solver_open_not_converged_log();

int ensure_q_neighbor_history_buffers(particles *p);
void capture_q_neighbor_snapshot(particles *g_particles, double *q_out, long int *idx_out);

void finalize_debug_log();

void dump_charge_assignment_diagnostics(particles *g_particles, grid *g_grid, double q_ref, double q_tot_loc);
void dump_q_neighbor_triplet_csv(
    const double *q_prev, const long int *ng_prev,
    const double *q_curr, const long int *ng_curr,
    const double *q_next, const long int *ng_next,
    long int nonconv_step, long int next_step, int n_p, int num_neighbors
);
void dump_q_neighbor_triplet_csv_from_rank0(
    const double *q_prev_local, const long int *ng_prev_local,
    const double *q_curr_local, const long int *ng_curr_local,
    const double *q_next_local, const long int *ng_next_local,
    long int nonconv_step, long int next_step, int n_p, int num_neighbors
);
void dump_mpi_particle_consistency(particles *g_particles);

#endif // DEBUG_LOG_H