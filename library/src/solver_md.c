#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mp_structs.h"
#include "mympi.h"

#define MAX_ITG_PARAMS 10

particles *g_particles = NULL;
integrator *g_integrator = NULL;
grid *g_grid = NULL;

double q_tot = 0.0;


void solver_initialize(int n_grid) {
    init_mpi();
    init_mpi_grid(n_grid);
}

void solverinitialize_grid(int n_grid, double L, double h, double tol, int grid_type) {
    g_grid = grid_init(n_grid, L, h, tol, grid_type);
}

void solverinitialize_particles(
    int n, double L, double h, int n_p, int pot_type,
    double *pos, double *vel, double *mass, long int *charges
) {
    g_particles = particles_init(n, n_p, L, h);

    memcpy(g_particles->pos, pos, n_p * 3 * sizeof(double));
    memcpy(g_particles->vel, vel, n_p * 3 * sizeof(double));
    memcpy(g_particles->mass, mass, n_p * sizeof(double));
    memcpy(g_particles->charges, charges, n_p * sizeof(long int));
    
    g_particles->init_potential(g_particles, pot_type);
}

void solverinitialize_integrator(
    int n_p, double dt, double T, double gamma, int itg_type, int itg_enabled
) {
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

    if (itg_enabled) g_integrator->init_thermostat(g_integrator, itg_params);
}

int solver_update_charges() {
    int res = 0;
    double q_tot_loc;
    
    g_particles->update_nearest_neighbors(g_particles);
    q_tot_loc = g_grid->update_charges(g_grid, g_particles);

    if (fabs(q_tot - q_tot_loc) < 1e-6)  res = 1;

    q_tot = q_tot_loc;

    return res;
}

void solver_update_field() {
    g_grid->update_field(g_grid);
}

void solver_compute_forces() {
    g_particles->compute_forces(g_particles, g_grid);
}

void solvernitialize_md(int preconditioning, int vel_rescale) {
    #pragma omp parallel for reduction(+:q_tot)
    for (int i = 0; i < g_particles->n_p; i++) {
        q_tot += g_particles->charges[i];
    }

    // Step 0 Verlet
    solver_update_charges();
    if (preconditioning == 1)
        solver_update_field();
    solver_compute_forces();

    // Step 1 Verlet
    g_integrator->part1(g_integrator, g_particles);
    solver_update_charges();
    if (preconditioning == 1) 
        solver_update_field();
    solver_compute_forces();
    g_integrator->part2(g_integrator, g_particles);

    if (vel_rescale == 1) 
        g_particles->rescale_velocities(g_particles);
}

void solver_md_loop_iter() {
    g_integrator->part1(g_integrator, g_particles);
    solver_update_charges();
    solver_update_field();
    solver_compute_forces();
    g_integrator->part2(g_integrator, g_particles);
}

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

void solver_run_n_steps(int n_steps) {
    for (int i = 0; i < n_steps; i++) {
        solver_md_loop_iter();
    }
}

void solver_finalize() {
    if (g_particles != NULL)  g_particles->free(g_particles);
    if (g_grid != NULL)  g_grid->free(g_grid);
    if (g_integrator != NULL)  g_integrator->free(g_integrator);

    cleanup_mpi();
}
