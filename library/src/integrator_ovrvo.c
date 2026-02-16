#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "constants.h"
#include "mpi_base.h"
#include "omp_base.h"

// Thread-local RNG using xorshift64* with Box-Muller for normals.
// Each thread owns its own RNG state to avoid OpenMP data races.
static unsigned long long *rng_state = NULL;
static double *rng_spare = NULL;
static int *rng_has_spare = NULL;
static int rng_nthreads = 0;

static unsigned long long splitmix64_next(unsigned long long *x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static unsigned long long xorshift64star(unsigned long long *x) {
    unsigned long long v = *x;
    v ^= v >> 12;
    v ^= v << 25;
    v ^= v >> 27;
    *x = v;
    return v * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform01(int tid) {
    // Convert to (0,1) double. Keep away from 0 to avoid log(0).
    unsigned long long r = xorshift64star(&rng_state[tid]);
    double u = (r >> 11) * (1.0 / 9007199254740992.0); // 53-bit mantissa
    if (u <= 0.0) {
        u = 1.0 / 9007199254740992.0;
    }
    return u;
}

static void rng_ensure_init() {
    int nthreads = get_omp_max_threads();
    if (nthreads < 1) {
        nthreads = 1;
    }
    if (rng_state != NULL && rng_nthreads == nthreads) {
        return;
    }
    free(rng_state);
    free(rng_spare);
    free(rng_has_spare);
    rng_state = (unsigned long long *)malloc((size_t)nthreads * sizeof(unsigned long long));
    rng_spare = (double *)malloc((size_t)nthreads * sizeof(double));
    rng_has_spare = (int *)calloc((size_t)nthreads, sizeof(int));
    rng_nthreads = nthreads;

    unsigned long long seed = 0xA5A5A5A5A5A5A5A5ULL;
    seed ^= (unsigned long long)rand();
    seed ^= ((unsigned long long)rand()) << 32;
    for (int i = 0; i < nthreads; i++) {
        rng_state[i] = splitmix64_next(&seed);
    }
}

static double randn_thread(int tid) {
    if (rng_has_spare[tid]) {
        rng_has_spare[tid] = 0;
        return rng_spare[tid];
    }
    double u1 = rng_uniform01(tid);
    double u2 = rng_uniform01(tid);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    rng_spare[tid] = r * sin(theta);
    rng_has_spare[tid] = 1;
    return r * cos(theta);
}

void ovrvo_integrator_init(integrator *integrator) {
    integrator->part1 = ovrvo_integrator_part1;
    integrator->part2 = ovrvo_integrator_part2;
    integrator->init_thermostat = ovrvo_integrator_init_thermostat;
    integrator->stop_thermostat = ovrvo_integrator_stop_thermostat;
}

void o_block(integrator *integrator, particles *p) {
    // This function needs to be MPI aware as it is possible for every process to generate
    // different random numbers leading to very process working with desynchronized velocities/positions.
    int rank = get_rank();
    long int ni;
    double dt = integrator->dt;
    double c1 = integrator->c1;
    double T = integrator->T;
    double var1, var2;
    
    int n_p = p->n_p;
    double *vel = p->vel;
    double *masses = p->mass;

    double c1_sqrt = sqrt(c1);

    if (rank == 0) {
        rng_ensure_init();
        // The original call to multivariate_normal had a diagonal covariance so we are fine
        // with using randn for each component to generate 3 independent random numbers with the respective
        // mean = 0.0 and variance = 1.0.
        var2 = (1 - c1) * kB * T;
        #pragma omp parallel for private(ni, var1)
        for (int i = 0; i < n_p; i++) {
            int tid = get_omp_thread_num();
            ni = i * 3;
            var1 = sqrt(var2 / masses[i]);
            for (int j = 0; j < 3; j++) {
                vel[ni + j] *= c1_sqrt;
                vel[ni + j] += var1 * randn_thread(tid);
            }
        }
    }

    bcast_double(vel, n_p * 3, 0);
}

void v_block(integrator *integrator, particles *p) {
    double dt = integrator->dt;
    double c2 = integrator->c2;
    int n_p = p->n_p;
    double *vel = p->vel;
    double *forces = p->fcs_tot;
    double *masses = p->mass;

    long int ni;
    #pragma omp parallel for private(ni)
    for (int i = 0; i < n_p; i++) {
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            vel[ni + j] += 0.5 * dt * c2 * forces[ni + j] / masses[i];
        }
    }
}

void r_block(integrator *integrator, particles *p) {
    int n_p = p->n_p;
    double dt = integrator->dt;
    double c2 = integrator->c2;

    double *pos = p->pos;
    double *vel = p->vel;
    double L = p->L;

    long int ni;
    double app;
    #pragma omp parallel for private(ni, app)
    for (int i = 0; i < n_p; i++) {
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            app = pos[ni + j] + c2 * dt * vel[ni + j];
            if (app < 0) {
                pos[ni + j] = app + L;
            } else if (app >= L) {
                pos[ni + j] = app - L;
            } else {
                pos[ni + j] = app;
            }
        }
    }
}

void ovrvo_integrator_part1(integrator *integrator, particles *p) {
    if (integrator->enabled == INTEGRATOR_ENABLED) {
        o_block(integrator, p);
    }
    v_block(integrator, p);
    r_block(integrator, p);
}

void ovrvo_integrator_part2(integrator *integrator, particles *p) {
    v_block(integrator, p);
    if (integrator->enabled == INTEGRATOR_ENABLED) {
        o_block(integrator, p);
    }
}

void ovrvo_integrator_init_thermostat(integrator *integrator, double *params) {
    integrator->T = params[0];
    double gamma = params[1];

    integrator->enabled = INTEGRATOR_ENABLED;

    integrator->c1 = exp(-gamma * integrator->dt);
    integrator->c2 = sqrt(2 / (gamma * integrator->dt) * tanh(0.5 * gamma * integrator->dt));
}

void ovrvo_integrator_stop_thermostat(integrator *integrator) {
    integrator->enabled = INTEGRATOR_DISABLED;
    integrator->c1 = 1.0;
    integrator->c2 = 1.0;
}
