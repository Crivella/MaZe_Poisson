#include <math.h>

#include "mp_structs.h"
#include "constants.h"

// double randn (double mu, double sigma)
// {
//   double U1, U2, W, mult;
//   static double X1, X2;
//   static int call = 0;

//   if (call == 1)
//     {
//       call = !call;
//       return (mu + sigma * (double) X2);
//     }

//   do
//     {
//       U1 = -1 + ((double) rand () / RAND_MAX) * 2;
//       U2 = -1 + ((double) rand () / RAND_MAX) * 2;
//       W = pow (U1, 2) + pow (U2, 2);
//     }
//   while (W >= 1 || W == 0);

//   mult = sqrt ((-2 * log (W)) / W);
//   X1 = U1 * mult;
//   X2 = U2 * mult;

//   call = !call;

//   return (mu + sigma * (double) X1);
// }

void o_block(integrator *integrator, particles *p) {
    double dt = integrator->dt;
    double c1 = integrator->c1;
    double T = integrator->T;
    int n_p = p->n_p;
    double *vel = p->vel;
    double *masses = p->mass;
    // c1 = self.c1
    // rnd = np.random.multivariate_normal(
    //     mean=[0.0, 0.0, 0.0],
    //     cov=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    //     size=vel.shape[0]
    // )
    double c1_sqrt = sqrt(c1);
    double cm1_sqrt = sqrt(1 - c1);
    // double rnd = 0;
    double rnd[3] = {0.0, 0.0, 0.0};
    // for 

    long int ni;
    #pragma omp parallel for private(ni)
    for (int i = 0; i < n_p; i++) {
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            vel[ni + j] = c1_sqrt * vel[ni + j] + cm1_sqrt * kB * T / masses[i] * rnd[j];
        }
        vel[i * 3]     = 0.0;
        vel[i * 3 + 1] = 0.0;
        vel[i * 3 + 2] = 0.0;
    }
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

void * ovrvo_integrator_part1(integrator *integrator, particles *p) {
    if (integrator->enabled == INTEGRATOR_ENABLED) o_block(integrator, p);
    v_block(integrator, p);
    r_block(integrator, p);
}

void * ovrvo_integrator_part2(integrator *integrator, particles *p) {
    v_block(integrator, p);
    if (integrator->enabled == INTEGRATOR_ENABLED) o_block(integrator, p);
}

void * ovrvo_integrator_init_thermostat(integrator *integrator, double *params) {
    integrator->T = params[0];
    double gamma = params[1];

    integrator->enabled = INTEGRATOR_ENABLED;

    integrator->c1 = exp(-gamma * integrator->dt);
    integrator->c2 = sqrt(2 / (gamma * integrator->dt) * tanh(0.5 * gamma * integrator->dt));
}

void * ovrvo_integrator_stop_thermostat(integrator *integrator) {
    integrator->enabled = INTEGRATOR_DISABLED;
    integrator->c1 = 1.0;
    integrator->c2 = 1.0;
}