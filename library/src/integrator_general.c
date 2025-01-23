#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "constants.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// General integrator functions
integrator * integrator_init(int n_p, double dt, int type) {
    integrator *new = (integrator *)malloc(sizeof(integrator));
    new->type = type;
    new->n_p = n_p;
    new->dt = dt;

    new->enabled = INTEGRATOR_DISABLED;
    new->c1 = 1.0;
    new->c2 = 1.0;

    switch (type) {
        case INTEGRATOR_TYPE_OVRVO:
            new->part1 = ovrvo_integrator_part1;
            new->part2 = ovrvo_integrator_part2;
            new->init_thermostat = ovrvo_integrator_init_thermostat;
            new->stop_thermostat = ovrvo_integrator_stop_thermostat;
            break;
        case INTEGRATOR_TYPE_VERLET:
            new->part1 = verlet_integrator_part1;
            new->part2 = verlet_integrator_part2;
            new->init_thermostat = verlet_integrator_init_thermostat;
            new->stop_thermostat = verlet_integrator_stop_thermostat;
            break;
    }

    new->free = integrator_free;

    return new;
}

void * integrator_free(integrator *integrator) {
    free(integrator);

    return NULL;
}
