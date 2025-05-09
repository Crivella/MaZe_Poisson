#include <stdio.h>
#include <stdlib.h>

#include "mp_structs.h"

char precond_type_str[PRECOND_TYPE_NUM][16] = {"NONE", "JACOBI", "MG"};

int get_precond_type_num() {
    return PRECOND_TYPE_NUM;
}

char *get_precond_type_str(int n) {
    return precond_type_str[n];
}

void precond_none_init(precond *p) {
    // Do nothing
}

precond * precond_init(int n, double L, double h, int type) {
    void (*init_func)(precond *);
    switch (type) {
        case PRECOND_TYPE_JACOBI:
            init_func = precond_jacobi_init;
            break;
        case PRECOND_TYPE_MG:
            init_func = precond_mg_init;
            break;
        default:
            init_func = precond_none_init;
            break;
    }
    precond *new = (precond *)malloc(sizeof(precond));
    new->type = type;
    new->n = n;
    new->L = L;
    new->h = h;

    new->grid2 = NULL;
    new->grid3 = NULL;

    new->apply = NULL;

    init_func(new);

    new->free = precond_cleanup;

    return new;
}

void precond_cleanup(precond *p) {
    switch (p->type) {
        case PRECOND_TYPE_JACOBI:
            precond_jacobi_cleanup(p);
            break;
        case PRECOND_TYPE_MG:
            precond_mg_cleanup(p);
            break;
        default:
            break;
    }
    free(p);
}
