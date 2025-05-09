#include <stdio.h>
#include <stdlib.h>

#include "mp_structs.h"

char precond_type_str[2][16] = {"JACOBI", "MG"};

int get_precond_type_num() {
    return PRECOND_TYPE_NUM;
}

char *get_precond_type_str(int n) {
    return precond_type_str[n];
}

void precond_jacobi_init(precond *p) {
    // Initialize Jacobi preconditioner
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
            break;
    }
    precond *new = (precond *)malloc(sizeof(precond));
    new->type = type;
    new->n = n;
    new->L = L;
    new->h = h;

    new->grid2 = NULL;
    new->grid3 = NULL;

    new->apply = precond_apply;
    new->prolong = precond_prolong;
    new->restriction = precond_restriction;
    new->smooth = precond_smooth;

    init_func(new);

    new->free = precond_cleanup;

    return new;
}

void precond_cleanup(precond *p) {
    switch (p->type) {
        case PRECOND_TYPE_JACOBI:
            // precond_jacobi_cleanup(p);
            break;
        case PRECOND_TYPE_MG:
            precond_mg_cleanup(p);
            break;
        default:
            break;
    }
    free(p);
}

void precond_apply(precond *p, double *in, double *out) {
    // Do nothing
}

void precond_prolong(double *in, double *out, int s1, int s2, int ts1, int ts2) {
    // Do nothing
}

void precond_restriction(double *in, double *out, int s1, int s2) {
    // Do nothing
}

void precond_smooth(double *in, double *out, int s1, int s2, double tol) {
    // Do nothing
}
