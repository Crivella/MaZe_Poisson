#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mympi.h"


double g(double x, double L, double h) {
    x = abs(x - round(x / L) * L);
    return 1.0 - x / h;
}

double update_charges_(
    int n_grid, int n_p, double h,
    double *pos, long int *neighbors, double *charges, double *q
) {
    int n_loc = get_n_loc();
    int n_loc_start = get_n_start();
    printf("chargs.c: N: %d, N_p: %d, h: %f, n_loc: %d, n_loc_start: %d\n", n_grid, n_p, h, n_loc, n_loc_start);

    int ni, nj, nk, ni_loc;
    long int i1, i2;
    long int n2 = n_grid * n_grid;
    long int n3 = n_loc * n2;

    double px, py, pz;

    double L = n_grid * h;
    double q_tot = 0.0;
    double app, upd, chg;

    // printf("---------- n3: %ld\n", n3);
    // // #pragma omp parallel for
    for (long int i=0; i < n3; i++) {
        q[i] = 0.0;
    }

    for (int i=0; i<n_p; i++) {
        i1 = i * 3;
        i2 = i * 24;

        chg = charges[i];
        px = pos[i1 + 0];
        py = pos[i1 + 1];
        pz = pos[i1 + 2];
        for (int j=0; j < 24; j+=3) {
            ni = neighbors[i2 + j + 0];
            ni_loc = ni - n_loc_start;
            if (ni_loc < 0 || ni_loc >= n_loc) {
                continue;
            }
            nj = neighbors[i2 + j + 1];
            nk = neighbors[i2 + j + 2];
            printf("px: %f, ni: %d, hn: %f, diff: %f\n", px, ni, h, px - ni*h);;
            app = g(px - ni*h, L, h) * g(py - nj*h, L, h) * g(pz - nk*h, L, h);
            upd = chg * app;
            q[ni_loc * n2 + nj * n_grid + nk] += upd;
            q_tot += upd;
        }
    }

    allreduce_double(&q_tot);

    return q_tot;
}
