#include <stdio.h>
#include <math.h>

#include "mympi.h"


double g(double x, double L, double h) {
    x = fabs(x - round(x / L) * L);
    if (x >= h) {
        return 0.0;
    }
    return 1.0 - x / h;

    // if (2*x > L) x = L - x;
    // return x < h ? 1.0 - x / h : 0.0;
}

#ifdef __MPI

double update_charges(
    int n_grid, int n_p, double h,
    double *pos, long int *neighbors, long int *charges, double *q
) {
    int n_loc = get_n_loc();
    int n_loc_start = get_n_start();

    long int ni, nj, nk, ni_loc;
    long int i, i1, i2;
    long int n2 = n_grid * n_grid;
    long int n3 = n_loc * n2;

    double px, py, pz;

    double L = n_grid * h;
    double q_tot = 0.0;
    long int chg;
    double app, upd;

    #pragma omp parallel for
    for (i=0; i < n3; i++) {
        q[i] = 0.0;
    }

    #pragma omp parallel for private(i1, i2, ni, nj, nk, ni_loc, px, py, pz, app, upd, chg) reduction(+:q_tot)
    for (i=0; i<n_p; i++) {
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
            app = g(px - ni*h, L, h) * g(py - nj*h, L, h) * g(pz - nk*h, L, h);
            upd = chg * app;
            q_tot += upd;
            q[ni_loc * n2 + nj * n_grid + nk] += upd;
        }
    }

    allreduce_double(&q_tot);

    return q_tot;
}

#else



double update_charges(
    int n_grid, int n_p, double h,
    double *pos, long int *neighbors, long int *charges, double *q
) {
    long int ni, nj, nk;
    long int i, i1, i2;
    long int n2 = n_grid * n_grid;
    long int n3 = n_grid * n2;

    double px, py, pz;

    double L = n_grid * h;
    double q_tot = 0.0;
    long int chg;
    double app, upd;

    #pragma omp parallel for
    for (i=0; i < n3; i++) {
        q[i] = 0.0;
    }

    #pragma omp parallel for private(i1, i2, ni, nj, nk, px, py, pz, app, upd, chg) reduction(+:q_tot)
    for (i=0; i<n_p; i++) {
        i1 = i * 3;
        i2 = i * 24;

        chg = charges[i];
        px = pos[i1 + 0];
        py = pos[i1 + 1];
        pz = pos[i1 + 2];
        for (int j=0; j < 24; j+=3) {
            ni = neighbors[i2 + j + 0];
            nj = neighbors[i2 + j + 1];
            nk = neighbors[i2 + j + 2];
            app = g(px - ni*h, L, h) * g(py - nj*h, L, h) * g(pz - nk*h, L, h);
            upd = chg * app;
            q_tot += upd;
            q[ni * n2 + nj * n_grid + nk] += upd;
        }
    }

    return q_tot;
}

#endif