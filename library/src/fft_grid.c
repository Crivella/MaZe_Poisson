#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fftw_wrap.h"
#include "charges.h"
#include "mympi.h"
#include "mp_structs.h"

fft_grid * fft_grid_init(int n, double L, double h) {
    fft_grid *grid = (fft_grid *)malloc(sizeof(fft_grid));
    grid->n = n;
    grid->L = L;
    grid->h = h;

    int nh = n / 2 + 1;

    grid->q = (double *)malloc(n * n * n * sizeof(double));
    grid->phi = (double *)malloc(n * n * n * sizeof(double));

    grid->ig2 = (double *)malloc(n * n * nh * sizeof(double));
    
    int half_limit = (n+1) / 2 - 1;
    double pi2 = 2 * M_PI;

    double app;
    double freq[n];
    double rfreq[nh];

    int i = 0;
    for (i = 0; i <= nh; i++) {
        app = pi2 * i / L;
        freq[i] = app;
        rfreq[i] = app;

    }
    for (i = n%2 ? nh+1: nh; i < n; i++) {
        freq[i] = pi2 * (i - n) / L;
    }

    double f1, f2;
    double const constant = 4 * M_PI / (h*h*h);
    for (i = 0; i < n; i++) {
        f1 = freq[i] * freq[i];
        for (int j = 0; j < n; j++) {
            f2 = f1 + freq[j] * freq[j];
            for (int k = 0; k < nh; k++) {
                app = f2 + rfreq[k] * rfreq[k];
                grid->ig2[i * n * nh + j * nh + k] = app == 0 ? 0 : constant / app;
            }
        }
    }


    grid->free = fft_grid_free;
    grid->init_field = fft_grid_init_field;
    grid->update_field = fft_grid_update_field;

    init_fftw_omp();
    init_rfft(n);

    return grid;
}

void * fft_grid_free(fft_grid *grid) {
    free(grid->q);
    free(grid->phi);
    free(grid->ig2);
    free(grid);

    cleanup_fftw();

    return NULL;
}

void * fft_grid_init_field(fft_grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi);

    return NULL;
}

int fft_grid_update_field(fft_grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi);

    return 0;
}

double fft_grid_update_charges(fft_grid *grid, particles *p) {
    return update_charges(grid->n, p->n_p, grid->h, p->pos, p->neighbors, p->charges, grid->q);
}
