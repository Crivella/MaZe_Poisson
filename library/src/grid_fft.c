#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "fftw_wrap.h"

void fft_grid_init(grid * grid) {
    int n = grid->n;
    double L = grid->L;
    double h = grid->h;

    int nh = n / 2 + 1;


    long int size = n * n * n;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->phi_n = (double *)malloc(size * sizeof(double));
    grid->ig2 = (double *)malloc(n * n * nh * sizeof(double));
    
    int half_limit = (n+1) / 2 - 1;
    double pi2 = 2 * M_PI;

    double app;
    double *freq = (double *)malloc(n * sizeof(double));
    double *rfreq = (double *)malloc(nh * sizeof(double));

    int i = 0;
    for (i = 0; i < nh; i++) {
        app = pi2 * i / L;
        freq[i] = app;
        rfreq[i] = app;
        // printf("freq[%d] = %f\n", i, freq[i]);

    }
    for (i = n%2 ? nh: nh-1; i < n; i++) {
        freq[i] = pi2 * (i - n) / L;
        // printf("freq[%d] = %f\n", i, freq[i]);
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

    init_fftw_omp();
    init_rfft(n);


    grid->init_field = fft_grid_init_field;
    grid->update_field = fft_grid_update_field;

    free(freq);
    free(rfreq);
}

void * fft_grid_init_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);
}

int fft_grid_update_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);

    return 0;
}
