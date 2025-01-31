#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi_base.h"
#include "mp_structs.h"
#include "fftw_wrap.h"

#ifdef __MPI
void fft_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();

    int n = grid->n;
    int rank = mpid->rank;
    int size = mpid->size;
    long int buffer_size = n * n;

    // int div, mod;
    int n_loc, n_start;

    mpid->n_loc = grid->n_local;
    mpid->n_start = grid->n_start;
    for (int i=0; i<size; i++) {
        n_loc = mpid->n_loc;
        n_start = mpid->n_start;
        MPI_Bcast(&n_loc, 1, MPI_INT, i, MPI_COMM_WORLD);
        MPI_Bcast(&n_start, 1, MPI_INT, i, MPI_COMM_WORLD);
        mpid->n_loc_list[i] = n_loc;
        mpid->n_start_list[i] = n_start;
        // printf("FFT MPI(%d %d): n_local = %d, n_start = %d\n", rank, i, n_loc, n_start);
    }
    if (rank < size-1) {
        if (mpid->n_loc_list[rank+1] == 0) {
            mpid->next_rank = 0;
        } 
    }
    if (rank == 0) {
        if (mpid->n_loc_list[size-1] == 0) {
            for (int i=size-1; i>=0; i--) {
                if (mpid->n_loc_list[i] > 0) {
                    mpid->prev_rank = i;
                    break;
                }
            }
        }
    }
    printf(
        "FFT MPI(%d): n_local = %d, n_start = %d, prev_rank=%d, nxt_rank=%d\n",
        rank, grid->n_local, grid->n_start, mpid->prev_rank, mpid->next_rank
    );
    mpid->buffer_size = buffer_size;
    if (size > 1) {
        mpid->bot = (double *)malloc(buffer_size * sizeof(double));
        mpid->top = (double *)malloc(buffer_size * sizeof(double));
    }
}

#else  // __MPI

void fft_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void fft_grid_init(grid * grid) {
    long int i1, j1;

    int n = grid->n;
    double L = grid->L;
    double h = grid->h;

    int n_loc, n_start;
    init_rfft(n, &n_loc, &n_start);
    grid->n_local = n_loc;
    grid->n_start = n_start;
    fft_grid_init_mpi(grid);

    int nh = n / 2 + 1;
    long int size = n_loc * n * n;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->phi_n = (double *)malloc(size * sizeof(double));
    grid->ig2 = (double *)malloc(n_loc * n * nh * sizeof(double));
    
    double const pi2 = 2 * M_PI;
    double app;
    double freq[n];
    double rfreq[nh];

    for (int i = 0; i < nh; i++) {
        app = pi2 * i / L;
        freq[i] = app;
        rfreq[i] = app;

    }
    for (int i = n%2 ? nh: nh-1; i < n; i++) {
        freq[i] = pi2 * (i - n) / L;
    }

    double f1, f2;
    double const constant = 4 * M_PI / (h*h*h) / (n * n * n);
    #pragma omp parallel for private(i1, f1, j1, f2)
    for (int i = 0; i < n_loc; i++) {
        i1 = i * n;
        f1 = pow(freq[i + n_start], 2);
        for (int j = 0; j < n; j++) {
            j1 = (i1 + j) * nh;
            f2 = f1 + pow(freq[j], 2);
            for (int k = 0; k < nh; k++) {
                grid->ig2[j1 + k] = constant / (f2 + pow(rfreq[k], 2) + 1E-10);  // Add small number to avoid division by zero
            }
        }
    }
    if (n_start == 0) {
        grid->ig2[0] = 0;
    }

    grid->init_field = fft_grid_init_field;
    grid->update_field = fft_grid_update_field;
}

void fft_grid_cleanup(grid * grid) {
    cleanup_fftw();
}   

void fft_grid_init_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);
}

int fft_grid_update_field(grid *grid) {
    rfft_solve(grid->n, grid->q, grid->ig2, grid->phi_n);

    return 0;
}
