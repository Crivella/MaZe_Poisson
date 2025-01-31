/*Wrappers for FFTW3 library*/
#include <stdio.h>
#include <stdlib.h>

#include "fftw_wrap.h"
#include "mpi_base.h"


#ifdef __FFTW
// int initialized_c = 0;
int initialized_r = FFTW_BLANK;

// fftw_plan c_fwd_plan;
// fftw_plan c_bwd_plan;
fftw_plan r_fwd_plan;
fftw_plan r_bwd_plan;

// fftw_complex *c_in;
// fftw_complex *c_out;

double *r_real;
fftw_complex *r_cmpx;

// int FLAG = FFTW_ESTIMATE;
int FLAG = FFTW_MEASURE;
// int FLAG = FFTW_PATIENT;

// void init_fft(int n){
//     if (initialized_c != 0) {
//         return;
//     }
//     initialized_c = 1;
//     #ifdef _OPENMP
//     int tid = omp_get_thread_num();
//     fftw_plan_with_nthreads(omp_get_max_threads());
//     #endif
//     c_in = (fftw_complex *)fftw_malloc(n * n * n * sizeof(fftw_complex));
//     c_out = (fftw_complex *)fftw_malloc(n * n * n * sizeof(fftw_complex));

//     #ifdef _OPENMP
//     if (tid == 0) {
//     #endif
//     printf("FFTW: Initializing C-C plans\n");
//     c_fwd_plan = fftw_plan_dft_3d(n, n, n, c_in, c_out, FFTW_FORWARD, FLAG | FFTW_DESTROY_INPUT);
//     c_bwd_plan = fftw_plan_dft_3d(n, n, n, c_in, c_out, FFTW_BACKWARD, FLAG | FFTW_DESTROY_INPUT);
//     printf("FFTW: ...DONE\n");
//     #ifdef _OPENMP
//     }
//     #endif
// }

#ifdef __FFTW_MPI

// int same = 0;

void init_rfft(int n) {
    if (initialized_r != FFTW_BLANK) {
        return;
    }
    int rank = get_rank();
    initialized_r = FFTW_DOCLEANUP;

    int nh = n / 2 + 1;

    fftw_mpi_init();
    ptrdiff_t loc0, loc_start, loc_size;
    loc_size = fftw_mpi_local_size_3d(n, n, nh, MPI_COMM_WORLD, &loc0, &loc_start);
    // ptrdiff_t fftw_mpi_local_size_many(
    //     int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
    //     ptrdiff_t block0, MPI_Comm comm,
    //     ptrdiff_t *local_n0, ptrdiff_t *local_0_start
    // );
    // loc_size = fftw_mpi_local_size_many(
    //     3, (ptrdiff_t[]){n, n, nh}, 1, n / get_size(), MPI_COMM_WORLD, &loc0, &loc_start
    // );

    // printf(
    //     "--- FFTW_MPI (%d): loc0 = %ld, n_loc = %d, loc_start = %ld, n_start = %d size=%d\n",
    //     rank, loc0, get_n_loc(), loc_start, get_n_start(), loc_size
    //     );

    int cnt = 0;
    if (loc0 != get_n_loc() || loc_start != get_n_start()) {
        cnt = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (cnt > 0) {
        fprintf(
            stderr,
            "FFTW_MPI (%d): loc0 = %ld, n_loc = %d, loc_start = %ld, n_start = %d\n",
            rank, loc0, get_n_loc(), loc_start, get_n_start()
        );
        // fprintf(stderr, "FFTW_MPI: Local size mismatch\n");
        exit(1);
    }

    // if (loc_size == n*n*nh) {
    //     same = 1;
    // }
  
    r_real = fftw_alloc_real(2*loc_size);
    r_cmpx = fftw_alloc_complex(loc_size);

    if (rank == 0) printf("FFTW: Initializing R-C-R plans with MPI\n");
    r_fwd_plan = fftw_mpi_plan_dft_r2c_3d(n, n, n, r_real, r_cmpx, MPI_COMM_WORLD, FLAG | FFTW_DESTROY_INPUT);
    r_bwd_plan = fftw_mpi_plan_dft_c2r_3d(n, n, n, r_cmpx, r_real, MPI_COMM_WORLD, FLAG | FFTW_DESTROY_INPUT);

    if (rank == 0) printf("FFTW: ...DONE\n");
}

// void rfft_solve_even(int n, double *b, double *ig2, double *x) {
//     int n_loc = get_n_loc();
//     int n_start = get_n_start();
//     int nh = n / 2 + 1;
//     long int size = n_loc * n * n;
//     long int n3r = n * n * n;

//     int np = 2 * (n / 2 + 1);
//     long int n2  = n * n;
//     long int n2p = n * np;
//     long int n2h = n * nh;
//     long int i0, j0, i1, j1;

//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         r_real[i] = b[i];
//     }

//     fftw_execute(r_fwd_plan);

//     r_cmpx[0] = 0;
//     #pragma omp parallel for private(i0, j0, i1)
//     for (int i = 0; i < n_loc; i++) {
//         i0 = i * n2h;
//         i1 = (i+n_start) * n2h;
//         for (int j=0; j < n; j++) {
//             j0 = j * nh;
//             for (int k=0; k < nh; k++) {
//                 r_cmpx[i0 + j0 + k] *= ig2[i1 + j0 + k];
//             }
//         }
//     }

//     fftw_execute(r_bwd_plan);

//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         x[i] = r_real[i] / n3r;
//     }
// }

void rfft_solve(int n, double *b, double *ig2, double *x) {
    int n_loc = get_n_loc();
    int n_start = get_n_start();
    int nh = n / 2 + 1;
    long int size = n_loc * n * nh;
    long int n3r = n * n * n;

    int np = 2 * (n / 2 + 1);
    long int n2  = n * n;
    long int n2p = n * np;
    long int n2h = n * nh;
    long int i0, j0, i1, j1;

    #pragma omp parallel for private(i0, j0, i1, j1)
    for (int i=0; i < n_loc; i++) {
        i0 = i * n2p;
        i1 = i * n2;
        for (int j=0; j < n; j++) {
            j0 = i0 + j * np;
            j1 = i1 + j * n;
            for (int k=0; k < n; k++) {
                r_real[j0 + k] = b[j1 + k];
            }
        }
    }

    fftw_execute(r_fwd_plan);

    r_cmpx[0] = 0;
    #pragma omp parallel for private(i0, j0, i1)
    for (int i = 0; i < n_loc; i++) {
        i0 = i * n2h;
        i1 = (i+n_start) * n2h;
        for (int j=0; j < n; j++) {
            j0 = j * nh;
            for (int k=0; k < nh; k++) {
                r_cmpx[i0 + j0 + k] *= ig2[i1 + j0 + k];
            }
        }
    }

    fftw_execute(r_bwd_plan);

    #pragma omp parallel for private(i0, j0, i1, j1)
    for (int i = 0; i < n_loc; i++) {
        i0 = i * n2p;
        i1 = i * n2;
        for (int j=0; j < n; j++) {
            j0 = i0 + j * np;
            j1 = i1 + j * n;
            for (int k=0; k < n; k++) {
                x[j1 + k] = r_real[j0 + k] / n3r;
            }
        }
    }
}

// void rfft_solve(int n, double *b, double *ig2, double *x) {
//     if (same == 0) {
//         rfft_solve_odd(n, b, ig2, x);
//     } else {
//         rfft_solve_even(n, b, ig2, x);
//     }
// }

#else // __FFTW_MPI not defined

void init_rfft(int n) {
    if (initialized_r != FFTW_BLANK) {
        return;
    }
    int nh = n / 2 + 1;
    initialized_r = FFTW_DOCLEANUP;
  
    r_real = fftw_alloc_real(n * n * n);
    r_cmpx = fftw_alloc_complex(n * n * nh);

    printf("FFTW: Initializing R-C-R plans SERIAL\n");
    r_fwd_plan = fftw_plan_dft_r2c_3d(n, n, n, r_real, r_cmpx, FLAG | FFTW_DESTROY_INPUT);
    r_bwd_plan = fftw_plan_dft_c2r_3d(n, n, n, r_cmpx, r_real, FLAG | FFTW_DESTROY_INPUT);
    printf("FFTW: ...DONE\n");
}

/*Solve Ax=b where A is the laplacian using real grid FFTS*/
void rfft_solve(int n, double *b, double *ig2, double *x) {
    int nh = n / 2 + 1;
    long int size = n * n * nh;
    long int n3r = n * n * n;

    #pragma omp parallel for
    for (long int i = 0; i < n3r; i++) {
        r_real[i] = b[i];
    }

    fftw_execute(r_fwd_plan);

    r_cmpx[0] = 0;
    #pragma omp parallel for
    for (long int i = 1; i < size; i++) {
        r_cmpx[i] *= ig2[i];
    }

    fftw_execute(r_bwd_plan);

    #pragma omp parallel for
    for (long int i = 0; i < n3r; i++) {
        x[i] = r_real[i] / n3r;
    }
}


#endif // __FFTW_MPI

void cleanup_fftw() {
    // printf("FFTW: Cleaning up\n");
    // if (initialized_c == 0) {
    //     fftw_destroy_plan(c_fwd_plan);
    //     fftw_destroy_plan(c_bwd_plan);
    //     fftw_free(c_in);
    //     fftw_free(c_out);
    //     initialized_c = 0;
    // }
    if (initialized_r == FFTW_DOCLEANUP) {
        fftw_destroy_plan(r_fwd_plan);
        fftw_destroy_plan(r_bwd_plan);
        fftw_free(r_real);
        fftw_free(r_cmpx);
    }
    initialized_r = FFTW_BLANK;
    // printf("FFTW: Cleaned up\n");
}

// void fft_3d(int n, double *in, complex *out) {
//     int size = n * n * n;
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         c_in[i] = in[i];
//     }
//     fftw_execute(c_fwd_plan);
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         out[i] = c_out[i];
//     }
// }

// void ifft_3d(int n, complex *in, double *out) {
//     long int size = n * n * n;
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         c_in[i] = in[i];
//     }
//     fftw_execute(c_bwd_plan);
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         out[i] = creal(c_out[i]) / size;
//     }
// }

// void rfft_3d(int n, double *in, complex *out) {
//     int nh = n / 2 + 1;
//     long int size = n * n * nh;
//     long int n3r = n * n * n;
//     #pragma omp parallel for
//     for (long int i = 0; i < n3r; i++) {
//         r_real[i] = in[i];
//     }
//     fftw_execute(r_fwd_plan);
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         out[i] = r_cmpx[i];
//     }
// }

// void irfft_3d(int n, complex *in, double *out) {
//     int nh = n / 2 + 1;
//     long int size = n * n * nh;
//     long int n3r = n * n * n;
//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         r_cmpx[i] = in[i];
//     }
//     fftw_execute(r_bwd_plan);
//     #pragma omp parallel for
//     for (long int i = 0; i < n3r; i++) {
//         out[i] = r_real[i] / n3r;
//     }
// }

/*Solve Ax=b where A is the laplacian using FFTS*/
// void fft_solve(int n, double *b, double *ig2, double *x) {
//     long int size = n * n * n;

//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         c_in[i] = b[i];
//     }

//     fftw_execute(c_fwd_plan);

//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         c_in[i] = c_out[i] * ig2[i];
//     }

//     fftw_execute(c_bwd_plan);

//     #pragma omp parallel for
//     for (long int i = 0; i < size; i++) {
//         x[i] = creal(c_out[i]) / size;
//     }
// }

#else // __FFTW

void init_rfft(int n) {
    fprintf(stderr, "TERMINATING: Library compiled without FFTW support\n");
    exit(1);
}

void cleanup_fftw() {
    fprintf(stderr, "TERMINATING: Library compiled without FFTW support\n");
    exit(1);
}

void rfft_solve(int n, double *b, double *ig2, double *x) {
    fprintf(stderr, "TERMINATING: Library compiled without FFTW support\n");
    exit(1);
}

#endif // __FFTW