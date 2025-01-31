/*Wrappers for FFTW3 library*/
#include <stdio.h>
#include <stdlib.h>

#include "fftw_wrap.h"
#include "mpi_base.h"


#ifdef __FFTW
int initialized_r = FFTW_BLANK;
double *r_real;
fftw_complex *r_cmpx;
fftw_plan r_fwd_plan;
fftw_plan r_bwd_plan;

// int initialized_c = FFTW_BLANK;
// fftw_complex *c_in;
// fftw_complex *c_out;
// fftw_plan c_fwd_plan;
// fftw_plan c_bwd_plan;


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

void init_rfft(int n, int *n_loc, int *n_start) {
    if (initialized_r != FFTW_BLANK) {
        return;
    }
    mpi_data *mpid = get_mpi_data();
    initialized_r = FFTW_DOCLEANUP;

    int nh = n / 2 + 1;

    fftw_mpi_init();
    ptrdiff_t loc0, loc_start, loc_size;
    loc_size = fftw_mpi_local_size_3d(n, n, nh, mpid->comm, &loc0, &loc_start);

    // int cnt = loc0 < 1 ? 1 : 0;
    // MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPI_INT, MPI_SUM, mpid->comm);
    // if (cnt > 0) {
    //     printf("FFTW_MPI(%d): n_local = %ld, n_start = %ld\n", mpid->rank, loc0, loc_start);
    //     if (mpid->rank == 0) {
    //         fprintf(
    //             stderr,
    //             "FFTW_MPI: The current #N_GRID and #NPROCS is resulting in\n"
    //             "          a group with 0 elements (currently not implemented).\n"
    //             "          Please change the number of grid points or processes.\n"
    //         );
    //     }
    //     exit(1);
    // }

    *n_loc = loc0;
    *n_start = loc_start;
  
    r_real = fftw_alloc_real(2*loc_size);
    r_cmpx = fftw_alloc_complex(loc_size);

    if (mpid->rank == 0) printf("FFTW: Initializing R-C-R plans with MPI\n");
    r_fwd_plan = fftw_mpi_plan_dft_r2c_3d(n, n, n, r_real, r_cmpx, mpid->comm, FLAG | FFTW_DESTROY_INPUT);
    r_bwd_plan = fftw_mpi_plan_dft_c2r_3d(n, n, n, r_cmpx, r_real, mpid->comm, FLAG | FFTW_DESTROY_INPUT);

    if (mpid->rank == 0) printf("FFTW: ...DONE\n");
}

void rfft_solve(int n, double *b, double *ig2, double *x) {
    int n_loc = get_n_loc();

    int nh = n / 2 + 1;
    int npad = 2 * nh;

    long int size = n_loc * n * nh;
    long int i0, j0, j1;

    // printf("FFTW_MPI(%d): n_local = %d, n_start = %d\n", get_rank(), n_loc, get_n_start());
    #pragma omp parallel for private(i0, j0, j1)
    for (int i=0; i < n_loc; i++) {
        i0 = i * n;
        for (int j=0; j < n; j++) {
            j0 = i0 + j;
            j1 = j0 * n;
            j0 *= npad;
            for (int k=0; k < n; k++) {
                r_real[j0 + k] = b[j1 + k];
            }
        }
    }

    fftw_execute(r_fwd_plan);

    #pragma omp parallel for
    for (long int i = 0; i < size; i++) {
        r_cmpx[i] *= ig2[i];
    }

    fftw_execute(r_bwd_plan);

    #pragma omp parallel for private(i0, j0, j1)
    for (int i = 0; i < n_loc; i++) {
        i0 = i * n;
        for (int j=0; j < n; j++) {
            j0 = i0 + j;
            j1 = j0 * n;
            j0 *= npad;
            for (int k=0; k < n; k++) {
                x[j1 + k] = r_real[j0 + k];  // Normalization moved inside ig2
            }
        }
    }
}

#else // __FFTW_MPI not defined

void init_rfft(int n, int *n_loc, int *n_start) {
    if (get_size() > 1) {
        if (get_rank() == 0) fprintf(stderr, "TERMINATING: Linked FFTW compiled without MPI support\n");
        exit(1);
    }
    if (initialized_r != FFTW_BLANK) {
        return;
    }
    *n_loc = n;
    *n_start = 0;

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

    #pragma omp parallel for
    for (long int i = 1; i < size; i++) {
        r_cmpx[i] *= ig2[i];
    }

    fftw_execute(r_bwd_plan);

    #pragma omp parallel for
    for (long int i = 0; i < n3r; i++) {
        x[i] = r_real[i];  // Normalization moved inside ig2
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

void init_rfft(int n, int *n_loc, int *n_start) {
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