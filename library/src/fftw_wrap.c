/*Wrappers for FFTW3 library*/

// Order matters here, including complex.h before fftw3.h makes fftw_complex be a complex instead of a double[2]
#include <stdio.h>
#include <stdlib.h>

#include "omp_base.h"

#ifdef __FFTW

#include <complex.h>
#include <fftw3.h>

#define FFTW_OMP_BLANK 0
#define FFTW_OMP_INITIALIZED 1
#define FFTW_OMP_DOCLEANUP 2

int initialized_omp = FFTW_OMP_BLANK;
// int initialized_c = 0;
int initialized_r = 0;

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

void init_fftw_omp() {
    if (initialized_omp != FFTW_OMP_BLANK) {
        return;
    }
    initialized_omp = FFTW_OMP_INITIALIZED;  // Set to 2 to avoid reinitializing but skip the cleanup

    int res;
    int num_threads = get_omp_max_threads();
    if (num_threads > 0) {
        initialized_omp = FFTW_OMP_DOCLEANUP;
        res = fftw_init_threads();
        if (res == 0) {
            printf("Error initializing FFTW threads\n");
            exit(1);
        }
        fftw_plan_with_nthreads(get_omp_max_threads());
        printf("FFTW: Running with %d threads\n", num_threads);
    }
}

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

void init_rfft(int n) {
    if (initialized_r != 0) {
        return;
    }
    int nh = n / 2 + 1;
    initialized_r = 1;
    int tid = get_omp_thread_num();
  
    r_real = (double *)fftw_malloc(n * n * n * sizeof(double));
    r_cmpx = (fftw_complex *)fftw_malloc(n * n * nh * sizeof(fftw_complex));

    if (tid == 0) {
        printf("FFTW: Initializing R-C-R plans\n");
        r_fwd_plan = fftw_plan_dft_r2c_3d(n, n, n, r_real, r_cmpx, FLAG | FFTW_DESTROY_INPUT);
        r_bwd_plan = fftw_plan_dft_c2r_3d(n, n, n, r_cmpx, r_real, FLAG | FFTW_DESTROY_INPUT);
        printf("FFTW: ...DONE\n");
    }
}

void cleanup_fftw() {
    // printf("FFTW: Cleaning up\n");
    // if (initialized_c == 0) {
    //     fftw_destroy_plan(c_fwd_plan);
    //     fftw_destroy_plan(c_bwd_plan);
    //     fftw_free(c_in);
    //     fftw_free(c_out);
    //     initialized_c = 0;
    // }
    if (initialized_r == 0) {
        fftw_destroy_plan(r_fwd_plan);
        fftw_destroy_plan(r_bwd_plan);
        fftw_free(r_real);
        fftw_free(r_cmpx);
        initialized_r = 0;
    }
    if (initialized_omp == FFTW_OMP_DOCLEANUP) {
        fftw_cleanup_threads();
        initialized_omp = FFTW_OMP_BLANK;
    }
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

#else // __FFTW

void init_fftw_omp() {
    fprintf(stderr, "FFTW not enabled\n");
    exit(1);
}

void init_rfft(int n) {
    fprintf(stderr, "FFTW not enabled\n");
    exit(1);
}

void cleanup_fftw() {
    fprintf(stderr, "FFTW not enabled\n");
    exit(1);
}

void rfft_solve(int n, double *b, double *ig2, double *x) {
    fprintf(stderr, "FFTW not enabled\n");
    exit(1);
}

#endif // __FFTW