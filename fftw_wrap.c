/*Wrappers for FFTW3 library*/

#include <stdio.h>
#include <stdbool.h>
#include <fftw3.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

bool initialized = false;

void init_fftw() {
    if (initialized) {
        return;
    }
    initialized = true;
    printf("Initializing fftw\n");
#ifdef _OPENMP
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    printf("Running fftw with %d threads\n", omp_get_max_threads());
#endif
}

/*Perform a 3D fftw real to complex*/
void fftw_3d(int n, double *in, complex *out) {
    init_fftw();
    fftw_plan plan;
    int i, j, k;
    long int n2 = n * n;
    long int n3 = n2 * n;
    fftw_complex *in_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));
    fftw_complex *out_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                in_c[i * n2 + j * n + k][0] = in[i * n2 + j * n + k];
                in_c[i * n2 + j * n + k][1] = 0.0;
            }
        }
    }

    plan = fftw_plan_dft_3d(n, n, n, in_c, out_c, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                out[i * n2 + j * n + k] = out_c[i * n2 + j * n + k][0] + I * out_c[i * n2 + j * n + k][1];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in_c);
    fftw_free(out_c);
}

/*Perform a inverse 3D fftw complex to real*/
void ifftw_3d(int n, complex *in, double *out) {
    init_fftw();
    fftw_plan plan;
    int i, j, k;
    long int n2 = n * n;
    long int n3 = n2 * n;
    fftw_complex *in_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));
    fftw_complex *out_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                in_c[i * n2 + j * n + k][0] = creal(in[i * n2 + j * n + k]);
                in_c[i * n2 + j * n + k][1] = cimag(in[i * n2 + j * n + k]);
            }
        }
    }

    plan = fftw_plan_dft_3d(n, n, n, in_c, out_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                out[i * n2 + j * n + k] = out_c[i * n2 + j * n + k][0] / n3;
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in_c);
    fftw_free(out_c);
}

/*Solve Ax=b where A is the laplacian using FFTS*/
void ffft_solve(int n, double *b, double *ig2, double *x) {
    init_fftw();
    long int i;
    long int n2 = n * n;
    long int n3 = n2 * n;
    fftw_plan plan;
    fftw_complex *a_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));
    fftw_complex *b_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));

    // #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        a_c[i][0] = b[i];
        a_c[i][1] = 0.0;
    }

    plan = fftw_plan_dft_3d(n, n, n, a_c, b_c, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // #pragma omp parallel
    for (i = 0; i < n3; i++) {
        b_c[i][0] *= ig2[i];
        b_c[i][1] *= ig2[i];
    }

    fftw_destroy_plan(plan);

    plan = fftw_plan_dft_3d(n, n, n, b_c, a_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        x[i] = a_c[i][0] / n3;
    }

    fftw_destroy_plan(plan);
    fftw_free(a_c);
    fftw_free(b_c);
}

/*Perform a 3D fftw real to complex using r2c*/
void rfftw_3d(int n, double *in, complex *out) {
    init_fftw();
    fftw_plan plan;
    int i, j, k;
    int nh = n / 2 + 1;
    long int n2 = n * n;
    long int n3 = n2 * nh;
    long int idx;
    fftw_complex *out_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));
    printf("n3: %ld\n", n3);
    fflush(stdout);

    plan = fftw_plan_dft_r2c_3d(n, n, n, in, out_c, FFTW_ESTIMATE);
    fftw_execute(plan);

    printf("RFFTW_3D: EXECUTE DONE\n");
    fflush(stdout);

    // OUT is of size N x N x (N/2 + 1)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < nh; k++) {
                idx = i * n2 + j * n + k;
                out[idx] = out_c[idx][0] + I * out_c[idx][1];
                printf("RFFTW_3D: out[%ld]: %f + %f i\n", idx, creal(out[idx]), cimag(out[idx]));
            }
        }
    }

    printf("RFFTW_3D: remap int out DONE\n");
    fflush(stdout);

    fftw_destroy_plan(plan);
    printf("RFFTW_3D: destroy_plan DONE\n");
    fflush(stdout);
    fftw_free(out_c);
    printf("RFFTW_3D: free DONE\n");
    fflush(stdout);
}

/*Perform a inverse 3D fftw complex to real using c2r*/
void irfftw_3d(int n, complex *in, double *out) {
    init_fftw();
    fftw_plan plan;
    int i, j, k;
    long int n2 = n * n;
    long int n3 = n2 * (n/2 + 1);
    fftw_complex *in_c = (fftw_complex *)fftw_malloc(n3 * sizeof(fftw_complex));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n / 2 + 1; k++) {
                in_c[i * n2 + j * (n / 2 + 1) + k][0] = creal(in[i * n2 + j * n + k]);
                in_c[i * n2 + j * (n / 2 + 1) + k][1] = cimag(in[i * n2 + j * n + k]);
            }
        }
    }

    plan = fftw_plan_dft_c2r_3d(n, n, n, in_c, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (i = 0; i < n3; i++) {
        out[i] /= n3;
    }

    fftw_destroy_plan(plan);
    fftw_free(in_c);
}
