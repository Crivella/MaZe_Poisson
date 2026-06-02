#ifndef __MP_FFTW_H
#define __MP_FFTW_H

// Order matters here, including complex.h before fftw3.h makes fftw_complex be a complex instead of a double[2]
#include <complex.h>

#ifdef __FFTW

#ifdef __FFTW_MPI
#include <fftw3-mpi.h>
#else  // __FFTW_MPI
#include <fftw3.h>
#endif  // __FFTW_MPI

#else  // __FFTW

typedef complex double fftw_complex;

#endif  // __FFTW

#define FFTW_BLANK 0
#define FFTW_INITIALIZED 1
#define FFTW_DOCLEANUP 2

void init_rfft(int n, int *n_loc, int *n_start);
void cleanup_fftw();
void rfft_3d(int n, int n_loc, double *in, fftw_complex *out);
void irfft_3d(int n, int n_loc, fftw_complex *in, double *out);
void rfft_solve(int n, int n_loc, double *b, double *ig2, double *x);

#endif  // __MP_FFTW_H