#ifndef __MP_FFTW_H
#define __MP_FFTW_H

#ifdef __FFTW
// Order matters here, including complex.h before fftw3.h makes fftw_complex be a complex instead of a double[2]
#include <complex.h>
#include <fftw3.h>
#endif

#define FFTW_BLANK 0
#define FFTW_INITIALIZED 1
#define FFTW_DOCLEANUP 2

void init_rfft(int n);
void cleanup_fftw();
void rfft_solve(int n, double *b, double *ig2, double *x);

#endif  // __MP_FFTW_H