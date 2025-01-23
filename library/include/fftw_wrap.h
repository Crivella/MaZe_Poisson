#ifndef __MP_FFTW_H
#define __MP_FFTW_H

void init_fftw_omp();
void init_rfft(int n);
void cleanup_fftw();
void rfft_solve(int n, double *b, double *ig2, double *x);

#endif  // __MP_FFTW_H
