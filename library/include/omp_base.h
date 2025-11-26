#ifndef __MP_OMP_H
#define __MP_OMP_H

#ifdef _OPENMP
#include <omp.h>
#endif

int get_omp_thread_num();
int get_omp_max_threads();

#endif // __MP_OMP_H