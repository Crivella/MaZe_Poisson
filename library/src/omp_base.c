#include "omp_base.h"

#ifdef _OPENMP
int get_omp_thread_num() {
    return omp_get_thread_num();
}

int get_omp_max_threads() {
    return omp_get_max_threads();
}

#else // _OPENMP

int get_omp_thread_num() {
    return 0;
}

int get_omp_max_threads() {
    return 0;
}

#endif // _OPENMP