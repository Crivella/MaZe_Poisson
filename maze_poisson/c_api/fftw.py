# Interface and fallbacks to fftw_wrap.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library

try:
    init_fftw_omp = library.init_fftw_omp
    init_fftw_omp.restype = None
    init_fftw_omp.argtypes = []

    # void cleanup_fftw()
    cleanup_fftw = library.cleanup_fftw
    cleanup_fftw.restype = None
    cleanup_fftw.argtypes = []

    # void init_rfft(int n)
    init_rfft = library.init_rfft
    init_rfft.restype = None
    init_rfft.argtypes = [ctypes.c_int]

    # void rfft_solve(int n, double *b, double *ig2, double *x)
    rfft_solve = library.rfft_solve
    rfft_solve.restype = None
    rfft_solve.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]
except:
    # logger.warning("C_API: Interface to FFTW not available. Using numpy instead.")
    from .fftw_fallbacks import (cleanup_fftw, init_fftw_omp, init_rfft,
                                 rfft_solve)
# else:
#     logger.info("C_API: Interface to FFTW available")