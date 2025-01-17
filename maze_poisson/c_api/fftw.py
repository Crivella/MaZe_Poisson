# Interface and fallbacks to fftw_wrap.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi
from .fftw_fallbacks import cleanup_fftw, init_fftw_omp, init_rfft, rfft_solve

capi.register_function('init_fftw_omp', None, [], init_fftw_omp)
capi.register_function('cleanup_fftw', None, [], cleanup_fftw)
capi.register_function('init_rfft', None, [ctypes.c_int], init_rfft)
capi.register_function(
    'rfft_solve', None, [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
    rfft_solve
)
