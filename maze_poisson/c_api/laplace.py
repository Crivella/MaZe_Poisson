# Interface and fallbacks to laplace.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi
from .laplace_fallbacks import conj_grad, laplace

capi.register_function(
    'laplace_filter', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int,
    ],
    laplace
)

capi.register_function(
    'conj_grad', ctypes.c_int, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int
    ],
    conj_grad
)
