# Interface and fallbacks to laplace.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library

try:
    # void laplace_filter(double *u, double *u_new, int n)
    c_laplace = library.laplace_filter
    c_laplace.restype = None
    c_laplace.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int,
    ]
except:
    from .laplace_fallbacks import c_laplace

try:
    # int conj_grad(double *b, double *x0, double *x, double tol, nt n) {
    c_conj_grad = library.conj_grad
    c_conj_grad.restype = ctypes.c_int
    c_conj_grad.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int
    ]
except:
    from .laplace_fallbacks import c_conj_grad
