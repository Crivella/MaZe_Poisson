# Interface and fallbacks to laplace.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library, mpi

try:
    print('Using C laplace')
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
    print('Using C conj_grad')
    # int conj_grad(double *b, double *x0, double *x, double tol, nt n) {
    _c_conj_grad = library.conj_grad
    _c_conj_grad.restype = ctypes.c_int
    _c_conj_grad.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int
    ]
except:
    from .laplace_fallbacks import c_conj_grad as _c_conj_grad

from .mympy import c_init_mpi


def c_conj_grad(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
    c_init_mpi(n, mpi.comm_address)
    return _c_conj_grad(b, x0, x, tol, n)
