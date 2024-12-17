import atexit
import ctypes
import os
import signal

import numpy as np
import numpy.ctypeslib as npct

from .loggers import logger

__all__ = ['c_laplace', 'c_ddot', 'c_daxpy', 'c_daxpy2']

# Import from shared library next to this file as a package
library = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libmaze_poisson.so'))

# int get_omp_info()
try:
    c_get_omp_info = library.get_omp_info
    c_get_omp_info.restype = ctypes.c_int
    c_get_omp_info.argtypes = []
except:
    c_get_omp_info = lambda: 0

# init_fftw_omp()
try:
    c_init_fftw_omp = library.init_fftw_omp
    c_init_fftw_omp.restype = None
    c_init_fftw_omp.argtypes = []
except:
    logger.warning("C Interface to FFTW not available. Using numpy instead.")
    init_fftw_omp = lambda: None
    init_fft = lambda n: None
    init_rfft = lambda n: None
    fft_3d = lambda n, in_, out: np.fft.fftn(in_, out=out)
    ifft_3d = lambda n, in_, out: np.fft.ifftn(in_, out=out)
    rfft_3d = lambda n, in_, out: np.fft.rfftn(in_, out=out)
    irfft_3d = lambda n, in_, out: np.fft.irfftn(in_, out=out)
else:
    logger.info("C_API: Interface to FFTW available")
    c_init_fftw_omp()

    # void cleanup_fftw()
    c_cleanup_fftw = library.cleanup_fftw
    c_cleanup_fftw.restype = None
    c_cleanup_fftw.argtypes = []

    atexit.register(c_cleanup_fftw)

    # void init_fft(int n)
    init_fft = library.init_fft
    init_fft.restype = None
    init_fft.argtypes = [ctypes.c_int]
    # c_init_fftw()

    # void init_rfft(int n)
    init_rfft = library.init_rfft
    init_rfft.restype = None
    init_rfft.argtypes = [ctypes.c_int]

    # void fft_3d(int n, double *in, complex *out)
    fft_3d = library.fft_3d
    fft_3d.restype = None
    fft_3d.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
    ]

    # void ifft_3d(int n, complex *in, double *out)
    ifft_3d = library.ifft_3d
    ifft_3d.restype = None
    ifft_3d.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]

    # void rfft_3d(int n, double *in, complex *out)
    rfft_3d = library.rfft_3d
    rfft_3d.restype = None
    rfft_3d.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
    ]

    # void irfft_3d(int n, complex *in, double *out)
    irfft_3d = library.irfft_3d
    irfft_3d.restype = None
    irfft_3d.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]

    # void fft_solve(int n, double *b, double *ig2, double *x)
    fft_solve = library.fft_solve
    fft_solve.restype = None
    fft_solve.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]

    # void rfft_solve(int n, double *b, double *ig2, double *x)
    rfft_solve = library.rfft_solve
    rfft_solve.restype = None
    rfft_solve.argtypes = [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]


# void laplace_filter(long double *u, long double *u_new, int n)
c_laplace = library.laplace_filter
c_laplace.restype = None
c_laplace.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_int,
]

# double ddot(double *u, double *v, int n)
c_ddot = library.ddot
c_ddot.restype = ctypes.c_double
c_ddot.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_int,
]

# void daxpy(double *u, double *v, double *result, double alpha, int n) {
c_daxpy = library.daxpy
c_daxpy.restype = None
c_daxpy.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int,
]

# void daxpy2(double *v, double *u, double alpha, int n)
c_daxpy2 = library.daxpy2
c_daxpy2.restype = None
c_daxpy2.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int,
]

# int conj_grad(double *b, double *x0, double *x, double tol, long int n) {
c_conj_grad = library.conj_grad
c_conj_grad.restype = ctypes.c_int
c_conj_grad.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int
]

# Enable Ctrl-C to interrupt the C code
signal.signal(signal.SIGINT, signal.SIG_DFL)

logger.info("C_API: Loaded successfully")
num_threads = c_get_omp_info()
if num_threads > 0:
    logger.info("C_API: Number of OpenMP threads: %d", c_get_omp_info())
else:
    logger.warning("C_API: OpenMP not enabled")
