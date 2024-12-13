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
c_init_fftw_omp = library.init_fftw_omp
c_init_fftw_omp.restype = None
c_init_fftw_omp.argtypes = []
c_init_fftw_omp()

# void init_fftw_c(int n)
c_init_fftw_c = library.init_fftw_c
c_init_fftw_c.restype = None
c_init_fftw_c.argtypes = [ctypes.c_int]
# c_init_fftw()

# void init_fftw_r(int n)
c_init_fftw_r = library.init_fftw_r
c_init_fftw_r.restype = None
c_init_fftw_r.argtypes = [ctypes.c_int]

# void cleanup_fftw()
c_cleanup_fftw = library.cleanup_fftw
c_cleanup_fftw.restype = None
c_cleanup_fftw.argtypes = []

atexit.register(c_cleanup_fftw)

# void c_fft_3d(int n, double *in, complex *out)
c_c_fft_3d = library.c_fft_3d
c_c_fft_3d.restype = None
c_c_fft_3d.argtypes = [
    ctypes.c_int,
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
]

# void c_ifft_3d(int n, complex *in, double *out)
c_c_ifft_3d = library.c_ifft_3d
c_c_ifft_3d.restype = None
c_c_ifft_3d.argtypes = [
    ctypes.c_int,
    npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
]

# void r_fft_3d(int n, double *in, complex *out)
c_r_fft_3d = library.r_fft_3d
c_r_fft_3d.restype = None
c_r_fft_3d.argtypes = [
    ctypes.c_int,
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
]

# void r_ifft_3d(int n, complex *in, double *out)
c_r_ifft_3d = library.r_ifft_3d
c_r_ifft_3d.restype = None
c_r_ifft_3d.argtypes = [
    ctypes.c_int,
    npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
]

# # void fftw_3d(int n, double *in, complex *out)
# c_fftw_3d = library.fftw_3d
# c_fftw_3d.restype = None
# c_fftw_3d.argtypes = [
#     ctypes.c_int,
#     npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
#     npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
# ]

# #void ifftw_3d(int n, complex *in, double *out)
# c_ifftw_3d = library.ifftw_3d
# c_ifftw_3d.restype = None
# c_ifftw_3d.argtypes = [
#     ctypes.c_int,
#     npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
#     npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
# ]

# # void rfftw_3d(int n, double *in, complex *out)
# c_rfftw_3d = library.rfftw_3d
# c_rfftw_3d.restype = None
# c_rfftw_3d.argtypes = [
#     ctypes.c_int,
#     npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
#     npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
# ]

# # void irfftw_3d(int n, complex *in, double *out)
# c_irfftw_3d = library.irfftw_3d
# c_irfftw_3d.restype = None
# c_irfftw_3d.argtypes = [
#     ctypes.c_int,
#     npct.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS'),
#     npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
# ]

# # TEST rfftw_3d
# n = 20
# inp = np.random.rand(n,n,n)
# out = np.empty((n,n,n//2 + 1), dtype=np.complex128)
# print(inp.shape, out.shape, out.size)
# c_rfftw_3d(n, inp, out)
# np_res = np.fft.rfftn(inp)
# print(np_res.shape)
# print(np.linalg.norm(out - np_res))
# # exit()
# back = np.empty((n,n,n), dtype=np.float64)
# c_irfftw_3d(n, out, back)
# back_np = np.fft.irfftn(out, s=(n,n,n))
# print(np.linalg.norm(back - back_np))
# # exit()

# # # TEST fftw_3d
# n = 20
# inp = np.random.rand(n,n,n)
# out = np.zeros((n,n,n), dtype=np.complex128)
# c_fftw_3d(n, inp, out)
# np_res = np.fft.fftn(inp)
# print(np.linalg.norm(out - np_res))
# back = np.zeros((n,n,n), dtype=np.float64)
# c_ifftw_3d(n, out, back)
# back_np = np.fft.ifftn(out)
# print(np.linalg.norm(back - back_np))
# # logger.info("fftw_3d test passed")
# exit()

# void ffft_solve(int n, double *b, double *ig2, double *x)
c_ffft_solve = library.ffft_solve
c_ffft_solve.restype = None
c_ffft_solve.argtypes = [
    ctypes.c_int,
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
]

# void ffft_solve_r(int n, double *b, double *ig2, double *x)
c_ffft_solve_r = library.ffft_solve_r
c_ffft_solve_r.restype = None
c_ffft_solve_r.argtypes = [
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

logger.info("C API loaded successfully")
num_threads = c_get_omp_info()
if num_threads > 0:
    logger.info("Number of OpenMP threads: %d", c_get_omp_info())
else:
    logger.warning("OpenMP not enabled")
