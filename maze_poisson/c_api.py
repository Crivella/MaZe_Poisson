import ctypes
import os
import signal

import numpy as np
import numpy.ctypeslib as npct
from scipy.sparse.linalg import LinearOperator, cg

from .loggers import logger

# Import from shared library next to this file as a package
try:
    library = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libmaze_poisson.so'))
except:
    logger.warning("C_API: Could not load shared library.")
    library = None
else:
    logger.info("C_API: Loaded successfully")

try:
    c_get_omp_info = library.get_omp_info
    c_get_omp_info.restype = ctypes.c_int
    c_get_omp_info.argtypes = []
except:
    c_get_omp_info = lambda: 0

# Interface and fallbacks to fftw_wrap.c
try:
    init_fftw_omp = library.init_fftw_omp
    init_fftw_omp.restype = None
    init_fftw_omp.argtypes = []
except:
    logger.warning("C_API: Interface to FFTW not available. Using numpy instead.")
    init_fftw_omp = lambda: None
    cleanup_fftw = lambda: None
    init_rfft = lambda n: None
    rfft_solve = lambda n, in_, ig2, out: np.fft.irfftn(np.fft.rfftn(in_) * ig2, out=out, s=in_.shape)
else:
    logger.info("C_API: Interface to FFTW available")
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


# Interface and fallbacks to laplace.c
try:
    # void laplace_filter(long double *u, long double *u_new, int n)
    c_laplace = library.laplace_filter
    c_laplace.restype = None
    c_laplace.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
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
except:
    logger.warning("C_API: laplace_filter not available. Using Python instead.")
    def c_laplace(u_in: np.ndarray, u_out: np.ndarray, n: int) -> None:
        """Apply the Laplace filter."""
        u_out[:] = -6 * u_in.reshape(n, n, n)

        u_out[1:,:,:] += u_in[:-1,:,:]
        u_out[:-1,:,:] += u_in[1:,:,:]
        u_out[-1,:,:] += u_in[0,:,:]
        u_out[0,:,:] += u_in[-1,:,:]

        u_out[:,1:,:] += u_in[:,:-1,:]
        u_out[:,:-1,:] += u_in[:,1:,:]
        u_out[:,-1,:] += u_in[:,0,:]
        u_out[:,0,:] += u_in[:,-1,:]

        u_out[:,:,1:] += u_in[:,:,:-1]
        u_out[:,:,:-1] += u_in[:,:,1:]
        u_out[:,:,-1] += u_in[:,:,0]
        u_out[:,:,0] += u_in[:,:,-1]

    def _laplace(u: np.ndarray, n: int) -> np.ndarray:
        u_out = np.empty_like(u)
        c_laplace(u, u_out, n)
        return u_out

    def c_conj_grad(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
        if x0 is not None:
            x0 = x0.flatten()
        func = lambda x: _laplace(x.reshape(n, n, n), n).flatten()
        res, i = cg(
            LinearOperator((b.size, b.size), matvec=func),
            b.flatten(),
            x0=x0,
            atol=tol
            )
        if i > 0:
            raise ValueError(f'Conjugate gradient did not converge {i} iterations')
        x[:] = res.reshape(n, n, n)
        return i

# Interface and fallbacks to forces.c
try:
    # double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces)
    c_compute_force_fd = library.compute_force_fd
    c_compute_force_fd.restype = ctypes.c_double
    c_compute_force_fd.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]
except:
    logger.warning("C_API: compute_force_fd not available. Using Python instead.")

    def c_compute_force_fd(
            N: int, N_p: int, h: float,
            phi_v: np.ndarray, q: np.ndarray, neighbors: np.ndarray,
            forces: np.ndarray
        ) -> float:
        """Compute the forces from the field using finite differences.

        Args:
            N (int): Grid size. (Not used only to match the signature)
            N_p (int): Number of particles. (Not used only to match the signature)
            h (float): Grid spacing.
            phi_v (np.ndarray): Electrostatic potential. Shape (N, N, N).
            q (np.ndarray): 3D Charge. Shape (N, N, N).
            neighbors (np.ndarray): Indices of the neighbors. Shape (N_p, 8, 3).
            forces (np.ndarray): Output forces. Shape (N_p, 3).

        Returns:
            float: Total charge contribution.
        """
        h *= 2  # Double the grid spacing
        q_neighbors = q[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]
        for axis in range(3):
            E_ax = (np.roll(phi_v, -1, axis=axis) - np.roll(phi_v, 1, axis=axis)) / h
            E_neighbors = E_ax[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]
            forces[:, axis] = -np.sum(q_neighbors * E_neighbors, axis=1)

        q_tot = np.sum(q_neighbors)
        return q_tot

# Enable Ctrl-C to interrupt the C code
signal.signal(signal.SIGINT, signal.SIG_DFL)

if library:
    # Check if OpenMP is enabled
    num_threads = c_get_omp_info()
    if num_threads > 0:
        logger.info("C_API: Number of OpenMP threads: %d", c_get_omp_info())
    else:
        logger.warning("C_API: OpenMP not enabled")
