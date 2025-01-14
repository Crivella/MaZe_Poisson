import ctypes
import os
import signal

import numpy as np
import numpy.ctypeslib as npct
from scipy.sparse.linalg import LinearOperator, cg

from .loggers import logger
from .mpi import MPIBase

mpi = MPIBase()

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
    # void laplace_filter(double *u, double *u_new, int n)
    c_laplace_single = library.laplace_filter
    c_laplace_single.restype = None
    c_laplace_single.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int,
    ]

    # int conj_grad(double *b, double *x0, double *x, double tol, nt n) {
    c_conj_grad_single = library.conj_grad
    c_conj_grad_single.restype = ctypes.c_int
    c_conj_grad_single.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int
    ]

    # void laplace_filter_mpi(double *u, double *u_new, double *top, double *bot, int n_loc, int n)
    _c_laplace_mpi = library.laplace_filter_mpi
    _c_laplace_mpi.restype = None
    _c_laplace_mpi.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
    ]

    # double conj_grad_mpi_iter1(double *Ap, double *p, int n_loc, int n)
    c_conj_grad_mpi_iter1 = library.conj_grad_mpi_iter1
    c_conj_grad_mpi_iter1.restype = ctypes.c_double
    c_conj_grad_mpi_iter1.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
    ]

    # double conj_grad_mpi_iter2(double *Ap, double *p, double *r, double *x, double alpha, int n_loc, int n)
    c_conj_grad_mpi_iter2 = library.conj_grad_mpi_iter2
    c_conj_grad_mpi_iter2.restype = ctypes.c_double
    c_conj_grad_mpi_iter2.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]

    # void conj_grad_mpi_iter3(double *r, double *p, double *r_dot, int n_loc, int n)
    c_conj_grad_mpi_iter3 = library.conj_grad_mpi_iter3
    c_conj_grad_mpi_iter3.restype = None
    c_conj_grad_mpi_iter3.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
    ]
    # print('c_api.py: c_conj_grad_mpi_iter3 is available')
except:
    logger.warning("C_API: laplace_filter not available. Using Python instead.")
    def c_laplace_single(u_in: np.ndarray, u_out: np.ndarray, n: int) -> None:
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

    def _laplace_single(u: np.ndarray, n: int) -> np.ndarray:
        u_out = np.empty_like(u)
        c_laplace_single(u, u_out, n)
        return u_out

    def c_conj_grad_single(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
        if x0 is not None:
            x0 = x0.flatten()
        func = lambda x: _laplace_single(x.reshape(n, n, n), n).flatten()
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

    def _c_laplace_mpi(u_in: np.ndarray, u_out: np.ndarray, top: np.ndarray, bot: np.ndarray, n_loc: int, n: int):
        u_out[:] = -6 * u_in
        u_out[1:, :, :] += u_in[:-1, :, :]
        u_out[:-1, :, :] += u_in[1:, :, :]
        u_out[-1, :, :] += top
        u_out[0, :, :] += bot

        u_out[:, 1:, :] += u_in[:, :-1, :]
        u_out[:, :-1, :] += u_in[:, 1:, :]
        u_out[:, -1, :] += u_in[:, 0, :]
        u_out[:, 0, :] += u_in[:, -1, :]

        u_out[:, :, 1:] += u_in[:, :, :-1]
        u_out[:, :, :-1] += u_in[:, :, 1:]
        u_out[:, :, -1] += u_in[:, :, 0]
        u_out[:, :, 0] += u_in[:, :, -1]

    def c_conj_grad_mpi_iter1(Ap: np.ndarray, p: np.ndarray, n_loc: int, n: int) -> float:
        return np.sum(p * Ap)

    def c_conj_grad_mpi_iter2(Ap: np.ndarray, p: np.ndarray, r: np.ndarray, x: np.ndarray, alpha: float, n_loc: int, n: int) -> float:
        x[:] = x + alpha * p
        r[:] = r + alpha * Ap
        return np.sum(r * r)

    def c_conj_grad_mpi_iter3(r: np.ndarray, p: np.ndarray, r_dot: list, n_loc: int, n: int):
        rn_dot_vn = - r_dot[1] / 6.0
        beta = rn_dot_vn / r_dot[0]
        r_dot[0] = rn_dot_vn
        p[:] = r / 6.0 + beta * p


def c_conj_grad_mpi(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
    """Conjugate gradient method with MPI. Every rank owns a slab of the grid and sends its
    top/bot slices to the next/previous rank. The ddots need to be all_reduce'd.

    Args:
        b (np.ndarray): Right-hand side. Shape (N_loc, N, N).
        x0 (np.ndarray): Initial guess. Shape (N_loc, N, N).
        x (np.ndarray): Solution. Shape (N_loc, N, N).
        tol (float): Tolerance.
        n (int): Grid size.

    Returns:
        int: Number of iterations. Returns -1 if the method did not converge.
    """
    n_loc = b.shape[0]
    limit = n * n // 2
    Ap = np.empty_like(b)
    if x0 is None:
        x0 = np.zeros_like(b)
        r = np.copy(b)
    else:
        r = np.empty_like(b)
        c_laplace_mpi(x0, r, n)
        r -= b
    x[:] = x0

    p = r / 6.0

    r_dot_v = -np.sum(r * r) / 6.0
    r_dot_v = mpi.all_reduce(r_dot_v)

    res = -1
    iter_ = 1
    r_dot = np.array([r_dot_v, 0])
    while iter_ < limit:
        c_laplace_mpi(p, Ap, n)
        pAp = c_conj_grad_mpi_iter1(Ap, p, n_loc, n)
        pAp = mpi.all_reduce(pAp)
        alpha = r_dot[0] / pAp
        app = c_conj_grad_mpi_iter2(Ap, p, r, x, alpha, n_loc, n)
        r_dot[1] = mpi.all_reduce(app)
        if r_dot[1] ** 0.5 < tol:
            res = iter_
            break
        c_conj_grad_mpi_iter3(r, p, r_dot, n_loc, n)
        iter_ += 1

    return res

def c_laplace_mpi(u_in: np.ndarray, u_out: np.ndarray, n: int):
    """Apply the Laplace filter with MPI. Every ranks owns a slab of the grid and sends its
    top/bot slices to the next/previous rank.

    Args:
        u_in (np.ndarray): Input grid. Shape (N_loc, N, N).
        u_out (np.ndarray): Output grid. Shape (N_loc, N, N).
        n (int): Grid size.
    """
    tmp_top = np.empty_like(u_in[0, :, :])
    tmp_bot = np.empty_like(u_in[-1, :, :])

    # print(f'Rank {mpi.rank}: SENDING')
    mpi.send_previous(u_in[0, :, :])
    mpi.send_next(u_in[-1, :, :])

    # print(f'Rank {mpi.rank}: RECEIVING')
    mpi.recv_previous(tmp_bot)
    mpi.recv_next(tmp_top)

    _c_laplace_mpi(u_in, u_out, tmp_top, tmp_bot, u_in.shape[0], n)

if mpi and mpi.size > 1:
    c_laplace = c_laplace_mpi
    c_conj_grad = c_conj_grad_mpi
else:
    c_laplace = c_laplace_single
    c_conj_grad = c_conj_grad_single

# Interface and fallbacks to forces.c
try:
    # double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces)
    c_compute_force_fd_single = library.compute_force_fd
    c_compute_force_fd_single.restype = ctypes.c_double
    c_compute_force_fd_single.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]

    # double compute_force_fd_mpi(int n_grid, int n_loc, int n_start, int n_p, double h, double *phi, double *bot, double *top, double *q, long int *neighbors, double *forces)
    _c_compute_forces_fd_mpi = library.compute_force_fd_mpi
    _c_compute_forces_fd_mpi.restype = ctypes.c_double
    _c_compute_forces_fd_mpi.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]
except:
    logger.warning("C_API: compute_force_fd not available. Using Python instead.")

    def c_compute_force_fd_single(
            N: int, N_p: int, h: float,
            phi_v: np.ndarray, q: np.ndarray, neighbors: np.ndarray,
            forces: np.ndarray,
            N_loc: int = None, N_start: int = None
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

    # double compute_force_fd_mpi(int n_grid, int n_loc, int n_start, int n_p, double h, double *phi, double *bot, double *top, double *q, long int *neighbors, double *forces)
    def _c_compute_forces_fd_mpi(
            N: int, N_loc: int, N_start: int, N_p: int, h: float,
            phi: np.ndarray, bot: np.ndarray, top: np.ndarray, q: np.ndarray, neighbors: np.ndarray,
            forces: np.ndarray
        ) -> float:
        N_end = N_start + N_loc
        h *= 2
        E_x = np.zeros_like(phi)
        E_x[:-1] += phi[1:]
        E_x[-1] += top
        E_x[1:] -= phi[:-1]
        E_x[0] -= bot
        E_x /= h
        E_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / h
        E_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / h

        # forces = self.particles.forces_elec
        forces.fill(0)
        q_tot = 0
        for i,neigh in enumerate(neighbors):
            for x,y,z in neigh:
                if x < N_start or x >= N_end:
                    continue
                s = x - N_start
                qn = q[s, y, z]
                q_tot += qn
                forces[i][0] -= qn * E_x[s, y, z]
                forces[i][1] -= qn * E_y[s, y, z]
                forces[i][2] -= qn * E_z[s, y, z]

        return q_tot

def c_compute_forces_fd_mpi(
        N: int, N_p: int, h: float,
        phi: np.ndarray, q: np.ndarray, neighbors: np.ndarray,
        forces: np.ndarray,
         N_loc: int = None, N_start: int = None
    ) -> float:
    tmp_top = np.empty_like(phi[0, :, :])
    tmp_bot = np.empty_like(phi[0, :, :])
    mpi.send_previous(phi[0, :, :])
    mpi.send_next(phi[-1, :, :])
    mpi.recv_previous(tmp_bot)
    mpi.recv_next(tmp_top)

    q_tot = _c_compute_forces_fd_mpi(N, N_loc, N_start, N_p, h, phi, tmp_bot, tmp_top, q, neighbors, forces)
    q_tot = mpi.comm.allreduce(q_tot)
    mpi.all_reduce_inplace(forces)

    return q_tot


if mpi and mpi.size > 1:
    c_compute_force_fd = c_compute_forces_fd_mpi
else:
    c_compute_force_fd = c_compute_force_fd_single

# Enable Ctrl-C to interrupt the C code
signal.signal(signal.SIGINT, signal.SIG_DFL)

if library:
    # Check if OpenMP is enabled
    num_threads = c_get_omp_info()
    if num_threads > 0:
        logger.info("C_API: Number of OpenMP threads: %d", c_get_omp_info())
    else:
        logger.warning("C_API: OpenMP not enabled")
