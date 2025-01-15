# Interface and fallbacks to forces.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library, logger, mpi

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

    # double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces)
    c_compute_tf_forces = library.compute_tf_forces
    c_compute_tf_forces.restype = ctypes.c_double
    c_compute_tf_forces.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]
except:
    logger.warning("C_API: compute_force_fd not available. Using Python instead.")

    from .forces_fallbacks import (_c_compute_forces_fd_mpi,
                                   c_compute_force_fd_single,
                                   c_compute_tf_forces)


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


c_compute_force_fd = c_compute_forces_fd_mpi if mpi else c_compute_force_fd_single
