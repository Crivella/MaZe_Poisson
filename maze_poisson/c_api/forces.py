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
