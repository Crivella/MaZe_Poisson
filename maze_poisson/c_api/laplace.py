# Interface and fallbacks to laplace.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct
from scipy.sparse.linalg import LinearOperator, cg

from . import library, logger, mpi

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

    mpi.send_previous(u_in[0, :, :])
    mpi.send_next(u_in[-1, :, :])

    mpi.recv_previous(tmp_bot)
    mpi.recv_next(tmp_top)

    _c_laplace_mpi(u_in, u_out, tmp_top, tmp_bot, u_in.shape[0], n)

if mpi and mpi.size > 1:
    c_laplace = c_laplace_mpi
    c_conj_grad = c_conj_grad_mpi
else:
    c_laplace = c_laplace_single
    c_conj_grad = c_conj_grad_single