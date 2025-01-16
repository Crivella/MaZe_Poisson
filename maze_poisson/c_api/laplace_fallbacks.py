"""Fallback implementations of the Laplace filter and conjugate gradient functions."""
import numpy as np

from . import mpi


def c_laplace(u_in: np.ndarray, u_out: np.ndarray, n: int) -> None:
    """Apply the Laplace filter."""
    bot, top = mpi.get_bot_top(u_in)

    u_out[:] = -6 * u_in

    u_out[1:  ,:   ,:  ] += u_in[:-1,:,:]
    u_out[:-1 ,:   ,:  ] += u_in[1:,:,:]
    u_out[-1  ,:   ,:  ] += top  # u_in[0,:,:]
    u_out[ 0  ,:   ,:  ] += bot  # u_in[-1,:,:]

    u_out[:   ,1:  ,:  ] += u_in[:,:-1,:]
    u_out[:   ,:-1 ,:  ] += u_in[:,1:,:]
    u_out[:   ,-1  ,:  ] += u_in[:,0,:]
    u_out[:   ,0   ,:  ] += u_in[:,-1,:]

    u_out[:   ,:   ,1: ] += u_in[:,:,:-1]
    u_out[:   ,:   ,:-1] += u_in[:,:,1:]
    u_out[:   ,:   ,-1 ] += u_in[:,:,0]
    u_out[:   ,:   ,0  ] += u_in[:,:,-1]


def ddot(u: np.ndarray, v: np.ndarray) -> float:
    res = np.sum(u * v)
    return mpi.all_reduce(res)

def c_conj_grad(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
    limit = n * n // 2
    Ap = np.empty_like(b)
    if x0 is None:
        x0 = np.zeros_like(b)
        r = -b
    else:
        r = np.empty_like(b)
        c_laplace(x0, r, n)
        r -= b
    x[:] = x0

    p = r / 6.0

    r_dot_v = - ddot(r, p)

    res = -1
    iter_ = 1
    while iter_ < limit:
        c_laplace(p, Ap, n)

        alpha = r_dot_v / ddot(p, Ap)
        x[:] = x + alpha * p
        r[:] = r + alpha * Ap

        rn_dot_rn = ddot(r, r)
        if rn_dot_rn ** 0.5 < tol:
            res = iter_
            break

        rn_dot_vn = - rn_dot_rn / 6.0
        beta = rn_dot_vn / r_dot_v
        r_dot_v = rn_dot_vn
        p[:] = r / 6.0 + beta * p

        iter_ += 1

    return res
