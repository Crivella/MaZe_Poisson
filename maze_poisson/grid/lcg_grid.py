from collections import deque

import numpy as np

from .. import c_api
from ..mpi import MPIBase
from .base_grid import BaseGrid

mpi = MPIBase()

class LCGGrid(BaseGrid):
    def init_grids(self):
        """Initialize the grids."""
        self.shape = (self.N,)*3
        self.y = np.zeros(self.shape, dtype=float)  # right-hand side of the preconditioned Poisson equation
        self.q = np.zeros(self.shape, dtype=float)  # charge vector - q for every grid point
        self.tmp = np.empty(self.shape, dtype=float)  # temporary array for the Poisson equation
        # 2-step phi for the Verlet algorithm
        self._phi = deque([np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)], maxlen=2)

    def initialize_field(self):
        """Initialize the field."""
        c_api.c_conj_grad(- 4 * np.pi * self.q / self.h, self.phi, self.tmp, self.tol, self.N)
        self._phi.append(np.copy(self.tmp))

    @BaseGrid.timeit
    def update_field(self):
        """Update the field."""
        phi = 2 * self.phi - self.phi_prev
        c_api.c_laplace(phi, self.tmp, self.N)
        sigma_p = 4 * np.pi * self.q / self.h + self.tmp

        y_new = np.empty_like(self.y)
        self.n_iters = c_api.c_conj_grad(sigma_p, self.y, y_new, self.tol, self.N)

        phi -= y_new
        self.y = y_new
        self._phi.append(phi)

    def gather(self, vec):
        self.gathered = vec

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]

def MatrixVectorProduct_manual(v):
    # N = int(v.shape[0]**(1/3) + 0.5)
    N = 100
    v = v.reshape((N,N,N))
    res = -6 * np.copy(v)

    res[1:,:,:] += v[:-1,:,:]
    res[:-1,:,:] += v[1:,:,:]
    res[-1,:,:] += v[0,:,:]
    res[0,:,:] += v[-1,:,:]

    res[:,1:,:] += v[:,:-1,:]
    res[:,:-1,:] += v[:,1:,:]
    res[:,-1,:] += v[:,0,:]
    res[:,0,:] += v[:,-1,:]

    res[:,:,1:] += v[:,:,:-1]
    res[:,:,:-1] += v[:,:,1:]
    res[:,:,-1] += v[:,:,0]
    res[:,:,0] += v[:,:,-1]

    return res

# Every process owns a slab of the grid
def laplace_filter_mpi(u_in: np.ndarray, u_out: np.ndarray, n: int):
    tmp_top = np.empty_like(u_in[0, :, :])
    tmp_bot = np.empty_like(u_in[-1, :, :])

    # print(f'Rank {mpi.rank}: SENDING')
    mpi.send_previous(u_in[0, :, :])
    mpi.send_next(u_in[-1, :, :])

    # print(f'Rank {mpi.rank}: RECEIVING')
    mpi.recv_previous(tmp_bot)
    mpi.recv_next(tmp_top)

    # print(f'Rank {mpi.rank}: {tmp_top.shape} {tmp_top[:5,:5]}')
    # print(f'Rank {mpi.rank}: {tmp_bot.shape} {tmp_bot[:5,:5]}')

    u_out[:] = -6 * u_in
    u_out[1:, :, :] += u_in[:-1, :, :]
    u_out[:-1, :, :] += u_in[1:, :, :]
    u_out[-1, :, :] += tmp_top
    u_out[0, :, :] += tmp_bot

    u_out[:, 1:, :] += u_in[:, :-1, :]
    u_out[:, :-1, :] += u_in[:, 1:, :]
    u_out[:, -1, :] += u_in[:, 0, :]
    u_out[:, 0, :] += u_in[:, -1, :]

    u_out[:, :, 1:] += u_in[:, :, :-1]
    u_out[:, :, :-1] += u_in[:, :, 1:]
    u_out[:, :, -1] += u_in[:, :, 0]
    u_out[:, :, 0] += u_in[:, :, -1]

def conj_grad_mpi(b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int) -> int:
    limit = n * n // 2
    Ap = np.empty_like(b)
    if x0 is None:
        x0 = np.zeros_like(b)
        r = np.copy(b)
    else:
        r = np.empty_like(b)
        laplace_filter_mpi(x0, r, n)
        r -= b
    x[:] = x0

    p = r / 6.0

    r_dot_v = -np.sum(r * r) / 6.0
    r_dot_v = mpi.all_reduce(r_dot_v)

    res = -1
    iter_ = 1
    while iter_ < limit:
        laplace_filter_mpi(p, Ap, n)

        pAp = np.sum(p * Ap)
        pAp = mpi.all_reduce(pAp)
        alpha = r_dot_v / pAp
        x += alpha * p
        r += alpha * Ap
        rn_dot_rn = np.sum(r * r)
        rn_dot_rn = mpi.all_reduce(rn_dot_rn)
        if rn_dot_rn ** 0.5 < tol:
            res = iter_
            break

        rn_dot_vn = - rn_dot_rn / 6.0
        beta = rn_dot_vn / r_dot_v
        r_dot_v = rn_dot_vn
        p = r / 6.0 + beta * p
        iter_ += 1

    return res

class LCGGrid_MPI(BaseGrid):
    def init_grids(self):
        """Initialize the grids."""
        div = self.N // mpi.size
        rem = self.N % mpi.size
        self.N_loc = div + (1 if mpi.rank < rem else 0)
        self.N_loc_start = div * mpi.rank + min(mpi.rank, rem)
        self.N_loc_end = self.N_loc_start + self.N_loc
        print('*'*100)
        print(f"Rank {mpi.rank}: {self.N_loc_start} - {self.N_loc_start + self.N_loc}")
        print('*'*100)

        self.shape = (self.N_loc, self.N, self.N)
        self.tmp_top = np.empty((self.N, self.N), dtype=float)
        self.tmp_bot = np.empty((self.N, self.N), dtype=float)
        self.y = np.zeros(self.shape, dtype=float)  # right-hand side of the preconditioned Poisson equation
        self.q = np.zeros(self.shape, dtype=float)  # charge vector - q for every grid point
        self.tmp = np.empty(self.shape, dtype=float)  # temporary array for the Poisson equation
        # 2-step phi for the Verlet algorithm
        self._phi = deque([np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)], maxlen=2)
        mpi.barrier()

    def initialize_field(self):
        """Initialize the field."""
        # c_api.c_conj_grad(- 4 * np.pi * self.q / self.h, self.phi, self.tmp, self.tol, self.N)
        # (b: np.ndarray, x0: np.ndarray, x: np.ndarray, tol: float, n: int, mpi: MPIBase)
        conj_grad_mpi(- 4 * np.pi * self.q / self.h, self.phi, self.tmp, self.tol, self.N)
        self._phi.append(np.copy(self.tmp))

    @BaseGrid.timeit
    def update_field(self):
        """Update the field."""
        phi = 2 * self.phi - self.phi_prev
        # c_api.c_laplace(phi, self.tmp, self.N)
        laplace_filter_mpi(phi, self.tmp, self.N)
        sigma_p = 4 * np.pi * self.q / self.h + self.tmp

        y_new = np.empty_like(self.y)
        # self.n_iters = c_api.c_conj_grad(sigma_p, self.y, y_new, self.tol, self.N)
        self.n_iters = conj_grad_mpi(sigma_p, self.y, y_new, self.tol, self.N)
        if self.n_iters == -1:
            raise ValueError(f'Conjugate gradient did not converge {self.n_iters} iterations')

        phi -= y_new
        self.y = y_new
        self._phi.append(phi)
        # print(self.phi.shape)

    def gather(self, vec):
        if mpi.rank == 0:
            self.app = np.empty((self.N, self.N, self.N), dtype=float)
            self.app[:self.N_loc] = vec
            for i in range(1, mpi.size):
                N_start = mpi.comm.recv(source=i)
                N_loc = mpi.comm.recv(source=i)
                mpi.comm.Recv(self.app[N_start:N_start+N_loc], source=i)
        else:
            self.app = None
            mpi.comm.send(self.N_loc_start, dest=0)
            mpi.comm.send(self.N_loc, dest=0)
            mpi.comm.Send(vec, dest=0)
        self.gathered = self.app

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]
