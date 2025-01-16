from collections import deque

import numpy as np

from .. import c_api
from .base_grid import BaseGrid


class LCGGrid(BaseGrid):
    mpi_enabled = True

    def init_grids(self):
        """Initialize the grids."""
        self.shape = (self.N_loc, self.N, self.N)

        self.y = np.zeros(self.shape, dtype=float)  # right-hand side of the preconditioned Poisson equation
        self.q = np.zeros(self.shape, dtype=float)  # charge vector - q for every grid point
        self.tmp = np.empty(self.shape, dtype=float)  # temporary array for the Poisson equation
        # 2-step phi for the Verlet algorithm
        self._phi = deque([np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)], maxlen=2)

    def initialize_field(self):
        """Initialize the field."""
        c_api.c_conj_grad(- 4 * np.pi * self.q / self.h, self.phi, self.tmp, self.tol, self.N)
        self._phi.append(np.copy(self.tmp))

    def update_field(self):
        """Update the field."""
        phi = 2 * self.phi - self.phi_prev
        c_api.c_laplace(phi, self.tmp, self.N)
        sigma_p = 4 * np.pi * self.q / self.h + self.tmp

        self.n_iters = c_api.c_conj_grad(sigma_p, self.y, self.tmp, self.tol, self.N)
        if self.n_iters == -1:
            self.logger.error(f'Conjugate gradient did not converge!!!')
            exit()

        phi -= self.tmp
        self.y = self.tmp
        self._phi.append(phi)

    def gather(self, vec):
        if not self.mpi:
            self.gathered = vec
        else:
            if self.mpi.rank == 0:
                app = np.empty((self.N, self.N, self.N), dtype=float)
                app[:self.N_loc] = vec
                for i in range(1, self.mpi.size):
                    N_start = self.mpi.comm.recv(source=i)
                    N_loc = self.mpi.comm.recv(source=i)
                    self.mpi.comm.Recv(app[N_start:N_start+N_loc], source=i)
            else:
                app = None
                self.mpi.comm.send(self.N_loc_start, dest=0)
                self.mpi.comm.send(self.N_loc, dest=0)
                self.mpi.comm.Send(vec, dest=0)
            self.gathered = app

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]
