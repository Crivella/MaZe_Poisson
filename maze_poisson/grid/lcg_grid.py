from collections import deque

import numpy as np

from ..c_api import capi
from ..c_api.mympi import collect_grid_buffer
from .base_grid import BaseGrid


class LCGGrid(BaseGrid):
    def init_grids(self):
        """Initialize the grids."""
        self.shape = (self.n_loc, self.N, self.N)

        self.y = np.zeros(self.shape, dtype=float)  # right-hand side of the preconditioned Poisson equation
        self.q = np.zeros(self.shape, dtype=float)  # charge vector - q for every grid point
        self.tmp = np.empty(self.shape, dtype=float)  # temporary array for the Poisson equation
        # 2-step phi for the Verlet algorithm
        self._phi = deque([np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)], maxlen=2)

    def initialize_field(self):
        """Initialize the field."""
        capi.conj_grad(- 4 * np.pi * self.q / self.h, self.phi, self.tmp, self.tol, self.N)
        self._phi.append(np.copy(self.tmp))

    def update_field(self):
        """Update the field."""
        phi = 2 * self.phi - self.phi_prev
        capi.laplace_filter(phi, self.tmp, self.N)
        sigma_p = 4 * np.pi * self.q / self.h + self.tmp

        self.n_iters = capi.conj_grad(sigma_p, self.y, self.tmp, self.tol, self.N)
        if self.n_iters == -1:
            self.logger.error(f'Conjugate gradient did not converge!!!')
            exit()

        phi -= self.tmp
        self.y = self.tmp
        self._phi.append(phi)

    def gather(self, vec):
        self.gathered = collect_grid_buffer(vec, self.N)

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]
