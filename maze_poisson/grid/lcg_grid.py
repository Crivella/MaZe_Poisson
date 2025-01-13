from collections import deque

import numpy as np

from .. import c_api
# from ..verlet import MatrixVectorProduct, PrecondLinearConjGradPoisson
from .base_grid import BaseGrid


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

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]
