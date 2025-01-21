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
        # 2-step phi for the Verlet algorithm
        self._phi = [np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)]

    def initialize_field(self):
        """Initialize the field."""
        self.phi_prev[:] = self.phi
        capi.conj_grad(- 4 * np.pi * self.q / self.h, self.phi, self.phi, self.tol, self.N)

    def update_field(self):
        """Update the field."""
        self.n_iters = capi.verlet_poisson(self.tol, self.h, self.phi, self.phi_prev, self.q, self.y, self.N)

        if self.n_iters == -1:
            self.logger.error(f'Conjugate gradient did not converge!!!')
            exit()

    def gather(self, vec):
        return collect_grid_buffer(vec, self.N)

    @property
    def phi_prev(self):
        return self._phi[0]

    @property
    def phi(self):
        return self._phi[1]
