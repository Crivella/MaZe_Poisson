from collections import deque

import numpy as np

from ..verlet import MatrixVectorProduct, PrecondLinearConjGradPoisson
from .base_grid import BaseGrid


class LCGGrid(BaseGrid):
    def init_grids(self):
        """Initialize the grids."""
        self.shape = (self.N,)*3
        self.y = np.zeros(self.shape, dtype=float)  # right-hand side of the Poisson equation
        self.q = np.zeros(self.shape, dtype=float)  # charge vector - q for every grid point
        self._phi = deque([np.zeros(self.shape, dtype=float), np.zeros(self.shape, dtype=float)], maxlen=2)

    def initialize_field(self):
        """Initialize the field."""
        new, _ = PrecondLinearConjGradPoisson(
            - 4 * np.pi * self.q / self.h,
            x0 = self.phi,
            tol = self.tol,
            )
        self._phi.append(new)

    def update_field(self):
        """Update the field."""
        phi = 2 * self.phi - self.phi_prev
        matrixmult = MatrixVectorProduct(phi)
        sigma_p = 4 * np.pi * self.q / self.h + matrixmult

        # print("sigma_p", sigma_p.shape)
        y_new, iter_conv = PrecondLinearConjGradPoisson(sigma_p, x0=self.y, tol=self.tol)
        # print("done", iter_conv)
        phi -= y_new
        self.y = y_new

        self._phi.append(phi)

        return iter_conv

    @property
    def phi(self):
        return self._phi[-1]

    @property
    def phi_prev(self):
        return self._phi[0]
