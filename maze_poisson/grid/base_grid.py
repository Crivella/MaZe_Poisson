from abc import ABC, abstractmethod

import numpy as np

from ..c_api import capi
from ..myio import Logger
from ..particles import Particles


class BaseGrid(Logger, ABC):
    """Base class for all grid classes."""
    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N
        self.h = h
        self.L = L
        self.tol = tol

        self.potential_notelec = 0

        self.time = 0
        self.n_iters = 0

        self.X = np.arange(0, L, h)
        self.field_j = 0
        self.field_k = 0

        self.n_loc = capi.init_mpi_grid(self.N)

        self.init_grids()

    @abstractmethod
    def init_grids(self):
        """Initialize the grids."""

    @abstractmethod
    def initialize_field(self):
        """Initialize the field."""

    @abstractmethod
    def update_field(self):
        """Update the field."""

    @property
    @abstractmethod 
    def phi(self):
        """Should return the field in REAL space."""

    def update_charges(self, particles: Particles) -> float:
        """Update the charges on the grid.
        
        Args:
            particles (Particles): Particles object.

        Returns:
            float: Total charge contribution.
        """
        N_p = particles.N_p
        pos = np.ascontiguousarray(particles.pos)
        neighbors = particles.neighbors
        charges = particles.charges

        q_tot = capi.update_charges(self.N, N_p, self.h, pos, neighbors, charges, self.q)

        return q_tot

    def cleanup(self):
        """Cleanup the grid."""
        pass
