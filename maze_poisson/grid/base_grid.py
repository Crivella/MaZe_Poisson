from abc import ABC, abstractmethod

import numpy as np

from .. import c_api
from ..c_api import mympi as mpi
from ..myio import Logger
from ..particles import Particles


class BaseGrid(Logger, ABC):
    """Base class for all grid classes."""
    mpi_enabled = False

    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mpi.MPIBase.mpi and not self.mpi_enabled:
            self.logger.error(f"MPI not implemented for {self.__class__.__name__}")
            exit()

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

        self.N_loc = mpi.c_init_mpi_grid(N)
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

    def init_grids(self):
        """Initialize the grids."""
        if mpi.MPIBase.mpi:
            self.init_grids_mpi()
        else:
            self.init_grids_single()

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

        q_tot = c_api.c_update_charges(self.N, N_p, self.h, pos, neighbors, charges, self.q)

        return q_tot

    def cleanup(self):
        """Cleanup the grid."""
        pass
