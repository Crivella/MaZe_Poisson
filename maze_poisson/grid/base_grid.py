import time
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from ..mpi import MPIBase
from ..myio import Logger
from ..particles import Particles, g


class BaseGrid(Logger, ABC):
    """Base class for all grid classes."""
    mpi_enabled = False

    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpi = MPIBase()
        if self.mpi and not self.mpi_enabled:
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

        div = self.N // self.mpi.size
        rem = self.N % self.mpi.size
        self.N_loc = div + (1 if self.mpi.rank < rem else 0)
        self.N_loc_start = div * self.mpi.rank + min(self.mpi.rank, rem)

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
        if self.mpi:
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
        pos = particles.pos
        neighbors = particles.neighbors
        charges = particles.charges

        q = self.q
        q.fill(0)
        diff = pos[:, np.newaxis, :] - neighbors * self.h

        indices = tuple(neighbors.reshape(-1, 3).T)

        updates = (charges[:, np.newaxis] * np.prod(g(diff, self.L, self.h), axis=2)).flatten()
        if self.mpi:
            for i,j,k,upd in zip(*indices, updates):
                i -= self.N_loc_start
                if 0 <= i < self.N_loc:
                    q[i, j, k] += upd
                    
        else:
            q[indices] += updates
  
        q_tot = np.sum(updates)
        if self.mpi:
            q_tot = self.mpi.all_reduce(q_tot)

        return q_tot

    def cleanup(self):
        """Cleanup the grid."""
        pass
