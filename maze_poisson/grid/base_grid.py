import time
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from ..mpi import MPIBase
from ..particles import Particles, g

mpi = MPIBase()

class BaseGrid(ABC):
    """Base class for all grid classes."""

    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7):
        self.mpi = MPIBase()

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

        self.N_loc = self.N
        self.N_loc_start = 0

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

    # def update_charges(self, pos: np.ndarray, neighbors: np.ndarray, charges: np.ndarray) -> float:
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
        if mpi and mpi.size > 1:
            for i,j,k,upd in zip(*indices, updates):
                i -= self.N_loc_start
                if 0 <= i < self.N_loc:
                    q[i, j, k] += upd
                    
        else:
            q[indices] += updates
  
        q_tot = np.sum(updates)
        if mpi and mpi.size > 1:
            q_tot = mpi.all_reduce(q_tot)

        return q_tot

    @staticmethod
    def timeit(func):
        """Decorator to measure the time of the function."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            func(self, *args, **kwargs)
            end = time.time()
            self.time = end - start
        return wrapper

    def cleanup(self):
        """Cleanup the grid."""
        pass
