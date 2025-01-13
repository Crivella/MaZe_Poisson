import time
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from ..mpi import MPIBase


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
