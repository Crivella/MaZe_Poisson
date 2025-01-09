from abc import ABC, abstractmethod

from ..particles import Particles


class BaseGrid(ABC):
    """Base class for all grid classes."""

    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7):
        self.N = N
        self.h = h
        self.L = L
        self.tol = tol

        self.potential_notelec = 0

        self.init_grids()

    @abstractmethod
    def init_grids(self):
        """Initialize the grids."""

    @abstractmethod
    def initialize_field(self):
        """Initialize the field."""

    @abstractmethod
    def update_field(self) -> int:
        """Update the field.

        Returns:
            int: Number of iterations to convergence.
        """

    @property
    @abstractmethod
    def phi(self):
        """Should return the field in REAL space."""
