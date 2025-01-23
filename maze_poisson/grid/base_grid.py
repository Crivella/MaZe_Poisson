from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..c_api import capi
from ..myio import Logger
from ..particles import Particles


class BaseGrid(Logger, ABC):
    """Base class for all grid classes."""
    def __init__(self, L: float, h: float, N: int, tol: float = 1e-7, field_file: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N
        self.h = h
        self.L = L
        self.tol = tol

        self.potential_notelec = 0

        self.q = None

        self.time = 0
        self.n_iters = 0

        self.n_loc = capi.init_mpi_grid(self.N)

        self.init_grids()

        # if field_file is not None:
        #     raise NotImplementedError("Field file loading not implemented yet.")

        if field_file is not None:
            self.logger.info(f"Loading field from file: {field_file}")
            df_field = pd.read_csv(field_file)
            self.phi[:] = df_field['phi'].values.reshape(self.shape)
            if self.phi_prev is not self.phi:
                self.phi_prev[:] = df_field['phi_prev'].values.reshape(self.shape)

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
    def phi_prev(self):
        """Should return the field in REAL space at the previous iteration if needed by the solver."""
        return self.phi

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
        pos = particles.pos
        neighbors = particles.neighbors
        charges = particles.charges

        q_tot = capi.update_charges(self.N, N_p, self.h, pos, neighbors, charges, self.q)

        return q_tot

    def gather(self, vec):
        return vec
    

    def cleanup(self):
        """Cleanup the grid."""
        pass
