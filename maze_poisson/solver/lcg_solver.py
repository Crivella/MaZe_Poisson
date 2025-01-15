import numpy as np

from .. import c_api
from ..grid import LCGGrid, LCGGrid_MPI
from ..mpi import MPIBase
from . import BaseSolver

mpi = MPIBase()


class LCGSolver(BaseSolver):
    def initialize_grid(self):
        """Initialize the grid."""
        if mpi and mpi.size > 1:
            self.grid = LCGGrid_MPI(self.gset.L, self.gset.h, self.gset.N, self.mdv.tol)
        else:
            self.grid = LCGGrid(self.gset.L, self.gset.h, self.gset.N, self.mdv.tol)

    
    def compute_forces_field(self) -> np.ndarray:
        """Compute the forces from the field."""
        h = self.grid.h  # h in angstrom
        N = self.grid.N
        N_loc = self.grid.N_loc
        N_loc_start = self.grid.N_loc_start
        N_p = self.particles.N_p
        phi = self.grid.phi
        q = self.grid.q

        neighbors = self.particles.neighbors

        out = self.particles.forces_elec
        self.q_tot = c_api.c_compute_force_fd(N, N_p, h, phi, q, neighbors, out, N_loc, N_loc_start)

        return out
