from functools import wraps

import numpy as np

from .. import c_api
from ..grid import FFTGrid
from . import BaseSolver


class FFTSolver(BaseSolver):
    @wraps(BaseSolver.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_init('fftw', self.init_fftw)
        self.register_finalize('fftw', self.cleanup_fftw)

    def init_fftw(self):
        """Initialize the FFTW."""
        c_api.init_fftw_omp()
        c_api.init_rfft(self.N)

    def cleanup_fftw(self):
        """Cleanup the FFTW."""
        c_api.cleanup_fftw()

    def initialize_grid(self):
        """Initialize the grid."""
        self.grid = FFTGrid(self.gset.L, self.gset.h, self.gset.N, self.mdv.tol)

    def compute_forces_field(self) -> np.ndarray:
        """Compute the forces from the field."""
        h = self.grid.h  # h in angstrom
        N = self.grid.N
        N_p = self.particles.N_p
        phi = self.grid.phi
        q = self.grid.q

        neighbors = self.particles.neighbors

        out = self.particles.forces_elec
        self.q_tot = c_api.c_compute_force_fd(N, N_p, h, phi, q, neighbors, out)

        return out
