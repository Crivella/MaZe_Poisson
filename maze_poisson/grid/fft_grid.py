import numpy as np

from .. import c_api
from .base_grid import BaseGrid


class FFTGrid(BaseGrid):
    def init_grids(self):
        """Initialize the grids."""
        self.shape = (self.N,)*3
        self.q = np.zeros(self.shape, dtype=float)
        self.phi_r = np.zeros(self.shape, dtype=np.float64)

        # Grids for FFT
        freqs = np.fft.fftfreq(self.N, d=self.h) * 2 * np.pi
        freqs_r = np.fft.rfftfreq(self.N, d=self.h) * 2 * np.pi
        gx, gy, gz = np.meshgrid(freqs, freqs, freqs_r, indexing='ij')
        g2 = gx**2 + gy**2 + gz**2
        g2[0, 0, 0] = 1  # to avoid division by zero

        self.ig2 = (4 * np.pi / self.h**3) / g2
        del g2, gx, gy, gz

        self.init_fftw()

    def init_fftw(self):
        """Initialize the FFTW."""
        c_api.init_fftw_omp()
        c_api.init_rfft(self.N)

    def cleanup_fftw(self):
        """Cleanup the FFTW."""
        c_api.cleanup_fftw()

    def calculate_phi(self):
        """Calculate the field."""
        c_api.rfft_solve(self.N, self.q, self.ig2, self.phi_r)

    def initialize_field(self):
        """Initialize the field."""
        self.calculate_phi()

    def update_field(self):
        """Update the field."""
        self.calculate_phi()

    @property
    def phi(self):
        return self.phi_r

    @property
    def phi_prev(self):
        return self.phi_r

    def cleanup(self):
        self.cleanup_fftw()
        return super().cleanup()
