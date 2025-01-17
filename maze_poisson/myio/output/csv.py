from abc import abstractmethod

import pandas as pd

from ...constants import Ha_to_eV, a0, conv_mass
from ...grid.base_grid import BaseGrid
from ...particles import Particles
from .base_out import BaseOutputFile, OutputFiles, ensure_enabled


class CSVOutputFile(BaseOutputFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_headers()

    @ensure_enabled
    def init_headers(self):
        pd.DataFrame(columns=self.headers).to_csv(self.buffer, index=False)

    @ensure_enabled
    def write_data(self, iter: int, grid: BaseGrid, particles: Particles, mode: str = 'a'):
        header = False
        if mode == 'w':
            self.buffer.truncate(0)
            self.buffer.seek(0)
            header = True
        df = self.get_data(iter, grid, particles)
        df.to_csv(self.buffer, columns=self.headers, header=header, index=False, mode=mode)

    @property
    @abstractmethod
    def headers(self):
        pass

    @abstractmethod
    def get_data(self, iter: int, grid: BaseGrid, particles: Particles) -> pd.DataFrame:
        pass

class EnergyCSVOutputFile(CSVOutputFile):
    name = 'energy'
    headers = ['iter', 'K', 'V_notelec']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'K': [particles.get_kinetic_energy()],
            'V_notelec': [grid.potential_notelec]
        })

class ForcesCSVOutputFile(CSVOutputFile):
    name = 'forces'
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'Fx': [particles.forces[:, 0].sum()],
            'Fy': [particles.forces[:, 1].sum()],
            'Fz': [particles.forces[:, 2].sum()]
        })

class TemperatureCSVOutputFile(CSVOutputFile):
    name = 'temperature'
    headers = ['iter', 'T']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'T': [particles.get_temperature()]
        })

class SolutesCSVOutputFile(CSVOutputFile):
    name = 'solute'
    headers = ['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        df = pd.DataFrame(particles.pos, columns=['x', 'y', 'z'])
        df['vx'] = particles.vel[:, 0]
        df['vy'] = particles.vel[:, 1]
        df['vz'] = particles.vel[:, 2]
        df['fx_elec'] = particles.forces_elec[:, 0]
        df['fy_elec'] = particles.forces_elec[:, 1]
        df['fz_elec'] = particles.forces_elec[:, 2]
        df['charge'] = particles.charges
        df['iter'] = iter
        df['particle'] = range(particles.N_p)
        return df

class PerformanceCSVOutputFile(CSVOutputFile):
    name =  'performance'
    headers = ['iter', 'time', 'n_iters']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'time': [grid.time],
            'n_iters': [grid.n_iters]
        })

class FieldCSVOutputFile(CSVOutputFile):
    name = 'field'
    headers = ['iter', 'x', 'MaZe']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        X = grid.X
        j = grid.field_j
        k = grid.field_k
        return pd.DataFrame({
            'iter': [iter]*len(X),
            'x': X,
            'MaZe': grid.phi[:, j, k] * Ha_to_eV
        })

class RestartCSVOutputFile(CSVOutputFile):
    name = 'restart'
    headers = ['charge', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        df = pd.DataFrame(particles.pos * a0, columns=['x', 'y', 'z'])
        df[['vx', 'vy', 'vz']] = particles.vel
        df['charge'] = particles.charges
        df['mass'] = particles.masses / conv_mass
        return df

OutputFiles.register_format(
    'csv',
    {
        'field': FieldCSVOutputFile,
        'performance': PerformanceCSVOutputFile,
        'energy': EnergyCSVOutputFile,
        'temperature': TemperatureCSVOutputFile,
        'solute': SolutesCSVOutputFile,
        'tot_force': ForcesCSVOutputFile,
        'restart': RestartCSVOutputFile
    }
)
