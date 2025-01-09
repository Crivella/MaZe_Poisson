"""Implement a base solver Class for maze_poisson."""

import functools
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..constants import Ha_to_eV, a0, conv_mass, kB
from ..grid.base_grid import BaseGrid
from ..input import GridSetting, MDVariables, OutputSettings
from ..integrators import BaseIntegrator, OVRVOIntegrator, VerletIntegrator
from ..loggers import setup_logger
from ..output_md import generate_output_files
from ..particles import Particles, g

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


def time_report(func):
    """Decorator to report the time taken by a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        dt = end - start
        if dt > 1:
            print(f"Time taken by {func.__name__}: {dt:.2f} s")
        return result
    return wrapper

class BaseSolver(ABC):
    """Base class for all solver classes."""

    def __init__(self, gset: GridSetting, mdv: MDVariables, outset: OutputSettings):
        self.logger = setup_logger(self.__class__.__name__)

        self.gset = gset
        self.mdv = mdv
        self.outset = outset

        self.L = gset.L
        self.h = gset.h
        self.N = gset.N
        self.N_p = gset.N_p

        self.integrator: BaseIntegrator = None
        self.grid: BaseGrid = None

        self.ofiles = generate_output_files(outset)
        self.out_stride = outset.stride
        self.out_flushstride = outset.flushstride * outset.stride

        self.q_tot = 0
        self.thermostat = self.mdv.thermostat

        self.to_init = []
        self.to_finalize = []

        self.register_init('grid', self.initialize_grid)
        self.register_init('particles', self.initialize_particles)
        self.register_init('integrator', self.initialize_integrator)

    @abstractmethod
    def initialize_grid(self):
        """Initialize the grid."""

    @abstractmethod
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""

    def register_init(self, name: str, func, args = None, kwargs = None):
        """Register an initialization function."""
        self.to_init.append((name, func, args or [], kwargs or {}))

    def register_finalize(self, name: str, func, args = None, kwargs = None):
        """Register a finalize function."""
        self.to_finalize.append((name, func, args or [], kwargs or {}))

    @time_report
    def initialize(self):
        """Initialize the solver."""
        for name, func, args, kwargs in self.to_init:
            self.logger.info(f"Initializing {name}...")
            func(*args, **kwargs)

    @time_report
    def finalize(self):
        """Finalize the solver."""
        for name, func, args, kwargs in self.to_finalize:
            self.logger.info(f"Finalizing {name}...")
            func(*args, **kwargs)

    def initialize_particles(self):
        """Initialize the particles."""
        start_file = self.gset.restart_file if self.outset.restart else self.gset.input_file
        df = pd.read_csv(start_file)
        charges = df['charge']
        mass = df['mass'] * conv_mass
        pos = df[['x', 'y', 'z']].values / a0
        self.particles = Particles(
            self.gset,
            self.mdv.potential,
            charges=charges,
            masses=mass,
            positions=pos
            )
        if self.outset.restart:
            self.logger.info(f"Restarting from file: {start_file}")
            self.particles.vel = df[['vx', 'vy', 'vz']].values
        else:
            self.particles.vel = np.random.normal(
                loc = 0.0,
                scale = np.sqrt(self.mdv.kBT / self.particles.masses[:, np.newaxis]),
                size=(self.N_p, 3)
            )

    def initialize_integrator(self):
        """Initialize the MD integrator.

        Raises:
            ValueError: If the integrator is not recognized.
        """
        name = self.mdv.integrator.upper()
        if name == 'OVRVO':
            self.integrator = OVRVOIntegrator(self.mdv.dt, self.mdv.kBT, self.grid.L)
            tstat_args = [self.mdv.gamma]
        elif name == 'VERLET':
            self.integrator = VerletIntegrator(self.mdv.dt, self.mdv.kBT, self.grid.L)
            tstat_args = [self.mdv.T]
        else:
            raise ValueError(f"Integrator {name} not recognized.") 

        if self.thermostat:
            self.integrator.init_thermostat(*tstat_args)

    def initialize_md(self):
        """Initialize the molecular dynamics."""
        self.q_tot = np.sum(self.particles.charges)
        self.logger.info(f"Total charge: {self.q_tot}")
        # STEP 0 Verlet
        self.particles.get_nearest_neighbors()
        self.update_charges()
        if self.mdv.preconditioning:
            self.grid.initialize_field()
        self.compute_forces()

        # STEP 1 Verlet
        self.integrator.part1(self.particles)
        self.particles.get_nearest_neighbors()
        self.update_charges()

        if self.mdv.preconditioning:
            self.grid.initialize_field()
        self.compute_forces()
        self.integrator.part2(self.particles)

        if self.mdv.rescale:
            self.particles.rescale_velocities()

    def compute_forces(self):
        """Compute the forces on the particles."""
        if self.mdv.elec:
            self.compute_forces_field()
        if self.mdv.not_elec:
            self.compute_forces_notelec()
        self.particles.forces = self.particles.forces_elec + self.particles.forces_notelec

    def compute_forces_notelec(self):
        """Compute the forces on the particles due to non-electric interactions."""
        self.grid.potential_notelec = self.particles.ComputeTFForces()

    def md_loop_output(self, i):
        """Output the results of the molecular dynamics loop."""
        if i % self.out_stride != 0:
            return
        # print(self.particles.pos[0])
        self.ofiles.energy.write_data(i, self.grid, self.particles)
        self.ofiles.tot_force.write_data(i, self.grid, self.particles)
        self.ofiles.temperature.write_data(i, self.grid, self.particles)
        #     raise NotImplementedError("Reimplement outputs in a more modular way")
        #     # self.ofiles.output_field(i, self.grid)
        #     # self.ofiles.output_iters(i, self.grid)
        #     # self.ofiles.output_energy(i, self.particles)
        #     # self.ofiles.output_temperature(i, self.particles)
        #     # self.ofiles.output_solute(i, self.particles)
        #     # self.ofiles.output_tot_force(i, self.particles)
        if self.out_flushstride:
            if i % self.out_flushstride == 0:
                self.ofiles.flush()

    def md_loop_iter(self):
        """Run one iteration of the molecular dynamics loop."""
        self.integrator.part1(self.particles)
        if self.mdv.elec:
            self.particles.get_nearest_neighbors()
            self.update_charges()
            self.grid.update_field()
        self.compute_forces()
        self.integrator.part2(self.particles)

    def md_check_thermostat(self):
        if self.thermostat:
            temperature = self.particles.get_temperature()
            if np.abs(temperature - self.mdv.T) <= 100:
                self.logger.info('End thermostating')
                self.thermostat = False
                self.integrator.stop_thermostat()

    @time_report
    def md_loop(self):
        """Run the molecular dynamics loop."""
        if self.mdv.init_steps:
            self.logger.info("Running MD loop initialization steps...")
            for i in tqdm(range(self.mdv.init_steps)):
                self.md_loop_iter()

        self.logger.info("Running MD loop...")
        for i in tqdm(range(self.mdv.N_steps)):
            self.md_loop_iter()
            self.md_check_thermostat()
            self.md_loop_output(i)

    def run(self):
        """Run the solver."""
        self.initialize()
        self.init_info()
        self.initialize_md()
        self.md_loop()
        self.finalize()

    def init_info(self):
        """Print information about the initialization."""
        self.logger.info(f'Running a MD simulation with:')
        self.logger.info(f'  N_p = {self.N_p}, N_steps = {self.mdv.N_steps}, tol = {self.mdv.tol}')
        self.logger.info(f'  N = {self.N}, L [a.u.] = {self.L}, h [a.u.] = {self.h}')
        self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: {self.mdv.integrator}, Method: {self.mdv.method}')
        self.logger.info(f'  Potential: {self.mdv.potential}')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')

    def update_charges(self):
        """Update the charge grid based on the particles position with function g to spread them on the grid."""
        # raise NotImplementedError("Need to move this in the Grid class")
        L = self.L
        h = self.h
        q = self.grid.q
        q.fill(0)

        # for m in range(len(self.particles.charges)):
        #     for i, j, k in self.particles.neighbors[m, :, :]:
        #         diff = self.particles.pos[m] - np.array([i,j, k]) * h
        #         self.q[i, j, k] += self.particles.charges[m] * g(diff[0], L, h) * g(diff[1], L, h) * g(diff[2], L, h)

        # Same as above using broadcasting
        diff = self.particles.pos[:, np.newaxis, :] - self.particles.neighbors * h
        
        # if Python 3.11 or newer uncomment below and comment lines 217-219
        #self.q[*self.particles.neighbors.reshape(-1, 3).T] += (self.particles.charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        
        # Version that works for Python 3.8.15
        indices = tuple(self.particles.neighbors.reshape(-1, 3).T)
        updates = (self.particles.charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        q[indices] += updates
  
        # q_tot_expected = np.sum(self.particles.charges)
        q_tot = np.sum(updates)

        # self.phi_q_updated = False

        if np.abs(self.q_tot - q_tot) > 1e-6:
            self.logger.error('Error: change initial position, charge is not preserved: q_tot = %.4f', q_tot)
            exit() # exits runinning otherwise it hangs the code

        # print(q_tot)
        self.q_tot = q_tot
