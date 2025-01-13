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
from ..loggers import Logger, setup_logger
from ..outputs import OutputFiles
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

class BaseSolver(Logger, ABC):
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

        self.ofiles = OutputFiles(self.outset)
        self.out_stride = outset.stride
        self.out_flushstride = outset.flushstride * outset.stride
        if self.outset.debug:
            for handler in self.logger.handlers:
                handler.setLevel(0)
            self.logger.debug("Set verbosity to DEBUG")

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
        start_file = self.gset.restart_file or self.gset.input_file
        self.particles = Particles.from_file(start_file, self.gset, self.mdv.potential, self.mdv.kBT)

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

        self.grid.field_j = int(self.particles.pos[0,1] / self.h)
        self.grid.field_k = int(self.particles.pos[0,2] / self.h)

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
            self.ofiles.output(i, self.grid, self.particles)

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
