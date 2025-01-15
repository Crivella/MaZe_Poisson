"""Implement a base solver Class for maze_poisson."""

import functools
import time

import numpy as np

from ..clocks import Clock
from ..grid import BaseGrid, FFTGrid, LCGGrid, LCGGrid_MPI
from ..input import GridSetting, MDVariables, OutputSettings
from ..integrators import BaseIntegrator, OVRVOIntegrator, VerletIntegrator
from ..loggers import Logger
from ..mpi import MPIBase
from ..outputs import OutputFiles
from ..particles import Particles

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

mpi = MPIBase()

np.random.seed(42)

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

method_grid_map: dict[str, tuple[BaseGrid, BaseGrid]] = {
    'LCG': (LCGGrid, LCGGrid_MPI),
    'FFT': (FFTGrid, None),
}

integrator_map: dict[str, BaseIntegrator] = {
    'OVRVO': OVRVOIntegrator,
    'VERLET': VerletIntegrator,
}

class SolverMD(Logger):
    """Base class for all solver classes."""

    def __init__(self, gset: GridSetting, mdv: MDVariables, outset: OutputSettings):
        super().__init__()

        self.gset = gset
        self.mdv = mdv
        self.outset = outset

        self.L = gset.L
        self.h = gset.h
        self.N = gset.N
        self.N_p = gset.N_p

        self.integrator: BaseIntegrator = None
        self.grid: BaseGrid = None
        self.particles: Particles = None

        self.ofiles = OutputFiles(self.outset)
        self.out_stride = outset.stride
        self.out_flushstride = outset.flushstride * outset.stride
        if self.outset.debug:
            # print(self.logger.handlers())
            for handler in self.logger.handlers:
                handler.setLevel(0)
            self.logger.debug("Set verbosity to DEBUG")

        self.q_tot = 0
        self.thermostat = self.mdv.thermostat

    @Clock('initialize')
    def initialize(self):
        """Initialize the solver."""
        self.initialize_grid()
        self.initialize_particles()
        self.initialize_integrator()

    def finalize(self):
        """Finalize the solver."""
        self.particles.cleanup()
        self.grid.cleanup()

    def initialize_grid(self):
        """Initialize the grid."""
        method = self.mdv.method.upper()
        if not method in method_grid_map:
            raise ValueError(f"Method {method} not recognized.")
        tpl = method_grid_map[method]
        if mpi and mpi.size > 1:
            grid_cls = tpl[1]
        else:
            grid_cls = tpl[0]
        if grid_cls is None:
            raise ValueError(f"Method {method} not supported with MPI.")

        self.grid = grid_cls(self.L, self.h, self.N, self.mdv.tol)

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
        if not name in integrator_map:
            raise ValueError(f"Integrator {name} not recognized.")
        cls = integrator_map[name]
        self.integrator = cls(self.mdv.dt, self.mdv.kBT, self.grid.L)

        if self.thermostat:
            tstat_args = cls.get_thermostat_variables(self.mdv)
            self.integrator.init_thermostat(*tstat_args)

    def initialize_md(self):
        """Initialize the molecular dynamics."""
        self.q_tot = np.sum(self.particles.charges)
        self.logger.info(f"Total charge: {self.q_tot}")
        # STEP 0 Verlet
        self.update_charges()
        if self.mdv.preconditioning:
            self.initialize_field()

        # STEP 1 Verlet
        self.integrator.part1(self.particles)
        self.update_charges()
        if self.mdv.preconditioning:
            self.initialize_field()
        self.compute_forces()
        self.integrator.part2(self.particles)

        if self.mdv.rescale:
            self.particles.rescale_velocities()

        self.grid.field_j = int(self.particles.pos[0,1] / self.h)
        self.grid.field_k = int(self.particles.pos[0,2] / self.h)

    @Clock('field')
    def initialize_field(self):
        """Initialize the field."""
        self.grid.initialize_field()

    @Clock('field')
    def update_field(self):
        """Update the field."""
        self.grid.update_field()

    @Clock('forces')
    def compute_forces(self):
        """Compute the forces on the particles."""
        if self.mdv.elec:
            start = time.time()
            self.compute_forces_field()
            end = time.time()
            self.logger.debug(f'Forces ELEC: {end - start}')
        if self.mdv.not_elec:
            start = time.time()
            self.compute_forces_notelec()
            end = time.time()
            self.logger.debug(f'Forces NOEL: {end - start}')
        self.particles.forces = self.particles.forces_elec + self.particles.forces_notelec

    @Clock('forces_field')
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""
        self.q_tot = self.particles.compute_forces_field(self.grid.phi, self.grid.q)

    @Clock('forces_notelec')
    def compute_forces_notelec(self):
        """Compute the forces on the particles due to non-electric interactions."""
        self.grid.potential_notelec = self.particles.ComputeTFForces()

    @Clock('file_output')
    def md_loop_output(self, i: int):
        """Output the data for the MD loop."""
        self.ofiles.output(i, self.grid, self.particles)

    @Clock('charges')
    def update_charges(self):
        """Update the charge grid based on the particles position with function g to spread them on the grid."""
        # raise NotImplementedError("Need to move this in the Grid class")
        self.particles.get_nearest_neighbors()
        q_tot = self.grid.update_charges(self.particles)

        if np.abs(self.q_tot - q_tot) > 1e-6:
            self.logger.error('Error: change initial position, charge is not preserved: q_tot = %.4f', q_tot)
            exit() # exits runinning otherwise it hangs the code

        # print(q_tot)
        self.q_tot = q_tot

    def md_loop_iter(self):
        """Run one iteration of the molecular dynamics loop."""
        self.integrator.part1(self.particles)
        if self.mdv.elec:
            self.update_charges()
            self.update_field()
        self.compute_forces()
        self.integrator.part2(self.particles)

    def md_check_thermostat(self, i: int):
        if self.thermostat:
            temperature = self.particles.get_temperature()
            if np.abs(temperature - self.mdv.T) <= 100:
                self.logger.info(f'End thermostating iteration {i}')
                self.thermostat = False
                self.integrator.stop_thermostat()

    def md_loop(self):
        """Run the molecular dynamics loop."""
        if self.mdv.init_steps:
            self.logger.info("Running MD loop initialization steps...")
            if mpi and mpi.rank > 1:
                loop = range(self.mdv.init_steps)
            else:
                loop = tqdm(range(self.mdv.init_steps))
            for i in loop:
                self.md_loop_iter()

        self.logger.info("Running MD loop...")
        if mpi and mpi.rank > 1:
            loop = range(self.mdv.N_steps)
        else:
            loop = tqdm(range(self.mdv.N_steps))
        for i in loop:
            self.md_loop_iter()
            self.md_check_thermostat(i)
            self.md_loop_output(i)

    @Clock('total')
    def run(self):
        """Run the solver."""
        self.initialize()
        self.init_info()
        self.initialize_md()
        self.md_loop()
        self.finalize()
        Clock.report_all()

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
