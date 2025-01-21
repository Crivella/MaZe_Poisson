"""Implement a base solver Class for maze_poisson."""
import atexit
from typing import Dict, Tuple

import numpy as np

from ..c_api import capi
from ..clocks import Clock
from ..grid import BaseGrid, FFTGrid, LCGGrid
from ..integrators import BaseIntegrator, OVRVOIntegrator, VerletIntegrator
from ..myio import Logger, OutputFiles, ProgressBar
from ..myio.input import GridSetting, MDVariables, OutputSettings
from ..particles import Particles

np.random.seed(42)

method_grid_map: Dict[str, Tuple[BaseGrid, BaseGrid]] = {
    'LCG': LCGGrid,
    'FFT': FFTGrid,
}

integrator_map: Dict[str, BaseIntegrator] = {
    'OVRVO': OVRVOIntegrator,
    'VERLET': VerletIntegrator,
}

class SolverMD(Logger):
    """Base class for all solver classes."""

    def __init__(self, gset: GridSetting, mdv: MDVariables, outset: OutputSettings):
        capi.initialize()
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
            for handler in self.logger.handlers:
                handler.setLevel(0)
            self.logger.debug("Set verbosity to DEBUG")

        self.q_tot = 0
        self.thermostat = self.mdv.thermostat

        atexit.register(self.finalize)

    @Clock('initialize')
    def initialize(self):
        """Initialize the solver."""
        self.initialize_grid()
        self.initialize_particles()
        self.initialize_integrator()
        self.initialize_md()

    def finalize(self):
        """Finalize the solver."""
        self.particles.cleanup()
        self.grid.cleanup()
        Clock.report_all()

    def initialize_grid(self):
        """Initialize the grid."""
        method = self.mdv.method.upper()
        if not method in method_grid_map:
            raise ValueError(f"Method {method} not recognized.")

        grid_cls = method_grid_map[method]
        restart_field = self.gset.restart_field_file
        self.grid = grid_cls(self.L, self.h, self.N, self.mdv.tol, field_file=restart_field)

    def initialize_particles(self):
        """Initialize the particles."""
        start_file = self.gset.restart_file or self.gset.input_file
        self.particles = Particles.from_file(start_file, self.gset, self.mdv.potential, self.mdv.kBT)

    def initialize_integrator(self):
        """Initialize the MD integrator."""
        name = self.mdv.integrator.upper()
        if not name in integrator_map:
            raise ValueError(f"Integrator {name} not recognized.")
        cls = integrator_map[name]
        self.integrator = cls(self.mdv.dt, self.mdv.kBT, self.grid.L)

        if self.thermostat:
            tstat_args = cls.get_thermostat_variables(self.mdv)
            self.integrator.init_thermostat(*tstat_args)

    def initialize_md(self):
        """Initialize the first 2 steps for the MD and forces."""
        self.q_tot = np.sum(self.particles.charges)
        self.logger.info(f"Total charge: {self.q_tot}")

        ffile = self.gset.restart_field_file
        if ffile is None:
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
        else:
            self.logger.info(f"Initialization step skipped due to field loaded from file.")

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
            self.compute_forces_field()
        if self.mdv.not_elec:
            self.compute_forces_notelec()
        self.particles.forces = self.particles.forces_elec + self.particles.forces_notelec

    @Clock('forces_field')
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""
        self.particles.compute_forces_field(self.grid)

    @Clock('forces_notelec')
    def compute_forces_notelec(self):
        """Compute the forces on the particles due to non-electric interactions."""
        self.grid.potential_notelec = self.particles.ComputeTFForces()

    @Clock('file_output')
    def md_loop_output(self, i: int, force: bool = False):
        """Output the data for the MD loop."""
        self.ofiles.output(i, self.grid, self.particles, force)

    @Clock('charges')
    def update_charges(self):
        """Update the charge grid based on the particles position with function g to spread them on the grid."""
        self.particles.get_nearest_neighbors()
        q_tot = self.grid.update_charges(self.particles)

        if np.abs(self.q_tot - q_tot) > 1e-6:
            self.logger.error('Error: change initial position, charge is not preserved: q_tot = %.6f, %.6f', q_tot, self.q_tot)
            exit()

        self.q_tot = q_tot

    def md_loop_iter(self):
        """Run one iteration of the molecular dynamics loop."""
        self.integrator.part1(self.particles)
        if self.mdv.elec:
            self.update_charges()
            self.update_field()
        self.compute_forces()
        self.integrator.part2(self.particles)

    def md_check_thermostat(self, iter: int):
        if self.thermostat:
            temperature = self.particles.get_temperature()
            if np.abs(temperature - self.mdv.T) <= 100:
                self.logger.info(f'End thermostating iteration {iter}')
                self.thermostat = False
                self.integrator.stop_thermostat()

    def md_loop(self):
        """Run the molecular dynamics loop."""
        if self.mdv.init_steps:
            self.logger.info("Running MD loop initialization steps...")
            for i in ProgressBar(self.mdv.init_steps):
                self.md_loop_iter()

        self.logger.info("Running MD loop...")
        for i in ProgressBar(self.mdv.N_steps):
            self.md_loop_iter()
            self.md_check_thermostat(i)
            self.md_loop_output(i)

    def run(self):
        """Run the MD calculation."""
        self.initialize()
        self.init_info()
        self.md_loop()
        self.md_loop_output(self.mdv.N_steps, force=True)

    def init_info(self):
        """Print information about the initialization."""
        self.logger.info(f'Running a MD simulation with:')
        self.logger.info(f'  N_p = {self.N_p}, N_steps = {self.mdv.N_steps}, tol = {self.mdv.tol}')
        self.logger.info(f'  N = {self.N}, L [a.u.] = {self.L}, h [a.u.] = {self.h}')
        self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: {self.mdv.integrator}, Method: {self.mdv.method} dt = {self.mdv.dt}')
        self.logger.info(f'  Potential: {self.mdv.potential}')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')
