"""Implement a base solver Class for maze_poisson."""
import atexit
from typing import Dict

import numpy as np
import pandas as pd

from .c_api import capi
from .clocks import Clock
from .constants import a0, conv_mass
from .myio import Logger, OutputFiles, ProgressBar
from .myio.input import GridSetting, MDVariables, OutputSettings

np.random.seed(42)

method_grid_map: Dict[str, int] = {
    'LCG': 0,
    'FFT': 1,
}

integrator_map: Dict[str, int] = {
    'OVRVO': 0,
    'VERLET': 1,
}

potential_map: Dict[str, int] = {
    'TF': 0,
    'LD': 1,
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

        self.ofiles = OutputFiles(self.outset)
        self.out_stride = outset.stride
        self.out_flushstride = outset.flushstride * outset.stride
        if self.outset.debug:
            for handler in self.logger.handlers:
                handler.setLevel(0)
            self.logger.debug("Set verbosity to DEBUG")

    @Clock('initialize')
    def initialize(self):
        """Initialize the solver."""
        capi.solver_initialize(self.N)

        self.initialize_grid()
        self.initialize_particles()
        self.initialize_integrator()
        self.initialize_md()

        atexit.register(self.finalize)

    def finalize(self):
        """Finalize the solver."""
        capi.solver_finalize()
        Clock.report_all()

    def initialize_grid(self):
        """Initialize the grid."""
        self.logger.info(f"Initializing grid with method: {self.mdv.method}")
        method = self.mdv.method.upper()
        if not method in method_grid_map:
            raise ValueError(f"Method {method} not recognized.")

        grid_id = method_grid_map[method]
        capi.solver_initialize_grid(self.N, self.L, self.h, self.mdv.tol, grid_id)

    def initialize_particles(self):
        """Initialize the particles."""
        self.logger.info(f"Initializing particles with potential: {self.mdv.potential}")
        potential = self.mdv.potential.upper()
        if not potential in potential_map:
            raise ValueError(f"Potential {potential} not recognized.")
        pot_id = potential_map[potential]

        start_file = self.gset.restart_file or self.gset.input_file
        kBT = self.mdv.kBT

        df = pd.read_csv(start_file)
        charges = np.ascontiguousarray(df['charge'].values, dtype=np.int64)
        mass = np.ascontiguousarray(df['mass'].values * conv_mass)
        pos = np.ascontiguousarray(df[['x', 'y', 'z']].values / a0)

        self.logger.info(f"Loaded starting positions from file: {start_file}")
        if 'vx' in df.columns:
            self.logger.info("Loading starting velocities from file.")
            vel = np.ascontiguousarray(df[['vx', 'vy', 'vz']].values)
        else:
            if kBT is None:
                raise ValueError("kBT must be provided to generate random velocities.")
            self.logger.info("Generating random velocities.")
            vel = np.random.normal(
                loc = 0.0,
                scale = np.sqrt(kBT / mass[:, np.newaxis]),
                size=(len(df), 3)
            )
        capi.solver_initialize_particles(
            self.N, self.L, self.h, self.N_p, pot_id,
            pos, vel, mass, charges
        )

    def initialize_integrator(self):
        """Initialize the MD integrator."""
        self.logger.info(f"Initializing integrator: {self.mdv.integrator}")
        name = self.mdv.integrator.upper()
        if not name in integrator_map:
            raise ValueError(f"Integrator {name} not recognized.")
        itg_id = integrator_map[name]

        enabled = 1 if self.mdv.thermostat else 0
        capi.solver_initialize_integrator(
            self.N_p, self.mdv.dt, self.mdv.T, self.mdv.gamma, itg_id, enabled
        )

    def initialize_md(self):
        """Initialize the first 2 steps for the MD and forces."""
        # if capi.solver_initialize_md(self.mdv.preconditioning, self.mdv.rescale) != 0:
        #     self.logger.error("Error initializing MD.")
        #     exit()
        ffile = self.gset.restart_field_file
        if ffile is None or self.mdv.invert_time:
            # STEP 0 Verlet
            self.update_charges()
            if self.mdv.preconditioning:
                self.initialize_field()
            self.compute_forces()

            # STEP 1 Verlet
            self.integrator_part1()
            self.update_charges()
            if self.mdv.preconditioning:
                self.initialize_field()
            self.compute_forces()
            self.integrator_part2()
        else:
            self.logger.info(f"Initialization step skipped due to field loaded from file.")

        if self.mdv.rescale:
            capi.solver_rescale_velocities()

    @Clock('field')
    def initialize_field(self):
        """Initialize the field."""
        capi.solver_init_field()

    @Clock('field')
    def update_field(self):
        """Update the field."""
        capi.solver_update_field()

    @Clock('forces')
    def compute_forces(self):
        """Compute the forces on the particles."""
        if self.mdv.elec:
            self.compute_forces_field()
        if self.mdv.not_elec:
            self.compute_forces_notelec()
        capi.solver_compute_forces_tot()

    @Clock('forces_field')
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""
        capi.solver_compute_forces_elec()

    @Clock('forces_notelec')
    def compute_forces_notelec(self):
        """Compute the forces on the particles due to non-electric interactions."""
        self.potential_notelec = capi.solver_compute_forces_noel()

    @Clock('file_output')
    def md_loop_output(self, i: int, force: bool = False):
        """Output the data for the MD loop."""
        self.ofiles.output(i, self, force)

    @Clock('charges')
    def update_charges(self):
        """Update the charge grid based on the particles position with function g to spread them on the grid."""
        if capi.solver_update_charges() != 0:
            self.logger.error('Error: change initial position, charge is not preserved.')
            exit()

    @Clock('integrator')
    def integrator_part1(self):
        """Update the position and velocity of the particles."""
        capi.integrator_part_1()

    @Clock('integrator')
    def integrator_part2(self):
        """Update the velocity of the particles."""
        capi.integrator_part_2()

    def md_loop_iter(self):
        """Run one iteration of the molecular dynamics loop."""
        self.integrator_part1()
        if self.mdv.elec:
            self.update_charges()
            self.update_field()
        self.compute_forces()
        self.integrator_part2()

    def md_check_thermostat(self, iter: int):
        if capi.solver_check_thermostat():
            self.logger.info(f'End thermostating iteration {iter}')

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
        self.init_info()
        self.initialize()
        self.md_loop()
        self.md_loop_output(self.mdv.N_steps, force=True)

    def init_info(self):
        """Print information about the initialization."""
        from .constants import density
        self.logger.info(f'Running a MD simulation with:')
        self.logger.info(f'  N_p = {self.N_p}, N_steps = {self.mdv.N_steps}, tol = {self.mdv.tol}')
        self.logger.info(f'  N = {self.N}, L [a.u.] = {self.L}, h [a.u.] = {self.h}')
        self.logger.info(f'  density = {density} g/cm^3')
        self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: {self.mdv.integrator},  Method: {self.mdv.method},  dt = {self.mdv.dt}')
        self.logger.info(f'  Potential: {self.mdv.potential}')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.mdv.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')
