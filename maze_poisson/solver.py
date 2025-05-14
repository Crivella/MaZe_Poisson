"""Implement a base solver Class for maze_poisson."""
import atexit
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from .c_api import capi
from .clocks import Clock
from .constants import a0, conv_mass
from .myio import OutputFiles, ProgressBar
from .myio.input import GridSetting, MDVariables, OutputSettings
from .myio.loggers import Logger
from .myio.output import save_json

np.random.seed(42)

method_grid_map: Dict[str, int] = {
    # 'LCG': 0,
    # 'FFT': 1,
}

integrator_map: Dict[str, int] = {
    # 'OVRVO': 0,
    # 'VERLET': 1,
}

potential_map: Dict[str, int] = {
    # 'TF': 0,
    # 'LD': 1,
}

ca_scheme_map: Dict[str, int] = {
    # 'CIC': 0,
    # 'SPL_QUADR': 1,
    # 'SPL_CUBIC': 2,
}

precond_map: Dict[str, int] = {
    # 'NONE': 0,  # Jacobi implicit
    # 'JACOBI': 1,  # Jacobi explicit
    # 'MG': 2,  # Multigrid
    # 'SSOR': 3,  # Symmetric Successive Over-Relaxation
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

        self.n_iters = 0

        self.ofiles = OutputFiles(self.outset)
        self.out_stride = outset.stride
        self.out_flushstride = outset.flushstride * outset.stride

        # Logging
        out_log = os.path.join(outset.path, 'log.txt')
        self.add_file_handler(out_log, level=logging.DEBUG)
        if self.outset.debug:
            self.set_log_level(logging.DEBUG)
            self.logger.debug("Set verbosity to DEBUG")

        self.save_input()

    @Clock('initialize')
    def initialize(self):
        """Initialize the solver."""
        capi.solver_initialize()

        self.initialize_str_maps()

        self.initialize_grid()
        self.initialize_particles()
        self.initialize_integrator()
        self.initialize_md()

        atexit.register(self.finalize)

    def finalize(self):
        """Finalize the solver."""
        capi.solver_finalize()
        Clock.report_all()

    def initialize_str_maps(self):
        """Initialize the string maps."""
        n = capi.get_grid_type_num()
        for i in range(n):
            ptr = capi.get_grid_type_str(i)
            method_grid_map[ptr.decode('utf-8').upper()] = i

        n = capi.get_potential_type_num()
        for i in range(n):
            ptr = capi.get_potential_type_str(i)
            potential_map[ptr.decode('utf-8').upper()] = i

        n = capi.get_ca_scheme_type_num()
        for i in range(n):
            ptr = capi.get_ca_scheme_type_str(i)
            ca_scheme_map[ptr.decode('utf-8').upper()] = i

        n = capi.get_integrator_type_num()
        for i in range(n):
            ptr = capi.get_integrator_type_str(i)
            integrator_map[ptr.decode('utf-8').upper()] = i

        n = capi.get_precond_type_num()
        for i in range(n):
            ptr = capi.get_precond_type_str(i)
            precond_map[ptr.decode('utf-8').upper()] = i

    def initialize_grid(self):
        """Initialize the grid."""
        self.logger.info(f"Initializing grid with method: {self.mdv.method}")
        method = self.mdv.method.upper()
        if not method in method_grid_map:
            raise ValueError(f"Method {method} not recognized.")
        precond = self.gset.precond.upper()
        if not precond in precond_map:
            raise ValueError(f"Preconditioner {precond} not recognized.")

        grid_id = method_grid_map[method]
        precond_id = precond_map[precond]
        capi.solver_initialize_grid(self.N, self.L, self.h, self.mdv.tol, grid_id, precond_id)

    def initialize_particles(self):
        """Initialize the particles."""
        self.logger.info(f"Initializing particles with potential: {self.mdv.potential}")
        potential = self.mdv.potential.upper()
        if not potential in potential_map:
            raise ValueError(f"Potential {potential} not recognized.")
        pot_id = potential_map[potential]

        cas_str = self.gset.charge_assignment.upper()
        if not cas_str in ca_scheme_map:
            raise ValueError(f"Charge assignment scheme {cas_str} not recognized.")
        ca_scheme_id = ca_scheme_map[cas_str]

        if self.gset.input_file and self.gset.restart_file:
            self.logger.warning("Both input and restart files provided. Using restart file.")
        start_file = self.gset.restart_file or self.gset.input_file
        kBT = self.mdv.kBT

        df = pd.read_csv(start_file)
        charges = np.ascontiguousarray(df['charge'].values, dtype=np.int64)
        mass = np.ascontiguousarray(df['mass'].values * conv_mass, dtype=np.float64)
        pos = np.ascontiguousarray(df[['x', 'y', 'z']].values / a0, dtype=np.float64)

        if not pos.size:
            raise ValueError(f"Empty or incorrect input file `{start_file}`.")
        if len(pos) != self.N_p:
            raise ValueError(f"Number of particles in file ({len(pos)}) does not match N_p ({self.N_p}).")

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
            self.N, self.L, self.h, self.N_p,
            pot_id, ca_scheme_id,
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
            self.logger.debug("---- Udpating charges")
            self.update_charges()
            # if self.mdv.preconditioning:
            self.logger.debug("---- Initializing field")
            self.initialize_field()
            self.logger.debug("---- Computing forces")
            self.compute_forces()

            # STEP 1 Verlet
            self.logger.debug("---- Integrator part 1")
            self.integrator_part1()
            self.logger.debug("---- Updating charges")
            self.update_charges()
            # if self.mdv.preconditioning:
            self.logger.debug("---- Initializing field")
            self.initialize_field()
            self.logger.debug("---- Computing forces")
            self.compute_forces()
            self.logger.debug("---- Integrator part 2")
            self.integrator_part2()
        elif ffile:
            df = pd.read_csv(ffile)
            phi = np.ascontiguousarray(df['phi'].values).reshape((self.N, self.N, self.N))
            capi.solver_set_field(phi)
            phi = np.ascontiguousarray(df['phi_prev'].values).reshape((self.N, self.N, self.N))
            capi.solver_set_field_prev(phi)

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
        res = capi.solver_update_field()
        if res == -1:
            self.logger.warning("Warning: CG did not converge.")
            # raise ValueError("Error CG did not converge.")
        return res

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
            self.n_iters = self.update_field()
            self.t_iters = Clock.get_clock('field').last_call
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

        temp = capi.get_temperature()
        self.logger.info(f"Temperature: {temp:.2f} K")

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

    def save_input(self):
        """Save the input parameters to a file."""
        filename = os.path.join(self.outset.path, 'input.json')
        self.logger.info(f"Saving input parameters to file {filename}")
        dct = {}
        dct['grid_setting'] = self.gset.to_dict()
        dct['md_variables'] = self.mdv.to_dict()
        dct['output_settings'] = self.outset.to_dict()

        save_json(filename, dct)

    def init_info(self):
        """Print information about the initialization."""
        from .constants import density
        self.logger.info(f'Running a MD simulation with:')
        self.logger.info(f'  N_p = {self.N_p}, N_steps = {self.mdv.N_steps}, tol = {self.mdv.tol}')
        self.logger.info(f'  N = {self.N}, L [a.u.] = {self.L}, h [a.u.] = {self.h}')
        self.logger.info(f'  density = {density} g/cm^3')
        self.logger.info(f'  Solver: {self.mdv.method},  Preconditioner: {self.gset.precond}')
        self.logger.info(f'  Charge assignment scheme: {self.gset.charge_assignment}')
        # self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: {self.mdv.integrator}, dt = {self.mdv.dt}')
        self.logger.info(f'  Potential: {self.mdv.potential}')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.mdv.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')
