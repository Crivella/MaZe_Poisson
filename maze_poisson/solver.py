"""Implement a base solver Class for maze_poisson."""
import atexit
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

from . import constants as cst
from .c_api import capi
from .clocks import Clock
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
    # 'SC': 2,
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
    # 'BLOCKJACOBI': 4,  # Symmetric Successive Over-Relaxation
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
        self.N_typs = gset.N_typs

        self.thermostat = mdv.thermostat

        self.n_iters = 0

        self.energy_nonpolar = 0.0

        if self.outset.print_restart:
            outset.restart_step = outset.restart_step or mdv.N_steps

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
        for _map, fname_num, fname_data in [
            (method_grid_map, 'get_grid_type_num', 'get_grid_type_str'),
            (potential_map, 'get_potential_type_num', 'get_potential_type_str'),
            (ca_scheme_map, 'get_ca_scheme_type_num', 'get_ca_scheme_type_str'),
            (integrator_map, 'get_integrator_type_num', 'get_integrator_type_str'),
            (precond_map, 'get_precond_type_num', 'get_precond_type_str'),
        ]:
            n = getattr(capi, fname_num)()
            for i in range(n):
                ptr = getattr(capi, fname_data)(i)
                _map[ptr.decode('utf-8').upper()] = i

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
        capi.solver_initialize_grid(
            self.N, self.L, self.h, self.mdv.tol, self.gset.eps_s,
            grid_id, precond_id
        )

        if self.mdv.poisson_boltzmann:
            eps_s = self.gset.eps_s
            kbar2 = (
                8 * np.pi * cst.NA * cst.EC**2 * self.gset.I * 1e3
            ) / (
                eps_s * cst.eps0 * cst.kB_si * self.mdv.T
            ) * cst.BR ** 2 * self.h ** 2
            self.logger.info("Initializing grid for Poisson-Boltzmann.")
            capi.solver_initialize_grid_pois_boltz(
                self.gset.w, kbar2
            )

    def get_tosi_fumi_params(self, particles) -> np.ndarray:
        """Get the Tosi-Fumi parameters for the particles."""
        if self.mdv.potential_params_file is None:
            raise ValueError("Potential parameters file must be provided for TF potential.")
        tf_params = pd.read_csv(self.mdv.potential_params_file)
        expected = self.N_typs * (self.N_typs + 1) // 2
        if len(tf_params) != expected:
            raise ValueError(
                f"Potential parameters file must have {expected} unique pairs of types."
            )
        try:
            particles.loc[tf_params['type1']]
            particles.loc[tf_params['type2']]
        except KeyError as e:
            raise ValueError(f"Particle type not found in particles file: {e}")

        tf_params.set_index(['type1', 'type2'], inplace=True)
        if len(tf_params) != len(tf_params.index.unique()):
            raise ValueError(
                "Potential parameters file must have unique pairs of types (type1, type2)."
            )
        tf_params_array = np.empty((self.N_typs, self.N_typs, 5), dtype=np.float64) * np.nan
        for t1, t2 in tf_params.index:
            t1_idx = particles.loc[t1, 'enum']
            t2_idx = particles.loc[t2, 'enum']
            if not np.isnan(tf_params_array[t1_idx, t2_idx, 0]):
                raise ValueError(f"Duplicate potential parameters for types {t1_idx} and {t2_idx}.")
            if not np.isnan(tf_params_array[t2_idx, t1_idx, 0]):
                raise ValueError(f"Potential parameters for types {t1_idx} and {t2_idx} must be symmetric.")
            tf_params_array[t1_idx, t2_idx] = tf_params.loc[(t1, t2), ['A', 'B', 'C', 'D', 'sigma']].values
            tf_params_array[t2_idx, t1_idx] = tf_params_array[t1_idx, t2_idx]
        if np.any(np.isnan(tf_params_array)):
            raise ValueError("Potential parameters for some particle types are missing.")
        tf_params_array[:, :, 0] *= cst.kJmol_to_hartree  # A  kJ/mol -> Hartree
        tf_params_array[:, :, 1] *= cst.a0  # B  1/ang -> a.u.
        tf_params_array[:, :, 2] *= cst.kJmol_to_hartree  / cst.a0**6  # C  kJ/mol*ang^6 -> Hartree*a.u.^6
        tf_params_array[:, :, 3] *= cst.kJmol_to_hartree / cst.a0**8  # D  kJ/mol*ang^8 -> Hartree*a.u.^8
        tf_params_array[:, :, 4] /= cst.a0  # sigma  ang -> a.u.
        tf_params_array = np.ascontiguousarray(tf_params_array.flatten(), dtype=np.float64)

        return tf_params_array

    def get_sc_params(self) -> np.ndarray:
        """Get the shared parameters for the SC potential."""
        self.logger.info("Using SC potential with shared parameters (nu, d, B).")
        if self.mdv.potential_params_file is None:
            raise ValueError("Potential parameters file must be provided for SC potential.")

        sc_params = pd.read_csv(self.mdv.potential_params_file)

        required_columns = {'nu', 'd'}
        if not required_columns.issubset(sc_params.columns):
            raise ValueError(f"Potential parameters file must contain columns: {required_columns}")

        if len(sc_params) != 1:
            raise ValueError("Potential parameters file for SC must contain exactly one row.")

        nu = sc_params['nu'].iloc[0]
        d = sc_params['d'].iloc[0]
        if nu <= 0 or d <= 0:
            raise ValueError("Parameters 'nu' and 'd' must be strictly positive.")

        Am = 1.74
        Nc = 6
        d_au = d / cst.a0  # convert d from Angstrom to Bohr
        B_au = Am / (Nc * nu * d_au)  # au
        
        # Salva come vettore (es. per uso diretto nei kernel)
        sc_params_array = np.array([nu, d_au, B_au], dtype=np.float64)

        return sc_params_array

    def initialize_particles(self):
        """Initialize the particles."""
        self.logger.info(f"Reading particle definitions from file: {self.gset.particles_file}")
        particles = pd.read_csv(self.gset.particles_file)
        if len(particles) != self.gset.N_typs:
            raise ValueError(
                f"Number of particle types in file ({len(particles)}) does not match N_typs ({self.gset.N_typs})."
            )
        if len(set(particles['type'])) != self.gset.N_typs:
            raise ValueError("Particle types in file must be unique.")
        particles.set_index('type', inplace=True)
        particles['enum'] = range(len(particles))

        self.logger.info(f"Initializing particles with potential: {self.mdv.potential}")
        potential = self.mdv.potential.upper()
        if not potential in potential_map:
            # print(potential_map, potential)
            raise ValueError(f"Potential {potential} not recognized among {','.join(potential_map.keys())}.")
        pot_id = potential_map[potential]

        cas_str = self.gset.cas.upper()
        if not cas_str in ca_scheme_map:
            raise ValueError(f"Charge assignment scheme {cas_str} not recognized.")
        ca_scheme_id = ca_scheme_map[cas_str]

        start_file = self.gset.input_file
        kBT = self.mdv.kBT

        df = pd.read_csv(start_file)
        types = np.ascontiguousarray(particles.loc[df['type'], 'enum'].values, dtype=np.int32)
        pos = np.ascontiguousarray(df[['x', 'y', 'z']].values / cst.a0, dtype=np.float64)
        charges = np.ascontiguousarray(particles.loc[df['type'], 'charge'].values, dtype=np.float64)
        mass = np.ascontiguousarray(particles.loc[df['type'], 'mass'].values, dtype=np.float64) * cst.conv_mass
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

        if potential == 'TF':
            pot_params = self.get_tosi_fumi_params(particles)
        elif potential == 'SC':
            pot_params = self.get_sc_params()

        # print(f"Using potential parameters: {pot_params}")

        capi.solver_initialize_particles(
            self.N, self.N_typs, self.L, self.h, self.N_p,
            pot_id, ca_scheme_id,
            types, pos, vel, mass, charges,
            pot_params
        )

        if self.mdv.poisson_boltzmann:
            if 'radius' not in particles.columns:
                raise ValueError("Probe radius must be provided in the input file for Poisson-Boltzmann.")
            radius = np.ascontiguousarray(particles.loc[df['type'], 'radius'].values, dtype=np.float64)
            radius = radius / cst.a0 + self.mdv.probe_radius
            self.logger.info("Initializing particles for Poisson-Boltzmann.")
            capi.solver_initialize_particles_pois_boltz(
                self.mdv.gamma_np_au, self.mdv.beta_np, radius
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
        self.logger.info("Initializing MD (first 2 steps)...")
        ffile = self.gset.restart_field_file
        if ffile is None or self.mdv.invert_time:
            # STEP 0 Verlet
            # self.logger.debug("Running first step of MD loop (Verlet)...")
            self.update_charges()
            # self.logger.debug("Updating k^2 grid for Poisson-Boltzmann...")
            self.update_eps_k2()
            # self.logger.debug("Initializing field...")
            self.initialize_field()
            # self.logger.debug("Computing forces...")
            self.compute_forces()

            # STEP 1 Verlet
            # self.logger.debug("Running second step of MD loop (Verlet)...")
            self.integrator_part1()
            # self.logger.debug("Updating charges...")
            self.update_charges()
            # self.logger.debug("Updating k^2 grid for Poisson-Boltzmann...")
            self.update_eps_k2()
            # self.logger.debug("Updating field...")
            self.initialize_field()
            # self.logger.debug("Computing forces...")
            self.compute_forces()
            # self.logger.debug("Running second part of integrator...")
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

    @Clock('update_eps_k2')
    def update_eps_k2(self):
        """Update the k^2 grid for Poisson-Boltzmann."""
        if self.mdv.poisson_boltzmann:
            capi.solver_update_eps_k2()

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
        if self.mdv.poisson_boltzmann:
            self.compute_forces_pb()
        # self.logger.debug("Computing total forces...")
        capi.solver_compute_forces_tot()

    @Clock('forces_field')
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""
        # self.logger.debug("Computing forces due to electric field...")
        capi.solver_compute_forces_elec()

    @Clock('forces_notelec')
    def compute_forces_notelec(self):
        """Compute the forces on the particles due to non-electric interactions."""
        # self.logger.debug("Computing forces due to non-electric interactions...")
        self.potential_notelec = capi.solver_compute_forces_noel()

    @Clock('forces_PBoltz')
    def compute_forces_pb(self):
        """Compute the forces on the particles due to Poisson-Boltzmann interactions."""
        # self.logger.debug("Computing forces due to Poisson-Boltzmann interactions...")
        self.energy_nonpolar = capi.solver_compute_forces_pb()

    @Clock('file_output')
    def md_loop_output(self, i: int, force: bool = False):
        """Output the data for the MD loop."""
        self.ofiles.output(i, self, force)

    @Clock('charges')
    def update_charges(self):
        """Update the charge grid based on the particles position with function g to spread them on the grid."""
        if capi.solver_update_charges() != 0:
            self.logger.error('Error: change initial position, charge is not preserved.')
            sys.exit(1)

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
            self.update_eps_k2()
            self.n_iters = self.update_field()
            self.t_iters = Clock.get_clock('field').last_call
        self.compute_forces()
        self.integrator_part2()

    def md_loop(self):
        """Run the molecular dynamics loop."""
        if self.mdv.init_steps:
            self.logger.info("Running MD loop initialization steps...")
            for i in ProgressBar(self.mdv.init_steps):
                self.md_loop_iter()
        
        temp = capi.get_temperature()
        self.logger.info(f"Temperature: {temp:.2f} K")

        self.logger.info("Running MD loop...")
        if self.thermostat:
            self.logger.info("Thermostat ON in production run")

        for i in ProgressBar(self.mdv.N_steps):
            self.md_loop_iter()
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
        self.logger.info(f'  N = {self.N}, L [A] = {self.L / cst.a0}, h [A] = {self.h / cst.a0}')
        self.logger.info(f'  density = {density} g/cm^3')
        self.logger.info(f'  Solvent dielectric constant: {self.gset.eps_s}')
        self.logger.info(f'  Solver: {self.mdv.method},  Preconditioner: {self.gset.precond}')
        self.logger.info(f'  Charge assignment scheme: {self.gset.cas}')
        # self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: {self.mdv.integrator}, dt = {self.mdv.dt}')
        self.logger.info(f'  Potential: {self.mdv.potential}')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.mdv.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')
        if self.outset.print_restart:
            self.logger.info(f'  Restart step: {self.outset.restart_step}')
        if self.mdv.poisson_boltzmann:
            w_ang = self.gset.w_ang
            h_ang = self.gset.h * cst.a0
            self.logger.info('  ***************************************')
            self.logger.info('  Poisson-Boltzmann: ENABLED')
            self.logger.info(f'  Transition region width: {w_ang} A')
            if 2 * w_ang < h_ang:
                self.logger.warning(
                    f'  Warning: transition region width ({w_ang:.2f} A) is smaller than grid spacing ({h_ang:.2f} A)'
                )
            self.logger.info(f'  Ionic strength: {self.gset.I} M')
            self.logger.info(f'  Gamma NP: {self.mdv.gamma_np}')
            self.logger.info(f'  Beta NP: {self.mdv.beta_np}')


