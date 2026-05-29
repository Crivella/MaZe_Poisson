"""Implement a base solver Class for maze_poisson."""
import atexit
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

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
    # 'MULTIGRID': 2,
    # 'MAZE-LCG': 3,
    # 'MAZE-MULTIGRID': 4,
}

integrator_map: Dict[str, int] = {
    # 'OVRVO': 0,
    # 'VERLET': 1,
}

potential_map: Dict[str, int] = {
    # 'TF': 0,
    # 'LJ': 1,
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

elec_corr_map: Dict[str, int] = {
    # 'SPREAD': 0,
    # 'SR': 1,
}

smoothing_map: Dict[str, int] = {
    # 'NONE': 0,
    # 'GAUSS': 1,
    # 'DIFFUSION': 2,
}

pneigh_method_map: Dict[str, int] = {
    # 'SPHERE': 0,
    # 'CELL_LIST': 1,
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
        self.t_charges = 0.0
        self.t_smoothing = 0.0
        self.t_field = 0.0
        self.t_elec_total = 0.0

        self.potential_notelec = 0.0
        self.energy_nonpolar = 0.0
        self.potential_short_range = 0.0
        self.energy_intra = 0.0
        self.energy_elec = 0.0
        self.energy_corr = 0.0
        self.potential_notelec = 0.0

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

        self.types_str_to_num = {}
        self.types_num_to_str = {}

    @Clock('initialize')
    def initialize(self):
        """Initialize the solver."""
        capi.solver_initialize()

        self.mpi_rank = capi.get_rank()
        self.mpi_size = capi.get_size()

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
            (elec_corr_map, 'get_water_electrostatic_type_num', 'get_water_electrostatic_type_str'),
            (ca_scheme_map, 'get_ca_scheme_type_num', 'get_ca_scheme_type_str'),
            (integrator_map, 'get_integrator_type_num', 'get_integrator_type_str'),
            (precond_map, 'get_precond_type_num', 'get_precond_type_str'),
            (smoothing_map, 'get_smoothing_type_num', 'get_smoothing_type_str'),
            (pneigh_method_map, 'get_particle_neighbor_type_num', 'get_particle_neighbor_type_str'),
        ]:
            n = getattr(capi, fname_num)()
            for i in range(n):
                ptr = getattr(capi, fname_data)(i)
                _map[ptr.decode('utf-8').upper()] = i

    def initialize_grid_smoothing(self):
        """Initialize the smoothing."""
        smoothing = self.smoothing_type = self.gset.charge_smoothing

        self.logger.info(f"Initializing smoothing with method: '{smoothing}'")
        method = smoothing.upper()
        if not method in smoothing_map:
            raise ValueError(f"Smoothing method {method} not recognized.")

        method_id = smoothing_map[method]
        self.smoothing_rcut = self.gset.smoothing_rcut / cst.a0
        if self.gset.smoothing_sigma is None:
            self.smoothing_sigma = self.smoothing_rcut / 3.0
        else:
            self.smoothing_sigma = self.gset.smoothing_sigma / cst.a0

        capi.solver_initialize_grid_smoothing(method_id, self.smoothing_rcut, self.smoothing_sigma)
    
    def initialize_grid_pb(self):
        """Initialize the grid for Poisson-Boltzmann."""
        if not self.mdv.poisson_boltzmann:
            return
        self.logger.info("Initializing grid for Poisson-Boltzmann.")
        eps_s = self.gset.eps_s
        # eps_int = self.gset.eps_int
        kbar2 = (
            8 * np.pi * cst.NA * cst.EC**2 * self.gset.I * 1e3
        ) / (
            eps_s * cst.eps0 * cst.kB_si * self.mdv.T
        ) * cst.BR ** 2 * self.h ** 2

        if self.mdv.nonpolar_forces:
            nonpolar_enabled = 1
        else:  
            nonpolar_enabled = 0   

        capi.solver_initialize_grid_pois_boltz(
            self.gset.w, kbar2, nonpolar_enabled
        )

    def initialize_grid(self):
        """Initialize the grid."""
        self.logger.info(f"Initializing grid with method: '{self.mdv.method}'")
        method = self.mdv.method.upper()
        if not method in method_grid_map:
            raise ValueError(f"Method {method} not recognized.")
        precond = self.gset.precond.upper()
        if not precond in precond_map:
            raise ValueError(f"Preconditioner {precond} not recognized.")

        grid_id = method_grid_map[method]
        precond_id = precond_map[precond]
        capi.solver_initialize_grid(
            self.N, self.L, self.h, self.mdv.tol, self.gset.eps_s, self.gset.eps_int,
            grid_id, precond_id
        )

        self.initialize_grid_pb()
        self.initialize_grid_smoothing()

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

    def get_lennard_jones_params(self, particles) -> np.ndarray:
        """Get the Lennard Jones parameters for the particles."""
        if self.mdv.potential_params_file is None:
            raise ValueError("Potential parameters file must be provided for LJ potential.")
        lj_params = pd.read_csv(self.mdv.potential_params_file)
        expected = self.N_typs * (self.N_typs + 1) // 2
        if len(lj_params) != expected:
            raise ValueError(
                f"Potential parameters file must have {expected} unique pairs of types."
            )
        try:
            particles.loc[lj_params['type1']]
            particles.loc[lj_params['type2']]
        except KeyError as e:
            raise ValueError(f"Particle type not found in particles file: {e}")

        lj_params.set_index(['type1', 'type2'], inplace=True)
        if len(lj_params) != len(lj_params.index.unique()):
            raise ValueError(
                "Potential parameters file must have unique pairs of types (type1, type2)."
            )
        lj_params_array = np.empty((self.N_typs, self.N_typs, 2), dtype=np.float64) * np.nan
        for t1, t2 in lj_params.index:
            t1_idx = particles.loc[t1, 'enum']
            t2_idx = particles.loc[t2, 'enum']
            if not np.isnan(lj_params_array[t1_idx, t2_idx, 0]):
                raise ValueError(f"Duplicate potential parameters for types {t1_idx} and {t2_idx}.")
            if not np.isnan(lj_params_array[t2_idx, t1_idx, 0]):
                raise ValueError(f"Potential parameters for types {t1_idx} and {t2_idx} must be symmetric.")
            lj_params_array[t1_idx, t2_idx] = lj_params.loc[(t1, t2), ['sigma', 'epsilon']].values
            lj_params_array[t2_idx, t1_idx] = lj_params_array[t1_idx, t2_idx]
        if np.any(np.isnan(lj_params_array)):
            raise ValueError("Potential parameters for some particle types are missing.")
        lj_params_array[:, :, 0] /= cst.a0  # ang -> a.u.
        lj_params_array[:, :, 1] *= cst.kJmol_to_hartree  # kJ/mol -> Hartree
        lj_params_array = np.ascontiguousarray(lj_params_array.flatten(), dtype=np.float64)

        return lj_params_array

    def validate_water_inputs(self, species_df: pd.DataFrame, coords_df: pd.DataFrame):
        """Check that water-specific inputs are consistent when iswater flag is set."""
        if not self.mdv.iswater:
            return
        required_col = 'type'
        for name, df in [('species file', species_df), ('input file', coords_df)]:
            if required_col not in df.columns:
                raise ValueError(f"When iswater=True, the {name} must contain a '{required_col}' column.")

        species_types = species_df['type'].astype(str).str.upper()
        if not {'O', 'H'}.issubset(set(species_types)):
            raise ValueError("When iswater=True, the species file must define both oxygen ('O') and hydrogen ('H').")

        coord_types = coords_df['type'].astype(str).str.upper().to_numpy()
        if len(coord_types) % 3 != 0:
            raise ValueError("When iswater=True, the input file must list atoms in O-H-H triplets (row count must be a multiple of 3).")

        unique_coord_types = set(coord_types)
        if not unique_coord_types.issubset({'O', 'H'}):
            extras = unique_coord_types.difference({'O', 'H'})
            raise ValueError(f"When iswater=True, input file must contain only oxygen and hydrogen types; found extra types: {extras}.")

        triplets = coord_types.reshape((-1, 3))
        pattern = np.array(['O', 'H', 'H'])
        mismatches = np.where((triplets != pattern).any(axis=1))[0]
        if mismatches.size:
            idx = mismatches[0]
            raise ValueError(
                f"When iswater=True, each molecule must be ordered as O-H-H in the 'type' column; "
                f"molecule {idx} has types {triplets[idx].tolist()}."
            )

    @staticmethod
    def pd_ensure_lowercase(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Ensure that a specified column in a DataFrame is lowercase."""
        if column not in df.columns:
            for col in df.columns:
                if col.lower() == column.lower():
                    df.rename(columns={col: column}, inplace=True)
                    break
        return df
    
    def _initialize_particle_pneigh(self, r_cut: float):
        """Initialize the particle neighbor list."""
        method = self.mdv.neighbor_method.upper()
        if not method in pneigh_method_map:
            raise ValueError(f"Particle neighbor method {method} not recognized.")
        method_id = pneigh_method_map[method]
        capi.solver_initialize_particle_pneigh(method_id, r_cut)

    def _initialize_particle_potential(self, particles: pd.DataFrame) -> tuple[int, np.ndarray, int]:
        """Initialize the particle potential."""
        self.logger.info(f"Initializing particles with potential: '{self.mdv.potential}'")
        potential = self.mdv.potential.upper()
        if not potential in potential_map:
            # print(potential_map, potential)
            raise ValueError(f"Potential {potential} not recognized among {','.join(potential_map.keys())}.")
        pot_id = potential_map[potential]

        if potential == 'TF':
            pot_params = self.get_tosi_fumi_params(particles)
            lj_force_shift = 1
        elif potential == 'LJ':
            pot_params = self.get_lennard_jones_params(particles)
            lj_force_shift = int(bool(self.mdv.lj_force_shift))
            if lj_force_shift:
                self.logger.info("Using force-shifted LJ potential.")
            else:
                self.logger.info("Using LAMMPS-like unshifted LJ potential with tail energy correction.")
        elif potential == 'SC':
            pot_params = self.get_sc_params()
            lj_force_shift = 1

        return pot_id, pot_params, lj_force_shift

    def _validate_rcut(self, r_cut: float) -> float:
        """Initialize the cutoff radius for non-electrostatic interactions."""
        if r_cut is None:
            r_cut = -1.0
        else:
            if r_cut <= 0.0:
                raise ValueError("Optional non-electrostatic cutoff must be positive.")
            self.logger.info(f"Using custom cutoff: {r_cut:.6f} a.u.")
            max_cut = self.L / 2.0
            if r_cut > max_cut:
                raise ValueError(
                    f"Requested cutoff {r_cut:.6f} a.u. exceeds the maximum allowed by minimum-image PBC, "
                    f"L/2 = {max_cut:.6f} a.u."
                )
        return r_cut

    def _initialize_particle_water(self):
        """Initialize the particle settings for water if iswater flag is set."""
        if not self.mdv.iswater:
            return
        estatic_corr = self.mdv.electrostatic_correction.upper()
        if estatic_corr not in elec_corr_map:
            raise ValueError(
                f"Electrostatic correction '{self.mdv.electrostatic_correction}' not recognized. "
                f"Use one of: {', '.join(elec_corr_map.keys())}."
            )
        estatic_corr_id = elec_corr_map[estatic_corr]
        capi.solver_initialize_particles_water(self.mdv.iswater, estatic_corr_id)

    def _initialize_particle_pb(self, particles: pd.DataFrame, df: pd.DataFrame):
        """Initialize the particle settings for Poisson-Boltzmann if poisson_boltzmann flag is set."""
        if not self.mdv.poisson_boltzmann:
            return
        if 'radius' not in particles.columns:
            raise ValueError("Probe radius must be provided in the input file for Poisson-Boltzmann.")
        radius = np.ascontiguousarray(particles.loc[df['type'], 'radius'].values, dtype=np.float64)
        radius = radius / cst.a0 + self.mdv.probe_radius
        self.logger.info("Initializing particles for Poisson-Boltzmann.")
        capi.solver_initialize_particles_pois_boltz(
            self.mdv.gamma_np_au, self.mdv.beta_np, radius
        )

    def initialize_particles(self):
        """Initialize the particles."""
        start_file = self.gset.input_file
        
        self.logger.info(f"Reading particle definitions from file: {self.gset.particles_file}")
        particles = pd.read_csv(self.gset.particles_file)
        self.logger.info(f"Reading starting positions from file: {start_file}")
        df = pd.read_csv(start_file)
        # Normalize column name for type if provided with different casing
        particles = self.pd_ensure_lowercase(particles, 'type')
        df = self.pd_ensure_lowercase(df, 'type')

        if self.mdv.iswater:
            self.logger.info("Water mode enabled (SPC): expecting O-H-H triplets (types O,H,H) in input coordinates.")
            self.validate_water_inputs(particles, df)
        if len(particles) != self.gset.N_typs:
            raise ValueError(
                f"Number of particle types in file ({len(particles)}) does not match N_typs ({self.gset.N_typs})."
            )
        if len(set(particles['type'])) != self.gset.N_typs:
            raise ValueError("Particle types in file must be unique.")
        for idx, part in enumerate(particles['type']):
            self.types_str_to_num[part] = idx
            self.types_num_to_str[idx] = part
        particles.set_index('type', inplace=True)
        particles['enum'] = range(len(particles))

        cas_str = self.gset.cas.upper()
        if not cas_str in ca_scheme_map:
            raise ValueError(f"Charge assignment scheme {cas_str} not recognized.")
        ca_scheme_id = ca_scheme_map[cas_str]

        kBT = self.mdv.kBT

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

        pot_id, pot_params, lj_force_shift = self._initialize_particle_potential(particles)
        r_cut = self._validate_rcut(self.mdv.neighbor_r_cut)

        capi.solver_initialize_particles(
            self.N, self.N_typs, self.L, self.h, self.N_p,
            pot_id, ca_scheme_id,
            types, pos, vel, mass, charges,
            pot_params, r_cut, lj_force_shift
        )

        self._initialize_particle_pneigh(r_cut)
        self._initialize_particle_water()
        self._initialize_particle_pb(particles, df)


    def initialize_integrator(self):
        """Initialize the MD integrator."""
        self.logger.info(f"Initializing integrator: '{self.mdv.integrator}'")
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
        if ffile is None or not self.mdv.invert_time:
            # STEP 0 Verlet
            # self.logger.debug("Running first step of MD loop (Verlet)...")
            # self.logger.debug("Updating charges...")
            self.update_charges()
            # self.logger.debug("Smoothing charges...")
            self.smoothing(); 
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
            self.smoothing(); 
            # self.logger.debug("Updating k^2 grid for Poisson-Boltzmann...")
            self.update_eps_k2()
            # self.logger.debug("Updating field...")
            self.initialize_field()
            # self.logger.debug("Computing forces...")
            self.compute_forces()
            # self.logger.debug("Running second part of integrator...")
            self.integrator_part2()
        elif ffile:
            if self.mpi_rank == 0:
                df = pd.read_csv(ffile)
                phi = np.ascontiguousarray(df['phi'].values).reshape((self.N, self.N, self.N))
            else:
                phi = np.empty((0, 0, 0), dtype=np.float64)  # Dummy array for non-root ranks
            capi.solver_set_field(phi)
            if self.mpi_rank == 0:
                phi = np.ascontiguousarray(df['phi_prev'].values).reshape((self.N, self.N, self.N))
            else:
                phi = np.empty((0, 0, 0), dtype=np.float64)
            capi.solver_set_field_prev(phi)

            self.logger.info(f"Initialization step skipped due to field loaded from file.")

        if self.mdv.rescale:
            capi.solver_rescale_velocities()

    def rescale_periodic(self, step: int):
        """Periodically remove total momentum when rescaling is enabled."""
        if not self.mdv.rescale or self.mdv.rescale_stride is None:
            return
        if step % self.mdv.rescale_stride == 0:
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
        if self.mdv.iswater:
            self.energy_intra = capi.solver_compute_intramolecular_forces()
            self.energy_corr = capi.solver_compute_forces_electrostatic_correction()
        else:
            self.energy_intra = 0.0
            self.energy_corr = 0.0
        capi.solver_compute_forces_tot()
        # Electrostatic energy from the grid (not printed in energy.csv per request)
        self.energy_elec = capi.get_energy_elec()

    @Clock('forces_field')
    def compute_forces_field(self):
        """Compute the forces on the particles due to the electric field."""
        # self.logger.debug("Computing forces due to electric field...")
        self.potential_short_range = capi.solver_compute_forces_elec()

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

    @Clock('smoothing')
    def smoothing(self):
        if self.smoothing_type.upper() == 'GAUSS':
            self._smoothing_gauss()
        else:
            capi.solver_smoothing()

    def _smoothing_gauss(self):
        if self.mpi_rank == 0:
            rho = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        else:
            rho = np.empty((0, 0, 0), dtype=np.float64)  # Dummy array for non-root ranks

        # Only rank 0 performs the smoothing and then broadcasts the smoothed charge density to all ranks.
        # The get/set functions have to be called by all ranks to collect/distribute the data
        # - For collection, only rank 0 gets the full buffer
        # - For broadcasting only rank 0 needs the actual data
        capi.get_q(rho)
        if self.mpi_rank == 0:
            # gaussian_filter expects sigma in grid-cell units, while sigma_gauss is in length units.
            sigma_grid = self.smoothing_sigma / self.h
            rho_smooth = gaussian_filter(rho, sigma=sigma_grid, mode='wrap')
            rho_smooth = np.ascontiguousarray(rho_smooth)    
        else:
            rho_smooth = np.empty((0, 0, 0), dtype=np.float64)

        # Injection in C
        capi.set_q(rho_smooth)
    
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
            self.t_charges = Clock.get_clock('charges').last_call
            self.smoothing()
            self.t_smoothing = Clock.get_clock('smoothing').last_call
            self.update_eps_k2()
            self.n_iters = self.update_field()
            self.t_field = Clock.get_clock('field').last_call
            self.t_elec_total = self.t_charges + self.t_smoothing + self.t_field
            self.t_iters = self.t_field
        self.compute_forces()
        self.integrator_part2()

    def md_loop(self):
        """Run the molecular dynamics loop."""
        if self.mdv.init_steps:
            self.logger.info("Running MD loop initialization steps...")
            for i in ProgressBar(self.mdv.init_steps, description="MD init"):
                self.md_loop_iter()
        
        temp = capi.get_temperature()
        self.logger.info(f"Temperature: {temp:.2f} K")

        self.logger.info("Running MD loop...")
        if self.thermostat:
            self.logger.info("Thermostat ON in production run")

        for i in ProgressBar(self.mdv.N_steps, description="MD steps"):
            self.md_loop_iter()
            self.rescale_periodic(i + 1)
            self.md_loop_output(i+1)  # Report step number as 1-indexed also to avoid double printing the final one

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
        self.logger.info(f'  N = {self.N}, L [A] = {self.L * cst.a0}, h [A] = {self.h * cst.a0}')
        self.logger.info(f'  density = {density} g/cm^3')
        self.logger.info(f'  Solvent dielectric constant: {self.gset.eps_s}')
        self.logger.info(f'  Solver: "{self.mdv.method}",  Preconditioner: "{self.gset.precond}"')
        self.logger.info(f'  Charge assignment scheme: "{self.gset.cas}"')
        # self.logger.info(f'  Preconditioning: {self.mdv.preconditioning}')
        self.logger.info(f'  Integrator: "{self.mdv.integrator}", dt = {self.mdv.dt} au = {self.mdv.dt * cst.t_au} fs')
        self.logger.info(f'  Potential: "{self.mdv.potential}"')
        self.logger.info(f'  Electrostatic correction: "{self.mdv.electrostatic_correction}"')
        self.logger.info(f'  Elec: {self.mdv.elec}    NotElec: {self.mdv.not_elec}')
        self.logger.info(f'  Temperature: {self.mdv.T} K,  Thermostat: {self.mdv.thermostat},  Gamma: {self.mdv.gamma}')
        self.logger.info(f'  Velocity rescaling: {self.mdv.rescale}')
        if self.mdv.rescale and self.mdv.rescale_stride is not None:
            self.logger.info(
                f'  Periodic velocity rescaling: enabled every {self.mdv.rescale_stride} MD steps'
            )
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
