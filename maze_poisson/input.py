from pathlib import Path

import numpy as np
import yaml

import os
from .constants import a0, density, kB, m_Cl, m_Na, t_au, ref_L, ref_N
from .loggers import logger
import argparse

###################################################################################

### Output settings ###

class OutputSettings:
    print_field = None # to move
    print_performance = None # to move
    print_solute = None # to move
    print_energy = None # to move
    print_temperature = None 
    print_tot_force = None 
    print_iters = False
    path = 'Outputs/'
    debug = False
    restart = None
    generate_restart_file = None 
    iter_restart = None

###################################################################################

### Grid and box settings ###
class GridSetting:
    def __init__(self):
        self._N = None
        self._L = None
        self._N_p = None
        self._N_tot = None
        self._h = None
        self._input_file = None
        self._restart_file = None
        self.cas = None # B-Spline or CIC
        self.rescale_force = None
        self.eps_s = None
        self.I = None
        self.w = None

    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        self._N = value
        self._N_tot = int(value ** 3)
        self._h = None

    @property
    def N_p(self):
        return self._N_p

    @N_p.setter
    def N_p(self, value):
        self._N_p = value
        # Compute L_ang and log/print it if L is set
        if hasattr(self, 'L_ang') and self.L_ang is not None:
            return  # avoid overwriting if L_ang already set
        if self._L is not None:
            self.L_ang = np.round(self._L * a0, 4)  # convert from a.u. to Å for logging
        else:
            self.L_ang = np.round((((self._N_p * (m_Cl + m_Na)) / (2 * density)) ** (1 / 3)) * 1.e9, 4)  # in Å
            self._L = self.L_ang / a0  # in a.u.

    @property
    def N_tot(self):
        return self._N_tot

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self.L_ang = value  # input is assumed in Å
        self._L = value / a0  # convert to a.u.
        self._h = None

    @property
    def h(self):
        if self._h is None:
            self._h = self.L / self.N
        return self._h

    @property
    def input_file(self):
        if self._input_file is None:
            self._input_file = 'input_files/input_coord'+str(self.N_p)+'.csv'
        return self._input_file

    @property
    def restart_file(self):
        if self._restart_file is None:
            self._restart_file = 'restart_files/restart_N'+str(self.N)+'_step9999.csv'
        return self._restart_file

###################################################################################

### MD variables ###
class MDVariables:
    def __init__(self):
        self._T = None
        self._kBT = None
        self.N_steps = None
        self.init_steps = None
        self.thermostat = None # to move
        self._dt_fs = None # dt in fs
        self._dt = None        # timestep for the solute evolution given in fs and converted in a.u. # to move
        self.stride = 1              # saves every stride steps
        self.initialization = 'CG'   # always CG
        self.preconditioning = 'Yes' # Yes or No
        self.rescale = None # rescaling of the initial momenta to have tot momenta = 0
        self.elec = None # to move
        self.not_elec = None # to move
        self.potential = 'TF' # Tosi Fumi (TF) or Leonard Jones (LJ)
        self.integrator = 'OVRVO'
        self.gamma = 1e-3 # OVRVO parameter
        self.method = None
        self.non_polar = False
        self.gamma_np = 0.0
        self.beta_np = 0.0
        self.probe_radius = 1.4

    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        self._T = value
        self._kBT = kB * value
    
    @property
    def kBT(self):
        return self._kBT
    
    @property
    def dt_fs(self):
        return self._dt_fs
    
    @dt_fs.setter
    def dt_fs(self, value):
        self._dt_fs = value
        self._dt = value / t_au
    
    @property
    def dt(self):
        return self._dt

required_inputs = {
    'grid_setting': ['N_p','cas', 'rescale_force', 'L'],
    'output_settings': ['restart'],
    'md_variables': ['N_steps', 'tol', 'rescale', 'T']
}

def initialize_from_yaml(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    if not isinstance(filename, Path):
        logger.error("filename must be a Path or a str")
        raise TypeError('filename must be a Path or a str')
        
    if not filename.exists():
        logger.error(f'Input file {filename} does not exist')
        raise FileNotFoundError(f'Input file {filename} does not exist')
    
    grid_setting = GridSetting()
    output_settings = OutputSettings()
    md_variables = MDVariables()

    with filename.open() as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    method = data.get("md_variables", {}).get("method", "Poisson MaZe")
    md_variables.method = method

    missing = []
    for key in ['output_settings', 'grid_setting', 'md_variables']:
        ptr = data.get(key, {})
        req = required_inputs.get(key, [])
        missing += [f'{key}.{r}' for r in req if r not in ptr]
        items = list(data[key].items())
        if key == 'grid_setting':
            items.sort(key=lambda item: 0 if item[0] == 'L' else 1)
        for k, v in items:
            setattr(eval(key), k, v)

    if method == "PB MaZe":
        grid_setting.eps_s = data["grid_setting"].get("eps_s", 80)
        grid_setting.k_b = data["grid_setting"].get("I", None)
        grid_setting.w = data["grid_setting"].get("w", None)

        md_variables.non_polar = data["md_variables"].get("non_polar", False)
        md_variables.gamma_np = data["md_variables"].get("gamma_np", 0.0)
        md_variables.beta_np = data["md_variables"].get("beta_np", 0.0)
        md_variables.probe_radius = data["md_variables"].get("probe_radius", 1.4)

    if missing:
        raise ValueError(f'Missing required inputs: {", ".join(missing)}')

    if output_settings.restart:
        if not grid_setting.restart_file:
            logger.error('restart_file must be provided if restart is True')
            raise ValueError('restart_file must be provided if restart is True')
        if not Path(grid_setting.restart_file).exists():
            logger.error(f'Restart file {grid_setting.restart_file} does not exist')
            raise FileNotFoundError(f'Restart file {grid_setting.restart_file} does not exist')

    return grid_setting, output_settings, md_variables