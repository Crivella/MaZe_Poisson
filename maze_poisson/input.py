from pathlib import Path

import numpy as np
import yaml

from .constants import a0, density, kB, m_Cl, m_Na, t_au
from .loggers import logger

###################################################################################

### Output settings ###

class OutputSettings:
    print_field = None # to move
    print_performance = None # to move
    print_solute = None # to move
    print_energy = None # to move
    print_temperature = None 
    print_momentum= None 
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
            self._input_file = f'input_files/input_coord{self.N_p}.csv'
        return self._input_file

    @property
    def restart_file(self):
        if self._restart_file is None:
            self._restart_file = f'restart_files/restart_Np{self.N_p}.csv'
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

    missing = []
    for key in ['output_settings', 'grid_setting', 'md_variables']:
        ptr = data.get(key, {})
        req = required_inputs.get(key, [])
        missing += [f'{key}.{r}' for r in req if r not in ptr]
        items = list(data[key].items())
        if key == 'grid_setting':
            # Ensure L is set before N_p
            items.sort(key=lambda item: 0 if item[0] == 'L' else 1)
        for k, v in items:
            setattr(eval(key), k, v)

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
