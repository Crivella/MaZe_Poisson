from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from ...constants import a0, density, kB, m_Cl, m_Na, t_au
from ..loggers import logger


class BaseSettings(ABC):
    tocheck = []
    def __init__(self):
        self.defaults()

    @abstractmethod
    def defaults(self):
        """Set default values."""

    def validate(self):
        """Check if all required values are set."""
        missing = []
        for key in self.tocheck:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required values: {missing}")

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        for key, value in data.items():
            setattr(obj, key, value)
        return obj

class OutputSettings(BaseSettings):
    tocheck = []

    def defaults(self):
        """Set default values."""
        # self.print_field = False
        self.print_solute = False
        self.print_performance = False
        self.print_momentum = False
        self.print_energy = False
        self.print_temperature = False
        self.print_tot_force = False
        self.print_restart = False
        self.print_restart_field = False
        self.path = 'Outputs/'
        self.format = 'csv'
        self.stride = 50
        self.flushstride = 0
        self.debug = False
        self.generate_restart_file = False
        self.restart_step = None

class GridSetting(BaseSettings):
    tocheck = ['N', 'N_p', 'L', 'h', 'file']
    def defaults(self):
        """Set default values."""
        self._N = None
        self.input_file = None
        self.restart_file = None
        self.restart_field_file = None
        self._L = None
        self._L_ang = None
        self.h = None
        
    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        self._N = value
        if self.L is not None:
            self.h = self.L / value
    @property
    def N_p(self):
        return self._N_p

    @N_p.setter
    def N_p(self, value):
        self._N_p = value
        # self.L_ang = np.round((((value*(m_Cl + m_Na)) / (2*density))  **(1/3)) *1.e9, 4) # in A
        # self.L = self.L_ang / a0 # in amu
        # if self.N is not None:
        #     self.h = self.L / self.N

    @property
    def L(self):
        return self._L
    @L.setter
    def L(self, value):
        self._L = value
        self._L_ang = value * a0
        if self.N is not None:
            self.h = value / self.N
    @property
    def L_ang(self):
        return self._L_ang
    @L_ang.setter
    def L_ang(self, value):
        self._L_ang = value
        self._L = value / a0
        if self.N is not None:
            self.h = self.L / self.N

    @property
    def file(self):
        return self.restart_file or self.input_file

class MDVariables(BaseSettings):
    tocheck = ['N_steps', 'T', 'dt']
    def defaults(self):
        """Set default values."""
        self._T = None
        self._kBT = None
        self.dt = None
        self.N_steps = None

        self.init_steps = 0
        self.thermostat = False
        self.preconditioning = True
        self.rescale = False
        self.elec = True
        self.not_elec = True
        self.potential = 'TF'
        self.integrator = 'OVRVO'
        self.method = 'FFT'
        self.tol = 1e-7
        self.gamma = 1e-3
        self._invert_time = False
    
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
        if value <= 0:
            raise ValueError("Timestep must be positive.")
        if self.invert_time:
            value = -value
        self._dt_fs = value
        self.dt = value / t_au

    @property
    def invert_time(self):
        return self._invert_time
    @invert_time.setter
    def invert_time(self, value):
        if value and self.dt_fs is not None and self.dt_fs > 0:
            dt_fs = -self.dt_fs
            self._dt_fs = dt_fs
            self.dt = dt_fs / t_au
        self._invert_time = value

def mpi_file_loader(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # obj = None
        # if MPIBase.master:
        #     obj = func(*args, **kwargs)
        # obj = mpi.comm.bcast(obj, root=0)
        obj = func(*args, **kwargs)
        return obj
    return wrapper

def validate_all(*args):
    """Call validate on all objects and raise a ValueError if any fail."""
    msg = []
    for obj in args:
        name = obj.__class__.__name__
        try:
            obj.validate()
        except ValueError as e:
            msg.append(f"{name}: {str(e)}")
    if msg:
        raise ValueError('\n'.join(msg))
