import atexit
import os
from abc import ABC, abstractmethod
from functools import wraps
from io import StringIO

import pandas as pd

from ..grid.base_grid import BaseGrid
from ..input import OutputSettings
from ..loggers import logger
from ..particles import Particles


def ensure_enabled(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return
        return func(self, *args, **kwargs)
    return wrapper

class BaseOutputFile(ABC):
    name = None
    def __init__(self, *args, path: str, enabled: bool = True, overwrite: bool = True, **kwargs):
        self.path = os.path.abspath(path)
        self.enabled = enabled
        if not enabled:
            return

        logger.info("Saving %s to '%s'", self.name, self.path)

        if os.path.exists(path):
            if overwrite:
                os.remove(path)
            else:
                raise ValueError(f"File {path} already exists")

        self.buffer = StringIO()
        self.file = open(path, 'w')
        atexit.register(self.close)

    @abstractmethod
    def write_data(self, df: pd.DataFrame, mode: str = 'a'):
        pass

    @ensure_enabled
    def flush(self):
        self.file.write(self.buffer.getvalue())
        self.buffer.truncate(0)
        self.buffer.seek(0)
        self.file.flush()

    @ensure_enabled
    def close(self):
        self.flush()
        self.file.close()


class OutputFiles:
    field = None
    performance = None
    energy = None
    temperature = None
    solute = None
    tot_force = None
    restart = None

    files = ['field', 'performance', 'energy', 'temperature', 'solute', 'tot_force', 'restart']

    format_classes = {}

    def __init__(self, oset: OutputSettings):
        self.oset = oset
        self.base_path = oset.path
        self.fmt = oset.format

        self.out_stride = oset.stride
        self.out_flushstride = (oset.flushstride or 0) * oset.stride
        self.restart_stride = oset.restart_stride

        self.init_files()

    def init_files(self):
        """Initialize the output files."""
        ptr = self.format_classes[self.fmt]
        for name in self.files:
            cls = ptr[name]
            _path = os.path.join(self.base_path, f'{name}.{self.fmt}')
            setattr(self, name, cls(
                path = _path,
                enabled = getattr(self.oset, f'print_{name}'),
                overwrite=True
            ))

    def flush(self):
        for fname in self.files:
            file = getattr(self, fname)
            if file:
                file.flush()

    def output(self, iter: int, grid: BaseGrid, particles: Particles):
        """Output the results of the molecular dynamics loop."""
        if iter % self.out_stride == 0:
            self.energy.write_data(iter, grid, particles)
            self.tot_force.write_data(iter, grid, particles)
            self.temperature.write_data(iter, grid, particles)
            self.solute.write_data(iter, grid, particles)
            self.performance.write_data(iter, grid, particles)
            self.field.write_data(iter, grid, particles)
            if self.out_flushstride and iter % self.out_flushstride == 0:
                self.flush()
        if self.restart_stride and iter % self.restart_stride == 0:
            self.restart.write_data(iter, grid, particles, mode='w')
            self.restart.flush()

    @classmethod
    def register_format(cls, name: str, classes: dict):
        cls.format_classes[name] = classes
