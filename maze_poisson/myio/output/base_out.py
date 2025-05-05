import atexit
import os
from abc import ABC, abstractmethod
from io import StringIO

import pandas as pd

from ...myio.loggers import Logger
from .. import get_enabled_io
from ..input import OutputSettings


class BaseOutputFile(Logger, ABC):
    name = None
    def __init__(self, *args, path: str, enabled: bool = True, overwrite: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = os.path.abspath(path)
        self._enabled = enabled
        self._io_enabled = get_enabled_io()

        if not self.enabled:
            return

        self.logger.info("Saving %s to '%s'", self.name, self.path)

        if os.path.exists(path):
            if overwrite:
                try:  # When running with MPI file could be locked
                    os.remove(path)
                except:
                    pass
            else:
                raise ValueError(f"File {path} already exists")

        self.buffer = StringIO()
        atexit.register(self.close)

    @property
    def enabled(self):
        return self._enabled and self._io_enabled

    @abstractmethod
    def write_data(self, df: pd.DataFrame, mode: str = 'a', mpi_bypass: bool = False):
        pass

    def flush(self):
        if self.enabled:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(self.buffer.getvalue())
            self.buffer.truncate(0)
            self.buffer.seek(0)

    def close(self):
        if self.enabled:
            self.flush()
            self.buffer.close()

class OutputFiles:
    performance = None
    energy = None
    momentum = None
    temperature = None
    solute = None
    tot_force = None
    restart = None
    restart_field = None

    files = ['performance', 'energy', 'momentum', 'temperature', 'solute', 'tot_force']
    files_restart = ['restart', 'restart_field']

    format_classes = {}

    last = -1

    def __init__(self, oset: OutputSettings):
        self.oset = oset
        self.base_path = oset.path
        self.fmt = oset.format

        if not os.path.exists(self.base_path) and get_enabled_io():
            os.makedirs(self.base_path, exist_ok=True)

        self.out_stride = oset.stride
        self.out_flushstride = (oset.flushstride or 0) * oset.stride
        self.restart_step = oset.restart_step

        self.init_files()

    def init_files(self):
        """Initialize the output files."""
        ptr = self.format_classes[self.fmt]
        for name in self.files + self.files_restart:
            cls = ptr[name]
            _path = os.path.join(self.base_path, f'{name}.{self.fmt}')
            setattr(self, name, cls(
                path = _path,
                enabled = getattr(self.oset, f'print_{name}'),
                overwrite=True
            ))

    @property
    def all_files(self):
        return self.files + self.files_restart

    def flush(self):
        for fname in self.all_files:
            file = getattr(self, fname)
            if file:
                file.flush()

    def output(self, itr: int, solver, force: bool = False):
        """Output the results of the molecular dynamics loop."""
        if force or itr % self.out_stride == 0:
            if self.last != itr:
                self.last = itr
                for name in self.files:
                    file = getattr(self, name)
                    if file:
                        file.write_data(itr, solver)
                if force or (self.out_flushstride and itr % self.out_flushstride == 0):
                    self.flush()
        if self.restart_step == itr:
            self.restart.write_data(itr, solver, mode='w')
            self.restart_field.write_data(itr, solver, mode='w', mpi_bypass=True)
            self.restart.flush()
            self.restart_field.flush()

    @classmethod
    def register_format(cls, name: str, classes: dict):
        cls.format_classes[name] = classes
