import atexit
import os
from abc import ABC, abstractmethod
from io import StringIO

import pandas as pd

from .grid.base_grid import BaseGrid
from .input import OutputSettings
from .particles import Particles


class BaseOutputFile(ABC):
    def __init__(self, *args, path: str, enabled: bool = True, overwrite: bool = True, **kwargs):
        if os.path.exists(path):
            if overwrite:
                os.remove(path)
            else:
                raise ValueError(f"File {path} already exists")
        self.enabled = enabled
        self.buffer = StringIO()
        self.file = open(path, 'w')

        atexit.register(self.close)

    @abstractmethod
    def write_data(self, df: pd.DataFrame):
        pass

    def flush(self):
        self.file.write(self.buffer.getvalue())
        self.buffer.truncate(0)
        self.buffer.seek(0)
        self.file.flush()

    def close(self):
        # print("Closing file")
        self.flush()
        self.file.close()

class NullOutputFile(BaseOutputFile):
    def __init__(self, *args, **kwargs):
        pass
    def write_data(self, df: pd.DataFrame):
        pass
    def flush(self):
        pass
    def close(self):
        pass

class CSVOutputFile(BaseOutputFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pd.DataFrame(columns=self.headers).to_csv(self.buffer, index=False)

    def write_data(self, iter: int, grid: BaseGrid, particles: Particles):
        if not self.enabled:
            return
        df = self.get_data(iter, grid, particles)
        df.to_csv(self.buffer, header=False, index=False, mode='a')

    @property
    @abstractmethod
    def headers(self):
        pass

    @abstractmethod
    def get_data(self, iter: int, grid: BaseGrid, particles: Particles) -> pd.DataFrame:
        pass

class EnergyCSVOutputFile(CSVOutputFile):
    headers = ['iter', 'K', 'V_notelec']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'K': [particles.get_kinetic_energy()],
            'V_notelec': [grid.potential_notelec]
        })

class ForcesCSVOutputFile(CSVOutputFile):
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'Fx': [particles.forces[:, 0].sum()],
            'Fy': [particles.forces[:, 1].sum()],
            'Fz': [particles.forces[:, 2].sum()]
        })

class TemperatureCSVOutputFile(CSVOutputFile):
    headers = ['iter', 'T']
    def get_data(self, iter: int, grid: BaseGrid = None, particles: Particles = None):
        return pd.DataFrame({
            'iter': [iter],
            'T': [particles.get_temperature()]
        })



def get_file_class(file_type: str, enabled: bool) -> BaseOutputFile:
    file_type = file_type.lower()
    if not enabled:
        return NullOutputFile
    if file_type == 'csv':
        return CSVOutputFile
    raise ValueError(f"Invalid file type: {file_type}")

class OutputFiles:
    # def __init__(self):
    #     self.files = []

    # def register_file(self, file: BaseOutputFile):
    #     self.files.append(file)

    # def write_headers(self):
    #     for file in self.files:
    #         file.write_headers()

    
    field = None
    performance = None
    iters = None
    energy = None
    temperature = None
    solute = None
    tot_force = None

    def flush(self):
        for file in [
            self.field, self.performance, self.iters, self.energy, self.temperature, self.solute, self.tot_force
            ]:
            if file:
                file.flush()   

# open_files = []

# def generate_output_file(out_path, overwrite=True):
#     if os.path.exists(out_path):
#         if overwrite:
#             os.remove(out_path)
#         else:
#             raise ValueError(f"File {out_path} already exists")
#     res = open(out_path, 'w')
#     open_files.append(res)
#     return res

csv_headers = {
    'field': ['iter', 'x', 'MaZe'],
    'performance': ['iter', 'time', 'n_iters'],
    'iters': ['iter', 'max_sigma', 'norm'],
    'energy': ['iter', 'K', 'V_notelec'],
    'temperature': ['iter', 'T'],
    'solute': ['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec'],
    'tot_force': ['iter', 'Fx', 'Fy', 'Fz']
}

def generate_output_files(oset: OutputSettings):
    # N = grid.N
    # N_p = grid.N_p
    # output_settings = grid.output_settings
    path = oset.path

    output_files = OutputFiles()

    # if md_variables.thermostat:
    #     thermostat_path = os.path.join(path, 'Thermostatted')
    #     os.makedirs(thermostat_path, exist_ok=True)
    #     path = thermostat_path  # if thermostating is true then this has to take place.

    # tag_str = '_N' + str(N) + '_N_p_' + str(N_p)
    tag_str = ''

    fmt = oset.format

    output_files.field = NullOutputFile()
    output_files.performance = NullOutputFile()
    output_files.iters = NullOutputFile()
    output_files.energy = EnergyCSVOutputFile(
        path = os.path.join(path, f'energy{tag_str}.csv'),
        enabled = oset.print_energy,
        overwrite=True
    )
    output_files.temperature = TemperatureCSVOutputFile(
        path = os.path.join(path, f'temperature{tag_str}.csv'),
        enabled = oset.print_temperature,
        overwrite=True
    )
    output_files.solute = NullOutputFile()
    output_files.tot_force = ForcesCSVOutputFile(
        path = os.path.join(path, f'tot_force{tag_str}.csv'),
        enabled = oset.print_tot_force,
        overwrite=True
    )
    # for name in ['field', 'performance', 'iters', 'energy', 'temperature', 'solute', 'tot_force']:
    #     enabled = getattr(oset, f'print_{name}')
    #     cls_type = get_file_class(fmt, enabled)
    #     out_file = cls_type(
    #         path = os.path.join(path, f'{name}{tag_str}.csv'),
    #         headers = csv_headers[name],
    #         overwrite=True
    #     )
            
    #     setattr(output_files, f'{name}', out_file)

    # cls_type = get_file_class(oset.file_type, oset.enabled)
    # if oset.print_field:
    #     os.makedirs(path, exist_ok=True)
    #     output_field = os.path.join(path, f'field{tag_str}.csv')
    #     output_files.file_output_field = generate_output_file(output_field)
    #     output_files.file_output_field.write("iter,x,MaZe\n")

    # if oset.print_performance:
    #     os.makedirs(path, exist_ok=True)
    #     output_performance = os.path.join(path, 'performance_tol'+str(md_variables.tol)+'.csv')
    #     output_files.file_output_performance = generate_output_file(output_performance)
    #     output_files.file_output_performance.write("iter,time,n_iters\n")

    # if oset.print_iters:
    #     os.makedirs(path, exist_ok=True)
    #     output_iters = os.path.join(path, f'iters{tag_str}.csv')
    #     output_files.file_output_iters = generate_output_file(output_iters)
    #     output_files.file_output_iters.write("iter,max_sigma,norm\n")

    # # prints only kinetic and non electrostatic potential
    # if oset.print_energy:
    #     os.makedirs(path, exist_ok=True)
    #     output_energy = os.path.join(path, f'energy{tag_str}.csv')
    #     output_files.file_output_energy = generate_output_file(output_energy)
    #     output_files.file_output_energy.write("iter,K,V_notelec\n")

    # if oset.print_temperature:
    #     os.makedirs(path, exist_ok=True)
    #     output_temperature = os.path.join(path, f'temperature{tag_str}.csv')
    #     output_files.file_output_temperature = generate_output_file(output_temperature)
    #     output_files.file_output_temperature.write("iter,T\n")

    # if oset.print_solute:
    #     os.makedirs(path, exist_ok=True)
    #     output_solute = os.path.join(path, f'solute{tag_str}.csv')
    #     output_files.file_output_solute = generate_output_file(output_solute)
    #     output_files.file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz,fx_elec,fy_elec,fz_elec\n")

    # if oset.print_tot_force:
    #     os.makedirs(path, exist_ok=True)
    #     output_tot_force = os.path.join(path, f'tot_force{tag_str}.csv')
    #     output_files.file_output_tot_force = generate_output_file(output_tot_force)
    #     output_files.file_output_tot_force.write("iter,Fx,Fy,Fz\n")

    return output_files

# @atexit.register
# def close_output_files():
#     not_closed = []
#     while open_files:
#         app = open_files.pop()
#         try:
#             app.close()
#         except Exception as e:
#             print(f"Error closing file: {e}")
#             not_closed.append(app)

#     if not_closed:
#         open_files.extend(not_closed)
#         print("Some files could not be closed")
            