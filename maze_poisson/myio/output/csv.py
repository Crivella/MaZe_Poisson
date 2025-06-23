from abc import abstractmethod

import numpy as np
import pandas as pd

from ...c_api import capi
from ...constants import a0, conv_mass
from .base_out import BaseOutputFile, OutputFiles


class CSVOutputFile(BaseOutputFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_headers()

    def init_headers(self):
        if not self.enabled:
            return
        pd.DataFrame(columns=self.headers).to_csv(self.buffer, index=False)

    def write_data(self, iter: int, solver = None, mode: str = 'a', mpi_bypass: bool = False):
        if not self.enabled:
            # Needed when the get_data method needs to be called on all ranks to avoid MPI deadlock
            if self._enabled and mpi_bypass:
                self.get_data(iter, solver)
            return
        header = False
        if mode == 'w':
            open(self.path, 'w').close()
            self.buffer.truncate(0)
            self.buffer.seek(0)
            header = True
        df = self.get_data(iter, solver)
        df.to_csv(self.buffer, columns=self.headers, header=header, index=False, mode=mode)

    @property
    @abstractmethod
    def headers(self):
        pass

    @abstractmethod
    def get_data(self, iter: int, solver) -> pd.DataFrame:
        pass

class EnergyCSVOutputFile(CSVOutputFile):
    name = 'energy'
    headers = ['iter', 'K', 'V_notelec']
    def get_data(self, iter: int, solver):
        kin = capi.get_kinetic_energy()
        return pd.DataFrame({
            'iter': [iter],
            'K': [kin],
            'V_notelec': [solver.potential_notelec]
        })

class MomentumCSVOutputFile(CSVOutputFile):
    name = 'momentum'
    headers = ['iter', 'Px', 'Py', 'Pz']
    def get_data(self, iter: int, solver):
        momentum = np.empty(3, dtype=np.float64)
        capi.get_momentum(momentum)
        return pd.DataFrame({
            'iter': [iter],
            'Px': [momentum[0]],
            'Py': [momentum[1]],
            'Pz': [momentum[2]]
        })

class TotForcesCSVOutputFile(CSVOutputFile):
    name = 'forces'
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, solver):
        forces = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_fcs_tot(forces)
        df = pd.DataFrame(forces.sum(axis=0).reshape(1,3), columns=['Fx', 'Fy', 'Fz'])
        df['iter'] = iter
        return df

class ForcesCSVOutputFile(CSVOutputFile):
    name = 'forces'
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, solver):
        # if output_settings.print_components_force:
        #     os.makedirs(path, exist_ok=True)
        #     output_force = os.path.join(path, 'force_N' + str(N) +'.csv')
        #     output_files.file_output_force = generate_output_file(output_force)
        #     output_files.file_output_force.write("iter,particle,fx_RF,fy_RF,fz_RF,fx_DB,fy_DB,fz_DB,fx_IB,fy_IB,fz_IB,fx_NP,fy_NP,fz_NP\n")    

        raise NotImplementedError("TODO")
        forces = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_fcs_tot(forces)
        df = pd.DataFrame(forces.sum(axis=0).reshape(1,3), columns=['Fx', 'Fy', 'Fz'])
        df['iter'] = iter
        return df

class TemperatureCSVOutputFile(CSVOutputFile):
    name = 'temperature'
    headers = ['iter', 'T']
    def get_data(self, iter: int, solver):
        temp = capi.get_temperature()
        return pd.DataFrame({
            'iter': [iter],
            'T': [temp]
        })

class SolutesCSVOutputFile(CSVOutputFile):
    name = 'solute'
    headers = ['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']
    def get_data(self, iter: int, solver):
        tmp = np.empty((solver.N_p, 3), dtype=np.float64)
        df = pd.DataFrame()
        
        capi.get_pos(tmp)
        df[['x', 'y', 'z']] = tmp
        capi.get_vel(tmp)
        df[['vx', 'vy', 'vz']] = tmp
        capi.get_fcs_elec(tmp)
        df[['fx_elec', 'fy_elec', 'fz_elec']] = tmp

        tmp = np.empty(solver.N_p, dtype=np.int64)
        capi.get_charges(tmp)
        df['charge'] = tmp

        df['iter'] = iter
        df['particle'] = range(solver.N_p)
        return df

class PerformanceCSVOutputFile(CSVOutputFile):
    name =  'performance'
    headers = ['iter', 'time', 'n_iters']
    def get_data(self, iter: int, solver):
        return pd.DataFrame({
            'iter': [iter],
            'time': [solver.t_iters],
            'n_iters': [solver.n_iters]
        })

class RestartCSVOutputFile(CSVOutputFile):
    name = 'restart'
    headers = ['charge', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()

        tmp = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_pos(tmp)
        df[['x', 'y', 'z']] = tmp * a0
        capi.get_vel(tmp)
        df[['vx', 'vy', 'vz']] = tmp

        tmp = np.empty(solver.N_p, dtype=np.int64)
        capi.get_charges(tmp)
        df['charge'] = tmp
        tmp = np.empty(solver.N_p, dtype=np.float64)
        capi.get_masses(tmp)
        df['mass'] = tmp / conv_mass

        return df

class RestartFieldCSVOutputFile(CSVOutputFile):
    name = 'restart_field'
    headers = ['phi_prev', 'phi']
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()
        tmp = np.empty((solver.N, solver.N, solver.N), dtype=np.float64)
        capi.get_field(tmp)
        df['phi'] = tmp.flatten()
        capi.get_field_prev(tmp)
        df['phi_prev'] = tmp.flatten()

        return df


OutputFiles.register_format(
    'csv',
    {
        'performance': PerformanceCSVOutputFile,
        'energy': EnergyCSVOutputFile,
        'momentum': MomentumCSVOutputFile,
        'temperature': TemperatureCSVOutputFile,
        'solute': SolutesCSVOutputFile,
        'tot_force': TotForcesCSVOutputFile,
        'force': ForcesCSVOutputFile,
        'restart': RestartCSVOutputFile,
        'restart_field': RestartFieldCSVOutputFile
    }
)
