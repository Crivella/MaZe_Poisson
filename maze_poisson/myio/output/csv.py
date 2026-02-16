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
            # Needed when the get_data method needs to be called on all ranks to avoid MPI deadlock/desyncs
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
    headers = ['iter', 'K', 'V_notelec', 'V_elec', 'V_intra', 'E_corr']
    
    def get_data(self, iter: int, solver):
        kin = capi.get_kinetic_energy()

        return pd.DataFrame({
            'iter': [iter],
            'K': [kin],
            'V_notelec': [solver.potential_notelec],
            # 1/2 \sum_i q_i phi_i
            'V_elec': [solver.energy_elec],
            'V_intra': [solver.energy_intra if solver.mdv.iswater else 0.0],
            # E_corr is stored with negative sign (subtractive correction).
            'E_corr': [solver.energy_corr],
            # 'DeltaG_nonpolar': [solver.energy_nonpolar],
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
    name = 'forces_tot'
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, solver):
        forces = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_fcs_tot(forces)
        df = pd.DataFrame(forces.sum(axis=0).reshape(1,3), columns=['Fx', 'Fy', 'Fz'])
        df['iter'] = iter
        return df

class ForceComponentsCSVOutputFile(CSVOutputFile):
    name = 'force_components'
    headers = [
        'iter',
        'Fx_elec', 'Fy_elec', 'Fz_elec',
        'Fx_noel', 'Fy_noel', 'Fz_noel',
        'Fx_intra', 'Fy_intra', 'Fz_intra',
        'Fx_corr', 'Fy_corr', 'Fz_corr',
    ]
    def get_data(self, iter: int, solver):
        forces = np.empty((solver.N_p, 3), dtype=np.float64)

        capi.get_fcs_elec(forces)
        f_elec = forces.sum(axis=0)

        capi.get_fcs_noel(forces)
        f_noel = forces.sum(axis=0)

        capi.get_fcs_intra(forces)
        f_intra = forces.sum(axis=0)

        capi.get_fcs_corr(forces)
        f_corr = forces.sum(axis=0)

        return pd.DataFrame({
            'iter': [iter],
            'Fx_elec': [f_elec[0]], 'Fy_elec': [f_elec[1]], 'Fz_elec': [f_elec[2]],
            'Fx_noel': [f_noel[0]], 'Fy_noel': [f_noel[1]], 'Fz_noel': [f_noel[2]],
            'Fx_intra': [f_intra[0]], 'Fy_intra': [f_intra[1]], 'Fz_intra': [f_intra[2]],
            'Fx_corr': [f_corr[0]], 'Fy_corr': [f_corr[1]], 'Fz_corr': [f_corr[2]],
        })

class ForceComponentsParticleCSVOutputFile(CSVOutputFile):
    name = 'force_components_particle'
    headers = [
        'iter', 'particle',
        'Fx_elec', 'Fy_elec', 'Fz_elec',
        'Fx_noel', 'Fy_noel', 'Fz_noel',
        'Fx_intra', 'Fy_intra', 'Fz_intra',
        'Fx_corr', 'Fy_corr', 'Fz_corr',
    ]
    def get_data(self, iter: int, solver):
        particle = solver.outset.force_components_particle
        if particle is None:
            particle = 0
        if particle < 0 or particle >= solver.N_p:
            raise ValueError(
                f"force_components_particle={particle} out of range (0..{solver.N_p - 1})"
            )

        forces = np.empty((solver.N_p, 3), dtype=np.float64)

        capi.get_fcs_elec(forces)
        f_elec = forces[particle].copy()

        capi.get_fcs_noel(forces)
        f_noel = forces[particle].copy()

        capi.get_fcs_intra(forces)
        f_intra = forces[particle].copy()

        capi.get_fcs_corr(forces)
        f_corr = forces[particle].copy()

        return pd.DataFrame({
            'iter': [iter],
            'particle': [particle],
            'Fx_elec': [f_elec[0]], 'Fy_elec': [f_elec[1]], 'Fz_elec': [f_elec[2]],
            'Fx_noel': [f_noel[0]], 'Fy_noel': [f_noel[1]], 'Fz_noel': [f_noel[2]],
            'Fx_intra': [f_intra[0]], 'Fy_intra': [f_intra[1]], 'Fz_intra': [f_intra[2]],
            'Fx_corr': [f_corr[0]], 'Fy_corr': [f_corr[1]], 'Fz_corr': [f_corr[2]],
        })

class ForcesPBoltzCSVOutputFile(CSVOutputFile):
    name = 'forces_pb'
    headers = [
        'iter', 'particle',
        'Fx_RF', 'Fy_RF', 'Fz_RF',
        'Fx_DB', 'Fy_DB', 'Fz_DB', 'Fx_IB', 'Fy_IB', 'Fz_IB', 'Fx_NP', 'Fy_NP', 'Fz_NP'
        ]
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()
        forces = np.empty((solver.N_p, 3), dtype=np.float64)

        capi.get_fcs_elec(forces)
        df[['Fx_RF', 'Fy_RF', 'Fz_RF']] = forces

        capi.get_fcs_db(forces)
        df[['Fx_DB', 'Fy_DB', 'Fz_DB']] = forces

        capi.get_fcs_ib(forces)
        df[['Fx_IB', 'Fy_IB', 'Fz_IB']] = forces

        capi.get_fcs_np(forces)
        df[['Fx_NP', 'Fy_NP', 'Fz_NP']] = forces

        df['iter'] = iter
        df['particle'] = range(solver.N_p)
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

        tmp = np.empty(solver.N_p, dtype=np.float64)
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
    headers = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()

        tmp = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_pos(tmp)
        df[['x', 'y', 'z']] = tmp * a0
        capi.get_vel(tmp)
        df[['vx', 'vy', 'vz']] = tmp

        tmp = np.empty(solver.N_p, dtype=np.int32)
        capi.get_types(tmp)
        df['type'] = [solver.types_num_to_str[t] for t in tmp]

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
        'force_components': ForceComponentsCSVOutputFile,
        'force_components_particle': ForceComponentsParticleCSVOutputFile,
        'forces_pb': ForcesPBoltzCSVOutputFile,
        'restart': RestartCSVOutputFile,
        'restart_field': RestartFieldCSVOutputFile
    }
)
