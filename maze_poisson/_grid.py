import numpy as np
import pandas as pd

from . import c_api
from .constants import a0, conv_mass, kB
from .loggers import logger
from .output_md import generate_output_files

# from .particles import Particles, g


### grid class to represent the grid and the fields operating on it ###
class Grid:
    def __init__(self, grid_setting, md_variables, output_settings):
        self.grid_setting = grid_setting
        # self.md_variables = md_variables
        # self.output_settings = output_settings
        # self.debug = output_settings.debug

        self.N = grid_setting.N
        # self.N_tot = grid_setting.N_tot
        # self.N_p = grid_setting.N_p
        self.h = grid_setting.h
        self.L = grid_setting.L
        # self.dt = md_variables.dt
        # self.elec = md_variables.elec
        # self.not_elec = md_variables.not_elec
        #self.kB = kB
        # self.kBT = md_variables.kBT

        # self.offset_update = np.array([
        #     [self.h/2, 0, 0],
        #     [self.h/2, 0, 0],
        #     [0, self.h/2, 0],
        #     [0, self.h/2, 0],
        #     [0, 0, self.h/2],
        #     [0, 0, self.h/2],
        #     [0, 0, 0]
        #     ])
        
        # self.output_files = generate_output_files(self, md_variables)

        # if output_settings.restart == False: # if False then it starts from a good initial config (BCC lattice) - i.e from an input file.
        #     df = pd.read_csv(grid_setting.input_file) # from file
        #     print('START new simulation from file:' + grid_setting.input_file)

        #     self.particles = Particles(
        #             self,
        #             md_variables,
        #             df['charge'],
        #             df['mass'] * conv_mass, # mass given in amu and converted in au
        #             np.array([df['x'], df['y'], df['z']]).T / a0
        #             )
        #     self.particles.vel = np.random.normal(loc = 0.0, scale =  np.sqrt(self.kBT / self.particles.masses[:, np.newaxis]), size=(self.N_p, 3))
            
        # else:
        #     df = pd.read_csv(grid_setting.restart_file)
        #     print('RESTART from file:' + grid_setting.restart_file)

        #     self.particles = Particles(
        #         self,
        #         md_variables,
        #         df['charge'],
        #         df['mass'] * conv_mass, # mass given in amu and converted in au
        #         np.array([df['x'], df['y'], df['z']]).T / a0
        #         )
        #     self.particles.vel = np.array([df['vx'], df['vy'], df['vz']]).T

        self.shape = (self.N,)*3
        self.q = np.zeros(self.shape, dtype=float)          # charge vector - q for every grid point
        self.phi = np.zeros(self.shape, dtype=float)          # electrostatic field updated with MaZe
        self.phi_prev = np.zeros(self.shape, dtype=float)     # electrostatic field for step t - 1 Verlet
        self.shape2 = (self.N, self.N, self.N//2 + 1)
        self.phi_r = np.zeros(self.shape, dtype=np.float64)          # electrostatic field updated with MaZe
        self.phi_q = np.zeros(self.shape2, dtype=np.complex128)          # electrostatic field updated with MaZe
        # self.phi_prev_q = np.zeros(self.shape2, dtype=np.complex128)     # electrostatic field for step t - 1 Verlet
        self.linked_cell = None
        # self.energy = 0
        # self.temperature = md_variables.T
        self.potential_notelec = 0

        # Grids for FFT
        # d = self.h * a0  # in Angstrom
        d = self.h  # in a.u

        freqs = np.fft.fftfreq(self.N, d=d) * 2 * np.pi
        freqs_r = np.fft.rfftfreq(self.N, d=d) * 2 * np.pi
        gx, gy, gz = np.meshgrid(freqs, freqs, freqs_r, indexing='ij')
        g2 = gx**2 + gy**2 + gz**2
        g2[0, 0, 0] = 1  # to avoid division by zero

        self.q_const = 4 * np.pi / self.h**3
        # self.igx = 1j * gx
        # self.igy = 1j * gy
        # self.igz = 1j * gz
        self.ig2 = self.q_const / g2
        del g2, gx, gy, gz

        self.phi_q_updated = False
        self.phi_updated = False

    def calculate_phi_q(self):
        if not self.phi_q_updated:
            c_api.rfft_3d(self.N, self.q, self.phi_q)
            self.phi_q *= self.ig2
            self.phi_q_updated = True
            self.phi_updated = False
        return self.phi_q

    def calculate_phi(self):
        if not self.phi_updated:
            self.phi_updated = True
            c_api.irfft_3d(self.N, self.phi_q, self.phi_r)
        return self.phi_r
        # self.phi = self.fq
     
    # def ComputeForcesLJBasic(self):
    #     pe = 0

    #     for p1 in range(self.N_p):
    #         for p2 in range(p1+1, self.N_p):
    #             pair_force, pair_potential = self.particles[p1].ComputeLJForcePotentialPair(self.particles[p2]) 
    #             self.particles[p1].force_notelec += pair_force
    #             self.particles[p2].force_notelec -= pair_force
                            
    #             pe += pair_potential
    #     self.potential_notelec = pe

                
    # # returns only kinetic energy and not electrostatic one
    # def Energy(self, iter, print_energy):
    #     # kinetic E
    #     kinetic = 0.5 * np.sum(self.particles.masses * np.sum(self.particles.vel**2, axis=1))
        
    #     if print_energy:
    #         self.output_files.file_output_energy.write(str(iter) + ',' +  str(kinetic) + ',' + str(self.potential_notelec) + '\n')


    # def Temperature(self, iter, print_temperature):
    #     mi_vi2 = self.particles.masses * np.sum(self.particles.vel**2, axis=1)
    #     self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)
        
    #     if print_temperature:
    #         self.output_files.file_output_temperature.write(str(iter) + ',' +  str(self.temperature) + '\n')
