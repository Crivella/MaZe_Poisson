import numpy as np
import pandas as pd

from .constants import a0, conv_mass, kB
from .loggers import logger
from .output_md import generate_output_files
from .particles import Particles, g


### grid class to represent the grid and the fields operating on it ###
class Grid:
    def __init__(self, grid_setting, md_variables, output_settings):
        self.grid_setting = grid_setting
        self.md_variables = md_variables
        self.output_settings = output_settings
        self.debug = output_settings.debug

        self.N = grid_setting.N
        self.N_tot = grid_setting.N_tot
        self.N_p = grid_setting.N_p
        self.h = grid_setting.h
        self.L = grid_setting.L
        self.dt = md_variables.dt
        self.elec = md_variables.elec
        self.not_elec = md_variables.not_elec
        #self.kB = kB
        self.kBT = md_variables.kBT

        self.offset_update = np.array([
            [self.h/2, 0, 0],
            [self.h/2, 0, 0],
            [0, self.h/2, 0],
            [0, self.h/2, 0],
            [0, 0, self.h/2],
            [0, 0, self.h/2],
            [0, 0, 0]
            ])
        
        self.output_files = generate_output_files(self, md_variables)

        self.particles = [] # list of class instances of class particle
        
        if output_settings.restart == False: # if False then it starts from a good initial config (BCC lattice) - i.e from an input file.
            df = pd.read_csv(grid_setting.input_file) # from file
            print('START new simulation from file:' + grid_setting.input_file)

            self.particles = Particles(
                    self,
                    md_variables,
                    df['charge'],
                    df['mass'] * conv_mass, # mass given in amu and converted in au
                    np.array([df['x'], df['y'], df['z']]).T / a0
                    )
            self.particles.vel = np.random.normal(loc = 0.0, scale =  np.sqrt(self.kBT / self.particles.masses[:, np.newaxis]), size=(self.N_p, 3))
            
        else:
            df = pd.read_csv(grid_setting.restart_file)
            print('RESTART from file:' + grid_setting.restart_file)

            self.particles = Particles(
                self,
                md_variables,
                df['charge'],
                df['mass'] * conv_mass, # mass given in amu and converted in au
                np.array([df['x'], df['y'], df['z']]).T / a0
                )
            self.particles.vel = np.array([df['vx'], df['vy'], df['vz']]).T

        self.shape = (self.N,)*3
        self.q = np.zeros(self.shape, dtype=float)          # charge vector - q for every grid point
        self.phi = np.zeros(self.shape, dtype=float)          # electrostatic field updated with MaZe
        self.phi_prev = np.zeros(self.shape, dtype=float)     # electrostatic field for step t - 1 Verlet
        self.linked_cell = None
        self.energy = 0
        self.temperature = md_variables.T
        self.potential_notelec = 0

        # Grids for FFT
        d = 1
        # d = grid_setting.L_ang / grid_setting.N
        d = self.h
        freqs = np.fft.fftfreq(self.N, d=d) * 2 * np.pi
        freqs_r = np.fft.rfftfreq(self.N, d=d) * 2 * np.pi
        gx, gy, gz = np.meshgrid(freqs, freqs, freqs_r, indexing='ij')
        g2 = gx**2 + gy**2 + gz**2
        g2[0, 0, 0] = 1  # to avoid division by zero
        self.ig2_r = -1 / g2

        freqs = np.fft.fftfreq(self.N, d=d) * 2 * np.pi
        gx, gy, gz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
        g2 = gx**2 + gy**2 + gz**2
        g2[0, 0, 0] = 1  # to avoid division by zero
        self.ig2 = -1 / g2

        freqs2 = np.fft.fftfreq(self.N*2, d=d) * 2 * np.pi
        freqs2_r = np.fft.rfftfreq(self.N*2, d=d) * 2 * np.pi
        gx, gy, gz = np.meshgrid(freqs2, freqs2, freqs2_r, indexing='ij')
        g2 = gx**2 + gy**2 + gz**2
        g2[0, 0, 0] = 1  # to avoid division by zero
        self.ig2_r2 = -1 / g2
    
    def RescaleVelocities(self):
        init_vel_Na = np.zeros(3)
        new_vel_Na = np.zeros(3)
        init_vel_Cl = np.zeros(3)
        new_vel_Cl = np.zeros(3)
        
        init_vel_Na = np.sum(self.particles.vel[self.particles.charges == 1.], axis=0)
        init_vel_Cl = np.sum(self.particles.vel[self.particles.charges == -1.], axis=0)

        mi_vi2 = self.particles.masses * np.sum(self.particles.vel**2, axis=1)
        self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)

        print(f'Total initial vel:\nNa = {init_vel_Na} \nCl = {init_vel_Cl}\nOld T = {self.temperature}\n')
        
        self.particles.vel[self.particles.charges == 1.] -= 2 * init_vel_Na / self.N_p
        self.particles.vel[self.particles.charges == -1.] -= 2 * init_vel_Cl / self.N_p
        
        new_vel_Na = np.sum(self.particles.vel[self.particles.charges == 1.], axis=0)
        new_vel_Cl = np.sum(self.particles.vel[self.particles.charges == -1.], axis=0)

        mi_vi2 = self.particles.masses * np.sum(self.particles.vel**2, axis=1)
        self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)

        print(f'Total scaled vel: \nNa = {new_vel_Na} \nCl = {new_vel_Cl}\nNew T = {self.temperature}\n')
    
     
    def ComputeForcesLJBasic(self):
        pe = 0

        for p1 in range(self.N_p):
            for p2 in range(p1+1, self.N_p):
                pair_force, pair_potential = self.particles[p1].ComputeLJForcePotentialPair(self.particles[p2]) 
                self.particles[p1].force_notelec += pair_force
                self.particles[p2].force_notelec -= pair_force
                            
                pe += pair_potential
        self.potential_notelec = pe

    ### Linked Cell Method ###
    def ComputeForcesLJLinkedcell(self):
        for p in self.particles:   
            p.force_notelec = np.zeros(3)
            
        pe = 0
        side_cells = self.linked_cell.side_cell_num
        for IX in range(side_cells):
            for IY in range(side_cells):
                for IZ in range(side_cells):
                    central_cell_particles, neighbor_cells_particles = self.linked_cell.interacting_particles(IX, IY, IZ)
                    num_particles_central = len(central_cell_particles)
                    
                    for p1 in range(num_particles_central):
                        #Interaction between particles in central cell
                        for p2 in range(p1+1,num_particles_central):
                            pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeLJForcePotentialPair(self.particles[central_cell_particles[p2]]) 
                            self.particles[central_cell_particles[p1]].force_notelec += pair_force
                            self.particles[central_cell_particles[p2]].force_notelec -= pair_force
                            
                            pe += pair_potential
                    
                        #Interaction between particles in central cell and neighbors
                        if side_cells > 2:
                            for pn in neighbor_cells_particles:
                                pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeLJForcePotentialPair(self.particles[pn]) 
                                self.particles[central_cell_particles[p1]].force_notelec += pair_force
                                self.particles[pn].force_notelec -= pair_force
                                pe += pair_potential

        self.potential_notelec = pe
    

    '''
    def ComputeForcesTFBasic(self): # 
        self.particles.force_notelec = np.zeros((self.N_p, 3), dtype=float)
        pe = 0
        
        for p1 in range(self.N_p):
            for p2 in range(p1+1, self.N_p):
                pair_force, pair_potential = self.particles[p1].ComputeTFForcePotentialPair(self.particles[p2]) 
                self.particles[p1].force_notelec += pair_force
                self.particles[p2].force_notelec -= pair_force
                            
                pe += pair_potential
                
        self.potential_notelec = pe


    def ComputeForcesTFLinkedcell(self):
        for p in self.particles:
            p.force_notelec = np.zeros(3)

        pe = 0
        side_cells = self.linked_cell.side_cell_num
        
        for IX in range(side_cells):
            for IY in range(side_cells):
                for IZ in range(side_cells):
                    central_cell_particles, neighbor_cells_particles = self.linked_cell.interacting_particles(IX, IY, IZ)
                    num_particles_central = len(central_cell_particles)
                    
                    for p1 in range(num_particles_central):
                        #Interaction between particles in central cell
                        for p2 in range(p1+1,num_particles_central):
                            pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeTFForcePotentialPair(self.particles[central_cell_particles[p2]]) 
                            self.particles[central_cell_particles[p1]].force_notelec += pair_force
                            self.particles[central_cell_particles[p2]].force_notelec -= pair_force
                            
                            pe += pair_potential
                    
                        #Interaction between particles in central cell and neighbors
                        if side_cells > 2:
                            for pn in neighbor_cells_particles:
                                #print(central_cell_particles[p1],pn)
                                pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeTFForcePotentialPair(self.particles[pn]) 
                                self.particles[central_cell_particles[p1]].force_notelec += pair_force
                                self.particles[pn].force_notelec -= pair_force
                                pe += pair_potential

        self.potential_notelec = pe
    '''

    # update charges with a weight function that spreads it on the grid
    def SetCharges(self):
        L = self.L
        h = self.h
        self.q = np.zeros(self.shape, dtype=float)
        
        # for m in range(len(self.particles.charges)):
        #     for i, j, k in self.particles.neighbors[m, :, :]:
        #         diff = self.particles.pos[m] - np.array([i,j, k]) * h
        #         self.q[i, j, k] += self.particles.charges[m] * g(diff[0], L, h) * g(diff[1], L, h) * g(diff[2], L, h)

        # Same as above using broadcasting
        diff = self.particles.pos[:, np.newaxis, :] - self.particles.neighbors * h
        
        # if Python 3.11 or newer uncomment below and comment lines 217-219
        #self.q[*self.particles.neighbors.reshape(-1, 3).T] += (self.particles.charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        
        # Version that works for Python 3.8.15
        indices = tuple(self.particles.neighbors.reshape(-1, 3).T)
        updates = (self.particles.charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        self.q[indices] += updates
  
        q_tot_expected = np.sum(self.particles.charges)
        q_tot = np.sum(self.q)

        if q_tot + 1e-6 < q_tot_expected:
            logger.error('Error: change initial position, charge is not preserved: q_tot ='+str(q_tot))
            exit() # exits runinning otherwise it hangs the code
                
    # returns only kinetic energy and not electrostatic one
    def Energy(self, iter, print_energy):
        # kinetic E
        kinetic = 0.5 * np.sum(self.particles.masses * np.sum(self.particles.vel**2, axis=1))
        
        if print_energy:
            self.output_files.file_output_energy.write(str(iter) + ',' +  str(kinetic) + ',' + str(self.potential_notelec) + '\n')


    def Temperature(self, iter, print_temperature):
        mi_vi2 = self.particles.masses * np.sum(self.particles.vel**2, axis=1)
        self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)
        
        if print_temperature:
            self.output_files.file_output_temperature.write(str(iter) + ',' +  str(self.temperature) + '\n')
