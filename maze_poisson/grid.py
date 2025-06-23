import numpy as np
import pandas as pd

from .constants import a0, conv_mass, kB
from .loggers import logger
from .output_md import generate_output_files
from .particles import Particles, g, cubic_bspline, quadratic_bspline
from scipy.constants import epsilon_0, k as k_B, N_A, elementary_charge as e, physical_constants

import matplotlib.pyplot as plt
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
        self.method = md_variables.method
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
        else:
            df = pd.read_csv(grid_setting.restart_file)
            print('RESTART from file:' + grid_setting.restart_file)

        if self.method == 'Poisson MaZe':
            self.particles = Particles(
                    self,
                    md_variables,
                    df['charge'],
                    df['mass'] * conv_mass, # mass given in amu and converted in au
                    np.array([df['x'], df['y'], df['z']]).T / a0 # positions given in Angstrom and converted in au
                    )
        elif self.method == 'PB MaZe':
            self.particles = Particles(
                    self,
                    md_variables,
                    df['charge'],
                    df['mass'] * conv_mass,
                    np.array([df['x'], df['y'], df['z']]).T / a0, # positions given in Angstrom and converted in au
                    df['radius'] / a0 # radii given in Angstrom and converted in au
                    )
            
            if output_settings.restart == False:
                self.particles.vel = np.random.normal(loc = 0.0, scale =  np.sqrt(self.kBT / self.particles.masses[:, np.newaxis]), size=(self.N_p, 3))
            else:
                self.particles.vel = np.array([df['vx'], df['vy'], df['vz']]).T

        self.linked_cell = None
        self.energy = 0
        self.temperature = md_variables.T
        self.potential_notelec = 0

        self.shape = (self.N,)*3
        self.q = np.zeros(self.shape, dtype=float)          # charge vector - q for every grid point
        self.phi = np.zeros(self.shape, dtype=float)          # electrostatic field updated with MaZe
        self.phi_prev = np.zeros(self.shape, dtype=float)     # electrostatic field for step t - 1 Verlet
        
        if self.method == 'PB MaZe':
            self.eps_s = grid_setting.eps_s     # relative permittivity of the solvent
            self.I = grid_setting.I             # ionic strength
          
            # electrostatic potential in the solvent
            self.phi_s = np.zeros(self.shape, dtype=float)          # updated with MaZe
            self.phi_s_prev = np.zeros(self.shape, dtype=float)     # for step t - 1 Verlet
            
            # dielectric field components
            self.eps_x = self.eps_s * np.ones(self.shape, dtype=float) 
            self.eps_y = self.eps_s * np.ones(self.shape, dtype=float)
            self.eps_z = self.eps_s * np.ones(self.shape, dtype=float)

            # screening factor
            meter_to_bohr = physical_constants["Bohr radius"][0]  # 1 bohr in meters
            self.kbar2 = (8 * np.pi * N_A * e**2 * self.I * 1e3) / (self.eps_s * epsilon_0 * k_B * self.md_variables.T) * (meter_to_bohr**2)
            self.k2 = self.kbar2 * np.ones(self.shape, dtype=float) 

            # transition region
            self.w = grid_setting.w / a0 # smoothing width for the solvation radius
            self.H = {}
            self.H_ratio = {}
            self.H_mask = {}
            self.r_hat = {}

            # non-polar solvation
            self.non_polar = md_variables.non_polar
            self.energy_np = 0.0
        
    
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
        
        if self.grid_setting.cas == 'CIC':
            updates = (self.particles.charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        elif self.grid_setting.cas == 'B-Spline':
            updates = (self.particles.charges[:, np.newaxis] * np.prod(cubic_bspline(diff, L, h), axis=2)).flatten()
        elif self.grid_setting.cas == 'Quadratic-B-Spline':
            # w = quadratic_bspline(diff[0], L, h)
            # print("Weights:", np.sum(np.prod(w, axis=1)))
            updates = (self.particles.charges[:, np.newaxis] * np.prod(quadratic_bspline(diff, L, h), axis=2)).flatten()
        else:
            raise ValueError(f"Unknown grid setting: {self.grid_setting.cas}")
        
        self.q[indices] += updates
  
        q_tot_expected = np.sum(self.particles.charges)
        q_tot = np.sum(self.q)

        if q_tot + 1e-6 < q_tot_expected:
            logger.error('Error: change initial position, charge is not preserved: q_tot ='+str(q_tot))
            exit() # exits running otherwise it hangs the code
                
    # returns only kinetic energy and not electrostatic one
    def Energy(self, iter, print_energy, prev=False):
        # kinetic E
        kinetic = 0.5 * np.sum(self.particles.masses * np.sum(self.particles.vel**2, axis=1))
        
        if prev:
            energy_elec = 0.5 * np.sum(self.q * (self.phi_s_prev - self.phi_prev))
        else:
            energy_elec = 0.5 * np.sum(self.q * (self.phi_s - self.phi))

        if print_energy:
            if self.method == 'Poisson MaZe':
                self.output_files.file_output_energy.write(str(iter) + ',' +  str(kinetic) + ',' + str(self.potential_notelec) + '\n')
            elif self.method == 'PB MaZe':
                self.output_files.file_output_energy.write(str(iter) + ',' + str(kinetic) + ',' + str(self.potential_notelec) + ',' + str(energy_elec) + ',' + str(self.energy_np) + '\n')

    def Temperature(self, iter, print_temperature):
        mi_vi2 = self.particles.masses * np.sum(self.particles.vel**2, axis=1)
        self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)
        
        if print_temperature:
            self.output_files.file_output_temperature.write(str(iter) + ',' +  str(self.temperature) + '\n')

    # def ComputeMask(self, center_x, center_y, center_z):
    #     X, Y, Z = np.meshgrid(center_x, center_y, center_z, indexing='ij')   # (N,N,N)
    #     coords = np.stack([X, Y, Z], axis=0)[None]                            # shape (1, 3, N, N, N)
    #     pos_exp = self.particles.pos[:, :, None, None, None]                                # shape (Np, 3, 1, 1, 1)

    #     delta = coords - pos_exp                                             # shape (Np, 3, N, N, N)
    #     delta -= self.L * np.round(delta / self.L)                                     # apply PBC
    #     r2 = np.sum(delta**2, axis=1)                                        # (Np, N, N, N)
    #     return np.any(r2 <= self.particles.solvation_radii2[:, None, None, None], axis=0)            # (N, N, N)

    # def UpdateEpsAndK2(self):
    #     N = self.N
    #     h = self.h
       
    #     eps_s = self.eps_s
    #     kbar2 = self.kbar2

    #     # Coordinates of voxel centers
    #     centers = (np.arange(N) + 0.5) * h

    #     # k2 update
    #     mask_k2 = self.ComputeMask(centers, centers, centers)
    #     self.k2[...] = np.where(mask_k2, kbar2, 0.0)

    #     # eps_x at x + h/2
    #     x_face = np.arange(N) * h + h/2
    #     mask_x = self.ComputeMask(x_face, centers, centers)
    #     self.eps_x[...] = np.where(mask_x, 1.0, eps_s)

    #     # eps_y at y + h/2
    #     y_face = np.arange(N) * h + h/2
    #     mask_y = self.ComputeMask(centers, y_face, centers)
    #     self.eps_y[...] = np.where(mask_y, 1.0, eps_s)

    #     # eps_z at z + h/2
    #     z_face = np.arange(N) * h + h/2
    #     mask_z = self.ComputeMask(centers, centers, z_face)
    #     self.eps_z[...] = np.where(mask_z, 1.0, eps_s)
    
    def ComputeH(self):
        "Compute the H function and its derivatives for the transition region."
        N, h = self.N, self.h
        L = self.L
        w = self.w
        pos = self.particles.pos
        radius = self.particles.solvation_radii

        def make_H_and_rhat(coord_x, coord_y, coord_z):
            # Build meshgrid and shift by h/2 to place edges between grid nodes
            X, Y, Z = np.meshgrid(coord_x, coord_y, coord_z, indexing='ij')
            coords = np.stack([X, Y, Z], axis=0)[None]   # (1, 3, N, N, N)

            # Compute vector distances and apply PBC
            rvec = coords - pos[:, :, None, None, None]  # (Np, 3, N, N, N)
            rvec -= L * np.rint(rvec / L)
            r = np.linalg.norm(rvec, axis=1) + 1e-12  # (Np, N, N, N)

            r_hat = -rvec / r[:, None, :, :, :]  # (Np, 3, N, N, N)
            x = r - radius[:, None, None, None]  # (Np, N, N, N)

            H_per_particle = np.zeros_like(r)
            H_prime = np.zeros_like(r)

            if w == 0.0:
                H_per_particle = (r >= radius[:, None, None, None]).astype(np.float64)
            else:
                mask_inner = x <= -w
                mask_outer = x >= w
                mask_transition = (x > -w) & (x < w)
                x_t = x[mask_transition]

                H_per_particle[mask_inner] = 0.0
                H_per_particle[mask_outer] = 1.0
                H_per_particle[mask_transition] = (
                    -(1 / (4 * w**3)) * (x_t + w)**3 + (3 / (4 * w**2)) * (x_t + w)**2
                )
                H_prime[mask_transition] = (
                    (-3 / (4 * w**3)) * (x_t + w)**2 + (3 / (2 * w**2)) * (x_t + w)
                )

            H_total = np.prod(H_per_particle, axis=0)  # (N, N, N)

            # Per-particle H'/H only in transition region
            H_ratio = []
            H_mask = []
            for a in range(pos.shape[0]):
                mask_a = (H_per_particle[a] > 1e-12) & (H_per_particle[a] < 1 - 1e-12)
                ratio_a = np.zeros_like(H_per_particle[a])
                ratio_a[mask_a] = H_prime[a][mask_a] / H_per_particle[a][mask_a]
                H_ratio.append(ratio_a)
                H_mask.append(mask_a)

            return H_total, np.array(H_ratio), np.array(H_mask), r_hat

        coords = np.arange(N) * h               # i*h → nodi della griglia
        centers = (np.arange(N) + 0.5) * h      # (i+1/2)*h → centro delle celle
        self.H = {}
        self.H_ratio = {}
        self.H_mask = {}
        self.r_hat = {}

        grid_points = {
            'x': (centers, coords, coords),
            'y': (coords, centers, coords),
            'z': (coords, coords, centers)
        }

        for axis, (cx, cy, cz) in grid_points.items():
            H_total, H_ratio, H_mask, r_hat = make_H_and_rhat(cx, cy, cz)
            self.H[axis] = H_total
            self.H_ratio[axis] = H_ratio
            self.H_mask[axis] = H_mask
            self.r_hat[axis] = r_hat

        # H al centro dei nodi: serve per IB force (phi, k2 definiti sui nodi)
        Hc, Hrc, Hmc, r_hatc = make_H_and_rhat(coords, coords, coords)
        self.H['node'] = Hc
        self.H_ratio['node'] = Hrc
        self.H_mask['node'] = Hmc
        self.r_hat['node'] = r_hatc
        
        # line = self.H['node'][:, self.N // 2, self.N // 2]
        # plt.plot(np.arange(self.N) * self.h * a0, line, label='H', marker='.', linestyle='-')
        # # plt.plot(np.arange(self.N) * self.h * a0, line_p, label='H_p', marker='.', linestyle='-')
        # plt.xlabel('x [Å]')
        # plt.axvline(pos[0,0] * a0, label='$x_1$', color='k', linestyle='-')
        
        # plt.axvline(a0 * (pos[0,0] + radius[0] - w), color='r', label='$R_1$ - w')
        # plt.axvline(a0 * (pos[0,0] + radius[0] + w), color='orange', label='$R_1$ + w')
        # plt.axvline(a0 * (pos[0,0] - (radius[0] - w)), color='r')
        # plt.axvline(a0 * (pos[0,0] - (radius[0] + w)), color='orange')
        # plt.axvline(pos[1,0] * a0, label='$x_2$', color='k', linestyle='--')
        # plt.axvline(a0 * (pos[1,0] - (radius[1] - w)), color='r', linestyle = '--', label='$R_2$ - w')
        # plt.axvline(a0 * (pos[1,0] - (radius[1] + w)), color='orange', linestyle = '--', label='$R_2$ + w')
        # plt.axvline(a0 * (pos[1,0] + (radius[1] - w)), color='r', linestyle = '--')
        # plt.axvline(a0 * (pos[1,0] + (radius[1] + w)), color='orange', linestyle = '--')

        # plt.legend()
        # plt.xlim([0, 13.65])
        # plt.ylabel('H')
        # plt.title('H along x at y = z = center')
        # plt.grid()
        # plt.show()

    def UpdateEpsAndK2_transition(self):
        self.ComputeH()

        eps_s = self.eps_s
        kbar2 = self.kbar2

        self.eps_x[...] = 1. + (eps_s - 1.) * self.H['x']
        self.eps_y[...] = 1. + (eps_s - 1.) * self.H['y']
        self.eps_z[...] = 1. + (eps_s - 1.) * self.H['z']
        self.k2[...] = kbar2 * self.H['node']

    # def ComputeH(self, center_coords):
    #     pos = self.particles.pos
    #     L = self.L
    #     w = self.w
    #     radii = self.solvation_radii

    #     cx, cy, cz = center_coords
    #     X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')  # (N,N,N)
    #     coords = np.stack([X, Y, Z], axis=0)[None]        # (1, 3, N, N, N)
    #     pos_exp = pos[:, :, None, None, None]             # (Np, 3, 1, 1, 1)

    #     delta = coords - pos_exp                          # (Np, 3, N, N, N)
    #     delta -= L * np.round(delta / L)                  # Apply PBC
    #     r2 = np.sum(delta**2, axis=1)                     # (Np, N, N, N)
    #     H = H_smooth(r2, radii, w)                        # (Np, N, N, N)
    #     return np.clip(np.sum(H, axis=0), 0.0, 1.0)       # sum over particles

    # def UpdateEpsAndK2_transition(self):
    #     N = self.N
    #     h = self.h

    #     eps_s = self.eps_s
    #     kbar2 = self.kbar2

    #     # Coordinate dei centri voxel
    #     centers = (np.arange(N) + 0.5) * h
    #     faces   = np.arange(N) * h + h/2

    #     # k2: da 0 a kbar2
    #     Hk = self.ComputeH((centers, centers, centers))  # (N,N,N)
    #     self.k2[...] = Hk * kbar2

    #     # eps_x: da eps_s a 1
    #     Hx = self.ComputeH((faces, centers, centers))
    #     self.eps_x[...] = eps_s + (1.0 - eps_s) * Hx

    #     # eps_y
    #     Hy = self.ComputeH((centers, faces, centers))
    #     self.eps_y[...] = eps_s + (1.0 - eps_s) * Hy

    #     # eps_z
    #     Hz = self.ComputeH((centers, centers, faces))
    #     self.eps_z[...] = eps_s + (1.0 - eps_s) * Hz


# def H_smooth(r2, R, w):
#     """
#     Smoothed transition function H(r) as defined in Im et al.
#     r2: squared distances array, shape (Np, N, N, N)
#     R: array of solvation radii, shape (Np,)
#     w: smoothing width (scalar)
#     Returns: H, shape (Np, N, N, N)
#     """
#     r = np.sqrt(r2)
#     x = r - R[:, None, None, None] + w

#     H = np.zeros_like(r)
#     mask_mid = (r > (R - w)[:, None, None, None]) & (r < (R + w)[:, None, None, None])
#     mask_out = (r >= (R + w)[:, None, None, None])

#     H[mask_mid] = (
#         (1 / (4 * w**3)) * x[mask_mid]**3
#         - (3 / (4 * w**2)) * x[mask_mid]**2
#     )
#     H[mask_out] = 1.0
#     return H