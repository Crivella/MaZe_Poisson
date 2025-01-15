from itertools import product

import numpy as np
import pandas as pd

from .c_api import c_compute_tf_forces
from .constants import a0, conv_mass, kB
from .indices import GetDictTF
from .input import GridSetting
from .loggers import Logger


class Particles(Logger):
    # def __init__(self, grid, md_variables, charges, masses, positions):
    def __init__(
            self, gset: GridSetting, pot: str, charges: np.ndarray, masses: np.ndarray, positions: np.ndarray
        ):
        super().__init__()
        self.N = gset.N
        self.h = gset.h
        self.L = gset.L
        N_p = self.N_p = gset.N_p

        # Particle properties as NumPy arrays
        self.masses = np.array(masses)        # Shape: (N_p,)
        self.pos = np.array(positions) # Shape: (N_p, 3)
        self.vel = np.zeros((N_p, 3), dtype=float) # Shape: (N_p, 3)
        self.charges = np.array(charges)     # Shape: (N_p,)
        self.forces = np.zeros((N_p, 3), dtype=float) # Electrostatic forces
        self.forces_elec = np.zeros((N_p, 3), dtype=float) # Electrostatic forces
        self.forces_notelec = np.zeros((N_p, 3), dtype=float) # Non-electrostatic forces
        self.potential_info = pot

        # Additional attributes based on grid potential
        self.init_potential(pot)

        # Pre-allocate for nearest neighbor indices
        self.neigh_diff = np.array(list(product([0, 1], repeat=3)))  # Shape: (8, 3)
        self.neighbors = np.empty((N_p, 8, 3), dtype=int)  # Shape: (N_p, 8, 3)

    @classmethod
    def from_file(cls, path: str, gset: GridSetting, pot: str, kBT: float = None) -> 'Particles':
        df = pd.read_csv(path)
        charges = df['charge']
        mass = df['mass'] * conv_mass
        pos = df[['x', 'y', 'z']].values / a0
        particles = cls(
            gset,
            pot,
            charges=charges,
            masses=mass,
            positions=pos
            )
        particles.logger.info(f"Loaded starting positions from file: {path}")
        if 'vx' in df.columns:
            particles.logger.info("Loading starting velocities from file.")
            particles.vel = df[['vx', 'vy', 'vz']].values
        else:
            if kBT is None:
                raise ValueError("kBT must be provided to generate random velocities.")
            particles.logger.info("Generating random velocities.")
            particles.vel = np.random.normal(
                loc = 0.0,
                scale = np.sqrt(kBT / particles.masses[:, np.newaxis]),
                size=(len(df), 3)
            )

        return particles


    def init_potential(self, potential: str):
        self.potential_info = potential
        if potential == 'TF':
            self._init_potential_tf()
        elif potential == 'LJ':
            self._init_potential_lj()
        else:
            raise ValueError(f"Invalid potential: {potential}")

    def _init_potential_tf(self):
        """Initialize the Tosi-Fumi potential parameters."""
        self.r_cutoff = 0.5 * self.L  # Limit to where the force acts
        self.B = 3.1546 * a0  # Parameter for TF potential
        self.ComputeForceNotElec = self.ComputeTFForces  # Set the function to compute non-electrostatic forces

        # Compute pairwise TF parameters
        tf_params = GetDictTF()
        charges_sum = self.charges[:, np.newaxis] + self.charges[np.newaxis, :]  # Shape: (N_p, N_p)
        A = np.vectorize(lambda q: tf_params[q][0])(charges_sum)
        C = np.vectorize(lambda q: tf_params[q][1])(charges_sum)
        D = np.vectorize(lambda q: tf_params[q][2])(charges_sum)
        sigma_TF = np.vectorize(lambda q: tf_params[q][3])(charges_sum)

        V_shift = A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - C / self.r_cutoff**6 - D / self.r_cutoff**8
        alpha = A * self.B * np.exp(self.B * (sigma_TF - self.r_cutoff)) - 6 * C / self.r_cutoff**7 - 8 * D / self.r_cutoff**9
        beta = - V_shift - alpha * self.r_cutoff

        self.tf_params = np.array([A, C, D, sigma_TF, alpha, beta])

    def _init_potential_lj(self):
        """Initialize the Lennard-Jones potential parameters."""
        self.sigma = 3.00512 * 2 / a0  # Lennard-Jones sigma
        self.epsilon = 5.48 * 1e-4  # Lennard-Jones epsilon (Hartree)
        self.r_cutoff = 2.5 * self.sigma
        self.ComputeForceNotElec = self.ComputeLJForce

    def get_nearest_neighbors(self):
        """Compute the nearest neighbors indeces on the grid for each particle.

        Returns:
            np.ndarray: The array of nearest neighbor indices for each particle [N_p, 8, 3].
        """
        indices = np.floor(self.pos / self.h).astype(int)  # Shape: (N_p, 3)
        self.neighbors = np.ascontiguousarray(indices[:, np.newaxis, :] + self.neigh_diff) % self.N
        return self.neighbors

    def ComputeTFForces(self) -> float:
        # Get all pairwise differences
        return c_compute_tf_forces(self.N_p, self.L, self.pos, self.B, self.tf_params, self.r_cutoff, self.forces_notelec)

    def ComputeLJForce(self, grid):
        raise NotImplementedError("Lennard-Jones forces not implemented yet")

    # def ComputeForce(self, grid, prev):
    #     L = grid.L
    #     h = grid.h

    #     # Select the correct potential
    #     phi_v = grid.phi_prev if prev else grid.phi

    #     # Convert neighbor indices to coordinates
    #     neigh_coords = np.array(self.neigh) * h  # Shape: (num_neighbors, 3)

    #     # Compute vector differences: r_alpha - r_i for all neighbors
    #     diffs = self.pos - neigh_coords  # Shape: (num_neighbors, 3)

    #     # Compute g and g_prime for all components
    #     g_x = g(diffs[:, 0], L, h)
    #     g_y = g(diffs[:, 1], L, h)
    #     g_z = g(diffs[:, 2], L, h)

    #     g_prime_x = g_prime(diffs[:, 0], L, h)
    #     g_prime_y = g_prime(diffs[:, 1], L, h)
    #     g_prime_z = g_prime(diffs[:, 2], L, h)

    #     # Extract the phi values for all neighbors
    #     phi_values = phi_v[tuple(np.array(self.neigh).T)]  # Shape: (num_neighbors,)

    #     # Compute force contributions for each component
    #     self.force = -self.charge * np.array([
    #         np.sum(phi_values * g_prime_x * g_y * g_z),
    #         np.sum(phi_values * g_x * g_prime_y * g_z),
    #         np.sum(phi_values * g_x * g_y * g_prime_z)
    #     ])

    def get_temperature(self):
        mi_vi2 = self.masses * np.sum(self.vel**2, axis=1)
        self.temperature = np.sum(mi_vi2) / (3 * self.N_p * kB)
        return self.temperature

    def get_kinetic_energy(self):
        self.kinetic = 0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1))
        return self.kinetic

    def rescale_velocities(self):
        init_vel_Na = np.zeros(3)
        new_vel_Na = np.zeros(3)
        init_vel_Cl = np.zeros(3)
        new_vel_Cl = np.zeros(3)
        
        init_vel_Na = np.sum(self.vel[self.charges == 1.], axis=0)
        init_vel_Cl = np.sum(self.vel[self.charges == -1.], axis=0)

        self.get_temperature()

        # print(f'Total initial vel:\nNa = {init_vel_Na} \nCl = {init_vel_Cl}\nOld T = {self.temperature}\n')
        
        self.vel[self.charges == 1.] -= 2 * init_vel_Na / self.N_p
        self.vel[self.charges == -1.] -= 2 * init_vel_Cl / self.N_p
        
        new_vel_Na = np.sum(self.vel[self.charges == 1.], axis=0)
        new_vel_Cl = np.sum(self.vel[self.charges == -1.], axis=0)

        self.get_temperature()

        # print(f'Total scaled vel: \nNa = {new_vel_Na} \nCl = {new_vel_Cl}\nNew T = {self.temperature}\n')

        return self.temperature


  
# distance with periodic boundary conditions
# returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScaleDistance(diff, L): 
    diff = diff - L * np.rint(diff / L)
    distance = np.linalg.norm(diff)
    return distance
    
# returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScaleDistance2(diff, L): 
    diff = diff - L * np.rint(diff / L)
    distance = np.linalg.norm(diff)
    return diff, distance

# returns a vector - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScale(diff, L): 
    diff = diff - L * np.rint(diff / L)
    return diff

# weight function as defined in the paper Im et al. (1998) - eqn 24
def g(x, L, h):
    x = x - L * np.rint(x / L)
    x = np.abs(x)
    
    # Use vectorized operations with NumPy
    result = np.where(x < h, 1 - x / h, 0)
    return result


# derivative of the weight function as defined in the paper Im et al. (1998) - eqn 27
def g_prime(x, L, h):
    x = x - L * np.rint(x / L)
    if x < 0:
        return 1 / h
    elif x == 0:
        return 0
    else:
        return - 1 / h
    
def LJPotential(r, epsilon, sigma):  
        V_mag = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        return V_mag

