from functools import wraps
from math import exp, sqrt, tanh

import numpy as np

from ..particles import Particles
from .base_integrator import BaseIntegrator


class OVRVOIntegrator(BaseIntegrator):
    @wraps(BaseIntegrator.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = 0

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        if self.gamma == 0:
            self.c1 = 1
            self.c2 = 1
        else:
            self.c1 = exp(-self.gamma * self.dt)
            self.c2 = sqrt(2 /(self.gamma * self.dt) * tanh(0.5 * self.gamma * self.dt))

    def part1(self, particles: Particles):
        """Update the position and velocity of the particles."""
        if self.gamma != 0:
            particles.vel = self.O_block(particles.vel, particles.masses)
        particles.vel = self.V_block(particles.vel, particles.forces, particles.masses)
        particles.pos = self.R_block(particles.pos, particles.vel)

    def part2(self, particles: Particles):
        """Update the velocity of the particles."""
        particles.vel = self.V_block(particles.vel, particles.forces, particles.masses)
        if self.gamma != 0:
            particles.vel = self.O_block(particles.vel, particles.masses)


    def O_block(self, vel: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Solves the equations for the O-block.

        Args:
            vel (np.ndarray): Velocities of the particles. Shape (N_p, 3).
            masses (np.ndarray): Masses of the particles. Shape (N_p,).

        Returns:
            np.ndarray: New velocities of the particles. Shape (N_p, 3).
        """
        c1 = self.c1
        rnd = np.random.multivariate_normal(
            mean=[0.0, 0.0, 0.0],
            cov=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            size=vel.shape[0]
        )

        return np.sqrt(c1) * vel + np.sqrt((1 - c1) * self.kBT / masses[:, np.newaxis]) * rnd
            
    def V_block(self, vel: np.ndarray, forces: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Solves the equations for the V-block.

        Args:
            vel (np.ndarray): The velocities of the particles. Shape (N_p, 3).
            forces (np.ndarray): The forces on the particles. Shape (N_p, 3).
            masses (np.ndarray): The masses of the particles. Shape (N_p,).

        Returns:
            np.ndarray: New velocities of the particles. Shape (N_p, 3).
        """
        return vel + 0.5 * self.c2 * self.dt * forces / masses[:, np.newaxis]

    def R_block(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Solves the equations for the R-block.

        Args:
            pos (np.ndarray): The positions of the particles. Shape (N_p, 3).
            vel (np.ndarray): The velocities of the particles. Shape (N_p, 3).

        Returns:
            np.ndarray: New positions of the particles. Shape (N_p, 3).
        """
        return (pos + self.c2 * self.dt * vel) % self.L

    def init_thermostat(self, gamma: float):
        """Initialize the thermostat.

        Args:
            gamma (float): OVRVO gamma parameter.
        """
        self.gamma = gamma

    def stop_thermostat(self):
        """Stop the thermostat."""
        self.gamma = 0
