from functools import wraps
from math import sqrt

import numpy as np

from ..particles import Particles
from .base_integrator import BaseIntegrator


class VerletIntegrator(BaseIntegrator):
    @wraps(BaseIntegrator.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = None

    def part1(self, particles: Particles):
        """Update the position and velocity of the particles."""
        if self.temp is not None:
            targ = self.temp
            curr = particles.get_temperature()
            particles.vel *= sqrt(targ / curr)

        particles.vel += 0.5 * self.dt * particles.forces / particles.masses[:, np.newaxis]
        particles.pos += self.dt * particles.vel
        particles.pos %= self.L

    def part2(self, particles: Particles):
        """Update the velocity of the particles."""
        particles.vel += 0.5 * self.dt * particles.forces / particles.masses[:, np.newaxis]

    def init_thermostat(self, temp: float):
        """Initialize the thermostat.

        Args:
            temp (float): Target temperature.
        """
        self.temp = temp

    def stop_thermostat(self):
        """Stop the thermostat."""
        self.temp = None

