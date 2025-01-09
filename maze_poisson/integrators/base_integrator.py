from abc import ABC, abstractmethod

from ..particles import Particles


class BaseIntegrator(ABC):
    """Base class for all 2-step MD integrators."""

    def __init__(self, dt: float, kBT: float, L: float):
        """Initializes the integrator.

        Args:
            dt (float): Time step.
            kBT (float): Temperature in energy units.
            L (float): Box size.
        """
        self.dt = dt
        self.kBT = kBT
        self.L = L
    
    @abstractmethod
    def part1(self, particles: Particles):
        """Update the position and velocity of the particles."""

    @abstractmethod
    def part2(self, particles: Particles):
        """Update the velocity of the particles."""

    @abstractmethod
    def init_thermostat(self):
        """Stop the thermostat."""

    @abstractmethod
    def stop_thermostat(self):
        """Stop the thermostat."""