from dataclasses import dataclass
from collections import namedtuple
import numpy as np


XY = namedtuple('XY', ['x', 'y'])
XYZ = namedtuple('XYZ', ['x', 'y', 'z'])

@dataclass
class Volume:
    latitude: float
    longitude: float
    base_altitude: float
    box_min: XYZ
    box_max: XYZ

@dataclass
class Direction:
    inclination: float
    declination: float

@dataclass
class PhysicalModel:
    """
    Physical model for the Aurora reconstruction.
    """
    name: str
    altitude_bins: np.ndarray
    energy_bins: np.ndarray
    emission_matrix: np.ndarray
    mag_field_direction: Direction
    reconstruction_volume: Volume

    def __repr__(self):
        return ", ".join(["PhysicalModel=(",
            f"altitude_bins={type(self.altitude_bins)} {self.altitude_bins.shape}",
            f"energy_bins={type(self.energy_bins)} {self.energy_bins.shape}",
            f"emission_matrix={type(self.emission_matrix)} {self.emission_matrix.shape}",
            f"mag_field_direction={self.mag_field_direction}",
            f"volume={self.reconstruction_volume}",
        ")"])
    
    def __str__(self):
        return self.__repr__(self)