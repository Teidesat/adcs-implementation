import numpy as np
import constants as c
from source.get_radiation_pressure_force import get_radiation_pressure_force

SOLAR_CONSTANT = 1358
SOLAR_RADIATION_PRESSURE = SOLAR_CONSTANT / c.LIGHT_SPEED

def solar_radiation_pressure(sunVector: np.array, faceVectors: np.array, massDisplacement: np.array, area: np.array) -> np.array:
  """
  Calculates the resulting force from solar radiation pressure on each face of the spacecraft.

  Arguments:
    sunVector: [m]
    faceVectors: [m]
    massDisplacement: [m]
    area: [m^2]

  Returns:
    totalSolarRadiationPressureForce: [N]
  """
  
  return get_radiation_pressure_force(sunVector, SOLAR_RADIATION_PRESSURE, faceVectors, massDisplacement, area)