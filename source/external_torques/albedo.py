import numpy as np
import constants as c
from source.get_radiation_pressure_force import get_radiation_pressure_force

ALBEDO_FORCE = 600
INFRARED_FORCE = 150
TOTAL_FORCE = ALBEDO_FORCE + INFRARED_FORCE
ALBEDO_RADIATION_PRESSURE = ALBEDO_FORCE / c.LIGHT_SPEED

def albedo(nadirVector: np.array, faceVectors: np.array, massDisplacement: np.array, area: np.array) -> np.array:
  """
  Calculates the total albedo force on each face of the spacecraft. 

  Arguments:
    nadirVector: [m]
    faceVectors: [m]
    massDisplacement: [m]
    area: [m^2]
      
  Returns:
    totalAlbedoForce: [N]
  """

  return get_radiation_pressure_force(nadirVector, ALBEDO_RADIATION_PRESSURE, faceVectors, massDisplacement, area)