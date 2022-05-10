import numpy as np
import constants as c

def magnetic_disturbance(externalMagneticField: np.array) -> np.array:
  """
  Calculates the magnetic disturbance force on the spacecraft.

  Arguments:
    externalMagneticField: [T]

  Returns:
    magneticDisturbanceForce: [N]
  """

  return np.cross(c.MAGNETIC_DISTURBANCE, externalMagneticField)