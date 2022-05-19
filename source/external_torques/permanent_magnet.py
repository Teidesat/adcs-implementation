import numpy as np
import constants as c
from typing import Tuple

def permanent_magnet(externalMagneticField: np.array) -> Tuple[np.array, float]:
  """
  Calculates the permanent magnet force on the spacecraft.

  Arguments:
    externalMagneticField: [T]

  Returns:
    permanentMagnetForce: [N]
    error: [degrees] Out for the moment
  """
  permanentMagnetForce = np.cross(c.MAGNETIZATION_VECTOR, externalMagneticField)
  #error = np.rad2deg(np.arcsin(np.linalg.norm(permanentMagnetForce) / np.linalg.norm(c.MAGNETIZATION_VECTOR) * np.linalg.norm(externalMagneticField)))
  return permanentMagnetForce
