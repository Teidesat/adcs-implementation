import numpy as np
import constants as c

def permanent_magnet(externalMagneticField: np.array) -> tuple(np.array, float):
  """
  Calculates the permanent magnet force on the spacecraft.

  Arguments:
    externalMagneticField: [T]

  Returns:
    permanentMagnetForce: [N]
    error: [degrees]
  """
  permanentMagnetForce = np.cross(c.MAGNETIZATION_VECTOR, externalMagneticField)
  error = np.rad2deg(np.arcsin(np.linalg.norm(permanentMagnetForce) / np.linalg.norm(c.MAGNETIC_FORCE) * np.linalg.norm(externalMagneticField)))
  return permanentMagnetForce, error
