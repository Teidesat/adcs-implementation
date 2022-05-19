import numpy as np
import constants as c

def aerodynamic(massDisplacement: np.array, faceVectors: np.array, atmosphericDensity: float, velocityBody: np.array, area: np.array) -> np.array:
  """
  Calculates the total aerodynamic force on the spacecraft.

  Arguments:
    massDisplacement: [m]
    faceVectors: [m]
    atmosphericDensity: [kg/m^3]
    velocityBody: [m/s]
    area: [m^2]

  Returns:
    totalAerodynamicForce: [N]
  """
  totalAerodynamicForce = np.zeros((1, 3))
  forceVector = np.zeros((3, c.NUMBER_OF_FACES))
  relativeVelocity = np.linalg.norm(velocityBody)
  for i in range(c.NUMBER_OF_FACES):
    if np.dot(velocityBody, faceVectors[:, i]) < 0:
      forceVector[:, i] = 0.5 * atmosphericDensity * c.DRAG_COEFFICIENT * area[i] * relativeVelocity * np.dot(velocityBody, faceVectors[:, i])
    else:
      forceVector[:, i] = 0
    totalAerodynamicForce += np.cross(massDisplacement[i], forceVector[:, i])
  return totalAerodynamicForce