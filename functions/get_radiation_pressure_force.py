import numpy as np
import constants as c

def get_radiation_pressure_force(directionVector: np.array, pressureConstant: float, 
  faceVectors: np.array, massDisplacement: np.array, area: np.array) -> np.array:
  """
  Calculates the resulting force resulting from radiation pressure on each face of the spacecraft,
  based on its direction and magnitude.
  Will be used to calculate the solar radiation pressure force and the albedo radiation pressure force.

  Arguments:
    directionVector: [m] The direction vector of the radiation
    pressureConstant: [Pa]
    faceVectors: [m]
    massDisplacement: [m]
    area: [m^2]
  
  Returns:
    totalRadiationPressureForce: [N]
  """
  solarRadiationDensity = np.ones(c.NUMBER_OF_FACES) * 0.8
  dragDensity = np.ones(c.NUMBER_OF_FACES) * 0.1

  totalRadiationPressureForce = np.zeros((1, 3))
  forceVector = np.zeros((3, c.NUMBER_OF_FACES))

  for i in range(c.NUMBER_OF_FACES):
    if np.dot(directionVector, faceVectors[:, i]) > 0:
      forceVector[:, i] = - pressureConstant * area[i] * np.dot(directionVector, faceVectors[:, i]) * \
        ((1 - solarRadiationDensity[i]) * directionVector + (2 * solarRadiationDensity[i] * 
        np.dot(directionVector, faceVectors[:, i]) + 2/3 * dragDensity[i]) * faceVectors[:, i])
    else:
      forceVector[:, i] = 0
    totalRadiationPressureForce += np.cross(massDisplacement[i], forceVector[:, i])
  
  return totalRadiationPressureForce