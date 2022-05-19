import numpy as np
import constants as c
from typing import Tuple

def geometry(velocityBody: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
  """
  Calculates all of the geometric parameters that are needed to calculate the torques.

  Arguments:
    velocityBody: [m/s]

  Returns:
    faceVectors: [m]
    massDisplacement: [m] Distance of the Geometric center of each face to the center of mass
    area: [m^2]
    areaCross: [m^2]
  """
  faceVectors = np.zeros((3, c.NUMBER_OF_FACES))

  faceVectors[:, 0] = np.array([1, 0, 0])  # (-X) Azimuth pointing, solar panel 
  faceVectors[:, 1] = np.array([0, 1, 0])  # (+Y) Velocity pointing
  faceVectors[:, 2] = np.array([0, 0, 1])  # (+Z) Pole pointing
  faceVectors[:, 3] = -faceVectors[:, 0]      # (+X) Nadir pointing
  faceVectors[:, 4] = -faceVectors[:, 1]      # (-Y) Drag pointing
  faceVectors[:, 5] = -faceVectors[:, 2]      # (-Z)

  massDisplacement = np.array([[-c.SATELLITE_LENGTH/2, -c.CENTER_OF_MASS_DISPLACEMENT, 0],
                               [0, -(c.SATELLITE_LENGTH/2 + c.CENTER_OF_MASS_DISPLACEMENT), 0],
                               [0, -c.CENTER_OF_MASS_DISPLACEMENT, -c.SATELLITE_LENGTH/2],
                               [c.SATELLITE_LENGTH/2, -c.CENTER_OF_MASS_DISPLACEMENT, 0],
                               [0, c.SATELLITE_LENGTH/2 - c.CENTER_OF_MASS_DISPLACEMENT, 0],
                               [0, -c.CENTER_OF_MASS_DISPLACEMENT, c.SATELLITE_LENGTH/2]])

  area = np.ones(c.NUMBER_OF_FACES) * 0.01
  areaCross = 0
  for i in range(c.NUMBER_OF_FACES):
    if np.dot(velocityBody, faceVectors[:,i]) < 0:
      areaCross = areaCross - area[i] * velocityBody / np.linalg.norm(velocityBody) * faceVectors[:,i]

  return faceVectors, massDisplacement, area, areaCross
