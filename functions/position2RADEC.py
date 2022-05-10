import constants
import numpy as np
from math import degrees

def position2RADEC(positionVector: np.array) -> tuple(float, float):
  """
  Calculates the right ascension (RA) and the declination (DEC) from
  the geocentric equatorial position vector.

  Arguments:
    positionVector: Geocentric equatorial position vector [km]

  Returns:
    rightAscension: [degrees]
    declination: [degrees]
  """
  # Direction cosines of the position vector
  l = positionVector[0] / np.linalg.norm(positionVector)
  m = positionVector[1] / np.linalg.norm(positionVector)
  n = positionVector[2] / np.linalg.norm(positionVector)

  declination = degrees(np.arcsin(n))
  if (m > 0):
    rightAscension = degrees(np.arccos(l / degrees(np.cos(declination))))
  else:
    rightAscension = 360 - degrees(np.arccos(l / degrees(np.cos(declination))))
  
  return rightAscension, declination