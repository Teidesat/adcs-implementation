import numpy as np
from typing import Tuple

def earth_position(positionVector: np.array, attitudeMatrix: np.array) -> Tuple[np.array, np.array]:
  """
  Gives the earth position given the position vector and attitude matrix.
  
  Arguments:
    positionVector: [m]
    attitudeMatrix: [rad]
  
  Returns:
    northVector: [m]
    nadirVector: [m]
  """
  bodyFramePosition = np.dot(attitudeMatrix, positionVector)
  northVector = -positionVector / np.linalg.norm(positionVector)
  nadirVector = -bodyFramePosition / np.linalg.norm(bodyFramePosition)

  return northVector, nadirVector