import numpy as np
from typing import Tuple

def determination_error(attitudeMatrix: np.array, estimatedAttitudeMatrix: np.array, attitudeLN: np.array) -> Tuple[float, float]:
  """
  Calculates the error of the attitude determination.

  Args:
    attitudeMatrix: [-]
    estimatedAttitudeMatrix: [-]
    attitudeLN: [-]

  Returns:
    determinationError: [-]
    trackingError: [-]
  """
  determinationError = np.trace(attitudeMatrix - estimatedAttitudeMatrix)
  attitudeBL = estimatedAttitudeMatrix * attitudeLN.T.conj()

  BLQuaternions = np.zeros(4)
  BLQuaternions[3] = 1/2 * np.sqrt((1 + attitudeBL[0, 0] + attitudeBL[1, 1] + attitudeBL[2, 2]))
  BLQuaternions[0] = 1 / (4 * BLQuaternions[3]) * (attitudeBL[1, 2] - attitudeBL[2, 1])
  BLQuaternions[1] = 1 / (4 * BLQuaternions[3]) * (attitudeBL[2, 0] - attitudeBL[0, 2])
  BLQuaternions[2] = 1 / (4 * BLQuaternions[3]) * (attitudeBL[0, 1] - attitudeBL[1, 0])
  quaternionEye = np.array([1, 0, 0, 0])

  trackingError = np.sum(BLQuaternions - quaternionEye)

  return determinationError, trackingError