import numpy as np
from typing import Tuple

def tracking_error(externalMagneticField: np.array, velocityBody: np.array, 
estimatedAttitudeMatrix: np.array, attitudeLN: np.array) -> Tuple[np.array, np.array, np.array]:
  """
  Calculates the tracking error of the spacecraft.

  Arguments:
    externalMagneticField: [T]
    velocityBody: [m/s]
    estimatedAttitudeMatrix: [-]
    attitudeLN: [-]

  Returns:
    velocityPointingError: [deg]
    nadirPointingError: [deg]
    polePointingError: [deg]
  """
  attitudeBL = estimatedAttitudeMatrix * attitudeLN.T.conj()
  xAxis = np.dot(attitudeBL, np.array([1, 0, 0]).T.conj())

  nadirPointingError = np.rad2deg(np.arccos(np.dot(np.array([-1, 0, 0]), xAxis) / np.linalg.norm(xAxis)))
  polePointingError = np.rad2deg(np.arccos(np.dot(np.array([0, 0, 1]), externalMagneticField) / np.linalg.norm(externalMagneticField)))
  velocityPointingError = np.rad2deg(np.arccos(np.dot(np.array([0, 1, 0]), velocityBody) / np.linalg.norm(velocityBody)))

  return nadirPointingError, polePointingError, velocityPointingError