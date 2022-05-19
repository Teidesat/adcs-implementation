import numpy as np

def velocity_body(relativeVelocityVector: np.array, attitudeMatrixError: np.array) -> np.array:
  """
  Calculates the velocity of the spacecraft in the body frame.

  Arguments:
    relativeVelocityVector: [m/s]
    attitudeMatrixError: [-]

  Returns:
    velocityBody: [m/s]
"""
  # Same as matlab's velocityBody = relativeVelocityVector'/(attitudeMatrixError)*1000:
  velocityBody = np.dot(relativeVelocityVector.T.conj(), np.linalg.pinv(attitudeMatrixError)) * 1000
  return velocityBody