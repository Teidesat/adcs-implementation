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
  return relativeVelocityVector.T.conj() / attitudeMatrixError * 1000