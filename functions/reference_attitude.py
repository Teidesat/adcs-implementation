import numpy as np

def reference_attitude(attitudeMatrix: np.array, meanVelocity: float, inclination: float, 
time: float, angularVelocity: np.array) -> tuple(np.array, np.array, np.array, np.array):
  """
  Reference attitude determination.

  Args:
    attitudeMatrix: [-]
    meanVelocity: [km/s]
    inclination (float): [rad/s]
    time: [s]
    angularVelocity: [rad/s]
  
  Returns:
    attitudeMatrixError: [-]
    idealTrackError: [-]
    attitudeLN: [-]
    angularVelocityError: [-]
  """

  attitudeLN = np.array([[np.cos(meanVelocity * time), np.sin(meanVelocity * time), 0],
                         [-np.sin(meanVelocity * time), np.cos(meanVelocity * time), 0],
                         [0, 0, 1]]) * \
               np.array([[1, 0, 0],
                         [0, np.cos(np.deg2rad(inclination)), np.sin(np.deg2rad(inclination))],
                         [0, -np.sin(np.deg2rad(inclination)), np.cos(np.deg2rad(inclination))]])
  attitudeMatrixError = attitudeMatrix * attitudeLN.T.conj()
  idealTrackError = np.trace(attitudeMatrixError)
  angularVelocityLN = np.array([0, 0, meanVelocity]) * \
                      np.array([[1, 0, 0],
                                [0, np.cos(np.deg2rad(inclination)), np.sin(np.deg2rad(inclination))],
                                [0, -np.sin(np.deg2rad(inclination)), np.cos(np.deg2rad(inclination))]])
  angularVelocityError = np.rad2deg(angularVelocity - attitudeMatrixError * angularVelocityLN.T.conj())

  return attitudeMatrixError, idealTrackError, attitudeLN, angularVelocityError