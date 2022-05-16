import numpy as np
import constants as c

def sun_sensor(time: float, attitudeMatrix: np.array) -> np.array:
  """
  Gets the sun vector and the sun sensor attitude.

  Args:
    time: [s]
    attitudeMatrix: [-]

  Returns:
    sunVector: [-]
    sunAttitude: [-]
  """
  N = 2 * np.pi / c.EARTH_ORBITAL_PERIOD
  EPSILON = 23.45

  sunAttitude = np.array([
    np.cos(N * time),
    np.sin(N * time) * np.cos(np.deg2rad(EPSILON)),
    np.sin(N * time) * np.sin(np.deg2rad(EPSILON))
  ]).T.conj()
  sunVector = attitudeMatrix * sunAttitude

  return sunVector, sunAttitude
