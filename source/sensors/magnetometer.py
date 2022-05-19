import numpy as np
import constants as c

def magnetometer(time: float, positionVector: np.array, attitudeMatrix: np.array) -> np.array:
  """
  Gets the magnetic field vector of the spacecraft and the magnetometer attitude.

  Args:
    time: [s]
    positionVector: [m]
    attitudeMatrix: [-]

  Returns:
    externalMagneticField: [T]
    magnetometerAttitude: [-]
  """
  magneticFieldVector = np.array([
    np.sin(11.5)*np.cos(c.EARTH_ANGULAR_VELOCITY * time),
    np.sin(11.5)*np.sin(c.EARTH_ANGULAR_VELOCITY * time),
    np.cos(11.5)
  ])

  g10 = -29615e-9
  g11 = -1728e-9
  h11 = 5186e-9
  h0 = np.sqrt(g10**2 + g11**2 + h11**2)
  
  position = np.linalg.norm(positionVector)
  positionDirection = positionVector / position
  magnetometerAttitude = (c.EARTH_RADIUS * 1000)**3 * h0 / position**3 *(3 * np.cross(magneticFieldVector, 
    positionDirection) * positionDirection - magneticFieldVector)
  externalMagneticField = np.dot(attitudeMatrix, magnetometerAttitude)

  return externalMagneticField, magnetometerAttitude