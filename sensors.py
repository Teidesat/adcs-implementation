import numpy as np
import constants as c
from functions.magnetometer import magnetometer
from functions.sun_sensor import sun_sensor
from functions.gyro_model import gyro_model

def sensors(positionVector: np.array, attitudeMatrix: np.array, angularVelocity: np.array, 
time: float) -> tuple(np.array, np.array, np.array, np.array, np.array):
  """
  Returns the sensor readings for the given state.
  Sensors: the magnetometer, the sun sensor and the gyroscope.

  Args:
    positionVector: [m]
    attitudeMatrix: [-]
    angularVelocity: [rad/s]
    time: [s]

  Returns:
    sunVector: [-]
    sunAttitude: [-]
    externalMagneticField: [T]
    magnetometerAttitude: [-]
    gyroAngularVelocity: [rad/s]
  """
  externalMagneticField, magnetometerAttitude = magnetometer(time, positionVector, attitudeMatrix)
  sunVector, sunAttitude = sun_sensor(time, attitudeMatrix)
  gyroAngularVelocity = gyro_model(angularVelocity)

  return sunVector, sunAttitude, externalMagneticField, magnetometerAttitude, gyroAngularVelocity

