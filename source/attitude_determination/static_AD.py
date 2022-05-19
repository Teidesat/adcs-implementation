from source.attitude_determination.static_attitude_determination import static_attitude_determination
import constants
import numpy as np

def realSunVector(sunVector: np.array) -> np.array:
  """
  Calculates the real sun vector based on the accuracy of the sun sensor.

  Arguments:
    sunVector: [-]

  Returns:
    realSunVector: [-]
  """
  errorMatrix = np.array([[np.cos(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY), np.cos(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)+np.sin(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY), -np.cos(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY)+np.sin(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)],
                    [-np.sin(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY), -np.sin(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)+np.cos(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY), np.sin(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY)+np.cos(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY)],
                    [np.sin(constants.SUN_SENSOR_ACCURACY), -np.cos(constants.SUN_SENSOR_ACCURACY)*np.sin(constants.SUN_SENSOR_ACCURACY), np.cos(constants.SUN_SENSOR_ACCURACY)*np.cos(constants.SUN_SENSOR_ACCURACY)]])
  return np.dot(errorMatrix, sunVector)

def realMagneticVector(magneticVector: np.array) -> np.array:
  """
  Calculates the real magnetic vector based on the accuracy of the magnetometer.

  Arguments:
    magneticVector: [-]

  Returns:
    realMagneticVector: [-]
  """
  errorMatrix = np.array([[np.cos(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY), np.cos(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)+np.sin(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY), -np.cos(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY)+np.sin(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)],
                    [-np.sin(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY), -np.sin(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)+np.cos(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY), np.sin(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY)+np.cos(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY)],
                    [np.sin(constants.MAGNETOMETER_ACCURACY), -np.cos(constants.MAGNETOMETER_ACCURACY)*np.sin(constants.MAGNETOMETER_ACCURACY), np.cos(constants.MAGNETOMETER_ACCURACY)*np.cos(constants.MAGNETOMETER_ACCURACY)]])
  return np.dot(errorMatrix, magneticVector)

def static_AD(sunVector: np.array, sunSensorAttitude: np.array, magneticVector: np.array, magnetometerAttitude: np.array) -> np.array:
  """
  Calculates the attitude by using the sun sensor and magentometer data.
  
  Arguments:
    sunVector: [-]
    sunSensorAttitude: [-]
    magneticVector: [-]
    magnetometer: [-]
  
  Returns:
    attitudeMatrix: [-]
  """
  sunSensor = realSunVector(sunVector)
  magnetometer = realMagneticVector(magneticVector)

  return static_attitude_determination(sunSensor, sunSensorAttitude, magnetometerAttitude, magnetometer)