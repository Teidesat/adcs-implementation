# B-Dot algorithm implementation for the detumbling of a spacecraft.

import numpy as np

SATURATION_RANGE = 0.131
TORQUE_THRESHOLD = 1e-2

def magnetic_moment(externalMagneticField: np.array, magneticFieldDerivative: np.array, k: float) -> np.array:
  """
  Taking as input the external magnetic field, calculates the resulting magnetic moment
  using the B-dot algorithm. As a final step, saturation is added to the magnetic moment.

  Args:
    externalMagneticField: External magnetic field in the body frame.
    magneticFieldDerivative: External magnetic field derivative in the body frame.
    k: Gain constant.

  Returns:
    Magnetic moment in the body frame as a result of the magnetic field variation.
  """

  magneticMoment = np.divide(np.multiply(magneticFieldDerivative, -k), (np.linalg.norm(externalMagneticField)**2))

  for i in range(len(magneticMoment)):
    if abs(magneticMoment[i]) > SATURATION_RANGE:
      magneticMoment[i] = np.sign(magneticMoment[i]) * SATURATION_RANGE

  return magneticMoment
  
def magnetorque(magneticMoment: np.array, externalMagneticField: np.array, angularVelocity: np.array) -> np.array:
  """
  Calculates the resulting torque of the spacecraft due to the magnetic moment.
  If the spacecraft's angular velocity is below a threshold, the torque is set to zero.

  Args:
    magneticMoment: Magnetic moment in the body frame.
    externalMagneticField: External magnetic field in the body frame.
    angularVelocity: Angular velocity of the spacecraft measured by the gyroscope.

  Returns:
    Torque in the body frame as a result of the magnetic moment to be passed to the magnetorquers.
  """

  magneticMoment = np.array(magneticMoment)
  externalMagneticField = np.array(externalMagneticField)
  angularVelocity = np.array(angularVelocity)

  if np.linalg.norm(angularVelocity) > TORQUE_THRESHOLD:
    torque = np.cross(magneticMoment, externalMagneticField)
  else:
    torque = np.zeros(3)

  return torque


def b_dot(externalMagneticField: np.array, magneticFieldDerivative: np.array, k: float, angularVelocity: np.array) -> np.array:
  """
  Calculates the resulting magnetic moment and torque of the spacecraft due to the magnetic field variation.
  If the spacecraft's angular velocity is below a threshold, the torque is set to zero.

  Args:
    externalMagneticField: External magnetic field in the body frame.
    magneticFieldDerivative: External magnetic field derivative in the body frame.
    k: Gain constant.
    angularVelocity: Angular velocity of the spacecraft measured by the gyroscope.

  Returns:
    Magnetic moment in the body frame as a result of the magnetic field variation.
    Torque in the body frame as a result of the magnetic moment to be passed to the magnetorquers.
  """

  magneticMoment = magnetic_moment(externalMagneticField, magneticFieldDerivative, k)
  torque = magnetorque(magneticMoment, externalMagneticField, angularVelocity)

  return magneticMoment, torque