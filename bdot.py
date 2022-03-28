# B-Dot algorithm implementation for the detumbling of a spacecraft.

import numpy as np
from typing import List

SATURATION_RANGE = 0.131
TORQUE_THRESHOLD = 1e-2

def magnetic_moment(bB: List[float], dbB: List[float], k: float) -> List[float]:
  """
  Taking as input the external magnetic field, calculates the resulting magnetic moment
  using the B-dot algorithm. As a final step, saturation is added to the magnetic moment.

  Args:
    bB: External magnetic field in the body frame.
    dbB: External magnetic field derivative in the body frame.
    k: Gain constant.

  Returns:
    Magnetic moment in the body frame as a result of the magnetic field variation.
  """

  bB = np.array(bB)
  dbB = np.array(dbB)

  magneticMoment = np.divide(np.multiply(dbB, -k), (np.linalg.norm(bB)**2))

  if magneticMoment > SATURATION_RANGE:
    magneticMoment = SATURATION_RANGE
  elif magneticMoment < -SATURATION_RANGE:
    magneticMoment = -SATURATION_RANGE

  return magneticMoment.tolist()
  
def magnetorque(magneticMoment: List[float], bB: List[float], angularVelocity: List[float]) -> List[float]:
  """
  Calculates the resulting torque of the spacecraft due to the magnetic moment.
  If the spacecraft's angular velocity is below a threshold, the torque is set to zero.

  Args:
    magneticMoment: Magnetic moment in the body frame.
    bB: External magnetic field in the body frame.
    angularVelocity: Angular velocity of the spacecraft measured by the gyroscope.

  Returns:
    Torque in the body frame as a result of the magnetic moment to be passed to the magnetorquers.
  """

  magneticMoment = np.array(magneticMoment)
  bB = np.array(bB)
  angularVelocity = np.array(angularVelocity)

  if np.linalg.norm(angularVelocity) > TORQUE_THRESHOLD:
    torque = np.cross(magneticMoment, bB)
  else:
    torque = np.zeros(3)

  return torque.tolist()


def b_dot(bB: List[float], dbB: List[float], k: float, angularVelocity: List[float]) -> List[float]:
  """
  Calculates the resulting magnetic moment and torque of the spacecraft due to the magnetic field variation.
  If the spacecraft's angular velocity is below a threshold, the torque is set to zero.

  Args:
    bB: External magnetic field in the body frame.
    dbB: External magnetic field derivative in the body frame.
    k: Gain constant.
    angularVelocity: Angular velocity of the spacecraft measured by the gyroscope.

  Returns:
    Magnetic moment in the body frame as a result of the magnetic field variation.
    Torque in the body frame as a result of the magnetic moment to be passed to the magnetorquers.
  """

  magneticMoment = magnetic_moment(bB, dbB, k)
  torque = magnetorque(magneticMoment, bB, angularVelocity)

  return magneticMoment, torque