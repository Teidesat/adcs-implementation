import numpy as np
from scipy import integrate
import constants as c

def get_quaternion(normalizedQuaternion: np.array, time: float, angularVelocity: np.array) -> np.array:
  """
  Calculates the derivative of the quaternion of the spacecraft in a given instant.

  Args:
    angularVelocity: [rad/s] The angular velocity of the spacecraft.
    normalizedQuaternion: The normalized quaternion of the spacecraft.
    time: [s]

  Returns:
    The derivative quaternion of the spacecraft.
  """
  angVelocityQuaternion = np.array([
    [0, angularVelocity[2], -angularVelocity[1], angularVelocity[0]],
    [-angularVelocity[2], 0, angularVelocity[0], angularVelocity[1]],
    [angularVelocity[1], -angularVelocity[0], 0, angularVelocity[2]],
    [-angularVelocity[0], -angularVelocity[1], -angularVelocity[2], 0]])

  return 1/2 * np.dot(angVelocityQuaternion, normalizedQuaternion)

def quaternion_to_attitude(quaternion: np.array) -> np.array:
  """
  Calculates the attitude matrix of the spacecraft given its quaternion.
    
  Args:
    quaternion: [rad/s]

  Returns:
    attitudeMatrix: [-]
  """
  return np.array([
    [quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2, 
      2*(quaternion[0]*quaternion[1] + quaternion[2]*quaternion[3]), 
      2*(quaternion[0]*quaternion[2] - quaternion[1]*quaternion[3])],
    [2*(quaternion[0]*quaternion[1] - quaternion[2]*quaternion[3]), 
      -quaternion[0]**2 + quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2, 
      2*(quaternion[1]*quaternion[2] + quaternion[0]*quaternion[3])],
    [2*(quaternion[0]*quaternion[2] + quaternion[1]*quaternion[3]), 
      2*(quaternion[1]*quaternion[2] - quaternion[0]*quaternion[3]), 
      -quaternion[0]**2 - quaternion[1]**2 + quaternion[2]**2 + quaternion[3]**2]])


def kinematics(angularVelocity: np.array, time: float) -> np.array:
  """
  Integrates the quaternion of the spacecraft through time and returns the attitude matrix.
  Probably needs normalization.

  Args:
    angularVelocity: [rad/s]
    time: [s]
      
  Returns:
    attitudeMatrix: [-]
  """

  integratedQuaternion = integrate.odeint(get_quaternion, 
    y0 = c.INITIAL_ATTITUDE_QUATERNIONS, t = np.array([time, time + c.DELTA_TIME]),
    args = (angularVelocity,))[1]
  return quaternion_to_attitude(integratedQuaternion)
