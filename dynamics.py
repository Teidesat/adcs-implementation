import numpy as np
import constants as c
from scipy import integrate

def get_angular_velocity_dot(disturbance: np.array, positionVector: np.array, attitudeBL: np.array,
  inertia: np.array, angularVelocity: np.array, time: float) -> np.array:
  """
  Calculates the derivative of the angular velocity of the spacecraft in a given instant.

  Args:
    disturbance: [rad/s]
    positionVector: [m]
    attitudeBL: [-]
    inertia: [kg*m^2]
    angularVelocity: [rad/s]
    time: [s]

  Returns:
    angularVelocityDot: [rad/s]
  """

  # Constants for the angular velocity dot calculation
  c1 = (inertia[1] - inertia[2]) / inertia[0]
  c2 = (inertia[2] - inertia[0]) / inertia[1]
  c3 = (inertia[0] - inertia[1]) / inertia[2]

  angularAcceleration = attitudeBL * np.array([1, 0, 0]).T.conj()
  normalizedPosition = np.linalg.norm(positionVector, 2)

  return np.array([
    [c1 * angularVelocity[1] * angularVelocity[2] - 
      (c1 * 3 * c.UNIVERSAL_GRAV_CONSTANT * c.EARTH_MASS / normalizedPosition**3 * 
      angularAcceleration[2] * angularAcceleration[1]) +  disturbance[0] / inertia[0]],
    [c2 * angularVelocity[0] * angularVelocity[2] -
      (c2 * 3 * c.UNIVERSAL_GRAV_CONSTANT * c.EARTH_MASS / normalizedPosition**3 * 
      angularAcceleration[2] * angularAcceleration[0]) + disturbance[1] / inertia[1]],
    [c3 * angularVelocity[1] * angularVelocity[0] -
      (c3 * 3 * c.UNIVERSAL_GRAV_CONSTANT * c.EARTH_MASS / normalizedPosition**3 *
      angularAcceleration[0] * angularAcceleration[1]) + disturbance[2] / inertia[2]]])

def dynamics(disturbance: np.array, positionVector: np.array, attitudeBL: np.array, 
  angularVelocity: np.array) -> np.array:
  """
  Calculates the angular velocity of the spacecraft over time.

  Args:
    disturbance: [rad/s]
    positionVector: [m]
    attitudeBL: [-]
    angularVelocity: [rad/s]
  
  Returns:
    angularVelocity: [rad/s]
  """

  return integrate.odeint(get_angular_velocity_dot, [disturbance, positionVector, attitudeBL, c.INERTIA, angularVelocity], t = c.TIMESTEPS)
  