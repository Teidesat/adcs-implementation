from ast import Return
import numpy as np
import constants as c
from scipy import integrate

def gyro_dynamics(stateVector: np.array, angularVelocity: np.array, time: float) -> tuple(np.array, np.array):
  """
  Defining the dynamics for the Gyro STIM 202 MODEL in a given instant

  Args:
    stateVector:            x
    angularVelocity:        w
    time:                   t
  
  Returns:
    stateVectorDerivative:  x_dot
    feedbackTorque:         M
  """

  # Gyro dynamics expressed as 1st order ODE
  stateVectorDerivative = np.array([
    stateVector[3], # va_x
    stateVector[4], # va_y
    stateVector[5], # va_z
    -(c.GYRO_CONSTANT / c.GYRO_RADIAL_INERTIA) * stateVector[3] - (c.GYRO_ELASTIC_COEF / c.GYRO_RADIAL_INERTIA) * 
      stateVector[0] - (c.GYRO_AXIAL_INERTIA / c.GYRO_RADIAL_INERTIA) * c.GYRO_ANGULAR_VELOCITY * angularVelocity[0],
    -(c.GYRO_CONSTANT / c.GYRO_RADIAL_INERTIA) * stateVector[4] - (c.GYRO_ELASTIC_COEF / c.GYRO_RADIAL_INERTIA) *
      stateVector[1] - (c.GYRO_AXIAL_INERTIA / c.GYRO_RADIAL_INERTIA) * c.GYRO_ANGULAR_VELOCITY * angularVelocity[1],
    -(c.GYRO_CONSTANT / c.GYRO_RADIAL_INERTIA) * stateVector[5] - (c.GYRO_ELASTIC_COEF / c.GYRO_RADIAL_INERTIA) *
      stateVector[2] - (c.GYRO_AXIAL_INERTIA / c.GYRO_RADIAL_INERTIA) * c.GYRO_ANGULAR_VELOCITY * angularVelocity[2]
  ]).T.conj()

  # Feedback torque applied to the gyros
  feedbackTorque = np.array([
    - c.GYRO_ELASTIC_COEF * stateVector[0] - c.GYRO_CONSTANT * stateVector[3],    # M_x
    - c.GYRO_ELASTIC_COEF * stateVector[1] - c.GYRO_CONSTANT * stateVector[4],    # M_y
    - c.GYRO_ELASTIC_COEF * stateVector[2] - c.GYRO_CONSTANT * stateVector[5]     # M_z
  ]).T.conj()

  return stateVectorDerivative, feedbackTorque

def filter(filteredAngularVelocity: np.array, estimatedAngularVelocity: np.array, time: float) -> np.array:
  """
  Defining the filter as a Linear Observer with euler equations.

  Args:
    filteredAngularVelocity:  w_filter
    estimatedAngularVelocity: w_est
    time:                     t

  Returns:
    filteredAngularVelocityDerivative:  w_filter
  """
  # Constants for the euler equations
  c1 = ((c.INERTIA_Y - c.INERTIA_Z) / c.INERTIA_X)
  c2 = ((c.INERTIA_Z - c.INERTIA_X) / c.INERTIA_Y)
  c3 = ((c.INERTIA_X - c.INERTIA_Y) / c.INERTIA_Z)

  # Euler equations
  return np.array([
    c1 * filteredAngularVelocity[1] * filteredAngularVelocity[2] + 
      c.OBSERVER_GAINS * (filteredAngularVelocity[0] - estimatedAngularVelocity[0]) / c.INERTIA_X,
    c2 * filteredAngularVelocity[0] * filteredAngularVelocity[2] +
      c.OBSERVER_GAINS * (filteredAngularVelocity[1] - estimatedAngularVelocity[1]) / c.INERTIA_Y,
    c3 * filteredAngularVelocity[1] * filteredAngularVelocity[0] +
      c.OBSERVER_GAINS * (filteredAngularVelocity[2] - estimatedAngularVelocity[2]) / c.INERTIA_Z
  ])

def gyro_model(angularVelocity: np.array) -> np.array:
  """
  Defining the model for the Gyro STIM 202

  Args:
    angularVelocity:        w

  Returns:
    gyroAngularVelocity:    w_gyro
  """
  _, feedbackTorque = integrate.odeint(gyro_dynamics, [np.zeros(6).T.conj(), angularVelocity], t = c.TIMESTEPS)

  # Recover angular velocity of spacecraft
  estimatedAngularVelocity = np.array([
    feedbackTorque[0] / (c.GYRO_AXIAL_INERTIA * c.GYRO_ANGULAR_VELOCITY),
    feedbackTorque[1] / (c.GYRO_AXIAL_INERTIA * c.GYRO_ANGULAR_VELOCITY),
    feedbackTorque[2] / (c.GYRO_AXIAL_INERTIA * c.GYRO_ANGULAR_VELOCITY)
  ])

  # Filter the angular velocity
  return integrate.odeint(filter, [c.INITIAL_ANGULAR_VELOCITY, estimatedAngularVelocity], t = c.TIMESTEPS)
