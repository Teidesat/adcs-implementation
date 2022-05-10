import numpy as np

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
  