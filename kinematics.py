import numpy as np

def kynematics(angularVelocity: np.array) -> np.array:
  """
  Calculates the attitude matrix of the spacecraft given its angular velocity.
    
  Args:
    angularVelocity: [rad/s]

  Returns:
    attitudeMatrix: [-]
  """

  