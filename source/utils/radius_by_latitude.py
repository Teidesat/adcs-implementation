import numpy as np

def radius_by_latitude(equatorRadius: float, oblatenessRadius: float, latitude: float) -> float:
  """
  Gives the radius given the latitude

  Arguments:
    equatorRadius: [m]
    oblatenessRadius: [m]
    latitude: [rad]

  Returns:
    radius: [m]
  """
  poleRadius = (1 - oblatenessRadius) * equatorRadius
  radiusSquared = (((equatorRadius ** 2 * np.cos(latitude)) ** 2 + (poleRadius ** 2 * np.sin(latitude)) ** 2) /
    ((equatorRadius * np.cos(latitude)) ** 2 + (poleRadius * np.sin(latitude)) ** 2))
  return np.sqrt(radiusSquared)