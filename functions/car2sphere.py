import numpy as np
from typing import Tuple

def car2sphere(x: float, y: float, z: float) -> Tuple[float, float, float]:
  """
  Transformation from cartesian coordinates to spherical coordinates

  Arguments:
    x: [km]
    y: [km]
    z: [km]

  Returns:
    azimuth: [rad]
    elevation: [rad]
    radius: [km]
  """
  hxy = np.hypot(x, y)
  radius = np.hypot(hxy, z)
  elevation = np.arctan2(z, hxy)
  azimuth = np.arctan2(y, x)
  return azimuth, elevation, radius