import constants
import numpy as np
from typing import Tuple

def kep2car(semimajorAxis: float, eccentricity: float, inclination: float, raan: float, 
  argOfPeriapsis: float, trueAnomaly: float) -> Tuple[np.array, np.array]:
  """
  Transformation from Keplerian elements to cartesian coordinates

  Arguments:
    semimajorAxis: [km]
    eccentricity: [-]
    inclination: [rad]
    raan: [rad]
    argOfPeriapsis: [rad]
    trueAnomaly: [rad]

  Returns:
    positionVector: [km]
    velocityVector: [km/s]
  """

  # First rotation of RAAN about K
  raanRotation = np.array([[np.cos(raan), np.sin(raan), 0],
                          [-np.sin(raan), np.cos(raan), 0],
                          [0, 0, 1]])

  # Second rotation of inclination about I'
  inclinationRotation = np.array([[1, 0, 0],
                                  [0, np.cos(inclination), np.sin(inclination)],
                                  [0, -np.sin(inclination), np.cos(inclination)]])

  # Third rotation of argument of periapsis about K''
  
  argOfPeriapsisRotation = np.array([[np.cos(argOfPeriapsis), np.sin(argOfPeriapsis), 0],
                                    [-np.sin(argOfPeriapsis), np.cos(argOfPeriapsis), 0],
                                    [0, 0, 1]])

  # Position and velocity in perifocal frame
  semilatusRectum = semimajorAxis * (1 - eccentricity ** 2)
  perifocalPositionMagnitude = semilatusRectum / (1 + eccentricity * np.cos(trueAnomaly))
  perifocalPosition = np.array([perifocalPositionMagnitude * np.cos(trueAnomaly),
                                perifocalPositionMagnitude * np.sin(trueAnomaly),
                                0]).conj().T
  perifocalVelocity = np.array([-np.sqrt(constants.EARTH_STD_GRAV_PARAMETER / semilatusRectum) * np.sin(trueAnomaly),
                                np.sqrt(constants.EARTH_STD_GRAV_PARAMETER / semilatusRectum) * (eccentricity + np.cos(trueAnomaly)),
                                0]).conj().T

  # Transformation to cartesian coordinates
  transformationMatrix = np.array(np.dot(np.dot(raanRotation, inclinationRotation), argOfPeriapsisRotation))
  positionVector = np.dot(transformationMatrix.T.conj(), perifocalPosition)
  velocityVector = np.dot(transformationMatrix.T.conj(), perifocalVelocity)

  return positionVector, velocityVector

