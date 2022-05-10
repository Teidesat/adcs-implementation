import constants
import numpy as np

def car2kep(positionVector: np.array, velocityVector: np.array) -> tuple(float, float, float, float, float, float):
  """
  Transformation from cartesian coordinates to Keplerian elements

  Arguments:
    positionVector: [km]
    velocityVector: [km/s]
      
  Returns:
    semimajorAxis: [km]
    eccentricity: [-]
    inclination: [rad]
    raan: [rad]
    argOfPeriapsis: [rad]
    trueAnomaly: [rad]
  """
  position = np.linalg.norm(positionVector)
  velocity = np.linalg.norm(velocityVector)

  specificOrbitalEnergy = (velocity ** 2 / 2) - (constants.EARTH_STD_GRAV_PARAMETER / position)
  semimajorAxis = -(constants.EARTH_STD_GRAV_PARAMETER / (2 * specificOrbitalEnergy))

  eccentricityVector = (1 / constants.EARTH_STD_GRAV_PARAMETER) * ((velocity ** 2 - 
    constants.EARTH_STD_GRAV_PARAMETER / position) * positionVector - np.dot(positionVector, velocityVector) * velocityVector)
  eccentricity = np.linalg.norm(eccentricityVector)

  angularMomentumVector = np.cross(positionVector, velocityVector)
  angularMomentum = np.linalg.norm(angularMomentumVector)

  nodesLineVector = np.cross(angularMomentumVector, np.array([0, 0, 1]))
  nodesLine = np.linalg.norm(nodesLineVector)

  inclination = np.arccos(angularMomentumVector[2] / angularMomentum)

  raan = np.arccos(nodesLineVector[0] / nodesLine)
  if nodesLineVector[1] < 0:
    raan = 2 * np.pi - raan

  argOfPeriapsis = np.arccos(np.dot(nodesLineVector, eccentricityVector) / (nodesLine * eccentricity))
  if (eccentricityVector[2] < 0):
    argOfPeriapsis = 2 * np.pi - argOfPeriapsis
  
  radialVelocity = np.dot(positionVector, velocityVector) / position

  trueAnomaly = np.arccos(np.dot(eccentricityVector, positionVector) / (eccentricity * position))
  if (radialVelocity < 0):
    trueAnomaly = 2 * np.pi - trueAnomaly

  return semimajorAxis, eccentricity, inclination, raan, argOfPeriapsis, trueAnomaly

