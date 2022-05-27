import numpy as np
import constants as c
from pyatmos import expo
from source.utils.kep2car import kep2car
from source.utils.car2sphere import car2sphere
from source.utils.radius_by_latitude import radius_by_latitude
from scipy import integrate
from typing import Tuple

def orbital_parameters(keplerianParameters: np.array) -> \
  np.array([float, float, float, float, float, float]):
  """
  Determination of the orbital parameters from Keplerian elements in a given instant.
  These include the variables that are later needed for the derivation and for the
  output of the Orbit Determination Block.

  Args:
    keplerianParameters (np.array): {
      semimajorAxis: [km]
      eccentricity: [-]
      inclination: [rad]
      raan: [rad]
      argOfPeriapsis: [rad]
      trueAnomaly: [rad]
    }
  
  Returns:
    perturbationAcceleration
    orbitRadius
    velocity
    argOfLatitude
    specificAngularMomentum
    positionVector
    relativeVelocityVector
    meanVelocity
    atmosphereDensity

  """
  # Keplerian parameters
  semimajorAxis = keplerianParameters[0]
  eccentricity = keplerianParameters[1]
  inclination = keplerianParameters[2]
  raan = keplerianParameters[3]
  argOfPeriapsis = keplerianParameters[4]
  trueAnomaly = keplerianParameters[5]

  semiminorAxis = semimajorAxis * np.sqrt(1 - eccentricity ** 2)
  semilatusRectum = semiminorAxis ** 2 / semimajorAxis
  meanVelocity = np.sqrt(c.EARTH_STD_GRAV_PARAMETER / semimajorAxis ** 3)
  specificAngularMomentum = meanVelocity * semimajorAxis * semiminorAxis
  orbitRadius = semilatusRectum / (1 + eccentricity * np.cos(trueAnomaly))
  velocity = np.sqrt(2* c.EARTH_STD_GRAV_PARAMETER / orbitRadius - 
    c.EARTH_STD_GRAV_PARAMETER / semimajorAxis)
  argOfLatitude = trueAnomaly + argOfPeriapsis

  # Perturbation acceleration
  positionVector, velocityVector = kep2car(semimajorAxis, eccentricity,
    inclination, raan, argOfPeriapsis, trueAnomaly)
  j2PerturbationAccelerationConstant = ((3/2) * c.J2_PERTURBATION * 
    c.EARTH_STD_GRAV_PARAMETER * c.EARTH_RADIUS ** 2 / orbitRadius ** 4)
  j2PerturbationAcceleration = j2PerturbationAccelerationConstant * np.array([
    positionVector[0] / orbitRadius * (5 * (positionVector[2] / orbitRadius) ** 2 - 1),
    positionVector[1] / orbitRadius * (5 * (positionVector[2] / orbitRadius) ** 2 - 1),
    positionVector[2] / orbitRadius * (5 * (positionVector[2] / orbitRadius) ** 2 - 3)])
  relativeVelocityVector = velocityVector - np.cross(c.EARTH_ANGULAR_VELOCITY_VECTOR, positionVector)
  relativeVelocity = np.linalg.norm(relativeVelocityVector)

  elevation = car2sphere(positionVector[0], positionVector[1], positionVector[2])[1]
  radius = radius_by_latitude(c.EARTH_RADIUS, c.EARTH_OBLATENESS, elevation)
  altitude = orbitRadius - radius
  atmosphereDensity = expo(altitude).rho[0]
  
  dragAcceleration = (-(1/2) * (c.SATELLITE_A_M_RATIO / 1e6) * 
    c.DRAG_COEFFICIENT * (atmosphereDensity * 1e9) * relativeVelocity) * relativeVelocityVector
  perturbationAcceleration = j2PerturbationAcceleration + dragAcceleration

  tangentialVelocity = velocityVector / np.linalg.norm(velocityVector)
  azimuthalVelocity = (np.cross(positionVector, velocityVector) / 
    np.linalg.norm(np.cross(positionVector, velocityVector)))
  nadirVelocity = np.cross(azimuthalVelocity, tangentialVelocity)
  rotationMatrix = np.array([tangentialVelocity, nadirVelocity, azimuthalVelocity])

  perturbationAcceleration = np.dot(rotationMatrix.T.conj(), perturbationAcceleration)
  
  return (perturbationAcceleration, orbitRadius, velocity, argOfLatitude, 
    specificAngularMomentum, positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity)

def keplerianParametersDerivatives(keplerianParameters: np.array, time: float, 
  perturbationAcceleration: np.array, orbitRadius: float, velocity: float, 
  argOfLatitude: float, specificAngularMomentum: float) -> np.array:
  """
  Calculates the derivative of the Keplerian parameters of a satellite.

  Args:
    keplerianParameters: Keplerian parameters of the satellite.
    time: Time since the beginning of the simulation.

  Returns:
    Derivative of the Keplerian parameters of the satellite.
  """
  # Keplerian parameters needed for derivation
  semimajorAxis = keplerianParameters[0]
  eccentricity = keplerianParameters[1]
  inclination = keplerianParameters[2]
  trueAnomaly = keplerianParameters[5]

  # Derivation of the orbital parameters for output
  semimajorAxisDerivative = (2 * semimajorAxis ** 2 * velocity / 
    c.EARTH_STD_GRAV_PARAMETER * perturbationAcceleration[0])
  eccentricityDerivative = 1 / velocity * (2 * (eccentricity + np.cos(trueAnomaly)) * 
    perturbationAcceleration[0] - orbitRadius / semimajorAxis * np.sin(trueAnomaly) * 
    perturbationAcceleration[1])
  inclinationDerivative = (orbitRadius * np.cos(argOfLatitude) / 
    specificAngularMomentum * perturbationAcceleration[2])
  raanDerivative = orbitRadius * np.sin(argOfLatitude) / (specificAngularMomentum * 
    np.sin(inclination)) * perturbationAcceleration[2]
  argOfPeriapsisDerivative = 1 / (eccentricity * velocity) * (2 * np.sin(trueAnomaly) *
    perturbationAcceleration[0] + (2 * eccentricity + orbitRadius / semimajorAxis *
    np.cos(trueAnomaly)) * perturbationAcceleration[1]) - (orbitRadius * np.sin(argOfLatitude) *
    np.cos(inclination) / (specificAngularMomentum * np.sin(inclination)) * perturbationAcceleration[2])
  trueAnomalyDerivative = specificAngularMomentum / orbitRadius ** 2 - 1 / (eccentricity * 
    velocity) * (2 * np.sin(trueAnomaly) * perturbationAcceleration[0] + (2 * eccentricity +
    orbitRadius / semimajorAxis * np.cos(trueAnomaly)) * perturbationAcceleration[1])
  
  return np.array([
    semimajorAxisDerivative,
    eccentricityDerivative,
    inclinationDerivative,
    raanDerivative,
    argOfPeriapsisDerivative,
    trueAnomalyDerivative
  ])

def orbit_determination(currentKeplerianParameters: np.array, time: float) -> Tuple[np.array, np.array, float, float, np.array]:
  """
  Determination of the orbit integrating the orbital parameters through time.

  Args:
    currentKeplerianParameters: Keplerian parameters. Will use the initial ones 
      or the ones from the previous time step according to the time variable.
    time: [s] Simultation time.

  Returns:
    positionVector (np.array): [km]
    relativeVelocityVector (np.array): [km/s]
    meanVelocity (float): [km/s]
    atmosphereDensity (float): [kg/m^3]
    keplerianParameters (np.array): {
      semimajorAxis (float): [km/s]
      eccentricity (float): [-]
      inclination (float): [rad/s]
      raan (float): [rad/s]
      argOfPeriapsis (float): [rad/s]
      trueAnomaly (float): [rad/s]
    }
  """
  perturbationAcceleration, orbitRadius, velocity, argOfLatitude, specificAngularMomentum, \
    positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity = \
    orbital_parameters(currentKeplerianParameters)

  keplerianParameters = integrate.odeint(keplerianParametersDerivatives, 
    y0 = currentKeplerianParameters, t = np.array([time, time + c.DELTA_TIME]),
    args = (perturbationAcceleration, orbitRadius, velocity, argOfLatitude, 
    specificAngularMomentum))[1]
  
  return (positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity,
    keplerianParameters)
