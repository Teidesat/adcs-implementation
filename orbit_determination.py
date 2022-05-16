import numpy as np
from ambiance import Atmosphere
import constants as c
import functions.kep2car as kep2car
import functions.car2sphere as car2sphere
import functions.radius_by_latitude as radius_by_latitude
from scipy import integrate

def orbital_parameters(semimajorAxis: float, eccentricity: float, inclination: float, 
  raan: float, argOfPeriapsis: float, trueAnomaly: float, time: float) -> \
  tuple(np.array, np.array, float, float, float, float, float, float, float, float):
  """
  Determination of the orbital parameters from Keplerian elements in a given instant.

  Args:
    semimajorAxis: [km]
    eccentricity: [-]
    inclination: [rad]
    raan: [rad]
    argOfPeriapsis: [rad]
    trueAnomaly: [rad]
    time: [s]
  
  Returns:
    positionVector: [km]
    relativeVelocityVector: [km/s]
    meanVelocity: [km/s]
    atmosphereDensity: [kg/m^3]
    semimajorAxisDerivative: [km/s]
    eccentricityDerivative: [-]
    inclinationDerivative: [rad/s]
    raanDerivative: [rad/s]
    argOfPeriapsisDerivative: [rad/s]
    trueAnomalyDerivative: [rad/s]
  """
  # Keplerian parameters - Many of these are constants and could be placed in the c file
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
  altitude = c.EARTH_RADIUS - radius
  atmosphereDensity = Atmosphere(altitude).density
  
  dragAcceleration = (-(1/2) * (c.SATELLITE_A_M_RATIO / 1e6) * 
    c.DRAG_COEFFICIENT * (atmosphereDensity * 1e9) * relativeVelocity) * relativeVelocityVector
  perturbationAcceleration = j2PerturbationAcceleration + dragAcceleration

  tangentialVelocity = velocityVector / np.linalg.norm(velocityVector)
  azimuthalVelocity = (np.cross(positionVector, velocityVector) / 
    np.linalg.norm(np.cross(positionVector, velocityVector)))
  nadirVelocity = np.cross(tangentialVelocity, azimuthalVelocity)
  rotationMatrix = np.array([tangentialVelocity, nadirVelocity, azimuthalVelocity])

  perturbationAcceleration = rotationMatrix.T.conj() * perturbationAcceleration

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
  
  return (positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity, 
    semimajorAxisDerivative, eccentricityDerivative, inclinationDerivative, 
    raanDerivative, argOfPeriapsisDerivative, trueAnomalyDerivative)

def orbit_determination() -> tuple(np.array, np.array, float, float, np.array):
  """
  Determination of the orbit integrating the orbital parameters through time.

  Returns:
    positionVector (np.array): [km]
    relativeVelocityVector (np.array): [km/s]
    meanVelocity (float): [km/s]
    atmosphereDensity (float): [kg/m^3]
    keplerianParameters (Dict): {
      semimajorAxis (float): [km/s]
      eccentricity (float): [-]
      inclination (float): [rad/s]
      raan (float): [rad/s]
      argOfPeriapsis (float): [rad/s]
      trueAnomaly (float): [rad/s]
    }
  """

  positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity, \
    semimajorAxis, eccentricity, inclination, raan, argOfPeriapsis, trueAnomaly = \
    integrate.odeint(orbital_parameters, *c.INITIAL_KEPLERIAN_PARAMETERS, t = c.TIMESTEPS)
  
  keplerianParameters = {
    'semimajorAxis': semimajorAxis,
    'eccentricity': eccentricity,
    'inclination': inclination,
    'raan': raan,
    'argOfPeriapsis': argOfPeriapsis,
    'trueAnomaly': trueAnomaly
  }

  return (positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity,
    keplerianParameters)


    

  
  