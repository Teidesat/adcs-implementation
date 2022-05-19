import numpy as np
import constants as c
from dynamics import dynamics
from functions.velocity_body import velocity_body
from orbit_determination import orbit_determination
from kinematics import kinematics
from functions.reference_attitude import reference_attitude
from sensors import sensors
from static_AD import static_AD
from external_torques import external_torques
from determination_error import determination_error
from tracking_error import tracking_error
from functions.earth_position import earth_position

def detumbling():
  """
  Model for simulating the detumbling phase of a spacecraft.
  The detumbling is the phase in ADCS where the spacecraft is set to a stable
  attitude.
  """
  angularVelocity = c.INITIAL_ANGULAR_VELOCITY
  time = c.INITIAL_TIME
  previousExternalMagneticField = np.zeros((3,))

  while time <= c.SECONDS:
    print('SIMULATING DETUMBLING:\t{0}/{1} seconds\r'.format(time, c.SECONDS), end="")

    positionVector, relativeVelocityVector, meanVelocity, atmosphereDensity, \
      keplerianParameters = orbit_determination(time)
    
    attitudeMatrix = kinematics(angularVelocity, time)

    attitudeMatrixError, idealTrackError, attitudeLN, angularVelocityError = reference_attitude(attitudeMatrix, 
      meanVelocity, keplerianParameters[2], time, angularVelocity)

    velocityBody = velocity_body(relativeVelocityVector, attitudeMatrixError)

    sunVector, sunAttitude, externalMagneticField, magnetometerAttitude, gyroAngularVelocity = sensors(positionVector, attitudeMatrix, angularVelocity, time)

    estimatedAttitudeMatrix = static_AD(sunVector, sunAttitude, externalMagneticField, magnetometerAttitude)

    determinationError, trackingError = determination_error(attitudeMatrix, estimatedAttitudeMatrix, attitudeLN)
    nadirPointingError, polePointingError, velocityPointingError = tracking_error(externalMagneticField, velocityBody, estimatedAttitudeMatrix, attitudeLN)

    northVector, nadirVector = earth_position(positionVector, attitudeMatrix)

    controlTorque, solarRadiationPressureTorque, magnDisturbance, dragTorque, \
      areaCross, albedoTorque, magnetotorquer  = external_torques(atmosphereDensity, 
      sunVector, externalMagneticField, previousExternalMagneticField, velocityBody, 
      nadirVector, gyroAngularVelocity)

    disturbances = solarRadiationPressureTorque + dragTorque + albedoTorque
    disturbancesAndControl = disturbances + controlTorque + magnetotorquer

    angularVelocity = dynamics(disturbancesAndControl, positionVector, attitudeMatrixError, angularVelocity, time)

    previousExternalMagneticField = externalMagneticField
    time += c.DELTA_TIME

    