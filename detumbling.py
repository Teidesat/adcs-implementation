import numpy as np
import constants as c
from source.dynamics import dynamics
from source.velocity_body import velocity_body
from source.orbit_determination import orbit_determination
from source.kinematics import kinematics
from source.reference_attitude import reference_attitude
from source.sensors.sensors import sensors
from source.attitude_determination.static_AD import static_AD
from source.external_torques.external_torques import external_torques
from source.error.determination_error import determination_error
from source.error.tracking_error import tracking_error
from source.earth_position import earth_position

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

    