import numpy as np
import matplotlib.pyplot as plt
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
  # Initializing needed variables
  angularVelocity = c.INITIAL_ANGULAR_VELOCITY
  time = c.INITIAL_TIME
  previousExternalMagneticField = np.zeros((3,))

  # Defining variables for plotting
  disturbancesPlot = np.empty((0,3))
  controlPlot = np.empty((0,3))
  angularVelocityPlot = np.empty((0,3))
  angularVelocityNormPlot = np.empty(0)
  polePointingPlot = np.empty(0)
  nadirPointingPlot = np.empty(0)
  measuredTrackingErrorPlot = np.empty(0)
  determinationErrorPlot = np.empty(0)
  keplerianParametersPlot = np.empty((0,6))

  # Simulation loop
  while time < c.SECONDS:
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

    # Updating variables for next iteration and plotting
    previousExternalMagneticField = externalMagneticField
    time += c.DELTA_TIME

    disturbancesPlot = np.append(disturbancesPlot, disturbances, axis=0)
    controlPlot = np.append(controlPlot, [controlTorque], axis=0)
    angularVelocityPlot = np.append(angularVelocityPlot, [angularVelocity], axis=0)
    angularVelocityNormPlot = np.append(angularVelocityNormPlot, np.linalg.norm(angularVelocity))
    polePointingPlot = np.append(polePointingPlot, polePointingError)
    nadirPointingPlot = np.append(nadirPointingPlot, nadirPointingError)
    measuredTrackingErrorPlot = np.append(measuredTrackingErrorPlot, trackingError)
    determinationErrorPlot = np.append(determinationErrorPlot, determinationError)
    keplerianParametersPlot = np.append(keplerianParametersPlot, [keplerianParameters], axis=0)

  # Plotting
  plt.figure('Disturbance Torques')
  plt.title('Disturbance Torques')
  plt.plot(c.TIMESTEPS, disturbancesPlot[:,0], label='X Disturbances')
  plt.plot(c.TIMESTEPS, disturbancesPlot[:,1], label='Y Disturbances')
  plt.plot(c.TIMESTEPS, disturbancesPlot[:,2], label='Z Disturbances')
  plt.legend()
  plt.show()

  plt.figure('Control Torques')
  plt.title('Control Torques')
  plt.plot(c.TIMESTEPS, controlPlot[:,0], label='X Control')
  plt.plot(c.TIMESTEPS, controlPlot[:,1], label='Y Control')
  plt.plot(c.TIMESTEPS, controlPlot[:,2], label='Z Control')
  plt.legend()
  plt.show()

  plt.figure('Angular Velocity')
  plt.title('Angular Velocity')
  plt.plot(c.TIMESTEPS, angularVelocityPlot[:,0], label='X Angular Velocity')
  plt.plot(c.TIMESTEPS, angularVelocityPlot[:,1], label='Y Angular Velocity')
  plt.plot(c.TIMESTEPS, angularVelocityPlot[:,2], label='Z Angular Velocity')
  plt.plot(c.TIMESTEPS, angularVelocityNormPlot, label='Angular Velocity Norm')
  plt.legend()
  plt.show()

  plt.figure('Pole Pointing Error')
  plt.subplot(211)
  plt.title('Angle between Z axis and magnetic field lines')
  plt.plot(c.TIMESTEPS, polePointingPlot)
  plt.subplot(212)
  plt.title('Angle between X axis and nadir direction')
  plt.plot(c.TIMESTEPS, nadirPointingPlot)
  plt.show()

  plt.figure('Determination Error')
  plt.subplot(311)
  plt.title('Determination Error')
  plt.plot(c.TIMESTEPS, determinationErrorPlot)
  plt.subplot(312)
  plt.title('Measured Tracking Error')
  plt.plot(c.TIMESTEPS, measuredTrackingErrorPlot)
  plt.show()

  plt.figure('Keplerian Parameters')
  plt.subplot(411)
  plt.title('Semi-major Axis')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,0])
  plt.subplot(412)
  plt.title('Eccentricity')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,1])
  plt.subplot(413)
  plt.title('Inclination')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,2])
  plt.subplot(414)
  plt.title('Right Ascension of the Ascending Node (RAAN)')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,3])
  plt.subplot(421)
  plt.title('Argument of Perigee')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,4])
  plt.subplot(422)
  plt.title('True Anomaly')
  plt.plot(c.TIMESTEPS, keplerianParametersPlot[:,5])
  plt.show()

    