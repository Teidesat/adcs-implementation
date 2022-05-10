import numpy as np
import constants as c
from functions.bdot import b_dot
from functions.magnetic_disturbance import magneticDisturbance
from functions.permanent_magnet import permanentMagnet
from functions.hysteresis_model import hysteresisModel
from functions.geometry import geometry
from functions.solar_radiation_pressure import solarRadiationPressure
from functions.aerodynamic import aerodynamic
from functions.albedo import albedo

def externalTorques(atmosphereDensity: float, sunVector: np.array, 
  externalMagneticField: np.array, velocityBody: np.array, nadirVector: np.array, 
  gyroscopeAngVelocity: np.array):
  """
  Calculates all of the external torques that act on the spacecraft.

  Arguments:
    atmosphereDensity: [kg/m^3]
    sunVector: [m]
    externalMagneticField: [T]
    velocityBody: [m/s]
    nadirVector: [m]
    gyroscopeAngVelocity: [rad/s]

  Returns:
    controlTorque: Caused by the hysteresis rods.
    solarRadiationPressureTorque
    magnDisturbance. Unused.
    dragTorque
    areaCross. Unused.
    albedoTorque
    magnetotorquer
  """
  _, magnetotorquer = b_dot(externalMagneticField, externalMagneticField, c.DETUMBLE_TORQUE, gyroscopeAngVelocity)  # Derivatives??
  magnDisturbance = magneticDisturbance(externalMagneticField)
  permaMagnetControlTorque, _ = permanentMagnet(externalMagneticField)

  hysteresisTorque1 = hysteresisModel(externalMagneticField, c.ROD1_DIPOLE_DIRECTION, c.ROD1_INDUCTION, c.ROD1_VOLUME)
  hysteresisTorque2 = hysteresisModel(externalMagneticField, c.ROD2_DIPOLE_DIRECTION, c.ROD2_INDUCTION, c.ROD2_VOLUME)
  hysteresisTorque3 = hysteresisModel(externalMagneticField, c.ROD3_DIPOLE_DIRECTION, c.ROD3_INDUCTION, c.ROD3_VOLUME)

  controlTorque = hysteresisTorque1 + hysteresisTorque2 + hysteresisTorque3 + permaMagnetControlTorque

  faceVectors, massDisplacement, area, areaCross = geometry(velocityBody)
  solarRadiationPressureTorque = solarRadiationPressure(sunVector, faceVectors, massDisplacement, area)
  dragTorque = aerodynamic(massDisplacement, faceVectors, atmosphereDensity, velocityBody)
  albedoTorque = albedo(nadirVector, faceVectors, massDisplacement, area)

  return controlTorque, solarRadiationPressureTorque, magnDisturbance, dragTorque, areaCross, albedoTorque, magnetotorquer