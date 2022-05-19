import numpy as np
import constants as c
from functions.bdot import b_dot
from functions.magnetic_disturbance import magnetic_disturbance
from functions.permanent_magnet import permanent_magnet
from functions.hysteresis_model import hysteresis_model
from functions.geometry import geometry
from functions.solar_radiation_pressure import solar_radiation_pressure
from functions.aerodynamic import aerodynamic
from functions.albedo import albedo
from typing import Tuple

def external_torques(atmosphereDensity: float, sunVector: np.array, 
  externalMagneticField: np.array, previousMagneticField: np.array, velocityBody: np.array, 
  nadirVector: np.array, gyroscopeAngVelocity: np.array) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
  """
  Calculates all of the external torques that act on the spacecraft.

  Arguments:
    atmosphereDensity: [kg/m^3]
    sunVector: [m]
    externalMagneticField: [T]
    previousMagneticField: [T]
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
  magneticFieldDerivative = (externalMagneticField - previousMagneticField) / c.DELTA_TIME
  _, magnetotorquer = b_dot(externalMagneticField, magneticFieldDerivative, c.DETUMBLE_TORQUE, gyroscopeAngVelocity)
  magnDisturbance = magnetic_disturbance(externalMagneticField)
  permaMagnetControlTorque = permanent_magnet(externalMagneticField)

  hysteresisTorque1 = hysteresis_model(externalMagneticField, c.ROD1_DIPOLE_DIRECTION, c.ROD1_INDUCTION, c.ROD1_VOLUME)
  hysteresisTorque2 = hysteresis_model(externalMagneticField, c.ROD2_DIPOLE_DIRECTION, c.ROD2_INDUCTION, c.ROD2_VOLUME)
  hysteresisTorque3 = hysteresis_model(externalMagneticField, c.ROD3_DIPOLE_DIRECTION, c.ROD3_INDUCTION, c.ROD3_VOLUME)

  controlTorque = hysteresisTorque1 + hysteresisTorque2 + hysteresisTorque3 + permaMagnetControlTorque

  faceVectors, massDisplacement, area, areaCross = geometry(velocityBody)
  solarRadiationPressureTorque = solar_radiation_pressure(sunVector, faceVectors, massDisplacement, area)
  dragTorque = aerodynamic(massDisplacement, faceVectors, atmosphereDensity, velocityBody, area)
  albedoTorque = albedo(nadirVector, faceVectors, massDisplacement, area)

  return controlTorque, solarRadiationPressureTorque, magnDisturbance, dragTorque, areaCross, albedoTorque, magnetotorquer