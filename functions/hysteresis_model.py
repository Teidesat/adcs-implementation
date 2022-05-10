import numpy as np
import constants as c

hysteresisLoopSwitch = [False, False, False]

def hysteresis_model(externalMagneticField: np.array, rodDipoleDirection: np.array, rodInduction: float, rodVolume: float) -> np.array:
  """
  This function implements the hysteresis model for a single hysteresis rod.

  Args:
    externalMagneticField: [T]
    rodDipoleDirection: [T]
    rodInduction: [T]
    rodVolume: [m^3]
  
  Returns:
    The total torque caused by the hysteresis rod.
  """
  hearthHyst = np.dot(externalMagneticField / c.VACUUM_PERMEABILITY, rodDipoleDirection)
  if c.HYSTERESIS_MODEL_SWITCH == 1:
    # Hysteresis loop with delay + ramp Br/H
    if hearthHyst >= c.REMANENCE:
      hysteresisLoopSwitch[0] = True
    elif hearthHyst <= -c.REMANENCE:
      hysteresisLoopSwitch[0] = False
    
    bHyst = hysteresis_loop() + hearthHyst * c.REMANENCE / c.COERCIVITY

    # Add in saturation
    if bHyst > rodInduction:
      bHyst = rodInduction
    elif bHyst < -rodInduction:
      bHyst = -rodInduction
  elif c.HYSTERESIS_MODEL_SWITCH == 2:
    # Hysteresis loop with delay + ramp Bs/2H
    if hearthHyst >= c.REMANENCE:
      hysteresisLoopSwitch[1] = True
    elif hearthHyst <= -c.REMANENCE:
      hysteresisLoopSwitch[1] = False

    bHyst = hysteresis_loop() + hearthHyst * (rodInduction / 2 / c.COERCIVITY)

    # Add in saturation
    if bHyst > rodInduction:
      bHyst = rodInduction
    elif bHyst < -rodInduction:
      bHyst = -rodInduction

  elif c.HYSTERESIS_MODEL_SWITCH == 3:
    # Hysteresis loop with delay only
    if hearthHyst >= c.REMANENCE:
      hysteresisLoopSwitch[2] = True
    elif hearthHyst <= -c.REMANENCE:
      hysteresisLoopSwitch[2] = False
    
    bHyst = hysteresis_loop()
  
  else:
    raise ValueError("Invalid hysteresis model switch value")

  return np.cross((bHyst * rodVolume / c.VACUUM_PERMEABILITY) * rodDipoleDirection, externalMagneticField)

def hysteresis_loop(loopSelector: int) -> float:
  """
  Returns the hysteresis loop parameters depending on whether the selected
  loop is on or off.
  """
  if hysteresisLoopSwitch[loopSelector]:
    return c.COERCIVITY
  else:
    return -c.COERCIVITY