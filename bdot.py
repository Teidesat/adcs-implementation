# Simple first approach to the B-Dot algorithm implementation.
# Uses B-Dot to return only the magnetic moment.
# To get the resulting torque, saturation for 'm' and a cross 
# product between 'm' and 'b_B' is needed.

import numpy as np

def b_dot(b_B, db_B, k):
  # Formatting vectors to work with numpy
  b_B = np.array(b_B)
  db_B = np.array(db_B)

  # Magnetic moment
  m = np.divide(np.multiply(db_B, -k), (np.linalg.norm(b_B)**2))
  return m