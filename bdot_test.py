# Unit testing for bdot.py

import unittest
import bdot

class TestBDot(unittest.TestCase):
  def testMagMoment(self):
    tests = [
      # magnetic_moment parameters: (bB, dbB, k)
      {
        'input': ([-0.131, -0.131, -0.131], [1, 1, 1], 1),
        'output': [-0.131, -0.131, -0.131]
      },
      {
        'input': ([0.131, 0.131, 0.131], [1, 1, 1], 1),
        'output': [-0.131, -0.131, -0.131]
      }
    ]
    for test in tests:
      self.assertEqual(bdot.magnetic_moment(*test['input']), test['output'])

  def testMagnetorque(self):
    tests = [
      # magnetorque parameters: (magneticMoment, bB, angularVelocity)
      {
        'input': ([-0.131, -0.131, -0.131], [1, 1, 1], [1, 1, 1]),
        'output': [0, 0, 0]
      },
      {
        'input': ([0.131, 0.131, 0.131], [1, 1, 1], [1, 1, 1]),
        'output': [0, 0, 0]
      }
    ]
    for test in tests:
      self.assertEqual(bdot.magnetorque(*test['input']), test['output'])

  def testBDot(self):
    tests = [
      # b_dot parameters: (bB, dbB, k, angularVelocity)
      {
        'input': [[1, 1, 1], [1, 1, 1], 1, [1, 1, 1]],
        'output': ([-0.131, -0.131, -0.131], [0, 0, 0])
      },
      {
        'input': [[1, 1, 1], [1, 1, 1], 1, [0, 0, 0]],
        'output': ([-0.131, -0.131, -0.131], [0.0, 0.0, 0.0])
      }
    ]
    for test in tests:
      self.assertEqual(bdot.b_dot(*test['input']), test['output'])

if __name__ == '__main__':
  unittest.main()