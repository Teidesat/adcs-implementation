import numpy as np
import scipy

TRIAD = 1
FMINSLSQP = 2
SVD = 3
Q_METHOD = 4
QUEST = 5

def triad(sunSensor: np.array, sunSensorAttitude: np.array, magnetometerAttitude: np.array, magnetometer: np.array) -> np.array:
  """
  Triad algorithm for attitude determination.

  Arguments:
    sunSensor (np.array): [rad]
    sunSensorAttitude (np.array): [rad]
    magnetometerAttitude (np.array): [rad]
    magnetometer (np.array): [rad]

  Returns:
    attitudeMatrix (np.array): [rad]
  """
  measure = np.array([np.array(sunSensor).T.conj(),
                      np.array(np.cross(sunSensor, magnetometer) 
                        / np.linalg.norm(np.cross(sunSensor, magnetometer))).T.conj(),
                      np.array(np.cross(sunSensor, np.cross(sunSensor, magnetometer) 
                        / np.linalg.norm(np.cross(sunSensor, magnetometer)))).T.conj()])

  model = np.array([np.array(sunSensorAttitude).T.conj(),
                    np.array(np.cross(sunSensorAttitude, magnetometerAttitude) 
                      / np.linalg.norm(np.cross(sunSensorAttitude, magnetometerAttitude))).T.conj(),
                    np.array(np.cross(sunSensorAttitude, np.cross(sunSensorAttitude, magnetometerAttitude) 
                      / np.linalg.norm(np.cross(sunSensorAttitude, magnetometerAttitude)))).T.conj()])

  return (measure * model).T.conj()

def fminslsqp(sunSensor, sunSensorAttitude, magnetometerAttitude: np.array, magnetometer: np.array) -> np.array:
  """
  Function minimizing using Sequential Least Squares Programming algorithm for attitude determination.

  Arguments:
    sunSensor: [rad] Unused parameter.
    sunSensorAttitude: [rad] Unused parameter.
    magnetometerAttitude: [rad]
    magnetometer: [rad]

  Returns:
    attitudeMatrix: [rad]
  """
  del sunSensor, sunSensorAttitude
  attitudeMatrix = np.zeros((3,3))
  functionToMinimize = lambda attitudeMatrix: np.linalg.norm(magnetometer - attitudeMatrix @ magnetometerAttitude) ** 2
  firstConstraint = np.trace(- attitudeMatrix.T.conj() @ attitudeMatrix + np.eye(3))
  secondConstraint = np.linalg.det(attitudeMatrix) - 1
  initialPoint = np.ones((3))
  return scipy.optimize.fmin_slsqp(functionToMinimize, initialPoint,
    eqcons=[firstConstraint, secondConstraint])
  
def svd(sunSensor: np.array, sunSensorAttitude: np.array, magnetometerAttitude: np.array, magnetometer: np.array) -> np.array:
  """
  Singular Value Decomposition algorithm for attitude determination.

  Arguments:
    sunSensor: [rad]
    sunSensorAttitude: [rad]
    magnetometerAttitude: [rad]
    magnetometer: [rad]

  Returns:
    attitudeMatrix: [rad]
  """
  alpha = np.array([0.2, 0.8]).T.conj()
  beta = (alpha[0] * magnetometer * np.array(magnetometerAttitude).T.conj() + 
    alpha[1] * sunSensor * np.array(sunSensorAttitude).T.conj())
  leftSingularMatrix, _, rightSingularMatrix = np.linalg.svd(beta)

  originalMatrix = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, np.linalg.det(leftSingularMatrix) * np.linalg.det(rightSingularMatrix)]])

  return leftSingularMatrix * originalMatrix * rightSingularMatrix.T.conj()

def q_method(sunSensor: np.array, sunSensorAttitude: np.array, magnetometerAttitude: np.array, magnetometer: np.array) -> np.array:
  """
  Q method algorithm for attitude determination.

  Arguments:
    sunSensor: [rad]
    sunSensorAttitude: [rad]
    magnetometerAttitude: [rad]
    magnetometer: [rad]

  Returns:
    attitudeMatrix: [rad]
  """
  alpha = np.array([0.2, 0.8]).T.conj()
  B = alpha[0] * magnetometer * np.array(magnetometerAttitude).T.conj() + \
    alpha[1] * sunSensor * np.array(sunSensorAttitude).T.conj()
  S = B.T.conj() + B
  s = np.trace(B)
  z = np.array([B[1,2] - B[2,1], B[2,0] - B[0,2], B[0,1] - B[1,0]])

  K = np.ones((4,4))
  K[0:3, 0:3] = S - s * np.eye(3)
  K[0:3, 3] = z.T.conj()
  K[3, 0:3] = z
  K[3, 3] = s
  V, D = np.linalg.eig(K)
  _, kcol = np.where(D == np.max(np.max(D)))
  q = V[:, kcol]
  attitudeMatrix = np.array([[q[0, 0] ** 2 - q[0, 1] ** 2 - q[0, 2] ** 2 + q[0, 3] ** 2,
                              2 * (q[0, 0] * q[0, 1] + q[0, 2] * q[0, 3]),
                              2 * (q[0, 0] * q[0, 2] - q[0, 1] * q[0, 3])],
                              [2 * (q[0, 0] * q[0, 1] - q[0, 2] * q[0, 3]),
                              - q[0, 0] ** 2 + q[0, 1] ** 2 - q[0, 2] ** 2 + q[0, 3] ** 2,
                              2 * (q[0, 1] * q[0, 2] + q[0, 0] * q[0, 3])],
                              [2 * (q[0, 0] * q[0, 2] + q[0, 1] * q[0, 3]),
                              2 * (q[0, 1] * q[0, 2] - q[0, 0] * q[0, 3]),
                              - q[0, 0] ** 2 - q[0, 1] ** 2 + q[0, 2] ** 2 + q[0, 3] ** 2]])
  return attitudeMatrix

def quest(sunSensor: np.array, sunSensorAttitude: np.array, magnetometerAttitude: np.array, magnetometer: np.array) -> np.array:
  """
  Quest algorithm for attitude determination.

  Arguments:
    sunSensor: [rad]
    sunSensorAttitude: [rad]
    magnetometerAttitude: [rad]
    magnetometer: [rad]

  Returns:
    attitudeMatrix: [rad]
  """
  alpha = np.array([0.2, 0.8]).T.conj()
  B = alpha[0] * magnetometer * np.array(magnetometerAttitude).T.conj() + \
    alpha[1] * sunSensor * np.array(sunSensorAttitude).T.conj()
  S = B.T.conj() + B
  s = np.trace(B)
  z = np.array([B[1,2] - B[2,1], B[2,0] - B[0,2], B[0,1] - B[1,0]])

  g = scipy.linalg.solve(S - (s + 1) * np.eye(3), -z)

  return (1/(1+g[0]**2+g[1]**2+g[2]**2)) * \
    np.array([[1+g[0]**2-g[1]**2-g[2]**2, 2*(g[0]*g[1]+g[2]), 2*(g[0]*g[2]+g[1])],
                [2*(g[0]*g[1]-g[2]), 1-g[0]**2+g[1]**2-g[2]**2, 2*(g[1]*g[2]+g[0])],
                [2*(g[0]*g[2]+g[1]), 2*(g[1]*g[2]-g[0]), 1-g[0]**2-g[1]**2+g[2]**2]])

def static_attitude_determination(sunSensor: np.array, sunSensorAttitude: np.array, magnetometerAttitude: np.array, magnetometer: np.array, algorithmSelector: int = SVD, idealAttitudeMatrix: np.array = None) -> tuple(np.array, float):
  """
  Static attitude determination algorithms. Needs at least two measures: sun sensor
  and magnetometer. Should be as orthogonal to the Sun Sensor as possible.
  
  Arguments:
    sunSensor: [-]
    sunSensorAttitude: [-]
    magnetometerAttitude: [-]
    magnetometer: [T]
    algorithmSelector: Triad: 1, FMINSLSQP: 2, SVD: 3 (Default), Q_METHOD: 4, QUEST: 5
    idealAttitudeMatrix: [-] If not given, the error is not calculated.
  
  Returns:
    attitudeMatrix: [-]
    error: [-]
  """

  ALGORITHM_SWITCH = {
    TRIAD: triad,
    FMINSLSQP: fminslsqp,
    SVD: svd,
    Q_METHOD: q_method,
    QUEST: quest
  }
  
  staticADAlgorithm = ALGORITHM_SWITCH.get(algorithmSelector, lambda: "Invalid algorithm selected.")

  attitudeMatrix = staticADAlgorithm(sunSensor, sunSensorAttitude, magnetometerAttitude, magnetometer)

  if idealAttitudeMatrix is not None:
    error = np.trace(idealAttitudeMatrix - attitudeMatrix)
    return attitudeMatrix, error
  return attitudeMatrix