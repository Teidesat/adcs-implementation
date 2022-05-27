# Passive Magnetic and Aerodynamic Stabilization for 1U CubeSat for
# Earth-Pointing for low latitudes

import numpy as np
import matplotlib.pyplot as plt
import math

# Earth/Universal constants
EARTH_STD_GRAV_PARAMETER = 3.98600433e5                                   # mu_earth: [km^3/s^2]
EARTH_MASS = 5.9736e24                                                    # m_T: [kg]
EARTH_RADIUS = 6378.137                                                   # r_earth: [km]
EARTH_ANGULAR_VELOCITY = 7.292115e-5                                      # w_earth: [rad/s]
EARTH_ANGULAR_VELOCITY_VECTOR = np.array([0, 0, EARTH_ANGULAR_VELOCITY])  # w_earth(vector): [rad/s]
EARTH_MAGNETIC_INDUCTION = 3e-5                                           # B_earth: [T]
EARTH_ORBITAL_PERIOD = 31558118.4                                         # T_earth: [s]
LIGHT_SPEED = 299792458                                                   # c: [m/s]
UNIVERSAL_GRAV_CONSTANT = 6.674e-11                                       # G: [m^3/kg/s^2] In matlab it was e-17, don't know why.

# Hysteresis loop constants
VACUUM_PERMITTIVITY = 8.854187817e-12                                     # E_0: [F/m]
VACUUM_PERMEABILITY = 1.2566370614e-6                                     # mu_0: [N/A^2]
REMANENCE = 0.004                                                         # B_r: [T]
COERCIVITY = 12                                                           # H_c: [T]
HYSTERESIS_MODEL_SWITCH = 3                                               # 1: delay+ramp Br/H,   2: delay+ramp Bs/2H,   3: delay only.

# Gravity perturbation
EARTH_OBLATENESS = 0.003352813                                            # f: [-]
J2_PERTURBATION = 0.0010826269                                            # J2: [-]

# Satellite data
SATELLITE_MASS = 1.33                                                     # m: [kg]
SATELLITE_LENGTH = 0.1                                                    # l: [m]
SATELLITE_MAX_FRONTAL_SECTION = SATELLITE_LENGTH * (SATELLITE_LENGTH +
  2 * SATELLITE_LENGTH * np.cos(np.radians(20)))                          # A: [m^2]
SATELLITE_A_M_RATIO = SATELLITE_MAX_FRONTAL_SECTION / SATELLITE_MASS      # A/m: [m^2/kg]
DRAG_COEFFICIENT = 2.1                                                    # C_d: [-]
BALLISTIC_COEFFICIENT = 1 / (DRAG_COEFFICIENT * SATELLITE_A_M_RATIO)      # B: [-]
CENTER_OF_MASS_DISPLACEMENT = 0.0                                         # d_CM: [m]
NUMBER_OF_FACES = 6                                                       # N: [-]

## Keplerian parameters
INITIAL_HEIGHT = 410                                                      # H_0: [km] Until 360.
MINIMUM_ELLIPSE_DISTANCE = INITIAL_HEIGHT + 6378                          # r_p: [km]
MAXIMUM_ELLIPSE_DISTANCE = INITIAL_HEIGHT + 6380                          # r_a: [km]
INITIAL_SEMIMAJOR_AXIS = 0.5 * (MINIMUM_ELLIPSE_DISTANCE + 
MAXIMUM_ELLIPSE_DISTANCE)                                                 # a_0: [m]
INITIAL_ECCENTRICITY = (MAXIMUM_ELLIPSE_DISTANCE - 
  MINIMUM_ELLIPSE_DISTANCE) / (MAXIMUM_ELLIPSE_DISTANCE + 
  MINIMUM_ELLIPSE_DISTANCE)                                               # e_0: [-]
MEAN_ANGULAR_MOTION = math.sqrt(EARTH_STD_GRAV_PARAMETER / 
  (INITIAL_SEMIMAJOR_AXIS ** 3))                                          # n_0: [rad/s]
ORBITAL_PERIOD = 2 * math.pi / MEAN_ANGULAR_MOTION                        # T_0: [s]
INITIAL_INCLINATION = math.radians(51.645)                                # i_0: [rad]
INITIAL_RAAN = math.radians(0.0)                                          # (Right Ascension of the Ascending Node) OMEGA_0: [rad]
INITIAL_ARGUMENT_OF_PERIAPSIS = math.radians(0.0)                         # omega_0: [rad]
INITIAL_TRUE_ANOMALY = math.radians(0.0)                                  # theta_0: [rad]

INITIAL_KEPLERIAN_PARAMETERS = np.array([
  INITIAL_SEMIMAJOR_AXIS,
  INITIAL_ECCENTRICITY,
  INITIAL_INCLINATION,
  INITIAL_RAAN,
  INITIAL_ARGUMENT_OF_PERIAPSIS,
  INITIAL_TRUE_ANOMALY
])                                                                        # [a, e, i, OMEGA, omega, theta]: [m, -, rad, rad, rad, rad]

## Magnetometer parameters
MAGNETOMETER_ACCURACY = math.radians(5.0)                                 # magn_acc: [rad]
SUN_SENSOR_ACCURACY = math.radians(1/8)                                   # sun_acc: [rad]
MAGNETIC_DISTURBANCE = np.array([0.0075, 0.0075, 0.0075])                 # [B_x, B_y, B_z]: [T]

## Magnetorquer parameters
DETUMBLE_TORQUE = 1e-5                                                    # G_detumble: [Nm]
PERMA_MAGNET_INDUCTION = 1.28                                             # B_mag: [T]
MAGNET_DIPOLE_DIRECTION = np.array([0, 0, 1]).T.conj()                    # u_mag: [B_x, B_y, B_z]: [T]
MAGNETIZATION_VECTOR = 0.04 * MAGNET_DIPOLE_DIRECTION                     # M_mag: [B_x, B_y, B_z]: [T]
MAGNET_VOLUME_ELEMENT = (np.linalg.norm(MAGNETIZATION_VECTOR) /
  (PERMA_MAGNET_INDUCTION * VACUUM_PERMEABILITY))                         # V_mag: [m^3]

## Hysteresis bars
ROD1_INDUCTION = 0.025                                                    # B_rod1: [T]
ROD1_THICKNESS = 1e-3                                                     # t_rod1: [m]
ROD1_VOLUME = 2 * np.pi * (ROD1_THICKNESS / 2) ** 2 * SATELLITE_LENGTH    # V_rod1: [m^3]
ROD1_DIPOLE_DIRECTION = np.array([1, 0, 0]).T.conj()                      # u_rod1: [B_x, B_y, B_z]: [T]
ROD1_MAGNETIC_MOMENT = (ROD1_INDUCTION * ROD1_VOLUME /
  VACUUM_PERMEABILITY * ROD1_DIPOLE_DIRECTION)                            # M_rod1: [B_x, B_y, B_z]: [T]

ROD2_INDUCTION = 0.025                                                    # B_rod2: [T]
ROD2_THICKNESS = 1e-3                                                     # t_rod2: [m]
ROD2_VOLUME = 2 * np.pi * (ROD2_THICKNESS / 2) ** 2 * SATELLITE_LENGTH    # V_rod2: [m^3]
ROD2_DIPOLE_DIRECTION = np.array([0, 1, 0]).T.conj()                       # u_rod2: [B_x, B_y, B_z]: [T]
ROD2_MAGNETIC_MOMENT = (ROD2_INDUCTION * ROD2_VOLUME /
  VACUUM_PERMEABILITY * ROD2_DIPOLE_DIRECTION)                            # M_rod2: [B_x, B_y, B_z]: [T]

ROD3_INDUCTION = 0.025                                                    # B_rod3: [T]
ROD3_THICKNESS = 1e-3                                                     # t_rod3: [m]
ROD3_VOLUME = 2 * np.pi * (ROD3_THICKNESS / 2) ** 2 * SATELLITE_LENGTH    # V_rod3: [m^3]
ROD3_DIPOLE_DIRECTION = np.array([0, 0, 1])                               # u_rod3: [B_x, B_y, B_z]: [T]
ROD3_MAGNETIC_MOMENT = (ROD3_INDUCTION * ROD3_VOLUME /
  VACUUM_PERMEABILITY * ROD3_DIPOLE_DIRECTION)                            # M_rod3: [B_x, B_y, B_z]: [T]

RODS_AMPERAGE = 78e-3                                                     # A_rod: [A]
RODS_VOLTAGE = 3.3                                                        # V_rod: [V]

## Gyroscope parameters
GYRO_AXIAL_INERTIA = 1e7                                                  # Ir: [kg*m^2]
GYRO_RADIAL_INERTIA = 2e7                                                 # Iz: [kg*m^2]
GYRO_ELASTIC_COEF = 10                                                    # k_gyro: [-]
GYRO_ANGULAR_VELOCITY = 1.745 * 2                                         # w_gyro: [rad/s]
GYRO_CONSTANT = 2 * np.sqrt(GYRO_ELASTIC_COEF * GYRO_RADIAL_INERTIA)      # c_gyro: [-]
OBSERVER_GAINS = -0.5                                                     # alpha   []

# Initial conditions
INITIAL_ANGULAR_VELOCITY = np.radians(np.array([10, 5, 5]))               # omega_0: [rad/s]
INITIAL_ATTITUDE_MATRIX = np.random.random((3, 3))                        # A_0: 3x3 matrix [-]

## Satellite Attitude as quaternions
q0_4 = 0.5 * math.sqrt(1 + INITIAL_ATTITUDE_MATRIX[0, 0] + INITIAL_ATTITUDE_MATRIX[1, 1] + INITIAL_ATTITUDE_MATRIX[2, 2])
q0_1 = 1 / (4 * q0_4) * (INITIAL_ATTITUDE_MATRIX[1, 2] - INITIAL_ATTITUDE_MATRIX[2, 1])
q0_2 = 1 / (4 * q0_4) * (INITIAL_ATTITUDE_MATRIX[2, 0] - INITIAL_ATTITUDE_MATRIX[0, 2])
q0_3 = 1 / (4 * q0_4) * (INITIAL_ATTITUDE_MATRIX[0, 1] - INITIAL_ATTITUDE_MATRIX[1, 0])

INITIAL_ATTITUDE_QUATERNIONS = np.array([q0_1, q0_2, q0_3, q0_4])         # q_0: [q_1, q_2, q_3, q_4]

# Inertia
INERTIA_X = 0.0022                                                         # I_x [kgm^2]
INERTIA_Y = 0.0027                                                         # I_y [kgm^2]
INERTIA_Z = 0.0028                                                         # I_z [kgm^2]
INERTIA = [INERTIA_X, INERTIA_Y, INERTIA_Z]                                # I_x, I_y, I_z: [kg*m^2]

## Time of integration
HOURS = 0.5                                                               # hours: [h]
SECONDS = HOURS * 3600                                                    # T_f: [s]
INITIAL_TIME = 0                                                          # T_0: [s]
TIME_SPAN = np.array([INITIAL_TIME, SECONDS])                             # T_span: [T_0, T_f]: [s]
DELTA_TIME = 1                                                            # delta_T: [s]
TIMESTEPS = np.linspace(0, SECONDS, round(SECONDS / DELTA_TIME))