"""Student exercise: implement the IMU prediction shared by both parts."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from support import Estimate
# from filter import noise_covariances


def predict(
    estimate: Estimate,
    angular_velocity: ArrayLike,
    acceleration: ArrayLike,
    dt: float,
    process_noise: ArrayLike,
) -> Estimate:
    """Return an Estimate with the propagated state (15,) and covariance (15, 15).

    IMU vectors have shape (3,) and dt is seconds. process_noise is the (12, 12)
    continuous-time noise intensity Q, ordered as gyroscope noise, accelerometer
    noise, gyroscope bias drift, accelerometer bias drift (three axes each).
    For gravity use 9.8 m/s^2.
    """
    # TODO: derive the dynamics and Jacobians, then propagate mean and covariance.

    # position -> linear veloticy
    # orientation -> G^{-1} (angular velocity - gyroscope bias - gyroscope noise)
    # linear velocity  -> g + R(orientation)(acceleration - accelerometer bias - noise a)
    # gyroscope bias -> n_{bg}
    # accelerometer bias -> n_{ba}

    x,y,z = estimate.state[0:3]
    roll,pitch,yaw = estimate.state[3:6] - estimate.state[9:12]
    vx,vy,vz = estimate.state[6:9]
    gx,gy,gz = estimate.state[9:12]

    acceler_bias = estimate.state[12:]

    #added bias corrected readings which werent used anywhere before
    omega = np.asarray(angular_velocity) - estimate.state[9:12]
    accel = np.asarray(acceleration) - acceler_bias

    #old G was ZXY not ZYX
    G = np.array([
        [1,0, -np.sin(pitch)],
        [0, np.cos(roll), np.sin(roll) * np.cos(pitch)],
        [0, -np.sin(roll), np.cos(roll) * np.cos(pitch)]
    ])
    G_inv = np.linalg.inv(G)
    #print("G",G)
    #print("G_inv",G_inv)

    # linear velocity
    Rz = np.transpose(np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ]))
    Ry = np.transpose(np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ]))
    Rx = np.transpose(np.array([
        [1, 0,0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]       
    ]))
    R = np.transpose(Rz) @ np.transpose(Ry) @ np.transpose(Rx)


    #recalculated these based on new G
    G_prime_roll = np.array([
        [0,0,0],
        [0, -np.sin(roll), np.cos(roll) * np.cos(pitch)],
        [0, -np.cos(roll), -np.sin(roll) * np.cos(pitch)]
    ])
    G_prime_pitch = np.array([
        [0, 0, -np.cos(pitch)],
        [0, 0, -np.sin(roll) * np.sin(pitch)],
        [0, 0, -np.cos(roll) * np.sin(pitch)]
    ])
    G_prime_yaw = np.zeros((3,3))

    #added minus sign since derive of G-1 = -G-1 dG G-1
    G_inv_prime_roll = -G_inv @ G_prime_roll @ G_inv
    G_inv_prime_pitch = -G_inv @ G_prime_pitch @ G_inv
    G_inv_prime_yaw = -G_inv @ G_prime_yaw @ G_inv

    G_inv_prime_roll_roll = G_inv_prime_roll @ omega
    G_inv_prime_pitch_pitch = G_inv_prime_pitch @ omega
    G_inv_prime_yaw_yaw = G_inv_prime_yaw @ omega

    #not needed anymore 
    #G_inv_prime_roll_g = G_inv_prime_roll @ -estimate.state[9:12]
    #G_inv_prime_pitch_g = G_inv_prime_pitch @ -estimate.state[9:12]
    #G_inv_prime_yaw_g = G_inv_prime_yaw @ -estimate.state[9:12]

    #G_inv_prime_roll_noise_g = G_inv_prime_roll @ -np.ones(3)
    #G_inv_prime_pitch_noise_g = G_inv_prime_pitch @ -np.ones(3)
    #G_inv_prime_yaw_noise_g = G_inv_prime_yaw @ -np.ones(3)

        #fixed Rz dot was missing a negative sign
    Rz_dot = np.array([
        [-np.sin(yaw), -np.cos(yaw), 0],
        [np.cos(yaw), -np.sin(yaw), 0],
        [0, 0, 0]
    ])
    Ry_dot = np.array([
        [-np.sin(pitch), 0, np.cos(pitch)],
        [0, 0, 0],
        [-np.cos(pitch), 0, -np.sin(pitch)]
    ])
    Rx_dot = np.array([
        [0, 0,0],
        [0, -np.sin(roll), -np.cos(roll)],
        [0, np.cos(roll), -np.sin(roll)]        
    ])

    #R_dot = Rx_dot @ np.transpose(Ry) @ np.transpose(Rz) + 2 * Rx * np.transpose(Rx) @ np.transpose(Ry) @ np.transpose(Rz) + (np.transpose(Rx) * np.transpose(Ry)) * Rz_dot

    #replaced D_dot with respective angle partials
    R_dot_roll = np.transpose(Rz) @ np.transpose(Ry) @ Rx_dot @ accel
    R_dot_pitch = np.transpose(Rz) @ Ry_dot @ np.transpose(Rx) @ accel
    R_dot_yaw = Rz_dot @ np.transpose(Ry) @ np.transpose(Rx) @ accel

    #corrected A 
        #G inv prime was entered as rows instead of columns
        #added R_dot_roll/pitch/yaw in coluns 3-5 this was missing 

    A = np.array([
        [0,0,0, 0,0,0, 1,0,0,0,0,0,0,0,0], # x
        [0,0,0, 0,0,0, 0,1,0,0,0,0,0,0,0], # y
        [0,0,0, 0,0,0, 0,0,1,0,0,0,0,0,0], # z
        [0,0,0, G_inv_prime_roll_roll[0], G_inv_prime_pitch_pitch[0], G_inv_prime_yaw_yaw[0], 0,0,0, -G_inv[0,0], -G_inv[0,1], -G_inv[0,2], 0,0,0], # roll
        [0,0,0, G_inv_prime_roll_roll[1], G_inv_prime_pitch_pitch[1], G_inv_prime_yaw_yaw[1], 0,0,0, -G_inv[1,0], -G_inv[1,1], -G_inv[1,2], 0,0,0], # pitch
        [0,0,0, G_inv_prime_roll_roll[2], G_inv_prime_pitch_pitch[2], G_inv_prime_yaw_yaw[2], 0,0,0, -G_inv[2,0], -G_inv[2,1], -G_inv[2,2], 0,0,0], # yaw
        [0,0,0, R_dot_roll[0], R_dot_pitch[0], R_dot_yaw[0], 0,0,0, 0,0,0, -R[0,0],-R[0,1],-R[0,2]], # vx
        [0,0,0, R_dot_roll[1], R_dot_pitch[1], R_dot_yaw[2], 0,0,0, 0,0,0, -R[1,0],-R[1,1],-R[1,2]], # vy
        [0,0,0, R_dot_roll[2], R_dot_pitch[2], R_dot_yaw[2], 0,0,0, 0,0,0, -R[2,0],-R[2,1],-R[2,2]], # vz
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # gyroscope bias x
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # gyroscope bias y
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # gyroscope bias z
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # accelerometer bias x
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # accelerometer bias y
        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0] # accelerometer bias z
    ])

    # process noise is: gyroscope noise, accelerometer noise, gyroscope bias drift, accelerometer bias drift
    U = np.array([
        [0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,0, 0,0,0],
        #
        [-G_inv[0,0], -G_inv[0,1], -G_inv[0,2], 0,0,0, 0,0,0, 0,0,0],
        [-G_inv[1,0], -G_inv[1,1], -G_inv[1,2], 0,0,0, 0,0,0, 0,0,0],
        [-G_inv[2,0], -G_inv[2,1], -G_inv[2,2], 0,0,0, 0,0,0, 0,0,0],
        #
        [0,0,0, -R[0,0],-R[0,1],-R[0,2], 0,0,0, 0,0,0],
        [0,0,0, -R[1,0],-R[1,1],-R[1,2], 0,0,0, 0,0,0],
        [0,0,0, -R[2,0],-R[2,1],-R[2,2], 0,0,0, 0,0,0],
        #
        [0,0,0, 0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1, 0,0,0],
        #
        [0,0,0, 0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,0, 0,0,1]
    ])

    g = np.array([0,0,-9.8])
    x_dot = np.concatenate([
	estimate.state[6:9],
	G_inv @ omega,
	g + R @ accel,
	np.zeros(6),
    ])

    F = np.eye(15) + dt * A

    #print("A",A)
    #print("F",F)
    #print("U",U)
    #print("process_noise",process_noise)
    #print("covariance",estimate.covariance)

    #before had state + dt * A @state which never used IMU readings (A only for covariance)
    new_estimate = Estimate(
        state=estimate.state + dt * x_dot,
        covariance=F @ estimate.covariance @ np.transpose(F) + dt * U @ process_noise @ np.transpose(U),
    ) #added dt term since Q is continuos
    return new_estimate

