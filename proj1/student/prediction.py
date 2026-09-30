"""Student exercise: implement the IMU prediction shared by both parts."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from support import Estimate
from filter import noise_covariances


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

 
    # linear velocity
    Rz = np.transpose(np.array([
        np.cos(yaw), -np.sin(yaw), 0 ;
        np.sin(yaw), np.cos(yaw), 0;
        0, 0, 1
    ]))
    Ry = np.transpose(np.array([
        np.cos(pitch), 0, np.sin(pitch);
        0, 1, 0;
        -np.sin(pitch), 0, np.cos(pitch);
    ]))
    Rx = np.transpose(np.array([
        1, 0,0;
        0, np.cos(roll), -np.sin(roll);
        0, np.sin(roll), np.cos(roll) ;        
    ]))


    Q,R = noise_covariances(1)

    x,y,z = estimate.state[0:3]
    roll,pitch,yaw = estimate.state[3:6]
    vx,vy,vz = estimate.state[6:9]
    gx,gy,gz = estimate.state[9:12]

    acceler_bias = estimate.state[12:]

    new_estimate = Estimate(np.zeros(12), np.zeros(12,12))

    # position
    new_estimate[0] = x + dt * vx
    new_estimate[1] = y + dt * vy
    new_estimate[2] = z + dt * vz

    # orientation
    G = np.array([
        np.cos(roll), 0, -np.cos(pitch) * np.sin(roll);
        0, 1, np.sin(pitch);
        np.sin(roll), 0, np.cos(pitch) * np.cos(roll)
    ])
    G_inv = np.inv(G)
    new_esimate[3:6] = dt * np.inv(G) @ (angular_velocity - estimate.state[9:12] - process_noise[3:6])

    # linear velocity
     
    R = np.transpose(Rz) @ np.transpose(Ry) @ np.transpose(Rx)
    new_estimate[6:9] = estimate.state[6:9] + dt * (np.array([0, 0, -9.8]) + R(np.array(acceleration) - np.array(acceler_bias) - process_noise[6:9]))


    # gyroscope bias
    new_estimate[9:12] = process_noise[9:12]

    # accelerometer bias
    new_estimate[12:] = process_noise[12:]

    # cov = F * Sigma * F^t + V Q V^T
    # F = np.eye(12) + dt * A
    

    G_prime_roll = np.array([
        0,0,np.sin(roll) * np.sin(pitch);
        0,0,np.cos(roll);
        0,0,-np.cos(pitch)*np.sin(roll)
    ])
    G_prime_pitch = np.array([
        -np.sin(pitch), 0, -np.cos(roll) * np.cos(pitch);
        0,0,0;
        np.cos(pitch), 0, -np.cos(roll)*np.sin(pitch)
    ])
    G_prime_yaw = np.zeros(3,3)

    G_inv_prime_roll = G_inv @ G_prime_roll @ G_inv
    G_inv_prime_pitch = G_inv @ G_prime_pitch @ G_inv
    G_inv_prime_yaw = G_inv @ G_prime_yaw @ G_inv

    G_inv_prime_roll_roll = G_inv_prime_roll @ estimate.state[3:6]
    G_inv_prime_pitch_pitch = G_inv_prime_pitch @ estimate.state[3:6]
    G_inv_prime_yaw_yaw = G_inv_prime_yaw @ estimate.state[3:6]

    G_inv_prime_roll_g = G_inv_prime_roll @ -estimate.state[9:12]
    G_inv_prime_pitch_g = G_inv_prime_pitch @ -estimate.state[9:12]
    G_inv_prime_yaw_g = G_inv_prime_yaw @ -estimate.state[9:12]

    Rz_dot = np.array([
        -np.sin(yaw), np.cos(yaw), 0 ;
        np.cos(yaw), -np.sin(yaw), 0;
        0, 0, 0
    ])
    Ry_dot = np.array([
        -np.sin(pitch), 0, np.cos(pitch);
        0, 0, 0;
        -np.cos(pitch), 0, -np.sin(pitch);
    ])
    Rx_dot = np.np.array([
        0, 0,0;
        0, -np.sin(roll), -np.cos(roll);
        0, np.cos(roll), -np.sin(roll) ;        
    ])

    R_dot = Rx_dot @ Ry.transpose @ Rz.transpose + 2 * Rx * Rx.transpose Ry.transpose * Rz.transpose * Rz + (Rx.transpose * Ry.transpose) * Rz_dot

    A = np.array([
        1,0,0, 0,0,0, 0,0,0,0,0,0,0,0,0;
        0,1,0, 0,0,0, 0,0,0,0,0,0,0,0,0;
        0,0,1, 0,0,0, 0,0,0,0,0,0,0,0,0;
        0,0,0, G_inv_prime_roll_roll[0], G_inv_prime_roll_roll[1], G_inv_prime_roll_roll[2], 0,0,0 G_inv_prime_roll_g[0], G_inv_prime_roll_g[1], G_inv_prime_roll_g[2];
        0,0,0, G_inv_prime_pitch_pitch[0], G_inv_prime_pitch_pitch[1], G_inv_prime_pitch_pitch[2], 0,0,0 G_inv_prime_pitch_g[0], G_inv_prime_pitch_g[1], G_inv_prime_pitch_g[2];
        0,0,0, G_inv_prime_yaw_yaw[0], G_inv_prime_yaw_yaw[1], G_inv_prime_yaw_yaw[2], 0,0,0 G_inv_prime_yaw_g[0], G_inv_prime_yaw_g[1], G_inv_prime_yaw_g[2];

    ])

    raise NotImplementedError("Complete prediction.py: predict")
