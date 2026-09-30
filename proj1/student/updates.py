"""Student exercises: the two observation models share the prediction step."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from support import Estimate

from filter import noise_covariances


def update_pose(
    estimate: Estimate, measurement: ArrayLike, measurement_noise: ArrayLike
) -> Estimate:
    """Part 1: position/orientation measurement (6,) and its covariance R (6, 6).

    Wrap orientation innovations to [-pi, pi); leave state angles unwrapped.
    """
    # TODO: build the observation model and apply the Kalman update.
    #C x_bar + v_t
    Q,R = noise_covariances(1)
    x = estimate.state[:6]
    C = np.eye(6)

    mu_bar = x
    cov_bar = estimate.covariance[:6,:6]

    mu = mu_bar + cov_bar @ np.transpose(C) @ np.inv(C @ cov_bar @ np.transpose(C) + R) @ (x - measurement)

    cov = cov_bar - cov_bar @ np.transpose(C) @ np.inv(C @ cov_bar @ np.transpose(C) + R)

    estimate.state[:6] = mu
    estimate.state[3:] = np.min(-np.pi, np.max(np.pi, estimate.state[3:]))
    estimate.cov[:6,:6]=cov
    return estimate # TODO where is our measurement_noise

    raise NotImplementedError("Complete updates.py: update_pose")


def update_velocity(
    estimate: Estimate, measurement: ArrayLike, measurement_noise: ArrayLike
) -> Estimate:
    """Part 2: world-frame velocity measurement (3,) and its covariance R (3, 3)."""
    # TODO: build the observation model and apply the Kalman update.
    raise NotImplementedError("Complete updates.py: update_velocity")
