"""Student exercises: the two observation models share the prediction step."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from support import Estimate

# from filter import noise_covariances


def update_pose(
    estimate: Estimate, measurement: ArrayLike, measurement_noise: ArrayLike
) -> Estimate:
    """Part 1: position/orientation measurement (6,) and its covariance R (6, 6).

    Wrap orientation innovations to [-pi, pi); leave state angles unwrapped.
    """
    # TODO: build the observation model and apply the Kalman update.
    #C x_bar + v_t
    R = measurement_noise
    x = estimate.state[:6]
    W = np.eye(6)
    
    # is the partial of g w.r.t .x TODO
    C = np.array([
        [1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
        [0, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
        [0, 0, 1, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
        [0, 0, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0],
        [0, 0, 0, 0, 1, 0, 0,0,0,0,0,0,0,0,0],
        [0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0]
    ])
    # mu_bar = x
    #print("pre inverse:", C @ estimate.covariance @ np.transpose(C) + W @ R @ np.transpose(W))
    K =  estimate.covariance @ np.transpose(C) @ np.linalg.inv(C @ estimate.covariance @ np.transpose(C) + W @ R @ np.transpose(W))

    #changed to measurement - x instead of x - measurement (divergence bug)
    innovation = measurement - x
    
    for s in range(3,6):
        if innovation[s] > np.pi:
            innovation[s] -= 2*np.pi
        elif innovation[s] < -np.pi:
            innovation[s] += 2*np.pi

    mu = estimate.state + K @ (innovation)

    cov = estimate.covariance - K @ C @ estimate.covariance

    #estimate.state = mu
    #estimate.covariance=cov
    return Estimate(state=mu, covariance=cov)

    #raise NotImplementedError("Complete updates.py: update_pose")

#finished this (forgot to complete last term)
def update_velocity(
    estimate: Estimate, measurement: ArrayLike, measurement_noise: ArrayLike
) -> Estimate:
    """Part 2: world-frame velocity measurement (3,) and its covariance R (3, 3)."""
    # TODO: build the observation model and apply the Kalman update.
    R = measurement_noise 
    x = estimate.state[6:9]
    W = np.eye(3)
	
    C = np.array([
	# [1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
	# [0,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
	# [0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
	# [0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0],
	# [0,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0],
	# [0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0],
	[0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0],
	[0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0],
	[0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0]
    ])
    # print("1", C @ estimate.covariance @  np.transpose(C))
    # print("2", W @ R @ np.transpose(W))
    innovation = measurement - x

    K =  estimate.covariance @ np.transpose(C) @ np.linalg.inv(C @ estimate.covariance @ np.transpose(C) + W @ R @ np.transpose(W))

    # print(measurement)
    mu = estimate.state + K @ (innovation)

    
    cov = estimate.covariance - K @ C @ estimate.covariance

    return Estimate(state=mu, covariance=cov)


	
    
    #raise NotImplementedError("Complete updates.py: update_velocity")
