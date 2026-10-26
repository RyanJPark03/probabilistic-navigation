"""Part 2: feature tracking, optical-flow velocity, and three-point RANSAC."""

import cv2
import numpy as np
import random

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose


def velocity_ransac(points: np.ndarray, flow: np.ndarray, depth: np.ndarray, *,
                    ransac: bool, threshold: float, iterations: int) -> np.ndarray:
    """Return camera-frame [vx, vy, vz, wx, wy, wz] (6,).

    points and flow are (N, 2) normalized image coordinates and their rates
    (1/s); depth is (N,) positive camera-z depth in metres, with N >= 3.
    With ransac=False, fit all points by least squares. Otherwise sample
    three distinct points per trial, reject rank-deficient systems, score
    each point by its 2D flow residual norm (< threshold, in 1/s), keep the
    best consensus, and refit using all its inliers.
    """

    if not ransac:
        A = np.asarray([])
        B = np.asarray([])

        for i in range(len(points)):
            corner = points[i]
            Z = depth[i]
            A = np.append(A, [-1/Z, 0, corner[0] / Z])
            A = np.append(A, [0, -1/Z, corner[1] / Z])
            B = np.append(B, [[corner[0] * corner[1], -(1+corner[0]**2), corner[1]]])
            B = np.append(B, [[1+corner[1]**2, -corner[0]*corner[1], -corner[0]]])
        A = A.reshape(-1,3)
        B = B.reshape(-1,3)
        H = np.concatenate((A, B), axis=1)
        temp1 = np.transpose(H) @ H
        temp2 = np.linalg.inv(temp1)
        temp3 = temp2 @ np.transpose(H)
        best_x = temp3 @ flow[list(inlier_indices)].ravel()

    # use velocity_ransac to estimate the camera-frame velocity
    indices_chosen = set()

    best_inlier_count = -1
    best_x = None

    for iteration in range(iterations):
        # choose three random points
        H = np.asarray([1])
        while np.linalg.matrix_rank(H) < 6:
            random_indices = tuple(sorted(random.sample(range(len(flow)), 3)))

            A = np.asarray([])
            B = np.asarray([])

            for i in random_indices:
                corner = points[i]
                Z = depth[i]
                A = np.append(A, [-1/Z, 0, corner[0] / Z])
                A = np.append(A, [0, -1/Z, corner[1] / Z])
                B = np.append(B, [[corner[0] * corner[1], -(1+corner[0]**2), corner[1]]])
                B = np.append(B, [[1+corner[1]**2, -corner[0]*corner[1], -corner[0]]])

            A = A.reshape(-1,3)
            B = B.reshape(-1,3)
            H = np.concatenate((A, B), axis=1)

        temp1 = np.transpose(H) @ H
        temp2 = np.linalg.inv(temp1)
        temp3 = temp2 @ np.transpose(H)
        x = temp3 @ flow[list(random_indices)].ravel()

        num_inliers = 0
        inlier_indices = []

        for i in range(len(points)):
            test_A = np.asarray([])
            test_B = np.asarray([])
            corner = points[i]
            Z = depth[i]
            test_A = np.append(test_A, [-1/Z, 0, corner[0] / Z])
            test_A = np.append(test_A, [0, -1/Z, corner[1] / Z])
            test_B = np.append(test_B, [[corner[0] * corner[1], -(1+corner[0]**2), corner[1]]])
            test_B = np.append(test_B, [[1+corner[1]**2, -corner[0]*corner[1], -corner[0]]])
            test_A = test_A.reshape(-1,3)
            test_B = test_B.reshape(-1,3)
            test_H = np.concatenate((test_A, test_B), axis=1)

            error = np.linalg.norm(np.asarray(test_H @ x)[:2] - flow[i].ravel())
            if error <= threshold:
                num_inliers = num_inliers + 1
                inlier_indices.append(i)

        if num_inliers > best_inlier_count:
            best_inlier_count = num_inliers
            A = np.asarray([])
            B = np.asarray([])
            for i in inlier_indices:
                corner = points[i]
                Z = depth[i]
                A = np.append(A, [-1/Z, 0, corner[0] / Z])
                A = np.append(A, [0, -1/Z, corner[1] / Z])
                B = np.append(B, [[corner[0] * corner[1], -(1+corner[0]**2), corner[1]]])
                B = np.append(B, [[1+corner[1]**2, -corner[0]*corner[1], -corner[0]]])
            A = A.reshape(-1,3)
            B = B.reshape(-1,3)
            H = np.concatenate((A, B), axis=1)
            temp1 = np.transpose(H) @ H
            temp2 = np.linalg.inv(temp1)
            temp3 = temp2 @ np.transpose(H)
            best_x = temp3 @ flow[list(inlier_indices)].ravel()

    return best_x


def estimate_velocity(previous: Frame, current: Frame, pose: Pose, *,
                      ransac: bool) -> np.ndarray | None:
    """Return [world linear velocity, body angular velocity] (6,).

    pose is the body pose at previous.time. Detect corners in previous.image
    (including scribble), track to current.image with KLT, and discard failed
    tracks. Normalize pixels with CAMERA_MATRIX; divide displacements by
    current.time - previous.time. Intersect previous-frame rays with world
    z=0 using the camera pose to obtain depth. Fit camera motion with
    velocity_ransac(), then account for the camera/body rotation AND offset.
    Choose detector/tracker and RANSAC parameters here. Return None when
    fewer than three usable tracks remain. The runner plots interval averages
    at midpoint timestamps; use the previous pose for frame transformations.
    """

    if (current.time - previous.time) <= 0:
        return None

    features_to_track = cv2.goodFeaturesToTrack(previous.image, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if features_to_track is None or len(features_to_track) < 3:
        return None

    next_points, status, err = cv2.calcOpticalFlowPyrLK(previous.image, current.image, features_to_track, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    good_tracks_mask = status.reshape(-1) == 1
    valid_prev_pts = features_to_track[good_tracks_mask]
    valid_next_pts = next_points[good_tracks_mask]
    if len(valid_prev_pts) < 3:
        return None


    expanded_prev_pts = np.concatenate((valid_prev_pts, np.ones((valid_prev_pts.shape[0], 1, 1))), axis=2)
    expanded_next_pts = np.concatenate((valid_next_pts, np.ones((valid_next_pts.shape[0], 1, 1))), axis=2)

    # Compute the flow in normalized image coordinates
    raw_prev_pts = np.linalg.inv(CAMERA_MATRIX) @ expanded_prev_pts.reshape(-1, 3, 1)
    raw_next_pts = np.linalg.inv(CAMERA_MATRIX) @ expanded_next_pts.reshape(-1, 3, 1)
   
    corrected_prev_pts = raw_prev_pts[:, :2, 0]
    corrected_next_pts = raw_next_pts[:, :2, 0]

    flow = corrected_next_pts - corrected_prev_pts
    flow /= (current.time - previous.time)

    #camera pose at previous.time
    R_CB = BODY_TO_CAMERA[:3, :3]
    t_CB = BODY_TO_CAMERA[:3, 3] 
    R_WC = pose.rotation @ np.transpose(R_CB)
    t_BC = -np.transpose(R_CB) @ t_CB
    H_WC = np.asarray(np.concatenate([np.transpose(R_WC)[:, :2], 
    np.transpose(np.asarray([-np.transpose(R_WC) @ (pose.position + pose.rotation @ t_BC)]))], axis = 1))

    rays_world = np.linalg.inv(H_WC) @ raw_prev_pts
    depth = 1 / rays_world[:,2]

    threshold = 0.5
    iterations = 12
    x = velocity_ransac(corrected_prev_pts, flow, depth.ravel(), ransac=ransac, threshold=threshold, iterations=iterations)

    #x is the camera's velocty in the camera frame
    V = x[:3]
    omega = x[3:]

    #V + omega x t_CB. then rotate to world/body frames
    v_world = R_WC @ (V + np.cross(omega, t_CB))
    omega_body = np.transpose(R_CB) @ omega

    x = np.concatenate((v_world, omega_body))

    return x

    raise NotImplementedError("Complete flow.py: estimate_velocity")
