"""Part 2: feature tracking, optical-flow velocity, and three-point RANSAC."""

import cv2
import numpy as np

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

    A = np.asarray([])
    B = np.asarray([])

    # use velocity_ransac to estimate the camera-frame velocity

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

    if not ransac:
        temp1 = np.transpose(H) @ H
        temp2 = np.linalg.inv(temp1)
        temp3 = temp2 @ np.transpose(H)
        x = temp3 @ flow.ravel()

        return x    


    raise NotImplementedError("Complete flow.py: velocity_ransac")


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

    # cv2.imshow("previous", previous.image)
    # cv2.waitKey()
    

    # harris_corners = cv2.cornerHarris(previous.image, 2, 3, 0.04)
    # print("corner", harris_corners)
    # for corner in harris_corners:
    #     cv2.circle(previous.image, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
    # # corners_current = cv2.cornerHarris(current.image, 2, 3, 0.04)
    # cv2.imwrite('harris_output.jpg', previous.image)

    features_to_track = cv2.goodFeaturesToTrack(previous.image, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # for corner in features_to_track:
    #     cv2.circle(previous.image, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0), -1)
    # cv2.imwrite('good_features_output.jpg', previous.image)
    if features_to_track is None or len(features_to_track) < 3:
        return None

    next_points, status, err = cv2.calcOpticalFlowPyrLK(previous.image, current.image, features_to_track, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    good_tracks_mask = status.reshape(-1) == 1
    valid_prev_pts = features_to_track[good_tracks_mask]
    valid_next_pts = next_points[good_tracks_mask]

    # print("shape of valid_prev_pts: ", valid_prev_pts.shape)
    # print("shape of ones: ", np.ones((valid_prev_pts.shape[0], 1)).shape)
    expanded_prev_pts = np.concatenate((valid_prev_pts, np.ones((valid_prev_pts.shape[0], 1, 1))), axis=2)
    expanded_next_pts = np.concatenate((valid_next_pts, np.ones((valid_next_pts.shape[0], 1, 1))), axis=2)
    # print("shape of expanded_prev_pts: ", expanded_prev_pts.shape)

    # Compute the flow in normalized image coordinates
    raw_prev_pts = np.linalg.inv(CAMERA_MATRIX) @ expanded_prev_pts.reshape(-1, 3, 1)
   # print("raw shape:", raw_prev_pts.shape )
    raw_next_pts = np.linalg.inv(CAMERA_MATRIX) @ expanded_next_pts.reshape(-1, 3, 1)

    # print("K_inv: ", np.linalg.inv(CAMERA_MATRIX))
    # print("corrected_prev_pts[0]: ", corrected_prev_pts[0])
    # print("expanded_prev_pts[0]: ", expanded_prev_pts[0])
    # print("reshaped expanded_prev_pts[0]: ", expanded_prev_pts.reshape(-1, 3, 1)[0])
    #raise NotImplementedError("Complete flow.py: estimate_velocity")
   
    corrected_prev_pts = raw_prev_pts[:, :2, 0]
    corrected_next_pts = raw_next_pts[:, :2, 0]

    flow = corrected_next_pts - corrected_prev_pts
    flow /= (current.time - previous.time)

    #camera pose at previous.time
    R_CB = BODY_TO_CAMERA[:3, :3]
    t_CB = BODY_TO_CAMERA[:3, 3] 
    R_WC = pose.rotation @ np.transpose(R_CB)
    # print("Shape: R_WC", R_WC.shape)
    # print("Shape: pose", pose.position.shape)
    # print("shape t_CB:", t_CB.shape)
    # print("shape: t_CB  rot pose:", (pose.rotation @ t_CB).shape)
    H_WC = np.asarray(np.concatenate([np.transpose(R_WC)[:, :2], np.transpose(np.asarray([-np.transpose(R_WC) @ (pose.position + pose.rotation @ t_CB)]))], axis = 1))

    # print("H_WC Shape:", H_WC.shape)

    rays_world = H_WC @ raw_prev_pts
    depth = -H_WC[2,-1] / rays_world[:,2]

    threshold = None # Set an appropriate threshold for RANSAC
    iterations = None # Set an appropriate number of iterations for RANSAC
    x = velocity_ransac(corrected_prev_pts, flow, depth.ravel(), ransac=ransac, threshold=threshold, iterations=iterations)

    #x is the camera's velocty in the camera frame
    V = x[:3]
    omega = x[3:]

    #V + omega x t_CB. then rotate to world/body frames
    v_world = R_WC @ (V + np.cross(omega, t_CB))
    omega_body = np.linalg.inv(R_WC) @ omega

    x = np.concatenate((-v_world, omega_body))

    return x

    raise NotImplementedError("Complete flow.py: estimate_velocity")
