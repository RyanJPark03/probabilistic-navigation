"""Part 1: AprilTag map geometry and pose from a planar homography."""

import numpy as np
from scipy.spatial.transform import Rotation

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose

TL,TR,BR,BL = 3,2,1,0  # corner order in get_corners() and Frame.corners


def get_corners(ids: np.ndarray) -> np.ndarray:
    """Return (N, 4, 3) world coordinates in metres for the given tag IDs.

    Preserve ID order. Each tag's corners are bottom left, bottom right,
    top right, top left (p1, p2, p3, p4); the mat is the world z=0 plane.
    """
    coordinates = [] # tl, tr, br, bl; tl = x y z
    # print(ids[-1])
    for id in ids:
        # id = 27
        row = id % 12
        col = id // 12
        coordinates.append([[],[],[],[]])
        coordinates[-1][TL] = [2 * 0.152 * row, 2 * 0.152 * col, 0]
        coordinates[-1][TR] = [2 * 0.152 * row, 2 * 0.152 * col + 0.152, 0]
        coordinates[-1][BR] = [2 * 0.152 * row + 0.152, 2 * 0.152 * col + 0.152, 0]
        coordinates[-1][BL] = [2 * 0.152 * row + 0.152, 2 * 0.152 * col, 0]
        if col >= 3:
            coordinates[-1][TL][1] += 0.026
            coordinates[-1][TR][1] += 0.026
            coordinates[-1][BR][1] += 0.026
            coordinates[-1][BL][1] += 0.026
        if col >= 6:
            coordinates[-1][TL][1] += 0.026
            coordinates[-1][TR][1] += 0.026
            coordinates[-1][BR][1] += 0.026
            coordinates[-1][BL][1] += 0.026
        # print("row: ", row, " col: ", col)
        # print(coordinates[-1])
        # raise NotImplementedError("Complete pose.py: get_corners")
    return np.array(coordinates)
    raise NotImplementedError("Complete pose.py: get_corners")


def estimate_pose(frame: Frame) -> Pose | None:
    """Estimate the body origin and body-to-world rotation from tag corners.

    Use get_corners(), the calibrated camera, and a homography estimated by
    SVD. Choose the sign giving positive camera depths and a proper rotation.
    Apply BODY_TO_CAMERA to recover the body pose. Return None if no tags
    are visible. Pixel coordinates are zero-based (u right, v down).
    """
    if len(frame.ids) == 0:
        return None
    world_frame_corners = get_corners(frame.ids)
    pixel_corners = frame.corners
    # print(world_frame_corners)
    # raise NotImplementedError("Complete pose.py: estimate_pose")

    A = []
    for id in range(len(world_frame_corners)):
        for corner in range(4):
            A.append([world_frame_corners[id][corner][0], world_frame_corners[id][corner][1], 1, 0, 0, 0, -pixel_corners[id][corner][0] * world_frame_corners[id][corner][0], -pixel_corners[id][corner][0] * world_frame_corners[id][corner][1], -pixel_corners[id][corner][0]])
            A.append([0, 0, 0, world_frame_corners[id][corner][0], world_frame_corners[id][corner][1], 1, -pixel_corners[id][corner][1] * world_frame_corners[id][corner][0], -pixel_corners[id][corner][1] * world_frame_corners[id][corner][1], -pixel_corners[id][corner][1]])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    pretend_corner = [world_frame_corners[0][0][0], world_frame_corners[0][0][1], 1]

    # reproject world corner
    temp = H @ pretend_corner
    if temp[2] < 0:
        H = -H
    RRT = np.linalg.inv(CAMERA_MATRIX) @ H

    R1 = RRT[:, 0]
    R2 = RRT[:, 1]
    t = RRT[:, 2]

    t = t / (np.linalg.norm(R1))

    R3 = np.cross(R1.flatten(), R2.flatten()).reshape(3, 1)

    U,S,Vt = np.linalg.svd(np.hstack((R1.reshape(-1, 1), R2.reshape(-1, 1), R3)))
    R = U @ np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, np.linalg.det(U @ Vt)]]
    ) @ Vt

    # t = -np.linalg.inv(R) @ t

    orientation = np.transpose(np.vstack((np.transpose(R), t)))
    orientation = np.vstack((orientation, [0, 0, 0, 1]))


    body_pose = np.linalg.inv(BODY_TO_CAMERA) @ orientation
    body_pose = np.linalg.inv(body_pose)

    return Pose(position=body_pose[:3, 3], rotation=body_pose[:3, :3])
    raise NotImplementedError("Complete pose.py: estimate_pose")
