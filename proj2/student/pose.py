"""Part 1: AprilTag map geometry and pose from a planar homography."""

import numpy as np
from scipy.spatial.transform import Rotation

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose

TL,TR,BR,BL = 0,1,2,3  # corner order in get_corners() and Frame.corners


def get_corners(ids: np.ndarray) -> np.ndarray:
    """Return (N, 4, 3) world coordinates in metres for the given tag IDs.

    Preserve ID order. Each tag's corners are bottom left, bottom right,
    top right, top left (p1, p2, p3, p4); the mat is the world z=0 plane.
    """
    coordinates = [] # tl, tr, br, bl; tl = x y z
    for id in ids:
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

    A = []
    for id in range(len(world_frame_corners)):
        for corner in range(4):
            A.append([world_frame_corners[id][corner][0], world_frame_corners[id][corner][1], 1, 0, 0, 0, -pixel_corners[id][corner][0] * world_frame_corners[id][corner][0], -pixel_corners[id][corner][0] * world_frame_corners[id][corner][1], -pixel_corners[id][corner][0]])
            A.append([0, 0, 0, world_frame_corners[id][corner][0], world_frame_corners[id][corner][1], 1, -pixel_corners[id][corner][1] * world_frame_corners[id][corner][0], -pixel_corners[id][corner][1] * world_frame_corners[id][corner][1], -pixel_corners[id][corner][1]])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    # print("Vt full", Vt)
    # print("Vt", Vt[-1])
    H = Vt[-1].reshape(3, 3)
    # print("H", H)
    # raise("Complete pose.py: estimate_pose")
    # print("Singular values:", S)

    RRT = np.linalg.inv(CAMERA_MATRIX) @ H
    # rrt_augmented = np.vstack((RRT, [0, 0, 1]))
    # corner = np.hstack((world_frame_corners[0][0], [1]))
    # world_frame_corners[0][0][2]=1
    # world_frame_corners[0][0].append(1)
    pretend_corner = [world_frame_corners[0][0][0], world_frame_corners[0][0][1], 1, 1]

    # reproject world corner
    print("world_frame_corners[0][0]:", pretend_corner)
    print("RRT:", RRT)
    R_test = RRT[:, :2]
    r3_test = np.cross(R_test[:, 0], R_test[:, 1])
    print("r3_test:", r3_test)
    R_test = np.hstack((R_test, r3_test.reshape(3, 1), RRT[:,2].reshape(3, 1)))
    R_test = np.vstack((R_test, [0, 0, 0, 1]))
    print("R_test:", R_test)
    temp = np.asarray(R_test @ pretend_corner)
    print("temp:", temp)
    # print("RRT @ world_frame_corners[0][0]:", temp)
    # print("norm of rrt:", np.linalg.norm(RRT))
    # print("norm of rr", np.linalg.norm(RRT[:, 0]), np.linalg.norm(RRT[:, 1]))
    if temp[2] < 0:
        print("Flipping RRT because depth is negative")
        RRT = -RRT
        # print("RRT @ world_frame_corners[0][0]:", np.asarray(RRT @ world_frame_corners[0][0]))

    R1 = RRT[:, :1]
    R2 = RRT[:, 1:2]
    t = RRT[:, 2]

    t = t / np.linalg.norm(R1)
    R1 = R1 / np.linalg.norm(R1)
    R2 = R2 / np.linalg.norm(R2)

    # if np.isclose(np.transpose(R1) @ R2, 0, atol=1e-6):
        # print("R1:", R1)
        # print("R2:", R2)
        # print("R1.T @ R2:", np.transpose(R1) @ R2)
        # raise ValueError("R1 and R2 are not orthogonal")
    # if np.isclose(np.linalg.norm(R1), np.linalg.norm(R2), atol=1e-6):
    #     print("R1 norm:", np.linalg.norm(R1))
    #     print("R2 norm:", np.linalg.norm(R2))
    #     raise ValueError("R1 and R2 do not have the same norm")
    # if np.isclose(np.linalg.norm(R1), 1.0, atol=1e-6):
    #     print("R1 norm:", np.linalg.norm(R1))
    #     raise ValueError("R1 is not a unit vector")

    
    R3 = np.cross(R1.flatten(), R2.flatten()).reshape(3, 1)

    U,S,Vt = np.linalg.svd(np.hstack((R1, R2, R3)))
    R = U @ np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, np.linalg.det(U @ Vt)]]
    ) @ Vt

    # R1 = np.transpose(R1.flatten())
    # R2 = np.transpose(R2.flatten())
    # R3 = np.transpose(R3.flatten())
    # t = np.transpose(t.flatten())

    # print("R1:", R1)
    # print("R1 shape:", R1.shape)
    # print("R2:", R2)
    # print("R2 shape:", R2.shape)
    # print("R3:", R3)
    # print("R3 shape:", R3.shape)
    # print("t:", t)
    # print("t shape:", t.shape)
    # R = np.vstack((R1, R2, R3))
    # print("R:", R)
    # print("R shape:", R.shape)
    print("R:", R)
    print("t:", t)
    orientation = np.transpose(np.vstack((np.transpose(R), t)))
    print("orientation:", orientation)
    orientation = np.vstack((orientation, [0, 0, 0, 1]))
    print("orientation:", orientation)
    # print("orientation:", orientation)
    # print("orientation shape:", orientation.shape)


    body_pose = np.linalg.inv(BODY_TO_CAMERA) @ orientation

    return Pose(position=body_pose[:3, 3], rotation=body_pose[:3, :3])
    raise NotImplementedError("Complete pose.py: estimate_pose")
