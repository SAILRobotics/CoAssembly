import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from enum import Enum

# from calib_and_object.object_pose_estimator import ObjectPoseEstimator
# OpenXR hand bones
HAND_BONES = [
    [0, 1], [0, 2], [2, 3], [3, 4], [4, 5],
    [0, 6], [6, 7], [7, 8], [8, 9],
    [0, 10], [10, 11], [11, 12], [12, 13],
    [0, 14], [14, 15], [15, 16], [16, 17],
    [0, 18], [18, 19], [19, 20], [20, 21],
]
# [0] = Wrist
# [1] = Palm
# [2–5] = Thumb (4 joints)
# [6–9] = Index
# [10–13] = Middle
# [14–17] = Ring
# [18–21] = Pinky

class RobotMode(Enum):
    IDLE = 0
    TELEOP = 1
    TELEOP_CA = 2
    FINE = 3
    
class RobotControlMode(Enum):
    FULLYMANUAL = 0
    AUTOGRASP = 1
    ONLYWHENARRIVED = 2
    PREDICTIVE = 3


def unity_to_open3d_vector(pos):
    return np.array([pos["x"], pos["z"], pos["y"]])

def unity_to_open3d_vector2(pos):
    return np.array([pos[0], pos[2], pos[1]])  # for [x, y, z] format

def unity_to_open3d_vector3(pos):
    return {
        'x': pos["x"],
        'y': pos["z"],
        'z': pos["y"]
    }

def unity_to_open3d_quaternion(q):
    w, x, y, z = q
    return [w, -x, -z, -y]

def open3d_to_unity_vector(pos):
    """Convert Open3D vector (X, Y, Z) to Unity (X, Z, Y) as np.array"""
    return np.array([pos[0], pos[2], pos[1]])



def open3d_to_unity_vector_array(points):
    """
    Convert an (N, 3) array of Open3D points to Unity format.
    Open3D (x, y, z) → Unity (x, z, y)
    """
    points = np.asarray(points)
    return np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)


def open3d_to_unity_quaternion(q):
    # Open3D uses [w, x, y, z]
    # Unity expects [w, -x, -z, -y] during import → reverse that
    w, x, y, z = q
    return [w, -x, -z, -y]  # Reversed to match Unity ↔ Open3D definition

def rotation_matrix_to_quaternion(R_mat):
    return R.from_matrix(R_mat).as_quat()  # returns [x, y, z, w]

def transform_point_world_to_robot(p_world, T_robot_in_world):
    """
    Transforms a 3D point from world coordinates to robot base frame coordinates.

    Parameters:
        p_world (np.ndarray): A 3D point in world frame (shape: [3])
        T_robot_in_world (np.ndarray): 4x4 transformation matrix of robot in world frame

    Returns:
        np.ndarray: A 3D point in robot base frame (shape: [3])
    """
    assert p_world.shape == (3,), "p_world must be a 3D point"
    assert T_robot_in_world.shape == (4, 4), "T_robot_in_world must be a 4x4 matrix"

    # Compute the inverse: World → Robot
    T_world_robot = np.linalg.inv(T_robot_in_world)

    # Convert point to homogeneous coordinates
    p_world_h = np.concatenate([p_world, [1]])

    # Transform to robot frame
    p_robot_h = T_world_robot @ p_world_h
    return p_robot_h[:3]

def transform_point_robot_to_world(p_robot, T_robot_in_world):
    """
    Transforms a 3D point from robot base frame coordinates to world coordinates.

    Parameters:
        p_robot (np.ndarray): A 3D point in robot base frame (shape: [3])
        T_robot_in_world (np.ndarray): 4x4 transformation matrix of robot in world frame

    Returns:
        np.ndarray: A 3D point in world frame (shape: [3])
    """
    assert p_robot.shape == (3,), "p_robot must be a 3D point"
    assert T_robot_in_world.shape == (4, 4), "T_robot_in_world must be a 4x4 matrix"
    # print (p_robot)
    # Convert to homogeneous coordinates
    p_robot_h = np.concatenate([p_robot, [1]])
    # print (T_robot_in_world)
    # Apply transformation
    p_world_h = T_robot_in_world @ p_robot_h
    return p_world_h[:3]

def get_2d_bbox(T, dimensions):
    # Returns a shapely Polygon for the 2D (XY) footprint of an object
    x, y = dimensions[0], dimensions[1]
    corners = np.array([
        [0, 0], [x, 0], [x, y], [0, y]
    ])
    # Rotate + translate
    R_2x2 = T[:2, :2]
    t_xy = T[:2, 3]
    transformed = (R_2x2 @ corners.T).T + t_xy
    return Polygon(transformed)

def find_the_closest_obj(T_robot_in_world, p, q, object_poses, object_geoms):
    if T_robot_in_world is None:
        print("❗ T_robot_in_world not available")
        return None

    # Robot TCP position in meters, robot frame → world frame
    p_robot_m = np.array(p[:3]) / 1000.0
    p_world = transform_point_robot_to_world(p_robot_m, T_robot_in_world)

    closest_obj = None
    closest_dist = float("inf")
    closest_obj_height = 0
    closet_obj_center = np.zeros(3)
    final_tcp_position = np.zeros(3)
    final_yaw_deg = 0.0
    interest_corner1 = np.zeros(3)
    interest_corner2 = np.zeros(3)

    ref = np.array([0.0, 1.0])  # +Y in robot base
    current_yaw = q[-1]

    for obj in object_poses.keys():
        lineset = object_geoms[obj]["box"]
        result = ObjectPoseEstimator.analyze_box_geometry(lineset)
        if result is None:
            continue

        obj_base = result["center"]
        dist = np.linalg.norm(obj_base[:2] - p_world[:2])

        dir_4_to_5 = result["dir_4_to_5"]
        dir_4_to_7 = result["dir_4_to_7"]
        corner_start_long_axis = np.array(result["v4"])
        corner_end_one_axis = np.array(result["v5"])
        corner_end_another_axis = np.array(result["v7"])

        if np.linalg.norm(dir_4_to_5) > np.linalg.norm(dir_4_to_7):
            long_axis = dir_4_to_5
            short_axis = dir_4_to_7
            long_along_45 = True
        else:
            long_axis = dir_4_to_7
            short_axis = dir_4_to_5
            long_along_45 = False

        long_axis_xy = long_axis.copy()
        long_axis_xy[2] = 0.0
        long_axis_xy_norm = long_axis_xy / (np.linalg.norm(long_axis_xy) + 1e-8)

        # Try both + and - directions
        offset_magnitude = 0.06  # 6 cm
        offset_vector_pos = offset_magnitude * long_axis_xy_norm
        offset_vector_neg = -offset_magnitude * long_axis_xy_norm

        candidate_pos_world = obj_base + offset_vector_pos
        candidate_neg_world = obj_base + offset_vector_neg

        candidate_pos_robot = transform_point_world_to_robot(candidate_pos_world, T_robot_in_world)
        candidate_neg_robot = transform_point_world_to_robot(candidate_neg_world, T_robot_in_world)

        # Compute yaw for each candidate
        def compute_yaw(from_pt, to_pt):
            v = to_pt - from_pt
            v[2] = 0.0
            v_norm = v[:2] / (np.linalg.norm(v[:2]) + 1e-8)
            yaw_rad = np.arctan2(np.cross(ref, v_norm), np.dot(ref, v_norm))
            yaw_deg = np.degrees(yaw_rad) + 90
            return (yaw_deg + 180) % 360 - 180

        yaw_pos = compute_yaw(candidate_pos_robot, candidate_neg_robot)
        yaw_neg = compute_yaw(candidate_neg_robot, candidate_pos_robot)

        delta_yaw_pos = (yaw_pos - current_yaw + 180) % 360 - 180
        delta_yaw_neg = (yaw_neg - current_yaw + 180) % 360 - 180

        # Evaluate which TCP point to use
        pos_x = candidate_pos_robot[0] if candidate_pos_robot[0] > 0 else float("inf")
        neg_x = candidate_neg_robot[0] if candidate_neg_robot[0] > 0 else float("inf")

        if pos_x < neg_x:
            selected_tcp = candidate_pos_world
            selected_yaw = yaw_pos
            selected_corners = (corner_start_long_axis, corner_end_one_axis if long_along_45 else corner_end_another_axis)
            # print(f"📍 Selected +long axis | x = {pos_x:.3f} m | Δyaw = {delta_yaw_pos:.2f}")
        else:
            selected_tcp = candidate_neg_world
            selected_yaw = yaw_neg
            selected_corners = (corner_end_one_axis if long_along_45 else corner_end_another_axis, corner_start_long_axis)
            # print(f"📍 Selected -long axis | x = {neg_x:.3f} m | Δyaw = {delta_yaw_neg:.2f}")

        if dist < closest_dist:
            closest_dist = dist
            closest_obj = obj
            closest_obj_height = result["height"]
            closet_obj_center = obj_base
            final_tcp_position = selected_tcp
            final_yaw_deg = selected_yaw
            interest_corner1, interest_corner2 = selected_corners

    return (
        closest_obj,
        closet_obj_center,
        closest_obj_height,
        final_yaw_deg,
        final_tcp_position,
        interest_corner1,
        interest_corner2
    )

def find_the_closest_obj_to_hand(T_robot_in_world, p, q, object_poses, object_geoms):
    if T_robot_in_world is None:
        print("❗ T_robot_in_world not available")
        return None

    # Robot TCP position in meters, robot frame → world frame
    p_robot_m = np.array(p[:3]) / 1000.0
    p_world = transform_point_robot_to_world(p_robot_m, T_robot_in_world)

    closest_obj = None
    closest_dist = float("inf")
    closest_obj_height = 0
    closet_obj_center = np.zeros(3)
    final_tcp_position = np.zeros(3)
    final_yaw_deg = 0.0
    interest_corner1 = np.zeros(3)
    interest_corner2 = np.zeros(3)

    ref = np.array([0.0, 1.0])  # +Y in robot base
    current_yaw = q[-1]

    for obj in object_poses.keys():
        lineset = object_geoms[obj]["box"]
        result = ObjectPoseEstimator.analyze_box_geometry(lineset)
        if result is None:
            continue

        obj_base = result["center"]
        dist = np.linalg.norm(obj_base[:2] - p_world[:2])

        dir_4_to_5 = result["dir_4_to_5"]
        dir_4_to_7 = result["dir_4_to_7"]
        corner_start_long_axis = np.array(result["v4"])
        corner_end_one_axis = np.array(result["v5"])
        corner_end_another_axis = np.array(result["v7"])

        if np.linalg.norm(dir_4_to_5) > np.linalg.norm(dir_4_to_7):
            long_axis = dir_4_to_5
            short_axis = dir_4_to_7
            long_along_45 = True
        else:
            long_axis = dir_4_to_7
            short_axis = dir_4_to_5
            long_along_45 = False

        long_axis_xy = long_axis.copy()
        long_axis_xy[2] = 0.0
        long_axis_xy_norm = long_axis_xy / (np.linalg.norm(long_axis_xy) + 1e-8)

        # Try both + and - directions
        offset_magnitude = 0.06  # 6 cm
        offset_vector_pos = offset_magnitude * long_axis_xy_norm
        offset_vector_neg = -offset_magnitude * long_axis_xy_norm

        candidate_pos_world = obj_base + offset_vector_pos
        candidate_neg_world = obj_base + offset_vector_neg

        candidate_pos_robot = transform_point_world_to_robot(candidate_pos_world, T_robot_in_world)
        candidate_neg_robot = transform_point_world_to_robot(candidate_neg_world, T_robot_in_world)

        # Compute yaw for each candidate
        def compute_yaw(from_pt, to_pt):
            v = to_pt - from_pt
            v[2] = 0.0
            v_norm = v[:2] / (np.linalg.norm(v[:2]) + 1e-8)
            yaw_rad = np.arctan2(np.cross(ref, v_norm), np.dot(ref, v_norm))
            yaw_deg = np.degrees(yaw_rad) + 90
            return (yaw_deg + 180) % 360 - 180

        yaw_pos = compute_yaw(candidate_pos_robot, candidate_neg_robot)
        yaw_neg = compute_yaw(candidate_neg_robot, candidate_pos_robot)

        delta_yaw_pos = (yaw_pos - current_yaw + 180) % 360 - 180
        delta_yaw_neg = (yaw_neg - current_yaw + 180) % 360 - 180

        # Evaluate which TCP point to use
        pos_x = candidate_pos_robot[0] if candidate_pos_robot[0] > 0 else float("inf")
        neg_x = candidate_neg_robot[0] if candidate_neg_robot[0] > 0 else float("inf")

        if pos_x < neg_x:
            selected_tcp = candidate_pos_world
            selected_yaw = yaw_pos
            selected_corners = (corner_start_long_axis, corner_end_one_axis if long_along_45 else corner_end_another_axis)
            # print(f"📍 Selected +long axis | x = {pos_x:.3f} m | Δyaw = {delta_yaw_pos:.2f}")
        else:
            selected_tcp = candidate_neg_world
            selected_yaw = yaw_neg
            selected_corners = (corner_end_one_axis if long_along_45 else corner_end_another_axis, corner_start_long_axis)
            # print(f"📍 Selected -long axis | x = {neg_x:.3f} m | Δyaw = {delta_yaw_neg:.2f}")

        if dist < closest_dist:
            closest_dist = dist
            closest_obj = obj
            closest_obj_height = result["height"]
            closet_obj_center = obj_base
            final_tcp_position = selected_tcp
            final_yaw_deg = selected_yaw
            interest_corner1, interest_corner2 = selected_corners

    return (
        closest_obj
    )


def get_optimal_robot_pose_for_object(obj_name, T_robot_in_world, q, object_poses, object_geoms):
    if T_robot_in_world is None or obj_name not in object_poses:
        print(f"❗ Invalid input or object '{obj_name}' not found.")

        
        return None

    ref = np.array([0.0, 1.0])  # +Y in robot base
    current_yaw = q[-1]

    lineset = object_geoms[obj_name]["box"]
    result = ObjectPoseEstimator.analyze_box_geometry(lineset)
    if result is None:
        return None

    obj_base = result["center"]

    dir_4_to_5 = result["dir_4_to_5"]
    dir_4_to_7 = result["dir_4_to_7"]
    corner_start_long_axis = np.array(result["v4"])
    corner_end_one_axis = np.array(result["v5"])
    corner_end_another_axis = np.array(result["v7"])

    if np.linalg.norm(dir_4_to_5) >= np.linalg.norm(dir_4_to_7):
        long_axis = dir_4_to_5
        long_along_45 = True
    else:
        long_axis = dir_4_to_7
        long_along_45 = False

    long_axis_xy = long_axis.copy()
    long_axis_xy[2] = 0.0
    long_axis_xy_norm = long_axis_xy / (np.linalg.norm(long_axis_xy) + 1e-8)

    offset_magnitude = 0.06  # 6 cm offset from object center
    offset_vector_pos = offset_magnitude * long_axis_xy_norm
    offset_vector_neg = -offset_magnitude * long_axis_xy_norm

    candidate_pos_world = obj_base + offset_vector_pos
    candidate_neg_world = obj_base + offset_vector_neg

    candidate_pos_robot = transform_point_world_to_robot(candidate_pos_world, T_robot_in_world)
    candidate_neg_robot = transform_point_world_to_robot(candidate_neg_world, T_robot_in_world)

    def compute_yaw(from_pt, to_pt):
        v = to_pt - from_pt
        v[2] = 0.0
        v_norm = v[:2] / (np.linalg.norm(v[:2]) + 1e-8)
        yaw_rad = np.arctan2(np.cross(ref, v_norm), np.dot(ref, v_norm))
        yaw_deg = np.degrees(yaw_rad) + 90
        return (yaw_deg + 180) % 360 - 180

    yaw_pos = compute_yaw(candidate_pos_robot, candidate_neg_robot)
    yaw_neg = compute_yaw(candidate_neg_robot, candidate_pos_robot)

    delta_yaw_pos = (yaw_pos - current_yaw + 180) % 360 - 180
    delta_yaw_neg = (yaw_neg - current_yaw + 180) % 360 - 180

    pos_x = candidate_pos_robot[0] if candidate_pos_robot[0] > 0 else float("inf")
    neg_x = candidate_neg_robot[0] if candidate_neg_robot[0] > 0 else float("inf")

    if pos_x < neg_x:
        selected_tcp = candidate_pos_world
        selected_yaw = yaw_pos
        selected_corners = (corner_start_long_axis, corner_end_one_axis if long_along_45 else corner_end_another_axis)
    else:
        selected_tcp = candidate_neg_world
        selected_yaw = yaw_neg
        selected_corners = (corner_end_one_axis if long_along_45 else corner_end_another_axis, corner_start_long_axis)

    return {
        "tcp_position": selected_tcp,
        "tcp_yaw_deg": selected_yaw,
        "object_center": obj_base,
        "object_height": result["height"],
        "corner1": selected_corners[0],
        "corner2": selected_corners[1]
    }

class Node:
    def __init__(self, idx, position, enforce_orientation, orientation_vector):
        self.idx = idx
        self.position = np.array(position)
        self.enforce_orientation = enforce_orientation
        self.orientation_vector = np.array(orientation_vector)

class Edge:
    def __init__(self, nodeA_idx, nodeB_idx):
        self.nodeA_idx = nodeA_idx
        self.nodeB_idx = nodeB_idx