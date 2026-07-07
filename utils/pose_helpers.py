import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

from .unity_conversion import (
    unity_to_open3d_vector, unity_to_open3d_quaternion,
    open3d_to_unity_vector, open3d_to_unity_quaternion, HAND_BONES,
)


# =============================================================================
# Pose helpers
# =============================================================================

_R_FIX = ScipyR.from_euler('x', -90.0, degrees=True).as_matrix()


def _unity_pose_to_T(pos_xyz, rot_xyzw) -> np.ndarray:
    pos_dict = {"x": float(pos_xyz[0]), "y": float(pos_xyz[1]), "z": float(pos_xyz[2])}
    p = unity_to_open3d_vector(pos_dict)
    x, y, z, w = rot_xyzw
    q_o3d = unity_to_open3d_quaternion([float(w), float(x), float(y), float(z)])
    R_cam = ScipyR.from_quat([q_o3d[1], q_o3d[2], q_o3d[3], q_o3d[0]]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam @ _R_FIX
    T[:3, 3]  = p
    return T


def _T_to_unity_pose(T: np.ndarray):
    """Exact inverse of _unity_pose_to_T. Given a camera-convention 4x4 (open3d
    frame, with the extra -90 deg X 'R_fix' baked in), recover the raw Unity
    (pos_xyz, rot_xyzw) that would reproduce it via _unity_pose_to_T.

    T_eye_offset (the calibrated centerEyeAnchor<->left-cam tilt) is defined
    in this same R_fix-laden convention (it's computed as inv(center_T) @ cam_T
    where both center_T and cam_T come from _unity_pose_to_T). Converting a
    pose composed with it back to Unity via the generic open3d<->Unity
    conversion (which knows nothing about R_fix) would leave a stray ~90 deg
    rotation error baked in — this function undoes R_fix as well.
    """
    R_cam = T[:3, :3] @ _R_FIX.T
    q_xyzw = ScipyR.from_matrix(R_cam).as_quat()
    q_o3d_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    w, x, y, z = unity_to_open3d_quaternion(q_o3d_wxyz)   # involution -> unity wxyz
    rot_xyzw = [float(x), float(y), float(z), float(w)]
    p = T[:3, 3]
    pos_xyz = [float(p[0]), float(p[2]), float(p[1])]
    return pos_xyz, rot_xyzw


def _adapt_cx_cy(fx, fy, cx, cy, sensor_w, sensor_h, img_w, img_h):
    if sensor_w is None or sensor_h is None:
        return fx, fy, cx, cy
    crop_x = (float(sensor_w) - float(img_w)) / 2.0
    crop_y = (float(sensor_h) - float(img_h)) / 2.0
    return fx, fy, cx - crop_x, cy - crop_y


def _transform_point(T: np.ndarray, p_local) -> np.ndarray:
    p = np.array([p_local[0], p_local[1], p_local[2], 1.0], dtype=np.float64)
    return (T @ p)[:3]


# =============================================================================
# Hand joint helpers
# =============================================================================

_BONES_NP  = np.array(HAND_BONES, dtype=np.int32)
_N_JOINTS  = int(_BONES_NP.max()) + 1
_HIDDEN_PT = np.array([[0., -100., 0.]])

_JOINT_GROUP_ORDER = ["Wrist", "Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"]


def _unity_to_o3d(pts_unity: np.ndarray) -> np.ndarray:
    return pts_unity[:, [0, 2, 1]]


def _to_world(joints_unity: np.ndarray, T_world_tracking: np.ndarray) -> np.ndarray:
    pts = _unity_to_o3d(joints_unity)
    R, t = T_world_tracking[:3, :3], T_world_tracking[:3, 3]
    return (pts @ R.T) + t


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _palm_quat(pts: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Quaternion (xyzw) for gripper orientation matching a pinch grasp.

    The approach axis is the palm outward normal (cross product of the jaw axis
    with the wrist→palm direction).  The jaw is always ⊥ to the forearm
    anatomically, so this cross product is never zero regardless of wrist
    rotation angle — no singularity at ±90°.
    """
    x_axis   = _unit(pts[6] - pts[3])        # jaw: thumb MCP → index MCP
    if is_left:
        x_axis = -x_axis                      # anatomical mirror for left hand

    palm_fwd = _unit(pts[1] - pts[0])          # wrist → palm, always ⊥ to jaw
    z_axis   = -_unit(np.cross(x_axis, palm_fwd))  # inward (opposite palm face)
    y_axis   = _unit(np.cross(z_axis, x_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    q_palm   = ScipyR.from_matrix(R)
    q_offset = ScipyR.from_euler('x', -90, degrees=True)
    return (q_palm * q_offset).as_quat()  # xyzw


def _tool_grasp_quat(R_world: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) for grasping a pegboard tool.

    After Rx(-90°), the physical gripper Z = y_axis of the pre-offset matrix R.
    So y_axis must equal the inward approach direction (-R_world[:,2]).
    cross(R[:,1], R[:,0]) = -R[:,2] for any right-handed rotation matrix,
    so setting z_axis = R_world[:,1] achieves this automatically.
    """
    x_axis = _unit(R_world[:, 0])
    z_axis = _unit(R_world[:, 1])              # gives y_axis = -R_world[:,2] = inward
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))   # reorthogonalise

    R = np.column_stack([x_axis, y_axis, z_axis])
    q_offset = ScipyR.from_euler('x', -90, degrees=True)
    target_q = (ScipyR.from_matrix(R) * q_offset).as_quat()

    # Keep camera facing up — same wrist-flip logic as TCP click
    if ScipyR.from_quat(target_q).apply([0.0, 1.0, 0.0])[2] > 0:
        target_q = (ScipyR.from_quat(target_q)
                    * ScipyR.from_euler('z', 180, degrees=True)).as_quat()
    return target_q


def _extract_joints(hand_block) -> np.ndarray | None:
    if hand_block is None:
        return None
    groups = hand_block.get("groups")
    if not groups:
        return None
    joints = []
    for group_name in _JOINT_GROUP_ORDER:
        for pose in groups.get(group_name) or []:
            if pose is None:
                joints.append([0.0, 0.0, 0.0])
            else:
                pos = pose.get("position") or {}
                joints.append([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)])
    return np.array(joints, dtype=np.float64) if joints else None


# =============================================================================
# Tracked board geometry — 250 x 200 x 30 mm board with an ArUco marker
# centred on each of its two large (250x200) faces, marker Y axes aligned
# in the same world direction. Board origin = geometric centre of the box.
# Each marker is inset 1mm from its exterior face, so its centre sits
# 14mm (half the 30mm thickness, minus the 1mm inset) from the board
# origin along its own -Z axis.
#
#   _BOARD_SIZE             : (X, Y, Z) full extents in the board's local frame.
#   _T_BOARD_FROM_MARKER_A/_B : fixed transforms from each marker's frame to
#                               the board frame. Marker A defines the board's
#                               Z axis directly; marker B is on the opposite
#                               face, related by a 180° rotation about Y.
# =============================================================================

BOARD_SIZE = (0.250, 0.200, 0.030)

T_BOARD_FROM_MARKER_A = np.array([
    [1.0, 0.0,  0.0,  0.0  ],
    [0.0, 1.0,  0.0,  0.0  ],
    [0.0, 0.0,  1.0, -0.014],
    [0.0, 0.0,  0.0,  1.0  ],
], dtype=np.float64)

T_BOARD_FROM_MARKER_B = np.array([
    [-1.0, 0.0,  0.0,  0.0  ],
    [ 0.0, 1.0,  0.0,  0.0  ],
    [ 0.0, 0.0, -1.0, -0.014],
    [ 0.0, 0.0,  0.0,  1.0  ],
], dtype=np.float64)
