# main_setting.py
#
# Single source of truth for all IPs, ports, paths, and ArUco settings used by
# main_with_robot.py (and any other script that imports this module).
# Edit values here — no need to touch the main script.

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ── File paths ────────────────────────────────────────────────────────────────
SCENE_LAYOUT_DIR = _HERE / "scene_layout"

# ── Machine IPs ───────────────────────────────────────────────────────────────
UNITY_IP   = "192.168.50.103"   # Quest / Windows machine running Unity
ROBOT_IP   = "192.168.50.70"    # UR10e robot controller

# ── Ports (Unity → Python) ────────────────────────────────────────────────────
CAM_FEED_PORT         = 5560   # camera feed frames + intrinsics
HAND1_PORT_FROM_UNITY = 5570   # real hand tracking stream
HAND2_PORT_FROM_UNITY = 5571   # synthetic hand tracking stream
TOOL_CLICK_PORT       = 5009   # tool click / hover events from Unity
TARGET_POSE_PORT      = 5013   # manipulated TCP target pose from Unity

# ── Ports (Python → Unity) ────────────────────────────────────────────────────
WORLD_ROOT_PORT    = 5005   # world root transform
SYNTH_OBJECTS_PORT = 5006   # synthetic bounding-box objects
PEGBOARD_ROOT_PORT = 5008   # pegboard root transform
TOOL_COLOR_PORT    = 5010   # per-tool highlight colour
TOOL_LAYOUT_PORT   = 5011   # full tool layout (positions + sizes)
GRIP_STATE_PORT    = 5012   # grip state + gripper box pose
BOARD_ROOT_PORT    = 5014   # tracked board root transform
ROBOT_BASE_PORT    = 5000   # robot base initial pose (matches RobotBaseInitialPoseReceiverNetMQ)
ROBOT_JOINT_PORT   = 5001   # live arm joint angles, radians (matches RobotJointNetMQReceiver)
WORKSPACE_BOUND_PORT = 5015 # robot workspace boundary wireframe (bounds + proximity)
CENTER_EYE_OVERRIDE_PORT = 5016 # override CenterEyeAnchor pose from Python

# ── ArUco marker IDs ──────────────────────────────────────────────────────────
ANCHOR_MARKER_ID   = 100   # world frame + PyBullet scene origin
PEGBOARD_MARKER_ID = 101   # pegboard origin (top-right corner)
BOARD_MARKER_A_ID  = 102   # one large face of the tracked board
BOARD_MARKER_B_ID  = 103   # opposite large face of the tracked board

# ── ArUco marker sizes (metres) ───────────────────────────────────────────────
ANCHOR_MARKER_SIZE   = 0.100   # marker 100:  10 cm
PEGBOARD_MARKER_SIZE = 0.100   # marker 101: 10 cm
BOARD_MARKER_SIZE    = 0.100   # markers 102/103: 10 cm
WORLD_MARKER_SIZE    = 0.100   # markers 104-107: 10 cm

# ── World marker continuous relocking (secondary markers) ──────────────────────
WORLD_MARKER_IDS           = (104, 105, 106, 107)   # secondary relock markers
WORLD_MARKERS_FILE         = SCENE_LAYOUT_DIR / "world_markers_T_ref_from_marker.json"
WORLD_MARKERS_PROXIMITY_MAX = 1.0                    # metres — max distance to relock
WORLD_MARKERS_TILT_MAX_DEG = 20.0                    # degrees from vertical to relock
WORLD_MARKERS_RELOCK_COOLDOWN = 1.0                  # seconds between secondary relocks

# ── Robot workspace boundary (world frame, metres) ────────────────────────────
# Used by the SceneVis boundary box, Unity workspace publisher, and frax CBF.
# WORKSPACE_LO = [-1.0474, -0.3082, 0.05]   # [x_min, y_min, z_min]
WORKSPACE_LO = [-0.6000, -0.3082, 0.05]   # [x_min, y_min, z_min]
WORKSPACE_HI = [ 0.7942,  0.4220, 0.750]   # [x_max, y_max, z_max]

# ── Conservative UR10e joint limits (degrees) ─────────────────────────────────
# Applied to both PyBullet IK and the frax CBF safety filter.
# Adjust to match your actual robot range / mounting pose.
#            J1      J2     J3     J4      J5      J6
# JOINT_MIN_DEG = [-162.00, -90,  50,  -160,  0,  0]
# JOINT_MAX_DEG = [ -90,    -20,  150,   60,   360,   360]


JOINT_MIN_DEG = [-162.00, -90,  50,  -360,  -360,  -360]
JOINT_MAX_DEG = [ -90,    -20,  150,   360,   360,   360]

# ── Runtime defaults ──────────────────────────────────────────────────────────
SIMULATION = True   # True → fixed joint angles; False → live RTDE

# ── Scene setup flags ─────────────────────────────────────────────────────────
# If True, load T_world_base + T_tcp_handcam from calibration_data/results/
# even in simulation mode (robot appears at its real calibrated position).
# Has no effect when SIMULATION=False (calibration is always used then).
USE_CALIBRATED_ROBOT_BASE_POSE = True

# If True, automatically load the pegboard pose from
# scene_layout/T_world100_pegboard101.npz the moment marker 100 is locked,
# so you don't need to show marker 101 separately.
LOAD_PEGBOARD_FROM_FILE = True

# ── Helpers (kept for compatibility with code that calls cfg.to_unity() etc.) ─
def to_unity(port: int) -> str:
    return f"tcp://{UNITY_IP}:{port}"

