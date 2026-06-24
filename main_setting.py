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
UNITY_IP   = "192.168.50.201"   # Quest / Windows machine running Unity
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

# ── ArUco marker IDs ──────────────────────────────────────────────────────────
ANCHOR_MARKER_ID   = 10   # world frame + PyBullet scene origin
PEGBOARD_MARKER_ID = 101   # pegboard origin (top-right corner)
BOARD_MARKER_A_ID  = 102   # one large face of the tracked board
BOARD_MARKER_B_ID  = 103   # opposite large face of the tracked board

# ── ArUco marker sizes (metres) ───────────────────────────────────────────────
ANCHOR_MARKER_SIZE   = 0.100   # marker 10:  9 cm
PEGBOARD_MARKER_SIZE = 0.100   # marker 101: 10 cm
BOARD_MARKER_SIZE    = 0.100   # markers 102/103: 10 cm

# ── Runtime defaults ──────────────────────────────────────────────────────────
SIMULATION = True   # True → fixed joint angles; False → live RTDE

# ── Scene setup flags ─────────────────────────────────────────────────────────
# If True, load T_world_base + T_tcp_handcam from calibration_data/results/
# even in simulation mode (robot appears at its real calibrated position).
# Has no effect when SIMULATION=False (calibration is always used then).
USE_CALIBRATED_ROBOT_BASE_POSE = True

# If True, automatically load the pegboard pose from
# scene_layout/T_world10_pegboard101.npz the moment marker 10 is locked,
# so you don't need to show marker 101 separately.
LOAD_PEGBOARD_FROM_FILE = True

# ── Helpers (kept for compatibility with code that calls cfg.to_unity() etc.) ─
def to_unity(port: int) -> str:
    return f"tcp://{UNITY_IP}:{port}"

