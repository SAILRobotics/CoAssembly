"""
pybullet_scene.py — Simplified PyBullet visualization scene for CoAssembly.

Loads UR10e robot + Robotiq 85 gripper + adapter at calibrated world pose.
Draws world frame, robot base frame, desk camera frustum, and pegboard frame.

Usage
-----
    scene = PyBulletScene.from_calibration("calibration_data/results")
    scene.build()

    while True:
        scene.update_robot(q_rad)      # optional — set arm joint angles
        scene.update_tcp_bodies()      # move gripper/adapter to TCP via FK
        scene.update_pegboard(T_wb)    # draw/update pegboard frame when locked
"""

from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as ScipyR

# ---------------------------------------------------------------------------
# Asset paths — local robot_assets/ folder next to this file
# ---------------------------------------------------------------------------
_ASSETS              = Path(__file__).resolve().parent / "robot_assets"
_ROBOT_URDF          = _ASSETS / "ur10e.urdf"
_GRIPPER_URDF        = _ASSETS / "robotiq85_2f" / "robotiq85_2f.urdf"
_ADAPTER_STL         = _ASSETS / "OnlyAdapter.stl"
_ROBOTIQ_ADAPTER_STL = _ASSETS / "RobotiqAdapter.stl"
_PEGBOARD_OBJ        = _ASSETS / "pegboard" / "Pegboard.obj"

# Tracked board (markers 102/103) — 250 x 200 x 25 mm, drawn as a wireframe box
_BOARD_SIZE = (0.250, 0.200, 0.025)

# ---------------------------------------------------------------------------
# Joint / gripper constants
# ---------------------------------------------------------------------------
_ARM_JOINT_NAMES = [
    b"shoulder_pan_joint", b"shoulder_lift_joint", b"elbow_joint",
    b"wrist_1_joint",      b"wrist_2_joint",       b"wrist_3_joint",
]

_GRIPPER_MIMIC = {
    b"robotiq_85_left_knuckle_joint":          1.0,
    b"robotiq_85_right_knuckle_joint":        -1.0,
    b"robotiq_85_left_inner_knuckle_joint":    1.0,
    b"robotiq_85_right_inner_knuckle_joint":  -1.0,
    b"robotiq_85_left_finger_tip_joint":      -1.0,
    b"robotiq_85_right_finger_tip_joint":      1.0,
}

# Rigid offsets along the tool chain
def _T_adapter_in_tool() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = ScipyR.from_euler('xyz', [0.0, 180.0, -180.0], degrees=True).as_matrix()
    return T

def _T_robotiq_adapt_in_adapt() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = ScipyR.from_euler('y', 180.0, degrees=True).as_matrix()
    T[:3, 3]  = [0.0, 0.0, -0.0100]
    return T

def _T_gripper_in_adapt() -> np.ndarray:
    T = np.eye(4)
    T[:3, 3]  = [0.0, 0.0, 0.039]
    return T


# ===========================================================================
# PyBulletScene
# ===========================================================================

class PyBulletScene:
    """
    Minimal PyBullet scene: robot + gripper + adapter placed at calibrated pose.

    Parameters
    ----------
    T_world_base    : 4×4 SE3 — robot base in world frame.
    T_world_deskcam : 4×4 SE3 — desk camera in world frame.
    T_tcp_handcam   : 4×4 SE3 or None — hand camera in TCP frame.
    K_desk / K_hand : 3×3 intrinsic matrices for frustum drawing (optional).
    gui             : bool — open PyBullet GUI window (default True).

    The pegboard mesh is placed at runtime via update_pegboard(T) using the
    pose of ArUco marker 101 detected by _WorldAnchor.
    """

    def __init__(
        self,
        T_world_base:    np.ndarray,
        T_world_deskcam: np.ndarray,
        T_tcp_handcam:   np.ndarray | None,
        *,
        K_desk: np.ndarray | None = None,
        K_hand: np.ndarray | None = None,
        gui:    bool              = True,
    ):
        self.T_world_base    = np.array(T_world_base,    dtype=float)
        self.T_world_deskcam = np.array(T_world_deskcam, dtype=float)
        self.T_tcp_handcam   = None if T_tcp_handcam is None else np.array(T_tcp_handcam, dtype=float)

        # Calibration-time poses (anchor marker frame) — preserved for set_scene_origin()
        self._T_calib_base    = self.T_world_base.copy()
        self._T_calib_deskcam = self.T_world_deskcam.copy()
        self.K_desk = K_desk
        self.K_hand = K_hand
        self.gui    = gui

        # Tool-chain offsets
        self._adapt_in_tool    = _T_adapter_in_tool()
        self._rq_adapt_in_adapt = _T_robotiq_adapt_in_adapt()
        self._grip_in_adapt    = _T_gripper_in_adapt()

        # PyBullet body handles
        self.robot_id:            int | None = None
        self.arm_indices:         list       = []
        self._jmap_robot:         dict       = {}
        self.adapter_id:          int | None = None
        self.robotiq_adapter_id:  int | None = None
        self.gripper_id:          int | None = None
        self._jmap_gripper:       dict       = {}
        self.tool0_link_idx:      int        = -1

        # 2×2 grid of pegboard mesh bodies (loaded once, teleported when anchor locks)
        # Each tile: 12" wide × 16" tall → total board: 24" wide × 32" tall
        # Marker centre is at (0.05, 0.05) from the board's top-right corner.
        # ArUco is on the FRONT face so the mesh is shifted back by board thickness
        # so its front face sits at Z=0. Adjust _BOARD_THICKNESS if the mesh looks off.
        _W, _H          = 12 * 0.0254, 16 * 0.0254
        _X0, _Y0        = 0.05, 0.05
        _BOARD_THICKNESS = 0.0127         # 1/2 inch
        self._PEGBOARD_GRID_OFFSETS = [
            [_X0 - _W,      _Y0,      -_BOARD_THICKNESS],
            [_X0 - 2 * _W,  _Y0,      -_BOARD_THICKNESS],
            [_X0 - _W,      _Y0 - _H, -_BOARD_THICKNESS],
            [_X0 - 2 * _W,  _Y0 - _H, -_BOARD_THICKNESS],
        ]
        self._pegboard_body_ids: list[int] = []
        self._reachability_ids:  list[int] = []
        self._table_id:           int | None  = None

        # Marker wall body (flat box in marker 100's XY plane)
        self._wall_id:            int | None  = None

        # Dynamic debug line IDs
        self._tcp_frame_ids:      list | None = None
        self._hand_frust_ids:     list | None = None
        self._pegboard_frame_ids: list | None = None
        self._board_frame_ids:    list | None = None
        self._board_box_ids:      list | None = None
        self._cached_T_tool0:     np.ndarray | None = None

        self.connected = False

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        results_dir: Path | str = "calibration_data/results",
        *,
        gui: bool = True,
    ) -> "PyBulletScene":
        """Build a PyBulletScene by loading the calibration NPZ files."""
        d = Path(results_dir)
        ph1 = np.load(str(d / "phase1_results.npz"), allow_pickle=True)
        ph2 = np.load(str(d / "phase2_results.npz"), allow_pickle=True)
        return cls(
            T_world_base    = ph1["T_world_base"],
            T_world_deskcam = ph1["T_world_deskcam"],
            T_tcp_handcam   = ph2["T_tcp_handcam"],
            gui=gui,
        )

    # -----------------------------------------------------------------------
    # Build / teardown
    # -----------------------------------------------------------------------

    def build(self):
        """Connect PyBullet and load all scene bodies. Call once before the loop."""
        if self.connected:
            return

        if self.gui:
            p.connect(p.GUI)
            p.resetSimulation()
            p.removeAllUserDebugItems()
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=self.T_world_base[:3, 3].tolist())
        else:
            p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._load_robot()
        self._load_adapter()
        self._load_gripper()
        self._load_table()
        self._load_pegboard_mesh()
        self._wall_id = self._create_wall_body(width=1.5, height=1.5,
                                               color=[0.75, 0.75, 0.70, 0.30])
        self._draw_static_debug()

        self.connected = True
        print("[PyBulletScene] Built.")

    def disconnect(self):
        if self.connected:
            p.disconnect()
            self.connected = False

    def set_scene_origin(self, T_world_anchor: np.ndarray):
        """Reposition robot and desk-cam relative to the runtime anchor marker pose.

        Calibration was captured with the anchor marker as origin, so at runtime:
            T_runtime_X = T_world_anchor @ T_calib_X
        """
        if not self.connected:
            return
        T_anchor = np.array(T_world_anchor, dtype=float)
        self.T_world_base    = T_anchor @ self._T_calib_base
        self.T_world_deskcam = T_anchor @ self._T_calib_deskcam

        if self.robot_id is not None:
            T_robot = self.T_world_base.copy()
            T_robot[:3, :3] = T_robot[:3, :3] @ ScipyR.from_euler('z', 180, degrees=True).as_matrix()
            pos, quat = self._mat_to_pb(T_robot)
            p.resetBasePositionAndOrientation(self.robot_id, pos, quat)

        if self._table_id is not None:
            T_table = T_anchor.copy()
            T_table[:3, 3] = T_table[:3, 3] + T_anchor[:3, :3] @ [0.0, 0.0, -self._table_half_z]
            self._teleport(self._table_id, T_table)

        p.removeAllUserDebugItems()
        # Reset all tracked IDs so the next draw creates fresh items
        # instead of trying replaceItemUniqueId with stale IDs.
        self._tcp_frame_ids      = None
        self._hand_frust_ids     = None
        self._pegboard_frame_ids = None
        self._board_frame_ids    = None
        self._board_box_ids      = None
        self._tool_box_ids       = []
        self._reachability_ids   = []
        self._draw_static_debug()
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=self.T_world_base[:3, 3].tolist())
        print(f"[PyBulletScene] Scene origin updated. Robot base at "
              f"{self.T_world_base[:3,3].tolist()}")

    # -----------------------------------------------------------------------
    # Per-frame update API
    # -----------------------------------------------------------------------

    def step(self):
        """No-op — PyBullet GUI renders in background when bodies are updated."""
        pass

    @property
    def current_q(self) -> np.ndarray:
        """Current arm joint angles in radians read from PyBullet."""
        return np.array([p.getJointState(self.robot_id, idx)[0]
                         for idx in self.arm_indices], dtype=np.float64)

    def get_arm_link_world_poses(self) -> list[np.ndarray]:
        """7 world-space 4×4 transforms for [base, shoulder, upper_arm, forearm,
        wrist1, wrist2, wrist3]. Apply URDF visual offsets separately to get
        mesh world poses. Returns identity matrices if not connected."""
        if not self.connected or self.robot_id is None:
            return [np.eye(4, dtype=np.float64)] * 7
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        poses = [self._pb_to_mat(pos, orn)]
        for idx in self.arm_indices:
            s = p.getLinkState(self.robot_id, idx, computeForwardKinematics=True)
            poses.append(self._pb_to_mat(s[4], s[5]))
        return poses

    def check_reachability(self,
                           T_pegboard_world: np.ndarray,
                           target_quat_xyzw: np.ndarray | None = None,
                           grid_nx: int = 12,
                           grid_ny: int = 10,
                           x_min: float = -0.05,   # marker is 5 cm from left edge
                           x_max: float =  0.560,  # 24 in (0.610 m) - 0.05 m
                           y_min: float = -0.05,   # marker is 5 cm from bottom edge
                           y_max: float =  0.763,  # 32 in (0.813 m) - 0.05 m
                           z_offset: float = 0.0,
                           reach_tol: float = 0.02) -> tuple[int, int]:
        """Sample a grid over the pegboard, test IK for each point, and draw
        green (reachable) / red (unreachable) dots in the PyBullet window.
        Restores the robot to its original pose when done.
        Returns (n_reachable, n_total).
        """
        # Default: approach perpendicular to the pegboard (along its Z axis)
        if target_quat_xyzw is None:
            board_z_world = T_pegboard_world[:3, 2]
            # build a rotation where tool-Z = -board_z (approaching from front)
            approach = -board_z_world / (np.linalg.norm(board_z_world) + 1e-9)
            # arbitrary perpendicular for tool-X
            ref = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_ax = np.cross(ref, approach); x_ax /= np.linalg.norm(x_ax)
            y_ax = np.cross(approach, x_ax)
            R = np.column_stack([x_ax, y_ax, approach])
            target_quat_xyzw = ScipyR.from_matrix(R).as_quat()

        for rid in self._reachability_ids:
            try:
                p.removeUserDebugItem(rid)
            except Exception:
                pass
        self._reachability_ids = []

        saved_q = self.current_q.copy()

        # Pre-compute null-space parameters (same for every IK call)
        n_joints = p.getNumJoints(self.robot_id)
        movable  = [j for j in range(n_joints)
                    if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED]
        lower_limits, upper_limits, joint_ranges = [], [], []
        for j in movable:
            info = p.getJointInfo(self.robot_id, j)
            ll, ul = float(info[8]), float(info[9])
            if ul <= ll:
                ll, ul = -np.pi, np.pi
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)
        arm_q_map  = dict(zip(self.arm_indices, saved_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in movable]

        reachable   = []
        unreachable = []

        for xi in np.linspace(x_min, x_max, grid_nx):
            for yi in np.linspace(y_min, y_max, grid_ny):
                p_local  = np.array([xi, yi, z_offset, 1.0])
                p_world  = (T_pegboard_world @ p_local)[:3]

                joint_q = p.calculateInverseKinematics(
                    self.robot_id, self.tool0_link_idx, p_world.tolist(),
                    targetOrientation=target_quat_xyzw,
                    lowerLimits=lower_limits, upperLimits=upper_limits,
                    jointRanges=joint_ranges, restPoses=rest_poses,
                    maxNumIterations=200, residualThreshold=1e-5)

                for idx, q in zip(self.arm_indices, joint_q[:len(self.arm_indices)]):
                    p.resetJointState(self.robot_id, idx, float(q))

                fk    = p.getLinkState(self.robot_id, self.tool0_link_idx,
                                       computeForwardKinematics=True)
                error = np.linalg.norm(np.array(fk[4]) - p_world)
                (reachable if error < reach_tol else unreachable).append(p_world.tolist())

        # Restore robot
        for idx, q in zip(self.arm_indices, saved_q):
            p.resetJointState(self.robot_id, idx, float(q))

        # Visualise
        if reachable:
            self._reachability_ids.append(
                p.addUserDebugPoints(reachable,   [[0.0, 1.0, 0.0]] * len(reachable),   pointSize=8))
        if unreachable:
            self._reachability_ids.append(
                p.addUserDebugPoints(unreachable, [[1.0, 0.0, 0.0]] * len(unreachable), pointSize=8))

        n_total = len(reachable) + len(unreachable)
        pct = 100 * len(reachable) / max(n_total, 1)
        print(f"[Reachability] {len(reachable)}/{n_total} points reachable ({pct:.0f}%)")
        return len(reachable), n_total

    def update_tool_boxes(self, boxes: list) -> None:
        """Draw wireframe bounding boxes for each tool. Clears previous boxes.
        boxes: list of (pos_world, R_world, size) from _ToolLayoutManager.world_boxes().
        """
        for lid in getattr(self, '_tool_box_ids', []):
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._tool_box_ids = []

        color = [1.0, 0.8, 0.0]
        for pos, R, size in boxes:
            hw, hd, hh = size[0] / 2, size[1] / 2, size[2] / 2
            local = np.array([
                [-hw, -hd, -hh], [+hw, -hd, -hh], [+hw, +hd, -hh], [-hw, +hd, -hh],
                [-hw, -hd, +hh], [+hw, -hd, +hh], [+hw, +hd, +hh], [-hw, +hd, +hh],
            ])
            corners = (local @ R.T) + pos
            for a, b in [(0,1),(1,2),(2,3),(3,0),
                         (4,5),(5,6),(6,7),(7,4),
                         (0,4),(1,5),(2,6),(3,7)]:
                lid = p.addUserDebugLine(
                    corners[a].tolist(), corners[b].tolist(),
                    color, lineWidth=4, lifeTime=0)
                self._tool_box_ids.append(lid)

            # Local coordinate frame at centroid
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = pos
            axis_len = max(size[0], size[1]) * 0.4
            self._tool_box_ids.extend(
                PyBulletScene.draw_frame(T, length=axis_len, width=2))

    def update_robot(self, q_rad: np.ndarray):
        """Set the 6 arm joint angles in radians."""
        for idx, q in zip(self.arm_indices, q_rad):
            p.resetJointState(self.robot_id, idx, float(q))

    def update_gripper(self, drive_q: float):
        """Set Robotiq 85 drive angle (0 = open, 0.8 = closed)."""
        if self.gripper_id is not None:
            self._set_gripper(self.gripper_id, self._jmap_gripper,
                              float(np.clip(drive_q, 0.0, 0.8)))

    def update_tcp_bodies(self, life_time: float = 0.35):
        """
        Teleport adapter + gripper to current FK tool0 pose and draw TCP debug.

        life_time controls how long TCP frame debug lines persist.
        Use 0.35 s at ~5 Hz to keep lines visible but auto-clear if FK stops.
        Returns the 4×4 tool0 transform or None if FK fails.
        """
        if self.tool0_link_idx < 0 or self.robot_id is None:
            return None
        try:
            s = p.getLinkState(self.robot_id, self.tool0_link_idx,
                               computeForwardKinematics=True)
            T_tool0 = self._pb_to_mat(s[4], s[5])
            self._cached_T_tool0 = T_tool0

            T_adapt  = T_tool0 @ self._adapt_in_tool
            T_rq     = T_adapt  @ self._rq_adapt_in_adapt
            T_grip   = T_rq     @ self._grip_in_adapt

            self._teleport(self.adapter_id,        T_adapt)
            self._teleport(self.robotiq_adapter_id, T_rq)
            self._teleport(self.gripper_id,         T_grip)

            self._draw_tcp_debug(life_time)
            return T_tool0
        except RuntimeError:
            return None

    def update_pegboard(self, T_world_pegboard: np.ndarray | None, life_time: float = 0.0):
        """
        Draw/update the pegboard coordinate frame and teleport the OBJ mesh.

        Pass None to hide both.  The mesh is centred at the marker origin by
        offsetting half the board width/height in the marker's local XY plane.
        """
        if T_world_pegboard is None:
            self._hide_lines(self._pegboard_frame_ids)
            self._pegboard_frame_ids = None
            for i, body_id in enumerate(self._pegboard_body_ids):
                p.resetBasePositionAndOrientation(body_id, [0, 0, -10 - i], [0, 0, 0, 1])
            return

        self._pegboard_frame_ids = self.draw_frame(
            T_world_pegboard, length=0.10, width=3,
            ids=self._pegboard_frame_ids, life_time=life_time)

        R = T_world_pegboard[:3, :3]
        t = T_world_pegboard[:3, 3]
        # Rotate each 16"×12" tile 90° around board Z so it stands 12"×16" (portrait)
        _tile_R = R @ ScipyR.from_euler('z', 90, degrees=True).as_matrix()
        for body_id, offset in zip(self._pegboard_body_ids, self._PEGBOARD_GRID_OFFSETS):
            T_board = np.eye(4, dtype=float)
            T_board[:3, :3] = _tile_R
            T_board[:3, 3]  = t + R @ np.array(offset)
            self._teleport(body_id, T_board)

    def update_tracked_board(self, T_world_board: np.ndarray | None, life_time: float = 0.0):
        """
        Draw/update a wireframe box + coordinate frame for the tracked
        250x200x25mm board (ArUco markers 102/103). Pass None to hide both.
        """
        if T_world_board is None:
            self._hide_lines(self._board_frame_ids)
            self._hide_lines(self._board_box_ids)
            self._board_frame_ids = None
            self._board_box_ids   = None
            return

        self._board_frame_ids = self.draw_frame(
            T_world_board, length=0.10, width=3,
            ids=self._board_frame_ids, life_time=life_time)
        self._board_box_ids = self._draw_wireframe_box(
            T_world_board, _BOARD_SIZE,
            ids=self._board_box_ids, life_time=life_time,
            color=[1.0, 0.9, 0.2])

    def update_wall(self, T_world_marker: np.ndarray):
        """Teleport the marker wall (flat box in marker XY plane) to pose T."""
        if self._wall_id is not None:
            self._teleport(self._wall_id, T_world_marker)

    # -----------------------------------------------------------------------
    # Static drawing helpers (public so callers can add extra debug geometry)
    # -----------------------------------------------------------------------

    @staticmethod
    def draw_frame(T, length=0.05, width=2, ids=None, life_time: float = 0.0):
        """Draw X (red) / Y (green) / Z (blue) frame arrows at pose T."""
        pos, R = T[:3, 3], T[:3, :3]
        if ids is None or len(ids) != 3:
            ids = [-1, -1, -1]
        new_ids = []
        for i, col in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            end    = (pos + R[:, i] * length).tolist()
            old_id = int(ids[i]) if (ids[i] is not None and int(ids[i]) >= 0) else -1
            kw = {"replaceItemUniqueId": old_id} if old_id >= 0 else {}
            lid = p.addUserDebugLine(
                pos.tolist(), end, col, lineWidth=width, lifeTime=life_time, **kw)
            new_ids.append(lid)
        return new_ids

    @staticmethod
    def draw_frustum(T_world_cam, depth=0.15, color=(0.2, 0.6, 1.0), ids=None,
                     fx=900.0, fy=900.0, cx=640.0, cy=360.0,
                     w=1280, h=720, life_time: float = 0.0):
        """Draw a camera frustum as 8 debug lines (4 rays + 4 rectangle edges)."""
        corners_n = [
            np.array([(0 - cx) / fx, (0 - cy) / fy, 1.0]),
            np.array([(w - cx) / fx, (0 - cy) / fy, 1.0]),
            np.array([(w - cx) / fx, (h - cy) / fy, 1.0]),
            np.array([(0 - cx) / fx, (h - cy) / fy, 1.0]),
        ]
        R, t   = T_world_cam[:3, :3], T_world_cam[:3, 3]
        origin = t.tolist()
        pts    = [(R @ (c / np.linalg.norm(c) * depth) + t).tolist() for c in corners_n]

        if ids is None or len(ids) != 8:
            ids = [-1] * 8
        new_ids = []
        for i, pt in enumerate(pts):
            old_id = int(ids[i]) if (ids[i] is not None and int(ids[i]) >= 0) else -1
            kw = {"replaceItemUniqueId": old_id} if old_id >= 0 else {}
            lid = p.addUserDebugLine(origin, pt, color, lineWidth=1.5, lifeTime=life_time, **kw)
            new_ids.append(lid)
        for i in range(4):
            old_id = int(ids[4 + i]) if (ids[4 + i] is not None and int(ids[4 + i]) >= 0) else -1
            kw = {"replaceItemUniqueId": old_id} if old_id >= 0 else {}
            lid = p.addUserDebugLine(pts[i], pts[(i + 1) % 4], color,
                                     lineWidth=1.5, lifeTime=life_time, **kw)
            new_ids.append(lid)
        return new_ids

    @staticmethod
    def _draw_wireframe_box(T, size, ids=None, life_time: float = 0.0,
                            color=(1.0, 0.9, 0.2)):
        """Draw a 12-edge wireframe box of (X, Y, Z) `size` at pose T (local
        frame centred on the box). Reuses `ids` via replaceItemUniqueId."""
        hw, hd, hh = size[0] / 2, size[1] / 2, size[2] / 2
        local = np.array([
            [-hw, -hd, -hh], [+hw, -hd, -hh], [+hw, +hd, -hh], [-hw, +hd, -hh],
            [-hw, -hd, +hh], [+hw, -hd, +hh], [+hw, +hd, +hh], [-hw, +hd, +hh],
        ])
        R, t   = T[:3, :3], T[:3, 3]
        corners = (local @ R.T) + t

        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        if ids is None or len(ids) != len(edges):
            ids = [-1] * len(edges)
        new_ids = []
        for i, (a, b) in enumerate(edges):
            old_id = int(ids[i]) if (ids[i] is not None and int(ids[i]) >= 0) else -1
            kw = {"replaceItemUniqueId": old_id} if old_id >= 0 else {}
            lid = p.addUserDebugLine(
                corners[a].tolist(), corners[b].tolist(), color,
                lineWidth=2, lifeTime=life_time, **kw)
            new_ids.append(lid)
        return new_ids

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _mat_to_pb(T):
        return T[:3, 3].tolist(), ScipyR.from_matrix(T[:3, :3]).as_quat().tolist()

    @staticmethod
    def _pb_to_mat(pos, quat):
        T = np.eye(4)
        T[:3, :3] = ScipyR.from_quat(quat).as_matrix()
        T[:3, 3]  = np.array(pos)
        return T

    def _teleport(self, body_id, T):
        if body_id is None:
            return
        pos, quat = self._mat_to_pb(T)
        p.resetBasePositionAndOrientation(body_id, pos, quat)

    def _set_gripper(self, body_id, jmap, drive_q):
        for name, mult in _GRIPPER_MIMIC.items():
            if name in jmap:
                p.resetJointState(body_id, jmap[name], drive_q * mult)

    @staticmethod
    def _get_joint_map(body_id):
        return {p.getJointInfo(body_id, i)[1]: i
                for i in range(p.getNumJoints(body_id))}

    @staticmethod
    def _find_link_index(body_id: int, link_name: bytes) -> int:
        for i in range(p.getNumJoints(body_id)):
            if p.getJointInfo(body_id, i)[12] == link_name:
                return i
        return -1

    @staticmethod
    def _disable_collisions(body_id: int):
        """Remove body from all collision groups (visual-only ghost)."""
        for i in range(-1, p.getNumJoints(body_id)):
            p.setCollisionFilterGroupMask(body_id, i, 0, 0)

    @staticmethod
    def _recolor(body_id: int, rgba):
        """Apply a uniform RGBA color to every link of a body."""
        for i in range(-1, p.getNumJoints(body_id)):
            try:
                p.changeVisualShape(body_id, i, rgbaColor=list(rgba))
            except Exception:
                pass

    def _create_wall_body(self, width: float = 0.5, height: float = 0.5,
                          thickness: float = 0.004, color=(0.8, 0.8, 0.7, 0.35)):
        """Return a bodyId for a thin flat box (wall in local XY plane)."""
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[width / 2, height / 2, thickness / 2],
            rgbaColor=list(color))
        body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                    basePosition=[0, 0, -10])
        return body_id

    @staticmethod
    def _hide_lines(ids):
        """Move existing debug lines off-screen so they disappear."""
        if ids is None:
            return
        far, far2 = [0.0, 0.0, -50.0], [0.0, 0.0, -50.001]
        for lid in ids:
            if lid is not None and int(lid) >= 0:
                try:
                    p.addUserDebugLine(far, far2, [0, 0, 0], lineWidth=1,
                                       lifeTime=0.01, replaceItemUniqueId=int(lid))
                except Exception:
                    pass

    # -----------------------------------------------------------------------
    # Scene loading
    # -----------------------------------------------------------------------

    def _load_robot(self):
        # UR mounting convention: extra 180° Z rotation on the base frame
        T_robot = self.T_world_base.copy()
        T_robot[:3, :3] = T_robot[:3, :3] @ ScipyR.from_euler('z', 180, degrees=True).as_matrix()
        base_pos, base_quat = self._mat_to_pb(T_robot)
        self.robot_id = p.loadURDF(
            str(_ROBOT_URDF),
            basePosition=base_pos,
            baseOrientation=base_quat,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        self._jmap_robot  = self._get_joint_map(self.robot_id)
        self.arm_indices  = [self._jmap_robot[n] for n in _ARM_JOINT_NAMES]
        self.tool0_link_idx = self._find_link_index(self.robot_id, b"tool0")
        print(f"[PyBulletScene] Robot loaded (id={self.robot_id}, "
              f"tool0_link={self.tool0_link_idx})")

    def _load_adapter(self):
        vis = p.createVisualShape(
            p.GEOM_MESH, fileName=str(_ADAPTER_STL),
            meshScale=[1.0, 1.0, 1.0], rgbaColor=[0.65, 0.65, 0.65, 1.0])
        if vis < 0:
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.031, length=0.011,
                                      rgbaColor=[0.65, 0.65, 0.65, 1.0])
            print("[PyBulletScene] Adapter: using cylinder placeholder.")
        self.adapter_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                            basePosition=[0, 0, -10])
        self._disable_collisions(self.adapter_id)

        rq_vis = p.createVisualShape(
            p.GEOM_MESH, fileName=str(_ROBOTIQ_ADAPTER_STL),
            meshScale=[0.001, 0.001, 0.001], rgbaColor=[0.30, 0.30, 0.30, 1.0])
        if rq_vis >= 0:
            self.robotiq_adapter_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=rq_vis, basePosition=[0, 0, -10])
            self._disable_collisions(self.robotiq_adapter_id)
        else:
            print("[PyBulletScene] WARNING: RobotiqAdapter.stl not loaded.")

    def _load_gripper(self):
        self.gripper_id    = p.loadURDF(str(_GRIPPER_URDF), useFixedBase=False,
                                        basePosition=[0, 0, -10])
        self._jmap_gripper = self._get_joint_map(self.gripper_id)
        self._disable_collisions(self.gripper_id)
        self._recolor(self.gripper_id, [0.20, 0.70, 0.20, 0.85])
        print(f"[PyBulletScene] Gripper loaded (id={self.gripper_id})")

    def _load_pegboard_mesh(self):
        if not _PEGBOARD_OBJ.exists():
            print(f"[PyBulletScene] WARNING: pegboard OBJ not found: {_PEGBOARD_OBJ}")
            return
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(_PEGBOARD_OBJ),
            meshScale=[1.0, 1.0, 1.0],
        )
        if vis < 0:
            print("[PyBulletScene] WARNING: pegboard OBJ failed to load as visual shape.")
            return
        for i in range(4):
            body_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
                basePosition=[0, 0, -10 - i])
            self._pegboard_body_ids.append(body_id)
        print(f"[PyBulletScene] Pegboard 2×2 grid loaded (ids={self._pegboard_body_ids})")

    def _load_table(self):
        half = [1.0, 0.5, 0.0125] #half the table, all dimensions in meters and total Z thickness is 2.5 cm
        self._table_half_z = half[2]
        vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                   rgbaColor=[0.55, 0.40, 0.25, 1.0])
        col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        self._table_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                           baseCollisionShapeIndex=col,
                                           basePosition=[0.0, 0.0, -half[2]]) #puts table top at z=0

    # -----------------------------------------------------------------------
    # Static debug geometry (drawn once at build())
    # -----------------------------------------------------------------------

    def _draw_static_debug(self):
        # World origin frame (large)
        self.draw_frame(np.eye(4), length=0.12, width=4)
        p.addUserDebugText("world", [0, 0, 0.14],
                           textColorRGB=[0.9, 0.9, 0.9], textSize=1.1)

        # Robot base frame
        self.draw_frame(self.T_world_base, length=0.08, width=3)
        p.addUserDebugText("robot base",
                           self.T_world_base[:3, 3].tolist(),
                           textColorRGB=[1.0, 0.6, 0.0], textSize=1.0)

        # Desk camera — frame + frustum
        self.draw_frame(self.T_world_deskcam, length=0.05, width=2)
        p.addUserDebugText("desk cam",
                           self.T_world_deskcam[:3, 3].tolist(),
                           textColorRGB=[0.0, 0.7, 1.0], textSize=1.0)
        kw = {}
        if self.K_desk is not None:
            kw = {"fx": float(self.K_desk[0, 0]), "fy": float(self.K_desk[1, 1]),
                  "cx": float(self.K_desk[0, 2]), "cy": float(self.K_desk[1, 2])}
        self.draw_frustum(self.T_world_deskcam, depth=0.18, color=(0.0, 0.55, 1.0), **kw)

    def _draw_pegboard_grid(self, T: np.ndarray,
                            board_w: float = 0.60, board_h: float = 0.90,
                            n_cols: int = 8, n_rows: int = 12):
        """Draw a green peg-hole grid in the XY plane of the marker frame."""
        R, origin = T[:3, :3], T[:3, 3]
        x_hat = R[:, 0]   # board horizontal axis
        y_hat = R[:, 1]   # board vertical axis
        color = [0.15, 0.85, 0.25]

        # Outer border
        corners = [
            origin - x_hat * board_w / 2 - y_hat * board_h / 2,
            origin + x_hat * board_w / 2 - y_hat * board_h / 2,
            origin + x_hat * board_w / 2 + y_hat * board_h / 2,
            origin - x_hat * board_w / 2 + y_hat * board_h / 2,
        ]
        for i in range(4):
            p.addUserDebugLine(corners[i].tolist(), corners[(i + 1) % 4].tolist(),
                               color, lineWidth=2.0)

        # Grid lines — columns (vertical lines across board height)
        for ci in range(n_cols + 1):
            t = -board_w / 2 + ci * board_w / n_cols
            start = origin + x_hat * t - y_hat * board_h / 2
            end   = origin + x_hat * t + y_hat * board_h / 2
            p.addUserDebugLine(start.tolist(), end.tolist(), color, lineWidth=0.5)

        # Grid lines — rows (horizontal lines across board width)
        for ri in range(n_rows + 1):
            t = -board_h / 2 + ri * board_h / n_rows
            start = origin - x_hat * board_w / 2 + y_hat * t
            end   = origin + x_hat * board_w / 2 + y_hat * t
            p.addUserDebugLine(start.tolist(), end.tolist(), color, lineWidth=0.5)

    # -----------------------------------------------------------------------
    # TCP debug (called each frame from update_tcp_bodies)
    # -----------------------------------------------------------------------

    def _draw_tcp_debug(self, life_time: float = 0.35):
        T = self._cached_T_tool0
        if T is None:
            return

        if self._tcp_frame_ids is None or len(self._tcp_frame_ids) != 3:
            self._tcp_frame_ids = [-1, -1, -1]
        self._tcp_frame_ids = self.draw_frame(
            T, length=0.05, width=2,
            ids=self._tcp_frame_ids, life_time=life_time)

        if self.T_tcp_handcam is not None:
            T_handcam = T @ self.T_tcp_handcam
            if self._hand_frust_ids is None or len(self._hand_frust_ids) != 8:
                self._hand_frust_ids = [-1] * 8
            kw = {}
            if self.K_hand is not None:
                kw = {"fx": float(self.K_hand[0, 0]), "fy": float(self.K_hand[1, 1]),
                      "cx": float(self.K_hand[0, 2]), "cy": float(self.K_hand[1, 2])}
            self._hand_frust_ids = self.draw_frustum(
                T_handcam, depth=0.12, color=(1.0, 0.15, 0.15),
                ids=self._hand_frust_ids, life_time=life_time, **kw)


# ===========================================================================
# Robot controller
# ===========================================================================

class RobotController:
    """
    Moves the UR10e from a start joint configuration to a hardcoded target
    TCP pose using PyBullet's built-in IK and linear joint interpolation.

    Usage
    -----
        ctrl = RobotController(robot_id, end_effector_link_index, start_q_rad,
                               arm_joint_indices)
        # each frame:
        ctrl.update(robot_id, arm_indices)
    """

    _IK_ITER      = 200   # PyBullet IK solver iterations
    _INTERP_STEPS = 60    # animation frames from start → target (~2 s at 26 Hz)

    def __init__(self, robot_id: int, end_effector_link: int,
                 start_q: np.ndarray, arm_indices: list,
                 target_pos: list, target_quat_xyzw: np.ndarray | None = None,
                 straight_line: bool = False,
                 straight_line_start: "list | None" = None):
        self._start_q = np.array(start_q, dtype=np.float64)
        self._step    = 0
        self.done     = False

        if target_quat_xyzw is None:
            target_quat_xyzw = ScipyR.from_euler(
                'xyz', [180.0, 0.0, 0.0], degrees=True).as_quat()

        # Build null-space parameters (shared by both modes)
        n_joints = p.getNumJoints(robot_id)
        movable  = [j for j in range(n_joints)
                    if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
        lower_limits, upper_limits, joint_ranges = [], [], []
        for j in movable:
            info = p.getJointInfo(robot_id, j)
            ll, ul = float(info[8]), float(info[9])
            if ul <= ll:
                ll, ul = -np.pi, np.pi
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)

        arm_q_map  = dict(zip(arm_indices, start_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in movable]

        # Solve IK for target (used as joint-space target and null-space bias)
        joint_q = p.calculateInverseKinematics(
            robot_id, end_effector_link, target_pos,
            targetOrientation=target_quat_xyzw,
            lowerLimits=lower_limits, upperLimits=upper_limits,
            jointRanges=joint_ranges, restPoses=rest_poses,
            maxNumIterations=self._IK_ITER, residualThreshold=1e-5)
        self._target_q = np.array(joint_q[:len(arm_indices)], dtype=np.float64)
        print(f"[RobotController] IK solution (deg): "
              f"{np.rad2deg(self._target_q).round(1).tolist()}")

        self._straight_line = straight_line
        if straight_line:
            # Store everything needed for per-step online IK
            if straight_line_start is not None:
                self._start_pos = np.array(straight_line_start, dtype=np.float64)
            else:
                fk = p.getLinkState(robot_id, end_effector_link,
                                    computeForwardKinematics=True)
                self._start_pos = np.array(fk[4], dtype=np.float64)
            self._end_pos     = np.array(target_pos, dtype=np.float64)
            self._quat        = np.array(target_quat_xyzw, dtype=np.float64)
            self._robot_id    = robot_id
            self._ee_link     = end_effector_link
            self._movable     = movable
            self._lower       = lower_limits
            self._upper       = upper_limits
            self._ranges      = joint_ranges
            self._arm_indices = arm_indices

    def update(self, robot_id: int, arm_indices: list) -> bool:
        """Advance one interpolation step. Returns True when done."""
        if self.done:
            return True

        t = min(self._step / self._INTERP_STEPS, 1.0)

        if self._straight_line:
            # Cartesian linear interpolation — IK called online each step
            pos = ((1.0 - t) * self._start_pos + t * self._end_pos).tolist()
            arm_q_map  = dict(zip(self._arm_indices, self._target_q))
            rest_poses = [float(arm_q_map.get(j, 0.0)) for j in self._movable]
            joint_q = p.calculateInverseKinematics(
                self._robot_id, self._ee_link, pos,
                targetOrientation=self._quat,
                lowerLimits=self._lower, upperLimits=self._upper,
                jointRanges=self._ranges, restPoses=rest_poses,
                maxNumIterations=self._IK_ITER, residualThreshold=1e-5)
            for idx, qi in zip(arm_indices, joint_q[:len(arm_indices)]):
                p.resetJointState(robot_id, idx, float(qi))
        else:
            q = (1.0 - t) * self._start_q + t * self._target_q
            for idx, qi in zip(arm_indices, q):
                p.resetJointState(robot_id, idx, float(qi))

        self._step += 1
        if self._step >= self._INTERP_STEPS:
            self.done = True
        return self.done


class HandTrackController:
    """
    Continuously re-targets the UR10e's TCP toward a live (moving) target
    pose, one rate-limited IK step per call — unlike RobotController, there
    is no fixed start/end interpolation; the target may change every call.

    The joint delta is clamped to _MAX_JOINT_SPEED * dt (a real angular
    velocity limit, not a per-call constant) so a noisy or jumpy tracked
    target (e.g. hand-tracking jitter) doesn't snap the arm, and the motion
    stays consistent regardless of how fast the calling loop is ticking; it
    smoothly chases whatever target is passed to step() each frame.

    Usage
    -----
        tracker = HandTrackController(robot_id, end_effector_link, arm_indices)
        # each frame while tracking:
        new_q = tracker.step(pb_scene.current_q, target_pos, target_quat_xyzw, dt)
    """

    _IK_ITER         = 200
    _MAX_JOINT_SPEED = np.deg2rad(45.0)   # rad/sec — max joint angular velocity

    def __init__(self, robot_id: int, end_effector_link: int, arm_indices: list):
        self._robot_id    = robot_id
        self._ee_link     = end_effector_link
        self._arm_indices = arm_indices

        n_joints = p.getNumJoints(robot_id)
        self._movable = [j for j in range(n_joints)
                         if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
        self._lower, self._upper, self._ranges = [], [], []
        for j in self._movable:
            info = p.getJointInfo(robot_id, j)
            ll, ul = float(info[8]), float(info[9])
            if ul <= ll:
                ll, ul = -np.pi, np.pi
            self._lower.append(ll)
            self._upper.append(ul)
            self._ranges.append(ul - ll)

    def step(self, current_q: np.ndarray, target_pos, target_quat_xyzw,
             dt: float) -> np.ndarray:
        """Solve IK for target_pos/quat, move the arm one rate-limited step
        toward the solution, and apply it directly. `dt` is the elapsed time
        (seconds) since the previous call, used to convert _MAX_JOINT_SPEED
        into a per-step joint delta. Returns the new arm joint angles
        (radians)."""
        arm_q_map  = dict(zip(self._arm_indices, current_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in self._movable]
        joint_q = p.calculateInverseKinematics(
            self._robot_id, self._ee_link, target_pos,
            targetOrientation=target_quat_xyzw,
            lowerLimits=self._lower, upperLimits=self._upper,
            jointRanges=self._ranges, restPoses=rest_poses,
            maxNumIterations=self._IK_ITER, residualThreshold=1e-5)
        target_q = np.array(joint_q[:len(self._arm_indices)], dtype=np.float64)

        max_step = self._MAX_JOINT_SPEED * dt
        delta = np.clip(target_q - current_q, -max_step, max_step)
        new_q = current_q + delta
        for idx, qi in zip(self._arm_indices, new_q):
            p.resetJointState(self._robot_id, idx, float(qi))
        return new_q


# ===========================================================================
# Live overlay — hand skeletons and camera frustums drawn in the PyBullet GUI
# ===========================================================================

# OpenXR hand bone pairs (matches utils/unity_conversion.py HAND_BONES)
_HAND_BONES = [
    [0, 1], [0, 2], [2, 3], [3, 4], [4, 5],
    [0, 6], [6, 7], [7, 8], [8, 9],
    [0, 10], [10, 11], [11, 12], [12, 13],
    [0, 14], [14, 15], [15, 16], [16, 17],
    [0, 18], [18, 19], [19, 20], [20, 21],
]


class SceneOverlay:
    """
    Draws hand skeletons and camera/head frustums directly in the PyBullet GUI
    window each frame.  All coordinates must be in the same world frame used
    by PyBulletScene.

    Usage
    -----
        overlay = SceneOverlay()
        # each frame:
        overlay.update_hands(left_pts, right_pts)
        overlay.update_cam_frustum(T_world_cam, fx, fy, cx, cy, w, h)
        overlay.update_head(T_world_head, fx, fy, cx, cy, w, h)
    """

    _COLOR_LEFT  = [0.3, 0.6, 1.0]   # blue
    _COLOR_RIGHT = [1.0, 0.55, 0.1]  # orange
    _COLOR_HEAD  = [1.0, 0.1, 0.9]   # magenta

    _FRUSTUM_DEPTH = 0.25  # metres — how far to draw frustum edges

    def __init__(self):
        n_bones = len(_HAND_BONES)
        self._ids_l    = [-1] * n_bones
        self._ids_r    = [-1] * n_bones
        self._ids_head = [-1] * 8

    # ── Hands ─────────────────────────────────────────────────────────────────

    def _draw_hand(self, pts: np.ndarray | None, ids: list, color: list) -> list:
        if pts is None or len(pts) == 0:
            return self._hide(ids)
        for k, (i, j) in enumerate(_HAND_BONES):
            if i < len(pts) and j < len(pts):
                kw = {"replaceItemUniqueId": ids[k]} if ids[k] >= 0 else {}
                ids[k] = p.addUserDebugLine(
                    pts[i].tolist(), pts[j].tolist(), color,
                    lineWidth=2, lifeTime=0, **kw)
        return ids

    def reset_ids(self) -> None:
        """Call after p.removeAllUserDebugItems() to prevent stale-ID failures."""
        n = len(_HAND_BONES)
        self._ids_l    = [-1] * n
        self._ids_r    = [-1] * n
        self._ids_head = [-1] * 8

    def update_hands(self, left_pts: np.ndarray | None,
                     right_pts: np.ndarray | None):
        self._ids_l = self._draw_hand(left_pts,  self._ids_l, self._COLOR_LEFT)
        self._ids_r = self._draw_hand(right_pts, self._ids_r, self._COLOR_RIGHT)

    # ── Frustums ──────────────────────────────────────────────────────────────

    def _frustum_pts(self, T: np.ndarray, fx: float, fy: float,
                     w: float, h: float) -> tuple:
        """Returns (origin, [tl, tr, br, bl]) all in world space."""
        depth = self._FRUSTUM_DEPTH
        hw = (w / 2.0) / fx * depth
        hh = (h / 2.0) / fy * depth
        corners_cam = np.array([
            [-hw, -hh, depth],
            [ hw, -hh, depth],
            [ hw,  hh, depth],
            [-hw,  hh, depth],
        ])
        R, t = T[:3, :3], T[:3, 3]
        corners = (R @ corners_cam.T).T + t
        return t, corners

    def _draw_frustum(self, T: np.ndarray | None, ids: list, color: list,
                      fx: float, fy: float, w: float, h: float) -> list:
        if T is None:
            return self._hide(ids)
        origin, corners = self._frustum_pts(T, fx, fy, w, h)
        edges = [
            (origin, corners[0]), (origin, corners[1]),
            (origin, corners[2]), (origin, corners[3]),
            (corners[0], corners[1]), (corners[1], corners[2]),
            (corners[2], corners[3]), (corners[3], corners[0]),
        ]
        for k, (a, b) in enumerate(edges):
            kw = {"replaceItemUniqueId": ids[k]} if ids[k] >= 0 else {}
            ids[k] = p.addUserDebugLine(
                a.tolist(), b.tolist(), color,
                lineWidth=2, lifeTime=0, **kw)
        return ids

    def update_head(self, T: np.ndarray | None,
                    fx: float = 400., fy: float = 400.,
                    w: float = 640., h: float = 480.):
        self._ids_head = self._draw_frustum(
            T, self._ids_head, self._COLOR_HEAD, fx, fy, w, h)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hide(ids: list) -> list:
        for i, item_id in enumerate(ids):
            if item_id >= 0:
                p.removeUserDebugItem(item_id)
                ids[i] = -1
        return ids
