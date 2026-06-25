"""
pybullet_ik.py — Minimal headless PyBullet IK/FK engine for CoAssembly.

Loads only the UR10e URDF (no gripper, no visual bodies, no debug lines).
Always runs in DIRECT mode. All visualisation is handled by Open3D.

Public API
----------
    scene = IKScene.from_calibration("calibration_data/results")
    scene.build()

    scene.update_robot(q_rad)                   # set joint angles
    T_tool0 = scene.update_tcp_bodies()          # read TCP world pose
    poses   = scene.get_arm_link_world_poses()   # for Open3D robot mesh
"""

from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as ScipyR

_ASSETS     = Path(__file__).resolve().parent / "robot_assets"
_ROBOT_URDF = _ASSETS / "ur10e.urdf"

_ARM_JOINT_NAMES = [
    b"shoulder_pan_joint", b"shoulder_lift_joint", b"elbow_joint",
    b"wrist_1_joint",      b"wrist_2_joint",       b"wrist_3_joint",
]


class IKScene:
    """Headless PyBullet scene: UR10e robot at calibrated world pose."""

    def __init__(self, T_world_base: np.ndarray):
        self.T_world_base    = np.array(T_world_base, dtype=float)
        self._T_calib_base   = self.T_world_base.copy()

        self.robot_id:       int | None = None
        self.arm_indices:    list       = []
        self._jmap_robot:    dict       = {}
        self.tool0_link_idx: int        = -1
        self.connected = False

    @classmethod
    def from_calibration(
        cls,
        results_dir: Path | str = "calibration_data/results",
    ) -> "IKScene":
        d = Path(results_dir)
        ph1 = np.load(str(d / "phase1_results.npz"), allow_pickle=True)
        return cls(T_world_base=ph1["T_world_base"])

    def build(self):
        if self.connected:
            return
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_robot()
        self.connected = True
        print("[IKScene] Built (headless).")

    def disconnect(self):
        if self.connected:
            p.disconnect()
            self.connected = False

    def set_scene_origin(self, T_world_marker10: np.ndarray):
        """Reposition the robot when marker 10 is locked at runtime."""
        if not self.connected:
            return
        T_m10 = np.array(T_world_marker10, dtype=float)
        self.T_world_base = T_m10 @ self._T_calib_base
        if self.robot_id is not None:
            T_robot = self.T_world_base.copy()
            T_robot[:3, :3] = (T_robot[:3, :3]
                               @ ScipyR.from_euler('z', 180, degrees=True).as_matrix())
            pos, quat = self._mat_to_pb(T_robot)
            p.resetBasePositionAndOrientation(self.robot_id, pos, quat)
        print(f"[IKScene] Scene origin updated → robot base {self.T_world_base[:3,3].tolist()}")

    # ── Per-frame API ─────────────────────────────────────────────────────────

    @property
    def current_q(self) -> np.ndarray:
        return np.array([p.getJointState(self.robot_id, idx)[0]
                         for idx in self.arm_indices], dtype=np.float64)

    def update_robot(self, q_rad: np.ndarray):
        for idx, q in zip(self.arm_indices, q_rad):
            p.resetJointState(self.robot_id, idx, float(q))

    def update_tcp_bodies(self) -> "np.ndarray | None":
        """Compute FK and return the 4×4 tool0 world transform, or None on failure."""
        if self.tool0_link_idx < 0 or self.robot_id is None:
            return None
        try:
            s = p.getLinkState(self.robot_id, self.tool0_link_idx,
                               computeForwardKinematics=True)
            return self._pb_to_mat(s[4], s[5])
        except RuntimeError:
            return None

    def get_arm_link_world_poses(self) -> list[np.ndarray]:
        """7 world-space 4×4 transforms for [base, shoulder, upper_arm, forearm,
        wrist1, wrist2, wrist3] — consumed by Open3D robot mesh visualisation."""
        if not self.connected or self.robot_id is None:
            return [np.eye(4, dtype=np.float64)] * 7
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        poses = [self._pb_to_mat(pos, orn)]
        for idx in self.arm_indices:
            s = p.getLinkState(self.robot_id, idx, computeForwardKinematics=True)
            poses.append(self._pb_to_mat(s[4], s[5]))
        return poses

    def check_reachability(
        self,
        T_pegboard_world: np.ndarray,
        target_quat_xyzw: np.ndarray | None = None,
        grid_nx: int = 12,
        grid_ny: int = 10,
        x_min: float = -0.553,
        x_max: float =  0.057,
        y_min: float = -0.756,
        y_max: float =  0.057,
        z_offset: float = 0.0,
        reach_tol: float = 0.02,
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Sample a grid of IK targets over the pegboard.
        Returns (n_reachable, n_total, points_world Nx3, reachable_flags N bool)."""
        if not self.connected or self.robot_id is None:
            return 0, 0, np.zeros((0, 3)), np.zeros(0, dtype=bool)

        if target_quat_xyzw is None:
            board_z = T_pegboard_world[:3, 2]
            approach = -board_z / (np.linalg.norm(board_z) + 1e-9)
            x_tmp = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(approach, x_tmp)) > 0.9:
                x_tmp = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(approach, x_tmp)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(approach, x_axis)
            R_target = np.column_stack([x_axis, y_axis, approach])
            target_quat_xyzw = ScipyR.from_matrix(R_target).as_quat()

        T_peg  = np.array(T_pegboard_world, dtype=float)
        saved_q = self.current_q.copy()

        n_joints = p.getNumJoints(self.robot_id)
        movable  = [j for j in range(n_joints)
                    if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED]
        lower_limits, upper_limits, joint_ranges = [], [], []
        arm_q_map = dict(zip(self.arm_indices, saved_q))
        rest_poses = []
        for j in movable:
            info = p.getJointInfo(self.robot_id, j)
            ll, ul = float(info[8]), float(info[9])
            if ul <= ll:
                ll, ul = -np.pi, np.pi
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)
            rest_poses.append(float(arm_q_map.get(j, 0.0)))

        points_world: list  = []
        reach_flags:  list  = []
        n_reachable, n_total = 0, 0
        for x in np.linspace(x_min, x_max, grid_nx):
            for y in np.linspace(y_min, y_max, grid_ny):
                p_world = (T_peg @ np.array([x, y, z_offset, 1.0]))[:3]
                joint_q = p.calculateInverseKinematics(
                    self.robot_id, self.tool0_link_idx, p_world.tolist(),
                    targetOrientation=target_quat_xyzw,
                    lowerLimits=lower_limits, upperLimits=upper_limits,
                    jointRanges=joint_ranges, restPoses=rest_poses,
                    maxNumIterations=200, residualThreshold=1e-5)
                for idx, q in zip(self.arm_indices, joint_q[:len(self.arm_indices)]):
                    p.resetJointState(self.robot_id, idx, float(q))
                fk = p.getLinkState(self.robot_id, self.tool0_link_idx,
                                    computeForwardKinematics=True)
                reachable = np.linalg.norm(np.array(fk[4]) - p_world) <= reach_tol
                points_world.append(p_world)
                reach_flags.append(reachable)
                n_total += 1
                if reachable:
                    n_reachable += 1

        for idx, q in zip(self.arm_indices, saved_q):
            p.resetJointState(self.robot_id, idx, float(q))

        pct = 100 * n_reachable / max(n_total, 1)
        print(f"[Reachability] {n_reachable}/{n_total} points reachable ({pct:.0f}%)")
        return (n_reachable, n_total,
                np.array(points_world, dtype=np.float64),
                np.array(reach_flags,  dtype=bool))

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mat_to_pb(T):
        return T[:3, 3].tolist(), ScipyR.from_matrix(T[:3, :3]).as_quat().tolist()

    @staticmethod
    def _pb_to_mat(pos, quat):
        T = np.eye(4)
        T[:3, :3] = ScipyR.from_quat(quat).as_matrix()
        T[:3, 3]  = np.array(pos)
        return T

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

    def _load_robot(self):
        T_robot = self.T_world_base.copy()
        T_robot[:3, :3] = (T_robot[:3, :3]
                           @ ScipyR.from_euler('z', 180, degrees=True).as_matrix())
        base_pos, base_quat = self._mat_to_pb(T_robot)
        self.robot_id = p.loadURDF(
            str(_ROBOT_URDF),
            basePosition=base_pos,
            baseOrientation=base_quat,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        self._jmap_robot    = self._get_joint_map(self.robot_id)
        self.arm_indices    = [self._jmap_robot[n] for n in _ARM_JOINT_NAMES]
        self.tool0_link_idx = self._find_link_index(self.robot_id, b"tool0")
        print(f"[IKScene] Robot loaded (id={self.robot_id}, "
              f"tool0_link={self.tool0_link_idx})")


# ===========================================================================
# Robot controller — re-exported here for convenience
# ===========================================================================
from pybullet_scene import RobotController  # noqa: E402 — keep original intact
