"""
pybullet_ik.py — Minimal headless PyBullet IK/FK engine for CoAssembly.

Loads only the UR10e URDF (no gripper, no visual bodies, no debug lines).
Always runs in DIRECT mode. All visualisation is handled by Open3D.

Public API
----------
    scene = IKScene.from_calibration("calibration_data/results")
    scene.build()

    scene.update_robot(q_rad)                        # set joint angles
    T_tool0 = scene.update_tcp_bodies()              # read TCP world pose
    poses   = scene.get_arm_link_world_poses()       # for Open3D robot mesh

    # Rate-limited IK step (use in servo loop instead of servoL):
    new_q = scene.step_ik(current_q, target_pos, target_quat_xyzw, dt)
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

_MAX_JOINT_SPEED = np.deg2rad(90.0)   # rad/s — used by step_ik
_IK_ITER         = 200


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

        # Cached joint-limit data (populated by _load_robot, used by step_ik
        # and check_reachability so we don't rebuild them every call)
        self._movable:      list = []
        self._lower_limits: list = []
        self._upper_limits: list = []
        self._joint_ranges: list = []
        self._rest_poses:   list = []   # preferred arm config bias for solve_ik; empty → limit midpoints

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

    def set_scene_origin(self, T_world_anchor: np.ndarray):
        """Reposition the robot when the anchor marker is locked at runtime."""
        if not self.connected:
            return
        T_anchor = np.array(T_world_anchor, dtype=float)
        self.T_world_base = T_anchor @ self._T_calib_base
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

    def set_joint_limits(self, lower: list[float], upper: list[float],
                         degrees: bool = False):
        """Override the URDF joint limits used by step_ik and check_reachability.
        Both lists must have one value per movable joint (6 for UR10e).
        Pass degrees=True to supply values in degrees. Call after build()."""
        if len(lower) != len(self._movable) or len(upper) != len(self._movable):
            raise ValueError(
                f"Expected {len(self._movable)} values, "
                f"got lower={len(lower)} upper={len(upper)}")
        conv = np.deg2rad if degrees else float
        self._lower_limits = [float(conv(v)) for v in lower]
        self._upper_limits = [float(conv(v)) for v in upper]
        self._joint_ranges = [u - l for l, u in zip(self._lower_limits, self._upper_limits)]

    def set_rest_poses(self, q: list, degrees: bool = False) -> None:
        """Set the preferred arm configuration used as the bias seed in solve_ik.
        Pass 6 values (one per arm joint). Empty → falls back to limit midpoints."""
        conv = np.deg2rad if degrees else float
        self._rest_poses = [float(conv(v)) for v in q]

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

    def solve_ik(self, current_q: np.ndarray, target_pos,
                 target_quat_xyzw=None, pos_tol: float = 0.005,
                 orient_tol: "float | None" = None) -> np.ndarray:
        """Solve IK and return joint angles immediately (no rate limiting).
        Use for one-shot moves (moveJ); use step_ik for streaming servo loops.

        target_quat_xyzw: target orientation; None → position-only IK.
        pos_tol:    warn if FK position error exceeds this (metres).
        orient_tol: warn if FK orientation error exceeds this (radians); ignored
                    when target_quat_xyzw is None.

        Tries two seeds and picks the solution with smaller FK position error:
          1. current_q    — stays close to the current configuration
          2. _rest_poses  — biases toward the preferred arm pose (set via
                            set_rest_poses); falls back to limit midpoints."""
        if self._rest_poses:
            bias_map  = dict(zip(self.arm_indices, self._rest_poses))
            bias_rest = [float(bias_map.get(j, (l + u) / 2.0))
                         for j, l, u in zip(self._movable,
                                            self._lower_limits, self._upper_limits)]
        else:
            bias_rest = [(l + u) / 2.0
                         for l, u in zip(self._lower_limits, self._upper_limits)]

        best_q, best_err = None, float('inf')
        for seed_q in (current_q, None):   # None → use bias_rest seed
            arm_q_map  = (dict(zip(self.arm_indices, seed_q))
                          if seed_q is not None else {})
            rest_poses = [float(arm_q_map.get(j, bias_rest[i]))
                          for i, j in enumerate(self._movable)]
            ik_kwargs  = dict(
                lowerLimits=self._lower_limits, upperLimits=self._upper_limits,
                jointRanges=self._joint_ranges, restPoses=rest_poses,
                maxNumIterations=_IK_ITER, residualThreshold=1e-5)
            if target_quat_xyzw is not None:
                ik_kwargs["targetOrientation"] = target_quat_xyzw
            joint_q = p.calculateInverseKinematics(
                self.robot_id, self.tool0_link_idx, target_pos, **ik_kwargs)
            q = np.array(joint_q[:len(self.arm_indices)], dtype=np.float64)
            for idx, qi in zip(self.arm_indices, q):
                p.resetJointState(self.robot_id, idx, float(qi))
            fk  = p.getLinkState(self.robot_id, self.tool0_link_idx,
                                  computeForwardKinematics=True)
            err = np.linalg.norm(np.array(fk[4]) - np.array(target_pos))
            if err < best_err:
                best_q, best_err = q, err
                best_fk = fk

        for idx, qi in zip(self.arm_indices, best_q):
            p.resetJointState(self.robot_id, idx, float(qi))

        # Report convergence
        arm_limit_idx = [self._movable.index(j) for j in self.arm_indices]
        lower      = np.array([self._lower_limits[i] for i in arm_limit_idx])
        upper      = np.array([self._upper_limits[i] for i in arm_limit_idx])
        violations = np.sum((best_q < lower) | (best_q > upper))
        if best_err > pos_tol:
            print(f"[IK] WARNING: pos error {best_err*1000:.1f} mm > tol  "
                  f"target={[round(v*1000,1) for v in target_pos]} mm")
        else:
            print(f"[IK] Converged  pos err {best_err*1000:.1f} mm", end="")
            if violations:
                print(f"  ({violations} joint(s) outside soft limits)")
            else:
                print()

        if orient_tol is not None and target_quat_xyzw is not None:
            q_achieved = np.array(best_fk[5])
            q_target   = np.array(target_quat_xyzw)
            dot        = float(np.clip(abs(np.dot(q_achieved, q_target)), 0.0, 1.0))
            orient_err = 2.0 * np.arccos(dot)
            if orient_err > orient_tol:
                print(f"[IK] WARNING: orient error {np.rad2deg(orient_err):.1f}° "
                      f"> tol {np.rad2deg(orient_tol):.1f}°")

        return best_q

    def step_ik(self, current_q: np.ndarray, target_pos, target_quat_xyzw,
                dt: float) -> np.ndarray:
        """Solve IK for target_pos/quat and return new joint angles after one
        rate-limited step.

        restPoses=current_q biases the solution toward the current configuration,
        preventing joint wraparound and unexpected configuration flips.
        The joint delta is clamped to _MAX_JOINT_SPEED * dt so slider jitter or
        a sudden target change never snaps the arm.
        """
        arm_q_map  = dict(zip(self.arm_indices, current_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in self._movable]
        joint_q = p.calculateInverseKinematics(
            self.robot_id, self.tool0_link_idx, target_pos,
            targetOrientation=target_quat_xyzw,
            lowerLimits=self._lower_limits, upperLimits=self._upper_limits,
            jointRanges=self._joint_ranges, restPoses=rest_poses,
            maxNumIterations=_IK_ITER, residualThreshold=1e-5)
        target_q = np.array(joint_q[:len(self.arm_indices)], dtype=np.float64)
        max_step  = _MAX_JOINT_SPEED * dt
        new_q     = current_q + np.clip(target_q - current_q, -max_step, max_step)
        for idx, qi in zip(self.arm_indices, new_q):
            p.resetJointState(self.robot_id, idx, float(qi))
        return new_q

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

        T_peg   = np.array(T_pegboard_world, dtype=float)
        saved_q = self.current_q.copy()
        arm_q_map  = dict(zip(self.arm_indices, saved_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in self._movable]

        points_world: list  = []
        reach_flags:  list  = []
        n_reachable, n_total = 0, 0
        for x in np.linspace(x_min, x_max, grid_nx):
            for y in np.linspace(y_min, y_max, grid_ny):
                p_world = (T_peg @ np.array([x, y, z_offset, 1.0]))[:3]
                joint_q = p.calculateInverseKinematics(
                    self.robot_id, self.tool0_link_idx, p_world.tolist(),
                    targetOrientation=target_quat_xyzw,
                    lowerLimits=self._lower_limits, upperLimits=self._upper_limits,
                    jointRanges=self._joint_ranges, restPoses=rest_poses,
                    maxNumIterations=_IK_ITER, residualThreshold=1e-5)
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

        # Cache joint limits for step_ik and check_reachability
        n_joints = p.getNumJoints(self.robot_id)
        self._movable = [j for j in range(n_joints)
                         if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED]
        for j in self._movable:
            info = p.getJointInfo(self.robot_id, j)
            ll, ul = float(info[8]), float(info[9])
            if ul <= ll:
                ll, ul = -np.pi, np.pi
            self._lower_limits.append(ll)
            self._upper_limits.append(ul)
            self._joint_ranges.append(ul - ll)

        print(f"[IKScene] Robot loaded (id={self.robot_id}, "
              f"tool0_link={self.tool0_link_idx})")


# ===========================================================================
# Robot controller — re-exported here for convenience
# ===========================================================================
from pybullet_scene import RobotController, HandTrackController  # noqa: E402 — keep original intact
