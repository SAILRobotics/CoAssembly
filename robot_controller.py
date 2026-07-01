"""robot_controller.py — Central robot controller for CoAssembly.

Handles all robot-related concerns in both simulation and real-robot modes:
  - PyBullet scene     : step_ik (hand tracking), waypoint runner (sim grasp)
  - RTDE receive/ctrl  : poll joint angles, servoJ, moveJ (real only)
  - IK solving         : via the shared PyBullet scene
  - Gripper            : Robotiq 2F-85 open / close (real only)
  - Grasp sequence     : approach → grasp → retract
                         sim  → PyBullet waypoint runner, driven by tick()
                         real → RTDE moveJ in a background thread
  - Unity publish      : base pose (port 5000) and joint angles (port 5001)

Typical usage in main_with_robot.py
-------------------------------------
    self.robot = RobotController(
        unity_ip      = quest_ip,
        pb_scene      = self.pb_scene,
        T_world_base  = self.pb_scene.T_world_base,
        robot_ip      = cfg.ROBOT_IP,   # omit / pass None for simulation
    )

    # Each frame:
    q = self.robot.poll_q()             # sim → pb_scene.current_q; real → RTDE
    self.robot.publish_joints(q)
    self.robot.tick()                   # advance sim waypoint runner

    # Hand tracking:
    self.robot.step_hand_track(target_pos, target_quat, dt)

    # Tool click:
    self.robot.execute_grasp(tool_data, grasp_joints=..., category='tool',
                             on_complete=lambda ok: ...)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, str(Path(__file__).parent / "robot_controller"))

# ── Optional hardware dependencies ─────────────────────────────────────────────

try:
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
    _RTDE_AVAILABLE = True
except ImportError:
    RTDEReceiveInterface = None   # type: ignore[assignment,misc]
    RTDEControlInterface = None   # type: ignore[assignment,misc]
    _RTDE_AVAILABLE = False

try:
    from robotiq_gripper import RobotiqGripper
    _GRIPPER_AVAILABLE = True
except ImportError:
    RobotiqGripper = None         # type: ignore[assignment,misc]
    _GRIPPER_AVAILABLE = False

# ── Simulation waypoint runner ──────────────────────────────────────────────────

try:
    from pybullet_scene import RobotController as _PbWaypointRunner
    _PB_RUNNER_AVAILABLE = True
except ImportError:
    _PbWaypointRunner = None      # type: ignore[assignment,misc]
    _PB_RUNNER_AVAILABLE = False


class _PbJointRunner:
    """Direct joint-space linear interpolation — sim equivalent of moveJ.

    No IK. Interpolates start_q → target_q in _INTERP_STEPS equal steps,
    calling resetJointState each frame so the PyBullet scene reflects the motion.
    """
    _INTERP_STEPS = 60

    def __init__(self, start_q: np.ndarray, target_q: np.ndarray):
        self.start_q  = np.asarray(start_q, dtype=float)
        self.target_q = np.asarray(target_q, dtype=float)
        self._step    = 0
        self.done     = False

    def update(self, robot_id: int, arm_indices: "list[int]") -> None:
        if self.done:
            return
        import pybullet as p
        self._step += 1
        t = min(self._step / self._INTERP_STEPS, 1.0)
        q = self.start_q + t * (self.target_q - self.start_q)
        for j_idx, q_j in zip(arm_indices, q):
            p.resetJointState(robot_id, j_idx, float(q_j))
        if self._step >= self._INTERP_STEPS:
            self.done = True

# ── Grasp orientation helper ────────────────────────────────────────────────────

try:
    from utils.pose_helpers import _tool_grasp_quat
except ImportError:
    def _tool_grasp_quat(R_world):  # type: ignore[misc]
        return ScipyR.from_matrix(R_world).as_quat()

# ── Unity coordinate helpers ────────────────────────────────────────────────────

try:
    from utils.unity_conversion import open3d_to_unity_vector, open3d_to_unity_quaternion
except ImportError:
    def open3d_to_unity_vector(v):       # type: ignore[misc]
        return np.array([v[0], v[2], v[1]], dtype=float)
    def open3d_to_unity_quaternion(q):   # type: ignore[misc]
        return np.array([q[0], q[2], q[1], -q[3]], dtype=float)

# ── UR10e collision sphere model ───────────────────────────────────────────────

try:
    from ur_collision_model import (
        positions_list as _UR_POSITIONS,
        radii_list     as _UR_RADII,
        pairs_sc       as _UR_SC_PAIRS,
        base_position  as _UR_BASE_POS,
        base_radius    as _UR_BASE_RADIUS,
        base_sc_idxs   as _UR_BASE_SC_IDXS,
    )
    _UR_COLLISION_AVAILABLE = True
except ImportError:
    _UR_COLLISION_AVAILABLE = False

# ── frax / CBF (optional) ─────────────────────────────────────────────────────

try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent / "frax"))
    _sys.path.insert(0, str(Path(__file__).parent / "frax" / "examples"))
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    from frax.robots.ur10e import load_ur10e
    from frax.utils.rotation_utils import orientation_error_3D
    from cbf_utils import OSCBFVelocityConfig
    from cbfpy import CBF
    _FRAX_AVAILABLE = True
except Exception:
    jnp = None                           # type: ignore[assignment]
    _FRAX_AVAILABLE = False

# Build frax-format collision dict from the UR10e model when both are available
if _FRAX_AVAILABLE and _UR_COLLISION_AVAILABLE:
    _UR_FRAX_COLLISION: "dict | None" = {
        "positions":      _UR_POSITIONS,
        "radii":          _UR_RADII,
        "root_positions": (_UR_BASE_POS,),
        "root_radii":     (_UR_BASE_RADIUS,),
        "root_sc_pairs":  tuple((0, idx) for idx in _UR_BASE_SC_IDXS),
        "root_sc_tols":   tuple(0.0 for _ in _UR_BASE_SC_IDXS),
        "body_sc_pairs":  _UR_SC_PAIRS,
        "body_sc_tols":   tuple(0.0 for _ in _UR_SC_PAIRS),
    }
else:
    _UR_FRAX_COLLISION = None

if _FRAX_AVAILABLE:
    class _RcCbfConfig(OSCBFVelocityConfig):
        """Velocity-control CBF: joint limits + floor + workspace + obstacle + self-collision.

        Uses per-link collision spheres from ur_collision_model when available, giving
        accurate floor clearance, per-link obstacle avoidance, and self-collision barriers.
        Falls back to EE-only checks when the collision model is not loaded.
        """
        def __init__(self, robot, base_pos, base_R_flat,
                     q_min, q_max, z_min, ws_lo, ws_hi,
                     obs_centers, obs_halves, obs_R_bw, alpha,
                     sc_pairs=()):
            self.base_pos    = base_pos
            self.base_R_flat = base_R_flat
            self.q_min_t     = q_min
            self.q_max_t     = q_max
            self.z_min       = z_min
            self.ws_lo       = ws_lo
            self.ws_hi       = ws_hi
            self.obs_centers = obs_centers
            self.obs_halves  = obs_halves
            self.obs_R_bw    = obs_R_bw
            self._alpha      = alpha
            self.sc_pairs_i  = tuple(int(i) for i, j in sc_pairs)
            self.sc_pairs_j  = tuple(int(j) for i, j in sc_pairs)
            self._has_links  = _UR_COLLISION_AVAILABLE
            super().__init__(robot)

        def h_1(self, z):
            q      = z
            h_jlo  = q - jnp.asarray(self.q_min_t)
            h_jhi  = jnp.asarray(self.q_max_t) - q
            base_R = jnp.asarray(self.base_R_flat).reshape(3, 3)
            base_p = jnp.asarray(self.base_pos)

            if self._has_links:
                # Per-link sphere positions in world frame
                link_pos_b, link_r = self.robot.link_collision_data(q)
                link_r     = jnp.asarray(link_r)   # radii are numpy constants — must be JAX for traced indexing
                link_pos_w = link_pos_b @ base_R.T + base_p

                # Floor clearance: every sphere must clear z_min
                h_floor = link_pos_w[:, 2] - link_r - self.z_min
                parts   = [h_jlo, h_jhi, h_floor]

                # Obstacle avoidance: all link spheres vs all obstacle boxes
                if self.obs_centers:
                    obs_c   = jnp.asarray(self.obs_centers)
                    obs_h   = jnp.asarray(self.obs_halves)
                    R_bw    = jnp.asarray(self.obs_R_bw).reshape(-1, 3, 3)
                    d_world = link_pos_w[:, None, :] - obs_c[None, :, :]
                    d_box   = jnp.einsum("oij,soj->soi", R_bw, d_world)
                    inside  = jnp.abs(d_box) - obs_h[None, :, :]
                    sdf     = jnp.linalg.norm(jnp.maximum(inside, 0.0), axis=2)
                    parts.append((sdf - link_r[:, None]).reshape(-1))

                # Workspace bounds on EE
                ee_t   = self.robot.ee_transform(q)
                ee_pos = base_R @ ee_t[:3, 3] + base_p
                parts.append(jnp.concatenate([
                    ee_pos - jnp.asarray(self.ws_lo),
                    jnp.asarray(self.ws_hi) - ee_pos,
                ]))

                # Self-collision: sphere pairs must keep a positive margin
                if self.sc_pairs_i:
                    sc_i = jnp.asarray(self.sc_pairs_i)
                    sc_j = jnp.asarray(self.sc_pairs_j)
                    dist = jnp.linalg.norm(link_pos_w[sc_i] - link_pos_w[sc_j], axis=1)
                    parts.append(dist - link_r[sc_i] - link_r[sc_j])
            else:
                # Fallback: EE-only constraints (no collision model loaded)
                ee_t   = self.robot.ee_transform(q)
                ee_pos = base_R @ ee_t[:3, 3] + base_p
                h_floor = jnp.array([ee_pos[2] - self.z_min])
                parts   = [h_jlo, h_jhi, h_floor, jnp.concatenate([
                    ee_pos - jnp.asarray(self.ws_lo),
                    jnp.asarray(self.ws_hi) - ee_pos,
                ])]
                if self.obs_centers:
                    obs_c   = jnp.asarray(self.obs_centers)
                    obs_h   = jnp.asarray(self.obs_halves)
                    R_bw    = jnp.asarray(self.obs_R_bw).reshape(-1, 3, 3)
                    d_world = ee_pos[None, :] - obs_c
                    d_box   = jnp.einsum("oij,oj->oi", R_bw, d_world)
                    inside  = jnp.abs(d_box) - obs_h
                    sdf     = jnp.linalg.norm(jnp.maximum(inside, 0.0), axis=1)
                    parts.append(sdf - 0.05)

            return jnp.concatenate(parts)

        def alpha(self, h):
            return self._alpha * h


# =============================================================================
# _FraxController — standalone frax OSC+CBF wrapper
# =============================================================================

if _FRAX_AVAILABLE:
    class _FraxController:
        """Self-contained OSC+CBF velocity controller for the UR10e.

        All frax and CBF parameters live here; RobotController only holds a
        reference to an instance and calls ``servo_step`` / ``ee_world_pos``.
        """

        def __init__(
            self,
            urdf_path:      str,
            T_world_base:   np.ndarray,
            kp_pos:         float = 100.0,
            kp_ori:         float = 50.0,
            qdot_max:       float = 1.5,
            q_min:          "list | None" = None,
            q_max:          "list | None" = None,
            cbf_alpha:      float = 10.0,
            z_min:          float = -0.05,
            ws_lo:          "list | None" = None,
            ws_hi:          "list | None" = None,
            obstacle_boxes: "list | None" = None,
        ) -> None:
            self._qdot_max = float(qdot_max)

            # Base frame: frax URDF convention has a Rz(180°) offset
            _Rz180       = ScipyR.from_euler('z', np.pi).as_matrix()
            self._base_pos = np.array(T_world_base[:3, 3], float)
            self._base_R   = np.array(T_world_base[:3, :3], float) @ _Rz180

            # Load robot model with collision spheres when available
            self.robot = load_ur10e(str(urdf_path),
                                    collision_data=_UR_FRAX_COLLISION)
            _col_status = "with collision model" if _UR_COLLISION_AVAILABLE else "EE-only"
            print(f"[FraxController] Loaded UR10e: {self.robot.num_joints} joints ({_col_status})")

            # JIT-compile nominal OSC
            kp_task = jnp.array([float(kp_pos)] * 3 + [float(kp_ori)] * 3)

            @jax.jit
            def _osc(ee_pos, ee_rot, des_pos, des_rot,
                     des_vel, des_omega, J, M_inv):
                task_inertia_inv = J @ M_inv @ J.T
                task_inertia     = jnp.linalg.inv(task_inertia_inv)
                J_bar    = M_inv @ J.T @ task_inertia
                task_err = jnp.concatenate([
                    ee_pos - des_pos,
                    orientation_error_3D(ee_rot, des_rot),
                ])
                return J_bar @ (jnp.concatenate([des_vel, des_omega]) - kp_task * task_err)

            self._osc = _osc

            # Warm-up (triggers JIT compilation once at startup)
            _q0 = jnp.zeros(self.robot.num_joints)
            _Mi, _J, _et = self.robot.dynamically_consistent_velocity_control_matrices(_q0)
            _ = _osc(_et[:3, 3], _et[:3, :3],
                     _et[:3, 3], _et[:3, :3],
                     jnp.zeros(3), jnp.zeros(3), _J, _Mi)
            print("[FraxController] OSC JIT compiled.")

            # Joint limits and workspace bounds
            _q_min = (np.asarray(q_min, float) if q_min is not None
                      else np.full(self.robot.num_joints, -2 * np.pi))
            _q_max = (np.asarray(q_max, float) if q_max is not None
                      else np.full(self.robot.num_joints,  2 * np.pi))
            _ws_lo = (np.asarray(ws_lo, float) if ws_lo is not None
                      else np.array([-1.2, -1.2, -0.05]))
            _ws_hi = (np.asarray(ws_hi, float) if ws_hi is not None
                      else np.array([ 1.2,  1.2,  1.50]))

            # Obstacle boxes → tuples (CBF config is immutable once built)
            obs_c, obs_h, obs_R = [], [], []
            for obs in (obstacle_boxes or []):
                if isinstance(obs, dict):
                    c, h, y = obs['center'], obs['half'], obs.get('yaw_deg', 0.0)
                elif len(obs) == 3:
                    c, h, y = obs
                else:
                    c, h = obs; y = 0.0
                obs_c.append(np.asarray(c, float))
                obs_h.append(np.asarray(h, float))
                obs_R.append(ScipyR.from_euler('z', float(y), degrees=True).as_matrix().T)

            _cbf_cfg = _RcCbfConfig(
                self.robot,
                tuple(float(v) for v in self._base_pos),
                tuple(float(v) for v in self._base_R.ravel()),
                tuple(float(v) for v in _q_min),
                tuple(float(v) for v in _q_max),
                float(z_min),
                tuple(float(v) for v in _ws_lo),
                tuple(float(v) for v in _ws_hi),
                tuple(map(tuple, obs_c)) if obs_c else (),
                tuple(map(tuple, obs_h)) if obs_h else (),
                tuple(tuple(float(v) for v in R.ravel()) for R in obs_R) if obs_R else (),
                float(cbf_alpha),
                sc_pairs=_UR_SC_PAIRS if _UR_COLLISION_AVAILABLE else (),
            )
            self._cbf_cfg = _cbf_cfg
            self._cbf = CBF.from_config(_cbf_cfg)
            _sc_msg = f"{len(_UR_SC_PAIRS)} self-collision pairs" if _UR_COLLISION_AVAILABLE else "no self-collision"
            print(f"[FraxController] CBF built — {len(obs_c)} obstacle(s), {_sc_msg}.")

        # ── Public API ────────────────────────────────────────────────────────

        def servo_step(self,
                       target_pos_world: np.ndarray,
                       target_rot_world: np.ndarray,
                       q_current:        np.ndarray,
                       dt:               float) -> np.ndarray:
            """Compute q_target for one servoJ step via OSC+CBF."""
            q       = jnp.array(q_current)
            des_pos = jnp.array(self._base_R.T @ (target_pos_world - self._base_pos))
            des_rot = jnp.array(self._base_R.T @ target_rot_world)

            M_inv, J, ee_t = self.robot.dynamically_consistent_velocity_control_matrices(q)
            qdot      = self._osc(ee_t[:3, 3], ee_t[:3, :3],
                                  des_pos, des_rot,
                                  jnp.zeros(3), jnp.zeros(3), J, M_inv)
            qdot_safe = np.asarray(self._cbf.safety_filter(q, qdot))
            qdot_np   = np.clip(qdot_safe, -self._qdot_max, self._qdot_max)
            return q_current + qdot_np * dt

        def ee_world_pos(self, q_current: np.ndarray) -> np.ndarray:
            """End-effector position in world frame."""
            q = jnp.array(q_current)
            _, _, ee_t = self.robot.dynamically_consistent_velocity_control_matrices(q)
            return self._base_R @ np.asarray(ee_t[:3, 3]) + self._base_pos

        def link_spheres_world(self, q_current: np.ndarray):
            """Return (positions, radii) of all collision spheres in world frame.

            Returns (None, None) when the UR collision model is not loaded.
            positions: np.ndarray (N, 3);  radii: np.ndarray (N,)
            """
            if not _UR_COLLISION_AVAILABLE:
                return None, None
            q = jnp.array(q_current)
            pos_b, radii = self.robot.link_collision_data(q)
            pos_w = np.asarray(pos_b) @ self._base_R.T + self._base_pos
            return pos_w, np.asarray(radii)

        def h_barriers(self, q_current: np.ndarray) -> np.ndarray:
            """Evaluate all CBF barrier values h(q).  h > 0 means safe."""
            return np.asarray(self._cbf_cfg.h_1(jnp.array(q_current)))

        def rebuild_cbf(self, new_alpha: float) -> None:
            """Rebuild the CBF QP with a new alpha value (triggers re-JIT, ~1-2 s)."""
            self._cbf_cfg._alpha = float(new_alpha)
            self._cbf = CBF.from_config(self._cbf_cfg)
            print(f"[FraxController] CBF rebuilt with alpha={new_alpha:.2f}")


# =============================================================================
# RobotController
# =============================================================================

class RobotController:
    """Unified robot interface for simulation and real-robot modes.

    Parameters
    ----------
    unity_ip       : Quest / Unity host IP for ZMQ publishing.
    pb_scene       : Shared PyBulletScene used for IK and simulation.
    T_world_base   : (4,4) robot base pose in world frame.
    robot_ip       : UR10e IP. Pass None (default) for simulation mode.
    base_pub_port  : ZMQ PUB port for robot base pose (default 5000).
    joint_pub_port : ZMQ PUB port for joint angles (default 5001).
    approach_dist  : Real-robot approach standoff in metres (default 0.10).
    speed / accel  : Default moveJ speed (rad/s) and acceleration (rad/s²).
    """

    # -90° yaw between the PyBullet URDF convention and Unity's URDF asset.
    _BASE_YAW_CORRECTION_DEG = -90.0

    def __init__(
        self,
        unity_ip:       str,
        pb_scene,
        T_world_base:   np.ndarray,
        robot_ip:       "str | None" = None,
        base_pub_port:  int   = 5000,
        joint_pub_port: int   = 5001,
        approach_dist:  float = 0.10,
        speed:          float = 0.15,
        accel:          float = 0.10,
        urdf_path:      "str | None" = None,   # enables frax OSC+CBF (real robot only)
        frax_q_min:     "list | None" = None,  # joint lower limits (rad) for CBF
        frax_q_max:     "list | None" = None,  # joint upper limits (rad) for CBF
        frax_ws_lo:     "list | None" = None,  # workspace lower corner [x,y,z] (world)
        frax_ws_hi:     "list | None" = None,  # workspace upper corner [x,y,z] (world)
    ) -> None:
        self.simulation    = (robot_ip is None)
        self._robot_ip     = robot_ip
        self._pb_scene     = pb_scene
        self._T_world_base = np.array(T_world_base, dtype=float)
        self._approach_dist = approach_dist
        self._speed        = speed
        self._accel        = accel
        self._last_q: "np.ndarray | None" = None

        # Real-robot only: lazy RTDE + gripper connections
        self._recv:    "RTDEReceiveInterface | None" = None
        self._rtde_ctrl: "RTDEControlInterface | None" = None
        self._gripper: "RobotiqGripper | None"       = None
        self._in_servo = False

        # Real-robot grasp thread
        self._grasp_thread: "threading.Thread | None" = None
        self._grasp_cancel  = threading.Event()

        # Simulation motion state machine
        self._sim_runner:       None                  = None
        self._sim_phase:        "str | None"          = None
        self._sim_on_complete:  "Callable | None"     = None
        self._sim_grasp_joints: "np.ndarray | None"   = None
        self._sim_q_approach:   "np.ndarray | None"   = None
        self._sim_q_above:      "np.ndarray | None"   = None

        # frax OSC+CBF (real robot + simulation; None if unavailable or no urdf_path given)
        self._frax: "None" = None
        if _FRAX_AVAILABLE and urdf_path is not None:
            try:
                self._frax = _FraxController(
                    urdf_path, self._T_world_base,
                    q_min  = frax_q_min,
                    q_max  = frax_q_max,
                    ws_lo  = frax_ws_lo,
                    ws_hi  = frax_ws_hi,
                )
            except Exception as exc:
                print(f"[Robot] frax init failed: {exc} — falling back to IK.")

        # ZMQ publishers (Unity)
        _ctx = zmq.Context.instance()
        self._base_pub  = _ctx.socket(zmq.PUB)
        self._base_pub.connect(f"tcp://{unity_ip}:{base_pub_port}")
        self._joint_pub = _ctx.socket(zmq.PUB)
        self._joint_pub.connect(f"tcp://{unity_ip}:{joint_pub_port}")

    # ── PyBullet scene accessors ──────────────────────────────────────────────

    @property
    def tcp_pose(self) -> "np.ndarray | None":
        """Current TCP transform (4×4 world frame) from PyBullet FK."""
        if self._pb_scene is None:
            return None
        return self._pb_scene.update_tcp_bodies()

    def arm_link_poses(self):
        """World-frame link poses for the 3D visualizer."""
        if self._pb_scene is None:
            return None
        return self._pb_scene.get_arm_link_world_poses()

    # ── Joint state ───────────────────────────────────────────────────────────

    @property
    def q(self) -> "np.ndarray | None":
        """Last polled joint angles (radians)."""
        return self._last_q

    def poll_q(self) -> "np.ndarray | None":
        """Return current joint angles.

        Simulation : reads pb_scene.current_q (no hardware).
        Real robot : queries RTDE; returns None on failure.
        """
        if self.simulation:
            if self._pb_scene is None:
                return None
            q = self._pb_scene.current_q.copy()
            self._last_q = q
            return q
        try:
            q = np.array(self._recv_conn().getActualQ(), dtype=float)
            self._last_q = q
            return q
        except Exception as e:
            print(f"[Robot] poll_q failed: {e}")
            return None

    # ── Hand tracking ─────────────────────────────────────────────────────────

    def step_hand_track(self, target_pos: "list | np.ndarray",
                        target_quat: "list | np.ndarray", dt: float) -> None:
        """Drive TCP toward a target this frame.

        Simulation : calls pb_scene.step_ik (updates pb_scene.current_q).
        Real robot : OSC+CBF via frax if available, else IK → servoJ.
        """
        if self._pb_scene is None:
            return
        if self.simulation:
            self._pb_scene.step_ik(
                self._pb_scene.current_q, list(target_pos), list(target_quat), dt)
        elif self._frax is not None and self._last_q is not None:
            target_rot = ScipyR.from_quat(list(target_quat)).as_matrix()
            q_target = self._frax.servo_step(
                np.array(target_pos), target_rot, self._last_q, dt)
            self.servoJ(q_target, dt)
        else:
            try:
                q = self.solve_ik(np.array(target_pos), list(target_quat), self._last_q)
            except RuntimeError:
                return
            self.servoJ(q, dt)

    # ── Simulation per-frame tick ─────────────────────────────────────────────

    def tick(self) -> None:
        """Advance the simulation runner one frame; call on_complete when done."""
        if not self.simulation or self._sim_runner is None:
            return
        if not self._sim_runner.done:
            self._sim_runner.update(self._pb_scene.robot_id,
                                    self._pb_scene.arm_indices)
            return
        # Phase complete — transition or finish.
        # Tool:  approach → grasp → retract
        # Part:  approach → above → grasp → ret_above → retract
        cq = self._pb_scene.current_q.copy()
        if self._sim_phase == 'approach':
            if self._sim_q_above is not None:
                self._sim_runner = _PbJointRunner(cq, self._sim_q_above)
                self._sim_phase  = 'above'
                print("[Robot sim] At approach → moveJ to above")
            else:
                self._sim_runner = _PbJointRunner(cq, self._sim_grasp_joints)
                self._sim_phase  = 'grasp'
                print("[Robot sim] At approach → moveJ to grasp")
        elif self._sim_phase == 'above':
            self._sim_runner = _PbJointRunner(cq, self._sim_grasp_joints)
            self._sim_phase  = 'grasp'
            print("[Robot sim] At above → moveJ to grasp")
        elif self._sim_phase == 'grasp':
            if self._sim_q_above is not None:
                self._sim_runner = _PbJointRunner(cq, self._sim_q_above)
                self._sim_phase  = 'ret_above'
                print("[Robot sim] Grasp done → lifting to above")
            elif self._sim_q_approach is not None:
                self._sim_runner = _PbJointRunner(cq, self._sim_q_approach)
                self._sim_phase  = 'retract'
                print("[Robot sim] Grasp done → retracting to approach")
            else:
                self._sim_finish(True)
        elif self._sim_phase == 'ret_above':
            self._sim_runner = _PbJointRunner(cq, self._sim_q_approach)
            self._sim_phase  = 'retract'
            print("[Robot sim] At above → retracting to approach")
        else:
            self._sim_finish(True)

    def _sim_finish(self, success: bool) -> None:
        cb = self._sim_on_complete
        self._sim_runner       = None
        self._sim_phase        = None
        self._sim_on_complete  = None
        self._sim_grasp_joints = None
        self._sim_q_approach   = None
        self._sim_q_above      = None
        if cb is not None:
            try:
                cb(success)
            except Exception:
                pass

    # ── Single TCP-pose move ──────────────────────────────────────────────────

    def move_tcp(self, pos: "list | np.ndarray",
                 quat: "list | np.ndarray",
                 on_complete: "Callable[[bool], None] | None" = None) -> None:
        """Move TCP to a Cartesian target in one motion phase.

        Simulation : starts a _PbWaypointRunner, advanced by tick().
        Real robot : solves IK and runs moveJ in a background thread.
        """
        if self.simulation:
            if self._pb_scene is None or not _PB_RUNNER_AVAILABLE:
                return
            try:
                self._sim_runner = _PbWaypointRunner(
                    self._pb_scene.robot_id,
                    self._pb_scene.tool0_link_idx,
                    self._pb_scene.current_q.copy(),
                    self._pb_scene.arm_indices,
                    list(pos),
                    target_quat_xyzw=np.array(quat),
                )
                self._sim_phase = 'move_tcp'
                self._sim_on_complete = on_complete
            except Exception as e:
                print(f"[Robot sim] move_tcp failed: {e}")
        elif self._frax is not None:
            target_pos = np.array(pos, float)
            target_rot = ScipyR.from_quat(list(quat)).as_matrix()
            def _frax_loop():
                dt      = 0.008   # ~125 Hz
                timeout = 30.0
                t0      = time.time()
                self._grasp_cancel.clear()
                while not self._grasp_cancel.is_set():
                    if time.time() - t0 > timeout:
                        print("[Robot] frax move_tcp: timeout.")
                        if on_complete:
                            on_complete(False)
                        return
                    q_cur = self._last_q
                    if q_cur is None:
                        time.sleep(dt)
                        continue
                    ee_pos = self._frax.ee_world_pos(q_cur)
                    if np.linalg.norm(ee_pos - target_pos) < 0.005:
                        self.servoStop()
                        if on_complete:
                            on_complete(True)
                        return
                    q_target = self._frax.servo_step(target_pos, target_rot, q_cur, dt)
                    self.servoJ(q_target, dt)
                    time.sleep(dt)
                self.servoStop()
            threading.Thread(target=_frax_loop, daemon=True).start()
        else:
            def _move():
                try:
                    q = self.solve_ik(np.array(pos), list(quat), self._last_q)
                    self.moveJ(q)
                    if on_complete:
                        on_complete(True)
                except Exception as e:
                    print(f"[Robot] move_tcp failed: {e}")
                    if on_complete:
                        on_complete(False)
            threading.Thread(target=_move, daemon=True).start()

    # ── Low-level control (real robot) ────────────────────────────────────────

    def servoJ(self, q: np.ndarray, dt: float,
               speed: float = 1.0, accel: float = 1.0,
               lookahead: float = 0.1, gain: int = 300) -> None:
        """Stream one servoJ command to the robot."""
        self._rtde_ctrl_conn().servoJ(list(q), speed, accel, dt, lookahead, gain)
        self._in_servo = True

    def servoStop(self) -> None:
        """Exit servoJ mode so moveJ can be accepted."""
        if self._in_servo and self._rtde_ctrl is not None:
            try:
                self._rtde_ctrl.servoStop()
            except Exception:
                pass
            self._in_servo = False

    def moveJ(self, q: "list | np.ndarray",
              speed: "float | None" = None,
              accel: "float | None" = None) -> None:
        """Blocking joint-space move (real robot only)."""
        self.servoStop()
        self._rtde_ctrl_conn().moveJ(
            list(q),
            speed if speed is not None else self._speed,
            accel if accel is not None else self._accel,
        )

    def stopJ(self, decel: float = 2.0) -> None:
        if self._rtde_ctrl is not None:
            try:
                self._rtde_ctrl.stopJ(decel)
            except Exception:
                pass

    # ── IK solving ────────────────────────────────────────────────────────────

    def solve_ik(self, pos_world: np.ndarray, quat_xyzw: list,
                 seed_q: "np.ndarray | None" = None,
                 tol: float = 0.005) -> np.ndarray:
        """Single-shot IK via PyBullet's calculateInverseKinematics.

        Uses seed_q as the rest-pose bias so the solver prefers a configuration
        near the current robot state. Raises RuntimeError if residual > 50 mm.
        """
        import pybullet as p
        if self._pb_scene is None:
            raise RuntimeError("No PyBullet scene — IK unavailable.")
        if seed_q is None:
            seed_q = self._last_q if self._last_q is not None else np.zeros(6)

        pb = self._pb_scene
        arm_q_map  = dict(zip(pb.arm_indices, seed_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in pb._movable]

        joint_q = p.calculateInverseKinematics(
            pb.robot_id, pb.tool0_link_idx,
            list(pos_world), list(quat_xyzw),
            lowerLimits=pb._lower_limits, upperLimits=pb._upper_limits,
            jointRanges=pb._joint_ranges, restPoses=rest_poses,
            maxNumIterations=200, residualThreshold=1e-5)
        q = np.array(joint_q[:len(pb.arm_indices)], dtype=np.float64)

        # Verify via FK
        pb.update_robot(q)
        T_fk = pb.update_tcp_bodies()
        err = (np.linalg.norm(T_fk[:3, 3] - pos_world) if T_fk is not None else float('inf'))
        if err > tol:
            print(f"[Robot IK] WARNING — residual {err*1000:.1f} mm")
        if err > 0.05:
            raise RuntimeError(
                f"IK failed: residual {err*1000:.0f} mm > 50 mm. "
                f"Target {np.round(pos_world, 3).tolist()} may be unreachable.")
        return q

    # ── Gripper (real robot) ──────────────────────────────────────────────────

    def open_gripper(self, speed: int = 255, force: int = 10) -> None:
        g = self._gripper_conn()
        g.move_and_wait_for_pos(g.get_open_position(), speed, force)

    def close_gripper(self, speed: int = 255, force: int = 100) -> None:
        g = self._gripper_conn()
        g.move_and_wait_for_pos(g.get_closed_position(), speed, force)

    # ── Grasp sequence (both modes) ───────────────────────────────────────────

    @property
    def tool_grasp_running(self) -> bool:
        if self.simulation:
            return self._sim_phase is not None
        return self._grasp_thread is not None and self._grasp_thread.is_alive()

    def execute_grasp(
        self,
        grasp_joints: "list | np.ndarray",
        on_complete:  "Callable[[bool], None] | None" = None,
        q_approach:   "list | np.ndarray | None"      = None,
        q_above:      "list | np.ndarray | None"      = None,
    ) -> None:
        """Grasp sequence in both modes.

        Tool  (q_above=None): approach → grasp → retract
        Part  (q_above given): approach → above → grasp → above → retract
        Bare  (q_approach=None): direct moveJ to grasp_joints
        """
        if self.tool_grasp_running:
            print("[Robot] Grasp already running — cancel first.")
            return
        if self.simulation:
            if self._pb_scene is None:
                if on_complete:
                    on_complete(False)
                return
            self._sim_on_complete  = on_complete
            self._sim_grasp_joints = np.array(grasp_joints, dtype=float)
            self._sim_q_approach   = (np.array(q_approach, dtype=float)
                                      if q_approach is not None else None)
            self._sim_q_above      = (np.array(q_above, dtype=float)
                                      if q_above is not None else None)
            if self._sim_q_approach is not None:
                self._sim_runner = _PbJointRunner(
                    self._pb_scene.current_q.copy(),
                    self._sim_q_approach,
                )
                self._sim_phase = 'approach'
                print("[Robot sim] moveJ → approach")
            else:
                self._sim_runner = _PbJointRunner(
                    self._pb_scene.current_q.copy(),
                    self._sim_grasp_joints,
                )
                self._sim_phase = 'grasp'
                print("[Robot sim] moveJ → grasp_joints")
        else:
            if not _RTDE_AVAILABLE:
                print("[Robot] rtde_control not installed.")
                return
            self._grasp_cancel.clear()
            self._grasp_thread = threading.Thread(
                target=self._grasp_sequence,
                args=(np.array(grasp_joints, dtype=float),
                      (np.array(q_approach, dtype=float) if q_approach is not None else None),
                      (np.array(q_above,    dtype=float) if q_above    is not None else None),
                      on_complete),
                daemon=True,
            )
            self._grasp_thread.start()

    def cancel_motion(self) -> None:
        """Abort any running grasp and stop the arm."""
        if self.simulation:
            cb = self._sim_on_complete
            self._sim_runner       = None
            self._sim_phase        = None
            self._sim_on_complete  = None
            self._sim_grasp_joints = None
            self._sim_q_approach   = None
            self._sim_q_above      = None
            if cb is not None:
                try:
                    cb(False)
                except Exception:
                    pass
        else:
            self._grasp_cancel.set()
            self.stopJ()

    # ── Unity publishing ──────────────────────────────────────────────────────

    def publish_base(self, T_world_base: "np.ndarray | None" = None) -> None:
        """Publish robot base pose to Unity (port 5000)."""
        T = T_world_base if T_world_base is not None else self._T_world_base
        R_o3d  = T[:3, :3] @ ScipyR.from_euler(
            'z', self._BASE_YAW_CORRECTION_DEG, degrees=True).as_matrix()
        t_o3d  = T[:3, 3]
        q_xyzw = ScipyR.from_matrix(R_o3d).as_quat()
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u    = open3d_to_unity_quaternion(q_wxyz)
        t_u    = open3d_to_unity_vector(t_o3d)
        q_u_xyzw = [float(q_u[1]), float(q_u[2]), float(q_u[3]), float(q_u[0])]
        R_u    = ScipyR.from_quat(q_u_xyzw).as_matrix()
        T_u    = np.eye(4, dtype=float)
        T_u[:3, :3] = R_u
        T_u[:3, 3]  = t_u
        try:
            self._base_pub.send_string(
                json.dumps({"robot_matrix": T_u.T.flatten().tolist()}))
        except Exception as e:
            print(f"[Robot] publish_base error: {e}")

    def publish_joints(self, q: np.ndarray) -> None:
        """Publish live joint angles (radians) to Unity (port 5001)."""
        try:
            self._joint_pub.send_string(
                json.dumps({"joint_values": [float(v) for v in q]}))
        except Exception as e:
            print(f"[Robot] publish_joints error: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.cancel_motion()
        if not self.simulation:
            self.servoStop()
            for attr in ("_recv", "_rtde_ctrl"):
                conn = getattr(self, attr)
                if conn is not None:
                    try:
                        conn.disconnect()
                    except Exception:
                        pass
                    setattr(self, attr, None)
            if self._gripper is not None:
                try:
                    self._gripper.disconnect()
                except Exception:
                    pass
                self._gripper = None
        for sock in (self._base_pub, self._joint_pub):
            try:
                sock.close(0)
            except Exception:
                pass

    # ── Internal: lazy real-robot connections ─────────────────────────────────

    def _recv_conn(self):
        if not _RTDE_AVAILABLE:
            raise RuntimeError("rtde_receive not installed.")
        if self._recv is None:
            self._recv = RTDEReceiveInterface(self._robot_ip)
            print(f"[Robot] RTDE receive → {self._robot_ip}")
        return self._recv

    def _rtde_ctrl_conn(self):
        if not _RTDE_AVAILABLE:
            raise RuntimeError("rtde_control not installed.")
        if self._rtde_ctrl is None:
            self._rtde_ctrl = RTDEControlInterface(self._robot_ip)
            print(f"[Robot] RTDE control → {self._robot_ip}")
        return self._rtde_ctrl

    def _gripper_conn(self):
        if not _GRIPPER_AVAILABLE:
            raise RuntimeError("RobotiqGripper not available.")
        if self._gripper is None:
            g = RobotiqGripper()
            g.connect(self._robot_ip, 63352)
            g.activate()
            self._gripper = g
            print("[Robot] Gripper ready.")
        return self._gripper

    # ── Internal: real-robot grasp sequence ───────────────────────────────────

    def _grasp_sequence(self, grasp_joints: np.ndarray,
                        q_approach: "np.ndarray | None",
                        q_above:    "np.ndarray | None",
                        on_complete: "Callable[[bool], None] | None") -> None:
        def _check():
            if self._grasp_cancel.is_set():
                raise InterruptedError("Grasp cancelled.")
        success = False
        try:
            self.servoStop()
            _check()
            self.open_gripper()
            _check()
            if q_approach is not None:
                self.moveJ(q_approach, speed=0.5, accel=0.5); _check()
            if q_above is not None:
                self.moveJ(q_above, speed=0.3, accel=0.3); _check()
            self.moveJ(grasp_joints, speed=0.2, accel=0.2); _check()
            self.close_gripper()
            time.sleep(1.0)   # 1 s visual check — gripper opens if object resists
            self.open_gripper()
            _check()
            # Retract (reverse)
            if q_above is not None:
                self.moveJ(q_above,    speed=0.2, accel=0.2); _check()
                self.moveJ(q_approach, speed=0.3, accel=0.3)
            elif q_approach is not None:
                self.moveJ(q_approach, speed=0.5, accel=0.5)
            success = True
        except InterruptedError:
            print("[Robot] Grasp cancelled.")
        except Exception as e:
            print(f"[Robot] Grasp error: {e}")
        if on_complete is not None:
            try:
                on_complete(success)
            except Exception:
                pass
