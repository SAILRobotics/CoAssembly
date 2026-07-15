"""robot_controller.py — Central robot controller for CoAssembly.

Handles all robot-related concerns in both simulation and real-robot modes:
  - PyBullet scene     : step_ik (hand tracking), waypoint runner (sim grasp)
  - RTDE receive/ctrl  : poll joint angles, servoJ, moveJ (real only)
  - IK solving         : via the shared PyBullet scene
  - Gripper            : Robotiq 2F-85 open / close (real only)
  - Grasp sequence     : approach → grasp → retract
                         sim  → PyBullet waypoint runner, driven by tick()
                         real → async RTDE moveJ state machine, driven by tick()
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

# ── Force-monitoring tunables ─────────────────────────────────────────────────

_FORCE_RELEASE_THRESHOLD = 20.0  # N — pull force delta to trigger tool release
_FORCE_GRASP_THRESHOLD   =  4.0  # N — contact force delta to trigger board grasp
_FORCE_DEBOUNCE_HITS     =  5    # consecutive over-threshold ticks before acting
_FORCE_POLL_HZ           = 20    # Hz — force polling rate

# ── Simulation waypoint runner ──────────────────────────────────────────────────


class _PbJointRunner:
    """Direct joint-space linear interpolation — sim equivalent of moveJ.

    No IK. Interpolates start_q → target_q over _DURATION_S seconds of
    wall-clock time (using the caller's dt), calling resetJointState each
    tick so the PyBullet scene reflects the motion. Time-based rather than
    step-count-based so the motion's speed is independent of tick()'s call
    rate — at a fixed step count, raising the control loop's Hz would make
    every sim moveJ/grasp waypoint complete proportionally faster.
    """
    _DURATION_S = 3.0

    def __init__(self, start_q: np.ndarray, target_q: np.ndarray):
        self.start_q  = np.asarray(start_q, dtype=float)
        self.target_q = np.asarray(target_q, dtype=float)
        self._elapsed = 0.0
        self.done     = False

    def update(self, robot_id: int, arm_indices: "list[int]", dt: float) -> None:
        if self.done:
            return
        import pybullet as p
        self._elapsed += dt
        t = min(self._elapsed / self._DURATION_S, 1.0)
        q = self.start_q + t * (self.target_q - self.start_q)
        for j_idx, q_j in zip(arm_indices, q):
            p.resetJointState(robot_id, j_idx, float(q_j))
        if t >= 1.0:
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
        pairs_sc              as _UR_SC_PAIRS,
        pairs_sc_with_gripper as _UR_SC_PAIRS_WG,
        make_collision_data   as _ur_make_collision_data,
        base_position         as _UR_BASE_POS,
        base_radius           as _UR_BASE_RADIUS,
        base_sc_idxs          as _UR_BASE_SC_IDXS,
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

# Precompute both arm-only (22-sphere) and arm+gripper (30-sphere) collision dicts.
# _FraxController selects between them via its gripper_collision parameter.
def _build_frax_collision(with_gripper: bool) -> "dict | None":
    if not (_FRAX_AVAILABLE and _UR_COLLISION_AVAILABLE):
        return None
    data     = _ur_make_collision_data(with_gripper=with_gripper)
    sc_pairs = _UR_SC_PAIRS_WG if with_gripper else _UR_SC_PAIRS
    return {
        "positions":      data["positions"],
        "radii":          data["radii"],
        "root_positions": (_UR_BASE_POS,),
        "root_radii":     (_UR_BASE_RADIUS,),
        "root_sc_pairs":  tuple((0, idx) for idx in _UR_BASE_SC_IDXS),
        "root_sc_tols":   tuple(0.0 for _ in _UR_BASE_SC_IDXS),
        "body_sc_pairs":  sc_pairs,
        "body_sc_tols":   tuple(0.0 for _ in sc_pairs),
    }

_UR_FRAX_COLLISION_ARM = _build_frax_collision(with_gripper=False)
_UR_FRAX_COLLISION_WG  = _build_frax_collision(with_gripper=True)

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
            urdf_path:         str,
            T_world_base:      np.ndarray,
            kp_pos:            float = 200.0,
            kp_ori:            float = 200.0,
            qdot_max:          float = 3.0,
            q_min:             "list | None" = None,
            q_max:             "list | None" = None,
            cbf_alpha:         float = 10.0,
            z_min:             float = 0.15,
            ws_lo:             "list | None" = None,
            ws_hi:             "list | None" = None,
            obstacle_boxes:    "list | None" = None,
            gripper_collision: bool = True,
        ) -> None:
            self._qdot_max = float(qdot_max)

            # Base frame: frax URDF convention has a Rz(180°) offset
            _Rz180       = ScipyR.from_euler('z', np.pi).as_matrix()
            self._base_pos = np.array(T_world_base[:3, 3], float)
            self._base_R   = np.array(T_world_base[:3, :3], float) @ _Rz180

            # Load robot model with collision spheres when available
            _col_data = (_UR_FRAX_COLLISION_WG if gripper_collision
                         else _UR_FRAX_COLLISION_ARM)
            self.robot = load_ur10e(str(urdf_path), collision_data=_col_data)
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
                sc_pairs=((_UR_SC_PAIRS_WG if gripper_collision else _UR_SC_PAIRS)
                          if _UR_COLLISION_AVAILABLE else ()),
            )
            self._cbf_cfg = _cbf_cfg
            self._cbf = CBF.from_config(_cbf_cfg)
            if _UR_COLLISION_AVAILABLE:
                _sc_pairs_used = _UR_SC_PAIRS_WG if gripper_collision else _UR_SC_PAIRS
                _sc_label = "w/ gripper" if gripper_collision else "arm only"
                _sc_msg = f"{len(_sc_pairs_used)} self-collision pairs ({_sc_label})"
            else:
                _sc_msg = "no self-collision"
            print(f"[FraxController] CBF built — {len(obs_c)} obstacle(s), {_sc_msg}.")

            # Warm up CBF safety_filter JIT so first real call doesn't stall
            # (and doesn't race with the Open3D render thread on GPU/XLA init).
            try:
                _q0_cbf = jnp.zeros(self.robot.num_joints)
                _ = self._cbf.safety_filter(_q0_cbf, _q0_cbf)
                print("[FraxController] CBF JIT compiled.")
            except Exception:
                pass

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
def _wrap_nearest(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Shift q_target by ±n·2π per joint to minimise travel from q_ref.

    If the shortest-path equivalent still falls outside ±2π (i.e. the robot has
    wound past 360°), adds another ±2π to bring it back inside the UR's hard
    range.  The result may be farther than the raw shortest path but is always
    within [-2π, 2π] so the UR controller won't reject the moveJ.
    """
    TWO_PI = 2.0 * np.pi
    q = q_target - TWO_PI * np.round((q_target - q_ref) / TWO_PI)
    q = np.where(q < -TWO_PI, q + TWO_PI, q)
    q = np.where(q >  TWO_PI, q - TWO_PI, q)
    return q


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
    approach_dist  : Real-robot approach standoff in metres (default 0.30).
    speed / accel  : Default moveJ speed (rad/s) and acceleration (rad/s²).
    speed_scale    : Multiplies real-hardware speed/accel and the pre-CBF
                     joint-delta rate limit (moveJ, servoJ-driven tracking,
                     grasp-sequence moveJs). 1.0 = full speed. Has no effect
                     on simulation. Intended as a temporary, blanket "slow
                     everything down" safety knob while testing on real
                     hardware.
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
        approach_dist:  float = 0.30,
        speed:          float = 0.15,
        accel:          float = 0.10,
        speed_scale:    float = 1.0,           # real-hardware speed/accel multiplier — see class docstring
        urdf_path:      "str | None" = None,   # enables frax OSC+CBF (real robot only)
        frax_q_min:     "list | None" = None,  # joint lower limits (rad) for CBF
        frax_q_max:     "list | None" = None,  # joint upper limits (rad) for CBF
        frax_ws_lo:            "list | None" = None,  # workspace lower corner [x,y,z] (world)
        frax_ws_hi:            "list | None" = None,  # workspace upper corner [x,y,z] (world)
        frax_z_min:            float = 0.15,          # floor clearance for link collision spheres (m)
        frax_gripper_collision: bool = True,           # include gripper spheres in CBF
    ) -> None:
        self.simulation    = (robot_ip is None)
        self._robot_ip     = robot_ip
        self._pb_scene     = pb_scene
        self._T_world_base = np.array(T_world_base, dtype=float)
        self._approach_dist = approach_dist
        self._speed_scale   = float(speed_scale)
        self._speed         = speed * self._speed_scale
        self._accel         = accel * self._speed_scale
        self._last_q: "np.ndarray | None" = None

        # Real-robot only: lazy RTDE + gripper connections
        self._recv:    "RTDEReceiveInterface | None" = None
        self._rtde_ctrl: "RTDEControlInterface | None" = None
        self._gripper: "RobotiqGripper | None"       = None
        self._in_servo = False

        # Real-robot grasp state machine (driven by tick())
        self._real_grasp_joints: "np.ndarray | None" = None
        self._real_q_approach:   "np.ndarray | None" = None
        self._real_q_above:      "np.ndarray | None" = None
        self._real_on_complete:  "Callable | None"   = None
        self._real_on_phase:     "Callable | None"   = None
        self._real_move_sent:    bool                 = False
        self._real_settle_start: "float | None"       = None

        # Simulation motion state machine
        self._sim_runner:       None                  = None
        self._sim_phase:        "str | None"          = None
        self._sim_on_complete:  "Callable | None"     = None
        self._sim_on_phase:     "Callable | None"     = None
        self._sim_grasp_joints: "np.ndarray | None"   = None
        self._sim_q_approach:   "np.ndarray | None"   = None
        self._sim_q_above:      "np.ndarray | None"   = None

        # Sim move_tcp target — advanced one step per tick(), no thread needed
        self._tracked_tcp_pos:  "np.ndarray | None" = None
        self._tracked_tcp_quat: "np.ndarray | None" = None
        self._tracked_tcp_cb:   "Callable | None"   = None
        self._move_tcp_smooth:  "np.ndarray | None" = None  # EMA state for real servoJ

        # Force monitoring — polled by tick(), fires callback on threshold (real robot only)
        self._force_mode:     "str | None"        = None   # 'release' | 'grasp'
        self._force_baseline: "np.ndarray | None" = None
        self._force_hits:     int                  = 0
        self._force_cb:       "Callable | None"    = None
        self._force_last_t:   "float | None"       = None
        self._force_threshold: "float | None"      = None   # active trigger delta (N); set by start_force_monitor

        # frax OSC+CBF (real robot + simulation; None if unavailable or no urdf_path given)
        self._frax: "None" = None
        if _FRAX_AVAILABLE and urdf_path is not None:
            try:
                self._frax = _FraxController(
                    urdf_path, self._T_world_base,
                    q_min              = frax_q_min,
                    q_max              = frax_q_max,
                    ws_lo              = frax_ws_lo,
                    ws_hi              = frax_ws_hi,
                    z_min              = frax_z_min,
                    gripper_collision  = frax_gripper_collision,
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
        try:
            return self._pb_scene.update_tcp_bodies()
        except Exception:
            return None

    def arm_link_poses(self):
        """World-frame link poses for the 3D visualizer."""
        if self._pb_scene is None:
            return None
        try:
            return self._pb_scene.get_arm_link_world_poses()
        except Exception:
            return None

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
                        target_quat: "list | np.ndarray", dt: float,
                        max_joint_speed: float = np.deg2rad(180.0)) -> None:
        """Drive TCP toward a target this frame.

        Simulation : pb_scene.step_ik (rate-limited, updates pb_scene.current_q).
        Real robot : PyBullet IK with the same rate-limit → servoJ.
                     frax.servo_step is intentionally NOT used — OSC doesn't converge.
        """
        if self._pb_scene is None:
            return
        q        = self._pb_scene.current_q.copy() if self.simulation else self._last_q
        if q is None:
            return
        # Sim has no hardware constraint — run 4× faster so tracking feels responsive
        max_step  = (max_joint_speed * 4.0 * dt) if self.simulation \
                    else (max_joint_speed * self._speed_scale * dt)
        q_ik      = self.query_ik_joints(target_pos, target_quat, seed_q=q)
        # calculateInverseKinematics doesn't enforce joint limits, and the
        # wrist joints (wide/near-continuous range) can come back wrapped by
        # a full ±2π or more relative to q — unwrap to the nearest equivalent
        # before rate-limiting, or the rate limiter chases the far solution
        # and winds the wrist past its intended range instead of converging
        # to the correct (short-path) configuration.
        q_ik      = _wrap_nearest(q_ik, q)
        q_limited = q + np.clip(q_ik - q, -max_step, max_step)
        if self._frax is not None:
            qdot = (q_limited - q) / max(dt, 1e-6)
            try:
                qdot_safe = np.asarray(
                    self._frax._cbf.safety_filter(jnp.array(q), jnp.array(qdot)))
            except Exception:
                qdot_safe = qdot
            q_target = q + qdot_safe * dt
        else:
            q_target = q_limited
        if self.simulation:
            self._pb_scene.update_robot(q_target)
        else:
            self.servoJ(q_target, dt)

    # ── Simulation per-frame tick ─────────────────────────────────────────────

    def tick(self, dt: float = 1.0 / 60.0) -> None:
        """Advance one frame for both sim and real robot.

        move_tcp: one IK+CBF step per call — sim writes pb_scene, real sends servoJ.
        Grasp state machine (sim only): advances _PbJointRunner one step.
        Call every frame regardless of simulation mode.
        """
        # ── Force monitoring (real robot only, runs at _FORCE_POLL_HZ) ──────────
        if self._force_mode is not None and not self.simulation:
            _ft_now = time.perf_counter()
            if self._force_last_t is None or _ft_now - self._force_last_t >= 1.0 / _FORCE_POLL_HZ:
                self._force_last_t = _ft_now
                _f = self.poll_tcp_force()
                if _f is not None:
                    if self._force_baseline is None:
                        self._force_baseline = _f
                    else:
                        _delta  = float(np.linalg.norm(_f - self._force_baseline))
                        _thresh = self._force_threshold
                        self._force_hits = self._force_hits + 1 if _delta > _thresh else 0
                        if self._force_hits >= _FORCE_DEBOUNCE_HITS:
                            _mode = self._force_mode
                            _cb   = self._force_cb
                            self.stop_force_monitor()
                            print(f"[Robot] Force trigger: {_mode} (delta={_delta:.1f} N)")
                            if _cb:
                                try:
                                    _cb()
                                except Exception as e:
                                    print(f"[Robot] force trigger callback error: {e}")

        # move_tcp: one IK+CBF step, driven by caller's frame rate
        if self._sim_phase == 'move_tcp':
            tgt_pos  = self._tracked_tcp_pos
            tgt_quat = self._tracked_tcp_quat
            if tgt_pos is None:
                self._sim_phase = None
                return
            q_cur = (self._pb_scene.current_q.copy() if self.simulation
                     else self._last_q)
            if q_cur is None:
                return
            if self.simulation:
                max_step = np.deg2rad(720.0) * dt
                conv_thr = 0.01
            else:
                max_step = np.deg2rad(180.0) * self._speed_scale * dt
                conv_thr = 0.005
            if self._frax is not None:
                if np.linalg.norm(self._frax.ee_world_pos(q_cur) - tgt_pos) < conv_thr:
                    cb = self._tracked_tcp_cb
                    self._tracked_tcp_pos  = None
                    self._tracked_tcp_quat = None
                    self._tracked_tcp_cb   = None
                    self._sim_phase        = None
                    if not self.simulation:
                        self.servoStop()
                    if cb:
                        cb(True)
                    return
            q_ik      = self.query_ik_joints(tgt_pos, tgt_quat, seed_q=q_cur)
            # See step_hand_track's comment — unwrap to the nearest equivalent
            # before rate-limiting so the wrist joints take the short path.
            q_ik      = _wrap_nearest(q_ik, q_cur)
            q_limited = q_cur + np.clip(q_ik - q_cur, -max_step, max_step)
            if self._frax is not None:
                qdot = (q_limited - q_cur) / max(dt, 1e-6)
                try:
                    qdot_safe = np.asarray(
                        self._frax._cbf.safety_filter(jnp.array(q_cur), jnp.array(qdot)))
                except Exception:
                    qdot_safe = qdot
                q_target = q_cur + qdot_safe * dt
            else:
                q_target = q_limited
                if np.linalg.norm(q_target - q_cur) < 1e-4:
                    cb = self._tracked_tcp_cb
                    self._tracked_tcp_pos = None
                    self._sim_phase       = None
                    if cb:
                        cb(True)
                    return
            if np.any(np.isnan(q_target)):
                return   # IK/CBF returned NaN (unreachable target) — hold current pose
            if self.simulation:
                self._pb_scene.update_robot(q_target)
            else:
                if self._move_tcp_smooth is None:
                    self._move_tcp_smooth = q_target.copy()
                else:
                    self._move_tcp_smooth = (0.25 * q_target
                                             + 0.75 * self._move_tcp_smooth)
                self.servoJ(self._move_tcp_smooth, dt, lookahead=0.195, gain=100)
            return

        if not self.simulation:
            if self._sim_phase is not None and self._sim_phase.startswith('real_'):
                self._tick_real_grasp(dt)
            return

        if self._sim_runner is None:
            return
        if not self._sim_runner.done:
            self._sim_runner.update(self._pb_scene.robot_id,
                                    self._pb_scene.arm_indices, dt)
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
                self._fire_phase(self._sim_on_phase, "grasping")
        elif self._sim_phase == 'above':
            self._sim_runner = _PbJointRunner(cq, self._sim_grasp_joints)
            self._sim_phase  = 'grasp'
            print("[Robot sim] At above → moveJ to grasp")
            self._fire_phase(self._sim_on_phase, "grasping")
        elif self._sim_phase == 'grasp':
            self._fire_phase(self._sim_on_phase, "grasped")
            if self._sim_q_above is not None:
                self._sim_runner = _PbJointRunner(cq, self._sim_q_above)
                self._sim_phase  = 'ret_above'
                print("[Robot sim] Grasp done → lifting to above")
                self._fire_phase(self._sim_on_phase, "retracting")
            elif self._sim_q_approach is not None:
                self._sim_runner = _PbJointRunner(cq, self._sim_q_approach)
                self._sim_phase  = 'retract'
                print("[Robot sim] Grasp done → retracting to approach")
                self._fire_phase(self._sim_on_phase, "retracting")
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

        Both sim and real use IK + CBF safety filter (when frax is loaded).
        Sim updates pb_scene directly; real sends servoJ.
        """
        if self.simulation:
            if self._pb_scene is None:
                return
            # Just store the target — tick() advances one IK+CBF step per frame
            self._tracked_tcp_pos  = np.array(pos, float)
            self._tracked_tcp_quat = np.array(list(quat), float)
            self._tracked_tcp_cb   = on_complete
            self._sim_phase        = 'move_tcp'
        else:
            # Store target — tick() sends servoJ each frame via frax CBF
            self._tracked_tcp_pos  = np.array(pos, float)
            self._tracked_tcp_quat = np.array(list(quat), float)
            self._tracked_tcp_cb   = on_complete
            self._move_tcp_smooth  = None   # reset EMA on new target
            self._sim_phase        = 'move_tcp'

    # ── Low-level control (real robot) ────────────────────────────────────────

    def servoJ(self, q: np.ndarray, dt: float,
               speed: float = 1.0, accel: float = 1.0,
               lookahead: float = 0.15, gain: int = 300) -> None:
        """Stream one servoJ command to the robot."""
        # UR firmware requires both time and lookahead_time ∈ [0.03, 0.2] s.
        t  = max(min(float(dt),       0.2), 0.03)
        la = max(min(float(lookahead), 0.2), 0.03)
        try:
            self._rtde_ctrl_conn().servoJ(list(q), speed, accel, t, la, gain)
            self._in_servo = True
        except Exception as e:
            # Reset so the next tick attempts a fresh reconnect.
            print(f"[Robot] servoJ failed ({e}); will retry next tick")
            self._rtde_ctrl = None

    def servoStop(self) -> None:
        """Exit servoJ mode so moveJ can be accepted."""
        if self._in_servo and self._rtde_ctrl is not None:
            try:
                self._rtde_ctrl.servoStop()
            except Exception:
                pass
            self._in_servo = False

    # ── IK solving ────────────────────────────────────────────────────────────

    def solve_ik_from_config(self, start_q: np.ndarray,
                             target_pos, target_quat_xyzw,
                             n_steps: int = 20) -> np.ndarray:
        """IK that walks incrementally from start_q toward target_pos.

        By re-seeding each sub-step from the previous result the solver stays
        in the same arm/wrist configuration as start_q and avoids wrist flips
        that occur when the target is far from the seed.
        """
        import pybullet as p
        if self._pb_scene is None:
            raise RuntimeError("No PyBullet scene.")
        pb  = self._pb_scene
        saved_q = pb.current_q.copy()
        try:
            pb.update_robot(start_q)
            T0 = pb.update_tcp_bodies()
            if T0 is None:
                raise RuntimeError("FK failed at start_q.")
            start_pos  = T0[:3, 3].copy()
            target_pos = np.asarray(target_pos, float)
            tgt_quat   = list(target_quat_xyzw)

            q = np.array(start_q, float)
            for step in range(n_steps):
                alpha      = (step + 1) / n_steps
                interp_pos = ((1.0 - alpha) * start_pos + alpha * target_pos).tolist()
                arm_q_map  = dict(zip(pb.arm_indices, q))
                rest_poses = [float(arm_q_map.get(j, 0.0)) for j in pb._movable]
                joint_q    = p.calculateInverseKinematics(
                    pb.robot_id, pb.tool0_link_idx,
                    interp_pos, tgt_quat,
                    lowerLimits=pb._lower_limits, upperLimits=pb._upper_limits,
                    jointRanges=pb._joint_ranges, restPoses=rest_poses,
                    maxNumIterations=200, residualThreshold=1e-5)
                q = np.array(joint_q[:len(pb.arm_indices)], dtype=float)
            return q
        finally:
            pb.update_robot(saved_q)

    def query_ik_joints(self, pos_world, quat_xyzw,
                        seed_q: "np.ndarray | None" = None) -> np.ndarray:
        """Raw IK solution without modifying PyBullet scene state."""
        import pybullet as p
        if self._pb_scene is None:
            raise RuntimeError("No PyBullet scene.")
        pb = self._pb_scene
        if seed_q is None:
            seed_q = pb.current_q.copy()
        arm_q_map  = dict(zip(pb.arm_indices, seed_q))
        rest_poses = [float(arm_q_map.get(j, 0.0)) for j in pb._movable]
        joint_q = p.calculateInverseKinematics(
            pb.robot_id, pb.tool0_link_idx,
            list(pos_world), list(quat_xyzw),
            lowerLimits=pb._lower_limits, upperLimits=pb._upper_limits,
            jointRanges=pb._joint_ranges, restPoses=rest_poses,
            maxNumIterations=200, residualThreshold=1e-5)
        return np.array(joint_q[:len(pb.arm_indices)], dtype=np.float64)

    # ── Gripper (real robot) ──────────────────────────────────────────────────

    def open_gripper(self, speed: int = 255, force: int = 10) -> None:
        g = self._gripper_conn()
        g.move_and_wait_for_pos(g.get_open_position(), speed, force)

    def close_gripper(self, speed: int = 255, force: int = 100) -> None:
        g = self._gripper_conn()
        g.move_and_wait_for_pos(g.get_closed_position(), speed, force)

    # ── Force monitoring ──────────────────────────────────────────────────────

    def poll_tcp_force(self) -> "np.ndarray | None":
        """Read TCP force [Fx,Fy,Fz] in Newtons (base frame). Real robot only."""
        if self._recv is None:
            try:
                self._recv = RTDEReceiveInterface(self._robot_ip)
            except Exception:
                return None
        try:
            return np.array(self._recv.getActualTCPForce()[:3], float)
        except Exception:
            return None

    def start_force_monitor(self, mode: str, on_trigger: "Callable",
                            threshold: "float | None" = None) -> None:
        """Start non-blocking force threshold monitoring (polled by tick()).

        mode='release': fires on_trigger when pull force > threshold
        mode='grasp':   fires on_trigger when contact force > threshold
        threshold: optional override (N) for the trigger delta; defaults to the
                   mode constant (_FORCE_RELEASE_THRESHOLD / _FORCE_GRASP_THRESHOLD).

        on_trigger() is called on the main thread (from tick()), so it is safe
        to update UI state directly.
        """
        self._force_baseline = self.poll_tcp_force()   # may be None — captured on first tick
        self._force_mode     = mode
        self._force_hits     = 0
        self._force_cb       = on_trigger
        self._force_last_t   = None
        self._force_threshold = (threshold if threshold is not None
                                 else (_FORCE_RELEASE_THRESHOLD if mode == 'release'
                                       else _FORCE_GRASP_THRESHOLD))
        print(f"[Robot] Force monitor started: mode={mode}, threshold={self._force_threshold:.1f} N")

    def stop_force_monitor(self) -> None:
        """Stop force monitoring without firing the callback."""
        if self._force_mode is not None:
            print(f"[Robot] Force monitor stopped (mode={self._force_mode})")
        self._force_mode     = None
        self._force_baseline = None
        self._force_hits     = 0
        self._force_cb       = None
        self._force_threshold = None


    # ── Grasp sequence (both modes) ───────────────────────────────────────────

    @property
    def tool_grasp_running(self) -> bool:
        if self.simulation:
            return self._sim_phase is not None
        return self._sim_phase is not None and self._sim_phase.startswith('real_')

    def execute_grasp(
        self,
        grasp_joints: "list | np.ndarray",
        on_complete:  "Callable[[bool], None] | None" = None,
        on_phase:     "Callable[[str], None] | None"  = None,
        q_approach:   "list | np.ndarray | None"      = None,
        q_above:      "list | np.ndarray | None"      = None,
        category:     str                              = "tool",
        board_normal: "list | np.ndarray | None"      = None,
    ) -> None:
        """Grasp sequence in both modes.

        on_phase(name), if given, fires as the sequence progresses through
        "approaching" -> "grasping" -> "grasped" -> "retracting" (in
        addition to the terminal on_complete(ok)).

        Tool  (q_above=None): approach → grasp → retract
        Part  (q_above given): approach → above → grasp → above → retract
        Bare  (q_approach=None): direct moveJ to grasp_joints

        When board_normal is supplied and q_approach is None, the method
        auto-computes q_approach (and q_above for category="part") via FK +
        solve_ik_from_config + _wrap_nearest — identical to robot_tester.py.
        """
        if self.tool_grasp_running:
            print("[Robot] Grasp already running — cancel first.")
            return

        # Auto-compute approach/above waypoints from board_normal when not pre-supplied
        if board_normal is not None and q_approach is None and self._pb_scene is not None:
            gj_arr  = np.array(grasp_joints, dtype=float)
            is_part = (category == "part")
            saved_q = self._pb_scene.current_q.copy()
            bn      = np.array(board_normal, float)
            bn     /= (np.linalg.norm(bn) + 1e-9)
            try:
                self._pb_scene.update_robot(gj_arr)
                grasp_T = self._pb_scene.update_tcp_bodies()
            finally:
                self._pb_scene.update_robot(saved_q)

            if grasp_T is not None:
                tcp_pos  = grasp_T[:3, 3]
                tcp_quat = ScipyR.from_matrix(grasp_T[:3, :3]).as_quat().tolist()
                if is_part:
                    pos_above    = tcp_pos + np.array([0., 0., 0.05])
                    pos_approach = pos_above + self._approach_dist * bn
                else:
                    pos_approach = tcp_pos + self._approach_dist * bn
                q_approach = _wrap_nearest(
                    self.solve_ik_from_config(gj_arr, pos_approach, tcp_quat), saved_q)
                if is_part:
                    q_above = _wrap_nearest(
                        self.solve_ik_from_config(gj_arr, pos_above, tcp_quat), q_approach)
                    grasp_joints = _wrap_nearest(gj_arr, q_above)
                else:
                    grasp_joints = _wrap_nearest(gj_arr, q_approach)

        if self.simulation:
            if self._pb_scene is None:
                if on_complete:
                    on_complete(False)
                return
            self._sim_on_complete  = on_complete
            self._sim_on_phase     = on_phase
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
                self._fire_phase(on_phase, "approaching")
            else:
                self._sim_runner = _PbJointRunner(
                    self._pb_scene.current_q.copy(),
                    self._sim_grasp_joints,
                )
                self._sim_phase = 'grasp'
                print("[Robot sim] moveJ → grasp_joints")
                self._fire_phase(on_phase, "grasping")
        else:
            if not _RTDE_AVAILABLE:
                print("[Robot] rtde_control not installed.")
                if on_complete:
                    on_complete(False)
                return
            # Stop servoJ and clear any active move_tcp before starting grasp
            self.servoStop()
            self._sim_phase        = None
            self._tracked_tcp_pos  = None
            self._tracked_tcp_quat = None
            self._tracked_tcp_cb   = None
            self._move_tcp_smooth  = None
            # Arm the real-robot grasp state machine — tick() drives it
            self._real_grasp_joints = np.array(grasp_joints, dtype=float)
            self._real_q_approach   = (np.array(q_approach, dtype=float)
                                       if q_approach is not None else None)
            self._real_q_above      = (np.array(q_above, dtype=float)
                                       if q_above is not None else None)
            self._real_on_complete  = on_complete
            self._real_on_phase     = on_phase
            self._real_move_sent    = False
            self._real_settle_start = None
            self._sim_phase         = 'real_open_pre'
            # Print current joints vs limits so we can spot near-limit starts
            q_now = self._last_q
            if q_now is not None and self._frax is not None:
                cfg  = self._frax._cbf_cfg
                lo   = np.degrees(cfg.q_min_t)
                hi   = np.degrees(cfg.q_max_t)
                qd   = np.degrees(q_now)
                names = ["Base","Shoulder","Elbow","Wrist1","Wrist2","Wrist3"]
                print("[Robot] Grasp start — joint positions vs CBF limits (deg):")
                for i, (n, v, l, h) in enumerate(zip(names, qd, lo, hi)):
                    margin_lo = v - l
                    margin_hi = h - v
                    warn = " *** CLOSE ***" if min(margin_lo, margin_hi) < 10 else ""
                    print(f"  J{i+1} {n:8s}: {v:7.2f}  [{l:.1f}, {h:.1f}]"
                          f"  margins +{margin_hi:.1f} / -{margin_lo:.1f}{warn}")
                print(f"  Target grasp joints (deg): {np.round(np.degrees(self._real_grasp_joints), 1).tolist()}")
            print("[Robot] Starting real grasp state machine")

    @staticmethod
    def _fire_phase(on_phase: "Callable[[str], None] | None", name: str) -> None:
        if on_phase is None:
            return
        try:
            on_phase(name)
        except Exception as e:
            print(f"[Robot] on_phase callback error: {e}")

    def cancel_motion(self) -> None:
        """Abort any running grasp/move and stop the arm."""
        self.stop_force_monitor()
        if self.simulation:
            cb = self._sim_on_complete
            self._sim_runner          = None
            self._sim_phase           = None
            self._sim_on_complete     = None
            self._sim_grasp_joints    = None
            self._sim_q_approach      = None
            self._sim_q_above         = None
            self._tracked_tcp_pos     = None
            self._tracked_tcp_quat    = None
            self._tracked_tcp_cb      = None
            if cb is not None:
                try:
                    cb(False)
                except Exception:
                    pass
        else:
            phase = self._sim_phase
            self._sim_phase         = None
            self._tracked_tcp_pos   = None
            self._tracked_tcp_quat  = None
            self._tracked_tcp_cb    = None
            self._move_tcp_smooth   = None
            cb = self._real_on_complete
            self._real_on_complete  = None
            self._real_on_phase     = None
            # Stop robot: if async moveJ was in flight, issue stopJ; otherwise servoStop
            if phase is not None and phase.startswith('real_') and self._real_move_sent:
                try:
                    self._rtde_ctrl_conn().stopJ(2.0)
                except Exception as e:
                    print(f"[Robot] cancel stopJ failed: {e}")
            else:
                self.servoStop()
            if cb is not None:
                try:
                    cb(False)
                except Exception:
                    pass

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

    def connect_gripper(self) -> None:
        """Pre-connect and calibrate the gripper at startup so the first grasp
        has no calibration delay. Called blocking during server init."""
        if self.simulation or not _GRIPPER_AVAILABLE:
            return
        try:
            self._gripper_conn()
        except Exception as e:
            print(f"[Robot] Gripper pre-connect failed: {e}")

    def _gripper_conn(self):
        if not _GRIPPER_AVAILABLE:
            raise RuntimeError("RobotiqGripper not available.")
        if self._gripper is None:
            g = RobotiqGripper()
            g.connect(self._robot_ip, 63352)
            try:
                g.activate()
            except Exception as e:
                print(f"[Robot] Gripper activation warning (continuing): {e}")
            self._gripper = g
            print("[Robot] Gripper ready.")
        return self._gripper

    # ── Internal: real-robot grasp state machine (driven by tick()) ─────────────

    def _tick_real_grasp(self, dt: float) -> None:
        """Advance the real-robot grasp state machine one tick.

        All RTDE access happens here, on the server's single loop thread —
        no concurrent access is possible, eliminating the 'another thread is
        controlling the robot' UR error.

        States: real_open_pre → real_approach? → real_above? → real_down
                → real_close → real_settle → real_open_post
                → real_ret_above? → real_retract? → done
        """
        phase = self._sim_phase
        sc    = self._speed_scale

        def _next(new_phase: str) -> None:
            self._sim_phase      = new_phase
            self._real_move_sent = False

        def _done(ok: bool) -> None:
            self._sim_phase        = None
            self._real_on_phase    = None
            cb = self._real_on_complete
            self._real_on_complete = None
            if cb is not None:
                try:
                    cb(ok)
                except Exception as e:
                    print(f"[Robot] grasp on_complete error: {e}")

        def _ctrl():
            try:
                return self._rtde_ctrl_conn()
            except Exception as e:
                print(f"[Robot] RTDE connect failed in grasp: {e}")
                _done(False)
                return None

        if phase == 'real_open_pre':
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] open_gripper failed: {e}")
            self._fire_phase(self._real_on_phase, "approaching")
            if self._real_q_approach is not None:
                _next('real_approach')
            elif self._real_q_above is not None:
                _next('real_above')
            else:
                _next('real_down')

        elif phase in ('real_approach', 'real_above', 'real_down',
                       'real_ret_above', 'real_retract'):
            target = {
                'real_approach':  self._real_q_approach,
                'real_above':     self._real_q_above,
                'real_down':      self._real_grasp_joints,
                'real_ret_above': self._real_q_above,
                'real_retract':   self._real_q_approach,
            }[phase]
            speed = {
                'real_approach':  0.5 * sc,
                'real_above':     0.3 * sc,
                'real_down':      0.2 * sc,
                'real_ret_above': 0.2 * sc,
                'real_retract':   (0.3 * sc if self._real_q_above is not None
                                   else 0.5 * sc),
            }[phase]

            if not self._real_move_sent:
                c = _ctrl()
                if c is None:
                    return
                try:
                    c.moveJ(list(target), float(speed), float(speed), asynchronous=True)
                    self._real_move_sent = True
                    print(f"[Robot] async moveJ → {phase}")
                except Exception as e:
                    print(f"[Robot] async moveJ failed ({phase}): {e}")
                    self._rtde_ctrl = None
                    _done(False)
            else:
                c = _ctrl()
                if c is None:
                    return
                try:
                    if c.isSteady():
                        if phase == 'real_approach':
                            _next('real_above' if self._real_q_above is not None
                                  else 'real_down')
                        elif phase == 'real_above':
                            _next('real_down')
                        elif phase == 'real_down':
                            self._fire_phase(self._real_on_phase, "grasping")
                            _next('real_close')
                        elif phase == 'real_ret_above':
                            _next('real_retract')
                        elif phase == 'real_retract':
                            _done(True)
                except Exception as e:
                    print(f"[Robot] isSteady failed ({phase}): {e}")
                    self._rtde_ctrl = None
                    _done(False)

        elif phase == 'real_close':
            try:
                self.close_gripper()
            except Exception as e:
                print(f"[Robot] close_gripper failed: {e}")
            self._real_settle_start = time.perf_counter()
            _next('real_settle')

        elif phase == 'real_settle':
            if time.perf_counter() - (self._real_settle_start or 0.0) >= 1.0:
                self._fire_phase(self._real_on_phase, "grasped")
                _next('real_open_post')

        elif phase == 'real_open_post':
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] open_gripper failed: {e}")
            self._fire_phase(self._real_on_phase, "retracting")
            if self._real_q_above is not None:
                _next('real_ret_above')
            elif self._real_q_approach is not None:
                _next('real_retract')
            else:
                _done(True)
