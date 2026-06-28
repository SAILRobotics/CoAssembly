"""OSCBFController — Frax OSC + CBF safety-filtered joint-velocity controller.

Requires the `proxy` conda environment (jax, cbfpy, frax all installed there).

Usage
-----
    ctrl = OSCBFController(T_world_base, urdf_path)
    ctrl.build(
        obstacle_boxes=[(center, half_extents), ...],   # world-frame AABB obstacles
        workspace_box=[xlo, ylo, zlo, xhi, yhi, zhi],  # world-frame workspace limits
    )

    # Inside a ~100 Hz servo loop:
    qdot = ctrl.compute_qdot(q, target_pos_world, target_rot_world)
    q_next = q + qdot * dt
    rtde_c.servoJ(q_next.tolist(), servo_speed, servo_accel, dt,
                  servo_lookahead, servo_gain)

    # When the scene changes (e.g. tool layout updated):
    ctrl.rebuild_cbf(new_obstacle_boxes)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

# ── sys.path: frax package lives inside this cbf/ folder ──────────────────────
_CBF_DIR = Path(__file__).parent
for _p in (_CBF_DIR / "frax", _CBF_DIR / "frax" / "examples", _CBF_DIR / "oscbf"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import jax
import jax.numpy as jnp
from cbfpy import CBF
from frax.robots.ur10e import load_ur10e
from frax.utils.rotation_utils import orientation_error_3D
from cbf_utils import OSCBFVelocityConfig
from oscbf.core.ur_collision_model import (
    positions_list, radii_list, pairs_sc,
    base_position, base_radius, base_sc_idxs,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")


# ── CBF config ─────────────────────────────────────────────────────────────────

class _CbfCfg(OSCBFVelocityConfig):
    """CBF barriers: joint limits + floor clearance + workspace box + obstacle boxes."""

    def __init__(self, robot, base_pos, base_R_flat,
                 obs_centers, obs_halves, obs_R_bw,
                 z_min, q_min, q_max, ws_lo, ws_hi, alpha):
        self.base_pos    = base_pos
        self.base_R_flat = base_R_flat
        self.obs_centers = obs_centers
        self.obs_halves  = obs_halves
        self.obs_R_bw    = obs_R_bw
        self.z_min       = z_min
        self.q_min_t     = q_min
        self.q_max_t     = q_max
        self.ws_lo       = ws_lo
        self.ws_hi       = ws_hi
        self._alpha      = alpha
        super().__init__(robot)

    def h_1(self, z):
        q       = z[: self.num_joints]
        base_R  = jnp.asarray(self.base_R_flat).reshape(3, 3)
        base_p  = jnp.asarray(self.base_pos)
        link_pos_b, link_r = self.robot.link_collision_data(q)
        link_pos = link_pos_b @ base_R.T + base_p          # world frame

        h_jlo   = q - jnp.asarray(self.q_min_t)
        h_jhi   = jnp.asarray(self.q_max_t) - q
        h_floor = link_pos[:, 2] - link_r - self.z_min
        parts   = [h_jlo, h_jhi, h_floor]

        if self.obs_centers:
            obs_c = jnp.asarray(self.obs_centers)
            obs_h = jnp.asarray(self.obs_halves)
            R_bw  = jnp.asarray(self.obs_R_bw).reshape(-1, 3, 3)
            d_world = link_pos[:, None, :] - obs_c[None, :, :]
            d_box   = jnp.einsum("oij,soj->soi", R_bw, d_world)
            inside  = jnp.abs(d_box) - obs_h[None, :, :]
            sdf     = jnp.linalg.norm(jnp.maximum(inside, 0.0), axis=2)
            parts.append((sdf - link_r[:, None]).reshape(-1))

        ee_tmat  = self.robot.ee_transform(q)
        ee_world = base_R @ ee_tmat[:3, 3] + base_p
        parts.append(jnp.concatenate([
            ee_world - jnp.asarray(self.ws_lo),
            jnp.asarray(self.ws_hi) - ee_world,
        ]))
        return jnp.concatenate(parts)

    def alpha(self, h):
        return self._alpha * h


# ── Main controller class ──────────────────────────────────────────────────────

class OSCBFController:
    """Frax OSC + CBF safety-filtered velocity controller for UR10e.

    Parameters
    ----------
    T_world_base : (4,4) ndarray
        Robot base pose in world frame (from calibration).
    urdf_path : str | Path
        Path to the UR10e URDF used by frax (same one PyBullet uses).
    kp_pos : float
        OSC position gain.
    kp_ori : float
        OSC orientation gain.
    cbf_alpha : float
        CBF class-K gain — higher = tighter safety margins.
    qdot_max : float
        Joint velocity clamp (rad/s) applied after CBF filter.
    """

    # UR10e has a 180° yaw offset between the URDF base convention and the
    # world-frame base pose from calibration (same correction as linux_controller4).
    _RZ180 = ScipyR.from_euler('z', np.pi).as_matrix()

    def __init__(
        self,
        T_world_base: np.ndarray,
        urdf_path: "str | Path",
        kp_pos:    float = 200.0,
        kp_ori:    float = 100.0,
        cbf_alpha: float = 15.0,
        qdot_max:  float = 5.0,
    ):
        self._base_pos     = T_world_base[:3, 3].copy()
        self._base_R       = T_world_base[:3, :3].copy()
        self._base_R_full  = self._base_R @ self._RZ180   # frax convention
        self._kp_task      = np.array([kp_pos] * 3 + [kp_ori] * 3, dtype=np.float64)
        self._cbf_alpha    = float(cbf_alpha)
        self._qdot_max     = float(qdot_max)

        frax_collision = {
            "positions":      positions_list,
            "radii":          radii_list,
            "root_positions": (base_position,),
            "root_radii":     (base_radius,),
            "root_sc_pairs":  tuple((0, idx) for idx in base_sc_idxs),
            "root_sc_tols":   tuple(0.0 for _ in base_sc_idxs),
            "body_sc_pairs":  pairs_sc,
            "body_sc_tols":   tuple(0.0 for _ in pairs_sc),
        }
        self.robot = load_ur10e(str(urdf_path), collision_data=frax_collision)
        print(f"[OSCBFController] frax Manipulator: {self.robot.num_joints} joints")

        kp_jnp = jnp.array(self._kp_task)
        robot   = self.robot

        @jax.jit
        def _nominal_osc(q, ee_pos, ee_rot, des_pos, des_rot,
                         des_vel, des_omega, J, M_inv):
            task_inertia_inv = J @ M_inv @ J.T
            task_inertia     = jnp.linalg.inv(task_inertia_inv)
            J_bar    = M_inv @ J.T @ task_inertia
            pos_err  = ee_pos - des_pos
            rot_err  = orientation_error_3D(ee_rot, des_rot)
            task_err = jnp.concatenate([pos_err, rot_err])
            twist    = jnp.concatenate([des_vel, des_omega])
            return J_bar @ (twist - kp_jnp * task_err)

        self._nominal_osc = _nominal_osc

        # Warm up JIT
        _q0  = jnp.zeros(robot.num_joints)
        _Mv, _J, _eet = robot.dynamically_consistent_velocity_control_matrices(_q0)
        _nominal_osc(_q0, _eet[:3, 3], _eet[:3, :3],
                     _eet[:3, 3], _eet[:3, :3],
                     jnp.zeros(3), jnp.zeros(3), _J, _Mv)
        print("[OSCBFController] OSC JIT compiled.")

        self.cbf: "CBF | None" = None
        self._cbf_q_min: tuple = ()
        self._cbf_q_max: tuple = ()
        self._cbf_z_min: float = 0.0
        self._cbf_ws_lo: tuple = (-2.0, -2.0, -0.05)
        self._cbf_ws_hi: tuple = ( 2.0,  2.0,  2.00)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        obstacle_boxes: list = (),
        workspace_box:  "list | np.ndarray | None" = None,
        q_min_deg: "list | np.ndarray | None" = None,
        q_max_deg: "list | np.ndarray | None" = None,
        joint_limit_margin_deg: float = 0.0,
    ) -> None:
        """Build frax model + CBF. Call once before entering the control loop.

        Parameters
        ----------
        obstacle_boxes : list of (center, half_extents) or (center, half_extents, yaw_deg)
            World-frame axis-aligned box obstacles (yaw_deg rotates around world Z).
        workspace_box : [xlo, ylo, zlo, xhi, yhi, zhi] in world frame.
        q_min_deg / q_max_deg : per-joint limits in degrees (length-6).
            Defaults to UR10e physical limits.
        joint_limit_margin_deg : extra inward margin on each joint limit.
        """
        if workspace_box is not None:
            ws = np.asarray(workspace_box, dtype=np.float64)
            self._cbf_ws_lo = tuple(float(v) for v in ws[:3])
            self._cbf_ws_hi = tuple(float(v) for v in ws[3:])
            self._cbf_z_min = float(ws[2])

        # Default UR10e joint limits (degrees)
        _default_min = np.array([-360., -360., -360., -360., -360., -360.])
        _default_max = np.array([ 360.,  360.,  360.,  360.,  360.,  360.])
        q_min = np.deg2rad((np.asarray(q_min_deg, float)
                            if q_min_deg is not None else _default_min)
                           + joint_limit_margin_deg)
        q_max = np.deg2rad((np.asarray(q_max_deg, float)
                            if q_max_deg is not None else _default_max)
                           - joint_limit_margin_deg)
        self._cbf_q_min = tuple(float(v) for v in q_min)
        self._cbf_q_max = tuple(float(v) for v in q_max)

        self.rebuild_cbf(obstacle_boxes)

    def rebuild_cbf(self, obstacle_boxes: list = ()) -> None:
        """Rebuild the CBF (e.g. when the tool layout changes)."""
        obstacles   = [self._unpack_box(o) for o in obstacle_boxes]
        obs_centers = np.array([c for c, _, _ in obstacles]) if obstacles else np.zeros((0, 3))
        obs_halves  = np.array([h for _, h, _ in obstacles]) if obstacles else np.zeros((0, 3))
        obs_yaws    = np.array([y for _, _, y in obstacles]) if obstacles else np.zeros(0)
        obs_R_bw    = np.array([
            ScipyR.from_euler('z', y, degrees=True).as_matrix().T
            for y in obs_yaws
        ]).reshape(-1, 3, 3)
        obs_c_t = tuple(map(tuple, obs_centers.tolist())) if len(obs_centers) else ()
        obs_h_t = tuple(map(tuple, obs_halves.tolist()))  if len(obs_halves)  else ()
        obs_R_t = tuple(tuple(float(v) for v in R.ravel())
                        for R in obs_R_bw) if len(obs_R_bw) else ()

        base_pos_t  = tuple(float(v) for v in self._base_pos)
        base_R_flat = tuple(float(v) for v in self._base_R_full.ravel())

        def _make(robot_):
            return _CbfCfg(
                robot_,
                base_pos_t, base_R_flat,
                obs_c_t, obs_h_t, obs_R_t,
                self._cbf_z_min,
                self._cbf_q_min, self._cbf_q_max,
                self._cbf_ws_lo, self._cbf_ws_hi,
                self._cbf_alpha,
            )

        self.cbf = CBF.from_config(_make(self.robot))
        print(f"[OSCBFController] CBF built — {len(obstacles)} obstacle(s) "
              "+ floor + joint limits + workspace.")

    def compute_qdot(
        self,
        q:                np.ndarray,
        target_pos_world: np.ndarray,
        target_rot_world: np.ndarray,
        des_vel:          "np.ndarray | None" = None,
        des_omega:        "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Compute CBF-filtered joint velocities.

        Parameters
        ----------
        q : (6,) current joint angles in radians.
        target_pos_world : (3,) desired EE position in world frame.
        target_rot_world : (3,3) desired EE rotation matrix in world frame.
        des_vel / des_omega : (3,) desired Cartesian / angular velocity feed-forwards.

        Returns
        -------
        qdot : (6,) safe joint velocities (rad/s), clipped to ±qdot_max.
        """
        if self.cbf is None:
            raise RuntimeError("Call build() before compute_qdot().")

        q_jnp = jnp.array(q, dtype=jnp.float64)
        M_inv, J, ee_tmat = self.robot.dynamically_consistent_velocity_control_matrices(q_jnp)

        ee_pos_base  = np.array(ee_tmat[:3, 3])
        ee_rot_base  = np.array(ee_tmat[:3, :3])

        # Convert world-frame targets to robot base frame
        target_pos_base = self._base_R_full.T @ (target_pos_world - self._base_pos)
        target_rot_base = self._base_R_full.T @ target_rot_world

        _dv = jnp.zeros(3) if des_vel   is None else jnp.array(des_vel,   dtype=jnp.float64)
        _dw = jnp.zeros(3) if des_omega is None else jnp.array(des_omega, dtype=jnp.float64)

        qdot = self._nominal_osc(
            q_jnp,
            jnp.array(ee_pos_base), jnp.array(ee_rot_base),
            jnp.array(target_pos_base), jnp.array(target_rot_base),
            _dv, _dw, J, M_inv,
        )
        qdot = np.asarray(self.cbf.safety_filter(q_jnp, qdot))
        return np.clip(qdot, -self._qdot_max, self._qdot_max)

    def filter_qdot(self, q: np.ndarray, qdot_nominal: np.ndarray) -> np.ndarray:
        """Apply only the CBF safety filter to a pre-computed nominal qdot.

        Use this when the nominal motion comes from an external source (e.g. PyBullet IK)
        rather than the internal frax OSC controller.

        Parameters
        ----------
        q            : (6,) current joint angles in radians.
        qdot_nominal : (6,) nominal joint velocities to be safety-filtered.

        Returns
        -------
        qdot : (6,) CBF-filtered joint velocities, clipped to ±qdot_max.
        """
        if self.cbf is None:
            raise RuntimeError("Call build() before filter_qdot().")
        q_jnp    = jnp.array(q,            dtype=jnp.float64)
        qdot_jnp = jnp.array(qdot_nominal, dtype=jnp.float64)
        qdot     = np.asarray(self.cbf.safety_filter(q_jnp, qdot_jnp))
        return np.clip(qdot, -self._qdot_max, self._qdot_max)

    def is_converged(
        self,
        q:                np.ndarray,
        target_pos_world: np.ndarray,
        target_rot_world: np.ndarray,
        pos_tol:  float = 0.01,
        ori_tol:  float = 0.05,
    ) -> bool:
        """True when EE is within pos_tol (m) and ori_tol (rad) of the target."""
        q_jnp = jnp.array(q, dtype=jnp.float64)
        _, _, ee_tmat = self.robot.dynamically_consistent_velocity_control_matrices(q_jnp)
        ee_pos_w = self._base_R_full @ np.array(ee_tmat[:3, 3]) + self._base_pos
        ee_rot_b = np.array(ee_tmat[:3, :3])
        target_rot_base = self._base_R_full.T @ target_rot_world
        R_err    = ee_rot_b @ target_rot_base.T
        ori_err  = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
        pos_err  = float(np.linalg.norm(ee_pos_w - target_pos_world))
        return pos_err < pos_tol and ori_err < ori_tol

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_box(obs) -> tuple:
        """Accept (center, half), (center, half, yaw_deg), or dict."""
        if isinstance(obs, dict):
            return (np.asarray(obs["center"],  float),
                    np.asarray(obs["half"],     float),
                    float(obs.get("yaw_deg", 0.0)))
        if len(obs) == 2:
            return np.asarray(obs[0], float), np.asarray(obs[1], float), 0.0
        c, h, y = obs
        return np.asarray(c, float), np.asarray(h, float), float(y)
