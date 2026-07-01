#!/usr/bin/env python3
"""linux_controller4.py — PyBullet-free variant of linux_controller3.

Key differences from lc3:
- No PyBullet / PyBulletScene dependency.
- ghost_pos / ghost_euler / ghost_q are plain instance variables.
- Simulated controller GUI uses Dear PyGui (pip install dearpygui) in a
  background thread instead of PyBullet debug parameters.
- PyBullet IK fallback removed — discrete point-to-point moves require MoveIt2.
- RViz remains the only visualization (joint_states, ghost_gripper, obstacles,
  gripper_collision_box, display_planned_path topics unchanged).
"""
import sys
import argparse
import json
import subprocess
import time
import threading
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

# ── Import paths ───────────────────────────────────────────────────────────────
def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "main_logic" / "linux_config.py").exists():
            return path
    return start.parent


def prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


ROOT = find_project_root(Path(__file__).resolve())
PYMOVEIT2_ROOT = ROOT / "ros_ws" / "src" / "pymoveit2"
print(f"[linux_controller4] Project root: {ROOT}")
print(f"[linux_controller4] PyMoveIt2 root: {PYMOVEIT2_ROOT}")

for path in (ROOT, ROOT / "main_logic", PYMOVEIT2_ROOT):
    prepend_sys_path(path)

# ── frax ──────────────────────────────────────────────────────────────────────
FRAX_ROOT = ROOT / "main_logic" / "frax"
for p in (str(FRAX_ROOT), str(FRAX_ROOT / "examples")):
    prepend_sys_path(Path(p))

import jax
import jax.numpy as jnp
from cbfpy import CBF

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from frax.robots.ur10e import load_ur10e
from frax.utils.rotation_utils import orientation_error_3D
from cbf_utils import OSCBFVelocityConfig

class _Lc4CbfCfg(OSCBFVelocityConfig):
    def __init__(self, robot_, base_pos, base_R_flat,
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
        super().__init__(robot_)

    def h_1(self, z):
        q      = z[: self.num_joints]                          # joint angles only (strip velocity state)
        base_R = jnp.asarray(self.base_R_flat).reshape(3, 3)  # robot base rotation (world frame)
        base_p = jnp.asarray(self.base_pos)                   # robot base position (world frame)
        robot_pos_b, robot_r = self.robot.link_collision_data(q)  # link centers (base frame) + radii
        robot_pos = robot_pos_b @ base_R.T + base_p           # link centers in world frame

        h_jlo   = q - jnp.asarray(self.q_min_t)              # > 0 when above lower joint limit
        h_jhi   = jnp.asarray(self.q_max_t) - q              # > 0 when below upper joint limit
        h_floor = robot_pos[:, 2] - robot_r - self.z_min     # clearance of each link above the floor
        parts   = [h_jlo, h_jhi, h_floor]

        if self.obs_centers:
            obs_c = jnp.asarray(self.obs_centers)             # obstacle box centers (world frame)
            obs_h = jnp.asarray(self.obs_halves)              # obstacle half-extents per axis
            R_bw  = jnp.asarray(self.obs_R_bw).reshape(-1, 3, 3)  # world→box rotations per obstacle
            d_world = robot_pos[:, None, :] - obs_c[None, :, :]    # link-to-obstacle vectors (world)
            d_box   = jnp.einsum("oij,soj->soi", R_bw, d_world)   # same vectors in each box's local frame
            inside  = jnp.abs(d_box) - obs_h[None, :, :]           # positive = outside box on that axis
            sdf     = jnp.linalg.norm(jnp.maximum(inside, 0.0), axis=2)  # box SDF (0 inside, >0 outside)
            parts.append((sdf - robot_r[:, None]).reshape(-1))      # margin = SDF minus link radius

        ee_tmat  = self.robot.ee_transform(q)                 # end-effector pose in base frame
        ee_world = base_R @ ee_tmat[:3, 3] + base_p          # end-effector position in world frame
        parts.append(jnp.concatenate([
            ee_world - jnp.asarray(self.ws_lo),               # distance from lower workspace boundary
            jnp.asarray(self.ws_hi) - ee_world,               # distance from upper workspace boundary
        ]))
        return jnp.concatenate(parts)  # all margins stacked; QP enforces each ≥ 0

    def alpha(self, h):
        return self._alpha * h  # linear class-K: push harder the closer you are to the boundary


# ── RTDE ──────────────────────────────────────────────────────────────────────
try:
    import rtde_receive
except ImportError:
    rtde_receive = None

try:
    import rtde_control
except ImportError:
    rtde_control = None

# ── ROS2 ──────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.duration import Duration as RclpyDuration
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, JointConstraint
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
    
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[linux_controller4] WARNING: rclpy not found — ROS2 publishing disabled.")

try:
    from pymoveit2 import MoveIt2
    HAS_PYMOVEIT2 = True
except ImportError:
    HAS_PYMOVEIT2 = False

try:
    from quest_direct_bridge import QuestDirectBridge
    HAS_QUEST_BRIDGE = True
except ImportError:
    HAS_QUEST_BRIDGE = False

# ── Dear PyGui ────────────────────────────────────────────────────────────────
try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except ImportError:
    HAS_DPG = False
    print("[linux_controller4] WARNING: dearpygui not found — simulated controller GUI disabled.")

from main_logic.ghost_gripper_manager import GhostGripperManager
from main_logic.linux_config2 import (
    Config,
    ARM_JOINT_NAMES,
    ROS_GRIPPER_JOINT_NAMES,
    ROS_GRIPPER_MIMIC,
    GRIPPER_MOUNT_OFFSET,
    obstacle_box,
)
from main_logic.utils.unity_conversion import (
    open3d_to_unity_vector,
    open3d_to_unity_quaternion,
)

# =============================================================================
# Main controller class
# =============================================================================

class LinuxController:

    def __init__(self, cfg: Config, robot_ip: str | None,
                 ctrl_port=5200, state_port=5201, config_port=5202,
                 ros_node=None, use_moveit: bool = False,
                 quest_bridge=None, quest_ip: str = "192.168.50.201"):

        self.cfg = cfg
        self.fixed_joints = list(cfg.fixed_joint_angles_deg.keys())
        self.fixed_joint_rads = np.deg2rad(
            [cfg.fixed_joint_angles_deg[j] for j in self.fixed_joints])

        self.active_preset = "default"
        if "default" in cfg.joint_limit_presets:
            _min, _max = cfg.joint_limit_presets["default"]
            cfg.joint_min_deg = np.asarray(_min, float)
            cfg.joint_max_deg = np.asarray(_max, float)

        # ── Calibration ───────────────────────────────────────────────────────
        self.load_calibration()
        print(f"[linux_controller4] Calibration loaded. "
              f"Markers: {list(self.T_world_markers.keys())}")

        # ── Ghost state (replaces PyBulletScene.ghost_*) ──────────────────────
        self.ghost_pos   = np.zeros(3)
        self.ghost_euler = np.zeros(3)
        self.ghost_q     = 0.0

        self.q_current = np.deg2rad(cfg.home_q_deg).copy()
        self.gripper_q = 0.0
        self._shortest_wrist3  = True
        self._exec_speed_scale = 1.0
        self._sim_btn_state    = {"A": False, "B": False, "add_wp": False,
                                  "wp_next": False, "wp_prev": False}
        self.wp_manager        = GhostGripperManager()

        # ── RTDE ──────────────────────────────────────────────────────────────
        self.rtde, self.rtde_c = self.connect_rtde(robot_ip)

        # ── frax / CBF ────────────────────────────────────────────────────────
        self.setup_oscbf()

        # ── ZMQ ───────────────────────────────────────────────────────────────
        self.quest_bridge = quest_bridge
        ctx = zmq.Context.instance()
        if quest_bridge is None:
            self.controller_recv = ctx.socket(zmq.PULL)
            self.controller_recv.setsockopt(zmq.CONFLATE, 1)
            self.controller_recv.setsockopt(zmq.RCVHWM, 1)
            self.controller_recv.bind(f"tcp://0.0.0.0:{ctrl_port}")
            self.state_send = ctx.socket(zmq.PUSH)
            self.state_send.setsockopt(zmq.CONFLATE, 1)
            self.state_send.setsockopt(zmq.SNDHWM, 1)
            self.state_send.setsockopt(zmq.LINGER, 0)
            self.state_send.bind(f"tcp://0.0.0.0:{state_port}")
            self.config_rep = ctx.socket(zmq.REP)
            self.config_rep.bind(f"tcp://0.0.0.0:{config_port}")
            print(f"[linux_controller4] controller_recv:{ctrl_port}  "
                  f"state_send:{state_port}  config_rep:{config_port}")
        else:
            self.controller_recv = self.state_send = self.config_rep = None
            print("[linux_controller4] Direct-connect mode — Windows ZMQ sockets disabled.")

        self.unity_base_pub = ctx.socket(zmq.PUB)
        self.unity_base_pub.setsockopt(zmq.LINGER, 0)
        self.unity_base_pub.connect(f"tcp://{quest_ip}:5000")
        self.unity_joint_pub = ctx.socket(zmq.PUB)
        self.unity_joint_pub.setsockopt(zmq.LINGER, 0)
        self.unity_joint_pub.connect(f"tcp://{quest_ip}:5001")
        self.unity_gripper_pub = ctx.socket(zmq.PUB)
        self.unity_gripper_pub.setsockopt(zmq.LINGER, 0)
        self.unity_gripper_pub.connect(f"tcp://{quest_ip}:5004")
        self.unity_ghost_gripper_pub = ctx.socket(zmq.PUB)
        self.unity_ghost_gripper_pub.setsockopt(zmq.LINGER, 0)
        self.unity_ghost_gripper_pub.connect(f"tcp://{quest_ip}:5007")
        self.unity_ghost_pub = ctx.socket(zmq.PUB)
        self.unity_ghost_pub.setsockopt(zmq.LINGER, 0)
        self.unity_ghost_pub.connect(f"tcp://{quest_ip}:5006")
        print(f"[linux_controller4] Unity publishers → {quest_ip}")

        self.ctrl_lock = threading.Lock()
        self.latest_right_T = None
        self.latest_left_T  = None
        self.latest_head_T  = None
        self.latest_right_buttons = {}
        self.latest_left_buttons  = {}
        self.sim_ctrl_active = False   # True once DPG GUI is running
        self._unity_scene_logged = False
        self._ghost_no_tf_ctr = 0  # consecutive frames without TF broadcast (for diagnostics)
        self._T_world_ghost = None  # cached world→quest_ctrl_base transform for TCP axes

        # ── ROS2 ──────────────────────────────────────────────────────────────
        self.ros    = ros_node
        self.js_pub = self.obj_pub = None
        self.moveit2 = None
        # T_scaled: casing frame → controller frame (from registration script)
        _t_path = (Path(__file__).resolve().parent
                   / "initialization" / "stl_files" / "casing_T_scaled.npy")
        if _t_path.exists():
            self._T_casing_to_ctrl = np.load(str(_t_path))
            print(f"[linux_controller4] Loaded casing registration T_scaled from {_t_path}")
        else:
            self._T_casing_to_ctrl = None
            print(f"[linux_controller4] WARNING: casing_T_scaled.npy not found — "
                  "ghost assembly will not be broadcast. Run align_controller_meshes.py first.")

        if ros_node is not None:
            self.setup_ros_publishers(ros_node)
            if use_moveit:
                self.setup_moveit_client(ros_node)

        self.running = True

    # ── Initialization helpers ─────────────────────────────────────────────────

    @staticmethod
    def unpack_obstacle_box(obstacle):
        if isinstance(obstacle, dict):
            center  = obstacle["center"]
            half    = obstacle["half"]
            yaw_deg = obstacle.get("yaw_deg", 0.0)
        elif len(obstacle) == 2:
            center, half = obstacle
            yaw_deg = 0.0
        elif len(obstacle) == 3:
            center, half, yaw_deg = obstacle
        else:
            raise ValueError(
                "Obstacle boxes must be (center, half), (center, half, yaw_deg), "
                "or {'center': ..., 'half': ..., 'yaw_deg': ...}."
            )
        return (
            np.asarray(center,  dtype=np.float64),
            np.asarray(half,    dtype=np.float64),
            float(yaw_deg),
        )

    def load_calibration(self):
        cfg = self.cfg
        p1 = cfg.results_dir / "phase1_results.npz"
        if not p1.exists():
            raise FileNotFoundError(f"Calibration not found: {p1}")
        d1 = np.load(p1)
        self.T_world_base    = d1["T_world_base"]
        self.T_world_deskcam = d1["T_world_deskcam"]
        self.T_world_markers = {}
        if "object_marker_ids" in d1:
            for mid in d1["object_marker_ids"]:
                key = f"T_world_marker_{mid}"
                if key in d1:
                    self.T_world_markers[int(mid)] = d1[key]
        self.T_tcp_handcam = None
        p2 = cfg.results_dir / "phase2_results.npz"
        if p2.exists():
            self.T_tcp_handcam = np.load(p2)["T_tcp_handcam"]
        self.K_desk = None
        intr1 = cfg.results_dir.parent / "phase1" / "intrinsics.npz"
        if intr1.exists():
            self.K_desk = np.load(intr1)["camera_matrix"]

    def connect_rtde(self, robot_ip: str | None):
        if robot_ip is None:
            return None, None
        rtde, rtde_c = None, None
        if rtde_receive is None:
            print("[linux_controller4] rtde_receive not installed — IK-only mode.")
        else:
            try:
                r = rtde_receive.RTDEReceiveInterface(robot_ip)
                _ = r.getActualQ()
                print(f"[linux_controller4] RTDE receive connected to {robot_ip}")
                rtde = r
            except Exception as e:
                print(f"[linux_controller4] RTDE unavailable ({e}) — IK-only mode.")
        if rtde_control is None:
            print("[linux_controller4] rtde_control not installed — no speedJ.")
        else:
            try:
                rtde_c = rtde_control.RTDEControlInterface(robot_ip)
                print(f"[linux_controller4] RTDE control connected to {robot_ip}")
            except Exception as e:
                print(f"[linux_controller4] RTDEControl unavailable ({e}) — no speedJ.")
        return rtde, rtde_c

    def setup_oscbf(self):
        cfg = self.cfg

        self._base_pos = self.T_world_base[:3, 3].copy()
        self._base_R   = self.T_world_base[:3, :3].copy()
        _Rz180 = ScipyR.from_euler('z', np.pi).as_matrix()
        self._base_R_full = self._base_R @ _Rz180


        from oscbf.core.ur_collision_model import (
            positions_list, radii_list, pairs_sc,
            base_position, base_radius, base_sc_idxs,
        )
        frax_collision_data = {
            "positions":      positions_list,
            "radii":          radii_list,
            "root_positions": (base_position,),
            "root_radii":     (base_radius,),
            "root_sc_pairs":  tuple((0, idx) for idx in base_sc_idxs),
            "root_sc_tols":   tuple(0.0 for _ in base_sc_idxs),
            "body_sc_pairs":  pairs_sc,
            "body_sc_tols":   tuple(0.0 for _ in pairs_sc),
        }

        self.robot = load_ur10e(str(cfg.urdf_path), collision_data=frax_collision_data)
        print(f"[linux_controller4] frax Manipulator: {self.robot.num_joints} joints")

        robot   = self.robot
        kp_task = jnp.array([float(cfg.kp_pos)] * 3 + [float(cfg.kp_ori)] * 3)

        @jax.jit
        def _nominal_osc(_q, ee_pos, ee_rot, des_pos, des_rot,
                         des_vel, des_omega, J, M_inv):
            task_inertia_inv = J @ M_inv @ J.T
            task_inertia     = jnp.linalg.inv(task_inertia_inv)
            J_bar = M_inv @ J.T @ task_inertia
            pos_err  = ee_pos - des_pos
            rot_err  = orientation_error_3D(ee_rot, des_rot)
            task_err = jnp.concatenate([pos_err, rot_err])
            twist_des = jnp.concatenate([des_vel, des_omega])
            return J_bar @ (twist_des - kp_task * task_err)

        self._nominal_osc = _nominal_osc
        _q0 = jnp.zeros(robot.num_joints)
        _M_inv, _J, _eet = robot.dynamically_consistent_velocity_control_matrices(_q0)
        _ = _nominal_osc(_q0, _eet[:3, 3], _eet[:3, :3],
                         _eet[:3, 3], _eet[:3, :3],
                         jnp.zeros(3), jnp.zeros(3), _J, _M_inv)
        print("[linux_controller4] frax OSC JIT compiled.")

        self._cbf_base_pos_t  = tuple(float(v) for v in self._base_pos)
        self._cbf_base_R_flat = tuple(float(v) for v in self._base_R_full.ravel())
        q_min = np.deg2rad(np.asarray(cfg.joint_min_deg, float)
                           + float(cfg.joint_limit_margin_deg))
        q_max = np.deg2rad(np.asarray(cfg.joint_max_deg, float)
                           - float(cfg.joint_limit_margin_deg))
        self._cbf_q_min_t = tuple(float(v) for v in q_min)
        self._cbf_q_max_t = tuple(float(v) for v in q_max)
        self._cbf_z_min   = float(cfg.workspace_box[2])
        self._cbf_ws_lo   = tuple(float(v) for v in cfg.workspace_box[:3])
        self._cbf_ws_hi   = tuple(float(v) for v in cfg.workspace_box[3:])
        self._cbf_alpha   = float(cfg.cbf_alpha)

        self._build_cbf()

    def _build_cbf(self):
        cfg = self.cfg
        # Re-derive joint limits from current cfg so preset switches take effect
        margin = float(cfg.joint_limit_margin_deg)
        self._cbf_q_min_t = tuple(float(v) for v in
                                   np.deg2rad(np.asarray(cfg.joint_min_deg, float) + margin))
        self._cbf_q_max_t = tuple(float(v) for v in
                                   np.deg2rad(np.asarray(cfg.joint_max_deg, float) - margin))
        all_boxes = list(cfg.obstacle_boxes)
        obstacles = [self.unpack_obstacle_box(o) for o in all_boxes]
        obs_centers = np.array([c for c, _, _ in obstacles]) if obstacles else np.zeros((0, 3))
        obs_halves  = np.array([h for _, h, _ in obstacles]) if obstacles else np.zeros((0, 3))
        obs_yaws    = np.array([y for _, _, y in obstacles]) if obstacles else np.zeros(0)
        obs_R_bw    = np.array([
            ScipyR.from_euler('z', yaw, degrees=True).as_matrix().T
            for yaw in obs_yaws
        ]).reshape(-1, 3, 3)
        obs_c_t = tuple(map(tuple, obs_centers.tolist())) if len(obs_centers) else ()
        obs_h_t = tuple(map(tuple, obs_halves.tolist()))  if len(obs_halves)  else ()
        obs_R_t = tuple(tuple(float(v) for v in R.ravel())
                        for R in obs_R_bw) if len(obs_R_bw) else ()

        def _make(robot_):
            return _Lc4CbfCfg(
                robot_,
                self._cbf_base_pos_t, self._cbf_base_R_flat,
                obs_c_t, obs_h_t, obs_R_t,
                self._cbf_z_min, self._cbf_q_min_t, self._cbf_q_max_t,
                self._cbf_ws_lo, self._cbf_ws_hi, self._cbf_alpha,
            )

        self.cbf = CBF.from_config(_make(self.robot))
        print(f"[linux_controller4] CBF built — {len(all_boxes)} obstacle box(es) "
              "+ floor + joint limits + workspace box.")

    # ── Simulated controller GUI (Dear PyGui) ──────────────────────────────────

    def setup_simulated_controller_gui(self, ee_pos=None, ee_euler_deg=None):
        if self.sim_ctrl_active:
            return
        if not HAS_DPG:
            print("[linux_controller4] dearpygui not available — simulated GUI disabled.")
            return
        x0, y0, z0    = (ee_pos       if ee_pos       is not None else [0.35, 0.0, 0.35])
        rx0, ry0, rz0 = (ee_euler_deg if ee_euler_deg is not None else [0.0,  0.0, 0.0 ])

        self._sim_btn_state = {"A": False, "B": False}
        self._exec_speed_scale = 1.0
        self._shortest_wrist3 = True

        # Snapshot obstacle list and preset names at GUI-creation time (safe for closure)
        _obs_list = [
            (f"obs_{i}", *self.unpack_obstacle_box(obs))
            for i, obs in enumerate(self.cfg.obstacle_boxes)
        ]
        _preset_names = list(self.cfg.joint_limit_presets.keys())

        def _gui_thread():
            dpg.create_context()
            with dpg.window(label="Simulated Controller", width=440, height=960, no_close=True):
                dpg.add_text("End-effector target")
                dpg.add_slider_float(label="x",          tag="sim_x",
                                     min_value=-1.0, max_value=1.0,   default_value=x0)
                dpg.add_slider_float(label="y",          tag="sim_y",
                                     min_value=-0.8, max_value=0.8,   default_value=y0)
                dpg.add_slider_float(label="z",          tag="sim_z",
                                     min_value=0.0,  max_value=1.0,   default_value=z0)
                dpg.add_separator()
                dpg.add_text("Orientation (XYZ Euler, degrees)")
                dpg.add_slider_float(label="rx_deg",     tag="sim_rx",
                                     min_value=-180., max_value=180., default_value=rx0)
                dpg.add_slider_float(label="ry_deg",     tag="sim_ry",
                                     min_value=-180., max_value=180., default_value=ry0)
                dpg.add_slider_float(label="rz_deg",     tag="sim_rz",
                                     min_value=-180., max_value=180., default_value=rz0)
                dpg.add_separator()
                dpg.add_text("Buttons / triggers")
                dpg.add_slider_float(label="index_trigger",  tag="sim_idx",
                                     min_value=0., max_value=1., default_value=0.)
                dpg.add_slider_float(label="middle_trigger", tag="sim_mid",
                                     min_value=0., max_value=1., default_value=0.)
                dpg.add_separator()
                dpg.add_button(label="A  (preview/execute)", tag="sim_A", width=200,
                               callback=lambda: self._sim_btn_state.__setitem__("A", True))
                dpg.add_button(label="B  (remove WP / cancel)", tag="sim_B", width=200,
                               callback=lambda: self._sim_btn_state.__setitem__("B", True))
                dpg.add_separator()
                dpg.add_text("Waypoints")
                dpg.add_button(label="Add Waypoint  (= index trigger)", tag="sim_add_wp", width=200,
                               callback=lambda: self._sim_btn_state.__setitem__("add_wp", True))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="← Prev WP", width=97,
                                   callback=lambda: self._sim_btn_state.__setitem__("wp_prev", True))
                    dpg.add_button(label="Next WP →", width=97,
                                   callback=lambda: self._sim_btn_state.__setitem__("wp_next", True))
                dpg.add_separator()
                dpg.add_text("Execution speed")
                dpg.add_slider_float(
                    label="speed ×", tag="exec_speed",
                    min_value=0.25, max_value=4.0, default_value=1.0,
                    callback=lambda s, a: setattr(self, "_exec_speed_scale", float(a)),
                )
                dpg.add_checkbox(
                    label="Shortest-path wrist3", tag="shortest_wrist3",
                    default_value=True,
                    callback=lambda s, a: setattr(self, "_shortest_wrist3", bool(a)),
                )
                dpg.add_separator()
                dpg.add_text("Log")
                dpg.add_input_text(tag="sim_log", multiline=True, readonly=True,
                                   width=-1, height=120, default_value="")
                dpg.add_separator()
                dpg.add_text("Joint Limit Preset")
                dpg.add_radio_button(
                    tag="preset_radio",
                    items=_preset_names,
                    default_value="default",
                    callback=lambda s, a: self.switch_joint_limit_preset(a),
                )
                dpg.add_separator()
                dpg.add_text("MoveIt Obstacles")
                for _name, _c, _h, _y in _obs_list:
                    _lbl = (f"{_name}  "
                            f"[{_h[0]*2:.2f}×{_h[1]*2:.2f}×{_h[2]*2:.2f} m]")

                    def _make_toggle(name_, c_, h_, y_):
                        def _cb(sender, app_data):
                            if app_data:
                                self.add_moveit_obstacle(name_, c_, h_, y_)
                            else:
                                self.remove_moveit_obstacle(name_)
                        return _cb

                    dpg.add_checkbox(
                        label=_lbl,
                        tag=f"dpg_{_name}",
                        default_value=True,
                        callback=_make_toggle(_name, _c, _h, _y),
                    )

            dpg.create_viewport(title="Robot Controller GUI", width=460, height=980)
            dpg.setup_dearpygui()
            dpg.show_viewport()
            while dpg.is_dearpygui_running() and self.running:
                dpg.render_dearpygui_frame()
            dpg.destroy_context()

        self.sim_ctrl_active = True
        threading.Thread(target=_gui_thread, daemon=True, name="dpg-gui").start()
        print("[linux_controller4] Simulated controller GUI started (dearpygui).")

    def read_simulated_controller_gui(self) -> dict:
        if not HAS_DPG or not self.sim_ctrl_active:
            return {}
        try:
            self.ghost_pos   = np.array([dpg.get_value("sim_x"),
                                          dpg.get_value("sim_y"),
                                          dpg.get_value("sim_z")], dtype=float)
            self.ghost_euler = np.array([dpg.get_value("sim_rx"),
                                          dpg.get_value("sim_ry"),
                                          dpg.get_value("sim_rz")], dtype=float)
            btn = self._sim_btn_state
            a_val      = btn["A"];      btn["A"]      = False
            b_val      = btn["B"];      btn["B"]      = False
            add_wp_val = btn["add_wp"]; btn["add_wp"] = False
            wp_next    = btn["wp_next"]; btn["wp_next"] = False
            wp_prev    = btn["wp_prev"]; btn["wp_prev"] = False
            return {
                "index_trigger":    float(dpg.get_value("sim_idx")),
                "middle_trigger":   float(dpg.get_value("sim_mid")),
                "A":                a_val,
                "B":                b_val,
                "add_wp":           add_wp_val,
                "thumbstick_x":     1.0 if wp_next else (-1.0 if wp_prev else 0.0),
                "thumbstick_y":     0.0,
                "thumbstick_click": False,
            }
        except Exception:
            return {}

    _GUI_LOG_MAX_LINES = 20

    def gui_log(self, msg: str):
        """Append a message to the GUI log panel (no-op if GUI is not running)."""
        if not HAS_DPG or not self.sim_ctrl_active:
            return
        try:
            current = dpg.get_value("sim_log") or ""
            lines = current.splitlines()
            lines.append(msg)
            if len(lines) > self._GUI_LOG_MAX_LINES:
                lines = lines[-self._GUI_LOG_MAX_LINES:]
            dpg.set_value("sim_log", "\n".join(lines))
        except Exception:
            pass

    # ── Config / ZMQ server ───────────────────────────────────────────────────

    def controller_receiver_thread(self):
        while self.running:
            latest_ctrl = None
            try:
                while True:
                    latest_ctrl = self.controller_recv.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                pass
            if latest_ctrl is None:
                time.sleep(0.001)
                continue
            try:
                msg = json.loads(latest_ctrl)
                right_T = self.pose_to_T(msg["right_ctrl"]) if msg.get("right_ctrl") else None
                left_T  = self.pose_to_T(msg["left_ctrl"])  if msg.get("left_ctrl")  else None
                head_T  = self.pose_to_T(msg["head"])        if msg.get("head")        else None
                right_buttons = msg.get("right_buttons", {})
                left_buttons  = msg.get("left_buttons",  {})
            except Exception as exc:
                print(f"[linux_controller4] Bad controller packet: {exc}")
                continue
            with self.ctrl_lock:
                if right_T is not None: self.latest_right_T = right_T
                if left_T  is not None: self.latest_left_T  = left_T
                if head_T  is not None: self.latest_head_T  = head_T
                self.latest_right_buttons = right_buttons
                self.latest_left_buttons  = left_buttons

    def config_server_thread(self):
        if self.config_rep is None:
            return
        cfg_str = json.dumps({
            "type":        "static_config",
            "robot_base":  self.T_to_pose(self.T_world_base),
            "markers":     {str(k): self.T_to_pose(v) for k, v in self.T_world_markers.items()},
            "joint_names": ARM_JOINT_NAMES,
        })
        print("[linux_controller4] Config server ready.")
        while self.running:
            try:
                self.config_rep.recv_string(zmq.NOBLOCK)
                self.config_rep.send_string(cfg_str)
                print("[linux_controller4] Served static config.")
            except zmq.Again:
                time.sleep(0.01)

    # ── Unity scene ───────────────────────────────────────────────────────────

    def make_unity_scene_objects(self) -> list[dict]:
        objects = []
        objects.append(self.T_to_unity_object(
            obj_id=0, name="Cube", T_world_object=np.eye(4, dtype=np.float64),
            scale_unity=[0.1, 0.1, 0.1]))
        T_baseboard = self.T_world_markers.get(
            self.cfg.scene_baseboard_marker, np.eye(4, dtype=np.float64))
        objects.append(self.T_to_unity_object(
            obj_id=1, name="baseboard", T_world_object=T_baseboard,
            mesh_correction=ScipyR.from_euler('x', -90.0, degrees=True),
            scale_unity=[self.cfg.scene_baseboard_scale] * 3))
        T_preassembly = self.T_world_markers.get(
            self.cfg.scene_gearstand_marker,
            self.T_world_markers.get(51, np.eye(4, dtype=np.float64)))
        objects.append(self.T_to_unity_object(
            obj_id=2, name="preassembly_gearstands", T_world_object=T_preassembly,
            mesh_correction=ScipyR.from_euler('x', -90.0, degrees=True),
            scale_unity=[self.cfg.scene_gearstand_scale] * 3))
        self.add_unity_end_gearstands(objects, T_baseboard, start_id=10)
        return objects

    def add_unity_end_gearstands(self, objects, T_world_baseboard, start_id=10):
        if not self.cfg.gear_stands_json.exists():
            return
        with open(self.cfg.gear_stands_json) as f:
            data = json.load(f)
        mesh_correction = ScipyR.from_euler('x', -90.0, degrees=True)
        groups = [
            ("gear_stands", "gearstand",    self.cfg.scene_stands_scale),
            ("stand_ups",   "gearstandcap", self.cfg.scene_stands_scale),
        ]
        obj_id = start_id
        for section, name_prefix, scale in groups:
            block = data.get(section, {}).get("end", {})
            for i, entry in enumerate(block.get("poses", [])[:4], start=1):
                T_world_object = T_world_baseboard @ self.gear_stand_entry_to_T(entry, scale)
                objects.append(self.T_to_unity_object(
                    obj_id=obj_id, name=f"{name_prefix}{i}",
                    T_world_object=T_world_object,
                    mesh_correction=mesh_correction,
                    scale_unity=[scale] * 3))
                obj_id += 1

    @staticmethod
    def gear_stand_entry_to_T(entry: dict, scale: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('z', float(entry.get("yaw_deg", 0.0)),
                                       degrees=True).as_matrix()
        T[:3, 3]  = np.array([float(entry["x"]), float(entry["y"]), float(entry["z"])],
                              dtype=np.float64) * float(scale)
        return T

    def publish_unity_scene_objects(self):
        if self.quest_bridge is None:
            return
        objects = self.make_unity_scene_objects()
        if not self._unity_scene_logged:
            for obj in objects:
                print(f"[unity_scene] {obj['name']}: pos={np.round(obj['position'], 4).tolist()}")
            self._unity_scene_logged = True
        self.quest_bridge.send_synthetic_objects(objects)

    # ── ROS2 ──────────────────────────────────────────────────────────────────

    def setup_moveit_client(self, node):
        if not HAS_PYMOVEIT2:
            print("[linux_controller4] pymoveit2 not found — MoveIt disabled.")
            self.moveit2 = None
            return
        try:
            self._moveit_cb_group = ReentrantCallbackGroup()
            self.moveit2 = MoveIt2(
                node              = node,
                joint_names       = ARM_JOINT_NAMES,
                base_link_name    = "base_link",
                end_effector_name = "tool0",
                group_name        = "ur_manipulator",
                callback_group    = self._moveit_cb_group,
            )
            self.moveit2.max_velocity     = float(cfg.moveit_vel_scale)
            self.moveit2.max_acceleration = float(cfg.moveit_accel_scale)
            print(f"[linux_controller4] MoveIt2 client ready  "
                  f"vel_scale={cfg.moveit_vel_scale}  accel_scale={cfg.moveit_accel_scale}.")
            # Push all obstacles into the planning scene ~1 s after startup
            # (gives MoveIt's /collision_object subscriber time to come up)
            self._moveit_sync_timer = node.create_timer(1.0, self._moveit_initial_sync)
        except Exception as exc:
            print(f"[linux_controller4] MoveIt2 setup ERROR: {exc}")
            self.moveit2 = None

    def _moveit_initial_sync(self):
        """One-shot timer callback: push obstacles + joint constraints into MoveIt."""
        self._moveit_sync_timer.cancel()
        self._sync_all_moveit_obstacles()
        self._apply_moveit_joint_constraints()

    def _sync_all_moveit_obstacles(self):
        for i, obs in enumerate(self.cfg.obstacle_boxes):
            center, half, yaw_deg = self.unpack_obstacle_box(obs)
            self.add_moveit_obstacle(f"obs_{i}", center, half, yaw_deg)

    def add_moveit_obstacle(self, name: str, center, half_extents, yaw_deg: float = 0.0):
        """Add (or replace) a box collision object in the MoveIt2 planning scene."""
        if self.moveit2 is None:
            return
        q = ScipyR.from_euler('z', float(yaw_deg), degrees=True).as_quat()
        self.moveit2.add_collision_box(
            id=name,
            size=(float(half_extents[0] * 2), float(half_extents[1] * 2),
                  float(half_extents[2] * 2)),
            position=(float(center[0]), float(center[1]), float(center[2])),
            quat_xyzw=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            frame_id="world",
        )
        print(f"[moveit] added collision box '{name}'  "
              f"center={np.round(np.asarray(center, float), 3).tolist()}", flush=True)

    def remove_moveit_obstacle(self, name: str):
        """Remove a collision object from the MoveIt2 planning scene."""
        if self.moveit2 is None:
            return
        self.moveit2.remove_collision_object(name)
        print(f"[moveit] removed collision object '{name}'", flush=True)

    def _apply_moveit_joint_constraints(self):
        """Push cfg.joint_min_deg / joint_max_deg (with margin) as MoveIt path constraints."""
        if self.moveit2 is None:
            return
        self.moveit2.clear_path_constraints()
        margin = float(self.cfg.joint_limit_margin_deg)
        min_rad = np.deg2rad(self.cfg.joint_min_deg + margin)
        max_rad = np.deg2rad(self.cfg.joint_max_deg - margin)
        for jname, lo, hi in zip(ARM_JOINT_NAMES, min_rad, max_rad):
            self.moveit2.set_path_joint_constraint(
                joint_positions=[(float(lo) + float(hi)) / 2.0],
                joint_names=[jname],
                tolerance=(float(hi) - float(lo)) / 2.0,
                weight=1.0,
            )

    def switch_joint_limit_preset(self, name: str):
        """Switch the active joint-limit preset, rebuild CBF, update MoveIt constraints."""
        presets = self.cfg.joint_limit_presets
        if name not in presets:
            print(f"[preset] Unknown preset '{name}'", flush=True)
            return
        preset = presets[name]
        min_deg, max_deg = preset[0], preset[1]
        self.cfg.joint_min_deg = np.asarray(min_deg, float)
        self.cfg.joint_max_deg = np.asarray(max_deg, float)
        self.active_preset = name
        self._build_cbf()
        self._apply_moveit_joint_constraints()
        print(f"[preset] → '{name}'  "
              f"min={np.round(self.cfg.joint_min_deg, 1).tolist()}", flush=True)
        self.gui_log(f"[preset] → {name}")

    def _plan_common(self, traj, label: str):
        if traj is None:
            print(f"[moveit] {label} FAILED — no trajectory returned.", flush=True)
            self.gui_log(f"[moveit] {label} FAILED")
            return [], []
        pts = traj.points
        waypoints = [np.array(pt.positions) for pt in pts]
        def _t(pt):
            return pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        seg_durations = [
            max(_t(pts[i + 1]) - _t(pts[i]), 1e-3)
            for i in range(len(pts) - 1)
        ] if len(pts) > 1 else []
        duration = _t(pts[-1]) if pts else 0.0
        path_len = sum(
            np.linalg.norm(np.array(pts[i + 1].positions) - np.array(pts[i].positions))
            for i in range(len(pts) - 1)
        ) if len(pts) > 1 else 0.0
        max_vel = max(
            (max(abs(v) for v in pt.velocities) for pt in pts if pt.velocities),
            default=float("nan"),
        )
        print(f"[moveit] {label} SUCCESS — {len(waypoints)} waypoints  "
              f"duration={duration:.2f}s  path={path_len:.3f}rad  "
              f"max-vel={max_vel:.3f}rad/s", flush=True)
        self.gui_log(f"[moveit] OK  {len(waypoints)}pts  {duration:.1f}s")
        return waypoints, seg_durations

    @staticmethod
    def _normalize_joints_to_bounds(q_rad, min_rad, max_rad):
        """Shift each joint by multiples of 2π to land inside [min, max]."""
        q_out = q_rad.copy()
        for i in range(len(q_out)):
            while q_out[i] < min_rad[i] - 1e-6:
                q_out[i] += 2 * np.pi
            while q_out[i] > max_rad[i] + 1e-6:
                q_out[i] -= 2 * np.pi
        return q_out

    def request_moveit_plan(self, target_pos, target_quat_xyzw):
        if self.moveit2 is None:
            return [], []
        min_rad = np.deg2rad(self.cfg.joint_min_deg)
        max_rad = np.deg2rad(self.cfg.joint_max_deg)
        q_normalized = self._normalize_joints_to_bounds(self.q_current, min_rad, max_rad)
        print(f"[moveit] raw q_current  (deg): {np.round(np.rad2deg(self.q_current), 1).tolist()}", flush=True)
        print(f"[moveit] normalized start(deg): {np.round(np.rad2deg(q_normalized), 1).tolist()}", flush=True)
        print(f"[moveit] preset min      (deg): {np.round(self.cfg.joint_min_deg, 1).tolist()}", flush=True)
        print(f"[moveit] preset max      (deg): {np.round(self.cfg.joint_max_deg, 1).tolist()}", flush=True)
        start_js = JointState()
        start_js.name     = ARM_JOINT_NAMES
        start_js.position = q_normalized.tolist()
        try:
            print("[moveit] planning (joint-space)…", flush=True)
            traj = self.moveit2.plan(
                position              = target_pos.tolist(),
                quat_xyzw             = target_quat_xyzw.tolist(),
                frame_id              = "world",
                start_joint_state     = start_js,
                tolerance_position    = 0.0005,
                tolerance_orientation = 0.005,
            )
        except Exception as exc:
            print(f"[moveit] EXCEPTION in plan(): {exc}", flush=True)
            import traceback; traceback.print_exc()
            self.gui_log(f"[moveit] EXCEPTION: {exc}")
            return [], []
        return self._plan_common(traj, "joint-space")

    _STL_DIR       = Path(__file__).resolve().parent / "initialization" / "stl_files"

    _QUEST_COLORED_STL = str(
        Path(__file__).resolve().parent
        / "initialization" / "stl_files"
        / "meta_quest3_right_controller_colored.obj"
    )
    _QUEST_CASING_STL = str(
        Path(__file__).resolve().parent
        / "initialization" / "stl_files"
        / "meta_quest3_right_controller_casing_registered.stl"
    )



    def setup_ros_publishers(self, node):
        latching = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.js_pub        = node.create_publisher(JointState,        "/joint_states",           10)
        self.obj_pub       = node.create_publisher(MarkerArray,       "/scene_objects",        latching)
        self.ghost_pub     = None  # kept for API compat; ghost gripper now uses TF + ghost_joint_states
        self.preview_pub   = node.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self.ghost_js_pub  = node.create_publisher(JointState,        "/ghost_joint_states",   10)
        self.obstacles_pub    = node.create_publisher(MarkerArray, "/obstacles",     latching)
        self.workspace_pub    = node.create_publisher(MarkerArray, "/workspace_box", latching)
        self.robot_tcp_pub    = node.create_publisher(MarkerArray, "/robot_tcp_axes",    10)
        self.proxy_tcp_pub    = node.create_publisher(MarkerArray, "/proxy_tcp_axes",    10)
        self.waypoint_pub     = node.create_publisher(MarkerArray, "/waypoint_markers",  10)
        self.tf_bc         = TransformBroadcaster(node)
        stamp = node.get_clock().now().to_msg()
        ma = MarkerArray()
        for m in self.make_scene_markers(stamp):
            ma.markers.append(m)
        self.obj_pub.publish(ma)
        self._publish_debug_static(stamp)
        node.create_timer(1.0, self._republish_scene_markers)
        # Spawn quest assembly robot_state_publisher (casing + adapters + Robotiq 85)
        # ghost RSP is launched by moveit.launch.py (quest_assembly_rsp in /ghost namespace).
        # linux_controller4 only needs to broadcast world→quest_ctrl_base TF
        # and publish /ghost_joint_states.

    def _republish_scene_markers(self):
        stamp = self.ros.get_clock().now().to_msg()
        ma = MarkerArray()
        for m in self.make_scene_markers(stamp):
            ma.markers.append(m)
        self.obj_pub.publish(ma)
        self._publish_debug_static(stamp)

    # ── Static debug markers (obstacles + workspace box) ──────────────────────

    def _publish_debug_static(self, stamp):
        if not hasattr(self, 'obstacles_pub'):
            return

        # Obstacle boxes — semi-transparent red cubes → /obstacles
        obs_ma = MarkerArray()
        for mid, obs in enumerate(self.cfg.obstacle_boxes):
            center, half, yaw_deg = self.unpack_obstacle_box(obs)
            m = Marker()
            m.header.frame_id = "world"; m.header.stamp = stamp
            m.ns = "obstacles"; m.id = mid
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, center)
            q = ScipyR.from_euler('z', yaw_deg, degrees=True).as_quat()
            m.pose.orientation.x, m.pose.orientation.y, \
                m.pose.orientation.z, m.pose.orientation.w = map(float, q)
            m.scale.x, m.scale.y, m.scale.z = float(half[0]*2), float(half[1]*2), float(half[2]*2)
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.2, 0.35
            obs_ma.markers.append(m)
        self.obstacles_pub.publish(obs_ma)

        # Workspace bounding box — cyan wireframe → /workspace_box
        ws = self.cfg.workspace_box
        lo, hi = ws[:3], ws[3:]
        x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
        x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
        # 8 corners
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        # 12 edges as index pairs
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        from geometry_msgs.msg import Point as GPoint
        m = Marker()
        m.header.frame_id = "world"; m.header.stamp = stamp
        m.ns = "workspace"; m.id = 0
        m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.004  # line width (m)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 1.0, 1.0
        for a, b in edges:
            for cx, cy, cz in (corners[a], corners[b]):
                p = GPoint(); p.x, p.y, p.z = cx, cy, cz
                m.points.append(p)
        ws_ma = MarkerArray(); ws_ma.markers.append(m)
        self.workspace_pub.publish(ws_ma)

    # ── Dynamic TCP axes (robot EE + proxy grip camera_adapter) ───────────────

    @staticmethod
    def _make_axis_arrow(ns, marker_id, stamp, pos, rot_col, color_rgba, length=0.08):
        """One arrow along rot_col direction from pos."""
        from geometry_msgs.msg import Point as GPoint
        m = Marker()
        m.header.frame_id = "world"; m.header.stamp = stamp
        m.ns = ns; m.id = marker_id; m.type = Marker.ARROW; m.action = Marker.ADD
        p0 = GPoint(); p0.x, p0.y, p0.z = float(pos[0]), float(pos[1]), float(pos[2])
        end = pos + rot_col * length
        p1 = GPoint(); p1.x, p1.y, p1.z = float(end[0]), float(end[1]), float(end[2])
        m.points = [p0, p1]
        m.scale.x = 0.005   # shaft diameter
        m.scale.y = 0.010   # head diameter
        r, g, b, a = color_rgba
        m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), float(a)
        return m

    def publish_tcp_axes(self, stamp):
        if not hasattr(self, 'robot_tcp_pub'):
            return
        length = 0.08  # 8 cm arrows

        # ── Robot TCP → /robot_tcp_axes ───────────────────────────────────────
        q_jnp = jnp.array(self.q_current)
        ee_pos, ee_rot = self._ee_world(q_jnp)
        robot_ma = MarkerArray()
        for col, rgba in enumerate([(1,0,0,1), (0,1,0,1), (0,0,1,1)]):  # X=red Y=green Z=blue
            robot_ma.markers.append(self._make_axis_arrow(
                "robot_tcp", col, stamp, ee_pos, ee_rot[:, col], rgba, length))
        self.robot_tcp_pub.publish(robot_ma)

        # ── Proxy grip TCP (camera_adapter = 19 mm along casing Y) → /proxy_tcp_axes
        T_world_ghost = self._T_world_ghost
        if T_world_ghost is not None and self._T_casing_to_ctrl is not None:
            T_y019 = np.eye(4); T_y019[1, 3] = 0.019
            T_world_cam = T_world_ghost @ self._T_casing_to_ctrl @ T_y019
            cam_pos = T_world_cam[:3, 3]
            cam_rot = T_world_cam[:3, :3]
            proxy_ma = MarkerArray()
            for col, rgba in enumerate([(1,0,0,1), (0,1,0,1), (0,0,1,1)]):
                proxy_ma.markers.append(self._make_axis_arrow(
                    "proxy_tcp", col, stamp, cam_pos, cam_rot[:, col], rgba, length))
            self.proxy_tcp_pub.publish(proxy_ma)

    def _do_add_waypoint(self):
        """Add or update a waypoint at the current camera_adapter TCP.

        If a waypoint is currently selected, moves it to the new pose.
        Otherwise appends a new waypoint.
        """
        _wp_pos, _wp_R = self._camera_adapter_target()
        if not (np.isfinite(_wp_R).all() and np.isfinite(_wp_pos).all()
                and abs(np.linalg.det(_wp_R) - 1.0) < 0.01):
            print("[waypoint] Skipped — invalid camera_adapter pose.", flush=True)
            return
        _T_wp = np.eye(4)
        _T_wp[:3, :3] = _wp_R
        _T_wp[:3, 3]  = _wp_pos
        sel = self.wp_manager.selected
        if sel is not None:
            self.wp_manager.update(sel, _T_wp)
            print(f"[waypoint] Moved #{sel} to {_wp_pos.round(3)}", flush=True)
        else:
            wp = self.wp_manager.add(_T_wp)
            print(f"[waypoint] Added #{wp.idx} at {_wp_pos.round(3)}", flush=True)
        self._send_waypoints_to_unity()
        if self.ros is not None:
            self.publish_waypoints(self.ros.get_clock().now().to_msg())

    def publish_waypoints(self, stamp):
        """Publish waypoint markers to RViz and the full waypoint list to Unity."""
        if not hasattr(self, 'waypoint_pub'):
            return
        from geometry_msgs.msg import Point as GPoint
        ma = MarkerArray()
        wps = self.wp_manager.get_all()

        # ── Delete stale markers first ────────────────────────────────────────
        del_m = Marker(); del_m.action = Marker.DELETEALL
        del_m.ns = "waypoints"; del_m.header.frame_id = "world"; del_m.header.stamp = stamp
        ma.markers.append(del_m)

        _gripper_stl = self._STL_DIR / "non-moving-gripper.stl"
        _gripper_stl_uri = f"file://{_gripper_stl}" if _gripper_stl.exists() else None

        # ── Ghost gripper mesh + sphere + axes per waypoint ───────────────────
        for wp in wps:
            pos = wp.T_world[:3, 3]
            rot = wp.T_world[:3, :3]
            is_sel = (wp.idx == self.wp_manager.selected)
            quat = ScipyR.from_matrix(rot).as_quat()  # xyzw

            # Ghost gripper mesh
            if _gripper_stl_uri is not None:
                gm = Marker()
                gm.header.frame_id = "world"; gm.header.stamp = stamp
                gm.ns = "waypoints"; gm.id = wp.idx * 10 + 6
                gm.type = Marker.MESH_RESOURCE; gm.action = Marker.ADD
                gm.mesh_resource = _gripper_stl_uri
                gm.pose.position.x, gm.pose.position.y, gm.pose.position.z = \
                    float(pos[0]), float(pos[1]), float(pos[2])
                gm.pose.orientation.x, gm.pose.orientation.y, \
                    gm.pose.orientation.z, gm.pose.orientation.w = \
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                gm.scale.x = gm.scale.y = gm.scale.z = 1.0  # STL in metres
                if is_sel:
                    gm.color.r, gm.color.g, gm.color.b, gm.color.a = 1.0, 1.0, 0.0, 0.85
                else:
                    gm.color.r, gm.color.g, gm.color.b, gm.color.a = 0.7, 0.7, 0.7, 0.5
                ma.markers.append(gm)

            # Sphere node
            sp = Marker()
            sp.header.frame_id = "world"; sp.header.stamp = stamp
            sp.ns = "waypoints"; sp.id = wp.idx * 10
            sp.type = Marker.SPHERE; sp.action = Marker.ADD
            sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = \
                float(pos[0]), float(pos[1]), float(pos[2])
            sp.pose.orientation.w = 1.0
            sp.scale.x = sp.scale.y = sp.scale.z = 0.02
            sp.color.r, sp.color.g, sp.color.b, sp.color.a = \
                (1.0, 1.0, 0.0, 1.0) if is_sel else (0.0, 0.8, 1.0, 1.0)
            ma.markers.append(sp)
            # Text label
            tx = Marker()
            tx.header.frame_id = "world"; tx.header.stamp = stamp
            tx.ns = "waypoints"; tx.id = wp.idx * 10 + 1
            tx.type = Marker.TEXT_VIEW_FACING; tx.action = Marker.ADD
            tx.text = str(wp.idx)
            tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = \
                float(pos[0]), float(pos[1]) + 0.04, float(pos[2])
            tx.pose.orientation.w = 1.0
            tx.scale.z = 0.025
            tx.color.r = tx.color.g = tx.color.b = tx.color.a = 1.0
            ma.markers.append(tx)
            # Orientation axes (X=red Y=green Z=blue)
            for col, rgba in enumerate([(1,0,0,1),(0,1,0,1),(0,0,1,1)]):
                ma.markers.append(self._make_axis_arrow(
                    "waypoints", wp.idx * 10 + 2 + col, stamp,
                    pos, rot[:, col], rgba, length=0.05))

        # ── Edge line strip ───────────────────────────────────────────────────
        if len(wps) >= 2:
            ln = Marker()
            ln.header.frame_id = "world"; ln.header.stamp = stamp
            ln.ns = "waypoints"; ln.id = 9000
            ln.type = Marker.LINE_STRIP; ln.action = Marker.ADD
            ln.scale.x = 0.004
            ln.color.r, ln.color.g, ln.color.b, ln.color.a = 1.0, 0.6, 0.0, 1.0
            for wp in wps:
                p = GPoint()
                p.x, p.y, p.z = float(wp.T_world[0,3]), float(wp.T_world[1,3]), float(wp.T_world[2,3])
                ln.points.append(p)
            ma.markers.append(ln)

        self.waypoint_pub.publish(ma)

        # ── Send to Unity ─────────────────────────────────────────────────────
        self._send_waypoints_to_unity()

    _WP_CORRECTION = (
        ScipyR.from_euler('x', -90.0, degrees=True) *
        ScipyR.from_euler('z', 180.0, degrees=True)
    ).as_matrix()

    def _send_waypoints_to_unity(self):
        """Send the full waypoint list + edges to Unity via port 5006."""
        wps = self.wp_manager.get_all()
        unity_wps = []
        for wp in wps:
            _T = wp.T_world.copy()
            _T[:3, :3] = _T[:3, :3] @ self._WP_CORRECTION
            _ghost_obj = self.T_to_unity_object(
                obj_id=wp.idx, name=f"wp_{wp.idx}", T_world_object=_T)
            unity_wps.append({
                "id":           _ghost_obj["id"],
                "position":     _ghost_obj["position"],
                "rotation_xyzw": _ghost_obj["rotation_xyzw"],
            })
        edges = [[a, b] for a, b in self.wp_manager.get_edges()]
        selected = self.wp_manager.selected
        msg = json.dumps({
            "type":      "waypoints",
            "waypoints": unity_wps,
            "edges":     edges,
            "selected":  selected if selected is not None else -1,
        }, separators=(',', ':'))  # compact — C# Contains("\"type\":\"waypoints\"") requires no spaces
        try:
            self.unity_ghost_pub.send_string(msg, zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception:
            pass

    def publish_preview_trajectory(self, waypoints: list,
                                   durations: list | None = None,
                                   dt: float = 1.0,
                                   target_frames: int = 200,
                                   preview_duration: float = 0.1):
        if not waypoints or not hasattr(self, 'preview_pub'):
            return
        
        times = [0.0]
        for i in range(len(waypoints) - 1):
            times.append(times[-1] + (durations[i] if durations and i < len(durations) else dt))
        total_t = times[-1]
        qs = [np.asarray(q, dtype=float) for q in waypoints]
        n_frames = max(2, target_frames)
        sample_times = np.linspace(0.0, total_t, n_frames)
        resampled = []
        for st in sample_times:
            idx = int(np.clip(np.searchsorted(times, st, side='right') - 1, 0, len(qs) - 2))
            seg_dt = times[idx + 1] - times[idx]
            alpha  = (st - times[idx]) / seg_dt if seg_dt > 1e-9 else 0.0
            resampled.append((1.0 - alpha) * qs[idx] + alpha * qs[idx + 1])
        traj = JointTrajectory(); traj.joint_names = ARM_JOINT_NAMES
        scale = float(preview_duration) / max(total_t, 1e-3)
        for q, st in zip(resampled, sample_times):
            pt = JointTrajectoryPoint()
            pt.positions = q.tolist()
            pt.time_from_start = RclpyDuration(seconds=float(st) * scale).to_msg()
            traj.points.append(pt)
        rt = RobotTrajectory(); rt.joint_trajectory = traj
        msg = DisplayTrajectory(); msg.model_id = "ur10e"
        msg.trajectory.append(rt)
        msg.trajectory_start.joint_state.name     = ARM_JOINT_NAMES
        msg.trajectory_start.joint_state.position = resampled[0].tolist()
        self.preview_pub.publish(msg)

    def publish_ros(self):
        node = self.ros
        now  = node.get_clock().now().to_msg()
        js   = JointState(); js.header.stamp = now
        drive_q = float(np.clip(self.gripper_q, 0.0, 0.8))
        gripper_positions = [drive_q * r for r in ROS_GRIPPER_MIMIC]
        js.name     = ARM_JOINT_NAMES + ROS_GRIPPER_JOINT_NAMES
        js.position = self.q_current.tolist() + gripper_positions
        self.js_pub.publish(js)
        if hasattr(self, 'ghost_pub'):
            self.publish_ghost_gripper(now)
        self.publish_tcp_axes(now)

    _T_R_Z180 = np.block([[ScipyR.from_euler('z', np.pi).as_matrix(), np.zeros((3,1))],
                           [np.zeros((1,3)),                            np.ones((1,1))]])

    def publish_ghost_gripper(self, stamp):
        if self._T_casing_to_ctrl is None:
            return

        T_world_ctrl = None
        if not getattr(self, '_simulate_controller', False):
            if self.quest_bridge is not None:
                with self.quest_bridge.ctrl_lock:
                    T_world_ctrl = self.quest_bridge.latest_right_T
            else:
                with self.ctrl_lock:
                    T_world_ctrl = self.latest_right_T

        if T_world_ctrl is None or not np.isfinite(T_world_ctrl).all():
            # Fallback: reconstruct approximate controller pose from ghost_pos / ghost_euler.
            # This keeps the assembly visible in simulate_controller mode and during
            # Quest start-up before the first tracking frame arrives.
            gp = getattr(self, 'ghost_pos',   None)
            ge = getattr(self, 'ghost_euler', None)
            if gp is None or ge is None:
                self._ghost_no_tf_ctr += 1
                if self._ghost_no_tf_ctr % 200 == 1:
                    print("[ghost] No controller data and no ghost_pos — TF not broadcast.",
                          flush=True)
                return
            ik_euler = self.ghost_visual_euler_to_ik_euler(np.asarray(ge))
            R_ctrl   = ScipyR.from_euler('xyz', ik_euler, degrees=True).as_matrix()
            # ghost_pos = t_ctrl + R_ctrl @ [0, 0.1, 0]; undo the 100 mm Y offset
            t_ctrl   = np.asarray(gp) - R_ctrl @ np.array([0.0, 0.1, 0.0])
            T_world_ctrl = np.eye(4, dtype=np.float64)
            T_world_ctrl[:3, :3] = R_ctrl
            T_world_ctrl[:3,  3] = t_ctrl
            if self._ghost_no_tf_ctr % 200 == 1:
                print("[ghost] Using ghost_pos fallback for TF broadcast "
                      f"(no live controller data). count={self._ghost_no_tf_ctr}", flush=True)
        else:
            if self._ghost_no_tf_ctr > 0:
                print(f"[ghost] Controller data restored after {self._ghost_no_tf_ctr} frames.",
                      flush=True)
            self._ghost_no_tf_ctr = 0

        # quest_ctrl_base = controller tracking frame rotated 180° around local Z
        # (aligns the mesh coordinate frame used in the registration with the Quest tracking frame)
        T_world_ghost = T_world_ctrl @ self._T_R_Z180
        if not np.isfinite(T_world_ghost[:3, :3]).all():
            return
        self._T_world_ghost = T_world_ghost  # cached for publish_tcp_axes

        ts = TransformStamped()
        ts.header.stamp    = stamp
        ts.header.frame_id = "world"
        ts.child_frame_id  = "quest_ctrl_base"
        pos  = T_world_ghost[:3, 3]
        quat = ScipyR.from_matrix(T_world_ghost[:3, :3]).as_quat()
        ts.transform.translation.x = float(pos[0])
        ts.transform.translation.y = float(pos[1])
        ts.transform.translation.z = float(pos[2])
        ts.transform.rotation.x = float(quat[0])
        ts.transform.rotation.y = float(quat[1])
        ts.transform.rotation.z = float(quat[2])
        ts.transform.rotation.w = float(quat[3])
        self.tf_bc.sendTransform(ts)

        # Robotiq finger joint states (mimic joints computed manually)
        q = float(np.clip(self.gripper_q, 0.0, 0.8))
        gjs = JointState()
        gjs.header.stamp = stamp
        gjs.name = [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint",
        ]
        gjs.position = [q, -q, q, -q, -q, q]
        self.ghost_js_pub.publish(gjs)

    def make_scene_markers(self, stamp) -> list:
        out, mid = [], 0
        def _mesh_marker(stl_path, T, scale, rgba):
            nonlocal mid
            m = Marker()
            m.header.frame_id = "world"; m.header.stamp = stamp
            m.ns = "scene_objects"; m.id = mid; mid += 1
            m.type = Marker.MESH_RESOURCE; m.action = Marker.ADD
            m.mesh_resource = f"file://{stl_path}"
            pos = T[:3, 3]; q = ScipyR.from_matrix(T[:3, :3]).as_quat()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
            m.pose.orientation.x, m.pose.orientation.y, \
                m.pose.orientation.z, m.pose.orientation.w = map(float, q)
            m.scale.x = m.scale.y = m.scale.z = float(scale)
            m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
            out.append(m)
        baseboard = self.cfg.stl_dir / "baseboard.stl"
        if baseboard.exists():
            _mesh_marker(baseboard, np.eye(4), 0.001, (0.05, 0.05, 0.05, 1.0))
        for stl_name, marker_id, rgba in [
            ("baseboard.stl",       11,  (0.7, 0.7, 0.75, 1.0)),
            ("PreAssembly_Aux.stl", 112, (0.7, 0.7, 0.75, 1.0)),
        ]:
            stl_path = self.cfg.stl_dir / stl_name
            T = self.T_world_markers.get(marker_id)
            if T is not None and stl_path.exists():
                _mesh_marker(stl_path, T, 0.001, rgba)
        stands_stl = self.cfg.stl_dir / "preassembly_stands.stl"
        T_stands = self.T_world_markers.get(self.cfg.scene_gearstand_marker)
        if T_stands is not None and stands_stl.exists():
            _mesh_marker(stands_stl, T_stands, 0.001, (0.7, 0.7, 0.75, 1.0))
        if self.cfg.gear_stands_json.exists():
            with open(self.cfg.gear_stands_json) as f:
                gs_data = json.load(f)
            scale = self.cfg.scene_stands_scale
            for stl_file, section, rgba in [
                ("stand_bottom.stl", "gear_stands", (0.6, 0.6, 0.65, 1.0)),
                ("stand_up.stl",     "stand_ups",   (0.55, 0.55, 0.60, 1.0)),
            ]:
                stl_path = self.cfg.stl_dir / stl_file
                if not stl_path.exists():
                    continue
                block = gs_data.get(section, {}).get(self.cfg.scene_assembly_state, {})
                ref_name = block.get("relative_to", "baseboard.stl")
                if ref_name == "baseboard.stl":
                    ref_T = np.eye(4)
                else:
                    mid_ref = self.cfg.stl_to_marker.get(ref_name) if hasattr(self.cfg, 'stl_to_marker') else None
                    ref_T = self.T_world_markers.get(mid_ref, np.eye(4)) if mid_ref else np.eye(4)
                for entry in block.get("poses", []):
                    T_local = self.gear_stand_entry_to_T(entry, scale)
                    _mesh_marker(stl_path, ref_T @ T_local, 0.001, rgba)
        return out

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _ee_world(self, q_jnp) -> tuple[np.ndarray, np.ndarray]:
        _, _, ee_tmat = self.robot.dynamically_consistent_velocity_control_matrices(q_jnp)
        ee_pos = self._base_R_full @ np.array(ee_tmat[:3, 3]) + self._base_pos
        ee_rot = self._base_R_full @ np.array(ee_tmat[:3, :3])
        return ee_pos, ee_rot

    def _camera_adapter_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_world, R_world) for the camera_adapter TCP.

        Uses the registered ICP chain (T_world_ghost → T_casing_to_ctrl → T_y019)
        when controller data is available — same transform the proxy TCP axes display.
        Falls back to ghost_pos / ghost_euler when _T_world_ghost is not yet set.
        """
        T_ghost = self._T_world_ghost
        if T_ghost is not None and self._T_casing_to_ctrl is not None:
            T_y019 = np.eye(4); T_y019[1, 3] = 0.019
            T_cam = T_ghost @ self._T_casing_to_ctrl @ T_y019
            return T_cam[:3, 3].copy(), T_cam[:3, :3].copy()
        # Fallback (first tick or no registration)
        ik_euler = self.ghost_visual_euler_to_ik_euler(self.ghost_euler)
        R = ScipyR.from_euler('xyz', ik_euler, degrees=True).as_matrix()
        return self.ghost_pos.copy(), R

    @staticmethod
    def ghost_visual_euler_to_ik_euler(ghost_euler: np.ndarray) -> np.ndarray:
        ik_euler = np.asarray(ghost_euler, dtype=np.float64).copy()
        ik_euler[0] = ((ik_euler[0] + 90.0 + 180.0) % 360.0) - 180.0
        return ik_euler

    def run(self, simulate_controller: bool = False):
        cfg = self.cfg
        self._simulate_controller = simulate_controller
        if self.quest_bridge is None:
            threading.Thread(target=self.config_server_thread, daemon=True).start()
            if not simulate_controller:
                threading.Thread(target=self.controller_receiver_thread, daemon=True).start()

        # Move real robot to home pose before anything else
        if self.rtde_c is not None:
            home_q_rad = np.deg2rad(cfg.home_q_deg).tolist()
            print(f"[linux_controller4] Moving to home pose: {np.round(cfg.home_q_deg, 1).tolist()}")
            self.rtde_c.moveJ(home_q_rad, cfg.movej_speed, cfg.movej_accel)
            print("[linux_controller4] Home pose reached.")

        # Snap ghost to current EE at startup
        if self.rtde is not None:
            try:
                self.q_current = np.array(self.rtde.getActualQ())
            except Exception:
                pass
        q_jnp = jnp.array(self.q_current)
        ee_pos, ee_rot = self._ee_world(q_jnp)
        R_ik   = ScipyR.from_matrix(ee_rot) * GRIPPER_MOUNT_OFFSET.inv()
        ik_euler = R_ik.as_euler('xyz', degrees=True)
        ghost_euler_init    = ik_euler.copy()
        ghost_euler_init[0] = ((ik_euler[0] - 90.0 + 180.0) % 360.0) - 180.0
        self.ghost_pos   = ee_pos.copy()
        self.ghost_euler = ghost_euler_init.copy()
        print(f"[linux_controller4] Ghost initialised at EE pos={ee_pos.round(3)}")

        if simulate_controller:
            self.setup_simulated_controller_gui(ee_pos=ee_pos, ee_euler_deg=ghost_euler_init)

        IK_DT  = 0.005 if self.rtde_c is None else 0.016
        loop_hz = 1.0 / IK_DT
        UNITY_SCENE_DT = 0.5
        print(f"[linux_controller4] Main loop — frax OSCBF@{loop_hz:.0f}Hz…")

        prev_grip_btn      = False
        prev_send_btn      = False
        prev_cancel_btn    = False
        prev_index_trigger = False
        prev_thumbstick_x  = 0.0
        prev_thumbstick_cl = False
        ctrl_interp_gen = None
        was_tracking    = False
        ctrl_state      = "idle"
        last_unity_scene_t = 0.0
        last_unity_base_t  = 0.0
        _moveit_waypoints = None
        _moveit_durations = None
        _moveit_start_q   = None

        try:
            while True:
                t0 = time.perf_counter()
                if self.rtde_c is not None:
                    t_start = self.rtde_c.initPeriod()

                # 1. Read controller frame
                if simulate_controller:
                    right_buttons = self.read_simulated_controller_gui()
                    if not right_buttons:
                        right_buttons = {"index_trigger": 0.0, "middle_trigger": 0.0,
                                         "A": False, "B": False, "add_wp": False,
                                         "thumbstick_x": 0.0, "thumbstick_y": 0.0,
                                         "thumbstick_click": False}
                elif self.quest_bridge is not None:
                    with self.quest_bridge.ctrl_lock:
                        last_right_T  = self.quest_bridge.latest_right_T
                        right_buttons = dict(self.quest_bridge.latest_right_buttons)
                else:
                    with self.ctrl_lock:
                        last_right_T  = self.latest_right_T
                        right_buttons = dict(self.latest_right_buttons)

                right_grip_btn    = bool(right_buttons.get("A",               False))
                right_send_btn    = float(right_buttons.get("middle_trigger", 0.0)) > 0.5
                right_index_btn   = float(right_buttons.get("index_trigger",  0.0)) > 0.5
                right_cancel_btn  = bool(right_buttons.get("B",              False))
                grip_val          = float(right_buttons.get("middle_trigger", 0.0))
                right_add_wp_btn  = bool(right_buttons.get("add_wp",         False))
                thumbstick_x      = float(right_buttons.get("thumbstick_x",  0.0))
                thumbstick_click  = bool(right_buttons.get("thumbstick_click", False))

                # 2. Update ghost from physical controller (not simulated)
                if not simulate_controller:
                    last_right_T = locals().get("last_right_T")
                    if last_right_T is not None:
                        R_ctrl = last_right_T[:3, :3]
                        t_ctrl = last_right_T[:3, 3]
                        R_off  = ScipyR.from_euler('x', -90, degrees=True).as_matrix()
                        self.ghost_pos   = t_ctrl + R_ctrl @ np.array([0.0, 1.0, 0.0]) * 0.10
                        self.ghost_euler = ScipyR.from_matrix(R_ctrl @ R_off).as_euler(
                            'xyz', degrees=True)

                # 3. Read robot joint state
                if self.rtde is not None:
                    try:
                        self.q_current = np.array(self.rtde.getActualQ())
                    except Exception:
                        print("[linux_controller4] RTDE lost — holding last q.")

                # 4. Grip button — plan / execute
                if right_grip_btn and not prev_grip_btn:
                    print(f"[grip_btn] triggered  moveit2={'ready' if self.moveit2 is not None else 'NONE'}",
                          flush=True)
                    sel = self.wp_manager.selected
                    sel_wp = next((w for w in self.wp_manager.get_all() if w.idx == sel), None) \
                             if sel is not None else None
                    if sel_wp is not None:
                        target_pos = sel_wp.T_world[:3, 3]
                        target_R   = sel_wp.T_world[:3, :3]
                        print(f"[grip_btn] Planning to selected waypoint #{sel} @ {target_pos.round(3)}",
                              flush=True)
                    else:
                        target_pos, target_R = self._camera_adapter_target()
                        print(f"[grip_btn] No waypoint selected — planning to live ghost @ {target_pos.round(3)}",
                              flush=True)
                    ghost_quat = (ScipyR.from_matrix(target_R) * GRIPPER_MOUNT_OFFSET).as_quat()

                    if self.moveit2 is not None:
                        if _moveit_waypoints is None:
                            print(f"[grip_btn] MoveIt planning → target={target_pos.round(3)}",
                                  flush=True)
                            plan, seg_dur = self.request_moveit_plan(target_pos, ghost_quat)
                            if plan:
                                if self._shortest_wrist3:
                                    plan = self.postprocess_shortest_wrist3(plan)
                                _moveit_waypoints  = plan
                                _moveit_durations  = seg_dur
                                _moveit_start_q    = self.q_current.copy()
                                self.publish_preview_trajectory(
                                    plan, durations=seg_dur,
                                    preview_duration=cfg.preview_duration)
                                print("[grip_btn] Preview running (RViz) — press again to execute.",
                                      flush=True)
                            else:
                                print("[grip_btn] MoveIt plan FAILED.", flush=True)
                        else:
                            plan              = _moveit_waypoints
                            _moveit_waypoints = None
                            self.q_current    = _moveit_start_q.copy()
                            spd = max(0.01, self._exec_speed_scale)
                            execute_steps = [max(1, round(d / IK_DT / spd))
                                             for d in _moveit_durations]
                            ctrl_interp_gen = self.waypoints_lerp(plan, steps_list=execute_steps)
                            print(f"[grip_btn] Executing planned trajectory  "
                                  f"speed×{spd:.2f}.", flush=True)
                            if self.rtde_c is not None:
                                self.rtde_c.moveJ(plan[-1].tolist(),
                                                  cfg.movej_speed * spd,
                                                  cfg.movej_accel * spd,
                                                  asynchronous=True)
                    else:
                        print("[grip_btn] MoveIt2 not available — discrete moves require MoveIt2.",
                              flush=True)
                prev_grip_btn = right_grip_btn

                # B button — remove selected waypoint, or cancel pending plan
                if right_cancel_btn and not prev_cancel_btn:
                    sel = self.wp_manager.selected
                    if sel is not None:
                        self.wp_manager.remove(sel)
                        print(f"[waypoint] Removed #{sel}", flush=True)
                        self._send_waypoints_to_unity()
                        if self.ros is not None:
                            self.publish_waypoints(self.ros.get_clock().now().to_msg())
                    elif _moveit_waypoints is not None:
                        _moveit_waypoints = None
                        _moveit_durations = None
                        print("[cancel] Plan/preview cancelled.", flush=True)
                prev_cancel_btn = right_cancel_btn

                if ctrl_interp_gen is not None:
                    try:
                        self.q_current = next(ctrl_interp_gen).copy()
                    except StopIteration:
                        ctrl_interp_gen = None

                # 5. Middle trigger — continuous OSC tracking (hold to follow)
                if not right_send_btn and prev_send_btn:
                    ctrl_state = "idle"
                    if self.rtde_c is not None:
                        self.rtde_c.speedJ([0.0] * 6, cfg.speedj_stop_accel, IK_DT)
                    print("[send] Stopped.")

                if right_send_btn and not prev_send_btn:
                    q_jnp = jnp.array(self.q_current)
                    ee_pos, ee_rot = self._ee_world(q_jnp)
                    R_ik      = ScipyR.from_matrix(ee_rot) * GRIPPER_MOUNT_OFFSET.inv()
                    ik_euler  = R_ik.as_euler('xyz', degrees=True)
                    ghost_euler_init    = ik_euler.copy()
                    ghost_euler_init[0] = ((ik_euler[0] - 90.0 + 180.0) % 360.0) - 180.0
                    self.ghost_pos   = ee_pos.copy()
                    self.ghost_euler = ghost_euler_init.copy()
                    ctrl_state = "tracking"
                    print(f"[send] Ghost snapped to EE — tracking. pos={ee_pos.round(3)}")

                # Index trigger (rising edge) — add / update waypoint
                if right_index_btn and not prev_index_trigger:
                    self._do_add_waypoint()

                # Add Waypoint button (simulate_controller GUI)
                if right_add_wp_btn:
                    self._do_add_waypoint()

                # Thumbstick X — cycle waypoints (rising edge on threshold cross)
                stick_right = thumbstick_x >  0.5
                stick_left  = thumbstick_x < -0.5
                prev_right  = prev_thumbstick_x >  0.5
                prev_left   = prev_thumbstick_x < -0.5
                if stick_right and not prev_right:
                    self.wp_manager.cycle_selection(forward=True)
                    self._send_waypoints_to_unity()
                    if self.ros is not None:
                        self.publish_waypoints(self.ros.get_clock().now().to_msg())
                    print(f"[waypoint] Selected #{self.wp_manager.selected}", flush=True)
                if stick_left and not prev_left:
                    self.wp_manager.cycle_selection(forward=False)
                    self._send_waypoints_to_unity()
                    if self.ros is not None:
                        self.publish_waypoints(self.ros.get_clock().now().to_msg())
                    print(f"[waypoint] Selected #{self.wp_manager.selected}", flush=True)
                prev_thumbstick_x = thumbstick_x

                # Proximity-based auto-select: select the nearest waypoint
                # within 15 cm; deselect when none are that close.
                if len(self.wp_manager) > 0:
                    _prox_pos, _ = self._camera_adapter_target()
                    if np.isfinite(_prox_pos).all():
                        _prev_sel = self.wp_manager.selected
                        self.wp_manager.select_nearest(_prox_pos, max_dist=0.15,
                                                      deselect_on_miss=False)
                        if self.wp_manager.selected != _prev_sel:
                            self._send_waypoints_to_unity()
                            if self.ros is not None:
                                self.publish_waypoints(self.ros.get_clock().now().to_msg())
                            if self.wp_manager.selected is not None:
                                print(f"[waypoint] Auto-selected #{self.wp_manager.selected}",
                                      flush=True)
                            else:
                                print("[waypoint] Deselected (moved away)", flush=True)

                if ctrl_state == "tracking":
                    q_jnp = jnp.array(self.q_current)
                    M_inv, J_jnp, ee_tmat = \
                        self.robot.dynamically_consistent_velocity_control_matrices(q_jnp)
                    ee_pos_base  = np.array(ee_tmat[:3, 3])
                    ee_rot_base  = np.array(ee_tmat[:3, :3])
                    ee_pos_world = self._base_R_full @ ee_pos_base + self._base_pos

                    target_pos_world, _target_R = self._camera_adapter_target()
                    target_rot_world = (ScipyR.from_matrix(_target_R) * GRIPPER_MOUNT_OFFSET).as_matrix()

                    pos_err = np.linalg.norm(ee_pos_world - target_pos_world)
                    target_pos_base = self._base_R_full.T @ (target_pos_world - self._base_pos)
                    target_rot_base = self._base_R_full.T @ target_rot_world

                    qdot = self._nominal_osc(
                        q_jnp,
                        jnp.array(ee_pos_base), jnp.array(ee_rot_base),
                        jnp.array(target_pos_base), jnp.array(target_rot_base),
                        jnp.zeros(3), jnp.zeros(3),
                        J_jnp, M_inv,
                    )
                    qdot_np = np.asarray(qdot)
                    qdot_np = np.asarray(self.cbf.safety_filter(q_jnp, jnp.array(qdot_np)))
                    qdot_np = np.clip(qdot_np, -cfg.qdot_max, cfg.qdot_max)
                    qdot_np[self.fixed_joints] = 0.0
                    self.q_current[self.fixed_joints] = self.fixed_joint_rads

                    _R_err   = ee_rot_base @ target_rot_base.T
                    _ori_err = float(np.arccos(np.clip(
                        (np.trace(_R_err) - 1.0) / 2.0, -1.0, 1.0)))
                    if pos_err < cfg.convergence_pos_m and _ori_err < cfg.convergence_ori_rad:
                        qdot_np[:] = 0.0

                    if self.rtde_c is not None:
                        q_target = self.q_current + qdot_np * IK_DT
                        self.rtde_c.servoJ(q_target.tolist(), cfg.servo_speed,
                                           cfg.servo_accel, IK_DT,
                                           cfg.servo_lookahead, cfg.servo_gain)
                    else:
                        self.q_current = self.q_current + qdot_np * IK_DT
                    was_tracking = True
                else:
                    if was_tracking and self.rtde_c is not None:
                        self.rtde_c.servoStop()
                    was_tracking = False

                prev_send_btn      = right_send_btn
                prev_index_trigger = right_index_btn
                self.gripper_q = float(np.clip(grip_val * 0.8, 0.0, 0.8))

                # 6. State broadcast
                if self.quest_bridge is None:
                    ghost_quat = ScipyR.from_euler('xyz', self.ghost_euler,
                                                   degrees=True).as_quat()
                    out = {
                        "stamp":           time.time(),
                        "joint_positions": self.q_current.tolist(),
                        "gripper_q":       float(self.gripper_q),
                        "ghost":           {"pos":       self.ghost_pos.tolist(),
                                            "quat_xyzw": ghost_quat.tolist()},
                    }
                    try:
                        self.state_send.send_string(json.dumps(out), zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                elif t0 - last_unity_scene_t >= UNITY_SCENE_DT:
                    last_unity_scene_t = t0
                    self.publish_unity_scene_objects()

                q_for_unity = self.q_current.copy()
                q_for_unity[0] += np.pi
                try:
                    self.unity_joint_pub.send_string(
                        json.dumps({"joint_values": q_for_unity.tolist()}), zmq.NOBLOCK)
                except zmq.Again:
                    pass
                try:
                    self.unity_gripper_pub.send_string(str(self.gripper_q), zmq.NOBLOCK)
                except zmq.Again:
                    pass
                try:
                    self.unity_ghost_gripper_pub.send_string(str(self.gripper_q), zmq.NOBLOCK)
                except zmq.Again:
                    pass
                try:
                    _ghost_pos, _ghost_R = self._camera_adapter_target()
                    if (np.isfinite(_ghost_R).all() and np.isfinite(_ghost_pos).all()
                            and abs(np.linalg.det(_ghost_R) - 1.0) < 0.01):
                        _ghost_extra_y_deg = 0.0    # ← tune: extra Y-axis correction (degrees)
                        _ghost_extra_z_deg = 180.0    # ← tune: extra Z-axis correction (degrees)
                        _ghost_correction = (
                            ScipyR.from_euler('x', -90.0, degrees=True) *
                            ScipyR.from_euler('y', _ghost_extra_y_deg, degrees=True) *
                            ScipyR.from_euler('z', _ghost_extra_z_deg, degrees=True)
                        ).as_matrix()
                        _T_send = np.eye(4)
                        _T_send[:3, :3] = _ghost_R @ _ghost_correction
                        _T_send[:3, 3]  = _ghost_pos #+ np.array([0.05, 0.0, 0.05])
                        _ghost_obj = self.T_to_unity_object(
                            obj_id=0, name="ghost_gripper", T_world_object=_T_send)
                        # print(f"[ghost→Unity] pos=({_ghost_pos[0]:.3f}, {_ghost_pos[1]:.3f}, "
                        #       f"{_ghost_pos[2]:.3f})")
                        self.unity_ghost_pub.send_string(json.dumps(_ghost_obj), zmq.NOBLOCK)
                    # # DEBUG: fixed test pose — (0, 0, 0.3m), zero rotation
                    # _T_test = np.eye(4); _T_test[2, 3] = 0.5
                    # _ghost_obj = self.T_to_unity_object(
                    #     obj_id=0, name="ghost_gripper", T_world_object=_T_test)
                    # self.unity_ghost_pub.send_string(json.dumps(_ghost_obj), zmq.NOBLOCK)
                    # print(f"[ghost→Unity] sent: {_ghost_obj}", flush=True)
                except zmq.Again:
                    pass
                except Exception:
                    pass
                if t0 - last_unity_base_t >= 1.0:
                    last_unity_base_t = t0
                    deg = np.rad2deg(self.q_current)
                    # print(f"[joints→Unity] deg={np.round(deg, 1).tolist()}", flush=True)
                    _ux, _uy, _uz = open3d_to_unity_vector(self.T_world_base[:3, 3])
                    # print(f"[base→Unity]   pos=({_ux:.3f}, {_uy:.3f}, {_uz:.3f})", flush=True)
                    try:
                        self.unity_base_pub.send_string(
                            json.dumps({"robot_matrix": self._make_unity_robot_matrix()}),
                            zmq.NOBLOCK)
                    except zmq.Again:
                        pass

                # 7. ROS2 publish
                if self.ros is not None and rclpy.ok():
                    try:
                        self.publish_ros()
                    except Exception as _pub_exc:
                        print(f"[linux_controller4] publish_ros error: {_pub_exc}", flush=True)

                # 8. Meet RT deadline
                elapsed = time.perf_counter() - t0
                if self.rtde_c is not None:
                    self.rtde_c.waitPeriod(t_start)
                else:
                    remaining = IK_DT - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("[linux_controller4] Shutting down.")
            if self.rtde_c is not None:
                try:
                    self.rtde_c.servoStop()
                    self.rtde_c.disconnect()
                except Exception:
                    pass
            if self.controller_recv is not None:
                self.controller_recv.close()
            if self.state_send is not None:
                self.state_send.close()
            if self.config_rep is not None:
                self.config_rep.close()
            if self.quest_bridge is not None:
                self.quest_bridge.stop()
            self.unity_base_pub.close()
            self.unity_joint_pub.close()
            self.unity_gripper_pub.close()
            self.unity_ghost_gripper_pub.close()

    # ── Unity helpers ─────────────────────────────────────────────────────────

    def _make_unity_robot_matrix(self) -> list:
        T = self.T_world_base
        pos_unity = open3d_to_unity_vector(T[:3, 3])
        q_xyzw    = ScipyR.from_matrix(T[:3, :3]).as_quat()
        q_wxyz    = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u_wxyz  = open3d_to_unity_quaternion(q_wxyz)
        q_u_xyzw  = [q_u_wxyz[1], q_u_wxyz[2], q_u_wxyz[3], q_u_wxyz[0]]
        R_unity   = ScipyR.from_quat(q_u_xyzw).as_matrix()
        R_unity   = R_unity @ ScipyR.from_euler('y', -90, degrees=True).as_matrix()
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R_unity
        mat[:3, 3]  = pos_unity
        return mat.flatten(order='F').tolist()

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def T_to_pose(T) -> dict | None:
        if T is None:
            return None
        return {"pos": T[:3, 3].tolist(),
                "quat_xyzw": ScipyR.from_matrix(T[:3, :3]).as_quat().tolist()}

    @staticmethod
    def T_to_unity_object(obj_id: int, name: str, T_world_object,
                          size_o3d=None, color=None, mesh_correction=None,
                          scale_unity=None) -> dict:
        q_xyzw   = ScipyR.from_matrix(T_world_object[:3, :3]).as_quat()
        q_wxyz   = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u_wxyz = open3d_to_unity_quaternion(q_wxyz)
        p_unity  = open3d_to_unity_vector(T_world_object[:3, 3])
        q_unity_xyzw = [float(q_u_wxyz[1]), float(q_u_wxyz[2]),
                        float(q_u_wxyz[3]), float(q_u_wxyz[0])]
        if mesh_correction is not None:
            q_unity_xyzw = (ScipyR.from_quat(q_unity_xyzw) * mesh_correction).as_quat().tolist()
        out = {"id": int(obj_id), "name": str(name),
               "position": [float(v) for v in p_unity],
               "rotation_xyzw": [float(v) for v in q_unity_xyzw]}
        if size_o3d   is not None: out["size"]  = [float(v) for v in open3d_to_unity_vector(np.asarray(size_o3d, dtype=np.float64))]
        if scale_unity is not None: out["scale"] = [float(v) for v in scale_unity]
        if color       is not None: out["color"] = [float(v) for v in color]
        return out

    @staticmethod
    def pose_to_T(pose: dict) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3]  = pose["pos"]
        T[:3, :3] = ScipyR.from_quat(pose["quat_xyzw"]).as_matrix()
        return T

    @staticmethod
    def gripper_q_to_joints(q: float) -> list:
        q = float(np.clip(q, 0.0, 0.8))
        return [q, q, -q, -q, q, -q]

    @staticmethod
    def lerp_joints(q0, q1, steps=90):
        for i in range(steps + 1):
            yield q0 + (q1 - q0) * i / steps

    @staticmethod
    def waypoints_lerp(waypoints, steps_per_seg=20, steps_list=None):
        for i in range(len(waypoints) - 1):
            q0, q1 = waypoints[i], waypoints[i + 1]
            n = steps_list[i] if steps_list is not None else steps_per_seg
            n = max(1, n)
            for j in range(n + 1):
                yield q0 + (q1 - q0) * j / n

    @staticmethod
    def postprocess_shortest_wrist3(waypoints: list) -> list:
        """Replace wrist_3 (joint 5) values with a shortest-path linear interpolation.

        Keeps joints 0-4 exactly as planned; wrist_3 takes the shortest angular
        route from its start to end value rather than whatever OMPL chose.
        """
        if len(waypoints) < 2:
            return waypoints
        wps = [np.asarray(q, float).copy() for q in waypoints]
        start_w3 = wps[0][5]
        end_w3   = wps[-1][5]
        # Signed shortest angular difference (wraps within ±π)
        diff = ((end_w3 - start_w3 + np.pi) % (2 * np.pi)) - np.pi
        n = len(wps) - 1
        for i, wp in enumerate(wps):
            wp[5] = start_w3 + diff * (i / n)
        return wps


# =============================================================================
# ROS2 node wrapper
# =============================================================================

def run_as_ros_node(cfg: Config, robot_ip: str | None,
                    simulate_controller: bool = False,
                    use_moveit: bool = False, quest_bridge=None,
                    quest_ip: str = "192.168.50.201"):
    rclpy.init()
    node = rclpy.create_node("linux_controller4")
    ctrl = LinuxController(cfg, robot_ip=robot_ip, ros_node=node,
                           use_moveit=use_moveit, quest_bridge=quest_bridge,
                           quest_ip=quest_ip)
    t = threading.Thread(target=ctrl.run,
                         kwargs={"simulate_controller": simulate_controller},
                         daemon=True)
    t.start()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.running = False
        t.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


# =============================================================================
# Entry point
# =============================================================================

# (proxy) skim3674@isye-saillin2:~/Desktop/ProxyGrip$ python main_logic/linux_controller4.py --moveit --ros --no-robot --simulate_controller --camera-access
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Linux controller node (frax, no PyBullet)")
    ap.add_argument("--robot-ip",    default="192.168.50.70")
    ap.add_argument("--no-robot",    action="store_true")
    ap.add_argument("--ros",         action="store_true")
    ap.add_argument("--ctrl-port",   type=int, default=5200)
    ap.add_argument("--state-port",  type=int, default=5201)
    ap.add_argument("--config-port", type=int, default=5202)
    ap.add_argument("--simulate_controller", action="store_true")
    ap.add_argument("--moveit",      action="store_true")
    ap.add_argument("--direct-connect",  action="store_true")
    ap.add_argument("--camera-access",   action="store_true",
                    help="Connect to Quest camera for ArUco/WorldRoot even with --simulate_controller")
    ap.add_argument("--quest-ip",    default="192.168.50.201")
    ap.add_argument("--marker-id",   type=int,   default=11)
    ap.add_argument("--marker-size", type=float, default=0.090)
    args, _ = ap.parse_known_args()

    # ── User-defined parameters ────────────────────────────────────────────────
    cfg = Config(
        project_root = ROOT,

        home_q_deg = np.array([-124.23, -58.60, 109.15, -140.21, -90.41, -37.54]),

        kp_pos   = 200.0,   # OSC position gain (N/m equivalent) [real/sim]
        kp_ori   = 100.0,   # OSC orientation gain [real/sim]
        qdot_max = 5.0,    # max joint velocity (rad/s) sent to speedj [real robot]
        speedj_stop_accel = 3.0,  # deceleration (rad/s²) used to stop with speedj [real robot]
        cbf_alpha = 15.0,  # CBF class-K gain — higher = tighter safety margins [real/sim]

        joint_limit_margin_deg = 0.0,  # extra safety margin shrunk inward from each limit [real/sim]

        servo_speed     = 1.0,   # max speed for servoJ (rad/s) [real robot]
        servo_accel     = 1.0,   # max accel for servoJ (rad/s²) [real robot]
        servo_lookahead = 0.1,   # servoJ lookahead time (s) — smooths the trajectory [real robot]
        servo_gain      = 300,   # servoJ proportional gain — higher = stiffer tracking [real robot]

        movej_speed = 1.0,   # joint speed for moveJ point-to-point moves (rad/s) [real robot]
        movej_accel = 1.0,   # joint accel for moveJ (rad/s²) [real robot]

        fixed_joint_angles_deg = {},

        # ── Joint-limit presets — fill in actual values for your use cases ──────
        # 0=shoulder_pan  1=shoulder_lift  2=elbow  3=wrist_1  4=wrist_2  5=wrist_3
        joint_limit_presets = {
            "default": (
                np.array([-190.0, -105.0,   0., -180.0, -180., -180.]),  # min
                np.array([ -90.0,  -20.0, 170.,   180.0,  180.,  180.]),  # max
            ),
            "top_down": (
                np.array([-190.0, -105.0,   0., 120, -180., -180.]),      # min
                np.array([ -90.0,  -20.0, 170.,   360.0,  0.,  180.]),   # max
            ),
            "sideways": (
                np.array([-190.0, -105.0,   0., -180.0, -180., -270.]),   # min
                np.array([ -90.0,  -20.0, 170.,   180.0,  180., -90.]),  # max
            ),
            

            # "default": (
            #     np.array([-160.0, -90.0,   0., -180.0, -180., -180.]),  # min
            #     np.array([ -90.0,  -20.0, 150.,   180.0,  180.,  180.]),  # max
            #     np.array([-125., -44., 109., 207., -90., -215.]),         # null-space target
            # ),
            # "top_down": (
            #     np.array([-160.0, -90.0,   0., 120, -180., -180.]),      # min
            #     np.array([ -90.0,  -20.0, 150.,   360.0,  0.,  180.]),   # max
            #     np.array([-125., -44., 109., 207., -90., -215.]),         # null-space target
            # ),
            # "sideways": (
            #     np.array([-160.0, -90.0,   0., -180.0, -180., -270.]),   # min
            #     np.array([ -90.0,  -20.0, 150.,   180.0,  180., -90.]),  # max
            #     np.array([-135., -43., 113., 293., -319., -182.]),        # null-space target
            # ),

        },

        workspace_box = np.array([
            -1.0474, -0.3082, 0.05,
             0.7942,  0.4220, 0.50,
        ]),

        obstacle_boxes = [
            ([-0.1266, 0.0569, -0.050], [0.7366, 0.2921, 0.025], 0.0),
            # Baseboard (201×250 mm STL, 1.2× margin, 200 mm tall) — set center from marker
            obstacle_box(center=[0.0, 0.0, 0.0], stl_xy_m=(0.201, 0.250),
                         scale=1.0, height=0.2, yaw_deg=0.0),
            # Gearstand (240×180 mm STL, 1.2× margin, 200 mm tall) — set center from marker
            obstacle_box(center=[0.01600, 0.220, 0.0], stl_xy_m=(0.240, 0.180),
                         scale=1.0, height=0.2, yaw_deg=0.0),
        ],

        scene_assembly_state  = "end",
        scene_baseboard_marker = 10,
        scene_baseboard_scale  = 0.001,
        scene_gearstand_marker = 51,
        scene_gearstand_scale  = 0.001,
        scene_stands_scale     = 0.001,
    )
    # ── End user parameters ────────────────────────────────────────────────────

    robot_ip = None if args.no_robot else args.robot_ip

    quest_bridge = None
    if args.direct_connect or args.camera_access:
        if not HAS_QUEST_BRIDGE:
            print("[main] ERROR: quest_direct_bridge not found.")
            sys.exit(1)
        quest_bridge = QuestDirectBridge(quest_ip=args.quest_ip)
        quest_bridge.start_aruco_display(marker_id=args.marker_id,
                                         marker_size_m=args.marker_size)

    if args.ros and HAS_ROS:
        run_as_ros_node(cfg, robot_ip,
                        simulate_controller=args.simulate_controller,
                        use_moveit=args.moveit,
                        quest_bridge=quest_bridge,
                        quest_ip=args.quest_ip)
    else:
        ctrl = LinuxController(cfg, robot_ip=robot_ip,
                               ctrl_port=args.ctrl_port,
                               state_port=args.state_port,
                               config_port=args.config_port,
                               quest_bridge=quest_bridge,
                               quest_ip=args.quest_ip)
        ctrl.run(simulate_controller=args.simulate_controller)
