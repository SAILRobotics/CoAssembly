"""robot_control_server.py — Dedicated process owning the UR10e hardware link.

Merged from the former robot_control_server.py (ZMQ loop) and
robot_controller.py (hardware abstraction). Handles only what is needed for
tool grasping:
  - PyBullet scene (IK + sim grasp animation)
  - RTDE receive / control (real robot only)
  - Gripper actuation (real robot only)
  - Grasp sequence: approach → grasp → retract (sim + real)
  - Unity publishing: base pose + joint angles

Talks to main_with_robot.py (via robot_client.RobotClient) over ZMQ:
    SUB cfg.ROBOT_CMD_PORT   (bind) — commands in
    PUB cfg.ROBOT_EVENT_PORT (bind) — state + events out

Usage
-----
    python robot_control_server.py
    python robot_control_server.py --robot-ip 192.168.50.70 --no-simulation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

try:
    from pybullet_ik import IKScene as PyBulletScene
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False
    print("[robot_control_server] pybullet_ik import failed — cannot run.")

try:
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
    _RTDE_AVAILABLE = True
except ImportError:
    RTDEReceiveInterface = None   # type: ignore[assignment,misc]
    RTDEControlInterface = None   # type: ignore[assignment,misc]
    _RTDE_AVAILABLE = False

try:
    from robot_controller.robotiq_gripper import RobotiqGripper
    _GRIPPER_AVAILABLE = True
except ImportError:
    RobotiqGripper = None         # type: ignore[assignment,misc]
    _GRIPPER_AVAILABLE = False

try:
    from utils.unity_conversion import open3d_to_unity_vector, open3d_to_unity_quaternion
except ImportError:
    def open3d_to_unity_vector(v):       # type: ignore[misc]
        return np.array([v[0], v[2], v[1]], dtype=float)
    def open3d_to_unity_quaternion(q):   # type: ignore[misc]
        return np.array([q[0], q[2], q[1], -q[3]], dtype=float)

import main_setting as cfg

_FILE_DIR = Path(__file__).resolve().parent
_URDF_PATH = _FILE_DIR / "robot_assets" / "ur10e.urdf"

_DEFAULT_Q_DEG = cfg.ROBOT_DEFAULT_JOINT_DEG

# ── frax / CBF (optional) ─────────────────────────────────────────────────────
try:
    from frax_controller import _FraxController as _RobotFraxController
    import jax.numpy as _jnp
    _HAS_FRAX = True
    print("[RobotServer] frax/CBF loaded OK.")
except Exception as _e:
    _RobotFraxController = None   # type: ignore[assignment,misc]
    _jnp = None
    _HAS_FRAX = False
    print(f"[RobotServer] frax/CBF NOT loaded — CBF inactive: {_e}")

# _HAS_FRAX = False
# ── Simulation waypoint runner ────────────────────────────────────────────────

class _PbJointRunner:
    """Linear joint interpolation over time — sim equivalent of moveJ."""
    _DURATION_S = 3.0

    def __init__(self, start_q: np.ndarray, target_q: np.ndarray,
                 duration_s: float | None = None):
        self.start_q  = np.asarray(start_q, dtype=float)
        self.target_q = np.asarray(target_q, dtype=float)
        self.duration_s = (self._DURATION_S if duration_s is None
                           else max(float(duration_s), 1e-3))
        self._elapsed = 0.0
        self.done     = False

    def update(self, robot_id: int, arm_indices: list, dt: float) -> None:
        if self.done:
            return
        import pybullet as p
        self._elapsed += dt
        t = min(self._elapsed / self.duration_s, 1.0)
        q = self.start_q + t * (self.target_q - self.start_q)
        for j_idx, q_j in zip(arm_indices, q):
            p.resetJointState(robot_id, j_idx, float(q_j))
        if t >= 1.0:
            self.done = True


def _wrap_nearest(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Shift q_target by ±n·2π per joint to minimise travel from q_ref."""
    TWO_PI = 2.0 * np.pi
    q = q_target - TWO_PI * np.round((q_target - q_ref) / TWO_PI)
    q = np.where(q < -TWO_PI, q + TWO_PI, q)
    q = np.where(q >  TWO_PI, q - TWO_PI, q)
    return q


def _build_pb_scene(simulation: bool, use_calibrated_robot_base: bool,
                    sim_q: np.ndarray) -> "PyBulletScene | None":
    calib_dir = _FILE_DIR / "hand_eye_data" / "results"
    scene: "PyBulletScene | None" = None
    if simulation:
        if use_calibrated_robot_base and calib_dir.exists():
            try:
                p1 = np.load(calib_dir / "phase1_results.npz")
                scene = PyBulletScene(T_world_base=p1["T_world_base"])
                scene.build()
                scene.update_robot(sim_q)
                print("[RobotServer] Simulation + calibrated base pose.")
            except Exception as e:
                print(f"[RobotServer] PyBullet (calibrated) failed: {e}")
        else:
            T = np.eye(4, dtype=float)
            T[:3, 3] = [-0.4, -0.8, 0.4]
            try:
                scene = PyBulletScene(T_world_base=T)
                scene.build()
                scene.update_robot(sim_q)
                print("[RobotServer] Simulation — hardcoded base pose.")
            except Exception as e:
                print(f"[RobotServer] PyBullet scene failed: {e}")
    else:
        if calib_dir.exists():
            try:
                scene = PyBulletScene.from_calibration(calib_dir)
                scene.build()
                scene.update_robot(sim_q)
            except Exception as e:
                print(f"[RobotServer] PyBullet scene failed: {e}")
        else:
            print(f"[RobotServer] Calibration dir not found: {calib_dir}")
    return scene


def _load_pegboard_cbf_obstacle() -> "list[dict]":
    """Build one oriented cuboid around the saved physical pegboard.

    The saved marker transform is the pegboard-local frame used by the scene
    viewer.  Inflate the board by 3 cm in-plane.  Along the board normal, keep
    3 cm behind it and extend 8 cm toward the robot so the CBF starts steering
    link collision spheres away well before contact.
    """
    path = cfg.SCENE_LAYOUT_DIR / "T_world10_pegboard101.npz"
    if not path.exists():
        print(f"[RobotServer] Pegboard CBF obstacle not found: {path}")
        return []
    try:
        data = np.load(path)
        T = np.asarray(data["T_world10_pegboard"], dtype=float)
        width = float(data["pegboard_width_m"])
        height = float(data["pegboard_height_m"])
        offset_x = float(data["marker_offset_right_m"])
        offset_y = float(data["marker_offset_top_m"])
        thickness = 0.03
        margin = 0.03
        rear_margin = 0.03
        robot_side_margin = 0.08
        z_min = -thickness - rear_margin
        z_max = robot_side_margin
        center_local = np.array([
            offset_x - width / 2.0,
            offset_y - height / 2.0,
            (z_min + z_max) / 2.0,
        ])
        center_world = T[:3, :3] @ center_local + T[:3, 3]
        obstacle = {
            "center": center_world.tolist(),
            "half": [width / 2.0 + margin,
                     height / 2.0 + margin,
                     (z_max - z_min) / 2.0],
            "rotation": T[:3, :3].tolist(),
        }
        print("[RobotServer] Pegboard CBF obstacle: center="
              f"{np.round(center_world, 3).tolist()} size="
              f"{np.round(2.0 * np.asarray(obstacle['half']), 3).tolist()}")
        return [obstacle]
    except Exception as exc:
        print(f"[RobotServer] Pegboard CBF obstacle load failed: {exc}")
        return []


# ── Main server class (merged controller + ZMQ loop) ─────────────────────────

class RobotControlServer:
    _STATE_INTERVAL          = 1.0 / 30.0
    _BASE_YAW_CORRECTION_DEG = -90.0
    _MOVE_POS_TOL_M          = 0.03
    _MOVE_ANGLE_TOL_RAD      = np.deg2rad(10.0)
    _BOARD_MOVE_POS_TOL_M    = 0.01
    _BOARD_MOVE_ANGLE_TOL_RAD = np.deg2rad(3.0)
    # Study completion is intentionally identical for AR and freedrive. Tight
    # 1 cm/3 deg values remain an IK-validation criterion, not a study result.
    _WORKHOLDING_MOVE_POS_TOL_M = 0.05
    _WORKHOLDING_MOVE_ANGLE_TOL_RAD = np.deg2rad(15.0)
    _HANDOVER_MOVE_POS_TOL_M = 0.015
    _HANDOVER_MOVE_ANGLE_TOL_RAD = np.deg2rad(6.0)
    _MOVE_DWELL_S            = 0.30
    _MOVE_STALL_TIMEOUT_S     = 3.0
    _MOVE_PROGRESS_POS_M      = 0.002
    _MOVE_PROGRESS_ANGLE_RAD  = np.deg2rad(1.0)
    _MOVE_BRAKE_DIST_M       = 0.05
    _MOVE_BRAKE_ANGLE_RAD    = np.deg2rad(8.0)
    _MOVE_SCALE_FLOOR        = 0.05
    _SERVO_EMA_ALPHA         = 0.70
    _BOARD_SERVO_EMA_ALPHA   = 0.85
    _SIM_BOARD_SMOOTH_ALPHA  = 0.55
    _SERVO_LOOKAHEAD_S       = 0.08
    _SERVO_GAIN              = 400
    _WORKHOLDING_EMA_ALPHA   = 1.00
    _WORKHOLDING_LOOKAHEAD_S = 0.03
    _WORKHOLDING_GAIN        = 500
    _WORKHOLDING_OSC_SPEED_SCALE = 1.15
    _WORKHOLDING_BRAKE_DIST_M = 0.06
    _WORKHOLDING_BRAKE_ANGLE_RAD = np.deg2rad(10.0)
    # Hand delivery stays reactive at long range, then increasingly filters and
    # brakes the command only inside the final 8 cm around the user's hand.
    _HANDOVER_OSC_SPEED_SCALE = 1.15
    _HANDOVER_BRAKE_DIST_M = 0.08
    _HANDOVER_BRAKE_ANGLE_RAD = np.deg2rad(12.0)
    _HANDOVER_EMA_FAR = 0.85
    _HANDOVER_EMA_NEAR = 0.62
    _HANDOVER_LOOKAHEAD_FAR_S = 0.06
    _HANDOVER_LOOKAHEAD_NEAR_S = 0.10
    _HANDOVER_GAIN_FAR = 450
    _HANDOVER_GAIN_NEAR = 380
    # Each approach/grasp/retract waypoint normally takes three seconds in the
    # visual simulator. Tool fetching is intentionally quicker for iteration;
    # this value is never used for physical-robot motion.
    _SIM_TOOL_GRASP_DURATION_S = 0.75

    def __init__(self, quest_ip: str, robot_ip: "str | None", simulation: bool,
                 use_calibrated_robot_base: bool, gripper_collision: bool,
                 use_diff_ik: bool = True):
        self.simulation   = simulation
        self.use_diff_ik  = use_diff_ik
        self._robot_ip    = robot_ip
        self.control_hz   = cfg.ROBOT_CONTROL_HZ if simulation else cfg.ROBOT_CONTROL_HZ_REAL

        self._default_q = np.deg2rad(_DEFAULT_Q_DEG)
        # Give simulation a visible startup move.  The real scene is synced to
        # the measured hardware joints before its first startup command.
        initial_q = (np.deg2rad(cfg.JOINT_REST_DEG)
                     if simulation else self._default_q)
        self.pb_scene = _build_pb_scene(
            simulation, use_calibrated_robot_base, initial_q)
        if self.pb_scene is None:
            raise RuntimeError("[RobotServer] pb_scene failed to build — cannot run.")
        self.pb_scene.set_joint_limits(cfg.JOINT_MIN_DEG, cfg.JOINT_MAX_DEG, degrees=True)
        self.pb_scene.set_rest_poses(cfg.JOINT_REST_DEG, degrees=True)

        self._T_world_base  = np.array(self.pb_scene.T_world_base, dtype=float)
        self._approach_dist = 0.30
        speed_scale         = 1.0 if simulation else cfg.REAL_ROBOT_SPEED_SCALE
        self._speed_scale   = float(speed_scale)
        self._last_q: "np.ndarray | None" = None

        # Real-robot hardware (lazy connections)
        self._recv:      "RTDEReceiveInterface | None" = None
        self._rtde_ctrl: "RTDEControlInterface | None" = None
        self._gripper:   "RobotiqGripper | None"       = None
        self._in_servo   = False

        # Grasp state machine (shared by sim and real)
        self._grasp_phase:        "str | None"        = None
        self._grasp_on_complete:  "Callable | None"   = None
        self._grasp_on_phase:     "Callable | None"   = None
        self._grasp_joints:            "np.ndarray | None" = None
        self._grasp_q_approach:        "np.ndarray | None" = None
        self._grasp_q_above:           "np.ndarray | None" = None
        self._grasp_q_above_approach:  "np.ndarray | None" = None
        self._grasp_runner       = None          # sim only: _PbJointRunner
        self._grasp_sim_duration_s = _PbJointRunner._DURATION_S
        self._grasp_move_sent:    bool           = False   # real only: async moveJ sent
        self._grasp_settle_start: "float | None" = None    # real only: close+settle timer
        self._grasp_ok:             bool           = False   # real only: object detected
        self._grasp_cancel_pending: bool           = False   # cancel in progress; all phases

        # Board-only force handoff.  This is intentionally independent of the
        # programmed tool/part grasp state machine above.
        self._board_state: str = "inactive"
        self._board_force_mode: "str | None" = None  # None | 'grasp' | 'release'
        self._board_force_baseline: "np.ndarray | None" = None
        self._board_force_hits = 0
        self._board_force_last_t = 0.0
        self._board_freedrive_active = False
        # Whether holding a board should auto-engage freedrive, and what a
        # completed board_move should hand back to.  Defaults on (today's
        # behaviour); the workholding study harness flips this per condition
        # so an AR-only trial never silently regains physical assist.
        self._board_freedrive_policy = True
        # General tool/part handover: hold the gripper closed until the human
        # applies a sustained pull, then open and acknowledge the request.
        self._handover_pull_active = False
        self._handover_force_baseline: "np.ndarray | None" = None
        self._handover_force_hits = 0
        self._handover_force_last_t = 0.0
        self._handover_request_id = None

        # Move-to-pose state machine (IK + CBF servo loop, driven by tick())
        self._move_phase:       "str | None"        = None
        self._move_target_pos:  "np.ndarray | None" = None
        self._move_target_quat: "np.ndarray | None" = None
        self._move_on_complete: "Callable | None"   = None
        self._move_smooth_q:    "np.ndarray | None" = None
        self._move_conv_start:  "float | None"      = None  # perf_counter when TCP first meets pose tolerances
        self._move_diag_t:      float               = 0.0   # last convergence-status print time
        self._move_pybullet_q_target: "np.ndarray | None" = None
        self._move_best_dist:   float               = float('inf')
        self._move_best_angle:  float               = float('inf')
        self._move_progress_t:  float               = 0.0
        self._move_is_board:    bool                = False
        self._move_motion_profile: str              = "default"

        # Direct joint move used by gearbox step selection.
        self._joint_move_target: "np.ndarray | None" = None
        self._joint_move_runner = None
        self._joint_move_sent = False
        self._joint_move_started = 0.0
        self._joint_move_on_complete: "Callable | None" = None

        # Automatic startup positioning, shared by simulation and hardware.
        self._startup_active = True
        self._startup_move_sent = False
        self._startup_runner = (
            _PbJointRunner(self.pb_scene.current_q.copy(), self._default_q)
            if simulation else None)

        # frax CBF controller (optional — falls back to rate-limited IK)
        self._frax = None
        if _HAS_FRAX and _URDF_PATH.exists():
            try:
                self._frax = _RobotFraxController(
                    urdf_path          = str(_URDF_PATH),
                    T_world_base       = self._T_world_base,
                    q_min              = np.deg2rad(cfg.JOINT_MIN_DEG).tolist(),
                    q_max              = np.deg2rad(cfg.JOINT_MAX_DEG).tolist(),
                    ws_lo              = list(cfg.WORKSPACE_LO),
                    ws_hi              = list(cfg.WORKSPACE_HI),
                    obstacle_boxes     = _load_pegboard_cbf_obstacle(),
                    gripper_collision  = gripper_collision,
                )
                print("[RobotServer] frax CBF ready.")
            except Exception as e:
                print(f"[RobotServer] frax init failed: {e}")

        _GRN = "\033[92m"; _YLW = "\033[93m"; _RST = "\033[0m"
        if self.use_diff_ik and self._frax is not None:
            print(f"{_GRN}[RobotServer] Motion mode: OSC + CBF differential IK{_RST}")
        elif self._frax is not None:
            print(f"{_YLW}[RobotServer] Motion mode: PyBullet IK + CBF filter (--no-diff-ik){_RST}")
        else:
            print(f"{_YLW}[RobotServer] Motion mode: PyBullet IK only (frax not loaded){_RST}")
        print(f"[RobotServer] Move tuning: brake={self._MOVE_BRAKE_DIST_M*100:.0f} cm/"
              f"{np.rad2deg(self._MOVE_BRAKE_ANGLE_RAD):.0f} deg, "
              f"EMA={self._SERVO_EMA_ALPHA:.2f}, "
              f"lookahead={self._SERVO_LOOKAHEAD_S:.2f} s, "
              f"gain={self._SERVO_GAIN}")
        print(f"[RobotServer] Workholding profile: EMA="
              f"{self._WORKHOLDING_EMA_ALPHA:.2f}, "
              f"lookahead={self._WORKHOLDING_LOOKAHEAD_S:.2f} s, "
              f"gain={self._WORKHOLDING_GAIN}, "
              f"OSC speed={self._WORKHOLDING_OSC_SPEED_SCALE:.2f}x")
        print(f"[RobotServer] Stop/clamp: "
              f"{self._MOVE_POS_TOL_M*100:.0f} cm/"
              f"{np.rad2deg(self._MOVE_ANGLE_TOL_RAD):.0f} deg for "
              f"{self._MOVE_DWELL_S:.1f} s, workspace inset="
              f"{cfg.ROBOT_TARGET_WORKSPACE_MARGIN_M*100:.0f} cm")

        # Gate publish_base until vision side has locked the anchor and called set_scene_origin
        self._scene_origin_set = False

        if not simulation:
            self.connect_gripper()

        ctx = zmq.Context.instance()

        # Unity publishers
        self._base_pub  = ctx.socket(zmq.PUB)
        self._base_pub.connect(f"tcp://{quest_ip}:{cfg.ROBOT_BASE_PORT}")
        self._joint_pub = ctx.socket(zmq.PUB)
        self._joint_pub.connect(f"tcp://{quest_ip}:{cfg.ROBOT_JOINT_PORT}")

        # main_with_robot.py ↔ server
        self._cmd_sub = ctx.socket(zmq.SUB)
        self._cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._cmd_sub.bind(f"tcp://{cfg.LOCALHOST}:{cfg.ROBOT_CMD_PORT}")
        self._event_pub = ctx.socket(zmq.PUB)
        self._event_pub.bind(f"tcp://{cfg.LOCALHOST}:{cfg.ROBOT_EVENT_PORT}")
        time.sleep(0.2)

        self._running = True
        print(f"[RobotServer] Ready — {'simulation' if simulation else f'live ({robot_ip})'}, "
              f"cmd:{cfg.ROBOT_CMD_PORT} event:{cfg.ROBOT_EVENT_PORT} @ {self.control_hz} Hz"
              + ("" if simulation else f"  (speed_scale={cfg.REAL_ROBOT_SPEED_SCALE})"))

    # ── ZMQ I/O ──────────────────────────────────────────────────────────────

    def _publish(self, payload: dict) -> None:
        try:
            self._event_pub.send_string(json.dumps(payload))
        except Exception as e:
            print(f"[RobotServer] Publish error: {e}")

    def _publish_event(self, name: str, request_id, **kw) -> None:
        self._publish({"type": "event", "name": name, "request_id": request_id, **kw})

    def _publish_state(self) -> None:
        q = self.pb_scene.current_q
        self.publish_joints(q)
        if self._scene_origin_set:
            self.publish_base(self.pb_scene.T_world_base)
        self._publish({
            "type":               "state",
            "q":                  q.tolist(),
            "tool_grasp_running": bool(self.tool_grasp_running),
            "move_running":       bool(self._startup_active
                                        or self._move_phase is not None
                                        or self._joint_move_target is not None),
            "board_state":        self._board_state,
            "board_interaction_active": self._board_force_mode is not None
                                        or self._board_state != "inactive",
            "board_freedrive_active": self._board_freedrive_active,
            "object_handover_waiting": self._handover_pull_active,
        })

    def _poll_commands(self) -> None:
        while True:
            try:
                raw = self._cmd_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[RobotServer] Bad command: {e}")
                continue
            try:
                self._handle_cmd(msg)
            except Exception as e:
                print(f"[RobotServer] Command '{msg.get('cmd')}' failed: {e}")

    def _handle_cmd(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        rid = msg.get("request_id")

        if cmd == "set_scene_origin":
            T = np.array(msg["T"], dtype=float).reshape(4, 4)
            self.pb_scene.set_scene_origin(T)
            self._T_world_base = np.array(self.pb_scene.T_world_base, dtype=float)
            self._scene_origin_set = True
            self.publish_base(self.pb_scene.T_world_base)

        elif cmd == "execute_grasp":
            self.execute_grasp(
                msg["grasp_joints"],
                category     = msg.get("category", "tool"),
                board_normal = msg.get("board_normal"),
                tool_type    = msg.get("tool_type", ""),
                on_complete  = lambda ok, rid=rid: self._publish_event(
                    "grasp_done", rid, ok=bool(ok)),
                on_phase     = lambda phase, rid=rid: self._publish_event(
                    "grasp_phase", rid, phase=phase))

        elif cmd == "move_to_pose":
            self.move_to_pose(
                pos  = msg["pos"],
                quat = msg.get("quat"),
                board_move      = bool(msg.get("board_move", False)),
                force_pybullet_ik = bool(msg.get("force_pybullet_ik", False)),
                motion_profile  = str(msg.get("motion_profile", "default")),
                on_complete = lambda ok, rid=rid: self._publish_event(
                    "move_done", rid, ok=bool(ok)))

        elif cmd == "move_to_joints":
            self.move_to_joints(
                msg["joints"],
                on_complete=lambda ok, rid=rid: self._publish_event(
                    "move_done", rid, ok=bool(ok)))

        elif cmd == "update_move_target":
            self.update_move_target(msg["pos"], msg.get("quat"))

        elif cmd == "cancel_move":
            self.cancel_move()

        elif cmd == "cancel":
            self.cancel_motion()
        elif cmd == "cancel_grasp":
            self.cancel_grasp()

        elif cmd == "wait_for_handover_pull":
            self.start_object_handover_pull(rid)

        elif cmd == "start_board_interaction":
            self.start_board_interaction(freedrive=bool(msg.get("freedrive", True)))

        elif cmd == "cancel_board_interaction":
            self.cancel_board_interaction()

        elif cmd == "set_board_freedrive":
            self.set_board_freedrive_policy(bool(msg.get("enabled", True)))

        elif cmd == "arm_board_release":
            self.arm_board_release()

        elif cmd == "simulate_board_pull":
            self.simulate_board_pull()

        else:
            print(f"[RobotServer] Unknown command: {cmd!r}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        target_dt      = 1.0 / self.control_hz
        last_tick_time = None
        last_state_pub = 0.0
        try:
            while self._running:
                iter_t0 = time.perf_counter()
                dt = (min(iter_t0 - last_tick_time, 0.1)
                      if last_tick_time is not None else target_dt)
                last_tick_time = iter_t0

                self._poll_commands()

                if not self.simulation:
                    q_fresh = self.poll_q()
                    if q_fresh is not None:
                        self.pb_scene.update_robot(q_fresh)

                self.tick(dt)

                now = time.time()
                if now - last_state_pub >= self._STATE_INTERVAL:
                    self._publish_state()
                    last_state_pub = now

                elapsed    = time.perf_counter() - iter_t0
                sleep_left = target_dt - elapsed
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
            self.pb_scene.disconnect()
            print("[RobotServer] Shutting down.")

    # ── Grasp state (property + tick) ─────────────────────────────────────────

    @property
    def tool_grasp_running(self) -> bool:
        return self._grasp_phase is not None

    def tick(self, dt: float = 1.0 / 60.0) -> None:
        if self._startup_active:
            self._tick_startup_pose(dt)
            return
        if self._joint_move_target is not None:
            self._tick_joint_move(dt)
            return
        if self._move_phase is not None:
            self._tick_move_to_pose(dt)
            return
        if self._board_force_mode is not None:
            self._tick_board_interaction()
        if self._handover_pull_active:
            self._tick_object_handover_pull()
        if not self.simulation:
            if self._grasp_phase is not None:
                self._tick_real_grasp(dt)
            return
        if self._grasp_runner is None:
            return
        if not self._grasp_runner.done:
            self._grasp_runner.update(self.pb_scene.robot_id,
                                    self.pb_scene.arm_indices, dt)
            return
        cq = self.pb_scene.current_q.copy()
        if self._grasp_phase == 'approach':
            # approach → grasp (same for tools and parts)
            self._grasp_runner = _PbJointRunner(
                cq, self._grasp_joints, self._grasp_sim_duration_s)
            self._grasp_phase  = 'grasp'
            print("[Robot sim] At approach → moveJ to grasp")
            self._fire_phase(self._grasp_on_phase, "grasping")
        elif self._grasp_phase == 'grasp':
            self._fire_phase(self._grasp_on_phase, "grasped")
            if self._grasp_q_above is not None:
                # parts: lift 5 cm above grasp
                self._grasp_runner = _PbJointRunner(
                    cq, self._grasp_q_above, self._grasp_sim_duration_s)
                self._grasp_phase  = 'lift'
                print("[Robot sim] Grasp done → lifting 5 cm")
                self._fire_phase(self._grasp_on_phase, "retracting")
            elif self._grasp_q_approach is not None:
                # tools: retract straight to approach
                self._grasp_runner = _PbJointRunner(
                    cq, self._grasp_q_approach, self._grasp_sim_duration_s)
                self._grasp_phase  = 'retract'
                print("[Robot sim] Grasp done → retracting to approach")
                self._fire_phase(self._grasp_on_phase, "retracting")
            else:
                self._finish_grasp(True)
        elif self._grasp_phase == 'lift':
            if self._grasp_q_above_approach is not None:
                # parts: retract to 5 cm above approach, stay there
                self._grasp_runner = _PbJointRunner(
                    cq, self._grasp_q_above_approach,
                    self._grasp_sim_duration_s)
                self._grasp_phase  = 'ret_above_approach'
                print("[Robot sim] Lifted → retracting to above_approach")
            elif self._grasp_q_approach is not None:
                self._grasp_runner = _PbJointRunner(
                    cq, self._grasp_q_approach, self._grasp_sim_duration_s)
                self._grasp_phase  = 'retract'
                print("[Robot sim] Lifted → retracting to approach")
            else:
                self._finish_grasp(True)
        else:
            # ret_above_approach or retract → done
            self._finish_grasp(True)

    def move_to_joints(self, joints, on_complete=None) -> None:
        """Start a guarded direct joint-space move; input is radians."""
        busy = (self._startup_active or self._move_phase is not None
                or self._grasp_phase is not None or self._handover_pull_active
                or self._board_force_mode is not None or self._board_state != "inactive"
                or self._joint_move_target is not None)
        if busy:
            print("[Robot] Gearbox joint move blocked — robot is busy")
            if on_complete:
                on_complete(False)
            return
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            print("[Robot] Gearbox joint move rejected — expected six finite joints")
            if on_complete:
                on_complete(False)
            return
        q_min = np.deg2rad(np.asarray(cfg.JOINT_MIN_DEG, dtype=float))
        q_max = np.deg2rad(np.asarray(cfg.JOINT_MAX_DEG, dtype=float))
        if np.any(q < q_min) or np.any(q > q_max):
            print(f"[Robot] Gearbox joint move rejected by limits: "
                  f"{np.round(np.rad2deg(q), 2).tolist()}")
            if on_complete:
                on_complete(False)
            return
        self._joint_move_target = _wrap_nearest(q, self.pb_scene.current_q.copy())
        self._joint_move_on_complete = on_complete
        self._joint_move_sent = False
        self._joint_move_started = time.monotonic()
        if self.simulation:
            self._joint_move_runner = _PbJointRunner(
                self.pb_scene.current_q.copy(), self._joint_move_target)
        print(f"[Robot] Gearbox moveJ → "
              f"{np.round(np.rad2deg(self._joint_move_target), 2).tolist()}")

    def _finish_joint_move(self, ok: bool) -> None:
        cb = self._joint_move_on_complete
        self._joint_move_target = None
        self._joint_move_runner = None
        self._joint_move_sent = False
        self._joint_move_on_complete = None
        if cb:
            cb(bool(ok))

    def _tick_joint_move(self, dt: float) -> None:
        target = self._joint_move_target
        if target is None:
            return
        if self.simulation:
            if not self._joint_move_runner.done:
                self._joint_move_runner.update(
                    self.pb_scene.robot_id, self.pb_scene.arm_indices, dt)
                return
            self.pb_scene.update_robot(target)
            print("[Robot sim] Gearbox joint pose reached")
            self._finish_joint_move(True)
            return
        try:
            ctrl = self._rtde_ctrl_conn()
            err = np.max(np.abs(_wrap_nearest(target, self.pb_scene.current_q)
                                - self.pb_scene.current_q))
            if not self._joint_move_sent:
                speed = 0.5 * self._speed_scale
                ctrl.moveJ(list(target), speed, speed, asynchronous=True)
                self._joint_move_sent = True
            elif err < np.deg2rad(1.0):
                print("[Robot] Gearbox joint pose reached")
                self._finish_joint_move(True)
            elif time.monotonic() - self._joint_move_started > 30.0:
                ctrl.stopJ(2.0)
                print("[Robot] Gearbox joint move timed out")
                self._finish_joint_move(False)
        except Exception as e:
            print(f"[Robot] Gearbox joint move failed: {e}")
            self._finish_joint_move(False)

    def _tick_startup_pose(self, dt: float) -> None:
        """Move both simulated and real robots to the shared default pose."""
        if self.simulation:
            runner = self._startup_runner
            if runner is None:
                self._startup_active = False
                return
            if not runner.done:
                runner.update(self.pb_scene.robot_id,
                              self.pb_scene.arm_indices, dt)
                return
            self.pb_scene.update_robot(self._default_q)
            self._startup_runner = None
            self._startup_active = False
            print("[Robot sim] Startup default pose reached")
            return

        q_cur = self.pb_scene.current_q.copy()
        target = _wrap_nearest(self._default_q, q_cur)
        joint_err = np.abs(_wrap_nearest(target, q_cur) - q_cur)
        if np.max(joint_err) < np.deg2rad(1.0):
            self._startup_active = False
            print("[Robot] Startup default pose reached")
            return

        try:
            ctrl = self._rtde_ctrl_conn()
            if not self._startup_move_sent:
                speed = 0.5 * self._speed_scale
                ctrl.moveJ(list(target), speed, speed, asynchronous=True)
                self._startup_move_sent = True
                print(f"[Robot] Startup moveJ → default pose "
                      f"{np.round(np.rad2deg(target), 2).tolist()}")
            elif ctrl.isSteady() and np.max(joint_err) < np.deg2rad(2.0):
                self._startup_active = False
                print("[Robot] Startup default pose reached")
        except Exception as e:
            print(f"[Robot] Startup default-pose move failed: {e}")
            self._startup_active = False
            self._startup_move_sent = False
            self._rtde_ctrl = None

    def _finish_grasp(self, success: bool) -> None:
        cb = self._grasp_on_complete
        self._grasp_runner            = None
        self._grasp_phase             = None
        self._grasp_on_complete       = None
        self._grasp_joints            = None
        self._grasp_q_approach        = None
        self._grasp_q_above           = None
        self._grasp_q_above_approach  = None
        if cb is not None:
            try:
                cb(success)
            except Exception:
                pass

    # ── Move-to-pose (IK + CBF servo loop) ───────────────────────────────────

    # Board force handoff is intentionally independent of tool/part grasps.
    def _set_board_state(self, state: str) -> None:
        if state != self._board_state:
            print(f"[Robot] board state: {self._board_state} → {state}")
        self._board_state = state

    def _arm_board_force(self, mode: "str | None") -> None:
        self._board_force_mode = mode
        self._board_force_baseline = None
        self._board_force_hits = 0
        self._board_force_last_t = 0.0
        if mode is not None:
            threshold = (cfg.BOARD_GRASP_FORCE_THRESHOLD_N
                         if mode == "grasp"
                         else cfg.BOARD_RELEASE_FORCE_THRESHOLD_N)
            print(f"[Robot] Board force monitor armed: {mode}, "
                  f"threshold={threshold:.1f} N")

    def start_board_interaction(self, freedrive: bool = True) -> None:
        """Open for a board and arm force-triggered grasp detection.

        `freedrive` sets the policy freedrive returns to whenever the board is
        held (here, and after each completed board_move) — off for an
        AR-only study condition, on otherwise."""
        if self._startup_active:
            print("[Robot] Board interaction blocked — startup move is active")
            return
        if self._grasp_phase is not None or self._move_phase is not None:
            print("[Robot] Board interaction blocked — robot motion is active")
            return
        self._board_freedrive_policy = bool(freedrive)
        if self.simulation:
            # Simulation has no force sensor or gripper.  Never carry a stale
            # real-robot force mode into its virtual held-board state.
            self._arm_board_force(None)
            self._set_board_state("holding_board")
            self._set_board_freedrive(self._board_freedrive_policy)
            return
        try:
            self.servoStop()
            self.open_gripper()
        except Exception as e:
            print(f"[Robot] Board gripper open failed: {e}")
            self._set_board_state("inactive")
            self._arm_board_force(None)
            return
        self._set_board_state("waiting_for_board")
        self._arm_board_force("grasp")

    def _set_board_freedrive(self, enabled: bool) -> bool:
        """Enter/leave freedrive without changing the board latch state."""
        enabled = bool(enabled)
        if enabled == self._board_freedrive_active:
            return True
        if self.simulation:
            self._board_freedrive_active = enabled
            print(f"[Robot sim] Board freedrive {'enabled' if enabled else 'disabled'}")
            return True
        try:
            ctrl = self._rtde_ctrl_conn()
            if enabled:
                self.servoStop()
                ctrl.freedriveMode()
            else:
                ctrl.endFreedriveMode()
            self._board_freedrive_active = enabled
            print(f"[Robot] Board freedrive {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            print(f"[Robot] Board freedrive transition failed: {e}")
            return False

    def arm_board_release(self) -> None:
        """Toggle a held board between locked and pull-to-release modes."""
        if self._board_state == "release_armed":
            # A second click means the first was accidental: cancel release,
            # restore the latch indication, and return to freedrive.
            self._arm_board_force(None)
            self._set_board_state("holding_board")
            self._set_board_freedrive(self._board_freedrive_policy)
            print("[Robot] Board release cancelled → locked again")
            return
        if self._board_state != "holding_board":
            print(f"[Robot] Board release ignored — state is '{self._board_state}'")
            return
        if not self._set_board_freedrive(False):
            return
        self._set_board_state("release_armed")
        self._arm_board_force(None if self.simulation else "release")
        if self.simulation:
            print("[Robot sim] Board release armed; press P to simulate pull")

    def simulate_board_pull(self) -> None:
        """Testing hook: complete a board pull only in simulation mode."""
        if not self.simulation:
            print("[Robot] simulate_board_pull ignored on real hardware")
            return
        if self._board_state != "release_armed":
            print(f"[Robot sim] Virtual pull ignored — state is '{self._board_state}'")
            return
        self._set_board_freedrive(False)
        self._arm_board_force(None)
        self._set_board_state("inactive")
        print("[Robot sim] Virtual board pull → release complete")

    def set_board_freedrive_policy(self, enabled: bool) -> None:
        """Flip whether a held board keeps freedrive on, without releasing or
        re-grasping it. Used to switch interaction-study conditions (e.g.
        AR-only ↔ freedrive-involved) mid-session."""
        self._board_freedrive_policy = bool(enabled)
        if self._board_state == "holding_board":
            self._set_board_freedrive(self._board_freedrive_policy)

    def cancel_board_interaction(self) -> None:
        """Stop board monitoring without automatically dropping a held board."""
        self._set_board_freedrive(False)
        self._arm_board_force(None)
        if self.simulation:
            # Simulation has no physical gripper/object to protect.  Its
            # holding_board state is only a virtual state used to expose the AR
            # handle, so cancellation can always return it to inactive.
            self._set_board_state("inactive")
            return
        if self._board_state not in ("holding_board", "moving_board", "release_armed"):
            self._set_board_state("inactive")

    def _prepare_board_state_for_motion(self, board_move: bool = False) -> None:
        """Release board states that do not represent a physical held object."""
        if (self.simulation and not board_move
                and self._board_state == "holding_board"):
            print("[Robot sim] New motion requested → dismissing virtual board hold")
            self._set_board_freedrive(False)
            self._arm_board_force(None)
            self._set_board_state("inactive")
            return
        if (self._board_state in ("inactive", "waiting_for_board")
                and self._board_force_mode == "grasp"):
            print("[Robot] New motion requested → cancelling board-contact wait")
            self._arm_board_force(None)
            self._set_board_state("inactive")

    def _poll_tcp_force(self) -> "np.ndarray | None":
        if self.simulation:
            return None
        try:
            return np.asarray(self._recv_conn().getActualTCPForce()[:3], float)
        except Exception as e:
            print(f"[Robot] TCP force read failed: {e}")
            return None

    def _tick_board_interaction(self) -> None:
        """Detect board insertion/pull as force change from a fresh baseline."""
        if self.simulation:
            return
        now = time.perf_counter()
        if (self._board_force_last_t > 0.0
                and now - self._board_force_last_t
                    < 1.0 / cfg.BOARD_FORCE_POLL_HZ):
            return
        self._board_force_last_t = now
        force = self._poll_tcp_force()
        if force is None:
            return
        if self._board_force_baseline is None:
            self._board_force_baseline = force
            print(f"[Robot] Board force baseline captured "
                  f"({np.round(force, 1).tolist()} N)")
            return

        delta = float(np.linalg.norm(force - self._board_force_baseline))
        mode = self._board_force_mode
        if mode is None:
            return
        threshold = (cfg.BOARD_GRASP_FORCE_THRESHOLD_N
                     if mode == "grasp"
                     else cfg.BOARD_RELEASE_FORCE_THRESHOLD_N)
        self._board_force_hits = (self._board_force_hits + 1
                                  if delta > threshold else 0)
        if self._board_force_hits < cfg.BOARD_FORCE_DEBOUNCE_HITS:
            return

        if mode == "grasp":
            print(f"[Robot] Board contact detected ({delta:.1f} N) → closing")
            try:
                has_object = self.close_gripper()
            except Exception as e:
                print(f"[Robot] Board gripper close failed: {e}")
                self._set_board_state("inactive")
                self._arm_board_force(None)
                return
            if has_object:
                self._set_board_state("holding_board")
                self._arm_board_force(None)
                self._set_board_freedrive(self._board_freedrive_policy)
            else:
                print("[Robot] Board not detected by gripper → reopening")
                try:
                    self.open_gripper()
                except Exception as e:
                    print(f"[Robot] Board gripper reopen failed: {e}")
                    self._arm_board_force(None)
                    return
                self._set_board_state("waiting_for_board")
                self._arm_board_force("grasp")
        elif mode == "release" and self._board_state == "release_armed":
            print(f"[Robot] Board pull detected ({delta:.1f} N) → opening")
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] Board gripper release failed: {e}")
                self._arm_board_force(None)
                return
            self._set_board_state("inactive")
            self._arm_board_force(None)

    def start_object_handover_pull(self, request_id) -> None:
        """Hold a grasped object until a sustained human pull is detected."""
        if self._startup_active or self._grasp_phase is not None or self._move_phase is not None:
            print("[Robot] Object handover pull monitor blocked — robot is moving")
            self._publish_event("release_done", request_id, ok=False)
            return
        if self._board_force_mode is not None or self._board_state != "inactive":
            print("[Robot] Object handover pull monitor blocked — board interaction active")
            self._publish_event("release_done", request_id, ok=False)
            return
        if self.simulation:
            # PyBullet has no wrist-force or gripper simulation. Treat the
            # virtual human pull as immediate so the full lifecycle remains testable.
            print("[Robot sim] Virtual handover pull → release acknowledged")
            self._publish_event("release_done", request_id, ok=True)
            return
        self.servoStop()
        self._handover_pull_active = True
        self._handover_force_baseline = None
        self._handover_force_hits = 0
        self._handover_force_last_t = 0.0
        self._handover_request_id = request_id
        print(f"[Robot] Waiting for human pull on object "
              f"(threshold={cfg.OBJECT_HANDOVER_PULL_THRESHOLD_N:.1f} N)")

    def _finish_object_handover_pull(self, ok: bool) -> None:
        request_id = self._handover_request_id
        self._handover_pull_active = False
        self._handover_force_baseline = None
        self._handover_force_hits = 0
        self._handover_force_last_t = 0.0
        self._handover_request_id = None
        self._publish_event("release_done", request_id, ok=bool(ok))

    def _tick_object_handover_pull(self) -> None:
        """Open the gripper only after a debounced force change from the baseline."""
        if self.simulation or not self._handover_pull_active:
            return
        now = time.perf_counter()
        if (self._handover_force_last_t > 0.0
                and now - self._handover_force_last_t
                    < 1.0 / cfg.OBJECT_HANDOVER_FORCE_POLL_HZ):
            return
        self._handover_force_last_t = now
        force = self._poll_tcp_force()
        if force is None:
            return
        if self._handover_force_baseline is None:
            self._handover_force_baseline = force
            print(f"[Robot] Object handover force baseline captured "
                  f"({np.round(force, 1).tolist()} N)")
            return
        delta = float(np.linalg.norm(force - self._handover_force_baseline))
        self._handover_force_hits = (
            self._handover_force_hits + 1
            if delta > cfg.OBJECT_HANDOVER_PULL_THRESHOLD_N else 0)
        if self._handover_force_hits < cfg.OBJECT_HANDOVER_DEBOUNCE_HITS:
            return
        print(f"[Robot] Human pull detected ({delta:.1f} N) → opening gripper")
        try:
            self.open_gripper()
        except Exception as e:
            print(f"[Robot] Object handover gripper release failed: {e}")
            self._finish_object_handover_pull(False)
            return
        self._finish_object_handover_pull(True)

    def move_to_pose(self, pos, quat=None, board_move: bool = False,
                     force_pybullet_ik: bool = False,
                     motion_profile: str = "default",
                     on_complete: "Callable[[bool], None] | None" = None) -> None:
        """Stream IK+CBF servoJ toward a fixed Cartesian target until convergence."""
        if self._startup_active:
            print("[Robot] move_to_pose blocked — startup move is active")
            if on_complete:
                on_complete(False)
            return
        if self._joint_move_target is not None:
            print("[Robot] move_to_pose blocked — gearbox joint move is active")
            if on_complete:
                on_complete(False)
            return
        if self._grasp_phase is not None:
            print("[Robot] move_to_pose blocked — grasp in progress")
            if on_complete:
                on_complete(False)
            return
        if self._handover_pull_active:
            print("[Robot] move_to_pose blocked — waiting for object handover pull")
            if on_complete:
                on_complete(False)
            return
        self._prepare_board_state_for_motion(board_move=board_move)
        if (self._board_state != "inactive"
                or self._board_force_mode is not None):
            valid_board_move = (board_move
                                and self._board_state == "holding_board")
            if not valid_board_move:
                print(f"[Robot] move_to_pose blocked — board state "
                      f"is '{self._board_state}', force mode "
                      f"is {self._board_force_mode!r}")
                if on_complete:
                    on_complete(False)
                return
        if board_move and not self._set_board_freedrive(False):
            print("[Robot] Board move blocked — could not disable freedrive")
            if on_complete:
                on_complete(False)
            return
        if not self.simulation:
            self.servoStop()
        T_tcp = self.pb_scene.update_tcp_bodies()
        origin = T_tcp[:3, 3] if T_tcp is not None else None
        self._move_target_pos = np.asarray(
            cfg.project_robot_target_position(pos, origin), float)
        self._move_target_quat = np.asarray(quat, float) if quat is not None else None
        self._move_on_complete = on_complete
        self._move_smooth_q    = None
        self._move_pybullet_q_target = None
        self._move_conv_start  = None
        self._move_diag_t      = 0.0
        self._move_force_pybullet_ik = force_pybullet_ik
        self._move_is_board     = bool(board_move)
        self._move_motion_profile = (
            motion_profile
            if motion_profile in ("workholding", "handover")
            else "default")
        self._reset_move_progress()
        self._move_phase       = 'moving_to_pose'
        if board_move:
            # Acceleration during motion must not look like a release pull.
            self._arm_board_force(None)
            self._set_board_state("moving_board")
        print(f"[Robot] move_to_pose start "
              f"(profile={self._move_motion_profile}) → "
              f"{np.round(self._move_target_pos, 3).tolist()}")

    def update_move_target(self, pos, quat=None) -> None:
        """Update the Cartesian target of an in-progress move without restarting the controller."""
        if self._move_phase is None:
            return
        T_tcp = self.pb_scene.update_tcp_bodies()
        origin = T_tcp[:3, 3] if T_tcp is not None else None
        self._move_target_pos = np.asarray(
            cfg.project_robot_target_position(pos, origin), float)
        if quat is not None:
            self._move_target_quat = np.asarray(quat, float)
        # The PyBullet controller uses one stable joint-space solution per
        # Cartesian target. A P/N target change requires a fresh solution.
        self._move_pybullet_q_target = None
        # A replacement target gets its own progress window.  This prevents an
        # old unreachable AR pose from timing out a newly released pose.
        self._reset_move_progress()

    def _reset_move_progress(self) -> None:
        self._move_best_dist = float('inf')
        self._move_best_angle = float('inf')
        self._move_progress_t = time.perf_counter()

    def cancel_move(self) -> None:
        if self._move_phase is None:
            return
        print("[Robot] move_to_pose cancelled")
        self._finish_move(False)

    def _finish_move(self, ok: bool) -> None:
        self._move_phase       = None
        self._move_target_pos  = None
        self._move_target_quat = None
        self._move_smooth_q    = None
        self._move_pybullet_q_target = None
        self._move_conv_start  = None
        self._move_is_board    = False
        self._move_motion_profile = "default"
        self._reset_move_progress()
        if not self.simulation:
            self.servoStop()
        if self._board_state == "moving_board":
            # The board remains latched. Return to freedrive without enabling
            # pull-to-release; that requires another explicit TCPMarker click.
            self._set_board_state("holding_board")
            self._arm_board_force(None)
            self._set_board_freedrive(self._board_freedrive_policy)
        cb = self._move_on_complete
        self._move_on_complete = None
        if cb is not None:
            try:
                cb(ok)
            except Exception as e:
                print(f"[Robot] move_on_complete error: {e}")

    def _tick_move_to_pose(self, dt: float) -> None:
        target_pos  = self._move_target_pos
        target_quat = self._move_target_quat
        if target_pos is None:
            self._move_phase = None
            return

        # Use the configured hardware period for integration and servoJ.  The
        # measured outer-loop dt can include servoJ blocking time and scheduling
        # jitter, which otherwise feeds back into the next command.
        control_dt = dt if self.simulation else 1.0 / self.control_hz
        if self._move_motion_profile == "workholding":
            pos_tol = self._WORKHOLDING_MOVE_POS_TOL_M
            angle_tol = self._WORKHOLDING_MOVE_ANGLE_TOL_RAD
        elif self._move_motion_profile == "handover":
            pos_tol = self._HANDOVER_MOVE_POS_TOL_M
            angle_tol = self._HANDOVER_MOVE_ANGLE_TOL_RAD
        else:
            pos_tol = (self._BOARD_MOVE_POS_TOL_M
                       if self._move_is_board else self._MOVE_POS_TOL_M)
            angle_tol = (self._BOARD_MOVE_ANGLE_TOL_RAD
                         if self._move_is_board else self._MOVE_ANGLE_TOL_RAD)

        q_cur = self.pb_scene.current_q.copy()

        _GRN = "\033[92m"; _RED = "\033[91m"; _RST = "\033[0m"

        # One-shot: compare frax FK vs PyBullet FK to detect frame mismatch
        if not getattr(self, '_fk_check_done', False) and self._frax is not None:
            self._fk_check_done = True
            _pb_tcp  = self.pb_scene.update_tcp_bodies()
            _fr_tcp  = self._frax.ee_world_pos(q_cur)
            if _pb_tcp is not None:
                print(f"[Robot] FK check  PyBullet TCP: {np.round(_pb_tcp[:3, 3], 3).tolist()}")
                print(f"[Robot] FK check    frax TCP:   {np.round(_fr_tcp, 3).tolist()}")
                print(f"[Robot] FK check    target:     {np.round(target_pos, 3).tolist()}")

        # Convergence requires both translation and rotation to remain aligned
        # for the dwell period.  Position-only completion can stop servoing
        # before the wrist reaches its requested orientation.
        T_tcp = self.pb_scene.update_tcp_bodies()
        dist = float('inf')
        angle_err = float('inf')
        if T_tcp is not None:
            dist = float(np.linalg.norm(T_tcp[:3, 3] - target_pos))
            R_err = T_tcp[:3, :3].T @ ScipyR.from_quat(target_quat).as_matrix()
            angle_err = float(ScipyR.from_matrix(R_err).magnitude())
            _now = time.perf_counter()

            # Fixed unreachable poses used to servo forever.  Treat meaningful
            # improvement in either pose component as progress, and report a
            # failed move after both have stalled outside the stop tolerance.
            _made_progress = (
                dist < self._move_best_dist - self._MOVE_PROGRESS_POS_M
                or angle_err < (self._move_best_angle
                                - self._MOVE_PROGRESS_ANGLE_RAD))
            if _made_progress:
                # Retain the last *meaningful* milestone so many small control
                # steps accumulate into progress instead of being discarded.
                self._move_best_dist = min(self._move_best_dist, dist)
                self._move_best_angle = min(self._move_best_angle, angle_err)
                self._move_progress_t = _now
            elif ((dist >= pos_tol
                   or angle_err >= angle_tol)
                  and _now - self._move_progress_t
                  >= self._MOVE_STALL_TIMEOUT_S):
                print(f"[Robot] move_to_pose unreachable/no progress for "
                      f"{self._MOVE_STALL_TIMEOUT_S:.1f} s "
                      f"({dist*100:.1f} cm, "
                      f"{np.rad2deg(angle_err):.1f}°) → cancelled")
                self._finish_move(False)
                return

            # Throttled colored status print at ~2 Hz
            if _now - self._move_diag_t >= 0.5:
                self._move_diag_t = _now
                _dc = _GRN if dist      < pos_tol   else _RED
                _ac = _GRN if angle_err < angle_tol else _RED
                print(f"[Robot] dist: {_dc}{dist*100:5.1f} cm{_RST}  "
                      f"angle: {_ac}{np.rad2deg(angle_err):5.1f}°{_RST}")

            if (dist < pos_tol and angle_err < angle_tol):
                if self._move_conv_start is None:
                    self._move_conv_start = time.perf_counter()
                elif (time.perf_counter() - self._move_conv_start
                      >= self._MOVE_DWELL_S):
                    print(f"[Robot] move_to_pose converged "
                          f"({dist*100:.1f} cm < {pos_tol*100:.1f} cm, "
                          f"{np.rad2deg(angle_err):.1f}° < "
                          f"{np.rad2deg(angle_tol):.1f}°, "
                          f"{self._MOVE_DWELL_S:.1f} s dwell) → idle")
                    self._finish_move(True)
                    return
            else:
                self._move_conv_start = None

        # Velocity command
        _use_pybullet = (getattr(self, '_move_force_pybullet_ik', False)
                         or not self.use_diff_ik)
        if not _use_pybullet and self._frax is not None:
            # OSC + CBF: Cartesian-space differential IK — deterministic, no PyBullet IK needed
            target_rot = ScipyR.from_quat(target_quat).as_matrix()
            try:
                q_target = self._frax.servo_step(
                    target_pos, target_rot, q_cur, control_dt,
                    speed_scale=(
                        self._WORKHOLDING_OSC_SPEED_SCALE
                        if self._move_motion_profile == "workholding" else
                        self._HANDOVER_OSC_SPEED_SCALE
                        if self._move_motion_profile == "handover" else 1.0))
            except Exception as _e:
                print(f"[Robot] servo_step error: {_e}")
                q_target = q_cur
            # Brake only when both pose components are close.  Taking the
            # larger scale preserves wrist motion when translation has already
            # converged but a significant orientation error remains.
            if T_tcp is not None:
                if self._move_motion_profile == "workholding":
                    # Smoothstep gives zero slope where braking begins and a
                    # gentler final approach than the previous linear scale.
                    # Full OSC speed is retained outside this short zone.
                    pos_x = min(dist / self._WORKHOLDING_BRAKE_DIST_M, 1.0)
                    rot_x = min(
                        angle_err / self._WORKHOLDING_BRAKE_ANGLE_RAD, 1.0)
                    pos_scale = pos_x * pos_x * (3.0 - 2.0 * pos_x)
                    rot_scale = rot_x * rot_x * (3.0 - 2.0 * rot_x)
                elif self._move_motion_profile == "handover":
                    pos_x = min(dist / self._HANDOVER_BRAKE_DIST_M, 1.0)
                    rot_x = min(
                        angle_err / self._HANDOVER_BRAKE_ANGLE_RAD, 1.0)
                    pos_scale = pos_x * pos_x * (3.0 - 2.0 * pos_x)
                    rot_scale = rot_x * rot_x * (3.0 - 2.0 * rot_x)
                else:
                    pos_scale = min(dist / self._MOVE_BRAKE_DIST_M, 1.0)
                    rot_scale = min(angle_err / self._MOVE_BRAKE_ANGLE_RAD, 1.0)
                motion_scale = max(self._MOVE_SCALE_FLOOR,
                                   pos_scale, rot_scale)
            else:
                motion_scale = 1.0
            q_target = q_cur + (q_target - q_cur) * motion_scale
        else:
            # PyBullet IK + rate-limit (fallback or --no-diff-ik)
            # Solve once for a fixed pose. Re-solving from the changing current
            # configuration can jump between redundant UR joint branches and
            # makes the CBF chase a moving joint-space target.
            if self._move_pybullet_q_target is None:
                ik_pos_tol = (self._BOARD_MOVE_POS_TOL_M
                              if self._move_motion_profile == "workholding"
                              else pos_tol)
                ik_angle_tol = (self._BOARD_MOVE_ANGLE_TOL_RAD
                                if self._move_motion_profile == "workholding"
                                else angle_tol)
                q_ik = self.pb_scene.solve_ik(
                    q_cur, target_pos, target_quat,
                    pos_tol=ik_pos_tol, orient_tol=ik_angle_tol)
                q_ik = _wrap_nearest(q_ik, q_cur)

                # solve_ik places the headless PyBullet robot at the candidate
                # while checking FK; restore the actual configuration before
                # beginning the rate-limited motion.
                self.pb_scene.update_robot(q_cur)
                self._move_pybullet_q_target = q_ik
                print("[Robot] PyBullet IK target fixed for this move")
            q_ik = self._move_pybullet_q_target
            sc        = self._speed_scale
            if T_tcp is not None:
                pos_scale = min(dist / self._MOVE_BRAKE_DIST_M, 1.0)
                rot_scale = min(angle_err / self._MOVE_BRAKE_ANGLE_RAD, 1.0)
                motion_scale = max(self._MOVE_SCALE_FLOOR,
                                   pos_scale, rot_scale)
            else:
                motion_scale = 1.0
            max_step = (np.deg2rad(90.0 if self.simulation else 180.0)
                        * (1.0 if self.simulation else sc)
                        * control_dt * motion_scale)
            q_target  = q_cur + np.clip(q_ik - q_cur, -max_step, max_step)
            if self._frax is not None:
                # Still apply CBF safety filter even in IK mode
                qdot = (q_target - q_cur) / max(control_dt, 1e-6)
                try:
                    qdot_safe = np.asarray(
                        self._frax._cbf.safety_filter(
                            _jnp.array(q_cur), _jnp.array(qdot)))
                    q_target = q_cur + qdot_safe * control_dt
                except Exception:
                    pass
            if np.linalg.norm(q_target - q_cur) < 1e-4:
                print("[Robot] move_to_pose IK stalled → cancelled")
                self._finish_move(False)
                return

        if np.any(np.isnan(q_target)):
            return

        if self.simulation:
            if self._move_smooth_q is None:
                self._move_smooth_q = q_target.copy()
            else:
                alpha = (self._SIM_BOARD_SMOOTH_ALPHA
                         if self._move_is_board else 0.3)
                self._move_smooth_q = (alpha * q_target
                                       + (1.0 - alpha) * self._move_smooth_q)
            self.pb_scene.update_robot(self._move_smooth_q)
        else:
            # EMA smoothing to damp servoJ jitter
            workholding_profile = self._move_motion_profile == "workholding"
            handover_profile = self._move_motion_profile == "handover"
            if self._move_smooth_q is None:
                self._move_smooth_q = q_target.copy()
            else:
                if handover_profile:
                    proximity = max(0.0, min(
                        1.0, 1.0 - dist / self._HANDOVER_BRAKE_DIST_M))
                    alpha = (self._HANDOVER_EMA_FAR
                             + proximity * (self._HANDOVER_EMA_NEAR
                                            - self._HANDOVER_EMA_FAR))
                else:
                    alpha = (self._WORKHOLDING_EMA_ALPHA
                             if workholding_profile else
                             (self._BOARD_SERVO_EMA_ALPHA
                              if self._move_is_board else self._SERVO_EMA_ALPHA))
                self._move_smooth_q = (alpha * q_target
                                       + (1.0 - alpha) * self._move_smooth_q)
            if handover_profile:
                proximity = max(0.0, min(
                    1.0, 1.0 - dist / self._HANDOVER_BRAKE_DIST_M))
                lookahead = (self._HANDOVER_LOOKAHEAD_FAR_S
                             + proximity * (self._HANDOVER_LOOKAHEAD_NEAR_S
                                            - self._HANDOVER_LOOKAHEAD_FAR_S))
                gain = (self._HANDOVER_GAIN_FAR
                        + proximity * (self._HANDOVER_GAIN_NEAR
                                       - self._HANDOVER_GAIN_FAR))
            else:
                lookahead = (self._WORKHOLDING_LOOKAHEAD_S
                             if workholding_profile else self._SERVO_LOOKAHEAD_S)
                gain = (self._WORKHOLDING_GAIN
                        if workholding_profile else self._SERVO_GAIN)
            self.servoJ(self._move_smooth_q, control_dt,
                        lookahead=lookahead, gain=gain)

    # ── Grasp command ─────────────────────────────────────────────────────────

    def execute_grasp(
        self,
        grasp_joints: "list | np.ndarray",
        on_complete:  "Callable[[bool], None] | None" = None,
        on_phase:     "Callable[[str], None] | None"  = None,
        q_approach:   "list | np.ndarray | None"      = None,
        q_above:      "list | np.ndarray | None"      = None,
        category:     str                              = "tool",
        board_normal: "list | np.ndarray | None"      = None,
        tool_type:    str                              = "",
    ) -> None:
        if self._startup_active:
            print("[Robot] Grasp blocked — startup move is active")
            if on_complete:
                on_complete(False)
            return
        if self._joint_move_target is not None:
            print("[Robot] Grasp blocked — gearbox joint move is active")
            if on_complete:
                on_complete(False)
            return
        if self._handover_pull_active:
            print("[Robot] Grasp blocked — waiting for object handover pull")
            if on_complete:
                on_complete(False)
            return
        self._prepare_board_state_for_motion()
        if (self._board_state != "inactive"
                or self._board_force_mode is not None):
            print(f"[Robot] Tool/part grasp blocked — board state "
                  f"is '{self._board_state}'")
            if on_complete:
                on_complete(False)
            return
        if self.tool_grasp_running:
            print("[Robot] Grasp already running — cancel first.")
            return

        # Auto-compute waypoints from board_normal when not pre-supplied.
        # Sequence (parts):  approach → grasp → lift → above_approach → done
        # Sequence (tools):  approach → grasp → retract to approach → done
        # Approach is at grasp height (board normal projected to XY so no Z drift).
        q_above_approach = None
        if board_normal is not None and q_approach is None and self.pb_scene is not None:
            gj_arr  = np.array(grasp_joints, dtype=float)
            is_part = (category != "tool")
            saved_q = self.pb_scene.current_q.copy()
            if tool_type.startswith("GEAR_ROD"):
                bn = np.array([1.0, 0.0, 0.0])        # gear rods: approach along world +X
            else:
                bn = np.array(board_normal, float)
                bn[2] = 0.0                            # project to XY — keeps approach at grasp height
                bn /= (np.linalg.norm(bn) + 1e-9)
            try:
                self.pb_scene.update_robot(gj_arr)
                grasp_T = self.pb_scene.update_tcp_bodies()
            finally:
                self.pb_scene.update_robot(saved_q)
            if grasp_T is not None:
                tcp_pos  = grasp_T[:3, 3]
                tcp_quat = ScipyR.from_matrix(grasp_T[:3, :3]).as_quat().tolist()
                pos_approach = tcp_pos + self._approach_dist * bn
                q_approach = _wrap_nearest(
                    self.solve_ik_from_config(gj_arr, pos_approach, tcp_quat), saved_q)
                if is_part:
                    pos_above          = tcp_pos + np.array([0., 0., 0.05])
                    pos_above_approach = pos_approach + np.array([0., 0., 0.05])
                    q_above = _wrap_nearest(
                        self.solve_ik_from_config(gj_arr, pos_above, tcp_quat),
                        _wrap_nearest(gj_arr, q_approach))
                    q_above_approach = _wrap_nearest(
                        self.solve_ik_from_config(gj_arr, pos_above_approach, tcp_quat),
                        q_approach)
                    grasp_joints = _wrap_nearest(gj_arr, q_above)
                else:
                    grasp_joints = _wrap_nearest(gj_arr, q_approach)

        if self.simulation:
            self._grasp_sim_duration_s = (
                self._SIM_TOOL_GRASP_DURATION_S
                if category == "tool" else _PbJointRunner._DURATION_S)
            self._grasp_on_complete       = on_complete
            self._grasp_on_phase          = on_phase
            self._grasp_joints            = np.array(grasp_joints, dtype=float)
            self._grasp_q_approach        = (np.array(q_approach, dtype=float)
                                             if q_approach is not None else None)
            self._grasp_q_above           = (np.array(q_above, dtype=float)
                                             if q_above is not None else None)
            self._grasp_q_above_approach  = (np.array(q_above_approach, dtype=float)
                                             if q_above_approach is not None else None)
            if self._grasp_q_approach is not None:
                self._grasp_runner = _PbJointRunner(
                    self.pb_scene.current_q.copy(), self._grasp_q_approach,
                    self._grasp_sim_duration_s)
                self._grasp_phase  = 'approach'
                print("[Robot sim] moveJ → approach")
                self._fire_phase(on_phase, "approaching")
            else:
                self._grasp_runner = _PbJointRunner(
                    self.pb_scene.current_q.copy(), self._grasp_joints,
                    self._grasp_sim_duration_s)
                self._grasp_phase  = 'grasp'
                print("[Robot sim] moveJ → grasp_joints")
                self._fire_phase(on_phase, "grasping")
        else:
            if not _RTDE_AVAILABLE:
                print("[Robot] rtde_control not installed.")
                if on_complete:
                    on_complete(False)
                return
            self.servoStop()
            self._grasp_phase             = None
            self._grasp_joints            = np.array(grasp_joints, dtype=float)
            self._grasp_q_approach        = (np.array(q_approach, dtype=float)
                                             if q_approach is not None else None)
            self._grasp_q_above           = (np.array(q_above, dtype=float)
                                             if q_above is not None else None)
            self._grasp_q_above_approach  = (np.array(q_above_approach, dtype=float)
                                             if q_above_approach is not None else None)
            self._grasp_on_complete       = on_complete
            self._grasp_on_phase          = on_phase
            self._grasp_move_sent    = False
            self._grasp_settle_start = None
            self._grasp_phase         = 'open_pre'
            print("[Robot] Starting real grasp state machine")

    def cancel_grasp(self) -> None:
        """Cancel at any grasp phase. Action depends on how far along the grasp is:
        - open_pre/approach/above: stop and retract immediately (gripper stays open)
        - grasp: reach position first, then close+open+retract
        - close/settle: let settle finish, then open gripper and retract
        - ret_above/retract: stop, go back to board, open gripper, re-retract
        - open_post/put_back/put_back_open: already releasing — ignore
        """
        phase = self._grasp_phase
        if phase is None:
            return
        if self._grasp_cancel_pending:
            print(f"[Robot] Cancel already in progress (phase='{phase}')")
            return
        if phase in ('open_post', 'put_back', 'put_back_open'):
            print(f"[Robot] Cancel ignored — already releasing tool (phase='{phase}')")
            return

        self._grasp_cancel_pending = True

        if self.simulation:
            self._finish_grasp(False)
            return

        if phase == 'grasp':
            print("[Robot] Cancel queued — will close+open gripper then retract")
            return

        if phase in ('close', 'settle'):
            print("[Robot] Cancel queued — will open gripper then retract")
            return

        if phase in ('open_pre', 'approach', 'grasp'):
            if self._grasp_move_sent:
                try:
                    self._rtde_ctrl_conn().stopJ(2.0)
                except Exception as e:
                    print(f"[Robot] cancel_grasp stopJ failed: {e}")
                self._grasp_move_sent = False
            if self._grasp_q_above_approach is not None:
                self._grasp_phase = 'ret_above_approach'
            elif self._grasp_q_approach is not None:
                self._grasp_phase = 'retract'
            else:
                self._grasp_cancel_pending = False
                self.cancel_motion()
                return
            self._fire_phase(self._grasp_on_phase, "retracting")
            print("[Robot] Grasp cancelled — retracting")
            return

        if phase in ('lift', 'ret_above_approach', 'retract'):
            if self._grasp_move_sent:
                try:
                    self._rtde_ctrl_conn().stopJ(2.0)
                except Exception as e:
                    print(f"[Robot] cancel_grasp stopJ failed: {e}")
                self._grasp_move_sent = False
            self._grasp_phase = 'put_back'
            print("[Robot] Grasp cancelled — returning tool to board")

    def cancel_motion(self) -> None:
        if self._handover_pull_active:
            print("[Robot] Object handover pull wait cancelled; gripper remains closed")
            self._finish_object_handover_pull(False)
        startup_was_active = self._startup_active
        joint_was_active = self._joint_move_target is not None
        joint_cb = self._joint_move_on_complete
        self._joint_move_target = None
        self._joint_move_runner = None
        self._joint_move_on_complete = None
        self._startup_active = False
        self._startup_runner = None
        phase              = self._grasp_phase
        cb                 = self._grasp_on_complete
        self._grasp_phase          = None
        self._grasp_on_complete    = None
        self._grasp_on_phase       = None
        self._grasp_runner         = None
        self._grasp_joints            = None
        self._grasp_q_approach        = None
        self._grasp_q_above           = None
        self._grasp_q_above_approach  = None
        self._grasp_cancel_pending    = False
        if not self.simulation:
            if joint_was_active and self._joint_move_sent:
                try:
                    self._rtde_ctrl_conn().stopJ(2.0)
                except Exception as e:
                    print(f"[Robot] cancel gearbox moveJ failed: {e}")
                self._joint_move_sent = False
            elif startup_was_active and self._startup_move_sent:
                try:
                    self._rtde_ctrl_conn().stopJ(2.0)
                except Exception as e:
                    print(f"[Robot] cancel startup stopJ failed: {e}")
                self._startup_move_sent = False
            elif phase is not None and self._grasp_move_sent:
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
        if joint_cb is not None:
            try:
                joint_cb(False)
            except Exception:
                pass

    @staticmethod
    def _fire_phase(on_phase: "Callable[[str], None] | None", name: str) -> None:
        if on_phase is None:
            return
        try:
            on_phase(name)
        except Exception as e:
            print(f"[Robot] on_phase callback error: {e}")

    # ── IK solving ────────────────────────────────────────────────────────────

    def solve_ik_from_config(self, start_q: np.ndarray,
                             target_pos, target_quat_xyzw,
                             n_steps: int = 20) -> np.ndarray:
        """IK that walks incrementally from start_q to avoid wrist flips."""
        import pybullet as p
        pb      = self.pb_scene
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

    # ── Joint state ───────────────────────────────────────────────────────────

    def poll_q(self) -> "np.ndarray | None":
        if self.simulation:
            q = self.pb_scene.current_q.copy()
            self._last_q = q
            return q
        try:
            q = np.array(self._recv_conn().getActualQ(), dtype=float)
            self._last_q = q
            return q
        except Exception as e:
            print(f"[Robot] poll_q failed: {e}")
            return None

    # ── Low-level control (real robot) ────────────────────────────────────────

    def servoJ(self, q: np.ndarray, dt: float,
               speed: float = 1.0, accel: float = 1.0,
               lookahead: float = 0.10, gain: int = 300) -> None:
        # `time` is the blocking command period.  Keep it at the configured
        # control period instead of stretching a 120 Hz command to 30 ms.
        t  = max(min(float(dt),       0.020), 0.002)
        la = max(min(float(lookahead), 0.2), 0.03)
        try:
            self._rtde_ctrl_conn().servoJ(list(q), speed, accel, t, la, gain)
            self._in_servo = True
        except Exception as e:
            print(f"[Robot] servoJ failed ({e})")
            self._rtde_ctrl = None

    def servoStop(self) -> None:
        if self._in_servo and self._rtde_ctrl is not None:
            try:
                self._rtde_ctrl.servoStop()
            except Exception:
                pass
            self._in_servo = False

    # ── Gripper ───────────────────────────────────────────────────────────────

    def open_gripper(self, speed: int = 255, force: int = 10) -> None:
        g = self._gripper_conn()
        g.move_and_wait_for_pos(g.get_open_position(), speed, force)

    def close_gripper(self, speed: int = 255, force: int = 100) -> bool:
        """Close gripper and return True if an object was detected."""
        g = self._gripper_conn()
        _, status = g.move_and_wait_for_pos(g.get_closed_position(), speed, force)
        return status != RobotiqGripper.ObjectStatus.AT_DEST

    def connect_gripper(self) -> None:
        if self.simulation or not _GRIPPER_AVAILABLE:
            return
        try:
            self._gripper_conn()
        except Exception as e:
            print(f"[Robot] Gripper pre-connect failed: {e}")

    # ── Unity publishing ──────────────────────────────────────────────────────

    def publish_base(self, T_world_base: "np.ndarray | None" = None) -> None:
        T      = T_world_base if T_world_base is not None else self._T_world_base
        R_o3d  = T[:3, :3] @ ScipyR.from_euler(
            'z', self._BASE_YAW_CORRECTION_DEG, degrees=True).as_matrix()
        q_xyzw = ScipyR.from_matrix(R_o3d).as_quat()
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u    = open3d_to_unity_quaternion(q_wxyz)
        t_u    = open3d_to_unity_vector(T[:3, 3])
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
        try:
            self._joint_pub.send_string(
                json.dumps({"joint_values": [float(v) for v in q]}))
        except Exception as e:
            print(f"[Robot] publish_joints error: {e}")

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
            try:
                g.activate()
            except Exception as e:
                print(f"[Robot] Gripper activation warning: {e}")
            self._gripper = g
            print("[Robot] Gripper ready.")
        return self._gripper

    # ── Real-robot grasp state machine (driven by tick()) ─────────────────────

    def _tick_real_grasp(self, dt: float) -> None:
        phase = self._grasp_phase
        sc    = self._speed_scale

        def _next(new_phase: str) -> None:
            self._grasp_phase      = new_phase
            self._grasp_move_sent = False

        def _done(ok: bool) -> None:
            result = False if self._grasp_cancel_pending else ok
            self._grasp_phase          = None
            self._grasp_on_phase       = None
            self._grasp_cancel_pending = False
            cb = self._grasp_on_complete
            self._grasp_on_complete = None
            if cb is not None:
                try:
                    cb(result)
                except Exception as e:
                    print(f"[Robot] grasp on_complete error: {e}")

        def _ctrl():
            try:
                return self._rtde_ctrl_conn()
            except Exception as e:
                print(f"[Robot] RTDE connect failed in grasp: {e}")
                _done(False)
                return None

        if phase == 'open_pre':
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] open_gripper failed: {e}")
            self._fire_phase(self._grasp_on_phase, "approaching")
            _next('approach' if self._grasp_q_approach is not None else 'grasp')

        elif phase in ('approach', 'grasp', 'lift', 'ret_above_approach', 'retract'):
            target = {
                'approach':          self._grasp_q_approach,
                'grasp':             self._grasp_joints,
                'lift':              self._grasp_q_above,
                'ret_above_approach': self._grasp_q_above_approach,
                'retract':           self._grasp_q_approach,
            }[phase]
            speed = {
                # Travel quickly through the known-clear waypoint segments,
                # but retain a slower final descent at the object itself.
                'approach':          0.85 * sc,
                'grasp':             0.30 * sc,
                'lift':              0.40 * sc,
                'ret_above_approach': 0.55 * sc,
                'retract':           0.85 * sc,
            }[phase]
            if not self._grasp_move_sent:
                c = _ctrl()
                if c is None:
                    return
                try:
                    c.moveJ(list(target), float(speed), float(speed), asynchronous=True)
                    self._grasp_move_sent = True
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
                        if phase == 'approach':
                            _next('grasp')
                        elif phase == 'grasp':
                            self._fire_phase(self._grasp_on_phase, "grasping")
                            _next('close')
                        elif phase == 'lift':
                            _next('ret_above_approach'
                                  if self._grasp_q_above_approach is not None
                                  else 'retract')
                        elif phase in ('ret_above_approach', 'retract'):
                            _done(self._grasp_ok)
                except Exception as e:
                    print(f"[Robot] isSteady failed ({phase}): {e}")
                    self._rtde_ctrl = None
                    _done(False)

        elif phase == 'close':
            try:
                self._grasp_ok = self.close_gripper()
            except Exception as e:
                print(f"[Robot] close_gripper failed: {e}")
                self._grasp_ok = False
            print(f"[Robot] Gripper closed — object="
                  f"{'detected' if self._grasp_ok else 'NOT detected'}")
            self._grasp_settle_start = time.perf_counter()
            _next('settle')

        elif phase == 'settle':
            if time.perf_counter() - (self._grasp_settle_start or 0.0) >= 1.0:
                if self._grasp_cancel_pending:
                    _next('open_post')
                else:
                    self._fire_phase(self._grasp_on_phase,
                                     "grasped" if self._grasp_ok else "grasp_failed")
                    self._fire_phase(self._grasp_on_phase, "retracting")
                    if self._grasp_q_above is not None:
                        _next('lift')
                    elif self._grasp_q_approach is not None:
                        _next('retract')
                    else:
                        _done(self._grasp_ok)

        elif phase == 'open_post':
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] open_gripper (cancel) failed: {e}")
            self._fire_phase(self._grasp_on_phase, "retracting")
            if self._grasp_q_above is not None:
                _next('lift')
            elif self._grasp_q_approach is not None:
                _next('retract')
            else:
                _done(False)

        elif phase == 'put_back':
            # Cancelled during ret_above/retract — drive back to the tool's
            # grasp joint position before opening the gripper.
            target = self._grasp_joints
            if not self._grasp_move_sent:
                c = _ctrl()
                if c is None:
                    return
                try:
                    c.moveJ(list(target), 0.2 * sc, 0.2 * sc, asynchronous=True)
                    self._grasp_move_sent = True
                    print("[Robot] async moveJ → put_back (returning tool)")
                except Exception as e:
                    print(f"[Robot] async moveJ failed (put_back): {e}")
                    self._rtde_ctrl = None
                    _done(False)
            else:
                c = _ctrl()
                if c is None:
                    return
                try:
                    if c.isSteady():
                        _next('put_back_open')
                except Exception as e:
                    print(f"[Robot] isSteady failed (put_back): {e}")
                    self._rtde_ctrl = None
                    _done(False)

        elif phase == 'put_back_open':
            try:
                self.open_gripper()
            except Exception as e:
                print(f"[Robot] open_gripper (put_back) failed: {e}")
            self._fire_phase(self._grasp_on_phase, "retracting")
            if self._grasp_q_above is not None:
                _next('lift')
            elif self._grasp_q_approach is not None:
                _next('retract')
            else:
                _done(False)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.cancel_board_interaction()
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
        for sock in (self._base_pub, self._joint_pub,
                     self._cmd_sub, self._event_pub):
            try:
                sock.close(0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="CoAssembly dedicated robot-control process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip", default=cfg.UNITY_IP)
    ap.add_argument("--robot-ip", default=cfg.ROBOT_IP,
                    help="UR robot controller IP for live RTDE control")
    ap.add_argument("--simulation", action=argparse.BooleanOptionalAction,
                    default=cfg.SIMULATION)
    ap.add_argument("--calibrated-robot-base", action=argparse.BooleanOptionalAction,
                    default=cfg.USE_CALIBRATED_ROBOT_BASE_POSE)
    ap.add_argument("--gripper-collision", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--diff-ik", action=argparse.BooleanOptionalAction,
                    default=True, help="Use OSC+CBF differential IK (default). --no-diff-ik falls back to PyBullet IK.")
    args = ap.parse_args()

    if not _PYBULLET_AVAILABLE:
        raise SystemExit("[RobotServer] pybullet_ik not available — aborting.")

    server = RobotControlServer(
        quest_ip                  = args.quest_ip,
        robot_ip                  = args.robot_ip,
        simulation                = args.simulation,
        use_calibrated_robot_base = args.calibrated_robot_base,
        gripper_collision         = args.gripper_collision,
        use_diff_ik               = args.diff_ik,
    )
    server.run()


if __name__ == "__main__":
    main()
