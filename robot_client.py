"""robot_client.py — Thin ZMQ-backed client for robot_control_server.py.

Used by main_with_robot.py in place of a direct RobotController instance.
Mirrors RobotController's public method surface (execute_grasp, move_tcp,
cancel_motion, tool_grasp_running, start_force_monitor,
open_gripper_async/close_gripper_async, servoStop, tcp_pose,
arm_link_poses, set_scene_origin, check_reachability) so the grip-state
machine in main_with_robot.py needs only minimal changes to use it.

Keeps a private, IK-free PyBullet scene (`self.pb_scene`) purely for forward
kinematics — robot-mesh rendering and reachability-arrow visualisation. Real
IK/FK for hardware control, the RTDE connection, the gripper, and the frax
CBF filter all live in robot_control_server.py; this class never touches
hardware directly and never blocks the caller except in
`check_reachability` (a rare, user-triggered, already-blocking action).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import zmq

try:
    from pybullet_ik import IKScene as PyBulletScene
    _PYBULLET_AVAILABLE = True
except ImportError:
    PyBulletScene = None   # type: ignore[assignment,misc]
    _PYBULLET_AVAILABLE = False

import main_setting as cfg

_FILE_DIR = Path(__file__).resolve().parent

# Mirrors robot_control_server._SIM_Q_DEG — only used to pose the local
# visualization mesh before the first real state broadcast arrives.
_SIM_Q_DEG = [-105.97, -29.43, 87.53, 33.17, 92.40, 168.95]


def _build_local_viz_scene(simulation: bool, use_calibrated_robot_base: bool,
                            sim_q: np.ndarray) -> "PyBulletScene | None":
    """FK-only mirror of the server's pb_scene base-pose selection, so the
    locally rendered mesh starts in the same place the server's robot does."""
    calib_dir = _FILE_DIR / "calibration_data" / "results"
    try:
        if simulation:
            if use_calibrated_robot_base and calib_dir.exists():
                p1 = np.load(calib_dir / "phase1_results.npz")
                scene = PyBulletScene(T_world_base=p1["T_world_base"])
            else:
                T = np.eye(4, dtype=float)
                T[:3, 3] = [-0.4, -0.8, 0.4]
                scene = PyBulletScene(T_world_base=T)
        elif calib_dir.exists():
            scene = PyBulletScene.from_calibration(calib_dir)
        else:
            print(f"[RobotClient] Calibration dir not found: {calib_dir}")
            return None
        scene.build()
        scene.update_robot(sim_q)
        # Same conservative limits as the server, so local scoring-only IK
        # queries (handover voxel scoring) stay consistent with real motion.
        scene.set_joint_limits(cfg.JOINT_MIN_DEG, cfg.JOINT_MAX_DEG, degrees=True)
    except Exception as e:
        print(f"[RobotClient] Local viz pb_scene failed to build: {e}")
        return None
    return scene


class RobotClient:
    """Drop-in replacement for a direct RobotController, backed by ZMQ."""

    def __init__(self, simulation: bool = True,
                 use_calibrated_robot_base: bool = cfg.USE_CALIBRATED_ROBOT_BASE_POSE,
                 host: str = "127.0.0.1",
                 cmd_port: int = cfg.ROBOT_CMD_PORT,
                 event_port: int = cfg.ROBOT_EVENT_PORT,
                 reachability_timeout: float = 5.0):
        if not _PYBULLET_AVAILABLE:
            raise RuntimeError("pybullet_ik not available")
        self.simulation = simulation
        self._sim_q = np.deg2rad(_SIM_Q_DEG)
        self.pb_scene = _build_local_viz_scene(
            simulation, use_calibrated_robot_base, self._sim_q)
        if self.pb_scene is None:
            raise RuntimeError("local visualization pb_scene failed to build")

        self._reachability_timeout = reachability_timeout
        self._tool_grasp_running   = False
        self._has_cbf              = False
        self._collision_spheres    = (None, None)   # (positions Nx3, radii N), from server state
        self._last_q_debug_t       = 0.0

        ctx = zmq.Context.instance()
        self._cmd_pub = ctx.socket(zmq.PUB)
        self._cmd_pub.connect(f"tcp://{host}:{cmd_port}")
        self._event_sub = ctx.socket(zmq.SUB)
        self._event_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._event_sub.connect(f"tcp://{host}:{event_port}")
        time.sleep(0.2)

        self._next_request_id = 1
        self._callbacks:       "dict[int, Callable]" = {}   # single-shot, popped on first event
        self._phase_callbacks: "dict[int, Callable]" = {}   # persist across grasp_phase events
        print(f"[RobotClient] Connected to robot_control_server "
              f"(cmd:{cmd_port} event:{event_port})")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        try:
            self._cmd_pub.send_string(json.dumps(payload))
        except Exception as e:
            print(f"[RobotClient] Send error: {e}")

    def _new_request(self, on_done: "Callable | None") -> "int | None":
        if on_done is None:
            return None
        rid = self._next_request_id
        self._next_request_id += 1
        self._callbacks[rid] = on_done
        return rid

    def _handle_event(self, msg: dict) -> None:
        if msg.get("type") == "state":
            q = msg.get("q")
            if q is not None:
                q_arr = np.array(q, dtype=float)
                self.pb_scene.update_robot(q_arr)
                # _now = time.time()
                # if _now - self._last_q_debug_t >= 1.0:
                #     self._last_q_debug_t = _now
                #     print(f"[RobotClient] q(deg) = {np.round(np.degrees(q_arr), 1).tolist()}")
            self._tool_grasp_running = bool(msg.get("tool_grasp_running", False))
            self._has_cbf = bool(msg.get("has_cbf", False))
            cs  = msg.get("collision_spheres") or {}
            pos = cs.get("positions")
            rad = cs.get("radii")
            self._collision_spheres = (
                (np.array(pos, dtype=float), np.array(rad, dtype=float))
                if pos is not None and rad is not None else (None, None))
            return
        if msg.get("type") != "event":
            return
        rid = msg.get("request_id")
        if msg.get("name") == "grasp_phase":
            # Not popped — more grasp_phase events (and a terminal grasp_done)
            # for this request_id are still to come.
            cb = self._phase_callbacks.get(rid) if rid is not None else None
            if cb is not None:
                try:
                    cb(msg.get("phase"))
                except Exception as e:
                    print(f"[RobotClient] Phase callback error: {e}")
            return
        if rid is not None:
            self._phase_callbacks.pop(rid, None)
        cb = self._callbacks.pop(rid, None) if rid is not None else None
        if cb is not None:
            try:
                cb(msg)
            except Exception as e:
                print(f"[RobotClient] Callback error: {e}")

    def poll(self) -> None:
        """Drain pending state/event messages. Call once per main-loop iteration."""
        while True:
            try:
                raw = self._event_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._handle_event(msg)

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def tool_grasp_running(self) -> bool:
        return self._tool_grasp_running

    @property
    def has_cbf(self) -> bool:
        """Whether the server's RobotController has a live frax CBF filter."""
        return self._has_cbf

    def collision_spheres(self):
        """(positions Nx3, radii N) for the CBF collision model, computed
        server-side and relayed via state broadcasts; (None, None) if no CBF."""
        return self._collision_spheres

    @property
    def q(self) -> "np.ndarray | None":
        """Latest joint angles (radians), kept in sync from state broadcasts."""
        return self.pb_scene.current_q

    @property
    def tcp_pose(self) -> "np.ndarray | None":
        try:
            return self.pb_scene.update_tcp_bodies()
        except Exception:
            return None

    def arm_link_poses(self):
        try:
            return self.pb_scene.get_arm_link_world_poses()
        except Exception:
            return None

    def query_ik_joints(self, pos_world, quat_xyzw, seed_q=None) -> "np.ndarray | None":
        """Scoring-only IK query (e.g. handover voxel joint-travel scoring).

        Solved locally against the client's own PyBullet client — independent
        physics state from the server's, so this never touches real hardware
        or the safety-critical control loop; it's purely a same-limits
        estimate used to rank candidate points.
        """
        import pybullet as p
        pb = self.pb_scene
        if seed_q is None:
            seed_q = pb.current_q.copy()
        try:
            arm_q_map  = dict(zip(pb.arm_indices, seed_q))
            rest_poses = [float(arm_q_map.get(j, 0.0)) for j in pb._movable]
            joint_q = p.calculateInverseKinematics(
                pb.robot_id, pb.tool0_link_idx,
                list(pos_world), list(quat_xyzw),
                lowerLimits=pb._lower_limits, upperLimits=pb._upper_limits,
                jointRanges=pb._joint_ranges, restPoses=rest_poses,
                maxNumIterations=200, residualThreshold=1e-5)
            return np.array(joint_q[:len(pb.arm_indices)], dtype=np.float64)
        except Exception as e:
            print(f"[RobotClient] query_ik_joints failed: {e}")
            return None

    # ── Scene origin ─────────────────────────────────────────────────────────

    def set_scene_origin(self, T_world_marker10: np.ndarray) -> None:
        """Update the local visualization scene immediately (no round-trip
        needed for rendering) and forward the same update to the
        authoritative hardware scene owned by the server."""
        self.pb_scene.set_scene_origin(T_world_marker10)
        self._send({"cmd": "set_scene_origin",
                    "T": np.asarray(T_world_marker10, dtype=float).flatten().tolist()})

    # ── Motion commands ──────────────────────────────────────────────────────

    def track_hand(self, target_pos, target_quat) -> None:
        self._send({"cmd": "track_hand",
                    "target_pos":  [float(v) for v in target_pos],
                    "target_quat": [float(v) for v in target_quat]})

    def track_hand_stop(self) -> None:
        self._send({"cmd": "track_hand_stop"})

    def move_tcp(self, pos, quat,
                 on_complete: "Callable[[bool], None] | None" = None) -> None:
        rid = self._new_request(
            (lambda msg: on_complete(bool(msg.get("ok", False)))) if on_complete else None)
        self._send({"cmd": "move_tcp",
                    "pos":  [float(v) for v in pos],
                    "quat": [float(v) for v in quat],
                    "request_id": rid})

    def execute_grasp(self, grasp_joints, on_complete: "Callable[[bool], None] | None" = None,
                      on_phase: "Callable[[str], None] | None" = None,
                      category: str = "tool", board_normal=None) -> None:
        rid = None
        if on_complete is not None or on_phase is not None:
            rid = self._next_request_id
            self._next_request_id += 1
            if on_complete is not None:
                self._callbacks[rid] = lambda msg: on_complete(bool(msg.get("ok", False)))
            if on_phase is not None:
                self._phase_callbacks[rid] = on_phase
        self._send({
            "cmd":           "execute_grasp",
            "grasp_joints":  np.asarray(grasp_joints, dtype=float).tolist(),
            "category":      category,
            "board_normal":  (np.asarray(board_normal, dtype=float).tolist()
                              if board_normal is not None else None),
            "request_id":    rid,
        })

    def cancel_motion(self) -> None:
        self._send({"cmd": "cancel"})

    def servoStop(self) -> None:
        self._send({"cmd": "servo_stop"})

    # ── Gripper ──────────────────────────────────────────────────────────────

    def open_gripper_async(self, on_done: "Callable | None" = None) -> None:
        rid = self._new_request((lambda msg: on_done()) if on_done else None)
        self._send({"cmd": "open_gripper", "request_id": rid})

    def close_gripper_async(self, on_done: "Callable | None" = None) -> None:
        rid = self._new_request((lambda msg: on_done()) if on_done else None)
        self._send({"cmd": "close_gripper", "request_id": rid})

    # ── Force monitor ────────────────────────────────────────────────────────

    def start_force_monitor(self, mode: str, on_trigger: "Callable",
                            threshold: "float | None" = None) -> None:
        rid = self._new_request(lambda msg: on_trigger())
        self._send({"cmd": "start_force_monitor", "mode": mode,
                    "threshold": threshold, "request_id": rid})

    def stop_force_monitor(self) -> None:
        self._send({"cmd": "stop_force_monitor"})

    # ── Reachability (blocking request/response — rare, user-triggered) ────────

    def check_reachability(self, T_pegboard_world: np.ndarray, target_quat_xyzw=None):
        """Blocks (up to `reachability_timeout`) for the server's result,
        mirroring the previous direct/blocking pb_scene call so the 'R'-key
        handler in main_with_robot.py needs no restructuring.

        Returns (n_reachable, n_total, points_world Nx3, reachable_flags N bool).
        """
        rid = self._next_request_id
        self._next_request_id += 1
        self._send({
            "cmd":               "check_reachability",
            "T_pegboard_world":  np.asarray(T_pegboard_world, dtype=float).flatten().tolist(),
            "target_quat_xyzw":  (np.asarray(target_quat_xyzw, dtype=float).tolist()
                                  if target_quat_xyzw is not None else None),
            "request_id":        rid,
        })
        deadline = time.time() + self._reachability_timeout
        while time.time() < deadline:
            try:
                raw = self._event_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (msg.get("type") == "event" and msg.get("name") == "reachability_result"
                    and msg.get("request_id") == rid):
                return (msg["n_reachable"], msg["n_total"],
                        np.array(msg["points"], dtype=np.float64),
                        np.array(msg["flags"], dtype=bool))
            self._handle_event(msg)   # don't drop state/other events while waiting
        print("[RobotClient] check_reachability timed out.")
        return (0, 0, np.zeros((0, 3)), np.zeros(0, dtype=bool))

    def close(self) -> None:
        try:
            self._cmd_pub.close(0)
            self._event_sub.close(0)
        except Exception:
            pass
