"""robot_client.py — Thin ZMQ-backed client for robot_control_server.py.

Used by main_with_robot.py in place of a direct RobotController instance.
Exposes motion, tool/part grasp, and the isolated AR board-interaction
commands while all hardware access remains in robot_control_server.py.

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
        scene.set_joint_limits(cfg.JOINT_MIN_DEG, cfg.JOINT_MAX_DEG, degrees=True)
        scene.set_rest_poses(cfg.JOINT_REST_DEG, degrees=True)
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
                 event_port: int = cfg.ROBOT_EVENT_PORT):
        if not _PYBULLET_AVAILABLE:
            raise RuntimeError("pybullet_ik not available")
        self.simulation = simulation
        self._sim_q = np.deg2rad(_SIM_Q_DEG)
        self.pb_scene = _build_local_viz_scene(
            simulation, use_calibrated_robot_base, self._sim_q)
        if self.pb_scene is None:
            raise RuntimeError("local visualization pb_scene failed to build")

        self._tool_grasp_running = False
        self._move_running       = False
        self._board_state        = "inactive"
        self._board_interaction_active = False

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

    def _handle_event(self, msg: dict) -> None:
        if msg.get("type") == "state":
            q = msg.get("q")
            if q is not None:
                self.pb_scene.update_robot(np.array(q, dtype=float))
            self._tool_grasp_running = bool(msg.get("tool_grasp_running", False))
            self._move_running       = bool(msg.get("move_running", False))
            self._board_state        = str(msg.get("board_state", "inactive"))
            self._board_interaction_active = bool(
                msg.get("board_interaction_active", False))
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
    def move_running(self) -> bool:
        return self._move_running

    @property
    def board_state(self) -> str:
        return self._board_state

    @property
    def board_interaction_active(self) -> bool:
        return self._board_interaction_active

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

    # ── Scene origin ─────────────────────────────────────────────────────────

    def set_scene_origin(self, T_world_anchor: np.ndarray) -> None:
        """Update the local visualization scene immediately (no round-trip
        needed for rendering) and forward the same update to the
        authoritative hardware scene owned by the server."""
        self.pb_scene.set_scene_origin(T_world_anchor)
        self._send({"cmd": "set_scene_origin",
                    "T": np.asarray(T_world_anchor, dtype=float).flatten().tolist()})

    # ── Motion commands ──────────────────────────────────────────────────────

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

    def move_to_pose(self, pos, quat=None, *, board_move: bool = False,
                     on_complete: "Callable[[bool], None] | None" = None) -> None:
        """Stream IK+CBF toward a fixed Cartesian target; fires on_complete(ok) on arrival."""
        rid = None
        if on_complete is not None:
            rid = self._next_request_id
            self._next_request_id += 1
            self._callbacks[rid] = lambda msg: on_complete(bool(msg.get("ok", False)))
        self._send({
            "cmd":        "move_to_pose",
            "pos":        np.asarray(pos, float).tolist(),
            "quat":       np.asarray(quat, float).tolist() if quat is not None else None,
            "board_move": bool(board_move),
            "request_id": rid,
        })

    def start_board_interaction(self) -> None:
        """Open for force-triggered board grasp (simulation enters held state)."""
        self._send({"cmd": "start_board_interaction"})

    def cancel_board_interaction(self) -> None:
        """Disable board force monitoring without opening a held gripper."""
        self._send({"cmd": "cancel_board_interaction"})

    def update_move_target(self, pos, quat=None) -> None:
        """Update the target of an in-progress move_to_pose without restarting it."""
        self._send({
            "cmd":  "update_move_target",
            "pos":  np.asarray(pos, float).tolist(),
            "quat": np.asarray(quat, float).tolist() if quat is not None else None,
        })

    def cancel_move(self) -> None:
        """Cancel an in-progress move_to_pose."""
        self._send({"cmd": "cancel_move"})

    def cancel_motion(self) -> None:
        self._send({"cmd": "cancel"})

    def cancel_grasp(self) -> None:
        """Cancel mid-grasp and retract to approach position."""
        self._send({"cmd": "cancel_grasp"})

    def close(self) -> None:
        try:
            self._cmd_pub.close(0)
            self._event_sub.close(0)
        except Exception:
            pass
