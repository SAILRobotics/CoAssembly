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

    # Simulation-mode grasp geometry (matches MainScene constants).
    _SIM_TCP_OFFSET    = 0.17   # tool0-to-gripper-tip offset along board normal
    _SIM_APPROACH_DIST = 0.30   # standoff added beyond the grasp point

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
        self._grasp_status: str = "idle"

        # Simulation motion state machine
        self._sim_runner  = None          # _PbWaypointRunner instance
        self._sim_phase: "str | None" = None  # 'approach' | 'final' | 'move_tcp'
        self._sim_on_complete: "Callable | None" = None
        self._sim_approach_pos: "np.ndarray | None" = None
        self._sim_grasp_pos:    "np.ndarray | None" = None
        self._sim_grasp_quat:   "np.ndarray | None" = None

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
        Real robot : solves IK then sends a servoJ command.
        """
        if self._pb_scene is None:
            return
        if self.simulation:
            self._pb_scene.step_ik(
                self._pb_scene.current_q, list(target_pos), list(target_quat), dt)
        else:
            try:
                q = self.solve_ik(np.array(target_pos), list(target_quat), self._last_q)
            except RuntimeError:
                return
            self.servoJ(q, dt)

    # ── Simulation per-frame tick ─────────────────────────────────────────────

    def tick(self) -> None:
        """Advance the simulation waypoint runner one frame.

        Call this every loop iteration regardless of whether a grasp or
        move_tcp is active — it is a no-op when nothing is running.
        """
        if not self.simulation or self._sim_runner is None:
            return
        if not self._sim_runner.done:
            self._sim_runner.update(self._pb_scene.robot_id,
                                    self._pb_scene.arm_indices)
            return
        self._on_sim_phase_done()

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

    @property
    def grasp_status(self) -> str:
        if self.simulation:
            return self._sim_phase or "idle"
        return self._grasp_status

    def execute_grasp(
        self,
        tool_data:    tuple,
        grasp_joints: "list | None"                   = None,
        category:     str                             = "tool",
        on_complete:  "Callable[[bool], None] | None" = None,
    ) -> None:
        """Approach → grasp in both simulation and real-robot modes.

        Parameters
        ----------
        tool_data    : (centroid_world, R_world, size) from get_world_data().
        grasp_joints : Pre-recorded joint angles [6 floats, rad].  Sim mode
                       ignores this and uses IK.
        category     : "tool" or "part" (affects retract direction for real).
        on_complete  : Called with success bool when the sequence finishes.
        """
        if self.tool_grasp_running:
            print("[Robot] Grasp already running — cancel first.")
            return
        if self.simulation:
            self._start_sim_grasp(tool_data, on_complete)
        else:
            if not _RTDE_AVAILABLE:
                print("[Robot] rtde_control not installed.")
                return
            self._grasp_cancel.clear()
            self._grasp_thread = threading.Thread(
                target=self._grasp_sequence,
                args=(tool_data, grasp_joints, category, on_complete),
                daemon=True,
            )
            self._grasp_thread.start()

    def cancel_motion(self) -> None:
        """Abort any running grasp and stop the arm."""
        if self.simulation:
            self._finish_sim(False)
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

    # ── Internal: simulation grasp state machine ──────────────────────────────

    def _start_sim_grasp(self, tool_data, on_complete):
        if self._pb_scene is None or not _PB_RUNNER_AVAILABLE:
            if on_complete:
                on_complete(False)
            return
        centroid, R_world, _sz = tool_data
        board_out  = R_world[:, 2]
        board_out  = board_out / (np.linalg.norm(board_out) + 1e-9)
        grasp_quat = _tool_grasp_quat(R_world)

        tcp_grasp    = centroid + self._SIM_TCP_OFFSET * board_out
        tcp_approach = centroid + (self._SIM_TCP_OFFSET + self._SIM_APPROACH_DIST) * board_out

        self._sim_grasp_pos    = tcp_grasp
        self._sim_approach_pos = tcp_approach
        self._sim_grasp_quat   = grasp_quat
        self._sim_on_complete  = on_complete

        try:
            self._sim_runner = _PbWaypointRunner(
                self._pb_scene.robot_id,
                self._pb_scene.tool0_link_idx,
                self._pb_scene.current_q.copy(),
                self._pb_scene.arm_indices,
                tcp_approach.tolist(),
                target_quat_xyzw=grasp_quat,
            )
            self._sim_phase = 'approach'
            print(f"[Robot sim] Grasp approach → {np.round(tcp_approach, 3).tolist()}")
        except Exception as e:
            print(f"[Robot sim] Grasp start failed: {e}")
            self._finish_sim(False)

    def _on_sim_phase_done(self):
        if self._sim_phase == 'approach':
            try:
                self._sim_runner = _PbWaypointRunner(
                    self._pb_scene.robot_id,
                    self._pb_scene.tool0_link_idx,
                    self._pb_scene.current_q.copy(),
                    self._pb_scene.arm_indices,
                    self._sim_grasp_pos.tolist(),
                    target_quat_xyzw=self._sim_grasp_quat,
                    straight_line=True,
                    straight_line_start=self._sim_approach_pos.tolist(),
                )
                self._sim_phase = 'final'
                print(f"[Robot sim] Grasp final → {np.round(self._sim_grasp_pos, 3).tolist()}")
            except Exception as e:
                print(f"[Robot sim] Grasp final failed: {e}")
                self._finish_sim(False)
        else:
            self._finish_sim(True)

    def _finish_sim(self, success: bool) -> None:
        cb = self._sim_on_complete
        self._sim_runner       = None
        self._sim_phase        = None
        self._sim_on_complete  = None
        self._sim_approach_pos = None
        self._sim_grasp_pos    = None
        self._sim_grasp_quat   = None
        if cb is not None:
            try:
                cb(success)
            except Exception:
                pass

    # ── Internal: real-robot grasp sequence ───────────────────────────────────

    def _check_cancel(self) -> None:
        if self._grasp_cancel.is_set():
            raise InterruptedError("Grasp cancelled.")

    def _grasp_sequence(self, tool_data, grasp_joints, category, on_complete):
        if grasp_joints is None:
            print("[Robot] No grasp_joints recorded — cannot execute grasp.")
            if on_complete:
                on_complete(False)
            return

        centroid, R_world, _sz = tool_data
        # R_world[:, 2] is the pegboard outward normal (same as _plane_n in annotator)
        normal  = R_world[:, 2]
        normal  = normal / (np.linalg.norm(normal) + 1e-9)
        is_part = (category == "part")
        success = False

        try:
            self.servoStop()
            time.sleep(0.05)

            current_q = self.poll_q()
            if current_q is None:
                raise RuntimeError("Could not read joint angles from robot.")
            self._check_cancel()

            # 1. FK of grasp_joints → TCP pose at grasp
            q_grasp = np.array(grasp_joints, dtype=float)
            self._pb_scene.update_robot(q_grasp)
            T_grasp = self._pb_scene.update_tcp_bodies()
            if T_grasp is None:
                raise RuntimeError("FK failed for grasp_joints.")
            pos_grasp  = T_grasp[:3, 3]
            quat_grasp = ScipyR.from_matrix(T_grasp[:3, :3]).as_quat().tolist()

            # 2. Compute approach standoff along pegboard outward normal
            if is_part:
                pos_above    = pos_grasp + np.array([0.0, 0.0, 0.05])
                pos_approach = pos_above + self._approach_dist * normal
                q_approach   = self.solve_ik(pos_approach, quat_grasp, current_q)
                q_above      = self.solve_ik(pos_above,    quat_grasp, q_approach)
            else:
                pos_approach = pos_grasp + self._approach_dist * normal
                q_approach   = self.solve_ik(pos_approach, quat_grasp, current_q)

            self._check_cancel()

            # 3. Open gripper
            self._set_status("Opening gripper")
            self.open_gripper()
            self._check_cancel()

            # 4. servoJ to approach standoff (smooth streaming, interruptible)
            self._set_status("Streaming to approach")
            self._servo_to(current_q, q_approach)
            self._check_cancel()

            if is_part:
                self._set_status("Moving above grasp")
                self._servo_to(q_approach, q_above)
                self._check_cancel()

            # 5. Final approach with exact pre-recorded joints (slow moveJ)
            self._set_status("Moving to grasp pose")
            self.moveJ(grasp_joints, speed=self._speed * 0.5, accel=self._accel * 0.5)
            self._check_cancel()

            # 6. Close gripper
            self._set_status("Closing gripper")
            self.close_gripper()
            time.sleep(0.3)
            self._check_cancel()

            # 7. Retract
            if is_part:
                self._set_status("Lifting")
                self.moveJ(q_above.tolist(), speed=self._speed * 0.5, accel=self._accel * 0.5)
            self._set_status("Retracting to approach")
            self.moveJ(q_approach.tolist())

            self._set_status("Done")
            success = True

        except InterruptedError:
            self._set_status("Cancelled")
        except Exception as e:
            self._set_status(f"Error: {e}")
            print(f"[Robot] Grasp error: {e}")

        if on_complete is not None:
            try:
                on_complete(success)
            except Exception:
                pass

    def _servo_to(self, q_start: np.ndarray, q_end: np.ndarray,
                  duration: float = 3.0, dt: float = 0.02) -> None:
        """Stream servoJ commands interpolating from q_start to q_end."""
        steps = max(1, int(duration / dt))
        for i in range(steps):
            if self._grasp_cancel.is_set():
                raise InterruptedError("Grasp cancelled.")
            alpha = (i + 1) / steps
            q_cmd = q_start + alpha * (q_end - q_start)
            self.servoJ(q_cmd, dt)
            time.sleep(dt)
        self.servoStop()
        time.sleep(0.1)

    def _set_status(self, msg: str) -> None:
        self._grasp_status = msg
        print(f"[Robot] {msg}")
