"""robot_control_server.py — Dedicated process owning the UR10e hardware link.

Runs the robot's real-time control loop (RTDE servoJ streaming, the frax CBF
safety filter, gripper actuation, force-triggered contact/release detection)
on its own fixed-rate timer (cfg.ROBOT_CONTROL_HZ), independent of
main_with_robot.py's vision-loop rate. Owns the single RTDE connection, the
single gripper connection, and the PyBullet IK scene used for real IK/FK —
none of that is touched from any other process.

Talks to main_with_robot.py (via robot_client.RobotClient) over ZMQ:
    SUB cfg.ROBOT_CMD_PORT   (bind) — commands in, one JSON dict per message
    PUB cfg.ROBOT_EVENT_PORT (bind) — periodic state + one-shot events out

Usage
-----
    python robot_control_server.py
    python robot_control_server.py --robot-ip 192.168.50.70 --no-simulation
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

try:
    from pybullet_ik import IKScene as PyBulletScene
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False
    print("[robot_control_server] pybullet_ik import failed — cannot run.")

from robot_controller import RobotController

import main_setting as cfg

_FILE_DIR = Path(__file__).resolve().parent

# Mirrors MainScene._SIM_Q_DEG in main_with_robot.py — the resting pose the
# arm is shown/initialised at before any real tracking/grasp is commanded.
_SIM_Q_DEG = [-105.97, -29.43, 87.53, 33.17, 92.40, 168.95]


def _build_pb_scene(simulation: bool, use_calibrated_robot_base: bool,
                     sim_q: np.ndarray) -> "PyBulletScene | None":
    """Mirrors MainScene's pb_scene construction (main_with_robot.py) exactly,
    so the server's robot pose/behaviour matches what main_with_robot.py used
    to build in-process."""
    calib_dir = _FILE_DIR / "calibration_data" / "results"
    scene: "PyBulletScene | None" = None
    if simulation:
        if use_calibrated_robot_base and calib_dir.exists():
            try:
                p1 = np.load(calib_dir / "phase1_results.npz")
                scene = PyBulletScene(T_world_base=p1["T_world_base"])
                scene.build()
                scene.update_robot(sim_q)
                print("[RobotServer] Simulation + calibrated base pose (headless).")
            except Exception as e:
                print(f"[RobotServer] PyBullet (calibrated) failed: {e}")
                scene = None
        else:
            T_world_base_sim = np.eye(4, dtype=float)
            T_world_base_sim[:3, 3] = [-0.4, -0.8, 0.4]
            try:
                scene = PyBulletScene(T_world_base=T_world_base_sim)
                scene.build()
                scene.update_robot(sim_q)
                print("[RobotServer] Simulation — hardcoded base pose (headless).")
            except Exception as e:
                print(f"[RobotServer] PyBullet scene failed to build: {e}")
                scene = None
    else:
        if calib_dir.exists():
            try:
                scene = PyBulletScene.from_calibration(calib_dir)
                scene.build()
                scene.update_robot(sim_q)
            except Exception as e:
                print(f"[RobotServer] PyBullet scene failed to build: {e}")
                scene = None
        else:
            print(f"[RobotServer] Calibration dir not found: {calib_dir}")
    return scene


class RobotControlServer:
    _STATE_INTERVAL = 1.0 / 30.0   # Hz — state broadcast rate (independent of control Hz)
    _Q_DEBUG_INTERVAL = 1.0        # seconds between "[RobotServer] q(deg)" debug prints

    def __init__(self, quest_ip: str, robot_ip: "str | None", simulation: bool,
                 use_calibrated_robot_base: bool, gripper_collision: bool):
        self._sim_q = np.deg2rad(_SIM_Q_DEG)
        self._last_q_debug_t = 0.0
        # Real hardware runs slower for now (safety while testing) — separate
        # loop Hz and a blanket speed/accel multiplier. Sim is unaffected.
        self.control_hz = cfg.ROBOT_CONTROL_HZ if simulation else cfg.ROBOT_CONTROL_HZ_REAL

        self.pb_scene = _build_pb_scene(simulation, use_calibrated_robot_base, self._sim_q)
        if self.pb_scene is None:
            raise RuntimeError("[RobotServer] pb_scene failed to build — cannot run.")
        self.pb_scene.set_joint_limits(cfg.JOINT_MIN_DEG, cfg.JOINT_MAX_DEG, degrees=True)

        urdf = cfg.SCENE_LAYOUT_DIR.parent / "robot_assets" / "ur10e.urdf"
        self.robot = RobotController(
            unity_ip               = quest_ip,
            pb_scene               = self.pb_scene,
            T_world_base           = self.pb_scene.T_world_base,
            robot_ip               = robot_ip if not simulation else None,
            speed_scale             = 1.0 if simulation else cfg.REAL_ROBOT_SPEED_SCALE,
            urdf_path               = str(urdf) if urdf.exists() else None,
            frax_q_min              = list(np.deg2rad(cfg.JOINT_MIN_DEG)),
            frax_q_max              = list(np.deg2rad(cfg.JOINT_MAX_DEG)),
            frax_ws_lo               = cfg.WORKSPACE_LO,
            frax_ws_hi               = cfg.WORKSPACE_HI,
            frax_gripper_collision  = gripper_collision,
        )
        if not simulation:
            self.robot.connect_gripper()

        # Base pose is only meaningful (and only published to Unity) once the
        # vision side has actually locked the world anchor and told us where
        # it is via set_scene_origin — mirrors the old main_with_robot.py
        # behavior of gating publish_base on `self.anchor.locked`. Publishing
        # unconditionally from server startup sent a stale/premature base
        # pose into an unlocked WorldRoot, which looked like the robot mesh
        # jumping/swimming around before the anchor lock settled.
        self._scene_origin_set = False

        # Latest hand-tracking target; server keeps stepping toward it every
        # loop iteration until a track_hand_stop / cancel command clears it.
        self._track_target: "tuple | None" = None

        ctx = zmq.Context.instance()
        self._cmd_sub = ctx.socket(zmq.SUB)
        self._cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._cmd_sub.bind(f"tcp://127.0.0.1:{cfg.ROBOT_CMD_PORT}")
        self._event_pub = ctx.socket(zmq.PUB)
        self._event_pub.bind(f"tcp://127.0.0.1:{cfg.ROBOT_EVENT_PORT}")
        time.sleep(0.2)

        self._running = True
        print(f"[RobotServer] Ready — {'simulation' if simulation else f'live ({robot_ip})'}, "
              f"cmd:{cfg.ROBOT_CMD_PORT} event:{cfg.ROBOT_EVENT_PORT} "
              f"@ {self.control_hz} Hz"
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
        # pb_scene.current_q mirrors reality in both modes: sim writes it
        # directly, real hardware is synced into it via poll_q() each
        # iteration in run() below — matches main_with_robot.py's prior
        # `self.robot.publish_joints(self.pb_scene.current_q)` usage.
        q = self.pb_scene.current_q
        self.robot.publish_joints(q)
        if self._scene_origin_set:
            self.robot.publish_base(self.pb_scene.T_world_base)

        # _now = time.time()
        # if _now - self._last_q_debug_t >= self._Q_DEBUG_INTERVAL:
        #     self._last_q_debug_t = _now
        #     print(f"[RobotServer] q(deg) = "
        #           f"{np.round(np.degrees(q), 1).tolist()}  "
        #           f"track_target={'set' if self._track_target is not None else 'None'}  "
        #           f"sim_phase={self.robot._sim_phase!r}")

        frax = self.robot._frax
        positions = radii = None
        if frax is not None:
            try:
                positions, radii = frax.link_spheres_world(q)
            except Exception as e:
                print(f"[RobotServer] link_spheres_world failed: {e}")

        self._publish({
            "type":               "state",
            "q":                  q.tolist(),
            "tool_grasp_running": bool(self.robot.tool_grasp_running),
            "has_cbf":            frax is not None,
            "collision_spheres": {
                "positions": positions.tolist() if positions is not None else None,
                "radii":     radii.tolist()     if radii     is not None else None,
            },
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

    # ── Command dispatch ─────────────────────────────────────────────────────

    def _handle_cmd(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        rid = msg.get("request_id")

        if cmd == "set_scene_origin":
            T = np.array(msg["T"], dtype=float).reshape(4, 4)
            self.pb_scene.set_scene_origin(T)
            self._scene_origin_set = True
            self.robot.publish_base(self.pb_scene.T_world_base)

        elif cmd == "track_hand":
            self._track_target = (msg["target_pos"], msg["target_quat"])

        elif cmd == "track_hand_stop":
            self._track_target = None

        elif cmd == "move_tcp":
            self.robot.move_tcp(
                msg["pos"], msg["quat"],
                on_complete=lambda ok, rid=rid: self._publish_event(
                    "move_tcp_done", rid, ok=bool(ok)))

        elif cmd == "execute_grasp":
            self._track_target = None   # stop hand-tracking servoJ before grasp moveJ starts
            self.robot.execute_grasp(
                msg["grasp_joints"],
                category      = msg.get("category", "tool"),
                board_normal  = msg.get("board_normal"),
                on_complete   = lambda ok, rid=rid: self._publish_event(
                    "grasp_done", rid, ok=bool(ok)),
                on_phase      = lambda phase, rid=rid: self._publish_event(
                    "grasp_phase", rid, phase=phase))

        elif cmd == "cancel":
            self._track_target = None
            self.robot.cancel_motion()

        elif cmd == "open_gripper":
            try:
                self.robot.open_gripper()
            except Exception as e:
                print(f"[RobotServer] open_gripper error: {e}")
            self._publish_event("gripper_done", rid, action="open")

        elif cmd == "close_gripper":
            try:
                self.robot.close_gripper()
            except Exception as e:
                print(f"[RobotServer] close_gripper error: {e}")
            self._publish_event("gripper_done", rid, action="close")

        elif cmd == "servo_stop":
            self.robot.servoStop()

        elif cmd == "start_force_monitor":
            mode      = msg["mode"]
            threshold = msg.get("threshold")
            self.robot.start_force_monitor(
                mode,
                lambda rid=rid, mode=mode: self._publish_event(
                    "force_triggered", rid, mode=mode),
                threshold=threshold)

        elif cmd == "stop_force_monitor":
            self.robot.stop_force_monitor()

        elif cmd == "check_reachability":
            # User-triggered ('R' key), rare — run inline (blocks the control
            # loop briefly, same as it did in-process before this refactor).
            T = np.array(msg["T_pegboard_world"], dtype=float).reshape(4, 4)
            quat = msg.get("target_quat_xyzw")
            quat = np.array(quat, dtype=float) if quat is not None else None
            n_reach, n_total, pts, flags = self.pb_scene.check_reachability(
                T, target_quat_xyzw=quat)
            self._publish_event(
                "reachability_result", rid,
                n_reachable=n_reach, n_total=n_total,
                points=pts.tolist(), flags=flags.tolist())

        else:
            print(f"[RobotServer] Unknown command: {cmd!r}")

    # ── Main loop ────────────────────────────────────────────────────────────

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

                if not self.robot.simulation:
                    q_fresh = self.robot.poll_q()
                    if q_fresh is not None:
                        self.pb_scene.update_robot(q_fresh)

                if self._track_target is not None:
                    pos, quat = self._track_target
                    self.robot.step_hand_track(pos, quat, dt)
                else:
                    self.robot.tick(dt)

                now = time.time()
                if now - last_state_pub >= self._STATE_INTERVAL:
                    self._publish_state()
                    last_state_pub = now

                elapsed = time.perf_counter() - iter_t0
                sleep_left = target_dt - elapsed
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            pass
        finally:
            self.robot.close()
            self.pb_scene.disconnect()
            print("[RobotServer] Shutting down.")


def main():
    ap = argparse.ArgumentParser(
        description="CoAssembly dedicated robot-control process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip", default=cfg.UNITY_IP)
    ap.add_argument("--robot-ip", default=cfg.ROBOT_IP,
                    help="UR robot controller IP for live RTDE control")
    ap.add_argument("--simulation", action=argparse.BooleanOptionalAction,
                    default=cfg.SIMULATION,
                    help="Fixed default joint angles (--simulation) or live RTDE (--no-simulation)")
    ap.add_argument("--calibrated-robot-base", action=argparse.BooleanOptionalAction,
                    default=cfg.USE_CALIBRATED_ROBOT_BASE_POSE,
                    help="Load robot base pose from calibration_data/ even in simulation mode")
    ap.add_argument("--gripper-collision", action=argparse.BooleanOptionalAction, default=False,
                    help="Include gripper spheres in CBF self-collision model")
    args = ap.parse_args()

    if not _PYBULLET_AVAILABLE:
        raise SystemExit("[RobotServer] pybullet_ik not available — aborting.")

    server = RobotControlServer(
        quest_ip                  = args.quest_ip,
        robot_ip                  = args.robot_ip,
        simulation                = args.simulation,
        use_calibrated_robot_base = args.calibrated_robot_base,
        gripper_collision         = args.gripper_collision,
    )
    server.run()


if __name__ == "__main__":
    main()
