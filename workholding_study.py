#!/usr/bin/env python3
"""workholding_study.py — Freedrive vs AR vs AR+Freedrive board-placement study.

A board is clamped in the robot's gripper for one independent session: grasp,
10 trials, release. --mode picks which interaction condition this session
tests:

    freedrive  — physically drag the arm+board by hand. No AR handle.
    ar         — grab the AR box handle in the headset, drag, release; the
                 robot then autonomously drives the board to the released
                 pose. No freedrive.
    hybrid     — both channels live at once: freedrive AND the AR handle.

Run the script once per mode (e.g. three separate invocations, one per
condition) to cover all three. For each of cfg.workholding_test_poses()'s 10
target poses, the robot first autonomously returns to its default pose
(still holding the board), then the participant repositions the board using
whichever control channel(s) --mode grants.

The manipulated variable is which control channel(s) are available — a
translucent "ghost" box at the current trial's target pose is shown in ALL
THREE modes, on cfg.WORKHOLDING_BOX_PORT via WorkholdingBoxReceiver.cs (same
mechanism workholding_testing.py already uses to park a static box).

Trials auto-complete once the board's actual pose is within tolerance of the
target for a dwell period (freedrive channel) or once an AR-triggered
autonomous move lands within tolerance (AR channel) — whichever happens
first. A manual override key force-completes the current trial (for
experimenter recovery if a trial stalls or a participant needs to bail).

Real hardware only (no --simulation) — freedrive has no effect in
simulation. Requires robot_control_server.py running separately with
--no-simulation, and this only talks to it over ZMQ via
robot_client.RobotClient, same as main_with_robot.py.

Reuses main_with_robot.py's receiver classes unchanged (import only), same
as workholding_testing.py:
  _CamFeedReceiver, _ArUcoWorker, _ArucoPoseEstimator, _HandDataReceiver,
  _WorldAnchor, _GripPoseBridge, _ToolSelectionManager, _RelockCubePublisher,
  _WorkspaceBoundPublisher, _OffsetTuner.

Usage
-----
    python workholding_study.py --session-name P01 --mode ar
    python workholding_study.py --session-name P01 --mode freedrive
    python workholding_study.py --session-name P01 --mode hybrid

Keys: ENTER lock/relock marker 100; F force-complete the current trial; ESC quit
(flushes whatever was logged so far).
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

import main_setting as cfg
from main_with_robot import (
    _CamFeedReceiver,
    _ArUcoWorker,
    _ArucoPoseEstimator,
    _HandDataReceiver,
    _WorldAnchor,
    _GripPoseBridge,
    _ToolSelectionManager,
    _RelockCubePublisher,
    _WorkspaceBoundPublisher,
    _OffsetTuner,
)
from robot_client import RobotClient

_FILE_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _FILE_DIR / "study_logs"

_MODES = ("freedrive", "ar", "hybrid")
# The manipulated variable: which control channel(s) the participant has.
# The AR ghost target box is shown in every mode regardless.
_CHANNELS = {
    "freedrive": {"freedrive": True,  "ar": False},
    "ar":        {"freedrive": False, "ar": True},
    "hybrid":    {"freedrive": True,  "ar": True},
}

_TRIAL_CSV_HEADER = [
    "session_name", "mode", "trial_idx", "pose_idx",
    "target_pos_x_m", "target_pos_y_m", "target_pos_z_m",
    "target_euler_x_deg", "target_euler_y_deg", "target_euler_z_deg",
    "start_time", "end_time", "duration_s",
    "final_pos_error_m", "final_angle_error_deg",
    "num_interactions", "completion_reason",
]
_TRAJ_CSV_HEADER = [
    "session_name", "mode", "trial_idx", "t_rel_s",
    "tcp_pos_x_m", "tcp_pos_y_m", "tcp_pos_z_m",
    "tcp_quat_x", "tcp_quat_y", "tcp_quat_z", "tcp_quat_w",
    "joint1_deg", "joint2_deg", "joint3_deg", "joint4_deg", "joint5_deg", "joint6_deg",
]


class WorkholdingStudy:
    # Same lock/relock tuning as MainScene / workholding_testing.py.
    _RELOCK_COOLDOWN        = 2.0
    _AUTO_LOCK_MAX_DIST     = 1.0    # metres — auto-lock-on-sight only within this range
    _AUTO_LOCK_MAX_TILT_DEG = 45.0   # degrees — max tilt from vertical to auto-lock

    # Trial completion / interaction-counting tuning (independent of the CBF
    # servo tolerances in robot_control_server.py — human placement is noisier).
    _STUDY_POS_TOL_M          = 0.03    # metres
    _STUDY_ANGLE_TOL_DEG      = 10.0    # degrees
    _STUDY_DWELL_S            = 1.0     # seconds within tolerance before auto-complete
    _STUDY_MOVE_THRESHOLD_MPS = 0.01    # m/s — freedrive movement-segment detector
    _STUDY_TRAJ_SAMPLE_HZ     = 10.0

    def __init__(self, quest_ip: str, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 hand_port: int, use_calibrated_robot_base: bool,
                 session_name: str, mode: str, seed: int, out_dir: Path):
        self.quest_ip           = quest_ip
        self.anchor_marker_id   = anchor_marker_id
        self.pegboard_marker_id = pegboard_marker_id
        self.hand_port          = hand_port
        self.session_name       = session_name
        self.mode                = mode
        self._freedrive_enabled = _CHANNELS[mode]["freedrive"]
        self._ar_enabled        = _CHANNELS[mode]["ar"]

        # ── Receivers (same construction as workholding_testing.py) ──────────
        self.cam = _CamFeedReceiver(quest_ip)
        _aruco = _ArucoPoseEstimator(
            anchor_marker_id       = anchor_marker_id,
            pegboard_marker_id     = pegboard_marker_id,
            anchor_marker_size_m   = anchor_marker_size_m,
            pegboard_marker_size_m = pegboard_marker_size_m,
            board_marker_ids       = (),
            board_marker_size_m    = cfg.WORLD_MARKER_SIZE)
        self.aruco_worker     = _ArUcoWorker(self.cam, _aruco)
        self.hands            = _HandDataReceiver(quest_ip, hand_port)
        self.anchor           = _WorldAnchor(quest_ip)
        self.tools            = _ToolSelectionManager(quest_ip)
        self.tuner            = _OffsetTuner()
        self.relock_cubes     = _RelockCubePublisher(quest_ip)
        self.relock_cubes.set_markers(self.anchor._T_world_marker)
        self.workspace_bound_pub = _WorkspaceBoundPublisher(quest_ip)

        # Ghost target box — always visible, current trial's target, all modes.
        self.ghost_bridge = _GripPoseBridge(quest_ip, grip_state_port=cfg.WORKHOLDING_BOX_PORT)
        # Interactive AR handle (grab/drag/release) — same port pair main_with_robot.py
        # uses for its board-AR flow. Only driven when self._ar_enabled.
        self.ar_bridge = _GripPoseBridge(quest_ip)

        # ── Robot control client (talks to robot_control_server.py) ──────────
        try:
            self.robot = RobotClient(
                simulation                = False,
                use_calibrated_robot_base = use_calibrated_robot_base)
        except Exception as e:
            raise RuntimeError(
                f"RobotClient failed to connect — is robot_control_server.py running? ({e})")

        self._ws_lo = np.array(cfg.WORKSPACE_LO, dtype=np.float64)
        self._ws_hi = np.array(cfg.WORKSPACE_HI, dtype=np.float64)
        self._BOUNDS_VIS_DIST = 0.5

        # ── Trial plan: the 10 poses, shuffled (seeded) ───────────────────────
        self._poses_raw = cfg.workholding_test_poses()
        self._poses_T   = [self._pose_to_T(pos, euler) for pos, euler in self._poses_raw]
        rng = random.Random(seed)
        self._pose_order = list(range(len(self._poses_raw)))
        rng.shuffle(self._pose_order)
        self._trial_cursor = 0

        # ── Lock/relock state (ported from workholding_testing.py) ───────────
        self._last_proximity_relock_time = 0.0
        self._anchor_highlight_until      = 0.0
        self._prev_relock_available       = False

        # ── Study state machine ───────────────────────────────────────────────
        self._study_started = False
        self._phase          = "lock_anchor"
        self._last_phase     = None
        self._default_tcp_pos: "np.ndarray | None"  = None
        self._default_tcp_quat: "np.ndarray | None" = None
        self._auto_move_pending = False
        self._auto_move_result: "bool | None" = None
        self._release_armed_sent = False
        self._release_pull_sent  = False
        self._quit = False

        self._trial_target_T: "np.ndarray | None" = None
        self._trial_start_t   = 0.0
        self._trial_dwell_start: "float | None" = None
        self._trial_interactions = 0
        self._trial_last_traj_t  = 0.0
        self._prev_tcp_pos_for_speed: "np.ndarray | None" = None
        self._prev_tcp_t_for_speed:  "float | None"       = None
        self._was_moving_freedrive = False
        self._force_complete_requested = False

        # ── Logging ────────────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._trials_path = out_dir / f"{session_name}-{mode}_{stamp}_trials.csv"
        self._traj_path   = out_dir / f"{session_name}-{mode}_{stamp}_trajectory.csv"
        self._trials_f = open(self._trials_path, "w", newline="")
        self._trials_writer = csv.writer(self._trials_f)
        self._trials_writer.writerow(_TRIAL_CSV_HEADER)
        self._traj_f = open(self._traj_path, "w", newline="")
        self._traj_writer = csv.writer(self._traj_f)
        self._traj_writer.writerow(_TRAJ_CSV_HEADER)
        print(f"[Study] Logging trials  → {self._trials_path}")
        print(f"[Study] Logging traj.   → {self._traj_path}")
        print(f"[Study] Mode: {mode}  (freedrive={'ON' if self._freedrive_enabled else 'off'}, "
              f"AR={'ON' if self._ar_enabled else 'off'})  "
              f"— {len(self._pose_order)} trials, seed={seed}")

        self._win = ("Workholding Study  "
                     "[ENTER=lock/relock  F=force-complete  ESC=quit]")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pose_to_T(pos, euler_deg) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('xyz', euler_deg, degrees=True).as_matrix()
        T[:3, 3]  = np.asarray(pos, dtype=np.float64)
        return T

    @staticmethod
    def _pose_error(T_a: np.ndarray, T_b: np.ndarray) -> "tuple[float, float]":
        pos_err = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
        R_err   = T_a[:3, :3].T @ T_b[:3, :3]
        ang_err = float(np.degrees(ScipyR.from_matrix(R_err).magnitude()))
        return pos_err, ang_err

    def _compute_default_tcp(self) -> "tuple[np.ndarray, np.ndarray]":
        """FK of cfg.ROBOT_DEFAULT_JOINT_DEG via the client's local (IK-free)
        pb_scene mirror — save/restore current_q so live visualization is
        undisturbed. Same trick as robot_control_server.execute_grasp's
        board_normal approach-waypoint computation."""
        scene = self.robot.pb_scene
        saved_q = scene.current_q.copy()
        try:
            scene.update_robot(np.deg2rad(cfg.ROBOT_DEFAULT_JOINT_DEG))
            T = scene.update_tcp_bodies()
        finally:
            scene.update_robot(saved_q)
        pos  = T[:3, 3].copy()
        quat = ScipyR.from_matrix(T[:3, :3]).as_quat()
        return pos, quat

    def _lock_initial(self, T_cam_anchor: np.ndarray) -> None:
        """First-time marker-100 lock (mirrors MainScene._lock_anchor_initial)."""
        self.anchor.lock(T_cam_anchor, self.cam.camera_T)
        self._last_proximity_relock_time = time.time()
        self.robot.set_scene_origin(np.eye(4))

    # ── Study state machine ─────────────────────────────────────────────────

    def _on_auto_move_complete(self, ok: bool) -> None:
        self._auto_move_pending = False
        self._auto_move_result  = bool(ok)

    def _start_auto_move(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self._auto_move_pending = True
        self._auto_move_result  = None
        self.robot.move_to_pose(pos, quat, board_move=True,
                                on_complete=self._on_auto_move_complete)

    def _begin_trial(self) -> None:
        pose_idx = self._pose_order[self._trial_cursor]
        self._trial_target_T          = self._poses_T[pose_idx]
        self._trial_start_t           = time.time()
        self._trial_dwell_start       = None
        self._trial_interactions      = 0
        self._trial_last_traj_t       = 0.0
        self._prev_tcp_pos_for_speed  = None
        self._prev_tcp_t_for_speed    = None
        self._was_moving_freedrive    = False
        self._force_complete_requested = False
        pos, euler = self._poses_raw[pose_idx]
        n = len(self._pose_order)
        print(f"[Trial] {self.mode} {self._trial_cursor + 1}/{n}  "
              f"pose#{pose_idx}  pos={np.round(pos, 3).tolist()}  euler={euler}")
        self._phase = "trial_running"

    def _start_next_trial_or_finish(self) -> None:
        if self._trial_cursor >= len(self._pose_order):
            self._phase = "release_board"
            return
        self._begin_trial()

    def _finish_trial(self, reason: str, pos_err: float, ang_err: float) -> None:
        now      = time.time()
        pose_idx = self._pose_order[self._trial_cursor]
        pos, euler = self._poses_raw[pose_idx]
        duration = now - self._trial_start_t
        self._trials_writer.writerow([
            self.session_name, self.mode, self._trial_cursor, pose_idx,
            pos[0], pos[1], pos[2], euler[0], euler[1], euler[2],
            self._trial_start_t, now, duration,
            pos_err, ang_err, self._trial_interactions, reason,
        ])
        self._trials_f.flush()
        print(f"[Trial] done ({reason}) — {duration:.1f}s, "
              f"err={pos_err * 100:.1f}cm/{ang_err:.1f}deg, "
              f"interactions={self._trial_interactions}")
        self._trial_cursor += 1
        self._phase = "reset_to_default"

    def _sample_trajectory(self, now: float) -> None:
        if now - self._trial_last_traj_t < 1.0 / self._STUDY_TRAJ_SAMPLE_HZ:
            return
        T_tcp = self.robot.tcp_pose
        q     = self.robot.q
        if T_tcp is None or q is None:
            return
        self._trial_last_traj_t = now
        quat = ScipyR.from_matrix(T_tcp[:3, :3]).as_quat()
        self._traj_writer.writerow([
            self.session_name, self.mode, self._trial_cursor,
            now - self._trial_start_t,
            *T_tcp[:3, 3].tolist(), *quat.tolist(), *np.degrees(q).tolist(),
        ])

    def _tick_freedrive_channel(self, now: float) -> None:
        T_tcp = self.robot.tcp_pose
        if T_tcp is None or self._trial_target_T is None:
            return
        pos_err, ang_err = self._pose_error(T_tcp, self._trial_target_T)
        within = (pos_err < self._STUDY_POS_TOL_M
                 and ang_err < self._STUDY_ANGLE_TOL_DEG)
        if within:
            if self._trial_dwell_start is None:
                self._trial_dwell_start = now
            elif now - self._trial_dwell_start >= self._STUDY_DWELL_S:
                self._finish_trial("converged", pos_err, ang_err)
                return
        else:
            self._trial_dwell_start = None

        pos = T_tcp[:3, 3]
        if (self._prev_tcp_pos_for_speed is not None
                and self._prev_tcp_t_for_speed is not None):
            dt = now - self._prev_tcp_t_for_speed
            if dt > 1e-3:
                speed  = float(np.linalg.norm(pos - self._prev_tcp_pos_for_speed)) / dt
                moving = speed > self._STUDY_MOVE_THRESHOLD_MPS
                if moving and not self._was_moving_freedrive:
                    self._trial_interactions += 1
                self._was_moving_freedrive = moving
        self._prev_tcp_pos_for_speed = pos.copy()
        self._prev_tcp_t_for_speed   = now

    def _tick_ar_channel(self, now: float) -> None:
        board_state = self.robot.board_state
        move_active = board_state == "moving_board" or self._auto_move_pending
        grip_state  = "moving" if move_active else (
            "grabbed" if board_state == "holding_board" else "idle")
        T_tcp = self.robot.tcp_pose
        if T_tcp is not None:
            self.ar_bridge.publish(grip_state, T_tcp)

        if self._auto_move_pending:
            return
        if self._auto_move_result is not None:
            ok = self._auto_move_result
            self._auto_move_result = None
            if ok and T_tcp is not None and self._trial_target_T is not None:
                pos_err, ang_err = self._pose_error(T_tcp, self._trial_target_T)
                if (pos_err < self._STUDY_POS_TOL_M
                        and ang_err < self._STUDY_ANGLE_TOL_DEG):
                    self._finish_trial("converged", pos_err, ang_err)
                    return
                print(f"[AR] Landed {pos_err*100:.1f}cm/{ang_err:.1f}deg from target "
                      f"— keep adjusting")
            elif not ok:
                print("[AR] Move cancelled/failed — try again")

        T_box_target = self.ar_bridge.poll()
        if T_box_target is not None:
            self._trial_interactions += 1
            tcp_pos  = (T_box_target[:3, 3]
                       - cfg.BOX_FORWARD_OFFSET * T_box_target[:3, 2])
            tcp_quat = ScipyR.from_matrix(T_box_target[:3, :3]).as_quat()
            print(f"[AR] Release #{self._trial_interactions} "
                  f"→ TCP {np.round(tcp_pos, 3).tolist()}")
            self._start_auto_move(tcp_pos, tcp_quat)

    def _tick_study(self, now: float) -> None:
        entering = self._phase != self._last_phase
        self._last_phase = self._phase

        if self._phase == "lock_anchor":
            return  # transition triggered from run() once self.anchor.locked

        if self._phase == "await_robot_ready":
            if entering:
                print("[Study] Waiting for robot to reach its default pose...")
            if self.robot.q is not None and not self.robot.move_running:
                self._default_tcp_pos, self._default_tcp_quat = self._compute_default_tcp()
                self.robot.start_board_interaction(freedrive=self._freedrive_enabled)
                self._phase = "await_grasp"

        elif self._phase == "await_grasp":
            if entering:
                print("[Study] Insert/attach the board...")
            if self.robot.board_state == "holding_board":
                print("[Study] Board grasped — beginning trials")
                self._phase = "reset_to_default"

        elif self._phase == "reset_to_default":
            if not self._auto_move_pending and self._auto_move_result is None:
                self._start_auto_move(self._default_tcp_pos, self._default_tcp_quat)
            elif self._auto_move_result is not None:
                ok = self._auto_move_result
                self._auto_move_result = None
                if ok:
                    self._start_next_trial_or_finish()
                else:
                    print("[Study] Reset-to-default move failed — retrying")

        elif self._phase == "trial_running":
            self._sample_trajectory(now)
            if self._freedrive_enabled:
                self._tick_freedrive_channel(now)
                if self._phase != "trial_running":
                    return
            if self._ar_enabled:
                self._tick_ar_channel(now)
                if self._phase != "trial_running":
                    return
            if self._force_complete_requested:
                self._force_complete_requested = False
                T_tcp = self.robot.tcp_pose
                if T_tcp is not None and self._trial_target_T is not None:
                    pos_err, ang_err = self._pose_error(T_tcp, self._trial_target_T)
                else:
                    pos_err, ang_err = float('nan'), float('nan')
                self._finish_trial("forced", pos_err, ang_err)

        elif self._phase == "release_board":
            if not self._release_armed_sent:
                self._release_armed_sent = True
                print("[Study] All trials complete — releasing board...")
                self.robot.set_board_freedrive(False)
                self.robot.arm_board_release()
            elif self.robot.board_state == "release_armed" and not self._release_pull_sent:
                self._release_pull_sent = True
                print("[Study] Pull the board to release it.")
            elif self.robot.board_state == "inactive":
                print("[Study] Session complete — board released.")
                self._phase = "done"

        elif self._phase == "done":
            self._quit = True

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while not self._quit:
                self.robot.poll()
                self.tools.poll(timeout_ms=0)
                self.hands.poll()

                T_cam_anchor, T_cam_pegboard, _T_cam_board, det_vis = self.aruco_worker.get()
                anchor_ok = T_cam_anchor is not None

                self.tuner.draw()
                pos_off, yaw_off = self.tuner.get()
                self.anchor.set_offset(pos_off, yaw_off)

                _now = time.time()

                if self.anchor.locked and not self._study_started:
                    self._study_started = True
                    self._phase = "await_robot_ready"

                if self.anchor.locked:
                    self.anchor.publish()
                    self.relock_cubes.publish()
                    self.workspace_bound_pub.publish(
                        self._ws_lo, self._ws_hi, self._BOUNDS_VIS_DIST)
                    if self._trial_target_T is not None:
                        T = self._trial_target_T
                        T_fake_tcp = np.eye(4)
                        T_fake_tcp[:3, :3] = T[:3, :3]
                        T_fake_tcp[:3, 3]  = T[:3, 3] - cfg.BOX_FORWARD_OFFSET * T[:3, 2]
                        self.ghost_bridge.publish("grabbed", T_fake_tcp)
                    self._tick_study(_now)

                dist_to_anchor = (float(np.linalg.norm(T_cam_anchor[:3, 3]))
                                  if anchor_ok and self.cam.camera_T is not None
                                  else float('inf'))

                _cos_tilt = (T_cam_anchor[2, 3] / dist_to_anchor
                            if anchor_ok and dist_to_anchor > 1e-6 else 0.0)
                _min_cos  = np.cos(np.deg2rad(self._AUTO_LOCK_MAX_TILT_DEG))
                if (not self.anchor.locked and anchor_ok
                        and self.cam.camera_T is not None
                        and dist_to_anchor < self._AUTO_LOCK_MAX_DIST
                        and _cos_tilt > _min_cos):
                    self._lock_initial(T_cam_anchor)
                    tilt_deg = float(np.degrees(np.arccos(np.clip(_cos_tilt, -1, 1))))
                    print(f"[AutoLock] Locked world to marker #{self.anchor_marker_id} "
                          f"({dist_to_anchor:.2f} m, {tilt_deg:.1f} deg tilt)")

                _relock_available = (self.anchor.locked and anchor_ok
                                     and self.cam.camera_T is not None
                                     and dist_to_anchor < 1.0)
                if self._anchor_highlight_until > 0.0 and _now >= self._anchor_highlight_until:
                    self._anchor_highlight_until = 0.0
                    self._prev_relock_available  = not _relock_available
                if (self._anchor_highlight_until == 0.0
                        and _relock_available != self._prev_relock_available):
                    self.tools.send_color(
                        self.anchor_marker_id,
                        _ToolSelectionManager.HOVER_COLOR if _relock_available
                        else _ToolSelectionManager.RESET_COLOR)
                    self._prev_relock_available = _relock_available
                if (self.tools.active_tool_id == self.anchor_marker_id
                        and _relock_available
                        and _now - self._last_proximity_relock_time >= self._RELOCK_COOLDOWN):
                    self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True)
                    self.tools.send_color(self.anchor_marker_id,
                                          _ToolSelectionManager.SELECTED_COLOR)
                    self._anchor_highlight_until      = _now + 1.0
                    self._prev_relock_available       = True
                    self._last_proximity_relock_time  = _now
                    print("[Relock] Relocked via proximity click")
                self.tools.deselect(self.anchor_marker_id)

                # ── OpenCV display + key input surface ────────────────────────
                disp = cv.resize(
                    det_vis if det_vis is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8),
                    (960, 540))
                locked = self.anchor.locked
                cv.putText(disp,
                           f"Marker #{self.anchor_marker_id}: "
                           f"{'DETECTED' if anchor_ok else 'searching...'}",
                           (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 80) if anchor_ok else (0, 80, 255), 2)
                cv.putText(disp,
                           "Anchor: " + ("LOCKED" if locked
                                         else f"waiting for marker #{self.anchor_marker_id}"),
                           (12, 58), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 150) if locked else (100, 100, 100), 2)
                cv.putText(disp,
                           f"Session: {self.session_name}   Mode: {self.mode}   "
                           f"Phase: {self._phase}",
                           (12, 88), cv.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                if self._phase == "trial_running" and self._trial_cursor < len(self._pose_order):
                    pose_idx = self._pose_order[self._trial_cursor]
                    pos, euler = self._poses_raw[pose_idx]
                    T_tcp = self.robot.tcp_pose
                    err_str = ""
                    if T_tcp is not None and self._trial_target_T is not None:
                        pe, ae = self._pose_error(T_tcp, self._trial_target_T)
                        err_str = f"  err={pe*100:.1f}cm/{ae:.1f}deg"
                    n = len(self._pose_order)
                    cv.putText(disp,
                               f"trial {self._trial_cursor+1}/{n}"
                               f"  target pos={np.round(pos,3).tolist()}",
                               (12, 116), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                    cv.putText(disp,
                               f"elapsed={time.time()-self._trial_start_t:.1f}s"
                               f"  interactions={self._trial_interactions}{err_str}",
                               (12, 142), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                cv.putText(disp,
                           "ENTER lock/relock     F force-complete trial     ESC quit",
                           (12, disp.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                           (200, 200, 200), 1)
                cv.imshow(self._win, disp)

                # ── Key handling ───────────────────────────────────────────────
                key = cv.waitKeyEx(1)
                if key != -1:
                    low = key & 0xFF
                    if low == 27:                                # ESC
                        break
                    elif low in (ord('f'), ord('F')):
                        if self._phase == "trial_running":
                            self._force_complete_requested = True
                    elif low == 13:                               # ENTER
                        if self.cam.camera_T is None:
                            print("[ENTER] No camera pose yet — skipping.")
                        elif anchor_ok:
                            if self.anchor.locked:
                                self.anchor.lock(T_cam_anchor, self.cam.camera_T,
                                                 require_locked=True)
                                self._last_proximity_relock_time = _now
                                print(f"[ENTER] Relocked world to marker #{self.anchor_marker_id}")
                            else:
                                self._lock_initial(T_cam_anchor)
                                print(f"[ENTER] Locked world to marker #{self.anchor_marker_id}")
                        else:
                            print(f"[ENTER] Marker #{self.anchor_marker_id} not visible.")
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.aruco_worker.stop()
        time.sleep(0.05)
        cv.destroyAllWindows()
        try:
            self.tuner.close()
        except Exception:
            pass
        self._trials_f.flush(); self._trials_f.close()
        self._traj_f.flush(); self._traj_f.close()
        print(f"[Study] Completed {self._trial_cursor}/{len(self._pose_order)} trials.")
        print(f"[Study] Trials CSV: {self._trials_path}")
        print(f"[Study] Trajectory CSV: {self._traj_path}")
        for obj in (self.anchor, self.relock_cubes, self.ghost_bridge, self.ar_bridge,
                    self.workspace_bound_pub, self.hands, self.cam, self.tools, self.robot):
            try:
                obj.close()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Freedrive vs AR vs AR+Freedrive board-placement study harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--session-name", required=True,
                    help="Identifies this run; logs are named "
                         "'{session-name}-{mode}_{timestamp}_*.csv'.")
    ap.add_argument("--mode", required=True, choices=_MODES,
                    help="Which interaction condition this session tests. Run the "
                         "script once per mode to cover all three.")
    ap.add_argument("--quest-ip",             default=cfg.UNITY_IP)
    ap.add_argument("--anchor-marker",        type=int,   default=cfg.ANCHOR_MARKER_ID)
    ap.add_argument("--pegboard-marker",      type=int,   default=cfg.PEGBOARD_MARKER_ID)
    ap.add_argument("--anchor-marker-size",   type=float, default=cfg.ANCHOR_MARKER_SIZE)
    ap.add_argument("--pegboard-marker-size", type=float, default=cfg.PEGBOARD_MARKER_SIZE)
    ap.add_argument("--hand-port",            type=int,   default=cfg.HAND1_PORT_FROM_UNITY)
    ap.add_argument("--calibrated-robot-base", action=argparse.BooleanOptionalAction,
                    default=cfg.USE_CALIBRATED_ROBOT_BASE_POSE)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds the pose shuffle (reproducible per run).")
    ap.add_argument("--out-dir", default=str(_DEFAULT_LOG_DIR))
    args = ap.parse_args()

    if args.anchor_marker == args.pegboard_marker:
        ap.error("--anchor-marker and --pegboard-marker must be different.")

    study = WorkholdingStudy(
        quest_ip                  = args.quest_ip,
        anchor_marker_id          = args.anchor_marker,
        pegboard_marker_id        = args.pegboard_marker,
        anchor_marker_size_m      = args.anchor_marker_size,
        pegboard_marker_size_m    = args.pegboard_marker_size,
        hand_port                 = args.hand_port,
        use_calibrated_robot_base = args.calibrated_robot_base,
        session_name              = args.session_name,
        mode                      = args.mode,
        seed                      = args.seed,
        out_dir                   = Path(args.out_dir),
    )
    print(f"\n[Study] Show marker #{args.anchor_marker} to lock the world "
          f"(auto within {WorkholdingStudy._AUTO_LOCK_MAX_DIST:.1f} m, or press ENTER).")
    study.run()
    print("Bye.")


if __name__ == "__main__":
    main()
