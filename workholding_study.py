#!/usr/bin/env python3
"""workholding_study.py — Freedrive vs AR vs AR+Freedrive board-placement study.

A board is clamped in the robot's gripper for one independent session: grasp,
10 trials, release. The robot returns to its configured default pose before
every trial. --mode picks which interaction condition this session
tests:

    freedrive  — physically drag the arm+board by hand. No AR handle.
    ar         — grab and drag the AR box handle in the headset; the robot
                 continuously follows it. No freedrive.
    hybrid     — starts with continuous AR following and freedrive locked out.
                 Clicking the stationary robot gripper (tool id 200) toggles
                 between AR following and freedrive-only control.

Run the script once per mode (e.g. three separate invocations, one per
condition) to cover all three.

The manipulated variable is which control channel(s) are available — a
translucent "ghost" box at the current trial's target pose is shown in ALL
THREE modes, on cfg.WORKHOLDING_BOX_PORT via WorkholdingBoxReceiver.cs (same
mechanism workholding_testing.py already uses to park a static box).

The experimenter presses ENTER once to start recording. Recording ends
automatically after the board remains within 5 cm and 15 degrees of the target
for 1 second. The robot remains stationary until ENTER authorizes the exact-
target snap outside recorded time. A manual override key can force-complete a
running trial for experimenter recovery.

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
    python workholding_study.py --session-name setup --mode freedrive \
        --teach-targets workholding_targets.json
    python workholding_study.py --session-name P01 --mode ar \
        --target-poses-file workholding_targets.json

Keys: P/N preview previous/next target; M mark and U undo in teaching mode;
ENTER starts/finishes recording during a trial and locks/relocks marker 100
outside trials; F force-completes the current trial; ESC quits and flushes logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import open3d as o3d
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
from scene_viewer_o3d import SceneVis as _SceneVis
from study_replay import ReplayRecorder

_FILE_DIR = Path(__file__).resolve().parent
_STUDY_LOG_ROOT = _FILE_DIR / "study_logs"
_DEFAULT_LOG_DIR = _STUDY_LOG_ROOT / "study2"

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
_HAND_TRAJ_CSV_HEADER = [
    "session_name", "mode", "trial_idx", "pose_idx", "t_rel_s",
    "sample_idx", "tracked", "joint_idx",
    "joint_pos_x_m", "joint_pos_y_m", "joint_pos_z_m",
]
_HEAD_TRAJ_CSV_HEADER = [
    "session_name", "mode", "trial_idx", "pose_idx", "t_rel_s",
    "sample_idx", "tracked",
    "head_pos_x_m", "head_pos_y_m", "head_pos_z_m",
    "head_quat_x", "head_quat_y", "head_quat_z", "head_quat_w",
]
_DETAILED_TRIAL_CSV_HEADER = [
    *_TRIAL_CSV_HEADER,
    "start_board_pos_x_m", "start_board_pos_y_m", "start_board_pos_z_m",
    "start_board_euler_x_deg", "start_board_euler_y_deg", "start_board_euler_z_deg",
    "start_pos_error_m", "start_angle_error_deg",
    "first_reach_time", "first_reach_elapsed_s",
    "first_reach_pos_error_m", "first_reach_angle_error_deg",
    "auto_stop_pos_error_m", "auto_stop_angle_error_deg",
    "enter_confirmation_time",
    "enter_confirmation_pos_error_m", "enter_confirmation_angle_error_deg",
    "post_stop_ar_interactions",
    "post_stop_freedrive_interactions", "post_stop_interactions",
    "post_stop_pos_error_improvement_m",
    "post_stop_angle_error_improvement_deg",
    "freedrive_interactions", "ar_interactions",
    "tcp_path_length_m", "tcp_angular_path_length_deg",
    "recording_start_source", "start_policy",
    "snap_success", "snap_duration_s",
    "post_snap_pos_error_m", "post_snap_angle_error_deg",
    "target_poses_file",
]


class _WorkholdingSceneVis(_SceneVis):
    """Open3D study view built on the main robot/hand scene visualizer."""

    _TARGET_DEFAULT_COLOR = [0.92, 0.92, 0.92]
    _TARGET_BLACK_COLOR = [0.0, 0.0, 0.0]
    _TARGET_UNREACHED_COLOR = [0.95, 0.08, 0.08]
    _TARGET_NEAR_COLOR = [1.00, 0.42, 0.02]
    _TARGET_REACHED_COLOR = [0.12, 0.90, 0.20]
    _TARGET_UNREACHABLE_COLOR = [0.95, 0.08, 0.08]
    _MODE_COLORS = {
        "ar": [0.10, 0.90, 0.20],
        "freedrive": [1.00, 0.48, 0.05],
        "hybrid": [0.10, 0.75, 1.00],
    }

    def __init__(self, title: str):
        super().__init__(title, board_ar_asset="HalfBoard.obj",
                         load_gearbox_mirror=False,
                         enable_key_callbacks=True)
        self._target_step_callback = None
        # GLFW key codes: LEFT=263, RIGHT=262. Action callbacks make the key
        # press explicit and work more reliably across Open3D builds.
        self.vis.register_key_action_callback(ord("P"), self._on_previous_target)
        self.vis.register_key_action_callback(ord("N"), self._on_next_target)
        self.vis.register_key_action_callback(263, self._on_previous_target)
        self.vis.register_key_action_callback(262, self._on_next_target)
        self._teach_mark_callback = None
        self._teach_undo_callback = None
        self.vis.register_key_action_callback(ord("M"), self._on_mark_target)
        self.vis.register_key_action_callback(ord("U"), self._on_undo_target)
        self._timer_toggle_callback = None
        # GLFW ENTER and keypad ENTER. The OpenCV window handles its backend-
        # specific Enter variants separately in run().
        self.vis.register_key_action_callback(257, self._on_toggle_timer)
        self.vis.register_key_action_callback(335, self._on_toggle_timer)
        asset_dir = _FILE_DIR / "robot_assets"
        asset = asset_dir / "HalfBoard.obj"
        self._study_target_mesh = None
        self._study_target_T = self._hidden_T()
        self._study_target_color_state: "str | None" = None
        self._study_target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.10)
        self._study_target_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._study_target_frame)
        self._study_target_frame_T = self._hidden_T()
        self._target_ghosts: list = []
        self._reached_target_indices: set[int] = set()
        self._selected_target_index: "int | None" = None
        self._target_board_asset = asset if asset.exists() else None
        if asset.exists():
            mesh = o3d.io.read_triangle_mesh(str(asset))
            if len(mesh.vertices):
                mesh.compute_vertex_normals()
                # HalfBoard.obj is authored directly in the BoardAR frame;
                # preserve its origin and rotate around its own local Y axis.
                T_local = np.eye(4, dtype=np.float64)
                T_local[:3, :3] = ScipyR.from_euler(
                    "y", 90.0, degrees=True).as_matrix()
                mesh.transform(T_local)
                mesh.paint_uniform_color(self._TARGET_UNREACHED_COLOR)
                mesh.transform(self._hidden_T())
                self.vis.add_geometry(mesh)
                self._study_target_mesh = mesh

        self._target_gripper_mesh = None
        self._target_gripper_T = self._hidden_T()
        self._target_gripper_color_state: "str | None" = None
        target_gripper_asset = asset_dir / "RobotiqGripperWithAdapters.obj"
        if target_gripper_asset.exists():
            mesh = o3d.io.read_triangle_mesh(str(target_gripper_asset))
            if len(mesh.vertices):
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(self._TARGET_UNREACHED_COLOR)
                # Preserve the established rigid-OBJ-to-TCP alignment used by
                # the original robot-attached gripper visualization.
                T_local = np.eye(4, dtype=np.float64)
                T_local[:3, :3] = (
                    ScipyR.from_euler("z", 180.0, degrees=True).as_matrix()
                    @ ScipyR.from_euler("x", 90.0, degrees=True).as_matrix())
                mesh.transform(T_local)
                mesh.transform(self._hidden_T())
                self.vis.add_geometry(mesh)
                self._target_gripper_mesh = mesh
                print("[StudyVis] Desired gripper loaded from "
                      "RobotiqGripperWithAdapters.obj")
        else:
            print(f"[StudyVis] Desired gripper asset not found: "
                  f"{target_gripper_asset}")

        self._ar_handle_mesh = None
        self._ar_handle_T = self._hidden_T()
        handle_asset = _FILE_DIR / "robot_assets" / "Handle.stl"
        if handle_asset.exists():
            handle_mesh = o3d.io.read_triangle_mesh(str(handle_asset))
            if len(handle_mesh.vertices):
                handle_mesh.compute_vertex_normals()
                handle_mesh.paint_uniform_color([0.15, 0.75, 1.0])
                handle_mesh.transform(self._hidden_T())
                self.vis.add_geometry(handle_mesh)
                self._ar_handle_mesh = handle_mesh
                print("[StudyVis] AR handle loaded from robot_assets/Handle.stl")

        self._mode_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
        self._mode_sphere.compute_vertex_normals()
        self._mode_sphere.paint_uniform_color(self._MODE_COLORS["freedrive"])
        self._mode_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._mode_sphere)
        self._mode_sphere_T = self._hidden_T()
        self._mode_sphere_state = None

    def configure_target_ghosts(self, target_poses: list[np.ndarray]) -> None:
        """Draw every saved board target in white against the black background."""
        if self._target_board_asset is None:
            return
        base = o3d.io.read_triangle_mesh(str(self._target_board_asset))
        if not len(base.vertices):
            return
        base.compute_vertex_normals()
        T_local = np.eye(4, dtype=np.float64)
        T_local[:3, :3] = ScipyR.from_euler(
            "y", 90.0, degrees=True).as_matrix()
        base.transform(T_local)
        for T_target in target_poses:
            mesh = o3d.geometry.TriangleMesh(base)
            mesh.transform(np.asarray(T_target, dtype=float))
            mesh.paint_uniform_color(self._TARGET_DEFAULT_COLOR)
            self.vis.add_geometry(mesh)
            self._target_ghosts.append(mesh)

    def select_target(self, target_index: int, proximity_state: str) -> None:
        """Color the active target by live board-to-target proximity."""
        previous = self._selected_target_index
        if (previous is not None and previous != target_index
                and previous < len(self._target_ghosts)):
            self._target_ghosts[previous].paint_uniform_color(
                self._TARGET_DEFAULT_COLOR)
            self.vis.update_geometry(self._target_ghosts[previous])
        self._selected_target_index = int(target_index)
        if target_index < len(self._target_ghosts):
            color = {
                "reached": self._TARGET_REACHED_COLOR,
                "near": self._TARGET_NEAR_COLOR,
                "far": self._TARGET_UNREACHED_COLOR,
                "black": self._TARGET_BLACK_COLOR,
            }.get(proximity_state, self._TARGET_DEFAULT_COLOR)
            self._target_ghosts[target_index].paint_uniform_color(color)
            self.vis.update_geometry(self._target_ghosts[target_index])

    def mark_target_reached(self, target_index: int,
                            T_target: np.ndarray) -> None:
        """Keep a reached target visible as a persistent solid green board."""
        if (target_index in self._reached_target_indices
                or target_index >= len(self._target_ghosts)):
            return
        self._reached_target_indices.add(int(target_index))
        mesh = self._target_ghosts[target_index]
        mesh.paint_uniform_color(self._TARGET_REACHED_COLOR)
        self.vis.update_geometry(mesh)

    def set_target_step_callback(self, callback) -> None:
        self._target_step_callback = callback

    def set_teach_callbacks(self, mark_callback, undo_callback) -> None:
        self._teach_mark_callback = mark_callback
        self._teach_undo_callback = undo_callback

    def set_timer_toggle_callback(self, callback) -> None:
        self._timer_toggle_callback = callback

    def _on_previous_target(self, _vis, action, _mods) -> bool:
        if action == 1 and self._target_step_callback is not None:
            self._target_step_callback(-1)
        return False

    def _on_next_target(self, _vis, action, _mods) -> bool:
        if action == 1 and self._target_step_callback is not None:
            self._target_step_callback(+1)
        return False

    def _on_mark_target(self, _vis, action, _mods) -> bool:
        if action == 1 and self._teach_mark_callback is not None:
            self._teach_mark_callback()
        return False

    def _on_undo_target(self, _vis, action, _mods) -> bool:
        if action == 1 and self._teach_undo_callback is not None:
            self._teach_undo_callback()
        return False

    def _on_toggle_timer(self, _vis, action, _mods) -> bool:
        if action == 1 and self._timer_toggle_callback is not None:
            self._timer_toggle_callback()
        return False

    def update_study_target(self, T_target: "np.ndarray | None",
                            reached: bool, reachable: bool) -> None:
        if self._study_target_mesh is None:
            return
        T_new = T_target if T_target is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._study_target_T)
        self._study_target_mesh.transform(delta)
        self._study_target_T = np.array(T_new, dtype=np.float64, copy=True)
        frame_delta = T_new @ np.linalg.inv(self._study_target_frame_T)
        self._study_target_frame.transform(frame_delta)
        self._study_target_frame_T = np.array(
            T_new, dtype=np.float64, copy=True)
        self.vis.update_geometry(self._study_target_frame)
        color_state = ("unreachable" if not reachable
                       else "reached" if reached else "unreached")
        if color_state != self._study_target_color_state:
            color = (self._TARGET_UNREACHABLE_COLOR if not reachable
                     else self._TARGET_REACHED_COLOR if reached
                     else self._TARGET_UNREACHED_COLOR)
            self._study_target_mesh.paint_uniform_color(color)
            self._study_target_color_state = color_state
        self.vis.update_geometry(self._study_target_mesh)

    def update_mode_indicator(self, T_tcp: "np.ndarray | None",
                              mode_state: str) -> None:
        if T_tcp is None:
            T_new = self._hidden_T()
        else:
            T_new = np.array(T_tcp, dtype=np.float64, copy=True)
            # Float above the gripper along TCP-local +Z.
            T_new[:3, 3] += 0.22 * T_new[:3, 2]
        delta = T_new @ np.linalg.inv(self._mode_sphere_T)
        self._mode_sphere.transform(delta)
        self._mode_sphere_T = T_new
        if mode_state != self._mode_sphere_state:
            self._mode_sphere.paint_uniform_color(
                self._MODE_COLORS.get(mode_state, [0.55, 0.55, 0.55]))
            self._mode_sphere_state = mode_state
        self.vis.update_geometry(self._mode_sphere)

    def update_target_gripper(self, T_target_board: "np.ndarray | None",
                              board_offset: float,
                              proximity_state: str) -> None:
        """Show the target TCP gripper using the target board's exact color."""
        if self._target_gripper_mesh is None:
            return
        if T_target_board is None:
            T_new = self._hidden_T()
        else:
            T_new = np.array(T_target_board, dtype=np.float64, copy=True)
            T_new[:3, 3] -= board_offset * T_new[:3, 2]
        delta = T_new @ np.linalg.inv(self._target_gripper_T)
        self._target_gripper_mesh.transform(delta)
        self._target_gripper_T = T_new
        color_state = proximity_state if T_target_board is not None else "hidden"
        if color_state != self._target_gripper_color_state:
            color = {
                "reached": self._TARGET_REACHED_COLOR,
                "near": self._TARGET_NEAR_COLOR,
                "far": self._TARGET_UNREACHED_COLOR,
                "black": self._TARGET_BLACK_COLOR,
            }.get(color_state, self._TARGET_DEFAULT_COLOR)
            self._target_gripper_mesh.paint_uniform_color(color)
            self._target_gripper_color_state = color_state
        self.vis.update_geometry(self._target_gripper_mesh)

    def update_ar_handle(self, T_board: "np.ndarray | None") -> None:
        """Show the handle at board-local [-7.5, -150, 0] mm, then Rx(90)."""
        if self._ar_handle_mesh is None:
            return
        if T_board is None:
            T_new = self._hidden_T()
        else:
            T_new = np.array(T_board, dtype=np.float64, copy=True)
            T_new[:3, 3] -= 0.0075 * T_new[:3, 0]
            T_new[:3, 3] -= 0.1500 * T_new[:3, 1]
            T_new[:3, :3] = (T_new[:3, :3]
                              @ ScipyR.from_euler(
                                  "x", 90.0, degrees=True).as_matrix())
        delta = T_new @ np.linalg.inv(self._ar_handle_T)
        self._ar_handle_mesh.transform(delta)
        self._ar_handle_T = T_new
        self.vis.update_geometry(self._ar_handle_mesh)


class WorkholdingStudy:
    # Same lock/relock tuning as MainScene / workholding_testing.py.
    _RELOCK_COOLDOWN        = 2.0
    _AUTO_LOCK_MAX_DIST     = 1.0    # metres — auto-lock-on-sight only within this range
    _AUTO_LOCK_MAX_TILT_DEG = 45.0   # degrees — max tilt from vertical to auto-lock

    # Trial completion / interaction-counting tuning (independent of the CBF
    # servo tolerances in robot_control_server.py — human placement is noisier).
    _STUDY_POS_TOL_M          = 0.05    # metres
    _STUDY_ANGLE_TOL_DEG      = 15.0    # degrees
    _STUDY_REACH_DWELL_S      = 1.0     # reject momentary tolerance crossings
    _TARGET_NEAR_POS_M         = 0.15    # visual orange proximity band
    _TARGET_NEAR_ANGLE_DEG     = 30.0
    # Strictly for IK candidate validation/preview. These do not change the
    # shared human/AR study completion criterion above.
    _IK_POS_TOL_M             = 0.01
    _IK_ANGLE_TOL_DEG         = 3.0
    _STUDY_MOVE_START_MPS      = 0.01    # begin a freedrive movement segment
    _STUDY_MOVE_STOP_MPS       = 0.004   # lower threshold prevents speed jitter
    _STUDY_ROT_START_DEGPS     = 5.0     # rotation can also begin a segment
    _STUDY_ROT_STOP_DEGPS      = 2.0     # angular-speed hysteresis threshold
    _STUDY_MOVE_STOP_DWELL_S   = 0.40    # stationary time required to end segment
    _AR_FOLLOW_POS_DEADBAND_M  = 0.005   # suppress hand-tracking position jitter
    _AR_FOLLOW_ANGLE_DEADBAND_DEG = 2.0  # suppress hand-tracking rotation jitter
    _TCP_TOOL_ID = _ToolSelectionManager.TCP_TOOL_ID
    # Live robot-gripper mode indicator. These are forced colors so hover and
    # selection feedback cannot obscure the current control mode.
    _AR_GRIPPER_RGBA = [0.0, 0.0, 0.0, 1.0]
    _FREEDRIVE_GRIPPER_RGBA = [1.0, 0.0, 0.0, 1.0]
    # The participant-controlled AR assembly is cyan in every AR-capable
    # condition; target/proximity colors are published on the separate ghost
    # channel and remain independent.
    _AR_ASSEMBLY_RGBA = {
        "ar": [0.0, 1.0, 1.0, 0.70],
        "hybrid": [0.0, 1.0, 1.0, 0.70],
    }
    _PALM_CBF_RADIUS_M = 0.03       # 6 cm diameter hand obstacle
    _PALM_CBF_CLEARANCE_M = 0.02
    _PALM_CBF_PUBLISH_INTERVAL_S = 1.0 / 30.0
    _STUDY_TRAJ_SAMPLE_HZ     = 10.0
    _REPLAY_SAMPLE_HZ         = 30.0

    # Robustness against robot_control_server.py not being up yet (or dropping a
    # command sent before its ZMQ socket finished connecting — PUB/SUB messages
    # sent before the peer connects are silently lost, not queued).
    _ROBOT_READY_NAG_INTERVAL_S = 5.0   # re-print "still waiting" while disconnected
    _GRASP_RETRY_INTERVAL_S     = 3.0   # resend start_board_interaction if board_state
                                         # never leaves "inactive"

    def __init__(self, quest_ip: str, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 hand_port: int, use_calibrated_robot_base: bool,
                 session_name: str, mode: str, out_dir: Path,
                 teach_targets_path: "Path | None" = None,
                 target_poses_path: "Path | None" = None,
                 target_navigation: str = "preview",
                 replay_log_path: "Path | None" = None,
                 resume: bool = True):
        self.quest_ip           = quest_ip
        self.anchor_marker_id   = anchor_marker_id
        self.pegboard_marker_id = pegboard_marker_id
        self.hand_port          = hand_port
        self.session_name       = session_name
        self.mode                = mode
        self._target_poses_path = target_poses_path
        self._target_poses_source = str(target_poses_path or "built_in")
        self._target_navigation = target_navigation
        self._teach_targets_path = teach_targets_path
        self._teach_mode = teach_targets_path is not None
        self._taught_poses: list[dict] = []
        if (self._teach_targets_path is not None
                and self._teach_targets_path.exists() and resume):
            self._taught_poses = self._load_target_records(
                self._teach_targets_path)
            print(f"[Teach] Resuming with {len(self._taught_poses)} target(s)")
        self._freedrive_enabled = (True if self._teach_mode
                                   else _CHANNELS[mode]["freedrive"])
        self._ar_enabled        = (False if self._teach_mode
                                   else _CHANNELS[mode]["ar"])

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

        # ── Trial plan: preserve the target-file/configuration order ─────────
        if target_poses_path is not None:
            target_records = self._load_target_records(target_poses_path)
            self._target_records = target_records
            self._poses_raw = [
                (record["position_m"], record["euler_xyz_deg"])
                for record in target_records
            ]
            self._target_joints = [
                (np.asarray(record["joint_positions_rad"], dtype=float)
                 if "joint_positions_rad" in record else None)
                for record in target_records
            ]
        else:
            self._poses_raw = cfg.workholding_test_poses()
            self._target_records = []
            self._target_joints = [None] * len(self._poses_raw)
        if target_poses_path is None and self._poses_raw:
            # Built-in targets describe BOARD poses. Derive Target 1 from the
            # configured default TCP using the same board mounting transform
            # used by motion, status checking, Unity, and Open3D.
            default_tcp_pos, default_tcp_quat = self._compute_default_tcp()
            T_default_tcp = np.eye(4, dtype=np.float64)
            T_default_tcp[:3, :3] = ScipyR.from_quat(
                default_tcp_quat).as_matrix()
            T_default_tcp[:3, 3] = default_tcp_pos
            T_default_board = self._board_pose_from_tcp(T_default_tcp)
            default_board_pos = T_default_board[:3, 3].tolist()
            default_board_euler = ScipyR.from_matrix(
                T_default_board[:3, :3]).as_euler(
                    "xyz", degrees=True).tolist()
            self._poses_raw[0] = (default_board_pos, default_board_euler)
            print("[Study] Target 1 set from default robot TCP: "
                  f"board={np.round(default_board_pos, 4).tolist()}  "
                  f"euler={np.round(default_board_euler, 2).tolist()}")
        self._poses_T   = [self._pose_to_T(pos, euler) for pos, euler in self._poses_raw]
        self._pose_order = list(range(len(self._poses_raw)))
        self._trial_cursor = 0
        self._target_preview_cursor = 0
        self._manual_target_preview = False
        self._target_proximity_state = "far"
        self._completion_flash_state = None
        self._preview_robot_link_poses: "list[np.ndarray] | None" = None
        self._preview_robot_tcp_T: "np.ndarray | None" = None
        self._target_reachability = ({}
                                     if self._teach_mode
                                     else self._precheck_target_reachability())

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
        self._robot_ready_last_nag = 0.0
        self._grasp_request_t      = 0.0

        self._trial_target_T: "np.ndarray | None" = None
        self._trial_start_t   = 0.0
        self._trial_timer_running = False
        self._trial_active_elapsed_s = 0.0
        self._trial_timer_last_t: "float | None" = None
        self._trial_reach_dwell_start: "float | None" = None
        self._trial_first_reach_time: "float | None" = None
        self._trial_first_reach_elapsed_s = float("nan")
        self._trial_first_reach_pos_error_m = float("nan")
        self._trial_first_reach_angle_error_deg = float("nan")
        self._trial_interactions = 0
        self._trial_freedrive_interactions = 0
        self._trial_ar_interactions = 0
        self._post_stop_ar_interactions = 0
        self._post_stop_freedrive_interactions = 0
        self._post_stop_interactions = 0
        self._post_stop_baseline_pos_error_m = float("nan")
        self._post_stop_baseline_angle_error_deg = float("nan")
        self._post_stop_freedrive_active = False
        self._post_stop_freedrive_start_errors = None
        self._ar_follow_last_board_T: "np.ndarray | None" = None
        self._hybrid_freedrive_only = False
        self._last_palm_cbf_pub_time = 0.0
        self._palm_cbf_active = False
        self._trial_tcp_path_length_m = 0.0
        self._trial_tcp_angular_path_length_deg = 0.0
        self._trial_path_prev_tcp_pos: "np.ndarray | None" = None
        self._trial_path_prev_tcp_rot: "np.ndarray | None" = None
        self._trial_recording_start_source = ""
        self._trial_start_policy = ""
        self._trial_start_board_T: "np.ndarray | None" = None
        self._trial_start_pos_error_m = float("nan")
        self._trial_start_angle_error_deg = float("nan")
        self._pending_trial_summary: "dict | None" = None
        self._snap_started_t: "float | None" = None
        self._trial_last_traj_t  = 0.0
        self._prev_tcp_pos_for_speed: "np.ndarray | None" = None
        self._prev_tcp_rot_for_speed: "np.ndarray | None" = None
        self._prev_tcp_t_for_speed:  "float | None"       = None
        self._was_moving_freedrive = False
        self._freedrive_stationary_since: "float | None" = None
        self._force_complete_requested = False
        self._completion_flash_started: "float | None" = None
        self._completion_flash_state: "str | None" = None
        self._status_trial_cursor: "int | None" = None   # for the inline live-offset line
        self._last_status_print_t = 0.0
        self._last_replay_t = 0.0
        self._last_replay_flush_t = 0.0
        self._session_id = f"{session_name}-{mode}-{int(time.time() * 1000)}"

        # ── Logging ────────────────────────────────────────────────────────
        # Everything lives in one replay JSONL per participant, shared
        # across all modes/conditions — every record carries a "mode"
        # field, so conditions stay distinguishable within the merged file.
        # Use study2_replay_to_csv.py to derive tabular CSVs on demand.
        out_dir.mkdir(parents=True, exist_ok=True)
        self._replay_path = (replay_log_path if replay_log_path is not None
                             else out_dir / f"{session_name}_replay.jsonl")
        self._replay_path.parent.mkdir(parents=True, exist_ok=True)
        self._trial_cursor = self._prepare_mode_replay(
            self._replay_path, mode,
            resume=bool(resume and not self._teach_mode))
        self._target_preview_cursor = (
            0 if self._trial_cursor >= len(self._pose_order)
            else self._trial_cursor)
        self._replay = ReplayRecorder(
            self._replay_path, "workholding_replay_v1", self._session_id,
            session_name=self.session_name, mode=self.mode)
        print("[Study] Logs are shared across modes for this participant "
              "— each record is distinguished by its 'mode' field.")
        if self._trial_cursor >= len(self._pose_order):
            print(f"[Study] All {len(self._pose_order)} {mode} trials are "
                  "already complete; target preview remains available.")
        elif self._trial_cursor:
            print(f"[Study] Resuming {mode} at trial "
                  f"{self._trial_cursor + 1}/{len(self._pose_order)}; "
                  "unfinished-trial records were removed.")
        else:
            print(f"[Study] Starting {mode} at trial 1; any unfinished "
                  "records for this mode were removed.")
        print(f"[Study] Logging replay → {self._replay_path}")
        print(f"[Study] Mode: {mode}  (freedrive={'ON' if self._freedrive_enabled else 'off'}, "
              f"AR={'ON' if self._ar_enabled else 'off'})  "
              f"— {len(self._pose_order)} trials in target-file order")
        print(f"[Study] P/N target navigation: {self._target_navigation}")

        self._win = ("Workholding Study  "
                     "[ENTER=start/snap  F=force-complete  ESC=quit]")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)
        self.vis = _WorkholdingSceneVis(
            f"Workholding Study — {session_name} — {mode}")
        if not self._teach_mode:
            self.vis.configure_target_ghosts(self._poses_T)
        self.vis.set_target_step_callback(self._step_target_preview)
        self.vis.set_teach_callbacks(
            self._mark_taught_target, self._undo_taught_target)
        self.vis.set_timer_toggle_callback(self._toggle_trial_timer)
        self._replay_event(
            "session_start", target_navigation=target_navigation,
            pose_order=self._pose_order, target_poses=self._poses_T,
            workspace_lo=self._ws_lo, workspace_hi=self._ws_hi,
            calibrated_robot_base=use_calibrated_robot_base)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_mode_replay(path: Path, mode: str, resume: bool) -> int:
        """Clean one mode's replay records and return its resume trial.

        A trial is complete only when its ``trial_summary`` event exists.
        Resume retains the contiguous completed prefix and drops every record
        for the first unfinished trial and later trials. A fresh start drops
        all records for the selected mode. Other modes are always preserved.
        """
        if not path.exists() or path.stat().st_size == 0:
            return 0
        parsed: list[tuple[str, "dict | None"]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                parsed.append((line, record))

        completed = set()
        if resume:
            for _line, record in parsed:
                if (record is not None and record.get("mode") == mode
                        and record.get("type") == "interaction"
                        and record.get("event") == "trial_summary"):
                    try:
                        completed.add(int(record["trial_idx"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        resume_trial = 0
        while resume_trial in completed:
            resume_trial += 1

        kept, removed = [], 0
        for line, record in parsed:
            if record is None or record.get("mode") != mode:
                kept.append(line)
                continue
            try:
                trial_idx = int(record.get("trial_idx", -1))
            except (TypeError, ValueError):
                trial_idx = -1
            if resume and 0 <= trial_idx < resume_trial:
                kept.append(line)
            else:
                removed += 1
        if removed:
            with path.open("w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
            action = "incomplete/later" if resume else "pre-existing"
            print(f"[Study] Removed {removed} {action} '{mode}' replay "
                  f"record(s) from {path.name}; other modes were kept.")
        return resume_trial

    def _replay_event(self, event: str, **payload) -> None:
        self._replay.record(
            "interaction", trial_idx=self._trial_cursor, phase=self._phase,
            event=event, **payload)

    def _sample_replay(self, now: float) -> None:
        if now - self._last_replay_t < 1.0 / self._REPLAY_SAMPLE_HZ:
            return
        self._last_replay_t = now
        T_tcp = self.robot.tcp_pose
        T_board = self._board_pose_from_tcp(T_tcp)
        head_T = None
        left_pts = right_pts = None
        if self.anchor.locked:
            center_eye = self.hands.center_eye_T()
            head_T = (self.anchor.world_T(center_eye)
                      if center_eye is not None else None)
            left_pts, right_pts = self.hands.world_joints(
                self.anchor.T_world_tracking)
        target_idx = None
        if self._trial_cursor < len(self._pose_order):
            target_idx = self._pose_order[self._trial_cursor]
        self._replay.record(
            "frame", trial_idx=self._trial_cursor,
            pose_idx=target_idx, phase=self._phase,
            timer_running=self._trial_timer_running,
            trial_elapsed_s=self._trial_elapsed(now),
            interaction_count=self._trial_interactions,
            freedrive_interaction_count=self._trial_freedrive_interactions,
            ar_interaction_count=self._trial_ar_interactions,
            post_stop_ar_interaction_count=self._post_stop_ar_interactions,
            post_stop_freedrive_interaction_count=
                self._post_stop_freedrive_interactions,
            post_stop_interaction_count=self._post_stop_interactions,
            tcp_path_length_m=self._trial_tcp_path_length_m,
            tcp_angular_path_length_deg=self._trial_tcp_angular_path_length_deg,
            recording_start_source=self._trial_recording_start_source,
            robot_q_rad=self.robot.q, tcp_world_T=T_tcp,
            robot_link_world_T=self.robot.arm_link_poses(),
            board_world_T=T_board, target_board_world_T=self._trial_target_T,
            head_world_T=head_T, left_hand_world=left_pts,
            right_hand_world=right_pts,
            world_tracking_T=self.anchor.T_world_tracking,
            robot_board_state=self.robot.board_state,
            robot_move_running=self.robot.move_running,
            freedrive_enabled=self._freedrive_enabled,
            ar_enabled=self._ar_enabled,
            ar_move_pending=self._auto_move_pending,
            target_color_state=self._target_proximity_state,
            completion_flash_state=self._completion_flash_state,
            target_navigation=self._target_navigation)

    @staticmethod
    def _pose_to_T(pos, euler_deg) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('xyz', euler_deg, degrees=True).as_matrix()
        T[:3, 3]  = np.asarray(pos, dtype=np.float64)
        return T

    @staticmethod
    def _load_target_records(path: Path) -> list[dict]:
        try:
            payload = json.loads(path.read_text())
            records = payload.get("poses", payload) if isinstance(payload, dict) else payload
            normalized = []
            for index, record in enumerate(records):
                if isinstance(record, dict):
                    pos = record["position_m"]
                    euler = record["euler_xyz_deg"]
                    joints = record.get("joint_positions_rad")
                else:
                    pos, euler = record
                    joints = None
                if len(pos) != 3 or len(euler) != 3:
                    raise ValueError("each target requires three position and Euler values")
                item = {
                    "target_index": index,
                    "position_m": [float(v) for v in pos],
                    "euler_xyz_deg": [float(v) for v in euler],
                }
                if joints is not None:
                    values = [float(v) for v in joints]
                    if len(values) != 6 or not np.all(np.isfinite(values)):
                        raise ValueError(
                            "joint_positions_rad must contain six finite values")
                    item["joint_positions_rad"] = values
                normalized.append(item)
            if not normalized:
                raise ValueError("target file contains no poses")
            fixed_count = sum("joint_positions_rad" in item
                              for item in normalized)
            print(f"[Study] Loaded {len(normalized)} taught target(s) from {path} "
                  f"({fixed_count} with fixed joint angles)")
            return normalized
        except Exception as exc:
            raise RuntimeError(f"Could not load target poses from {path}: {exc}") from exc

    @staticmethod
    def _load_target_poses(path: Path):
        """Backward-compatible pose-only view of the target file."""
        return [(record["position_m"], record["euler_xyz_deg"])
                for record in WorkholdingStudy._load_target_records(path)]

    def _store_generated_target_joints(self) -> None:
        """Atomically add generated fixed joint angles to the target JSON."""
        if self._target_poses_path is None or not self._target_records:
            return
        for index, joints in enumerate(self._target_joints):
            if joints is not None:
                self._target_records[index]["joint_positions_rad"] = [
                    float(value) for value in joints]
        payload = {
            "format": "coassembly_workholding_board_targets_v2",
            "box_forward_offset_m": cfg.BOX_FORWARD_OFFSET,
            "poses": self._target_records,
        }
        temp_path = self._target_poses_path.with_suffix(
            self._target_poses_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n")
        temp_path.replace(self._target_poses_path)

    def _historical_target_joint_seeds(self, T_board: np.ndarray) -> list[np.ndarray]:
        """Find prior snap solutions for this exact target pose as IK seeds."""
        if self._target_poses_path is None:
            return []
        seeds = []
        for replay_path in self._target_poses_path.parent.glob("*_replay.jsonl"):
            try:
                lines = replay_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                    saved_target = np.asarray(
                        record.get("target_board_world_T"), dtype=float)
                    joints = np.asarray(record.get("target_joints_rad"), dtype=float)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if (record.get("event") == "exact_target_snap_started"
                        and saved_target.shape == (4, 4)
                        and joints.shape == (6,)
                        and np.all(np.isfinite(joints))
                        and np.allclose(saved_target, T_board,
                                        atol=1e-8, rtol=1e-8)):
                    seeds.append(joints)
        return seeds

    def _save_taught_targets(self) -> None:
        if self._teach_targets_path is None:
            return
        self._teach_targets_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "coassembly_workholding_board_targets_v1",
            "box_forward_offset_m": cfg.BOX_FORWARD_OFFSET,
            "poses": self._taught_poses,
        }
        tmp_path = self._teach_targets_path.with_suffix(
            self._teach_targets_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
        tmp_path.replace(self._teach_targets_path)

    def _mark_taught_target(self) -> None:
        if not self._teach_mode:
            return
        if self.robot.board_state not in (
                "holding_board", "moving_board", "release_armed"):
            print("[Teach] Grasp the board before marking a target")
            return
        T_board = self._board_pose_from_tcp(self.robot.tcp_pose)
        if T_board is None:
            print("[Teach] No live TCP pose; target not recorded")
            return
        q_taught = self.robot.q
        if q_taught is None or np.asarray(q_taught).shape != (6,):
            print("[Teach] No valid six-joint robot state; target not recorded")
            return
        record = {
            "target_index": len(self._taught_poses),
            "position_m": [float(v) for v in T_board[:3, 3]],
            "euler_xyz_deg": [float(v) for v in ScipyR.from_matrix(
                T_board[:3, :3]).as_euler("xyz", degrees=True)],
            "joint_positions_rad": [float(v) for v in q_taught],
        }
        self._taught_poses.append(record)
        self._save_taught_targets()
        print(f"[Teach] Marked target {len(self._taught_poses)}: "
              f"pos={np.round(record['position_m'], 4).tolist()}  "
              f"euler={np.round(record['euler_xyz_deg'], 1).tolist()}")

    def _undo_taught_target(self) -> None:
        if not self._teach_mode:
            return
        if not self._taught_poses:
            print("[Teach] No marked target to remove")
            return
        removed = self._taught_poses.pop()
        self._save_taught_targets()
        print(f"[Teach] Removed target {removed['target_index'] + 1}; "
              f"{len(self._taught_poses)} remain")

    @staticmethod
    def _pose_error(T_a: np.ndarray, T_b: np.ndarray) -> "tuple[float, float]":
        pos_err = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
        R_err   = T_a[:3, :3].T @ T_b[:3, :3]
        ang_err = float(np.degrees(ScipyR.from_matrix(R_err).magnitude()))
        return pos_err, ang_err

    @staticmethod
    def _board_pose_from_tcp(T_tcp: "np.ndarray | None") -> "np.ndarray | None":
        if T_tcp is None:
            return None
        T_board = np.array(T_tcp, dtype=np.float64, copy=True)
        T_board[:3, 3] += cfg.BOX_FORWARD_OFFSET * T_board[:3, 2]
        return T_board

    def _target_display_state(self, T_target: "np.ndarray | None") -> str:
        """Return the immediate visual state, independent of completion dwell."""
        T_actual_board = self._board_pose_from_tcp(self.robot.tcp_pose)
        if T_target is None or T_actual_board is None:
            proximity_state = "far"
        else:
            pos_err, ang_err = self._pose_error(T_actual_board, T_target)
            if (pos_err < self._STUDY_POS_TOL_M
                    and ang_err < self._STUDY_ANGLE_TOL_DEG):
                proximity_state = "reached"
            elif (pos_err < self._TARGET_NEAR_POS_M
                  and ang_err < self._TARGET_NEAR_ANGLE_DEG):
                proximity_state = "near"
            else:
                proximity_state = "far"
        return self._completion_flash_state or proximity_state

    @staticmethod
    def _quest_target_color(proximity_state: str) -> list[float]:
        """Return the same discrete target color used by the Open3D view."""
        color = {
            "reached": _WorkholdingSceneVis._TARGET_REACHED_COLOR,
            "near": _WorkholdingSceneVis._TARGET_NEAR_COLOR,
            "far": _WorkholdingSceneVis._TARGET_UNREACHED_COLOR,
            "black": _WorkholdingSceneVis._TARGET_BLACK_COLOR,
        }.get(proximity_state, _WorkholdingSceneVis._TARGET_DEFAULT_COLOR)
        return [float(c) for c in color] + [0.45]
    def _precheck_target_reachability(self) -> "dict[int, bool]":
        """Validate fixed joints or deterministically generate and store them."""
        scene = self.robot.pb_scene
        saved_q = scene.current_q.copy()
        saved_rest = list(scene._rest_poses)
        results = {}
        generated = 0
        pos_tol = self._IK_POS_TOL_M
        angle_tol_deg = self._IK_ANGLE_TOL_DEG
        lower = np.deg2rad(np.asarray(cfg.JOINT_MIN_DEG, dtype=float))
        upper = np.deg2rad(np.asarray(cfg.JOINT_MAX_DEG, dtype=float))
        rest_q = np.deg2rad(np.asarray(cfg.JOINT_REST_DEG, dtype=float))
        rng = np.random.default_rng(20260831)
        try:
            for pose_idx, T_board in enumerate(self._poses_T):
                T_tcp = np.array(T_board, dtype=np.float64, copy=True)
                T_tcp[:3, 3] -= cfg.BOX_FORWARD_OFFSET * T_tcp[:3, 2]
                quat = ScipyR.from_matrix(T_tcp[:3, :3]).as_quat()
                try:
                    fixed_q = self._target_joints[pose_idx]
                    if fixed_q is not None:
                        q_ik = np.asarray(fixed_q, dtype=float)
                        scene.update_robot(q_ik)
                    else:
                        seeds = [saved_q, rest_q]
                        if pose_idx and self._target_joints[pose_idx - 1] is not None:
                            seeds.append(np.asarray(
                                self._target_joints[pose_idx - 1], dtype=float))
                        historical_seeds = self._historical_target_joint_seeds(
                            T_board)
                        seeds.extend(rng.uniform(lower, upper) for _ in range(12))
                        valid_candidates = []
                        best_failed = None
                        candidate_inputs = (
                            [(seed, True) for seed in historical_seeds]
                            + [(seed, False) for seed in seeds])
                        for seed, use_directly in candidate_inputs:
                            if use_directly:
                                candidate = np.asarray(seed, dtype=float)
                            else:
                                scene.set_rest_poses(seed)
                                candidate = scene.solve_ik(
                                    seed, T_tcp[:3, 3], quat,
                                    pos_tol=pos_tol,
                                    orient_tol=np.deg2rad(angle_tol_deg))
                            candidate = np.asarray(candidate, dtype=float)
                            scene.update_robot(candidate)
                            candidate_fk = scene.update_tcp_bodies()
                            if candidate_fk is None:
                                continue
                            candidate_pos_err, candidate_ang_err = self._pose_error(
                                candidate_fk, T_tcp)
                            candidate_within_limits = bool(
                                len(candidate) == len(lower)
                                and np.all(candidate >= lower)
                                and np.all(candidate <= upper))
                            score = candidate_pos_err + np.deg2rad(
                                candidate_ang_err) * 0.01
                            if best_failed is None or score < best_failed[0]:
                                best_failed = (score, candidate,
                                               candidate_pos_err,
                                               candidate_ang_err)
                            if (candidate_within_limits
                                    and candidate_pos_err < pos_tol
                                    and candidate_ang_err < angle_tol_deg):
                                joint_distance = float(np.linalg.norm(
                                    (candidate - saved_q) /
                                    np.maximum(upper - lower, 1e-6)))
                                valid_candidates.append(
                                    (joint_distance, candidate,
                                     candidate_pos_err, candidate_ang_err))
                        if valid_candidates:
                            reference_q = saved_q
                            rescored = []
                            for _old_distance, candidate, candidate_pos_err, candidate_ang_err in valid_candidates:
                                shortest_delta = (
                                    (candidate - reference_q + np.pi)
                                    % (2.0 * np.pi) - np.pi)
                                joint_distance = float(np.linalg.norm(
                                    shortest_delta))
                                rescored.append(
                                    (joint_distance, candidate,
                                     candidate_pos_err, candidate_ang_err))
                            _, q_ik, _, _ = min(rescored,
                                                key=lambda item: item[0])
                            self._target_joints[pose_idx] = q_ik.copy()
                            generated += 1
                        elif best_failed is not None:
                            _, q_ik, _, _ = best_failed
                        else:
                            raise RuntimeError("IK produced no candidates")
                        scene.update_robot(q_ik)
                    T_fk = scene.update_tcp_bodies()
                    if T_fk is None:
                        reachable = False
                        pos_err = ang_err = float("inf")
                    else:
                        pos_err, ang_err = self._pose_error(T_fk, T_tcp)
                        within_limits = bool(
                            len(q_ik) == len(lower)
                            and np.all(q_ik >= lower)
                            and np.all(q_ik <= upper))
                        reachable = bool(
                            within_limits
                            and pos_err < pos_tol
                            and ang_err < angle_tol_deg)
                except Exception as exc:
                    reachable = False
                    pos_err = ang_err = float("inf")
                    print(f"[StudyIK] pose#{pose_idx} check failed: {exc}")
                finally:
                    scene.update_robot(saved_q)
                results[pose_idx] = reachable
                label = "REACHABLE" if reachable else "UNREACHABLE"
                source = "fixed joints" if self._target_joints[pose_idx] is not None else "IK"
                print(f"[StudyIK] pose#{pose_idx}: {label} via {source} "
                      f"({pos_err*100:.1f} cm, {ang_err:.1f} deg)")
        finally:
            scene.set_rest_poses(saved_rest)
            scene.update_robot(saved_q)
        if generated:
            self._store_generated_target_joints()
            print(f"[StudyIK] Stored {generated} generated fixed-joint target(s) "
                  f"in {self._target_poses_path}")
        return results

    def _step_target_preview(self, delta: int) -> None:
        if not self._pose_order:
            return
        if (self._target_navigation == "move"
                and (self._auto_move_pending
                     or self.robot.board_state == "moving_board")):
            print("[StudyVis] P/N ignored — wait for the current robot move "
                  "to stop before selecting another target")
            return
        self._manual_target_preview = True
        self._target_proximity_state = "far"
        self._target_preview_cursor = (
            self._target_preview_cursor + delta) % len(self._pose_order)
        pose_idx = self._pose_order[self._target_preview_cursor]
        self._replay_event("target_preview_changed", delta=int(delta),
                           preview_trial_cursor=self._target_preview_cursor,
                           pose_idx=pose_idx,
                           target_navigation=self._target_navigation)
        pos, euler = self._poses_raw[pose_idx]
        print(f"[StudyVis] Preview target "
              f"{self._target_preview_cursor + 1}/{len(self._pose_order)} "
              f"(pose#{pose_idx})  pos={np.round(pos, 3).tolist()} "
              f"euler={euler}")

        T_board = self._poses_T[pose_idx]
        T_tcp = np.array(T_board, dtype=np.float64, copy=True)
        T_tcp[:3, 3] -= cfg.BOX_FORWARD_OFFSET * T_tcp[:3, 2]
        tcp_pos = T_tcp[:3, 3]
        tcp_quat = ScipyR.from_matrix(T_tcp[:3, :3]).as_quat()

        if self._target_navigation == "preview":
            # Compute a hypothetical configuration in the client's local
            # PyBullet scene, capture its link poses, then restore the live
            # robot state. This changes only the Open3D rendering.
            scene = self.robot.pb_scene
            saved_q = scene.current_q.copy()
            try:
                fixed_q = self._target_joints[pose_idx]
                if fixed_q is None:
                    raise RuntimeError(
                        "no validated fixed joint angles for this target")
                q_ik = np.asarray(fixed_q, dtype=float)
                scene.update_robot(q_ik)
                self._preview_robot_link_poses = [
                    np.asarray(T, dtype=float).copy()
                    for T in scene.get_arm_link_world_poses()]
                preview_tcp = scene.update_tcp_bodies()
                self._preview_robot_tcp_T = (
                    np.asarray(preview_tcp, dtype=float).copy()
                    if preview_tcp is not None else None)
                print("[StudyVis] Showing fixed-joint preview (hardware unchanged)")
            except Exception as exc:
                self._preview_robot_link_poses = None
                self._preview_robot_tcp_T = None
                print(f"[StudyVis] Robot IK preview failed: {exc}")
            finally:
                scene.update_robot(saved_q)
            return

        # In explicit move mode, P/N is also a motion command:
        # convert the selected board pose to its corresponding TCP pose and
        # either retarget the current servo motion or start a new one.
        if (self._target_navigation != "move"
                or self._teach_mode or not self._ar_enabled):
            return
        self._preview_robot_link_poses = None
        self._preview_robot_tcp_T = None
        board_held = self.robot.board_state in (
            "holding_board", "moving_board", "release_armed")
        self._start_auto_move(tcp_pos, tcp_quat, board_move=board_held)
        payload = "held board" if board_held else "bare gripper"
        print(f"[StudyVis] Moving {payload} to selected target")

    def _update_visualizer(self) -> None:
        """Mirror the live study state into the embedded Open3D window."""
        T_tcp = self.robot.tcp_pose
        link_poses = self.robot.arm_link_poses()
        if (self._target_navigation == "preview"
                and self._manual_target_preview
                and self._preview_robot_link_poses is not None):
            link_poses = self._preview_robot_link_poses
        display_tcp = (self._preview_robot_tcp_T
                       if (self._target_navigation == "preview"
                           and self._manual_target_preview
                           and self._preview_robot_tcp_T is not None)
                       else T_tcp)
        board_held = self.robot.board_state in (
            "holding_board", "moving_board", "release_armed")

        self.vis.set_tcp_gripper_closed(board_held)
        self.vis.update_tcp(display_tcp)
        if link_poses is not None:
            self.vis.update_robot(link_poses)
        # Keep the current/actual board visible at the live TCP throughout the
        # study, including the initial insertion phase.  Its gripper remains
        # open until the server reports that the board has been grasped.
        self.vis.update_board_ar_from_tcp(T_tcp, cfg.BOX_FORWARD_OFFSET)
        # The controller workspace is defined in the calibrated world frame,
        # so its Open3D wireframe can remain visible before marker locking.
        self.vis.update_workspace_bound(self._ws_lo, self._ws_hi)

        if self.anchor.locked:
            T_wt = self.anchor.T_world_tracking
            center_eye = self.hands.center_eye_T()
            T_world_head = (self.anchor.world_T(center_eye)
                            if center_eye is not None else None)
            left_pts, right_pts = self.hands.world_joints(T_wt)
            self.vis.update_head(T_world_head)
            self.vis.update_hands(left_pts, right_pts)
        else:
            self.vis.update_head(None)
            self.vis.update_hands(None, None)

        # Show the current planned target even before marker 100 is locked and
        # while the robot is resetting between trials.
        T_target = None
        target_reachable = False
        if self._pose_order and not self._teach_mode:
            visual_cursor = (self._target_preview_cursor
                             if self._manual_target_preview
                             else self._trial_cursor)
            visual_cursor = min(visual_cursor, len(self._pose_order) - 1)
            target_pose_idx = self._pose_order[visual_cursor]
            T_target = self._poses_T[target_pose_idx]
            target_reachable = self._target_reachability.get(
                target_pose_idx, False)
        # The cyan BoardAR preview is shown at the live TCP from startup, even
        # before marker lock or physical grasp. Drive the Open3D handle from
        # that same pose so the two never appear separated by state gating.
        T_preview_board = self._board_pose_from_tcp(T_tcp)
        self.vis.update_ar_handle(
            T_preview_board if self._ar_enabled else None)
        display_state = self._target_display_state(T_target)
        if T_target is not None:
            self.vis.select_target(target_pose_idx, display_state)
        self._target_proximity_state = display_state
        self.vis.update_target_gripper(
            T_target, cfg.BOX_FORWARD_OFFSET, display_state)

        if self._teach_mode:
            mode_state = "freedrive"
        elif self.mode == "hybrid":
            mode_state = ("freedrive" if self._hybrid_freedrive_only else "ar")
        else:
            mode_state = self.mode
        self.vis.update_mode_indicator(T_tcp, mode_state)
        self.vis.tick()

    def _update_palm_cbf_obstacle(self, now: float) -> None:
        """Publish the closest tracked palm during autonomous AR following."""
        following_ar = bool(
            self._ar_enabled
            and not self._hybrid_freedrive_only
            and self._phase in ("trial_running", "await_snap_confirmation")
            and (self._auto_move_pending
                 or self.robot.board_state == "moving_board"))
        if not following_ar or not self.anchor.locked:
            if self._palm_cbf_active:
                self.robot.update_palm_obstacle(None)
                self._palm_cbf_active = False
            return
        if (now - self._last_palm_cbf_pub_time
                < self._PALM_CBF_PUBLISH_INTERVAL_S):
            return

        left_pts, right_pts = self.hands.world_joints(
            self.anchor.T_world_tracking)
        candidates = []
        for points in (left_pts, right_pts):
            if points is None or len(points) <= 6:
                continue
            palm = (np.asarray(points[3], float)
                    + np.asarray(points[1], float)
                    + np.asarray(points[6], float)) / 3.0
            if palm.shape == (3,) and np.all(np.isfinite(palm)):
                candidates.append(palm)
        T_tcp = self.robot.tcp_pose
        if candidates:
            if T_tcp is not None:
                tcp_pos = T_tcp[:3, 3]
                palm_center = min(
                    candidates, key=lambda p: float(np.linalg.norm(p - tcp_pos)))
            else:
                palm_center = candidates[0]
            self.robot.update_palm_obstacle(
                palm_center, radius=self._PALM_CBF_RADIUS_M,
                clearance=self._PALM_CBF_CLEARANCE_M)
            self._last_palm_cbf_pub_time = now
            self._palm_cbf_active = True
        elif self._palm_cbf_active:
            self.robot.update_palm_obstacle(None)
            self._palm_cbf_active = False

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
        self._replay_event("anchor_locked", marker_id=self.anchor_marker_id,
                           world_tracking_T=self.anchor.T_world_tracking)

    # ── Study state machine ─────────────────────────────────────────────────

    def _on_auto_move_complete(self, ok: bool) -> None:
        self._auto_move_pending = False
        # Cancelling continuous AR following at the Hybrid handoff is an
        # intentional control-mode transition, not a failed move that should
        # poison the next reset-to-default state.
        self._auto_move_result = (
            None if self._hybrid_freedrive_only and not ok else bool(ok))
        self._replay_event("ar_move_complete", success=bool(ok))

    def _start_auto_move(self, pos: np.ndarray, quat: np.ndarray,
                         board_move: bool = True) -> None:
        self._auto_move_pending = True
        self._auto_move_result  = None
        self._replay_event("ar_move_started", target_tcp_position=pos,
                           target_tcp_quaternion=quat,
                           board_move=bool(board_move))
        self.robot.move_to_pose(np.asarray(pos, dtype=float),
                                np.asarray(quat, dtype=float),
                                board_move=board_move,
                                motion_profile="workholding",
                                on_complete=self._on_auto_move_complete)

    def _start_default_joint_reset(self) -> None:
        """Return to the known default configuration without Cartesian IK."""
        self._auto_move_pending = True
        self._auto_move_result = None
        self._replay_event(
            "default_reset_started",
            target_joints_deg=list(cfg.ROBOT_DEFAULT_JOINT_DEG))

        def _completed(ok: bool) -> None:
            self._auto_move_pending = False
            self._auto_move_result = bool(ok)
            self._replay_event("default_reset_complete", success=bool(ok))

        self.robot.move_to_joints(
            cfg.ROBOT_DEFAULT_JOINT_DEG, degrees=True,
            board_move=True, speed_multiplier=0.6,
            on_complete=_completed)

    def _begin_trial(self) -> None:
        pose_idx = self._pose_order[self._trial_cursor]
        self._target_preview_cursor = self._trial_cursor
        self._manual_target_preview = False
        self._target_proximity_state = "far"
        self._completion_flash_state = None
        self._trial_target_T          = self._poses_T[pose_idx]
        self._trial_start_t           = 0.0
        self._trial_timer_running     = False
        self._trial_active_elapsed_s  = 0.0
        self._trial_timer_last_t      = None
        self._trial_reach_dwell_start = None
        self._trial_first_reach_time = None
        self._trial_first_reach_elapsed_s = float("nan")
        self._trial_first_reach_pos_error_m = float("nan")
        self._trial_first_reach_angle_error_deg = float("nan")
        self._trial_interactions      = 0
        self._trial_freedrive_interactions = 0
        self._trial_ar_interactions = 0
        self._post_stop_ar_interactions = 0
        self._post_stop_freedrive_interactions = 0
        self._post_stop_interactions = 0
        self._post_stop_baseline_pos_error_m = float("nan")
        self._post_stop_baseline_angle_error_deg = float("nan")
        self._post_stop_freedrive_active = False
        self._post_stop_freedrive_start_errors = None
        self._ar_follow_last_board_T = None
        self._hybrid_freedrive_only = False
        if self.mode == "hybrid":
            # Hybrid starts each trial in AR-follow mode. The explicit TCP
            # click is the only way to enable freedrive.
            self.robot.set_board_freedrive(False)
        gripper_color = (self._FREEDRIVE_GRIPPER_RGBA
                         if self.mode == "freedrive"
                         else self._AR_GRIPPER_RGBA)
        self.tools.set_forced_color(self._TCP_TOOL_ID, gripper_color)
        self._trial_tcp_path_length_m = 0.0
        self._trial_tcp_angular_path_length_deg = 0.0
        self._trial_recording_start_source = ""
        self._trial_start_policy = "default"
        self._pending_trial_summary = None
        self._snap_started_t = None
        self._trial_last_traj_t       = 0.0
        self._prev_tcp_pos_for_speed  = None
        self._prev_tcp_rot_for_speed  = None
        self._prev_tcp_t_for_speed    = None
        self._was_moving_freedrive    = False
        self._freedrive_stationary_since = None
        self._force_complete_requested = False
        self._trial_start_board_T = self._board_pose_from_tcp(self.robot.tcp_pose)
        self._trial_path_prev_tcp_pos = (
            self.robot.tcp_pose[:3, 3].copy()
            if self.robot.tcp_pose is not None else None)
        self._trial_path_prev_tcp_rot = (
            self.robot.tcp_pose[:3, :3].copy()
            if self.robot.tcp_pose is not None else None)
        if self._trial_start_board_T is not None:
            (self._trial_start_pos_error_m,
             self._trial_start_angle_error_deg) = self._pose_error(
                self._trial_start_board_T, self._trial_target_T)
        else:
            self._trial_start_pos_error_m = float("nan")
            self._trial_start_angle_error_deg = float("nan")
        pos, euler = self._poses_raw[pose_idx]
        n = len(self._pose_order)
        print(f"[Trial] {self.mode} {self._trial_cursor + 1}/{n}  "
              f"pose#{pose_idx}  start={self._trial_start_policy}  "
              f"pos={np.round(pos, 3).tolist()}  euler={euler}")
        print("[Trial] Ready — press ENTER to start. The target criterion "
              "ends recording automatically; then press ENTER to snap.")
        if self._ar_enabled:
            print("[Trial] AR robot commands are disabled until ENTER starts "
                  "the timer.")
        self._phase = "trial_running"
        self._replay_event("trial_ready", pose_idx=pose_idx,
                           start_policy=self._trial_start_policy,
                           target_board_world_T=self._trial_target_T)

    def _handle_hybrid_gripper_click(self, event: dict) -> None:
        """Toggle stationary Hybrid trials between AR and freedrive control."""
        if (self.mode != "hybrid"
                or event.get("event_type") != "selected"
                or int(event.get("tool_id", -1)) != self._TCP_TOOL_ID):
            return
        self.tools.deselect(self._TCP_TOOL_ID)
        active_phase = (
            (self._phase == "trial_running" and self._trial_timer_running)
            or self._phase == "await_snap_confirmation")
        if not active_phase:
            print("[Hybrid] Gripper toggle ignored — start the trial first.")
            return
        T_tcp = self.robot.tcp_pose
        if T_tcp is None:
            print("[Hybrid] Gripper toggle ignored — no TCP pose.")
            return
        robot_stopped = bool(
            not self._auto_move_pending
            and not self.robot.move_running
            and self.robot.board_state == "holding_board")
        if not robot_stopped:
            print("[Hybrid] Gripper toggle ignored — wait until the robot "
                  "is fully stopped.")
            return

        self._hybrid_freedrive_only = not self._hybrid_freedrive_only
        if self._hybrid_freedrive_only:
            self.ar_bridge.publish("idle", T_tcp)
            self.robot.set_board_freedrive(True)
            color = self._FREEDRIVE_GRIPPER_RGBA
            mode_label = "FREEDRIVE"
        else:
            self._ar_follow_last_board_T = None
            self.ar_bridge.publish("grabbed", T_tcp)
            self.robot.set_board_freedrive(False)
            color = self._AR_GRIPPER_RGBA
            mode_label = "AR FOLLOW"
        self.tools.set_forced_color(self._TCP_TOOL_ID, color)
        self._prev_tcp_pos_for_speed = None
        self._prev_tcp_rot_for_speed = None
        self._prev_tcp_t_for_speed = None
        self._was_moving_freedrive = False
        self._freedrive_stationary_since = None
        self._close_status_line()
        print(f"[Hybrid] Gripper clicked → {mode_label}")
        self._replay_event(
            "hybrid_control_mode_toggled",
            control_mode=("freedrive" if self._hybrid_freedrive_only else "ar"),
            clicking_hand=event.get("hand", "unknown"),
            tcp_world_T=T_tcp)

    def _trial_elapsed(self, now: "float | None" = None) -> float:
        if now is None:
            now = time.time()
        elapsed = self._trial_active_elapsed_s
        if self._trial_timer_running and self._trial_timer_last_t is not None:
            elapsed += now - self._trial_timer_last_t
        return elapsed

    def _toggle_trial_timer(self) -> None:
        if self._phase == "snap_failed":
            print("[Study] Retrying exact-target snap...")
            self._start_exact_target_snap(time.time())
            return
        if self._phase == "await_snap_confirmation":
            if (self._auto_move_pending
                    or self.robot.board_state == "moving_board"
                    or self._post_stop_freedrive_active):
                print("[Study] An adjustment is still in progress — "
                      "wait for it to finish, then press ENTER.")
                return
            T_tcp = self.robot.tcp_pose
            if T_tcp is not None and self._trial_target_T is not None:
                T_board = self._board_pose_from_tcp(T_tcp)
                enter_pos_err, enter_ang_err = self._pose_error(
                    T_board, self._trial_target_T)
            else:
                enter_pos_err = enter_ang_err = float("nan")
            auto_pos_err = auto_ang_err = float("nan")
            if self._pending_trial_summary is not None:
                auto_pos_err = self._pending_trial_summary.get(
                    "auto_stop_pos_error_m", float("nan"))
                auto_ang_err = self._pending_trial_summary.get(
                    "auto_stop_angle_error_deg", float("nan"))
                self._pending_trial_summary.update({
                    "enter_confirmation_time": time.time(),
                    "enter_confirmation_pos_error_m": enter_pos_err,
                    "enter_confirmation_angle_error_deg": enter_ang_err,
                    "post_stop_ar_interactions":
                        self._post_stop_ar_interactions,
                    "post_stop_freedrive_interactions":
                        self._post_stop_freedrive_interactions,
                    "post_stop_interactions": self._post_stop_interactions,
                    "post_stop_pos_error_improvement_m":
                        auto_pos_err - enter_pos_err,
                    "post_stop_angle_error_improvement_deg":
                        auto_ang_err - enter_ang_err,
                })
            self._replay_event(
                "snap_confirmation_entered",
                enter_confirmation_pos_error_m=enter_pos_err,
                enter_confirmation_angle_error_deg=enter_ang_err,
                post_stop_ar_interactions=self._post_stop_ar_interactions,
                post_stop_freedrive_interactions=
                    self._post_stop_freedrive_interactions,
                post_stop_interactions=self._post_stop_interactions,
                post_stop_pos_error_improvement_m=auto_pos_err - enter_pos_err,
                post_stop_angle_error_improvement_deg=auto_ang_err - enter_ang_err)
            print("[Study] Snap confirmed — moving to the exact target...")
            self._start_exact_target_snap(time.time())
            return
        if self._phase != "trial_running":
            print("[Trial] ENTER is available while a trial is ready/running "
                  "or when retrying a failed snap.")
            return
        now = time.time()
        if self._trial_timer_running:
            print("[Trial] Already recording. Reach within 5 cm and 15 degrees "
                  "for 1 second; the trial will stop automatically.")
        else:
            if self._trial_start_t <= 0.0:
                self._trial_start_t = now
                self._trial_recording_start_source = "manual_enter"
                T_tcp = self.robot.tcp_pose
                self._trial_path_prev_tcp_pos = (
                    T_tcp[:3, 3].copy() if T_tcp is not None else None)
                self._trial_path_prev_tcp_rot = (
                    T_tcp[:3, :3].copy() if T_tcp is not None else None)
            self._trial_timer_last_t = now
            self._trial_timer_running = True
            self._prev_tcp_pos_for_speed = None
            self._prev_tcp_rot_for_speed = None
            self._prev_tcp_t_for_speed = None
            self._was_moving_freedrive = False
            self._freedrive_stationary_since = None
            print("[Trial] RUNNING — recording stops automatically at the target.")
            self._replay_event("trial_started",
                               elapsed_s=self._trial_elapsed(now))

    def _start_next_trial_or_finish(self) -> None:
        if self._trial_cursor >= len(self._pose_order):
            self._phase = "release_board"
            return
        if self._trial_cursor > 0:
            print(f"[Study] Preparing trial {self._trial_cursor + 1}: "
                  "returning to the default pose.")
            self._phase = "reset_to_default"
            return
        self._begin_trial()

    def _start_exact_target_snap(self, now: float) -> None:
        """Correct with a deterministic taught moveJ, or IK for legacy files."""
        if self._trial_target_T is None:
            self._phase = "snap_failed"
            return
        pose_idx = self._pose_order[self._trial_cursor]
        T_tcp = np.array(self._trial_target_T, dtype=np.float64, copy=True)
        T_tcp[:3, 3] -= cfg.BOX_FORWARD_OFFSET * T_tcp[:3, 2]
        tcp_quat = ScipyR.from_matrix(T_tcp[:3, :3]).as_quat()
        scene = self.robot.pb_scene
        q_current = scene.current_q.copy()
        fixed_q = self._target_joints[pose_idx]
        controller = "taught_fixed_joint_moveJ" if fixed_q is not None else "ik_moveJ"
        try:
            if fixed_q is not None:
                q_target = np.asarray(fixed_q, dtype=float).copy()
                lower = np.deg2rad(np.asarray(cfg.JOINT_MIN_DEG, dtype=float))
                upper = np.deg2rad(np.asarray(cfg.JOINT_MAX_DEG, dtype=float))
                if np.any(q_target < lower) or np.any(q_target > upper):
                    raise RuntimeError("taught joint angles are outside limits")
                scene.update_robot(q_target)
            else:
                q_target = scene.solve_ik(
                    q_current, T_tcp[:3, 3], tcp_quat,
                    pos_tol=self._IK_POS_TOL_M,
                    orient_tol=np.deg2rad(self._IK_ANGLE_TOL_DEG))
            T_check = scene.update_tcp_bodies()
            if T_check is None:
                raise RuntimeError("target joints produced no TCP pose")
            pos_err, ang_err = self._pose_error(T_check, T_tcp)
            if (pos_err >= self._IK_POS_TOL_M
                    or ang_err >= self._IK_ANGLE_TOL_DEG):
                raise RuntimeError(
                    f"target validation failed ({pos_err * 100:.2f} cm/"
                    f"{ang_err:.2f} deg)")
        except Exception as exc:
            print(f"[Study] Exact-target {controller} failed: {exc}")
            self._replay_event("exact_target_movej_planning_failed",
                               controller=controller, error=str(exc))
            self._phase = "snap_failed"
            return
        finally:
            scene.update_robot(q_current)
        self._snap_started_t = now
        self._auto_move_pending = True
        self._auto_move_result = None
        self.robot.move_to_joints(
            q_target, board_move=True, blocking=True,
            on_complete=self._on_auto_move_complete)
        self._phase = "snap_to_target"
        self._replay_event("exact_target_snap_started",
                           controller=controller,
                           target_joints_rad=q_target,
                           target_board_world_T=self._trial_target_T)

    def _finalize_snapped_trial(self, now: float,
                                snap_success: bool = True) -> None:
        T_board = self._board_pose_from_tcp(self.robot.tcp_pose)
        if T_board is not None and self._trial_target_T is not None:
            post_pos_err, post_ang_err = self._pose_error(
                T_board, self._trial_target_T)
        else:
            post_pos_err = post_ang_err = float("nan")
        snap_duration = (now - self._snap_started_t
                         if self._snap_started_t is not None else float("nan"))
        if self._pending_trial_summary is None:
            raise RuntimeError("Exact-target snap completed without a pending trial row")
        self._replay_event(
            "trial_summary", **self._pending_trial_summary,
            snap_success=bool(snap_success), snap_duration_s=snap_duration,
            post_snap_pos_error_m=post_pos_err,
            post_snap_angle_error_deg=post_ang_err,
            target_poses_file=self._target_poses_source)
        outcome = "complete" if snap_success else "skipped after failure"
        print(f"[Study] Exact-target snap {outcome} in {snap_duration:.1f}s "
              f"({post_pos_err * 100:.2f}cm/{post_ang_err:.2f}deg).")
        self._pending_trial_summary = None
        self._snap_started_t = None
        self._trial_cursor += 1
        self._start_next_trial_or_finish()

    def _close_status_line(self) -> None:
        """End the in-place live-offset line with a real newline, if one is open,
        so a subsequent normal print() doesn't land mid-line."""
        if self._status_trial_cursor is not None:
            print()
            self._status_trial_cursor = None

    def _print_live_status(self, now: float) -> None:
        """Print the current trial's live offset-from-target + state in place
        (overwriting the same terminal line each tick); starts a fresh line
        whenever the trial index changes."""
        if self._status_trial_cursor != self._trial_cursor:
            self._close_status_line()
            self._status_trial_cursor = self._trial_cursor
        if now - self._last_status_print_t < 0.2:
            return
        self._last_status_print_t = now
        n = len(self._pose_order)
        timer_state = "RUN" if self._trial_timer_running else "READY"
        bits = [f"{self.mode}",
                f"T{min(self._trial_cursor + 1, n)}/{n}",
                timer_state,
                f"t={self._trial_elapsed(now):.1f}s"]
        T_tcp = self.robot.tcp_pose
        if T_tcp is not None and self._trial_target_T is not None:
            T_board = self._board_pose_from_tcp(T_tcp)
            pos_err, ang_err = self._pose_error(
                T_board, self._trial_target_T)
            bits.append(f"err={pos_err * 100:.1f}cm/{ang_err:.1f}deg")
        bits.append(f"n={self._trial_interactions}")
        # Clear the full current terminal row before redrawing. Keeping this
        # deliberately short prevents wrapping, which a carriage return alone
        # cannot overwrite reliably.
        print(f"\r\033[2K[Trial] {'  '.join(bits)}", end="", flush=True)

    def _finish_trial(self, reason: str, pos_err: float, ang_err: float) -> None:
        now      = time.time()
        pose_idx = self._pose_order[self._trial_cursor]
        pos, euler = self._poses_raw[pose_idx]
        duration = self._trial_elapsed(now)
        T_tcp_now = self.robot.tcp_pose
        if T_tcp_now is not None and self._trial_path_prev_tcp_pos is not None:
            self._trial_tcp_path_length_m += float(np.linalg.norm(
                T_tcp_now[:3, 3] - self._trial_path_prev_tcp_pos))
            self._trial_path_prev_tcp_pos = T_tcp_now[:3, 3].copy()
        if T_tcp_now is not None and self._trial_path_prev_tcp_rot is not None:
            relative_rot = self._trial_path_prev_tcp_rot.T @ T_tcp_now[:3, :3]
            self._trial_tcp_angular_path_length_deg += float(np.degrees(
                ScipyR.from_matrix(relative_rot).magnitude()))
            self._trial_path_prev_tcp_rot = T_tcp_now[:3, :3].copy()
        if self._trial_timer_running:
            self._trial_active_elapsed_s = duration
            self._trial_timer_running = False
            self._trial_timer_last_t = None
        if self._trial_start_board_T is not None:
            start_pos = self._trial_start_board_T[:3, 3].tolist()
            start_euler = ScipyR.from_matrix(
                self._trial_start_board_T[:3, :3]).as_euler(
                    "xyz", degrees=True).tolist()
        else:
            start_pos = [float("nan")] * 3
            start_euler = [float("nan")] * 3
        # Held until _finalize_snapped_trial extends it with the post-snap
        # fields and logs the complete "trial_summary" replay event.
        self._pending_trial_summary = dict(
            pose_idx=pose_idx,
            target_pos_m=list(pos), target_euler_deg=list(euler),
            start_time=self._trial_start_t, end_time=now, duration_s=duration,
            final_pos_error_m=pos_err, final_angle_error_deg=ang_err,
            auto_stop_pos_error_m=pos_err,
            auto_stop_angle_error_deg=ang_err,
            first_reach_time=self._trial_first_reach_time,
            first_reach_elapsed_s=self._trial_first_reach_elapsed_s,
            first_reach_pos_error_m=self._trial_first_reach_pos_error_m,
            first_reach_angle_error_deg=self._trial_first_reach_angle_error_deg,
            num_interactions=self._trial_interactions, completion_reason=reason,
            start_board_pos_m=start_pos, start_board_euler_deg=start_euler,
            start_pos_error_m=self._trial_start_pos_error_m,
            start_angle_error_deg=self._trial_start_angle_error_deg,
            freedrive_interactions=self._trial_freedrive_interactions,
            ar_interactions=self._trial_ar_interactions,
            tcp_path_length_m=self._trial_tcp_path_length_m,
            tcp_angular_path_length_deg=self._trial_tcp_angular_path_length_deg,
            recording_start_source=self._trial_recording_start_source,
            start_policy=self._trial_start_policy,
        )
        # Logged immediately (rather than only once the snap finalizes) so a
        # trial's outcome survives even if the session ends before snapping.
        self._replay_event("trial_base_summary", **self._pending_trial_summary)
        self._close_status_line()
        print(f"[Trial] OVER ({reason}) — logging stopped at {duration:.1f}s, "
              f"err={pos_err * 100:.1f}cm/{ang_err:.1f}deg, "
              f"interactions={self._trial_interactions}")
        self._completion_flash_state = "reached"
        self._phase = "await_snap_confirmation"
        self._post_stop_baseline_pos_error_m = pos_err
        self._post_stop_baseline_angle_error_deg = ang_err
        self._was_moving_freedrive = False
        self._freedrive_stationary_since = None
        self._prev_tcp_pos_for_speed = None
        self._prev_tcp_rot_for_speed = None
        self._prev_tcp_t_for_speed = None
        if self._ar_enabled:
            print("[Study] Trial timing is over. You may continue adjusting "
                  "the AR board; these adjustments are not included in the "
                  "timed trial. Press ENTER when satisfied to record the "
                  "adjusted final error and snap to the exact target.")
        else:
            print("[Study] Trial is over. Press ENTER to record the final "
                  "error, snap to the exact target, and prepare the next "
                  "trial.")

    def _update_path_length_accumulators(self, now: float) -> None:
        """Accumulate TCP path length for the trial summary. Raw trajectory,
        hand, and head samples are no longer written here — they're already
        captured (at a higher 30 Hz rate) by _sample_replay's "frame"
        events in the single per-participant replay JSONL."""
        if now - self._trial_last_traj_t < 1.0 / self._STUDY_TRAJ_SAMPLE_HZ:
            return
        self._trial_last_traj_t = now
        T_tcp = self.robot.tcp_pose
        if T_tcp is None or self.robot.q is None:
            return
        if self._trial_path_prev_tcp_pos is not None:
            self._trial_tcp_path_length_m += float(np.linalg.norm(
                T_tcp[:3, 3] - self._trial_path_prev_tcp_pos))
        if self._trial_path_prev_tcp_rot is not None:
            relative_rot = self._trial_path_prev_tcp_rot.T @ T_tcp[:3, :3]
            self._trial_tcp_angular_path_length_deg += float(np.degrees(
                ScipyR.from_matrix(relative_rot).magnitude()))
        self._trial_path_prev_tcp_pos = T_tcp[:3, 3].copy()
        self._trial_path_prev_tcp_rot = T_tcp[:3, :3].copy()

    def _tick_target_completion(self, now: float) -> None:
        """End the trial after a sustained target match."""
        if self._auto_move_pending:
            self._trial_reach_dwell_start = None
            return
        T_tcp = self.robot.tcp_pose
        if T_tcp is None or self._trial_target_T is None:
            self._trial_reach_dwell_start = None
            return
        T_board = self._board_pose_from_tcp(T_tcp)
        pos_err, ang_err = self._pose_error(T_board, self._trial_target_T)
        reached = (pos_err < self._STUDY_POS_TOL_M
                   and ang_err < self._STUDY_ANGLE_TOL_DEG)
        if not reached:
            self._trial_reach_dwell_start = None
            return
        if self._trial_reach_dwell_start is None:
            self._trial_reach_dwell_start = now
            if self._trial_first_reach_time is None:
                self._trial_first_reach_time = now
                self._trial_first_reach_elapsed_s = self._trial_elapsed(now)
                self._trial_first_reach_pos_error_m = pos_err
                self._trial_first_reach_angle_error_deg = ang_err
                self._replay_event(
                    "target_first_reached",
                    first_reach_elapsed_s=self._trial_first_reach_elapsed_s,
                    first_reach_pos_error_m=pos_err,
                    first_reach_angle_error_deg=ang_err)
            return
        if now - self._trial_reach_dwell_start >= self._STUDY_REACH_DWELL_S:
            self._finish_trial("target_reached", pos_err, ang_err)

    def _tick_freedrive_channel(self, now: float) -> None:
        """Count physical movement segments; freedrive itself remains server-side."""
        T_tcp = self.robot.tcp_pose
        if T_tcp is None:
            return

        pos = T_tcp[:3, 3]
        rot = T_tcp[:3, :3]
        if (self._prev_tcp_pos_for_speed is not None
                and self._prev_tcp_rot_for_speed is not None
                and self._prev_tcp_t_for_speed is not None):
            dt = now - self._prev_tcp_t_for_speed
            if dt > 1e-3:
                speed = float(np.linalg.norm(
                    pos - self._prev_tcp_pos_for_speed)) / dt
                relative_rot = self._prev_tcp_rot_for_speed.T @ rot
                angular_speed_deg = float(np.degrees(
                    ScipyR.from_matrix(relative_rot).magnitude())) / dt
                moving_fast = (speed > self._STUDY_MOVE_START_MPS
                               or angular_speed_deg > self._STUDY_ROT_START_DEGPS)
                fully_still = (speed < self._STUDY_MOVE_STOP_MPS
                               and angular_speed_deg < self._STUDY_ROT_STOP_DEGPS)
                if moving_fast:
                    if not self._was_moving_freedrive:
                        self._trial_interactions += 1
                        self._trial_freedrive_interactions += 1
                    self._was_moving_freedrive = True
                    self._freedrive_stationary_since = None
                elif self._was_moving_freedrive and fully_still:
                    if self._freedrive_stationary_since is None:
                        self._freedrive_stationary_since = now
                    elif (now - self._freedrive_stationary_since
                          >= self._STUDY_MOVE_STOP_DWELL_S):
                        self._was_moving_freedrive = False
                        self._freedrive_stationary_since = None
                elif self._was_moving_freedrive:
                    # Speed is inside the hysteresis band, so this is still
                    # the same continuous movement segment.
                    self._freedrive_stationary_since = None
        self._prev_tcp_pos_for_speed = pos.copy()
        self._prev_tcp_rot_for_speed = rot.copy()
        self._prev_tcp_t_for_speed   = now

    def _tick_ar_channel(self, now: float, recording: bool,
                         accept_commands: bool = True) -> None:
        board_state = self.robot.board_state
        move_active = board_state == "moving_board" or self._auto_move_pending
        grip_state  = "moving" if move_active else (
            "grabbed" if board_state == "holding_board" else "idle")
        T_tcp = self.robot.tcp_pose
        if T_tcp is not None:
            self.ar_bridge.publish(
                grip_state, T_tcp,
                box_color=self._AR_ASSEMBLY_RGBA.get(self.mode))

        if self._auto_move_result is not None:
            ok = self._auto_move_result
            self._auto_move_result = None
            if ok and T_tcp is not None and self._trial_target_T is not None:
                T_board = self._board_pose_from_tcp(T_tcp)
                pos_err, ang_err = self._pose_error(
                    T_board, self._trial_target_T)
                if self._phase == "await_snap_confirmation":
                    before_pos = self._post_stop_baseline_pos_error_m
                    before_ang = self._post_stop_baseline_angle_error_deg
                    self._replay_event(
                        "post_stop_adjustment_completed",
                        post_stop_interaction_idx=self._post_stop_interactions,
                        interaction_type="ar",
                        landed_board_world_T=T_board,
                        before_pos_error_m=before_pos,
                        before_angle_error_deg=before_ang,
                        landed_pos_error_m=pos_err,
                        landed_angle_error_deg=ang_err,
                        pos_error_improvement_m=before_pos - pos_err,
                        angle_error_improvement_deg=before_ang - ang_err)
                    self._post_stop_baseline_pos_error_m = pos_err
                    self._post_stop_baseline_angle_error_deg = ang_err
                    # In hybrid mode, do not misclassify this commanded AR
                    # motion as a physical freedrive interaction.
                    self._prev_tcp_pos_for_speed = T_tcp[:3, 3].copy()
                    self._prev_tcp_rot_for_speed = T_tcp[:3, :3].copy()
                    self._prev_tcp_t_for_speed = now
                self._close_status_line()
                if self._phase == "await_snap_confirmation":
                    next_action = "— adjust again or press ENTER when satisfied"
                elif not recording:
                    next_action = "— press ENTER to start recording"
                else:
                    next_action = "— press ENTER to finish when ready"
                print(f"[AR] Landed {pos_err*100:.1f}cm/{ang_err:.1f}deg from target "
                      + next_action)
            elif not ok:
                self._close_status_line()
                print("[AR] Move cancelled/failed — try again")

        manipulation_event = self.ar_bridge.poll_event()
        if manipulation_event is not None:
            manipulation_state, T_box_target = manipulation_event
            if not accept_commands:
                if manipulation_state == "released":
                    self._close_status_line()
                    print("[AR] Manipulation ignored — press ENTER to start "
                          "the trial before commanding the robot.")
                    self._replay_event(
                        "ar_handle_release_ignored_before_recording",
                        released_board_world_T=T_box_target,
                        reason="trial_not_started")
                return
            post_stop = self._phase == "await_snap_confirmation"
            tcp_pos  = (T_box_target[:3, 3]
                       - cfg.BOX_FORWARD_OFFSET * T_box_target[:3, 2])
            tcp_quat = ScipyR.from_matrix(T_box_target[:3, :3]).as_quat()
            should_retarget = True
            if self._ar_follow_last_board_T is not None:
                follow_pos_delta, follow_angle_delta = self._pose_error(
                    T_box_target, self._ar_follow_last_board_T)
                should_retarget = bool(
                    follow_pos_delta >= self._AR_FOLLOW_POS_DEADBAND_M
                    or follow_angle_delta >= self._AR_FOLLOW_ANGLE_DEADBAND_DEG)
            if should_retarget or manipulation_state == "released":
                self._ar_follow_last_board_T = T_box_target.copy()
                if self._auto_move_pending or board_state == "moving_board":
                    self.robot.update_move_target(tcp_pos, tcp_quat)
                else:
                    self._start_auto_move(tcp_pos, tcp_quat)

            if manipulation_state == "released":
                if recording:
                    self._trial_interactions += 1
                    self._trial_ar_interactions += 1
                if post_stop:
                    self._post_stop_ar_interactions += 1
                    self._post_stop_interactions += 1
                if self._trial_target_T is not None:
                    released_pos_err, released_ang_err = self._pose_error(
                        T_box_target, self._trial_target_T)
                else:
                    released_pos_err = released_ang_err = float("nan")
                self._close_status_line()
                release_label = (f"#{self._trial_interactions}"
                                 if recording else "(not recorded)")
                print(f"[AR] Release {release_label}; robot following final "
                      f"TCP {np.round(tcp_pos, 3).tolist()}")
                self._replay_event(
                    "ar_handle_released",
                    released_board_world_T=T_box_target,
                    commanded_tcp_position=tcp_pos,
                    commanded_tcp_quaternion=tcp_quat,
                    recording=bool(recording), post_stop=post_stop,
                    post_stop_ar_interaction_idx=(
                        self._post_stop_ar_interactions if post_stop else None),
                    post_stop_interaction_idx=(
                        self._post_stop_interactions if post_stop else None),
                    released_pos_error_m=released_pos_err,
                    released_angle_error_deg=released_ang_err)

    def _tick_post_stop_freedrive_channel(self, now: float) -> None:
        """Log each completed physical adjustment after timed completion."""
        if self._auto_move_pending or self.robot.board_state == "moving_board":
            return
        T_tcp = self.robot.tcp_pose
        if T_tcp is None:
            return
        pos, rot = T_tcp[:3, 3], T_tcp[:3, :3]
        if (self._prev_tcp_pos_for_speed is not None
                and self._prev_tcp_rot_for_speed is not None
                and self._prev_tcp_t_for_speed is not None):
            dt = now - self._prev_tcp_t_for_speed
            if dt > 1e-3:
                speed = float(np.linalg.norm(
                    pos - self._prev_tcp_pos_for_speed)) / dt
                rel_rot = self._prev_tcp_rot_for_speed.T @ rot
                angular_speed = float(np.degrees(
                    ScipyR.from_matrix(rel_rot).magnitude())) / dt
                moving = (speed > self._STUDY_MOVE_START_MPS
                          or angular_speed > self._STUDY_ROT_START_DEGPS)
                still = (speed < self._STUDY_MOVE_STOP_MPS
                         and angular_speed < self._STUDY_ROT_STOP_DEGPS)
                if moving:
                    if not self._post_stop_freedrive_active:
                        self._post_stop_freedrive_active = True
                        self._post_stop_freedrive_interactions += 1
                        self._post_stop_interactions += 1
                        self._post_stop_freedrive_start_errors = (
                            self._post_stop_baseline_pos_error_m,
                            self._post_stop_baseline_angle_error_deg)
                        self._replay_event(
                            "post_stop_adjustment_started",
                            post_stop_interaction_idx=
                                self._post_stop_interactions,
                            interaction_type="freedrive",
                            before_pos_error_m=
                                self._post_stop_baseline_pos_error_m,
                            before_angle_error_deg=
                                self._post_stop_baseline_angle_error_deg)
                    self._freedrive_stationary_since = None
                elif self._post_stop_freedrive_active and still:
                    if self._freedrive_stationary_since is None:
                        self._freedrive_stationary_since = now
                    elif (now - self._freedrive_stationary_since
                          >= self._STUDY_MOVE_STOP_DWELL_S):
                        T_board = self._board_pose_from_tcp(T_tcp)
                        after_pos, after_ang = self._pose_error(
                            T_board, self._trial_target_T)
                        before_pos, before_ang = (
                            self._post_stop_freedrive_start_errors)
                        self._replay_event(
                            "post_stop_adjustment_completed",
                            post_stop_interaction_idx=
                                self._post_stop_interactions,
                            interaction_type="freedrive",
                            landed_board_world_T=T_board,
                            before_pos_error_m=before_pos,
                            before_angle_error_deg=before_ang,
                            landed_pos_error_m=after_pos,
                            landed_angle_error_deg=after_ang,
                            pos_error_improvement_m=before_pos - after_pos,
                            angle_error_improvement_deg=before_ang - after_ang)
                        self._post_stop_baseline_pos_error_m = after_pos
                        self._post_stop_baseline_angle_error_deg = after_ang
                        self._post_stop_freedrive_active = False
                        self._post_stop_freedrive_start_errors = None
                        self._freedrive_stationary_since = None
        self._prev_tcp_pos_for_speed = pos.copy()
        self._prev_tcp_rot_for_speed = rot.copy()
        self._prev_tcp_t_for_speed = now

    def _tick_study(self, now: float) -> None:
        entering = self._phase != self._last_phase
        self._last_phase = self._phase

        if self._phase == "lock_anchor":
            return  # transition triggered from run() once self.anchor.locked

        if self._phase == "await_robot_ready":
            if entering:
                print("[Study] Waiting for robot_control_server.py connection...")
                self._robot_ready_last_nag = now
            if not self.robot.connected:
                if now - self._robot_ready_last_nag >= self._ROBOT_READY_NAG_INTERVAL_S:
                    self._robot_ready_last_nag = now
                    print("[Study] Still no connection to robot_control_server.py — "
                          "is it running? (start it BEFORE this script)")
                return
            if self.robot.move_running:
                return
            self._default_tcp_pos, self._default_tcp_quat = self._compute_default_tcp()
            self.robot.start_board_interaction(freedrive=self._freedrive_enabled)
            self._grasp_request_t = now
            self._phase = "await_grasp"

        elif self._phase == "await_grasp":
            if entering:
                print("[Study] Insert/attach the board...")
            if self.robot.board_state == "holding_board":
                if self._teach_mode:
                    print("[Teach] Board grasped — freedrive enabled. "
                          "Move the board and press M to mark each target.")
                    self._phase = "teach_running"
                elif self._trial_cursor >= len(self._pose_order):
                    print("[Study] All trials for this mode are already complete.")
                    self._phase = "release_board"
                else:
                    print("[Study] Board grasped — returning to the default "
                          "pose before the next trial")
                    self._phase = "reset_to_default"
            elif (self.robot.board_state == "inactive"
                  and now - self._grasp_request_t >= self._GRASP_RETRY_INTERVAL_S):
                print("[Study] No response — resending grasp request...")
                self.robot.start_board_interaction(freedrive=self._freedrive_enabled)
                self._grasp_request_t = now

        elif self._phase == "teach_running":
            # Physical placement itself establishes reachability; target
            # capture is explicitly triggered by the M key.
            return

        elif self._phase == "reset_to_default":
            if not self._auto_move_pending and self._auto_move_result is None:
                self._start_default_joint_reset()
            elif self._auto_move_result is not None:
                ok = self._auto_move_result
                self._auto_move_result = None
                if ok:
                    # The requested reset is complete; begin directly so the
                    # start-policy router does not request the same reset again.
                    self._begin_trial()
                else:
                    print("[Study] Reset-to-default move failed — stopped. "
                          "Check the preceding [Robot] rejection message; "
                          "restart the study after correcting the target or "
                          "constraint.")
                    self._phase = "reset_failed"

        elif self._phase == "reset_failed":
            # Do not automatically resend an identical failed motion every
            # render tick. The robot remains holding the board in its safe
            # post-failure state.
            return

        elif self._phase == "trial_running":
            recording = self._trial_timer_running
            if recording:
                self._update_path_length_accumulators(now)
            if recording:
                if (self._freedrive_enabled
                        and (self.mode != "hybrid"
                             or self._hybrid_freedrive_only)):
                    self._tick_freedrive_channel(now)
            if self._ar_enabled and not self._hybrid_freedrive_only:
                self._tick_ar_channel(
                    now, recording=recording,
                    accept_commands=recording)
                if self._phase != "trial_running":
                    return
            if recording:
                self._tick_target_completion(now)
                if self._phase != "trial_running":
                    return
            if recording and self._force_complete_requested:
                self._force_complete_requested = False
                T_tcp = self.robot.tcp_pose
                if T_tcp is not None and self._trial_target_T is not None:
                    T_board = self._board_pose_from_tcp(T_tcp)
                    pos_err, ang_err = self._pose_error(
                        T_board, self._trial_target_T)
                else:
                    pos_err, ang_err = float('nan'), float('nan')
                self._finish_trial("forced", pos_err, ang_err)

        elif self._phase == "await_snap_confirmation":
            # The timed trial has ended, but AR participants may keep refining
            # the board placement.  Continue servicing release commands while
            # deliberately excluding these moves from timed interaction and
            # path metrics.  ENTER captures the resulting placement as the
            # enter-confirmation error before the deterministic exact snap.
            if self._ar_enabled and not self._hybrid_freedrive_only:
                self._tick_ar_channel(now, recording=False)
            if (self._freedrive_enabled
                    and (self.mode != "hybrid"
                         or self._hybrid_freedrive_only)):
                self._tick_post_stop_freedrive_channel(now)

        elif self._phase == "snap_to_target":
            if self._auto_move_result is None:
                return
            ok = self._auto_move_result
            self._auto_move_result = None
            if ok:
                self._finalize_snapped_trial(now)
            else:
                self._close_status_line()
                print("[Study] Exact-target snap failed — session stopped before "
                      "advancing. Correct the robot/target issue and resume the session.")
                self._replay_event("exact_target_snap_failed")
                self._phase = "snap_failed"

        elif self._phase == "snap_failed":
            return

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
                # Keep the Unity ID-200 robot gripper on the live physical TCP
                # in every phase and mode. GripStateReceiver treats pose_only
                # as transform data and leaves the AR UI state unchanged.
                _live_tcp = self.robot.tcp_pose
                if _live_tcp is not None:
                    self.ar_bridge.publish("pose_only", _live_tcp)
                for interaction_event in self.tools.pop_interaction_events():
                    self._handle_hybrid_gripper_click(interaction_event)

                # Keep the Quest target board visible before marker 100 is
                # locked, matching the always-available Open3D preview.  Once
                # a trial starts, its assigned target takes precedence over
                # the preview cursor.
                T_quest_target = self._trial_target_T
                if (T_quest_target is None and self._pose_order
                        and not self._teach_mode):
                    visual_cursor = (self._target_preview_cursor
                                     if self._manual_target_preview
                                     else self._trial_cursor)
                    visual_cursor = min(visual_cursor,
                                        len(self._pose_order) - 1)
                    T_quest_target = self._poses_T[
                        self._pose_order[visual_cursor]]
                if T_quest_target is not None:
                    T_fake_tcp = np.eye(4)
                    T_fake_tcp[:3, :3] = T_quest_target[:3, :3]
                    T_fake_tcp[:3, 3] = (
                        T_quest_target[:3, 3]
                        - cfg.BOX_FORWARD_OFFSET * T_quest_target[:3, 2])
                    # Calculate immediately from the latest TCP pose instead
                    # of publishing the Open3D state from the previous loop.
                    quest_proximity_state = self._target_display_state(
                        T_quest_target)
                    target_color = self._quest_target_color(
                        quest_proximity_state)
                    self.ghost_bridge.publish(
                        quest_proximity_state, T_fake_tcp,
                        box_color=target_color,
                        gripper_color=target_color)

                if self.anchor.locked and not self._study_started:
                    self._study_started = True
                    self._phase = "await_robot_ready"

                if self.anchor.locked:
                    self.anchor.publish()
                    self.relock_cubes.publish()
                    self.workspace_bound_pub.publish(
                        self._ws_lo, self._ws_hi, self._BOUNDS_VIS_DIST)
                    self._tick_study(_now)
                    self._update_palm_cbf_obstacle(_now)
                    if self._phase == "trial_running":
                        self._print_live_status(_now)

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
                    self._replay_event(
                        "anchor_relocked_by_click",
                        marker_id=self.anchor_marker_id,
                        world_tracking_T=self.anchor.T_world_tracking)
                self.tools.deselect(self.anchor_marker_id)

                # Embedded Open3D mirror: robot + articulated gripper/adapters,
                # actual and target boards, tracking, bounds, and study state.
                self._update_visualizer()
                self._sample_replay(_now)

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
                        T_board = self._board_pose_from_tcp(T_tcp)
                        pe, ae = self._pose_error(T_board, self._trial_target_T)
                        err_str = f"  err={pe*100:.1f}cm/{ae:.1f}deg"
                    n = len(self._pose_order)
                    cv.putText(disp,
                               f"trial {self._trial_cursor+1}/{n}"
                               f"  target pos={np.round(pos,3).tolist()}",
                               (12, 116), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                    cv.putText(disp,
                               f"timer={'RUNNING' if self._trial_timer_running else 'READY'}"
                               f"  elapsed={self._trial_elapsed():.1f}s"
                               f"  interactions={self._trial_interactions}{err_str}",
                               (12, 142), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                _key_help = (
                    "M mark target   U undo   ENTER lock/relock   ESC quit"
                    if self._teach_mode else
                    "ENTER start/snap   P/N or arrows preview targets   "
                    "F force-complete/skip snap   ESC quit")
                cv.putText(disp, _key_help,
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
                        if (self._phase == "trial_running"
                                and self._trial_timer_running):
                            self._force_complete_requested = True
                            self._replay_event("force_complete_requested")
                        elif self._phase in ("await_snap_confirmation",
                                             "snap_failed"):
                            print("[Study] Snap explicitly skipped with F; "
                                  "advancing to the next target from the "
                                  "current board pose.")
                            self._replay_event("exact_target_snap_force_skipped",
                                               move_completed=False)
                            self._finalize_snapped_trial(_now, snap_success=False)
                        elif self._phase == "snap_to_target":
                            print("[Study] The single-thread moveJ is blocking; "
                                  "wait for it to finish before skipping.")
                        elif self._phase == "trial_running":
                            print("[Trial] Force-complete ignored before recording; "
                                  "press ENTER to start first.")
                    elif low in (ord('m'), ord('M')):
                        self._mark_taught_target()
                    elif low in (ord('u'), ord('U')):
                        self._undo_taught_target()
                    elif low in (ord('p'), ord('P')):
                        self._step_target_preview(-1)
                    elif low in (ord('n'), ord('N')):
                        self._step_target_preview(+1)
                    elif (low in (10, 13)
                          or key in (16777220, 16777221)):          # ENTER/RETURN
                        if self._phase in ("trial_running",
                                           "await_snap_confirmation",
                                           "snap_failed"):
                            self._toggle_trial_timer()
                        elif self.cam.camera_T is None:
                            print("[ENTER] No camera pose yet — skipping.")
                        elif anchor_ok:
                            if self.anchor.locked:
                                self.anchor.lock(T_cam_anchor, self.cam.camera_T,
                                                 require_locked=True)
                                self._last_proximity_relock_time = _now
                                print(f"[ENTER] Relocked world to marker #{self.anchor_marker_id}")
                            else:
                                self._lock_initial(T_cam_anchor)
                                self._replay_event(
                                    "anchor_locked_by_keyboard",
                                    marker_id=self.anchor_marker_id)
                                print(f"[ENTER] Locked world to marker #{self.anchor_marker_id}")
                        else:
                            print(f"[ENTER] Marker #{self.anchor_marker_id} not visible.")
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._close_status_line()
        if self._palm_cbf_active:
            self.robot.update_palm_obstacle(None)
            self._palm_cbf_active = False
        self.aruco_worker.stop()
        time.sleep(0.05)
        cv.destroyAllWindows()
        try:
            self.tuner.close()
        except Exception:
            pass
        try:
            self.vis.close()
        except Exception:
            pass
        self._replay_event("session_end", completed_trials=self._trial_cursor)
        self._replay.close()
        print(f"[Study] Completed {self._trial_cursor}/{len(self._pose_order)} trials.")
        print(f"[Study] Replay JSONL: {self._replay_path}")
        print(f"[Study] Replay JSONL: {self._replay_path}")
        if self._teach_mode:
            self._save_taught_targets()
            print(f"[Teach] Saved {len(self._taught_poses)} target(s) → "
                  f"{self._teach_targets_path}")
        self.tools.set_forced_color(self._TCP_TOOL_ID, None)
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
                    help="Identifies this participant; the single replay "
                         "log is named '{session-name}_replay.jsonl' and "
                         "shared across all modes run under this name — "
                         "each mode's records are distinguished by the "
                         "'mode' field. Restarting resumes that mode after its "
                         "last completed trial and leaves other modes intact.")
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
    ap.add_argument("--out-dir", default=str(_DEFAULT_LOG_DIR))
    ap.add_argument(
        "--replay-log", type=Path,
        help="Optional JSONL replay path. By default it is "
             "{out-dir}/{session-name}_replay.jsonl, shared across modes — "
             "this is the only log file; use study2_replay_to_csv.py to "
             "derive tabular CSVs from it.")
    ap.add_argument(
        "--teach-targets", type=Path, metavar="FILE",
        help="Freedrive teaching mode: press M to append the current held-board "
             "pose to FILE and U to remove the last pose.")
    ap.add_argument(
        "--target-poses-file", type=Path, metavar="FILE",
        help="Use board target poses previously recorded by --teach-targets.")
    ap.add_argument(
        "--target-navigation", choices=("preview", "move"), default="preview",
        help="P/N behavior: 'preview' only changes the displayed target; "
             "'move' also commands the held board toward that target using "
             "differential OSC + CBF.")
    ap.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume after the last completed trial for this participant and "
             "mode. The unfinished trial is discarded and repeated. Use "
             "--no-resume to restart this mode from trial 1.")
    args = ap.parse_args()

    if args.anchor_marker == args.pegboard_marker:
        ap.error("--anchor-marker and --pegboard-marker must be different.")
    if args.teach_targets is not None and args.target_poses_file is not None:
        ap.error("--teach-targets and --target-poses-file cannot be used together.")

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
        out_dir                   = Path(args.out_dir),
        teach_targets_path        = args.teach_targets,
        target_poses_path         = args.target_poses_file,
        target_navigation         = args.target_navigation,
        replay_log_path           = args.replay_log,
        resume                    = args.resume,
    )
    print(f"\n[Study] Show marker #{args.anchor_marker} to lock the world "
          f"(auto within {WorkholdingStudy._AUTO_LOCK_MAX_DIST:.1f} m, or press ENTER).")
    study.run()
    print("Bye.")


if __name__ == "__main__":
    main()
