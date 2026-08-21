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
    python workholding_study.py --session-name setup --mode freedrive \
        --teach-targets workholding_targets.json
    python workholding_study.py --session-name P01 --mode ar \
        --target-poses-file workholding_targets.json

Keys: P/N preview previous/next target; M mark and U undo in teaching mode;
ENTER lock/relock marker 100; F force-complete the current trial; ESC quit
(flushes whatever was logged so far).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
        self.vis.register_key_action_callback(ord("S"), self._on_toggle_timer)
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
    _TARGET_NEAR_POS_M         = 0.15    # visual orange proximity band
    _TARGET_NEAR_ANGLE_DEG     = 30.0
    # Strictly for IK candidate validation/preview. These do not change the
    # shared human/AR study completion criterion above.
    _IK_POS_TOL_M             = 0.01
    _IK_ANGLE_TOL_DEG         = 3.0
    _STUDY_DWELL_S            = 1.0     # seconds within tolerance before auto-complete
    _COMPLETION_FLASH_COUNT    = 3
    _COMPLETION_WARNING_S      = 3.0     # stationary warning before reset motion
    _STUDY_MOVE_THRESHOLD_MPS = 0.01    # m/s — freedrive movement-segment detector
    _STUDY_TRAJ_SAMPLE_HZ     = 10.0
    _TARGET_COLOR_NEAR_M      = _STUDY_POS_TOL_M
    _TARGET_COLOR_FAR_M       = 0.30

    # Robustness against robot_control_server.py not being up yet (or dropping a
    # command sent before its ZMQ socket finished connecting — PUB/SUB messages
    # sent before the peer connects are silently lost, not queued).
    _ROBOT_READY_NAG_INTERVAL_S = 5.0   # re-print "still waiting" while disconnected
    _GRASP_RETRY_INTERVAL_S     = 3.0   # resend start_board_interaction if board_state
                                         # never leaves "inactive"

    def __init__(self, quest_ip: str, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 hand_port: int, use_calibrated_robot_base: bool,
                 session_name: str, mode: str, seed: int, out_dir: Path,
                 teach_targets_path: "Path | None" = None,
                 target_poses_path: "Path | None" = None,
                 target_navigation: str = "preview"):
        self.quest_ip           = quest_ip
        self.anchor_marker_id   = anchor_marker_id
        self.pegboard_marker_id = pegboard_marker_id
        self.hand_port          = hand_port
        self.session_name       = session_name
        self.mode                = mode
        self._target_navigation = target_navigation
        self._teach_targets_path = teach_targets_path
        self._teach_mode = teach_targets_path is not None
        self._taught_poses: list[dict] = []
        if (self._teach_targets_path is not None
                and self._teach_targets_path.exists()):
            for index, (pos, euler) in enumerate(
                    self._load_target_poses(self._teach_targets_path)):
                self._taught_poses.append({
                    "target_index": index,
                    "position_m": pos,
                    "euler_xyz_deg": euler,
                })
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

        # ── Trial plan: the 10 poses, shuffled (seeded) ───────────────────────
        self._poses_raw = (self._load_target_poses(target_poses_path)
                           if target_poses_path is not None
                           else cfg.workholding_test_poses())
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
        rng = random.Random(seed)
        self._pose_order = list(range(len(self._poses_raw)))
        # Target 1 is derived from the configured default robot pose. Keep it
        # first so it is also the initial Unity/Open3D preview; randomise the
        # study poses that follow it.
        if len(self._pose_order) > 1:
            shuffled_tail = self._pose_order[1:]
            rng.shuffle(shuffled_tail)
            self._pose_order[1:] = shuffled_tail
        self._trial_cursor = 0
        self._target_preview_cursor = 0
        self._manual_target_preview = False
        self._target_proximity_state = "far"
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
        self._trial_dwell_start: "float | None" = None
        self._trial_interactions = 0
        self._trial_last_traj_t  = 0.0
        self._prev_tcp_pos_for_speed: "np.ndarray | None" = None
        self._prev_tcp_t_for_speed:  "float | None"       = None
        self._was_moving_freedrive = False
        self._force_complete_requested = False
        self._completion_flash_started: "float | None" = None
        self._completion_flash_state: "str | None" = None
        self._status_trial_cursor: "int | None" = None   # for the inline live-offset line
        self._last_status_print_t = 0.0

        # ── Logging ────────────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        self._trials_path = out_dir / f"{session_name}-{mode}_trials.csv"
        self._traj_path   = out_dir / f"{session_name}-{mode}_trajectory.csv"

        # A stable session name is resumable. Completed trials remain in the
        # same files and the deterministic pose order resumes at the next one.
        completed_trial_indices: list[int] = []
        completed_pose_by_trial: dict[int, int] = {}
        if self._trials_path.exists() and self._trials_path.stat().st_size > 0:
            with open(self._trials_path, newline="") as existing_f:
                for row in csv.DictReader(existing_f):
                    if (row.get("session_name") == session_name
                            and row.get("mode") == mode):
                        try:
                            trial_idx = int(row["trial_idx"])
                            pose_idx = int(row["pose_idx"])
                            completed_trial_indices.append(trial_idx)
                            completed_pose_by_trial[trial_idx] = pose_idx
                        except (KeyError, TypeError, ValueError):
                            pass
        self._completed_trial_indices = set(completed_trial_indices)
        while self._trial_cursor in self._completed_trial_indices:
            self._trial_cursor += 1
        if completed_trial_indices:
            for trial_idx, pose_idx in completed_pose_by_trial.items():
                if (0 <= trial_idx < len(self._pose_order)
                        and self._pose_order[trial_idx] != pose_idx):
                    raise RuntimeError(
                        f"Session {session_name}/{mode} was created with a "
                        "different target order (trial "
                        f"{trial_idx + 1}: logged pose#{pose_idx}, current "
                        f"pose#{self._pose_order[trial_idx]}). Use the original "
                        "--seed and target-pose file, or choose a new session name.")
            print(f"[Study] Resuming {session_name}/{mode} at trial "
                  f"{self._trial_cursor + 1}/{len(self._pose_order)}")

        # Trial rows are flushed only on completion, but trajectory samples are
        # streamed during a trial. Remove samples from the first unfinished
        # trial so an interrupted run restarts that trial cleanly from t=0.
        if self._traj_path.exists() and self._traj_path.stat().st_size > 0:
            with open(self._traj_path, newline="") as existing_f:
                reader = csv.reader(existing_f)
                rows = list(reader)
            if rows:
                header, data_rows = rows[0], rows[1:]
                try:
                    session_col = header.index("session_name")
                    mode_col = header.index("mode")
                    trial_col = header.index("trial_idx")
                except ValueError:
                    session_col = mode_col = trial_col = -1
                if trial_col >= 0:
                    kept_rows = []
                    removed_rows = 0
                    for row in data_rows:
                        remove = False
                        try:
                            row_trial_idx = int(row[trial_col])
                            remove = (row[session_col] == session_name
                                      and row[mode_col] == mode
                                      and row_trial_idx not in
                                      self._completed_trial_indices)
                        except (IndexError, TypeError, ValueError):
                            pass
                        if remove:
                            removed_rows += 1
                        else:
                            kept_rows.append(row)
                    if removed_rows:
                        tmp_path = self._traj_path.with_name(
                            self._traj_path.name + ".resume_tmp")
                        with open(tmp_path, "w", newline="") as clean_f:
                            writer = csv.writer(clean_f)
                            writer.writerow(header)
                            writer.writerows(kept_rows)
                        tmp_path.replace(self._traj_path)
                        print(f"[Study] Removed {removed_rows} partial trajectory "
                              f"sample(s) from unfinished trial "
                              f"{self._trial_cursor + 1}; restarting it from zero.")

        trials_need_header = (not self._trials_path.exists()
                              or self._trials_path.stat().st_size == 0)
        traj_need_header = (not self._traj_path.exists()
                            or self._traj_path.stat().st_size == 0)
        self._trials_f = open(self._trials_path, "a", newline="")
        self._trials_writer = csv.writer(self._trials_f)
        if trials_need_header:
            self._trials_writer.writerow(_TRIAL_CSV_HEADER)
            self._trials_f.flush()
        self._traj_f = open(self._traj_path, "a", newline="")
        self._traj_writer = csv.writer(self._traj_f)
        if traj_need_header:
            self._traj_writer.writerow(_TRAJ_CSV_HEADER)
            self._traj_f.flush()
        print(f"[Study] Logging trials  → {self._trials_path}")
        print(f"[Study] Logging traj.   → {self._traj_path}")
        print(f"[Study] Mode: {mode}  (freedrive={'ON' if self._freedrive_enabled else 'off'}, "
              f"AR={'ON' if self._ar_enabled else 'off'})  "
              f"— {len(self._pose_order)} trials, seed={seed}")
        print(f"[Study] P/N target navigation: {self._target_navigation}")

        self._win = ("Workholding Study  "
                     "[S=start/pause  ENTER=lock/relock  F=force-complete  ESC=quit]")
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

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pose_to_T(pos, euler_deg) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('xyz', euler_deg, degrees=True).as_matrix()
        T[:3, 3]  = np.asarray(pos, dtype=np.float64)
        return T

    @staticmethod
    def _load_target_poses(path: Path):
        try:
            payload = json.loads(path.read_text())
            records = payload.get("poses", payload) if isinstance(payload, dict) else payload
            poses = []
            for record in records:
                if isinstance(record, dict):
                    pos = record["position_m"]
                    euler = record["euler_xyz_deg"]
                else:
                    pos, euler = record
                if len(pos) != 3 or len(euler) != 3:
                    raise ValueError("each target requires three position and Euler values")
                poses.append(([float(v) for v in pos],
                              [float(v) for v in euler]))
            if not poses:
                raise ValueError("target file contains no poses")
            print(f"[Study] Loaded {len(poses)} taught target(s) from {path}")
            return poses
        except Exception as exc:
            raise RuntimeError(f"Could not load target poses from {path}: {exc}") from exc

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
        record = {
            "target_index": len(self._taught_poses),
            "position_m": [float(v) for v in T_board[:3, 3]],
            "euler_xyz_deg": [float(v) for v in ScipyR.from_matrix(
                T_board[:3, :3]).as_euler("xyz", degrees=True)],
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

    @classmethod
    def _quest_target_color(cls, T_actual_board: "np.ndarray | None",
                            T_target: np.ndarray) -> list[float]:
        if T_actual_board is None:
            return [0.95, 0.75, 0.08, 0.45]
        pos_err, ang_err = cls._pose_error(T_actual_board, T_target)
        pos_score = 1.0 - np.clip(
            (pos_err - cls._TARGET_COLOR_NEAR_M)
            / max(cls._TARGET_COLOR_FAR_M - cls._TARGET_COLOR_NEAR_M, 1e-6),
            0.0, 1.0)
        ang_score = 1.0 - np.clip(
            (ang_err - cls._STUDY_ANGLE_TOL_DEG) / 45.0, 0.0, 1.0)
        score = float(min(pos_score, ang_score))
        if score < 0.5:
            t = score / 0.5
            color = [0.95, 0.08 + 0.67 * t, 0.08 * (1.0 - t)]
        else:
            t = (score - 0.5) / 0.5
            color = [0.95 * (1.0 - t) + 0.12 * t,
                     0.75 * (1.0 - t) + 0.90 * t,
                     0.08 * (1.0 - t) + 0.20 * t]
        return [float(c) for c in color] + [0.45]
    def _precheck_target_reachability(self) -> "dict[int, bool]":
        """Solve local PyBullet IK once for every desired target TCP pose."""
        scene = self.robot.pb_scene
        saved_q = scene.current_q.copy()
        results = {}
        pos_tol = self._IK_POS_TOL_M
        angle_tol_deg = self._IK_ANGLE_TOL_DEG
        lower = np.deg2rad(np.asarray(cfg.JOINT_MIN_DEG, dtype=float))
        upper = np.deg2rad(np.asarray(cfg.JOINT_MAX_DEG, dtype=float))
        try:
            for pose_idx, T_board in enumerate(self._poses_T):
                T_tcp = np.array(T_board, dtype=np.float64, copy=True)
                T_tcp[:3, 3] -= cfg.BOX_FORWARD_OFFSET * T_tcp[:3, 2]
                quat = ScipyR.from_matrix(T_tcp[:3, :3]).as_quat()
                try:
                    q_ik = scene.solve_ik(
                        saved_q, T_tcp[:3, 3], quat,
                        pos_tol=pos_tol,
                        orient_tol=np.deg2rad(angle_tol_deg))
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
                print(f"[StudyIK] pose#{pose_idx}: {label} "
                      f"({pos_err*100:.1f} cm, {ang_err:.1f} deg)")
        finally:
            scene.update_robot(saved_q)
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
                q_ik = scene.solve_ik(
                    saved_q, tcp_pos, tcp_quat,
                    pos_tol=self._IK_POS_TOL_M,
                    orient_tol=np.deg2rad(self._IK_ANGLE_TOL_DEG))
                scene.update_robot(q_ik)
                self._preview_robot_link_poses = [
                    np.asarray(T, dtype=float).copy()
                    for T in scene.get_arm_link_world_poses()]
                preview_tcp = scene.update_tcp_bodies()
                self._preview_robot_tcp_T = (
                    np.asarray(preview_tcp, dtype=float).copy()
                    if preview_tcp is not None else None)
                print("[StudyVis] Showing robot IK preview (hardware unchanged)")
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
        # The TCP-derived board pose is meaningful for both a held board and
        # the bare-gripper target-navigation check.
        T_actual_board = T_preview_board
        self.vis.update_ar_handle(
            T_preview_board if self._ar_enabled else None)
        reached = False
        proximity_state = "far"
        if T_target is not None and T_actual_board is not None:
            pos_err, ang_err = self._pose_error(T_actual_board, T_target)
            within_reached = (pos_err < self._STUDY_POS_TOL_M
                              and ang_err < self._STUDY_ANGLE_TOL_DEG)
            if within_reached:
                reached = True
                proximity_state = "reached"
            else:
                nearby = (pos_err < self._TARGET_NEAR_POS_M
                          and ang_err < self._TARGET_NEAR_ANGLE_DEG)
                proximity_state = "near" if nearby else "far"
        if T_target is not None:
            display_state = self._completion_flash_state or proximity_state
            self.vis.select_target(target_pose_idx, display_state)
        else:
            display_state = proximity_state
        self._target_proximity_state = display_state
        self.vis.update_target_gripper(
            T_target, cfg.BOX_FORWARD_OFFSET, display_state)

        if self._teach_mode:
            mode_state = "freedrive"
        elif self.mode == "hybrid":
            mode_state = ("ar" if (self._auto_move_pending
                                    or self.robot.board_state == "moving_board")
                          else "freedrive")
        else:
            mode_state = self.mode
        self.vis.update_mode_indicator(T_tcp, mode_state)
        self.vis.tick()

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

    def _start_auto_move(self, pos: np.ndarray, quat: np.ndarray,
                         board_move: bool = True) -> None:
        self._auto_move_pending = True
        self._auto_move_result  = None
        self.robot.move_to_pose(np.asarray(pos, dtype=float),
                                np.asarray(quat, dtype=float),
                                board_move=board_move,
                                motion_profile="workholding",
                                on_complete=self._on_auto_move_complete)

    def _begin_trial(self) -> None:
        pose_idx = self._pose_order[self._trial_cursor]
        self._target_preview_cursor = self._trial_cursor
        self._manual_target_preview = False
        self._target_proximity_state = "far"
        self._trial_target_T          = self._poses_T[pose_idx]
        self._trial_start_t           = 0.0
        self._trial_timer_running     = False
        self._trial_active_elapsed_s  = 0.0
        self._trial_timer_last_t      = None
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
        print("[Trial] Ready — press S to start timing and interaction counting.")
        self._phase = "trial_running"

    def _trial_elapsed(self, now: "float | None" = None) -> float:
        if now is None:
            now = time.time()
        elapsed = self._trial_active_elapsed_s
        if self._trial_timer_running and self._trial_timer_last_t is not None:
            elapsed += now - self._trial_timer_last_t
        return elapsed

    def _toggle_trial_timer(self) -> None:
        if self._phase != "trial_running":
            print("[Trial] S is available while a trial is ready or running.")
            return
        now = time.time()
        if self._trial_timer_running:
            self._trial_active_elapsed_s = self._trial_elapsed(now)
            self._trial_timer_last_t = None
            self._trial_timer_running = False
            self._trial_dwell_start = None
            self._prev_tcp_pos_for_speed = None
            self._prev_tcp_t_for_speed = None
            self._was_moving_freedrive = False
            self._close_status_line()
            print(f"[Trial] PAUSED at {self._trial_active_elapsed_s:.1f}s — "
                  "press S to resume.")
        else:
            if self._trial_start_t <= 0.0:
                self._trial_start_t = now
            self._trial_timer_last_t = now
            self._trial_timer_running = True
            self._trial_dwell_start = None
            self._prev_tcp_pos_for_speed = None
            self._prev_tcp_t_for_speed = None
            self._was_moving_freedrive = False
            print("[Trial] RUNNING — timing and interaction counting enabled.")

    def _start_next_trial_or_finish(self) -> None:
        while self._trial_cursor in self._completed_trial_indices:
            self._trial_cursor += 1
        if self._trial_cursor >= len(self._pose_order):
            self._phase = "release_board"
            return
        self._begin_trial()

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
        timer_state = "RUN" if self._trial_timer_running else "PAUSE"
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
        if self._trial_timer_running:
            self._trial_active_elapsed_s = duration
            self._trial_timer_running = False
            self._trial_timer_last_t = None
        self._trials_writer.writerow([
            self.session_name, self.mode, self._trial_cursor, pose_idx,
            pos[0], pos[1], pos[2], euler[0], euler[1], euler[2],
            self._trial_start_t, now, duration,
            pos_err, ang_err, self._trial_interactions, reason,
        ])
        self._trials_f.flush()
        self._completed_trial_indices.add(self._trial_cursor)
        self._close_status_line()
        print(f"[Trial] done ({reason}) — {duration:.1f}s, "
              f"err={pos_err * 100:.1f}cm/{ang_err:.1f}deg, "
              f"interactions={self._trial_interactions}")
        self._begin_completion_flash()

    def _advance_unrecorded_trial(self, pos_err: float, ang_err: float) -> None:
        """Advance operationally while leaving this trial absent from the CSV."""
        self._close_status_line()
        print(f"[Trial] reached while PAUSED — not recorded "
              f"({pos_err * 100:.1f}cm/{ang_err:.1f}deg); advancing")
        self._trial_dwell_start = None
        self._begin_completion_flash()

    def _begin_completion_flash(self) -> None:
        self._completion_flash_started = time.time()
        self._completion_flash_state = "reached"
        self._phase = "completion_feedback"
        print(f"[Study] Target complete — robot remains stationary for "
              f"{self._COMPLETION_WARNING_S:.1f}s before resetting to default.")

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
            self._trial_elapsed(now),
            *T_tcp[:3, 3].tolist(), *quat.tolist(), *np.degrees(q).tolist(),
        ])

    def _tick_target_completion(self, now: float, recording: bool) -> None:
        """Apply the target dwell; write data only when recording is enabled."""
        if self._auto_move_pending:
            self._trial_dwell_start = None
            return
        T_tcp = self.robot.tcp_pose
        if T_tcp is None or self._trial_target_T is None:
            return
        T_board = self._board_pose_from_tcp(T_tcp)
        pos_err, ang_err = self._pose_error(T_board, self._trial_target_T)
        within = (pos_err < self._STUDY_POS_TOL_M
                 and ang_err < self._STUDY_ANGLE_TOL_DEG)
        if within:
            if self._trial_dwell_start is None:
                self._trial_dwell_start = now
            elif now - self._trial_dwell_start >= self._STUDY_DWELL_S:
                if recording:
                    self._finish_trial("converged", pos_err, ang_err)
                else:
                    self._advance_unrecorded_trial(pos_err, ang_err)
                return
        else:
            self._trial_dwell_start = None

    def _tick_freedrive_channel(self, now: float) -> None:
        """Count physical movement segments; freedrive itself remains server-side."""
        T_tcp = self.robot.tcp_pose
        if T_tcp is None:
            return

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

    def _tick_ar_channel(self, now: float, recording: bool) -> None:
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
                T_board = self._board_pose_from_tcp(T_tcp)
                pos_err, ang_err = self._pose_error(
                    T_board, self._trial_target_T)
                self._close_status_line()
                print(f"[AR] Landed {pos_err*100:.1f}cm/{ang_err:.1f}deg from target "
                      + ("— target completion is paused"
                         if not recording else "— checking target dwell"))
            elif not ok:
                self._close_status_line()
                print("[AR] Move cancelled/failed — try again")

        T_box_target = self.ar_bridge.poll()
        if T_box_target is not None:
            if recording:
                self._trial_interactions += 1
            tcp_pos  = (T_box_target[:3, 3]
                       - cfg.BOX_FORWARD_OFFSET * T_box_target[:3, 2])
            tcp_quat = ScipyR.from_matrix(T_box_target[:3, :3]).as_quat()
            self._close_status_line()
            release_label = (f"#{self._trial_interactions}"
                             if recording else "(not recorded)")
            print(f"[AR] Release {release_label} "
                  f"→ TCP {np.round(tcp_pos, 3).tolist()}")
            self._start_auto_move(tcp_pos, tcp_quat)

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
                else:
                    print("[Study] Board grasped — beginning trials")
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
                self._start_auto_move(self._default_tcp_pos, self._default_tcp_quat)
            elif self._auto_move_result is not None:
                ok = self._auto_move_result
                self._auto_move_result = None
                if ok:
                    self._start_next_trial_or_finish()
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
                self._sample_trajectory(now)
            self._tick_target_completion(now, recording=recording)
            if self._phase != "trial_running":
                return
            if recording:
                if self._freedrive_enabled:
                    self._tick_freedrive_channel(now)
            if self._ar_enabled:
                self._tick_ar_channel(now, recording=recording)
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

        elif self._phase == "completion_feedback":
            if self._completion_flash_started is None:
                self._completion_flash_started = now
            # Three pulses are G-B-G-B-G; there is no trailing black segment
            # before the reset motion begins.
            total_segments = self._COMPLETION_FLASH_COUNT * 2 - 1
            segment_duration = self._COMPLETION_WARNING_S / total_segments
            segment = int((now - self._completion_flash_started)
                          / segment_duration)
            if segment >= total_segments:
                self._completion_flash_state = None
                self._completion_flash_started = None
                self._trial_cursor += 1
                self._phase = "reset_to_default"
            else:
                self._completion_flash_state = (
                    "reached" if segment % 2 == 0 else "black")

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
                    target_color = self._quest_target_color(
                        self._board_pose_from_tcp(self.robot.tcp_pose),
                        T_quest_target)
                    self.ghost_bridge.publish(
                        self._target_proximity_state, T_fake_tcp,
                        box_color=target_color)

                if self.anchor.locked and not self._study_started:
                    self._study_started = True
                    self._phase = "await_robot_ready"

                if self.anchor.locked:
                    self.anchor.publish()
                    self.relock_cubes.publish()
                    self.workspace_bound_pub.publish(
                        self._ws_lo, self._ws_hi, self._BOUNDS_VIS_DIST)
                    self._tick_study(_now)
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
                self.tools.deselect(self.anchor_marker_id)

                # Embedded Open3D mirror: robot + articulated gripper/adapters,
                # actual and target boards, tracking, bounds, and study state.
                self._update_visualizer()

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
                               f"timer={'RUNNING' if self._trial_timer_running else 'PAUSED'}"
                               f"  elapsed={self._trial_elapsed():.1f}s"
                               f"  interactions={self._trial_interactions}{err_str}",
                               (12, 142), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                _key_help = (
                    "M mark target   U undo   ENTER lock/relock   ESC quit"
                    if self._teach_mode else
                    "S start/pause   P/N or arrows preview targets   "
                    "ENTER lock/relock   F force-complete   ESC quit")
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
                        elif self._phase == "trial_running":
                            print("[Trial] Force-complete ignored while paused; "
                                  "press S to resume recording first.")
                    elif low in (ord('s'), ord('S')):
                        self._toggle_trial_timer()
                    elif low in (ord('m'), ord('M')):
                        self._mark_taught_target()
                    elif low in (ord('u'), ord('U')):
                        self._undo_taught_target()
                    elif low in (ord('p'), ord('P')):
                        self._step_target_preview(-1)
                    elif low in (ord('n'), ord('N')):
                        self._step_target_preview(+1)
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
        self._close_status_line()
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
        self._trials_f.flush(); self._trials_f.close()
        self._traj_f.flush(); self._traj_f.close()
        print(f"[Study] Completed {self._trial_cursor}/{len(self._pose_order)} trials.")
        print(f"[Study] Trials CSV: {self._trials_path}")
        print(f"[Study] Trajectory CSV: {self._traj_path}")
        if self._teach_mode:
            self._save_taught_targets()
            print(f"[Teach] Saved {len(self._taught_poses)} target(s) → "
                  f"{self._teach_targets_path}")
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
                         "'{session-name}-{mode}_*.csv'. Reusing the same "
                         "name and mode appends and resumes the session.")
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
        seed                      = args.seed,
        out_dir                   = Path(args.out_dir),
        teach_targets_path        = args.teach_targets,
        target_poses_path         = args.target_poses_file,
        target_navigation         = args.target_navigation,
    )
    print(f"\n[Study] Show marker #{args.anchor_marker} to lock the world "
          f"(auto within {WorkholdingStudy._AUTO_LOCK_MAX_DIST:.1f} m, or press ENTER).")
    study.run()
    print("Bye.")


if __name__ == "__main__":
    main()
