#!/usr/bin/env python3
"""Study 4: task-aware multimodal part-acquisition experiment.

The model is intentionally independent from the GUI so that the dependency and
part-transformation rules can be tested without opening a window.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import multiprocessing
import queue
import re
import sys
import time
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Iterable

import dearpygui.dearpygui as dpg
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

# Ensure task_graph/ (this dir) and the repo root (its parent) are both importable, so we can pull
# the canonical ports from main_setting.py even though the viewer runs from the task_graph/ subdir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study4_vlm_assistant import VLMAssistant  # noqa: E402
from speech_listener import SpeechListener, DEFAULT_PULSEAUDIO_DEVICE  # noqa: E402
from tts import TTSService, DEFAULT_PIPER_MODEL  # noqa: E402
import gearbox_control  # noqa: E402  (--with-controller: run the controller in-process)

# Ports live canonically in main_setting.py; fall back to literals so the viewer still runs if that
# import is unavailable.
try:
    import main_setting
    _LOCALHOST                    = main_setting.LOCALHOST                  # loopback for same-machine Python IPC
    # IN  — gearbox_control.py PUBs semantic assembly events (show/complete/uncomplete/reset) here
    #        after every Unity click; this viewer SUBs and drives its TaskGraph model + node colors.
    _DEFAULT_CTRL_EVENTS_IN_PORT  = main_setting.GEARBOX_TASKGRAPH_PORT    # 5022
    # OUT — when a step node is selected in this viewer, it PUBs the matching (row, stage) here so
    #        that a gearbox_control.py --open-3d window can highlight the corresponding pegboard tools.
    _DEFAULT_STEP_OUT_PORT        = main_setting.GEARBOX_STEP_SELECT_PORT  # 5025
    #In one sentence: 5022 is the viewer listening to assembly progress from Unity/controller; 
    # 5025 is the viewer broadcasting which step you've clicked so a 3D tool viewer can react.
except Exception:
    _LOCALHOST                = "127.0.0.1"
    _DEFAULT_CTRL_EVENTS_IN_PORT   = 5022   # IN  — gearbox_control.py -> this viewer (assembly events)
    _DEFAULT_STEP_OUT_PORT = 5025   # OUT — this viewer -> gearbox_control.py --open-3d (selected step)


def _study4_tool_layout(condition: str | None) -> Path:
    """Return the shared physical layout used by every Study 4 condition."""
    return (Path(__file__).resolve().parent.parent
            / "scene_layout" / "tool_layout1.json")


@dataclass(frozen=True)
class Step:
    id: str                    # e.g. "r2_bearing_left"
    title: str                 # e.g. "Row 2.1: bearing into left stand"
    row: int                   # e.g. 2  (1-4 for regular rows; 0 for finish_gearbox)
    stage: int                 # e.g. 0  (0=bearings, 1=gear rod, 2=fasten first, 3=insert rod+fit second, 4=fasten second, 5=crank handle, 6=verify)
    inputs: tuple[str, ...]    # e.g. ("BEARING_ROW2_LEFT", "STAND_ROW2_LEFT")
    output: str                # e.g. "BEARING_STAND_ROW2_LEFT_ASSEMBLY"
    description: str           # e.g. "Insert BEARING_ROW2_LEFT into STAND_ROW2_LEFT."
    context: tuple[str, ...] = ()  # e.g. ("BASE_BOARD",) — parts present but not consumed
    requires: tuple[str, ...] = () # completed step IDs; unlike context, their outputs may be consumed


class AssistantInteractionLogger:
    """Append machine-readable part-reference decisions for study analysis."""

    FIELDS = (
        "event_index", "timestamp", "study_condition", "transcript", "vlm_prediction", "vlm_raw",
        "selected_step", "selected_step_state", "graph_decision", "matched_parts",
        "current_assemblies", "highlight_ids", "assembly_highlight_parts",
        "spoken_response",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_index = 1
        if self.path.exists() and self.path.stat().st_size:
            try:
                with self.path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    old_fields = tuple(reader.fieldnames or ())
                    rows = list(reader)
                    indices = [int(row.get("event_index", 0) or 0) for row in rows]
                self._next_index = max(indices, default=0) + 1
                if old_fields != self.FIELDS:
                    with self.path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                        writer.writeheader()
                        for row in rows:
                            writer.writerow({field: row.get(field, "")
                                             for field in self.FIELDS})
            except (OSError, ValueError):
                self._next_index = 1

    def append(self, **values) -> None:
        with self._lock:
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            row = {field: values.get(field, "") for field in self.FIELDS}
            row["event_index"] = self._next_index
            row["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
            for field in ("matched_parts", "current_assemblies", "highlight_ids",
                          "assembly_highlight_parts"):
                if not isinstance(row[field], str):
                    row[field] = json.dumps(row[field], ensure_ascii=False)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self._next_index += 1


class PartAcquisitionClickLogger:
    """Append every scored physical pegboard click for error/latency analysis."""

    FIELDS = (
        "timestamp", "participant_id", "condition", "event_type", "selected_step",
        "selected_step_state", "step_selected_at", "response_time_s",
        "clicked_tool_id", "clicked_tool_name", "expected_tool_ids",
        "acquired_tool_ids", "required_count", "acquired_count",
        "correct", "newly_acquired", "hand",
    )

    def __init__(self, path: str | Path, participant_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.participant_id = participant_id
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                fields = tuple(csv.DictReader(handle).fieldnames or ())
            if fields != self.FIELDS:
                backup = self.path.with_name(
                    f"{self.path.stem}_legacy_{int(time.time())}{self.path.suffix}")
                self.path.replace(backup)
                print(f"[StudyLog] Archived legacy click log -> {backup}")

    def append(self, **values) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        row = {field: values.get(field, "") for field in self.FIELDS}
        row["timestamp"] = datetime.now().astimezone().isoformat(
            timespec="milliseconds")
        row["participant_id"] = self.participant_id
        for field in ("expected_tool_ids", "acquired_tool_ids"):
            if not isinstance(row[field], str):
                row[field] = json.dumps(row[field])
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


class PartAcquisitionStepLogger:
    """One analysis-ready row per completed acquisition step."""

    FIELDS = (
        "timestamp", "participant_id", "condition", "step_id", "step_title",
        "duration_s", "required_tool_ids", "required_tool_names",
        "total_clicks", "wrong_clicks", "repeated_correct_clicks",
    )

    def __init__(self, path: str | Path, participant_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.participant_id = participant_id

    def append(self, **values) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        row = {field: values.get(field, "") for field in self.FIELDS}
        row["timestamp"] = datetime.now().astimezone().isoformat(
            timespec="milliseconds")
        row["participant_id"] = self.participant_id
        for field in ("required_tool_ids", "required_tool_names"):
            if not isinstance(row[field], str):
                row[field] = json.dumps(row[field], ensure_ascii=False)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
                writer.writerow(row)


class Study4SessionLogger:
    """Completed-step CSV plus an atomic live JSON resume checkpoint."""

    FIELDS = (
        "timestamp", "participant_id", "condition", "event_type", "modality",
        "step_id", "step_title", "step_state", "step_attempt",
        "step_event_index",
        "tool_id", "tool_name", "part_elapsed_s",
        "step_elapsed_s", "head_translation_m", "head_rotation_deg",
        "correct", "transcript", "vlm_prediction",
        "vlm_raw", "graph_decision", "spoken_response", "state_json",
    )

    def __init__(self, path: str | Path, participant_id: str,
                 condition: str, *, fresh_run: bool = False) -> None:
        self.path = Path(path)
        self.state_path = self.path.with_name(f"{self.path.stem}_state.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.participant_id = participant_id
        self.condition = condition
        self._lock = threading.Lock()
        self._pending_rows: list[dict[str, object]] = []
        self._latest_state: dict[str, object] | None = None
        if fresh_run:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for existing in (self.path, self.state_path):
                if existing.exists() and existing.stat().st_size:
                    backup = existing.with_name(
                        f"{existing.stem}_archived_{stamp}{existing.suffix}")
                    existing.replace(backup)
                    print(f"[StudyLog] Archived fresh-run file -> {backup}")
        if self.path.exists() and self.path.stat().st_size:
            try:
                with self.path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    fields = tuple(reader.fieldnames or ())
                    rows = list(reader)
                if fields != self.FIELDS:
                    backup = self.path.with_name(
                        f"{self.path.stem}_legacy_{int(time.time())}"
                        f"{self.path.suffix}")
                    self.path.replace(backup)
                    attempts: dict[tuple[str, str], int] = {}
                    event_indices: dict[tuple[str, str], int] = {}
                    with self.path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                        writer.writeheader()
                        for old_row in rows:
                            key = (old_row.get("participant_id", ""),
                                   old_row.get("condition", ""))
                            if old_row.get("event_type") in {
                                    "step_selected", "step_started"}:
                                attempts[key] = attempts.get(key, 0) + 1
                                event_indices[key] = 0
                            attempt = attempts.get(key, 0)
                            if attempt:
                                event_indices[key] = event_indices.get(key, 0) + 1
                            migrated = {
                                field: old_row.get(field, "")
                                for field in self.FIELDS
                            }
                            migrated["step_attempt"] = attempt or ""
                            migrated["step_event_index"] = (
                                event_indices.get(key, "") if attempt else "")
                            if old_row.get("state_json"):
                                try:
                                    state = json.loads(old_row["state_json"])
                                    if isinstance(state, dict):
                                        state["step_attempt"] = attempt
                                        state["step_event_index"] = (
                                            event_indices.get(key, 0))
                                        migrated["state_json"] = json.dumps(
                                            state, ensure_ascii=False,
                                            sort_keys=True)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            writer.writerow(migrated)
                    print(f"[StudyLog] Migrated session log; backup -> {backup}")
            except OSError:
                pass
        if not fresh_run:
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.state_path.exists() or not self.state_path.stat().st_size:
            self._import_legacy_csv_checkpoint()
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (payload.get("participant_id") != self.participant_id
                    or payload.get("condition") != self.condition):
                return
            state = payload.get("state")
            pending = payload.get("pending_events", [])
            self._latest_state = state if isinstance(state, dict) else None
            self._pending_rows = (
                list(pending) if isinstance(pending, list) else [])
        except (OSError, json.JSONDecodeError, TypeError):
            self._latest_state = None
            self._pending_rows = []

    def _import_legacy_csv_checkpoint(self) -> None:
        """Convert the former all-events CSV into the hybrid representation."""
        if not self.path.exists() or not self.path.stat().st_size:
            return
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            completed_attempts = {
                (row.get("step_id", ""), row.get("step_attempt", ""))
                for row in rows if row.get("event_type") == "graph_complete"
                and row.get("step_attempt")
            }
            committed = [
                row for row in rows
                if (row.get("step_id", ""), row.get("step_attempt", ""))
                in completed_attempts
            ]
            pending = [
                row for row in rows
                if row.get("step_attempt")
                and (row.get("step_id", ""), row.get("step_attempt", ""))
                not in completed_attempts
            ]
            for row in reversed(rows):
                raw_state = row.get("state_json", "")
                if not raw_state:
                    continue
                state = json.loads(raw_state)
                if isinstance(state, dict):
                    self._latest_state = state
                    break
            self._pending_rows = pending
            if len(committed) != len(rows):
                backup = self.path.with_name(
                    f"{self.path.stem}_pre_hybrid_{int(time.time())}"
                    f"{self.path.suffix}")
                self.path.replace(backup)
                if committed:
                    self._commit_rows(committed)
                print(f"[StudyLog] Converted legacy CSV; backup -> {backup}")
            if self._latest_state is not None:
                self._write_checkpoint()
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            self._latest_state = None
            self._pending_rows = []

    def _write_checkpoint(self) -> None:
        payload = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"),
            "state": self._latest_state or {},
            "pending_events": self._pending_rows,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
            handle.flush()
        temporary.replace(self.state_path)

    def _commit_rows(self, rows: list[dict[str, object]]) -> None:
        """Atomically store one latest completed event block per task step."""
        if not rows:
            return
        replacement_steps = {
            str(row.get("step_id", "")) for row in rows
            if row.get("step_id")
        }
        retained: list[dict[str, object]] = []
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                retained = [
                    dict(existing) for existing in csv.DictReader(handle)
                    if existing.get("step_id", "") not in replacement_steps
                ]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(retained)
            writer.writerows(rows)
            handle.flush()
        temporary.replace(self.path)

    def append(self, **values) -> None:
        with self._lock:
            row = {field: values.get(field, "") for field in self.FIELDS}
            row["timestamp"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds")
            row["participant_id"] = self.participant_id
            row["condition"] = self.condition
            state = row["state_json"]
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    state = {}
            self._latest_state = state if isinstance(state, dict) else {}
            row["state_json"] = json.dumps(
                self._latest_state, ensure_ascii=False, sort_keys=True)

            event_type = str(row["event_type"])
            has_attempt = bool(row["step_attempt"])
            if event_type == "step_started":
                self._pending_rows = [row]
            elif event_type == "step_attempt_abandoned":
                self._pending_rows = []
            elif has_attempt:
                self._pending_rows.append(row)

            if event_type == "graph_complete":
                if self._pending_rows:
                    self._commit_rows(self._pending_rows)
                    self._pending_rows = []
                else:
                    self._commit_rows([row])
            self._write_checkpoint()

    def latest_state(self) -> dict[str, object] | None:
        return dict(self._latest_state) if self._latest_state is not None else None

    def max_step_attempt(self) -> int:
        """Return the largest attempt number for this participant/condition."""
        maximum = 0
        if self.path.exists() and self.path.stat().st_size:
            try:
                with self.path.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        if (row.get("participant_id") == self.participant_id
                                and row.get("condition") == self.condition):
                            maximum = max(
                                maximum, int(row.get("step_attempt", 0) or 0))
            except (OSError, ValueError):
                maximum = 0
        if self._latest_state is not None:
            maximum = max(
                maximum, int(self._latest_state.get("step_attempt", 0) or 0))
        return maximum


class Study4Open3DScene:
    """Live Study 4 scene without any robot-control responsibilities.

    The perception/anchoring primitives are shared with ``main_with_robot`` so
    Study 4 uses the identical Unity coordinate conversion and marker-locking
    protocol.  Importing those classes does *not* construct MainScene or a
    RobotClient; this class owns only camera, hand, marker, pegboard, relock,
    tool-interaction, BoardAR, and Open3D resources.
    """

    AUTO_LOCK_MAX_DIST = 1.0
    AUTO_LOCK_MAX_TILT_DEG = 45.0
    RELOCK_COOLDOWN = 2.0

    def __init__(self, unity_ip: str, layout_path: str | Path,
                 on_part_click=None, on_interaction=None) -> None:
        scene_started = time.perf_counter()

        def scene_startup(message: str) -> None:
            elapsed = time.perf_counter() - scene_started
            print(f"[SCENE   +{elapsed:6.2f}s] {message}", flush=True)

        scene_startup(
            "Importing Open3D (native-library loading can take 30–60 seconds)...")
        from scene_viewer_o3d import SceneVis
        scene_startup("Open3D import complete; importing shared marker runtime...")
        import main_with_robot as scene_runtime
        scene_startup("Shared marker runtime import complete; importing ZMQ...")
        import zmq
        scene_startup("All scene runtime imports complete")
        # Empty placeholder geometries are normal in this dynamic scene; hide
        # Open3D's repeated zero-point AABB warnings while retaining errors.
        scene_runtime.o3d.utility.set_verbosity_level(
            scene_runtime.o3d.utility.VerbosityLevel.Error)
        self._rt = scene_runtime
        self._on_part_click = on_part_click
        self._on_interaction = on_interaction
        self.vis = SceneVis("Study 4 — Live Part Acquisition",
                            board_ar_asset="HalfBoard.obj",
                            load_robot_assets=False)
        scene_startup("Open3D boards and gearbox mirror loaded (robot assets skipped)")
        self._color_pub = zmq.Context.instance().socket(zmq.PUB)
        self._color_pub.connect(
            f"tcp://{unity_ip}:{main_setting.TOOL_COLOR_PORT}")
        scene_layout_dir = Path(__file__).resolve().parent.parent / "scene_layout"
        layout_path = Path(layout_path)
        pose_path = scene_layout_dir / "T_world10_pegboard101.npz"
        layout = json.loads(layout_path.read_text())
        self.tool_defs = list(layout.get("tools", []))
        data = np.load(pose_path)
        self.T_pegboard = np.asarray(data["T_world10_pegboard"], dtype=float)
        self.id_to_index = {
            int(tool["id"]): index for index, tool in enumerate(self.tool_defs)
        }
        self._step_ids: set[int] = set()
        self._reference_colors: dict[int, list[float]] = {}
        self._reference_colors_until = 0.0
        self.vis.set_pegboard_outline(
            offset_x=float(data["marker_offset_right_m"]),
            offset_y=float(data["marker_offset_top_m"]),
            width=float(data["pegboard_width_m"]),
            height=float(data["pegboard_height_m"]))
        self.vis.update_pegboard(self.T_pegboard)
        # Live Quest/Unity stack.  Deliberately absent: RobotClient, IK, CBF,
        # workspace publisher, gripper bridge, and all grasp/fetch logic.
        cfg = scene_runtime.cfg
        self.cam = scene_runtime._CamFeedReceiver(unity_ip)
        marker_ids = scene_runtime._load_prescan_marker_ids()
        aruco = scene_runtime._ArucoPoseEstimator(
            anchor_marker_id=cfg.ANCHOR_MARKER_ID,
            pegboard_marker_id=cfg.PEGBOARD_MARKER_ID,
            anchor_marker_size_m=cfg.ANCHOR_MARKER_SIZE,
            pegboard_marker_size_m=cfg.PEGBOARD_MARKER_SIZE,
            board_marker_ids=(cfg.BOARD_MARKER_A_ID,
                              cfg.BOARD_MARKER_B_ID, *marker_ids),
            board_marker_size_m=cfg.WORLD_MARKER_SIZE)
        self.aruco_worker = scene_runtime._ArUcoWorker(self.cam, aruco)
        self.hands = scene_runtime._HandDataReceiver(
            unity_ip, cfg.HAND1_PORT_FROM_UNITY)
        self.anchor = scene_runtime._WorldAnchor(unity_ip)
        self.tool_interaction = scene_runtime._ToolSelectionManager(unity_ip)
        self.tool_layout = scene_runtime._ToolLayoutManager(layout_path, unity_ip)
        self.relock_cubes = scene_runtime._RelockCubePublisher(unity_ip)
        self.relock_cubes.set_markers(self.anchor._T_world_marker)
        self.gearbox_pose = scene_runtime._GearboxPoseReceiver(unity_ip)
        scene_startup("Camera, markers, hands, and Unity receivers initialized")
        self._board_offsets = {
            cfg.BOARD_MARKER_A_ID: scene_runtime.T_BOARD_FROM_MARKER_A,
            cfg.BOARD_MARKER_B_ID: scene_runtime.T_BOARD_FROM_MARKER_B,
        }
        self._last_publish = 0.0
        self._last_relock = 0.0
        self._anchor_flash_until = 0.0
        self._anchor_available = False
        self._manual_lock_requested = False
        self._world_available = {mid: False for mid in self.anchor._T_world_marker}
        self._world_last_relock = {mid: 0.0 for mid in self.anchor._T_world_marker}
        self._wrong_color_until: dict[int, float] = {}
        self._removed_tool_ids: set[int] = set()
        self._acquired_tool_ids: set[int] = set()
        self._head_motion_recording = False
        self._head_motion_last: np.ndarray | None = None
        self._head_translation_m = 0.0
        self._head_rotation_rad = 0.0
        self._camera_window = "Study 4 - Quest passthrough (Enter=lock)"
        self._camera_window_ok = False
        try:
            scene_runtime.cv.namedWindow(
                self._camera_window, scene_runtime.cv.WINDOW_NORMAL)
            scene_runtime.cv.resizeWindow(self._camera_window, 960, 540)
            self._camera_window_ok = True
        except Exception as error:
            print(f"[Study4Scene] Camera window disabled: {error}")

        # A saved pegboard is visible immediately; marker 101 can replace this
        # pose after marker 100 is locked and Enter is pressed.
        self.anchor.set_pegboard(self.T_pegboard)
        self._update_pegboard(self.T_pegboard, publish=False)
        self._set_category_colors()
        scene_startup("Pegboard geometry and initial colors ready")
        print(f"[Study4Scene] Live camera/markers/hands enabled; "
              f"loaded {len(self.tool_defs)} pegboard objects")
        print("[Study4Scene] Marker 100 auto-locks nearby; Enter manually locks/relocks "
              "marker 100 and captures marker 101")

    def request_manual_lock(self) -> None:
        self._manual_lock_requested = True

    def start_head_motion_summary(self) -> None:
        """Begin accumulating head-path translation and orientation change."""
        self._head_motion_recording = True
        self._head_motion_last = None
        self._head_translation_m = 0.0
        self._head_rotation_rad = 0.0

    def stop_head_motion_summary(self) -> dict[str, float]:
        """Stop and return accumulated motion for the current timed step."""
        self._head_motion_recording = False
        self._head_motion_last = None
        return {
            "head_translation_m": self._head_translation_m,
            "head_rotation_deg": float(np.degrees(self._head_rotation_rad)),
        }

    def head_motion_summary(self) -> dict[str, float]:
        return {
            "head_translation_m": self._head_translation_m,
            "head_rotation_deg": float(np.degrees(self._head_rotation_rad)),
        }

    def reset_acquisition_colors(self) -> None:
        """Restore pegboard colors when a new step-acquisition trial starts."""
        self._acquired_tool_ids.clear()
        for tool_id in self.id_to_index:
            self.tool_interaction.set_forced_color(tool_id, None)
        self._wrong_color_until.clear()

    def mark_acquired_objects(self, tool_ids: Iterable[int]) -> None:
        """Keep found objects visible in green until their step completes."""
        acquired = {
            int(tool_id) for tool_id in tool_ids
            if int(tool_id) in self.id_to_index
        }
        self._acquired_tool_ids.update(acquired)
        for tool_id in acquired:
            self.tool_interaction.set_forced_color(
                tool_id, self._rt._ToolSelectionManager.SELECTED_COLOR)
            self.tool_interaction.deselect(tool_id)
        self._refresh_pegboard_colors()

    def remove_acquired_objects(self, tool_ids: Iterable[int]) -> None:
        """Remove newly acquired objects from Unity and Open3D immediately."""
        removable = {
            int(tool_id) for tool_id in tool_ids
            if int(tool_id) in self.id_to_index
        }
        self._removed_tool_ids.update(removable)
        self._acquired_tool_ids.difference_update(removable)
        for tool_id in self._removed_tool_ids:
            self.tool_layout.mark_delivered(tool_id)
            self.tool_interaction.deselect(tool_id)
        hidden = [self.id_to_index[tool_id]
                  for tool_id in self._removed_tool_ids
                  if tool_id in self.id_to_index]
        self.vis.set_tool_hidden_indices(hidden)
        self.vis.update_tool_boxes(self.tool_layout.world_boxes(self.T_pegboard))
        self.tool_layout.publish(self.T_pegboard)
        if removable:
            print(f"[Study4Scene] Removed consumed pegboard parts: "
                  f"{sorted(removable)}")

    def restore_acquired_objects(self) -> None:
        """Restock every Study 4 pegboard object after a full study reset."""
        self._removed_tool_ids.clear()
        self._acquired_tool_ids.clear()
        self.tool_layout.reset_delivered()
        self.vis.set_tool_hidden_indices([])
        self.vis.update_tool_boxes(self.tool_layout.world_boxes(self.T_pegboard))
        self.tool_layout.publish(self.T_pegboard)
        self.reset_acquisition_colors()

    def _set_category_colors(self) -> None:
        manager = self._rt._ToolSelectionManager
        for tool in self.tool_defs:
            color = (manager.PART_COLOR if tool.get("category") == "part"
                     else manager.TOOL_COLOR)
            self.tool_interaction.set_category_color(int(tool["id"]), color)

    def _update_pegboard(self, T: np.ndarray, publish: bool = True) -> None:
        self.T_pegboard = np.asarray(T, dtype=float)
        self.vis.update_pegboard(self.T_pegboard)
        self.vis.update_tool_boxes(self.tool_layout.world_boxes(self.T_pegboard))
        if publish:
            self.tool_layout.publish(self.T_pegboard)

    def _world_box(self, tool: dict) -> tuple:
        size = np.asarray(tool.get("size", [0.05, 0.05, 0.05]), float)
        rotation = tool.get("rotation_deg", [0.0, 0.0, 0.0])
        R_world = ScipyR.from_euler(
            "z", float(rotation[2]), degrees=True).as_matrix()
        base = ((self.T_pegboard
                 @ np.append(tool["peg_pos"], 1.0))[:3]
                if "peg_pos" in tool else
                np.asarray(tool.get("world_pos", [0.0, 0.0, 0.0]), float))
        center = base + R_world @ np.array([0.0, 0.0, size[2] / 2.0])
        return center, R_world, size

    def _refresh_pegboard_colors(self) -> None:
        # Keep the canonical interaction manager aware of semantic colors so
        # its periodic late-subscriber refresh cannot overwrite Study 4's
        # cyan step context or yellow language referent.
        self.tool_interaction._apply_layered_highlight(
            self._step_ids, self._reference_colors.keys())
        colors = {
            self.id_to_index[tool_id]: [0.0, 1.0, 1.0]
            for tool_id in self._step_ids if tool_id in self.id_to_index
        }
        colors.update({
            self.id_to_index[tool_id]: color[:3]
            for tool_id, color in self._reference_colors.items()
            if tool_id in self.id_to_index
        })
        colors.update({
            self.id_to_index[tool_id]: [0.1, 1.0, 0.1]
            for tool_id in self._acquired_tool_ids
            if tool_id in self.id_to_index
        })
        self.vis.set_tool_highlight_colors(colors)
        for index, tool in enumerate(self.tool_defs):
            rgb = colors.get(index)
            rgba = ((list(rgb) + [0.15]) if rgb is not None else
                    ([1.0, 0.78, 0.78, 0.15]
                     if tool.get("category") == "part" else
                     [0.80, 0.88, 1.0, 0.15]))
            self._color_pub.send_string(json.dumps({
                "tool_id": int(tool["id"]), "color": rgba,
            }))

    def apply(self, event: dict) -> None:
        name = event.get("event")
        if name == "step_context_highlight":
            self._step_ids = {int(item) for item in event.get("step_ids", [])}
            self._refresh_pegboard_colors()
        elif name == "reference_highlight":
            color = list(event.get("color", [0.0, 1.0, 1.0]))
            referred = {int(item) for item in event.get("referred_ids", [])}
            # Reference feedback is an overlay, not a replacement for the
            # persistent Condition-3 cyan step context.
            self._reference_colors = {
                int(item): list(event.get(
                    "referent_color", [1.0, 0.92, 0.02, 0.15]))
                for item in referred
            }
            self._reference_colors_until = (
                time.monotonic() + 1.5 if referred else 0.0)
            self._refresh_pegboard_colors()
            self.vis.apply_gearbox_assembly_event({
                "event": "assembly_reference",
                "parts": event.get("assembly_parts", []),
                "color": event.get("assembly_color", color),
            })
        elif name == "board_step_highlight":
            self.vis.apply_gearbox_assembly_event({
                "event": "assembly_reference",
                "parts": event.get("assembly_parts", []),
                "color": event.get("assembly_color", [0.0, 1.0, 1.0]),
            })
        elif name == "select":
            # Selecting/opening a task step must not reveal which physical
            # pegboard objects are needed. Study 4 measures acquisition, so
            # that automatic cyan cue would disclose the answer.
            self._step_ids.clear()
            self._reference_colors.clear()
            self._reference_colors_until = 0.0
            self._refresh_pegboard_colors()
        elif name in {"clear", "reset"}:
            self._step_ids.clear()
            self._reference_colors.clear()
            self._reference_colors_until = 0.0
            self.tool_interaction._apply_highlight_clear(
                clear_step_context=True)
            self._refresh_pegboard_colors()
            self.vis.apply_gearbox_assembly_event({"event": "reset"})
        elif name in {"complete", "uncomplete", "show", "close"}:
            self.vis.apply_gearbox_assembly_event(event)

    def tick(self) -> None:
        self.tool_interaction.poll(timeout_ms=0)
        for interaction in self.tool_interaction.pop_interaction_events():
            tool_id = int(interaction["tool_id"])
            if (interaction.get("event_type") != "selected"
                    and self._on_interaction is not None):
                self._on_interaction(interaction)
            if (interaction.get("event_type") == "selected"
                    and tool_id in self.id_to_index
                    and self._on_part_click is not None):
                correct = bool(self._on_part_click(interaction))
                if not correct:
                    self.tool_interaction.set_forced_color(
                        tool_id, [1.0, 0.0, 0.0, 0.25])
                    self._wrong_color_until[tool_id] = time.monotonic() + 1.0
                    self.tool_interaction.deselect(tool_id)
                else:
                    # Keep every acquired object green until the participant
                    # has collected the complete set for this step.
                    self.tool_interaction.set_forced_color(
                        tool_id, self._rt._ToolSelectionManager.SELECTED_COLOR)
        for tool_id, expires_at in list(self._wrong_color_until.items()):
            if time.monotonic() >= expires_at:
                self.tool_interaction.set_forced_color(tool_id, None)
                self._wrong_color_until.pop(tool_id, None)
        if (self._reference_colors
                and time.monotonic() >= self._reference_colors_until):
            self._reference_colors.clear()
            self._reference_colors_until = 0.0
            self._refresh_pegboard_colors()
        self.tool_interaction.refresh_colors()
        self.hands.poll(timeout_ms=0)
        gearbox_states = self.gearbox_pose.poll()
        if gearbox_states is not None:
            self.vis.update_gearbox_mirror(gearbox_states)

        T_cam_anchor, T_cam_pegboard, T_cam_board, det_vis = self.aruco_worker.get()
        if self._camera_window_ok and det_vis is not None:
            try:
                self._rt.cv.imshow(self._camera_window, det_vis)
                if (self._rt.cv.waitKey(1) & 0xFF) in (10, 13):
                    self._manual_lock_requested = True
            except Exception:
                self._camera_window_ok = False
        center_T = self.hands.center_eye_T()
        now = time.monotonic()
        dist = (float(np.linalg.norm(T_cam_anchor[:3, 3]))
                if T_cam_anchor is not None else float("inf"))
        cos_tilt = (float(T_cam_anchor[2, 3]) / dist
                    if dist > 1e-6 and np.isfinite(dist) else 0.0)
        suitable = (T_cam_anchor is not None and self.cam.camera_T is not None
                    and dist < self.AUTO_LOCK_MAX_DIST
                    and cos_tilt > np.cos(np.deg2rad(self.AUTO_LOCK_MAX_TILT_DEG)))

        # Same first-sight auto-lock as MainScene. Enter also works from farther
        # away and doubles as the explicit marker-101 pegboard capture action.
        if ((not self.anchor.locked and suitable)
                or (self._manual_lock_requested and T_cam_anchor is not None
                    and self.cam.camera_T is not None)):
            if self.anchor.lock(T_cam_anchor, self.cam.camera_T):
                print(f"[Study4Scene] World locked to marker 100 ({dist:.2f} m)")
        if (self._manual_lock_requested and self.anchor.locked
                and T_cam_pegboard is not None and self.cam.camera_T is not None
                and self.anchor.update_pegboard_from_tracking(
                    self.cam.camera_T, T_cam_pegboard)):
            self._update_pegboard(self.anchor.T_pegboard_in_world)
            print("[Study4Scene] Pegboard captured from marker 101")
        self._manual_lock_requested = False

        # Marker-100 proximity feedback/click-to-relock.
        available = self.anchor.locked and suitable
        if available != self._anchor_available and now >= self._anchor_flash_until:
            self.tool_interaction.send_color(
                self._rt.cfg.ANCHOR_MARKER_ID,
                self._rt._ToolSelectionManager.HOVER_COLOR if available
                else self._rt._ToolSelectionManager.RESET_COLOR)
            self._anchor_available = available
        if (available
                and self.tool_interaction.active_tool_id == self._rt.cfg.ANCHOR_MARKER_ID
                and now - self._last_relock >= self.RELOCK_COOLDOWN):
            if self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True):
                self.tool_interaction.send_color(
                    self._rt.cfg.ANCHOR_MARKER_ID,
                    self._rt._ToolSelectionManager.SELECTED_COLOR)
                self._anchor_flash_until = now + 1.0
                self._last_relock = now
                print("[Study4Scene] Relocked from marker 100 click")
            self.tool_interaction.deselect(self._rt.cfg.ANCHOR_MARKER_ID)

        # Prescanned secondary-marker relock cubes.
        for mid in self.anchor._T_world_marker:
            T_cam_mid = T_cam_board.get(mid)
            secondary_ok = False
            if self.anchor.locked and T_cam_mid is not None and self.cam.camera_T is not None:
                marker_dist = float(np.linalg.norm(T_cam_mid[:3, 3]))
                obliquity = abs(float(T_cam_mid[2, 2]))
                secondary_ok = (
                    marker_dist <= self._rt.cfg.WORLD_MARKERS_PROXIMITY_MAX
                    and obliquity >= np.cos(np.deg2rad(
                        self._rt.cfg.WORLD_MARKERS_TILT_MAX_DEG)))
            if secondary_ok != self._world_available[mid]:
                self.tool_interaction.send_color(
                    mid, self._rt._ToolSelectionManager.HOVER_COLOR
                    if secondary_ok else self._rt._ToolSelectionManager.RESET_COLOR)
                self._world_available[mid] = secondary_ok
            if (secondary_ok and self.tool_interaction.active_tool_id == mid
                    and now - self._world_last_relock[mid]
                    >= self._rt.cfg.WORLD_MARKERS_RELOCK_COOLDOWN):
                if self.anchor.relock_from_world_marker(
                        mid, T_cam_mid, self.cam.camera_T):
                    self.tool_interaction.send_color(
                        mid, self._rt._ToolSelectionManager.SELECTED_COLOR)
                    self._world_last_relock[mid] = now
                    print(f"[Study4Scene] Relocked from marker {mid} click")
                self.tool_interaction.deselect(mid)

        # Track the physical assembly board from markers 102/103.
        if self.anchor.locked:
            for mid, offset in self._board_offsets.items():
                if mid in T_cam_board:
                    self.anchor.update_board_from_tracking(
                        self.cam.camera_T, T_cam_board[mid], offset)
                    break
            self.anchor.publish()
            self.anchor.publish_pegboard()
            self.anchor.publish_board()
            if now - self._last_publish >= 1.0 / 30.0:
                self.relock_cubes.publish()
                self.tool_layout.publish(self.anchor.T_pegboard_in_world)
                self._last_publish = now

        T_wt = self.anchor.T_world_tracking
        left, right = self.hands.world_joints(T_wt)
        T_world_head = (self.anchor.world_T(center_T)
                        if center_T is not None else None)
        if self._head_motion_recording:
            if T_world_head is None:
                self._head_motion_last = None
            else:
                current_head = np.asarray(T_world_head, dtype=float)
                if self._head_motion_last is not None:
                    self._head_translation_m += float(np.linalg.norm(
                        current_head[:3, 3] - self._head_motion_last[:3, 3]))
                    relative_rotation = (
                        self._head_motion_last[:3, :3].T
                        @ current_head[:3, :3])
                    cosine = float(np.clip(
                        (np.trace(relative_rotation) - 1.0) / 2.0,
                        -1.0, 1.0))
                    self._head_rotation_rad += float(np.arccos(cosine))
                self._head_motion_last = current_head.copy()
        effective_cam = self.anchor._effective_cam_T(self.cam.camera_T, center_T)
        T_world_cam = (self.anchor.world_T(effective_cam)
                       if effective_cam is not None else None)
        self.vis.update_hands(left, right)
        self.vis.update_head(T_world_head)
        self.vis.update_passthrough_cam(T_world_cam)
        self.vis.update_tracking(T_wt)
        self.vis.update_board(self.anchor.T_board_in_world)
        self.vis.tick()

    def close(self) -> None:
        self.aruco_worker.stop()
        self.cam.close()
        self.hands.close()
        self.anchor.close()
        self.tool_interaction.close()
        self.tool_layout.close()
        self.relock_cubes.close()
        self.gearbox_pose.close()
        if self._camera_window_ok:
            try:
                self._rt.cv.destroyWindow(self._camera_window)
            except Exception:
                pass
        self._color_pub.close(0)
        self.vis.close()


def _study4_open3d_worker(
        unity_ip: str, layout_path: str, commands, events, responses,
        motion_summaries) -> None:
    """Own Open3D in a separate process so it cannot corrupt Dear PyGui's GL context."""
    request_id = 0

    def score_click(interaction: dict) -> bool:
        nonlocal request_id
        request_id += 1
        current = request_id
        events.put(("click", current, interaction))
        while True:
            response_id, correct = responses.get()
            if response_id == current:
                return bool(correct)

    def forward_interaction(interaction: dict) -> None:
        events.put(("interaction", None, interaction))

    scene = None
    last_motion_publish = 0.0
    try:
        scene = Study4Open3DScene(
            unity_ip, layout_path, on_part_click=score_click,
            on_interaction=forward_interaction)
        events.put(("ready", None, None))
        running = True
        while running:
            while True:
                try:
                    operation, payload = commands.get_nowait()
                except queue.Empty:
                    break
                if operation == "close":
                    running = False
                    break
                if operation == "apply":
                    scene.apply(payload)
                elif operation == "manual_lock":
                    scene.request_manual_lock()
                elif operation == "head_motion_start":
                    scene.start_head_motion_summary()
                elif operation == "head_motion_stop":
                    motion_summaries.put(scene.stop_head_motion_summary())
                elif operation == "remove":
                    scene.remove_acquired_objects(payload)
                elif operation == "mark_acquired":
                    scene.mark_acquired_objects(payload)
                elif operation == "reset_colors":
                    scene.reset_acquisition_colors()
                elif operation == "restore":
                    scene.restore_acquired_objects()
            if running:
                scene.tick()
                now = time.monotonic()
                if (scene._head_motion_recording
                        and now - last_motion_publish >= 0.05):
                    motion_summaries.put(scene.head_motion_summary())
                    last_motion_publish = now
                time.sleep(0.001)
    except Exception as error:
        events.put(("error", None, repr(error)))
    finally:
        if scene is not None:
            scene.close()


class Study4Open3DProcess:
    """Queue-backed facade exposing the Study4Open3DScene methods used by the GUI."""

    def __init__(self, unity_ip: str, layout_path: str | Path, on_part_click,
                 on_interaction=None) -> None:
        context = multiprocessing.get_context("spawn")
        self._commands = context.Queue()
        self._events = context.Queue()
        self._responses = context.Queue()
        self._motion_summaries = context.Queue()
        self._on_part_click = on_part_click
        self._on_interaction = on_interaction
        self._process = context.Process(
            target=_study4_open3d_worker,
            args=(unity_ip, str(layout_path), self._commands, self._events,
                  self._responses, self._motion_summaries),
            name="study4-open3d",
            daemon=True,
        )
        self._process.start()

    def apply(self, event: dict) -> None:
        self._commands.put(("apply", event))

    def request_manual_lock(self) -> None:
        self._commands.put(("manual_lock", None))

    def start_head_motion_summary(self) -> None:
        while True:
            try:
                self._motion_summaries.get_nowait()
            except queue.Empty:
                break
        self._commands.put(("head_motion_start", None))

    def stop_head_motion_summary(self) -> dict[str, float]:
        latest: dict[str, float] = {}
        while True:
            try:
                latest = dict(self._motion_summaries.get_nowait())
            except queue.Empty:
                break
        self._commands.put(("head_motion_stop", None))
        return latest

    def remove_acquired_objects(self, tool_ids: Iterable[int]) -> None:
        self._commands.put(("remove", list(tool_ids)))

    def mark_acquired_objects(self, tool_ids: Iterable[int]) -> None:
        self._commands.put(("mark_acquired", list(tool_ids)))

    def reset_acquisition_colors(self) -> None:
        self._commands.put(("reset_colors", None))

    def restore_acquired_objects(self) -> None:
        self._commands.put(("restore", None))

    def tick(self) -> None:
        while True:
            try:
                name, request_id, payload = self._events.get_nowait()
            except queue.Empty:
                return
            if name == "click":
                correct = self._on_part_click(payload)
                self._responses.put((request_id, correct))
            elif name == "interaction" and self._on_interaction is not None:
                self._on_interaction(payload)
            elif name == "ready":
                print("[Study4Open3D] Child scene ready", flush=True)
            elif name == "error":
                print(f"[Study4Open3D] Child scene disabled: {payload}", flush=True)

    def close(self) -> None:
        if not self._process.is_alive():
            return
        self._commands.put(("close", None))
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


PROVIDED_PARTS = {
    "BASE_BOARD",

    "BEARING_ROW1_LEFT", "BEARING_ROW1_RIGHT", 
    "CRANK_HANDLE_ROW1",
    "GEAR_ROD_ROW1", 
    "GEAR_ROW1_LEFT", 
    "PIN_ROW1_LEFT", "PIN_ROW1_RIGHT",
    "SCREW_ROW1_LEFT", "SCREW_ROW1_RIGHT", 
    "STAND_ROW1_LEFT", "STAND_ROW1_RIGHT",

    "BEARING_ROW2_LEFT", "BEARING_ROW2_RIGHT", 
    "GEAR_ROD_ROW2",
    "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", 
    "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT",
    "SCREW_ROW2_LEFT", "SCREW_ROW2_RIGHT", 
    "STAND_ROW2_LEFT", "STAND_ROW2_RIGHT",

    "BEARING_ROW3_LEFT", "BEARING_ROW3_RIGHT", 
    "GEAR_ROD_ROW3",
    "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", 
    "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT",
    "SCREW_ROW3_LEFT", "SCREW_ROW3_RIGHT", 
    "STAND_ROW3_LEFT", "STAND_ROW3_RIGHT",

    "BEARING_ROW4_LEFT", "BEARING_ROW4_RIGHT", 
    "GEAR_ROD_ROW4",
    "GEAR_ROW4_LEFT", 
    "PIN_ROW4_LEFT", 
    "SCREW_ROW4_LEFT", "SCREW_ROW4_RIGHT",
    "STAND_ROW4_LEFT", "STAND_ROW4_RIGHT",
}


def build_steps() -> list[Step]:
    steps: list[Step] = []
    gear_inputs = {
        1: ("GEAR_ROD_ROW1", "GEAR_ROW1_LEFT", "PIN_ROW1_LEFT"),
        2: ("GEAR_ROD_ROW2", "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT"),
        3: ("GEAR_ROD_ROW3", "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT"),
        4: ("GEAR_ROD_ROW4", "GEAR_ROW4_LEFT", "PIN_ROW4_LEFT"),
    }
    gear_text = {
        1: "Attach the single large gear and secure it with the left wooden pin.",
        2: "Attach the small and medium gears and secure both with their pins.",
        3: "Attach the small and medium gears and secure both with their pins.",
        4: "Attach the single large gear and secure it with the pin.",
    }

    for row in range(1, 5):
        for side in ("Right", "Left"):
            steps.append(Step(
                id=f"r{row}_bearing_{side.lower()}",
                title=f"Row {row}.{1 if side == 'Right' else 3}: bearing into {side.lower()} stand",
                row=row,
                stage=0,
                inputs=(f"BEARING_ROW{row}_{side.upper()}", f"STAND_ROW{row}_{side.upper()}"),
                output=f"BEARING_STAND_ROW{row}_{side.upper()}_ASSEMBLY",
                description=(f"Insert BEARING_ROW{row}_{side.upper()} into "
                             f"STAND_ROW{row}_{side.upper()}."),
                requires=((f"r{row}_bearing_right",) if side == "Left" else ()),
            ))

        steps.append(Step(
            id=f"r{row}_gear_rod",
            title=f"Row {row}.2: assemble gear rod",
            row=row,
            stage=1,
            inputs=gear_inputs[row],
            output=f"GEAR_ROD_ROW{row}_ASSEMBLY",
            description=gear_text[row],
        ))

        first, second = "Right", "Left"
        steps.append(Step(
            id=f"r{row}_fasten_first_stand",
            title=f"Row {row}.4: fasten {first.lower()} stand",
            row=row,
            stage=2,
            inputs=(f"BEARING_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
                    f"SCREW_ROW{row}_{first.upper()}"),
            output=f"FASTENED_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Fasten the {first.lower()} bearing-stand assembly to BASE_BOARD "
                         f"with SCREW_ROW{row}_{first.upper()} before inserting the gear rod."),
        ))
        steps.append(Step(
            id=f"r{row}_insert_rod_and_fit_second",
            title=f"Row {row}.5: insert rod and fit {second.lower()} stand",
            row=row,
            stage=3,
            inputs=(f"FASTENED_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
                    f"GEAR_ROD_ROW{row}_ASSEMBLY",
                    f"BEARING_STAND_ROW{row}_{second.upper()}_ASSEMBLY"),
            output=f"UNFASTENED_SECOND_STAND_ROW{row}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Insert the assembled gear rod through the fastened {first.lower()} "
                         f"stand while fitting the unfastened {second.lower()} stand over the "
                         "other end of the rod."),
        ))
        steps.append(Step(
            id=f"r{row}_fasten_second_stand",
            title=f"Row {row}.6: fasten {second.lower()} stand",
            row=row,
            stage=4,
            inputs=(f"UNFASTENED_SECOND_STAND_ROW{row}_ASSEMBLY",
                    f"SCREW_ROW{row}_{second.upper()}"),
            output=f"MOUNTED_ROW{row}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Fasten the {second.lower()} stand to BASE_BOARD with "
                         f"SCREW_ROW{row}_{second.upper()} after the rod and stand are in place."),
        ))

    steps.append(Step(
        id="r1_attach_handle",
        title="Row 1.7: attach crank handle",
        row=1,
        stage=5,
        inputs=("MOUNTED_ROW1_ASSEMBLY", "CRANK_HANDLE_ROW1", "PIN_ROW1_RIGHT"),
        output="CRANK_MOUNTED_ROW1_ASSEMBLY",
        description=("Attach CRANK_HANDLE_ROW1 only after GEAR_ROD_ROW1 is between both stands, "
                     "then secure the handle with PIN_ROW1_RIGHT."),
    ))
    steps.append(Step(
        id="finish_gearbox",
        title="Verify and finish gearbox",
        row=0,
        stage=6,
        inputs=("CRANK_MOUNTED_ROW1_ASSEMBLY", "MOUNTED_ROW2_ASSEMBLY",
                "MOUNTED_ROW3_ASSEMBLY", "MOUNTED_ROW4_ASSEMBLY"),
        output="COMPLETED_GEARBOX_ASSEMBLY",
        context=("BASE_BOARD",),
        description="Verify alignment, gear meshing, and free rotation of all four rows.",
    ))
    return steps


class TaskGraph:
    def __init__(self, available_parts: Iterable[str] = PROVIDED_PARTS):
        self.steps = build_steps()
        self.by_id = {step.id: step for step in self.steps}

        # print (len(self.by_id)) #26

        self.initial_parts = set(available_parts)
        self.active_parts = set(self.initial_parts)
        self.completed: list[str] = []
        # Row most recently changed by a successful complete/undo operation.
        # Recommendation stays with this row while it still has READY work.
        self.last_worked_row: int | None = None

        # In-memory mapping of each consumed input part to the assembly produced
        # from it: {input_part: output_assembly}. For example, completing a step
        # that transforms BEARING + ROD into BEARING_ROD_ASSEMBLY records:
        # {
        #     "BEARING": "BEARING_ROD_ASSEMBLY",
        #     "ROD": "BEARING_ROD_ASSEMBLY",
        # }
        # If that assembly is consumed later, another entry creates a chain such
        # as BEARING -> BEARING_ROD_ASSEMBLY -> GEAR_ROW_ASSEMBLY. trace_part()
        # follows this chain to find what an original part has currently become.
        # complete() adds entries, undo() removes the entries it reverses, and
        # reset() clears the dictionary. This history is not persisted to a file
        # or database, so it lasts only for the lifetime of this TaskGraph object.
        self.transform_history: dict[str, str] = {}
 
    def reset(self) -> None:
        self.active_parts = set(self.initial_parts)
        self.completed.clear()
        self.transform_history.clear()
        self.last_worked_row = None

    def missing(self, step: Step) -> list[str]:
        # Inputs are consumed and context parts must remain active. `requires`
        # records workflow precedence independently of whether the producer's
        # output has subsequently been consumed by another assembly step.
        missing_parts = [part for part in (*step.inputs, *step.context)
                         if part not in self.active_parts]
        missing_steps = [required for required in step.requires
                         if required not in self.completed]
        return missing_parts + missing_steps

    def is_ready(self, step: Step) -> bool:
        # A step is ready when: It has not already been completed. Its missing-parts list is empty.
        return step.id not in self.completed and not self.missing(step)

    def state(self, step: Step) -> str:
        if step.id in self.completed:
            return "complete"
        return "ready" if self.is_ready(step) else "blocked"

    def complete(self, step: Step) -> tuple[bool, str]:
        if step.id in self.completed:
            return False, f"{step.id} is already complete; its result is {step.output}."
        missing = self.missing(step)
        if missing:
            return False, (f"WARNING: {step.id} cannot happen yet. Missing prerequisite(s): "
                           f"{', '.join(missing)}.")
        for part in step.inputs:
            self.active_parts.remove(part)
            self.transform_history[part] = step.output
        self.active_parts.add(step.output)
        self.completed.append(step.id)
        if step.row > 0:
            self.last_worked_row = step.row
        return True, (f"COMPLETED {step.id}: {', '.join(step.inputs)} transformed into "
                      f"{step.output}.")

    def is_frontier(self, step: Step) -> bool:
        """A completed step is frontier-safe when no completed step consumed its output.
        A “frontier” step is a completed step whose output still exists in the inventory.
        Suppose:
        Step A: X + Y → XY
        Step B: XY + Z → XYZ
        After Step B, XY no longer exists because it was consumed. 
        Therefore, Step A cannot be undone until Step B is undone."""
        return (step.id in self.completed
                and step.output in self.active_parts
                and not self.completed_consumers(step))

    def completed_consumers(self, step: Step) -> list[Step]:
        # Context dependencies also block undo even though they do not consume
        # the producer's output.
        return [candidate for candidate in self.steps
                if candidate.id in self.completed
                and (step.output in (*candidate.inputs, *candidate.context)
                     or step.id in candidate.requires)]

    def frontier_steps(self) -> list[Step]:
        return [self.by_id[step_id] for step_id in self.completed
                if self.is_frontier(self.by_id[step_id])]

    def undo(self, step: Step | None = None) -> tuple[bool, str]:
        """Reverse a completed assembly step when doing so is dependency-safe.

        If ``step`` is omitted, the most recently completed step is selected.
        The target must be on the completed frontier: it must be complete and
        its output must still exist in ``active_parts``. If a later completed
        step has consumed that output, that dependent step must be undone first.

        A successful undo reverses the inventory changes made by ``complete``:
        it removes the assembled output, restores the consumed input parts,
        deletes this step's transformation-history entries, and removes the
        step ID from ``completed``. Context parts are not restored because
        completing a step never consumes them.

        Args:
            step: The completed step to undo. When ``None``, use the most
                recently completed step.

        Returns:
            A ``(success, message)`` tuple. ``success`` is ``True`` only when
            the step was undone; ``message`` describes the result or explains
            why the request was rejected.
        """
        # Nothing can be undone until at least one step has been completed.
        if not self.completed:
            return False, "WARNING: there is no completed step to undo."

        # Use the requested step, or default to the latest completed step.
        target = step or self.by_id[self.completed[-1]]

        # A step that has not happened cannot be reversed.
        if target.id not in self.completed:
            return False, f"WARNING: {target.id} is not complete, so it cannot be undone."

        # Do not invalidate dependent state. If another completed step consumed
        # this output, the consumer must be undone before this producer.
        if not self.is_frontier(target):
            consumers = self.completed_consumers(target)
            blocked_by = ", ".join(consumer.id for consumer in consumers) or "a dependent assembly"
            return False, (f"WARNING: {target.id} is not on the completed frontier. "
                           f"Undo its completed dependent step(s) first: {blocked_by}.")

        # Reverse the inventory transformation: remove the assembled result and
        # return all inputs that were consumed while completing the step.
        self.active_parts.remove(target.output)
        self.active_parts.update(target.inputs)

        # Delete only history entries created by this particular transformation.
        for part in target.inputs:
            if self.transform_history.get(part) == target.output:
                del self.transform_history[part]

        # Mark the target step as no longer completed.
        self.completed.remove(target.id)
        if target.row > 0:
            # Reverting a specific row makes that row the current work focus.
            self.last_worked_row = target.row
        return True, (f"UNDONE {target.id}: removed {target.output} and restored "
                      f"{', '.join(target.inputs)}.")

    def active_parts_text(self) -> str:
        parts = sorted(self.active_parts)
        return f"ACTIVE PARTS ({len(parts)}):\n" + "\n".join(f"  {part}" for part in parts)

    def producer_for(self, part: str) -> Step | None:
        return next((step for step in self.steps if step.output == part), None)

    def add_part(self, part: str) -> str:
        if part in self.active_parts:
            return f"{part} is already active."
        self.active_parts.add(part)
        self.initial_parts.add(part)
        return f"Added {part} to the inventory."

    def find_steps(self, query: str) -> list[Step]:
        needle = query.strip().lower()
        if not needle:
            return []
        exact = self.by_id.get(needle)
        if exact:
            return [exact]
        matches = []
        for step in self.steps:
            searchable = " ".join((step.id, step.title, step.output, *step.inputs)).lower()
            if needle in searchable:
                matches.append(step)
        return matches

    def trace_part(self, part: str) -> str:
        """Describe a part's current state and the next step that can use it.

        The lookup is case-insensitive. The requested name is first searched
        among active inventory items and then among consumed items recorded in
        ``transform_history``. If the part was consumed, its transformation
        chain is followed until the currently active assembly is found.

        The method then examines incomplete steps that directly use the active
        part or assembly as either an input or a context requirement. It reports
        ready steps when available; otherwise it explains which prerequisites
        are blocking the relevant steps.

        When there is no exact active or consumed match, the method searches all
        known input, output, and context names for partial matches. At most 12
        suggestions are returned.

        Args:
            part: Part or assembly name to trace.

        Returns:
            A human-readable status message containing the current assembly,
            possible name matches, a ready next step, or blocking details.
        """
        # Look for an exact name in both the current inventory and the history
        # of parts that have already been consumed. Comparisons ignore case,
        # while the stored spelling is preserved for display and later lookup.
        exact = next((p for p in self.active_parts if p.lower() == part.lower()), None)

        # Example: after completing
        #   BEARING_ROW1_LEFT + STAND_ROW1_LEFT
        #       -> BEARING_STAND_ROW1_LEFT_ASSEMBLY
        # transform_history contains:
        # {
        #     "BEARING_ROW1_LEFT": "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
        #     "STAND_ROW1_LEFT": "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
        # }
        # This expression searches those dictionary keys case-insensitively and
        # returns the stored key spelling, or None when the part was not consumed.
        consumed = next((p for p in self.transform_history if p.lower() == part.lower()), None) 
        #Then the code follows the chain starting from transform_history["BEARING_ROW1_LEFT"] to find what it eventually became
        canonical = exact or consumed or part

        # A consumed part may have passed through several transformations.
        # Follow that chain to identify its latest assembly, for example:
        # BEARING -> BEARING_ROD_ASSEMBLY -> GEAR_ROW_ASSEMBLY.
        if consumed:
            current = self.transform_history[consumed]
            while current in self.transform_history:
                current = self.transform_history[current]
            prefix = f"{consumed} has already been transformed; its current assembly is {current}. "
            canonical = current

        # An exact active match can be used directly when searching for the next
        # applicable assembly step.
        elif exact:
            prefix = f"Found active part {exact}. "

        # If no exact match exists, offer partial matches from every part and
        # assembly name appearing anywhere in the graph.
        else:
            known = {p for s in self.steps for p in (*s.inputs, s.output, *s.context)}
            candidates = sorted(p for p in known if part.lower() in p.lower())
            if candidates:
                return "Possible matches: " + ", ".join(candidates[:12])
            return f"WARNING: no part or assembly named '{part}' exists in this task graph."

        # Find incomplete steps that directly require the current part or
        # assembly. Context requirements count even though they are not consumed.
        candidates = [s for s in self.steps
                      if s.id not in self.completed and canonical in (*s.inputs, *s.context)]
        if not candidates:
            return prefix + "No remaining step directly uses it."

        # Prefer reporting usable steps. Multiple steps can be ready at once
        # because the task graph may contain independent assembly branches.
        ready = [s for s in candidates if self.is_ready(s)]
        if ready:
            return prefix + "Next READY step: " + "; ".join(f"{s.id} ({s.title})" for s in ready)

        # Relevant steps exist, but none are ready. Explain the missing parts for
        # each step so the caller knows which prerequisites must happen first.
        details = []
        for step in candidates:
            # Example: if step.id is "r1_gear_rod" and missing(step) returns
            # ["GEAR_ROW1", "ROD_ROW1"], this appends:
            # "r1_gear_rod is blocked by GEAR_ROW1, ROD_ROW1"
            # join() converts the list of missing-part names into one
            # comma-separated string for the user-facing warning.
            details.append(f"{step.id} is blocked by {', '.join(self.missing(step))}")
        return prefix + "WARNING: its next step should not happen yet: " + "; ".join(details)

    @staticmethod
    def parts_for_reference_label(label: str) -> list[str]:
        """Expand one VLM dataset label to physical task-graph identifiers."""
        label = str(label).strip().upper()
        if label == "BEARING":
            return sorted(p for p in PROVIDED_PARTS if p.startswith("BEARING_"))
        if label == "PIN":
            return sorted(p for p in PROVIDED_PARTS if p.startswith("PIN_"))
        row_group = re.fullmatch(r"(BEARING|PIN)_ROW([1-4])", label)
        if row_group:
            kind, row = row_group.groups()
            return [f"{kind}_ROW{row}_LEFT", f"{kind}_ROW{row}_RIGHT"]
        kit = re.fullmatch(r"ROW([1-4])_KIT", label)
        if kit:
            row = kit.group(1)
            prefixes = ("BEARING_", "SCREW_", "PIN_")
            contents = [
                part for part in PROVIDED_PARTS
                if f"_ROW{row}_" in part and part.startswith(prefixes)
            ]
            if row == "1":
                contents.append("CRANK_HANDLE_ROW1")
            return sorted(contents)
        screw = re.fullmatch(r"SCREW_ROW([1-4])", label)
        if screw:
            row = screw.group(1)
            return [f"SCREW_ROW{row}_LEFT", f"SCREW_ROW{row}_RIGHT"]
        return [label] if label in PROVIDED_PARTS else []

    @staticmethod
    def parts_for_layout_object(tool_name: str) -> list[str]:
        """Map one physical pegboard object to the graph parts/tools it supplies.

        This path is required for gesture-originated fetches, which have a
        layout ID/type but no VLM ``matched_parts`` payload.
        """
        name = str(tool_name).strip().upper()
        if re.fullmatch(r"ROW[1-4]_KIT", name):
            return TaskGraph.parts_for_reference_label(name)
        stand = re.fullmatch(r"GEAR_STAND_ROW([1-4])_(LEFT|RIGHT)", name)
        if stand:
            row, side = stand.groups()
            return [f"STAND_ROW{row}_{side}"]
        tool_contents = {
            "BIT_WRENCH": ["BIT_WRENCH"],
            "BIT_SCREWDRIVER": ["BIT_SCREWDRIVER"],
            # tool_layout1.json retains the historical one-l spelling.
            "PHILIPS_SCREWDRIVER": ["PHILLIPS_SCREWDRIVER"],
            "PHILLIPS_SCREWDRIVER": ["PHILLIPS_SCREWDRIVER"],
            "BITHOLDER1": ["BIT_HOLDER1", "H5_HEX_BIT", "H3_HEX_BIT"],
            "BIT_HOLDER1": ["BIT_HOLDER1", "H5_HEX_BIT", "H3_HEX_BIT"],
            "BITHOLDER2": ["BIT_HOLDER2", "T25_TORX_BIT"],
            "BIT_HOLDER2": ["BIT_HOLDER2", "T25_TORX_BIT"],
        }
        if name in tool_contents:
            return tool_contents[name]
        return [name] if name in PROVIDED_PARTS else []

    def current_container(self, part: str) -> str | None:
        """Return an active raw part or the active assembly containing it."""
        if part in self.active_parts:
            return part
        if part not in self.transform_history:
            return None
        current = self.transform_history[part]
        visited = {part}
        while current in self.transform_history and current not in visited:
            visited.add(current)
            current = self.transform_history[current]
        return current if current in self.active_parts else None

    @staticmethod
    def friendly_part(part: str) -> str:
        """Translate canonical graph identifiers into concise spoken language."""
        name = str(part).upper()
        match = re.fullmatch(
            r"(BEARING|STAND|SCREW|PIN|GEAR)_ROW([1-4])_(LEFT|RIGHT)", name)
        if match:
            kind, row, side = match.groups()
            noun = {
                "BEARING": "bearing",
                "STAND": "stand",
                "SCREW": "screw",
                "PIN": "wooden pin",
                "GEAR": "gear",
            }[kind]
            return f"the Row {row} {side.lower()} {noun}"
        match = re.fullmatch(r"GEAR_ROD_ROW([1-4])", name)
        if match:
            return f"the Row {match.group(1)} gear rod"
        if name == "CRANK_HANDLE_ROW1":
            return "the crank handle"
        if name == "BASE_BOARD":
            return "the gearbox baseboard"
        patterns = [
            (r"BEARING_STAND_ROW([1-4])_(LEFT|RIGHT)_ASSEMBLY",
             lambda m: (f"the Row {m.group(1)} {m.group(2).lower()} stand "
                        "with its bearing")),
            (r"GEAR_ROD_ROW([1-4])_ASSEMBLY",
             lambda m: f"the assembled Row {m.group(1)} gear rod"),
            (r"FASTENED_STAND_ROW([1-4])_RIGHT_ASSEMBLY",
             lambda m: f"the Row {m.group(1)} right stand fastened to the board"),
            (r"UNFASTENED_SECOND_STAND_ROW([1-4])_ASSEMBLY",
             lambda m: f"the Row {m.group(1)} gear rod fitted between both stands"),
            (r"MOUNTED_ROW([1-4])_ASSEMBLY",
             lambda m: f"the mounted Row {m.group(1)} assembly"),
            (r"CRANK_MOUNTED_ROW1_ASSEMBLY",
             lambda _m: "the completed Row 1 assembly with its handle"),
            (r"COMPLETED_GEARBOX_ASSEMBLY",
             lambda _m: "the completed gearbox"),
        ]
        for pattern, describe in patterns:
            match = re.fullmatch(pattern, name)
            if match:
                return describe(match)
        return name.replace("_", " ").lower()

    @staticmethod
    def friendly_step_action(step: Step) -> str:
        """Return only the short, familiar action phrase for a step."""
        if step.id.endswith("bearing_right"):
            return "put the bearing into the right stand"
        if step.id.endswith("bearing_left"):
            return "put the bearing into the left stand"
        if step.id.endswith("gear_rod"):
            return "put the gears on the gear rod and pin them"
        if step.id.endswith("fasten_first_stand"):
            return "screw the right stand to the board"
        if step.id.endswith("insert_rod_and_fit_second"):
            return "insert the gear rod and fit the left stand"
        if step.id.endswith("fasten_second_stand"):
            return "screw the left stand to the board"
        if step.id == "r1_attach_handle":
            return "attach and pin the crank handle"
        return "check that the gears line up and turn freely"

    def friendly_step_label(self, step: Step) -> str:
        """Return the compact user-facing stage label, such as ``Step 2.4``."""
        row_stage = self.control_coords_for(step.id)
        return (f"Step {row_stage[0]}.{row_stage[1]}"
                if row_stage and row_stage[0] else "Final check")

    def friendly_step(self, step: Step) -> str:
        """Return one compact stage-and-action description."""
        return f"{self.friendly_step_label(step)}: {self.friendly_step_action(step)}"

    @classmethod
    def validate(cls) -> list[str]:
        """Return model problems; missing raw inventory is reported separately by the GUI."""
        graph = cls()
        errors: list[str] = []
        ids = [s.id for s in graph.steps]
        outputs = [s.output for s in graph.steps]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate step ID")
        if len(outputs) != len(set(outputs)):
            errors.append("Duplicate assembled-part output")

        # Collect every assembly that can be created by a graph step. Each step
        # input must come from one of two valid sources:
        #   1. a raw part included in the graph's initial inventory, or
        #   2. an assembly produced as the output of another graph step.
        # If an input appears in neither collection, its name is probably
        # misspelled or the step that should produce it has not been defined.
        produced = set(outputs)
        for step in graph.steps:
            for required in step.requires:
                if required not in graph.by_id:
                    errors.append(f"Unknown required step {required} in {step.id}")
            for part in step.inputs:
                if part not in graph.initial_parts and part not in produced:
                    errors.append(f"Unknown input {part} in {step.id}")
        return errors

    @staticmethod
    def steps_for_control(row: int, stage: int) -> list[str]:
        """Bridge a gearbox_control.py (row, control-stage) to this graph's step id(s).

        The two files number "stage" differently: gearbox_control uses per-row control-stages 1-8,
        while this graph uses named steps. The control stages now map one-to-one onto task steps:
        stage 5 is the rod-insert + left-stand fit, stage 6 the left-stand fastening."""
        if stage == 8 or row == 0:
            return ["finish_gearbox"]
        if stage == 7:
            return ["r1_attach_handle"] if row == 1 else []
        return {
            1: [f"r{row}_bearing_right"],
            2: [f"r{row}_gear_rod"],
            3: [f"r{row}_bearing_left"],
            4: [f"r{row}_fasten_first_stand"],
            5: [f"r{row}_insert_rod_and_fit_second"],
            6: [f"r{row}_fasten_second_stand"],
        }.get(stage, [])

    @staticmethod
    def control_coords_for(step_id: str):
        """Inverse of steps_for_control: a task-step id -> the gearbox_control (row, control-stage)
        that opens it, or None if the id isn't a mapped control step. Lets this viewer tell an
        external (row, stage) consumer — e.g. gearbox_control.py --open-3d — which step was
        just selected."""
        if step_id == "finish_gearbox":
            return 0, 8
        for row in range(1, 5):
            for stage in range(1, 8):
                if step_id in TaskGraph.steps_for_control(row, stage):
                    return row, stage
        return None

    def state_summary(self) -> str:
        """Build a concise state description for the VLM system context."""
        completed = self.completed
        ready = [s for s in self.steps if self.is_ready(s)]
        assemblies = sorted(p for p in self.active_parts if p.endswith("_ASSEMBLY"))

        lines = [f"Progress: {len(completed)} / {len(self.steps)} steps completed"]
        lines.append(
            f"Recommendation focus row: {self.last_worked_row}"
            if self.last_worked_row is not None
            else "Recommendation focus row: none yet")

        if completed:
            lines.append("\nCompleted steps (most recent last):")
            # Include every completed step so the VLM has the full progress
            # history instead of only a truncated recent subset.
            for sid in completed:
                step = self.by_id[sid]
                lines.append(f"  - [{step.id}] {step.title}  ->  {step.output}")

        if ready:
            lines.append(
                "\nREADY steps — ALL of these are currently unlocked and INDEPENDENT."
                "\nThey have NO ordering requirement among themselves; the user may perform"
                " them in any order they prefer:"
            )
            # Include every ready step so no currently valid option is hidden
            # from the VLM.
            for step in ready:
                lines.append(f"  - [{step.id}] {step.title}")
        elif len(completed) == len(self.steps):
            lines.append("\nNo READY steps: the assembly is complete.")
        else:
            lines.append(
                "\nNo READY steps: assembly is stalled because required parts "
                "or prerequisite assemblies are missing."
            )

        if assemblies:
            lines.append("\nActive assemblies in inventory:")
            for a in assemblies:
                lines.append(f"  - {a}")

        lines.append(
            "\nIMPORTANT: Any task step not listed as COMPLETED or READY is BLOCKED."
            " Recommend only READY steps. Every listed READY step is valid and may"
            " be performed because all of its prerequisites are satisfied. When"
            " choosing among multiple READY steps, follow the same deterministic"
            " rule as the interface: first prefer the most recently completed or"
            " reverted row while it has READY work, then choose the lowest"
            " user-facing stage in that row. If that row has no READY work (or no"
            " row has been worked yet), use the lowest-numbered READY row and stage."
            " Recommend the final gearbox step only when it is READY and no row"
            " assembly step remains READY. This preference selects among valid"
            " options; it is not an additional dependency or ordering constraint."
            " Do NOT infer dependencies from descriptions or display order."
        )

        return "\n".join(lines)

    def recommend_next_step(self, exclude_step_id: str | None = None) -> "Step | None":
        """Prefer READY work in the last changed row, then row/stage order.

        ``Step.stage`` represents dependency depth and is not the stage number
        displayed by the interface or used by the controller. Use
        ``control_coords_for`` here so the rule-based recommendation follows the
        documented Stage 1–8 sequence exactly. The global finish step is treated
        as the highest row and therefore comes after all ready row work.
        """
        ready = [s for s in self.steps
                 if self.is_ready(s) and s.id != exclude_step_id]
        if not ready:
            return None

        if self.last_worked_row is not None:
            same_row = [s for s in ready if s.row == self.last_worked_row]
            if same_row:
                ready = same_row

        def _priority(s: Step) -> tuple:
            row = s.row if s.row > 0 else 999
            control_coords = self.control_coords_for(s.id)
            display_stage = control_coords[1] if control_coords is not None else 999
            return (row, display_stage, s.id)

        return min(ready, key=_priority)


class DearPyGuiTaskGraphApp:
    """Dear PyGui task-graph view and terminal controller."""

    COLORS = {
        "complete": (46, 157, 96, 255),   # green
        "ready":    (39, 132, 216, 255),  # blue
        "blocked":  (209, 138, 39, 255),  # orange
    }

    # Status → (RGBA color, display label)
    _VOICE_STATUS_STYLE: dict[str, tuple[tuple, str]] = {
        "loading":     ((180, 180, 180, 255), "Loading model..."),   # grey
        "idle":        ((255, 220, 50,  255), "Idle"),               # yellow
        "speech":      ((50,  200, 255, 255), "Speech detected..."), # cyan
        "queued":      ((200, 160, 50,  255), "Queued..."),          # amber
        "transcribing":((255, 165, 50,  255), "Transcribing..."),    # orange
        "listening":   ((50,  220, 80,  255), "Listening"),          # green
        "error":       ((255, 80,  80,  255), "Error"),              # red
    }

    # ── Initialization and application lifecycle ─────────────────────────────

    def __init__(self, study4_condition: str | None = None) -> None:
        """Initialize application state before any GUI widgets are created."""
        # Keep the Dear PyGui module available to every instance method.
        self.dpg = dpg
        # Create the assembly dependency graph and initial inventory.
        self.graph = TaskGraph()
        if study4_condition not in (None, "gesture", "language", "task_aware"):
            raise ValueError("invalid Study 4 condition")
        self.study4_condition = study4_condition
        self._tool_layout_path = _study4_tool_layout(study4_condition)
        # Store the currently selected task-step ID, or None when unselected.
        self.selected_id: str | None = None
        # Map task-step IDs to their Dear PyGui node tags.
        self.node_tags: dict[str, str] = {}
        # Map task-step IDs to their node input-pin tags.
        self.input_attributes: dict[str, str] = {}
        # Map task-step IDs to their node output-pin tags.
        self.output_attributes: dict[str, str] = {}
        # Map state names such as "ready" to Dear PyGui theme tags.
        self.themes: dict[str, str] = {}

        # Live-mirror state
        # Track steps currently opened by the external controller.
        self.active_ids: set[str] = set()
        # Track the step highlighted by the recommendation policy.
        self.recommended_id: str | None = None
        # Transfer ZMQ messages safely from the listener thread to the GUI thread.
        self._live_queue: "queue.Queue[dict]" = queue.Queue()
        # Tell the listener thread whether it should continue polling.
        self._live_running = False
        # Retain the background ZMQ listener thread after startup.
        self._live_thread: threading.Thread | None = None
        # Retain the ZeroMQ subscriber that receives controller events.
        self._live_sub = None
        # Retain the ZeroMQ publisher that sends selected-step events.
        self._select_pub = None   # PUB -> gearbox_control.py --open-3d viewer (selected step)

        # Optional in-process controller (set by main() when --with-controller is used).
        # Gives direct access to send commands to Unity without going through ZMQ.
        self.controller: "gearbox_control.GearboxController | None" = None

        # run() sets this to a SpeechListener when voice input is enabled.
        self._speech = None   # SpeechListener, set in run() if enabled
        # run() sets this to a VLMAssistant when a model is enabled.
        self._vlm    = None   # VLMAssistant, set in run() if enabled
        # Asynchronous speech output; initialized by run() unless --no-tts.
        self._tts: TTSService | None = None
        self._assistant_logger: AssistantInteractionLogger | None = None
        self._click_logger: PartAcquisitionClickLogger | None = None
        self._step_logger: PartAcquisitionStepLogger | None = None
        self._session_logger: Study4SessionLogger | None = None
        self._step_attempt = 0
        self._step_event_index = 0
        self._trial_recording = False
        self._trial_start_pending = False
        self._trial_start_pending_at: float | None = None
        self._trial_finished_elapsed_s: float | None = None
        self._trial_head_translation_m: float | None = None
        self._trial_head_rotation_deg: float | None = None
        self._last_terminal_status_at = 0.0
        self._step_selected_at: float | None = None
        self._last_acquisition_at: float | None = None
        self._trial_required_tool_ids: set[int] = set()
        self._acquired_tool_ids: set[int] = set()
        self._acquisition_complete = False
        self._removed_tool_ids: set[int] = set()
        self._trial_total_clicks = 0
        self._trial_wrong_clicks = 0
        self._trial_repeated_correct_clicks = 0
        # Condition-3 visual discourse: VLM candidate set is cyan; one current
        # referent is yellow and can rotate on alternative_to_previous.
        self._vlm_candidate_ids: list[int] = []
        self._vlm_referent_id: int | None = None
        self._last_hover_spoken_id: int | None = None
        self._last_hover_spoken_at = 0.0
        self._study4_scene: Study4Open3DProcess | None = None
        self._tool_index = gearbox_control.load_tool_index(
            self._tool_layout_path)
        try:
            layout = json.loads(self._tool_layout_path.read_text())
            self._tool_defs_by_id = {
                int(item["id"]): item for item in layout.get("tools", [])
            }
            self._fetchable_tool_ids = {
                int(item["id"])
                for item in layout.get("tools", [])
                if item.get("grasp_joints")
            }
        except (OSError, ValueError, TypeError, KeyError):
            self._tool_defs_by_id = {}
            self._fetchable_tool_ids = set()
        # At most one object can wait for spoken fetch confirmation.
        self._pending_fetch: dict[str, object] | None = None
        # Robot supply state is separate from assembly state: grasping or
        # handing over a part does not consume it in the task graph.
        self._robot_part_states: dict[str, dict[str, object]] = {}
        # A restarted task-graph process asks the still-running Open3D/main
        # process to replay its physical grasp/handover state. PUB/SUB may drop
        # the first message while sockets connect, so the request is retried.
        self._robot_state_sync_pending = False
        self._robot_state_sync_next_at = 0.0
        self._gui_built = False

    def build(self) -> None:
        """Construct, configure, and reveal the Dear PyGui interface."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Initialize Dear PyGui's internal item and resource registries.
        dpg.create_context()
        # Create the operating-system window that contains the application.
        # Keep the initial window within a normal desktop.  The previous
        # 3420x1400 viewport could be placed partly off-screen by X11/GLFW and
        # left the stretch-layout table with a collapsed content region.
        dpg.create_viewport(title="Gearbox Assembly Task Graph", width=1920, height=1080,
                            min_width=1000, min_height=700)
        # Register themes before any GUI item tries to use them.
        self._create_themes()

        # Create the root window whose contents fill the viewport.
        with dpg.window(tag="primary_window", label="Gearbox Assembly Task Graph"):
            # Add the application heading.
            dpg.add_text("Gearbox Assembly Dependency Graph", color=(225, 232, 240),
                         tag="main_title")
            # Add a legend explaining the node-state colors.
            dpg.add_text("Blue = ready     Orange = blocked     Green = complete"
                         "     Purple = selected / active     Gold = recommended",
                         color=(170, 180, 195))
            # Use a resizable table as a three-column page layout.
            with dpg.table(header_row=False, resizable=True, height=-1,
                           policy=dpg.mvTable_SizingStretchProp):
                # Give the dependency graph the largest proportional width.
                dpg.add_table_column(init_width_or_weight=2.8)
                # Give the controls panel a narrower proportional width.
                dpg.add_table_column(init_width_or_weight=1.0)
                # Give the VLM panel the remaining proportional width.
                dpg.add_table_column(init_width_or_weight=1.1)
                # Put the graph, controls, and VLM into one horizontal row.
                with dpg.table_row():
                    # First cell: dependency-graph canvas.
                    with dpg.table_cell():
                        # Fill the cell and allow a wide graph to scroll.
                        with dpg.child_window(height=-1, horizontal_scrollbar=True):
                            # Create the interactive graph canvas and minimap.
                            with dpg.node_editor(tag="task_node_editor", minimap=True,
                                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                                # Add one visual node per assembly step.
                                self._create_nodes()
                                # Connect producer nodes to consumer nodes.
                                self._create_links()
                    # Second cell: task details and application controls.
                    with dpg.table_cell():
                        # Fill the cell vertically with its own panel.
                        with dpg.child_window(height=-1):
                            self._create_side_panel()
                    # Third cell: VLM assistant.
                    with dpg.table_cell():
                        # Fill the cell vertically with its own panel.
                        with dpg.child_window(height=-1):
                            # Build the assistant UI when a VLM is enabled.
                            if self._vlm is not None:
                                self._vlm.build_inline()
                            # Otherwise explain how to enable the assistant.
                            else:
                                dpg.add_text("VLM Assistant",
                                             color=(225, 232, 240))
                                dpg.add_separator()
                                dpg.add_text(
                                    "Not enabled.\n\nRun with:\n"
                                    "  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct",
                                    color=(140, 155, 175))

        # Make the root window automatically fill the viewport.
        dpg.set_primary_window("primary_window", True)
        # Apply the previously created link and pin theme to the node editor.
        dpg.bind_item_theme("task_node_editor", "node_editor_theme")
        # Render current graph state into node colors, labels, and panels.
        self.refresh()
        # Finalize Dear PyGui after declaring the initial widgets.
        dpg.setup_dearpygui()
        # Make the operating-system window visible.
        dpg.show_viewport()
        self._gui_built = True

    def run(self, live_port: int | None = None,
            voice_device: str | None = None,
            asr_device: str = "cuda",
            wake_word: str | None = None,
            vlm_model: str | None = None,
            task_description_path: str | None = None,
            select_port: int | None = None,
            tts_engine: str | None = "piper",
            piper_model: str | Path = DEFAULT_PIPER_MODEL,
            tts_rate: float = 0.85,
            assistant_log_path: str | Path | None = None,
            click_log_path: str | Path | None = None,
            step_log_path: str | Path | None = None,
            session_log_path: str | Path | None = None,
            resume_session: bool = True,
            participant_id: str = "anonymous",
            open3d_scene: bool = True,
            unity_ip: str = gearbox_control._DEFAULT_IP) -> None:
        """Start optional services, run the GUI loop, and clean up on exit."""
        startup_started = time.perf_counter()

        def startup(message: str) -> None:
            elapsed = time.perf_counter() - startup_started
            print(f"[STARTUP +{elapsed:6.2f}s] {message}", flush=True)

        startup(
            f"Beginning Study 4 condition={self.study4_condition or 'none'} "
            f"(voice={'on' if voice_device is not None else 'off'}, "
            f"vlm={'on' if vlm_model is not None else 'off'}, "
            f"Open3D={'on' if open3d_scene else 'off'})")
        # The Study 4 VLM loads synchronously while its Dear PyGui panel is
        # built. This status line precedes that main-thread checkpoint load.
        if vlm_model is not None or voice_device is not None:
            startup("Transformers/model loading will run on the main thread")
        else:
            startup("Skipping transformers (voice and VLM disabled)")

        # Load ASR before the much larger VLM. Recognition readiness is
        # independent of trial routing: transcripts remain gated until the
        # participant selects a step and presses Enter.
        if voice_device is not None:
            startup(
                f"Loading speech recognizer first on {asr_device}: "
                f"{voice_device}")
            try:
                self._speech = SpeechListener(
                    device=voice_device, wake_word=wake_word,
                    compute_device=asr_device)
                # Load without accumulating startup-room audio while Qwen and
                # the visual interfaces are still being initialized.
                self._speech.set_input_enabled(False)
                self._speech.start()
                if self._speech.wait_until_model_ready(timeout=300.0):
                    startup("Speech recognizer ready; continuing with VLM startup")
                else:
                    startup(
                        "Speech recognizer did not become ready; continuing so "
                        "its detailed error can be shown in the GUI")
            except Exception as error:
                self._speech = None
                print(f"[Voice] Failed to start: {error}")

        # Create the VLM assistant only when the caller supplies a model.
        if vlm_model is not None:
            startup(f"Creating VLM assistant: {vlm_model}")
            # Use the canonical gearbox task description for every Study 4
            # condition.  Callers may still provide an explicit override.
            desc_path = task_description_path or str(
                Path(__file__).parent / "task_description.md")
            # Construct the assistant before build() so its panel can be created.
            self._vlm = VLMAssistant(
                self.dpg,
                desc_path,
                model_name=vlm_model,
                study4_condition=self.study4_condition,
            )
            # Report model selection in the operating-system terminal.
            print(f"[VLM] Assistant created: {vlm_model}")
        if tts_engine is not None:
            startup(f"Starting TTS backend: {tts_engine}")
            try:
                self._tts = TTSService(
                    tts_engine, Path(piper_model), speech_rate=tts_rate)
                self._tts.start()
                print(f"[TTS] Started asynchronous {tts_engine} speech output "
                      f"(rate={tts_rate:.2f}x)")
            except Exception as error:
                print(f"[TTS] Disabled: {error}")
        if assistant_log_path is not None:
            self._assistant_logger = AssistantInteractionLogger(assistant_log_path)
            print(f"[StudyLog] Part-reference decisions -> {assistant_log_path}")
        if click_log_path is not None:
            self._click_logger = PartAcquisitionClickLogger(
                click_log_path, participant_id)
            print(f"[StudyLog] Pegboard acquisition clicks -> {click_log_path}")
        if step_log_path is not None:
            self._step_logger = PartAcquisitionStepLogger(
                step_log_path, participant_id)
            print(f"[StudyLog] Per-step acquisition summary -> {step_log_path}")
        if session_log_path is not None:
            self._session_logger = Study4SessionLogger(
                session_log_path, participant_id,
                self.study4_condition or "", fresh_run=not resume_session)
            self._step_attempt = self._session_logger.max_step_attempt()
            print(f"[StudyLog] Unified resumable session -> {session_log_path}")
        # Open3D and Dear PyGui both use GLFW/OpenGL and crash or black out when
        # they own windows in one process on this Linux setup.  Spawn the scene
        # renderer; its queue-backed facade preserves click scoring and updates.
        if open3d_scene:
            startup("Starting camera/marker/Open3D scene process...")
            try:
                self._study4_scene = Study4Open3DProcess(
                    unity_ip, self._tool_layout_path,
                    on_part_click=self._score_part_click,
                    on_interaction=self._handle_pegboard_interaction)
                startup("Open3D scene process launched")
            except Exception as error:
                print(f"[Study4Open3D] Disabled: {error}", flush=True)
        startup("Building Dear PyGui task graph...")
        self.build()
        self.dpg.render_dearpygui_frame()
        startup("Task graph window rendered")
        # Seed the model with the graph before the first question; subsequent
        # complete/undo/reset events replace this live-state block.
        self._notify_vlm("INITIAL STATE")
        # Start receiving controller events when a live port was supplied.
        if live_port is not None:
            startup(f"Starting live event listener on port {live_port}")
            self.start_live_listener(live_port)
        # Start publishing node selections when an output port was supplied.
        if select_port is not None:
            startup(f"Starting step-selection publisher on port {select_port}")
            self.start_select_publisher(select_port)
        if self._speech is not None:
            self._speech.set_input_enabled(True)
            self.log(f"[Voice] Started on device: {voice_device}")
        # Enable drag-and-drop images only when the VLM assistant exists.
        if self._vlm is not None:
            try:
                # Route dropped files to the VLM assistant's image handler.
                self.dpg.set_viewport_drop_callback(self._vlm.on_file_drop)
            except Exception:
                pass  # older DPG versions may not support this

        if resume_session:
            self._resume_session()
        else:
            self._log_session_event("session_start")

        startup("Startup complete; entering GUI/Open3D render loop")

        # Use a local reference in the high-frequency render loop.
        dpg = self.dpg
        # Process background events and render until the window closes.
        while dpg.is_dearpygui_running():
            # Apply the Enter transition before polling speech so a transcript
            # arriving on the same frame sees RUNNING rather than stale READY.
            enter_pressed = dpg.is_key_pressed(dpg.mvKey_Return)
            shift_down = (dpg.is_key_down(dpg.mvKey_LShift)
                          or dpg.is_key_down(dpg.mvKey_RShift))
            enter_used = (self._handle_trial_enter()
                          if enter_pressed and not shift_down else False)
            self._finalize_pending_trial_start()
            # Apply queued controller messages on the main GUI thread.
            self._drain_live_queue()
            self._request_robot_state_sync_if_due()
            # Apply speech status and transcript events on the main GUI thread.
            self._poll_speech()
            # Display VLM status changes and completed responses.
            if self._vlm is not None:
                self._vlm.tick()
                self._poll_vlm_part_references()
                self._poll_vlm_answers()
            self._poll_tts()
            if self._study4_scene is not None:
                # Shift+Enter always retains manual marker relocking; plain
                # Enter does so only when no prepared trial consumes it.
                if enter_pressed and (shift_down or not enter_used):
                    self._study4_scene.request_manual_lock()
                self._study4_scene.tick()
            self._update_terminal_trial_status()
            # Draw one frame and process Dear PyGui interaction.
            dpg.render_dearpygui_frame()
        print("\r\033[2K", end="", flush=True)
        # Tell the live listener thread to stop polling.
        self._live_running = False
        # Release audio resources if speech recognition was active.
        if self._speech is not None:
            self._speech.close()
        # Stop the model worker if the assistant was active.
        if self._vlm is not None:
            self._vlm.close()
        if self._tts is not None:
            self._tts.close()
        if self._study4_scene is not None:
            self._study4_scene.close()
        # Release all Dear PyGui resources.
        dpg.destroy_context()

    # ── ZeroMQ sending, receiving, and live-controller events ────────────────

    def start_live_listener(self, port: int) -> bool:
        """Bind a SUB (receiver binds, per the repo's Python<->Python convention) and drain
        controller events on a background thread. Failures are non-fatal — the GUI still opens."""
        # Import ZeroMQ lazily so the GUI can still run without that dependency.
        try:
            import zmq
        except Exception as e:
            self.log(f"Live link disabled (zmq unavailable: {e}).")
            return False
        # Create and bind the subscriber that receives controller events.
        try:
            # Reuse ZeroMQ's process-wide context.
            ctx = zmq.Context.instance()
            # Create a subscriber socket.
            sub = ctx.socket(zmq.SUB)
            # Subscribe to every topic because messages are plain JSON strings.
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            # Bind the receiver to the configured local port.
            sub.bind(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Live link disabled (could not bind :{port}: {e}).")
            return False
        # Retain the socket for the lifetime of the application.
        self._live_sub = sub
        # Allow the listener loop to begin processing events.
        self._live_running = True

        # Define the work performed by the background listener thread.
        def _loop() -> None:
            # Use a poller so the thread can periodically check its stop flag.
            poller = zmq.Poller()
            # Watch the subscriber for incoming messages.
            poller.register(sub, zmq.POLLIN)
            # Continue until run() begins application shutdown.
            while self._live_running:
                # Wait briefly and retry when no message has arrived.
                if not dict(poller.poll(timeout=200)):
                    continue
                # Drain every message currently waiting on the socket.
                while True:
                    try:
                        # Receive without blocking after the poller reported data.
                        raw = sub.recv_string(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        # Decode JSON and hand it to the thread-safe GUI queue.
                        self._live_queue.put(json.loads(raw))
                    except json.JSONDecodeError:
                        # Ignore malformed external messages.
                        continue

        # Run network polling outside the GUI thread.
        self._live_thread = threading.Thread(target=_loop, daemon=True)
        # Start receiving controller events immediately.
        self._live_thread.start()
        # Report the active endpoint in the application's terminal.
        self.log(f"Live link listening on tcp://{_LOCALHOST}:{port} (controller mirror).")
        return True

    def start_select_publisher(self, port: int) -> bool:
        """Connect a PUB (sender connects, per convention) so pressing 'Select step' can tell an
        (row, stage) consumer — gearbox_control.py --open-3d — which step is selected. Non-fatal."""
        # Import ZeroMQ lazily so this optional feature can fail independently.
        try:
            import zmq
        except Exception as e:
            self.log(f"Step-select link disabled (zmq unavailable: {e}).")
            return False
        # Create and connect the selected-step publisher.
        try:
            # Reuse ZeroMQ's process-wide context.
            ctx = zmq.Context.instance()
            # Create a publisher socket for outgoing JSON events.
            pub = ctx.socket(zmq.PUB)
            # Connect the sender to the external consumer.
            pub.connect(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Step-select link disabled (could not connect :{port}: {e}).")
            return False
        # Retain the publisher for later selection callbacks.
        self._select_pub = pub
        self._robot_state_sync_pending = True
        self._robot_state_sync_next_at = time.monotonic() + 0.25
        # Report the outgoing endpoint in the application's terminal.
        self.log(f"Step-select link -> tcp://{_LOCALHOST}:{port} (open3d viewer).")
        return True

    def _request_robot_state_sync_if_due(self) -> None:
        """Request retained robot supply state until main acknowledges it."""
        if not self._robot_state_sync_pending or self._select_pub is None:
            return
        now = time.monotonic()
        if now < self._robot_state_sync_next_at:
            return
        self._send_select({"event": "robot_part_state_request"})
        self._robot_state_sync_next_at = now + 1.0

    def _send_select(self, msg: dict) -> None:
        """Serialize and publish one selected-step event when connected."""
        if self._study4_scene is not None:
            self._study4_scene.apply(msg)
        # Do nothing when the optional publisher was not started.
        if self._select_pub is None:
            return
        try:
            # Convert the dictionary to JSON and send it as a text message.
            self._select_pub.send_string(json.dumps(msg))
        except Exception:
            # Keep GUI selection usable if the external consumer disconnects.
            pass

    def _drain_live_queue(self) -> None:
        """Apply all controller messages currently waiting for the GUI thread."""
        # Continue until the thread-safe queue becomes empty.
        while True:
            try:
                # Retrieve immediately rather than freezing the GUI while waiting.
                msg = self._live_queue.get_nowait()
            except queue.Empty:
                return
            # Apply the event on the main thread, where GUI updates are safe.
            self._apply_live_event(msg)

    def _apply_live_event(self, msg: dict) -> None:
        """Apply one controller event (main/UI thread): purple overlay + built-in complete/undo."""
        # Read the event type and default missing coordinates to the global step.
        event = msg.get("event")
        # Unity-originated controller events must drive the embedded Open3D
        # BoardAR mirror too. Local GUI actions already take the direct
        # _send_select path; applying these state events twice is idempotent.
        if (self._study4_scene is not None
                and event in {"show", "close", "complete", "uncomplete", "reset"}):
            self._study4_scene.apply(msg)
        if event == "robot_part_state_snapshot":
            restored, rejected = self._restore_completed_task_stages(
                msg.get("completed_stages", []),
                last_worked_row=msg.get("last_worked_row"),
            )
            # The main process owns physical supply history. Replace only its
            # prior records; keep graph-progression inference independent.
            for label, info in list(self._robot_part_states.items()):
                if info.get("source") != "task_progression":
                    self._robot_part_states.pop(label, None)
            for item in msg.get("items", []):
                if isinstance(item, dict):
                    self._apply_robot_part_status(item, notify=False)
            self._robot_state_sync_pending = False
            self.log(
                f"[TaskRecovery] Restored {restored} completed step(s) and "
                f"{len(msg.get('items', []))} physical object state(s) from "
                f"main_with_robot.py"
                + (f"; rejected {rejected}" if rejected else ""))
            self._notify_vlm("TASK AND ROBOT STATE RESTORED AFTER RESTART")
            if self._gui_built:
                self.refresh()
            return
        if event == "robot_part_status":
            self._apply_robot_part_status(msg)
            return
        row, stage = msg.get("row", 0), msg.get("stage", 0)
        # Translate controller coordinates into task IDs that exist in this graph.
        ids = [i for i in TaskGraph.steps_for_control(row, stage) if i in self.graph.by_id]
        # Highlight the step whose controller view was opened.
        if event == "show":
            already_selected = bool(ids and self.selected_id == ids[0])
            self.active_ids = set(ids)
            if ids:
                self.selected_id = ids[0]
                if not already_selected:
                    self._start_acquisition_trial()
                selected = self.graph.by_id[ids[0]]
                # Unity mirrors programmatic recommendations and part-specific
                # selections back as ``show``. Avoid speaking the same selection
                # twice when it was already selected locally.
                if not already_selected:
                    self._announce_step_selection(selected)
                self._focus_vlm_on_step(selected)
                self._publish_selected_step_board_highlight()
                self._auto_complete_no_retrieval()
        elif event == "acquisition_blocked":
            if self._study4_scene is not None:
                self._study4_scene.apply({**msg, "event": "show", "blocked": True})
            self.log(f"[Acquisition] COMPLETION BLOCKED: "
                     f"{msg.get('reason', 'acquisition incomplete')}")
            return
        # Clear controller-driven highlighting when that view closes.
        elif event == "close":
            self.active_ids = set()
        # Restore the graph to its initial state after a controller reset.
        elif event == "reset":
            self.graph.reset()
            self.active_ids     = set()
            self.recommended_id = None
            self.log("Live: controller reset all progress.")
            self._notify_vlm("RESET: controller reset all progress")
        # Complete every task step represented by this controller stage.
        elif event == "complete":
            for sid in ids:
                step = self.graph.by_id[sid]
                previous_state = self.graph.state(step)
                ok, message = self.graph.complete(step)
                self.log("Live: " + message)
                if ok:
                    self._speak(
                        f"Step complete. You have created "
                        f"{self.graph.friendly_part(step.output)}.")
                elif previous_state == "complete":
                    self._speak(
                        f"{self.graph.friendly_step(step)} is already complete.")
                else:
                    self._speak(
                        f"That step cannot be completed yet. First prepare "
                        f"{self._friendly_missing(self.graph.missing(step))}.",
                        warning=True)
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"COMPLETE (live): {', '.join(ids)}")
            if self.selected_id in ids:
                # A completed stage must not remain the referent of "this
                # step". Clear its acquisition/focus state and require the
                # participant to select the next trial explicitly.
                self.selected_id = None
                self.active_ids.clear()
                self._step_selected_at = None
                self._acquired_tool_ids.clear()
                self._acquisition_complete = False
                self._vlm_candidate_ids = []
                self._vlm_referent_id = None
                if self._vlm is not None:
                    self._vlm.set_focused_step("")
                    self._vlm.set_resolved_part_candidates([])
                self._publish_selected_step_board_highlight()
            # Study 4 does not score Row 1 stages 1.5 or 1.6. As soon as all
            # prerequisites for 1.5 exist, complete both sequentially; 1.5
            # unlocks 1.6. The controller emits their normal semantic events,
            # keeping Unity, Open3D, and this graph synchronized.
            if row == 1 and self.controller is not None:
                self.controller.sm.complete_automatically(1, 5)
                self.controller.sm.complete_automatically(1, 6)
        # Undo mapped steps in reverse dependency order.
        elif event == "uncomplete":
            for sid in reversed(ids):
                ok, message = self.graph.undo(self.graph.by_id[sid])
                self.log("Live: " + message)
                if ok:
                    self._speak(
                        f"Undid {self.graph.friendly_step(self.graph.by_id[sid])}.")
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"UNDO (live): {', '.join(ids)}")
        # Ignore external event types this application does not recognize.
        else:
            return
        if event in {"complete", "uncomplete", "reset"}:
            self._log_session_event(f"graph_{event}")
        # Redraw nodes and panels after applying the controller event.
        self.refresh()

    def _restore_completed_task_stages(
            self, stage_records, *, last_worked_row=None) -> tuple[int, int]:
        """Rebuild the graph deterministically from main's row/stage snapshot."""
        requested: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        rejected = 0
        for record in stage_records if isinstance(stage_records, list) else []:
            try:
                row = int(record.get("row", 0))
                stage = int(record.get("stage", 0))
            except (AttributeError, TypeError, ValueError):
                rejected += 1
                continue
            if not ((1 <= row <= 4 and 1 <= stage <= 7)
                    or (row == 0 and stage == 8)):
                rejected += 1
                continue
            coords = (row, stage)
            if coords not in seen:
                seen.add(coords)
                requested.append(coords)

        self.graph.reset()
        pending = [
            step_id
            for row, stage in requested
            for step_id in TaskGraph.steps_for_control(row, stage)
            if step_id in self.graph.by_id
        ]
        # Multiple passes make recovery robust if future graph dependencies no
        # longer match the controller's numeric stage ordering.
        while pending:
            progressed = False
            for step_id in list(pending):
                step = self.graph.by_id[step_id]
                if not self.graph.is_ready(step):
                    continue
                ok, _message = self.graph.complete(step)
                if ok:
                    pending.remove(step_id)
                    progressed = True
            if not progressed:
                break
        rejected += len(pending)

        try:
            row_hint = int(last_worked_row)
        except (TypeError, ValueError):
            row_hint = 0
        if 1 <= row_hint <= 4:
            self.graph.last_worked_row = row_hint

        self._sync_sm_from_graph()
        if self.controller is not None:
            self.controller.sm.history = [
                TaskGraph.control_coords_for(step_id)
                for step_id in self.graph.completed
                if TaskGraph.control_coords_for(step_id) is not None
            ]
        return len(self.graph.completed), rejected

    def _apply_robot_part_status(self, msg: dict, *, notify: bool = True) -> None:
        """Remember which semantic parts the robot holds or already supplied."""
        status = str(msg.get("status", "")).strip().lower()
        if status not in {"in_robot_gripper", "handed_over", "grasp_failed"}:
            return

        requested_parts = [
            str(part).strip().upper()
            for part in msg.get("requested_parts", [])
            if str(part).strip()
        ]
        tool_name = str(msg.get("tool_name", "")).strip().upper()
        supplied_parts = list(requested_parts)
        # Always resolve the physical layout object as well. Voice fetches add
        # requested_parts, while gesture fetches depend entirely on this map.
        # A row kit therefore expands to all of its contents; an individual
        # gear, rod, or stand maps to its one canonical graph identifier.
        supplied_parts.extend(self.graph.parts_for_layout_object(tool_name))
        supply_labels = PROVIDED_PARTS | set(
            gearbox_control.FASTENING_TOOL_SPECS)
        supplied_parts = list(dict.fromkeys(
            part for part in supplied_parts if part in supply_labels))
        if not supplied_parts:
            return

        if status == "grasp_failed":
            for part in supplied_parts:
                existing = self._robot_part_states.get(part)
                if (existing
                        and existing.get("source") != "task_progression"
                        and existing.get("tool_id") == msg.get("tool_id")):
                    self._robot_part_states.pop(part, None)
        else:
            for part in supplied_parts:
                self._robot_part_states[part] = {
                    "status": status,
                    "tool_id": msg.get("tool_id"),
                    "tool_name": tool_name,
                    "requested": part in requested_parts,
                    "source": "robot",
                }

        friendly_requested = [
            (self._friendly_reference_label(part, [])
             if part in gearbox_control.FASTENING_TOOL_SPECS
             else self.graph.friendly_part(part))
            for part in requested_parts
        ]
        requested_text = ", ".join(friendly_requested) or tool_name
        self.log(
            f"[RobotSupply] {requested_text}: {status} via "
            f"{tool_name} (id={msg.get('tool_id')})")
        if notify:
            self._notify_vlm(
                f"ROBOT PART STATUS: {requested_text} -> {status.upper()}")

    # ── Dear PyGui construction ───────────────────────────────────────────────

    def _create_themes(self) -> None:
        """Create and register every theme used by the task-graph interface."""
        # Use the application's Dear PyGui module through a shorter local name.
        dpg = self.dpg

        # Create a theme for assembly entries in the active-parts tree.
        with dpg.theme(tag="assembly_tree_theme"):
            # Apply the enclosed settings specifically to tree-node widgets.
            with dpg.theme_component(dpg.mvTreeNode):
                # Display assembly tree-node text in green.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (85, 235, 130, 255),
                                    category=dpg.mvThemeCat_Core)

        # Create a theme for the node-editor canvas and its dependency links.
        with dpg.theme(tag="node_editor_theme"):
            # Apply the enclosed settings specifically to node-editor widgets.
            with dpg.theme_component(dpg.mvNodeEditor):
                # Draw dependency links in light blue normally.
                dpg.add_theme_color(dpg.mvNodeCol_Link, (105, 190, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten a dependency link while the pointer is over it.
                dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (180, 225, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Draw a selected dependency link in yellow.
                dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (255, 235, 120, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Make dependency-link lines three pixels thick.
                dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, 3.0,
                                    category=dpg.mvThemeCat_Nodes)
                # Set the radius of each input/output connection pin to five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_PinCircleRadius, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Build one node theme for each TaskGraph state: complete, ready, and blocked.
        for state, color in self.COLORS.items():
            # Give the theme a stable tag such as "node_theme_complete".
            tag = f"node_theme_{state}"
            # Darken the state's RGB color for the normal node body; append full opacity.
            background = tuple(max(18, int(channel * 0.48)) for channel in color[:3]) + (255,)
            # Use a brighter version of the same color for hover and selection.
            selected_background = tuple(max(25, int(channel * 0.68)) for channel in color[:3]) + (255,)

            # Create the named theme that refresh() will later bind to nodes.
            with dpg.theme(tag=tag):
                # Apply the enclosed settings specifically to node widgets.
                with dpg.theme_component(dpg.mvNode):
                    # Keep text white for contrast against colored backgrounds.
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                        category=dpg.mvThemeCat_Core)
                    # Set the node body's normal background color.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Set the node body's background while hovered.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Set the node body's background while selected.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Color the border with the state's full-strength color.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Color the title bar with the state's full-strength color.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Keep the same title-bar color while hovered.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Keep the same title-bar color while selected.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Round the node's corners by five pixels.
                    dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                        category=dpg.mvThemeCat_Nodes)

            # Save state -> theme-tag so refresh() can select the correct theme.
            self.themes[state] = tag

        # Purple "active" overlay theme — bound to a node while its stage is open in the live
        # controller (a part was clicked in Unity), overriding its state color until the menu closes.
        # Define the full-strength purple used for the outline and title bar.
        active_color = (168, 85, 247, 255)
        # Create a darker purple for the normal node body.
        background = tuple(max(18, int(channel * 0.48)) for channel in active_color[:3]) + (255,)
        # Create a brighter purple for the hovered or selected node body.
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in active_color[:3]) + (255,)

        # Create the active-node theme under a stable Dear PyGui tag.
        with dpg.theme(tag="node_theme_active"):
            # Apply the enclosed settings specifically to node widgets.
            with dpg.theme_component(dpg.mvNode):
                # Keep active-node text white for contrast.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                # Set the active node's normal body to dark purple.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the active node's body while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the active node's body while selected.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the active node's outline in full-strength purple.
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the active node's title bar in full-strength purple.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the purple title bar while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the purple title bar while selected.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Round the active node's corners by five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Make the active theme available through self.themes["active"].
        self.themes["active"] = "node_theme_active"

        # Gold/yellow "recommended" overlay theme — shown when the system recommends
        # a step as the next to perform.
        # Define the full-strength gold used for the outline and title bar.
        rec_color  = (255, 195, 0, 255)
        # Create a darker gold for the normal node body.
        background = tuple(max(18, int(channel * 0.48)) for channel in rec_color[:3]) + (255,)
        # Create a brighter gold for the hovered or selected node body.
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in rec_color[:3]) + (255,)

        # Create the recommendation theme under a stable Dear PyGui tag.
        with dpg.theme(tag="node_theme_recommended"):
            # Apply the enclosed settings specifically to node widgets.
            with dpg.theme_component(dpg.mvNode):
                # Keep recommended-node text white for contrast.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                # Set the recommended node's normal body to dark gold.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the recommended node's body while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the recommended node's body while selected.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the recommended node's outline in full-strength gold.
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the recommended node's title bar in full-strength gold.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the gold title bar while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the gold title bar while selected.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Round the recommended node's corners by five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Make the recommendation theme available through self.themes["recommended"].
        self.themes["recommended"] = "node_theme_recommended"

    def _create_nodes(self) -> None:
        """Create and position one node-editor node for every task step."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Displayed stages 1-3 form a vertical stack. Stage 4 sits beside stage 1,
        # while stages 5-7 form a horizontal line at the height of stage 2.
        # Display stages differ from Step.stage, which represents dependency depth.
        stage_x = {
            1: 20, 2: 20, 3: 20,
            4: 500, 5: 980, 6: 1460,
            7: 1940, 8: 2420,
        }
        # Build a visual node for every Step in the task graph.
        for step in self.graph.steps:
            # Create stable item tags from the unique step ID.
            node_tag = f"node::{step.id}"
            in_tag = f"node_in::{step.id}"
            out_tag = f"node_out::{step.id}"
            # Save tags so refresh() and _create_links() can find these items.
            self.node_tags[step.id] = node_tag
            self.input_attributes[step.id] = in_tag
            self.output_attributes[step.id] = out_tag
            # Create the node using the step title as its visible heading.
            with dpg.node(label=step.title, tag=node_tag):
                # Create the connection pin for dependencies entering this step.
                with dpg.node_attribute(tag=in_tag, attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text("INPUTS", color=(255, 255, 255))
                # Create non-connectable content in the middle of the node.
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    # Display the step instructions with wrapping.
                    dpg.add_text(step.description, color=(255, 255, 255), wrap=270)
                    # Add a small visual gap before the dynamic state label.
                    dpg.add_spacer(height=3)
                    # Reserve a tagged text item that refresh() updates.
                    dpg.add_text("", tag=f"node_state::{step.id}",
                                 color=(255, 235, 120))
                    # Pass the step ID to the callback when this button is pressed.
                    dpg.add_button(label="Select step", callback=self._select_callback,
                                   user_data=step.id, width=120)
                # Create the connection pin representing this step's output.
                with dpg.node_attribute(tag=out_tag, attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(step.output, color=(175, 255, 210), wrap=260)

            # Convert the task ID to the displayed/controller stage used for layout.
            control_coords = TaskGraph.control_coords_for(step.id)
            display_stage = control_coords[1] if control_coords else 8

            # Put stages 1 and 4 on the top level; put stages 2, 5, 6, and 7
            # on the middle level; leave stage 3 on the bottom level.
            if step.row:
                row_top = 60 + (step.row - 1) * 720
                stage_offset = {
                    1: 0, 2: 240, 3: 480,
                    4: 0, 5: 240, 6: 240,
                    7: 240,
                }
                y = row_top + stage_offset.get(display_stage, 0)
            # Center the global finish step between the four row groups.
            else:
                y = 1140

            # Position the node by displayed stage horizontally and row vertically.
            dpg.set_item_pos(node_tag, (stage_x[display_stage], y))

    def _create_links(self) -> None:
        """Draw consumed-input and explicit step-prerequisite links."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Map each produced assembly name to the step ID that creates it.
        producer = {step.output: step.id for step in self.graph.steps}
        made: set[tuple[str, str]] = set()
        # Inspect every step that may consume another step's output or name an
        # explicit workflow prerequisite through Step.requires.
        for step in self.graph.steps:
            sources: list[str] = []
            # Check each consumed input required by the consumer step.
            for part in step.inputs:
                # Find the producer, or None when the input is a raw part.
                source_id = producer.get(part)
                if source_id:
                    sources.append(source_id)
            # A requires edge represents precedence without consuming the
            # producer's output (for example N.1 right stand before N.3 left).
            sources.extend(required for required in step.requires
                           if required in self.output_attributes)
            for source_id in sources:
                edge = (source_id, step.id)
                if edge in made:
                    continue
                made.add(edge)
                dpg.add_node_link(self.output_attributes[source_id],
                                  self.input_attributes[step.id],
                                  parent="task_node_editor")

    def _create_side_panel(self) -> None:
        """Create selection, action, inventory, and voice controls."""
        dpg = self.dpg
        dpg.add_text("Selected step", color=(225, 232, 240))
        dpg.add_separator()
        dpg.add_text("", tag="selection_notice", wrap=400, show=False)
        dpg.add_text("Select a graph node to inspect it.", tag="step_details", wrap=400)
        dpg.add_spacer(height=5)
        dpg.add_button(label="Mark selected step complete", tag="complete_button",
                       callback=self._complete_selected, enabled=False, width=-1)
        dpg.add_button(label="Animate in Unity", tag="animate_unity_button",
                       callback=self._animate_unity_callback, enabled=False, width=-1)
        dpg.add_button(label="Recommend next step", tag="recommend_button",
                       callback=self._recommend_callback, width=-1)
        dpg.add_button(label="Reset all progress", callback=self._reset_callback, width=-1)
        dpg.add_button(label="Reset progress + handovers",
                       callback=self._reset_progress_and_handovers_callback,
                       width=-1)
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Active parts and assemblies")
        dpg.add_text("Top-level entries are active. Expand an assembly to see its composition.",
                     color=(165, 180, 195), wrap=400)
        with dpg.child_window(tag="active_parts_panel", height=245,
                              horizontal_scrollbar=True):
            dpg.add_group(tag="active_parts_tree")

        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Voice Input  (Parakeet ASR)", color=(225, 232, 240))
        dpg.add_checkbox(label="Enable voice input", tag="voice_input_enabled",
                         default_value=True, callback=self._toggle_voice_input)
        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=(160, 170, 185))
            dpg.add_text("—", tag="voice_status", color=(180, 180, 180, 255))
        with dpg.group(horizontal=True):
            dpg.add_text("Timer: ", color=(160, 170, 185))
            dpg.add_text("", tag="voice_timer", color=(100, 220, 255, 255))
        dpg.add_text("Level:", color=(160, 170, 185))
        dpg.add_progress_bar(tag="voice_rms", default_value=0.0, width=-1)
        dpg.add_spacer(height=4)
        dpg.add_checkbox(label="Route transcripts to VLM", tag="voice_to_vlm",
                         default_value=True)
        dpg.add_spacer(height=4)
        dpg.add_text("Transcripts:", color=(160, 170, 185))
        dpg.add_input_text(tag="voice_transcripts", multiline=True, readonly=True,
                           width=-1, height=120,
                           default_value="Speak normally; no wake word is required.")

    # ── Voice input and VLM integration ──────────────────────────────────────

    def _toggle_voice_input(self, sender, app_data, user_data=None) -> None:
        """Pause recognition while retaining the loaded ASR and VLM models."""
        enabled = bool(app_data)
        if self._speech is not None:
            self._speech.set_input_enabled(enabled)
        self.log(f"[Voice] Input {'enabled' if enabled else 'disabled'}.")

    def _poll_speech(self) -> None:
        """Refresh voice widgets and handle all newly available speech events."""
        if self._speech is None:
            return
        dpg = self.dpg
        events = self._speech.poll()
        if dpg.get_value("voice_input_enabled") != self._speech.input_enabled:
            dpg.set_value("voice_input_enabled", self._speech.input_enabled)

        # Status label + color
        # loading=model starting; idle=waiting for an optional wake word;
        # speech=someone is talking and audio is being captured right now;
        # queued=captured audio is waiting for ASR; transcribing=ASR is running;
        # listening=continuous capture is active (or an optional wake word was
        # accepted); error=capture or recognition failed.
        status = self._speech.current_status
        color, label = self._VOICE_STATUS_STYLE.get(
            status, ((200, 200, 200, 255), status))
        if not self._speech.input_enabled:
            label = "Disabled — microphone input is ignored"
        elif self._speech.always_listening:
            label = "Listening — no wake word required"
        elif self._speech.listening_active:
            label = f"Listening  (wake word: \"{self._speech.wake_word}\")"
        elif status == "idle":
            label = f"Idle — say \"{self._speech.wake_word}\""
        dpg.set_value("voice_status", label)
        dpg.configure_item("voice_status", color=list(color))

        # Timer
        if self._speech.listening_active and not self._speech.always_listening:
            dpg.set_value("voice_timer", f"{self._speech.remaining_time:.0f}s remaining")
        else:
            dpg.set_value("voice_timer", "")

        # RMS bar (clamp to [0, 1])
        dpg.set_value("voice_rms", min(self._speech.current_rms * 10.0, 1.0))

        # Transcript log
        history = list(self._speech.transcript_history)
        if history:
            dpg.set_value("voice_transcripts", "\n".join(f"› {t}" for t in history))

        # Log notable events to the terminal; optionally route transcripts to the VLM.
        route_to_vlm = (
            self._vlm is not None
            and dpg.get_value("voice_to_vlm")
            and self.selected_id is not None
            and self._trial_recording)
        for kind, payload in events:
            if kind == "wake_word":
                self.log("[Voice] Wake word detected — listening.")
            elif kind == "transcript":
                self.log(f"[Voice] {payload}")
                if self._tts is not None and self._tts.is_speaking:
                    self.log("[Voice] Ignored transcript produced during TTS playback.")
                    continue
                if route_to_vlm:
                    if not self._vlm.submit_question(payload):
                        self.log("[Voice] VLM busy — transcript skipped.")
                    else:
                        self.log("[Voice] Transcript routed to VLM.")
                elif self._vlm is not None and dpg.get_value("voice_to_vlm"):
                    self.log(
                        "[Voice] Transcript not routed to VLM — select a step "
                        "and press ENTER to start the trial.")
            elif kind == "timeout":
                self.log("[Voice] Timed out — back to idle.")
            elif kind == "error":
                self.log(f"[Voice error] {payload}")

    def _speak(self, text: str, warning: bool = False,
               allow_before_start: bool = False) -> None:
        """Queue concise guidance without blocking the Dear PyGui frame loop."""
        if self._tts is None:
            return
        if (self.selected_id and not self._trial_recording
                and not self._acquisition_complete
                and not allow_before_start):
            return
        self.log(f"[TTS] {text}")
        self._tts.speak(text, replace=warning)

    def _poll_tts(self) -> None:
        if self._tts is None:
            return
        for kind, payload in self._tts.poll():
            if kind == "error":
                self.log(f"[TTS error] {payload}")

    def _poll_vlm_part_references(self) -> None:
        if self._vlm is None:
            return
        for result in self._vlm.poll_part_references():
            self._handle_part_reference(result)

    def _poll_vlm_answers(self) -> None:
        """Speak validated natural-language answers produced by the VLM."""
        if self._vlm is None:
            return
        for result in self._vlm.poll_answers():
            answer = result.get("answer", "").strip()
            self._log_session_event(
                "vlm_answer", modality="language",
                transcript=result.get("text", ""),
                vlm_raw=result.get("raw", ""),
                graph_decision=result.get("intent", ""),
                spoken_response=answer)
            if answer:
                self._speak(
                    answer,
                    warning=result.get("intent") == "invalid",
                )

    def _clear_pending_fetch(self) -> None:
        self._pending_fetch = None
        if self._vlm is not None:
            self._vlm.set_pending_fetch(None)

    @staticmethod
    def _row_of_part(part: str) -> int | None:
        match = re.search(r"_ROW([1-4])(?:_|$)", str(part).upper())
        return int(match.group(1)) if match else None

    def _friendly_reference_label(self, label: str,
                                  parts: list[str]) -> str:
        label = label.upper()
        kit = re.fullmatch(r"ROW([1-4])_KIT", label)
        if kit:
            return f"Row {kit.group(1)} kit"
        tool_spec = gearbox_control.FASTENING_TOOL_SPECS.get(label)
        if tool_spec is not None:
            return str(tool_spec["friendly"])
        if label == "BEARING":
            return "bearing"
        if label == "PIN":
            return "wooden retaining pin"
        row_group = re.fullmatch(r"(BEARING|PIN)_ROW([1-4])", label)
        if row_group:
            kind, row = row_group.groups()
            noun = "bearings" if kind == "BEARING" else "wooden retaining pins"
            return f"Row {row} {noun}"
        if label.startswith("SCREW_ROW"):
            return f"Row {label[-1]} mounting screw"
        friendly = self.graph.friendly_part(parts[0]) if parts else "part"
        # Reference-response templates add their own article ("The"/"That").
        # friendly_part() includes "the" when it is used as a standalone noun
        # phrase, so remove it here to avoid speech such as "The the Row 1...".
        return friendly[4:] if friendly.lower().startswith("the ") else friendly

    def _selected_step_board_parts(self) -> list[str]:
        """Return raw BoardAR components relevant to the selected step."""
        if not self.selected_id or self.selected_id not in self.graph.by_id:
            return []
        step = self.graph.by_id[self.selected_id]
        required = set((*step.inputs, *step.context))
        parts: list[str] = []
        for part in sorted(PROVIDED_PARTS):
            current = self.graph.current_container(part)
            if part in required or current in required:
                parts.append(part)
        return parts

    def _publish_selected_step_board_highlight(self) -> None:
        """Clear semantic highlights after an ordinary step selection.

        The physical pegboard is the search space being measured in Study 4;
        automatically coloring its required objects would give every condition
        an unintended visual hint. BoardAR also retains only its normal
        animation/progression appearance for ordinary selections. Condition 3
        reveals the selected step's cyan pegboard cue only after the user asks
        what is needed through the STEP_ITEMS language-request path.
        """
        self._send_select({
            "event": "step_context_highlight",
            "step_ids": [],
        })
        self._send_select({
            "event": "board_step_highlight",
            "assembly_parts": [],
            "assembly_color": [0.0, 1.0, 1.0, 0.0],
        })
        if self.controller is not None:
            self.controller.send({"command": "reference_clear"})

    def _publish_selected_step_pegboard_highlight(
            self, *, announce: bool = False) -> None:
        """Highlight Condition-3 selected-step objects in both 3D views."""
        ids = self._selected_step_pegboard_ids()
        # step_context_highlight is mirrored by Study4Scene to its Open3D tool
        # boxes and to Unity's ToolColorReceiver for the physical pegboard.
        self._send_select({
            "event": "step_context_highlight",
            "step_ids": ids,
        })
        # Keep the assembled AR board free of semantic recommendation color.
        self._send_select({
            "event": "board_step_highlight",
            "assembly_parts": [],
            "assembly_color": [0.0, 1.0, 1.0, 0.0],
        })
        if self.controller is not None:
            self.controller.send({"command": "reference_clear"})
        if announce:
            self.log(
                f"[Step] Cyan highlight → {len(ids)} pegboard object(s) "
                "in Unity and Open3D")

    def _selected_step_pegboard_ids(self) -> list[int]:
        """Return pegboard objects entering the currently selected stage."""
        if not self.selected_id:
            return []
        coords = TaskGraph.control_coords_for(self.selected_id)
        if coords is None or coords[0] <= 0:
            return []
        index = gearbox_control.load_tool_index(self._tool_layout_path)
        row, stage = coords
        ids = list(gearbox_control.appearing_ids(index, row, stage))
        if row > 0 and stage in (4, 6):
            for tool_id in gearbox_control.fastening_tool_ids(index, row):
                if tool_id not in ids:
                    ids.append(tool_id)
        return [tool_id for tool_id in ids
                if tool_id not in self._removed_tool_ids
                and tool_id not in self._acquired_tool_ids]

    def _removable_consumable_ids(self, tool_ids: Iterable[int]) -> set[int]:
        """Consume parts while keeping reusable tools available for later rows."""
        return {int(tool_id) for tool_id in tool_ids
                if (int(tool_id) in self._tool_defs_by_id
                    and self._tool_defs_by_id[int(tool_id)].get("category")
                    == "part")}

    def _handle_pegboard_interaction(self, interaction: dict) -> None:
        """Log hovers and speak the hovered part during an active trial."""
        if interaction.get("event_type") != "hover_enter":
            return
        tool_id = int(interaction["tool_id"])
        now = time.monotonic()
        if (tool_id == self._last_hover_spoken_id
                and now - self._last_hover_spoken_at < 0.75):
            return
        self._last_hover_spoken_id = tool_id
        self._last_hover_spoken_at = now
        tool_name = next(
            (name for name, indexed_id in self._tool_index.items()
             if int(indexed_id) == tool_id),
            f"object {tool_id}",
        )
        friendly = self._friendly_reference_label(
            tool_name, self.graph.parts_for_layout_object(tool_name))
        self.log(f"[Hover] {friendly} (id={tool_id})")
        self._log_session_event(
            "hover", modality="hover", tool_id=tool_id,
            tool_name=tool_name)
        if self._tts is not None and self._trial_recording:
            # Every Study 4 condition receives the same hover-name feedback
            # after ENTER starts the trial. Replace stale hover speech so the
            # audio always describes the participant's current focus.
            self._tts.speak(friendly, replace=True)

    def _score_part_click(self, interaction: dict) -> bool:
        """Score a click against graph truth without exposing truth visually."""
        tool_id = int(interaction["tool_id"])
        modality = str(interaction.get("modality", "click"))
        if not self._trial_recording:
            if self._acquisition_complete:
                self._speak("This step is already complete.", warning=True)
            else:
                self._speak("Press Enter to start this trial.", warning=True)
                self._log_session_event(
                    "selection_before_start", modality=modality,
                    tool_id=tool_id)
            return False
        expected = sorted(self._trial_required_tool_ids)
        correct = bool(self.selected_id and tool_id in expected)
        newly_acquired = correct and tool_id not in self._acquired_tool_ids
        self._trial_total_clicks += 1
        if not correct:
            self._trial_wrong_clicks += 1
        elif not newly_acquired:
            self._trial_repeated_correct_clicks += 1
        if newly_acquired:
            self._acquired_tool_ids.add(tool_id)
        # Never reveal the initial target count. Report only what remains after
        # the participant has made a correct acquisition.
        if not self.selected_id:
            self._speak("Please select a step before selecting a part.", warning=True)
        elif not correct:
            self._speak("Wrong part.", warning=True)
        tool_name = next(
            (name for name, indexed_id in self._tool_index.items()
             if int(indexed_id) == tool_id),
            f"id={tool_id}",
        )
        step = (self.graph.by_id.get(self.selected_id)
                if self.selected_id else None)
        now = time.monotonic()
        part_elapsed = (
            now - (self._last_acquisition_at or self._step_selected_at or now))
        step_elapsed = (
            0.0 if self._step_selected_at is None
            else now - self._step_selected_at)
        if self._click_logger is not None:
            self._click_logger.append(
                condition=self.study4_condition or "",
                event_type=("language_selection"
                            if modality == "language" else "click"),
                selected_step=self.selected_id or "",
                selected_step_state=(self.graph.state(step) if step else "none"),
                step_selected_at=("" if self._step_selected_at is None
                                  else f"{self._step_selected_at:.6f}"),
                response_time_s=("" if self._step_selected_at is None else
                                 f"{time.monotonic() - self._step_selected_at:.6f}"),
                clicked_tool_id=tool_id,
                clicked_tool_name=tool_name,
                expected_tool_ids=expected,
                acquired_tool_ids=sorted(self._acquired_tool_ids),
                required_count=len(expected),
                acquired_count=len(self._acquired_tool_ids),
                correct=int(correct),
                newly_acquired=int(newly_acquired),
                hand=interaction.get("hand", "unknown"),
            )
        self.log(f"[Acquisition] {'CORRECT' if correct else 'WRONG'} "
                 f"{modality} selection {tool_name} (id={tool_id}); "
                 f"expected={expected}")
        self._log_session_event(
            "part_selection", modality=modality, tool_id=tool_id,
            tool_name=tool_name, part_elapsed_s=f"{part_elapsed:.6f}",
            step_elapsed_s=f"{step_elapsed:.6f}", correct=int(correct))
        if newly_acquired:
            self._last_acquisition_at = now
            if self._study4_scene is not None:
                self._study4_scene.mark_acquired_objects([tool_id])
            self._log_session_event(
                "object_acquired", modality=modality, tool_id=tool_id,
                tool_name=tool_name, part_elapsed_s=f"{part_elapsed:.6f}",
                step_elapsed_s=f"{step_elapsed:.6f}", correct=1)
        if (expected and not self._acquisition_complete
                and self._acquired_tool_ids == set(expected)):
            self._acquisition_complete = True
            elapsed = (time.monotonic() - self._step_selected_at
                       if self._step_selected_at is not None else None)
            self._trial_finished_elapsed_s = elapsed
            self._trial_recording = False
            head_motion = (
                self._study4_scene.stop_head_motion_summary()
                if self._study4_scene is not None else {})
            self._trial_head_translation_m = head_motion.get(
                "head_translation_m")
            self._trial_head_rotation_deg = head_motion.get(
                "head_rotation_deg")
            removed = self._removable_consumable_ids(self._acquired_tool_ids)
            self._removed_tool_ids.update(removed)
            if self._study4_scene is not None:
                self._study4_scene.remove_acquired_objects(removed)
            if self._click_logger is not None:
                self._click_logger.append(
                    condition=self.study4_condition or "",
                    event_type="step_acquisition_complete",
                    selected_step=self.selected_id or "",
                    selected_step_state=(self.graph.state(step) if step else "none"),
                    step_selected_at=("" if self._step_selected_at is None
                                      else f"{self._step_selected_at:.6f}"),
                    response_time_s=("" if elapsed is None else f"{elapsed:.6f}"),
                    clicked_tool_id=tool_id,
                    clicked_tool_name=tool_name,
                    expected_tool_ids=expected,
                    acquired_tool_ids=sorted(self._acquired_tool_ids),
                    required_count=len(expected),
                    acquired_count=len(self._acquired_tool_ids),
                    correct=1,
                    newly_acquired=int(newly_acquired),
                    hand=interaction.get("hand", "unknown"),
                )
            if self._step_logger is not None and step is not None:
                names = [next(
                    (name for name, indexed_id in self._tool_index.items()
                     if int(indexed_id) == required_id),
                    f"id={required_id}") for required_id in expected]
                self._step_logger.append(
                    condition=self.study4_condition or "",
                    step_id=step.id,
                    step_title=step.title,
                    duration_s=("" if elapsed is None else f"{elapsed:.6f}"),
                    required_tool_ids=expected,
                    required_tool_names=names,
                    total_clicks=self._trial_total_clicks,
                    wrong_clicks=self._trial_wrong_clicks,
                    repeated_correct_clicks=self._trial_repeated_correct_clicks,
                )
            self.log(f"[Acquisition] STEP COMPLETE {self.selected_id}: "
                     f"{len(expected)}/{len(expected)} objects in "
                     f"{elapsed:.3f}s" if elapsed is not None else
                     f"[Acquisition] STEP COMPLETE {self.selected_id}")
            self._print_step_result(elapsed, len(expected))
            self._log_session_event(
                "step_acquisition_complete", modality=modality,
                step_elapsed_s=("" if elapsed is None else f"{elapsed:.6f}"),
                head_translation_m=(
                    "" if self._trial_head_translation_m is None
                    else f"{self._trial_head_translation_m:.6f}"),
                head_rotation_deg=(
                    "" if self._trial_head_rotation_deg is None
                    else f"{self._trial_head_rotation_deg:.6f}"))
            self._auto_complete_acquired_step()
        elif correct:
            remaining = len(set(expected) - self._acquired_tool_ids)
            noun = "part" if remaining == 1 else "parts"
            self._speak(f"{remaining} {noun} left.", warning=True)
        return correct

    def _start_acquisition_trial(self) -> None:
        # Selecting a step prepares it, but the participant controls the timer
        # explicitly with Enter after they have understood the task.
        # Physical inventory persists: every previously acquired pegboard
        # object remains with the participant and outside later search spaces.
        self._step_event_index = 0
        self._step_selected_at = None
        self._last_acquisition_at = None
        self._trial_recording = False
        self._trial_start_pending = False
        self._trial_start_pending_at = None
        self._trial_finished_elapsed_s = None
        self._trial_head_translation_m = None
        self._trial_head_rotation_deg = None
        if self._speech is not None and not self._speech.input_enabled:
            self._speech.set_input_enabled(True)
        if self._tts is not None:
            # Do not let guidance queued for the prior state disclose object
            # names while the participant is studying the new READY step.
            self._tts.clear()
        self._trial_required_tool_ids = set(
            self._selected_step_pegboard_ids())
        self._acquired_tool_ids.clear()
        self._acquisition_complete = False
        self._trial_total_clicks = 0
        self._trial_wrong_clicks = 0
        self._trial_repeated_correct_clicks = 0
        self._vlm_candidate_ids = []
        self._vlm_referent_id = None
        if self._study4_scene is not None:
            self._study4_scene.reset_acquisition_colors()
        self._print_step_header()
        self._log_session_event("step_prepared")
        self.log("[Trial] READY — press ENTER when the participant is ready.")

    def _handle_trial_enter(self) -> bool:
        """Arm a prepared trial; timing starts after audio is clean."""
        if not self.selected_id:
            return False
        if self._trial_recording or self._trial_start_pending:
            return True
        if self._acquisition_complete:
            return True
        self._trial_start_pending = True
        self._trial_start_pending_at = time.monotonic()
        if self._tts is not None:
            self._tts.clear()
        if self._speech is not None:
            # This clears queued/raw audio and makes the VAD discard its active
            # utterance before the recorded interval begins.
            self._speech.set_input_enabled(False)
        self.log(
            "[Trial] STARTING — clearing TTS and microphone buffers; timer "
            "has not started yet.")
        return True

    def _finalize_pending_trial_start(self) -> None:
        """Enter RUNNING only after TTS stops and the audio path settles."""
        if not self._trial_start_pending:
            return
        now = time.monotonic()
        if now - (self._trial_start_pending_at or now) < 0.50:
            return
        if self._tts is not None and self._tts.is_speaking:
            return
        if self._vlm is not None and not self._vlm.is_ready:
            return
        if self._speech is not None and not self._speech.model_ready:
            return
        if self._speech is not None:
            self._speech.set_input_enabled(True)
        self._trial_start_pending = False
        self._trial_start_pending_at = None
        self._step_attempt += 1
        self._step_event_index = 0
        self._step_selected_at = now
        self._last_acquisition_at = self._step_selected_at
        self._trial_recording = True
        self._trial_finished_elapsed_s = None
        self._trial_head_translation_m = None
        self._trial_head_rotation_deg = None
        if self._study4_scene is not None:
            self._study4_scene.start_head_motion_summary()
        self._log_session_event("step_started", modality="keyboard")
        self.log(
            f"[Trial] RUNNING attempt={self._step_attempt} — timer started; "
            "fetch all required parts to stop it.")

    def _update_terminal_trial_status(self) -> None:
        """Keep one replace-in-place trial status line at the terminal bottom."""
        now = time.monotonic()
        if now - self._last_terminal_status_at < 0.10:
            return
        self._last_terminal_status_at = now
        if not self.selected_id:
            state, elapsed = "NO STEP", 0.0
        elif self._acquisition_complete:
            state = "COMPLETE"
            elapsed = self._trial_finished_elapsed_s or 0.0
        elif self._trial_start_pending:
            state, elapsed = "STARTING — preparing audio", 0.0
        elif self._trial_recording:
            state = "RUNNING"
            elapsed = now - (self._step_selected_at or now)
        else:
            state, elapsed = "READY — press ENTER", 0.0
        acquired = len(self._acquired_tool_ids)
        required = len(self._trial_required_tool_ids)
        print(
            f"\r\033[2K[TRIAL STATUS] {state} | "
            f"step={self.selected_id or '-'} | time={elapsed:7.1f}s | "
            f"parts={acquired}/{required}",
            end="", flush=True)

    def _print_step_header(self) -> None:
        """Visually separate each participant acquisition trial in stdout."""
        if not self.selected_id or self.selected_id not in self.graph.by_id:
            return
        step = self.graph.by_id[self.selected_id]
        coords = TaskGraph.control_coords_for(step.id)
        location = (f"Row {coords[0]} · Stage {coords[1]}"
                    if coords is not None else "Unnumbered step")
        expected = sorted(self._trial_required_tool_ids)
        names_by_id = {
            int(tool_id): name for name, tool_id in self._tool_index.items()
        }
        names = [names_by_id.get(tool_id, f"id={tool_id}")
                 for tool_id in expected]
        required = ", ".join(names) if names else "none (auto-complete)"
        border = "=" * 78
        print(
            f"\n{border}\n"
            f"[TRIAL ] START  {location}  |  {step.id}\n"
            f"[TRIAL ] {step.title}\n"
            f"[TRIAL ] REQUIRED ({len(expected)}): {required}\n"
            f"{border}",
            flush=True,
        )

    def _print_step_result(self, elapsed: float | None,
                           required_count: int) -> None:
        """Print a compact end-of-trial summary without changing study data."""
        border = "-" * 78
        duration = "unknown" if elapsed is None else f"{elapsed:.3f} s"
        print(
            f"{border}\n"
            f"[TRIAL ] COMPLETE  {self.selected_id or 'unknown'}\n"
            f"[TRIAL ] TIME={duration}  REQUIRED={required_count}  "
            f"CLICKS={self._trial_total_clicks}  "
            f"WRONG={self._trial_wrong_clicks}  "
            f"REPEATED={self._trial_repeated_correct_clicks}\n"
            f"{border}\n",
            flush=True,
        )

    def _completion_guard(self, row: int, stage: int) -> tuple[bool, str]:
        coords = (int(row), int(stage))
        selected_coords = (TaskGraph.control_coords_for(self.selected_id)
                           if self.selected_id else None)
        if selected_coords != coords:
            return False, "selected step does not match this checkbox"
        expected = sorted(self._trial_required_tool_ids)
        missing_count = len(set(expected) - self._acquired_tool_ids)
        if not self._acquisition_complete:
            return False, f"{missing_count} required object(s) not clicked"
        return True, "acquisition complete"

    def _auto_complete_no_retrieval(self) -> None:
        """Complete an assembly-only trial and emit green Unity/Open3D state."""
        if self._acquisition_complete or not self.selected_id:
            return
        expected = sorted(self._trial_required_tool_ids)
        if expected:
            return
        step = self.graph.by_id[self.selected_id]
        self._acquisition_complete = True
        elapsed = (time.monotonic() - self._step_selected_at
                   if self._step_selected_at is not None else 0.0)
        if self._step_logger is not None:
            self._step_logger.append(
                condition=self.study4_condition or "",
                step_id=step.id,
                step_title=step.title,
                duration_s=f"{elapsed:.6f}",
                required_tool_ids=[],
                required_tool_names=[],
                total_clicks=0,
                wrong_clicks=0,
                repeated_correct_clicks=0,
            )
        self.log(f"[Acquisition] STEP COMPLETE {step.id}: "
                 "auto_completed_no_retrieval")
        self._print_step_result(elapsed, 0)
        coords = TaskGraph.control_coords_for(step.id)
        if self.controller is not None and coords is not None:
            self.controller.sm.current_row, self.controller.sm.current_stage = coords
            self.controller.sm._handle_checkbox()

    def _send_reference_highlight(self, ids: list[int], color: list[float],
                                  status: str, label: str,
                                  assembly_parts: Iterable[str] = (),
                                  result: dict[str, object] | None = None) -> None:
        # Only explicitly referenced objects are highlighted. Merely selecting
        # a step never adds its pegboard IDs or BoardAR parts to this response.
        assembly_parts = list(dict.fromkeys(assembly_parts))
        assembly_color = list(color)
        # Cyan is reserved for VLM candidate locations on the pegboard. Never
        # mirror that cyan layer onto BoardAR; doing so makes the assembly board
        # an additional answer cue. Non-cyan referent/warning colors may still
        # be used for an explicitly referenced assembled component.
        is_cyan = (len(assembly_color) >= 3
                   and assembly_color[0] <= 0.05
                   and assembly_color[1] >= 0.95
                   and assembly_color[2] >= 0.95)
        if is_cyan:
            assembly_parts = []
        candidate_ids = sorted({int(tool_id) for tool_id in ids})
        referred_ids = (candidate_ids if len(candidate_ids) == 1
                        and label != "AMBIGUOUS" else [])
        if self.study4_condition == "task_aware":
            relation = str((result or {}).get(
                "reference_relation", "direct"))
            # Candidate labels are VLM output. Expand only those predictions to
            # physical IDs; do not inject selected-step ground truth here.
            inferred_ids: list[int] = []
            for candidate_label in (result or {}).get("candidate_labels", []):
                candidate_label = str(candidate_label)
                if candidate_label in gearbox_control.FASTENING_TOOL_SPECS:
                    mapped = gearbox_control.tool_ids_for_reference(
                        self._tool_index, candidate_label)
                else:
                    mapped = [
                        gearbox_control.tool_id_for_graph_part(
                            self._tool_index, part)
                        for part in self.graph.parts_for_reference_label(
                            candidate_label)
                    ]
                for mapped_id in mapped:
                    if mapped_id is not None and mapped_id not in inferred_ids:
                        inferred_ids.append(int(mapped_id))
            proposed = list(dict.fromkeys((*inferred_ids, *candidate_ids)))
            if relation == "alternative_to_previous" and self._vlm_candidate_ids:
                previous_referent = self._vlm_referent_id
                pool = list(self._vlm_candidate_ids)
                for proposed_id in proposed:
                    if proposed_id not in pool:
                        pool.append(proposed_id)
                alternatives = [
                    item for item in (proposed or pool)
                    if item != previous_referent
                ]
                if alternatives:
                    self._vlm_referent_id = (
                        None if label == "AMBIGUOUS" and len(alternatives) > 1
                        else alternatives[0])
                else:
                    self._vlm_referent_id = None
                self._vlm_candidate_ids = alternatives
            elif proposed:
                self._vlm_candidate_ids = proposed
                self._vlm_referent_id = (
                    None if label == "AMBIGUOUS" and len(proposed) > 1
                    else proposed[0])
            else:
                self._vlm_candidate_ids = []
                self._vlm_referent_id = None
            candidate_ids = list(self._vlm_candidate_ids)
            referred_ids = ([] if self._vlm_referent_id is None
                            else [self._vlm_referent_id])
            # Condition 3 uses the selected step's outstanding pegboard
            # objects as a hard visual search-space boundary. Never allow a
            # VLM candidate or discourse carry-over to color another object.
            if self.selected_id:
                allowed_ids = set(self._selected_step_pegboard_ids())
                candidate_ids = [
                    tool_id for tool_id in candidate_ids
                    if tool_id in allowed_ids
                ]
                referred_ids = [
                    tool_id for tool_id in referred_ids
                    if tool_id in allowed_ids
                ]
                self._vlm_candidate_ids = list(candidate_ids)
                self._vlm_referent_id = (
                    referred_ids[0] if referred_ids else None)
        self._send_select({
            "event": "reference_highlight",
            "ids": candidate_ids,
            "candidate_ids": candidate_ids,
            "step_ids": [],
            "referred_ids": referred_ids,
            "color": color,
            "assembly_color": assembly_color,
            "status": status,
            "label": label,
            "assembly_parts": assembly_parts,
        })
        # In --with-controller mode this also works when main_with_robot.py is
        # not running. When main is present, receiving the same idempotent
        # reference color a second time is harmless.
        if self.controller is not None:
            self.controller.send(
                {"command": "reference_color", "parts": assembly_parts,
                 "color": assembly_color}
                if assembly_parts else {"command": "reference_clear"})

    def _record_reference_decision(self, result: dict[str, object], decision: str,
                                   matched_parts: Iterable[str],
                                   current_assemblies: Iterable[str], ids: list[int],
                                   assembly_parts: Iterable[str], spoken: str) -> None:
        step = self.graph.by_id.get(self.selected_id) if self.selected_id else None
        if self._assistant_logger is not None:
            try:
                self._assistant_logger.append(
                study_condition=self.study4_condition or "",
                transcript=result.get("text", ""),
                vlm_prediction=result.get("label", "INVALID_OUTPUT"),
                vlm_raw=result.get("raw", ""),
                selected_step=step.id if step else "",
                selected_step_state=self.graph.state(step) if step else "none",
                graph_decision=decision,
                matched_parts=list(matched_parts),
                current_assemblies=list(current_assemblies),
                highlight_ids=ids,
                assembly_highlight_parts=list(assembly_parts),
                    spoken_response=spoken,
                )
            except OSError as error:
                self.log(f"[StudyLog error] {error}")
        self._log_session_event(
            "vlm_interaction", modality="language",
            transcript=result.get("text", ""),
            vlm_prediction=result.get("label", "INVALID_OUTPUT"),
            vlm_raw=result.get("raw", ""), graph_decision=decision,
            spoken_response=spoken)

    def _emit_reference_decision(
            self, result: dict[str, object], decision: str, color: list[float],
            spoken: str, *, ids: list[int] | None = None,
            matched_parts: Iterable[str] = (),
            current_assemblies: Iterable[str] = (),
            assembly_parts: Iterable[str] = (), warning: bool = False) -> None:
        ids = sorted(set(ids or []))
        assembly_parts = list(dict.fromkeys(assembly_parts))
        self._send_reference_highlight(
            ids, color, decision, result.get("label", "INVALID_OUTPUT"),
            assembly_parts=assembly_parts, result=result)
        if self._vlm is not None:
            self._vlm.apply_policy_response(
                result.get("text", ""),
                result.get("label", "INVALID_OUTPUT"),
                spoken,
            )
        # Study 4 part/tool referral feedback is visual and logged only. Do
        # not reveal or reinforce referred objects through spoken responses in
        # any condition.
        self._record_reference_decision(
            result, decision, matched_parts, current_assemblies, ids,
            assembly_parts, spoken)
        # A single valid yellow language referent is an acquisition, exactly
        # like a correct Unity click. Ambiguous/cyan sets and warnings remain
        # non-selecting visual context.
        if (not warning and len(ids) == 1
                and self.study4_condition in {"language", "task_aware"}
                and self.selected_id and ids[0] in self._trial_required_tool_ids
                and ids[0] not in self._acquired_tool_ids):
            self._score_part_click({
                "tool_id": ids[0],
                "event_type": "language_selection",
                "modality": "language",
                "hand": "voice",
            })

    @staticmethod
    def _ambiguity_question(text: str) -> str:
        words = str(text).lower()
        if any(word in words for word in ("stand", "bracket", "support")):
            return "Which row and side do you mean: left or right?"
        if "gear" in words and "rod" not in words and "shaft" not in words:
            return "Which gear do you mean? Please give its row, color, size, or position."
        if "holder" in words:
            return ("Which bit holder do you mean: Bit Holder 1 with the H5 and "
                    "H3 bits, or Bit Holder 2 with the T25 bit?")
        if any(word in words for word in ("screwdriver", "driver", "bit", "wrench")):
            return ("Which fastening tool do you mean: H5, T25, H3, or the "
                    "Phillips screwdriver? You can also tell me the row.")
        if any(word in words for word in ("screw", "bearing", "pin")):
            return "Which row are you assembling?"
        return "Which object do you mean? Please add its row, side, color, size, or position."

    def _candidate_ambiguity_question(
            self, text: str, candidates: Iterable[str]) -> str:
        """Ask only for the attribute that distinguishes known candidates."""
        parts = list(candidates)
        rows = {self._row_of_part(part) for part in parts}
        rows.discard(None)
        sides = {
            match.group(1).lower()
            for part in parts
            if (match := re.search(r"_(LEFT|RIGHT)$", part)) is not None
        }
        if len(rows) == 1 and sides == {"left", "right"}:
            row = next(iter(rows))
            if all(part.startswith("GEAR_ROW") for part in parts):
                noun = "gear"
            elif all(part.startswith("STAND_ROW") for part in parts):
                noun = "stand"
            else:
                noun = "object"
            return f"Do you mean the left {noun} or the right {noun} in Row {row}?"
        return self._ambiguity_question(text)

    @staticmethod
    def _is_explicit_plural_reference(text: str) -> bool:
        """Recognize an explicit plural noun without guessing from candidate count."""
        return bool(re.search(
            r"\b(gears|stands|bearings|pins|screws|holders|tools|parts|objects|everything|all)\b",
            str(text).lower()))

    @staticmethod
    def _ambiguous_part_category(text: str) -> tuple[str, list[str]]:
        """Expand an ambiguous common noun without guessing a specific object."""
        words = str(text).lower()
        if any(word in words for word in ("stand", "bracket", "support")):
            return "Gear stands", sorted(
                part for part in PROVIDED_PARTS if part.startswith("STAND_"))
        if "gear" in words and any(word in words for word in ("rod", "shaft")):
            return "Gear rods", sorted(
                part for part in PROVIDED_PARTS if part.startswith("GEAR_ROD_"))
        if "gear" in words:
            return "Gears", sorted(
                part for part in PROVIDED_PARTS if part.startswith("GEAR_ROW"))
        if "bearing" in words:
            return "Bearings", sorted(
                part for part in PROVIDED_PARTS if part.startswith("BEARING_"))
        if "screw" in words:
            return "Screws", sorted(
                part for part in PROVIDED_PARTS if part.startswith("SCREW_"))
        if "pin" in words:
            return "Wooden pins", sorted(
                part for part in PROVIDED_PARTS if part.startswith("PIN_"))
        return "", []

    def _candidate_parts_from_vlm(self, labels: Iterable[str]) -> list[str]:
        """Expand validated VLM ambiguity candidates into graph part names."""
        parts: list[str] = []
        for label in labels:
            for part in self.graph.parts_for_reference_label(label):
                if part not in parts:
                    parts.append(part)
        return parts

    def _physical_ids_for_vlm_candidates(
            self, labels: Iterable[str], parts: Iterable[str] = ()) -> list[int]:
        """Map only VLM-proposed candidates to visible pegboard objects."""
        ids: list[int] = []
        for label in labels:
            label = str(label)
            if label in gearbox_control.FASTENING_TOOL_SPECS:
                mapped = gearbox_control.tool_ids_for_reference(
                    self._tool_index, label)
                for tool_id in mapped:
                    if tool_id not in ids:
                        ids.append(tool_id)
        for part in parts:
            tool_id = gearbox_control.tool_id_for_graph_part(
                self._tool_index, str(part))
            if tool_id is not None and tool_id not in ids:
                ids.append(tool_id)
        return ids

    def _friendly_candidate_category(self, candidates: Iterable[str]) -> str:
        """Describe a VLM-provided candidate set without reparsing the speech."""
        parts = list(candidates)
        if not parts:
            return ""
        if all(part.startswith("STAND_") for part in parts):
            category = "Gear stands"
        elif all(part.startswith("GEAR_ROD_") for part in parts):
            category = "Gear rods"
        elif all(part.startswith("GEAR_ROW") for part in parts):
            category = "Gears"
        elif all(part.startswith("BEARING_") for part in parts):
            category = "Bearings"
        elif all(part.startswith("SCREW_") for part in parts):
            category = "Screws"
        elif all(part.startswith("PIN_") for part in parts):
            category = "Wooden pins"
        else:
            return "Parts"
        rows = {self._row_of_part(part) for part in parts}
        rows.discard(None)
        if len(rows) == 1:
            return f"Row {next(iter(rows))} {category.lower()}"
        return category

    def _arm_fetch_confirmation(
            self, result: dict[str, object], ids: Iterable[int], friendly: str,
            *, matched_parts: Iterable[str],
            assembly_parts: Iterable[str] = ()) -> bool:
        """Highlight one graspable object and ask before moving the robot."""
        unique_ids = sorted({int(tool_id) for tool_id in ids})
        if len(unique_ids) != 1:
            self._pending_fetch = None
            if self._vlm is not None:
                self._vlm.set_pending_fetch(None)
            spoken = (
                "I need one specific fetchable object. Please name only one part "
                "or tool, including its row and side when necessary.")
            self._emit_reference_decision(
                result, "fetch_not_specific", [1.0, 0.72, 0.0, 0.2],
                spoken, warning=True)
            return False

        tool_id = unique_ids[0]
        if self.study4_condition is not None:
            # Study 4 measures correct object requests, not robot delivery.
            # Treat fetch-like wording ("get/give/bring") as a grounded
            # selection and never create confirmation or motion commands.
            self._pending_fetch = None
            if self._vlm is not None:
                self._vlm.set_pending_fetch(None)
            spoken = f"I identified and highlighted the {friendly}."
            self._emit_reference_decision(
                result, "object_request_grounded", [0.0, 1.0, 1.0, 0.2],
                spoken, ids=[tool_id], matched_parts=matched_parts,
                assembly_parts=assembly_parts)
            return True

        if tool_id not in self._fetchable_tool_ids:
            self._pending_fetch = None
            if self._vlm is not None:
                self._vlm.set_pending_fetch(None)
            spoken = (
                f"I highlighted the {friendly}, but it has no recorded robot "
                "grasp pose, so I cannot fetch it automatically.")
            self._emit_reference_decision(
                result, "fetch_pose_unavailable", [1.0, 0.72, 0.0, 0.2],
                spoken, ids=[tool_id], matched_parts=matched_parts,
                assembly_parts=assembly_parts, warning=True)
            return False

        previous_friendly = (
            str(self._pending_fetch.get("friendly", ""))
            if self._pending_fetch is not None else "")
        self._pending_fetch = {
            "tool_id": tool_id,
            "label": str(result.get("label", "")),
            "friendly": friendly,
            "matched_parts": list(matched_parts),
            "assembly_parts": list(assembly_parts),
        }
        if self._vlm is not None:
            self._vlm.set_pending_fetch(friendly)
        replacement = (
            f"I cancelled the pending request for the {previous_friendly}. "
            if previous_friendly and previous_friendly != friendly else "")
        spoken = (f"{replacement}I highlighted the {friendly}. "
                  "Do you want me to get it?")
        self._emit_reference_decision(
            result, "fetch_confirmation_pending", [0.0, 1.0, 1.0, 0.2],
            spoken, ids=[tool_id], matched_parts=matched_parts,
            assembly_parts=assembly_parts)
        return True

    def _handle_fetch_confirmation(self, result: dict[str, str]) -> None:
        """Accept or cancel the single pending voice-requested robot fetch."""
        pending = self._pending_fetch
        question = result.get("text", "")
        if pending is None:
            spoken = "There is no pending robot fetch to confirm."
            self._speak(spoken, warning=True)
            if self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "FETCH_CANCELLED", spoken)
            return

        confirmed = result.get("confirmation") == "yes"
        self._pending_fetch = None
        if self._vlm is not None:
            self._vlm.set_pending_fetch(None)
        friendly = str(pending["friendly"])
        if confirmed:
            self._send_select({
                "event": "fetch_confirmed",
                "tool_id": int(pending["tool_id"]),
                "label": str(pending["label"]),
                "matched_parts": list(pending.get("matched_parts", [])),
            })
            spoken = f"Okay. I asked the robot to get the {friendly}."
            display_label = "FETCH_CONFIRMED"
        else:
            self._send_reference_highlight(
                [], [0.0, 1.0, 1.0, 0.2], "fetch_cancelled",
                str(pending["label"]))
            spoken = f"Okay. I cancelled the fetch for the {friendly}."
            display_label = "FETCH_CANCELLED"
        self._speak(spoken, warning=not confirmed)
        if self._vlm is not None:
            self._vlm.apply_policy_response(question, display_label, spoken)

    def _part_step_priority(self, step: Step) -> tuple[int, int, str]:
        """Use the interface's deterministic ordering for a filtered step set."""
        row = step.row if step.row > 0 else 999
        coords = self.graph.control_coords_for(step.id)
        stage = coords[1] if coords is not None else 999
        return row, stage, step.id

    def _tool_is_used_by_step(self, label: str, step: Step) -> bool:
        """Return whether one fastening-tool label is required by ``step``."""
        coords = self.graph.control_coords_for(step.id)
        if coords is None or coords[1] not in (4, 6):
            return False
        required = set(gearbox_control.fastening_tool_labels(step.row))
        if label == "BIT_HOLDER1":
            return bool(required & {"H5_HEX_BIT", "H3_HEX_BIT"})
        if label == "BIT_HOLDER2":
            return "T25_TORX_BIT" in required
        return label in required

    def _tools_retained_from_completed_steps(self) -> dict[str, Step]:
        """Tools used by a completed fastening step remain with the operator."""
        retained: dict[str, Step] = {}
        for step_id in self.graph.completed:
            step = self.graph.by_id[step_id]
            coords = self.graph.control_coords_for(step.id)
            if coords is None or coords[1] not in (4, 6):
                continue
            for label in gearbox_control.fastening_tool_labels(step.row):
                retained[label] = step
        return retained

    def _parts_supplied_by_completed_steps(self) -> dict[str, Step]:
        """Return raw parts whose physical containers must already be supplied.

        When one item from a row kit has been used, the entire physical kit is
        treated as supplied because it is represented by one pegboard object.
        The same grouping rule applies to any future shared physical container.
        """
        used_parts: dict[str, Step] = {}
        for step_id in self.graph.completed:
            step = self.graph.by_id[step_id]
            for part in step.inputs:
                if part in PROVIDED_PARTS:
                    used_parts[part] = step

        supplied: dict[str, Step] = {}
        for used_part, step in used_parts.items():
            tool_id = gearbox_control.tool_id_for_graph_part(
                self._tool_index, used_part)
            if tool_id is None:
                continue
            # Expand a physical row-kit box back to every semantic item stored
            # inside it so the VLM does not offer a hidden kit item later.
            container_parts = [
                part for part in PROVIDED_PARTS
                if gearbox_control.tool_id_for_graph_part(
                    self._tool_index, part) == tool_id
            ]
            for part in container_parts or [used_part]:
                supplied.setdefault(part, step)
        return supplied

    def _sync_progress_inferred_supply_statuses(self) -> None:
        """Mirror parts/tools implied by completed work into handed-over state.

        Consumed raw parts and tools used by completed fastening stages must
        have been available to the operator. This records that fact in the same
        state table used for robot handovers, with ``source=task_progression``
        so it remains distinguishable and reversible.
        """
        inferred_supply = self._parts_supplied_by_completed_steps()
        inferred_supply.update(self._tools_retained_from_completed_steps())

        # Undo/reset must remove only inferred records. A real robot handover
        # remains valid independently of subsequent task-graph edits.
        for label, info in list(self._robot_part_states.items()):
            if (info.get("source") == "task_progression"
                    and label not in inferred_supply):
                self._robot_part_states.pop(label, None)
                self.log(
                    f"[ProgressSupply] Removed inferred handover for {label} "
                    "after task progression changed")

        for label, step in inferred_supply.items():
            existing = self._robot_part_states.get(label)
            # A physical robot grasp/handover is stronger evidence and must not
            # be overwritten by task progression.
            if existing is not None and existing.get("source") != "task_progression":
                continue
            if label in gearbox_control.FASTENING_TOOL_SPECS:
                ids = gearbox_control.tool_ids_for_reference(
                    self._tool_index, label)
            else:
                tool_id = gearbox_control.tool_id_for_graph_part(
                    self._tool_index, label)
                ids = [] if tool_id is None else [tool_id]
            layout_name = next(
                (name for name, tool_id in self._tool_index.items()
                 if ids and tool_id == ids[0]),
                label,
            )
            new_info = {
                "status": "handed_over",
                "tool_id": ids[0] if ids else None,
                "tool_name": layout_name,
                "requested": False,
                "source": "task_progression",
                "inferred_from_step": step.id,
            }
            if existing != new_info:
                self._robot_part_states[label] = new_info
                self.log(
                    f"[ProgressSupply] {label}: handed_over inferred from "
                    f"completed step {step.id}")

    def _publish_progress_handover_state(self) -> None:
        """Synchronize progression-inferred pegboard removals with Open3D/Unity."""
        inferred = {
            label: info
            for label, info in self._robot_part_states.items()
            if info.get("source") == "task_progression"
        }
        tool_ids = sorted({
            int(info["tool_id"])
            for info in inferred.values()
            if info.get("tool_id") is not None
        })
        self._send_select({
            "event": "progress_handover_sync",
            "tool_ids": tool_ids,
            "items": [
                {
                    "label": label,
                    "tool_id": info.get("tool_id"),
                    "inferred_from_step": info.get("inferred_from_step"),
                }
                for label, info in sorted(inferred.items())
            ],
        })

    def _handle_part_step_lookup(
            self, result: dict[str, object], label: str,
            parts: Iterable[str], *, is_tool: bool) -> None:
        """Select the next READY graph step that uses a named part or tool."""
        parts = list(parts)
        if is_tool:
            candidates = [
                step for step in self.graph.steps
                if step.id not in self.graph.completed
                and self._tool_is_used_by_step(label, step)
            ]
        else:
            current_objects = {
                current for part in parts
                if (current := self.graph.current_container(part)) is not None
            }
            candidates = [
                step for step in self.graph.steps
                if step.id not in self.graph.completed
                and any(item in (*step.inputs, *step.context)
                        for item in current_objects)
            ]

        # Preserve the interface preference inside this part-filtered set.
        ready = [step for step in candidates if self.graph.is_ready(step)]
        if ready and self.graph.last_worked_row is not None:
            same_row = [step for step in ready
                        if step.row == self.graph.last_worked_row]
            if same_row:
                ready = same_row

        question = str(result.get("text", ""))
        friendly = self._friendly_reference_label(label, parts)
        if ready:
            step = min(ready, key=self._part_step_priority)
            self.recommended_id = step.id
            self._select_step(step.id, announce=False)
            if self.controller is not None:
                self._animate_unity_callback()
            action = self.graph.friendly_step_action(step)
            spoken = (
                f"I selected the next ready step that uses the {friendly}. "
                f"{action.capitalize()}.")
            self.log(f"[Part step] Auto-selected [{step.id}] for {label}")
            self._speak(spoken)
            coords = self.graph.control_coords_for(step.id)
            highlight_ids = (
                gearbox_control.appearing_ids(
                    self._tool_index, coords[0], coords[1])
                if coords is not None and coords[0] > 0 else [])
            self._record_reference_decision(
                result, "part_step_selected", parts or [label], (),
                highlight_ids, (), spoken)
            if self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "PART_STEP_SELECTED", spoken)
            return

        if candidates:
            blocked = min(candidates, key=self._part_step_priority)
            missing = self._friendly_missing(self.graph.missing(blocked))
            spoken = (
                f"The next step that uses the {friendly} is not ready. "
                f"First, {missing}.")
        else:
            spoken = (
                f"There is no remaining assembly step that uses the {friendly}.")
        self._speak(spoken, warning=True)
        self._record_reference_decision(
            result, "part_step_unavailable", parts or [label], (), [], (),
            spoken)
        if self._vlm is not None:
            self._vlm.apply_policy_response(
                question, "PART_STEP_UNAVAILABLE", spoken)

    def _robot_supply_response(
            self, label: str, parts: Iterable[str]) -> tuple[str, str]:
        """Return a user-facing response for parts already carried by the robot."""
        parts = list(parts)
        infos = [self._robot_part_states[part] for part in parts]
        statuses = {str(info.get("status", "")) for info in infos}
        friendly = self._friendly_reference_label(label, parts)
        physical_names = {str(info.get("tool_name", "")) for info in infos}
        kit_name = next(iter(physical_names)) if len(physical_names) == 1 else ""
        kit_match = re.fullmatch(r"ROW([1-4])_KIT", kit_name)
        kit_suffix = (f" as part of the Row {kit_match.group(1)} kit"
                      if kit_match and label.upper() != kit_name else "")
        if statuses == {"handed_over"}:
            return (
                "already_handed_over",
                f"The {friendly} has already been delivered to you{kit_suffix}.",
            )
        return (
            "already_in_robot_gripper",
            f"The robot is already holding the {friendly}{kit_suffix}.",
        )

    def _handle_task_independent_reference(
            self, result: dict[str, object], label: str) -> None:
        """Render VLM-grounded references under the active study condition.

        C2 expands labels without task context. In C3, a generic category is
        intersected with the selected step before it is mapped to physical
        pegboard objects.
        """
        is_tool = label in gearbox_control.FASTENING_TOOL_SPECS
        parts = self.graph.parts_for_reference_label(label)
        if label == "AMBIGUOUS":
            candidate_labels = list(result.get("candidate_labels", []))
            had_explicit_candidates = bool(candidate_labels)
            candidates = self._candidate_parts_from_vlm(candidate_labels)
            ids = self._physical_ids_for_vlm_candidates(
                candidate_labels, candidates)
            if self.study4_condition == "task_aware" and self.selected_id:
                allowed_ids = set(self._selected_step_pegboard_ids())
                ids = [tool_id for tool_id in ids if tool_id in allowed_ids]
                candidates = [
                    part for part in candidates
                    if gearbox_control.tool_id_for_graph_part(
                        self._tool_index, part) in allowed_ids
                ]
                candidate_labels = [
                    candidate for candidate in candidate_labels
                    if set(self._physical_ids_for_vlm_candidates(
                        [candidate], self.graph.parts_for_reference_label(
                            candidate))) & allowed_ids
                ]
                result = {
                    **result,
                    "candidate_labels": candidate_labels,
                }
                if (str(result.get("reference_relation", "direct"))
                        == "alternative_to_previous"
                        and not had_explicit_candidates):
                    # "The other part" is defined over selectable physical
                    # objects. Derive the alternatives from the complete
                    # selected-step search space, even if Qwen omitted them.
                    if self._vlm_referent_id is not None:
                        ids = sorted(allowed_ids - {self._vlm_referent_id})
                    elif self._vlm_candidate_ids:
                        # After a plural response, "the other part" refers to
                        # one member of that recently highlighted remainder.
                        ids = sorted(
                            allowed_ids & set(self._vlm_candidate_ids))
                    else:
                        ids = sorted(allowed_ids)
                    candidate_labels = [
                        name for name, tool_id in self._tool_index.items()
                        if int(tool_id) in ids
                    ]
                    candidates = self._candidate_parts_from_vlm(
                        candidate_labels)
                    result = {
                        **result,
                        "candidate_labels": candidate_labels,
                    }
            text = str(result.get("text", ""))
            if len(ids) == 1:
                # Several semantic contents can occupy one selectable physical
                # container (for example the Row 1 crank and pin are both in
                # ROW1_KIT). That is not a physical acquisition ambiguity.
                physical_label = next(
                    (name for name, tool_id in self._tool_index.items()
                     if int(tool_id) == ids[0]),
                    "physical object",
                )
                friendly = self._friendly_reference_label(
                    physical_label, candidates)
                spoken = f"I identified and highlighted the {friendly}."
                decision = "language_colocated_grounded"
                warning = False
                result = {
                    **result,
                    "label": physical_label,
                    "candidate_labels": [physical_label],
                }
            elif (len(ids) > 1 and candidates
                    and self._is_explicit_plural_reference(text)):
                category = self._friendly_candidate_category(candidates).lower()
                spoken = f"I identified and highlighted the {category}."
                decision = "language_plural_grounded"
                warning = False
            elif (len(ids) > 1
                    and str(result.get("reference_relation", "direct"))
                    == "alternative_to_previous"):
                choices = []
                for candidate_label in candidate_labels:
                    candidate_parts = self.graph.parts_for_layout_object(
                        candidate_label)
                    friendly = self._friendly_reference_label(
                        candidate_label, candidate_parts)
                    if friendly not in choices:
                        choices.append(friendly)
                spoken = (
                    "Which other object do you mean: "
                    + ", or ".join(choices) + "?")
                decision = "language_physical_alternative_ambiguous"
                warning = True
            else:
                spoken = self._candidate_ambiguity_question(text, candidates)
                decision = "language_ambiguous"
                warning = True
            if (self._vlm is not None and len(ids) > 1
                    and candidate_labels):
                self._vlm.set_resolved_part_candidates(candidate_labels)
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2],
                spoken, ids=ids, matched_parts=candidates, warning=warning)
            return
        if label in {"INVALID_OUTPUT", "STEP_ITEMS"}:
            spoken = (self._ambiguity_question(result.get("text", ""))
                      if label == "AMBIGUOUS" else
                      "Please name one specific physical part or tool.")
            self._emit_reference_decision(
                result, "language_unresolved", [1.0, 0.72, 0.0, 0.2],
                spoken, matched_parts=parts, warning=True)
            return
        if (self.study4_condition == "task_aware"
                and self.selected_id and parts and not is_tool):
            step = self.graph.by_id[self.selected_id]
            step_parts = set((*step.inputs, *step.context))
            relevant = [
                part for part in parts
                if (part in step_parts
                    or self.graph.current_container(part) in step_parts)
            ]
            if relevant:
                parts = relevant
        ids = (gearbox_control.tool_ids_for_reference(self._tool_index, label)
               if is_tool else [
                   tool_id for tool_id in (
                       gearbox_control.tool_id_for_graph_part(self._tool_index, part)
                       for part in parts)
                   if tool_id is not None
               ])
        ids = list(dict.fromkeys(ids))
        referenced_ids = list(ids)
        if self.study4_condition == "task_aware" and self.selected_id:
            allowed_ids = set(self._selected_step_pegboard_ids())
            ids = [tool_id for tool_id in ids if tool_id in allowed_ids]
            if not ids:
                step = self.graph.by_id[self.selected_id]
                already_acquired = [
                    tool_id for tool_id in referenced_ids
                    if tool_id in self._acquired_tool_ids
                ]
                if already_acquired:
                    acquired_names = [
                        self._friendly_reference_label(
                            physical_label,
                            self.graph.parts_for_layout_object(physical_label))
                        for tool_id in already_acquired
                        for physical_label in [next(
                            (name for name, indexed_id in self._tool_index.items()
                             if int(indexed_id) == tool_id),
                            f"object {tool_id}")]
                    ]
                    remaining_ids = self._selected_step_pegboard_ids()
                    remaining_names = [
                        self._friendly_reference_label(
                            physical_label,
                            self.graph.parts_for_layout_object(physical_label))
                        for tool_id in remaining_ids
                        for physical_label in [next(
                            (name for name, indexed_id in self._tool_index.items()
                             if int(indexed_id) == tool_id),
                            f"object {tool_id}")]
                    ]
                    acquired_text = " and ".join(acquired_names)
                    requested_text = self._friendly_reference_label(
                        label, parts or [label])
                    acknowledgement = (
                        f"{requested_text.capitalize()} is already covered by "
                        f"{acquired_text}, which you selected. ")
                    if remaining_names:
                        remaining_text = " and ".join(remaining_names)
                        spoken = (
                            acknowledgement
                            +
                            f"The remaining object for this step is "
                            f"{remaining_text}.")
                    else:
                        spoken = (
                            acknowledgement + "There are "
                            "no remaining objects for this step.")
                    self._emit_reference_decision(
                        result, "selected_step_object_already_acquired",
                        [0.1, 1.0, 0.1, 0.25], spoken,
                        ids=[], matched_parts=parts or [label])
                    return
                spoken = (
                    "That object is not required for the selected step. "
                    f"This step is to {self.graph.friendly_step_action(step)}.")
                self._emit_reference_decision(
                    result, "outside_selected_step_search_space",
                    [1.0, 0.0, 0.0, 0.25], spoken,
                    ids=[], matched_parts=parts, warning=True)
                return
        friendly = self._friendly_reference_label(label, parts)
        # Study 4 has no robot and therefore no fetch/confirmation state. A
        # request phrased as "get/fetch" is treated as an object-grounding
        # request; the participant completes it by clicking the object.
        spoken = f"I identified and highlighted the {friendly}."
        self._emit_reference_decision(
            result, "language_grounded", [0.0, 1.0, 1.0, 0.2], spoken,
            ids=ids, matched_parts=parts or [label])

    def _handle_selected_step_object_cycle(
            self, result: dict[str, object], *, recent_set_only: bool = False,
            highlight_all: bool = False
            ) -> None:
        """Advance Condition 3 to the next required physical pegboard object."""
        ids = list(dict.fromkeys(self._selected_step_pegboard_ids()))
        if recent_set_only and self._vlm_candidate_ids:
            recent_ids = set(self._vlm_candidate_ids)
            ids = [tool_id for tool_id in ids if tool_id in recent_ids]
        if not ids:
            self._emit_reference_decision(
                result, "step_cycle_empty", [1.0, 0.72, 0.0, 0.2],
                "There are no remaining pegboard objects for this step.",
                warning=True)
            return
        if highlight_all:
            physical_labels = [
                name for name, tool_id in self._tool_index.items()
                if int(tool_id) in ids
            ]
            resolved = {
                **result,
                "label": "AMBIGUOUS",
                "candidate_labels": physical_labels,
            }
            self._emit_reference_decision(
                resolved, "selected_step_full_search_space",
                [0.0, 1.0, 1.0, 0.2],
                "I highlighted all of the objects required for this step.",
                ids=ids,
                matched_parts=[
                    part
                    for physical_label in physical_labels
                    for part in self.graph.parts_for_layout_object(
                        physical_label)
                ])
            return
        if self._vlm_referent_id in ids:
            next_index = (ids.index(self._vlm_referent_id) + 1) % len(ids)
        else:
            next_index = 0
        next_id = ids[next_index]
        physical_label = next(
            (name for name, tool_id in self._tool_index.items()
             if int(tool_id) == next_id),
            f"object {next_id}",
        )
        parts = self.graph.parts_for_layout_object(physical_label)
        friendly = self._friendly_reference_label(physical_label, parts)
        spoken = f"I identified and highlighted the {friendly}."
        resolved = {
            **result,
            "label": physical_label,
            "candidate_labels": [physical_label],
        }
        self._emit_reference_decision(
            resolved, "selected_step_physical_cycle",
            [0.0, 1.0, 1.0, 0.2], spoken,
            ids=[next_id], matched_parts=parts or [physical_label])

    def _handle_part_reference(self, result: dict[str, object]) -> None:
        """Apply deterministic task-state policy to one VLM-resolved label."""
        label = str(result.get("label", "INVALID_OUTPUT"))
        if (label == "RECOMMEND_NEXT_STEP"
                and self.study4_condition == "task_aware"):
            self._activate_recommended_step(str(result.get("text", "")))
            return
        if (self.selected_id and not self._trial_recording
                and not self._acquisition_complete):
            self._emit_reference_decision(
                result, "trial_not_started", [1.0, 0.72, 0.0, 0.2],
                "Press Enter when you are ready to start this trial.",
                warning=True)
            return
        if (label == "STEP_ITEMS"
                and self.study4_condition == "task_aware"
                and not self.selected_id):
            self._emit_reference_decision(
                result, "no_selected_step", [0.0, 1.0, 1.0, 0.2],
                "Please select an assembly step first.", warning=True)
            return
        if (self.study4_condition == "task_aware"
                and result.get("origin_intent") == "step_items_request"
                and str(result.get("reference_relation", "direct"))
                != "alternative_to_previous"):
            # The assistant performs a second, model-based candidate pass for
            # evaluation, whose output can legitimately be AMBIGUOUS. The
            # user's validated needs request—not that predicted label—is the
            # Condition-3 unlock signal. Use graph-backed physical IDs so an
            # invented or non-physical candidate cannot erase the cyan cue.
            ids = self._selected_step_pegboard_ids()
            physical_labels = [
                name for name, tool_id in self._tool_index.items()
                if int(tool_id) in ids
            ]
            # reference_highlight intentionally renders only one resolved
            # referent, so a multi-object needs response must first populate
            # the persistent cyan step-context layer used by both views.
            self._publish_selected_step_pegboard_highlight(announce=True)
            self._emit_reference_decision(
                result, "selected_step_needs_highlight",
                [0.0, 1.0, 1.0, 0.2],
                "I identified and highlighted the parts.",
                ids=ids,
                matched_parts=[
                    part
                    for physical_label in physical_labels
                    for part in self.graph.parts_for_layout_object(
                        physical_label)
                ],
            )
            return
        if (self.study4_condition == "task_aware"
                and label == "STEP_ITEMS"
                and str(result.get("reference_relation", "direct"))
                == "alternative_to_previous"):
            self._handle_selected_step_object_cycle(result)
            return
        if (self.study4_condition == "task_aware"
                and label == "AMBIGUOUS"
                and str(result.get("reference_relation", "direct"))
                == "alternative_to_previous"
                and not result.get("candidate_labels")):
            self._handle_selected_step_object_cycle(result)
            return
        if (self.study4_condition == "task_aware"
                and label == "AMBIGUOUS"
                and str(result.get("reference_relation", "direct"))
                == "member_of_recent_set"):
            request_text = str(result.get("text", "")).casefold()
            highlight_all = bool(re.search(
                r"\b(all|everything|every one)\b", request_text))
            self._handle_selected_step_object_cycle(
                result, recent_set_only=not highlight_all,
                highlight_all=highlight_all)
            return
        if (self.study4_condition == "task_aware"
                and label == "STEP_ITEMS"):
            # In Condition 3 this explicit needs question is what unlocks the
            # cyan task-context cue. Selecting (or recommending) a step alone
            # must not reveal its required pegboard objects.
            if str(result.get("step_scope", "selected_step")) == "selected_step":
                self._publish_selected_step_pegboard_highlight(announce=True)
            item_scope = str(result.get("item_scope", "all"))
            if item_scope in {"parts", "all"}:
                self._handle_step_parts_request(result)
            if item_scope in {"tools", "all"}:
                self._handle_step_tools_request(result)
            return
        if self.study4_condition in {"language", "task_aware"}:
            self._handle_task_independent_reference(result, label)
            return
        if label == "STEP_ITEMS":
            item_scope = str(result.get("item_scope", "all"))
            if item_scope in {"parts", "all"}:
                self._handle_step_parts_request(result)
            if item_scope in {"tools", "all"}:
                self._handle_step_tools_request(result)
            return
        if label == "AMBIGUOUS":
            candidate_labels = result.get("candidate_labels", [])
            candidates = self._candidate_parts_from_vlm(candidate_labels)
            category = self._friendly_candidate_category(candidates)
            # Backward compatibility for explicit callers of
            # submit_part_reference(), which return only AMBIGUOUS. Normal
            # voice/text questions now receive candidates directly from Qwen.
            if not candidates:
                category, candidates = self._ambiguous_part_category(
                    result.get("text", ""))
            if (result.get("part_action") != "find_step"
                    and self.selected_id and candidates):
                step = self.graph.by_id[self.selected_id]
                step_parts = set((*step.inputs, *step.context))
                relevant = [
                    part for part in candidates
                    if (part in step_parts
                        or self.graph.current_container(part) in step_parts)
                ]
                if len(relevant) == 1:
                    # The selected step supplies enough context to resolve an
                    # otherwise generic expression, such as "the stand" when
                    # only its right stand is an input.
                    result = dict(result)
                    result["label"] = relevant[0]
                    result["candidate_labels"] = [relevant[0]]
                    label = relevant[0]
                elif relevant:
                    # Keep the clarification grounded in the selected step.
                    # The original ontology candidate (for example BEARING)
                    # may expand across every row, but only these instances are
                    # plausible antecedents of "one of them" here.
                    candidates = relevant
                    result = dict(result)
                    result["candidate_labels"] = list(relevant)
                elif not relevant:
                    ids = [
                        gearbox_control.tool_id_for_graph_part(self._tool_index, part)
                        for part in candidates if part in self.graph.active_parts
                    ]
                    ids = [tool_id for tool_id in ids if tool_id is not None]
                    spoken = (
                        f"{category} are not relevant to this step. "
                        f"This step is to {self.graph.friendly_step_action(step)}.")
                    self._emit_reference_decision(
                        result, "not_relevant_ambiguous",
                        [1.0, 0.0, 0.0, 0.25], spoken,
                        ids=ids, matched_parts=candidates, warning=True)
                    return
            if label == "AMBIGUOUS":
                ids = self._physical_ids_for_vlm_candidates(
                    result.get("candidate_labels", []), candidates)
                if 1 < len(candidates) <= 6:
                    choices = []
                    for part in candidates:
                        friendly = self.graph.friendly_part(part)
                        if friendly.lower().startswith("the "):
                            friendly = friendly[4:]
                        if friendly not in choices:
                            choices.append(friendly)
                    if len(choices) == 2:
                        choice_text = f"{choices[0]} or {choices[1]}"
                    else:
                        choice_text = (", ".join(choices[:-1])
                                       + f", or {choices[-1]}")
                    spoken = f"Which one do you mean: {choice_text}?"
                else:
                    spoken = self._ambiguity_question(result.get("text", ""))
                self._emit_reference_decision(
                    result, "ambiguous", [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=ids, matched_parts=candidates, warning=True)
                return

        is_tool = label in gearbox_control.FASTENING_TOOL_SPECS
        is_assembly = self.graph.producer_for(label) is not None
        parts = self.graph.parts_for_reference_label(label)
        if is_assembly:
            self._handle_assembly_reference(result, label)
            return
        if label == "INVALID_OUTPUT" or (not parts and not is_tool):
            spoken = ("I could not identify one specific part or tool. Please mention "
                      "its row, side, color, shape, or driver marking.")
            self._emit_reference_decision(
                result, "unresolved", [0.0, 1.0, 1.0, 0.2], spoken,
                warning=True)
            return

        if result.get("part_action") == "find_step":
            self._handle_part_step_lookup(
                result, label, parts, is_tool=is_tool)
            return

        # A fully supplied specific/grouped reference has an answer even when
        # no assembly step is selected. Do not ask the user to select a step or
        # offer to fetch the same physical kit again.
        if (parts and not is_tool
                and all(part in self._robot_part_states for part in parts)):
            decision, spoken = self._robot_supply_response(label, parts)
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=parts)
            return

        if not self.selected_id:
            spoken = ("Please select an assembly step first, so I can check whether "
                      "that part or tool belongs to the current task.")
            self._emit_reference_decision(
                result, "no_selected_step", [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=parts, warning=True)
            return

        step = self.graph.by_id[self.selected_id]
        if is_tool:
            coords = self.graph.control_coords_for(step.id)
            required = set(gearbox_control.fastening_tool_labels(step.row))
            # Each holder is relevant only when the row needs an insert stored
            # in that holder: H5/H3 in Holder 1 and T25 in Holder 2.
            holder1_needed = bool(required & {"H5_HEX_BIT", "H3_HEX_BIT"})
            holder2_needed = "T25_TORX_BIT" in required
            relevant = (coords is not None and coords[1] in (4, 6)
                        and (label in required
                             or (label == "BIT_HOLDER1" and holder1_needed)
                             or (label == "BIT_HOLDER2" and holder2_needed)))
            ids = gearbox_control.tool_ids_for_reference(self._tool_index, label)
            spoken_tool = self._friendly_reference_label(label, [])
            supplied_info = self._robot_part_states.get(label)
            if supplied_info is not None:
                if supplied_info.get("source") == "task_progression":
                    retained_step = self.graph.by_id.get(
                        str(supplied_info.get("inferred_from_step", "")))
                    prior_action = (
                        self.graph.friendly_step_action(retained_step)
                        if retained_step is not None else "complete earlier work")
                    spoken = (
                        f"You should already have the {spoken_tool}. It was "
                        f"used earlier to {prior_action} and should still be "
                        "with you.")
                    decision = "tool_retained_by_operator"
                elif supplied_info.get("status") == "in_robot_gripper":
                    spoken = f"The robot is already holding the {spoken_tool}."
                    decision = "tool_in_robot_gripper"
                else:
                    spoken = f"The {spoken_tool} has already been delivered to you."
                    decision = "tool_handed_over"
                self._emit_reference_decision(
                    result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=[], matched_parts=[label])
                return
            retained_step = self._tools_retained_from_completed_steps().get(label)
            if retained_step is not None:
                spoken = (
                    f"The {spoken_tool} was already used to "
                    f"{self.graph.friendly_step_action(retained_step)}, so it "
                    "should still be with you for this step.")
                self._emit_reference_decision(
                    result, "tool_retained_by_operator",
                    [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=[], matched_parts=[label])
                return
            if relevant:
                if result.get("part_action") == "fetch":
                    if self.graph.state(step) != "ready":
                        spoken = (
                            f"The {spoken_tool} belongs to this step, but this step "
                            "is not ready, so I will not fetch it yet.")
                        self._emit_reference_decision(
                            result, "fetch_blocked_step",
                            [1.0, 0.72, 0.0, 0.2], spoken,
                            ids=[], matched_parts=[label], warning=True)
                        return
                    self._arm_fetch_confirmation(
                        result, ids, spoken_tool, matched_parts=[label])
                    return
                if result.get("part_action") == "status":
                    spoken = (f"No. The {spoken_tool} has not been delivered yet. "
                              f"It is needed to "
                              f"{self.graph.friendly_step_action(step)}. "
                              "I have highlighted it.")
                else:
                    spoken = (f"The {spoken_tool} is needed to "
                              f"{self.graph.friendly_step_action(step)}. "
                              "I have highlighted it, but it has not been "
                              "delivered yet.")
                self._emit_reference_decision(
                    result, "relevant_tool", [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=ids, matched_parts=[label])
            else:
                required_text = self._fastening_tool_guidance(step)
                suffix = (f" Use {required_text}." if required_text else
                          " This selected step does not require a fastening tool.")
                spoken = f"The {spoken_tool} is not relevant to this step.{suffix}"
                self._emit_reference_decision(
                    result, "not_relevant_tool", [1.0, 0.0, 0.0, 0.25], spoken,
                    ids=ids, matched_parts=[label], warning=True)
            return

        step_parts = set((*step.inputs, *step.context))
        # Grouped labels such as BEARING and PIN are disambiguated by the
        # selected row before considering other rows.
        same_row = [part for part in parts
                    if step.row and self._row_of_part(part) == step.row]
        candidates = same_row or parts
        # Row context may turn a broad label such as BEARING into a fully
        # supplied row-specific set. Delivery state takes precedence over the
        # selected-step relevance warning.
        if (candidates
                and all(part in self._robot_part_states for part in candidates)):
            decision, spoken = self._robot_supply_response(label, candidates)
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=candidates)
            return
        active_relevant = [part for part in candidates
                           if part in self.graph.active_parts
                           and part in step_parts
                           and part not in self._robot_part_states]
        robot_supplied_relevant = [
            part for part in candidates
            if part in step_parts and part in self._robot_part_states
        ]
        consumed = [(part, self.graph.current_container(part))
                    for part in candidates
                    if part not in self.graph.active_parts
                    and self.graph.current_container(part) is not None]
        consumed_relevant = [(part, current) for part, current in consumed
                             if current in step_parts]
        spoken_part = self._friendly_reference_label(label, candidates)

        if robot_supplied_relevant and not active_relevant:
            states = {
                str(self._robot_part_states[part].get("status", ""))
                for part in robot_supplied_relevant
            }
            friendly_parts = [self.graph.friendly_part(part)
                              for part in robot_supplied_relevant]
            supplied_text = (friendly_parts[0] if len(friendly_parts) == 1
                             else ", and ".join(friendly_parts))
            if "in_robot_gripper" in states:
                spoken = f"The robot is already holding {supplied_text}."
                decision = "already_in_robot_gripper"
            else:
                spoken = f"{supplied_text.capitalize()} has already been handed to you."
                decision = "already_handed_over"
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=robot_supplied_relevant)
            return

        if consumed_relevant or (consumed and not active_relevant):
            consumed_source = consumed_relevant or consumed
            locations = list(dict.fromkeys(
                current for _part, current in consumed_source
                if current is not None))
            friendly_locations = [self.graph.friendly_part(item)
                                  for item in locations]
            location_text = (friendly_locations[0] if len(friendly_locations) == 1
                             else ", and ".join(friendly_locations))
            assembly_parts = [part for part, _current in consumed_source]
            spoken = (f"That {spoken_part} has already been used. It is now part of "
                      f"{location_text}. I have highlighted its location on the gearbox.")
            self._emit_reference_decision(
                result, "already_used", [1.0, 0.92, 0.02, 1.0], spoken,
                matched_parts=assembly_parts, current_assemblies=locations,
                assembly_parts=assembly_parts)
            return

        if active_relevant:
            ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
                   for part in active_relevant]
            ids = [tool_id for tool_id in ids if tool_id is not None]
            if result.get("part_action") == "fetch":
                if self.graph.state(step) != "ready":
                    spoken = (
                        f"The {spoken_part} belongs to this step, but this step "
                        "is not ready, so I will not fetch it yet.")
                    self._emit_reference_decision(
                        result, "fetch_blocked_step",
                        [1.0, 0.72, 0.0, 0.2], spoken,
                        ids=[], matched_parts=active_relevant,
                        warning=True)
                    return
                self._arm_fetch_confirmation(
                    result, ids, spoken_part,
                    matched_parts=active_relevant,
                    assembly_parts=active_relevant)
                return
            if self.graph.state(step) == "blocked":
                missing = self._friendly_missing(self.graph.missing(step))
                spoken = (f"The {spoken_part} belongs to this task, and I have highlighted "
                          f"its storage location. Do not start yet; first complete or prepare "
                          f"{missing}.")
                warning = True
            else:
                spoken = (f"Yes. The {spoken_part} is needed to "
                          f"{self.graph.friendly_step_action(step)}. "
                          "I have highlighted its "
                          "storage location.")
                warning = False
            self._emit_reference_decision(
                result, "relevant", [0.0, 1.0, 1.0, 0.2], spoken,
                ids=ids, matched_parts=active_relevant,
                # Mirror the same cyan semantic highlight onto the component
                # in the assembled BoardAR gearbox, not only its pegboard
                # storage object or row-kit box.
                assembly_parts=active_relevant, warning=warning)
            return

        ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
               for part in candidates if part in self.graph.active_parts]
        ids = [tool_id for tool_id in ids if tool_id is not None]
        spoken = (f"The {spoken_part} is not relevant to this step. "
                  f"This step is to {self.graph.friendly_step_action(step)}.")
        self._emit_reference_decision(
            result, "not_relevant", [1.0, 0.0, 0.0, 0.25], spoken,
            ids=ids, matched_parts=candidates, warning=True)

    def _handle_assembly_reference(
            self, result: dict[str, object], label: str) -> None:
        """Explain where a graph subassembly is without treating it as fetchable.

        Subassemblies are semantic task-graph objects rather than independent
        pegboard objects. Their visible highlight is therefore formed from the
        original physical components currently contained in that assembly.
        """
        producer = self.graph.producer_for(label)
        current = self.graph.current_container(label)
        if label in self.graph.active_parts:
            current = label

        highlight_container = current or label
        components = sorted(
            part for part in PROVIDED_PARTS
            if self.graph.current_container(part) == highlight_container
        )
        friendly = self.graph.friendly_part(label)
        current_friendly = (
            self.graph.friendly_part(current) if current is not None else "")
        selected = (
            self.graph.by_id.get(self.selected_id) if self.selected_id else None)
        selected_items = (
            set((*selected.inputs, *selected.context)) if selected else set())
        relevant = bool(
            selected and (label in selected_items or current in selected_items))

        if current == label:
            bearing_stand = re.fullmatch(
                r"BEARING_STAND_ROW([1-4])_(LEFT|RIGHT)_ASSEMBLY", label)
            if bearing_stand:
                row, side = bearing_stand.groups()
                spoken = (
                    f"Yes. The Row {row} {side.lower()} stand was already "
                    f"supplied. You fitted its bearing, so it is now {friendly}.")
            elif producer is not None:
                made_by = self.graph.friendly_step_action(producer)
                spoken = f"You already created {friendly} when you {made_by}."
            else:
                spoken = f"{friendly.capitalize()} is already assembled."
            if relevant and selected is not None:
                spoken += (
                    f" It is ready for this step: "
                    f"{self.graph.friendly_step_action(selected)}.")
                decision = "active_subassembly_relevant"
                color = [0.0, 1.0, 1.0, 0.2]
                warning = False
            elif selected is not None:
                spoken += (
                    f" It is not needed for this step. This step is to "
                    f"{self.graph.friendly_step_action(selected)}.")
                decision = "active_subassembly_not_relevant"
                color = [1.0, 0.0, 0.0, 0.25]
                warning = True
            else:
                decision = "active_subassembly"
                color = [1.0, 0.92, 0.02, 1.0]
                warning = False
            if result.get("part_action") == "fetch":
                spoken += " It is on the gearbox, so the robot does not need to fetch it."
            self._emit_reference_decision(
                result, decision, color, spoken,
                matched_parts=[label], current_assemblies=[label],
                assembly_parts=components, warning=warning)
            return

        if current is not None:
            spoken = (
                f"{friendly.capitalize()} has already been used. It is now "
                f"part of {current_friendly}. I have highlighted it on the gearbox.")
            self._emit_reference_decision(
                result, "subassembly_consumed", [1.0, 0.92, 0.02, 1.0],
                spoken, matched_parts=[label], current_assemblies=[current],
                assembly_parts=components)
            return

        if producer is None:
            spoken = "I could not locate that subassembly in the task graph."
            decision = "subassembly_unknown"
        elif self.graph.state(producer) == "ready":
            spoken = (
                f"{friendly.capitalize()} has not been assembled yet. You can "
                f"make it now: {self.graph.friendly_step_action(producer)}.")
            decision = "subassembly_not_created_ready"
        else:
            missing = self._friendly_missing(self.graph.missing(producer))
            spoken = (
                f"{friendly.capitalize()} has not been assembled yet. First, "
                f"{missing}.")
            decision = "subassembly_not_created_blocked"
        self._emit_reference_decision(
            result, decision, [1.0, 0.72, 0.0, 0.2], spoken,
            matched_parts=[label], warning=True)

    def _handle_step_tools_request(self, result: dict[str, object]) -> None:
        """Report/fetch the fastening tools required by a selected or next step."""
        scope = str(result.get("step_scope", ""))
        if scope == "selected_step":
            if not self.selected_id:
                self._emit_reference_decision(
                    result, "no_selected_step", [0.0, 1.0, 1.0, 0.2],
                    "Please select an assembly step first.", warning=True)
                return
            step = self.graph.by_id[self.selected_id]
        elif scope == "next_step":
            step = self.graph.recommend_next_step()
            if step is None:
                self._emit_reference_decision(
                    result, "no_next_step", [0.0, 1.0, 1.0, 0.2],
                    "There is no ready next step right now.", warning=True)
                return
        else:
            self._emit_reference_decision(
                result, "invalid_step_scope", [0.0, 1.0, 1.0, 0.2],
                "Please say whether you mean this step or the next step.",
                warning=True)
            return

        coords = self.graph.control_coords_for(step.id)
        required = list(
            gearbox_control.fastening_tool_labels(step.row)
            if coords is not None and coords[1] in (4, 6) else ())
        if not required:
            self._emit_reference_decision(
                result, "no_step_tools", [0.0, 1.0, 1.0, 0.2],
                "This step does not require a fastening tool.")
            return

        retained = self._tools_retained_from_completed_steps()
        supplied = [label for label in required
                    if label in self._robot_part_states or label in retained]
        outstanding = [label for label in required
                       if label not in self._robot_part_states
                       and label not in retained]
        friendly = [self._friendly_reference_label(label, [])
                    for label in required]
        friendly_text = (friendly[0] if len(friendly) == 1 else
                         " and ".join(friendly))
        outstanding_ids = sorted({
            tool_id
            for label in outstanding
            for tool_id in gearbox_control.tool_ids_for_reference(
                self._tool_index, label)
        })
        action = str(result.get("part_action", "reference"))

        # Make pronoun follow-ups refer to the tools just described, rather
        # than an older set of assembly parts.  This is especially important
        # after "Which tools do I need?" followed by "get one of them".
        if self._vlm is not None:
            self._vlm.set_resolved_part_candidates(outstanding or required)

        if not outstanding:
            directly_delivered = all(
                label in self._robot_part_states
                and self._robot_part_states[label].get("source")
                != "task_progression"
                for label in required)
            if directly_delivered:
                spoken = (f"Yes. The {friendly_text} "
                          f"{'has' if len(required) == 1 else 'have'} already "
                          "been delivered to you.")
                decision = "step_tools_already_delivered"
            else:
                prior_steps = list(dict.fromkeys(
                    self.graph.friendly_step_action(retained[label])
                    for label in required if label in retained))
                prior_text = " and ".join(prior_steps)
                spoken = (f"You should already have the {friendly_text}. "
                          f"{'It was' if len(required) == 1 else 'They were'} "
                          f"used earlier to {prior_text} and should still be "
                          "with you.")
                decision = "step_tools_retained_by_operator"
            self._emit_reference_decision(
                result, decision,
                [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=required)
            return

        outstanding_friendly = [self._friendly_reference_label(label, [])
                                for label in outstanding]
        outstanding_text = (outstanding_friendly[0]
                            if len(outstanding_friendly) == 1 else
                            " and ".join(outstanding_friendly))
        if action == "fetch":
            if self.graph.state(step) != "ready":
                spoken = "This step is not ready, so I will not fetch its tools yet."
                self._emit_reference_decision(
                    result, "fetch_blocked_step_tools",
                    [1.0, 0.72, 0.0, 0.2], spoken,
                    ids=[], matched_parts=outstanding,
                    warning=True)
                return
            # A fastening operation can require a driver plus an insert. The
            # robot fetches one physical object per confirmation, so offer the
            # primary driver first instead of asking the user to name it again.
            primary_label = outstanding[0]
            primary_ids = gearbox_control.tool_ids_for_reference(
                self._tool_index, primary_label)
            if len(primary_ids) == 1:
                primary_friendly = self._friendly_reference_label(
                    primary_label, [])
                if len(outstanding) > 1:
                    insert_text = " and ".join(
                        self._friendly_reference_label(label, [])
                        for label in outstanding[1:])
                    primary_friendly += f", used with the {insert_text}"
                self._arm_fetch_confirmation(
                    result, primary_ids, primary_friendly,
                    matched_parts=[primary_label])
                return
            spoken = (f"This step still needs the {outstanding_text}. "
                      "Please name the one you want the robot to get first.")
            self._emit_reference_decision(
                result, "fetch_step_tools_not_specific",
                [0.0, 1.0, 1.0, 0.2], spoken,
                ids=outstanding_ids, matched_parts=outstanding)
            return

        if supplied:
            supplied_text = " and ".join(
                self._friendly_reference_label(label, []) for label in supplied)
            spoken = (f"You should already have the {supplied_text}. "
                      f"You still need the {outstanding_text}. "
                      "I have highlighted it.")
        elif action == "status":
            spoken = (f"No. This step requires the {friendly_text}, and "
                      f"{'it has' if len(required) == 1 else 'they have'} not "
                      "been delivered yet. I have highlighted them.")
        else:
            spoken = (f"This step requires the {friendly_text}. "
                      "They have not been delivered yet, so I have highlighted them.")
        self._emit_reference_decision(
            result, "step_tools_outstanding", [0.0, 1.0, 1.0, 0.2],
            spoken, ids=outstanding_ids, matched_parts=required)

    def _handle_step_parts_request(self, result: dict[str, object]) -> None:
        """Resolve task-relative phrases such as "parts for the next step"."""
        scope = str(result.get("step_scope", ""))
        if scope == "selected_step":
            if not self.selected_id:
                self._emit_reference_decision(
                    result, "no_selected_step", [0.0, 1.0, 1.0, 0.2],
                    "Please select an assembly step first.", warning=True)
                return
            step = self.graph.by_id[self.selected_id]
            scope_text = "This step"
        elif scope == "next_step":
            # Never skip over the user's current work. "Next" advances only
            # after the selected step has actually been marked complete.
            if self.selected_id:
                selected = self.graph.by_id[self.selected_id]
                if self.graph.state(selected) != "complete":
                    self._emit_reference_decision(
                        result,
                        "current_step_incomplete",
                        [1.0, 0.72, 0.0, 0.2],
                        ("The current step is not complete yet. Finish it and "
                         "mark it complete before asking for the next step."),
                        warning=True,
                    )
                    return
            step = self.graph.recommend_next_step()
            if step is None:
                self._emit_reference_decision(
                    result, "no_next_step", [0.0, 1.0, 1.0, 0.2],
                    "There is no other ready assembly step right now.",
                    warning=True)
                return
            scope_text = "The next step"
        else:
            self._emit_reference_decision(
                result, "invalid_step_scope", [0.0, 1.0, 1.0, 0.2],
                "Please say whether you mean this step or the next step.",
                warning=True)
            return

        excluded_parts: set[str] = set()
        for label in result.get("exclude_labels", []):
            excluded_parts.update(self.graph.parts_for_reference_label(str(label)))
        supplied_inputs = [part for part in step.inputs
                           if part in self._robot_part_states
                           and part not in excluded_parts]
        active_inputs = [part for part in step.inputs
                         if part in self.graph.active_parts
                         and part not in excluded_parts
                         and part not in self._robot_part_states]
        if not active_inputs:
            outstanding_all = [
                part for part in step.inputs
                if part in self.graph.active_parts
                and part not in self._robot_part_states
            ]
            if outstanding_all:
                outstanding_text = ", and ".join(
                    self.graph.friendly_part(part) for part in outstanding_all)
                self._emit_reference_decision(
                    result,
                    "excluded_outstanding_step_parts",
                    [1.0, 0.72, 0.0, 0.2],
                    (f"You still need {outstanding_text}. The requested relation "
                     "excluded that outstanding part, so I will not report all "
                     "required parts as supplied."),
                    matched_parts=outstanding_all,
                    warning=True,
                )
                return
            if supplied_inputs:
                descriptions = [self.graph.friendly_part(part)
                                for part in supplied_inputs]
                supplied_text = (descriptions[0] if len(descriptions) == 1
                                 else ", and ".join(descriptions))
                self._emit_reference_decision(
                    result,
                    ("no_other_next_step_parts" if scope == "next_step"
                     else "no_other_selected_step_parts"),
                    [0.0, 1.0, 1.0, 0.2],
                    f"All required parts are already supplied. You have {supplied_text}.",
                    matched_parts=supplied_inputs,
                )
                return
            self._emit_reference_decision(
                result,
                ("no_other_next_step_parts" if scope == "next_step"
                 else "no_other_selected_step_parts"),
                [0.0, 1.0, 1.0, 0.2],
                f"There are no other active parts required for {scope_text.lower()}.",
            )
            return
        raw_active = [part for part in active_inputs if part in PROVIDED_PARTS]
        component_parts: list[str] = list(raw_active)
        # When an input is already a subassembly, highlight its consumed
        # components on the assembled gearbox instead of looking for a removed
        # pegboard object.
        for assembly in active_inputs:
            if assembly in PROVIDED_PARTS:
                continue
            for part in PROVIDED_PARTS:
                if (self.graph.current_container(part) == assembly
                        and part not in component_parts):
                    component_parts.append(part)

        ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
               for part in raw_active]
        ids = sorted({tool_id for tool_id in ids if tool_id is not None})
        descriptions = [self.graph.friendly_part(part) for part in active_inputs]
        if not descriptions:
            needed_text = "its required active assembly"
        elif len(descriptions) == 1:
            needed_text = descriptions[0]
        elif len(descriptions) == 2:
            needed_text = f"{descriptions[0]} and {descriptions[1]}"
        else:
            needed_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
        action = self.graph.friendly_step_action(step)
        if result.get("part_action") == "fetch":
            if self.graph.state(step) != "ready":
                spoken = (
                    f"{scope_text} is not ready, so I will not fetch its parts yet.")
                self._emit_reference_decision(
                    result, "fetch_blocked_step_parts",
                    [1.0, 0.72, 0.0, 0.2], spoken,
                    ids=[], matched_parts=active_inputs,
                    assembly_parts=[], warning=True)
                return
            if not raw_active:
                available_text = (
                    descriptions[0] if len(descriptions) == 1 else needed_text)
                supplied_prefix = ""
                if supplied_inputs:
                    supplied_descriptions = [
                        self.graph.friendly_part(part)
                        for part in supplied_inputs
                    ]
                    supplied_text = (
                        supplied_descriptions[0]
                        if len(supplied_descriptions) == 1
                        else ", and ".join(supplied_descriptions))
                    supplied_prefix = (
                        f"{supplied_text.capitalize()} "
                        f"{'has' if len(supplied_descriptions) == 1 else 'have'} "
                        "already been "
                        "supplied. ")
                spoken = (
                    f"{supplied_prefix}{available_text.capitalize()} is already "
                    "assembled and in the work area. There is no separate "
                    "pegboard part for the robot to fetch.")
                self._emit_reference_decision(
                    result, "step_parts_already_in_work_area",
                    [0.0, 1.0, 1.0, 0.2], spoken,
                    matched_parts=active_inputs,
                    current_assemblies=active_inputs,
                    assembly_parts=component_parts)
                return
            if len(active_inputs) == 1 and len(ids) == 1:
                friendly = descriptions[0]
                if friendly.lower().startswith("the "):
                    friendly = friendly[4:]
                self._arm_fetch_confirmation(
                    result, ids, friendly,
                    matched_parts=active_inputs,
                    assembly_parts=component_parts)
                if self._vlm is not None:
                    self._vlm.set_resolved_part_context(active_inputs[0])
                return
            spoken = (
                f"You still need {needed_text}. Please name the one you want "
                "the robot to get.")
            self._emit_reference_decision(
                result, "fetch_step_parts_not_specific",
                [0.0, 1.0, 1.0, 0.2], spoken,
                ids=ids, matched_parts=active_inputs,
                assembly_parts=component_parts)
            if self._vlm is not None:
                self._vlm.set_resolved_part_candidates(active_inputs)
            return

        highlight_pronoun = "it" if len(active_inputs) == 1 else "them"
        supplied_prefix = ""
        if supplied_inputs:
            supplied_descriptions = [self.graph.friendly_part(part)
                                     for part in supplied_inputs]
            already_text = (supplied_descriptions[0]
                            if len(supplied_descriptions) == 1
                            else ", and ".join(supplied_descriptions))
            statuses = {
                str(self._robot_part_states[part].get("status", ""))
                for part in supplied_inputs
            }
            location = ("is already in the robot gripper"
                        if "in_robot_gripper" in statuses
                        else "has already been handed to you")
            supplied_prefix = f"{already_text.capitalize()} {location}. "
        spoken = (
            f"{supplied_prefix}{scope_text} is to {action}. "
            f"You still need {needed_text}. "
            f"I have highlighted {highlight_pronoun}.")
        self._emit_reference_decision(
            result,
            "next_step_parts" if scope == "next_step" else "selected_step_parts",
            [0.0, 1.0, 1.0, 0.2],
            spoken,
            ids=ids,
            matched_parts=active_inputs,
            assembly_parts=component_parts,
        )
        if self._vlm is not None:
            self._vlm.set_resolved_part_candidates(active_inputs)

    def _friendly_missing(self, missing: list[str]) -> str:
        descriptions = []
        for item in missing[:3]:
            if item in self.graph.by_id:
                descriptions.append(
                    self.graph.friendly_step_action(self.graph.by_id[item]))
            else:
                producer = self.graph.producer_for(item)
                descriptions.append(
                    self.graph.friendly_step_action(producer)
                    if producer is not None else self.graph.friendly_part(item))
        if not descriptions:
            return "the earlier steps"
        if len(descriptions) == 1:
            return descriptions[0]
        return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

    def _fastening_tool_guidance(self, step: Step) -> str:
        coords = self.graph.control_coords_for(step.id)
        if coords is None or coords[1] not in (4, 6):
            return ""
        return {
            1: "the H5 bit from Holder 1 with the bit screwdriver",
            2: "the T25 bit from Holder 2 with the bit screwdriver",
            3: "the H3 bit from Holder 1 with the bit screwdriver",
            4: "the Phillips screwdriver",
        }.get(step.row, "")

    def _announce_step_selection(self, step: Step) -> None:
        state = self.graph.state(step)
        if state == "blocked":
            self._speak(
                f"Selected step: {step.title}. This step is not ready.",
                warning=True, allow_before_start=True)
        elif state == "complete":
            self._speak(
                f"Selected step: {step.title}. This step is already complete.",
                allow_before_start=True)
        else:
            self._speak(
                f"Selected step: {step.title}. Press Enter when you are ready.",
                allow_before_start=True)

    def _focus_vlm_on_step(self, step: Step) -> None:
        """Keep GUI- and Unity-originated selections identical for the VLM."""
        if self._vlm is None:
            return
        state = self.graph.state(step)
        focused = (
            f"Title: {step.title}\n"
            f"State: {state.upper()}\n"
            f"Action: {self.graph.friendly_step_action(step)}"
        )
        self._vlm.set_focused_step(focused)
        # Do not seed discourse with graph-ground-truth inputs. Candidate
        # highlights and follow-ups such as "the other one" must be based on a
        # candidate set the VLM actually predicted after this selection.
        self._vlm.set_resolved_part_candidates([])

    def _notify_vlm(self, event_label: str) -> None:
        """Replace the VLM's stored graph context after a state-changing event."""
        self._clear_pending_fetch()
        # Once an assembly step consumes a supplied raw part, its location is
        # represented by transform_history/current_container instead of the
        # robot-supply layer.
        for part in list(self._robot_part_states):
            if (part in PROVIDED_PARTS
                    and part not in self.graph.active_parts
                    and self._robot_part_states[part].get("source")
                    != "task_progression"):
                self._robot_part_states.pop(part, None)
        # A completed fastening step proves that its tools were available to
        # the operator. Add/remove those inferred handovers after every graph
        # change so complete, undo, live events, and reset behave identically.
        self._sync_progress_inferred_supply_statuses()
        self._publish_progress_handover_state()
        if self._vlm is not None:
            self._vlm.notify_graph_event(
                event_label, self._assistant_state_summary())

    def _assistant_state_summary(self) -> str:
        """Combine assembly progress with live robot grasp/handover state."""
        summary = self.graph.state_summary()
        lines = [summary]
        if self._robot_part_states:
            lines.extend(["", "ROBOT-SUPPLIED PART/TOOL STATE:"])
            for part in sorted(self._robot_part_states):
                info = self._robot_part_states[part]
                status = str(info.get("status", "")).upper()
                if info.get("source") == "task_progression":
                    lines.append(
                        f"  - {part}: {status} (inferred from completed step "
                        f"{info.get('inferred_from_step', 'unknown')})")
                else:
                    lines.append(
                        f"  - {part}: {status} (physical object: "
                        f"{info.get('tool_name', 'unknown')}, "
                        f"id={info.get('tool_id')})")
            lines.append(
                "Items marked IN_ROBOT_GRIPPER or HANDED_OVER are already "
                "supplied. Do not offer to fetch them again or describe them "
                "as outstanding. Supplied raw parts remain active task inputs "
                "until an assembly step consumes them; supplied tools remain "
                "available to the operator.")
        retained = self._tools_retained_from_completed_steps()
        retained = {label: step for label, step in retained.items()
                    if label not in self._robot_part_states}
        if retained:
            lines.extend([
                "",
                "TOOLS RETAINED BY THE OPERATOR FROM EARLIER STEPS:",
            ])
            for label, step in sorted(retained.items()):
                lines.append(
                    f"  - {label}: already used in {step.id} and assumed to "
                    "remain with the operator")
            lines.append(
                "Treat these tools as already available for later fastening "
                "steps. Do not claim that a separate robot handover was "
                "recorded unless their status above is HANDED_OVER.")
        return "\n".join(lines)

    # ── User actions and GUI callbacks ───────────────────────────────────────

    def _recommend_callback(self) -> None:
        """Select the preferred READY step in the GUI and Unity."""
        self._activate_recommended_step()

    def _activate_recommended_step(self, question: str = "") -> None:
        """Apply the graph recommendation as a synchronized step selection."""
        step = self.graph.recommend_next_step()
        if step is None:
            spoken = "There is no ready assembly step to recommend right now."
            self.log("[Recommend] No READY steps — assembly may be complete or stalled.")
            self._speak(spoken, warning=True)
            if question and self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "RECOMMENDED_STEP", spoken)
            return
        self.recommended_id = step.id
        self._select_step(step.id, announce=False)
        self._announce_step_selection(step)
        if self.controller is not None:
            self._animate_unity_callback()
        action = self.graph.friendly_step_action(step)
        tool_guidance = self._fastening_tool_guidance(step)
        suffix = f" Use {tool_guidance}." if tool_guidance else ""
        spoken = f"I selected the recommended step. {action.capitalize()}.{suffix}"
        self.log(f"[Recommend] Auto-selected: [{step.id}]  {step.title}")
        self._speak(spoken)
        if question and self._vlm is not None:
            self._vlm.apply_policy_response(
                question, "RECOMMENDED_STEP", spoken)

    def _animate_unity_callback(self) -> None:
        """Ask the in-process controller to animate the selected task stage."""
        if self.controller is None or not self.selected_id:
            return
        coords = TaskGraph.control_coords_for(self.selected_id)
        if coords is None:
            self.log("[Animate] This step has no Unity stage mapping.")
            return
        row, stage = coords
        step = self.graph.by_id[self.selected_id]
        if row > 0:
            done_stages = self.controller.sm._completed_stages(row)
            blocked     = not self.controller.sm.unlocked(row, stage)
            checked     = self.controller.sm.done[row][stage]
        else:
            done_stages = [s for s in range(1, 8)]
            blocked     = False
            checked     = self.controller.sm.done8
        # A Unity checkbox click after programmatic selection must apply to the
        # same recommended stage, just as if the user had clicked its part.
        self.controller.sm.current_row = row
        self.controller.sm.current_stage = stage
        self.controller.send({"command": "stage", "row": row, "stage": stage,
                               "done_stages": done_stages,
                               "step_delay": gearbox_control.STEP_DELAY,
                               "slide_seconds": gearbox_control.SLIDE_SECONDS})
        self.controller.send({"command": "ui", "show": True, "row": row,
                               "checked": checked, "blocked": blocked})
        self.log(f"[Animate] Unity → row {row}, stage {stage}  [{step.id}]")

    def _select_callback(self, _sender, _app_data, user_data) -> None:
        """Select a graph step and update external viewers and VLM focus."""
        self._select_step(user_data, announce=True)

    def _select_step(self, step_id: str, *, announce: bool) -> None:
        """Synchronize one selection with DearPyGui and external 3D viewers."""
        self._clear_pending_fetch()
        self.selected_id = step_id
        self._start_acquisition_trial()
        self.refresh()
        step = self.graph.by_id[step_id]
        if announce:
            self._announce_step_selection(step)
        if self.controller is not None:
            self.controller.send({"command": "reference_clear"})
        coords = TaskGraph.control_coords_for(step_id)
        if coords is not None:
            row, stage = coords
            index = gearbox_control.load_tool_index(self._tool_layout_path)
            ids = (gearbox_control.appearing_ids(index, row, stage)
                   if row > 0 else [])
            self._send_select({"event": "select", "row": row, "stage": stage,
                               "step": step_id, "ids": ids,
                               "blocked": self.graph.state(step) == "blocked"})
        self._publish_selected_step_board_highlight()
        self._focus_vlm_on_step(step)

    def _sync_sm_from_graph(self) -> None:
        """Rebuild GearboxStateMachine.done from TaskGraph.completed.
        Called after every GUI action that changes TaskGraph so the controller
        always reflects the same state — TaskGraph is the single source of truth."""
        if self.controller is None:
            return
        sm = self.controller.sm
        for row in range(1, 5):
            for stage in range(1, 8):
                step_ids = TaskGraph.steps_for_control(row, stage)
                sm.done[row][stage] = bool(step_ids) and all(
                    sid in self.graph.completed
                    for sid in step_ids if sid in self.graph.by_id)
        sm.done8 = "finish_gearbox" in self.graph.completed

    def _complete_selected(self) -> None:
        """Complete a ready selected step or undo a completed frontier step."""
        if not self.selected_id:
            return
        step = self.graph.by_id[self.selected_id]
        was_complete = self.graph.state(step) == "complete"
        if was_complete:
            ok, message = self.graph.undo(step)
            if ok:
                self._notify_vlm(f"UNDO: {step.id}")
        else:
            ok, message = self.graph.complete(step)
            if ok:
                self._notify_vlm(f"COMPLETE: {step.id} -> {step.output}")
        if ok and self.controller is not None:
            self._sync_sm_from_graph()
            coords = TaskGraph.control_coords_for(self.selected_id)
            if coords is not None and coords[0] > 0:
                row = coords[0]
                self.controller.send({"command": "recolor", "row": row,
                                      "done_stages": self.controller.sm._completed_stages(row)})
        if ok:
            coords = TaskGraph.control_coords_for(self.selected_id)
            if coords is not None:
                self._send_select({"event": "uncomplete" if was_complete else "complete",
                                   "row": coords[0], "stage": coords[1],
                                   "step": self.selected_id})
            self._log_session_event(
                "graph_uncomplete" if was_complete else "graph_complete")
        self.log(message)
        if ok and was_complete:
            self._speak(f"Undid {self.graph.friendly_step(step)}.")
        elif ok:
            self._speak(
                f"Step complete. You have created "
                f"{self.graph.friendly_part(step.output)}.")
        else:
            self._speak(
                f"That action is not available. {self._friendly_missing(self.graph.missing(step))} "
                "must be completed first.", warning=True)
        self.refresh()

    def _auto_complete_acquired_step(self) -> None:
        """Complete and synchronize a step after its final acquisition."""
        if not self.selected_id:
            return
        step = self.graph.by_id[self.selected_id]
        if self.graph.state(step) == "complete":
            return
        self.log(
            f"[Acquisition] All required objects retrieved — auto-completing "
            f"{step.id} in the task graph and Unity.")
        self._complete_selected()

    def _reset_state(self, *, clear_handovers: bool) -> None:
        """Reset progress, optionally treating the physical inventory as restocked."""
        self.graph.reset()
        self.selected_id    = None
        self.recommended_id = None
        if clear_handovers:
            self._robot_part_states.clear()
            self._removed_tool_ids.clear()
            if self._study4_scene is not None:
                self._study4_scene.restore_acquired_objects()
        self._send_select({"event": "reset"})
        if clear_handovers:
            self._send_select({"event": "reset_all_handovers"})
        if self.controller is not None:
            # Sync sm state from the now-empty graph (all done = False), then send
            # Unity commands directly — bypassing the ZMQ round-trip so we don't
            # get a spurious "Live: controller reset" log from _apply_live_event.
            self._sync_sm_from_graph()
            self.controller.sm.history.clear()
            self.controller.sm.current_row = self.controller.sm.current_stage = None
            self.controller.send({"command": "reset"})    # clear Unity part colors
            self.controller.send({"command": "show_all"}) # make all rows visible
            self.controller.send({"command": "ui", "show": False})  # hide in-headset UI
        if clear_handovers:
            self.log("All assembly progress and handover records were reset.")
            self._speak(
                "Assembly progress and handed-over inventory have been reset.",
                warning=True)
        else:
            self.log("All assembly progress was reset.")
            self._speak("Assembly progress has been reset.", warning=True)
        self.refresh()
        self._send_select({"event": "clear"})
        event = ("RESET: progress and all handovers cleared"
                 if clear_handovers else "RESET: all progress cleared")
        self._notify_vlm(event)
        self._log_session_event(
            "session_reset_all" if clear_handovers else "graph_reset")

    def _reset_callback(self) -> None:
        """Reset task progress while preserving confirmed robot handovers."""
        self._reset_state(clear_handovers=False)

    def _reset_progress_and_handovers_callback(self) -> None:
        """Reset progress and repopulate the complete physical inventory."""
        self._reset_state(clear_handovers=True)

    def log(self, message: str) -> None:
        """Write one compact, consistently categorized live-study line."""
        raw = " ".join(str(message).split())
        category = "SYSTEM"
        payload = raw
        match = re.match(r"^\[([^]]+)\]\s*(.*)$", raw)
        if match:
            source, payload = match.groups()
            source_lower = source.casefold()
            if source_lower.startswith("voice"):
                category = "SPEECH"
            elif source_lower.startswith("tts"):
                category = "TTS"
            elif source_lower == "acquisition":
                category = ("STEP" if payload.startswith("STEP COMPLETE")
                            else "CLICK")
            elif source_lower.startswith(("vlm", "part step")):
                category = "VLM"
            elif source_lower.startswith(("recommend", "animate")):
                category = "STEP"
            elif source_lower.startswith("studylog"):
                category = "LOG"
            else:
                category = source.upper().replace(" ", "_")
        elif raw.startswith(("Live:", "COMPLETED", "UNDONE")):
            category = "STEP"
        print(f"\r\033[2K[{category:<6}] {payload}", flush=True)

    def _session_state(self) -> dict[str, object]:
        elapsed = (
            self._trial_finished_elapsed_s
            if self._trial_finished_elapsed_s is not None else
            (0.0 if self._step_selected_at is None else
             max(0.0, time.monotonic() - self._step_selected_at)))
        return {
            "selected_step": self.selected_id or "",
            "completed_steps": list(self.graph.completed),
            "removed_tool_ids": sorted(self._removed_tool_ids),
            "trial_required_tool_ids": sorted(self._trial_required_tool_ids),
            "acquired_tool_ids": sorted(self._acquired_tool_ids),
            "acquisition_complete": self._acquisition_complete,
            "step_elapsed_s": elapsed,
            "step_attempt": self._step_attempt,
            "step_event_index": self._step_event_index,
            "trial_recording": self._trial_recording,
            "trial_finished_elapsed_s": self._trial_finished_elapsed_s,
            "head_translation_m": self._trial_head_translation_m,
            "head_rotation_deg": self._trial_head_rotation_deg,
        }

    def _log_session_event(self, event_type: str, **values) -> None:
        if self._session_logger is None:
            return
        grouped = bool(
            self._step_attempt > 0
            and (self._trial_recording or self._acquisition_complete))
        if grouped:
            self._step_event_index += 1
        step = self.graph.by_id.get(self.selected_id) if self.selected_id else None
        self._session_logger.append(
            event_type=event_type,
            step_id=self.selected_id or "",
            step_title=step.title if step is not None else "",
            step_state=self.graph.state(step) if step is not None else "none",
            step_attempt=(self._step_attempt if grouped else ""),
            step_event_index=(self._step_event_index if grouped else ""),
            state_json=self._session_state(),
            **values)

    def _resume_session(self) -> None:
        """Restore the latest participant-condition checkpoint, if present."""
        if self._session_logger is None:
            return
        state = self._session_logger.latest_state()
        if not state:
            self._log_session_event("session_start")
            return
        self.graph.reset()
        restored = 0
        for step_id in state.get("completed_steps", []):
            step = self.graph.by_id.get(str(step_id))
            if step is None:
                continue
            ok, _message = self.graph.complete(step)
            restored += int(ok)
        # Older checkpoints may have incorrectly removed reusable screwdrivers,
        # bits, or holders. Reconcile them through the current consumable rule
        # so they reappear and can be acquired again in later rows.
        self._removed_tool_ids = self._removable_consumable_ids(
            int(value) for value in state.get("removed_tool_ids", []))
        self._trial_required_tool_ids = {
            int(value) for value in state.get("trial_required_tool_ids", [])}
        self._acquired_tool_ids = {
            int(value) for value in state.get("acquired_tool_ids", [])}
        self._acquisition_complete = bool(
            state.get("acquisition_complete", False))
        self._step_attempt = int(
            state.get("step_attempt", self._step_attempt) or self._step_attempt)
        self._step_event_index = int(state.get("step_event_index", 0) or 0)
        selected = str(state.get("selected_step", ""))
        self.selected_id = selected if selected in self.graph.by_id else None
        resume_existing_attempt = bool(
            self.selected_id
            and self.graph.state(self.graph.by_id[self.selected_id])
            != "complete")
        if resume_existing_attempt:
            # Resume restores completed work, but an interrupted current step
            # always restarts from zero. Preserve its old rows for audit while
            # explicitly excluding that attempt from analysis.
            had_active_attempt = bool(
                state.get("trial_recording", False)
                or self._acquired_tool_ids
                or self._acquisition_complete)
            if had_active_attempt and self._session_logger is not None:
                self._step_event_index += 1
                abandoned_state = dict(state)
                abandoned_state.update({
                    "attempt_status": "abandoned",
                    "trial_recording": False,
                })
                step = self.graph.by_id[self.selected_id]
                self._session_logger.append(
                    event_type="step_attempt_abandoned",
                    modality="resume_restart",
                    step_id=self.selected_id,
                    step_title=step.title,
                    step_state=self.graph.state(step),
                    step_attempt=(self._step_attempt or ""),
                    step_event_index=(self._step_event_index
                                      if self._step_attempt else ""),
                    state_json=abandoned_state,
                )
            # Only objects acquired during the interrupted step are restocked;
            # objects consumed by earlier completed steps stay removed.
            self._removed_tool_ids.difference_update(self._acquired_tool_ids)
            self._acquired_tool_ids.clear()
            self._acquisition_complete = False
            self._start_acquisition_trial()
            self.log(
                f"[StudyLog] Restarted unfinished step {self.selected_id} "
                "from zero; press ENTER to begin.")
        else:
            # A completed saved selection is not the resume target. Select the
            # first graph-ready unfinished step and give it a new attempt block.
            next_step = self.graph.recommend_next_step()
            self.selected_id = next_step.id if next_step is not None else None
            self._step_selected_at = None
            self._last_acquisition_at = None
            self._trial_recording = False
            self._trial_finished_elapsed_s = None
            if self.selected_id:
                self._start_acquisition_trial()
        if self._study4_scene is not None and self._removed_tool_ids:
            self._study4_scene.remove_acquired_objects(self._removed_tool_ids)
        self._sync_sm_from_graph()
        if self.controller is not None:
            for row in range(1, 5):
                self.controller.send({
                    "command": "recolor", "row": row,
                    "done_stages": self.controller.sm._completed_stages(row),
                })
        for step_id in self.graph.completed:
            step = self.graph.by_id[step_id]
            coords = TaskGraph.control_coords_for(step_id)
            if coords is not None:
                self._send_select({
                    "event": "complete", "row": coords[0],
                    "stage": coords[1], "step": step_id,
                    "output": step.output,
                })
        self.refresh()
        if self.selected_id:
            step = self.graph.by_id[self.selected_id]
            coords = TaskGraph.control_coords_for(self.selected_id)
            if coords is not None:
                row, stage = coords
                ids = gearbox_control.appearing_ids(
                    self._tool_index, row, stage) if row > 0 else []
                self._send_select({
                    "event": "select", "row": row, "stage": stage,
                    "step": self.selected_id, "ids": ids,
                    "blocked": self.graph.state(step) == "blocked"})
                if self.controller is not None:
                    self._animate_unity_callback()
            self._publish_selected_step_board_highlight()
            self._focus_vlm_on_step(step)
        self._log_session_event("session_resumed")
        self.log(
            f"[StudyLog] Resumed {restored} completed step(s), selected="
            f"{self.selected_id or 'none'}, acquired="
            f"{sorted(self._acquired_tool_ids)}")

    # ── GUI state rendering and inventory tree ───────────────────────────────

    def refresh(self) -> None:
        """Redraw node themes, labels, inventory, and selected-step controls."""
        dpg = self.dpg
        # Auto-clear recommendation once the recommended step is completed.
        if (self.recommended_id
                and self.graph.state(self.graph.by_id[self.recommended_id]) == "complete"):
            self.recommended_id = None
        for step in self.graph.steps:
            state  = self.graph.state(step)
            active = step.id in self.active_ids or step.id == self.selected_id
            rec    = step.id == self.recommended_id and not active

            # Completion color has the highest priority. Keep the node selected
            # so its details remain visible, but render it green immediately
            # after completion instead of leaving the active/selected purple
            # theme in place until another node is clicked.
            if state == "complete":
                theme = self.themes["complete"]
            elif active:
                theme = self.themes["active"]
            elif rec:
                theme = self.themes["recommended"]
            else:
                theme = self.themes[state]
            dpg.bind_item_theme(self.node_tags[step.id], theme)
            label = f"{step.title}  [{state.upper()}]"
            if active:
                label += "  (open)"
            elif rec:
                label += "  [RECOMMENDED]"
            dpg.configure_item(self.node_tags[step.id], label=label)
            dpg.set_value(f"node_state::{step.id}", f"State: {state.upper()}")

        self._refresh_active_parts_tree()

        if not self.selected_id:
            dpg.configure_item("selection_notice", show=False)
            dpg.set_value("step_details", "Select a graph node to see its inputs, output, and readiness.")
            dpg.configure_item("complete_button", label="Mark selected step complete", enabled=False)
            dpg.configure_item("animate_unity_button", enabled=False)
            return
        step = self.graph.by_id[self.selected_id]
        state = self.graph.state(step)
        missing = self.graph.missing(step)
        detail = (f"{step.id}\n\n{step.description}\n\n"
                  f"Inputs: {', '.join(step.inputs)}\n\n"
                  f"Produces: {step.output}\n\n"
                  f"State: {state.upper()}")
        if missing and step.id not in self.graph.completed:
            detail += "\n\nBlocked by: " + ", ".join(missing)
        dpg.set_value("step_details", detail)
        if state == "blocked":
            notice = ("WARNING: This step should not be performed yet.\n"
                      "Complete or provide these prerequisites first: " + ", ".join(missing))
            dpg.set_value("selection_notice", notice)
            dpg.configure_item("selection_notice", show=True, color=(255, 85, 85))
        elif state == "complete":
            if self.graph.is_frontier(step):
                dpg.set_value("selection_notice",
                              f"COMPLETED: Created {step.output}\n"
                              "This step is on the completed frontier and can be undone.")
                dpg.configure_item("selection_notice", show=True, color=(90, 225, 135))
            else:
                consumers = self.graph.completed_consumers(step)
                blocked_by = ", ".join(item.id for item in consumers) or "a dependent step"
                dpg.set_value("selection_notice",
                              f"COMPLETED: Created {step.output}\n"
                              f"Undo completed dependent step(s) first: {blocked_by}.")
                dpg.configure_item("selection_notice", show=True, color=(255, 195, 90))
        else:
            dpg.set_value("selection_notice", "READY: This step can be performed now.")
            dpg.configure_item("selection_notice", show=True, color=(100, 185, 255))
        if state == "complete" and self.graph.is_frontier(step):
            dpg.configure_item("complete_button", label="Undo selected frontier step", enabled=True)
        elif state == "complete":
            dpg.configure_item("complete_button", label="Undo dependent step(s) first", enabled=False)
        else:
            dpg.configure_item("complete_button", label="Mark selected step complete",
                               enabled=self.graph.is_ready(step))
        can_animate = (self.controller is not None
                       and TaskGraph.control_coords_for(self.selected_id) is not None)
        dpg.configure_item("animate_unity_button", enabled=can_animate)

    # Example: if active_parts contains
    # {
    #     "BASE_BOARD",
    #     "BEARING_ROW1_RIGHT",
    #     "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
    # }
    # this method rebuilds the visible tree as:
    #
    # ROW 1 (2 active)
    #   - BEARING_ROW1_RIGHT
    #   - [ASSEMBLY] BEARING_STAND_ROW1_LEFT_ASSEMBLY
    #       - BEARING_ROW1_LEFT
    #       - STAND_ROW1_LEFT
    # BASE / FINAL (1 active)
    #   - BASE_BOARD
    #
    # Only active parts appear at the top level. _add_part_branch() recursively
    # shows the consumed components inside an active assembly.
    def _refresh_active_parts_tree(self) -> None:
        """Rebuild the inventory tree from the graph's current active parts."""
        dpg = self.dpg
        dpg.delete_item("active_parts_tree", children_only=True)
        grouped: dict[str, list[str]] = {
            "ROW 1": [], "ROW 2": [], "ROW 3": [], "ROW 4": [], "BASE / FINAL": []
        }
        for part in sorted(self.graph.active_parts):
            group = "BASE / FINAL"
            for row in range(1, 5):
                if f"_ROW{row}" in part:
                    group = f"ROW {row}"
                    break
            grouped[group].append(part)

        for group_name, parts in grouped.items():
            if not parts:
                continue
            group_node = dpg.add_tree_node(
                label=f"{group_name} ({len(parts)} active)", parent="active_parts_tree",
                default_open=True, span_full_width=True)
            for part in parts:
                self._add_part_branch(part, group_node)

    def _add_part_branch(self, part: str, parent) -> None:
        """Recursively add a raw part or expandable assembly to the inventory tree."""
        dpg = self.dpg
        producer = self.graph.producer_for(part)
        if producer is None:
            dpg.add_tree_node(label=part, parent=parent, leaf=True, bullet=True,
                              span_text_width=True)
            return
        assembly_node = dpg.add_tree_node(
            label=f"[ASSEMBLY] {part}", parent=parent, default_open=False,
            span_full_width=True)
        dpg.bind_item_theme(assembly_node, "assembly_tree_theme")
        for component in producer.inputs:
            self._add_part_branch(component, assembly_node)

def run_self_test() -> None:
    errors = TaskGraph.validate()
    assert not errors, errors
    recommendation_graph = TaskGraph()
    assert recommendation_graph.recommend_next_step().id == "r1_bearing_right"
    recommendation_graph.complete(recommendation_graph.by_id["r1_bearing_right"])
    # Both Stage 2 and Stage 3 are now READY; the documented controller-stage
    # ordering must select gear-rod Stage 2 before left-bearing Stage 3.
    assert recommendation_graph.recommend_next_step().id == "r1_gear_rod"
    recommendation_graph.complete(recommendation_graph.by_id["r1_gear_rod"])
    assert recommendation_graph.recommend_next_step().id == "r1_bearing_left"

    # Recommendation follows the row most recently changed by completion or
    # undo, even while lower-numbered rows also contain READY work.
    focus_graph = TaskGraph()
    ok, _ = focus_graph.complete(focus_graph.by_id["r2_bearing_right"])
    assert ok and focus_graph.last_worked_row == 2
    assert focus_graph.recommend_next_step().id == "r2_gear_rod"
    ok, _ = focus_graph.complete(focus_graph.by_id["r3_bearing_right"])
    assert ok and focus_graph.last_worked_row == 3
    assert focus_graph.recommend_next_step().id == "r3_gear_rod"
    ok, _ = focus_graph.undo(focus_graph.by_id["r2_bearing_right"])
    assert ok and focus_graph.last_worked_row == 2
    assert focus_graph.recommend_next_step().id == "r2_bearing_right"
    focus_graph.reset()
    assert focus_graph.last_worked_row is None
    assert focus_graph.recommend_next_step().id == "r1_bearing_right"

    graph = TaskGraph()
    assert graph.is_ready(graph.by_id["r1_bearing_right"])
    assert not graph.is_ready(graph.by_id["r1_bearing_left"])
    ok, _ = graph.complete(graph.by_id["r1_bearing_right"])
    assert ok
    assert graph.is_ready(graph.by_id["r1_bearing_left"])
    ok, _ = graph.complete(graph.by_id["r1_bearing_left"])
    assert ok and "BEARING_STAND_ROW1_LEFT_ASSEMBLY" in graph.active_parts
    ok, warning = graph.undo(graph.by_id["r1_bearing_right"])
    assert not ok and "r1_bearing_left" in warning
    ok, _ = graph.undo(graph.by_id["r1_bearing_left"])
    assert ok
    ok, _ = graph.undo(graph.by_id["r1_bearing_right"])
    assert ok and "BEARING_ROW1_RIGHT" in graph.active_parts
    graph.complete(graph.by_id["r1_bearing_right"])
    graph.complete(graph.by_id["r1_bearing_left"])
    graph.complete(graph.by_id["r1_fasten_first_stand"])
    # Fastening the right stand does not consume or depend on the independently
    # assembled left stand, so the left step remains undoable here.
    ok, _ = graph.undo(graph.by_id["r1_bearing_left"])
    assert ok
    ok, _ = graph.undo(graph.by_id["r1_fasten_first_stand"])
    assert ok
    ok, warning = graph.complete(graph.by_id["r1_insert_rod_and_fit_second"])
    assert not ok and warning.startswith("WARNING")
    # Complete every currently-ready step until the graph reaches its fixed point.
    while True:
        ready = [s for s in graph.steps if graph.is_ready(s)]
        if not ready:
            break
        for step in ready:
            graph.complete(step)
    assert "COMPLETED_GEARBOX_ASSEMBLY" in graph.active_parts
    assert len(graph.completed) == len(graph.steps)
    print(f"Self-test passed: {len(graph.steps)} steps; final part COMPLETED_GEARBOX_ASSEMBLY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="test the task model without opening the GUI")
    parser.add_argument("--no-live", action="store_true",
                        help="Disable the live controller link (open the viewer standalone).")
    parser.add_argument(
        "--voice-device", default=DEFAULT_PULSEAUDIO_DEVICE,
        help=("PulseAudio source name for voice input "
              f"(default: {DEFAULT_PULSEAUDIO_DEVICE})."))
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice input.")
    parser.add_argument(
        "--asr-device", choices=("cpu", "cuda", "auto"), default="cuda",
        help=("Device for Parakeet speech recognition. Study 4 loads and "
              "warms ASR on CUDA before loading the VLM."))
    parser.add_argument(
        "--wake-word", default="",
        help="Optional wake word. By default speech is accepted continuously without one.")
    parser.add_argument("--vlm-model", default=None,
                        help="Enable VLM assistant with this model name "
                             "(for example Qwen/Qwen3-VL-8B-Instruct). Omit to disable.")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable spoken task guidance and warnings.")
    parser.add_argument("--tts-engine", choices=("piper", "nemo"), default="piper",
                        help="Speech-output backend (default: piper).")
    parser.add_argument("--piper-model", default=str(DEFAULT_PIPER_MODEL),
                        help="Piper voice .onnx file used for spoken guidance.")
    parser.add_argument(
        "--tts-rate", type=float, default=0.85,
        help="Piper speaking-rate multiplier; values below 1.0 speak more slowly.")
    parser.add_argument(
        "--assistant-log",
        default=None,
        help="Optional legacy CSV for part-reference decisions.")
    parser.add_argument("--no-assistant-log", action="store_true",
                        help="Disable part-reference study logging.")
    parser.add_argument("--participant-id", default="anonymous",
                        help="Participant identifier written to the Study 4 session log.")
    parser.add_argument(
        "--click-log",
        default=None,
        help="Optional legacy CSV receiving physical pegboard clicks.")
    parser.add_argument("--no-click-log", action="store_true",
                        help="Disable physical pegboard click logging.")
    parser.add_argument(
        "--step-log",
        default=None,
        help="Optional legacy CSV containing completed-step summaries.")
    parser.add_argument("--no-step-log", action="store_true",
                        help="Disable per-step acquisition summary logging.")
    parser.add_argument(
        "--session-log",
        default=None,
        help=("Override the condition CSV path. By default logs are saved as "
              "study_logs/study4/<participant>/<condition>.csv."))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume the latest matching participant/condition checkpoint.")
    parser.add_argument(
        "--condition", dest="study4_condition",
        choices=("gesture", "language", "task_aware"), required=True,
        help=("Study 4 ablation: gesture disables language; language grounds parts "
              "without task-state policy; task_aware enables the full policy."))
    controller_group = parser.add_mutually_exclusive_group()
    controller_group.add_argument(
        "--with-controller", dest="with_controller", action="store_true",
        help="Run the Unity gearbox controller in this Study 4 process (default).")
    controller_group.add_argument(
        "--no-controller", dest="with_controller", action="store_false",
        help="Disable the in-process Unity controller.")
    parser.set_defaults(with_controller=True)
    parser.add_argument(
        "--open3d-scene", action=argparse.BooleanOptionalAction, default=True,
        help="Show the standalone pegboard and BoardAR Open3D mirror.")
    parser.add_argument("--unity-ip", default=gearbox_control._DEFAULT_IP,
                        help=f"With --with-controller: Unity host (default: {gearbox_control._DEFAULT_IP}).")
    parser.add_argument("--cmd-port", type=int, default=gearbox_control.DEFAULT_CMD_PORT,
                        help=f"With --with-controller: port this script PUBs commands on to Unity "
                             f"(show_row, stage, recolor, reset, …). "
                             f"OUT: this script -> Unity. (default: {gearbox_control.DEFAULT_CMD_PORT})")
    parser.add_argument("--click-port", type=int, default=gearbox_control.DEFAULT_CLICK_PORT,
                        help=f"With --with-controller: port this script SUBs on to receive part-click "
                             f"events from Unity (part name + event type). "
                             f"IN: Unity -> this script. (default: {gearbox_control.DEFAULT_CLICK_PORT})")
    parser.add_argument("--no-highlight", action="store_true",
                        help=("Deprecated compatibility flag; Condition 3 controls "
                              "its selected-step cyan search space automatically."))
    args = parser.parse_args()
    if args.tts_rate <= 0.0:
        parser.error("--tts-rate must be greater than zero.")
    if args.self_test:
        run_self_test()
        return
    if args.study4_condition == "gesture" and args.vlm_model is not None:
        parser.error("Study 4 gesture condition must omit --vlm-model.")
    if args.study4_condition in {"language", "task_aware"} and args.vlm_model is None:
        parser.error(f"Study 4 {args.study4_condition} condition requires --vlm-model.")
    live_port    = None if args.no_live else _DEFAULT_CTRL_EVENTS_IN_PORT
    # Study 4 applies highlights directly to its embedded Open3D scene. Port
    # 5025 remains unused, so main_with_robot.py is neither needed nor able to
    # receive accidental robot-fetch requests.
    select_port  = None
    # C1 is gesture-only: disable ASR/VLM logs, but retain concise acquisition
    # TTS feedback unless the caller explicitly passes --no-tts.
    gesture_only = args.study4_condition == "gesture"
    voice_device = None if (args.no_voice or gesture_only) else args.voice_device
    safe_participant_id = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", args.participant_id.strip()).strip("._")
    if not safe_participant_id:
        safe_participant_id = "anonymous"
    session_log_path = args.session_log or str(
        Path(__file__).resolve().parent.parent / "study_logs" / "study4"
        / safe_participant_id / f"{args.study4_condition}.csv")

    # Optionally co-launch gearbox_control.py in-process. DearPyGui owns the main thread, so the
    # controller's click listener (+ optional REPL) run on daemon threads behind it; the two keep
    # talking over the localhost live link (task-graph port). Both files still run standalone.
    app = DearPyGuiTaskGraphApp(study4_condition=args.study4_condition)
    controller = None
    controller_click_thread = None
    if args.with_controller:
        if args.no_live:
            parser.error("--with-controller needs the live link; do not pass --no-live.")
        controller = gearbox_control.GearboxController(
            args.unity_ip, args.cmd_port, args.click_port,
            _LOCALHOST, _DEFAULT_CTRL_EVENTS_IN_PORT,
            # Automatic step->pegboard highlighting discloses the target and
            # is therefore disabled for every Study 4 condition. Explicit
            # language-reference feedback still uses the color channel above.
            no_highlight=True, tool_json=app._tool_layout_path)
        app.controller = controller
        controller.sm.completion_guard = app._completion_guard
        controller_click_thread = threading.Thread(
            target=controller.run_click_loop, daemon=True)
        controller_click_thread.start()
        print(f"[Controller] in-process — cmd -> {args.unity_ip}:{args.cmd_port}, "
              f"clicks <- {args.unity_ip}:{args.click_port}, "
              f"mirror -> {_LOCALHOST}:{_DEFAULT_CTRL_EVENTS_IN_PORT}")
        # Python restarted with a fresh (empty) TaskGraph — wipe any stale Unity visuals
        # left over from the previous session before the user sees the app.
        controller.send({"command": "reset"})
        controller.send({"command": "show_all"})
        controller.send({"command": "ui", "show": False})

    try:
        app.run(
            live_port,
            voice_device=voice_device,
            asr_device=args.asr_device,
            wake_word=args.wake_word,
            vlm_model=args.vlm_model,
            select_port=select_port,
            tts_engine=None if args.no_tts else args.tts_engine,
            piper_model=args.piper_model,
            tts_rate=args.tts_rate,
            assistant_log_path=(None if (args.no_assistant_log or gesture_only)
                                else args.assistant_log),
            click_log_path=(None if args.no_click_log else args.click_log),
            step_log_path=(None if args.no_step_log else args.step_log),
            session_log_path=session_log_path,
            resume_session=args.resume,
            participant_id=args.participant_id,
            open3d_scene=args.open3d_scene,
            unity_ip=args.unity_ip,
        )
    finally:
        if controller is not None:
            controller.stop()
            if controller_click_thread is not None:
                controller_click_thread.join(timeout=1.0)
            controller.close()


if __name__ == "__main__":
    main()
