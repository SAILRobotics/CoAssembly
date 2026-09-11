"""Interactive tool to manually annotate handle-grasp times in Study 2 replays.

Automatic grasp detection infers grab onset from AR-handle proximity +
stillness because exact grab/release events are rarely logged. This tool
instead steps a human annotator frame-by-frame
through the Open3D replay scene so they can mark the true moment the
participant's hand grasps the handle, for every trial of a session.

Controls
  TIMELINE    drag the Frame slider to scrub; green=recorded grab,
              cyan=manual annotation, red=release
  SPACE       play / pause
  LEFT/RIGHT  step one frame back / forward (pauses)
  HOME/END    jump to first / last frame of the trial
  [ / ]       decrease / increase playback speed
  G           mark a grasp at the current frame's time
  Z           undo the last grasp mark for this trial
  E           mark a manual release at the current frame's time
  T           undo the last manual release for this trial
  D           remove/exclude the grasp-onset/release pair nearest the slider
  R           restore the excluded pair nearest the slider
  C           mark orientation-noise start/end at the current slider time
  V           cancel pending start or undo the last orientation-noise interval
  H           print all annotation-related keyboard controls
  X           mark trial reviewed with no visible manual grasp
  N / B       next / previous trial (auto-saves)
  S           save annotations to disk now
  Q / ESC     save and quit

Usage
  python study2_annotate_grasp.py mahya
  python study2_annotate_grasp.py pantea
  python study2_annotate_grasp.py parisa
  python study2_annotate_grasp.py austin
  python study2_annotate_grasp.py study_logs/study2/other_replay.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

import main_setting as cfg
from workholding_study import _WorkholdingSceneVis

_MODES = ("freedrive", "ar", "hybrid")
_LOG_DIR = Path("study_logs/study2")

_KEY_SPACE, _KEY_LEFT, _KEY_RIGHT = 32, 263, 262
_KEY_HOME, _KEY_END = 268, 269
_KEY_LBRACKET, _KEY_RBRACKET = 91, 93
_KEY_ESCAPE = 256


def _array(value, matrix=False):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result.reshape(4, 4) if matrix else result


def _trial_key_str(key: tuple) -> str:
    session_id, mode, trial_idx = key
    return f"{session_id}|{mode}|{trial_idx}"


def _load(path: Path):
    """Return (metadata_by_session, trials) from a workholding replay log."""
    metadata_by_session: dict[str, dict] = {}
    trials: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[Annotate] Skipping malformed line {line_number}: {error}")
                continue
            if record.get("schema") != "workholding_replay_v1":
                continue
            if record.get("event") == "session_start":
                metadata_by_session[str(record.get("session_id", ""))] = record
            mode, trial_idx = record.get("mode"), record.get("trial_idx")
            if mode not in _MODES or not isinstance(trial_idx, int):
                continue
            key = (str(record.get("session_id", "")), mode, trial_idx)
            trials.setdefault(key, []).append(record)
    if not trials:
        raise SystemExit(f"No Study 2 trials found in {path}")
    return metadata_by_session, trials


def _annotations_path(participant: str) -> Path:
    return _LOG_DIR / f"{participant}_grasp_annotations.json"


def _edited_annotations_path(participant: str) -> Path:
    return _LOG_DIR / f"{participant}_grasp_annotations_edited.json"


def _noise_annotations_path(participant: str) -> Path:
    return _LOG_DIR / f"{participant}_orientation_noise_annotations.json"


def _load_annotations(path: Path, participant: str, source: Path) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("trials", {})
        return data
    return {"participant": participant, "source_log": str(source), "trials": {}}


def _save_annotations(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    tmp.replace(path)


class _Annotator:
    def __init__(self, participant: str, path: Path):
        self.participant = participant
        self.original_annotations_path = _annotations_path(participant)
        self.out_path = _edited_annotations_path(participant)
        self.noise_path = _noise_annotations_path(participant)
        metadata_by_session, trials = _load(path)
        self.metadata_by_session = metadata_by_session
        # Keep all modes available for contextual replay; only AR and hybrid
        # accept grasp annotations.
        self.keys = list(trials)
        self.frames = {key: [r for r in records
                             if r.get("type") == "frame"
                             and r.get("timer_running")]
                       for key, records in trials.items()}
        self.summaries = {
            key: next((r for r in records if r.get("event") == "trial_summary"),
                      None)
            for key, records in trials.items()
        }
        self.records = {key: trials[key] for key in self.keys}
        # Continue an edited copy if one exists.  Otherwise start from a deep
        # copy of the original annotations.  The original JSON and replay log
        # are deliberately never used as save targets by this tool.
        if self.out_path.is_file():
            self.data = _load_annotations(self.out_path, participant, path)
        else:
            self.data = copy.deepcopy(_load_annotations(
                self.original_annotations_path, participant, path))
        self.data["derived_from_annotations"] = str(
            self.original_annotations_path)
        self.noise_data = _load_annotations(self.noise_path, participant, path)
        self.noise_data["annotation_type"] = "orientation_confusion_noise"
        self.recorded_grasps = {
            key: self._recorded_grasp_indices(key) for key in self.keys}
        self.release_indices = {
            key: self._event_frame_indices(key, "ar_handle_released")
            for key in self.keys}
        self.control_states = {
            key: self._control_states(key) for key in self.keys}
        self.freedrive_intervals = {
            key: self._freedrive_intervals(key) for key in self.keys}

        self.trial_i = 0
        self.frame_i = 0
        self.playing = False
        self.play_speed = 1.0
        self._play_anchor_wall = 0.0
        self._play_anchor_t = 0.0
        self._quit = False
        self._last_status_wall = 0.0
        self._current_targets: list = []
        self._timeline_window = "Study 2 grasp timeline"
        self._slider_ready = False
        self._updating_slider = False
        self._pending_noise_start: int | None = None

        self.vis = _WorkholdingSceneVis(f"Grasp Annotator — {participant}")
        v = self.vis.vis
        v.register_key_action_callback(_KEY_SPACE, self._on_toggle_play)
        v.register_key_action_callback(_KEY_LEFT, self._on_step_back)
        v.register_key_action_callback(_KEY_RIGHT, self._on_step_fwd)
        v.register_key_action_callback(_KEY_HOME, self._on_jump_start)
        v.register_key_action_callback(_KEY_END, self._on_jump_end)
        v.register_key_action_callback(_KEY_LBRACKET, self._on_slower)
        v.register_key_action_callback(_KEY_RBRACKET, self._on_faster)
        v.register_key_action_callback(ord("G"), self._on_mark_grasp)
        v.register_key_action_callback(ord("Z"), self._on_undo)
        v.register_key_action_callback(ord("E"), self._on_mark_release)
        v.register_key_action_callback(ord("T"), self._on_undo_release)
        v.register_key_action_callback(ord("D"), self._on_remove_pair)
        v.register_key_action_callback(ord("R"), self._on_restore_pair)
        v.register_key_action_callback(ord("C"), self._on_mark_noise)
        v.register_key_action_callback(ord("V"), self._on_undo_noise)
        v.register_key_action_callback(ord("H"), self._on_annotation_help)
        v.register_key_action_callback(ord("X"), self._on_mark_no_grasp)
        v.register_key_action_callback(ord("N"), self._on_next_trial)
        v.register_key_action_callback(ord("B"), self._on_prev_trial)
        v.register_key_action_callback(ord("S"), self._on_save_key)
        v.register_key_action_callback(ord("Q"), self._on_quit)
        v.register_key_action_callback(_KEY_ESCAPE, self._on_quit)

        print(__doc__)
        self._goto_trial(self._first_unannotated())
        self._create_timeline()

    def _nearest_frame(self, key: tuple, timestamp: float) -> int:
        frames = self.frames[key]
        return min(range(len(frames)), key=lambda i: abs(
            float(frames[i]["time_monotonic_s"]) - timestamp))

    def _event_frame_indices(self, key: tuple, event: str) -> list[int]:
        return [self._nearest_frame(key, float(record["time_monotonic_s"]))
                for record in self.records[key]
                if (record.get("event") == event
                    and record.get("recording", True)
                    and self.frames[key])]

    def _recorded_grasp_indices(self, key: tuple) -> list[int]:
        frames = self.frames[key]
        indices: list[int] = []
        was_grabbed = False
        for index, frame in enumerate(frames):
            grabbed = bool(frame.get("user_board_grabbed", False))
            if grabbed and not was_grabbed:
                indices.append(index)
            was_grabbed = grabbed
        indices.extend(self._event_frame_indices(key, "ar_handle_grabbed"))
        return sorted(set(indices))

    def _grasp_pairs(self, key: tuple) -> list[dict]:
        """Pair each displayed onset with the next available release."""
        entry = self.data["trials"].get(_trial_key_str(key), {})
        manual = [int(i) for i in entry.get("grasp_frame_indices", [])]
        recorded = self.recorded_grasps[key]
        suppressed = self._excluded_onset_indices(key)
        onsets = sorted(set(recorded + manual) - suppressed)
        releases = self._all_release_indices(key)
        pairs: list[dict] = []
        release_i = 0
        for onset in onsets:
            while release_i < len(releases) and releases[release_i] < onset:
                release_i += 1
            if release_i >= len(releases):
                break
            release = releases[release_i]
            release_i += 1
            pairs.append({
                "onset_frame_index": onset,
                "release_frame_index": release,
                "onset_source": "recorded" if onset in recorded else "manual",
            })
        return pairs

    def _manual_release_indices(self, key: tuple) -> list[int]:
        entry = self.data["trials"].get(_trial_key_str(key), {})
        return [int(index) for index in entry.get("release_frame_indices", [])]

    def _all_release_indices(self, key: tuple) -> list[int]:
        return sorted((set(self.release_indices[key]
                           + self._manual_release_indices(key))
                       - self._excluded_release_indices(key)))

    def _excluded_release_indices(self, key: tuple) -> set[int]:
        entry = self.data["trials"].get(_trial_key_str(key), {})
        return {
            int(release["frame_index"])
            for release in entry.get("excluded_release_events", [])
            if "frame_index" in release
        }

    @staticmethod
    def _pair_id(pair: dict) -> tuple[int, int]:
        return (int(pair["onset_frame_index"]),
                int(pair["release_frame_index"]))

    def _excluded_pair_ids(self, key: tuple) -> set[tuple[int, int]]:
        entry = self.data["trials"].get(_trial_key_str(key), {})
        return {
            (int(pair["onset_frame_index"]), int(pair["release_frame_index"]))
            for pair in entry.get("excluded_grasp_pairs", [])
            if "onset_frame_index" in pair and "release_frame_index" in pair
        }

    def _excluded_onset_indices(self, key: tuple) -> set[int]:
        """Return individually suppressed, audit-confirmed onset frames."""
        entry = self.data["trials"].get(_trial_key_str(key), {})
        return {
            int(onset["frame_index"])
            for onset in entry.get("excluded_grasp_onsets", [])
            if "frame_index" in onset
        }

    def _nearest_pair(self, *, excluded: bool) -> dict | None:
        key = self.keys[self.trial_i]
        excluded_ids = self._excluded_pair_ids(key)
        candidates = [
            pair for pair in self._grasp_pairs(key)
            if (self._pair_id(pair) in excluded_ids) == excluded
        ]
        if not candidates:
            return None

        def distance(pair: dict) -> int:
            start, end = self._pair_id(pair)
            if start <= self.frame_i <= end:
                return 0
            return min(abs(self.frame_i - start), abs(self.frame_i - end))

        return min(candidates, key=distance)

    def _control_states(self, key: tuple) -> list[str]:
        """Return the effective AR/freedrive channel at every replay frame."""
        frames = self.frames[key]
        if key[1] != "hybrid":
            return [key[1]] * len(frames)
        toggles = sorted(
            (float(record["time_monotonic_s"]), record.get("control_mode", "ar"))
            for record in self.records[key]
            if record.get("event") == "hybrid_control_mode_toggled")
        states: list[str] = []
        active = "ar"
        toggle_i = 0
        for frame in frames:
            timestamp = float(frame["time_monotonic_s"])
            while toggle_i < len(toggles) and toggles[toggle_i][0] <= timestamp:
                active = str(toggles[toggle_i][1])
                toggle_i += 1
            states.append(active)
        return states

    def _freedrive_intervals(self, key: tuple) -> list[tuple[int, int]]:
        """Recreate the study's TCP-speed freedrive interaction detector."""
        frames = self.frames[key]
        states = self.control_states[key]
        intervals: list[tuple[int, int]] = []
        moving = False
        start = 0
        still_since_t: float | None = None
        still_since_i: int | None = None
        previous: tuple[float, np.ndarray] | None = None
        for index, frame in enumerate(frames):
            if index >= len(states) or states[index] != "freedrive":
                if moving:
                    intervals.append((start, max(start, index - 1)))
                moving = False
                still_since_t = still_since_i = None
                previous = None
                continue
            tcp = _array(frame.get("tcp_world_T"), matrix=True)
            timestamp = float(frame.get("time_monotonic_s", 0.0))
            if tcp is None:
                previous = None
                continue
            if previous is not None:
                dt = timestamp - previous[0]
                if dt > 1e-3:
                    old = previous[1]
                    speed = float(np.linalg.norm(tcp[:3, 3] - old[:3, 3])) / dt
                    relative = old[:3, :3].T @ tcp[:3, :3]
                    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0,
                                           -1.0, 1.0))
                    angular_speed = float(np.degrees(np.arccos(cosine))) / dt
                    moving_fast = speed > 0.01 or angular_speed > 5.0
                    fully_still = speed < 0.004 and angular_speed < 2.0
                    if moving_fast:
                        if not moving:
                            start = index
                        moving = True
                        still_since_t = still_since_i = None
                    elif moving and fully_still:
                        if still_since_t is None:
                            still_since_t, still_since_i = timestamp, index
                        elif timestamp - still_since_t >= 0.10:
                            intervals.append((start, still_since_i or index))
                            moving = False
                            still_since_t = still_since_i = None
                    elif moving:
                        still_since_t = still_since_i = None
            previous = (timestamp, tcp)
        if moving and frames:
            intervals.append((start, len(frames) - 1))
        return intervals

    def _create_timeline(self) -> None:
        try:
            cv2.namedWindow(self._timeline_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._timeline_window, 1000, 170)
            maximum = max(1, len(self._current_frames()) - 1)
            cv2.createTrackbar(
                "Frame", self._timeline_window, self.frame_i, maximum,
                self._on_slider)
            self._slider_ready = True
            self._draw_timeline()
        except Exception as error:
            print(f"[Annotate] Timeline slider unavailable: {error}")

    def _on_slider(self, value: int) -> None:
        if not self._slider_ready or self._updating_slider:
            return
        self.playing = False
        self._set_frame(value, update_slider=False)

    def _sync_slider(self) -> None:
        """Move the UI thumb without treating it as a manual scrub."""
        if not self._slider_ready:
            return
        self._updating_slider = True
        try:
            cv2.setTrackbarPos(
                "Frame", self._timeline_window, self.frame_i)
        finally:
            self._updating_slider = False

    def _reset_slider_for_trial(self) -> None:
        """Resize and reset the slider without replaying its old position."""
        if not self._slider_ready:
            return
        self._updating_slider = True
        try:
            cv2.setTrackbarMax(
                "Frame", self._timeline_window,
                max(1, len(self._current_frames()) - 1))
            cv2.setTrackbarPos("Frame", self._timeline_window, 0)
        finally:
            self._updating_slider = False

    def _draw_timeline(self) -> None:
        if not self._slider_ready:
            return
        image = np.full((120, 1000, 3), 28, dtype=np.uint8)
        frames = self._current_frames()
        count = max(1, len(frames) - 1)
        x_of = lambda index: 20 + int(960 * index / count)
        cv2.line(image, (20, 65), (980, 65), (150, 150, 150), 2)
        key = self.keys[self.trial_i]
        state_colors = {"ar": (220, 110, 20), "freedrive": (0, 150, 255)}
        states = self.control_states[key]
        for index in range(max(0, len(states) - 1)):
            cv2.line(image, (x_of(index), 65), (x_of(index + 1), 65),
                     state_colors.get(states[index], (180, 80, 180)), 5)
        excluded_ids = self._excluded_pair_ids(key)
        excluded_onsets = ({pair[0] for pair in excluded_ids}
                           | self._excluded_onset_indices(key))
        excluded_releases = {pair[1] for pair in excluded_ids}
        excluded_releases |= self._excluded_release_indices(key)
        for index in self.release_indices[key]:
            if index in excluded_releases:
                continue
            cv2.line(image, (x_of(index), 48), (x_of(index), 82), (0, 0, 255), 2)
        for index in self._manual_release_indices(key):
            if index in excluded_releases:
                continue
            cv2.line(image, (x_of(index), 44), (x_of(index), 86),
                     (180, 80, 255), 3)
        for start, end in self.freedrive_intervals[key]:
            cv2.line(image, (x_of(start), 76), (x_of(end), 76),
                     (0, 230, 230), 7)
            cv2.line(image, (x_of(start), 69), (x_of(start), 84),
                     (0, 255, 255), 2)
            cv2.line(image, (x_of(end), 69), (x_of(end), 84),
                     (0, 255, 255), 2)
        for index in self.recorded_grasps[key]:
            if index in excluded_onsets:
                continue
            cv2.line(image, (x_of(index), 38), (x_of(index), 92), (0, 220, 0), 3)
        entry = self.data["trials"].get(_trial_key_str(key), {})
        for index in entry.get("grasp_frame_indices", []):
            if index in excluded_onsets:
                continue
            cv2.line(image, (x_of(index), 42), (x_of(index), 88), (255, 220, 0), 3)
        for start, end in sorted(excluded_ids):
            cv2.line(image, (x_of(start), 58), (x_of(end), 58),
                     (120, 120, 120), 3)
            cv2.line(image, (x_of(start), 48), (x_of(start), 68),
                     (180, 180, 180), 2)
            cv2.line(image, (x_of(end), 48), (x_of(end), 68),
                     (180, 180, 180), 2)
        noise_entry = self.noise_data["trials"].get(_trial_key_str(key), {})
        for interval in noise_entry.get("intervals", []):
            start = int(interval.get("start_frame_index", 0))
            end = int(interval.get("end_frame_index", start))
            cv2.line(image, (x_of(start), 34), (x_of(end), 34),
                     (220, 0, 220), 7)
            cv2.line(image, (x_of(start), 27), (x_of(start), 41),
                     (255, 80, 255), 2)
            cv2.line(image, (x_of(end), 27), (x_of(end), 41),
                     (255, 80, 255), 2)
        if self._pending_noise_start is not None:
            cv2.line(image, (x_of(self._pending_noise_start), 25),
                     (x_of(self._pending_noise_start), 44), (255, 80, 255), 3)
        cv2.circle(image, (x_of(self.frame_i), 65), 6, (255, 255, 255), -1)
        active = states[min(self.frame_i, len(states) - 1)] if states else key[1]
        label = (f"{self.participant} | condition={key[1]} | "
                 f"control={active} | trial {key[2]}")
        cv2.putText(image, label, (20, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(image,
                    "blue=AR  orange=freedrive  yellow=TCP start/span/end  "
                    "green=grab cyan=manual pink=manual-release red=release gray=removed",
                    (20, 108), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (215, 215, 215), 1, cv2.LINE_AA)
        cv2.imshow(self._timeline_window, image)

    # ── trial / frame navigation ────────────────────────────────────────

    def _first_unannotated(self) -> int:
        for i, key in enumerate(self.keys):
            if key[1] == "freedrive":
                continue
            if not self.data["trials"].get(_trial_key_str(key), {}).get("reviewed"):
                return i
        return 0

    def _goto_trial(self, index: int) -> None:
        self.trial_i = max(0, min(index, len(self.keys) - 1))
        self.frame_i = 0
        self.playing = False
        self._pending_noise_start = None
        key = self.keys[self.trial_i]
        metadata = self.metadata_by_session.get(key[0], {})
        targets = [_array(T, matrix=True)
                   for T in metadata.get("target_poses", [])]
        self._current_targets = targets
        if targets:
            frames = self.frames[key]
            visible_target = next(
                (frame.get("pose_idx") for frame in frames
                 if isinstance(frame.get("pose_idx"), int)), None)
            if visible_target is None and 0 <= key[2] < len(targets):
                visible_target = key[2]
            self.vis.configure_target_ghosts(
                targets, visible_index=visible_target)
        lo, hi = (_array(metadata.get("workspace_lo")),
                  _array(metadata.get("workspace_hi")))
        if lo is not None and hi is not None:
            self.vis.update_workspace_bound(lo, hi)
        self._render_frame()
        self._print_trial_header()
        if self._slider_ready:
            self._reset_slider_for_trial()
            self._draw_timeline()

    def _print_trial_header(self) -> None:
        key = self.keys[self.trial_i]
        session_id, mode, trial_idx = key
        n_frames = len(self.frames[key])
        summary = self.summaries[key]
        duration = f"{summary.get('duration_s'):.2f}s" if summary else "n/a"
        entry = self.data["trials"].get(_trial_key_str(key), {})
        grasps = entry.get("grasp_times_monotonic_s", [])
        status = ("no grasp marked" if entry.get("no_grasp")
                  else f"{len(grasps)} grasp(s)" if grasps else "not yet reviewed")
        print(f"\n=== Trial {self.trial_i + 1}/{len(self.keys)} "
              f"— {mode} #{trial_idx} (session {session_id}) "
              f"— {n_frames} frames, duration {duration} — {status} ===")

    def _on_next_trial(self, _vis, action, _mods) -> bool:
        if action == 1:
            self._save()
            self._goto_trial(self.trial_i + 1)
        return False

    def _on_prev_trial(self, _vis, action, _mods) -> bool:
        if action == 1:
            self._save()
            self._goto_trial(self.trial_i - 1)
        return False

    # ── playback ─────────────────────────────────────────────────────────

    def _current_frames(self) -> list[dict]:
        return self.frames[self.keys[self.trial_i]]

    def _on_toggle_play(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        frames = self._current_frames()
        if not frames:
            return False
        self.playing = not self.playing
        if self.playing:
            self._play_anchor_wall = time.perf_counter()
            self._play_anchor_t = float(
                frames[self.frame_i]["time_monotonic_s"])
        return False

    def _on_step_back(self, _vis, action, _mods) -> bool:
        if action in (1, 2):
            self.playing = False
            self._set_frame(self.frame_i - 1)
        return False

    def _on_step_fwd(self, _vis, action, _mods) -> bool:
        if action in (1, 2):
            self.playing = False
            self._set_frame(self.frame_i + 1)
        return False

    def _on_jump_start(self, _vis, action, _mods) -> bool:
        if action == 1:
            self.playing = False
            self._set_frame(0)
        return False

    def _on_jump_end(self, _vis, action, _mods) -> bool:
        if action == 1:
            self.playing = False
            self._set_frame(len(self._current_frames()) - 1)
        return False

    def _on_slower(self, _vis, action, _mods) -> bool:
        if action == 1:
            self.play_speed = max(0.1, self.play_speed / 2.0)
            print(f"[Annotate] speed = {self.play_speed:.2f}x")
        return False

    def _on_faster(self, _vis, action, _mods) -> bool:
        if action == 1:
            self.play_speed = min(8.0, self.play_speed * 2.0)
            print(f"[Annotate] speed = {self.play_speed:.2f}x")
        return False

    def _set_frame(self, index: int, *, update_slider: bool = True) -> None:
        frames = self._current_frames()
        if not frames:
            return
        self.frame_i = max(0, min(index, len(frames) - 1))
        self._render_frame()
        if self._slider_ready and update_slider:
            self._sync_slider()
        self._draw_timeline()
        self._print_status()

    def _advance_playback(self) -> None:
        if not self.playing:
            return
        frames = self._current_frames()
        target_t = self._play_anchor_t + (
            (time.perf_counter() - self._play_anchor_wall) * self.play_speed)
        advanced = False
        while (self.frame_i < len(frames) - 1
               and float(frames[self.frame_i + 1]["time_monotonic_s"]) <= target_t):
            self.frame_i += 1
            advanced = True
        if self.frame_i >= len(frames) - 1:
            self.playing = False
        if advanced:
            self._render_frame()
            if self._slider_ready:
                self._sync_slider()
            self._draw_timeline()
        now = time.perf_counter()
        if now - self._last_status_wall > 0.5:
            self._last_status_wall = now
            self._print_status()

    # ── grasp marking ────────────────────────────────────────────────────

    def _entry(self) -> dict:
        key = self.keys[self.trial_i]
        entry = self.data["trials"].setdefault(_trial_key_str(key), {})
        # Audit-only entries may already exist without the manual-annotation
        # fields. Backfill every required key instead of relying on setdefault
        # to insert the entire dictionary only for brand-new trials.
        defaults = {
            "session_id": key[0],
            "mode": key[1],
            "trial_idx": key[2],
            "grasp_times_monotonic_s": [],
            "grasp_frame_indices": [],
            "release_times_monotonic_s": [],
            "release_frame_indices": [],
            "no_grasp": False,
            "reviewed": False,
        }
        for field, value in defaults.items():
            entry.setdefault(field, value)
        return entry

    def _on_mark_grasp(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        if self.keys[self.trial_i][1] == "freedrive":
            print("[Annotate] Freedrive has no virtual-handle grasp to mark.")
            return False
        frames = self._current_frames()
        if not frames:
            return False
        t = float(frames[self.frame_i]["time_monotonic_s"])
        entry = self._entry()
        entry["no_grasp"] = False
        entry["reviewed"] = True
        entry["grasp_times_monotonic_s"].append(t)
        entry["grasp_frame_indices"].append(self.frame_i)
        entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[Annotate] grasp marked @ t={t:.3f}s "
              f"(frame {self.frame_i}/{len(frames) - 1})")
        self._save()
        self._draw_timeline()
        return False

    def _on_undo(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        entry = self._entry()
        if entry["grasp_times_monotonic_s"]:
            removed = entry["grasp_times_monotonic_s"].pop()
            entry["grasp_frame_indices"].pop()
            print(f"[Annotate] removed grasp mark @ t={removed:.3f}s")
            self._save()
            self._draw_timeline()
        return False

    def _on_mark_release(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        if self.keys[self.trial_i][1] == "freedrive":
            print("[Annotate] Freedrive has no virtual-handle release to mark.")
            return False
        frames = self._current_frames()
        if not frames:
            return False
        timestamp = float(frames[self.frame_i]["time_monotonic_s"])
        entry = self._entry()
        entry["no_grasp"] = False
        entry["reviewed"] = True
        entry["release_times_monotonic_s"].append(timestamp)
        entry["release_frame_indices"].append(self.frame_i)
        entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        self._draw_timeline()
        print(f"[Annotate] manual release marked @ t={timestamp:.3f}s "
              f"(frame {self.frame_i}/{len(frames) - 1})")
        return False

    def _on_undo_release(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        entry = self._entry()
        if entry["release_times_monotonic_s"]:
            removed = entry["release_times_monotonic_s"].pop()
            entry["release_frame_indices"].pop()
            entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            self._draw_timeline()
            print(f"[Annotate] removed manual release @ t={removed:.3f}s")
        else:
            print("[Annotate] no manual release to undo in this trial")
        return False

    def _on_remove_pair(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        if self.keys[self.trial_i][1] == "freedrive":
            print("[Annotate] Freedrive has no AR grasp pair to remove.")
            return False
        pair = self._nearest_pair(excluded=False)
        if pair is None:
            print("[Annotate] No complete onset/release pair in this trial.")
            return False
        entry = self._entry()
        excluded = entry.setdefault("excluded_grasp_pairs", [])
        start, end = self._pair_id(pair)
        frames = self._current_frames()
        excluded.append({
            **pair,
            "onset_time_monotonic_s": float(
                frames[start]["time_monotonic_s"]),
            "release_time_monotonic_s": float(
                frames[end]["time_monotonic_s"]),
            "excluded_at": datetime.now(timezone.utc).isoformat(),
        })
        entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        self._draw_timeline()
        print(f"[Annotate] removed pair frames {start}-{end}; "
              f"saved non-destructively -> {self.out_path}")
        return False

    def _on_restore_pair(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        pair = self._nearest_pair(excluded=True)
        if pair is None:
            print("[Annotate] No excluded pair in this trial to restore.")
            return False
        entry = self._entry()
        pair_id = self._pair_id(pair)
        entry["excluded_grasp_pairs"] = [
            saved for saved in entry.get("excluded_grasp_pairs", [])
            if (int(saved.get("onset_frame_index", -1)),
                int(saved.get("release_frame_index", -1))) != pair_id
        ]
        entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        self._draw_timeline()
        print(f"[Annotate] restored pair frames {pair_id[0]}-{pair_id[1]}")
        return False

    def _noise_entry(self) -> dict:
        key = self.keys[self.trial_i]
        entry = self.noise_data["trials"].setdefault(_trial_key_str(key), {})
        entry.setdefault("session_id", key[0])
        entry.setdefault("mode", key[1])
        entry.setdefault("trial_idx", key[2])
        entry.setdefault("intervals", [])
        return entry

    def _on_mark_noise(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        frames = self._current_frames()
        if not frames:
            return False
        if self._pending_noise_start is None:
            self._pending_noise_start = self.frame_i
            timestamp = float(frames[self.frame_i]["time_monotonic_s"])
            print(f"[Annotate] orientation-noise START @ frame {self.frame_i}, "
                  f"t={timestamp:.3f}s; move slider and press C for END")
        else:
            start, end = sorted((self._pending_noise_start, self.frame_i))
            entry = self._noise_entry()
            entry["intervals"].append({
                "start_frame_index": start,
                "end_frame_index": end,
                "start_time_monotonic_s": float(
                    frames[start]["time_monotonic_s"]),
                "end_time_monotonic_s": float(frames[end]["time_monotonic_s"]),
                "reason": "orientation_confusion",
                "treatment": "exclude_as_noise",
                "annotated_at": datetime.now(timezone.utc).isoformat(),
            })
            entry["annotation_note"] = (
                "Participant manipulated the board with an incorrect/confused "
                "orientation; interval marked manually for sensitivity analysis.")
            self._pending_noise_start = None
            self._save_noise()
            print(f"[Annotate] orientation-noise interval saved: frames "
                  f"{start}-{end} -> {self.noise_path}")
        self._draw_timeline()
        return False

    def _on_undo_noise(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        if self._pending_noise_start is not None:
            self._pending_noise_start = None
            print("[Annotate] pending orientation-noise start cancelled")
        else:
            entry = self._noise_entry()
            intervals = entry.get("intervals", [])
            if intervals:
                removed = intervals.pop()
                self._save_noise()
                print("[Annotate] removed orientation-noise interval frames "
                      f"{removed['start_frame_index']}-"
                      f"{removed['end_frame_index']}")
            else:
                print("[Annotate] no orientation-noise interval to undo")
        self._draw_timeline()
        return False

    def _on_annotation_help(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        print("""
=== Study 2 annotation keys ===
G  mark manual grasp onset at the current frame
Z  undo the last manual grasp onset in the current trial
E  mark a manual release at the current frame
T  undo the last manual release in the current trial
D  remove/exclude the grasp-onset/release pair nearest the slider
R  restore the excluded grasp-onset/release pair nearest the slider
C  mark orientation-noise START; press C again to mark END
V  cancel a pending noise start, or undo the last noise interval
X  mark the current trial reviewed with no visible manual grasp
N  save and go to the next trial
B  save and go to the previous trial
S  save annotations now
H  print this help
Q / ESC  save and quit

Playback: SPACE play/pause, LEFT/RIGHT step, HOME/END jump, [/] speed
===============================
""")
        return False

    def _on_mark_no_grasp(self, _vis, action, _mods) -> bool:
        if action != 1:
            return False
        entry = self._entry()
        entry["grasp_times_monotonic_s"] = []
        entry["grasp_frame_indices"] = []
        entry["release_times_monotonic_s"] = []
        entry["release_frame_indices"] = []
        entry["no_grasp"] = True
        entry["reviewed"] = True
        entry["annotated_at"] = datetime.now(timezone.utc).isoformat()
        print("[Annotate] trial marked: no visible manual grasp")
        self._save()
        return False

    def _on_save_key(self, _vis, action, _mods) -> bool:
        if action == 1:
            self._save()
            print(f"[Annotate] saved -> {self.out_path}")
        return False

    def _on_quit(self, _vis, action, _mods) -> bool:
        if action == 1:
            self._quit = True
        return False

    def _save(self) -> None:
        _save_annotations(self.out_path, self.data)
        self._save_noise()

    def _save_noise(self) -> None:
        _save_annotations(self.noise_path, self.noise_data)

    def _print_status(self) -> None:
        frames = self._current_frames()
        if not frames:
            return
        t0 = float(frames[0]["time_monotonic_s"])
        t = float(frames[self.frame_i]["time_monotonic_s"]) - t0
        state = "playing" if self.playing else "paused"
        print(f"[Annotate] frame {self.frame_i}/{len(frames) - 1} "
              f"t={t:.3f}s ({state}, {self.play_speed:.2f}x)")

    # ── rendering ────────────────────────────────────────────────────────

    def _render_frame(self) -> None:
        frames = self._current_frames()
        if not frames:
            return
        record = frames[self.frame_i]
        tcp = _array(record.get("tcp_world_T"), matrix=True)
        head = _array(record.get("head_world_T"), matrix=True)
        left = _array(record.get("left_hand_world"))
        right = _array(record.get("right_hand_world"))
        links_raw = record.get("robot_link_world_T")
        links = ([_array(T, matrix=True) for T in links_raw]
                 if links_raw is not None else None)
        self.vis.update_head(head)
        self.vis.update_hands(left, right)
        self.vis.update_tcp(tcp)
        if links is not None:
            self.vis.update_robot(links)
        held = record.get("robot_board_state") in {
            "holding_board", "moving_board", "release_armed"}
        self.vis.set_tcp_gripper_closed(held)
        self.vis.update_board_ar_from_tcp(tcp, cfg.BOX_FORWARD_OFFSET)
        pose_idx = record.get("pose_idx")
        state = record.get("target_color_state", "far")
        if (self._current_targets and isinstance(pose_idx, int)
                and 0 <= pose_idx < len(self._current_targets)):
            self.vis.select_target(pose_idx, state)
            self.vis.update_target_gripper(
                self._current_targets[pose_idx], cfg.BOX_FORWARD_OFFSET, state)
        virtual_board = _array(
            record.get("user_manipulated_board_world_T"), matrix=True)
        if virtual_board is None:
            virtual_board = _array(record.get("board_world_T"), matrix=True)
        self.vis.update_ar_handle(
            virtual_board if record.get("ar_enabled") else None)
        key = self.keys[self.trial_i]
        states = self.control_states[key]
        active = states[self.frame_i] if self.frame_i < len(states) else key[1]
        # A hybrid trial is identified in the timeline label; the sphere shows
        # which control channel is active at this exact frame.
        self.vis.update_mode_indicator(tcp, active)

    # ── main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while not self._quit:
                self._advance_playback()
                alive = self.vis.vis.poll_events()
                self.vis.vis.update_renderer()
                if self._slider_ready:
                    cv2.waitKey(1)
                if not alive:
                    break
                time.sleep(0.005)
        finally:
            self._save()
            self.vis.close()
            if self._slider_ready:
                cv2.destroyWindow(self._timeline_window)
            reviewed = sum(1 for e in self.data["trials"].values()
                          if e.get("reviewed"))
            print(f"\n[Annotate] saved {reviewed}/{len(self.keys)} trials "
                  f"reviewed -> {self.out_path}")
            print(f"[Annotate] orientation-noise intervals -> {self.noise_path}")


def _resolve_log_path(arg: str) -> tuple[str, Path]:
    candidate = Path(arg)
    if candidate.is_file():
        stem = candidate.stem
        participant = stem[:-7] if stem.endswith("_replay") else stem
        return participant, candidate
    participant = arg
    path = _LOG_DIR / f"{participant}_replay.jsonl"
    if not path.is_file():
        raise SystemExit(f"No replay log found for {arg!r} (tried {path})")
    return participant, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "participant",
        help="Participant name (e.g. mahya, pantea, parisa) or a replay log path")
    args = parser.parse_args()

    participant, path = _resolve_log_path(args.participant)
    annotator = _Annotator(participant, path)
    annotator.run()


if __name__ == "__main__":
    main()
