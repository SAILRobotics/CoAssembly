"""Standalone Study 2 efficiency analysis, cut off at first target reach.

Hybrid is attributed to AR, freedrive, or both according to the interaction
channels actually used. Later reinforcement and snap-confirmation are excluded.

Example:
  python3 study2_replay_to_csv.py study_logs/study2/*_replay.jsonl --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ("freedrive", "ar")
TRIAL_MODES = (*METHODS, "hybrid")
METRICS = (
    ("Right-hand movement (m)", "hand_movement_m"),
    ("Head translation before reach (m)", "head_translation_m"),
    ("Head rotation before reach (deg)", "head_rotation_deg"),
    ("Translation error reduction (m)", "translation_error_reduction_m"),
    ("Rotation error reduction (deg)", "rotation_error_reduction_deg"),
    ("Translation reduction / hand movement (m/m)", "translation_per_hand_m"),
    ("Rotation reduction / hand movement (deg/m)", "rotation_per_hand_m"),
    ("Interaction time (s)", "interaction_time_s"),
    ("Translation reduction / interaction time (m/s)", "translation_per_interaction_s"),
    ("Rotation reduction / interaction time (deg/s)", "rotation_per_interaction_s"),
    ("Interaction time / completion time", "interaction_completion_ratio"),
    ("Completion time to first reach (s)", "completion_time_s"),
    ("Number of interactions (#)", "num_interactions"),
)


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def matrix(value):
    try:
        value = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    return value if np.all(np.isfinite(value)) else None


def hand_centroid(value):
    points = []
    if isinstance(value, list):
        for item in value:
            try:
                point = np.asarray(item, dtype=float).reshape(-1)[:3]
            except (TypeError, ValueError):
                continue
            if point.size == 3 and np.all(np.isfinite(point)):
                points.append(point)
    return np.mean(points, axis=0) if points else None


def pose_error(current, target):
    current, target = matrix(current), matrix(target)
    if current is None or target is None:
        return None
    position = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    relative = current[:3, :3].T @ target[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return position, float(np.degrees(np.arccos(cosine)))


def nearest_frame(frames, timestamp):
    return min(range(len(frames)), key=lambda i: abs(
        number(frames[i].get("time_monotonic_s")) - timestamp))


def hand_path(frames, start, end, max_gap, max_step):
    samples = []
    for frame in frames[start:end + 1]:
        point = hand_centroid(frame.get("right_hand_world"))
        timestamp = number(frame.get("time_monotonic_s"))
        if point is not None and math.isfinite(timestamp):
            samples.append((timestamp, point))
    distance = tracked = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt, step = t1 - t0, float(np.linalg.norm(p1 - p0))
        if 0 < dt <= max_gap and step <= max_step:
            distance += step
            tracked += dt
    return distance, tracked


def head_movement(frames, max_gap, max_step, max_angle_step):
    """Cumulative head translation and rotation over the pre-reach trial."""
    samples = []
    for frame in frames:
        pose = matrix(frame.get("head_world_T"))
        timestamp = number(frame.get("time_monotonic_s"))
        if pose is not None and math.isfinite(timestamp):
            samples.append((timestamp, pose))
    translation = rotation = tracked = 0.0
    for (t0, first), (t1, second) in zip(samples, samples[1:]):
        dt = t1 - t0
        if not 0 < dt <= max_gap:
            continue
        step = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
        relative = first[:3, :3].T @ second[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        angle = float(np.degrees(np.arccos(cosine)))
        if step <= max_step and angle <= max_angle_step:
            translation += step
            rotation += angle
            tracked += dt
    return translation, rotation, tracked


def participant_name(path, rows):
    named = next((row.get("session_name") for row in rows
                  if row.get("session_name")), None)
    if named:
        return str(named)
    return path.stem[:-7] if path.stem.endswith("_replay") else path.stem


def annotations_for(path, participant):
    edited = path.parent / f"{participant}_grasp_annotations_edited.json"
    original = path.parent / f"{participant}_grasp_annotations.json"
    selected = edited if edited.is_file() else original
    if not selected.is_file():
        return {}
    try:
        return json.loads(selected.read_text()).get("trials", {})
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: cannot read {selected}: {error}")
        return {}


def orientation_noise_for(path, participant):
    selected = path.parent / f"{participant}_orientation_noise_annotations.json"
    if not selected.is_file():
        return {}
    try:
        return json.loads(selected.read_text()).get("trials", {})
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: cannot read {selected}: {error}")
        return {}


def subtract_intervals(intervals, excluded):
    """Remove inclusive excluded spans without bridging hand paths across them."""
    output = []
    for start, end in intervals:
        pieces = [(start, end)]
        for cut_start, cut_end in excluded:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_end < piece_start or cut_start > piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < cut_start:
                    next_pieces.append((piece_start, cut_start - 1))
                if cut_end < piece_end:
                    next_pieces.append((cut_end + 1, piece_end))
            pieces = next_pieces
        output.extend((start, end) for start, end in pieces if start <= end)
    return output


def freedrive_noise_rotation_penalty(frames, intervals, noise_intervals):
    """Magnitude of negative rotation-error changes inside marked noise."""
    penalty = 0.0
    for move_start, move_end in intervals:
        for noise_start, noise_end in noise_intervals:
            start, end = max(move_start, noise_start), min(move_end, noise_end)
            for index in range(start + 1, end + 1):
                target = frames[index].get("target_board_world_T")
                before = pose_error(frames[index - 1].get("board_world_T"), target)
                after = pose_error(frames[index].get("board_world_T"), target)
                if before and after:
                    reduction = before[1] - after[1]
                    if reduction < 0:
                        penalty += -reduction
    return penalty


def control_states(key, frames, records):
    if key[1] != "hybrid":
        return [key[1]] * len(frames)
    toggles = sorted((number(row.get("time_monotonic_s")),
                      str(row.get("control_mode", "ar")))
                     for row in records
                     if row.get("event") == "hybrid_control_mode_toggled")
    states, active, toggle_i = [], "ar", 0
    for frame in frames:
        timestamp = number(frame.get("time_monotonic_s"))
        while toggle_i < len(toggles) and toggles[toggle_i][0] <= timestamp:
            active = toggles[toggle_i][1]
            toggle_i += 1
        states.append(active)
    return states


def freedrive_intervals(frames, states):
    """Recreate logged physical movement segments with 0.10 s stop dwell."""
    intervals, moving, start, previous = [], False, 0, None
    still_t = still_i = None
    for i, frame in enumerate(frames):
        if states[i] != "freedrive":
            if moving:
                intervals.append((start, max(start, i - 1)))
            moving, previous, still_t, still_i = False, None, None, None
            continue
        tcp, timestamp = matrix(frame.get("tcp_world_T")), number(frame.get("time_monotonic_s"))
        if tcp is None or not math.isfinite(timestamp):
            previous = None
            continue
        if previous is not None and timestamp > previous[0] + 1e-3:
            dt, old = timestamp - previous[0], previous[1]
            speed = float(np.linalg.norm(tcp[:3, 3] - old[:3, 3])) / dt
            relative = old[:3, :3].T @ tcp[:3, :3]
            cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
            angular = float(np.degrees(np.arccos(cosine))) / dt
            starts = speed > .01 or angular > 5
            stops = speed < .004 and angular < 2
            if starts:
                if not moving:
                    start = i
                moving, still_t, still_i = True, None, None
            elif moving and stops:
                if still_t is None:
                    still_t, still_i = timestamp, i
                elif timestamp - still_t >= .10:
                    intervals.append((start, still_i or i))
                    moving, still_t, still_i = False, None, None
            elif moving:
                still_t = still_i = None
        previous = timestamp, tcp
    if moving:
        intervals.append((start, len(frames) - 1))
    return intervals


def handle_distance(frame):
    board = matrix(frame.get("user_manipulated_board_world_T"))
    if board is None:
        board = matrix(frame.get("board_world_T"))
    if board is None:
        return None
    handle = board[:3, 3] + board[:3, :3] @ np.array([0., -.09, 0.])
    hand = frame.get("right_hand_world")
    distances = []
    if isinstance(hand, list):
        for item in hand:
            try:
                point = np.asarray(item, dtype=float).reshape(-1)[:3]
            except (TypeError, ValueError):
                continue
            if point.size == 3 and np.all(np.isfinite(point)):
                distances.append(float(np.linalg.norm(point - handle)))
    return min(distances) if distances else None


def infer_onset(frames, release, lower, radius, still_speed, still_window):
    contacts = []
    for i in range(lower, release + 1):
        separation = handle_distance(frames[i])
        if separation is not None and separation <= radius:
            contacts.append(i)
    if contacts:
        start = contacts[-1]
        for i in reversed(contacts[:-1]):
            if i != start - 1:
                break
            start = i
        return start
    still, start = 0., release
    for i in range(release, lower, -1):
        p0 = hand_centroid(frames[i - 1].get("right_hand_world"))
        p1 = hand_centroid(frames[i].get("right_hand_world"))
        t0 = number(frames[i - 1].get("time_monotonic_s"))
        t1 = number(frames[i].get("time_monotonic_s"))
        if p0 is None or p1 is None or not 0 < t1 - t0 <= .25:
            still = 0
        elif float(np.linalg.norm(p1 - p0)) / (t1 - t0) < still_speed:
            still += t1 - t0
            start = i - 1
            if still >= still_window:
                break
        else:
            still = 0
    return start


def ar_intervals(key, frames, records, annotations, radius, still_speed, still_window):
    onsets, grabbed = [], False
    for i, frame in enumerate(frames):
        current = bool(frame.get("user_board_grabbed", False))
        if current and not grabbed:
            onsets.append(i)
        grabbed = current
    onsets += [nearest_frame(frames, number(row.get("time_monotonic_s")))
               for row in records if row.get("event") == "ar_handle_grabbed"
               and row.get("recording", True)]
    entry = annotations.get(f"{key[0]}|{key[1]}|{key[2]}", {})
    onsets += [int(i) for i in entry.get("grasp_frame_indices", [])]
    suppressed = {int(item["frame_index"])
                  for item in entry.get("excluded_grasp_onsets", [])
                  if "frame_index" in item}
    onsets = sorted(set(onsets) - suppressed)
    releases = {
        nearest_frame(frames, number(row.get("time_monotonic_s")))
        for row in records
        if row.get("event") == "ar_handle_released"
        and row.get("recording", True)
        and number(row.get("time_monotonic_s"))
        <= number(frames[-1].get("time_monotonic_s"))
    }
    releases.update(int(index) for index in entry.get(
        "release_frame_indices", []) if 0 <= int(index) < len(frames))
    releases -= {int(item["frame_index"])
                 for item in entry.get("excluded_release_events", [])
                 if "frame_index" in item}
    releases = sorted(releases)
    excluded = {(int(item["onset_frame_index"]), int(item["release_frame_index"]))
                for item in entry.get("excluded_grasp_pairs", [])
                if "onset_frame_index" in item and "release_frame_index" in item}
    intervals, previous = [], -1
    for release in releases:
        candidates = [i for i in onsets if previous < i <= release]
        onset = candidates[-1] if candidates else infer_onset(
            frames, release, previous + 1, radius, still_speed, still_window)
        if (onset, release) not in excluded:
            intervals.append((onset, release))
        previous = release
    return intervals


def interval_metrics(frames, intervals, method, noise_intervals,
                     max_gap, max_step):
    result = dict(hand_movement_m=0., hand_tracked_s=0., interaction_time_s=0.,
                  num_interactions=len(intervals), translation_error_reduction_m=0.,
                  rotation_error_reduction_deg=0.)
    raw_hand = 0.0
    for start, end in intervals:
        distance, tracked = hand_path(frames, start, end, max_gap, max_step)
        raw_hand += distance
        result["interaction_time_s"] += max(
            0., number(frames[end].get("time_monotonic_s"))
            - number(frames[start].get("time_monotonic_s")))
        before_pose = frames[start].get("board_world_T")
        after_pose = (frames[end].get("user_manipulated_board_world_T")
                      if method == "ar" else frames[end].get("board_world_T"))
        after_pose = after_pose or frames[end].get("board_world_T")
        target = frames[end].get("target_board_world_T")
        before, after = pose_error(before_pose, target), pose_error(after_pose, target)
        if before and after:
            result["translation_error_reduction_m"] += before[0] - after[0]
            result["rotation_error_reduction_deg"] += before[1] - after[1]
    hand_intervals = (subtract_intervals(intervals, noise_intervals)
                      if method == "ar" else intervals)
    for start, end in hand_intervals:
        distance, tracked = hand_path(frames, start, end, max_gap, max_step)
        result["hand_movement_m"] += distance
        result["hand_tracked_s"] += tracked
    result["raw_hand_movement_m"] = raw_hand
    result["orientation_noise_hand_removed_m"] = (
        raw_hand - result["hand_movement_m"])
    result["raw_rotation_error_reduction_deg"] = result[
        "rotation_error_reduction_deg"]
    penalty = (freedrive_noise_rotation_penalty(
        frames, intervals, noise_intervals) if method == "freedrive" else 0.0)
    result["orientation_noise_rotation_penalty_removed_deg"] = penalty
    result["rotation_error_reduction_deg"] += penalty
    return result


def merge_intervals(intervals):
    """Return the union of intervals so Hybrid time/path is not double-counted."""
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(span) for span in merged]


def condition_metrics(frames, intervals, noise_intervals, start_index,
                      max_gap, max_step):
    """Whole-trial effectiveness for Freedrive-only, AR-only, or Hybrid."""
    combined = merge_intervals([
        span for method_spans in intervals.values() for span in method_spans
    ])
    raw_hand_movement = 0.0
    for start, end in combined:
        distance, _tracked = hand_path(frames, start, end, max_gap, max_step)
        raw_hand_movement += distance
    adjusted_hand_intervals = merge_intervals(
        intervals["freedrive"]
        + subtract_intervals(intervals["ar"], noise_intervals))
    hand_movement = hand_tracked = interaction_time = 0.0
    for start, end in adjusted_hand_intervals:
        distance, tracked = hand_path(frames, start, end, max_gap, max_step)
        hand_movement += distance
        hand_tracked += tracked
    for start, end in combined:
        interaction_time += max(
            0.0, number(frames[end].get("time_monotonic_s"))
            - number(frames[start].get("time_monotonic_s")))
    target = frames[-1].get("target_board_world_T")
    before = pose_error(frames[start_index].get("board_world_T"), target)
    after = pose_error(frames[-1].get("board_world_T"), target)
    translation = before[0] - after[0] if before and after else float("nan")
    raw_rotation = before[1] - after[1] if before and after else float("nan")
    penalty = freedrive_noise_rotation_penalty(
        frames, intervals["freedrive"], noise_intervals)
    rotation = raw_rotation + penalty
    return {
        "hand_movement_m": hand_movement,
        "hand_tracked_s": hand_tracked,
        "interaction_time_s": interaction_time,
        "num_interactions": sum(len(spans) for spans in intervals.values()),
        "translation_error_reduction_m": translation,
        "rotation_error_reduction_deg": rotation,
        "raw_hand_movement_m": raw_hand_movement,
        "orientation_noise_hand_removed_m": raw_hand_movement - hand_movement,
        "raw_rotation_error_reduction_deg": raw_rotation,
        "orientation_noise_rotation_penalty_removed_deg": penalty,
        "ar_interactions": len(intervals["ar"]),
        "freedrive_interactions": len(intervals["freedrive"]),
    }


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def load(paths, args):
    method_output, condition_output = [], []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Replay log not found: {path}")
        trials, all_rows = defaultdict(list), []
        for line_number, line in enumerate(path.open(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Warning: {path}:{line_number}: {error}")
                continue
            all_rows.append(row)
            if row.get("mode") in TRIAL_MODES and isinstance(row.get("trial_idx"), int):
                trials[(str(row.get("session_id", "")), row["mode"], row["trial_idx"])].append(row)
        participant = participant_name(path, all_rows)
        annotations = annotations_for(path, participant)
        orientation_noise = orientation_noise_for(path, participant)
        for key, records in trials.items():
            frames = [row for row in records if row.get("type") == "frame"
                      and row.get("timer_running")]
            reach = next((row for row in records
                          if row.get("event") == "target_first_reached"), None)
            if not frames or not reach:
                continue
            cutoff = number(reach.get("time_monotonic_s"))
            frames = [row for row in frames
                      if number(row.get("time_monotonic_s")) <= cutoff]
            if not frames:
                continue
            states = control_states(key, frames, records)
            intervals = {
                "freedrive": freedrive_intervals(frames, states),
                "ar": ar_intervals(key, frames, records, annotations,
                                   args.contact_radius, args.still_speed,
                                   args.still_window),
            }
            noise_entry = orientation_noise.get(
                f"{key[0]}|{key[1]}|{key[2]}", {})
            noise_intervals = [
                (max(0, int(item["start_frame_index"])),
                 min(len(frames) - 1, int(item["end_frame_index"])))
                for item in noise_entry.get("intervals", [])
                if "start_frame_index" in item and "end_frame_index" in item
                and int(item["start_frame_index"]) < len(frames)
            ]
            for method, spans in intervals.items():
                if not spans:
                    continue
                start_index = min(start for start, _end in spans)
                start_time = number(frames[start_index].get("time_monotonic_s"))
                completion = max(0.0, cutoff - start_time)
                head_translation, head_rotation, head_tracked = head_movement(
                    frames[start_index:], args.max_gap, args.max_head_step,
                    args.max_head_angle_step)
                values = interval_metrics(frames, spans, method, noise_intervals,
                                          args.max_gap, args.max_hand_step)
                values.update(participant=participant, session_id=key[0],
                              trial_condition=key[1], trial_idx=key[2],
                              trial_block=("first_5" if key[2] < 5 else "last_3"),
                              method=method, completion_time_s=completion,
                              analysis_start_frame=start_index,
                              analysis_start_time_monotonic_s=start_time,
                              head_translation_m=head_translation,
                              head_rotation_deg=head_rotation,
                              head_tracked_s=head_tracked)
                values["interaction_completion_ratio"] = ratio(
                    values["interaction_time_s"], completion)
                values["translation_per_hand_m"] = ratio(
                    values["translation_error_reduction_m"], values["hand_movement_m"])
                values["rotation_per_hand_m"] = ratio(
                    values["rotation_error_reduction_deg"], values["hand_movement_m"])
                values["translation_per_interaction_s"] = ratio(
                    values["translation_error_reduction_m"], values["interaction_time_s"])
                values["rotation_per_interaction_s"] = ratio(
                    values["rotation_error_reduction_deg"], values["interaction_time_s"])
                method_output.append(values)
            combined_spans = [span for spans in intervals.values() for span in spans]
            if not combined_spans:
                continue
            start_index = min(start for start, _end in combined_spans)
            start_time = number(frames[start_index].get("time_monotonic_s"))
            completion = max(0.0, cutoff - start_time)
            head_translation, head_rotation, head_tracked = head_movement(
                frames[start_index:], args.max_gap, args.max_head_step,
                args.max_head_angle_step)
            values = condition_metrics(
                frames, intervals, noise_intervals, start_index,
                args.max_gap, args.max_hand_step)
            values.update(participant=participant, session_id=key[0],
                          trial_condition=key[1], trial_idx=key[2],
                          trial_block=("first_5" if key[2] < 5 else "last_3"),
                          completion_time_s=completion,
                          analysis_start_frame=start_index,
                          analysis_start_time_monotonic_s=start_time,
                          head_translation_m=head_translation,
                          head_rotation_deg=head_rotation,
                          head_tracked_s=head_tracked)
            values["interaction_completion_ratio"] = ratio(
                values["interaction_time_s"], completion)
            values["translation_per_hand_m"] = ratio(
                values["translation_error_reduction_m"], values["hand_movement_m"])
            values["rotation_per_hand_m"] = ratio(
                values["rotation_error_reduction_deg"], values["hand_movement_m"])
            values["translation_per_interaction_s"] = ratio(
                values["translation_error_reduction_m"], values["interaction_time_s"])
            values["rotation_per_interaction_s"] = ratio(
                values["rotation_error_reduction_deg"], values["interaction_time_s"])
            condition_output.append(values)
    if not method_output or not condition_output:
        raise SystemExit("No pre-first-reach Study 2 interactions found.")
    return method_output, condition_output


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def print_summary(rows, methods, title):
    if not rows:
        print(f"\n{title}\nNo actual-method rows available.")
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["participant"], row["method"])].append(row)
    people = {method: sorted({r["participant"] for r in rows if r["method"] == method})
              for method in methods}
    counts = {method: sum(r["method"] == method for r in rows) for method in methods}
    cells = {}
    for label, field in METRICS:
        for method in methods:
            person_means = [mean(r[field] for r in grouped[(person, method)])
                            for person in people[method]]
            person_means = [v for v in person_means if math.isfinite(v)]
            if not person_means:
                cells[label, method] = "—"
            else:
                avg = float(np.mean(person_means))
                sd = float(np.std(person_means, ddof=1)) if len(person_means) > 1 else float("nan")
                cells[label, method] = (f"{avg:.3f} ± {sd:.3f}" if math.isfinite(sd)
                                        else f"{avg:.3f} ± n/a")
    headers = ["Metric", *[f"{m} (P={len(people[m])}, T={counts[m]})" for m in methods]]
    table = [[label, *[cells[label, method] for method in methods]] for label, _ in METRICS]
    widths = [max([len(headers[i]), *[len(row[i]) for row in table]])
              for i in range(len(headers))]
    print(f"\n{title}")
    print("Study 2 pre-first-reach efficiency by method actually used")
    print("Participant means; group mean ± sample SD. Hybrid is attributed by channel.")
    print("  ".join(v.ljust(w) for v, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                        for i, (v, w) in enumerate(zip(row, widths))))


def print_condition_summary(rows, title):
    if not rows:
        print(f"\n{title}\nNo assigned-condition rows available.")
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["participant"], row["trial_condition"])].append(row)
    conditions = list(TRIAL_MODES)
    people = {condition: sorted({row["participant"] for row in rows
                                 if row["trial_condition"] == condition})
              for condition in conditions}
    counts = {condition: sum(row["trial_condition"] == condition for row in rows)
              for condition in conditions}
    cells = {}
    for label, field in METRICS:
        for condition in conditions:
            person_means = [mean(row[field] for row in grouped[(person, condition)])
                            for person in people[condition]]
            person_means = [value for value in person_means if math.isfinite(value)]
            if not person_means:
                cells[label, condition] = "—"
            else:
                average = float(np.mean(person_means))
                sd = (float(np.std(person_means, ddof=1))
                      if len(person_means) > 1 else float("nan"))
                cells[label, condition] = (
                    f"{average:.3f} ± {sd:.3f}" if math.isfinite(sd)
                    else f"{average:.3f} ± n/a")
    headers = ["Metric", *[f"{condition} (P={len(people[condition])}, T={counts[condition]})"
                            for condition in conditions]]
    table = [[label, *[cells[label, condition] for condition in conditions]]
             for label, _field in METRICS]
    widths = [max([len(headers[i]), *[len(row[i]) for row in table]])
              for i in range(len(headers))]
    print(f"\n{title}")
    print("Study 2 pre-first-reach effectiveness by assigned condition")
    print("Hybrid combines actual AR + freedrive activity; overlapping time/path is counted once.")
    print("Participant means; group mean ± sample SD.")
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(width) if i == 0 else value.rjust(width)
                        for i, (value, width) in enumerate(zip(row, widths))))


def write_csv(path, rows, *, include_method):
    fields = ["participant", "session_id", "trial_condition", "trial_idx",
              "trial_block",
              "analysis_start_frame", "analysis_start_time_monotonic_s"]
    if include_method:
        fields.append("method")
    fields += [field for _, field in METRICS]
    fields += ["hand_tracked_s", "head_tracked_s"]
    fields += ["raw_hand_movement_m", "orientation_noise_hand_removed_m",
               "raw_rotation_error_reduction_deg",
               "orientation_noise_rotation_penalty_removed_deg"]
    if not include_method:
        fields += ["ar_interactions", "freedrive_interactions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay_logs", type=Path, nargs="+")
    parser.add_argument("--method", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--max-gap", type=float, default=.25)
    parser.add_argument("--max-hand-step", type=float, default=.25)
    parser.add_argument("--max-head-step", type=float, default=.25)
    parser.add_argument("--max-head-angle-step", type=float, default=45.)
    parser.add_argument("--contact-radius", type=float, default=.09)
    parser.add_argument("--still-speed", type=float, default=.05)
    parser.add_argument("--still-window", type=float, default=.5)
    args = parser.parse_args()
    method_rows, condition_rows = load(args.replay_logs, args)
    excluded = {name.casefold() for name in args.exclude}
    method_rows = [row for row in method_rows
                   if row["participant"].casefold() not in excluded
                   and row["method"] in args.method
                   and 0 <= int(row["trial_idx"]) < 8]
    condition_rows = [row for row in condition_rows
                      if row["participant"].casefold() not in excluded
                      and 0 <= int(row["trial_idx"]) < 8]
    blocks = (
        ("OVERALL (T1–T8)", lambda row: True),
        ("FIRST FIVE TRIALS (T1–T5)",
         lambda row: int(row["trial_idx"]) < 5),
        ("FINAL THREE TRIALS (T6–T8)",
         lambda row: int(row["trial_idx"]) >= 5),
    )
    for title, select in blocks:
        print_condition_summary(
            [row for row in condition_rows if select(row)], title)
        print_summary([row for row in method_rows if select(row)],
                      args.method, title)
    if args.csv:
        write_csv(args.csv, method_rows, include_method=True)
        condition_path = args.csv.with_name(
            f"{args.csv.stem}_conditions{args.csv.suffix or '.csv'}")
        write_csv(condition_path, condition_rows, include_method=False)
        print(f"\nDetailed actual-method rows saved to {args.csv}")
        print(f"Detailed assigned-condition rows saved to {condition_path}")


if __name__ == "__main__":
    main()
