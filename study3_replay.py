"""Replay Study 3 JSONL trajectories in the existing Open3D SceneVis.

Usage:
    python3 study3_replay.py study3_handover_replay.jsonl
    python3 study3_replay.py study3_handover_replay.jsonl --speed 0.5
    python3 study3_replay.py study3_handover_replay.jsonl --session-id P01-...
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scene_viewer_o3d import SceneVis


def _array(value, *, matrix: bool = False):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result.reshape(4, 4) if matrix else result


def _load_records(path: Path, requested_session: str | None):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[Replay] Skipping malformed line {line_number}: {error}")
                continue
            if record.get("schema") == "study3_replay_v1":
                records.append(record)
    if not records:
        raise SystemExit(f"[Replay] No Study 3 records in {path}")
    sessions = list(dict.fromkeys(
        record.get("session_id") for record in records
        if record.get("session_id")))
    session_id = requested_session or sessions[-1]
    selected = [record for record in records
                if record.get("session_id") == session_id]
    if not selected:
        raise SystemExit(
            f"[Replay] Session {session_id!r} not found; available={sessions}")
    return session_id, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Study 3 JSONL log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")

    session_id, records = _load_records(args.log, args.session_id)
    metadata = next(
        (record for record in records if record.get("type") == "session_start"),
        {})
    vis = SceneVis(f"Study 3 Replay — {session_id}")
    workspace_lo = _array(metadata.get("workspace_lo"))
    workspace_hi = _array(metadata.get("workspace_hi"))
    if workspace_lo is not None and workspace_hi is not None:
        vis.update_workspace_bound(workspace_lo, workspace_hi)

    first_t = float(records[0]["time_monotonic_s"])
    replay_started = time.perf_counter()
    try:
        for record in records:
            target_elapsed = (
                float(record["time_monotonic_s"]) - first_t) / args.speed
            while time.perf_counter() - replay_started < target_elapsed:
                vis.tick()
                time.sleep(0.005)

            record_type = record.get("type")
            if record_type != "frame":
                if record_type in {"interaction", "unity_interaction"}:
                    event = record.get("event", record.get("event_type", ""))
                    print(f"[Replay {target_elapsed:8.3f}s] {event}: {record}")
                continue

            head_T = _array(record.get("head_world_T"), matrix=True)
            left = _array(record.get("left_hand_world"))
            right = _array(record.get("right_hand_world"))
            tcp_T = _array(record.get("tcp_world_T"), matrix=True)
            target_T = _array(
                record.get("commanded_target_world_T"), matrix=True)
            ghost_T = (_array(record.get("ghost_world_T"), matrix=True)
                       if record.get("ghost_visible", False) else None)
            links_raw = record.get("robot_link_world_T")
            links = ([_array(transform, matrix=True) for transform in links_raw]
                     if links_raw is not None else None)

            vis.update_head(head_T)
            vis.update_hands(left, right)
            vis.update_palm_triangles(left, right)
            vis.set_tcp_gripper_closed(bool(record.get("gripper_closed", False)))
            vis.update_tcp(tcp_T)
            vis.update_tcp_target(target_T)
            vis.update_left_hand_gripper(ghost_T)
            color = record.get("ghost_color_rgba")
            if color is not None:
                vis.set_left_hand_gripper_color(color[:3])
            if links is not None:
                vis.update_robot(links)
            vis.tick()
    finally:
        vis.close()


if __name__ == "__main__":
    main()
