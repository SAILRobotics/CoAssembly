"""Replay a workholding-study JSONL recording in the Open3D study scene."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import main_setting as cfg
from workholding_study import _WorkholdingSceneVis


def _array(value, matrix=False):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result.reshape(4, 4) if matrix else result


def _load(path: Path, requested_session: str | None,
          requested_mode: str | None = None):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[Replay] Skipping malformed line {line_number}: {error}")
                continue
            if record.get("schema") == "workholding_replay_v1":
                records.append(record)
    sessions = list(dict.fromkeys(record.get("session_id") for record in records))
    eligible_sessions = list(dict.fromkeys(
        record.get("session_id") for record in records
        if requested_mode is None or record.get("mode") == requested_mode))
    session_id = requested_session or (
        eligible_sessions[-1] if eligible_sessions else None)
    selected = [record for record in records
                if (record.get("session_id") == session_id
                    and (requested_mode is None
                         or record.get("mode") == requested_mode))]
    if not selected:
        available_modes = list(dict.fromkeys(
            record.get("mode") for record in records
            if record.get("session_id") == session_id))
        raise SystemExit(
            "No matching workholding replay; "
            f"sessions={sessions}, modes={available_modes}")
    return session_id, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a workholding JSONL log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--mode", choices=("freedrive", "ar", "hybrid"),
                        help="Replay only this study condition")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")

    session_id, records = _load(args.log, args.session_id, args.mode)
    metadata = next((r for r in records if r.get("event") == "session_start"), {})
    mode_label = f" — {args.mode}" if args.mode else ""
    vis = _WorkholdingSceneVis(
        f"Workholding Replay — {session_id}{mode_label}")
    targets = [_array(T, matrix=True) for T in metadata.get("target_poses", [])]
    if targets:
        vis.configure_target_ghosts(targets)
    lo, hi = _array(metadata.get("workspace_lo")), _array(metadata.get("workspace_hi"))
    if lo is not None and hi is not None:
        vis.update_workspace_bound(lo, hi)

    first_t = float(records[0]["time_monotonic_s"])
    started = time.perf_counter()
    try:
        for record in records:
            elapsed = (float(record["time_monotonic_s"]) - first_t) / args.speed
            while time.perf_counter() - started < elapsed:
                vis.tick()
                time.sleep(0.005)
            if record.get("type") != "frame":
                print(f"[Replay {elapsed:8.3f}s] {record.get('event', '')}")
                continue
            tcp = _array(record.get("tcp_world_T"), matrix=True)
            head = _array(record.get("head_world_T"), matrix=True)
            left, right = _array(record.get("left_hand_world")), _array(record.get("right_hand_world"))
            links_raw = record.get("robot_link_world_T")
            links = ([_array(T, matrix=True) for T in links_raw]
                     if links_raw is not None else None)
            vis.update_head(head)
            vis.update_hands(left, right)
            vis.update_tcp(tcp)
            if links is not None:
                vis.update_robot(links)
            held = record.get("robot_board_state") in {
                "holding_board", "moving_board", "release_armed"}
            vis.set_tcp_gripper_closed(held)
            vis.update_board_ar_from_tcp(tcp, cfg.BOX_FORWARD_OFFSET)
            pose_idx = record.get("pose_idx")
            state = record.get("target_color_state", "far")
            if targets and isinstance(pose_idx, int) and 0 <= pose_idx < len(targets):
                vis.select_target(pose_idx, state)
                vis.update_target_gripper(targets[pose_idx],
                                          cfg.BOX_FORWARD_OFFSET, state)
            vis.update_ar_handle(
                _array(record.get("board_world_T"), matrix=True)
                if record.get("ar_enabled") else None)
            vis.tick()
    finally:
        vis.close()


if __name__ == "__main__":
    main()
