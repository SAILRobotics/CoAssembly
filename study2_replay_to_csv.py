"""Derive Study 2 (workholding) trial/trajectory CSVs from a replay JSONL.

workholding_study.py logs only one file per participant now —
{session_name}_replay.jsonl, shared across all modes. This script
reconstructs the tabular CSVs analysts actually want to open in
Excel/pandas (trial summaries and the raw trajectory/hand/head time
series) from that single source of truth.

Usage
-----
    python3 study2_replay_to_csv.py study_logs/study2/P01_replay.jsonl
    python3 study2_replay_to_csv.py study_logs/study2/P01_replay.jsonl --out-dir out/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

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
    "freedrive_interactions", "ar_interactions",
    "tcp_path_length_m", "tcp_angular_path_length_deg",
    "recording_start_source", "start_policy",
    "snap_success", "snap_duration_s",
    "post_snap_pos_error_m", "post_snap_angle_error_deg",
    "target_poses_file",
]


def _load_records(path: Path) -> list:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _T_to_pos_quat(T) -> "tuple | None":
    if T is None:
        return None
    T = np.asarray(T, dtype=float)
    pos = T[:3, 3]
    quat = ScipyR.from_matrix(T[:3, :3]).as_quat()
    return pos, quat


def _trial_base_row(rec: dict) -> list:
    pos = rec.get("target_pos_m") or [float("nan")] * 3
    euler = rec.get("target_euler_deg") or [float("nan")] * 3
    return [
        rec.get("session_name", ""), rec.get("mode", ""),
        rec.get("trial_idx", ""), rec.get("pose_idx", ""),
        *pos, *euler,
        rec.get("start_time", ""), rec.get("end_time", ""),
        rec.get("duration_s", ""),
        rec.get("final_pos_error_m", ""), rec.get("final_angle_error_deg", ""),
        rec.get("num_interactions", ""), rec.get("completion_reason", ""),
    ]


def _trial_detailed_row(rec: dict) -> list:
    start_pos = rec.get("start_board_pos_m") or [float("nan")] * 3
    start_euler = rec.get("start_board_euler_deg") or [float("nan")] * 3
    return [
        *_trial_base_row(rec),
        *start_pos, *start_euler,
        rec.get("start_pos_error_m", ""), rec.get("start_angle_error_deg", ""),
        rec.get("freedrive_interactions", ""), rec.get("ar_interactions", ""),
        rec.get("tcp_path_length_m", ""), rec.get("tcp_angular_path_length_deg", ""),
        rec.get("recording_start_source", ""), rec.get("start_policy", ""),
        rec.get("snap_success", ""), rec.get("snap_duration_s", ""),
        rec.get("post_snap_pos_error_m", ""), rec.get("post_snap_angle_error_deg", ""),
        rec.get("target_poses_file", ""),
    ]


def convert(replay_path: Path, out_dir: Path) -> None:
    records = _load_records(replay_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = replay_path.stem
    if stem.endswith("_replay"):
        stem = stem[: -len("_replay")]

    trials_rows, detailed_rows = [], []
    traj_rows, left_hand_rows, right_hand_rows, head_rows = [], [], [], []
    sample_idx_by_trial: dict = {}

    for rec in records:
        rtype = rec.get("type")
        if rtype == "trial_base_summary":
            trials_rows.append(_trial_base_row(rec))
            continue
        if rtype == "trial_summary":
            detailed_rows.append(_trial_detailed_row(rec))
            continue
        if rtype != "frame":
            continue

        session_name, mode = rec.get("session_name", ""), rec.get("mode", "")
        trial_idx, pose_idx = rec.get("trial_idx", ""), rec.get("pose_idx", "")
        t_rel = rec.get("trial_elapsed_s", "")
        key = (mode, trial_idx)
        sample_idx = sample_idx_by_trial.get(key, 0)
        sample_idx_by_trial[key] = sample_idx + 1

        tcp = _T_to_pos_quat(rec.get("tcp_world_T"))
        q = rec.get("robot_q_rad")
        if tcp is not None and q is not None:
            pos, quat = tcp
            traj_rows.append([
                session_name, mode, trial_idx, t_rel,
                *pos.tolist(), *quat.tolist(),
                *np.degrees(np.asarray(q, dtype=float)).tolist(),
            ])

        def _hand_rows(points) -> list:
            prefix = [session_name, mode, trial_idx, pose_idx, t_rel, sample_idx]
            if not points:
                return [[*prefix, False, "", "", "", ""]]
            return [
                [*prefix, True, j, *[float(v) for v in pt]]
                for j, pt in enumerate(points)
            ]

        left_hand_rows.extend(_hand_rows(rec.get("left_hand_world")))
        right_hand_rows.extend(_hand_rows(rec.get("right_hand_world")))

        head = _T_to_pos_quat(rec.get("head_world_T"))
        head_prefix = [session_name, mode, trial_idx, pose_idx, t_rel, sample_idx]
        if head is None:
            head_rows.append([*head_prefix, False, "", "", "", "", "", "", ""])
        else:
            pos, quat = head
            head_rows.append([*head_prefix, True, *pos.tolist(), *quat.tolist()])

    def _write(name: str, header: list, rows: list) -> None:
        path = out_dir / f"{stem}_{name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[study2_replay_to_csv] {path} ({len(rows)} rows)")

    _write("trials", _TRIAL_CSV_HEADER, trials_rows)
    _write("trials_detailed", _DETAILED_TRIAL_CSV_HEADER, detailed_rows)
    _write("trajectory", _TRAJ_CSV_HEADER, traj_rows)
    _write("left_hand_trajectory", _HAND_TRAJ_CSV_HEADER, left_hand_rows)
    _write("right_hand_trajectory", _HAND_TRAJ_CSV_HEADER, right_hand_rows)
    _write("head_trajectory", _HEAD_TRAJ_CSV_HEADER, head_rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive Study 2 trial/trajectory CSVs from a merged "
                     "workholding replay JSONL.")
    ap.add_argument("replay_log", type=Path,
                    help="Path to {session_name}_replay.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Defaults to the replay log's own directory.")
    args = ap.parse_args()
    out_dir = args.out_dir or args.replay_log.parent
    convert(args.replay_log, out_dir)


if __name__ == "__main__":
    main()
