"""Rank Pantea's Study 2 methods by right-hand and robot movement.

Lower movement is treated as more efficient.  Right-hand distance is computed
from the centroid of the tracked hand joints during each timed trial; robot
distance comes from the trial summary's TCP path length.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


MODES = ("freedrive", "ar", "hybrid")


def hand_centroid(points: object) -> tuple[float, float, float] | None:
    if not isinstance(points, list) or not points:
        return None
    valid = [p for p in points if isinstance(p, list) and len(p) >= 3]
    if not valid:
        return None
    return tuple(sum(float(p[i]) for p in valid) / len(valid) for i in range(3))


def handle_distance(points: object, board: object) -> float | None:
    """Distance from the nearest tracked right-hand joint to the AR handle."""
    if (not isinstance(points, list) or not points
            or not isinstance(board, list) or len(board) < 3):
        return None
    try:
        # Handle origin is board-local [-7.5, -150, 0] mm.
        handle = tuple(float(board[i][3])
                       - 0.0075 * float(board[i][0])
                       - 0.1500 * float(board[i][1]) for i in range(3))
        joints = [tuple(float(value) for value in point[:3])
                  for point in points
                  if isinstance(point, list) and len(point) >= 3]
    except (IndexError, TypeError, ValueError):
        return None
    return min((distance(joint, handle) for joint in joints), default=None)


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _path(samples: list[tuple[float, tuple[float, ...]]], max_gap_s: float,
          max_step_m: float) -> tuple[float, float]:
    path_m = tracked_s = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt, step = t1 - t0, distance(p0, p1)
        if 0.0 < dt <= max_gap_s and step <= max_step_m:
            path_m += step
            tracked_s += dt
    return path_m, tracked_s


def _ar_hand_path(samples: list[tuple[float, tuple[float, ...]]],
                  contacts: list[tuple[float, float]], releases: list[float],
                  grabs: list[float], max_gap_s: float,
                  max_step_m: float, still_speed_m_s: float,
                  still_window_s: float, contact_radius_m: float,
                  contact_dwell_s: float) -> tuple[float, float, str]:
    """Measure exact grabs, else infer them from palm-to-handle contact."""
    windows: list[tuple[float, float]] = []
    if grabs:
        unused = list(grabs)
        for released in releases:
            eligible = [t for t in unused if t <= released]
            if eligible:
                grabbed = eligible[-1]
                windows.append((grabbed, released))
                unused.remove(grabbed)
        source = "events"
    else:
        previous_release = -math.inf
        proximity_windows = 0
        for released in releases:
            candidate = [(t, p) for t, p in samples
                         if previous_release < t <= released]
            nearby = [(t, d) for t, d in contacts
                      if previous_release < t <= released]
            runs: list[tuple[float, float]] = []
            run_start = run_end = None
            for t, d in nearby:
                contiguous = run_end is not None and t - run_end <= max_gap_s
                if d <= contact_radius_m:
                    if run_start is None or not contiguous:
                        run_start = t
                    run_end = t
                elif run_start is not None:
                    if run_end - run_start >= contact_dwell_s:
                        runs.append((run_start, run_end))
                    run_start = run_end = None
            if run_start is not None and run_end - run_start >= contact_dwell_s:
                runs.append((run_start, run_end))
            if runs:
                # The final sustained touch before release is the most likely
                # handle acquisition; the hand can then leave the physical
                # board because the virtual board follows it.
                start = runs[-1][0]
                proximity_windows += 1
            else:
                start = candidate[0][0] if candidate else released
                still_s = 0.0
                for (t0, p0), (t1, p1) in reversed(
                        list(zip(candidate, candidate[1:]))):
                    dt = t1 - t0
                    speed = (distance(p0, p1) / dt
                             if 0.0 < dt <= max_gap_s else math.inf)
                    if speed < still_speed_m_s:
                        still_s += dt
                        if still_s >= still_window_s:
                            start = t1
                            break
                    else:
                        still_s = 0.0
            windows.append((start, released))
            previous_release = released
        source = ("handle proximity" if proximity_windows == len(releases)
                  else "handle proximity + stillness fallback")
    chosen = [(t, p) for t, p in samples
              if any(start <= t <= end for start, end in windows)]
    path_m, tracked_s = _path(chosen, max_gap_s, max_step_m)
    return path_m, tracked_s, source


def load(path: Path, max_gap_s: float, max_step_m: float,
         still_speed_m_s: float, still_window_s: float,
         contact_radius_m: float, contact_dwell_s: float) -> list[dict]:
    trials: dict[tuple[str, str, int], dict] = defaultdict(
        lambda: {"samples": [], "contacts": [], "releases": [], "grabs": []})
    summaries: dict[tuple[str, str, int], dict] = {}

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON on line {line_number}")
                continue
            mode, trial_idx = row.get("mode"), row.get("trial_idx")
            if mode not in MODES or not isinstance(trial_idx, int):
                continue
            session_id = str(row.get("session_id", ""))
            key = (session_id, mode, trial_idx)
            if row.get("type") == "frame" and row.get("timer_running"):
                point = hand_centroid(row.get("right_hand_world"))
                now = row.get("time_monotonic_s")
                if point is not None and isinstance(now, (int, float)):
                    trials[key]["samples"].append((float(now), point))
                    separation = handle_distance(
                        row.get("right_hand_world"), row.get("board_world_T"))
                    if separation is not None:
                        trials[key]["contacts"].append((float(now), separation))
            event = row.get("event")
            event_time = row.get("time_monotonic_s")
            if event == "ar_handle_released" and row.get("recording", True):
                trials[key]["releases"].append(float(event_time))
            elif event == "ar_handle_grabbed" and row.get("recording", True):
                trials[key]["grabs"].append(float(event_time))
            if row.get("event") == "trial_summary":
                summaries[key] = row

    result = []
    for key, summary in summaries.items():
        trial = trials[key]
        duration = float(summary.get("duration_s", 0.0))
        robot_m = float(summary.get("tcp_path_length_m", float("nan")))
        if key[1] in {"ar", "hybrid"}:
            hand_m, tracked_s, source = _ar_hand_path(
                trial["samples"], trial["contacts"], trial["releases"],
                trial["grabs"],
                max_gap_s, max_step_m, still_speed_m_s, still_window_s,
                contact_radius_m, contact_dwell_s)
        else:
            hand_m, tracked_s = _path(
                trial["samples"], max_gap_s, max_step_m)
            source = "timed task"
        result.append({
            "mode": key[1], "trial": key[2], "hand_m": hand_m,
            "robot_m": robot_m, "duration_s": duration,
            "coverage": tracked_s / duration if duration > 0 else 0.0,
            "hand_source": source,
        })
    if not result:
        raise SystemExit("No completed Study 2 trials found.")
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def normalized(values: dict[str, float]) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    span = high - low
    return {key: (value - low) / span if span else 0.0
            for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", nargs="?", type=Path,
        default=Path("study_logs/study2/pantea_replay.jsonl"))
    parser.add_argument("--max-gap", type=float, default=0.25,
                        help="maximum consecutive tracking gap in seconds")
    parser.add_argument("--max-hand-step", type=float, default=0.25,
                        help="reject larger hand jumps as tracking errors (m)")
    parser.add_argument("--still-speed", type=float, default=0.05,
                        help="speed below which the hand is considered still (m/s)")
    parser.add_argument("--still-window", type=float, default=0.5,
                        help="stillness needed to mark inferred grab onset (s)")
    parser.add_argument("--contact-radius", type=float, default=0.09,
                        help="maximum nearest-joint-to-handle distance (m)")
    parser.add_argument("--contact-dwell", type=float, default=0.12,
                        help="required continuous handle proximity (s)")
    args = parser.parse_args()

    rows = load(args.log, args.max_gap, args.max_hand_step,
                args.still_speed, args.still_window,
                args.contact_radius, args.contact_dwell)
    grouped = {mode: [r for r in rows if r["mode"] == mode] for mode in MODES}
    grouped = {mode: items for mode, items in grouped.items() if items}
    hand = {mode: mean([r["hand_m"] for r in items])
            for mode, items in grouped.items()}
    robot = {mode: mean([r["robot_m"] for r in items])
             for mode, items in grouped.items()}
    duration = {mode: mean([r["duration_s"] for r in items])
                for mode, items in grouped.items()}
    coverage = {mode: mean([r["coverage"] for r in items]) * 100.0
                for mode, items in grouped.items()}
    hand_n, robot_n = normalized(hand), normalized(robot)
    score = {mode: (hand_n[mode] + robot_n[mode]) / 2.0 for mode in grouped}
    ranking = sorted(grouped, key=score.get)

    participant = args.log.stem.removesuffix("_replay").replace("_", " ").title()
    print(f"Study 2 — {participant} movement efficiency (lower is better)")
    print(f"{'Rank':<5} {'Method':<11} {'Trials':>6} {'Hand m':>9} "
          f"{'Robot m':>9} {'Time s':>8} {'Hand span':>10} {'Score':>8}")
    for rank, mode in enumerate(ranking, 1):
        print(f"{rank:<5} {mode:<11} {len(grouped[mode]):>6} "
              f"{hand[mode]:>9.3f} {robot[mode]:>9.3f} {duration[mode]:>8.2f} "
              f"{coverage[mode]:>8.1f}% {score[mode]:>8.3f}")
    print(f"\nMost movement-efficient method: {ranking[0]}")
    print("Score = equal-weight mean of min-max-normalized hand and robot path; "
          "0 is best. Values are means across completed trials.")
    sources = sorted({r["hand_source"] for r in rows})
    print("Hand windows: " + ", ".join(sources) + ".")


if __name__ == "__main__":
    main()
