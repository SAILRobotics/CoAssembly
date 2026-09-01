"""Read-only Study 2 group analysis from replay JSONL files.

Despite the historical filename, this script creates no files. Trials are
averaged within participant/mode first, then group mean and sample SD are
calculated across participants so every participant has equal weight.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

_MODES = ("freedrive", "ar", "hybrid")

# Display label, replay field, unit scale. Position errors are shown in cm.
_METRICS = (
    ("Completion time (s)", "duration_s", 1.0),
    ("Timed interactions (#)", "num_interactions", 1.0),
    ("Timed freedrive interactions (#)", "freedrive_interactions", 1.0),
    ("Timed AR interactions (#)", "ar_interactions", 1.0),
    ("Robot translation during task (m)", "tcp_path_length_m", 1.0),
    ("Robot rotation during task (deg)", "tcp_angular_path_length_deg", 1.0),
    ("Start position error (cm)", "start_pos_error_m", 100.0),
    ("Start angle error (deg)", "start_angle_error_deg", 1.0),
    ("First-reach time (s)", "first_reach_elapsed_s", 1.0),
    ("First-reach position error (cm)", "first_reach_pos_error_m", 100.0),
    ("First-reach angle error (deg)", "first_reach_angle_error_deg", 1.0),
    ("Trials with qualifying AR command (%)",
     "qualifying_ar_command_found", 100.0),
    ("Time to qualifying AR command (s)",
     "qualifying_ar_command_elapsed_s", 1.0),
    ("Qualifying AR command number (#)",
     "qualifying_ar_command_idx", 1.0),
    ("Qualifying command position error (cm)",
     "qualifying_ar_command_pos_error_m", 100.0),
    ("Qualifying command angle error (deg)",
     "qualifying_ar_command_angle_error_deg", 1.0),
    ("Qualifying command → first reach (s)",
     "qualifying_ar_command_to_first_reach_s", 1.0),
    ("Qualifying command → auto-stop (s)",
     "qualifying_ar_command_to_auto_stop_s", 1.0),
    ("Auto-stop position error (cm)", "auto_stop_pos_error_m", 100.0),
    ("Auto-stop angle error (deg)", "auto_stop_angle_error_deg", 1.0),
    ("Post-stop interactions (#)", "post_stop_interactions", 1.0),
    ("Post-stop freedrive interactions (#)",
     "post_stop_freedrive_interactions", 1.0),
    ("Post-stop AR interactions (#)", "post_stop_ar_interactions", 1.0),
    ("Post-stop position improvement (cm)",
     "post_stop_pos_error_improvement_m", 100.0),
    ("Post-stop angle improvement (deg)",
     "post_stop_angle_error_improvement_deg", 1.0),
    ("Enter position error (cm)", "enter_confirmation_pos_error_m", 100.0),
    ("Enter angle error (deg)", "enter_confirmation_angle_error_deg", 1.0),
    ("Exact-snap duration (s)", "snap_duration_s", 1.0),
    ("Post-snap position error (cm)", "post_snap_pos_error_m", 100.0),
    ("Post-snap angle error (deg)", "post_snap_angle_error_deg", 1.0),
)


def _participant_name(path: Path, record: dict) -> str:
    if record.get("session_name"):
        return str(record["session_name"])
    stem = path.stem
    return stem[:-7] if stem.endswith("_replay") else stem


def _load_trial_summaries(paths: list[Path]) -> list[dict]:
    summaries = []
    participant_paths: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Replay log not found: {path}")
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    print(f"Warning: {path}:{line_number}: {error}")
                    continue
                records.append(record)
        releases: dict[tuple, list[dict]] = defaultdict(list)
        for record in records:
            if (record.get("event") == "ar_handle_released"
                    and record.get("recording")):
                key = (record.get("session_id"), record.get("mode"),
                       record.get("trial_idx"))
                releases[key].append(record)
        for commands in releases.values():
            commands.sort(key=lambda record: record.get("time_unix_s", 0.0))

        for record in records:
            if (record.get("event") != "trial_summary"
                    and record.get("type") != "trial_summary"):
                continue
            participant = _participant_name(path, record)
            resolved = path.resolve()
            previous = participant_paths.setdefault(participant, resolved)
            if previous != resolved:
                raise SystemExit(
                    f"Participant {participant!r} occurs in both {previous} "
                    f"and {path}; pass each participant only once.")
            enriched = {**record, "_participant": participant}
            mode = record.get("mode")
            if mode in {"ar", "hybrid"}:
                enriched["qualifying_ar_command_found"] = 0.0
                key = (record.get("session_id"), mode,
                       record.get("trial_idx"))
                for command_idx, command in enumerate(
                        releases.get(key, []), 1):
                    pos_error = _number(command.get("released_pos_error_m"))
                    angle_error = _number(command.get(
                        "released_angle_error_deg"))
                    if pos_error < 0.05 and angle_error < 15.0:
                        command_time = _number(command.get("time_unix_s"))
                        enriched.update({
                            "qualifying_ar_command_found": 1.0,
                            "qualifying_ar_command_elapsed_s":
                                command_time - _number(record.get("start_time")),
                            "qualifying_ar_command_idx": command_idx,
                            "qualifying_ar_command_pos_error_m": pos_error,
                            "qualifying_ar_command_angle_error_deg": angle_error,
                            "qualifying_ar_command_to_first_reach_s":
                                _number(record.get("first_reach_time"))
                                - command_time,
                            "qualifying_ar_command_to_auto_stop_s":
                                _number(record.get("end_time")) - command_time,
                        })
                        break
            summaries.append(enriched)
    if not summaries:
        raise SystemExit("No completed Study 2 trial_summary records found.")
    return summaries


def _number(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _analyze(summaries: list[dict], modes: list[str], title: str) -> None:
    selected = [record for record in summaries if record.get("mode") in modes]
    if not selected:
        available = sorted({record.get("mode") for record in summaries})
        raise SystemExit(f"No trials for requested modes; available={available}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in selected:
        grouped[(record["_participant"], record["mode"])].append(record)

    participants = {
        mode: sorted({person for person, item_mode in grouped
                      if item_mode == mode}) for mode in modes
    }
    trial_counts = {
        mode: sum(len(rows) for (person, item_mode), rows in grouped.items()
                  if item_mode == mode) for mode in modes
    }

    cells = {}
    for label, field, scale in _METRICS:
        for mode in modes:
            participant_means = []
            for person in participants[mode]:
                values = np.asarray([
                    _number(record.get(field)) * scale
                    for record in grouped[(person, mode)]
                ])
                values = values[np.isfinite(values)]
                if values.size:
                    participant_means.append(float(np.mean(values)))
            if not participant_means:
                cells[(label, mode)] = "—"
                continue
            mean = float(np.mean(participant_means))
            sd = (float(np.std(participant_means, ddof=1))
                  if len(participant_means) > 1 else float("nan"))
            cells[(label, mode)] = (f"{mean:.3f} ± {sd:.3f}"
                                    if math.isfinite(sd)
                                    else f"{mean:.3f} ± n/a")

    headers = ["Metric", *[
        f"{mode} (P={len(participants[mode])}, T={trial_counts[mode]})"
        for mode in modes
    ]]
    rows = [[label, *[cells[(label, mode)] for mode in modes]]
            for label, _, _ in _METRICS]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print(f"\n{title}")
    print("Study 2 mode comparison (participant means; group mean ± sample SD)")
    print("P = participants, T = completed trials; — = not logged")
    print("  ".join(value.ljust(width)
                    for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(
            value.ljust(width) if i == 0 else value.rjust(width)
            for i, (value, width) in enumerate(zip(row, widths))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Study 2 comparison across participants/modes.")
    parser.add_argument("replay_logs", type=Path, nargs="+",
                        help="One or more participant *_replay.jsonl files")
    parser.add_argument("--mode", nargs="+", choices=_MODES,
                        default=list(_MODES),
                        help="Modes included in the printed comparison")
    args = parser.parse_args()
    summaries = _load_trial_summaries(args.replay_logs)
    _analyze(summaries, args.mode, "ALL TRIALS (T1–T10)")
    far = [record for record in summaries
           if 0 <= int(record.get("trial_idx", -1)) < 5]
    close = [record for record in summaries
             if 5 <= int(record.get("trial_idx", -1)) < 10]
    _analyze(far, args.mode, "ORIGINAL TARGETS / DEFAULT START (T1–T5)")
    _analyze(close, args.mode, "DISPERSED DUPLICATES / DEFAULT START (T6–T10)")


if __name__ == "__main__":
    main()
