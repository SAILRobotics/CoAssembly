#!/usr/bin/env python3
"""Consolidated Study 4 objective-log report: everything needed for the
full write-up and dashboard artifact in one JSON, plus companion CSVs.

Outputs (all in paper_related/):
  study4_report.json                  - everything below, one file
  study4_condition_summary.csv        - per-condition mean/SD across participants
  study4_step_summary.csv             - per (condition, step_id) mean/SD
  study4_part_requests.csv            - one row per acquired part (time to get it)
  study4_conversations.csv            - one row per language/task_aware turn
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "study_logs" / "study4"
CONDITIONS = ("gesture", "language", "task_aware")
CONDITION_LABELS = {"gesture": "Gesture", "language": "Language + Gesture",
                    "task_aware": "Task-Aware"}
STEP_ORDER = ("r1_bearing_right", "r1_bearing_left", "r1_gear_rod",
             "r1_fasten_first_stand", "r2_bearing_right", "r2_bearing_left",
             "r2_gear_rod", "r2_fasten_first_stand")
# Row 1 and Row 2 fastening are the same action performed twice; collapse them
# into one "Fastening Step" group for per-step reporting. Per participant,
# average both instances if they did both, use whichever one they did if not.
STEP_GROUP = {
    "r1_fasten_first_stand": "fasten_stand", "r2_fasten_first_stand": "fasten_stand",
}
STEP_GROUP_ORDER = ("r1_bearing_right", "r1_bearing_left", "r1_gear_rod",
                    "r2_bearing_right", "r2_bearing_left", "r2_gear_rod",
                    "fasten_stand")
STEP_GROUP_LABELS = {"fasten_stand": "Fastening Step"}
EXCLUDE_PARTICIPANTS = {"test"}
CLEAN_DECISIONS = {"language_grounded", "language_colocated_grounded",
                   "language_plural_grounded"}
# Mahya's active gesture.csv is a stray 2-step re-run from the day after her
# real session; her actual 7/8-step gesture run is archived. Read that instead
# without touching the raw log files.
SOURCE_OVERRIDE = {
    ("mahya", "gesture"): "gesture_archived_20260901_144901.csv",
}


def _num(value: str) -> "float | None":
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _elapsed(row: dict) -> "float | None":
    """step_elapsed_s, falling back to the state_json snapshot when the flat
    CSV column is blank (true for hover, vlm_interaction, and other event
    types that don't call _log_session_event with an explicit value)."""
    direct = _num(row.get("step_elapsed_s"))
    if direct is not None:
        return direct
    try:
        return json.loads(row.get("state_json", "") or "{}").get("step_elapsed_s")
    except (json.JSONDecodeError, AttributeError):
        return None


def _discover_participants() -> list[str]:
    participants = []
    for entry in sorted(LOG_DIR.iterdir()):
        if not entry.is_dir() or entry.name in EXCLUDE_PARTICIPANTS:
            continue
        if all((entry / f"{c}.csv").is_file() for c in CONDITIONS):
            participants.append(entry.name)
    return participants


def _mean_sd(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(values), "mean": mean(values),
            "sd": stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    participants = _discover_participants()

    steps: list[dict] = []          # one row per completed step
    part_requests: list[dict] = []  # one row per acquired part
    conversations: list[dict] = []  # one row per language/task_aware turn
    inference_times: list[dict] = []  # one row per vlm_request -> vlm_completed pair

    for participant in participants:
        for condition in CONDITIONS:
            path = LOG_DIR / participant / SOURCE_OVERRIDE.get(
                (participant, condition), f"{condition}.csv")
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))

            # Per-step interaction counts: hover for gesture, vlm turns else.
            interactions_by_step: dict[str, int] = defaultdict(int)
            errors_by_step: dict[str, int] = defaultdict(int)
            selections_by_step: dict[str, int] = defaultdict(int)
            # First registered-input time per step: first hover (gesture) or
            # first typed/spoken request actually submitted to the system
            # (language/task_aware) -- only present for participants whose
            # sessions logged participant_speech_submitted (see SOURCE note).
            first_input_by_step: dict[str, float] = {}
            pending_request_elapsed: "float | None" = None
            acquired_seen_in_step: set = set()
            for row in rows:
                event = row.get("event_type", "")
                step_id = row.get("step_id", "")
                is_input_event = (
                    event == "hover" if condition == "gesture"
                    else event == "participant_speech_submitted")
                if is_input_event and step_id not in first_input_by_step:
                    t = _elapsed(row)
                    if t is not None:
                        first_input_by_step[step_id] = t
                if event == "vlm_request_submitted":
                    pending_request_elapsed = _elapsed(row)
                if event == "vlm_inference_completed" and pending_request_elapsed is not None:
                    completed = _elapsed(row)
                    if completed is not None:
                        inference_times.append({
                            "participant": participant, "condition": condition,
                            "step_id": step_id,
                            "inference_s": completed - pending_request_elapsed,
                        })
                    pending_request_elapsed = None
                if event == "part_selection":
                    selections_by_step[step_id] += 1
                    if row.get("correct", "") == "0":
                        errors_by_step[step_id] += 1
                if event == "object_acquired":
                    first_in_step = step_id not in acquired_seen_in_step
                    acquired_seen_in_step.add(step_id)
                    part_requests.append({
                        "participant": participant, "condition": condition,
                        "step_id": step_id, "tool_name": row.get("tool_name", ""),
                        "part_elapsed_s": _num(row.get("part_elapsed_s")),
                        "first_in_step": first_in_step,
                    })
                if condition != "gesture" and event in ("vlm_interaction", "vlm_answer"):
                    interactions_by_step[step_id] += 1
                    decision = row.get("graph_decision", "")
                    conversations.append({
                        "participant": participant, "condition": condition,
                        "step_id": step_id, "step_title": row.get("step_title", ""),
                        "timestamp": row.get("timestamp", ""),
                        "transcript": row.get("transcript", ""),
                        "spoken_response": row.get("spoken_response", ""),
                        "vlm_prediction": row.get("vlm_prediction", ""),
                        "graph_decision": decision,
                        "clean": decision in CLEAN_DECISIONS,
                    })
                elif condition == "gesture" and event == "hover":
                    interactions_by_step[step_id] += 1
                if event == "step_acquisition_complete":
                    step_time = _num(row.get("step_elapsed_s"))
                    first_input = first_input_by_step.get(step_id)
                    steps.append({
                        "participant": participant, "condition": condition,
                        "step_id": step_id,
                        "step_elapsed_s": step_time,
                        "active_time_s": (step_time - first_input
                                         if step_time is not None and first_input is not None
                                         else None),
                        "head_translation_m": _num(row.get("head_translation_m")),
                        "head_rotation_deg": _num(row.get("head_rotation_deg")),
                        "left_hand_translation_m": _num(row.get("left_hand_translation_m")),
                        "right_hand_translation_m": _num(row.get("right_hand_translation_m")),
                        "total_hand_translation_m": _num(row.get("total_hand_translation_m")),
                        "interactions": interactions_by_step.get(step_id, 0),
                        "wrong_selections": errors_by_step.get(step_id, 0),
                        "total_selections": selections_by_step.get(step_id, 0),
                    })

    # ---- condition-level summary (per participant totals -> mean/SD) ----
    per_participant_condition: dict[tuple, dict] = defaultdict(lambda: {
        "total_time_s": 0.0, "n_steps": 0, "interactions": 0,
        "wrong_selections": 0, "total_selections": 0,
        "head_translation_m": 0.0, "head_rotation_deg": 0.0,
        "left_hand_translation_m": 0.0, "right_hand_translation_m": 0.0,
        "total_hand_translation_m": 0.0,
        "active_time_s": 0.0, "active_time_steps": 0,
    })
    for s in steps:
        key = (s["participant"], s["condition"])
        agg = per_participant_condition[key]
        agg["n_steps"] += 1
        for field in ("step_elapsed_s",):
            if s[field] is not None:
                agg["total_time_s"] += s[field]
        if s["active_time_s"] is not None:
            agg["active_time_s"] += s["active_time_s"]
            agg["active_time_steps"] += 1
        agg["interactions"] += s["interactions"]
        agg["wrong_selections"] += s["wrong_selections"]
        agg["total_selections"] += s["total_selections"]
        for field in ("head_translation_m", "head_rotation_deg",
                      "left_hand_translation_m", "right_hand_translation_m",
                      "total_hand_translation_m"):
            if s[field] is not None:
                agg[field] += s[field]

    condition_summary = {}
    for condition in CONDITIONS:
        entries = [v for (p, c), v in per_participant_condition.items() if c == condition]
        error_rates = [e["wrong_selections"] / e["total_selections"]
                      if e["total_selections"] else 0.0 for e in entries]
        condition_summary[condition] = {
            "label": CONDITION_LABELS[condition],
            "n_participants": len(entries),
            "total_time_s": _mean_sd([e["total_time_s"] for e in entries]),
            "interactions_total": _mean_sd([e["interactions"] for e in entries]),
            "error_rate": _mean_sd(error_rates),
            "head_translation_total_m": _mean_sd([e["head_translation_m"] for e in entries]),
            "head_rotation_total_deg": _mean_sd([e["head_rotation_deg"] for e in entries]),
            "left_hand_translation_total_m": _mean_sd([e["left_hand_translation_m"] for e in entries]),
            "right_hand_translation_total_m": _mean_sd([e["right_hand_translation_m"] for e in entries]),
            "total_hand_translation_total_m": _mean_sd([e["total_hand_translation_m"] for e in entries]),
            # Time from the first registered input (first hover for gesture;
            # first participant_speech_submitted for language/task_aware) to
            # step completion, summed like total_time_s. Only participants
            # whose sessions logged a registered-input timestamp for every
            # step contribute -- see the n on this field specifically, which
            # can be lower than n_participants above.
            "active_time_s": _mean_sd([e["active_time_s"] for e in entries
                                       if e["active_time_steps"] == e["n_steps"]]),
        }
    # Part-time and interactions-per-part, mean/SD across participants.
    # Split into first-part-of-step (carries the step's startup/search cost)
    # vs later parts (no startup cost -- closer to "real interaction" time).
    for condition in CONDITIONS:
        part_times_by_participant: dict[str, list[float]] = defaultdict(list)
        first_by_participant: dict[str, list[float]] = defaultdict(list)
        later_by_participant: dict[str, list[float]] = defaultdict(list)
        for pr in part_requests:
            if pr["condition"] != condition or pr["part_elapsed_s"] is None:
                continue
            part_times_by_participant[pr["participant"]].append(pr["part_elapsed_s"])
            (first_by_participant if pr["first_in_step"] else later_by_participant)[
                pr["participant"]].append(pr["part_elapsed_s"])
        condition_summary[condition]["part_time_s"] = _mean_sd(
            [mean(v) for v in part_times_by_participant.values() if v])
        condition_summary[condition]["part_time_first_s"] = _mean_sd(
            [mean(v) for v in first_by_participant.values() if v])
        condition_summary[condition]["part_time_later_s"] = _mean_sd(
            [mean(v) for v in later_by_participant.values() if v])
    # Model inference time (vlm_request_submitted -> vlm_inference_completed),
    # per-participant mean then mean/SD across participants. language/
    # task_aware only, and only for the 7 participants with this logged.
    for condition in CONDITIONS:
        by_participant: dict[str, list[float]] = defaultdict(list)
        for it in inference_times:
            if it["condition"] == condition:
                by_participant[it["participant"]].append(it["inference_s"])
        per_participant_mean = [mean(v) for v in by_participant.values() if v]
        condition_summary[condition]["inference_s"] = _mean_sd(per_participant_mean)

    # ---- per-step summary (mean/SD across participants) ----
    # Row 1 and Row 2 fastening are grouped into one "Fastening Step": each
    # participant contributes a single value (their own r1/r2 average if they
    # did both, whichever one they did if not), so no one is double-weighted.
    NUMERIC_STEP_FIELDS = ("step_elapsed_s", "interactions", "head_translation_m",
                          "head_rotation_deg", "left_hand_translation_m",
                          "right_hand_translation_m")
    step_summary = []
    for condition in CONDITIONS:
        for group_key in STEP_GROUP_ORDER:
            member_ids = {sid for sid in STEP_ORDER if STEP_GROUP.get(sid, sid) == group_key}
            matches = [s for s in steps if s["condition"] == condition and s["step_id"] in member_ids]
            if not matches:
                continue
            by_participant: dict[str, list[dict]] = defaultdict(list)
            for m in matches:
                by_participant[m["participant"]].append(m)
            per_participant_values = {field: [] for field in NUMERIC_STEP_FIELDS}
            for entries in by_participant.values():
                for field in NUMERIC_STEP_FIELDS:
                    vals = [e[field] for e in entries if e[field] is not None]
                    if vals:
                        per_participant_values[field].append(mean(vals))
            title = STEP_GROUP_LABELS.get(group_key) or next(
                (s.get("step_title", "") for s in steps if s["step_id"] == group_key), "")
            step_summary.append({
                "condition": condition, "step_id": group_key, "step_title": title,
                "n": len(by_participant),
                "step_time_s": _mean_sd(per_participant_values["step_elapsed_s"]),
                "interactions": _mean_sd(per_participant_values["interactions"]),
                "head_translation_m": _mean_sd(per_participant_values["head_translation_m"]),
                "head_rotation_deg": _mean_sd(per_participant_values["head_rotation_deg"]),
                "left_hand_translation_m": _mean_sd(per_participant_values["left_hand_translation_m"]),
                "right_hand_translation_m": _mean_sd(per_participant_values["right_hand_translation_m"]),
            })

    # Components ever acquired for each raw step_id, across every participant
    # and condition -- context for what a given step's requests were about.
    step_components: dict[str, list[str]] = defaultdict(set)
    for pr in part_requests:
        if pr["tool_name"]:
            step_components[pr["step_id"]].add(pr["tool_name"])
    step_components = {k: sorted(v) for k, v in step_components.items()}

    report = {
        "participants": participants,
        "condition_summary": condition_summary,
        "step_summary": step_summary,
        "steps_raw": steps,
        "part_requests": part_requests,
        "conversations": conversations,
        "step_components": step_components,
    }
    (ROOT / "study4_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Created {ROOT / 'study4_report.json'}")

    with (ROOT / "study4_condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "n", "mean", "sd"])
        for condition, data in condition_summary.items():
            for metric, stats in data.items():
                if isinstance(stats, dict) and "mean" in stats:
                    writer.writerow([CONDITION_LABELS[condition], metric,
                                    stats["n"], stats["mean"], stats["sd"]])
    print(f"Created {ROOT / 'study4_condition_summary.csv'}")

    with (ROOT / "study4_step_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "step_id", "step_title", "n",
                         "step_time_mean_s", "step_time_sd_s",
                         "interactions_mean", "interactions_sd",
                         "head_translation_mean_m", "head_translation_sd_m",
                         "head_rotation_mean_deg", "head_rotation_sd_deg",
                         "left_hand_mean_m", "right_hand_mean_m"])
        for row in step_summary:
            writer.writerow([
                CONDITION_LABELS[row["condition"]], row["step_id"], row["step_title"], row["n"],
                row["step_time_s"]["mean"], row["step_time_s"]["sd"],
                row["interactions"]["mean"], row["interactions"]["sd"],
                row["head_translation_m"]["mean"], row["head_translation_m"]["sd"],
                row["head_rotation_deg"]["mean"], row["head_rotation_deg"]["sd"],
                row["left_hand_translation_m"]["mean"], row["right_hand_translation_m"]["mean"],
            ])
    print(f"Created {ROOT / 'study4_step_summary.csv'}")

    with (ROOT / "study4_part_requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "condition", "step_id", "tool_name", "part_elapsed_s"])
        for pr in part_requests:
            writer.writerow([pr["participant"], CONDITION_LABELS[pr["condition"]],
                             pr["step_id"], pr["tool_name"], pr["part_elapsed_s"]])
    print(f"Created {ROOT / 'study4_part_requests.csv'}")

    with (ROOT / "study4_conversations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "condition", "step_id", "step_title",
                         "timestamp", "transcript", "spoken_response",
                         "vlm_prediction", "graph_decision", "clean"])
        for c in conversations:
            writer.writerow([c["participant"], CONDITION_LABELS[c["condition"]],
                             c["step_id"], c["step_title"], c["timestamp"],
                             c["transcript"], c["spoken_response"],
                             c["vlm_prediction"], c["graph_decision"], c["clean"]])
    print(f"Created {ROOT / 'study4_conversations.csv'}")


if __name__ == "__main__":
    main()
