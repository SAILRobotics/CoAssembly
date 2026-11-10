#!/usr/bin/env python3
"""Objective Study 4 behavioral analysis from the per-condition session CSVs.

For each participant/condition (gesture, language, task_aware), computes:
  - time taken       : mean per-step duration (step_elapsed_s at
                        step_acquisition_complete); normalized by completed
                        step count so participants missing steps aren't biased.
  - error rate        : wrong part_selection events / total part_selection
                        events (correct field).
  - interaction count : mean interactions per completed step. Gesture counts
                        `hover` events; language/task_aware count `vlm_interaction`
                        and `vlm_answer` events (spoken request/response turns).
  - physical effort    : mean per-step head translation (m), total hand
                        translation (m), and head rotation (deg).

Condition comparisons use the same nonparametric approach as
format_qualtrics_condition_comparison.py: Friedman omnibus tests with
Kendall's W, followed by Holm-corrected paired Wilcoxon signed-rank tests,
computed only over participants with valid data in all three conditions.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "study_logs" / "study4"
CONDITIONS = ("gesture", "language", "task_aware")
CONDITION_LABELS = {"gesture": "Gesture", "language": "Language + Gesture",
                    "task_aware": "Task-Aware"}
EXCLUDE_PARTICIPANTS = {"test"}
# Mahya's active gesture.csv is a stray 2-step re-run from the day after her
# real session; her actual 7/8-step gesture run is archived. Read that instead
# without touching the raw log files.
SOURCE_OVERRIDE = {
    ("mahya", "gesture"): "gesture_archived_20260901_144901.csv",
}

SUMMARY_OUTPUT = ROOT / "study4_behavioral_condition_summary.csv"
TESTS_OUTPUT = ROOT / "study4_behavioral_pairwise_tests.csv"
PER_PARTICIPANT_OUTPUT = ROOT / "study4_behavioral_per_participant.csv"
REPORT_OUTPUT = ROOT / "study4_behavioral_analysis_report.md"

METRICS = (
    # (key, label, higher_is_better)
    ("mean_step_time_s", "Time per step (s)", False),
    ("mean_part_time_s", "Time per part (s)", False),
    ("error_rate", "Error rate (wrong / total selections)", False),
    ("interactions_per_step", "Interactions per step", False),
    ("interactions_per_part", "Interactions per part", False),
    ("mean_head_translation_m", "Head translation per step (m)", False),
    ("mean_hand_translation_m", "Hand translation per step (m)", False),
    ("mean_head_rotation_deg", "Head rotation per step (deg)", False),
)


def _num(value: str) -> "float | None":
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _discover_participants() -> list[str]:
    participants = []
    for entry in sorted(LOG_DIR.iterdir()):
        if not entry.is_dir() or entry.name in EXCLUDE_PARTICIPANTS:
            continue
        if all((entry / f"{condition}.csv").is_file() for condition in CONDITIONS):
            participants.append(entry.name)
    return participants


def _load_condition(path: Path) -> dict:
    """Return per-condition aggregates for one participant/condition CSV."""
    steps = []          # (step_id, step_elapsed_s, head_m, hand_m, head_deg)
    total_selections = 0
    wrong_selections = 0
    interaction_count = 0
    part_times = []      # part_elapsed_s at each object_acquired (per-part time)
    n_parts_acquired = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = row.get("event_type", "")
            if event == "step_acquisition_complete":
                steps.append((
                    row.get("step_id", ""),
                    _num(row.get("step_elapsed_s")),
                    _num(row.get("head_translation_m")),
                    _num(row.get("total_hand_translation_m")),
                    _num(row.get("head_rotation_deg")),
                ))
            elif event == "part_selection":
                total_selections += 1
                if row.get("correct", "") == "0":
                    wrong_selections += 1
            elif event == "object_acquired":
                n_parts_acquired += 1
                part_time = _num(row.get("part_elapsed_s"))
                if part_time is not None:
                    part_times.append(part_time)
            if path.stem == "gesture":
                if event == "hover":
                    interaction_count += 1
            else:
                if event in ("vlm_interaction", "vlm_answer"):
                    interaction_count += 1

    n_steps = len(steps)
    step_times = [s[1] for s in steps if s[1] is not None]
    head_m = [s[2] for s in steps if s[2] is not None]
    hand_m = [s[3] for s in steps if s[3] is not None]
    head_deg = [s[4] for s in steps if s[4] is not None]
    return {
        "n_steps": n_steps,
        "mean_step_time_s": mean(step_times) if step_times else None,
        "total_time_s": sum(step_times) if step_times else None,
        "error_rate": (wrong_selections / total_selections
                      if total_selections else None),
        "total_selections": total_selections,
        "wrong_selections": wrong_selections,
        "interactions_per_step": (interaction_count / n_steps
                                  if n_steps else None),
        "interaction_count": interaction_count,
        "n_parts_acquired": n_parts_acquired,
        "mean_part_time_s": mean(part_times) if part_times else None,
        "interactions_per_part": (interaction_count / n_parts_acquired
                                  if n_parts_acquired else None),
        "mean_head_translation_m": mean(head_m) if head_m else None,
        "mean_hand_translation_m": mean(hand_m) if hand_m else None,
        "mean_head_rotation_deg": mean(head_deg) if head_deg else None,
    }


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(values) > 1 else float("nan"),
        "median": float(np.median(array)),
    }


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def p_text(value: float) -> str:
    return "<.001" if value < .001 else f"{value:.3f}"


def main() -> None:
    participants = _discover_participants()
    per_participant: dict[str, dict[str, dict]] = {}
    for participant in participants:
        per_participant[participant] = {
            condition: _load_condition(LOG_DIR / participant / SOURCE_OVERRIDE.get(
                (participant, condition), f"{condition}.csv"))
            for condition in CONDITIONS
        }

    with PER_PARTICIPANT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "condition", "n_steps", "mean_step_time_s",
                         "total_time_s", "n_parts_acquired", "mean_part_time_s",
                         "error_rate", "total_selections",
                         "wrong_selections", "interactions_per_step",
                         "interactions_per_part", "interaction_count",
                         "mean_head_translation_m",
                         "mean_hand_translation_m", "mean_head_rotation_deg"])
        for participant in participants:
            for condition in CONDITIONS:
                data = per_participant[participant][condition]
                writer.writerow([participant, condition, data["n_steps"],
                                 data["mean_step_time_s"], data["total_time_s"],
                                 data["n_parts_acquired"], data["mean_part_time_s"],
                                 data["error_rate"], data["total_selections"],
                                 data["wrong_selections"],
                                 data["interactions_per_step"],
                                 data["interactions_per_part"],
                                 data["interaction_count"],
                                 data["mean_head_translation_m"],
                                 data["mean_hand_translation_m"],
                                 data["mean_head_rotation_deg"]])
    print(f"Created {PER_PARTICIPANT_OUTPUT}")

    incomplete = [(participant, condition, per_participant[participant][condition]["n_steps"])
                 for participant in participants for condition in CONDITIONS
                 if per_participant[participant][condition]["n_steps"] < 8]

    summary_rows = []
    omnibus_rows = []
    pairwise_rows = []
    for key, label, _ in METRICS:
        for condition in CONDITIONS:
            observed = [per_participant[p][condition][key] for p in participants
                       if per_participant[p][condition][key] is not None]
            stats = describe(observed)
            summary_rows.append((label, CONDITION_LABELS[condition], stats))

        complete = [p for p in participants
                   if all(per_participant[p][c][key] is not None for c in CONDITIONS)]
        columns = [[per_participant[p][c][key] for p in complete] for c in CONDITIONS]
        try:
            result = friedmanchisquare(*columns)
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
        effect = statistic / (len(complete) * 2) if complete else 0.0
        omnibus_rows.append((label, len(complete), statistic, p_value, effect))

        tests = []
        for left_index, right_index in [(0, 1), (0, 2), (1, 2)]:
            left, right = CONDITIONS[left_index], CONDITIONS[right_index]
            x = [per_participant[p][left][key] for p in complete]
            y = [per_participant[p][right][key] for p in complete]
            try:
                result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            tests.append([label, CONDITION_LABELS[left], CONDITION_LABELS[right],
                         len(complete), mean(a - b for a, b in zip(x, y)),
                         statistic, p_value])
        for test, adjusted in zip(tests, holm([test[-1] for test in tests])):
            pairwise_rows.append(tuple(test + [adjusted]))

    with SUMMARY_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "condition", "n", "mean", "sd", "median"])
        for label, condition, stats in summary_rows:
            writer.writerow([label, condition, stats["n"], stats["mean"],
                             stats["sd"], stats["median"]])
    print(f"Created {SUMMARY_OUTPUT}")

    with TESTS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "condition_a", "condition_b", "n",
                         "mean_a_minus_b", "wilcoxon_W", "p_raw", "p_holm"])
        writer.writerows(pairwise_rows)
    print(f"Created {TESTS_OUTPUT}")

    report = [
        "# Study 4 Behavioral Analysis (Objective Logs)", "",
        f"Participants with complete gesture/language/task_aware logs: **{len(participants)}** "
        f"({', '.join(participants)}).",
        "Time, interaction count, and physical effort are per-step means, "
        "normalized by each participant's completed step count so incomplete "
        "sessions aren't biased by having fewer steps to sum over.",
        "Interaction count: Gesture counts `hover` events; Language + Gesture "
        "and Task-Aware count `vlm_interaction`/`vlm_answer` events (spoken "
        "request/response turns).",
        "", "## Data completeness", "",
    ]
    if incomplete:
        report.append("Sessions with fewer than 8 completed steps (out of 8 expected):")
        report.append("")
        for participant, condition, n_steps in incomplete:
            report.append(f"- {participant} / {CONDITION_LABELS[condition]}: {n_steps}/8 steps")
    else:
        report.append("All participants completed all 8 steps in every condition.")

    report += ["", "## Condition summaries", "",
               "| Metric | Condition | n | Mean ± SD | Median |",
               "|---|---|---:|---:|---:|"]
    for label, condition, stats in summary_rows:
        report.append(f"| {label} | {condition} | {stats['n']} | "
                      f"{stats['mean']:.3f} ± {stats['sd']:.3f} | {stats['median']:.3f} |")

    report += ["", "## Friedman omnibus tests", "",
               "| Metric | n | χ²(2) | p | Kendall's W |", "|---|---:|---:|---:|---:|"]
    for label, n, statistic, p_value, effect in omnibus_rows:
        report.append(f"| {label} | {n} | {statistic:.2f} | {p_text(p_value)} | {effect:.2f} |")

    report += ["", "## Holm-corrected pairwise Wilcoxon tests", "",
               "| Metric | Condition A | Condition B | n | A − B | W | Raw p | Holm p |",
               "|---|---|---|---:|---:|---:|---:|---:|"]
    for label, left, right, n, difference, statistic, raw_p, adjusted_p in pairwise_rows:
        report.append(f"| {label} | {left} | {right} | {n} | {difference:.3f} | "
                      f"{statistic:.2f} | {p_text(raw_p)} | {p_text(adjusted_p)} |")

    REPORT_OUTPUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
