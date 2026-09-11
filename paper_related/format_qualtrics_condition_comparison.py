#!/usr/bin/env python3
"""Put the three condition responses side by side for every repeated item."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "HumanRobotCoassembly_September 9, 2026_15.18.csv"
S2_OUTPUT = ROOT / "study2_responses_side_by_side.csv"
S4_OUTPUT = ROOT / "study4_responses_side_by_side.csv"
EDITED_S2_SOURCE = ROOT / "H2R_UserStudy_Analysis - study2_responses_side_by_side.csv"
S2_SUMMARY_OUTPUT = ROOT / "study2_edited_condition_summary.csv"
S2_TESTS_OUTPUT = ROOT / "study2_edited_pairwise_tests.csv"
S2_ANALYSIS_OUTPUT = ROOT / "study2_edited_analysis_report.md"
EDITED_S4_SOURCE = ROOT / "H2R_UserStudy_Analysis - study4_responses_side_by_side.csv"
S4_SUMMARY_OUTPUT = ROOT / "study4_edited_condition_summary.csv"
S4_TESTS_OUTPUT = ROOT / "study4_edited_pairwise_tests.csv"
S4_ANALYSIS_OUTPUT = ROOT / "study4_edited_analysis_report.md"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load():
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.reader(handle))
    fields, questions = raw[0], raw[1]
    question_by_field = {field: clean(questions[index]) for index, field in enumerate(fields)}
    retained = []
    for source_row, values in enumerate(raw[3:], start=4):
        values += [""] * (len(fields) - len(values))
        participant = values[28].strip()
        metric_count = sum(bool(values[index].strip()) for index, field in enumerate(fields)
                           if field.startswith(("S2_", "S4_")))
        if metric_count and participant.casefold() != "sandeep john philip":
            retained.append((source_row, participant, dict(zip(fields, values))))
    return question_by_field, retained


def short_question(full_question: str) -> str:
    # Remove the repeated scale instructions while retaining the exact item
    # prompt following the Qualtrics matrix separator.
    return full_question.rsplit(" - ", 1)[-1].strip()


def numeric(value: str | None) -> float | None:
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", value or "")
    return float(match.group(1)) if match else None


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(values) > 1 else float("nan"),
        "median": float(np.median(array)), "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
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


def analyze_edited_study(source: Path, conditions: list[str], scales: dict[str, int],
                         summary_output: Path, tests_output: Path,
                         report_output: Path, title: str,
                         interpretation: list[str],
                         nasa_items: list[int],
                         excluded_items: dict[str, dict[str, set[int]]] | None = None,
                         scale_labels: dict[str, str] | None = None,
                         excluded_participants: dict[str, str] | None = None) -> None:
    """Analyze a manually completed side-by-side worksheet.

    excluded_participants: {participant_name: reason}. Excluded for a
    documented, outcome-independent protocol reason (e.g., a deviation from
    task instructions given uniformly to all participants), applied to every
    scale and measure in this study rather than selectively.
    """
    excluded_participants = excluded_participants or {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("participant") or "").strip()
               and row["participant"].strip() not in excluded_participants]

    participants = list(dict.fromkeys(row["participant"].strip() for row in rows))
    excluded_items = excluded_items or {}
    scale_labels = scale_labels or {}
    indexed = {(row["participant"].strip(), row["scale"], int(row["item_number"])): row
               for row in rows}
    missing = []
    for row in rows:
        if row["scale"] in {"Additional Comments"}:
            continue
        for condition in conditions:
            if not (row.get(condition) or "").strip():
                missing.append((row["participant"], row["scale"], row["item_number"],
                                row["question"], condition))

    composites = {scale: {condition: {} for condition in conditions} for scale in scales}
    summary_rows = []
    omnibus_rows = []
    pairwise_rows = []
    for scale, item_count in scales.items():
        display_scale = scale_labels.get(scale, scale)
        included_items = (nasa_items if scale == "NASA-TLX"
                          else list(range(1, item_count + 1)))
        for participant in participants:
            for condition in conditions:
                values = []
                condition_items = [
                    item for item in included_items
                    if item not in excluded_items.get(scale, {}).get(condition, set())
                ]
                for item in condition_items:
                    value = numeric(indexed[(participant, scale, item)].get(condition))
                    # Performance is success-oriented, unlike the other
                    # lower-is-better workload dimensions.
                    if scale == "NASA-TLX" and item == 4 and value is not None:
                        value = 100 - value
                    if value is not None:
                        values.append(value)
                minimum = len(condition_items)
                composites[scale][condition][participant] = (
                    mean(values) if len(values) >= minimum else None
                )

        for condition in conditions:
            observed = [value for value in composites[scale][condition].values()
                        if value is not None]
            stats = describe(observed)
            summary_rows.append((display_scale, condition, stats))

        complete = [participant for participant in participants
                    if all(composites[scale][condition][participant] is not None
                           for condition in conditions)]
        columns = [[composites[scale][condition][participant] for participant in complete]
                   for condition in conditions]
        try:
            result = friedmanchisquare(*columns)
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
        effect = statistic / (len(complete) * 2)
        omnibus_rows.append((display_scale, len(complete), statistic, p_value, effect))

        tests = []
        for left_index, right_index in [(0, 1), (0, 2), (1, 2)]:
            left, right = conditions[left_index], conditions[right_index]
            paired = [participant for participant in participants
                      if composites[scale][left][participant] is not None
                      and composites[scale][right][participant] is not None]
            x = [composites[scale][left][participant] for participant in paired]
            y = [composites[scale][right][participant] for participant in paired]
            try:
                result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            tests.append([display_scale, left, right, len(paired), mean(a - b for a, b in zip(x, y)),
                          statistic, p_value])
        for test, adjusted in zip(tests, holm([test[-1] for test in tests])):
            pairwise_rows.append(tuple(test + [adjusted]))

    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scale", "condition", "n", "mean", "sd", "median", "q1", "q3"])
        for scale, condition, stats in summary_rows:
            writer.writerow([scale, condition, stats["n"], stats["mean"], stats["sd"],
                             stats["median"], stats["q1"], stats["q3"]])
    rank_rows = {participant: indexed[(participant, "Overall Preference Rank", 1)]
                 for participant in participants}
    rank_values = [[numeric(rank_rows[participant][condition]) for participant in participants]
                   for condition in conditions]
    rank_test = friedmanchisquare(*rank_values)
    rank_effect = rank_test.statistic / (len(participants) * 2)
    rank_pairs = []
    for left_index, right_index in [(0, 1), (0, 2), (1, 2)]:
        left, right = conditions[left_index], conditions[right_index]
        result = wilcoxon(rank_values[left_index], rank_values[right_index],
                          alternative="two-sided")
        rank_pairs.append(["Overall Preference Rank", left, right, len(participants),
                           mean(a - b for a, b in zip(rank_values[left_index],
                                                     rank_values[right_index])),
                           float(result.statistic), float(result.pvalue)])
    for test, adjusted in zip(rank_pairs, holm([test[-1] for test in rank_pairs])):
        pairwise_rows.append(tuple(test + [adjusted]))

    with tests_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scale", "condition_a", "condition_b", "n", "mean_a_minus_b",
                         "wilcoxon_W", "p_raw", "p_holm"])
        writer.writerows(pairwise_rows)

    workload_note = (
        "The task-relevant workload composite excludes Temporal Demand and Performance; "
        "it averages Mental Demand, Physical Demand, Effort, and Frustration "
        "(all lower-is-better)."
        if nasa_items == [1, 2, 5, 6]
        else "NASA-TLX uses all six dimensions; the success-oriented Performance item "
             "was reverse-scored so that every dimension is lower-is-better."
    )
    applicability_notes = []
    for scale, by_condition in excluded_items.items():
        for condition, items in by_condition.items():
            item_text = ", ".join(str(item) for item in sorted(items))
            applicability_notes.append(
                f"For {condition}, {scale} item(s) {item_text} were structurally "
                "inapplicable and excluded from its composite."
            )
    exclusion_notes = [
        f"Excluded: **{name}** — {reason}."
        for name, reason in excluded_participants.items()
    ]
    report = [
        f"# {title}", "",
        f"Participants: **{len(participants)}**. Required missing responses: **{len(missing)}**.",
        workload_note, *exclusion_notes, *applicability_notes, "",
        "## Condition summaries", "",
        "| Scale | Condition | n | Mean ± SD | Median [IQR] |", "|---|---|---:|---:|---:|",
    ]
    for scale, condition, stats in summary_rows:
        report.append(f"| {scale} | {condition} | {stats['n']} | {stats['mean']:.2f} ± {stats['sd']:.2f} | {stats['median']:.2f} [{stats['q1']:.2f}, {stats['q3']:.2f}] |")
    report += ["", "## Friedman omnibus tests", "",
               "| Scale | n | χ²(2) | p | Kendall's W |", "|---|---:|---:|---:|---:|"]
    for scale, n, statistic, p_value, effect in omnibus_rows:
        report.append(f"| {scale} | {n} | {statistic:.2f} | {p_text(p_value)} | {effect:.2f} |")

    report += ["", "## Holm-corrected pairwise Wilcoxon tests", "",
               "| Scale | Condition A | Condition B | n | A − B | W | Raw p | Holm p |",
               "|---|---|---|---:|---:|---:|---:|---:|"]
    for scale, left, right, n, difference, statistic, raw_p, adjusted_p in pairwise_rows:
        report.append(f"| {scale} | {left} | {right} | {n} | {difference:.2f} | {statistic:.2f} | {p_text(raw_p)} | {p_text(adjusted_p)} |")

    report += ["", "## Preference ranking", "",
               "| Condition | Mean rank | Rank 1 / 2 / 3 |", "|---|---:|---:|"]
    for condition, values in zip(conditions, rank_values):
        counts = Counter(int(value) for value in values)
        report.append(f"| {condition} | {mean(values):.2f} | {counts[1]} / {counts[2]} / {counts[3]} |")
    rank_p = "p<.001" if rank_test.pvalue < .001 else f"p={rank_test.pvalue:.3f}"
    report += ["", f"Friedman: χ²(2)={rank_test.statistic:.2f}, {rank_p}, Kendall's W={rank_effect:.2f}.",
               "", "## Interpretation", ""]
    report.extend(f"- {line}" for line in interpretation)
    report += ["", "## Missing required responses", ""]
    if missing:
        for participant, scale, item, question, condition in missing:
            report.append(f"- {participant}: {condition}, {scale} item {item} — {question}")
    else:
        report.append("None.")
    report_output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {summary_output}")
    print(f"Created {tests_output}")
    print(f"Created {report_output}")


def write_study(output: Path, question_by_field, participants, study: str,
                conditions: list[tuple[str, str]], scales: list[tuple[str, str, int]],
                comment_fields: dict[str, str], rank_prefix: str) -> None:
    condition_names = [name for name, _ in conditions]
    columns = ["participant", "source_csv_row", "scale", "item_number", "question"] + condition_names
    records = []

    for source_row, participant, response in participants:
        for scale_name, scale_token, count in scales:
            for item in range(1, count + 1):
                reference = f"{study}_{scale_token}_{conditions[0][1]}_{item}"
                row = {
                    "participant": participant,
                    "source_csv_row": source_row,
                    "scale": scale_name,
                    "item_number": item,
                    "question": short_question(question_by_field[reference]),
                }
                for condition_name, condition_token in conditions:
                    row[condition_name] = response.get(
                        f"{study}_{scale_token}_{condition_token}_{item}", ""
                    )
                records.append(row)

        comment_reference = comment_fields[conditions[0][0]]
        comment_row = {
            "participant": participant,
            "source_csv_row": source_row,
            "scale": "Additional Comments",
            "item_number": 1,
            "question": short_question(question_by_field[comment_reference]),
        }
        for condition_name, _ in conditions:
            comment_row[condition_name] = response.get(comment_fields[condition_name], "")
        records.append(comment_row)

        rank_reference = f"{rank_prefix}_1"
        rank_question = question_by_field[rank_reference].rsplit(" - ", 1)[0]
        rank_row = {
            "participant": participant,
            "source_csv_row": source_row,
            "scale": "Overall Preference Rank",
            "item_number": 1,
            "question": rank_question,
        }
        for rank_index, (condition_name, _) in enumerate(conditions, start=1):
            rank_row[condition_name] = response.get(f"{rank_prefix}_{rank_index}", "")
        records.append(rank_row)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Created {output}: {len(records)} rows")


def main() -> None:
    # Retain the original Qualtrics-to-side-by-side formatter when that export
    # is available, but allow the edited side-by-side analyses to run alone.
    if SOURCE.exists():
        question_by_field, participants = load()
        write_study(
            S2_OUTPUT, question_by_field, participants, "S2",
            [("Freedrive", "Freedrive"), ("AR Handle", "ARHandle"), ("Hybrid", "Hybrid")],
            [("NASA-TLX", "TLX", 6), ("Controllability", "Controllability", 3),
             ("Predictability", "Predictability", 3), ("Safety", "Safety", 2),
             ("Trust", "Trust", 1)],
            {"Freedrive": "Q149", "AR Handle": "Q150", "Hybrid": "Q151"},
            "S2_Rank_Overall",
        )
        write_study(
            S4_OUTPUT, question_by_field, participants, "S4",
            [("Gesture", "Gesture"), ("Language + Gesture", "LangGesture"),
             ("Task-Aware", "TaskAware")],
            [("NASA-TLX", "TLX", 6), ("Confidence", "Confidence", 2),
             ("Usability", "Usability", 2), ("Adaptability", "Adaptability", 2)],
            {"Gesture": "Q152", "Language + Gesture": "Q153", "Task-Aware": "Q154"},
            "S4_Rank_Overall",
        )
    else:
        print(f"Qualtrics source not found; skipping format-only step: {SOURCE}")
    analyze_edited_study(
        EDITED_S2_SOURCE,
        ["Freedrive", "AR Handle", "Hybrid"],
        {"NASA-TLX": 6, "Controllability": 3, "Predictability": 3,
         "Safety": 2, "Trust": 1},
        S2_SUMMARY_OUTPUT, S2_TESTS_OUTPUT, S2_ANALYSIS_OUTPUT,
        "Study 2 Questionnaire Analysis (Edited Side-by-Side Responses)",
        [
            "Hybrid had the lowest descriptive task-specific workload, but the omnibus difference was not significant.",
            "Adjustment Controllability differed strongly: Hybrid was significantly higher than both Freedrive and AR Handle after Holm correction.",
            "Robot Motion Predictability differed: both Freedrive and Hybrid were significantly higher than AR Handle; Freedrive and Hybrid did not differ.",
            "Perceived Operational Safety differed in the omnibus test, but no pairwise comparison survived Holm correction. Trust in Safe Positioning did not differ.",
            "Hybrid was the clear preference: 9 of 11 participants ranked it first, and it ranked significantly above both alternatives after correction.",
        ],
        nasa_items=[1, 2, 5, 6],
        excluded_items={},
        scale_labels={
            "NASA-TLX": "Task-Specific Workload",
            "Controllability": "Adjustment Controllability",
            "Predictability": "Robot Motion Predictability",
            "Safety": "Perceived Operational Safety",
            "Trust": "Trust in Safe Positioning",
        },
    )
    analyze_edited_study(
        EDITED_S4_SOURCE,
        ["Gesture", "Language + Gesture", "Task-Aware"],
        {"NASA-TLX": 6, "Confidence": 2, "Usability": 2, "Adaptability": 2},
        S4_SUMMARY_OUTPUT, S4_TESTS_OUTPUT, S4_ANALYSIS_OUTPUT,
        "Study 4 Questionnaire Analysis (Edited Side-by-Side Responses)",
        [
            "Task-Aware had the lowest workload. The omnibus difference was significant; Gesture workload was significantly higher than Language + Gesture and Task-Aware after Holm correction, while the latter two did not differ.",
            "Request Clarity and Ease differed in the omnibus test, although both corrected pairwise comparisons with Task-Aware narrowly missed significance (p_Holm = .053).",
            "Assistance Confidence did not differ in the omnibus test, so the isolated corrected Language + Gesture versus Task-Aware comparison is treated as exploratory. Interaction Adaptation Burden did not differ.",
            "Task-Aware was ranked first by all 11 participants and ranked significantly above both alternatives after Holm correction.",
        ],
        nasa_items=[1, 2, 3, 4, 5, 6],
        excluded_items={
            "Confidence": {"Gesture": {1}},
            "Adaptability": {"Gesture": {1}},
        },
        scale_labels={
            "Confidence": "Assistance Confidence",
            "Usability": "Request Clarity and Ease",
            "Adaptability": "Interaction Adaptation Burden",
        },
    )


if __name__ == "__main__":
    main()
