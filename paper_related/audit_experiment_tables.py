#!/usr/bin/env python3
"""Recompute provenance values used by Experiments 2 and 3 in root.tex."""

from pathlib import Path
import csv
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def study2_objective():
    path = PROJECT / "study_logs/study2/study2_efficiency_conditions.csv"
    fields = {
        "Right-hand path (m)": "hand_movement_m",
        "Interaction time (s)": "interaction_time_s",
        "Completion time (s)": "completion_time_s",
        "Interactions (#)": "num_interactions",
        "Trans. gain/hand (m/m)": "translation_per_hand_m",
        "Rot. gain/hand (deg/m)": "rotation_per_hand_m",
    }
    conditions = ("freedrive", "ar", "hybrid")
    raw = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for label, field in fields.items():
                value = number(row[field])
                if value is not None:
                    raw[(row["participant"], row["trial_condition"])][label].append(value)
    participants = sorted({p for p, _ in raw})
    print("STUDY 2 OBJECTIVE")
    for label in fields:
        columns = []
        for condition in conditions:
            values = [mean(raw[(p, condition)][label]) for p in participants
                      if raw[(p, condition)][label]]
            columns.append(values)
            print(label, condition, len(values), mean(values), stdev(values))
        result = friedmanchisquare(*columns)
        print("  Friedman", result.statistic, result.pvalue,
              "W", result.statistic / (len(columns[0]) * 2))
        pair_results = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            result_pair = wilcoxon(columns[a], columns[b], zero_method="wilcox",
                                   alternative="two-sided")
            pair_results.append((conditions[a], conditions[b],
                                 result_pair.statistic, result_pair.pvalue))
        adjusted = holm([x[3] for x in pair_results])
        for pair, p_holm in zip(pair_results, adjusted):
            print("  Pair", *pair, "Holm", p_holm)


if __name__ == "__main__":
    study2_objective()
