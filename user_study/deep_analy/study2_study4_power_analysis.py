#!/usr/bin/env python3
"""Approximate power/sensitivity analysis for the three-condition Friedman tests."""

from __future__ import annotations

import csv
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


ROOT = Path(__file__).resolve().parent
ALPHA = 0.05
N = 11
CONDITIONS = 3
DF = CONDITIONS - 1
CRITICAL_VALUE = chi2.ppf(1 - ALPHA, DF)

EFFECTS = {
    "Study 2": {
        "Task-Specific Workload": 0.13,
        "Adjustment Controllability": 0.69,
        "Robot Motion Predictability": 0.48,
        "Perceived Operational Safety": 0.31,
        "Trust in Safe Positioning": 0.03,
        "Preference Rank": 0.52,
    },
    "Study 4": {
        "NASA-TLX": 0.42,
        "Assistance Confidence": 0.14,
        "Request Clarity and Ease": 0.39,
        "Interaction Adaptation Burden": 0.22,
        "Preference Rank": 0.75,
    },
}


def approximate_power(n: int, kendalls_w: float) -> float:
    """Noncentral-chi-square approximation to Friedman-test power."""
    noncentrality = n * (CONDITIONS - 1) * kendalls_w
    return float(ncx2.sf(CRITICAL_VALUE, DF, noncentrality))


def required_n(kendalls_w: float, target: float) -> int | None:
    if kendalls_w <= 0:
        return None
    for sample_size in range(3, 10001):
        if approximate_power(sample_size, kendalls_w) >= target:
            return sample_size
    return None


def detectable_w(target: float) -> float:
    return float(brentq(lambda w: approximate_power(N, w) - target, 1e-9, 1.0))


def main() -> None:
    rows = []
    for study, measures in EFFECTS.items():
        for measure, effect in measures.items():
            rows.append({
                "study": study,
                "measure": measure,
                "n": N,
                "kendalls_w": effect,
                "approx_achieved_power": approximate_power(N, effect),
                "n_for_80_percent_power": required_n(effect, 0.80),
                "n_for_90_percent_power": required_n(effect, 0.90),
            })

    csv_path = ROOT / "study2_study4_power_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_path = ROOT / "study2_study4_power_analysis.md"
    report = [
        "# Study 2 and Study 4 Power Analysis", "",
        f"Three-condition Friedman tests; alpha={ALPHA:.2f}, N={N}. Power is based on a "
        "noncentral-chi-square approximation with lambda = N(k-1)W.", "",
        f"At N={N}, the minimum detectable Kendall's W is {detectable_w(.80):.3f} for "
        f"80% power and {detectable_w(.90):.3f} for 90% power.", "",
        "| Study | Measure | W | Approx. power | N for 80% | N for 90% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['study']} | {row['measure']} | {row['kendalls_w']:.2f} | "
            f"{row['approx_achieved_power']:.3f} | {row['n_for_80_percent_power']} | "
            f"{row['n_for_90_percent_power']} |"
        )
    report += [
        "", "Post-hoc power is descriptive and should not be used to reinterpret a "
        "nonsignificant p-value. The sensitivity threshold and prospective sample-size "
        "estimates are the more useful planning results.",
    ]
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {csv_path}")
    print(f"Created {md_path}")


if __name__ == "__main__":
    main()
