#!/usr/bin/env python3
"""Combine per-model result CSVs in inference_summary/ into one unified table.

Run this after evaluate_referring_expression_models.py has been run once per
model with --results task_graph/inference_summary/<alias>_results.csv. It
concatenates every *_results.csv found there into a single long-form
DataFrame (one row per model/sample, matching RESULT_COLUMNS) and rebuilds a
combined leaderboard from it.

Usage
-----
    python3 task_graph/combine_inference_summary.py
"""

from __future__ import annotations

from pathlib import Path

from evaluate_referring_expression_models import (
    RESULT_COLUMNS,
    build_leaderboard,
    pandas_module,
)

HERE = Path(__file__).resolve().parent
SUMMARY_DIR = HERE / "inference_summary"
COMBINED_RESULTS = SUMMARY_DIR / "all_models_results.csv"
COMBINED_LEADERBOARD = SUMMARY_DIR / "all_models_leaderboard.csv"


def main() -> int:
    pd = pandas_module()
    result_files = sorted(
        p for p in SUMMARY_DIR.glob("*_results.csv")
        if p.name != COMBINED_RESULTS.name
    )
    if not result_files:
        print(f"No *_results.csv files found in {SUMMARY_DIR}")
        return 1

    frames = []
    for path in result_files:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in RESULT_COLUMNS:
            if column not in frame:
                frame[column] = ""
        frames.append(frame[RESULT_COLUMNS])
        print(f"  loaded {len(frame):>4} rows  <-  {path.name}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["model_id", "index"], keep="last")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_RESULTS, index=False)
    print(f"\nCombined results:    {COMBINED_RESULTS}  ({len(combined)} rows)")

    leaderboard = build_leaderboard(combined, COMBINED_LEADERBOARD)
    print(f"Combined leaderboard:{COMBINED_LEADERBOARD}")
    print()
    print(leaderboard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
