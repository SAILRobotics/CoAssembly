#!/usr/bin/env python3
"""Pull every non-clean language-grounding moment from Study 4 for manual review.

A `vlm_interaction` row is "clean" when graph_decision is language_grounded /
language_colocated_grounded / language_plural_grounded (the request resolved
to exactly one correct, acquirable part and drove a real acquisition). Every
other decision is printed with its transcript, the system's spoken response,
and step context, grouped by decision type, so a human can judge whether the
moment reflects a user-side issue (ambiguous phrasing, referencing something
out of scope or already acquired) or a system-side issue (the model failed to
resolve a request that, from the transcript, looks clear).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "study_logs" / "study4"
CONDITIONS = ("language", "task_aware")
CLEAN_DECISIONS = {"language_grounded", "language_colocated_grounded",
                   "language_plural_grounded"}
EXCLUDE_PARTICIPANTS = {"test"}
OUTPUT = ROOT / "study4_conversation_review.md"


def main() -> None:
    by_decision: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for condition in CONDITIONS:
        for path in sorted(LOG_DIR.glob(f"*/{condition}.csv")):
            participant = path.parent.name
            if participant in EXCLUDE_PARTICIPANTS:
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if row.get("event_type") != "vlm_interaction":
                        continue
                    total += 1
                    decision = row.get("graph_decision", "")
                    if decision in CLEAN_DECISIONS:
                        continue
                    by_decision[decision].append({
                        "participant": participant,
                        "condition": condition,
                        "step_id": row.get("step_id", ""),
                        "step_title": row.get("step_title", ""),
                        "transcript": row.get("transcript", ""),
                        "spoken_response": row.get("spoken_response", ""),
                        "vlm_prediction": row.get("vlm_prediction", ""),
                        "timestamp": row.get("timestamp", ""),
                    })

    non_clean = sum(len(v) for v in by_decision.values())
    lines = [
        "# Study 4 Non-Clean Language Grounding — Full Transcripts", "",
        f"Total `vlm_interaction` events (language + task_aware): **{total}**. "
        f"Non-clean-grounding: **{non_clean}** ({non_clean/total:.1%}).",
        "",
        "Each entry: participant, condition, step, what the user said, what "
        "the system said back, and the system's structured prediction. Judge "
        "each as user-side (ambiguous/out-of-scope phrasing) or system-side "
        "(a clear request the model still failed to resolve).",
        "",
    ]
    for decision, entries in sorted(by_decision.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"## `{decision}` ({len(entries)})")
        lines.append("")
        for entry in entries:
            lines.append(
                f"- **{entry['participant']}** / {entry['condition']} / "
                f"{entry['step_id']} ({entry['step_title']})"
            )
            lines.append(f"  - Said: \"{entry['transcript']}\"")
            lines.append(f"  - System replied: \"{entry['spoken_response']}\"")
            if entry["vlm_prediction"]:
                lines.append(f"  - Prediction: `{entry['vlm_prediction']}`")
        lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT}: {non_clean} non-clean events across "
          f"{len(by_decision)} decision types.")


if __name__ == "__main__":
    main()
