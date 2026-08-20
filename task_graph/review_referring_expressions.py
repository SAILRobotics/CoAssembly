#!/usr/bin/env python3
"""Fast keyboard reviewer for single-target and ambiguous expressions.

Keys:
    O       mark the current expression verified and advance
    X       mark the current expression rejected and advance
    C       clear the current decision
    Left    previous expression
    Right   next expression
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import messagebox


ROOT = Path(__file__).resolve().parents[1]
TASK_GRAPH_DIR = Path(__file__).resolve().parent
DATASETS = {
    "responses": TASK_GRAPH_DIR / "referring_expression_responses.csv",
    "ambiguous": TASK_GRAPH_DIR / "referring_expression_ambiguous.csv",
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import referring_expression_test_babylon as study_app  # noqa: E402


class Reviewer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Referring-expression verifier")
        self.root.geometry("1040x760")
        self.root.minsize(760, 560)
        self.root.configure(bg="#111820")
        self.only_unreviewed = tk.BooleanVar(value=True)
        self.dataset_mode = tk.StringVar(value="responses")
        self.dataset_key = "responses"
        self.csv_path = DATASETS[self.dataset_key]
        self.fields: list[str] = []
        self.rows: list[dict[str, str]] = []
        self.position = 0
        self.load_dataset(self.dataset_key)

        modes = tk.Frame(self.root, bg="#111820")
        modes.pack(pady=(20, 3))
        tk.Label(modes, text="Dataset:", bg="#111820", fg="#a9bacb",
                 font=("Sans", 12, "bold")).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            modes, text="Single-target responses", value="responses",
            variable=self.dataset_mode, command=self.switch_dataset,
            bg="#111820", fg="#dbe7f2", activebackground="#111820",
            activeforeground="white", selectcolor="#263746",
            font=("Sans", 12),
        ).pack(side="left", padx=5)
        tk.Radiobutton(
            modes, text="Ambiguous / plural references", value="ambiguous",
            variable=self.dataset_mode, command=self.switch_dataset,
            bg="#111820", fg="#dbe7f2", activebackground="#111820",
            activeforeground="white", selectcolor="#263746",
            font=("Sans", 12),
        ).pack(side="left", padx=5)

        self.progress = tk.Label(
            self.root, bg="#111820", fg="#a9bacb", font=("Sans", 13))
        self.progress.pack(pady=(8, 4))
        self.target = tk.Label(
            self.root, bg="#111820", fg="#e4edf5", font=("Sans", 19, "bold"))
        self.target.pack(pady=5)
        self.metadata = tk.Label(
            self.root, bg="#111820", fg="#91a5b8", font=("Sans", 12),
            wraplength=940, justify="center")
        self.metadata.pack(pady=3)
        self.context = tk.Label(
            self.root, bg="#111820", fg="#b7c7d6", font=("Sans", 11),
            wraplength=940, justify="left")
        self.context.pack(padx=40, pady=3)
        self.description = tk.Label(
            self.root, bg="#1a2530", fg="white", font=("Sans", 22),
            wraplength=840, justify="center", padx=35, pady=45)
        self.description.pack(fill="both", expand=True, padx=50, pady=24)
        self.decision = tk.Label(
            self.root, bg="#111820", font=("Sans", 16, "bold"))
        self.decision.pack(pady=4)

        tk.Checkbutton(
            self.root, text="Only unreviewed", variable=self.only_unreviewed,
            command=self.filter_changed, bg="#111820", fg="#dbe7f2",
            activebackground="#111820", activeforeground="white",
            selectcolor="#263746", font=("Sans", 12),
        ).pack(pady=(4, 2))

        buttons = tk.Frame(self.root, bg="#111820")
        buttons.pack(pady=(8, 25))
        tk.Button(buttons, text="← Previous", command=lambda: self.move(-1), width=14).pack(side="left", padx=5)
        tk.Button(buttons, text="O  Verified", command=lambda: self.mark("o"), width=14,
                  bg="#247a49", fg="white").pack(side="left", padx=5)
        tk.Button(buttons, text="X  Reject", command=lambda: self.mark("x"), width=14,
                  bg="#9a3030", fg="white").pack(side="left", padx=5)
        tk.Button(buttons, text="C  Clear", command=self.clear, width=12).pack(side="left", padx=5)
        tk.Button(buttons, text="Next →", command=lambda: self.move(1), width=14).pack(side="left", padx=5)

        for key in ("o", "O"):
            self.root.bind(key, lambda _event: self.mark("o"))
        for key in ("x", "X"):
            self.root.bind(key, lambda _event: self.mark("x"))
        for key in ("c", "C"):
            self.root.bind(key, lambda _event: self.clear())
        self.root.bind("<Left>", lambda _event: self.move(-1))
        self.root.bind("<Right>", lambda _event: self.move(1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.show()

    def load_dataset(self, key: str) -> None:
        path = DATASETS[key]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"No responses found in {path}")
        if "Verified" not in fields:
            raise RuntimeError(f"The dataset has no Verified column: {path}")
        self.dataset_key = key
        self.csv_path = path
        self.fields = fields
        self.rows = rows
        self.position = next(
            (index for index, row in enumerate(rows)
             if not row.get("Verified", "").strip()),
            0,
        )
        self.root.title(f"Referring-expression verifier — {path.name}")

    def switch_dataset(self) -> None:
        requested = self.dataset_mode.get()
        if requested == self.dataset_key:
            return
        try:
            self.save()
            self.load_dataset(requested)
            self.show()
        except (OSError, RuntimeError) as exc:
            self.dataset_mode.set(self.dataset_key)
            messagebox.showerror("Could not switch dataset", str(exc))

    def save(self) -> None:
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.csv_path)
        if self.dataset_key == "responses":
            study_app.rebuild_xlsx()

    def show(self) -> None:
        row = self.rows[self.position]
        counts = Counter(item.get("Verified", "").lower() for item in self.rows)
        self.progress.configure(
            text=(f"{self.csv_path.name}     {self.position + 1} / {len(self.rows)}     "
                  f"verified: {counts['o']}     rejected: {counts['x']}     "
                  f"unreviewed: {counts['']}"))
        if self.dataset_key == "responses":
            self.target.configure(text=row["target_name"])
            self.metadata.configure(
                text=(f"CSV index {row['index']}  •  presentation "
                      f"{row['presentation_number']}/{row['target_presentations']}  •  "
                      f"{row['referring_type']}  •  {row['traits_used']}"))
            self.context.configure(text=f"Target ID: {row['target_id']}")
        else:
            self.target.configure(text=f"Ambiguous {row['object_category']}")
            self.metadata.configure(
                text=(f"CSV index {row['index']}  •  {row['reference_intent']}  •  "
                      f"expected behavior: {row['expected_behavior']}  •  "
                      f"known: {row['known_traits']}  •  missing: {row['missing_traits']}"))
            self.context.configure(
                text=(f"Candidates ({row['candidate_count']}): "
                      f"{row['candidate_ids'].replace('|', ', ')}\n"
                      f"Expected response: {row['expected_response']}"))
        self.description.configure(text=row["description"])
        status = row.get("Verified", "").strip().lower()
        if status == "o":
            self.decision.configure(text="VERIFIED (O)", fg="#58d68d")
        elif status == "x":
            self.decision.configure(text="REJECTED (X)", fg="#ff6b6b")
        else:
            self.decision.configure(text="UNREVIEWED", fg="#e6c75a")

    def mark(self, decision: str) -> None:
        self.rows[self.position]["Verified"] = decision
        self.save()
        if self.only_unreviewed.get():
            candidates = self.unreviewed_positions()
            next_unreviewed = next(
                (index for index in candidates if index > self.position),
                candidates[0] if candidates else None,
            )
            if next_unreviewed is not None:
                self.position = next_unreviewed
        elif self.position < len(self.rows) - 1:
            self.position += 1
        self.show()

    def clear(self) -> None:
        self.rows[self.position]["Verified"] = ""
        self.save()
        self.show()

    def move(self, delta: int) -> None:
        if self.only_unreviewed.get():
            positions = self.unreviewed_positions()
            if not positions:
                self.show()
                return
            if delta < 0:
                self.position = next(
                    (index for index in reversed(positions) if index < self.position),
                    positions[0],
                )
            else:
                self.position = next(
                    (index for index in positions if index > self.position),
                    positions[-1],
                )
        else:
            self.position = max(0, min(len(self.rows) - 1, self.position + delta))
        self.show()

    def unreviewed_positions(self) -> list[int]:
        return [
            index for index, row in enumerate(self.rows)
            if not row.get("Verified", "").strip()
        ]

    def filter_changed(self) -> None:
        if self.only_unreviewed.get():
            positions = self.unreviewed_positions()
            if positions and self.position not in positions:
                self.position = next(
                    (index for index in positions if index > self.position),
                    positions[0],
                )
        self.show()

    def close(self) -> None:
        try:
            self.save()
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Reviewer().run()
