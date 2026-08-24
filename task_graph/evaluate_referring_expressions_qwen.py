#!/usr/bin/env python3
"""Evaluate Qwen on the individual-part referring-expression dataset.

The evaluator supplies ``task_description.md`` as system context, submits one
CSV ``description`` at a time, asks for exactly one dataset ``target_name``, and
reports top-1 exact-match accuracy.  It never edits the source dataset; raw
answers and parsed predictions are saved incrementally to a separate CSV.

Examples
--------
Evaluate the full verified dataset::

    python3 task_graph/evaluate_referring_expressions_qwen.py

Quick ten-sample check::

    python3 task_graph/evaluate_referring_expressions_qwen.py --limit 10

Resume an interrupted run::

    python3 task_graph/evaluate_referring_expressions_qwen.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "referring_expression_responses.csv"
DEFAULT_DESCRIPTION = HERE / "task_description.md"
DEFAULT_OUTPUT = HERE / "referring_expression_qwen_predictions.csv"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

RESULT_FIELDS = [
    "index",
    "description",
    "target_name",
    "prediction",
    "correct",
    "raw_model_answer",
    "model_name",
    "inference_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Qwen top-1 part-reference classification accuracy.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-description", type=Path,
                        default=DEFAULT_DESCRIPTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face Qwen VL model name or local path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate at most this many remaining rows")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Evaluate CSV rows whose integer index is at least this value")
    parser.add_argument("--include-unverified", action="store_true",
                        help="Include rows not marked 'o' in the Verified column")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the output and skip source indices already evaluated")
    parser.add_argument(
        "--reparse-existing", action="store_true",
        help="Reparse/rescore saved raw answers without loading or running Qwen")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def load_dataset(path: Path, include_unverified: bool) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"index", "description", "target_name"}
    missing = required.difference(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")
    if not include_unverified and rows and "Verified" in rows[0]:
        rows = [row for row in rows
                if row.get("Verified", "").strip().casefold() == "o"]
    return [row for row in rows
            if row.get("description", "").strip()
            and row.get("target_name", "").strip()]


def build_system_prompt(task_description: str,
                        candidate_names: list[str]) -> str:
    candidates = "\n".join(f"- {name}" for name in candidate_names)
    return f"""You are evaluating referring expressions for a gearbox assembly.
Identify the single physical part meant by the user's expression.

Use the task description for terminology, colors, shapes, rows, left/right,
spatial relations, and part functions. Select exactly one label from the
allowed dataset labels below. Return only that label, with identical spelling.
Do not explain your choice and do not return JSON.

Most allowed labels are exactly the canonical task-graph identifiers used in
the task description below (e.g. `BASE_BOARD`, `STAND_ROW2_LEFT`,
`GEAR_ROW2_RIGHT`) — no conversion is needed for those. Three labels are
exceptions: `BEARING`, `PIN`, and `SCREW_ROW1`..`SCREW_ROW4` each group
several row/side-specific task-graph parts that are physically identical and
rendered as a single part in this dataset. Use the group label rather than a
specific row/side identifier for those, e.g. `BEARING` (not
`BEARING_ROW2_LEFT`) and `SCREW_ROW2` (not `SCREW_ROW2_LEFT` or
`SCREW_ROW2_RIGHT`).

The task description below also has a "Produced Subassembly Names" section
(identifiers ending in `_ASSEMBLY`, e.g. `GEAR_ROD_ROW1_ASSEMBLY`). Those
describe build state — the intermediate or finished products of assembly
steps — not which individual part a referring expression means. None of
them are allowed labels; never answer with an `_ASSEMBLY` name.

--- ALLOWED DATASET LABELS ---
{candidates}
--- END ALLOWED DATASET LABELS ---

--- TASK DESCRIPTION ---
{task_description}
--- END TASK DESCRIPTION ---
"""


def load_qwen(model_name: str) -> tuple[Any, Any]:
    """Use the same Qwen2.5-VL/Qwen2-VL fallback as ``vlm_assistant.py``."""
    try:
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
    except (ImportError, AttributeError, OSError):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def infer_one(model: Any, processor: Any, system_prompt: str,
              description: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "Referring expression: " + description +
                "\n\nWhich one part does this refer to? Return one allowed "
                "dataset label only.")},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True,
                       return_tensors="pt").to(model.device)

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [output[len(source):]
               for source, output in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()


def parse_prediction(raw_answer: str, candidates: list[str]) -> str:
    """Convert a minimally non-compliant answer to one allowed exact label."""
    # Reasoning models (e.g. DeepSeek-R1 distills) prefix their answer with a
    # <think>...</think> block. The opening tag is usually injected by the
    # chat template into the prompt rather than generated, so only the
    # closing tag reliably appears in the output; keep just what follows it
    # so reasoning-time label mentions aren't mistaken for the final answer.
    if "</think>" in raw_answer:
        raw_answer = raw_answer.rsplit("</think>", 1)[1].strip()
    by_folded = {name.casefold(): name for name in candidates}
    cleaned = raw_answer.strip().strip("`\"' .,:;\n\t")
    if cleaned.casefold() in by_folded:
        return by_folded[cleaned.casefold()]

    # target_name now reuses the task-graph identifiers from
    # task_description.md directly, so well-formed answers already matched
    # above. The one remaining gap: task_description.md exhaustively lists
    # 43 row/side-specific identifiers (e.g. BEARING_ROW2_LEFT), while
    # BEARING/PIN/SCREW_ROW{n} are grouped rendering classes with no single
    # row/side identifier of their own (see task_graph/part_naming_mapping.md).
    # Collapse a specific identifier down to its group when that's all this
    # dataset can distinguish.
    canonical = cleaned.upper().removesuffix(".STL")
    if "BEARING" in canonical:
        return "BEARING"
    if "PIN" in canonical:
        return "PIN"
    screw = re.search(r"SCREW_ROW([1-4])", canonical)
    if screw:
        return f"SCREW_ROW{screw.group(1)}"

    # Also accept a JSON value if a model ignores the no-JSON instruction.
    try:
        value = json.loads(raw_answer)
        if isinstance(value, str) and value.casefold() in by_folded:
            return by_folded[value.casefold()]
        if isinstance(value, dict):
            for key in ("target_name", "prediction", "label"):
                candidate = str(value.get(key, "")).strip().casefold()
                if candidate in by_folded:
                    return by_folded[candidate]
    except (json.JSONDecodeError, TypeError):
        pass

    # Last-resort extraction permits prose but only when exactly one distinct
    # allowed label occurs. Multiple mentioned labels remain an invalid answer.
    matches: list[str] = []
    for name in sorted(candidates, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                     raw_answer, flags=re.IGNORECASE):
            matches.append(name)
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else "INVALID_OUTPUT"


def load_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["index"]: row for row in csv.DictReader(handle)
                if row.get("index")}


def reparse_existing_output(path: Path, candidates: list[str]) -> list[dict[str, str]]:
    """Rescore existing raw answers atomically without another model run."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        prediction = parse_prediction(row.get("raw_model_answer", ""), candidates)
        row["prediction"] = prediction
        row["correct"] = str(prediction == row.get("target_name", ""))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in RESULT_FIELDS}
                         for row in rows)
    temporary.replace(path)
    return rows


def print_metrics(results: list[dict[str, str]]) -> None:
    total = len(results)
    correct = sum(row.get("correct", "").casefold() == "true"
                  for row in results)
    invalid = sum(row.get("prediction") == "INVALID_OUTPUT"
                  for row in results)
    accuracy = correct / total if total else 0.0
    print("\n=== EVALUATION RESULT ===")
    print(f"Samples:        {total}")
    print(f"Correct:        {correct}")
    print(f"Incorrect:      {total - correct}")
    print(f"Invalid output: {invalid}")
    print(f"Top-1 accuracy: {accuracy:.2%}")


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.dataset.resolve():
        raise ValueError("--output must differ from --dataset; the source CSV is read-only")
    all_rows = load_dataset(args.dataset, args.include_unverified)
    # Keep the answer vocabulary global even when evaluating a range/limit;
    # filtering samples must not accidentally remove valid classes.
    candidate_names = sorted({row["target_name"].strip() for row in all_rows})
    rows = all_rows
    if args.start_index is not None:
        rows = [row for row in rows
                if int(row["index"]) >= args.start_index]
    if not rows:
        print("No matching dataset rows to evaluate.", file=sys.stderr)
        return 2

    if args.reparse_existing:
        reparsed = reparse_existing_output(args.output, candidate_names)
        print(f"Reparsed {len(reparsed)} saved raw answers in {args.output}")
        print_metrics(reparsed)
        return 0

    completed = load_completed(args.output) if args.resume else {}
    pending = [row for row in rows if row["index"] not in completed]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        pending = pending[:args.limit]

    task_description = args.task_description.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(task_description, candidate_names)
    print(f"Dataset:    {args.dataset}")
    print(f"Context:    {args.task_description}")
    print(f"Model:      {args.model}")
    print(f"Classes:    {len(candidate_names)}")
    print(f"Pending:    {len(pending)} / {len(rows)}")
    print(f"Predictions:{args.output}")

    if not pending:
        print_metrics(list(completed.values()))
        return 0

    print("Loading Qwen model ...", flush=True)
    model, processor = load_qwen(args.model)
    print("Model ready.\n", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if mode == "w" or args.output.stat().st_size == 0:
            writer.writeheader()

        for position, row in enumerate(pending, start=1):
            start = time.perf_counter()
            raw = infer_one(model, processor, system_prompt,
                            row["description"], args.max_new_tokens)
            elapsed = time.perf_counter() - start
            prediction = parse_prediction(raw, candidate_names)
            correct = prediction == row["target_name"]
            result = {
                "index": row["index"],
                "description": row["description"],
                "target_name": row["target_name"],
                "prediction": prediction,
                "correct": str(correct),
                "raw_model_answer": raw,
                "model_name": args.model,
                "inference_seconds": f"{elapsed:.3f}",
            }
            writer.writerow(result)
            handle.flush()
            completed[row["index"]] = result
            mark = "OK" if correct else "WRONG"
            print(f"[{position:>3}/{len(pending)}] index={row['index']} "
                  f"{mark}: {prediction} (truth={row['target_name']}, "
                  f"{elapsed:.2f}s)", flush=True)

    # Report only rows relevant to this invocation's filtered dataset, while
    # including prior resumed predictions for those indices.
    evaluated = [completed[row["index"]] for row in rows
                 if row["index"] in completed]
    print_metrics(evaluated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
