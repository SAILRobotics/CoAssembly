#!/usr/bin/env python3
"""Benchmark multiple Hugging Face language and vision-language models.

Results use a long-form pandas table: one row per (model, referring expression).
This makes it easy to filter errors, calculate per-model/per-class statistics,
and add further models without creating a separate output schema for each run.

Examples
--------
List configured models::

    python3 task_graph/evaluate_referring_expression_models.py --list-models

Compare three models::

    python3 task_graph/evaluate_referring_expression_models.py \
      --models qwen25-vl-3b smolvlm2-2.2b gemma3-4b

Resume missing model/sample pairs::

    python3 task_graph/evaluate_referring_expression_models.py \
      --models qwen25-vl-3b smolvlm2-2.2b gemma3-4b --resume
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_referring_expressions_qwen import (
    build_system_prompt,
    parse_prediction,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "referring_expression_responses.csv"
DEFAULT_DESCRIPTION = HERE / "task_description.md"
DEFAULT_RESULTS = HERE / "referring_expression_benchmark_results.csv"
DEFAULT_LEADERBOARD = HERE / "referring_expression_benchmark_leaderboard.csv"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    loader: str = "auto_vlm"
    prompt_style: str = "chat_template"
    trust_remote_code: bool = False
    gated: bool = False
    notes: str = ""
    max_new_tokens: int | None = None  # overrides --max-new-tokens when set
    is_reasoning: bool = False  # always emits <think>...</think> before answering


# One authoritative place for benchmark model names and loading requirements.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen25-vl-3b": ModelSpec(
        "qwen25-vl-3b", "Qwen/Qwen2.5-VL-3B-Instruct", loader="qwen25"),
    "qwen25-3b": ModelSpec(
        "qwen25-3b", "Qwen/Qwen2.5-3B-Instruct",
        loader="causal_lm", prompt_style="causal_chat",
        notes="Text-only control for qwen25-vl-3b — this task never sends images, "
              "so this checks whether the vision tower costs or helps accuracy."),
    "qwen25-vl-7b": ModelSpec(
        "qwen25-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", loader="qwen25"),
    "qwen25-7b": ModelSpec(
        "qwen25-7b", "Qwen/Qwen2.5-7B-Instruct",
        loader="causal_lm", prompt_style="causal_chat",
        notes="Text-only control for qwen25-vl-7b."),
    "qwen3-vl-4b": ModelSpec(
        "qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct"),
    "qwen3-4b": ModelSpec(
        "qwen3-4b", "Qwen/Qwen3-4B-Instruct-2507",
        loader="causal_lm", prompt_style="causal_chat",
        notes="Text-only control for qwen3-vl-4b. Non-thinking instruct release "
              "(the plain Qwen3-4B checkpoint supports a thinking mode that would "
              "need a larger --max-new-tokens budget than this task uses)."),
    "qwen3-vl-8b": ModelSpec(
        "qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct",
        notes="Size-matched to qwen25-vl-7b/qwen25-7b — Qwen3 skips a 7B tier "
              "(0.6B/1.7B/4B/8B/14B/32B), so 8B is the closest comparison point."),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "Qwen/Qwen3-8B",
        loader="causal_lm", prompt_style="causal_chat",
        max_new_tokens=2400, is_reasoning=True,
        notes="Text-only control for qwen3-vl-8b. Unlike qwen3-4b, there is no "
              "non-thinking '-Instruct-2507' release at 8B, so this checkpoint "
              "defaults to thinking mode and needs a larger token budget to "
              "finish its <think> block before answering."),
    "smolvlm2-2.2b": ModelSpec(
        "smolvlm2-2.2b", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    "gemma3-4b": ModelSpec(
        "gemma3-4b", "google/gemma-3-4b-it", gated=True,
        notes="Accept Google's Hugging Face license before downloading."),
    "gemma3-12b": ModelSpec(
        "gemma3-12b", "google/gemma-3-12b-it", gated=True,
        notes="Accept Google's Hugging Face license before downloading. "
              "Larger sibling of gemma3-4b."),
    "gemma4-e2b": ModelSpec(
        "gemma4-e2b", "google/gemma-4-E2B-it",
        notes="Gemma 4 (released 2026-04-02), Apache 2.0 — not gated, unlike "
              "Gemma 3. Edge-optimized size; natively multimodal."),
    "gemma4-e4b": ModelSpec(
        "gemma4-e4b", "google/gemma-4-E4B-it",
        notes="Gemma 4 (released 2026-04-02), Apache 2.0 — not gated, unlike "
              "Gemma 3. Edge-optimized size; natively multimodal."),
    "internvl35-4b": ModelSpec(
        "internvl35-4b", "OpenGVLab/InternVL3_5-4B",
        trust_remote_code=True),
    "llama32-vision-11b": ModelSpec(
        "llama32-vision-11b", "meta-llama/Llama-3.2-11B-Vision-Instruct",
        gated=True,
        notes="Accept Meta's Hugging Face license before downloading."),
}


RESULT_COLUMNS = [
    "run_timestamp",
    "model_alias",
    "model_id",
    "index",
    "description",
    "referring_type",
    "traits_used",
    "target_name",
    "prediction",
    "correct",
    "invalid_output",
    "strict_correct",
    "raw_model_answer",
    "inference_seconds",
    "error",
]


def pandas_module():
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas is required for the multi-model evaluator. Install/fix it "
            "in the same environment used to run the models, for example: "
            "python3 -m pip install --upgrade pandas") from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face LLMs/VLMs on part references.")
    parser.add_argument("--models", nargs="+", default=["qwen25-vl-3b"],
                        help="Registry aliases or arbitrary Hugging Face model IDs")
    parser.add_argument("--all-models", action="store_true",
                        help="Run every configured model (large download/VRAM cost)")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-description", type=Path,
                        default=DEFAULT_DESCRIPTION)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip model/index pairs already stored in results")
    parser.add_argument("--reparse-existing", action="store_true",
                        help="Reparse saved raw answers and rebuild leaderboard only")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def print_registry() -> None:
    print("Configured models:\n")
    for spec in MODEL_REGISTRY.values():
        flags = []
        if spec.gated:
            flags.append("gated")
        if spec.trust_remote_code:
            flags.append("trust_remote_code")
        if spec.max_new_tokens:
            flags.append(f"max_new_tokens={spec.max_new_tokens}")
        suffix = f"  [{' / '.join(flags)}]" if flags else ""
        print(f"  {spec.alias:<22} {spec.model_id}{suffix}")
        if spec.notes:
            print(f"  {'':<22} {spec.notes}")


def resolve_models(names: list[str], all_models: bool) -> list[ModelSpec]:
    if all_models:
        return list(MODEL_REGISTRY.values())
    resolved = []
    for name in names:
        resolved.append(MODEL_REGISTRY.get(
            name, ModelSpec(alias=name.replace("/", "__"), model_id=name)))
    # Preserve requested order while removing duplicates.
    return list({spec.model_id: spec for spec in resolved}.values())


def load_dataset_frame(path: Path, include_unverified: bool):
    pd = pandas_module()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"index", "description", "target_name"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not include_unverified and "Verified" in frame.columns:
        frame = frame[frame["Verified"].str.strip().str.casefold() == "o"]
    frame = frame[(frame["description"].str.strip() != "")
                  & (frame["target_name"].str.strip() != "")]
    return frame.copy()


def load_model(spec: ModelSpec) -> tuple[Any, Any]:
    """Load supported VLM or causal-language-model architectures."""
    from transformers import AutoProcessor

    kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.loader == "qwen25":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                spec.model_id, **kwargs)
        except (ImportError, AttributeError):
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                spec.model_id, **kwargs)
    elif spec.loader == "causal_lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code)
        model.eval()
        return model, tokenizer
    else:
        try:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_class = AutoModelForVision2Seq
        model = model_class.from_pretrained(spec.model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code)
    model.eval()
    return model, processor


def infer_one(model: Any, processor: Any, full_prompt: str,
              max_new_tokens: int, prompt_style: str = "chat_template") -> str:
    """Text-only chat through a VLM processor, consistent across architectures."""
    if prompt_style == "causal_chat":
        messages = [{"role": "user", "content": full_prompt}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": full_prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output_ids[:, prompt_length:]
    return processor.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()


def read_results(path: Path):
    pd = pandas_module()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in RESULT_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[RESULT_COLUMNS]


def save_results(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame[RESULT_COLUMNS].to_csv(temporary, index=False)
    temporary.replace(path)


def build_leaderboard(results: Any, path: Path):
    pd = pandas_module()
    if results.empty:
        leaderboard = pd.DataFrame(columns=[
            "model_alias", "model_id", "samples", "correct", "incorrect",
            "invalid_outputs", "accuracy", "strict_correct", "strict_accuracy",
            "mean_inference_seconds", "std_inference_seconds",
            "total_inference_seconds",
        ])
    else:
        work = results.copy()
        work["correct_bool"] = work["correct"].astype(str).str.casefold() == "true"
        work["invalid_bool"] = (work["prediction"] == "INVALID_OUTPUT")
        work["strict_bool"] = (
            work.get("strict_correct", "").astype(str).str.casefold() == "true")
        work["seconds"] = pd.to_numeric(work["inference_seconds"], errors="coerce")
        leaderboard = work.groupby(
            ["model_alias", "model_id"], as_index=False).agg(
                samples=("index", "count"),
                correct=("correct_bool", "sum"),
                invalid_outputs=("invalid_bool", "sum"),
                accuracy=("correct_bool", "mean"),
                strict_correct=("strict_bool", "sum"),
                strict_accuracy=("strict_bool", "mean"),
                mean_inference_seconds=("seconds", "mean"),
                std_inference_seconds=("seconds", "std"),
                total_inference_seconds=("seconds", "sum"),
            )
        leaderboard["incorrect"] = (
            leaderboard["samples"] - leaderboard["correct"])
        leaderboard = leaderboard[[
            "model_alias", "model_id", "samples", "correct", "incorrect",
            "invalid_outputs", "accuracy", "strict_correct", "strict_accuracy",
            "mean_inference_seconds", "std_inference_seconds",
            "total_inference_seconds",
        ]].sort_values(["accuracy", "mean_inference_seconds"],
                       ascending=[False, True])
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False)
    return leaderboard


def release_model(model: Any, processor: Any) -> None:
    del model, processor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    if args.list_models:
        print_registry()
        return 0
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")

    pd = pandas_module()
    dataset = load_dataset_frame(args.dataset, args.include_unverified)
    candidates = sorted(dataset["target_name"].unique().tolist())
    if args.start_index is not None:
        numeric_index = pd.to_numeric(dataset["index"], errors="coerce")
        dataset = dataset[numeric_index >= args.start_index]
    if args.limit is not None:
        dataset = dataset.head(args.limit)

    results = read_results(args.results)
    if args.reparse_existing:
        reasoning_ids = {spec.model_id for spec in MODEL_REGISTRY.values()
                         if spec.is_reasoning}

        def reparse_prediction(raw: str, model_id: str) -> str:
            if model_id in reasoning_ids and "</think>" not in raw:
                return "INVALID_OUTPUT"
            return parse_prediction(raw, candidates)

        results["prediction"] = results.apply(
            lambda row: reparse_prediction(row["raw_model_answer"], row["model_id"]),
            axis=1)
        results["correct"] = (
            results["prediction"] == results["target_name"]).astype(str)
        results["invalid_output"] = (
            results["prediction"] == "INVALID_OUTPUT").astype(str)
        results["strict_correct"] = (
            results["raw_model_answer"].str.strip()
            == results["target_name"]).astype(str)
        save_results(results, args.results)
        leaderboard = build_leaderboard(results, args.leaderboard)
        print(leaderboard.to_string(index=False))
        return 0

    task_description = args.task_description.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(task_description, candidates)
    specs = resolve_models(args.models, args.all_models)
    if not args.resume:
        # Replace only rows belonging to requested models. Results from models
        # not in this invocation remain available in the combined benchmark.
        requested = {spec.model_id for spec in specs}
        results = results[~results["model_id"].isin(requested)].copy()

    load_failures: list[tuple[ModelSpec, str]] = []
    for spec in specs:
        done = set()
        if args.resume and not results.empty:
            done = set(results.loc[
                results["model_id"] == spec.model_id, "index"].astype(str))
        pending = dataset[~dataset["index"].astype(str).isin(done)]
        print(f"\n=== {spec.alias}: {spec.model_id} ===")
        print(f"Pending {len(pending)} / {len(dataset)} samples")
        if pending.empty:
            continue

        print("Loading model ...", flush=True)
        try:
            model, processor = load_model(spec)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            load_failures.append((spec, message))
            print(f"SKIPPED: could not load {spec.model_id}\n  {message}",
                  file=sys.stderr, flush=True)
            continue
        checkpoint_rows: list[dict[str, Any]] = []
        timestamp = datetime.now().isoformat(timespec="seconds")
        max_new_tokens = spec.max_new_tokens or args.max_new_tokens
        try:
            for position, (_, row) in enumerate(pending.iterrows(), start=1):
                user_prompt = (
                    system_prompt
                    + "\n\n--- CURRENT REFERRING EXPRESSION ---\n"
                    + str(row["description"])
                    + "\n--- END EXPRESSION ---\n\n"
                    + "Return exactly one allowed dataset label and nothing else.")
                started = time.perf_counter()
                error = ""
                try:
                    raw = infer_one(model, processor, user_prompt,
                                    max_new_tokens, spec.prompt_style)
                except Exception as exc:
                    raw = ""
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                # A reasoning model that never closes its <think> block ran out of
                # budget before committing to an answer at all -- whatever labels
                # appear in the unfinished reasoning are not a real answer, so this
                # is unresolved/invalid rather than something to extract from.
                truncated_reasoning = (
                    spec.is_reasoning and not error and "</think>" not in raw)
                if error or truncated_reasoning:
                    prediction = "INVALID_OUTPUT"
                else:
                    prediction = parse_prediction(raw, candidates)
                correct = prediction == row["target_name"]
                # No normalization at all: the model is only "strictly"
                # correct if it followed the "return only that label, with
                # identical spelling" instruction to the letter.
                strict_correct = raw.strip() == row["target_name"]
                checkpoint_rows.append({
                    "run_timestamp": timestamp,
                    "model_alias": spec.alias,
                    "model_id": spec.model_id,
                    "index": str(row["index"]),
                    "description": str(row["description"]),
                    "referring_type": str(row.get("referring_type", "")),
                    "traits_used": str(row.get("traits_used", "")),
                    "target_name": str(row["target_name"]),
                    "prediction": prediction,
                    "correct": str(correct),
                    "invalid_output": str(prediction == "INVALID_OUTPUT"),
                    "strict_correct": str(strict_correct),
                    "raw_model_answer": raw,
                    "inference_seconds": f"{elapsed:.4f}",
                    "error": error,
                })
                status = "OK" if correct else "WRONG"
                print(f"[{position:>3}/{len(pending)}] index={row['index']} "
                      f"{status}: {prediction} "
                      f"(truth={row['target_name']}, {elapsed:.2f}s)", flush=True)

                if (len(checkpoint_rows) >= args.checkpoint_every
                        or position == len(pending)):
                    results = pd.concat(
                        [results, pd.DataFrame(checkpoint_rows)], ignore_index=True)
                    checkpoint_rows.clear()
                    save_results(results, args.results)
                    build_leaderboard(results, args.leaderboard)
        finally:
            release_model(model, processor)

    leaderboard = build_leaderboard(results, args.leaderboard)
    print("\n=== FINAL LEADERBOARD ===")
    print(leaderboard.to_string(index=False))
    print(f"\nDetailed results: {args.results}")
    print(f"Leaderboard:      {args.leaderboard}")
    if load_failures:
        print("\nModels skipped because loading failed:", file=sys.stderr)
        for spec, message in load_failures:
            print(f"  {spec.alias}: {message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
