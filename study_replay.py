"""Shared JSONL recorder and auto-detecting replay entry point for user studies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class ReplayRecorder:
    """Line-buffered recorder shared by Study 2 and Study 3."""

    def __init__(self, path: str | Path, schema: str, session_id: str,
                 *, overwrite: bool = False, **base_fields):
        self.path = Path(path)
        self.schema = schema
        self.session_id = session_id
        self.base_fields = dict(base_fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "w" if overwrite else "a", encoding="utf-8", buffering=1)

    def record(self, record_type: str, **payload) -> None:
        record = {
            "schema": self.schema,
            "type": record_type,
            "time_unix_s": time.time(),
            "time_monotonic_s": time.perf_counter(),
            "session_id": self.session_id,
            **self.base_fields,
            **payload,
        }
        self._handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def _detect_schema(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                schema = json.loads(line).get("schema")
            except json.JSONDecodeError:
                continue
            if schema in {"study3_replay_v1", "workholding_replay_v1"}:
                return schema
    raise SystemExit(f"No supported replay records found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Study 2 or Study 3 JSONL (schema auto-detected)")
    parser.add_argument("log", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    schema = _detect_schema(args.log)
    forwarded = [str(args.log), "--speed", str(args.speed)]
    if args.session_id:
        forwarded += ["--session-id", args.session_id]
    sys.argv = [sys.argv[0], *forwarded]
    if schema == "study3_replay_v1":
        from study3_replay import main as replay_main
    else:
        from workholding_replay import main as replay_main
    replay_main()


if __name__ == "__main__":
    main()
