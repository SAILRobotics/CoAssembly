"""Model adapters for Study 1 referring-expression grounding.

The experiment talks only to :class:`ReferringExpressionResolver`.  Individual
models translate their native output into a stable part/mesh prediction so the
Babylon UI and study logs do not depend on one model family.
"""

from __future__ import annotations

import io
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One visible assembly mesh and its screen-space bounding box."""

    part_file: str
    mesh_name: str
    bbox: tuple[float, float, float, float]

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> "Candidate":
        bbox = value.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each candidate requires a four-value bbox")
        coords = tuple(float(item) for item in bbox)
        if not all(math.isfinite(item) for item in coords):
            raise ValueError("Candidate bbox values must be finite")
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Candidate bbox must have positive area")
        return cls(
            part_file=str(value.get("part_file", "")),
            mesh_name=str(value.get("mesh_name", "")),
            bbox=coords,
        )


@dataclass(frozen=True)
class Resolution:
    predicted_part_file: str | None
    predicted_mesh_name: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    mapping_score: float | None
    latency_ms: float
    resolver: str
    model: str
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferringExpressionResolver(ABC):
    """Stable boundary between the study and a grounding model."""

    name = "base"
    model_name = ""

    @abstractmethod
    def resolve(
        self,
        *,
        image_bytes: bytes,
        expression: str,
        candidates: list[Candidate],
    ) -> Resolution:
        """Resolve ``expression`` to one visible candidate without ground truth."""


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    if intersection == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def match_box_to_candidate(
    grounded_box: tuple[float, float, float, float],
    candidates: list[Candidate],
) -> tuple[Candidate | None, float]:
    """Map a model box to a Babylon mesh using overlap, then center distance."""

    if not candidates:
        return None, 0.0
    overlaps = [(_intersection_over_union(grounded_box, item.bbox), item) for item in candidates]
    best_iou, best = max(overlaps, key=lambda pair: pair[0])
    if best_iou > 0:
        return best, best_iou

    gx = (grounded_box[0] + grounded_box[2]) / 2
    gy = (grounded_box[1] + grounded_box[3]) / 2
    distances = []
    for item in candidates:
        cx = (item.bbox[0] + item.bbox[2]) / 2
        cy = (item.bbox[1] + item.bbox[3]) / 2
        distances.append((math.hypot(gx - cx, gy - cy), item))
    distance, best = min(distances, key=lambda pair: pair[0])
    diagonal = max(1.0, math.hypot(grounded_box[2] - grounded_box[0], grounded_box[3] - grounded_box[1]))
    return best, 1.0 / (1.0 + distance / diagonal)


class Florence2Resolver(ReferringExpressionResolver):
    """Phrase-ground an expression with Florence-2 and map it to a scene mesh."""

    name = "florence2"

    def __init__(self, model_name: str = "microsoft/Florence-2-base-ft") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "Florence-2 requires torch, transformers, Pillow, and accelerate"
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            try:
                # Current Transformers releases ship Florence-2 natively. Use
                # those classes so Microsoft model repositories cannot select
                # an older, incompatible remote configuration implementation.
                from transformers import Florence2ForConditionalGeneration, Florence2Processor

                model = Florence2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                ).to(device)
                processor = Florence2Processor.from_pretrained(self.model_name)
            except ImportError:
                # Compatibility path for older Transformers releases that
                # predate the native Florence-2 implementation.
                try:
                    from transformers import AutoModelForCausalLM, AutoProcessor
                except ImportError as exc:
                    raise RuntimeError(
                        "Florence-2 requires a compatible Transformers installation"
                    ) from exc
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
                processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
            model.eval()
            self._model, self._processor = model, processor
            self._device, self._dtype = device, dtype

    def resolve(
        self,
        *,
        image_bytes: bytes,
        expression: str,
        candidates: list[Candidate],
    ) -> Resolution:
        started = time.perf_counter()
        self._load()
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Florence-2 requires torch and Pillow") from exc

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task + expression
        with self._inference_lock, torch.inference_mode():
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                key: value.to(self._device, self._dtype)
                if key == "pixel_values"
                else value.to(self._device)
                for key, value in inputs.items()
            }
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task,
                image_size=image.size,
            )

        task_result = parsed.get(task, {}) if isinstance(parsed, dict) else {}
        boxes = task_result.get("bboxes", []) if isinstance(task_result, dict) else []
        labels = task_result.get("labels", []) if isinstance(task_result, dict) else []
        candidate = None
        match_score = 0.0
        grounded_box = None
        if boxes:
            grounded_box = tuple(float(value) for value in boxes[0])
            candidate, match_score = match_box_to_candidate(grounded_box, candidates)

        latency_ms = (time.perf_counter() - started) * 1000
        label = str(labels[0]) if labels else ""
        return Resolution(
            predicted_part_file=candidate.part_file if candidate else None,
            predicted_mesh_name=candidate.mesh_name if candidate else None,
            bbox=grounded_box,
            # Florence-2 does not expose a calibrated model confidence. This
            # score describes only how its box matched a Babylon mesh.
            confidence=None,
            mapping_score=match_score if candidate else None,
            latency_ms=latency_ms,
            resolver=self.name,
            model=self.model_name,
            raw_output=f"label={label!r}; generated={generated_text}",
        )


def create_resolver(name: str, model_name: str | None = None) -> ReferringExpressionResolver | None:
    normalized = name.strip().lower()
    if normalized in {"", "none", "disabled"}:
        return None
    if normalized == "florence2":
        return Florence2Resolver(model_name or "microsoft/Florence-2-base-ft")
    raise ValueError(f"Unknown resolver: {name}")
