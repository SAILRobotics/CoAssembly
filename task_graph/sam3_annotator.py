#!/usr/bin/env python3
"""Dear PyGui bounding-box annotator with SAM 3 concept suggestions."""

from __future__ import annotations

import argparse
import queue
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CLASSES = [
    "gear",
    "gear_rod",
    "gear_stand",
    "baseboard",
    "left_hand",
    "right_hand",
    "tool",
    "bearing",
]
# Add or remove phrases here. Every alias maps back to its canonical class.
CLASS_ALIASES = {
    "gear": ["gear"],
    "gear_rod": ["gear rod", "gear shaft", "geared axle"],
    "gear_stand": ["gear stand", "bearing stand", "gear support bracket"],
    "baseboard": ["baseboard", "gearbox base plate", "mounting board"],
    "left_hand": ["left hand", "human left hand"],
    "right_hand": ["right hand", "human right hand"],
    "bearing": ["bearing"],
    "tool": ["tool", "hand tool"],
}
SAM3_PROMPTS = [
    prompt for class_name in CLASSES for prompt in CLASS_ALIASES[class_name]
]
TEXT_PROMPT_TO_CLASS = {
    prompt_id: CLASSES.index(class_name)
    for prompt_id, class_name in enumerate(
        class_name
        for class_name in CLASSES
        for _prompt in CLASS_ALIASES[class_name]
    )
}
COLORS = [
    (255, 99, 71, 255),
    (70, 180, 255, 255),
    (255, 200, 70, 255),
    (110, 220, 120, 255),
    (210, 120, 255, 255),
    (255, 130, 190, 255),
    (80, 235, 210, 255),
    (190, 190, 190, 255),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            ((self.x1 + self.x2) / 2) / width,
            ((self.y1 + self.y2) / 2) / height,
            (self.x2 - self.x1) / width,
            (self.y2 - self.y1) / height,
        )


def discover_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stat().st_size
    )


def select_frames(images: list[Path], frame_step: int) -> list[Path]:
    """Select every Nth valid frame, starting with the first frame."""
    if frame_step <= 1:
        return images
    return images[::frame_step]


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        parts = raw_line.split()
        if len(parts) != 5:
            print(f"Warning: ignoring malformed {path}:{line_number}", file=sys.stderr)
            continue
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            print(f"Warning: ignoring malformed {path}:{line_number}", file=sys.stderr)
            continue
        if not 0 <= class_id < len(CLASSES):
            print(f"Warning: ignoring unknown class in {path}:{line_number}", file=sys.stderr)
            continue
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        if x2 > x1 and y2 > y1:
            boxes.append(Box(class_id, x1, y1, x2, y2))
    return boxes


def write_yolo_labels(path: Path, boxes: list[Box], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        cx, cy, bw, bh = box.normalized(width, height)
        lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


class AnnotationApp:
    def __init__(
        self,
        image_folder: Path,
        labels_folder: Path,
        model_name: str,
        confidence: float,
        device: str | None,
        frame_step: int,
    ) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.image_folder = image_folder
        self.labels_folder = labels_folder
        all_images = discover_images(image_folder)
        if not all_images:
            raise RuntimeError(f"No non-empty images found in {image_folder}")
        self.total_image_count = len(all_images)
        self.images = select_frames(all_images, frame_step)

        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self.index = 0
        self.image: Image.Image | None = None
        self.image_width = 0
        self.image_height = 0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.edit_mode: str | None = None
        self.edit_start: tuple[float, float] | None = None
        self.edit_original: Box | None = None
        self.dirty = False
        self.sam_model = None
        self.sam_processor = None
        self.predictor_image: Path | None = None
        self.vision_embeds = None
        self.vision_original_sizes = None
        self.model_busy = False
        self.results_queue: queue.Queue[tuple[Path, list[Box] | None, str | None]] = queue.Queue()
        self.pending_inference: tuple | None = None
        self.reference_path: Path | None = None
        self.reference_boxes: list[Box] = []
        self.annotated_count = 0

    def label_path(self, image_path: Path | None = None) -> Path:
        image_path = image_path or self.images[self.index]
        relative = image_path.relative_to(self.image_folder)
        return (self.labels_folder / relative).with_suffix(".txt")

    def run(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        self._build_gui()
        dpg.create_viewport(
            title="CoAssembly SAM 3 Annotator",
            width=1320,
            height=880,
            min_width=1050,
            min_height=760,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        self._load_image(0)
        self._show_model_readiness()
        self._maybe_auto_suggest()
        while dpg.is_dearpygui_running():
            self._poll_model_results()
            dpg.render_dearpygui_frame()
            # Keep CUDA model loading/inference on Dear PyGui's main thread.
            # Loading SAM 3 from a background thread can segfault in native code.
            if self.pending_inference is not None:
                job = self.pending_inference
                self.pending_inference = None
                self._prediction_worker(*job)
        self._save()
        dpg.destroy_context()

    def _build_gui(self) -> None:
        dpg = self.dpg
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                [0.08, 0.08, 0.08, 1.0] * CANVAS_WIDTH * CANVAS_HEIGHT,
                tag="image_texture",
            )

        with dpg.window(tag="primary_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=990, border=False):
                    with dpg.drawlist(
                        width=CANVAS_WIDTH,
                        height=CANVAS_HEIGHT,
                        tag="canvas",
                    ):
                        dpg.draw_image(
                            "image_texture",
                            (0, 0),
                            (CANVAS_WIDTH, CANVAS_HEIGHT),
                            tag="canvas_image",
                        )
                    dpg.add_text("", tag="image_status")
                with dpg.child_window(width=-1, border=True):
                    dpg.add_text("Classes")
                    dpg.add_radio_button(
                        CLASSES,
                        default_value=CLASSES[0],
                        tag="class_selector",
                        callback=self._change_selected_class,
                    )
                    dpg.add_separator()
                    dpg.add_button(
                        label="Suggest with SAM 3",
                        callback=self._suggest,
                        width=-1,
                        tag="suggest_button",
                    )
                    dpg.add_checkbox(
                        label="Auto-suggest on unlabeled images",
                        default_value=False,
                        tag="auto_suggest",
                    )
                    dpg.add_checkbox(label="Use text prompts", default_value=True, tag="use_text")
                    dpg.add_checkbox(
                        label="Use selected exemplar",
                        default_value=False,
                        tag="use_visual",
                    )
                    dpg.add_button(
                        label="Set selected box as exemplar",
                        callback=self._set_visual_reference,
                    )
                    dpg.add_text("Image exemplar: none", tag="reference_status", wrap=270)
                    dpg.add_slider_float(
                        label="Confidence",
                        min_value=0.05,
                        max_value=0.90,
                        default_value=self.confidence,
                        format="%.2f",
                        tag="confidence",
                    )
                    dpg.add_text("SAM 3 loads on first use.", color=(150, 160, 175))
                    dpg.add_text("", tag="model_status", wrap=270)
                    dpg.add_separator()
                    dpg.add_button(label="Delete selected  [Del]", callback=self._delete_selected)
                    dpg.add_button(label="Clear all boxes", callback=self._clear_boxes)
                    dpg.add_separator()
                    dpg.add_button(label="< Previous  [A]", callback=lambda: self._navigate(-1))
                    dpg.add_button(label="Save  [Ctrl+S]", callback=self._save)
                    dpg.add_button(label="Next  [D]", callback=lambda: self._navigate(1))
                    dpg.add_input_int(
                        label="Image index",
                        default_value=1,
                        min_value=1,
                        max_value=len(self.images),
                        min_clamped=True,
                        max_clamped=True,
                        on_enter=True,
                        callback=self._go_to_index,
                        tag="image_index",
                        width=130,
                    )
                    dpg.add_button(label="Go to image", callback=self._go_to_index)
                    dpg.add_separator()
                    dpg.add_text("Draw: drag on image")
                    dpg.add_text("Select: click inside box")
                    dpg.add_text("Resize: drag selected box handles")
                    dpg.add_text("Move: drag inside selected box")
                    dpg.add_text("Class keys: 1–8")
                    dpg.add_text("", tag="dataset_status", wrap=270)

        with dpg.item_handler_registry(tag="canvas_handlers"):
            dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=self._mouse_down)
        dpg.bind_item_handler_registry("canvas", "canvas_handlers")

        with dpg.handler_registry():
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._mouse_up)
            dpg.add_mouse_move_handler(callback=self._mouse_move)
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self._delete_selected)
            dpg.add_key_press_handler(key=dpg.mvKey_A, callback=lambda: self._navigate(-1))
            dpg.add_key_press_handler(key=dpg.mvKey_D, callback=lambda: self._navigate(1))
            for number, key in enumerate(
                [
                    dpg.mvKey_1,
                    dpg.mvKey_2,
                    dpg.mvKey_3,
                    dpg.mvKey_4,
                    dpg.mvKey_5,
                    dpg.mvKey_6,
                    dpg.mvKey_7,
                    dpg.mvKey_8,
                ]
            ):
                dpg.add_key_press_handler(
                    key=key, callback=lambda _s, _a, i=number: self._select_class(i)
                )

    def _load_image(self, index: int) -> None:
        self.index = index % len(self.images)
        path = self.images[self.index]
        with Image.open(path) as source:
            image = source.convert("RGBA")
        self.image = image
        self.image_width, self.image_height = image.size
        self.scale = min(CANVAS_WIDTH / self.image_width, CANVAS_HEIGHT / self.image_height)
        shown_width = max(1, round(self.image_width * self.scale))
        shown_height = max(1, round(self.image_height * self.scale))
        self.offset_x = (CANVAS_WIDTH - shown_width) / 2
        self.offset_y = (CANVAS_HEIGHT - shown_height) / 2

        fitted = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (20, 20, 20, 255))
        resized = image.resize((shown_width, shown_height), Image.Resampling.BILINEAR)
        fitted.alpha_composite(resized, (round(self.offset_x), round(self.offset_y)))
        pixel_data = (
            fitted.get_flattened_data()
            if hasattr(fitted, "get_flattened_data")
            else fitted.getdata()
        )
        pixels = [channel / 255.0 for pixel in pixel_data for channel in pixel]
        self.dpg.set_value("image_texture", pixels)

        self.boxes = read_yolo_labels(
            self.label_path(), self.image_width, self.image_height
        )
        self.selected = None
        self.dirty = False
        self._refresh()

    def _canvas_to_image(
        self, mouse: tuple[float, float], clamp: bool = False
    ) -> tuple[float, float] | None:
        left, top = self.dpg.get_item_rect_min("canvas")
        x = (mouse[0] - left - self.offset_x) / self.scale
        y = (mouse[1] - top - self.offset_y) / self.scale
        if clamp:
            return (
                min(max(x, 0.0), float(self.image_width)),
                min(max(y, 0.0), float(self.image_height)),
            )
        if 0 <= x <= self.image_width and 0 <= y <= self.image_height:
            return x, y
        return None

    @staticmethod
    def _handle_points(box: Box) -> dict[str, tuple[float, float]]:
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        return {
            "nw": (box.x1, box.y1),
            "n": (center_x, box.y1),
            "ne": (box.x2, box.y1),
            "e": (box.x2, center_y),
            "se": (box.x2, box.y2),
            "s": (center_x, box.y2),
            "sw": (box.x1, box.y2),
            "w": (box.x1, center_y),
        }

    def _mouse_down(self, _sender=None, _app_data=None) -> None:
        point = self._canvas_to_image(tuple(self.dpg.get_mouse_pos(local=False)))
        if point is None:
            return
        x, y = point
        if self.selected is not None and self.selected < len(self.boxes):
            selected_box = self.boxes[self.selected]
            tolerance = 7.0 / self.scale
            for handle, (handle_x, handle_y) in self._handle_points(selected_box).items():
                if abs(x - handle_x) <= tolerance and abs(y - handle_y) <= tolerance:
                    self.edit_mode = handle
                    self.edit_start = point
                    self.edit_original = Box(**vars(selected_box))
                    self.drag_start = None
                    return
        for index in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[index]
            if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
                if index == self.selected:
                    self.edit_mode = "move"
                    self.edit_start = point
                    self.edit_original = Box(**vars(box))
                    self.drag_start = None
                    return
                self.selected = index
                self.drag_start = None
                self.dpg.set_value("class_selector", CLASSES[box.class_id])
                self._refresh()
                return
        self.selected = None
        self.drag_start = point
        self._refresh()

    def _mouse_move(self, _sender=None, _app_data=None) -> None:
        if (
            self.edit_mode is None
            or self.edit_start is None
            or self.edit_original is None
            or self.selected is None
            or self.selected >= len(self.boxes)
        ):
            return
        point = self._canvas_to_image(
            tuple(self.dpg.get_mouse_pos(local=False)), clamp=True
        )
        if point is None:
            return
        original = self.edit_original
        x, y = point
        if self.edit_mode == "move":
            dx = x - self.edit_start[0]
            dy = y - self.edit_start[1]
            dx = min(max(dx, -original.x1), self.image_width - original.x2)
            dy = min(max(dy, -original.y1), self.image_height - original.y2)
            updated = Box(
                original.class_id,
                original.x1 + dx,
                original.y1 + dy,
                original.x2 + dx,
                original.y2 + dy,
                original.confidence,
            )
        else:
            x1, y1, x2, y2 = original.x1, original.y1, original.x2, original.y2
            if "w" in self.edit_mode:
                x1 = min(x, x2 - 3)
            if "e" in self.edit_mode:
                x2 = max(x, x1 + 3)
            if "n" in self.edit_mode:
                y1 = min(y, y2 - 3)
            if "s" in self.edit_mode:
                y2 = max(y, y1 + 3)
            updated = Box(
                original.class_id, x1, y1, x2, y2, original.confidence
            )
        self.boxes[self.selected] = updated
        self._refresh()

    def _mouse_up(self, _sender=None, _app_data=None) -> None:
        if self.edit_mode is not None:
            self._mouse_move()
            self.edit_mode = None
            self.edit_start = None
            self.edit_original = None
            self.dirty = True
            self._refresh()
            return
        if self.drag_start is None:
            return
        point = self._canvas_to_image(tuple(self.dpg.get_mouse_pos(local=False)))
        start = self.drag_start
        self.drag_start = None
        if point is None:
            return
        x1, x2 = sorted((start[0], point[0]))
        y1, y2 = sorted((start[1], point[1]))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return
        class_id = CLASSES.index(self.dpg.get_value("class_selector"))
        self.boxes.append(Box(class_id, x1, y1, x2, y2))
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self._refresh()

    def _refresh(self) -> None:
        dpg = self.dpg
        dpg.delete_item("canvas", children_only=True)
        dpg.draw_image(
            "image_texture",
            (0, 0),
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            parent="canvas",
            tag="canvas_image",
        )
        for index, box in enumerate(self.boxes):
            x1 = self.offset_x + box.x1 * self.scale
            y1 = self.offset_y + box.y1 * self.scale
            x2 = self.offset_x + box.x2 * self.scale
            y2 = self.offset_y + box.y2 * self.scale
            color = COLORS[box.class_id]
            thickness = 4 if index == self.selected else 2
            dpg.draw_rectangle((x1, y1), (x2, y2), color=color, thickness=thickness, parent="canvas")
            confidence = f" {box.confidence:.2f}" if box.confidence is not None else ""
            dpg.draw_text(
                (x1 + 3, max(2, y1 - 18)),
                CLASSES[box.class_id] + confidence,
                color=color,
                size=16,
                parent="canvas",
            )
            if index == self.selected:
                for handle_x, handle_y in self._handle_points(box).values():
                    screen_x = self.offset_x + handle_x * self.scale
                    screen_y = self.offset_y + handle_y * self.scale
                    dpg.draw_rectangle(
                        (screen_x - 5, screen_y - 5),
                        (screen_x + 5, screen_y + 5),
                        color=(255, 255, 255, 255),
                        fill=color,
                        thickness=1,
                        parent="canvas",
                    )
        path = self.images[self.index]
        marker = "*" if self.dirty else ""
        dpg.set_value(
            "image_status",
            f"{self.index + 1}/{len(self.images)}  {path.name}{marker}  "
            f"({self.image_width}x{self.image_height})  {len(self.boxes)} box(es)",
        )
        dpg.set_value("image_index", self.index + 1)
        self.annotated_count = sum(1 for image in self.images if self.label_path(image).exists())
        dpg.set_value(
            "dataset_status",
            f"Labels: {self.labels_folder}\n"
            f"Selected frames: {len(self.images)}/{self.total_image_count}\n"
            f"Annotated selected frames: {self.annotated_count}/{len(self.images)}",
        )

    def _save(self, *_args) -> None:
        if not self.image:
            return
        write_yolo_labels(
            self.label_path(), self.boxes, self.image_width, self.image_height
        )
        self.dirty = False
        self._refresh()
        self.dpg.set_value("model_status", f"Saved {self.label_path().name}")

    def _navigate(self, step: int) -> None:
        if self.model_busy:
            self.dpg.set_value("model_status", "Wait for SAM 3 inference to finish.")
            return
        if self.dirty:
            self._save()
        self._load_image(self.index + step)
        self._maybe_auto_suggest()

    def _go_to_index(self, *_args) -> None:
        target = int(self.dpg.get_value("image_index")) - 1
        if target == self.index:
            return
        if self.model_busy:
            self.dpg.set_value("model_status", "Wait for SAM 3 inference to finish.")
            self.dpg.set_value("image_index", self.index + 1)
            return
        if self.dirty:
            self._save()
        self._load_image(target)
        self._maybe_auto_suggest()

    def _maybe_auto_suggest(self) -> None:
        if not self.dpg.get_value("auto_suggest"):
            return
        if self.label_path().exists():
            self.dpg.set_value(
                "model_status", "Loaded saved annotations; auto-suggest skipped."
            )
            return
        self._suggest()

    def _delete_selected(self, *_args) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            return
        self.boxes.pop(self.selected)
        self.selected = None
        self.dirty = True
        self._refresh()

    def _clear_boxes(self, *_args) -> None:
        if self.boxes:
            self.boxes.clear()
            self.selected = None
            self.dirty = True
            self._refresh()

    def _select_class(self, class_id: int) -> None:
        self.dpg.set_value("class_selector", CLASSES[class_id])
        self._change_selected_class()

    def _change_selected_class(self, *_args) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            return
        self.boxes[self.selected].class_id = CLASSES.index(
            self.dpg.get_value("class_selector")
        )
        self.dirty = True
        self._refresh()

    def _suggest(self, *_args) -> None:
        if self.model_busy:
            return
        use_text = bool(self.dpg.get_value("use_text"))
        use_visual = bool(self.dpg.get_value("use_visual"))
        if not use_text and not use_visual:
            self.dpg.set_value("model_status", "Enable text and/or visual prompting.")
            return
        if use_visual and (self.reference_path is None or not self.reference_boxes):
            if not use_text:
                self.dpg.set_value(
                    "model_status", "Select and set a same-image exemplar first."
                )
                return
            use_visual = False
        self.model_busy = True
        path = self.images[self.index]
        confidence = float(self.dpg.get_value("confidence"))
        modes = " + ".join(
            name for name, enabled in (("text", use_text), ("visual", use_visual)) if enabled
        )
        self.dpg.set_value(
            "model_status", f"Running SAM 3 ({modes}) on {path.name}…"
        )
        self.pending_inference = (
            path,
            confidence,
            use_text,
            use_visual,
            self.reference_path,
            [Box(**vars(box)) for box in self.reference_boxes],
        )

    def _show_model_readiness(self) -> None:
        try:
            import transformers
            from transformers import Sam3Model, Sam3Processor  # noqa: F401
        except (ImportError, AttributeError):
            version = getattr(transformers, "__version__", "not installed") if "transformers" in locals() else "not installed"
            self.dpg.configure_item("suggest_button", enabled=False)
            self.dpg.set_value(
                "model_status",
                f"SAM 3 NOT READY\nTransformers: {version}\n"
                "Install a release containing Sam3Model and Sam3Processor, then restart.",
            )
            return

        try:
            from huggingface_hub import get_token
            authenticated = bool(get_token())
        except ImportError:
            authenticated = False
        self.dpg.configure_item("suggest_button", enabled=True)
        if authenticated:
            self.dpg.configure_item("suggest_button", enabled=True)
            self.dpg.set_value(
                "model_status",
                f"SAM 3 READY\nModel: {self.model_name}\n"
                "Hugging Face authentication found. The first run downloads weights.",
            )
        else:
            self.dpg.set_value(
                "model_status",
                f"SAM 3 AUTH REQUIRED\nModel: {self.model_name}\n"
                "Run `hf auth login`, accept access at huggingface.co/facebook/sam3, "
                "then click Suggest.",
            )

    def _set_visual_reference(self, *_args) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            self.dpg.set_value(
                "model_status", "Select one reviewed box before setting an exemplar."
            )
            return
        self.reference_path = self.images[self.index]
        self.reference_boxes = [Box(**vars(self.boxes[self.selected]))]
        exemplar = self.reference_boxes[0]
        self.dpg.set_value(
            "reference_status",
            f"{self.reference_path.name}: {CLASSES[exemplar.class_id]}",
        )
        self.dpg.set_value("use_visual", True)
        self.dpg.set_value(
            "model_status",
            "Same-image exemplar set. Click Suggest with SAM 3 on this image.",
        )

    @staticmethod
    def _iou(left: Box, right: Box) -> float:
        intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
        intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = intersection_width * intersection_height
        if not intersection:
            return 0.0
        left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
        right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
        return intersection / (left_area + right_area - intersection)

    @classmethod
    def _merge_predictions(cls, boxes: list[Box], iou_threshold: float = 0.55) -> list[Box]:
        kept: list[Box] = []
        for candidate in sorted(
            boxes,
            key=lambda box: box.confidence if box.confidence is not None else 0.0,
            reverse=True,
        ):
            if any(
                candidate.class_id == existing.class_id
                and cls._iou(candidate, existing) >= iou_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _transformers_result_boxes(result: dict, class_id: int) -> list[Box]:
        predictions: list[Box] = []
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        for coordinates, score in zip(boxes, scores):
            if len(coordinates) != 4:
                continue
            x1, y1, x2, y2 = map(float, coordinates)
            if x2 > x1 and y2 > y1:
                predictions.append(
                    Box(class_id, x1, y1, x2, y2, confidence=float(score))
                )
        return predictions

    def _prediction_worker(
        self,
        path: Path,
        confidence: float,
        use_text: bool,
        use_visual: bool,
        reference_path: Path | None,
        reference_boxes: list[Box],
    ) -> None:
        try:
            predictions = self._run_sam3(
                path,
                confidence,
                use_text,
                use_visual,
                reference_path,
                reference_boxes,
            )
            self.results_queue.put((path, self._merge_predictions(predictions), None))
        except Exception as error:
            self.results_queue.put((path, None, str(error)))

    def _run_sam3(
        self,
        path: Path,
        confidence: float,
        use_text: bool,
        use_visual: bool,
        reference_path: Path | None,
        reference_boxes: list[Box],
    ) -> list[Box]:
        try:
            if self.sam_model is None or self.sam_processor is None:
                try:
                    from transformers import Sam3Model, Sam3Processor
                except ImportError:
                    raise RuntimeError(
                        "SAM 3 requires a current Transformers release. "
                        "Run: pip install -U transformers accelerate"
                    ) from None
                load_kwargs = {"dtype": "auto"}
                if self.device:
                    self.sam_model = Sam3Model.from_pretrained(
                        self.model_name, **load_kwargs
                    ).to(self.device)
                else:
                    self.sam_model = Sam3Model.from_pretrained(
                        self.model_name, device_map="auto", **load_kwargs
                    )
                self.sam_model.eval()
                self.sam_processor = Sam3Processor.from_pretrained(self.model_name)
            if self.predictor_image != path:
                import torch

                with Image.open(path) as source:
                    image = source.convert("RGB")
                image_inputs = self.sam_processor(
                    images=image, return_tensors="pt"
                ).to(self.sam_model.device)
                with torch.no_grad():
                    self.vision_embeds = self.sam_model.get_vision_features(
                        pixel_values=image_inputs.pixel_values
                    )
                self.vision_original_sizes = image_inputs.get("original_sizes")
                self.predictor_image = path
            predictions: list[Box] = []
            if use_text:
                import torch

                for prompt_id, prompt in enumerate(SAM3_PROMPTS):
                    text_inputs = self.sam_processor(
                        text=prompt, return_tensors="pt"
                    ).to(self.sam_model.device)
                    with torch.no_grad():
                        outputs = self.sam_model(
                            vision_embeds=self.vision_embeds,
                            **text_inputs,
                        )
                    result = self.sam_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=confidence,
                        mask_threshold=0.5,
                        target_sizes=self.vision_original_sizes.tolist(),
                    )[0]
                    predictions.extend(
                        self._transformers_result_boxes(
                            result, TEXT_PROMPT_TO_CLASS[prompt_id]
                        )
                    )
            if use_visual and reference_path == path and reference_boxes:
                import torch

                exemplar = reference_boxes[0]
                with Image.open(path) as source:
                    image = source.convert("RGB")
                exemplar_inputs = self.sam_processor(
                    images=image,
                    input_boxes=[
                        [[exemplar.x1, exemplar.y1, exemplar.x2, exemplar.y2]]
                    ],
                    input_boxes_labels=[[1]],
                    return_tensors="pt",
                ).to(self.sam_model.device)
                with torch.no_grad():
                    outputs = self.sam_model(**exemplar_inputs)
                result = self.sam_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=confidence,
                    mask_threshold=0.5,
                    target_sizes=exemplar_inputs.get("original_sizes").tolist(),
                )
                predictions.extend(
                    self._transformers_result_boxes(result[0], exemplar.class_id)
                )
            return predictions
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"SAM 3: {error}") from error

    def _poll_model_results(self) -> None:
        try:
            path, predictions, error = self.results_queue.get_nowait()
        except queue.Empty:
            return
        self.model_busy = False
        if error:
            self.dpg.set_value("model_status", f"SAM 3 error: {error}")
            return
        if path != self.images[self.index]:
            self.dpg.set_value("model_status", "Prediction finished for a different image.")
            return
        self.boxes = predictions or []
        self.selected = None
        self.dirty = True
        self._refresh()
        self.dpg.set_value(
            "model_status",
            f"Suggested {len(self.boxes)} box(es). Review, then save.",
        )


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "label.txt"
        expected = [Box(2, 10, 20, 110, 220)]
        write_yolo_labels(path, expected, 200, 400)
        actual = read_yolo_labels(path, 200, 400)
        assert len(actual) == 1
        assert actual[0].class_id == 2
        assert all(
            abs(left - right) < 1e-4
            for left, right in zip(
                (actual[0].x1, actual[0].y1, actual[0].x2, actual[0].y2),
                (10, 20, 110, 220),
            )
        )
    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("dante_captures"),
        help="image folder (default: task_graph/dante_captures)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="YOLO label folder (default: <images>/labels)",
    )
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="load every Nth valid image; 1 uses all images (default: 3)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", help="inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    images = args.images.expanduser().resolve()
    if not images.is_dir():
        print(f"Error: image folder does not exist: {images}", file=sys.stderr)
        return 2
    if not 0 < args.confidence <= 1:
        print("Error: --confidence must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.frame_step < 1:
        print("Error: --frame-step must be >= 1.", file=sys.stderr)
        return 2
    labels = (args.labels or images.with_name(images.name + "_labels")).expanduser().resolve()
    try:
        AnnotationApp(
            images,
            labels,
            args.model,
            args.confidence,
            args.device,
            args.frame_step,
        ).run()
    except (ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
