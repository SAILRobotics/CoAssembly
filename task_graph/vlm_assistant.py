"""vlm_assistant.py — Qwen2.5-VL assistant panel for DearPyGui.

Features:
  - Task description as system context
  - Live task-graph state injection (updated on every complete/undo/reset)
  - Conversation history (up to MAX_HISTORY_PAIRS Q&A pairs); cleared on state change
  - Drag-and-drop image loading (+ file browser fallback)
  - Formatted answer output with Q/A separation

The VLM runs on a background thread so the UI never freezes.
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

_THUMB_W, _THUMB_H = 96, 72
_ANSWER_H          = 520
_IMAGE_LIST_H      = 180
_WRAP_WIDTH        = 900

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert assembly assistant for a gearbox assembly task.
Use the task description below as your primary reference when answering questions.
When images are provided, identify components and their assembly state from the visuals.

--- TASK DESCRIPTION ---
{task_description}
--- END TASK DESCRIPTION ---
"""

_MD_PATTERNS = [
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
    (re.compile(r'\*(.+?)\*'),     r'\1'),
    (re.compile(r'`(.+?)`'),       r'\1'),
    (re.compile(r'^#{1,6}\s+', re.MULTILINE), ''),
]

def _strip_md(text: str) -> str:
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ── Background VLM worker ─────────────────────────────────────────────────────

class _VLMWorker:
    MAX_HISTORY_PAIRS = 10  # keep last N (user, assistant) pairs in context

    def __init__(self, model_name: str, system_prompt: str) -> None:
        self._model_name    = model_name
        self._system_prompt = system_prompt
        self._job_queue: "queue.Queue[tuple | None]" = queue.Queue(maxsize=1)
        self._result_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

        self._history: list[dict] = []
        self._history_lock = threading.Lock()
        self._task_state: str = ""
        self._focused_step: str = ""   # injected before each inference

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, question: str, image_paths: list[str]) -> None:
        try:
            self._job_queue.put_nowait((question, list(image_paths)))
        except queue.Full:
            self._result_queue.put(("error", "Previous query still running — please wait."))

    def update_task_state(self, state: str) -> None:
        """Called on every graph state change. Replaces injected state and clears history."""
        with self._history_lock:
            self._task_state = state
            self._history.clear()
        self._result_queue.put(("state_updated", state))

    def set_focused_step(self, focused: str) -> None:
        """Update which step the user has selected. Does NOT clear history."""
        with self._history_lock:
            self._focused_step = focused

    def reset_history(self) -> None:
        with self._history_lock:
            self._history.clear()
        self._result_queue.put(("history_reset", ""))

    def poll(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while True:
            try:
                events.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def close(self) -> None:
        self._running = False
        try:
            self._job_queue.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self) -> None:
        self._result_queue.put(("status", f"Loading {self._model_name} ..."))
        try:
            model, processor = self._load_model()
        except Exception as e:
            self._result_queue.put(("error", f"Model load failed: {e}"))
            return
        self._result_queue.put(("status", "ready"))

        while self._running:
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            question, image_paths = job
            self._result_queue.put(("status", "thinking"))
            try:
                answer = self._infer(model, processor, question, image_paths)
                self._result_queue.put(("answer", answer))
            except Exception as e:
                self._result_queue.put(("error", f"Inference error: {e}"))
            self._result_queue.put(("status", "ready"))

    def _load_model(self):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_name, torch_dtype="auto", device_map="auto")
        except (ImportError, AttributeError, OSError):
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_name, torch_dtype="auto", device_map="auto")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self._model_name)
        model.eval()
        return model, processor

    def _infer(self, model, processor, question: str, image_paths: list[str]) -> str:
        from PIL import Image

        with self._history_lock:
            task_state   = self._task_state
            focused_step = self._focused_step
            history_snap = list(self._history[-(self.MAX_HISTORY_PAIRS * 2):])

        # Prepend state + focused step directly into the question text.
        # Small models attend to the user message far more reliably than
        # a second system message, so we inject context here instead.
        prefix_parts: list[str] = []
        if task_state:
            prefix_parts.append(
                "=== CURRENT ASSEMBLY STATE ===\n" + task_state + "\n=== END STATE ===")
        if focused_step:
            prefix_parts.append(
                "=== CURRENTLY SELECTED STEP ===\n" + focused_step +
                "\n=== END SELECTED STEP ===")
        question_with_state = ("\n\n".join(prefix_parts) + "\n\n" + question
                               if prefix_parts else question)

        content = []
        pil_images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            pil_images.append(img)
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question_with_state})

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history_snap)
        messages.append({"role": "user", "content": content})

        try:
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(model.device)
        except (ImportError, Exception):
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=pil_images if pil_images else None,
                padding=True, return_tensors="pt",
            ).to(model.device)

        import torch
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        # Store text-only version of this turn in history (images are too large to repeat)
        with self._history_lock:
            self._history.append({
                "role": "user",
                "content": [{"type": "text", "text": question}],
            })
            self._history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })
            # Trim to keep within MAX_HISTORY_PAIRS
            max_msgs = self.MAX_HISTORY_PAIRS * 2
            if len(self._history) > max_msgs:
                self._history = self._history[-max_msgs:]

        return answer


# ── DPG panel ─────────────────────────────────────────────────────────────────

class VLMAssistant:
    _STATUS_COLOR = {
        "loading":  (180, 180, 180, 255),
        "ready":    (50,  220, 80,  255),
        "thinking": (255, 165, 50,  255),
        "error":    (255, 80,  80,  255),
    }
    _STATUS_LABEL = {
        "loading":  "Loading model ...",
        "ready":    "Ready",
        "thinking": "Thinking ...",
        "error":    "Error",
    }

    def __init__(self, dpg, task_description_path: str,
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.dpg = dpg
        self._model_name = model_name

        desc_path = Path(task_description_path)
        task_description = desc_path.read_text() if desc_path.exists() else "(not found)"
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description)
        self._desc_info = f"{desc_path.name}  ({len(task_description)} chars)"

        self._worker = _VLMWorker(model_name, self._system_prompt)
        self._image_paths:   list[str] = []
        self._texture_tags:  list[str] = []
        self._tex_registry_tag = "vlm_tex_registry"
        self._next_tex_id = 0
        self._exchanges: list[tuple[str, str]] = []
        self._pending_question: str = ""
        self._current_status = "loading"
        self._history_count = 0   # tracks number of Q/A pairs stored in worker

    # ── Public: called by the task graph on every state change ────────────────

    def notify_graph_event(self, event_label: str, state_summary: str) -> None:
        """Call this whenever the task graph changes (complete/undo/reset/live)."""
        self._worker.update_task_state(f"Event: {event_label}\n\n{state_summary}")

    def set_focused_step(self, focused_text: str) -> None:
        """Tell the VLM which step the user has selected. Does NOT clear history."""
        self._worker.set_focused_step(focused_text)

    # ── Public: voice input ───────────────────────────────────────────────────

    def submit_question(self, text: str) -> bool:
        """Route an external text (e.g. voice transcript) as a VLM question.
        Returns False if the model is busy; caller can log a warning."""
        if self._current_status != "ready":
            return False
        self._pending_question = text
        self._render_history(pending=True)
        self._worker.submit(text, self._image_paths)
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def build_inline(self) -> None:
        """Add VLM content into the current DPG context (no window created)."""
        dpg = self.dpg
        with dpg.texture_registry(tag=self._tex_registry_tag):
            pass

        dpg.add_text("VLM Assistant", color=(225, 232, 240))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=(160, 170, 185))
            dpg.add_text("Loading model ...", tag="vlm_status",
                         color=list(self._STATUS_COLOR["loading"]))
        with dpg.group(horizontal=True):
            dpg.add_text("History:", color=(160, 170, 185))
            dpg.add_text("0 / 10 pairs", tag="vlm_history_count",
                         color=(150, 165, 185))
        dpg.add_text(f"Model: {self._model_name}", color=(120, 135, 160))
        dpg.add_text(f"Context: {self._desc_info}", color=(100, 180, 120))
        dpg.add_separator()

        dpg.add_text("Images", color=(200, 210, 225))
        dpg.add_text("Drop image files anywhere in the window, or use Browse.",
                     color=(130, 145, 165), wrap=_WRAP_WIDTH)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Browse ...", callback=self._open_file_dialog)
            dpg.add_button(label="Clear images", callback=self._clear_images)
        dpg.add_spacer(height=4)
        with dpg.child_window(tag="vlm_image_list", height=_IMAGE_LIST_H,
                              horizontal_scrollbar=False):
            dpg.add_text("No images loaded.", tag="vlm_no_images_hint",
                         color=(120, 130, 145))
        dpg.add_separator()

        dpg.add_text("Question", color=(200, 210, 225))
        dpg.add_input_text(tag="vlm_question",
                           hint="Type your question here ...",
                           width=-1, multiline=False, on_enter=False)
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Ask", tag="vlm_ask_button",
                           callback=self._ask, width=-1, enabled=False)
            dpg.add_button(label="Clear history", callback=self._clear_history, width=110)
        dpg.add_separator()

        dpg.add_text("Conversation", color=(200, 210, 225))
        dpg.add_input_text(tag="vlm_answer", multiline=True, readonly=True,
                           width=-1, height=_ANSWER_H,
                           default_value="Answers will appear here once the model is ready.")

        self._worker.start()

    def close(self) -> None:
        self._worker.close()

    # ── Render-loop tick ──────────────────────────────────────────────────────

    def tick(self) -> None:
        for kind, payload in self._worker.poll():
            if kind == "status":
                self._set_status(payload)
                if payload == "ready":
                    self.dpg.configure_item("vlm_ask_button", enabled=True)
                elif payload == "thinking":
                    self.dpg.configure_item("vlm_ask_button", enabled=False)

            elif kind == "answer":
                answer = _strip_md(payload)
                self._exchanges.append((self._pending_question, answer))
                self._history_count = min(
                    self._history_count + 1, _VLMWorker.MAX_HISTORY_PAIRS)
                self._pending_question = ""
                self._render_history()
                self._update_history_label()
                self.dpg.configure_item("vlm_ask_button", enabled=True)

            elif kind == "error":
                error_text = _strip_md(payload)
                self._exchanges.append(
                    (self._pending_question or "?", f"[ERROR] {error_text}"))
                self._pending_question = ""
                self._render_history()
                self._set_status("error")
                self.dpg.configure_item("vlm_ask_button", enabled=True)

            elif kind == "state_updated":
                # Graph state changed — clear conversation and show a banner
                self._exchanges.clear()
                self._pending_question = ""
                self._history_count = 0
                self._update_history_label()
                banner = "[Graph state changed — conversation reset]\n\n" + payload
                self.dpg.set_value("vlm_answer", banner)

            elif kind == "history_reset":
                self._exchanges.clear()
                self._pending_question = ""
                self._history_count = 0
                self._update_history_label()
                self.dpg.set_value("vlm_answer",
                                   "Conversation cleared. Ask a new question.")

    # ── File drop ─────────────────────────────────────────────────────────────

    def on_file_drop(self, _sender, app_data) -> None:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
        paths = app_data if isinstance(app_data, (list, tuple)) else [app_data]
        for p in paths:
            if Path(p).suffix.lower() in IMAGE_EXTS:
                self._add_image(p)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_status(self, status: str) -> None:
        self._current_status = status
        label = self._STATUS_LABEL.get(status, status)
        color = list(self._STATUS_COLOR.get(status, (200, 200, 200, 255)))
        self.dpg.set_value("vlm_status", label)
        self.dpg.configure_item("vlm_status", color=color)
        if status == "thinking" and self._pending_question:
            self._render_history(pending=True)

    def _update_history_label(self) -> None:
        self.dpg.set_value(
            "vlm_history_count",
            f"{self._history_count} / {_VLMWorker.MAX_HISTORY_PAIRS} pairs")

    def _render_history(self, pending: bool = False) -> None:
        lines: list[str] = []
        sep = "-" * 60
        for i, (q, a) in enumerate(self._exchanges):
            lines.append(f"Q: {q}")
            lines.append("")
            lines.append(f"A: {a}")
            if i < len(self._exchanges) - 1 or pending:
                lines.append(sep)
        if pending and self._pending_question:
            lines.append(f"Q: {self._pending_question}")
            lines.append("")
            lines.append("A: [Thinking ...]")
        self.dpg.set_value("vlm_answer", "\n".join(lines))

    def _ask(self) -> None:
        question = (self.dpg.get_value("vlm_question") or "").strip()
        if not question:
            return
        self._pending_question = question
        self.dpg.set_value("vlm_question", "")
        self._render_history(pending=True)
        self._worker.submit(question, self._image_paths)

    def _clear_history(self) -> None:
        self._worker.reset_history()
        # UI update happens via the "history_reset" event in tick()

    def _add_image(self, path: str) -> None:
        if path in self._image_paths:
            return
        dpg = self.dpg
        if dpg.does_item_exist("vlm_no_images_hint"):
            dpg.configure_item("vlm_no_images_hint", show=False)

        tex_tag = f"vlm_tex_{self._next_tex_id}"
        self._next_tex_id += 1
        try:
            from PIL import Image
            img = Image.open(path).convert("RGBA")
            img.thumbnail((_THUMB_W, _THUMB_H), Image.LANCZOS)
            bg = Image.new("RGBA", (_THUMB_W, _THUMB_H), (30, 32, 36, 255))
            bg.paste(img, ((_THUMB_W - img.width) // 2,
                            (_THUMB_H - img.height) // 2))
            flat = [v / 255.0 for pixel in bg.getdata() for v in pixel]
            dpg.add_static_texture(_THUMB_W, _THUMB_H, flat,
                                   tag=tex_tag, parent=self._tex_registry_tag)
        except Exception:
            tex_tag = None

        row_tag = f"vlm_img_row_{self._next_tex_id}"
        name    = Path(path).name
        with dpg.group(horizontal=True, parent="vlm_image_list", tag=row_tag):
            if tex_tag:
                dpg.add_image(tex_tag, width=_THUMB_W, height=_THUMB_H)
            dpg.add_text(f"  {name}", color=(190, 205, 225))
            dpg.add_button(label="X", width=28,
                           callback=self._remove_image, user_data=path)

        self._image_paths.append(path)
        self._texture_tags.append(tex_tag or "")

    def _remove_image(self, _sender, _app_data, path: str) -> None:
        if path not in self._image_paths:
            return
        idx = self._image_paths.index(path)
        self._image_paths.pop(idx)
        tex = self._texture_tags.pop(idx)
        self._rebuild_image_list()
        if tex and self.dpg.does_item_exist(tex):
            self.dpg.delete_item(tex)

    def _clear_images(self) -> None:
        for tex in self._texture_tags:
            if tex and self.dpg.does_item_exist(tex):
                self.dpg.delete_item(tex)
        self._image_paths.clear()
        self._texture_tags.clear()
        self._rebuild_image_list()

    def _rebuild_image_list(self) -> None:
        dpg = self.dpg
        dpg.delete_item("vlm_image_list", children_only=True)
        if not self._image_paths:
            dpg.add_text("No images loaded.", tag="vlm_no_images_hint",
                         color=(120, 130, 145), parent="vlm_image_list")
            return
        existing_paths    = list(self._image_paths)
        existing_textures = list(self._texture_tags)
        self._image_paths.clear()
        self._texture_tags.clear()
        for path, tex in zip(existing_paths, existing_textures):
            name    = Path(path).name
            row_tag = f"vlm_img_row_{self._next_tex_id}"
            self._next_tex_id += 1
            with dpg.group(horizontal=True, parent="vlm_image_list", tag=row_tag):
                if tex and dpg.does_item_exist(tex):
                    dpg.add_image(tex, width=_THUMB_W, height=_THUMB_H)
                dpg.add_text(f"  {name}", color=(190, 205, 225))
                dpg.add_button(label="X", width=28,
                               callback=self._remove_image, user_data=path)
            self._image_paths.append(path)
            self._texture_tags.append(tex)

    def _open_file_dialog(self) -> None:
        dpg = self.dpg

        def _selected(_sender, app_data):
            for path in (app_data.get("selections") or {}).values():
                self._add_image(path)

        with dpg.file_dialog(
            label="Select images", width=700, height=450,
            callback=_selected, modal=True,
            file_count=0,
        ):
            dpg.add_file_extension(".jpg",  color=(180, 220, 180, 255))
            dpg.add_file_extension(".jpeg", color=(180, 220, 180, 255))
            dpg.add_file_extension(".png",  color=(180, 220, 255, 255))
            dpg.add_file_extension(".bmp",  color=(220, 200, 180, 255))
            dpg.add_file_extension(".webp", color=(220, 200, 255, 255))
