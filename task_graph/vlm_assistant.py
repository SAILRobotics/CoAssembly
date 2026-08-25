"""vlm_assistant.py — Qwen3-VL assistant panel for DearPyGui.

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
import sys
import threading
import csv
import json
from pathlib import Path

_THUMB_W, _THUMB_H = 96, 72
_ANSWER_H          = 520
_IMAGE_LIST_H      = 180
_WRAP_WIDTH        = 900

FASTENING_TOOL_LABELS = (
    "H5_HEX_BIT",
    "T25_TORX_BIT",
    "H3_HEX_BIT",
    "BIT_SCREWDRIVER",
    "BIT_WRENCH",
    "BIT_HOLDER1",
    "BIT_HOLDER2",
    "PHILLIPS_SCREWDRIVER",
)

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert assembly assistant for a gearbox assembly task.
Use the task description below as your primary reference when answering questions.
When images are provided, identify components and their assembly state from the visuals.

Infer the user's intent from the complete utterance rather than relying on a
fixed list of request words. A physical-part reference may be phrased as a
request, a question, or only a noun phrase. Greetings, task-status questions,
how-to questions, and ordinary conversation are not physical-part references.

When a prompt requests a structured intent envelope, obey its JSON schema
exactly. For a physical-part reference, resolve the part or tool to a label from
the allowed list below. If multiple non-interchangeable objects fit, report the
reference as ambiguous and list every plausible allowed label as candidates.
Never guess one label when multiple labels fit. Most labels are canonical graph
identifiers, such as `BASE_BOARD`, `STAND_ROW2_LEFT`, and `GEAR_ROW2_RIGHT`.
`BEARING`, `PIN`, and `SCREW_ROW1`..`SCREW_ROW4` group physically co-located
or interchangeable items. `H5_HEX_BIT` and `H3_HEX_BIT` are stored in
`BIT_HOLDER1`; `T25_TORX_BIT` is stored in `BIT_HOLDER2`. For every other kind
of question (how-to, status, explanation, general conversation), provide a
concise natural-language answer grounded in the current task state.
Natural-language answers are for assembly operators: use familiar phrases such
as "the Row 1 right bearing" and "put the bearing into the right stand." Never
expose internal graph IDs such as `[r1_bearing_right]`, uppercase canonical
labels such as `BEARING_ROW1_RIGHT`, filenames, or interface numbering such as
`Row 1.1` unless the user explicitly asks for internal/debug identifiers.

--- ALLOWED PART LABELS ---
{part_labels}
--- END ALLOWED PART LABELS ---

--- TASK DESCRIPTION ---
{task_description}
--- END TASK DESCRIPTION ---
"""


def _load_part_label_names() -> list[str]:
    """Load dataset parts plus the task's fastening-tool ontology."""
    try:
        path = Path(__file__).resolve().parent / "referring_expression_responses.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            labels = {row.get("target_name", "").strip() for row in rows
                      if row.get("target_name", "").strip()
                      and row.get("Verified", "o").strip().casefold() == "o"}
            labels.update(FASTENING_TOOL_LABELS)
            return sorted(labels)
    except Exception:
        return []


def _load_part_labels() -> str:
    labels = _load_part_label_names()
    return ("\n".join(f"- {name}" for name in labels)
            if labels else "(part label list unavailable)")

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


def _humanize_general_answer(text: str) -> str:
    """Remove internal graph notation from user-facing conversational answers."""
    answer = _strip_md(str(text))
    # Graph IDs and row.stage prefixes are useful for controller debugging but
    # unfamiliar and distracting in speech output.
    answer = re.sub(r"\[(?:r[1-4]_[a-z0-9_]+|finish_gearbox)\]\s*", "", answer,
                    flags=re.IGNORECASE)
    answer = re.sub(r"\bRow\s+([1-4])\.[1-7]\s*:\s*", r"Row \1: ", answer,
                    flags=re.IGNORECASE)

    answer = re.sub(r"\bBASE_BOARD\b", "the baseboard", answer)
    answer = re.sub(r"\bCRANK_HANDLE_ROW1\b", "the Row 1 crank handle", answer)

    def _gear_rod(match: re.Match) -> str:
        return f"the Row {match.group(1)} gear rod"

    answer = re.sub(r"\bGEAR_ROD_ROW([1-4])\b", _gear_rod, answer)

    friendly_nouns = {
        "BEARING": "bearing",
        "STAND": "stand",
        "SCREW": "screw",
        "PIN": "wooden pin",
        "GEAR": "gear",
    }

    def _sided_part(match: re.Match) -> str:
        kind, row, side = match.groups()
        return f"the Row {row} {side.lower()} {friendly_nouns[kind.upper()]}"

    answer = re.sub(
        r"\b(BEARING|STAND|SCREW|PIN|GEAR)_ROW([1-4])_(LEFT|RIGHT)\b",
        _sided_part,
        answer,
    )
    tool_names = {
        "H5_HEX_BIT": "H5 hex bit",
        "H3_HEX_BIT": "H3 hex bit",
        "T25_TORX_BIT": "T25 Torx bit",
        "PHILLIPS_SCREWDRIVER": "Phillips screwdriver",
        "BIT_SCREWDRIVER": "bit screwdriver",
        "BIT_WRENCH": "bit wrench",
        "BIT_HOLDER1": "Bit Holder 1",
        "BIT_HOLDER2": "Bit Holder 2",
    }
    for canonical, friendly in tool_names.items():
        answer = re.sub(rf"\b{canonical}\b", friendly, answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
    return answer


def _print_console_block(label: str, text: str) -> None:
    """Mirror important VLM-panel content to the copyable system terminal."""
    print(f"\n=== {label} ===\n{text}\n=== END {label} ===", flush=True)


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

    def submit(
        self,
        question: str,
        image_paths: list[str],
        result_kind: str = "answer",
    ) -> bool:
        try:
            self._job_queue.put_nowait((question, list(image_paths), result_kind))
            return True
        except queue.Full:
            self._result_queue.put(("error", "Previous query still running — please wait."))
            return False

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

    def apply_policy_response(self, question: str, response: str) -> None:
        """Replace a raw label with the user-facing policy response in history.

        Part identification first produces a machine label such as
        ``GEAR_ROW1_LEFT`` or ``AMBIGUOUS``. The task graph then turns that label
        into the text shown and spoken to the user. Keeping that final response
        in model history lets a follow-up such as "the left one" refer to the
        clarification the assistant actually gave.
        """
        with self._history_lock:
            replaced = False
            if len(self._history) >= 2:
                user_msg = self._history[-2]
                assistant_msg = self._history[-1]
                user_content = user_msg.get("content", [])
                same_question = bool(
                    user_msg.get("role") == "user"
                    and user_content
                    and user_content[-1].get("text") == question)
                if same_question and assistant_msg.get("role") == "assistant":
                    assistant_msg["content"] = [{"type": "text", "text": response}]
                    replaced = True
            if not replaced:
                self._history.extend([
                    {"role": "user", "content": [{"type": "text", "text": question}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ])
                max_msgs = self.MAX_HISTORY_PAIRS * 2
                if len(self._history) > max_msgs:
                    self._history = self._history[-max_msgs:]

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
            question, image_paths, result_kind = job
            self._result_queue.put(("status", "thinking"))
            try:
                answer = self._infer(
                    model,
                    processor,
                    question,
                    image_paths,
                    record_history=(result_kind == "answer"),
                    use_history=(result_kind in {"answer", "intent"}),
                )
                self._result_queue.put((result_kind, answer))
            except Exception as e:
                self._result_queue.put(("error", f"Inference error: {e}"))
            self._result_queue.put(("status", "ready"))

    def _load_model(self):
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass
        model = ModelClass.from_pretrained(
            self._model_name, torch_dtype="auto", device_map="auto")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self._model_name)
        model.eval()
        return model, processor

    def _infer(
        self,
        model,
        processor,
        question: str,
        image_paths: list[str],
        record_history: bool = True,
        use_history: bool = True,
    ) -> str:
        from PIL import Image

        with self._history_lock:
            task_state   = self._task_state
            focused_step = self._focused_step
            history_snap = (
                list(self._history[-(self.MAX_HISTORY_PAIRS * 2):])
                if use_history else []
            )

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

        # Proactive decisions are machine-facing and should not pollute the
        # human conversation history used for later questions.
        if record_history:
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

    def __init__(
        self,
        dpg,
        task_description_path: str,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    ) -> None:
        self.dpg = dpg
        self._model_name = model_name

        desc_path = Path(task_description_path)
        task_description = desc_path.read_text() if desc_path.exists() else "(not found)"
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description, part_labels=_load_part_labels())
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
        self._part_labels = _load_part_label_names()
        self._part_events: "queue.Queue[dict[str, object]]" = queue.Queue()
        self._answer_events: "queue.Queue[dict[str, str]]" = queue.Queue()
        self._recommendation_events: "queue.Queue[dict[str, str]]" = queue.Queue()
        self._fetch_confirmation_events: "queue.Queue[dict[str, str]]" = queue.Queue()
        self._pending_part_text = ""
        self._awaiting_part_clarification = False
        self._ambiguous_part_text = ""
        self._pending_fetch_description = ""

    # ── Public: called by the task graph on every state change ────────────────

    def notify_graph_event(self, event_label: str, state_summary: str) -> None:
        """Call this whenever the task graph changes (complete/undo/reset/live)."""
        self._worker.update_task_state(f"Event: {event_label}\n\n{state_summary}")

    def set_focused_step(self, focused_text: str) -> None:
        """Tell the VLM which step the user has selected. Does NOT clear history."""
        self._worker.set_focused_step(focused_text)

    def set_pending_fetch(self, description: str | None) -> None:
        """Tell intent inference whether a yes/no robot-fetch question is active."""
        self._pending_fetch_description = str(description or "").strip()

    # ── Public: voice input ───────────────────────────────────────────────────

    def submit_question(self, text: str) -> bool:
        """Ask Qwen to infer intent, resolve references, or answer normally.

        The model—not a Python keyword list—decides whether the utterance is a
        physical-part reference. Python subsequently validates the structured
        result before the task graph is allowed to act on it.
        """
        if self._current_status != "ready":
            return False
        text = str(text).strip()
        if not text:
            return False
        self._pending_question = text
        self._pending_part_text = text
        self._render_history(pending=True)

        clarification = ""
        if self._awaiting_part_clarification and self._ambiguous_part_text:
            clarification = (
                "The assistant previously asked the user to clarify this "
                f"ambiguous reference: {self._ambiguous_part_text!r}.\n"
                "Decide whether the current utterance clarifies that physical "
                "part or instead starts an unrelated/general question.\n\n")
        fetch_context = ""
        if self._pending_fetch_description:
            fetch_context = (
                "A robot-fetch confirmation is currently pending for: "
                f"{self._pending_fetch_description}. Interpret a direct affirmative "
                "or negative response as `fetch_confirmation`; otherwise classify "
                "the new utterance normally.\n\n")
        prompt = f"""\
Classify and answer the CURRENT USER UTTERANCE below. Infer its intent from its
meaning; do not classify it by matching a fixed set of request verbs, and do
not mistake part names appearing in the injected assembly state for user intent.

Return ONLY one valid JSON object using exactly this schema:
{{"intent":"part_reference|step_parts_request|recommendation_request|fetch_confirmation|general_question","part_action":"reference|fetch|","confirmation":"yes|no|","step_scope":"selected_step|next_step|","target":"ALLOWED_LABEL|AMBIGUOUS|","candidates":["ALLOWED_LABEL"],"exclude_labels":["ALLOWED_LABEL"],"answer":"text"}}

Rules:
- Use `part_reference` when the user is referring to, identifying, locating,
  requesting, or naming a physical part/tool—even when they use only a noun
  phrase or unfamiliar wording. Set `part_action` to `fetch` only when they ask
  the robot to get, fetch, bring, retrieve, or hand over the object; otherwise
  use `reference`. Infer this semantically rather than by matching one verb.
- For one unambiguous part/tool: put its exact allowed label in `target`, use an
  empty `candidates` list, and leave `answer` empty.
- For a genuinely ambiguous physical reference: set `target` to `AMBIGUOUS`,
  list every plausible exact allowed label in `candidates`, and leave `answer`
  empty. Do not let the selected assembly step erase an ambiguity explicitly
  present in the user's words.
- Use `step_parts_request` when the user asks generically for the part or parts
  needed by a step without naming a physical object, such as "the part for this
  step" or "what do I need for the next step?" Set `step_scope` to
  `selected_step` or `next_step` exactly as requested. Leave `target`,
  `candidates`, and `answer` empty. A generic singular word such as "part" does
  NOT authorize choosing the first input; the graph must return all required
  inputs. Put any explicitly excluded physical objects, such as "other than the
  bearing", in `exclude_labels`. The task graph—not the language model—will
  decide whether the current step is complete enough to advance.
- Use `general_question` for greetings, how-to/status questions, and ordinary
  conversation. Leave `step_scope` and `target` empty, use an empty
  `candidates` list, and put a concise operator-facing response in `answer`.
  Do not copy internal step IDs, canonical labels, or row.stage numbers into
  that answer. For a recommendation, state the physical action directly.
- Use `recommendation_request` when the user asks what to do next, what to
  start with, or requests the recommended step. Leave every other field empty.
  Python will select the READY step using the graph's deterministic policy and
  synchronize that selection with the GUI and Unity; do not choose the step in
  `answer` yourself.
- Use `fetch_confirmation` only when a robot-fetch confirmation is pending and
  the user accepts or rejects it. Set `confirmation` to `yes` or `no`.
- Do not include Markdown or any text outside the JSON object.

Intent examples:
- "Can you give me the part required for this step?" ->
  {{"intent":"step_parts_request","step_scope":"selected_step","target":"","candidates":[],"exclude_labels":[],"answer":""}}
- "Other than the bearing, what other part do I need for this step?" ->
  {{"intent":"step_parts_request","step_scope":"selected_step","target":"","candidates":[],"exclude_labels":["BEARING"],"answer":""}}
- "Can you get me the bearing?" ->
  {{"intent":"part_reference","part_action":"fetch","confirmation":"","step_scope":"","target":"BEARING","candidates":[],"exclude_labels":[],"answer":""}}
- "Where is the bearing?" ->
  {{"intent":"part_reference","part_action":"reference","confirmation":"","step_scope":"","target":"BEARING","candidates":[],"exclude_labels":[],"answer":""}}
- "What should I start with?" ->
  {{"intent":"recommendation_request","step_scope":"","target":"","candidates":[],"exclude_labels":[],"answer":""}}
- "What is the next recommended step?" ->
  {{"intent":"recommendation_request","step_scope":"","target":"","candidates":[],"exclude_labels":[],"answer":""}}

{clarification}{fetch_context}CURRENT USER UTTERANCE:
{text}
"""
        queued = self._worker.submit(
            prompt, self._image_paths, result_kind="intent")
        if queued:
            _print_console_block("VLM QUESTION", text)
        else:
            self._pending_question = ""
            self._pending_part_text = ""
        return queued

    def submit_part_reference(self, text: str) -> bool:
        """Resolve free-form text to one dataset part label without conversation."""
        if self._current_status != "ready":
            return False
        self._pending_part_text = text
        clarification = ""
        if self._awaiting_part_clarification and self._ambiguous_part_text:
            clarification = (
                f"Previous ambiguous expression: {self._ambiguous_part_text}\n"
                f"User clarification: {text}\n\n")
        prompt = (
            "The user is referring to one physical gearbox part. Resolve the "
            "following expression and return exactly one label from the allowed "
            "part-and-tool label list. If more than one non-interchangeable "
            "object fits, return AMBIGUOUS. Return only the label or AMBIGUOUS.\n\n"
            f"{clarification}Expression: {text}"
        )
        queued = self._worker.submit(prompt, self._image_paths,
                                     result_kind="part_reference")
        if queued:
            _print_console_block("VLM PART REFERENCE", text)
        else:
            self._pending_part_text = ""
        return queued

    def poll_part_references(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        while True:
            try:
                events.append(self._part_events.get_nowait())
            except queue.Empty:
                return events

    def poll_answers(self) -> list[dict[str, str]]:
        """Return natural-language VLM answers that should be spoken by TTS."""
        events: list[dict[str, str]] = []
        while True:
            try:
                events.append(self._answer_events.get_nowait())
            except queue.Empty:
                return events

    def poll_recommendation_requests(self) -> list[dict[str, str]]:
        """Return model-recognized requests to select the recommended step."""
        events: list[dict[str, str]] = []
        while True:
            try:
                events.append(self._recommendation_events.get_nowait())
            except queue.Empty:
                return events

    def poll_fetch_confirmations(self) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        while True:
            try:
                events.append(self._fetch_confirmation_events.get_nowait())
            except queue.Empty:
                return events

    def apply_policy_response(self, question: str, label: str, spoken: str) -> None:
        """Show and remember the same friendly response that TTS receives."""
        question = str(question).strip()
        if not question:
            return
        if label == "STEP_PARTS":
            resolved = "Resolved request: parts required by the step"
        elif label == "RECOMMENDED_STEP":
            resolved = "Action: recommended step selected"
        else:
            resolved = f"Resolved label: {label}"
        display_answer = f"{spoken}\n\n{resolved}"
        self._awaiting_part_clarification = label == "AMBIGUOUS"
        self._ambiguous_part_text = question if label == "AMBIGUOUS" else ""
        replaced = False
        for index in range(len(self._exchanges) - 1, -1, -1):
            prior_question, _prior_answer = self._exchanges[index]
            if prior_question == question:
                self._exchanges[index] = (prior_question, display_answer)
                replaced = True
                break
        if not replaced:
            self._exchanges.append((question, display_answer))
            self._history_count = min(
                self._history_count + 1, _VLMWorker.MAX_HISTORY_PAIRS)
        # The debug label stays visible in the panel but is not fed back into
        # model history, where it previously caused repeated BEARING answers.
        self._worker.apply_policy_response(question, spoken)
        self._render_history()
        self._update_history_label()

    def _normalize_part_label(self, raw: str) -> str:
        """Accept explicit labels only; never derive a label from chat prose."""
        cleaned = _strip_md(raw).strip().strip("`\"' .,:;\n\t")
        if cleaned.casefold() == "ambiguous":
            return "AMBIGUOUS"
        exact = {name.casefold(): name for name in self._part_labels}
        if cleaned.casefold() in exact:
            return exact[cleaned.casefold()]

        # Group a specific bearing/pin/screw identifier only if the complete
        # model answer is that identifier. A substring match is unsafe: a
        # normal greeting response can legitimately mention BEARING_ROW1_RIGHT.
        canonical = cleaned.upper().removesuffix(".STL")
        if re.fullmatch(r"BEARING_ROW[1-4]_(LEFT|RIGHT)", canonical):
            return "BEARING"
        if re.fullmatch(r"PIN_ROW[1-4]_(LEFT|RIGHT)", canonical):
            return "PIN"
        screw = re.fullmatch(r"SCREW_ROW([1-4])_(LEFT|RIGHT)", canonical)
        if screw:
            return f"SCREW_ROW{screw.group(1)}"

        try:
            value = json.loads(raw)
            if isinstance(value, str):
                return exact.get(value.strip().casefold(), "INVALID_OUTPUT")
            if isinstance(value, dict):
                for key in ("label", "prediction", "target_name"):
                    candidate = str(value.get(key, "")).strip().casefold()
                    if candidate in exact:
                        return exact[candidate]
        except (json.JSONDecodeError, TypeError):
            pass
        return "INVALID_OUTPUT"

    def _parse_intent_envelope(self, raw: str) -> dict[str, object]:
        """Validate Qwen's intent JSON without inferring semantics in Python."""
        text = str(raw).strip()
        value = None
        # Accept a JSON object surrounded by accidental whitespace or a fenced
        # block, but never recover a part label from arbitrary prose.
        decoder = json.JSONDecoder()
        for offset, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _end = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            return {"intent": "invalid", "raw": raw}

        intent = str(value.get("intent", "")).strip().casefold()
        if intent == "fetch_confirmation":
            confirmation = str(value.get("confirmation", "")).strip().casefold()
            if confirmation not in {"yes", "no"}:
                return {"intent": "invalid", "raw": raw}
            return {
                "intent": intent,
                "confirmation": confirmation,
                "raw": raw,
            }
        if intent == "recommendation_request":
            return {"intent": intent, "raw": raw}
        if intent == "step_parts_request":
            step_scope = str(value.get("step_scope", "")).strip().casefold()
            if step_scope not in {"selected_step", "next_step"}:
                return {"intent": "invalid", "raw": raw}
            exclude_labels: list[str] = []
            raw_exclusions = value.get("exclude_labels", [])
            if isinstance(raw_exclusions, list):
                for candidate in raw_exclusions:
                    label = self._normalize_part_label(str(candidate))
                    if label in self._part_labels and label not in exclude_labels:
                        exclude_labels.append(label)
            return {
                "intent": intent,
                "step_scope": step_scope,
                "exclude_labels": exclude_labels,
                "raw": raw,
            }
        if intent == "general_question":
            answer = str(value.get("answer", "")).strip()
            if not answer:
                return {"intent": "invalid", "raw": raw}
            return {"intent": intent, "answer": answer, "raw": raw}
        if intent != "part_reference":
            return {"intent": "invalid", "raw": raw}

        target = self._normalize_part_label(str(value.get("target", "")))
        if target not in {*self._part_labels, "AMBIGUOUS"}:
            return {"intent": "invalid", "raw": raw}

        candidate_labels: list[str] = []
        raw_candidates = value.get("candidates", [])
        if isinstance(raw_candidates, list):
            for candidate in raw_candidates:
                label = self._normalize_part_label(str(candidate))
                if label in self._part_labels and label not in candidate_labels:
                    candidate_labels.append(label)
        if target != "AMBIGUOUS":
            candidate_labels.clear()
        part_action = str(value.get("part_action", "reference")).strip().casefold()
        if part_action not in {"reference", "fetch"}:
            return {"intent": "invalid", "raw": raw}
        return {
            "intent": intent,
            "target": target,
            "part_action": part_action,
            "candidate_labels": candidate_labels,
            "raw": raw,
        }

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
                           callback=self._ask, width=90, enabled=False)
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

            elif kind == "intent":
                question = self._pending_question
                envelope = self._parse_intent_envelope(payload)
                intent = envelope.get("intent")
                if intent == "fetch_confirmation":
                    self._fetch_confirmation_events.put({
                        "text": question,
                        "confirmation": str(envelope["confirmation"]),
                        "raw": payload,
                    })
                    _print_console_block(
                        "VLM INTENT",
                        f"fetch_confirmation -> {envelope['confirmation']}\nraw: {payload}",
                    )
                    self._pending_question = ""
                    self._pending_part_text = ""
                elif intent == "recommendation_request":
                    self._recommendation_events.put({
                        "text": question,
                        "raw": payload,
                    })
                    _print_console_block(
                        "VLM INTENT",
                        f"recommendation_request\nraw: {payload}",
                    )
                    self._pending_question = ""
                    self._pending_part_text = ""
                elif intent in {"part_reference", "step_parts_request"}:
                    is_step_parts = intent == "step_parts_request"
                    label = ("STEP_PARTS" if is_step_parts
                             else str(envelope["target"]))
                    self._part_events.put({
                        "text": question,
                        "label": label,
                        "part_action": str(envelope.get("part_action", "reference")),
                        "step_scope": str(envelope.get("step_scope", "")),
                        "exclude_labels": list(
                            envelope.get("exclude_labels", [])),
                        "candidate_labels": list(
                            envelope.get("candidate_labels", [])),
                        "raw": payload,
                    })
                    _print_console_block(
                        "VLM INTENT",
                        f"{intent} -> {label}\n"
                        f"step_scope: {envelope.get('step_scope', '')}\n"
                        f"exclude_labels: {envelope.get('exclude_labels', [])}\n"
                        f"candidates: {envelope.get('candidate_labels', [])}\n"
                        f"raw: {payload}",
                    )
                    # The task-graph policy will add its validated, friendly
                    # response to both the UI and conversation history.
                    self._pending_question = ""
                    self._pending_part_text = ""
                else:
                    if intent == "general_question":
                        answer = _humanize_general_answer(str(envelope["answer"]))
                    else:
                        answer = (
                            "I could not understand that reliably. Please rephrase "
                            "your request or question.")
                    _print_console_block(
                        "VLM INTENT",
                        "general_question" if intent == "general_question"
                        else f"invalid\nraw: {payload}",
                    )
                    _print_console_block("VLM ANSWER", answer)
                    self._answer_events.put({
                        "text": question,
                        "answer": answer,
                        "intent": str(intent),
                    })
                    self._exchanges.append((question, answer))
                    self._history_count = min(
                        self._history_count + 1,
                        _VLMWorker.MAX_HISTORY_PAIRS,
                    )
                    self._worker.apply_policy_response(question, answer)
                    self._pending_question = ""
                    self._pending_part_text = ""
                    self._awaiting_part_clarification = False
                    self._ambiguous_part_text = ""
                    self._render_history()
                    self._update_history_label()
                    self.dpg.configure_item("vlm_ask_button", enabled=True)

            elif kind == "answer":
                answer = _humanize_general_answer(payload)
                _print_console_block("VLM ANSWER", answer)
                self._exchanges.append((self._pending_question, answer))
                self._history_count = min(
                    self._history_count + 1, _VLMWorker.MAX_HISTORY_PAIRS)
                self._pending_question = ""
                self._render_history()
                self._update_history_label()
                self.dpg.configure_item("vlm_ask_button", enabled=True)
                # Retained for explicit legacy callers. Normal typed and voice
                # questions use the structured ``intent`` result above.

            elif kind == "part_reference":
                label = self._normalize_part_label(payload)
                self._part_events.put({
                    "text": self._pending_part_text,
                    "label": label,
                    "raw": payload,
                })
                _print_console_block("VLM PART RESULT", f"{label}\nraw: {payload}")
                self._pending_part_text = ""

            elif kind == "error":
                error_text = _strip_md(payload)
                _print_console_block("VLM ERROR", error_text)
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
                self._awaiting_part_clarification = False
                self._ambiguous_part_text = ""
                self._history_count = 0
                self._update_history_label()
                banner = "[Graph state changed — conversation reset]\n\n" + payload
                _print_console_block("VLM GRAPH STATE", banner)
                self.dpg.set_value("vlm_answer", banner)

            elif kind == "history_reset":
                self._exchanges.clear()
                self._pending_question = ""
                self._awaiting_part_clarification = False
                self._ambiguous_part_text = ""
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
        print(f"[VLM status] {label}", flush=True)
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
        self.dpg.set_value("vlm_question", "")
        self.submit_question(question)

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
