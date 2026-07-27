#!/usr/bin/env python3
"""Interactive gearbox assembly task graph.

The model is intentionally independent from the GUI so that the dependency and
part-transformation rules can be tested without opening a window.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Iterable

# Ensure task_graph/ (this dir) and the repo root (its parent) are both importable, so we can pull
# the canonical ports from main_setting.py even though the viewer runs from the task_graph/ subdir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_assistant import VLMAssistant  # noqa: E402
from speech_listener import SpeechListener  # noqa: E402
import gearbox_control  # noqa: E402  (--with-controller: run the controller in-process)

# Ports live canonically in main_setting.py; fall back to literals so the viewer still runs if that
# import is unavailable.
try:
    import main_setting
    _LOCALHOST                    = main_setting.LOCALHOST                  # loopback for same-machine Python IPC
    # IN  — gearbox_control.py PUBs semantic assembly events (show/complete/uncomplete/reset) here
    #        after every Unity click; this viewer SUBs and drives its TaskGraph model + node colors.
    _DEFAULT_CTRL_EVENTS_IN_PORT  = main_setting.GEARBOX_TASKGRAPH_PORT    # 5022
    # OUT — when a step node is selected in this viewer, it PUBs the matching (row, stage) here so
    #        that a gearbox_control.py --open-3d window can highlight the corresponding pegboard tools.
    _DEFAULT_STEP_OUT_PORT        = main_setting.GEARBOX_STEP_SELECT_PORT  # 5025
    #In one sentence: 5022 is the viewer listening to assembly progress from Unity/controller; 
    # 5025 is the viewer broadcasting which step you've clicked so a 3D tool viewer can react.
except Exception:
    _LOCALHOST                = "127.0.0.1"
    _DEFAULT_CTRL_EVENTS_IN_PORT   = 5022   # IN  — gearbox_control.py -> this viewer (assembly events)
    _DEFAULT_STEP_OUT_PORT = 5025   # OUT — this viewer -> gearbox_control.py --open-3d (selected step)


@dataclass(frozen=True)
class Step:
    id: str                    # e.g. "r2_bearing_left"
    title: str                 # e.g. "Row 2.1: bearing into left stand"
    row: int                   # e.g. 2  (1-4 for regular rows; 0 for finish_gearbox)
    stage: int                 # e.g. 0  (0=bearings, 1=gear rod, 2=fasten first, 3=insert rod+fit second, 4=fasten second, 5=crank handle, 6=verify)
    inputs: tuple[str, ...]    # e.g. ("BEARING_ROW2_LEFT", "STAND_ROW2_LEFT")
    output: str                # e.g. "BEARING_STAND_ROW2_LEFT_ASSEMBLY"
    description: str           # e.g. "Insert BEARING_ROW2_LEFT into STAND_ROW2_LEFT."
    context: tuple[str, ...] = ()  # e.g. ("BASE_BOARD",) — parts present but not consumed


PROVIDED_PARTS = {
    "BASE_BOARD",

    "BEARING_ROW1_LEFT", "BEARING_ROW1_RIGHT", 
    "CRANK_HANDLE_ROW1",
    "GEAR_ROD_ROW1", 
    "GEAR_ROW1_LEFT", 
    "PIN_ROW1_LEFT", "PIN_ROW1_RIGHT",
    "SCREW_ROW1_LEFT", "SCREW_ROW1_RIGHT", 
    "STAND_ROW1_LEFT", "STAND_ROW1_RIGHT",

    "BEARING_ROW2_LEFT", "BEARING_ROW2_RIGHT", 
    "GEAR_ROD_ROW2",
    "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", 
    "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT",
    "SCREW_ROW2_LEFT", "SCREW_ROW2_RIGHT", 
    "STAND_ROW2_LEFT", "STAND_ROW2_RIGHT",

    "BEARING_ROW3_LEFT", "BEARING_ROW3_RIGHT", 
    "GEAR_ROD_ROW3",
    "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", 
    "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT",
    "SCREW_ROW3_LEFT", "SCREW_ROW3_RIGHT", 
    "STAND_ROW3_LEFT", "STAND_ROW3_RIGHT",

    "BEARING_ROW4_LEFT", "BEARING_ROW4_RIGHT", 
    "GEAR_ROD_ROW4",
    "GEAR_ROW4_LEFT", 
    "PIN_ROW4_LEFT", 
    "SCREW_ROW4_LEFT", "SCREW_ROW4_RIGHT",
    "STAND_ROW4_LEFT", "STAND_ROW4_RIGHT",
}


def build_steps() -> list[Step]:
    steps: list[Step] = []
    gear_inputs = {
        1: ("GEAR_ROD_ROW1", "GEAR_ROW1_LEFT", "PIN_ROW1_LEFT", "PIN_ROW1_RIGHT"),
        2: ("GEAR_ROD_ROW2", "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT"),
        3: ("GEAR_ROD_ROW3", "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT"),
        4: ("GEAR_ROD_ROW4", "GEAR_ROW4_LEFT", "PIN_ROW4_LEFT"),
    }
    gear_text = {
        1: "Attach the single large gear and secure it with the Row 1 pins.",
        2: "Attach the small and medium gears and secure both with their pins.",
        3: "Attach the small and medium gears and secure both with their pins.",
        4: "Attach the single large gear and secure it with the supplied Row 4 pin.",
    }

    for row in range(1, 5):
        for side in ("Left", "Right"):
            steps.append(Step(
                id=f"r{row}_bearing_{side.lower()}",
                title=f"Row {row}.{1 if side == 'Left' else 3}: bearing into {side.lower()} stand",
                row=row,
                stage=0,
                inputs=(f"BEARING_ROW{row}_{side.upper()}", f"STAND_ROW{row}_{side.upper()}"),
                output=f"BEARING_STAND_ROW{row}_{side.upper()}_ASSEMBLY",
                description=(f"Insert BEARING_ROW{row}_{side.upper()} into "
                             f"STAND_ROW{row}_{side.upper()}."),
            ))

        steps.append(Step(
            id=f"r{row}_gear_rod",
            title=f"Row {row}.2: assemble gear rod",
            row=row,
            stage=1,
            inputs=gear_inputs[row],
            output=f"GEAR_ROD_ROW{row}_ASSEMBLY",
            description=gear_text[row],
        ))

        first, second = "Left", "Right"
        steps.append(Step(
            id=f"r{row}_fasten_first_stand",
            title=f"Row {row}.4: fasten {first.lower()} stand",
            row=row,
            stage=2,
            inputs=(f"BEARING_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
                    f"SCREW_ROW{row}_{first.upper()}"),
            output=f"FASTENED_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Fasten the {first.lower()} bearing-stand assembly to BASE_BOARD "
                         f"with SCREW_ROW{row}_{first.upper()} before inserting the gear rod."),
        ))
        steps.append(Step(
            id=f"r{row}_insert_rod_and_fit_second",
            title=f"Row {row}.5: insert rod and fit {second.lower()} stand",
            row=row,
            stage=3,
            inputs=(f"FASTENED_STAND_ROW{row}_{first.upper()}_ASSEMBLY",
                    f"GEAR_ROD_ROW{row}_ASSEMBLY",
                    f"BEARING_STAND_ROW{row}_{second.upper()}_ASSEMBLY"),
            output=f"UNFASTENED_SECOND_STAND_ROW{row}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Insert the assembled gear rod through the fastened {first.lower()} "
                         f"stand while fitting the unfastened {second.lower()} stand over the "
                         "other end of the rod."),
        ))
        steps.append(Step(
            id=f"r{row}_fasten_second_stand",
            title=f"Row {row}.6: fasten {second.lower()} stand",
            row=row,
            stage=4,
            inputs=(f"UNFASTENED_SECOND_STAND_ROW{row}_ASSEMBLY",
                    f"SCREW_ROW{row}_{second.upper()}"),
            output=f"MOUNTED_ROW{row}_ASSEMBLY",
            context=("BASE_BOARD",),
            description=(f"Fasten the {second.lower()} stand to BASE_BOARD with "
                         f"SCREW_ROW{row}_{second.upper()} after the rod and stand are in place."),
        ))

    steps.append(Step(
        id="r1_attach_handle",
        title="Row 1.7: attach crank handle",
        row=1,
        stage=5,
        inputs=("MOUNTED_ROW1_ASSEMBLY", "CRANK_HANDLE_ROW1"),
        output="CRANK_MOUNTED_ROW1_ASSEMBLY",
        description="Attach CRANK_HANDLE_ROW1 only after GEAR_ROD_ROW1 is between both stands.",
    ))
    steps.append(Step(
        id="finish_gearbox",
        title="Verify and finish gearbox",
        row=0,
        stage=6,
        inputs=("CRANK_MOUNTED_ROW1_ASSEMBLY", "MOUNTED_ROW2_ASSEMBLY",
                "MOUNTED_ROW3_ASSEMBLY", "MOUNTED_ROW4_ASSEMBLY"),
        output="COMPLETED_GEARBOX_ASSEMBLY",
        context=("BASE_BOARD",),
        description="Verify alignment, gear meshing, and free rotation of all four rows.",
    ))
    return steps


class TaskGraph:
    def __init__(self, available_parts: Iterable[str] = PROVIDED_PARTS):
        self.steps = build_steps()
        self.by_id = {step.id: step for step in self.steps}
        self.initial_parts = set(available_parts)
        self.active_parts = set(self.initial_parts)
        self.completed: list[str] = []
        self.transform_history: dict[str, str] = {}

    def reset(self) -> None:
        self.active_parts = set(self.initial_parts)
        self.completed.clear()
        self.transform_history.clear()

    def missing(self, step: Step) -> list[str]:
        return [part for part in (*step.inputs, *step.context)
                if part not in self.active_parts]

    def is_ready(self, step: Step) -> bool:
        return step.id not in self.completed and not self.missing(step)

    def state(self, step: Step) -> str:
        if step.id in self.completed:
            return "complete"
        return "ready" if self.is_ready(step) else "blocked"

    def complete(self, step: Step) -> tuple[bool, str]:
        if step.id in self.completed:
            return False, f"{step.id} is already complete; its result is {step.output}."
        missing = self.missing(step)
        if missing:
            return False, (f"WARNING: {step.id} cannot happen yet. Missing active input(s): "
                           f"{', '.join(missing)}.")
        for part in step.inputs:
            self.active_parts.remove(part)
            self.transform_history[part] = step.output
        self.active_parts.add(step.output)
        self.completed.append(step.id)
        return True, (f"COMPLETED {step.id}: {', '.join(step.inputs)} transformed into "
                      f"{step.output}.")

    def is_frontier(self, step: Step) -> bool:
        """A completed step is frontier-safe when no completed step consumed its output."""
        return step.id in self.completed and step.output in self.active_parts

    def completed_consumers(self, step: Step) -> list[Step]:
        return [candidate for candidate in self.steps
                if candidate.id in self.completed and step.output in candidate.inputs]

    def frontier_steps(self) -> list[Step]:
        return [self.by_id[step_id] for step_id in self.completed
                if self.is_frontier(self.by_id[step_id])]

    def undo(self, step: Step | None = None) -> tuple[bool, str]:
        """Undo any completed frontier node without invalidating dependent state."""
        if not self.completed:
            return False, "WARNING: there is no completed step to undo."
        target = step or self.by_id[self.completed[-1]]
        if target.id not in self.completed:
            return False, f"WARNING: {target.id} is not complete, so it cannot be undone."
        if not self.is_frontier(target):
            consumers = self.completed_consumers(target)
            blocked_by = ", ".join(consumer.id for consumer in consumers) or "a dependent assembly"
            return False, (f"WARNING: {target.id} is not on the completed frontier. "
                           f"Undo its completed dependent step(s) first: {blocked_by}.")
        self.active_parts.remove(target.output)
        self.active_parts.update(target.inputs)
        for part in target.inputs:
            if self.transform_history.get(part) == target.output:
                del self.transform_history[part]
        self.completed.remove(target.id)
        return True, (f"UNDONE {target.id}: removed {target.output} and restored "
                      f"{', '.join(target.inputs)}.")

    def active_parts_text(self) -> str:
        parts = sorted(self.active_parts)
        return f"ACTIVE PARTS ({len(parts)}):\n" + "\n".join(f"  {part}" for part in parts)

    def producer_for(self, part: str) -> Step | None:
        return next((step for step in self.steps if step.output == part), None)

    def add_part(self, part: str) -> str:
        if part in self.active_parts:
            return f"{part} is already active."
        self.active_parts.add(part)
        self.initial_parts.add(part)
        return f"Added {part} to the inventory."

    def find_steps(self, query: str) -> list[Step]:
        needle = query.strip().lower()
        if not needle:
            return []
        exact = self.by_id.get(needle)
        if exact:
            return [exact]
        matches = []
        for step in self.steps:
            searchable = " ".join((step.id, step.title, step.output, *step.inputs)).lower()
            if needle in searchable:
                matches.append(step)
        return matches

    def trace_part(self, part: str) -> str:
        exact = next((p for p in self.active_parts if p.lower() == part.lower()), None)
        consumed = next((p for p in self.transform_history if p.lower() == part.lower()), None)
        canonical = exact or consumed or part
        if consumed:
            current = self.transform_history[consumed]
            while current in self.transform_history:
                current = self.transform_history[current]
            prefix = f"{consumed} has already been transformed; its current assembly is {current}. "
            canonical = current
        elif exact:
            prefix = f"Found active part {exact}. "
        else:
            known = {p for s in self.steps for p in (*s.inputs, s.output, *s.context)}
            candidates = sorted(p for p in known if part.lower() in p.lower())
            if candidates:
                return "Possible matches: " + ", ".join(candidates[:12])
            return f"WARNING: no part or assembly named '{part}' exists in this task graph."

        candidates = [s for s in self.steps
                      if s.id not in self.completed and canonical in (*s.inputs, *s.context)]
        if not candidates:
            return prefix + "No remaining step directly uses it."
        ready = [s for s in candidates if self.is_ready(s)]
        if ready:
            return prefix + "Next READY step: " + "; ".join(f"{s.id} ({s.title})" for s in ready)
        details = []
        for step in candidates:
            details.append(f"{step.id} is blocked by {', '.join(self.missing(step))}")
        return prefix + "WARNING: its next step should not happen yet: " + "; ".join(details)

    @classmethod
    def validate(cls) -> list[str]:
        """Return model problems; missing raw inventory is reported separately by the GUI."""
        graph = cls()
        errors: list[str] = []
        ids = [s.id for s in graph.steps]
        outputs = [s.output for s in graph.steps]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate step ID")
        if len(outputs) != len(set(outputs)):
            errors.append("Duplicate assembled-part output")
        produced = set(outputs)
        for step in graph.steps:
            for part in step.inputs:
                if part not in graph.initial_parts and part not in produced:
                    errors.append(f"Unknown input {part} in {step.id}")
        return errors

    @staticmethod
    def steps_for_control(row: int, stage: int) -> list[str]:
        """Bridge a gearbox_control.py (row, control-stage) to this graph's step id(s).

        The two files number "stage" differently: gearbox_control uses per-row control-stages 1-8,
        while this graph uses named steps. The control stages now map one-to-one onto task steps:
        stage 5 is the rod-insert + right-stand fit, stage 6 the right-stand fastening."""
        if stage == 8 or row == 0:
            return ["finish_gearbox"]
        if stage == 7:
            return ["r1_attach_handle"] if row == 1 else []
        return {
            1: [f"r{row}_bearing_left"],
            2: [f"r{row}_gear_rod"],
            3: [f"r{row}_bearing_right"],
            4: [f"r{row}_fasten_first_stand"],
            5: [f"r{row}_insert_rod_and_fit_second"],
            6: [f"r{row}_fasten_second_stand"],
        }.get(stage, [])

    @staticmethod
    def control_coords_for(step_id: str):
        """Inverse of steps_for_control: a task-step id -> the gearbox_control (row, control-stage)
        that opens it, or None if the id isn't a mapped control step. Lets this viewer tell an
        external (row, stage) consumer — e.g. gearbox_control.py --open-3d — which step was
        just selected."""
        if step_id == "finish_gearbox":
            return 0, 8
        for row in range(1, 5):
            for stage in range(1, 8):
                if step_id in TaskGraph.steps_for_control(row, stage):
                    return row, stage
        return None

    def state_summary(self) -> str:
        """Build a concise state description for the VLM system context."""
        completed  = self.completed
        ready      = [s for s in self.steps if self.is_ready(s)]
        blocked    = [s for s in self.steps
                      if s.id not in self.completed and not self.is_ready(s)]
        assemblies = sorted(p for p in self.active_parts if p.endswith("_ASSEMBLY"))

        lines = [f"Progress: {len(completed)} / {len(self.steps)} steps completed"]

        if completed:
            lines.append("\nCompleted steps (most recent last):")
            for sid in completed[-8:]:
                step = self.by_id[sid]
                lines.append(f"  - {step.title}  ->  {step.output}")

        if ready:
            lines.append(
                "\nREADY steps — ALL of these are currently unlocked and INDEPENDENT."
                "\nThey have NO ordering requirement among themselves; the user may perform"
                " them in any order they prefer:"
            )
            for step in ready[:8]:
                lines.append(f"  - [{step.id}] {step.title}")

        if blocked:
            lines.append("\nBLOCKED steps (prerequisites not yet met):")
            for step in blocked[:6]:
                missing_parts  = self.missing(step)
                missing_steps  = []
                missing_raw    = []
                for part in missing_parts:
                    producer = self.producer_for(part)
                    if producer:
                        missing_steps.append(f"[{producer.id}]")
                    else:
                        missing_raw.append(part)
                needs = ", ".join(missing_steps + missing_raw)
                lines.append(f"  - [{step.id}] {step.title}  (needs these steps first: {needs})")

        if assemblies:
            lines.append("\nActive assemblies in inventory:")
            for a in assemblies[:12]:
                lines.append(f"  - {a}")

        lines.append(
            "\nIMPORTANT: Base your answers ONLY on the dependency structure above."
            " Do NOT infer ordering constraints from step descriptions — only the"
            " BLOCKED list represents real prerequisites. If a step is READY, the"
            " user may do it at any time regardless of the order steps are listed."
        )

        return "\n".join(lines)

    def recommend_next_step(self) -> "Step | None":
        """Row-by-row recommendation policy: pick the READY step with the lowest row
        number, then lowest stage within that row. The finish step (row=0) is treated
        as the highest row so it is only recommended when everything else is done."""
        ready = [s for s in self.steps if self.is_ready(s)]
        if not ready:
            return None

        def _priority(s: Step) -> tuple:
            row = s.row if s.row > 0 else 999
            return (row, s.stage, s.id)

        return min(ready, key=_priority)


class DearPyGuiTaskGraphApp:
    """Dear PyGui task-graph view and terminal controller."""

    COLORS = {
        "complete": (46, 157, 96, 255),
        "ready": (39, 132, 216, 255),
        "blocked": (209, 138, 39, 255),
    }

    # Status → (RGBA color, display label)
    _VOICE_STATUS_STYLE: dict[str, tuple[tuple, str]] = {
        "loading":     ((180, 180, 180, 255), "Loading model..."),
        "idle":        ((255, 220, 50,  255), "Idle"),
        "speech":      ((50,  200, 255, 255), "Speech detected..."),
        "queued":      ((200, 160, 50,  255), "Queued..."),
        "transcribing":((255, 165, 50,  255), "Transcribing..."),
        "listening":   ((50,  220, 80,  255), "Listening"),
        "error":       ((255, 80,  80,  255), "Error"),
    }

    def __init__(self) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.graph = TaskGraph()
        self.selected_id: str | None = None
        self.log_lines: list[str] = []
        self.node_tags: dict[str, str] = {}
        self.input_attributes: dict[str, str] = {}
        self.output_attributes: dict[str, str] = {}
        self.themes: dict[str, str] = {}

        # Live-mirror state
        self.active_ids: set[str] = set()
        self.recommended_id: str | None = None
        self._live_queue: "queue.Queue[dict]" = queue.Queue()
        self._live_running = False
        self._live_thread: threading.Thread | None = None
        self._live_sub = None
        self._select_pub = None   # PUB -> gearbox_control.py --open-3d viewer (selected step)

        self._speech = None   # SpeechListener, set in run() if enabled
        self._vlm    = None   # VLMAssistant, set in run() if enabled

    def build(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        dpg.create_viewport(title="Gearbox Assembly Task Graph", width=3420, height=1400,
                            min_width=1400, min_height=800)
        self._create_themes()

        with dpg.window(tag="primary_window", label="Gearbox Assembly Task Graph"):
            dpg.add_text("Gearbox Assembly Dependency Graph", color=(225, 232, 240),
                         tag="main_title")
            dpg.add_text("Blue = ready     Orange = blocked     Green = complete"
                         "     Purple = selected / active     Gold = recommended",
                         color=(170, 180, 195))
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=2.8)
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.1)
                with dpg.table_row():
                    with dpg.table_cell():
                        with dpg.child_window(height=-1, horizontal_scrollbar=True):
                            with dpg.node_editor(tag="task_node_editor", minimap=True,
                                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                                self._create_nodes()
                                self._create_links()
                    with dpg.table_cell():
                        with dpg.child_window(height=-1):
                            self._create_side_panel()
                    with dpg.table_cell():
                        with dpg.child_window(height=-1):
                            if self._vlm is not None:
                                self._vlm.build_inline()
                            else:
                                dpg.add_text("VLM Assistant",
                                             color=(225, 232, 240))
                                dpg.add_separator()
                                dpg.add_text(
                                    "Not enabled.\n\nRun with:\n"
                                    "  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct",
                                    color=(140, 155, 175))

        dpg.set_primary_window("primary_window", True)
        dpg.bind_item_theme("task_node_editor", "node_editor_theme")
        self.refresh()
        self.log("Type a part name, or type 'help' for commands.")
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self, live_port: int | None = None,
            voice_device: str | None = None,
            wake_word: str = "hey robot",
            vlm_model: str | None = None,
            task_description_path: str | None = None,
            select_port: int | None = None) -> None:
        # Pre-import transformers on the main thread before any worker threads start.
        # The VLM thread and ASR (NeMo) thread both import from transformers; if they
        # race during the initial import, Python's partially-initialized sys.modules
        # entry causes "cannot import name X from transformers".
        try:
            import transformers as _tf  # noqa: F401
        except Exception:
            pass

        if vlm_model is not None:
            desc_path = task_description_path or str(
                Path(__file__).parent / "task_description.md")
            self._vlm = VLMAssistant(self.dpg, desc_path, model_name=vlm_model)
            print(f"[VLM] Assistant created: {vlm_model}")
        self.build()
        if live_port is not None:
            self.start_live_listener(live_port)
        if select_port is not None:
            self.start_select_publisher(select_port)
        if voice_device is not None:
            try:
                self._speech = SpeechListener(device=voice_device, wake_word=wake_word)
                self._speech.start()
                self.log(f"[Voice] Started on device: {voice_device}")
            except Exception as e:
                self.log(f"[Voice] Failed to start: {e}")
        if self._vlm is not None:
            try:
                self.dpg.set_viewport_drop_callback(self._vlm.on_file_drop)
            except Exception:
                pass  # older DPG versions may not support this

        dpg = self.dpg
        while dpg.is_dearpygui_running():
            self._drain_live_queue()
            self._poll_speech()
            if self._vlm is not None:
                self._vlm.tick()
            dpg.render_dearpygui_frame()
        self._live_running = False
        if self._speech is not None:
            self._speech.close()
        if self._vlm is not None:
            self._vlm.close()
        dpg.destroy_context()

    # ── Live mirror of gearbox_control.py ────────────────────────────────────
    def start_live_listener(self, port: int) -> bool:
        """Bind a SUB (receiver binds, per the repo's Python<->Python convention) and drain
        controller events on a background thread. Failures are non-fatal — the GUI still opens."""
        try:
            import zmq
        except Exception as e:
            self.log(f"Live link disabled (zmq unavailable: {e}).")
            return False
        try:
            ctx = zmq.Context.instance()
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            sub.bind(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Live link disabled (could not bind :{port}: {e}).")
            return False
        self._live_sub = sub
        self._live_running = True

        def _loop() -> None:
            poller = zmq.Poller()
            poller.register(sub, zmq.POLLIN)
            while self._live_running:
                if not dict(poller.poll(timeout=200)):
                    continue
                while True:
                    try:
                        raw = sub.recv_string(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        self._live_queue.put(json.loads(raw))
                    except json.JSONDecodeError:
                        continue

        self._live_thread = threading.Thread(target=_loop, daemon=True)
        self._live_thread.start()
        self.log(f"Live link listening on tcp://{_LOCALHOST}:{port} (controller mirror).")
        return True

    def start_select_publisher(self, port: int) -> bool:
        """Connect a PUB (sender connects, per convention) so pressing 'Select step' can tell an
        (row, stage) consumer — gearbox_control.py --open-3d — which step is selected. Non-fatal."""
        try:
            import zmq
        except Exception as e:
            self.log(f"Step-select link disabled (zmq unavailable: {e}).")
            return False
        try:
            ctx = zmq.Context.instance()
            pub = ctx.socket(zmq.PUB)
            pub.connect(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Step-select link disabled (could not connect :{port}: {e}).")
            return False
        self._select_pub = pub
        self.log(f"Step-select link -> tcp://{_LOCALHOST}:{port} (open3d viewer).")
        return True

    def _send_select(self, msg: dict) -> None:
        if self._select_pub is None:
            return
        try:
            self._select_pub.send_string(json.dumps(msg))
        except Exception:
            pass

    def _drain_live_queue(self) -> None:
        while True:
            try:
                msg = self._live_queue.get_nowait()
            except queue.Empty:
                return
            self._apply_live_event(msg)

    def _poll_speech(self) -> None:
        if self._speech is None:
            return
        dpg = self.dpg
        events = self._speech.poll()

        # Status label + color
        status = self._speech.current_status
        color, label = self._VOICE_STATUS_STYLE.get(
            status, ((200, 200, 200, 255), status))
        if self._speech.listening_active:
            label = f"Listening  (wake word: \"{self._speech.wake_word}\")"
        elif status == "idle":
            label = f"Idle — say \"{self._speech.wake_word}\""
        dpg.set_value("voice_status", label)
        dpg.configure_item("voice_status", color=list(color))

        # Timer
        if self._speech.listening_active:
            dpg.set_value("voice_timer", f"{self._speech.remaining_time:.0f}s remaining")
        else:
            dpg.set_value("voice_timer", "")

        # RMS bar (clamp to [0, 1])
        dpg.set_value("voice_rms", min(self._speech.current_rms * 10.0, 1.0))

        # Transcript log
        history = list(self._speech.transcript_history)
        if history:
            dpg.set_value("voice_transcripts", "\n".join(f"› {t}" for t in history))

        # Log notable events to the terminal; optionally route transcripts to VLM
        route_to_vlm = (self._vlm is not None
                        and dpg.get_value("voice_to_vlm"))
        for kind, payload in events:
            if kind == "wake_word":
                self.log("[Voice] Wake word detected — listening.")
            elif kind == "transcript":
                self.log(f"[Voice] {payload}")
                if route_to_vlm:
                    sent = self._vlm.submit_question(payload)
                    if not sent:
                        self.log("[Voice->VLM] Model busy — transcript dropped.")
            elif kind == "timeout":
                self.log("[Voice] Timed out — back to idle.")
            elif kind == "error":
                self.log(f"[Voice error] {payload}")

    def _apply_live_event(self, msg: dict) -> None:
        """Apply one controller event (main/UI thread): purple overlay + built-in complete/undo."""
        event = msg.get("event")
        row, stage = msg.get("row", 0), msg.get("stage", 0)
        ids = [i for i in TaskGraph.steps_for_control(row, stage) if i in self.graph.by_id]
        if event == "show":
            self.active_ids = set(ids)
        elif event == "close":
            self.active_ids = set()
        elif event == "reset":
            self.graph.reset()
            self.active_ids     = set()
            self.recommended_id = None
            self.log("Live: controller reset all progress.")
            self._notify_vlm("RESET: controller reset all progress")
        elif event == "complete":
            for sid in ids:
                ok, message = self.graph.complete(self.graph.by_id[sid])
                self.log("Live: " + message)
            if ids:
                self._notify_vlm(f"COMPLETE (live): {', '.join(ids)}")
        elif event == "uncomplete":
            for sid in reversed(ids):
                ok, message = self.graph.undo(self.graph.by_id[sid])
                self.log("Live: " + message)
            if ids:
                self._notify_vlm(f"UNDO (live): {', '.join(ids)}")
        else:
            return
        self.refresh()

    def _create_themes(self) -> None:
        dpg = self.dpg
        with dpg.theme(tag="assembly_tree_theme"):
            with dpg.theme_component(dpg.mvTreeNode):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (85, 235, 130, 255),
                                    category=dpg.mvThemeCat_Core)
        with dpg.theme(tag="node_editor_theme"):
            with dpg.theme_component(dpg.mvNodeEditor):
                dpg.add_theme_color(dpg.mvNodeCol_Link, (105, 190, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (180, 225, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (255, 235, 120, 255),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, 3.0,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_PinCircleRadius, 5.0,
                                    category=dpg.mvThemeCat_Nodes)
        for state, color in self.COLORS.items():
            tag = f"node_theme_{state}"
            background = tuple(max(18, int(channel * 0.48)) for channel in color[:3]) + (255,)
            selected_background = tuple(max(25, int(channel * 0.68)) for channel in color[:3]) + (255,)
            with dpg.theme(tag=tag):
                with dpg.theme_component(dpg.mvNode):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                        category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, color,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, color,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, color,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, color,
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                        category=dpg.mvThemeCat_Nodes)
            self.themes[state] = tag

        # Purple "active" overlay theme — bound to a node while its stage is open in the live
        # controller (a part was clicked in Unity), overriding its state color until the menu closes.
        active_color = (168, 85, 247, 255)
        background = tuple(max(18, int(channel * 0.48)) for channel in active_color[:3]) + (255,)
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in active_color[:3]) + (255,)
        with dpg.theme(tag="node_theme_active"):
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)
        self.themes["active"] = "node_theme_active"

        # Gold/yellow "recommended" overlay theme — shown when the system recommends
        # a step as the next to perform.
        rec_color  = (255, 195, 0, 255)
        background = tuple(max(18, int(channel * 0.48)) for channel in rec_color[:3]) + (255,)
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in rec_color[:3]) + (255,)
        with dpg.theme(tag="node_theme_recommended"):
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)
        self.themes["recommended"] = "node_theme_recommended"

    def _create_nodes(self) -> None:
        dpg = self.dpg
        stage_x = {0: 20, 1: 500, 2: 980, 3: 1460, 4: 1940, 5: 2420, 6: 2900}
        for step in self.graph.steps:
            node_tag = f"node::{step.id}"
            in_tag = f"node_in::{step.id}"
            out_tag = f"node_out::{step.id}"
            self.node_tags[step.id] = node_tag
            self.input_attributes[step.id] = in_tag
            self.output_attributes[step.id] = out_tag
            with dpg.node(label=step.title, tag=node_tag):
                with dpg.node_attribute(tag=in_tag, attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text("INPUTS", color=(255, 255, 255))
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text(step.description, color=(255, 255, 255), wrap=270)
                    dpg.add_spacer(height=3)
                    dpg.add_text("", tag=f"node_state::{step.id}",
                                 color=(255, 235, 120))
                    dpg.add_button(label="Select step", callback=self._select_callback,
                                   user_data=step.id, width=120)
                with dpg.node_attribute(tag=out_tag, attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(step.output, color=(175, 255, 210), wrap=260)

            if step.stage == 0:
                side_offset = 0 if step.id.endswith("left") else 240
                y = 60 + (step.row - 1) * 520 + side_offset
            elif step.row:
                y = 180 + (step.row - 1) * 520
            else:
                y = 960
            dpg.set_item_pos(node_tag, (stage_x[step.stage], y))

    def _create_links(self) -> None:
        dpg = self.dpg
        producer = {step.output: step.id for step in self.graph.steps}
        for step in self.graph.steps:
            for part in step.inputs:
                source_id = producer.get(part)
                if source_id:
                    dpg.add_node_link(self.output_attributes[source_id],
                                      self.input_attributes[step.id],
                                      parent="task_node_editor")

    def _create_side_panel(self) -> None:
        dpg = self.dpg
        dpg.add_text("Selected step", color=(225, 232, 240))
        dpg.add_separator()
        dpg.add_text("", tag="selection_notice", wrap=400, show=False)
        dpg.add_text("Select a graph node to inspect it.", tag="step_details", wrap=400)
        dpg.add_spacer(height=5)
        dpg.add_button(label="Mark selected step complete", tag="complete_button",
                       callback=self._complete_selected, enabled=False, width=-1)
        dpg.add_button(label="Recommend next step", tag="recommend_button",
                       callback=self._recommend_callback, width=-1)
        dpg.add_button(label="Reset all progress", callback=self._reset_callback, width=-1)
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Active parts and assemblies")
        dpg.add_text("Top-level entries are active. Expand an assembly to see its composition.",
                     color=(165, 180, 195), wrap=400)
        with dpg.child_window(tag="active_parts_panel", height=245,
                              horizontal_scrollbar=True):
            dpg.add_group(tag="active_parts_tree")
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Part / step terminal")
        dpg.add_input_text(tag="terminal_output", multiline=True, readonly=True,
                           width=-1, height=165)
        dpg.add_input_text(tag="command_input", hint="Enter a part name or command...",
                           width=-1, on_enter=True, callback=self._command_callback)

        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Voice Input  (Parakeet ASR)", color=(225, 232, 240))
        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=(160, 170, 185))
            dpg.add_text("—", tag="voice_status", color=(180, 180, 180, 255))
        with dpg.group(horizontal=True):
            dpg.add_text("Timer: ", color=(160, 170, 185))
            dpg.add_text("", tag="voice_timer", color=(100, 220, 255, 255))
        dpg.add_text("Level:", color=(160, 170, 185))
        dpg.add_progress_bar(tag="voice_rms", default_value=0.0, width=-1)
        dpg.add_spacer(height=4)
        dpg.add_checkbox(label="Route transcripts to VLM", tag="voice_to_vlm",
                         default_value=False)
        dpg.add_spacer(height=4)
        dpg.add_text("Transcripts:", color=(160, 170, 185))
        dpg.add_input_text(tag="voice_transcripts", multiline=True, readonly=True,
                           width=-1, height=120,
                           default_value='Say "hey robot" to start...')

    def _notify_vlm(self, event_label: str) -> None:
        if self._vlm is not None:
            self._vlm.notify_graph_event(event_label, self.graph.state_summary())

    def _recommend_callback(self) -> None:
        step = self.graph.recommend_next_step()
        if step is None:
            self.log("[Recommend] No READY steps — assembly may be complete or stalled.")
            return
        self.recommended_id = step.id
        self.refresh()
        self.log(f"[Recommend] Highlighted: [{step.id}]  {step.title}")
        if self._vlm is not None:
            q = (
                f"The system recommends the next step as:\n"
                f"  [{step.id}] {step.title}\n\n"
                f"Briefly explain what this step involves and why it is the right"
                f" choice now according to the row-by-row assembly policy."
            )
            sent = self._vlm.submit_question(q)
            if not sent:
                self.log("[Recommend] VLM busy — explanation skipped.")

    def _select_callback(self, _sender, _app_data, user_data) -> None:
        self.selected_id = user_data
        self.refresh()
        coords = TaskGraph.control_coords_for(user_data)
        if coords is not None:
            self._send_select({"event": "select", "row": coords[0], "stage": coords[1],
                               "step": user_data})
        if self._vlm is not None:
            step = self.graph.by_id[user_data]
            state = self.graph.state(step)
            missing = self.graph.missing(step)
            focused = (
                f"[{step.id}] {step.title}\n"
                f"State: {state.upper()}\n"
                f"Description: {step.description}\n"
                f"Inputs: {', '.join(step.inputs)}\n"
                f"Produces: {step.output}"
            )
            if missing:
                focused += f"\nBlocked by: {', '.join(missing)}"
            self._vlm.set_focused_step(focused)

    def _complete_selected(self) -> None:
        if not self.selected_id:
            return
        step = self.graph.by_id[self.selected_id]
        if self.graph.state(step) == "complete":
            ok, message = self.graph.undo(step)
            if ok:
                self._notify_vlm(f"UNDO: {step.id}")
        else:
            ok, message = self.graph.complete(step)
            if ok:
                self._notify_vlm(f"COMPLETE: {step.id} -> {step.output}")
        self.log(message)
        self.refresh()

    def _reset_callback(self) -> None:
        self.graph.reset()
        self.selected_id   = None
        self.recommended_id = None
        self.log("All assembly progress was reset.")
        self.refresh()
        self._send_select({"event": "clear"})
        self._notify_vlm("RESET: all progress cleared")

    def _command_callback(self, sender, app_data) -> None:
        raw = (app_data or self.dpg.get_value(sender) or "").strip()
        self.dpg.set_value(sender, "")
        if raw:
            self.execute_command(raw)

    def log(self, message: str) -> None:
        self.log_lines.append(message)
        self.log_lines = self.log_lines[-200:]
        self.dpg.set_value("terminal_output", "\n".join(self.log_lines))

    def refresh(self) -> None:
        dpg = self.dpg
        # Auto-clear recommendation once the recommended step is completed.
        if (self.recommended_id
                and self.graph.state(self.graph.by_id[self.recommended_id]) == "complete"):
            self.recommended_id = None
        for step in self.graph.steps:
            state  = self.graph.state(step)
            active = step.id in self.active_ids or step.id == self.selected_id
            rec    = step.id == self.recommended_id and not active
            if active:
                theme = self.themes["active"]
            elif rec:
                theme = self.themes["recommended"]
            else:
                theme = self.themes[state]
            dpg.bind_item_theme(self.node_tags[step.id], theme)
            label = f"{step.title}  [{state.upper()}]"
            if active:
                label += "  (open)"
            elif rec:
                label += "  [RECOMMENDED]"
            dpg.configure_item(self.node_tags[step.id], label=label)
            dpg.set_value(f"node_state::{step.id}", f"State: {state.upper()}")

        self._refresh_active_parts_tree()

        if not self.selected_id:
            dpg.configure_item("selection_notice", show=False)
            dpg.set_value("step_details", "Select a graph node to see its inputs, output, and readiness.")
            dpg.configure_item("complete_button", label="Mark selected step complete", enabled=False)
            return
        step = self.graph.by_id[self.selected_id]
        state = self.graph.state(step)
        missing = self.graph.missing(step)
        detail = (f"{step.id}\n\n{step.description}\n\n"
                  f"Inputs: {', '.join(step.inputs)}\n\n"
                  f"Produces: {step.output}\n\n"
                  f"State: {state.upper()}")
        if missing and step.id not in self.graph.completed:
            detail += "\n\nBlocked by: " + ", ".join(missing)
        dpg.set_value("step_details", detail)
        if state == "blocked":
            notice = ("WARNING: This step should not be performed yet.\n"
                      "Complete or provide these prerequisites first: " + ", ".join(missing))
            dpg.set_value("selection_notice", notice)
            dpg.configure_item("selection_notice", show=True, color=(255, 85, 85))
        elif state == "complete":
            if self.graph.is_frontier(step):
                dpg.set_value("selection_notice",
                              f"COMPLETED: Created {step.output}\n"
                              "This step is on the completed frontier and can be undone.")
                dpg.configure_item("selection_notice", show=True, color=(90, 225, 135))
            else:
                consumers = self.graph.completed_consumers(step)
                blocked_by = ", ".join(item.id for item in consumers) or "a dependent step"
                dpg.set_value("selection_notice",
                              f"COMPLETED: Created {step.output}\n"
                              f"Undo completed dependent step(s) first: {blocked_by}.")
                dpg.configure_item("selection_notice", show=True, color=(255, 195, 90))
        else:
            dpg.set_value("selection_notice", "READY: This step can be performed now.")
            dpg.configure_item("selection_notice", show=True, color=(100, 185, 255))
        if state == "complete" and self.graph.is_frontier(step):
            dpg.configure_item("complete_button", label="Undo selected frontier step", enabled=True)
        elif state == "complete":
            dpg.configure_item("complete_button", label="Undo dependent step(s) first", enabled=False)
        else:
            dpg.configure_item("complete_button", label="Mark selected step complete",
                               enabled=self.graph.is_ready(step))

    def _refresh_active_parts_tree(self) -> None:
        dpg = self.dpg
        dpg.delete_item("active_parts_tree", children_only=True)
        grouped: dict[str, list[str]] = {
            "ROW 1": [], "ROW 2": [], "ROW 3": [], "ROW 4": [], "BASE / FINAL": []
        }
        for part in sorted(self.graph.active_parts):
            group = "BASE / FINAL"
            for row in range(1, 5):
                if f"_ROW{row}" in part:
                    group = f"ROW {row}"
                    break
            grouped[group].append(part)

        for group_name, parts in grouped.items():
            if not parts:
                continue
            group_node = dpg.add_tree_node(
                label=f"{group_name} ({len(parts)} active)", parent="active_parts_tree",
                default_open=True, span_full_width=True)
            for part in parts:
                self._add_part_branch(part, group_node)

    def _add_part_branch(self, part: str, parent) -> None:
        dpg = self.dpg
        producer = self.graph.producer_for(part)
        if producer is None:
            dpg.add_tree_node(label=part, parent=parent, leaf=True, bullet=True,
                              span_text_width=True)
            return
        assembly_node = dpg.add_tree_node(
            label=f"[ASSEMBLY] {part}", parent=parent, default_open=False,
            span_full_width=True)
        dpg.bind_item_theme(assembly_node, "assembly_tree_theme")
        for component in producer.inputs:
            self._add_part_branch(component, assembly_node)

    def execute_command(self, raw: str) -> None:
        self.log("> " + raw)
        command, _, argument = raw.partition(" ")
        command_lower = command.lower()
        if command_lower == "help":
            self.log("Commands: <part name>, part <name>, step <id/title>, complete <id/title>, "
                     "parts, status, frontier, missing, undo [step], add <part name>, reset, help")
        elif command_lower in {"parts", "inventory", "list"}:
            self.log(self.graph.active_parts_text())
        elif command_lower in {"frontier", "undoable"}:
            frontier = [step.id for step in self.graph.frontier_steps()]
            self.log("UNDOABLE FRONTIER: " + (", ".join(frontier) or "none"))
        elif command_lower in {"part", "find"}:
            self.log(self.graph.trace_part(argument))
        elif command_lower == "step":
            self._inspect_steps(argument)
        elif command_lower == "complete":
            matches = self.graph.find_steps(argument)
            if len(matches) == 1:
                self.selected_id = matches[0].id
                ok, message = self.graph.complete(matches[0])
                self.log(message)
                self.refresh()
                if ok:
                    self._notify_vlm(f"COMPLETE: {matches[0].id} -> {matches[0].output}")
            elif matches:
                self.log("Ambiguous step. Matches: " + ", ".join(s.id for s in matches))
            else:
                self.log(f"WARNING: no step matches '{argument}'.")
        elif command_lower == "status":
            ready = [s.id for s in self.graph.steps if self.graph.is_ready(s)]
            frontier = [s.id for s in self.graph.frontier_steps()]
            self.log(f"Completed {len(self.graph.completed)}/{len(self.graph.steps)}. "
                     f"Ready: {', '.join(ready) if ready else 'none'}\n"
                     f"Undoable frontier: {', '.join(frontier) if frontier else 'none'}")
        elif command_lower == "missing":
            produced = {other.output for other in self.graph.steps}
            missing_raw = sorted({p for s in self.graph.steps for p in self.graph.missing(s)
                                  if p not in produced})
            self.log("Missing raw inventory: " + (", ".join(missing_raw) or "none"))
        elif command_lower == "undo":
            requested = None
            if argument:
                matches = self.graph.find_steps(argument)
                if len(matches) != 1:
                    self.log("WARNING: specify exactly one completed step to undo.")
                    return
                requested = matches[0]
                self.selected_id = requested.id
            elif self.graph.completed:
                self.selected_id = self.graph.completed[-1]
            ok, message = self.graph.undo(requested)
            self.log(message)
            self.refresh()
            if ok:
                self._notify_vlm(f"UNDO: {requested.id if requested else 'last step'}")
        elif command_lower == "add":
            if not argument:
                self.log("Usage: add <part name>")
            else:
                self.log(self.graph.add_part(argument.strip()))
                self.refresh()
                self._notify_vlm(f"ADD: {argument.strip()} added to inventory")
        elif command_lower == "reset":
            self._reset_callback()
        else:
            self.log(self.graph.trace_part(raw))

    def _inspect_steps(self, query: str) -> None:
        matches = self.graph.find_steps(query)
        if not matches:
            self.log(f"WARNING: no step matches '{query}'.")
            return
        if len(matches) == 1:
            self.selected_id = matches[0].id
            self.refresh()
        self.log("; ".join(
            f"{s.id} [{self.graph.state(s)}]"
            + (f" blocked by {', '.join(self.graph.missing(s))}" if self.graph.missing(s) else "")
            for s in matches[:12]))


def run_self_test() -> None:
    errors = TaskGraph.validate()
    assert not errors, errors
    graph = TaskGraph()
    assert graph.is_ready(graph.by_id["r1_bearing_left"])
    ok, _ = graph.complete(graph.by_id["r1_bearing_left"])
    assert ok and "BEARING_STAND_ROW1_LEFT_ASSEMBLY" in graph.active_parts
    assert "BEARING_ROW1_LEFT" not in graph.active_parts
    assert "STAND_ROW1_LEFT" not in graph.active_parts
    ok, _ = graph.complete(graph.by_id["r1_bearing_right"])
    assert ok
    # Both independent bearing steps are frontier nodes, so the older one can
    # be undone even though it was not the most recently completed step.
    ok, _ = graph.undo(graph.by_id["r1_bearing_left"])
    assert ok and "BEARING_ROW1_LEFT" in graph.active_parts
    ok, _ = graph.undo(graph.by_id["r1_bearing_right"])
    assert ok and "BEARING_ROW1_RIGHT" in graph.active_parts
    graph.complete(graph.by_id["r1_bearing_left"])
    graph.complete(graph.by_id["r1_fasten_first_stand"])
    ok, warning = graph.undo(graph.by_id["r1_bearing_left"])
    assert not ok and "not on the completed frontier" in warning
    ok, _ = graph.undo(graph.by_id["r1_fasten_first_stand"])
    assert ok
    ok, warning = graph.complete(graph.by_id["r1_insert_rod_and_fit_second"])
    assert not ok and warning.startswith("WARNING")
    # Complete every currently-ready step until the graph reaches its fixed point.
    while True:
        ready = [s for s in graph.steps if graph.is_ready(s)]
        if not ready:
            break
        for step in ready:
            graph.complete(step)
    assert "COMPLETED_GEARBOX_ASSEMBLY" in graph.active_parts
    assert len(graph.completed) == len(graph.steps)
    print(f"Self-test passed: {len(graph.steps)} steps; final part COMPLETED_GEARBOX_ASSEMBLY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="test the task model without opening the GUI")
    parser.add_argument("--no-live", action="store_true",
                        help="Disable the live controller link (open the viewer standalone).")
    parser.add_argument("--voice-device", default="bluez_source.50_C2_ED_43_95_C8.handsfree_head_unit",
                        help="PulseAudio source name for voice input (pass empty string to disable).")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice input.")
    parser.add_argument("--wake-word", default="hey robot",
                        help="Wake word to activate transcription (default: 'hey robot').")
    parser.add_argument("--vlm-model", default=None,
                        help="Enable VLM assistant with this model name "
                             "(default when enabled: Qwen/Qwen2.5-VL-3B-Instruct). Omit to disable.")
    parser.add_argument("--with-controller", action="store_true",
                        help="Also run gearbox_control.py in-process (one launch = viewer + "
                             "controller). The controller drives Unity and mirrors here over the "
                             "live link; incompatible with --no-live and with the controller's "
                             "--open-3d mode (which needs the main thread).")
    parser.add_argument("--unity-ip", default=gearbox_control._DEFAULT_IP,
                        help=f"With --with-controller: Unity host (default: {gearbox_control._DEFAULT_IP}).")
    parser.add_argument("--cmd-port", type=int, default=gearbox_control.DEFAULT_CMD_PORT,
                        help=f"With --with-controller: port this script PUBs commands on to Unity "
                             f"(show_row, stage, recolor, reset, …). "
                             f"OUT: this script -> Unity. (default: {gearbox_control.DEFAULT_CMD_PORT})")
    parser.add_argument("--click-port", type=int, default=gearbox_control.DEFAULT_CLICK_PORT,
                        help=f"With --with-controller: port this script SUBs on to receive part-click "
                             f"events from Unity (part name + event type). "
                             f"IN: Unity -> this script. (default: {gearbox_control.DEFAULT_CLICK_PORT})")
    parser.add_argument("--no-highlight", action="store_true",
                        help="With --with-controller: disable the controller's pegboard tool highlighting.")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    live_port    = None if args.no_live else _DEFAULT_CTRL_EVENTS_IN_PORT
    select_port  = _DEFAULT_STEP_OUT_PORT
    voice_device = None if args.no_voice else args.voice_device

    # Optionally co-launch gearbox_control.py in-process. DearPyGui owns the main thread, so the
    # controller's click listener (+ optional REPL) run on daemon threads behind it; the two keep
    # talking over the localhost live link (task-graph port). Both files still run standalone.
    controller = None
    if args.with_controller:
        if args.no_live:
            parser.error("--with-controller needs the live link; do not pass --no-live.")
        controller = gearbox_control.GearboxController(
            args.unity_ip, args.cmd_port, args.click_port,
            _LOCALHOST, _DEFAULT_CTRL_EVENTS_IN_PORT, no_highlight=args.no_highlight)
        threading.Thread(target=controller.run_click_loop, daemon=True).start()
        print(f"[Controller] in-process — cmd -> {args.unity_ip}:{args.cmd_port}, "
              f"clicks <- {args.unity_ip}:{args.click_port}, "
              f"mirror -> {_LOCALHOST}:{_DEFAULT_CTRL_EVENTS_IN_PORT}")

    try:
        DearPyGuiTaskGraphApp().run(
            live_port,
            voice_device=voice_device,
            wake_word=args.wake_word,
            vlm_model=args.vlm_model,
            select_port=select_port,
        )
    finally:
        if controller is not None:
            controller.stop()
            controller.close()


if __name__ == "__main__":
    main()
