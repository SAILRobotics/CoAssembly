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

import dearpygui.dearpygui as dpg

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
        1: "Attach the single large gear and secure it with the pins.",
        2: "Attach the small and medium gears and secure both with their pins.",
        3: "Attach the small and medium gears and secure both with their pins.",
        4: "Attach the single large gear and secure it with the pin.",
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

        # print (len(self.by_id)) #26

        self.initial_parts = set(available_parts)
        self.active_parts = set(self.initial_parts)
        self.completed: list[str] = []

        # In-memory mapping of each consumed input part to the assembly produced
        # from it: {input_part: output_assembly}. For example, completing a step
        # that transforms BEARING + ROD into BEARING_ROD_ASSEMBLY records:
        # {
        #     "BEARING": "BEARING_ROD_ASSEMBLY",
        #     "ROD": "BEARING_ROD_ASSEMBLY",
        # }
        # If that assembly is consumed later, another entry creates a chain such
        # as BEARING -> BEARING_ROD_ASSEMBLY -> GEAR_ROW_ASSEMBLY. trace_part()
        # follows this chain to find what an original part has currently become.
        # complete() adds entries, undo() removes the entries it reverses, and
        # reset() clears the dictionary. This history is not persisted to a file
        # or database, so it lasts only for the lifetime of this TaskGraph object.
        self.transform_history: dict[str, str] = {}
 
    def reset(self) -> None:
        self.active_parts = set(self.initial_parts)
        self.completed.clear()
        self.transform_history.clear()

    def missing(self, step: Step) -> list[str]:
        #combines the two collections into one tuple. inputs are consumed by the step. context must be present but is not consumed.
        return [part for part in (*step.inputs, *step.context) 
                if part not in self.active_parts]

    def is_ready(self, step: Step) -> bool:
        # A step is ready when: It has not already been completed. Its missing-parts list is empty.
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
        """A completed step is frontier-safe when no completed step consumed its output.
        A “frontier” step is a completed step whose output still exists in the inventory.
        Suppose:
        Step A: X + Y → XY
        Step B: XY + Z → XYZ
        After Step B, XY no longer exists because it was consumed. 
        Therefore, Step A cannot be undone until Step B is undone."""
        return step.id in self.completed and step.output in self.active_parts

    def completed_consumers(self, step: Step) -> list[Step]:
        #This method finds all completed steps that used the given step’s output as an input.
        return [candidate for candidate in self.steps
                if candidate.id in self.completed and step.output in candidate.inputs]

    def frontier_steps(self) -> list[Step]:
        return [self.by_id[step_id] for step_id in self.completed
                if self.is_frontier(self.by_id[step_id])]

    def undo(self, step: Step | None = None) -> tuple[bool, str]:
        """Reverse a completed assembly step when doing so is dependency-safe.

        If ``step`` is omitted, the most recently completed step is selected.
        The target must be on the completed frontier: it must be complete and
        its output must still exist in ``active_parts``. If a later completed
        step has consumed that output, that dependent step must be undone first.

        A successful undo reverses the inventory changes made by ``complete``:
        it removes the assembled output, restores the consumed input parts,
        deletes this step's transformation-history entries, and removes the
        step ID from ``completed``. Context parts are not restored because
        completing a step never consumes them.

        Args:
            step: The completed step to undo. When ``None``, use the most
                recently completed step.

        Returns:
            A ``(success, message)`` tuple. ``success`` is ``True`` only when
            the step was undone; ``message`` describes the result or explains
            why the request was rejected.
        """
        # Nothing can be undone until at least one step has been completed.
        if not self.completed:
            return False, "WARNING: there is no completed step to undo."

        # Use the requested step, or default to the latest completed step.
        target = step or self.by_id[self.completed[-1]]

        # A step that has not happened cannot be reversed.
        if target.id not in self.completed:
            return False, f"WARNING: {target.id} is not complete, so it cannot be undone."

        # Do not invalidate dependent state. If another completed step consumed
        # this output, the consumer must be undone before this producer.
        if not self.is_frontier(target):
            consumers = self.completed_consumers(target)
            blocked_by = ", ".join(consumer.id for consumer in consumers) or "a dependent assembly"
            return False, (f"WARNING: {target.id} is not on the completed frontier. "
                           f"Undo its completed dependent step(s) first: {blocked_by}.")

        # Reverse the inventory transformation: remove the assembled result and
        # return all inputs that were consumed while completing the step.
        self.active_parts.remove(target.output)
        self.active_parts.update(target.inputs)

        # Delete only history entries created by this particular transformation.
        for part in target.inputs:
            if self.transform_history.get(part) == target.output:
                del self.transform_history[part]

        # Mark the target step as no longer completed.
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
        """Describe a part's current state and the next step that can use it.

        The lookup is case-insensitive. The requested name is first searched
        among active inventory items and then among consumed items recorded in
        ``transform_history``. If the part was consumed, its transformation
        chain is followed until the currently active assembly is found.

        The method then examines incomplete steps that directly use the active
        part or assembly as either an input or a context requirement. It reports
        ready steps when available; otherwise it explains which prerequisites
        are blocking the relevant steps.

        When there is no exact active or consumed match, the method searches all
        known input, output, and context names for partial matches. At most 12
        suggestions are returned.

        Args:
            part: Part or assembly name to trace.

        Returns:
            A human-readable status message containing the current assembly,
            possible name matches, a ready next step, or blocking details.
        """
        # Look for an exact name in both the current inventory and the history
        # of parts that have already been consumed. Comparisons ignore case,
        # while the stored spelling is preserved for display and later lookup.
        exact = next((p for p in self.active_parts if p.lower() == part.lower()), None)

        # Example: after completing
        #   BEARING_ROW1_LEFT + STAND_ROW1_LEFT
        #       -> BEARING_STAND_ROW1_LEFT_ASSEMBLY
        # transform_history contains:
        # {
        #     "BEARING_ROW1_LEFT": "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
        #     "STAND_ROW1_LEFT": "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
        # }
        # This expression searches those dictionary keys case-insensitively and
        # returns the stored key spelling, or None when the part was not consumed.
        consumed = next((p for p in self.transform_history if p.lower() == part.lower()), None) 
        #Then the code follows the chain starting from transform_history["BEARING_ROW1_LEFT"] to find what it eventually became
        canonical = exact or consumed or part

        # A consumed part may have passed through several transformations.
        # Follow that chain to identify its latest assembly, for example:
        # BEARING -> BEARING_ROD_ASSEMBLY -> GEAR_ROW_ASSEMBLY.
        if consumed:
            current = self.transform_history[consumed]
            while current in self.transform_history:
                current = self.transform_history[current]
            prefix = f"{consumed} has already been transformed; its current assembly is {current}. "
            canonical = current

        # An exact active match can be used directly when searching for the next
        # applicable assembly step.
        elif exact:
            prefix = f"Found active part {exact}. "

        # If no exact match exists, offer partial matches from every part and
        # assembly name appearing anywhere in the graph.
        else:
            known = {p for s in self.steps for p in (*s.inputs, s.output, *s.context)}
            candidates = sorted(p for p in known if part.lower() in p.lower())
            if candidates:
                return "Possible matches: " + ", ".join(candidates[:12])
            return f"WARNING: no part or assembly named '{part}' exists in this task graph."

        # Find incomplete steps that directly require the current part or
        # assembly. Context requirements count even though they are not consumed.
        candidates = [s for s in self.steps
                      if s.id not in self.completed and canonical in (*s.inputs, *s.context)]
        if not candidates:
            return prefix + "No remaining step directly uses it."

        # Prefer reporting usable steps. Multiple steps can be ready at once
        # because the task graph may contain independent assembly branches.
        ready = [s for s in candidates if self.is_ready(s)]
        if ready:
            return prefix + "Next READY step: " + "; ".join(f"{s.id} ({s.title})" for s in ready)

        # Relevant steps exist, but none are ready. Explain the missing parts for
        # each step so the caller knows which prerequisites must happen first.
        details = []
        for step in candidates:
            # Example: if step.id is "r1_gear_rod" and missing(step) returns
            # ["GEAR_ROW1", "ROD_ROW1"], this appends:
            # "r1_gear_rod is blocked by GEAR_ROW1, ROD_ROW1"
            # join() converts the list of missing-part names into one
            # comma-separated string for the user-facing warning.
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

        # Collect every assembly that can be created by a graph step. Each step
        # input must come from one of two valid sources:
        #   1. a raw part included in the graph's initial inventory, or
        #   2. an assembly produced as the output of another graph step.
        # If an input appears in neither collection, its name is probably
        # misspelled or the step that should produce it has not been defined.
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
        completed = self.completed
        ready = [s for s in self.steps if self.is_ready(s)]
        assemblies = sorted(p for p in self.active_parts if p.endswith("_ASSEMBLY"))

        lines = [f"Progress: {len(completed)} / {len(self.steps)} steps completed"]

        if completed:
            lines.append("\nCompleted steps (most recent last):")
            # Include every completed step so the VLM has the full progress
            # history instead of only a truncated recent subset.
            for sid in completed:
                step = self.by_id[sid]
                lines.append(f"  - [{step.id}] {step.title}  ->  {step.output}")

        if ready:
            lines.append(
                "\nREADY steps — ALL of these are currently unlocked and INDEPENDENT."
                "\nThey have NO ordering requirement among themselves; the user may perform"
                " them in any order they prefer:"
            )
            # Include every ready step so no currently valid option is hidden
            # from the VLM.
            for step in ready:
                lines.append(f"  - [{step.id}] {step.title}")
        elif len(completed) == len(self.steps):
            lines.append("\nNo READY steps: the assembly is complete.")
        else:
            lines.append(
                "\nNo READY steps: assembly is stalled because required parts "
                "or prerequisite assemblies are missing."
            )

        if assemblies:
            lines.append("\nActive assemblies in inventory:")
            for a in assemblies:
                lines.append(f"  - {a}")

        lines.append(
            "\nIMPORTANT: Any task step not listed as COMPLETED or READY is BLOCKED."
            " Recommend only READY steps. Every listed READY step is valid and may"
            " be performed because all of its prerequisites are satisfied. When"
            " choosing among multiple READY steps, follow the user's recommendation"
            " preference from the task description: prefer Row 1, then Row 2, then"
            " Row 3, then Row 4, and choose the lowest stage within that row."
            " Recommend the final gearbox step only when it is READY and no row"
            " assembly step remains READY. This preference selects among valid"
            " options; it is not an additional dependency or ordering constraint."
            " Do NOT infer dependencies from descriptions or display order."
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
        "complete": (46, 157, 96, 255),   # green
        "ready":    (39, 132, 216, 255),  # blue
        "blocked":  (209, 138, 39, 255),  # orange
    }

    # Status → (RGBA color, display label)
    _VOICE_STATUS_STYLE: dict[str, tuple[tuple, str]] = {
        "loading":     ((180, 180, 180, 255), "Loading model..."),   # grey
        "idle":        ((255, 220, 50,  255), "Idle"),               # yellow
        "speech":      ((50,  200, 255, 255), "Speech detected..."), # cyan
        "queued":      ((200, 160, 50,  255), "Queued..."),          # amber
        "transcribing":((255, 165, 50,  255), "Transcribing..."),    # orange
        "listening":   ((50,  220, 80,  255), "Listening"),          # green
        "error":       ((255, 80,  80,  255), "Error"),              # red
    }

    # ── Initialization and application lifecycle ─────────────────────────────

    def __init__(self) -> None:
        """Initialize application state before any GUI widgets are created."""
        # Keep the Dear PyGui module available to every instance method.
        self.dpg = dpg
        # Create the assembly dependency graph and initial inventory.
        self.graph = TaskGraph()
        # Store the currently selected task-step ID, or None when unselected.
        self.selected_id: str | None = None
        # Map task-step IDs to their Dear PyGui node tags.
        self.node_tags: dict[str, str] = {}
        # Map task-step IDs to their node input-pin tags.
        self.input_attributes: dict[str, str] = {}
        # Map task-step IDs to their node output-pin tags.
        self.output_attributes: dict[str, str] = {}
        # Map state names such as "ready" to Dear PyGui theme tags.
        self.themes: dict[str, str] = {}

        # Live-mirror state
        # Track steps currently opened by the external controller.
        self.active_ids: set[str] = set()
        # Track the step highlighted by the recommendation policy.
        self.recommended_id: str | None = None
        # Transfer ZMQ messages safely from the listener thread to the GUI thread.
        self._live_queue: "queue.Queue[dict]" = queue.Queue()
        # Tell the listener thread whether it should continue polling.
        self._live_running = False
        # Retain the background ZMQ listener thread after startup.
        self._live_thread: threading.Thread | None = None
        # Retain the ZeroMQ subscriber that receives controller events.
        self._live_sub = None
        # Retain the ZeroMQ publisher that sends selected-step events.
        self._select_pub = None   # PUB -> gearbox_control.py --open-3d viewer (selected step)

        # Optional in-process controller (set by main() when --with-controller is used).
        # Gives direct access to send commands to Unity without going through ZMQ.
        self.controller: "gearbox_control.GearboxController | None" = None

        # run() sets this to a SpeechListener when voice input is enabled.
        self._speech = None   # SpeechListener, set in run() if enabled
        # run() sets this to a VLMAssistant when a model is enabled.
        self._vlm    = None   # VLMAssistant, set in run() if enabled

    def build(self) -> None:
        """Construct, configure, and reveal the Dear PyGui interface."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Initialize Dear PyGui's internal item and resource registries.
        dpg.create_context()
        # Create the operating-system window that contains the application.
        dpg.create_viewport(title="Gearbox Assembly Task Graph", width=3420, height=1400,
                            min_width=1400, min_height=800)
        # Register themes before any GUI item tries to use them.
        self._create_themes()

        # Create the root window whose contents fill the viewport.
        with dpg.window(tag="primary_window", label="Gearbox Assembly Task Graph"):
            # Add the application heading.
            dpg.add_text("Gearbox Assembly Dependency Graph", color=(225, 232, 240),
                         tag="main_title")
            # Add a legend explaining the node-state colors.
            dpg.add_text("Blue = ready     Orange = blocked     Green = complete"
                         "     Purple = selected / active     Gold = recommended",
                         color=(170, 180, 195))
            # Use a resizable table as a three-column page layout.
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                # Give the dependency graph the largest proportional width.
                dpg.add_table_column(init_width_or_weight=2.8)
                # Give the controls panel a narrower proportional width.
                dpg.add_table_column(init_width_or_weight=1.0)
                # Give the VLM panel the remaining proportional width.
                dpg.add_table_column(init_width_or_weight=1.1)
                # Put the graph, controls, and VLM into one horizontal row.
                with dpg.table_row():
                    # First cell: dependency-graph canvas.
                    with dpg.table_cell():
                        # Fill the cell and allow a wide graph to scroll.
                        with dpg.child_window(height=-1, horizontal_scrollbar=True):
                            # Create the interactive graph canvas and minimap.
                            with dpg.node_editor(tag="task_node_editor", minimap=True,
                                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                                # Add one visual node per assembly step.
                                self._create_nodes()
                                # Connect producer nodes to consumer nodes.
                                self._create_links()
                    # Second cell: task details and application controls.
                    with dpg.table_cell():
                        # Fill the cell vertically with its own panel.
                        with dpg.child_window(height=-1):
                            self._create_side_panel()
                    # Third cell: VLM assistant.
                    with dpg.table_cell():
                        # Fill the cell vertically with its own panel.
                        with dpg.child_window(height=-1):
                            # Build the assistant UI when a VLM is enabled.
                            if self._vlm is not None:
                                self._vlm.build_inline()
                            # Otherwise explain how to enable the assistant.
                            else:
                                dpg.add_text("VLM Assistant",
                                             color=(225, 232, 240))
                                dpg.add_separator()
                                dpg.add_text(
                                    "Not enabled.\n\nRun with:\n"
                                    "  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct",
                                    color=(140, 155, 175))

        # Make the root window automatically fill the viewport.
        dpg.set_primary_window("primary_window", True)
        # Apply the previously created link and pin theme to the node editor.
        dpg.bind_item_theme("task_node_editor", "node_editor_theme")
        # Render current graph state into node colors, labels, and panels.
        self.refresh()
        # Finalize Dear PyGui after declaring the initial widgets.
        dpg.setup_dearpygui()
        # Make the operating-system window visible.
        dpg.show_viewport()

    def run(self, live_port: int | None = None,
            voice_device: str | None = None,
            wake_word: str = "hey robot",
            vlm_model: str | None = None,
            task_description_path: str | None = None,
            select_port: int | None = None) -> None:
        """Start optional services, run the GUI loop, and clean up on exit."""
        # Pre-import transformers on the main thread before any worker threads start.
        # The VLM thread and ASR (NeMo) thread both import from transformers; if they
        # race during the initial import, Python's partially-initialized sys.modules
        # entry causes "cannot import name X from transformers".
        try:
            import transformers as _tf  # noqa: F401
        except Exception:
            # Voice/VLM initialization will report a useful error later if needed.
            pass

        # Create the VLM assistant only when the caller supplies a model.
        if vlm_model is not None:
            # Use the requested prompt file or the repository's default description.
            desc_path = task_description_path or str(
                Path(__file__).parent / "task_description.md")
            # Construct the assistant before build() so its panel can be created.
            self._vlm = VLMAssistant(
                self.dpg,
                desc_path,
                model_name=vlm_model,
            )
            # Report model selection in the operating-system terminal.
            print(f"[VLM] Assistant created: {vlm_model}")
        # Build and show the Dear PyGui interface.
        self.build()
        # Start receiving controller events when a live port was supplied.
        if live_port is not None:
            self.start_live_listener(live_port)
        # Start publishing node selections when an output port was supplied.
        if select_port is not None:
            self.start_select_publisher(select_port)
        # Start speech recognition only when an audio device was supplied.
        if voice_device is not None:
            try:
                # Create and launch the background speech listener.
                self._speech = SpeechListener(device=voice_device, wake_word=wake_word)
                self._speech.start()
                self.log(f"[Voice] Started on device: {voice_device}")
            except Exception as e:
                # Keep the rest of the GUI usable if voice startup fails.
                self.log(f"[Voice] Failed to start: {e}")
        # Enable drag-and-drop images only when the VLM assistant exists.
        if self._vlm is not None:
            try:
                # Route dropped files to the VLM assistant's image handler.
                self.dpg.set_viewport_drop_callback(self._vlm.on_file_drop)
            except Exception:
                pass  # older DPG versions may not support this

        # Use a local reference in the high-frequency render loop.
        dpg = self.dpg
        # Process background events and render until the window closes.
        while dpg.is_dearpygui_running():
            # Apply queued controller messages on the main GUI thread.
            self._drain_live_queue()
            # Apply speech status and transcript events on the main GUI thread.
            self._poll_speech()
            # Display VLM status changes and completed responses.
            if self._vlm is not None:
                self._vlm.tick()
            # Draw one frame and process Dear PyGui interaction.
            dpg.render_dearpygui_frame()
        # Tell the live listener thread to stop polling.
        self._live_running = False
        # Release audio resources if speech recognition was active.
        if self._speech is not None:
            self._speech.close()
        # Stop the model worker if the assistant was active.
        if self._vlm is not None:
            self._vlm.close()
        # Release all Dear PyGui resources.
        dpg.destroy_context()

    # ── ZeroMQ sending, receiving, and live-controller events ────────────────

    def start_live_listener(self, port: int) -> bool:
        """Bind a SUB (receiver binds, per the repo's Python<->Python convention) and drain
        controller events on a background thread. Failures are non-fatal — the GUI still opens."""
        # Import ZeroMQ lazily so the GUI can still run without that dependency.
        try:
            import zmq
        except Exception as e:
            self.log(f"Live link disabled (zmq unavailable: {e}).")
            return False
        # Create and bind the subscriber that receives controller events.
        try:
            # Reuse ZeroMQ's process-wide context.
            ctx = zmq.Context.instance()
            # Create a subscriber socket.
            sub = ctx.socket(zmq.SUB)
            # Subscribe to every topic because messages are plain JSON strings.
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            # Bind the receiver to the configured local port.
            sub.bind(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Live link disabled (could not bind :{port}: {e}).")
            return False
        # Retain the socket for the lifetime of the application.
        self._live_sub = sub
        # Allow the listener loop to begin processing events.
        self._live_running = True

        # Define the work performed by the background listener thread.
        def _loop() -> None:
            # Use a poller so the thread can periodically check its stop flag.
            poller = zmq.Poller()
            # Watch the subscriber for incoming messages.
            poller.register(sub, zmq.POLLIN)
            # Continue until run() begins application shutdown.
            while self._live_running:
                # Wait briefly and retry when no message has arrived.
                if not dict(poller.poll(timeout=200)):
                    continue
                # Drain every message currently waiting on the socket.
                while True:
                    try:
                        # Receive without blocking after the poller reported data.
                        raw = sub.recv_string(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        # Decode JSON and hand it to the thread-safe GUI queue.
                        self._live_queue.put(json.loads(raw))
                    except json.JSONDecodeError:
                        # Ignore malformed external messages.
                        continue

        # Run network polling outside the GUI thread.
        self._live_thread = threading.Thread(target=_loop, daemon=True)
        # Start receiving controller events immediately.
        self._live_thread.start()
        # Report the active endpoint in the application's terminal.
        self.log(f"Live link listening on tcp://{_LOCALHOST}:{port} (controller mirror).")
        return True

    def start_select_publisher(self, port: int) -> bool:
        """Connect a PUB (sender connects, per convention) so pressing 'Select step' can tell an
        (row, stage) consumer — gearbox_control.py --open-3d — which step is selected. Non-fatal."""
        # Import ZeroMQ lazily so this optional feature can fail independently.
        try:
            import zmq
        except Exception as e:
            self.log(f"Step-select link disabled (zmq unavailable: {e}).")
            return False
        # Create and connect the selected-step publisher.
        try:
            # Reuse ZeroMQ's process-wide context.
            ctx = zmq.Context.instance()
            # Create a publisher socket for outgoing JSON events.
            pub = ctx.socket(zmq.PUB)
            # Connect the sender to the external consumer.
            pub.connect(f"tcp://{_LOCALHOST}:{port}")
        except Exception as e:
            self.log(f"Step-select link disabled (could not connect :{port}: {e}).")
            return False
        # Retain the publisher for later selection callbacks.
        self._select_pub = pub
        # Report the outgoing endpoint in the application's terminal.
        self.log(f"Step-select link -> tcp://{_LOCALHOST}:{port} (open3d viewer).")
        return True

    def _send_select(self, msg: dict) -> None:
        """Serialize and publish one selected-step event when connected."""
        # Do nothing when the optional publisher was not started.
        if self._select_pub is None:
            return
        try:
            # Convert the dictionary to JSON and send it as a text message.
            self._select_pub.send_string(json.dumps(msg))
        except Exception:
            # Keep GUI selection usable if the external consumer disconnects.
            pass

    def _drain_live_queue(self) -> None:
        """Apply all controller messages currently waiting for the GUI thread."""
        # Continue until the thread-safe queue becomes empty.
        while True:
            try:
                # Retrieve immediately rather than freezing the GUI while waiting.
                msg = self._live_queue.get_nowait()
            except queue.Empty:
                return
            # Apply the event on the main thread, where GUI updates are safe.
            self._apply_live_event(msg)

    def _apply_live_event(self, msg: dict) -> None:
        """Apply one controller event (main/UI thread): purple overlay + built-in complete/undo."""
        # Read the event type and default missing coordinates to the global step.
        event = msg.get("event")
        row, stage = msg.get("row", 0), msg.get("stage", 0)
        # Translate controller coordinates into task IDs that exist in this graph.
        ids = [i for i in TaskGraph.steps_for_control(row, stage) if i in self.graph.by_id]
        # Highlight the step whose controller view was opened.
        if event == "show":
            self.active_ids = set(ids)
        # Clear controller-driven highlighting when that view closes.
        elif event == "close":
            self.active_ids = set()
        # Restore the graph to its initial state after a controller reset.
        elif event == "reset":
            self.graph.reset()
            self.active_ids     = set()
            self.recommended_id = None
            self.log("Live: controller reset all progress.")
            self._notify_vlm("RESET: controller reset all progress")
        # Complete every task step represented by this controller stage.
        elif event == "complete":
            for sid in ids:
                ok, message = self.graph.complete(self.graph.by_id[sid])
                self.log("Live: " + message)
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"COMPLETE (live): {', '.join(ids)}")
        # Undo mapped steps in reverse dependency order.
        elif event == "uncomplete":
            for sid in reversed(ids):
                ok, message = self.graph.undo(self.graph.by_id[sid])
                self.log("Live: " + message)
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"UNDO (live): {', '.join(ids)}")
        # Ignore external event types this application does not recognize.
        else:
            return
        # Redraw nodes and panels after applying the controller event.
        self.refresh()

    # ── Dear PyGui construction ───────────────────────────────────────────────

    def _create_themes(self) -> None:
        """Create and register every theme used by the task-graph interface."""
        # Use the application's Dear PyGui module through a shorter local name.
        dpg = self.dpg

        # Create a theme for assembly entries in the active-parts tree.
        with dpg.theme(tag="assembly_tree_theme"):
            # Apply the enclosed settings specifically to tree-node widgets.
            with dpg.theme_component(dpg.mvTreeNode):
                # Display assembly tree-node text in green.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (85, 235, 130, 255),
                                    category=dpg.mvThemeCat_Core)

        # Create a theme for the node-editor canvas and its dependency links.
        with dpg.theme(tag="node_editor_theme"):
            # Apply the enclosed settings specifically to node-editor widgets.
            with dpg.theme_component(dpg.mvNodeEditor):
                # Draw dependency links in light blue normally.
                dpg.add_theme_color(dpg.mvNodeCol_Link, (105, 190, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten a dependency link while the pointer is over it.
                dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (180, 225, 255, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Draw a selected dependency link in yellow.
                dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (255, 235, 120, 255),
                                    category=dpg.mvThemeCat_Nodes)
                # Make dependency-link lines three pixels thick.
                dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, 3.0,
                                    category=dpg.mvThemeCat_Nodes)
                # Set the radius of each input/output connection pin to five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_PinCircleRadius, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Build one node theme for each TaskGraph state: complete, ready, and blocked.
        for state, color in self.COLORS.items():
            # Give the theme a stable tag such as "node_theme_complete".
            tag = f"node_theme_{state}"
            # Darken the state's RGB color for the normal node body; append full opacity.
            background = tuple(max(18, int(channel * 0.48)) for channel in color[:3]) + (255,)
            # Use a brighter version of the same color for hover and selection.
            selected_background = tuple(max(25, int(channel * 0.68)) for channel in color[:3]) + (255,)

            # Create the named theme that refresh() will later bind to nodes.
            with dpg.theme(tag=tag):
                # Apply the enclosed settings specifically to node widgets.
                with dpg.theme_component(dpg.mvNode):
                    # Keep text white for contrast against colored backgrounds.
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                        category=dpg.mvThemeCat_Core)
                    # Set the node body's normal background color.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Set the node body's background while hovered.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Set the node body's background while selected.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                        category=dpg.mvThemeCat_Nodes)
                    # Color the border with the state's full-strength color.
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Color the title bar with the state's full-strength color.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Keep the same title-bar color while hovered.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Keep the same title-bar color while selected.
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, color,
                                        category=dpg.mvThemeCat_Nodes)
                    # Round the node's corners by five pixels.
                    dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                        category=dpg.mvThemeCat_Nodes)

            # Save state -> theme-tag so refresh() can select the correct theme.
            self.themes[state] = tag

        # Purple "active" overlay theme — bound to a node while its stage is open in the live
        # controller (a part was clicked in Unity), overriding its state color until the menu closes.
        # Define the full-strength purple used for the outline and title bar.
        active_color = (168, 85, 247, 255)
        # Create a darker purple for the normal node body.
        background = tuple(max(18, int(channel * 0.48)) for channel in active_color[:3]) + (255,)
        # Create a brighter purple for the hovered or selected node body.
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in active_color[:3]) + (255,)

        # Create the active-node theme under a stable Dear PyGui tag.
        with dpg.theme(tag="node_theme_active"):
            # Apply the enclosed settings specifically to node widgets.
            with dpg.theme_component(dpg.mvNode):
                # Keep active-node text white for contrast.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                # Set the active node's normal body to dark purple.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the active node's body while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the active node's body while selected.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the active node's outline in full-strength purple.
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the active node's title bar in full-strength purple.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the purple title bar while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the purple title bar while selected.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, active_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Round the active node's corners by five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Make the active theme available through self.themes["active"].
        self.themes["active"] = "node_theme_active"

        # Gold/yellow "recommended" overlay theme — shown when the system recommends
        # a step as the next to perform.
        # Define the full-strength gold used for the outline and title bar.
        rec_color  = (255, 195, 0, 255)
        # Create a darker gold for the normal node body.
        background = tuple(max(18, int(channel * 0.48)) for channel in rec_color[:3]) + (255,)
        # Create a brighter gold for the hovered or selected node body.
        selected_background = tuple(max(25, int(channel * 0.68)) for channel in rec_color[:3]) + (255,)

        # Create the recommendation theme under a stable Dear PyGui tag.
        with dpg.theme(tag="node_theme_recommended"):
            # Apply the enclosed settings specifically to node widgets.
            with dpg.theme_component(dpg.mvNode):
                # Keep recommended-node text white for contrast.
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                    category=dpg.mvThemeCat_Core)
                # Set the recommended node's normal body to dark gold.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the recommended node's body while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Brighten the recommended node's body while selected.
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, selected_background,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the recommended node's outline in full-strength gold.
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Draw the recommended node's title bar in full-strength gold.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the gold title bar while hovered.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Preserve the gold title bar while selected.
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, rec_color,
                                    category=dpg.mvThemeCat_Nodes)
                # Round the recommended node's corners by five pixels.
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 5.0,
                                    category=dpg.mvThemeCat_Nodes)

        # Make the recommendation theme available through self.themes["recommended"].
        self.themes["recommended"] = "node_theme_recommended"

    def _create_nodes(self) -> None:
        """Create and position one node-editor node for every task step."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Displayed stages 1-3 form a vertical stack. Stage 4 sits beside stage 1,
        # while stages 5-7 form a horizontal line at the height of stage 2.
        # Display stages differ from Step.stage, which represents dependency depth.
        stage_x = {
            1: 20, 2: 20, 3: 20,
            4: 500, 5: 980, 6: 1460,
            7: 1940, 8: 2420,
        }
        # Build a visual node for every Step in the task graph.
        for step in self.graph.steps:
            # Create stable item tags from the unique step ID.
            node_tag = f"node::{step.id}"
            in_tag = f"node_in::{step.id}"
            out_tag = f"node_out::{step.id}"
            # Save tags so refresh() and _create_links() can find these items.
            self.node_tags[step.id] = node_tag
            self.input_attributes[step.id] = in_tag
            self.output_attributes[step.id] = out_tag
            # Create the node using the step title as its visible heading.
            with dpg.node(label=step.title, tag=node_tag):
                # Create the connection pin for dependencies entering this step.
                with dpg.node_attribute(tag=in_tag, attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text("INPUTS", color=(255, 255, 255))
                # Create non-connectable content in the middle of the node.
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    # Display the step instructions with wrapping.
                    dpg.add_text(step.description, color=(255, 255, 255), wrap=270)
                    # Add a small visual gap before the dynamic state label.
                    dpg.add_spacer(height=3)
                    # Reserve a tagged text item that refresh() updates.
                    dpg.add_text("", tag=f"node_state::{step.id}",
                                 color=(255, 235, 120))
                    # Pass the step ID to the callback when this button is pressed.
                    dpg.add_button(label="Select step", callback=self._select_callback,
                                   user_data=step.id, width=120)
                # Create the connection pin representing this step's output.
                with dpg.node_attribute(tag=out_tag, attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(step.output, color=(175, 255, 210), wrap=260)

            # Convert the task ID to the displayed/controller stage used for layout.
            control_coords = TaskGraph.control_coords_for(step.id)
            display_stage = control_coords[1] if control_coords else 8

            # Put stages 1 and 4 on the top level; put stages 2, 5, 6, and 7
            # on the middle level; leave stage 3 on the bottom level.
            if step.row:
                row_top = 60 + (step.row - 1) * 720
                stage_offset = {
                    1: 0, 2: 240, 3: 480,
                    4: 0, 5: 240, 6: 240,
                    7: 240,
                }
                y = row_top + stage_offset.get(display_stage, 0)
            # Center the global finish step between the four row groups.
            else:
                y = 1140

            # Position the node by displayed stage horizontally and row vertically.
            dpg.set_item_pos(node_tag, (stage_x[display_stage], y))

    def _create_links(self) -> None:
        """Draw dependency links from producer steps to consumer steps."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Map each produced assembly name to the step ID that creates it.
        producer = {step.output: step.id for step in self.graph.steps}
        # Inspect every step that may consume another step's output.
        for step in self.graph.steps:
            # Check each input required by the consumer step.
            for part in step.inputs:
                # Find the producer, or None when the input is a raw part.
                source_id = producer.get(part)
                # Raw inputs have no producer node, so only link assembled inputs.
                if source_id:
                    # Connect the producer output pin to the consumer input pin.
                    dpg.add_node_link(self.output_attributes[source_id],
                                      self.input_attributes[step.id],
                                      parent="task_node_editor")

    def _create_side_panel(self) -> None:
        """Create selection, action, inventory, and voice controls."""
        dpg = self.dpg
        dpg.add_text("Selected step", color=(225, 232, 240))
        dpg.add_separator()
        dpg.add_text("", tag="selection_notice", wrap=400, show=False)
        dpg.add_text("Select a graph node to inspect it.", tag="step_details", wrap=400)
        dpg.add_spacer(height=5)
        dpg.add_button(label="Mark selected step complete", tag="complete_button",
                       callback=self._complete_selected, enabled=False, width=-1)
        dpg.add_button(label="Animate in Unity", tag="animate_unity_button",
                       callback=self._animate_unity_callback, enabled=False, width=-1)
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

    # ── Voice input and VLM integration ──────────────────────────────────────

    def _poll_speech(self) -> None:
        """Refresh voice widgets and handle all newly available speech events."""
        if self._speech is None:
            return
        dpg = self.dpg
        events = self._speech.poll()

        # Status label + color
        # loading=model starting; idle=waiting for wake word;
        # speech=someone is talking and audio is being captured right now;
        # queued=captured audio is waiting for ASR; transcribing=ASR is running;
        # listening=wake word was accepted and command mode remains active, even
        # while the user is silent; error=capture or recognition failed.
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

        # Log notable events to the terminal; optionally route transcripts to the VLM.
        route_to_vlm = (self._vlm is not None
                        and dpg.get_value("voice_to_vlm"))
        for kind, payload in events:
            if kind == "wake_word":
                self.log("[Voice] Wake word detected — listening.")
            elif kind == "transcript":
                self.log(f"[Voice] {payload}")
                if route_to_vlm:
                    if not self._vlm.submit_question(payload):
                        self.log("[Voice] VLM busy — transcript skipped.")
            elif kind == "timeout":
                self.log("[Voice] Timed out — back to idle.")
            elif kind == "error":
                self.log(f"[Voice error] {payload}")

    def _notify_vlm(self, event_label: str) -> None:
        """Replace the VLM's stored graph context after a state-changing event."""
        if self._vlm is not None:
            self._vlm.notify_graph_event(event_label, self.graph.state_summary())

    # ── User actions and GUI callbacks ───────────────────────────────────────

    def _recommend_callback(self) -> None:
        """Highlight the preferred ready step and ask the VLM to explain it."""
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

    def _animate_unity_callback(self) -> None:
        """Ask the in-process controller to animate the selected task stage."""
        if self.controller is None or not self.selected_id:
            return
        coords = TaskGraph.control_coords_for(self.selected_id)
        if coords is None:
            self.log("[Animate] This step has no Unity stage mapping.")
            return
        row, stage = coords
        step = self.graph.by_id[self.selected_id]
        if row > 0:
            done_stages = self.controller.sm._completed_stages(row)
            blocked     = not self.controller.sm.unlocked(row, stage)
            checked     = self.controller.sm.done[row][stage]
        else:
            done_stages = [s for s in range(1, 8)]
            blocked     = False
            checked     = self.controller.sm.done8
        self.controller.send({"command": "stage", "row": row, "stage": stage,
                               "done_stages": done_stages,
                               "step_delay": gearbox_control.STEP_DELAY,
                               "slide_seconds": gearbox_control.SLIDE_SECONDS})
        self.controller.send({"command": "ui", "show": True, "row": row,
                               "checked": checked, "blocked": blocked})
        self.log(f"[Animate] Unity → row {row}, stage {stage}  [{step.id}]")

    def _select_callback(self, _sender, _app_data, user_data) -> None:
        """Select a graph step and update external viewers and VLM focus."""
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

    def _sync_sm_from_graph(self) -> None:
        """Rebuild GearboxStateMachine.done from TaskGraph.completed.
        Called after every GUI action that changes TaskGraph so the controller
        always reflects the same state — TaskGraph is the single source of truth."""
        if self.controller is None:
            return
        sm = self.controller.sm
        for row in range(1, 5):
            for stage in range(1, 8):
                step_ids = TaskGraph.steps_for_control(row, stage)
                sm.done[row][stage] = bool(step_ids) and all(
                    sid in self.graph.completed
                    for sid in step_ids if sid in self.graph.by_id)
        sm.done8 = "finish_gearbox" in self.graph.completed

    def _complete_selected(self) -> None:
        """Complete a ready selected step or undo a completed frontier step."""
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
        if ok and self.controller is not None:
            self._sync_sm_from_graph()
            coords = TaskGraph.control_coords_for(self.selected_id)
            if coords is not None and coords[0] > 0:
                row = coords[0]
                self.controller.send({"command": "recolor", "row": row,
                                      "done_stages": self.controller.sm._completed_stages(row)})
        self.log(message)
        self.refresh()

    def _reset_callback(self) -> None:
        """Reset graph, controller, Unity visualization, selection, and VLM state."""
        self.graph.reset()
        self.selected_id    = None
        self.recommended_id = None
        if self.controller is not None:
            # Sync sm state from the now-empty graph (all done = False), then send
            # Unity commands directly — bypassing the ZMQ round-trip so we don't
            # get a spurious "Live: controller reset" log from _apply_live_event.
            self._sync_sm_from_graph()
            self.controller.sm.history.clear()
            self.controller.sm.current_row = self.controller.sm.current_stage = None
            self.controller.send({"command": "reset"})    # clear Unity part colors
            self.controller.send({"command": "show_all"}) # make all rows visible
            self.controller.send({"command": "ui", "show": False})  # hide in-headset UI
        self.log("All assembly progress was reset.")
        self.refresh()
        self._send_select({"event": "clear"})
        self._notify_vlm("RESET: all progress cleared")

    def log(self, message: str) -> None:
        """Write an operational message to the system console."""
        print(message)

    # ── GUI state rendering and inventory tree ───────────────────────────────

    def refresh(self) -> None:
        """Redraw node themes, labels, inventory, and selected-step controls."""
        dpg = self.dpg
        # Auto-clear recommendation once the recommended step is completed.
        if (self.recommended_id
                and self.graph.state(self.graph.by_id[self.recommended_id]) == "complete"):
            self.recommended_id = None
        for step in self.graph.steps:
            state  = self.graph.state(step)
            active = step.id in self.active_ids or step.id == self.selected_id
            rec    = step.id == self.recommended_id and not active

            # Completion color has the highest priority. Keep the node selected
            # so its details remain visible, but render it green immediately
            # after completion instead of leaving the active/selected purple
            # theme in place until another node is clicked.
            if state == "complete":
                theme = self.themes["complete"]
            elif active:
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
            dpg.configure_item("animate_unity_button", enabled=False)
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
        can_animate = (self.controller is not None
                       and TaskGraph.control_coords_for(self.selected_id) is not None)
        dpg.configure_item("animate_unity_button", enabled=can_animate)

    # Example: if active_parts contains
    # {
    #     "BASE_BOARD",
    #     "BEARING_ROW1_RIGHT",
    #     "BEARING_STAND_ROW1_LEFT_ASSEMBLY",
    # }
    # this method rebuilds the visible tree as:
    #
    # ROW 1 (2 active)
    #   - BEARING_ROW1_RIGHT
    #   - [ASSEMBLY] BEARING_STAND_ROW1_LEFT_ASSEMBLY
    #       - BEARING_ROW1_LEFT
    #       - STAND_ROW1_LEFT
    # BASE / FINAL (1 active)
    #   - BASE_BOARD
    #
    # Only active parts appear at the top level. _add_part_branch() recursively
    # shows the consumed components inside an active assembly.
    def _refresh_active_parts_tree(self) -> None:
        """Rebuild the inventory tree from the graph's current active parts."""
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
        """Recursively add a raw part or expandable assembly to the inventory tree."""
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
    app = DearPyGuiTaskGraphApp()
    controller = None
    if args.with_controller:
        if args.no_live:
            parser.error("--with-controller needs the live link; do not pass --no-live.")
        controller = gearbox_control.GearboxController(
            args.unity_ip, args.cmd_port, args.click_port,
            _LOCALHOST, _DEFAULT_CTRL_EVENTS_IN_PORT, no_highlight=args.no_highlight)
        app.controller = controller
        threading.Thread(target=controller.run_click_loop, daemon=True).start()
        print(f"[Controller] in-process — cmd -> {args.unity_ip}:{args.cmd_port}, "
              f"clicks <- {args.unity_ip}:{args.click_port}, "
              f"mirror -> {_LOCALHOST}:{_DEFAULT_CTRL_EVENTS_IN_PORT}")
        # Python restarted with a fresh (empty) TaskGraph — wipe any stale Unity visuals
        # left over from the previous session before the user sees the app.
        controller.send({"command": "reset"})
        controller.send({"command": "show_all"})
        controller.send({"command": "ui", "show": False})

    try:
        app.run(
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
