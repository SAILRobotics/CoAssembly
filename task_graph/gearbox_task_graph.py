#!/usr/bin/env python3
"""Interactive gearbox assembly task graph.

The model is intentionally independent from the GUI so that the dependency and
part-transformation rules can be tested without opening a window.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import queue
import re
import sys
import time
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
from speech_listener import SpeechListener, DEFAULT_PULSEAUDIO_DEVICE  # noqa: E402
from tts import TTSService, DEFAULT_PIPER_MODEL  # noqa: E402
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
    requires: tuple[str, ...] = () # completed step IDs; unlike context, their outputs may be consumed


class AssistantInteractionLogger:
    """Append machine-readable part-reference decisions for study analysis."""

    FIELDS = (
        "event_index", "timestamp", "transcript", "vlm_prediction", "vlm_raw",
        "selected_step", "selected_step_state", "graph_decision", "matched_parts",
        "current_assemblies", "highlight_ids", "assembly_highlight_parts",
        "spoken_response",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_index = 1
        if self.path.exists() and self.path.stat().st_size:
            try:
                with self.path.open(newline="", encoding="utf-8-sig") as handle:
                    indices = [int(row.get("event_index", 0) or 0)
                               for row in csv.DictReader(handle)]
                self._next_index = max(indices, default=0) + 1
            except (OSError, ValueError):
                self._next_index = 1

    def append(self, **values) -> None:
        with self._lock:
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            row = {field: values.get(field, "") for field in self.FIELDS}
            row["event_index"] = self._next_index
            row["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
            for field in ("matched_parts", "current_assemblies", "highlight_ids",
                          "assembly_highlight_parts"):
                if not isinstance(row[field], str):
                    row[field] = json.dumps(row[field], ensure_ascii=False)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self._next_index += 1


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
        1: ("GEAR_ROD_ROW1", "GEAR_ROW1_LEFT", "PIN_ROW1_LEFT"),
        2: ("GEAR_ROD_ROW2", "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT"),
        3: ("GEAR_ROD_ROW3", "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT"),
        4: ("GEAR_ROD_ROW4", "GEAR_ROW4_LEFT", "PIN_ROW4_LEFT"),
    }
    gear_text = {
        1: "Attach the single large gear and secure it with the left wooden pin.",
        2: "Attach the small and medium gears and secure both with their pins.",
        3: "Attach the small and medium gears and secure both with their pins.",
        4: "Attach the single large gear and secure it with the pin.",
    }

    for row in range(1, 5):
        for side in ("Right", "Left"):
            steps.append(Step(
                id=f"r{row}_bearing_{side.lower()}",
                title=f"Row {row}.{1 if side == 'Right' else 3}: bearing into {side.lower()} stand",
                row=row,
                stage=0,
                inputs=(f"BEARING_ROW{row}_{side.upper()}", f"STAND_ROW{row}_{side.upper()}"),
                output=f"BEARING_STAND_ROW{row}_{side.upper()}_ASSEMBLY",
                description=(f"Insert BEARING_ROW{row}_{side.upper()} into "
                             f"STAND_ROW{row}_{side.upper()}."),
                requires=((f"r{row}_bearing_right",) if side == "Left" else ()),
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

        first, second = "Right", "Left"
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
        inputs=("MOUNTED_ROW1_ASSEMBLY", "CRANK_HANDLE_ROW1", "PIN_ROW1_RIGHT"),
        output="CRANK_MOUNTED_ROW1_ASSEMBLY",
        description=("Attach CRANK_HANDLE_ROW1 only after GEAR_ROD_ROW1 is between both stands, "
                     "then secure the handle with PIN_ROW1_RIGHT."),
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
        # Row most recently changed by a successful complete/undo operation.
        # Recommendation stays with this row while it still has READY work.
        self.last_worked_row: int | None = None

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
        self.last_worked_row = None

    def missing(self, step: Step) -> list[str]:
        # Inputs are consumed and context parts must remain active. `requires`
        # records workflow precedence independently of whether the producer's
        # output has subsequently been consumed by another assembly step.
        missing_parts = [part for part in (*step.inputs, *step.context)
                         if part not in self.active_parts]
        missing_steps = [required for required in step.requires
                         if required not in self.completed]
        return missing_parts + missing_steps

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
            return False, (f"WARNING: {step.id} cannot happen yet. Missing prerequisite(s): "
                           f"{', '.join(missing)}.")
        for part in step.inputs:
            self.active_parts.remove(part)
            self.transform_history[part] = step.output
        self.active_parts.add(step.output)
        self.completed.append(step.id)
        if step.row > 0:
            self.last_worked_row = step.row
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
        return (step.id in self.completed
                and step.output in self.active_parts
                and not self.completed_consumers(step))

    def completed_consumers(self, step: Step) -> list[Step]:
        # Context dependencies also block undo even though they do not consume
        # the producer's output.
        return [candidate for candidate in self.steps
                if candidate.id in self.completed
                and (step.output in (*candidate.inputs, *candidate.context)
                     or step.id in candidate.requires)]

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
        if target.row > 0:
            # Reverting a specific row makes that row the current work focus.
            self.last_worked_row = target.row
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

    @staticmethod
    def parts_for_reference_label(label: str) -> list[str]:
        """Expand one VLM dataset label to physical task-graph identifiers."""
        label = str(label).strip().upper()
        if label == "BEARING":
            return sorted(p for p in PROVIDED_PARTS if p.startswith("BEARING_"))
        if label == "PIN":
            return sorted(p for p in PROVIDED_PARTS if p.startswith("PIN_"))
        kit = re.fullmatch(r"ROW([1-4])_KIT", label)
        if kit:
            row = kit.group(1)
            prefixes = ("BEARING_", "SCREW_", "PIN_")
            contents = [
                part for part in PROVIDED_PARTS
                if f"_ROW{row}_" in part and part.startswith(prefixes)
            ]
            if row == "1":
                contents.append("CRANK_HANDLE_ROW1")
            return sorted(contents)
        screw = re.fullmatch(r"SCREW_ROW([1-4])", label)
        if screw:
            row = screw.group(1)
            return [f"SCREW_ROW{row}_LEFT", f"SCREW_ROW{row}_RIGHT"]
        return [label] if label in PROVIDED_PARTS else []

    @staticmethod
    def parts_for_layout_object(tool_name: str) -> list[str]:
        """Map one physical pegboard object to the graph parts/tools it supplies.

        This path is required for gesture-originated fetches, which have a
        layout ID/type but no VLM ``matched_parts`` payload.
        """
        name = str(tool_name).strip().upper()
        if re.fullmatch(r"ROW[1-4]_KIT", name):
            return TaskGraph.parts_for_reference_label(name)
        stand = re.fullmatch(r"GEAR_STAND_ROW([1-4])_(LEFT|RIGHT)", name)
        if stand:
            row, side = stand.groups()
            return [f"STAND_ROW{row}_{side}"]
        tool_contents = {
            "BIT_WRENCH": ["BIT_WRENCH"],
            "BIT_SCREWDRIVER": ["BIT_SCREWDRIVER"],
            # tool_layout1.json retains the historical one-l spelling.
            "PHILIPS_SCREWDRIVER": ["PHILLIPS_SCREWDRIVER"],
            "PHILLIPS_SCREWDRIVER": ["PHILLIPS_SCREWDRIVER"],
            "BITHOLDER1": ["BIT_HOLDER1", "H5_HEX_BIT", "H3_HEX_BIT"],
            "BIT_HOLDER1": ["BIT_HOLDER1", "H5_HEX_BIT", "H3_HEX_BIT"],
            "BITHOLDER2": ["BIT_HOLDER2", "T25_TORX_BIT"],
            "BIT_HOLDER2": ["BIT_HOLDER2", "T25_TORX_BIT"],
        }
        if name in tool_contents:
            return tool_contents[name]
        return [name] if name in PROVIDED_PARTS else []

    def current_container(self, part: str) -> str | None:
        """Return an active raw part or the active assembly containing it."""
        if part in self.active_parts:
            return part
        if part not in self.transform_history:
            return None
        current = self.transform_history[part]
        visited = {part}
        while current in self.transform_history and current not in visited:
            visited.add(current)
            current = self.transform_history[current]
        return current if current in self.active_parts else None

    @staticmethod
    def friendly_part(part: str) -> str:
        """Translate canonical graph identifiers into concise spoken language."""
        name = str(part).upper()
        match = re.fullmatch(
            r"(BEARING|STAND|SCREW|PIN|GEAR)_ROW([1-4])_(LEFT|RIGHT)", name)
        if match:
            kind, row, side = match.groups()
            noun = {
                "BEARING": "bearing",
                "STAND": "stand",
                "SCREW": "screw",
                "PIN": "wooden pin",
                "GEAR": "gear",
            }[kind]
            return f"the Row {row} {side.lower()} {noun}"
        match = re.fullmatch(r"GEAR_ROD_ROW([1-4])", name)
        if match:
            return f"the Row {match.group(1)} gear rod"
        if name == "CRANK_HANDLE_ROW1":
            return "the crank handle"
        if name == "BASE_BOARD":
            return "the gearbox baseboard"
        patterns = [
            (r"BEARING_STAND_ROW([1-4])_(LEFT|RIGHT)_ASSEMBLY",
             lambda m: (f"the Row {m.group(1)} {m.group(2).lower()} stand "
                        "with its bearing")),
            (r"GEAR_ROD_ROW([1-4])_ASSEMBLY",
             lambda m: f"the assembled Row {m.group(1)} gear rod"),
            (r"FASTENED_STAND_ROW([1-4])_RIGHT_ASSEMBLY",
             lambda m: f"the Row {m.group(1)} right stand fastened to the board"),
            (r"UNFASTENED_SECOND_STAND_ROW([1-4])_ASSEMBLY",
             lambda m: f"the Row {m.group(1)} gear rod fitted between both stands"),
            (r"MOUNTED_ROW([1-4])_ASSEMBLY",
             lambda m: f"the mounted Row {m.group(1)} assembly"),
            (r"CRANK_MOUNTED_ROW1_ASSEMBLY",
             lambda _m: "the completed Row 1 assembly with its handle"),
            (r"COMPLETED_GEARBOX_ASSEMBLY",
             lambda _m: "the completed gearbox"),
        ]
        for pattern, describe in patterns:
            match = re.fullmatch(pattern, name)
            if match:
                return describe(match)
        return name.replace("_", " ").lower()

    @staticmethod
    def friendly_step_action(step: Step) -> str:
        """Return only the short, familiar action phrase for a step."""
        if step.id.endswith("bearing_right"):
            return "put the bearing into the right stand"
        if step.id.endswith("bearing_left"):
            return "put the bearing into the left stand"
        if step.id.endswith("gear_rod"):
            return "put the gears on the gear rod and pin them"
        if step.id.endswith("fasten_first_stand"):
            return "screw the right stand to the board"
        if step.id.endswith("insert_rod_and_fit_second"):
            return "insert the gear rod and fit the left stand"
        if step.id.endswith("fasten_second_stand"):
            return "screw the left stand to the board"
        if step.id == "r1_attach_handle":
            return "attach and pin the crank handle"
        return "check that the gears line up and turn freely"

    def friendly_step_label(self, step: Step) -> str:
        """Return the compact user-facing stage label, such as ``Step 2.4``."""
        row_stage = self.control_coords_for(step.id)
        return (f"Step {row_stage[0]}.{row_stage[1]}"
                if row_stage and row_stage[0] else "Final check")

    def friendly_step(self, step: Step) -> str:
        """Return one compact stage-and-action description."""
        return f"{self.friendly_step_label(step)}: {self.friendly_step_action(step)}"

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
            for required in step.requires:
                if required not in graph.by_id:
                    errors.append(f"Unknown required step {required} in {step.id}")
            for part in step.inputs:
                if part not in graph.initial_parts and part not in produced:
                    errors.append(f"Unknown input {part} in {step.id}")
        return errors

    @staticmethod
    def steps_for_control(row: int, stage: int) -> list[str]:
        """Bridge a gearbox_control.py (row, control-stage) to this graph's step id(s).

        The two files number "stage" differently: gearbox_control uses per-row control-stages 1-8,
        while this graph uses named steps. The control stages now map one-to-one onto task steps:
        stage 5 is the rod-insert + left-stand fit, stage 6 the left-stand fastening."""
        if stage == 8 or row == 0:
            return ["finish_gearbox"]
        if stage == 7:
            return ["r1_attach_handle"] if row == 1 else []
        return {
            1: [f"r{row}_bearing_right"],
            2: [f"r{row}_gear_rod"],
            3: [f"r{row}_bearing_left"],
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
        lines.append(
            f"Recommendation focus row: {self.last_worked_row}"
            if self.last_worked_row is not None
            else "Recommendation focus row: none yet")

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
            " choosing among multiple READY steps, follow the same deterministic"
            " rule as the interface: first prefer the most recently completed or"
            " reverted row while it has READY work, then choose the lowest"
            " user-facing stage in that row. If that row has no READY work (or no"
            " row has been worked yet), use the lowest-numbered READY row and stage."
            " Recommend the final gearbox step only when it is READY and no row"
            " assembly step remains READY. This preference selects among valid"
            " options; it is not an additional dependency or ordering constraint."
            " Do NOT infer dependencies from descriptions or display order."
        )

        return "\n".join(lines)

    def recommend_next_step(self, exclude_step_id: str | None = None) -> "Step | None":
        """Prefer READY work in the last changed row, then row/stage order.

        ``Step.stage`` represents dependency depth and is not the stage number
        displayed by the interface or used by the controller. Use
        ``control_coords_for`` here so the rule-based recommendation follows the
        documented Stage 1–8 sequence exactly. The global finish step is treated
        as the highest row and therefore comes after all ready row work.
        """
        ready = [s for s in self.steps
                 if self.is_ready(s) and s.id != exclude_step_id]
        if not ready:
            return None

        if self.last_worked_row is not None:
            same_row = [s for s in ready if s.row == self.last_worked_row]
            if same_row:
                ready = same_row

        def _priority(s: Step) -> tuple:
            row = s.row if s.row > 0 else 999
            control_coords = self.control_coords_for(s.id)
            display_stage = control_coords[1] if control_coords is not None else 999
            return (row, display_stage, s.id)

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
        # Asynchronous speech output; initialized by run() unless --no-tts.
        self._tts: TTSService | None = None
        self._assistant_logger: AssistantInteractionLogger | None = None
        self._tool_index = gearbox_control.load_tool_index(
            gearbox_control._DEFAULT_TOOL_JSON)
        try:
            layout = json.loads(Path(gearbox_control._DEFAULT_TOOL_JSON).read_text())
            self._fetchable_tool_ids = {
                int(item["id"])
                for item in layout.get("tools", [])
                if item.get("grasp_joints")
            }
        except (OSError, ValueError, TypeError, KeyError):
            self._fetchable_tool_ids = set()
        # At most one object can wait for spoken fetch confirmation.
        self._pending_fetch: dict[str, object] | None = None
        # Robot supply state is separate from assembly state: grasping or
        # handing over a part does not consume it in the task graph.
        self._robot_part_states: dict[str, dict[str, object]] = {}
        # A restarted task-graph process asks the still-running Open3D/main
        # process to replay its physical grasp/handover state. PUB/SUB may drop
        # the first message while sockets connect, so the request is retried.
        self._robot_state_sync_pending = False
        self._robot_state_sync_next_at = 0.0
        self._gui_built = False

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
        self._gui_built = True

    def run(self, live_port: int | None = None,
            voice_device: str | None = None,
            wake_word: str | None = None,
            vlm_model: str | None = None,
            task_description_path: str | None = None,
            select_port: int | None = None,
            tts_engine: str | None = "piper",
            piper_model: str | Path = DEFAULT_PIPER_MODEL,
            tts_rate: float = 0.85,
            assistant_log_path: str | Path | None = None) -> None:
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
        if tts_engine is not None:
            try:
                self._tts = TTSService(
                    tts_engine, Path(piper_model), speech_rate=tts_rate)
                self._tts.start()
                print(f"[TTS] Started asynchronous {tts_engine} speech output "
                      f"(rate={tts_rate:.2f}x)")
            except Exception as error:
                print(f"[TTS] Disabled: {error}")
        if assistant_log_path is not None:
            self._assistant_logger = AssistantInteractionLogger(assistant_log_path)
            print(f"[StudyLog] Part-reference decisions -> {assistant_log_path}")
        # Build and show the Dear PyGui interface.
        self.build()
        # Seed the model with the graph before the first question; subsequent
        # complete/undo/reset events replace this live-state block.
        self._notify_vlm("INITIAL STATE")
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
            self._request_robot_state_sync_if_due()
            # Apply speech status and transcript events on the main GUI thread.
            self._poll_speech()
            # Display VLM status changes and completed responses.
            if self._vlm is not None:
                self._vlm.tick()
                self._poll_vlm_part_references()
                self._poll_vlm_fetch_confirmations()
                self._poll_vlm_recommendation_requests()
                self._poll_vlm_answers()
            self._poll_tts()
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
        if self._tts is not None:
            self._tts.close()
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
        self._robot_state_sync_pending = True
        self._robot_state_sync_next_at = time.monotonic() + 0.25
        # Report the outgoing endpoint in the application's terminal.
        self.log(f"Step-select link -> tcp://{_LOCALHOST}:{port} (open3d viewer).")
        return True

    def _request_robot_state_sync_if_due(self) -> None:
        """Request retained robot supply state until main acknowledges it."""
        if not self._robot_state_sync_pending or self._select_pub is None:
            return
        now = time.monotonic()
        if now < self._robot_state_sync_next_at:
            return
        self._send_select({"event": "robot_part_state_request"})
        self._robot_state_sync_next_at = now + 1.0

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
        if event == "robot_part_state_snapshot":
            restored, rejected = self._restore_completed_task_stages(
                msg.get("completed_stages", []),
                last_worked_row=msg.get("last_worked_row"),
            )
            # The main process owns physical supply history. Replace only its
            # prior records; keep graph-progression inference independent.
            for label, info in list(self._robot_part_states.items()):
                if info.get("source") != "task_progression":
                    self._robot_part_states.pop(label, None)
            for item in msg.get("items", []):
                if isinstance(item, dict):
                    self._apply_robot_part_status(item, notify=False)
            self._robot_state_sync_pending = False
            self.log(
                f"[TaskRecovery] Restored {restored} completed step(s) and "
                f"{len(msg.get('items', []))} physical object state(s) from "
                f"main_with_robot.py"
                + (f"; rejected {rejected}" if rejected else ""))
            self._notify_vlm("TASK AND ROBOT STATE RESTORED AFTER RESTART")
            if self._gui_built:
                self.refresh()
            return
        if event == "robot_part_status":
            self._apply_robot_part_status(msg)
            return
        row, stage = msg.get("row", 0), msg.get("stage", 0)
        # Translate controller coordinates into task IDs that exist in this graph.
        ids = [i for i in TaskGraph.steps_for_control(row, stage) if i in self.graph.by_id]
        # Highlight the step whose controller view was opened.
        if event == "show":
            already_selected = bool(ids and self.selected_id == ids[0])
            self.active_ids = set(ids)
            if ids:
                self.selected_id = ids[0]
                selected = self.graph.by_id[ids[0]]
                # Unity mirrors programmatic recommendations and part-specific
                # selections back as ``show``. Avoid speaking the same selection
                # twice when it was already selected locally.
                if not already_selected:
                    self._announce_step_selection(selected)
                self._focus_vlm_on_step(selected)
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
                step = self.graph.by_id[sid]
                previous_state = self.graph.state(step)
                ok, message = self.graph.complete(step)
                self.log("Live: " + message)
                if ok:
                    self._speak(
                        f"Step complete. You have created "
                        f"{self.graph.friendly_part(step.output)}.")
                elif previous_state == "complete":
                    self._speak(
                        f"{self.graph.friendly_step(step)} is already complete.")
                else:
                    self._speak(
                        f"That step cannot be completed yet. First prepare "
                        f"{self._friendly_missing(self.graph.missing(step))}.",
                        warning=True)
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"COMPLETE (live): {', '.join(ids)}")
        # Undo mapped steps in reverse dependency order.
        elif event == "uncomplete":
            for sid in reversed(ids):
                ok, message = self.graph.undo(self.graph.by_id[sid])
                self.log("Live: " + message)
                if ok:
                    self._speak(
                        f"Undid {self.graph.friendly_step(self.graph.by_id[sid])}.")
            # Send the updated state to the VLM when the event mapped to a step.
            if ids:
                self._notify_vlm(f"UNDO (live): {', '.join(ids)}")
        # Ignore external event types this application does not recognize.
        else:
            return
        # Redraw nodes and panels after applying the controller event.
        self.refresh()

    def _restore_completed_task_stages(
            self, stage_records, *, last_worked_row=None) -> tuple[int, int]:
        """Rebuild the graph deterministically from main's row/stage snapshot."""
        requested: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        rejected = 0
        for record in stage_records if isinstance(stage_records, list) else []:
            try:
                row = int(record.get("row", 0))
                stage = int(record.get("stage", 0))
            except (AttributeError, TypeError, ValueError):
                rejected += 1
                continue
            if not ((1 <= row <= 4 and 1 <= stage <= 7)
                    or (row == 0 and stage == 8)):
                rejected += 1
                continue
            coords = (row, stage)
            if coords not in seen:
                seen.add(coords)
                requested.append(coords)

        self.graph.reset()
        pending = [
            step_id
            for row, stage in requested
            for step_id in TaskGraph.steps_for_control(row, stage)
            if step_id in self.graph.by_id
        ]
        # Multiple passes make recovery robust if future graph dependencies no
        # longer match the controller's numeric stage ordering.
        while pending:
            progressed = False
            for step_id in list(pending):
                step = self.graph.by_id[step_id]
                if not self.graph.is_ready(step):
                    continue
                ok, _message = self.graph.complete(step)
                if ok:
                    pending.remove(step_id)
                    progressed = True
            if not progressed:
                break
        rejected += len(pending)

        try:
            row_hint = int(last_worked_row)
        except (TypeError, ValueError):
            row_hint = 0
        if 1 <= row_hint <= 4:
            self.graph.last_worked_row = row_hint

        self._sync_sm_from_graph()
        if self.controller is not None:
            self.controller.sm.history = [
                TaskGraph.control_coords_for(step_id)
                for step_id in self.graph.completed
                if TaskGraph.control_coords_for(step_id) is not None
            ]
        return len(self.graph.completed), rejected

    def _apply_robot_part_status(self, msg: dict, *, notify: bool = True) -> None:
        """Remember which semantic parts the robot holds or already supplied."""
        status = str(msg.get("status", "")).strip().lower()
        if status not in {"in_robot_gripper", "handed_over", "grasp_failed"}:
            return

        requested_parts = [
            str(part).strip().upper()
            for part in msg.get("requested_parts", [])
            if str(part).strip()
        ]
        tool_name = str(msg.get("tool_name", "")).strip().upper()
        supplied_parts = list(requested_parts)
        # Always resolve the physical layout object as well. Voice fetches add
        # requested_parts, while gesture fetches depend entirely on this map.
        # A row kit therefore expands to all of its contents; an individual
        # gear, rod, or stand maps to its one canonical graph identifier.
        supplied_parts.extend(self.graph.parts_for_layout_object(tool_name))
        supply_labels = PROVIDED_PARTS | set(
            gearbox_control.FASTENING_TOOL_SPECS)
        supplied_parts = list(dict.fromkeys(
            part for part in supplied_parts if part in supply_labels))
        if not supplied_parts:
            return

        if status == "grasp_failed":
            for part in supplied_parts:
                existing = self._robot_part_states.get(part)
                if (existing
                        and existing.get("source") != "task_progression"
                        and existing.get("tool_id") == msg.get("tool_id")):
                    self._robot_part_states.pop(part, None)
        else:
            for part in supplied_parts:
                self._robot_part_states[part] = {
                    "status": status,
                    "tool_id": msg.get("tool_id"),
                    "tool_name": tool_name,
                    "requested": part in requested_parts,
                    "source": "robot",
                }

        friendly_requested = [
            (self._friendly_reference_label(part, [])
             if part in gearbox_control.FASTENING_TOOL_SPECS
             else self.graph.friendly_part(part))
            for part in requested_parts
        ]
        requested_text = ", ".join(friendly_requested) or tool_name
        self.log(
            f"[RobotSupply] {requested_text}: {status} via "
            f"{tool_name} (id={msg.get('tool_id')})")
        if notify:
            self._notify_vlm(
                f"ROBOT PART STATUS: {requested_text} -> {status.upper()}")

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
        """Draw consumed-input and explicit step-prerequisite links."""
        # Use a short local reference for repeated Dear PyGui calls.
        dpg = self.dpg
        # Map each produced assembly name to the step ID that creates it.
        producer = {step.output: step.id for step in self.graph.steps}
        made: set[tuple[str, str]] = set()
        # Inspect every step that may consume another step's output or name an
        # explicit workflow prerequisite through Step.requires.
        for step in self.graph.steps:
            sources: list[str] = []
            # Check each consumed input required by the consumer step.
            for part in step.inputs:
                # Find the producer, or None when the input is a raw part.
                source_id = producer.get(part)
                if source_id:
                    sources.append(source_id)
            # A requires edge represents precedence without consuming the
            # producer's output (for example N.1 right stand before N.3 left).
            sources.extend(required for required in step.requires
                           if required in self.output_attributes)
            for source_id in sources:
                edge = (source_id, step.id)
                if edge in made:
                    continue
                made.add(edge)
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
        dpg.add_button(label="Reset progress + handovers",
                       callback=self._reset_progress_and_handovers_callback,
                       width=-1)
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
                         default_value=True)
        dpg.add_spacer(height=4)
        dpg.add_text("Transcripts:", color=(160, 170, 185))
        dpg.add_input_text(tag="voice_transcripts", multiline=True, readonly=True,
                           width=-1, height=120,
                           default_value="Speak normally; no wake word is required.")

    # ── Voice input and VLM integration ──────────────────────────────────────

    def _poll_speech(self) -> None:
        """Refresh voice widgets and handle all newly available speech events."""
        if self._speech is None:
            return
        dpg = self.dpg
        events = self._speech.poll()

        # Status label + color
        # loading=model starting; idle=waiting for an optional wake word;
        # speech=someone is talking and audio is being captured right now;
        # queued=captured audio is waiting for ASR; transcribing=ASR is running;
        # listening=continuous capture is active (or an optional wake word was
        # accepted); error=capture or recognition failed.
        status = self._speech.current_status
        color, label = self._VOICE_STATUS_STYLE.get(
            status, ((200, 200, 200, 255), status))
        if self._speech.always_listening:
            label = "Listening — no wake word required"
        elif self._speech.listening_active:
            label = f"Listening  (wake word: \"{self._speech.wake_word}\")"
        elif status == "idle":
            label = f"Idle — say \"{self._speech.wake_word}\""
        dpg.set_value("voice_status", label)
        dpg.configure_item("voice_status", color=list(color))

        # Timer
        if self._speech.listening_active and not self._speech.always_listening:
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
                if self._tts is not None and self._tts.is_speaking:
                    self.log("[Voice] Ignored transcript produced during TTS playback.")
                    continue
                if route_to_vlm:
                    if not self._vlm.submit_question(payload):
                        self.log("[Voice] VLM busy — transcript skipped.")
            elif kind == "timeout":
                self.log("[Voice] Timed out — back to idle.")
            elif kind == "error":
                self.log(f"[Voice error] {payload}")

    def _speak(self, text: str, warning: bool = False) -> None:
        """Queue concise guidance without blocking the Dear PyGui frame loop."""
        self.log(f"[TTS] {text}")
        if self._tts is not None:
            self._tts.speak(text, replace=warning)

    def _poll_tts(self) -> None:
        if self._tts is None:
            return
        for kind, payload in self._tts.poll():
            if kind == "error":
                self.log(f"[TTS error] {payload}")

    def _poll_vlm_part_references(self) -> None:
        if self._vlm is None:
            return
        for result in self._vlm.poll_part_references():
            self._handle_part_reference(result)

    def _poll_vlm_answers(self) -> None:
        """Speak validated natural-language answers produced by the VLM."""
        if self._vlm is None:
            return
        for result in self._vlm.poll_answers():
            answer = result.get("answer", "").strip()
            if answer:
                self._speak(
                    answer,
                    warning=result.get("intent") == "invalid",
                )

    def _poll_vlm_recommendation_requests(self) -> None:
        if self._vlm is None:
            return
        for result in self._vlm.poll_recommendation_requests():
            self._activate_recommended_step(result.get("text", ""))

    def _poll_vlm_fetch_confirmations(self) -> None:
        if self._vlm is None:
            return
        for result in self._vlm.poll_fetch_confirmations():
            self._handle_fetch_confirmation(result)

    def _clear_pending_fetch(self) -> None:
        self._pending_fetch = None
        if self._vlm is not None:
            self._vlm.set_pending_fetch(None)

    @staticmethod
    def _row_of_part(part: str) -> int | None:
        match = re.search(r"_ROW([1-4])(?:_|$)", str(part).upper())
        return int(match.group(1)) if match else None

    @staticmethod
    def _row_hint_from_reference(text: str) -> int | None:
        """Extract row numbers, including common Parakeet row/roll variants."""
        words = str(text).lower()
        number_words = {"one": 1, "two": 2, "three": 3, "four": 4}
        match = re.search(r"\b(?:row|roll)\s*([1-4])\b", words)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(?:row|roll)\s+(one|two|three|four)\b", words)
        if match:
            return number_words[match.group(1)]
        color_rows = {"white": 1, "red": 2, "green": 3, "blue": 4}
        mentioned = [row for color, row in color_rows.items()
                     if re.search(rf"\b{color}\b", words)]
        return mentioned[0] if len(mentioned) == 1 else None

    def _friendly_reference_label(self, label: str,
                                  parts: list[str]) -> str:
        label = label.upper()
        kit = re.fullmatch(r"ROW([1-4])_KIT", label)
        if kit:
            return f"Row {kit.group(1)} kit"
        tool_spec = gearbox_control.FASTENING_TOOL_SPECS.get(label)
        if tool_spec is not None:
            return str(tool_spec["friendly"])
        if label == "BEARING":
            return "bearing"
        if label == "PIN":
            return "wooden retaining pin"
        if label.startswith("SCREW_ROW"):
            return f"Row {label[-1]} mounting screw"
        friendly = self.graph.friendly_part(parts[0]) if parts else "part"
        # Reference-response templates add their own article ("The"/"That").
        # friendly_part() includes "the" when it is used as a standalone noun
        # phrase, so remove it here to avoid speech such as "The the Row 1...".
        return friendly[4:] if friendly.lower().startswith("the ") else friendly

    def _send_reference_highlight(self, ids: list[int], color: list[float],
                                  status: str, label: str,
                                  assembly_parts: Iterable[str] = ()) -> None:
        assembly_parts = list(dict.fromkeys(assembly_parts))
        self._send_select({
            "event": "reference_highlight",
            "ids": sorted({int(tool_id) for tool_id in ids}),
            "color": color,
            "status": status,
            "label": label,
            "assembly_parts": assembly_parts,
        })
        # In --with-controller mode this also works when main_with_robot.py is
        # not running. When main is present, receiving the same idempotent
        # reference color a second time is harmless.
        if self.controller is not None:
            self.controller.send(
                {"command": "reference_color", "parts": assembly_parts,
                 "color": color}
                if assembly_parts else {"command": "reference_clear"})

    def _record_reference_decision(self, result: dict[str, object], decision: str,
                                   matched_parts: Iterable[str],
                                   current_assemblies: Iterable[str], ids: list[int],
                                   assembly_parts: Iterable[str], spoken: str) -> None:
        if self._assistant_logger is None:
            return
        step = self.graph.by_id.get(self.selected_id) if self.selected_id else None
        try:
            self._assistant_logger.append(
                transcript=result.get("text", ""),
                vlm_prediction=result.get("label", "INVALID_OUTPUT"),
                vlm_raw=result.get("raw", ""),
                selected_step=step.id if step else "",
                selected_step_state=self.graph.state(step) if step else "none",
                graph_decision=decision,
                matched_parts=list(matched_parts),
                current_assemblies=list(current_assemblies),
                highlight_ids=ids,
                assembly_highlight_parts=list(assembly_parts),
                spoken_response=spoken,
            )
        except OSError as error:
            self.log(f"[StudyLog error] {error}")

    def _emit_reference_decision(
            self, result: dict[str, object], decision: str, color: list[float],
            spoken: str, *, ids: list[int] | None = None,
            matched_parts: Iterable[str] = (),
            current_assemblies: Iterable[str] = (),
            assembly_parts: Iterable[str] = (), warning: bool = False) -> None:
        ids = sorted(set(ids or []))
        assembly_parts = list(dict.fromkeys(assembly_parts))
        self._send_reference_highlight(
            ids, color, decision, result.get("label", "INVALID_OUTPUT"),
            assembly_parts=assembly_parts)
        if self._vlm is not None:
            self._vlm.apply_policy_response(
                result.get("text", ""),
                result.get("label", "INVALID_OUTPUT"),
                spoken,
            )
        self._speak(spoken, warning=warning)
        self._record_reference_decision(
            result, decision, matched_parts, current_assemblies, ids,
            assembly_parts, spoken)

    @staticmethod
    def _ambiguity_question(text: str) -> str:
        words = str(text).lower()
        if any(word in words for word in ("stand", "bracket", "support")):
            if DearPyGuiTaskGraphApp._row_hint_from_reference(text) is not None:
                return "Which side do you mean: left or right?"
            return "Which row and side do you mean: left or right?"
        if "gear" in words and "rod" not in words and "shaft" not in words:
            return "Which gear do you mean? Please give its row, color, size, or position."
        if "holder" in words:
            return ("Which bit holder do you mean: Bit Holder 1 with the H5 and "
                    "H3 bits, or Bit Holder 2 with the T25 bit?")
        if any(word in words for word in ("screwdriver", "driver", "bit", "wrench")):
            return ("Which fastening tool do you mean: H5, T25, H3, or the "
                    "Phillips screwdriver? You can also tell me the row.")
        if any(word in words for word in ("screw", "bearing", "pin")):
            return "Which row are you assembling?"
        return "Which object do you mean? Please add its row, side, color, size, or position."

    @staticmethod
    def _ambiguous_part_category(text: str) -> tuple[str, list[str]]:
        """Expand an ambiguous common noun without guessing a specific object."""
        words = str(text).lower()
        if any(word in words for word in ("stand", "bracket", "support")):
            return "Gear stands", sorted(
                part for part in PROVIDED_PARTS if part.startswith("STAND_"))
        if "gear" in words and any(word in words for word in ("rod", "shaft")):
            return "Gear rods", sorted(
                part for part in PROVIDED_PARTS if part.startswith("GEAR_ROD_"))
        if "gear" in words:
            return "Gears", sorted(
                part for part in PROVIDED_PARTS if part.startswith("GEAR_ROW"))
        if "bearing" in words:
            return "Bearings", sorted(
                part for part in PROVIDED_PARTS if part.startswith("BEARING_"))
        if "screw" in words:
            return "Screws", sorted(
                part for part in PROVIDED_PARTS if part.startswith("SCREW_"))
        if "pin" in words:
            return "Wooden pins", sorted(
                part for part in PROVIDED_PARTS if part.startswith("PIN_"))
        return "", []

    def _candidate_parts_from_vlm(self, labels: Iterable[str]) -> list[str]:
        """Expand validated VLM ambiguity candidates into graph part names."""
        parts: list[str] = []
        for label in labels:
            for part in self.graph.parts_for_reference_label(label):
                if part not in parts:
                    parts.append(part)
        return parts

    def _friendly_candidate_category(self, candidates: Iterable[str]) -> str:
        """Describe a VLM-provided candidate set without reparsing the speech."""
        parts = list(candidates)
        if not parts:
            return ""
        if all(part.startswith("STAND_") for part in parts):
            category = "Gear stands"
        elif all(part.startswith("GEAR_ROD_") for part in parts):
            category = "Gear rods"
        elif all(part.startswith("GEAR_ROW") for part in parts):
            category = "Gears"
        elif all(part.startswith("BEARING_") for part in parts):
            category = "Bearings"
        elif all(part.startswith("SCREW_") for part in parts):
            category = "Screws"
        elif all(part.startswith("PIN_") for part in parts):
            category = "Wooden pins"
        else:
            return "Parts"
        rows = {self._row_of_part(part) for part in parts}
        rows.discard(None)
        if len(rows) == 1:
            return f"Row {next(iter(rows))} {category.lower()}"
        return category

    def _arm_fetch_confirmation(
            self, result: dict[str, object], ids: Iterable[int], friendly: str,
            *, matched_parts: Iterable[str],
            assembly_parts: Iterable[str] = ()) -> bool:
        """Highlight one graspable object and ask before moving the robot."""
        unique_ids = sorted({int(tool_id) for tool_id in ids})
        if len(unique_ids) != 1:
            self._pending_fetch = None
            if self._vlm is not None:
                self._vlm.set_pending_fetch(None)
            spoken = (
                "I need one specific fetchable object. Please name only one part "
                "or tool, including its row and side when necessary.")
            self._emit_reference_decision(
                result, "fetch_not_specific", [1.0, 0.72, 0.0, 0.2],
                spoken, warning=True)
            return False

        tool_id = unique_ids[0]
        if tool_id not in self._fetchable_tool_ids:
            self._pending_fetch = None
            if self._vlm is not None:
                self._vlm.set_pending_fetch(None)
            spoken = (
                f"I highlighted the {friendly}, but it has no recorded robot "
                "grasp pose, so I cannot fetch it automatically.")
            self._emit_reference_decision(
                result, "fetch_pose_unavailable", [1.0, 0.72, 0.0, 0.2],
                spoken, ids=[tool_id], matched_parts=matched_parts,
                assembly_parts=assembly_parts, warning=True)
            return False

        self._pending_fetch = {
            "tool_id": tool_id,
            "label": str(result.get("label", "")),
            "friendly": friendly,
            "matched_parts": list(matched_parts),
            "assembly_parts": list(assembly_parts),
        }
        if self._vlm is not None:
            self._vlm.set_pending_fetch(friendly)
        spoken = f"I highlighted the {friendly}. Do you want me to get it?"
        self._emit_reference_decision(
            result, "fetch_confirmation_pending", [0.0, 1.0, 1.0, 0.2],
            spoken, ids=[tool_id], matched_parts=matched_parts,
            assembly_parts=assembly_parts)
        return True

    def _handle_fetch_confirmation(self, result: dict[str, str]) -> None:
        """Accept or cancel the single pending voice-requested robot fetch."""
        pending = self._pending_fetch
        question = result.get("text", "")
        if pending is None:
            spoken = "There is no pending robot fetch to confirm."
            self._speak(spoken, warning=True)
            if self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "FETCH_CANCELLED", spoken)
            return

        confirmed = result.get("confirmation") == "yes"
        self._pending_fetch = None
        if self._vlm is not None:
            self._vlm.set_pending_fetch(None)
        friendly = str(pending["friendly"])
        if confirmed:
            self._send_select({
                "event": "fetch_confirmed",
                "tool_id": int(pending["tool_id"]),
                "label": str(pending["label"]),
                "matched_parts": list(pending.get("matched_parts", [])),
            })
            spoken = f"Okay. I asked the robot to get the {friendly}."
            display_label = "FETCH_CONFIRMED"
        else:
            self._send_reference_highlight(
                [], [0.0, 1.0, 1.0, 0.2], "fetch_cancelled",
                str(pending["label"]))
            spoken = f"Okay. I cancelled the fetch for the {friendly}."
            display_label = "FETCH_CANCELLED"
        self._speak(spoken, warning=not confirmed)
        if self._vlm is not None:
            self._vlm.apply_policy_response(question, display_label, spoken)

    def _part_step_priority(self, step: Step) -> tuple[int, int, str]:
        """Use the interface's deterministic ordering for a filtered step set."""
        row = step.row if step.row > 0 else 999
        coords = self.graph.control_coords_for(step.id)
        stage = coords[1] if coords is not None else 999
        return row, stage, step.id

    def _tool_is_used_by_step(self, label: str, step: Step) -> bool:
        """Return whether one fastening-tool label is required by ``step``."""
        coords = self.graph.control_coords_for(step.id)
        if coords is None or coords[1] not in (4, 6):
            return False
        required = set(gearbox_control.fastening_tool_labels(step.row))
        if label == "BIT_HOLDER1":
            return bool(required & {"H5_HEX_BIT", "H3_HEX_BIT"})
        if label == "BIT_HOLDER2":
            return "T25_TORX_BIT" in required
        return label in required

    def _tools_retained_from_completed_steps(self) -> dict[str, Step]:
        """Tools used by a completed fastening step remain with the operator."""
        retained: dict[str, Step] = {}
        for step_id in self.graph.completed:
            step = self.graph.by_id[step_id]
            coords = self.graph.control_coords_for(step.id)
            if coords is None or coords[1] not in (4, 6):
                continue
            for label in gearbox_control.fastening_tool_labels(step.row):
                retained[label] = step
        return retained

    def _parts_supplied_by_completed_steps(self) -> dict[str, Step]:
        """Return raw parts whose physical containers must already be supplied.

        When one item from a row kit has been used, the entire physical kit is
        treated as supplied because it is represented by one pegboard object.
        The same grouping rule applies to any future shared physical container.
        """
        used_parts: dict[str, Step] = {}
        for step_id in self.graph.completed:
            step = self.graph.by_id[step_id]
            for part in step.inputs:
                if part in PROVIDED_PARTS:
                    used_parts[part] = step

        supplied: dict[str, Step] = {}
        for used_part, step in used_parts.items():
            tool_id = gearbox_control.tool_id_for_graph_part(
                self._tool_index, used_part)
            if tool_id is None:
                continue
            # Expand a physical row-kit box back to every semantic item stored
            # inside it so the VLM does not offer a hidden kit item later.
            container_parts = [
                part for part in PROVIDED_PARTS
                if gearbox_control.tool_id_for_graph_part(
                    self._tool_index, part) == tool_id
            ]
            for part in container_parts or [used_part]:
                supplied.setdefault(part, step)
        return supplied

    def _sync_progress_inferred_supply_statuses(self) -> None:
        """Mirror parts/tools implied by completed work into handed-over state.

        Consumed raw parts and tools used by completed fastening stages must
        have been available to the operator. This records that fact in the same
        state table used for robot handovers, with ``source=task_progression``
        so it remains distinguishable and reversible.
        """
        inferred_supply = self._parts_supplied_by_completed_steps()
        inferred_supply.update(self._tools_retained_from_completed_steps())

        # Undo/reset must remove only inferred records. A real robot handover
        # remains valid independently of subsequent task-graph edits.
        for label, info in list(self._robot_part_states.items()):
            if (info.get("source") == "task_progression"
                    and label not in inferred_supply):
                self._robot_part_states.pop(label, None)
                self.log(
                    f"[ProgressSupply] Removed inferred handover for {label} "
                    "after task progression changed")

        for label, step in inferred_supply.items():
            existing = self._robot_part_states.get(label)
            # A physical robot grasp/handover is stronger evidence and must not
            # be overwritten by task progression.
            if existing is not None and existing.get("source") != "task_progression":
                continue
            if label in gearbox_control.FASTENING_TOOL_SPECS:
                ids = gearbox_control.tool_ids_for_reference(
                    self._tool_index, label)
            else:
                tool_id = gearbox_control.tool_id_for_graph_part(
                    self._tool_index, label)
                ids = [] if tool_id is None else [tool_id]
            layout_name = next(
                (name for name, tool_id in self._tool_index.items()
                 if ids and tool_id == ids[0]),
                label,
            )
            new_info = {
                "status": "handed_over",
                "tool_id": ids[0] if ids else None,
                "tool_name": layout_name,
                "requested": False,
                "source": "task_progression",
                "inferred_from_step": step.id,
            }
            if existing != new_info:
                self._robot_part_states[label] = new_info
                self.log(
                    f"[ProgressSupply] {label}: handed_over inferred from "
                    f"completed step {step.id}")

    def _publish_progress_handover_state(self) -> None:
        """Synchronize progression-inferred pegboard removals with Open3D/Unity."""
        inferred = {
            label: info
            for label, info in self._robot_part_states.items()
            if info.get("source") == "task_progression"
        }
        tool_ids = sorted({
            int(info["tool_id"])
            for info in inferred.values()
            if info.get("tool_id") is not None
        })
        self._send_select({
            "event": "progress_handover_sync",
            "tool_ids": tool_ids,
            "items": [
                {
                    "label": label,
                    "tool_id": info.get("tool_id"),
                    "inferred_from_step": info.get("inferred_from_step"),
                }
                for label, info in sorted(inferred.items())
            ],
        })

    def _handle_part_step_lookup(
            self, result: dict[str, object], label: str,
            parts: Iterable[str], *, is_tool: bool) -> None:
        """Select the next READY graph step that uses a named part or tool."""
        parts = list(parts)
        if is_tool:
            candidates = [
                step for step in self.graph.steps
                if step.id not in self.graph.completed
                and self._tool_is_used_by_step(label, step)
            ]
        else:
            current_objects = {
                current for part in parts
                if (current := self.graph.current_container(part)) is not None
            }
            candidates = [
                step for step in self.graph.steps
                if step.id not in self.graph.completed
                and any(item in (*step.inputs, *step.context)
                        for item in current_objects)
            ]

        # Preserve the interface preference inside this part-filtered set.
        ready = [step for step in candidates if self.graph.is_ready(step)]
        if ready and self.graph.last_worked_row is not None:
            same_row = [step for step in ready
                        if step.row == self.graph.last_worked_row]
            if same_row:
                ready = same_row

        question = str(result.get("text", ""))
        friendly = self._friendly_reference_label(label, parts)
        if ready:
            step = min(ready, key=self._part_step_priority)
            self.recommended_id = step.id
            self._select_step(step.id, announce=False)
            if self.controller is not None:
                self._animate_unity_callback()
            action = self.graph.friendly_step_action(step)
            spoken = (
                f"I selected the next ready step that uses the {friendly}. "
                f"{action.capitalize()}.")
            self.log(f"[Part step] Auto-selected [{step.id}] for {label}")
            self._speak(spoken)
            coords = self.graph.control_coords_for(step.id)
            highlight_ids = (
                gearbox_control.appearing_ids(
                    self._tool_index, coords[0], coords[1])
                if coords is not None and coords[0] > 0 else [])
            self._record_reference_decision(
                result, "part_step_selected", parts or [label], (),
                highlight_ids, (), spoken)
            if self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "PART_STEP_SELECTED", spoken)
            return

        if candidates:
            blocked = min(candidates, key=self._part_step_priority)
            missing = self._friendly_missing(self.graph.missing(blocked))
            spoken = (
                f"The next step that uses the {friendly} is not ready. "
                f"First, {missing}.")
        else:
            spoken = (
                f"There is no remaining assembly step that uses the {friendly}.")
        self._speak(spoken, warning=True)
        self._record_reference_decision(
            result, "part_step_unavailable", parts or [label], (), [], (),
            spoken)
        if self._vlm is not None:
            self._vlm.apply_policy_response(
                question, "PART_STEP_UNAVAILABLE", spoken)

    def _robot_supply_response(
            self, label: str, parts: Iterable[str]) -> tuple[str, str]:
        """Return a user-facing response for parts already carried by the robot."""
        parts = list(parts)
        infos = [self._robot_part_states[part] for part in parts]
        statuses = {str(info.get("status", "")) for info in infos}
        friendly = self._friendly_reference_label(label, parts)
        physical_names = {str(info.get("tool_name", "")) for info in infos}
        kit_name = next(iter(physical_names)) if len(physical_names) == 1 else ""
        kit_match = re.fullmatch(r"ROW([1-4])_KIT", kit_name)
        kit_suffix = (f" as part of the Row {kit_match.group(1)} kit"
                      if kit_match and label.upper() != kit_name else "")
        if statuses == {"handed_over"}:
            return (
                "already_handed_over",
                f"The {friendly} has already been delivered to you{kit_suffix}.",
            )
        return (
            "already_in_robot_gripper",
            f"The robot is already holding the {friendly}{kit_suffix}.",
        )

    def _handle_part_reference(self, result: dict[str, object]) -> None:
        """Apply deterministic task-state policy to one VLM-resolved label."""
        label = str(result.get("label", "INVALID_OUTPUT"))
        if label == "STEP_PARTS":
            self._handle_step_parts_request(result)
            return
        if label == "STEP_TOOLS":
            self._handle_step_tools_request(result)
            return
        if label == "AMBIGUOUS":
            candidate_labels = result.get("candidate_labels", [])
            candidates = self._candidate_parts_from_vlm(candidate_labels)
            category = self._friendly_candidate_category(candidates)
            # Backward compatibility for explicit callers of
            # submit_part_reference(), which return only AMBIGUOUS. Normal
            # voice/text questions now receive candidates directly from Qwen.
            if not candidates:
                category, candidates = self._ambiguous_part_category(
                    result.get("text", ""))
                row_hint = self._row_hint_from_reference(result.get("text", ""))
                if row_hint is not None:
                    candidates = [part for part in candidates
                                  if self._row_of_part(part) == row_hint]
                    if category:
                        category = f"Row {row_hint} {category.lower()}"
            if (result.get("part_action") != "find_step"
                    and self.selected_id and candidates):
                step = self.graph.by_id[self.selected_id]
                step_parts = set((*step.inputs, *step.context))
                relevant = [
                    part for part in candidates
                    if (part in step_parts
                        or self.graph.current_container(part) in step_parts)
                ]
                if len(relevant) == 1:
                    # The selected step supplies enough context to resolve an
                    # otherwise generic expression, such as "the stand" when
                    # only its right stand is an input.
                    result = dict(result)
                    result["label"] = relevant[0]
                    label = relevant[0]
                elif not relevant:
                    ids = [
                        gearbox_control.tool_id_for_graph_part(self._tool_index, part)
                        for part in candidates if part in self.graph.active_parts
                    ]
                    ids = [tool_id for tool_id in ids if tool_id is not None]
                    spoken = (
                        f"{category} are not relevant to this step. "
                        f"This step is to {self.graph.friendly_step_action(step)}.")
                    self._emit_reference_decision(
                        result, "not_relevant_ambiguous",
                        [1.0, 0.0, 0.0, 0.25], spoken,
                        ids=ids, matched_parts=candidates, warning=True)
                    return
            if label == "AMBIGUOUS":
                spoken = self._ambiguity_question(result.get("text", ""))
                self._emit_reference_decision(
                    result, "ambiguous", [1.0, 0.72, 0.0, 0.2], spoken,
                    warning=True)
                return

        is_tool = label in gearbox_control.FASTENING_TOOL_SPECS
        is_assembly = self.graph.producer_for(label) is not None
        parts = self.graph.parts_for_reference_label(label)
        if is_assembly:
            self._handle_assembly_reference(result, label)
            return
        if label == "INVALID_OUTPUT" or (not parts and not is_tool):
            spoken = ("I could not identify one specific part or tool. Please mention "
                      "its row, side, color, shape, or driver marking.")
            self._emit_reference_decision(
                result, "unresolved", [0.0, 1.0, 1.0, 0.2], spoken,
                warning=True)
            return

        if result.get("part_action") == "find_step":
            self._handle_part_step_lookup(
                result, label, parts, is_tool=is_tool)
            return

        # A fully supplied specific/grouped reference has an answer even when
        # no assembly step is selected. Do not ask the user to select a step or
        # offer to fetch the same physical kit again.
        if (parts and not is_tool
                and all(part in self._robot_part_states for part in parts)):
            decision, spoken = self._robot_supply_response(label, parts)
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=parts)
            return

        if not self.selected_id:
            spoken = ("Please select an assembly step first, so I can check whether "
                      "that part or tool belongs to the current task.")
            self._emit_reference_decision(
                result, "no_selected_step", [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=parts, warning=True)
            return

        step = self.graph.by_id[self.selected_id]
        if is_tool:
            coords = self.graph.control_coords_for(step.id)
            required = set(gearbox_control.fastening_tool_labels(step.row))
            # Each holder is relevant only when the row needs an insert stored
            # in that holder: H5/H3 in Holder 1 and T25 in Holder 2.
            holder1_needed = bool(required & {"H5_HEX_BIT", "H3_HEX_BIT"})
            holder2_needed = "T25_TORX_BIT" in required
            relevant = (coords is not None and coords[1] in (4, 6)
                        and (label in required
                             or (label == "BIT_HOLDER1" and holder1_needed)
                             or (label == "BIT_HOLDER2" and holder2_needed)))
            ids = gearbox_control.tool_ids_for_reference(self._tool_index, label)
            spoken_tool = self._friendly_reference_label(label, [])
            supplied_info = self._robot_part_states.get(label)
            if supplied_info is not None:
                if supplied_info.get("source") == "task_progression":
                    retained_step = self.graph.by_id.get(
                        str(supplied_info.get("inferred_from_step", "")))
                    prior_action = (
                        self.graph.friendly_step_action(retained_step)
                        if retained_step is not None else "complete earlier work")
                    spoken = (
                        f"You should already have the {spoken_tool}. It was "
                        f"used earlier to {prior_action} and should still be "
                        "with you.")
                    decision = "tool_retained_by_operator"
                elif supplied_info.get("status") == "in_robot_gripper":
                    spoken = f"The robot is already holding the {spoken_tool}."
                    decision = "tool_in_robot_gripper"
                else:
                    spoken = f"The {spoken_tool} has already been delivered to you."
                    decision = "tool_handed_over"
                self._emit_reference_decision(
                    result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=[], matched_parts=[label])
                return
            retained_step = self._tools_retained_from_completed_steps().get(label)
            if retained_step is not None:
                spoken = (
                    f"The {spoken_tool} was already used to "
                    f"{self.graph.friendly_step_action(retained_step)}, so it "
                    "should still be with you for this step.")
                self._emit_reference_decision(
                    result, "tool_retained_by_operator",
                    [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=[], matched_parts=[label])
                return
            if relevant:
                if result.get("part_action") == "fetch":
                    if self.graph.state(step) != "ready":
                        spoken = (
                            f"The {spoken_tool} belongs to this step, but this step "
                            "is not ready, so I will not fetch it yet.")
                        self._emit_reference_decision(
                            result, "fetch_blocked_step",
                            [1.0, 0.72, 0.0, 0.2], spoken,
                            ids=ids, matched_parts=[label], warning=True)
                        return
                    self._arm_fetch_confirmation(
                        result, ids, spoken_tool, matched_parts=[label])
                    return
                if result.get("part_action") == "status":
                    spoken = (f"No. The {spoken_tool} has not been delivered yet. "
                              f"It is needed to "
                              f"{self.graph.friendly_step_action(step)}. "
                              "I have highlighted it.")
                else:
                    spoken = (f"The {spoken_tool} is needed to "
                              f"{self.graph.friendly_step_action(step)}. "
                              "I have highlighted it, but it has not been "
                              "delivered yet.")
                self._emit_reference_decision(
                    result, "relevant_tool", [0.0, 1.0, 1.0, 0.2], spoken,
                    ids=ids, matched_parts=[label])
            else:
                required_text = self._fastening_tool_guidance(step)
                suffix = (f" Use {required_text}." if required_text else
                          " This selected step does not require a fastening tool.")
                spoken = f"The {spoken_tool} is not relevant to this step.{suffix}"
                self._emit_reference_decision(
                    result, "not_relevant_tool", [1.0, 0.0, 0.0, 0.25], spoken,
                    ids=ids, matched_parts=[label], warning=True)
            return

        step_parts = set((*step.inputs, *step.context))
        # Grouped labels such as BEARING and PIN are disambiguated by the
        # selected row before considering other rows.
        same_row = [part for part in parts
                    if step.row and self._row_of_part(part) == step.row]
        candidates = same_row or parts
        # Row context may turn a broad label such as BEARING into a fully
        # supplied row-specific set. Delivery state takes precedence over the
        # selected-step relevance warning.
        if (candidates
                and all(part in self._robot_part_states for part in candidates)):
            decision, spoken = self._robot_supply_response(label, candidates)
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=candidates)
            return
        active_relevant = [part for part in candidates
                           if part in self.graph.active_parts
                           and part in step_parts
                           and part not in self._robot_part_states]
        robot_supplied_relevant = [
            part for part in candidates
            if part in step_parts and part in self._robot_part_states
        ]
        consumed = [(part, self.graph.current_container(part))
                    for part in candidates
                    if part not in self.graph.active_parts
                    and self.graph.current_container(part) is not None]
        consumed_relevant = [(part, current) for part, current in consumed
                             if current in step_parts]
        spoken_part = self._friendly_reference_label(label, candidates)

        if robot_supplied_relevant and not active_relevant:
            states = {
                str(self._robot_part_states[part].get("status", ""))
                for part in robot_supplied_relevant
            }
            friendly_parts = [self.graph.friendly_part(part)
                              for part in robot_supplied_relevant]
            supplied_text = (friendly_parts[0] if len(friendly_parts) == 1
                             else ", and ".join(friendly_parts))
            if "in_robot_gripper" in states:
                spoken = f"The robot is already holding {supplied_text}."
                decision = "already_in_robot_gripper"
            else:
                spoken = f"{supplied_text.capitalize()} has already been handed to you."
                decision = "already_handed_over"
            self._emit_reference_decision(
                result, decision, [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=robot_supplied_relevant)
            return

        if consumed_relevant or (consumed and not active_relevant):
            consumed_source = consumed_relevant or consumed
            locations = list(dict.fromkeys(
                current for _part, current in consumed_source
                if current is not None))
            friendly_locations = [self.graph.friendly_part(item)
                                  for item in locations]
            location_text = (friendly_locations[0] if len(friendly_locations) == 1
                             else ", and ".join(friendly_locations))
            assembly_parts = [part for part, _current in consumed_source]
            spoken = (f"That {spoken_part} has already been used. It is now part of "
                      f"{location_text}. I have highlighted its location on the gearbox.")
            self._emit_reference_decision(
                result, "already_used", [1.0, 0.92, 0.02, 1.0], spoken,
                matched_parts=assembly_parts, current_assemblies=locations,
                assembly_parts=assembly_parts)
            return

        if active_relevant:
            ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
                   for part in active_relevant]
            ids = [tool_id for tool_id in ids if tool_id is not None]
            if result.get("part_action") == "fetch":
                if self.graph.state(step) != "ready":
                    spoken = (
                        f"The {spoken_part} belongs to this step, but this step "
                        "is not ready, so I will not fetch it yet.")
                    self._emit_reference_decision(
                        result, "fetch_blocked_step",
                        [1.0, 0.72, 0.0, 0.2], spoken,
                        ids=ids, matched_parts=active_relevant,
                        warning=True)
                    return
                self._arm_fetch_confirmation(
                    result, ids, spoken_part,
                    matched_parts=active_relevant,
                    assembly_parts=active_relevant)
                return
            if self.graph.state(step) == "blocked":
                missing = self._friendly_missing(self.graph.missing(step))
                spoken = (f"The {spoken_part} belongs to this task, and I have highlighted "
                          f"its storage location. Do not start yet; first complete or prepare "
                          f"{missing}.")
                warning = True
            else:
                spoken = (f"Yes. The {spoken_part} is needed to "
                          f"{self.graph.friendly_step_action(step)}. "
                          "I have highlighted its "
                          "storage location.")
                warning = False
            self._emit_reference_decision(
                result, "relevant", [0.0, 1.0, 1.0, 0.2], spoken,
                ids=ids, matched_parts=active_relevant,
                # Mirror the same cyan semantic highlight onto the component
                # in the assembled BoardAR gearbox, not only its pegboard
                # storage object or row-kit box.
                assembly_parts=active_relevant, warning=warning)
            return

        ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
               for part in candidates if part in self.graph.active_parts]
        ids = [tool_id for tool_id in ids if tool_id is not None]
        spoken = (f"The {spoken_part} is not relevant to this step. "
                  f"This step is to {self.graph.friendly_step_action(step)}.")
        self._emit_reference_decision(
            result, "not_relevant", [1.0, 0.0, 0.0, 0.25], spoken,
            ids=ids, matched_parts=candidates, warning=True)

    def _handle_assembly_reference(
            self, result: dict[str, object], label: str) -> None:
        """Explain where a graph subassembly is without treating it as fetchable.

        Subassemblies are semantic task-graph objects rather than independent
        pegboard objects. Their visible highlight is therefore formed from the
        original physical components currently contained in that assembly.
        """
        producer = self.graph.producer_for(label)
        current = self.graph.current_container(label)
        if label in self.graph.active_parts:
            current = label

        highlight_container = current or label
        components = sorted(
            part for part in PROVIDED_PARTS
            if self.graph.current_container(part) == highlight_container
        )
        friendly = self.graph.friendly_part(label)
        current_friendly = (
            self.graph.friendly_part(current) if current is not None else "")
        selected = (
            self.graph.by_id.get(self.selected_id) if self.selected_id else None)
        selected_items = (
            set((*selected.inputs, *selected.context)) if selected else set())
        relevant = bool(
            selected and (label in selected_items or current in selected_items))

        if current == label:
            bearing_stand = re.fullmatch(
                r"BEARING_STAND_ROW([1-4])_(LEFT|RIGHT)_ASSEMBLY", label)
            if bearing_stand:
                row, side = bearing_stand.groups()
                spoken = (
                    f"Yes. The Row {row} {side.lower()} stand was already "
                    f"supplied. You fitted its bearing, so it is now {friendly}.")
            elif producer is not None:
                made_by = self.graph.friendly_step_action(producer)
                spoken = f"You already created {friendly} when you {made_by}."
            else:
                spoken = f"{friendly.capitalize()} is already assembled."
            if relevant and selected is not None:
                spoken += (
                    f" It is ready for this step: "
                    f"{self.graph.friendly_step_action(selected)}.")
                decision = "active_subassembly_relevant"
                color = [0.0, 1.0, 1.0, 0.2]
                warning = False
            elif selected is not None:
                spoken += (
                    f" It is not needed for this step. This step is to "
                    f"{self.graph.friendly_step_action(selected)}.")
                decision = "active_subassembly_not_relevant"
                color = [1.0, 0.0, 0.0, 0.25]
                warning = True
            else:
                decision = "active_subassembly"
                color = [1.0, 0.92, 0.02, 1.0]
                warning = False
            if result.get("part_action") == "fetch":
                spoken += " It is on the gearbox, so the robot does not need to fetch it."
            self._emit_reference_decision(
                result, decision, color, spoken,
                matched_parts=[label], current_assemblies=[label],
                assembly_parts=components, warning=warning)
            return

        if current is not None:
            spoken = (
                f"{friendly.capitalize()} has already been used. It is now "
                f"part of {current_friendly}. I have highlighted it on the gearbox.")
            self._emit_reference_decision(
                result, "subassembly_consumed", [1.0, 0.92, 0.02, 1.0],
                spoken, matched_parts=[label], current_assemblies=[current],
                assembly_parts=components)
            return

        if producer is None:
            spoken = "I could not locate that subassembly in the task graph."
            decision = "subassembly_unknown"
        elif self.graph.state(producer) == "ready":
            spoken = (
                f"{friendly.capitalize()} has not been assembled yet. You can "
                f"make it now: {self.graph.friendly_step_action(producer)}.")
            decision = "subassembly_not_created_ready"
        else:
            missing = self._friendly_missing(self.graph.missing(producer))
            spoken = (
                f"{friendly.capitalize()} has not been assembled yet. First, "
                f"{missing}.")
            decision = "subassembly_not_created_blocked"
        self._emit_reference_decision(
            result, decision, [1.0, 0.72, 0.0, 0.2], spoken,
            matched_parts=[label], warning=True)

    def _handle_step_tools_request(self, result: dict[str, object]) -> None:
        """Report/fetch the fastening tools required by a selected or next step."""
        scope = str(result.get("step_scope", ""))
        if scope == "selected_step":
            if not self.selected_id:
                self._emit_reference_decision(
                    result, "no_selected_step", [0.0, 1.0, 1.0, 0.2],
                    "Please select an assembly step first.", warning=True)
                return
            step = self.graph.by_id[self.selected_id]
        elif scope == "next_step":
            step = self.graph.recommend_next_step()
            if step is None:
                self._emit_reference_decision(
                    result, "no_next_step", [0.0, 1.0, 1.0, 0.2],
                    "There is no ready next step right now.", warning=True)
                return
        else:
            self._emit_reference_decision(
                result, "invalid_step_scope", [0.0, 1.0, 1.0, 0.2],
                "Please say whether you mean this step or the next step.",
                warning=True)
            return

        coords = self.graph.control_coords_for(step.id)
        required = list(
            gearbox_control.fastening_tool_labels(step.row)
            if coords is not None and coords[1] in (4, 6) else ())
        if not required:
            self._emit_reference_decision(
                result, "no_step_tools", [0.0, 1.0, 1.0, 0.2],
                "This step does not require a fastening tool.")
            return

        retained = self._tools_retained_from_completed_steps()
        supplied = [label for label in required
                    if label in self._robot_part_states or label in retained]
        outstanding = [label for label in required
                       if label not in self._robot_part_states
                       and label not in retained]
        friendly = [self._friendly_reference_label(label, [])
                    for label in required]
        friendly_text = (friendly[0] if len(friendly) == 1 else
                         " and ".join(friendly))
        outstanding_ids = sorted({
            tool_id
            for label in outstanding
            for tool_id in gearbox_control.tool_ids_for_reference(
                self._tool_index, label)
        })
        action = str(result.get("part_action", "reference"))

        if not outstanding:
            directly_delivered = all(
                label in self._robot_part_states
                and self._robot_part_states[label].get("source")
                != "task_progression"
                for label in required)
            if directly_delivered:
                spoken = (f"Yes. The {friendly_text} "
                          f"{'has' if len(required) == 1 else 'have'} already "
                          "been delivered to you.")
                decision = "step_tools_already_delivered"
            else:
                prior_steps = list(dict.fromkeys(
                    self.graph.friendly_step_action(retained[label])
                    for label in required if label in retained))
                prior_text = " and ".join(prior_steps)
                spoken = (f"You should already have the {friendly_text}. "
                          f"{'It was' if len(required) == 1 else 'They were'} "
                          f"used earlier to {prior_text} and should still be "
                          "with you.")
                decision = "step_tools_retained_by_operator"
            self._emit_reference_decision(
                result, decision,
                [0.0, 1.0, 1.0, 0.2], spoken,
                matched_parts=required)
            return

        outstanding_friendly = [self._friendly_reference_label(label, [])
                                for label in outstanding]
        outstanding_text = (outstanding_friendly[0]
                            if len(outstanding_friendly) == 1 else
                            " and ".join(outstanding_friendly))
        if action == "fetch":
            if self.graph.state(step) != "ready":
                spoken = "This step is not ready, so I will not fetch its tools yet."
                self._emit_reference_decision(
                    result, "fetch_blocked_step_tools",
                    [1.0, 0.72, 0.0, 0.2], spoken,
                    ids=outstanding_ids, matched_parts=outstanding,
                    warning=True)
                return
            # A fastening operation can require a driver plus an insert. The
            # robot fetches one physical object per confirmation, so offer the
            # primary driver first instead of asking the user to name it again.
            primary_label = outstanding[0]
            primary_ids = gearbox_control.tool_ids_for_reference(
                self._tool_index, primary_label)
            if len(primary_ids) == 1:
                primary_friendly = self._friendly_reference_label(
                    primary_label, [])
                if len(outstanding) > 1:
                    insert_text = " and ".join(
                        self._friendly_reference_label(label, [])
                        for label in outstanding[1:])
                    primary_friendly += f", used with the {insert_text}"
                self._arm_fetch_confirmation(
                    result, primary_ids, primary_friendly,
                    matched_parts=[primary_label])
                return
            spoken = (f"This step still needs the {outstanding_text}. "
                      "Please name the one you want the robot to get first.")
            self._emit_reference_decision(
                result, "fetch_step_tools_not_specific",
                [0.0, 1.0, 1.0, 0.2], spoken,
                ids=outstanding_ids, matched_parts=outstanding)
            return

        if supplied:
            supplied_text = " and ".join(
                self._friendly_reference_label(label, []) for label in supplied)
            spoken = (f"You should already have the {supplied_text}. "
                      f"You still need the {outstanding_text}. "
                      "I have highlighted it.")
        elif action == "status":
            spoken = (f"No. This step requires the {friendly_text}, and "
                      f"{'it has' if len(required) == 1 else 'they have'} not "
                      "been delivered yet. I have highlighted them.")
        else:
            spoken = (f"This step requires the {friendly_text}. "
                      "They have not been delivered yet, so I have highlighted them.")
        self._emit_reference_decision(
            result, "step_tools_outstanding", [0.0, 1.0, 1.0, 0.2],
            spoken, ids=outstanding_ids, matched_parts=required)

    def _handle_step_parts_request(self, result: dict[str, object]) -> None:
        """Resolve task-relative phrases such as "parts for the next step"."""
        scope = str(result.get("step_scope", ""))
        if scope == "selected_step":
            if not self.selected_id:
                self._emit_reference_decision(
                    result, "no_selected_step", [0.0, 1.0, 1.0, 0.2],
                    "Please select an assembly step first.", warning=True)
                return
            step = self.graph.by_id[self.selected_id]
            scope_text = "This step"
        elif scope == "next_step":
            # Never skip over the user's current work. "Next" advances only
            # after the selected step has actually been marked complete.
            if self.selected_id:
                selected = self.graph.by_id[self.selected_id]
                if self.graph.state(selected) != "complete":
                    self._emit_reference_decision(
                        result,
                        "current_step_incomplete",
                        [1.0, 0.72, 0.0, 0.2],
                        ("The current step is not complete yet. Finish it and "
                         "mark it complete before asking for the next step."),
                        warning=True,
                    )
                    return
            step = self.graph.recommend_next_step()
            if step is None:
                self._emit_reference_decision(
                    result, "no_next_step", [0.0, 1.0, 1.0, 0.2],
                    "There is no other ready assembly step right now.",
                    warning=True)
                return
            scope_text = "The next step"
        else:
            self._emit_reference_decision(
                result, "invalid_step_scope", [0.0, 1.0, 1.0, 0.2],
                "Please say whether you mean this step or the next step.",
                warning=True)
            return

        excluded_parts: set[str] = set()
        for label in result.get("exclude_labels", []):
            excluded_parts.update(self.graph.parts_for_reference_label(str(label)))
        supplied_inputs = [part for part in step.inputs
                           if part in self._robot_part_states
                           and part not in excluded_parts]
        active_inputs = [part for part in step.inputs
                         if part in self.graph.active_parts
                         and part not in excluded_parts
                         and part not in self._robot_part_states]
        if not active_inputs:
            if supplied_inputs:
                descriptions = [self.graph.friendly_part(part)
                                for part in supplied_inputs]
                supplied_text = (descriptions[0] if len(descriptions) == 1
                                 else ", and ".join(descriptions))
                self._emit_reference_decision(
                    result,
                    ("no_other_next_step_parts" if scope == "next_step"
                     else "no_other_selected_step_parts"),
                    [0.0, 1.0, 1.0, 0.2],
                    f"All required parts are already supplied. You have {supplied_text}.",
                    matched_parts=supplied_inputs,
                )
                return
            self._emit_reference_decision(
                result,
                ("no_other_next_step_parts" if scope == "next_step"
                 else "no_other_selected_step_parts"),
                [0.0, 1.0, 1.0, 0.2],
                f"There are no other active parts required for {scope_text.lower()}.",
            )
            return
        raw_active = [part for part in active_inputs if part in PROVIDED_PARTS]
        component_parts: list[str] = list(raw_active)
        # When an input is already a subassembly, highlight its consumed
        # components on the assembled gearbox instead of looking for a removed
        # pegboard object.
        for assembly in active_inputs:
            if assembly in PROVIDED_PARTS:
                continue
            for part in PROVIDED_PARTS:
                if (self.graph.current_container(part) == assembly
                        and part not in component_parts):
                    component_parts.append(part)

        ids = [gearbox_control.tool_id_for_graph_part(self._tool_index, part)
               for part in raw_active]
        ids = sorted({tool_id for tool_id in ids if tool_id is not None})
        descriptions = [self.graph.friendly_part(part) for part in active_inputs]
        if not descriptions:
            needed_text = "its required active assembly"
        elif len(descriptions) == 1:
            needed_text = descriptions[0]
        elif len(descriptions) == 2:
            needed_text = f"{descriptions[0]} and {descriptions[1]}"
        else:
            needed_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
        action = self.graph.friendly_step_action(step)
        if result.get("part_action") == "fetch":
            if self.graph.state(step) != "ready":
                spoken = (
                    f"{scope_text} is not ready, so I will not fetch its parts yet.")
                self._emit_reference_decision(
                    result, "fetch_blocked_step_parts",
                    [1.0, 0.72, 0.0, 0.2], spoken,
                    ids=ids, matched_parts=active_inputs,
                    assembly_parts=component_parts, warning=True)
                return
            if not raw_active:
                available_text = (
                    descriptions[0] if len(descriptions) == 1 else needed_text)
                supplied_prefix = ""
                if supplied_inputs:
                    supplied_descriptions = [
                        self.graph.friendly_part(part)
                        for part in supplied_inputs
                    ]
                    supplied_text = (
                        supplied_descriptions[0]
                        if len(supplied_descriptions) == 1
                        else ", and ".join(supplied_descriptions))
                    supplied_prefix = (
                        f"{supplied_text.capitalize()} "
                        f"{'has' if len(supplied_descriptions) == 1 else 'have'} "
                        "already been "
                        "supplied. ")
                spoken = (
                    f"{supplied_prefix}{available_text.capitalize()} is already "
                    "assembled and in the work area. There is no separate "
                    "pegboard part for the robot to fetch.")
                self._emit_reference_decision(
                    result, "step_parts_already_in_work_area",
                    [0.0, 1.0, 1.0, 0.2], spoken,
                    matched_parts=active_inputs,
                    current_assemblies=active_inputs,
                    assembly_parts=component_parts)
                return
            if len(active_inputs) == 1 and len(ids) == 1:
                friendly = descriptions[0]
                if friendly.lower().startswith("the "):
                    friendly = friendly[4:]
                self._arm_fetch_confirmation(
                    result, ids, friendly,
                    matched_parts=active_inputs,
                    assembly_parts=component_parts)
                if self._vlm is not None:
                    self._vlm.set_resolved_part_context(active_inputs[0])
                return
            spoken = (
                f"You still need {needed_text}. Please name the one you want "
                "the robot to get.")
            self._emit_reference_decision(
                result, "fetch_step_parts_not_specific",
                [0.0, 1.0, 1.0, 0.2], spoken,
                ids=ids, matched_parts=active_inputs,
                assembly_parts=component_parts)
            if self._vlm is not None:
                self._vlm.set_resolved_part_candidates(active_inputs)
            return

        highlight_pronoun = "it" if len(active_inputs) == 1 else "them"
        supplied_prefix = ""
        if supplied_inputs:
            supplied_descriptions = [self.graph.friendly_part(part)
                                     for part in supplied_inputs]
            already_text = (supplied_descriptions[0]
                            if len(supplied_descriptions) == 1
                            else ", and ".join(supplied_descriptions))
            statuses = {
                str(self._robot_part_states[part].get("status", ""))
                for part in supplied_inputs
            }
            location = ("is already in the robot gripper"
                        if "in_robot_gripper" in statuses
                        else "has already been handed to you")
            supplied_prefix = f"{already_text.capitalize()} {location}. "
        spoken = (
            f"{supplied_prefix}{scope_text} is to {action}. "
            f"You still need {needed_text}. "
            f"I have highlighted {highlight_pronoun}.")
        self._emit_reference_decision(
            result,
            "next_step_parts" if scope == "next_step" else "selected_step_parts",
            [0.0, 1.0, 1.0, 0.2],
            spoken,
            ids=ids,
            matched_parts=active_inputs,
            assembly_parts=component_parts,
        )
        if self._vlm is not None:
            self._vlm.set_resolved_part_candidates(active_inputs)

    def _friendly_missing(self, missing: list[str]) -> str:
        descriptions = []
        for item in missing[:3]:
            if item in self.graph.by_id:
                descriptions.append(
                    self.graph.friendly_step_action(self.graph.by_id[item]))
            else:
                producer = self.graph.producer_for(item)
                descriptions.append(
                    self.graph.friendly_step_action(producer)
                    if producer is not None else self.graph.friendly_part(item))
        if not descriptions:
            return "the earlier steps"
        if len(descriptions) == 1:
            return descriptions[0]
        return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

    def _fastening_tool_guidance(self, step: Step) -> str:
        coords = self.graph.control_coords_for(step.id)
        if coords is None or coords[1] not in (4, 6):
            return ""
        return {
            1: "the H5 bit from Holder 1 with the bit wrench",
            2: "the T25 bit from Holder 2 with the bit screwdriver",
            3: "the H3 bit from Holder 1 with the bit wrench",
            4: "the Phillips screwdriver",
        }.get(step.row, "")

    def _announce_step_selection(self, step: Step) -> None:
        state = self.graph.state(step)
        action = self.graph.friendly_step_action(step)
        if state == "blocked":
            missing = self._friendly_missing(self.graph.missing(step))
            self._speak(
                f"This step is not ready. First, {missing}.",
                warning=True)
        elif state == "complete":
            self._speak("This step is already complete.")
        else:
            tool_guidance = self._fastening_tool_guidance(step)
            spoken_action = action[0].upper() + action[1:]
            suffix = f" Use {tool_guidance}." if tool_guidance else ""
            self._speak(f"This step is ready. {spoken_action}.{suffix}")

    def _focus_vlm_on_step(self, step: Step) -> None:
        """Keep GUI- and Unity-originated selections identical for the VLM."""
        if self._vlm is None:
            return
        state = self.graph.state(step)
        missing = self.graph.missing(step)
        coords = self.graph.control_coords_for(step.id)
        fastening_labels = (gearbox_control.fastening_tool_labels(step.row)
                            if coords is not None and coords[1] in (4, 6)
                            else ())
        focused = (
            f"[{step.id}] {step.title}\n"
            f"State: {state.upper()}\n"
            f"Description: {step.description}\n"
            f"Inputs: {', '.join(step.inputs)}\n"
            f"Context: {', '.join(step.context) or '(none)'}\n"
            f"Fastening tools: {', '.join(fastening_labels) or '(none)'}\n"
            f"Produces: {step.output}"
        )
        if missing:
            focused += f"\nBlocked by: {', '.join(missing)}"
        self._vlm.set_focused_step(focused)

    def _notify_vlm(self, event_label: str) -> None:
        """Replace the VLM's stored graph context after a state-changing event."""
        self._clear_pending_fetch()
        # Once an assembly step consumes a supplied raw part, its location is
        # represented by transform_history/current_container instead of the
        # robot-supply layer.
        for part in list(self._robot_part_states):
            if (part in PROVIDED_PARTS
                    and part not in self.graph.active_parts
                    and self._robot_part_states[part].get("source")
                    != "task_progression"):
                self._robot_part_states.pop(part, None)
        # A completed fastening step proves that its tools were available to
        # the operator. Add/remove those inferred handovers after every graph
        # change so complete, undo, live events, and reset behave identically.
        self._sync_progress_inferred_supply_statuses()
        self._publish_progress_handover_state()
        if self._vlm is not None:
            self._vlm.notify_graph_event(
                event_label, self._assistant_state_summary())

    def _assistant_state_summary(self) -> str:
        """Combine assembly progress with live robot grasp/handover state."""
        summary = self.graph.state_summary()
        lines = [summary]
        if self._robot_part_states:
            lines.extend(["", "ROBOT-SUPPLIED PART/TOOL STATE:"])
            for part in sorted(self._robot_part_states):
                info = self._robot_part_states[part]
                status = str(info.get("status", "")).upper()
                if info.get("source") == "task_progression":
                    lines.append(
                        f"  - {part}: {status} (inferred from completed step "
                        f"{info.get('inferred_from_step', 'unknown')})")
                else:
                    lines.append(
                        f"  - {part}: {status} (physical object: "
                        f"{info.get('tool_name', 'unknown')}, "
                        f"id={info.get('tool_id')})")
            lines.append(
                "Items marked IN_ROBOT_GRIPPER or HANDED_OVER are already "
                "supplied. Do not offer to fetch them again or describe them "
                "as outstanding. Supplied raw parts remain active task inputs "
                "until an assembly step consumes them; supplied tools remain "
                "available to the operator.")
        retained = self._tools_retained_from_completed_steps()
        retained = {label: step for label, step in retained.items()
                    if label not in self._robot_part_states}
        if retained:
            lines.extend([
                "",
                "TOOLS RETAINED BY THE OPERATOR FROM EARLIER STEPS:",
            ])
            for label, step in sorted(retained.items()):
                lines.append(
                    f"  - {label}: already used in {step.id} and assumed to "
                    "remain with the operator")
            lines.append(
                "Treat these tools as already available for later fastening "
                "steps. Do not claim that a separate robot handover was "
                "recorded unless their status above is HANDED_OVER.")
        return "\n".join(lines)

    # ── User actions and GUI callbacks ───────────────────────────────────────

    def _recommend_callback(self) -> None:
        """Select the preferred READY step in the GUI and Unity."""
        self._activate_recommended_step()

    def _activate_recommended_step(self, question: str = "") -> None:
        """Apply the graph recommendation as a synchronized step selection."""
        step = self.graph.recommend_next_step()
        if step is None:
            spoken = "There is no ready assembly step to recommend right now."
            self.log("[Recommend] No READY steps — assembly may be complete or stalled.")
            self._speak(spoken, warning=True)
            if question and self._vlm is not None:
                self._vlm.apply_policy_response(
                    question, "RECOMMENDED_STEP", spoken)
            return
        self.recommended_id = step.id
        self._select_step(step.id, announce=False)
        if self.controller is not None:
            self._animate_unity_callback()
        action = self.graph.friendly_step_action(step)
        tool_guidance = self._fastening_tool_guidance(step)
        suffix = f" Use {tool_guidance}." if tool_guidance else ""
        spoken = f"I selected the recommended step. {action.capitalize()}.{suffix}"
        self.log(f"[Recommend] Auto-selected: [{step.id}]  {step.title}")
        self._speak(spoken)
        if question and self._vlm is not None:
            self._vlm.apply_policy_response(
                question, "RECOMMENDED_STEP", spoken)

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
        # A Unity checkbox click after programmatic selection must apply to the
        # same recommended stage, just as if the user had clicked its part.
        self.controller.sm.current_row = row
        self.controller.sm.current_stage = stage
        self.controller.send({"command": "stage", "row": row, "stage": stage,
                               "done_stages": done_stages,
                               "step_delay": gearbox_control.STEP_DELAY,
                               "slide_seconds": gearbox_control.SLIDE_SECONDS})
        self.controller.send({"command": "ui", "show": True, "row": row,
                               "checked": checked, "blocked": blocked})
        self.log(f"[Animate] Unity → row {row}, stage {stage}  [{step.id}]")

    def _select_callback(self, _sender, _app_data, user_data) -> None:
        """Select a graph step and update external viewers and VLM focus."""
        self._select_step(user_data, announce=True)

    def _select_step(self, step_id: str, *, announce: bool) -> None:
        """Synchronize one selection with DearPyGui and external 3D viewers."""
        self._clear_pending_fetch()
        self.selected_id = step_id
        self.refresh()
        step = self.graph.by_id[step_id]
        if announce:
            self._announce_step_selection(step)
        if self.controller is not None:
            self.controller.send({"command": "reference_clear"})
        coords = TaskGraph.control_coords_for(step_id)
        if coords is not None:
            row, stage = coords
            index = gearbox_control.load_tool_index(gearbox_control._DEFAULT_TOOL_JSON)
            ids = (gearbox_control.appearing_ids(index, row, stage)
                   if row > 0 else [])
            self._send_select({"event": "select", "row": row, "stage": stage,
                               "step": step_id, "ids": ids,
                               "blocked": self.graph.state(step) == "blocked"})
        self._focus_vlm_on_step(step)

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
        was_complete = self.graph.state(step) == "complete"
        if was_complete:
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
        if ok:
            coords = TaskGraph.control_coords_for(self.selected_id)
            if coords is not None:
                self._send_select({"event": "uncomplete" if was_complete else "complete",
                                   "row": coords[0], "stage": coords[1],
                                   "step": self.selected_id})
        self.log(message)
        if ok and was_complete:
            self._speak(f"Undid {self.graph.friendly_step(step)}.")
        elif ok:
            self._speak(
                f"Step complete. You have created "
                f"{self.graph.friendly_part(step.output)}.")
        else:
            self._speak(
                f"That action is not available. {self._friendly_missing(self.graph.missing(step))} "
                "must be completed first.", warning=True)
        self.refresh()

    def _reset_state(self, *, clear_handovers: bool) -> None:
        """Reset progress, optionally treating the physical inventory as restocked."""
        self.graph.reset()
        self.selected_id    = None
        self.recommended_id = None
        if clear_handovers:
            self._robot_part_states.clear()
        self._send_select({"event": "reset"})
        if clear_handovers:
            self._send_select({"event": "reset_all_handovers"})
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
        if clear_handovers:
            self.log("All assembly progress and handover records were reset.")
            self._speak(
                "Assembly progress and handed-over inventory have been reset.",
                warning=True)
        else:
            self.log("All assembly progress was reset.")
            self._speak("Assembly progress has been reset.", warning=True)
        self.refresh()
        self._send_select({"event": "clear"})
        event = ("RESET: progress and all handovers cleared"
                 if clear_handovers else "RESET: all progress cleared")
        self._notify_vlm(event)

    def _reset_callback(self) -> None:
        """Reset task progress while preserving confirmed robot handovers."""
        self._reset_state(clear_handovers=False)

    def _reset_progress_and_handovers_callback(self) -> None:
        """Reset progress and repopulate the complete physical inventory."""
        self._reset_state(clear_handovers=True)

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
    recommendation_graph = TaskGraph()
    assert recommendation_graph.recommend_next_step().id == "r1_bearing_right"
    recommendation_graph.complete(recommendation_graph.by_id["r1_bearing_right"])
    # Both Stage 2 and Stage 3 are now READY; the documented controller-stage
    # ordering must select gear-rod Stage 2 before left-bearing Stage 3.
    assert recommendation_graph.recommend_next_step().id == "r1_gear_rod"
    recommendation_graph.complete(recommendation_graph.by_id["r1_gear_rod"])
    assert recommendation_graph.recommend_next_step().id == "r1_bearing_left"

    # Recommendation follows the row most recently changed by completion or
    # undo, even while lower-numbered rows also contain READY work.
    focus_graph = TaskGraph()
    ok, _ = focus_graph.complete(focus_graph.by_id["r2_bearing_right"])
    assert ok and focus_graph.last_worked_row == 2
    assert focus_graph.recommend_next_step().id == "r2_gear_rod"
    ok, _ = focus_graph.complete(focus_graph.by_id["r3_bearing_right"])
    assert ok and focus_graph.last_worked_row == 3
    assert focus_graph.recommend_next_step().id == "r3_gear_rod"
    ok, _ = focus_graph.undo(focus_graph.by_id["r2_bearing_right"])
    assert ok and focus_graph.last_worked_row == 2
    assert focus_graph.recommend_next_step().id == "r2_bearing_right"
    focus_graph.reset()
    assert focus_graph.last_worked_row is None
    assert focus_graph.recommend_next_step().id == "r1_bearing_right"

    graph = TaskGraph()
    assert graph.is_ready(graph.by_id["r1_bearing_right"])
    assert not graph.is_ready(graph.by_id["r1_bearing_left"])
    ok, _ = graph.complete(graph.by_id["r1_bearing_right"])
    assert ok
    assert graph.is_ready(graph.by_id["r1_bearing_left"])
    ok, _ = graph.complete(graph.by_id["r1_bearing_left"])
    assert ok and "BEARING_STAND_ROW1_LEFT_ASSEMBLY" in graph.active_parts
    ok, warning = graph.undo(graph.by_id["r1_bearing_right"])
    assert not ok and "r1_bearing_left" in warning
    ok, _ = graph.undo(graph.by_id["r1_bearing_left"])
    assert ok
    ok, _ = graph.undo(graph.by_id["r1_bearing_right"])
    assert ok and "BEARING_ROW1_RIGHT" in graph.active_parts
    graph.complete(graph.by_id["r1_bearing_right"])
    graph.complete(graph.by_id["r1_bearing_left"])
    graph.complete(graph.by_id["r1_fasten_first_stand"])
    # Fastening the right stand does not consume or depend on the independently
    # assembled left stand, so the left step remains undoable here.
    ok, _ = graph.undo(graph.by_id["r1_bearing_left"])
    assert ok
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
    parser.add_argument(
        "--voice-device", default=DEFAULT_PULSEAUDIO_DEVICE,
        help=("PulseAudio source name for voice input "
              f"(default: {DEFAULT_PULSEAUDIO_DEVICE})."))
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice input.")
    parser.add_argument(
        "--wake-word", default="",
        help="Optional wake word. By default speech is accepted continuously without one.")
    parser.add_argument("--vlm-model", default=None,
                        help="Enable VLM assistant with this model name "
                             "(for example Qwen/Qwen3-VL-8B-Instruct). Omit to disable.")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable spoken task guidance and warnings.")
    parser.add_argument("--tts-engine", choices=("piper", "nemo"), default="piper",
                        help="Speech-output backend (default: piper).")
    parser.add_argument("--piper-model", default=str(DEFAULT_PIPER_MODEL),
                        help="Piper voice .onnx file used for spoken guidance.")
    parser.add_argument(
        "--tts-rate", type=float, default=0.85,
        help="Piper speaking-rate multiplier; values below 1.0 speak more slowly.")
    parser.add_argument(
        "--assistant-log",
        default=str(Path(__file__).resolve().parent / "task_assistant_interactions.csv"),
        help="Append part-reference decisions to this CSV file.")
    parser.add_argument("--no-assistant-log", action="store_true",
                        help="Disable part-reference study logging.")
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
    if args.tts_rate <= 0.0:
        parser.error("--tts-rate must be greater than zero.")
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
    controller_click_thread = None
    if args.with_controller:
        if args.no_live:
            parser.error("--with-controller needs the live link; do not pass --no-live.")
        controller = gearbox_control.GearboxController(
            args.unity_ip, args.cmd_port, args.click_port,
            _LOCALHOST, _DEFAULT_CTRL_EVENTS_IN_PORT, no_highlight=args.no_highlight)
        app.controller = controller
        controller_click_thread = threading.Thread(
            target=controller.run_click_loop, daemon=True)
        controller_click_thread.start()
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
            tts_engine=None if args.no_tts else args.tts_engine,
            piper_model=args.piper_model,
            tts_rate=args.tts_rate,
            assistant_log_path=(None if args.no_assistant_log
                                else args.assistant_log),
        )
    finally:
        if controller is not None:
            controller.stop()
            if controller_click_thread is not None:
                controller_click_thread.join(timeout=1.0)
            controller.close()


if __name__ == "__main__":
    main()
