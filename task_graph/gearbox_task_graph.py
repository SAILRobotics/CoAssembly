#!/usr/bin/env python3
"""Interactive gearbox assembly task graph.

The model is intentionally independent from the GUI so that the dependency and
part-transformation rules can be tested without opening a window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    row: int
    stage: int
    inputs: tuple[str, ...]
    output: str
    description: str
    context: tuple[str, ...] = ()


PROVIDED_PARTS = {
    "BASE_BOARD",
    "BEARING_ROW1_LEFT", "BEARING_ROW1_RIGHT", "CRANK_HANDLE_ROW1",
    "GEAR_ROD_ROW1", "GEAR_ROW1_LEFT", "PIN_ROW1_LEFT", "PIN_ROW1_RIGHT",
    "SCREW_ROW1_LEFT", "SCREW_ROW1_RIGHT", "STAND_ROW1_LEFT", "STAND_ROW1_RIGHT",
    "BEARING_ROW2_LEFT", "BEARING_ROW2_RIGHT", "GEAR_ROD_ROW2",
    "GEAR_ROW2_LEFT", "GEAR_ROW2_RIGHT", "PIN_ROW2_LEFT", "PIN_ROW2_RIGHT",
    "SCREW_ROW2_LEFT", "SCREW_ROW2_RIGHT", "STAND_ROW2_LEFT", "STAND_ROW2_RIGHT",
    "BEARING_ROW3_LEFT", "BEARING_ROW3_RIGHT", "GEAR_ROD_ROW3",
    "GEAR_ROW3_LEFT", "GEAR_ROW3_RIGHT", "PIN_ROW3_LEFT", "PIN_ROW3_RIGHT",
    "SCREW_ROW3_LEFT", "SCREW_ROW3_RIGHT", "STAND_ROW3_LEFT", "STAND_ROW3_RIGHT",
    "BEARING_ROW4_LEFT", "BEARING_ROW4_RIGHT", "GEAR_ROD_ROW4",
    "GEAR_ROW4_LEFT", "PIN_ROW4_LEFT", "SCREW_ROW4_LEFT", "SCREW_ROW4_RIGHT",
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
                title=f"Row {row}: bearing into {side.lower()} stand",
                row=row,
                stage=0,
                inputs=(f"BEARING_ROW{row}_{side.upper()}", f"STAND_ROW{row}_{side.upper()}"),
                output=f"BEARING_STAND_ROW{row}_{side.upper()}_ASSEMBLY",
                description=(f"Insert BEARING_ROW{row}_{side.upper()} into "
                             f"STAND_ROW{row}_{side.upper()}."),
            ))

        steps.append(Step(
            id=f"r{row}_gear_rod",
            title=f"Row {row}: assemble gear rod",
            row=row,
            stage=1,
            inputs=gear_inputs[row],
            output=f"GEAR_ROD_ROW{row}_ASSEMBLY",
            description=gear_text[row],
        ))

        first, second = "Left", "Right"
        steps.append(Step(
            id=f"r{row}_fasten_first_stand",
            title=f"Row {row}: fasten {first.lower()} stand",
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
            title=f"Row {row}: insert rod and fit {second.lower()} stand",
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
            title=f"Row {row}: fasten {second.lower()} stand",
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
        title="Row 1: attach crank handle",
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


def validate_model() -> list[str]:
    """Return model problems; missing raw inventory is reported separately by the GUI."""
    graph = TaskGraph()
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


class TkTaskGraphApp:
    COLORS = {"complete": "#2e9d60", "ready": "#2784d8", "blocked": "#d18a27"}

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.graph = TaskGraph()
        self.selected_id: str | None = None
        root.title("Gearbox Assembly Task Graph")
        root.geometry("1500x900")
        root.minsize(1080, 680)

        outer = ttk.Panedwindow(root, orient="horizontal")
        outer.pack(fill="both", expand=True, padx=8, pady=8)
        graph_frame = ttk.Frame(outer)
        side = ttk.Frame(outer, width=390)
        outer.add(graph_frame, weight=4)
        outer.add(side, weight=1)

        ttk.Label(graph_frame, text="Gearbox Assembly Dependency Graph",
                  font=("TkDefaultFont", 15, "bold")).pack(anchor="w", pady=(0, 5))
        legend = ttk.Label(graph_frame,
                           text="Blue = ready   Orange = blocked   Green = complete   Click a node for details")
        legend.pack(anchor="w")
        self.canvas = tk.Canvas(graph_frame, bg="#f5f6f8", highlightthickness=1,
                                highlightbackground="#b8bec8", scrollregion=(0, 0, 1800, 900))
        xbar = ttk.Scrollbar(graph_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(graph_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.pack(fill="both", expand=True, side="left", pady=(6, 0))
        ybar.pack(fill="y", side="right", pady=(6, 0))
        xbar.pack(fill="x", side="bottom")

        ttk.Label(side, text="Selected step", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.details = tk.Text(side, height=13, wrap="word", state="disabled")
        self.details.pack(fill="x", pady=5)
        self.complete_button = ttk.Button(side, text="Mark selected step complete",
                                          command=self.complete_selected)
        self.complete_button.pack(fill="x")
        ttk.Button(side, text="Reset all progress", command=self.reset).pack(fill="x", pady=(4, 10))

        ttk.Label(side, text="Active assembled parts",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.assemblies = tk.Listbox(side, height=8)
        self.assemblies.pack(fill="x", pady=(4, 10))

        ttk.Label(side, text="Part / step terminal",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.terminal = tk.Text(side, height=13, bg="#17202a", fg="#e8f0f7",
                                insertbackground="white", wrap="word", state="disabled")
        self.terminal.pack(fill="both", expand=True, pady=4)
        self.command = ttk.Entry(side)
        self.command.pack(fill="x")
        self.command.bind("<Return>", self.run_command)
        self.command.focus_set()

        self.positions: dict[str, tuple[float, float]] = {}
        self.draw_graph()
        self.refresh()
        self.log("Type 'parts' to list the current parts, or type 'help' for all commands.")

    def log(self, message: str) -> None:
        self.terminal.configure(state="normal")
        self.terminal.insert("end", message + "\n")
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def draw_graph(self) -> None:
        self.canvas.delete("all")
        stage_x = {0: 145, 1: 405, 2: 665, 3: 925, 4: 1185, 5: 1425, 6: 1645}
        headers = {0: "Prepare stands", 1: "Assemble rods", 2: "Fasten first stand",
                   3: "Insert rod & fit stand", 4: "Fasten second stand",
                   5: "Add handle", 6: "Verify"}
        for stage, title in headers.items():
            self.canvas.create_text(stage_x[stage], 25, text=title, font=("TkDefaultFont", 10, "bold"))

        stage_counts: dict[int, int] = {}
        for step in self.graph.steps:
            if step.stage == 0:
                side_offset = 0 if step.id.endswith("left") else 75
                y = 90 + (step.row - 1) * 190 + side_offset
            elif step.row:
                y = 125 + (step.row - 1) * 190
            else:
                y = 410
            stage_counts[step.stage] = stage_counts.get(step.stage, 0) + 1
            self.positions[step.id] = (stage_x[step.stage], y)

        producer = {s.output: s.id for s in self.graph.steps}
        for step in self.graph.steps:
            x2, y2 = self.positions[step.id]
            for part in step.inputs:
                source = producer.get(part)
                if source:
                    x1, y1 = self.positions[source]
                    self.canvas.create_line(x1 + 102, y1, x2 - 102, y2, arrow="last",
                                            width=2, fill="#8b95a1")

        for step in self.graph.steps:
            x, y = self.positions[step.id]
            state = self.graph.state(step)
            outline = "#1b2631" if step.id == self.selected_id else "#ffffff"
            width = 4 if step.id == self.selected_id else 2
            tag = f"step:{step.id}"
            self.canvas.create_rectangle(x - 102, y - 31, x + 102, y + 31,
                                         fill=self.COLORS[state], outline=outline,
                                         width=width, tags=(tag, "node"))
            label = step.title.replace(": ", ":\n", 1)
            self.canvas.create_text(x, y, text=label, width=190, fill="white",
                                    justify="center", font=("TkDefaultFont", 9, "bold"),
                                    tags=(tag, "node"))
            self.canvas.tag_bind(tag, "<Button-1>", lambda _e, sid=step.id: self.select(sid))

    def select(self, step_id: str) -> None:
        self.selected_id = step_id
        self.refresh()

    def set_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def refresh(self) -> None:
        self.draw_graph()
        self.assemblies.delete(0, "end")
        for part in sorted(p for p in self.graph.active_parts
                           if p.endswith("_ASSEMBLY")):
            self.assemblies.insert("end", part)
        if not self.selected_id:
            self.set_details("Select a graph node to see its inputs, output, and readiness.")
            self.complete_button.configure(state="disabled")
            return
        step = self.graph.by_id[self.selected_id]
        missing = self.graph.missing(step)
        text = (f"{step.id}\n\n{step.description}\n\n"
                f"Inputs: {', '.join(step.inputs)}\n"
                f"Produces: {step.output}\n"
                f"State: {self.graph.state(step).upper()}")
        if missing and step.id not in self.graph.completed:
            text += "\nBlocked by: " + ", ".join(missing)
        self.set_details(text)
        self.complete_button.configure(
            state="normal" if self.graph.is_ready(step) else "disabled")

    def complete_selected(self) -> None:
        if not self.selected_id:
            return
        _, message = self.graph.complete(self.graph.by_id[self.selected_id])
        self.log(message)
        self.refresh()

    def reset(self) -> None:
        self.graph.reset()
        self.selected_id = None
        self.log("All assembly progress was reset.")
        self.refresh()

    def run_command(self, _event=None) -> None:
        raw = self.command.get().strip()
        self.command.delete(0, "end")
        if not raw:
            return
        self.log("> " + raw)
        command, _, argument = raw.partition(" ")
        command_lower = command.lower()
        if command_lower == "help":
            self.log("Commands: <part name>, part <name>, step <id/title>, complete <id/title>, "
                     "status, missing, add <part name>, reset, help")
        elif command_lower in {"part", "find"}:
            self.log(self.graph.trace_part(argument))
        elif command_lower == "step":
            self.inspect_steps(argument)
        elif command_lower == "complete":
            matches = self.graph.find_steps(argument)
            if len(matches) == 1:
                self.selected_id = matches[0].id
                _, message = self.graph.complete(matches[0])
                self.log(message)
                self.refresh()
            elif matches:
                self.log("Ambiguous step. Matches: " + ", ".join(s.id for s in matches))
            else:
                self.log(f"WARNING: no step matches '{argument}'.")
        elif command_lower == "status":
            ready = [s.id for s in self.graph.steps if self.graph.is_ready(s)]
            self.log(f"Completed {len(self.graph.completed)}/{len(self.graph.steps)}. "
                     f"Ready: {', '.join(ready) if ready else 'none'}")
        elif command_lower == "missing":
            missing_raw = sorted({p for s in self.graph.steps for p in self.graph.missing(s)
                                  if p not in {other.output for other in self.graph.steps}})
            self.log("Missing raw inventory: " + (", ".join(missing_raw) or "none"))
        elif command_lower == "add":
            if not argument:
                self.log("Usage: add <part name>")
            else:
                self.log(self.graph.add_part(argument.strip()))
                self.refresh()
        elif command_lower == "reset":
            self.reset()
        else:
            # A bare part name is the primary terminal interaction.
            self.log(self.graph.trace_part(raw))

    def inspect_steps(self, query: str) -> None:
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


class DearPyGuiTaskGraphApp:
    """Dear PyGui task-graph view and terminal controller."""

    COLORS = {
        "complete": (46, 157, 96, 255),
        "ready": (39, 132, 216, 255),
        "blocked": (209, 138, 39, 255),
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

    def build(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        dpg.create_viewport(title="Gearbox Assembly Task Graph", width=1540, height=920,
                            min_width=1080, min_height=700)
        self._create_themes()

        with dpg.window(tag="primary_window", label="Gearbox Assembly Task Graph"):
            dpg.add_text("Gearbox Assembly Dependency Graph", color=(225, 232, 240),
                         tag="main_title")
            dpg.add_text("Blue = ready     Orange = blocked     Green = complete",
                         color=(170, 180, 195))
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=3.2)
                dpg.add_table_column(init_width_or_weight=1.25)
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

        dpg.set_primary_window("primary_window", True)
        dpg.bind_item_theme("task_node_editor", "node_editor_theme")
        self.refresh()
        self.log("Type a part name, or type 'help' for commands.")
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self) -> None:
        self.build()
        self.dpg.start_dearpygui()
        self.dpg.destroy_context()

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
                           width=-1, height=205)
        dpg.add_input_text(tag="command_input", hint="Enter a part name or command...",
                           width=-1, on_enter=True, callback=self._command_callback)

    def _select_callback(self, _sender, _app_data, user_data) -> None:
        self.selected_id = user_data
        self.refresh()

    def _complete_selected(self) -> None:
        if not self.selected_id:
            return
        step = self.graph.by_id[self.selected_id]
        if self.graph.state(step) == "complete":
            _, message = self.graph.undo(step)
        else:
            _, message = self.graph.complete(step)
        self.log(message)
        self.refresh()

    def _reset_callback(self) -> None:
        self.graph.reset()
        self.selected_id = None
        self.log("All assembly progress was reset.")
        self.refresh()

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
        for step in self.graph.steps:
            state = self.graph.state(step)
            dpg.bind_item_theme(self.node_tags[step.id], self.themes[state])
            dpg.configure_item(self.node_tags[step.id],
                               label=f"{step.title}  [{state.upper()}]")
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
                _, message = self.graph.complete(matches[0])
                self.log(message)
                self.refresh()
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
            _, message = self.graph.undo(requested)
            self.log(message)
            self.refresh()
        elif command_lower == "add":
            if not argument:
                self.log("Usage: add <part name>")
            else:
                self.log(self.graph.add_part(argument.strip()))
                self.refresh()
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
    errors = validate_model()
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
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    DearPyGuiTaskGraphApp().run()


if __name__ == "__main__":
    main()
