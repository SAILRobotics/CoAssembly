#!/usr/bin/env python3
"""Read-only Dear PyGui selector for Babylon referring-expression subassemblies.

Run the Flask/Babylon app first, open its browser page, then run this file.
Clicking a task node's button displays that step's output subassembly in Babylon.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import dearpygui.dearpygui as dpg

from task_graph.gearbox_task_graph import TaskGraph


SERVER = "http://127.0.0.1:5000"


class ReferringExpressionTaskGraph:
    ROW_COLORS = {
        1: (210, 215, 225, 255),
        2: (220, 70, 70, 255),
        3: (45, 170, 80, 255),
        4: (55, 100, 225, 255),
    }

    def __init__(self) -> None:
        self.graph = TaskGraph()
        # The completed gearbox is the full model, not a subassembly study target.
        self.steps = [step for step in self.graph.steps
                      if step.output != "COMPLETED_GEARBOX_ASSEMBLY"]
        self.input_attrs: dict[str, str] = {}
        self.output_attrs: dict[str, str] = {}
        self.subassembly_buttons: list[str] = []
        self.saved_responses: list[dict[str, str]] = []
        self.review_position = 0

    def _set_individual_parts_only(self, _sender, app_data: bool, _user_data=None) -> None:
        enabled = not bool(app_data)
        body = json.dumps({"individual_parts_only": bool(app_data)}).encode("utf-8")
        request = urllib.request.Request(
            f"{SERVER}/api/study-filter", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
            for button in self.subassembly_buttons:
                dpg.configure_item(button, enabled=enabled)
            dpg.set_value(
                "selection_status",
                "Individual parts only" if app_data else "Parts and subassemblies enabled")
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            dpg.set_value("parts_only_checkbox", not bool(app_data))
            dpg.set_value("selection_status", f"Could not update Babylon filter: {exc}")

    def _post_review_selection(self, active: bool, capture: bool = False) -> None:
        draw_boxes = bool(dpg.get_value("review_boxes")) if dpg.does_item_exist("review_boxes") else False
        payload: dict[str, object] = {
            "active": active, "capture": capture, "draw_boxes": draw_boxes,
        }
        if active and self.saved_responses:
            payload["response_index"] = int(
                self.saved_responses[self.review_position]["index"])
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{SERVER}/api/review-selection", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=1.5) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")

    def _refresh_review_controls(self) -> None:
        if not self.saved_responses:
            dpg.set_value("review_detail", "No saved CSV responses.")
            dpg.configure_item("review_previous", enabled=False)
            dpg.configure_item("review_next", enabled=False)
            dpg.configure_item("review_capture", enabled=False)
            dpg.set_value("review_image_status", "No image status available.")
            return
        row = self.saved_responses[self.review_position]
        dpg.set_value("review_index_input", int(row["index"]))
        dpg.set_value(
            "review_detail",
            f"{self.review_position + 1} / {len(self.saved_responses)}\n"
            f"CSV index: {row['index']}\nTarget: {row['target_id']}\n"
            f"Referring type: {row.get('referring_type', '') or 'unclassified'}\n"
            f"Traits used: {row.get('traits_used', '') or 'unclassified'}\n"
            f"Presentation: {row['presentation_number']} / {row['target_presentations']}\n\n"
            f"{row['description']}")
        dpg.configure_item("review_previous", enabled=self.review_position > 0)
        dpg.configure_item(
            "review_next", enabled=self.review_position < len(self.saved_responses) - 1)
        dpg.configure_item("review_capture", enabled=True)
        self._check_review_images()

    def _check_review_images(self, *_args) -> None:
        if not self.saved_responses:
            dpg.set_value("review_image_status", "No CSV sample selected.")
            return
        row = self.saved_responses[self.review_position]
        try:
            url = f"{SERVER}/api/rendering-status?response_index={int(row['index'])}"
            with urllib.request.urlopen(url, timeout=1.5) as response:
                status = json.loads(response.read().decode("utf-8"))
            clean = status.get("clean_images", [])
            detection = status.get("detection_images", [])
            annotations = status.get("annotations", [])
            if not status.get("has_images"):
                message = f"CSV index {row['index']}: no saved images"
            else:
                message = (f"CSV index {row['index']}: {len(clean)} clean PNG, "
                           f"{len(detection)} boxed PNG, {len(annotations)} JSON annotation")
            dpg.set_value("review_image_status", message)
        except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
            dpg.set_value("review_image_status", f"Could not check images: {exc}")

    def _jump_review(self, *_args) -> None:
        if not self.saved_responses:
            return
        requested = int(dpg.get_value("review_index_input"))
        position = next((index for index, row in enumerate(self.saved_responses)
                         if int(row["index"]) == requested), None)
        if position is None:
            dpg.set_value("selection_status", f"CSV index {requested} does not exist")
            return
        self.review_position = position
        try:
            self._post_review_selection(True)
            self._refresh_review_controls()
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            dpg.set_value("selection_status", f"Could not jump to CSV index: {exc}")

    def _set_csv_review(self, _sender, app_data: bool, _user_data=None) -> None:
        try:
            if app_data:
                with urllib.request.urlopen(f"{SERVER}/api/saved-responses", timeout=1.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.saved_responses = sorted(
                    payload.get("responses", []), key=lambda row: int(row["index"]))
                self.review_position = 0
                self._refresh_review_controls()
                if self.saved_responses:
                    self._post_review_selection(True)
            else:
                self._post_review_selection(False)
                self.saved_responses = []
                self.review_position = 0
                self._refresh_review_controls()
        except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
            dpg.set_value("csv_review_checkbox", not bool(app_data))
            dpg.set_value("selection_status", f"Could not update CSV review mode: {exc}")

    def _move_review(self, delta: int) -> None:
        if not self.saved_responses:
            return
        self.review_position = max(
            0, min(len(self.saved_responses) - 1, self.review_position + delta))
        try:
            self._post_review_selection(True)
            self._refresh_review_controls()
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            dpg.set_value("selection_status", f"Could not change reviewed row: {exc}")

    def _capture_review(self, *_args) -> None:
        if not self.saved_responses:
            return
        try:
            self._post_review_selection(True, capture=True)
            row = self.saved_responses[self.review_position]
            dpg.set_value("selection_status", f"Capture requested for CSV index {row['index']}")
            dpg.set_value("review_image_status",
                          "Capture is running in Babylon; press 'Check saved images' shortly.")
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            dpg.set_value("selection_status", f"Could not request capture: {exc}")

    def _select(self, _sender, _app_data, step_id: str) -> None:
        step = self.graph.by_id[step_id]
        body = json.dumps({"target": step.output}).encode("utf-8")
        request = urllib.request.Request(
            f"{SERVER}/api/graph-selection", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
            dpg.set_value("selection_status", f"Showing: {step.output}")
            dpg.set_value(
                "selection_detail",
                f"{step.title}\n\n{step.description}\n\nSubassembly:\n{step.output}")
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            dpg.set_value(
                "selection_status",
                f"Could not reach Babylon study at {SERVER}: {exc}")

    def _create_theme(self, row: int) -> str:
        tag = f"row_theme::{row}"
        color = self.ROW_COLORS[row]
        body = tuple(max(18, int(c * 0.35)) for c in color[:3]) + (255,)
        hover = tuple(max(25, int(c * 0.55)) for c in color[:3]) + (255,)
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, body,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, hover,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, hover,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, color,
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, color,
                                    category=dpg.mvThemeCat_Nodes)
        return tag

    def _create_nodes(self, themes: dict[int, str]) -> None:
        # Dependency-aware lanes. Keeping stages 1/4 above, 2/5/6/7 in the
        # middle, and 3 below leaves an open fan-in corridor before stage 5;
        # direct links no longer have to run underneath unrelated nodes.
        stage_x = {
            1: 40, 2: 40, 3: 560,
            4: 560, 5: 1080, 6: 1600, 7: 2120,
        }
        stage_y = {
            1: 0, 2: 300, 3: 600,
            4: 0, 5: 300, 6: 300, 7: 300,
        }
        row_height = 940
        for step in self.steps:
            node = f"node::{step.id}"
            input_attr = f"input::{step.id}"
            output_attr = f"output::{step.id}"
            self.input_attrs[step.id] = input_attr
            self.output_attrs[step.id] = output_attr
            with dpg.node(label=step.title, tag=node, parent="study_graph"):
                with dpg.node_attribute(tag=input_attr,
                                        attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text("INPUT")
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text(step.description, wrap=275)
                    button = f"show::{step.id}"
                    dpg.add_button(label="Show this subassembly", tag=button,
                                   callback=self._select, user_data=step.id,
                                   width=190, enabled=False)
                    self.subassembly_buttons.append(button)
                with dpg.node_attribute(tag=output_attr,
                                        attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(step.output, color=(175, 255, 210), wrap=275)
            dpg.bind_item_theme(node, themes[step.row])
            coords = TaskGraph.control_coords_for(step.id)
            stage = coords[1] if coords else 7
            dpg.set_item_pos(node, (
                stage_x[stage],
                50 + (step.row - 1) * row_height + stage_y[stage],
            ))

    def _create_links(self) -> None:
        producer = {step.output: step.id for step in self.steps}
        made: set[tuple[str, str]] = set()
        for step in self.steps:
            sources = [producer[part] for part in step.inputs if part in producer]
            sources.extend(required for required in step.requires
                           if required in self.output_attrs)
            for source in sources:
                edge = (source, step.id)
                if edge in made:
                    continue
                made.add(edge)
                dpg.add_node_link(self.output_attrs[source],
                                  self.input_attrs[step.id], parent="study_graph")

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Gearbox Subassembly Study Selector",
                            width=2200, height=1200, min_width=1200, min_height=750)
        themes = {row: self._create_theme(row) for row in range(1, 5)}
        with dpg.window(tag="primary"):
            dpg.add_text("Gearbox Subassembly Task Graph",
                         color=(225, 232, 240))
            dpg.add_text(
                "Click 'Show this subassembly' on any step. The Babylon browser updates automatically."
                " Individual parts remain selectable by clicking the full gearbox in the browser.",
                color=(165, 180, 195), wrap=1500)
            with dpg.table(header_row=False, resizable=True,
                           policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=4.2)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        with dpg.child_window(height=-1, horizontal_scrollbar=True):
                            with dpg.node_editor(tag="study_graph", minimap=True,
                                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                                self._create_nodes(themes)
                                self._create_links()
                    with dpg.table_cell():
                        dpg.add_text("Selection", color=(225, 232, 240))
                        dpg.add_separator()
                        dpg.add_checkbox(
                            label="Individual parts only (hide subassemblies)",
                            tag="parts_only_checkbox", default_value=True,
                            callback=self._set_individual_parts_only)
                        dpg.add_spacer(height=8)
                        dpg.add_checkbox(
                            label="Review saved CSV responses",
                            tag="csv_review_checkbox", default_value=False,
                            callback=self._set_csv_review)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="← Previous", tag="review_previous",
                                           callback=lambda _s, _a, _u: self._move_review(-1), enabled=False)
                            dpg.add_button(label="Next →", tag="review_next",
                                           callback=lambda _s, _a, _u: self._move_review(1), enabled=False)
                        with dpg.group(horizontal=True):
                            dpg.add_input_int(label="CSV index", tag="review_index_input",
                                              default_value=1, min_value=1, width=150)
                            dpg.add_button(label="Go", callback=self._jump_review)
                        dpg.add_button(label="Capture images for this CSV index",
                                       tag="review_capture", callback=self._capture_review,
                                       enabled=False, width=-1)
                        dpg.add_checkbox(
                            label="Draw labeled 2D bounding boxes in captures",
                            tag="review_boxes", default_value=False)
                        dpg.add_button(label="Check saved images", callback=self._check_review_images,
                                       width=-1)
                        dpg.add_text("No image status available.", tag="review_image_status",
                                     wrap=340)
                        dpg.add_text("Review mode is off.", tag="review_detail", wrap=340)
                        dpg.add_spacer(height=12)
                        dpg.add_text("No subassembly selected.", tag="selection_status",
                                     wrap=340)
                        dpg.add_spacer(height=10)
                        dpg.add_text("", tag="selection_detail", wrap=340)
                        dpg.add_spacer(height=20)
                        dpg.add_text(f"Babylon server: {SERVER}",
                                     color=(150, 165, 185), wrap=340)
        dpg.set_primary_window("primary", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    ReferringExpressionTaskGraph().run()
