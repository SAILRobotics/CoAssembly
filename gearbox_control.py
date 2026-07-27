#!/usr/bin/env python3
"""
gearbox_control.py — one tool to drive the Unity gearbox visualization, via typed commands
AND via clicks in Unity.

Two input paths share a single process, a single command channel to Unity, and a single
per-row state machine:

  • REPL (main thread) — type commands; the Unity scene reacts live:
        row 1 / row 2 / row 3 / row 4   show ONLY that row of gears (hide the rest)
        show all  (or: all)             make every row visible again
        reset                           full reset: whole model, no highlights, progress cleared
        <part name>                     toggle a single part's color (yellow <-> original);
                                        e.g. BearingRow3Right, GearRow2Left, GearRodRow4
        quit / exit / q                 leave

  • Click listener (background thread) — Unity publishes gearbox clicks (part NAME) on port
    5020 (GearboxClickPublisher). Clicking runs the per-row assembly-state machine:
        State 1: show only GearRod, Gear, Pin of the row
        State 2: show only Stand, Bearing, Screw of the row
        State 3: whole row (shown on the next click once BOTH states are marked done)
    A "completed" checkbox (__checkbox__) and a close X (__reset__) — also clickable — mark a
    state done / close the menu (show everything again, keeping completion state).

Networking mirrors the repo's convention: Unity BINDS sockets, this script CONNECTS. Commands
go out on 5019 (GearboxCommandReceiver); clicks come in on 5023 (GearboxClickPublisher). A third
channel PUBs stage events to the task-graph viewer (gearbox_task_graph.py) on 5022, and a fourth
PUBs the pegboard tool ids to fetch for the current stage on 5024 (consumed by test_tool_layout.py
/ eventually main_with_robot.py, which picks the highlight colour itself) — both mirrors follow the
Python<->Python convention (the consumer BINDS a SUB, this script CONNECTS a PUB).
ZeroMQ sockets are not thread-safe, so every outbound send is serialized through one lock.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import zmq

# IPs and ports live canonically in main_setting.py (single source of truth); fall back to literals
# so the script still runs standalone if that import is unavailable.
try:
    import main_setting
    _DEFAULT_IP    = main_setting.UNITY_IP
    _TASKGRAPH_IP  = main_setting.LOCALHOST       # viewer runs on this machine (loopback)
    DEFAULT_CMD_PORT       = main_setting.GEARBOX_CMD_PORT
    DEFAULT_CLICK_PORT     = main_setting.GEARBOX_CLICK_PORT
    DEFAULT_TASKGRAPH_PORT = main_setting.GEARBOX_TASKGRAPH_PORT
    DEFAULT_HIGHLIGHT_PORT = main_setting.GEARBOX_HIGHLIGHT_PORT
    DEFAULT_STEP_SELECT_PORT = main_setting.GEARBOX_STEP_SELECT_PORT
    _SCENE_DIR             = main_setting.SCENE_LAYOUT_DIR
except Exception:
    _DEFAULT_IP    = "127.0.0.1"
    _TASKGRAPH_IP  = "127.0.0.1"
    DEFAULT_CMD_PORT       = 5019   # Python -> Unity (commands), GearboxCommandReceiver.
    DEFAULT_CLICK_PORT     = 5023   # Unity -> Python (clicks), GearboxClickPublisher.
    DEFAULT_TASKGRAPH_PORT = 5022   # Python -> task-graph viewer (live mirror).
    DEFAULT_HIGHLIGHT_PORT = 5024   # Python -> pegboard tool-highlight consumer (ids only).
    DEFAULT_STEP_SELECT_PORT = 5025 # task-graph viewer -> this script's --open-3d viewer.
    _SCENE_DIR             = Path(__file__).resolve().parent / "scene_layout"

_DEFAULT_TOOL_JSON = _SCENE_DIR / "tool_layout1.json"

# Timing for the assembly animation (Unity owns the actual motion/choreography).
STEP_DELAY    = 0.35   # seconds between one animation group finishing and the next starting
SLIDE_SECONDS = 0.50   # per-part slide duration

CHECKBOX_NAME = "__checkbox__"
RESET_NAME    = "__reset__"
REVERT_NAME   = "__revert__"

# 8-stage per-row assembly dependency model (mirrors task_graph/gearbox_task_graph.py):
#   1 left bearing->stand   2 gears->rod+pins   3 right bearing->stand
#   4 fasten left stand to board (needs 1)
#   5 insert rod + fit right bearing/stand together (needs 2, 3 & 4)
#   6 drive right screw (needs 5)
#   7 crank handle (row 1 only, needs 6)        8 verify (global, all rows done)
FINAL_STAGE = {1: 7, 2: 6, 3: 6, 4: 6}   # last per-row stage that must be complete


def parse_part(name: str):
    """'BearingRow3Left' / 'Bearing_Row3_Left' -> ('Bearing', 3, 'Left').
    Underscore-agnostic. Returns (None, None, None) if it isn't a Row part."""
    clean = name.replace("_", "")
    idx = clean.find("Row")
    if idx < 0:
        return None, None, None
    ptype = clean[:idx]
    i = idx + 3
    digits = ""
    while i < len(clean) and clean[i].isdigit():
        digits += clean[i]
        i += 1
    if not digits:
        return None, None, None
    side = clean[i:]                 # "Left" / "Right" / ""
    return ptype, int(digits), side


def part_stages(ptype: str, side: str, row: int):
    """Map a clicked part to its (appear, place, seat) stages — mirrors Unity's StagesOf.
    A part is interacted with at up to three stages. Returns (None, None, None) if unmapped.
        appear = the stage that first reveals the part
        place  = the stage it drops into its final position (>= appear)
        seat   = the stage it is fastened down / turns green (>= place)
    These coincide for every part EXCEPT the right bearing-stand, which drops into place with the
    rod at stage 5 (place=5) but isn't fastened until the right screw drives at stage 6 (seat=6).
    Clicking a placed-but-unfastened part advances to its seat stage (see _handle_part)."""
    left = side == "Left"
    if ptype in ("Bearing", "Stand"):
        return (1, 4, 4) if left else (3, 5, 6)
    if ptype == "Pin" and row == 1 and side == "Right":
        return 7, 7, 7                  # Row-1 right pin secures the crank handle at stage 7
    if ptype in ("GearRod", "Gear", "Pin"):
        return 2, 5, 5
    if ptype == "Screw":
        return (4, 4, 4) if left else (6, 6, 6)
    if ptype == "CrankHandle":
        return 7, 7, 7
    return None, None, None


# ── pegboard tool highlighting: appearing parts -> tool ids ───────────────────
# Fixed (type, side) set to consider when deciding which pegboard tools to fetch for a stage.
# For each, part_stages(...).appear picks out the ones entering the assembly at the clicked stage.
_HIGHLIGHT_PARTS = [
    ("Bearing", "Left"), ("Bearing", "Right"),
    ("Stand",   "Left"), ("Stand",   "Right"),
    ("Screw",   "Left"), ("Screw",   "Right"),
    ("GearRod", ""),
    ("Gear",    "Left"), ("Gear",    "Right"),
    ("Pin",     "Left"), ("Pin",     "Right"),
    ("CrankHandle", ""),
]


def load_tool_index(json_path) -> dict:
    """Build {TYPE_UPPER: id} from a tool_layout JSON. Returns {} (feature disables silently) if
    the file can't be read — the same soft-fail style as the optional main_setting import."""
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception as e:
        print(f"  (tool highlight disabled: can't read {json_path}: {e})")
        return {}
    index = {}
    for t in data.get("tools", []):
        try:
            index[str(t["type"]).upper()] = int(t["id"])
        except (KeyError, ValueError, TypeError):
            continue
    return index


def _candidate_tool_names(ptype: str, side: str, row: int):
    """Ordered pegboard-name candidates for a part; the first present in the index wins.
    Tiers: exact per-part name, then the GEAR_STAND alias, then the shared bin by type."""
    s = side.upper()
    if ptype == "Bearing":
        return [f"BEARING_ROW{row}_{s}", "BEARINGS"]
    if ptype == "Stand":
        return [f"STAND_ROW{row}_{s}", f"GEAR_STAND_ROW{row}_{s}"]
    if ptype == "GearRod":
        return [f"GEAR_ROD_ROW{row}"]
    if ptype == "Gear":
        return [f"GEAR_ROW{row}_{s}"]
    if ptype == "Pin":
        return [f"PIN_ROW{row}_{s}", "PINS"]
    if ptype == "Screw":
        return [f"SCREW_ROW{row}_{s}", "SCREWS"]
    if ptype == "CrankHandle":
        # The crank is a row-1-only part; other rows have no crank step (and no MISC fetch).
        return [f"CRANK_HANDLE_ROW{row}", "MISC"] if row == 1 else []
    return []


def appearing_ids(index: dict, row: int, stage: int) -> list:
    """Pegboard tool ids for the parts ENTERING the assembly at (row, stage) — those whose
    part_stages(...).appear == stage. A part already built into a sub-assembly has appear < stage
    (it's orange, off the pegboard) so it's excluded for free — that IS the 'appear state only'
    rule. Each part resolves to a tool id via _candidate_tool_names; a (type, side) that doesn't
    exist for the row simply fails to resolve and is skipped. Ids are de-duplicated in order."""
    ids: list = []
    for ptype, side in _HIGHLIGHT_PARTS:
        appear, _place, _seat = part_stages(ptype, side, row)
        if appear != stage:
            continue
        for name in _candidate_tool_names(ptype, side, row):
            tid = index.get(name.upper())
            if tid is not None:
                if tid not in ids:
                    ids.append(tid)
                break
    return ids


# ── typed-command parsing ─────────────────────────────────────────────────────
def build_command(raw: str):
    """Parse a line of REPL input into a command dict, or None to ignore."""
    text = raw.strip()
    if not text:
        return None

    low = text.lower()

    if low in ("quit", "exit", "q"):
        return {"command": "quit"}
    if low in ("show all", "showall", "all"):
        return {"command": "show_all"}
    if low == "reset":
        return {"command": "reset"}
    if low == "verify":
        return {"command": "verify"}

    # "row N" or "rowN"
    tokens = low.split()
    row_token = None
    if len(tokens) == 2 and tokens[0] == "row":
        row_token = tokens[1]
    elif len(tokens) == 1 and low.startswith("row") and low[3:].isdigit():
        row_token = low[3:]

    if row_token is not None:
        if row_token.isdigit() and 1 <= int(row_token) <= 4:
            return {"command": "row", "row": int(row_token)}
        print(f"  ! Invalid row '{row_token}'. Use row 1, row 2, row 3, or row 4.")
        return None

    # Anything else: treat the original-case text as a part name to toggle its color.
    return {"command": "toggle", "part": text}


def describe(cmd: dict) -> str:
    c = cmd.get("command")
    if c == "row":
        return f"show only Row{cmd['row']}"
    if c == "show_all":
        return "show all rows"
    if c == "toggle":
        return f"toggle color of '{cmd['part']}'"
    return c


class GearboxStateMachine:
    """7-stage per-row assembly dependency machine, driven by clicks. Owns only the
    dependency/lock/completion logic; Unity choreographs the motion. State lives as attributes,
    matching the house style of _ToolSelectionManager (main_with_robot.py)."""

    def __init__(self, send, notify=None, highlight=None):
        self._send = send                       # send(dict) -> publishes a command to Unity
        # notify(dict) -> publishes a semantic (row, stage) event to the task-graph viewer.
        # Optional: defaults to a no-op so headless tests can construct GearboxStateMachine(send).
        self._notify = notify or (lambda _msg: None)
        # highlight(dict) -> forwards a pegboard tool-highlight event (the controller resolves the
        # appearing parts to tool ids). Same optional add-on pattern as notify, so the older
        # GearboxStateMachine(send) / (send, notify) constructions keep working unchanged.
        self._highlight = highlight or (lambda _msg: None)
        self.done = {r: {s: False for s in range(1, 8)} for r in range(1, 5)}
        self.done8 = False                      # global "verify" stage (stage 8)
        self.current_row = None
        self.current_stage = None
        # Completed (row, stage) steps in the order they were checked off, so the revert button can
        # undo the most recent one. Stays frontier-safe: completions respect dependency order, so the
        # last-completed step can never have a completed dependent (that would be the last instead).
        self.history: list[tuple[int, int]] = []

    def handle_click(self, name: str):
        if name == CHECKBOX_NAME:
            self._handle_checkbox()
        elif name == RESET_NAME:
            self.close_menu()
        elif name == REVERT_NAME:
            self.revert_last()
        else:
            self._handle_part(name)

    def _forget_history(self, row: int, stage: int):
        """Drop a (row, stage) from the completion history if present (on un-check / revert)."""
        try:
            self.history.remove((row, stage))
        except ValueError:
            pass

    # ── dependency logic ──────────────────────────────────────────────────────
    def unlocked(self, row: int, stage: int) -> bool:
        d = self.done[row]
        if stage in (1, 2, 3):
            return True
        if stage == 4:
            return d[1]
        if stage == 5:
            return d[2] and d[3] and d[4]
        if stage == 6:
            return d[5]
        if stage == 7:
            return row == 1 and d[6]
        return False

    def dependents_done(self, row: int, stage: int) -> bool:
        """True if a *completed* stage depends on (row, stage) — blocks un-checking (frontier)."""
        d = self.done[row]
        if stage == 1:
            return d[4]
        if stage in (2, 3, 4):
            return d[5]
        if stage == 5:
            return d[6]
        if stage == 6:
            return d[7] if row == 1 else self.done8
        if stage == 7:
            return self.done8
        return False

    def all_rows_done(self) -> bool:
        return all(self.done[r][FINAL_STAGE[r]] for r in range(1, 5))

    def _completed_stages(self, row: int):
        return [s for s in range(1, 8) if self.done[row][s]]

    # ── click handlers ────────────────────────────────────────────────────────
    def _handle_part(self, name: str):
        ptype, row, side = parse_part(name)
        if row is None:
            print(f"  (ignored '{name}': not a Row part)")
            return
        appear, place, seat = part_stages(ptype, side, row)
        if appear is None:
            print(f"  (ignored '{name}': type '{ptype}' not stage-mapped)")
            return
        # A part participates in its appear, place, and seat stages. Clicking it jumps FORWARD to
        # the furthest of those whose prerequisites are met — so once stages 1 & 2 are done a
        # stage-1/2 part triggers the stage-4 drop, and a part that's already dropped into place but
        # not yet fastened (the right bearing-stand: place 5, seat 6) advances to its fasten stage
        # so it can be completed. Falls back to `appear` (possibly blocked) when nothing is unlocked.
        stage = appear
        for cand in (seat, place):
            if cand > appear and self.unlocked(row, cand):
                stage = cand
                break
        if stage == 7 and row != 1:
            print(f"  (ignored '{name}': stage 7 is row 1 only)")
            return

        self.current_row, self.current_stage = row, stage
        blocked = not self.unlocked(row, stage)
        self._send({"command": "stage", "row": row, "stage": stage,
                    "done_stages": self._completed_stages(row),
                    "step_delay": STEP_DELAY, "slide_seconds": SLIDE_SECONDS})
        self._send({"command": "ui", "show": True, "row": row,
                    "checked": self.done[row][stage], "blocked": blocked})
        self._notify({"event": "show", "row": row, "stage": stage})
        # Light up the pegboard tools for the parts appearing at this stage. A stage that's already
        # complete has no appearing parts (they're orange, off the pegboard), so pass complete=True
        # to request an empty set. Locked/blocked stages still highlight (informational).
        self._highlight({"event": "highlight", "row": row, "stage": stage,
                         "complete": self.done[row][stage]})
        print(f"  row {row}: stage {stage}  "
              f"({'LOCKED' if blocked else 'ready'}, done={self.done[row][stage]})")

    def _handle_checkbox(self):
        if self.current_stage == 8:                       # global verify
            self.done8 = not self.done8
            if self.done8:
                self.history.append((0, 8))
            else:
                self._forget_history(0, 8)
            self._send({"command": "ui", "show": True, "row": 0,
                        "checked": self.done8, "blocked": False})
            self._notify({"event": "complete" if self.done8 else "uncomplete",
                          "row": 0, "stage": 8})
            print(f"  verify done={self.done8}")
            return
        if self.current_row is None or self.current_stage is None:
            print("  (checkbox ignored: no active stage)")
            return
        row, stage = self.current_row, self.current_stage
        if not self.unlocked(row, stage):
            print(f"  (checkbox ignored: stage {stage} is locked)")
            return
        if not self.done[row][stage]:
            self.done[row][stage] = True
            self.history.append((row, stage))
        else:
            if self.dependents_done(row, stage):
                print(f"  (can't un-check stage {stage}: a later stage depends on it)")
                return
            self.done[row][stage] = False
            self._forget_history(row, stage)

        # Recolor the whole row from its completed stages: green where seated, orange where a
        # sub-assembly's build step is done but it isn't seated yet, original otherwise.
        self._send({"command": "recolor", "row": row,
                    "done_stages": self._completed_stages(row)})
        self._send({"command": "ui", "show": True, "row": row,
                    "checked": self.done[row][stage], "blocked": False})
        self._notify({"event": "complete" if self.done[row][stage] else "uncomplete",
                      "row": row, "stage": stage})
        print(f"  row {row}: stage {stage} done={self.done[row][stage]}")

        if self.done[row][stage] and self.all_rows_done():
            self._show_stage8()

    def revert_last(self):
        """The revert button: turn the most-recently completed step back to incomplete. Updates the
        part colors (Unity) and the task-graph progression (viewer). Popping in reverse completion
        order keeps it frontier-safe — the newest completion never has a completed dependent."""
        if not self.history:
            print("  (nothing to revert)")
            return
        row, stage = self.history.pop()
        if stage == 8:
            self.done8 = False
        else:
            self.done[row][stage] = False
            # Whole model is on screen (no stage menu open), so just recolor the affected row.
            self._send({"command": "recolor", "row": row,
                        "done_stages": self._completed_stages(row)})
        self._notify({"event": "uncomplete", "row": row, "stage": stage})
        print(f"  reverted: row {row} stage {stage} -> incomplete")

    def _show_stage8(self):
        self.current_row, self.current_stage = 0, 8
        self._send({"command": "stage", "row": 0, "stage": 8,
                    "done_stages": [1, 2, 3, 4, 5, 6, 7]})   # everything is done -> all green
        self._send({"command": "ui", "show": True, "row": 0,
                    "checked": self.done8, "blocked": False})
        self._notify({"event": "show", "row": 0, "stage": 8})
        # Verify has no parts to fetch; the empty set also clears any lingering highlight.
        self._highlight({"event": "highlight", "row": 0, "stage": 8, "complete": True})
        print("  all rows complete -> stage 8 (verify)")

    def show_verify(self):
        """Typed 'verify': re-open the stage-8 view if the whole gearbox is done."""
        if self.all_rows_done():
            self._show_stage8()
        else:
            print("  (verify unavailable: not all rows are complete)")

    def close_menu(self):
        """The X button: show the whole model again and hide the UI, KEEPING completion state
        (and the green 'done' coloring)."""
        self.current_row = self.current_stage = None
        self._send({"command": "show_all"})
        self._send({"command": "ui", "show": False})
        self._notify({"event": "close"})
        self._highlight({"event": "clear"})

    def reset(self):
        """Full restart: clear all completion, show the whole model, hide the UI.
        Reached only via the typed 'reset' command, not the X button."""
        for r in self.done:
            for s in self.done[r]:
                self.done[r][s] = False
        self.done8 = False
        self.history.clear()
        self.close_menu()
        self._notify({"event": "reset"})


class PegboardBoxViewer:
    """Open3D window that draws every pegboard tool as a wireframe box and recolours the ones for
    the currently-selected task-graph step. A local stand-in for Unity (used via --open-3d when the
    headset pipeline is unavailable). Box geometry matches test_tool_layout.build_payload: pegboard
    peg_pos -> world via T, plus a half-height offset along the part's z to reach the centroid — but
    kept in the Open3D/world frame (no Open3D->Unity axis swap, since this renders in Open3D).

    Open3D's Visualizer is single-threaded and must live on the main thread; the owning code drives
    tick()/set_highlight() from there. open3d/numpy/scipy are imported lazily so the normal
    (non-open3d) modes of this script don't need them."""

    NEUTRAL   = [0.55, 0.55, 0.58]
    HIGHLIGHT = [0.0, 1.0, 1.0]
    _EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

    def __init__(self, tools: list, T, title: str = "Pegboard tools"):
        import numpy as np
        import open3d as o3d
        self.np, self.o3d = np, o3d

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=1000, height=700)
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.10])
        ro.line_width = 2.0
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10))

        self._ls_by_id: dict = {}
        self._color_by_id: dict = {}
        for t in tools:
            try:
                tid = int(t["id"])
            except (KeyError, ValueError, TypeError):
                continue
            center, R, size = self._box_geom(t, T)
            ls = self._make_box(center, R, size, self.NEUTRAL)
            self.vis.add_geometry(ls)
            self._ls_by_id[tid] = ls
            self._color_by_id[tid] = self.NEUTRAL

        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, -0.4, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.6)

    def _box_geom(self, t: dict, T):
        np = self.np
        from scipy.spatial.transform import Rotation as ScipyR
        sz  = t.get("size", [0.05, 0.05, 0.05])
        rot = t.get("rotation_deg", [0.0, 0.0, 0.0])
        R = ScipyR.from_euler('z', float(rot[2]), degrees=True).as_matrix()
        if "peg_pos" in t:
            base = (np.asarray(T) @ np.append(t["peg_pos"], 1.0))[:3]
        else:
            base = np.array(t.get("world_pos", [0.0, 0.0, 0.0]))
        center = base + R @ np.array([0.0, 0.0, sz[2] / 2.0])
        return center, R, sz

    def _make_box(self, pos, R, size, color):
        np, o3d = self.np, self.o3d
        w, d, h = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        corners_local = np.array([[-w, -d, -h], [w, -d, -h], [w, d, -h], [-w, d, -h],
                                  [-w, -d,  h], [w, -d,  h], [w, d,  h], [-w, d,  h]])
        corners = (R @ corners_local.T).T + pos
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines  = o3d.utility.Vector2iVector(self._EDGES)
        ls.colors = o3d.utility.Vector3dVector([list(color)] * 12)
        return ls

    def set_highlight(self, ids):
        """Colour the given tool ids HIGHLIGHT and everything else NEUTRAL (only touching boxes that
        actually change, so a repeated selection is cheap)."""
        want_ids = set(int(i) for i in ids)
        for tid, ls in self._ls_by_id.items():
            want = self.HIGHLIGHT if tid in want_ids else self.NEUTRAL
            if self._color_by_id[tid] is not want:
                ls.colors = self.o3d.utility.Vector3dVector([list(want)] * 12)
                self.vis.update_geometry(ls)
                self._color_by_id[tid] = want

    def tick(self) -> bool:
        """Pump one frame. Returns False once the window has been closed."""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


class GearboxController:
    """Owns the ZMQ sockets, the shared state machine, and the click-listener loop."""

    def __init__(self, ip: str, cmd_port: int, click_port: int,
                 tg_ip: str = _TASKGRAPH_IP, tg_port: int = DEFAULT_TASKGRAPH_PORT,
                 hl_ip: str = None, hl_port: int = DEFAULT_HIGHLIGHT_PORT,
                 no_highlight: bool = False, tool_json=None,
                 open3d: bool = False, step_select_port: int = DEFAULT_STEP_SELECT_PORT):
        self.ip, self.cmd_port, self.click_port = ip, cmd_port, click_port
        self.tg_ip, self.tg_port = tg_ip, tg_port
        # The tool-highlight consumer (main_with_robot.py / test_tool_layout.py) is a Python process
        # on THIS machine — a Python↔Python link like the task-graph mirror — so it defaults to
        # loopback, NOT the Unity IP. (When Unity ran on localhost these were the same; on an
        # Android/WiFi headset build the Unity IP is remote and would send highlights nowhere.)
        self.hl_ip, self.hl_port = hl_ip or _TASKGRAPH_IP, hl_port
        self.open3d, self.step_select_port = open3d, step_select_port
        self._tool_json = tool_json or _DEFAULT_TOOL_JSON
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{cmd_port}")

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip}:{click_port}")

        # Live mirror to the task-graph viewer (Python<->Python: viewer BINDs a SUB, we CONNECT).
        # Harmless if no viewer is running — a PUB with no subscriber just drops.
        self._tg_pub = self._ctx.socket(zmq.PUB)
        self._tg_pub.connect(f"tcp://{tg_ip}:{tg_port}")

        # Pegboard tool-highlight channel (ids only; the consumer picks the colour). We resolve the
        # appearing parts to tool ids via the tool_layout JSON. Same Python<->Python convention as
        # the task-graph mirror: the consumer BINDs a SUB, we CONNECT a PUB. Disabled silently if the
        # JSON can't be loaded or --no-highlight is set.
        # The name->id index feeds both the 5024 highlight wire and the --open-3d viewer, so it is
        # loaded regardless; --no-highlight only silences the 5024 PUB.
        self._tool_index = load_tool_index(self._tool_json)
        self._highlight_enabled = bool(self._tool_index) and not no_highlight
        if self._highlight_enabled:
            self._hl_pub = self._ctx.socket(zmq.PUB)
            self._hl_pub.connect(f"tcp://{self.hl_ip}:{self.hl_port}")
        else:
            self._hl_pub = None

        # --open-3d: receive the task-graph viewer's selected step (Python<->Python: we BIND a SUB,
        # gearbox_task_graph.py CONNECTs a PUB) and recolour the local pegboard boxes accordingly.
        if open3d:
            self._ss_sub = self._ctx.socket(zmq.SUB)
            self._ss_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self._ss_sub.bind(f"tcp://0.0.0.0:{step_select_port}")
        else:
            self._ss_sub = None

        time.sleep(0.2)  # let PUB/SUB settle (slow-joiner guard)

        self._send_lock = threading.Lock()
        self._running = True
        self.sm = GearboxStateMachine(self.send, self.send_taskgraph, self._on_highlight)

    def send(self, msg: dict):
        """Thread-safe outbound send (called from both the REPL and the click thread)."""
        with self._send_lock:
            self._pub.send_string(json.dumps(msg))

    def send_taskgraph(self, msg: dict):
        """Thread-safe send of a semantic (row, stage) event to the task-graph viewer."""
        with self._send_lock:
            self._tg_pub.send_string(json.dumps(msg))

    def send_highlight(self, msg: dict):
        """Thread-safe send of a pegboard tool-highlight event (ids only, no colour)."""
        if self._hl_pub is None:
            return
        with self._send_lock:
            self._hl_pub.send_string(json.dumps(msg))

    def _on_highlight(self, msg: dict):
        """State-machine highlight hook: resolve the appearing parts of (row, stage) to pegboard
        tool ids and publish them (or forward a clear). This is where 'gearbox_control sends the
        ids to be highlighted' — the state machine only forwards semantic (row, stage) events."""
        if not self._highlight_enabled:
            return
        if msg.get("event") == "clear":
            self.send_highlight({"event": "clear"})
            return
        row, stage = msg["row"], msg["stage"]
        ids = [] if msg.get("complete") else appearing_ids(self._tool_index, row, stage)
        self.send_highlight({"event": "highlight", "ids": ids, "row": row, "stage": stage})

    def _poll_step_select(self):
        """Drain the task-graph select channel; return the most recent event dict, or None. Only the
        latest is applied — intermediate selections in one tick are superseded anyway."""
        if self._ss_sub is None:
            return None
        latest = None
        while True:
            try:
                raw = self._ss_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                latest = json.loads(raw)
            except json.JSONDecodeError:
                continue
        return latest

    def run_open3d(self, identity: bool = False):
        """--open-3d: open the local pegboard-box window and recolour it from the task-graph viewer's
        selected step. Blocks on the Open3D loop (main thread) until the window closes or stop()."""
        from test_tool_layout import load_tools, load_pegboard_T
        tools = load_tools(self._tool_json)
        T     = load_pegboard_T(identity)
        viewer = PegboardBoxViewer(tools, T)
        print(f"[open3d] window open ({len(tools)} tools). "
              f"Select a step in the task graph to highlight its pegboard tools.")
        try:
            while self._running:
                ev = self._poll_step_select()
                if ev is not None:
                    if ev.get("event") == "clear":
                        viewer.set_highlight([])
                        print("[open3d] cleared")
                    elif ev.get("event") == "select":
                        row, stage = int(ev["row"]), int(ev["stage"])
                        ids = appearing_ids(self._tool_index, row, stage)
                        viewer.set_highlight(ids)
                        print(f"[open3d] step '{ev.get('step','?')}' "
                              f"(row {row}, stage {stage}) -> tools {ids}")
                if not viewer.tick():
                    break
                time.sleep(0.02)
        finally:
            viewer.close()
            self._running = False

    def full_reset(self):
        """Typed 'reset': clear color highlights too, then reset progress/visibility/UI."""
        self.send({"command": "reset"})   # clear color highlights
        self.sm.reset()                    # clear progress + show_all + hide UI

    def run_click_loop(self):
        """Poll for Unity clicks and drive the state machine. Blocks until stop()."""
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        while self._running:
            if not dict(poller.poll(timeout=100)):
                continue
            while True:
                try:
                    raw = self._sub.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    msg = json.loads(raw)
                    name = msg["name"]
                    event = msg.get("event_type", "selected")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  bad click message: {e}")
                    continue
                if event != "selected":
                    continue
                print(f"\nclick: {name}")
                self.sm.handle_click(name)

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        try: self._sub.close()
        except Exception: pass
        try: self._pub.close()
        except Exception: pass
        try: self._tg_pub.close()
        except Exception: pass
        if self._hl_pub is not None:
            try: self._hl_pub.close()
            except Exception: pass
        if self._ss_sub is not None:
            try: self._ss_sub.close()
            except Exception: pass
        # Context is shared (Context.instance()); leave it for the process to reclaim.


BANNER = """\
--------------------------------------------------------------
 Gearbox control  <->  Unity   (typed commands + clicks)
--------------------------------------------------------------
 Type:
   row 1 | row 2 | row 3 | row 4   show only that row
   show all                        show every row
   verify                          open the final verify view (once all rows done)
   reset                           full restart (model + highlights + clear progress)
   <PartName>                      toggle a part's color (yellow)
                                   e.g. BearingRow3Right, GearRodRow4
   quit                            exit
 Click (in Unity):
   a part -> drive its row through the 7-stage assembly (locked stages show red)
   the checkbox marks a stage done (turns it green); the X closes the menu
--------------------------------------------------------------"""


def main():
    parser = argparse.ArgumentParser(description="Typed + click controller for the Unity gearbox scene.")
    parser.add_argument("--ip", default=_DEFAULT_IP,
                        help=f"IP of the machine running Unity (default: {_DEFAULT_IP})")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT,
                        help=f"Port GearboxCommandReceiver binds to (default: {DEFAULT_CMD_PORT})")
    parser.add_argument("--click-port", type=int, default=DEFAULT_CLICK_PORT,
                        help=f"Port Unity publishes clicks on (default: {DEFAULT_CLICK_PORT})")
    parser.add_argument("--task-graph-ip", default=_TASKGRAPH_IP,
                        help=f"IP of the task-graph viewer (default: {_TASKGRAPH_IP})")
    parser.add_argument("--task-graph-port", type=int, default=DEFAULT_TASKGRAPH_PORT,
                        help=f"Port the task-graph viewer binds to (default: {DEFAULT_TASKGRAPH_PORT})")
    parser.add_argument("--highlight-ip", default=None,
                        help="IP of the pegboard tool-highlight consumer, a same-machine Python "
                             "process (default: loopback, NOT the Unity IP)")
    parser.add_argument("--highlight-port", type=int, default=DEFAULT_HIGHLIGHT_PORT,
                        help=f"Port the tool-highlight consumer binds to (default: {DEFAULT_HIGHLIGHT_PORT})")
    parser.add_argument("--tool-json", default=None,
                        help=f"tool_layout JSON for name->id resolution (default: {_DEFAULT_TOOL_JSON})")
    parser.add_argument("--no-highlight", action="store_true",
                        help="Disable pegboard tool highlighting.")
    parser.add_argument("--open-3d", "--open3d", dest="open_3d", action="store_true",
                        help="Open a local Open3D pegboard-box window (stand-in for Unity) and "
                             "highlight the tools for whichever step is selected in the task graph.")
    parser.add_argument("--step-select-port", type=int, default=DEFAULT_STEP_SELECT_PORT,
                        help=f"Port to receive the task graph's selected step on "
                             f"(default: {DEFAULT_STEP_SELECT_PORT}; --open-3d only)")
    parser.add_argument("--identity", action="store_true",
                        help="Lay out the --open-3d boxes in the pegboard frame (ignore the saved "
                             "scan pose).")
    parser.add_argument("--no-repl", action="store_true",
                        help="Run only the click listener (no typed REPL).")
    args = parser.parse_args()

    ctrl = GearboxController(args.ip, args.cmd_port, args.click_port,
                             args.task_graph_ip, args.task_graph_port,
                             hl_ip=args.highlight_ip, hl_port=args.highlight_port,
                             no_highlight=args.no_highlight, tool_json=args.tool_json,
                             open3d=args.open_3d, step_select_port=args.step_select_port)

    print(BANNER)
    print(f" commands   OUT -> tcp://{args.ip}:{args.cmd_port}")
    print(f" clicks     IN  <- tcp://{args.ip}:{args.click_port}")
    print(f" task-graph OUT -> tcp://{args.task_graph_ip}:{args.task_graph_port}")
    if ctrl._highlight_enabled:
        print(f" highlight  OUT -> tcp://{ctrl.hl_ip}:{ctrl.hl_port}  "
              f"({len(ctrl._tool_index)} tools indexed)")
    else:
        print(" highlight  OFF")
    if args.open_3d:
        print(f" open3d     IN  <- tcp://0.0.0.0:{args.step_select_port}  (task-graph step select)")
    print()

    click_thread = repl_thread = None
    try:
        if args.open_3d:
            # Open3D owns the main thread; the click listener and (optionally) the REPL run behind it.
            click_thread = threading.Thread(target=ctrl.run_click_loop, daemon=True)
            click_thread.start()
            if not args.no_repl:
                repl_thread = threading.Thread(target=_run_repl, args=(ctrl,), daemon=True)
                repl_thread.start()
            ctrl.run_open3d(identity=args.identity)
        elif args.no_repl:
            print(" (click-listener only; Ctrl-C to quit)")
            ctrl.run_click_loop()
        else:
            click_thread = threading.Thread(target=ctrl.run_click_loop, daemon=True)
            click_thread.start()
            _run_repl(ctrl)
    except KeyboardInterrupt:
        print()
    finally:
        ctrl.stop()
        if click_thread is not None:
            click_thread.join(timeout=1.0)
        ctrl.close()
        print("Bye.")


def _run_repl(ctrl: GearboxController):
    while True:
        try:
            raw = input("gearbox> ")
        except EOFError:
            print()
            return

        cmd = build_command(raw)
        if cmd is None:
            continue
        if cmd["command"] == "quit":
            return
        if cmd["command"] == "reset":
            ctrl.full_reset()
            print("  -> full reset")
            continue
        if cmd["command"] == "verify":
            ctrl.sm.show_verify()
            continue

        ctrl.send(cmd)
        print(f"  -> {describe(cmd)}")


if __name__ == "__main__":
    sys.exit(main())
