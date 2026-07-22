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
channel PUBs stage events to the task-graph viewer (gearbox_task_graph.py) on 5022 — that mirror
follows the Python<->Python convention (the viewer BINDS a SUB, this script CONNECTS a PUB).
ZeroMQ sockets are not thread-safe, so every outbound send is serialized through one lock.
"""

import argparse
import json
import sys
import threading
import time

import zmq

try:
    import ip_setting
    _DEFAULT_IP    = ip_setting.UNITY_IP
    _TASKGRAPH_IP  = ip_setting.LOCALHOST         # viewer runs on this machine
except Exception:
    _DEFAULT_IP    = "127.0.0.1"
    _TASKGRAPH_IP  = "127.0.0.1"

# Ports live canonically in main_setting.py (single source of truth); fall back to literals so the
# script still runs standalone if that import is unavailable.
try:
    import main_setting
    DEFAULT_CMD_PORT       = main_setting.GEARBOX_CMD_PORT
    DEFAULT_CLICK_PORT     = main_setting.GEARBOX_CLICK_PORT
    DEFAULT_TASKGRAPH_PORT = main_setting.GEARBOX_TASKGRAPH_PORT
except Exception:
    DEFAULT_CMD_PORT       = 5019   # Python -> Unity (commands), GearboxCommandReceiver.
    DEFAULT_CLICK_PORT     = 5023   # Unity -> Python (clicks), GearboxClickPublisher.
    DEFAULT_TASKGRAPH_PORT = 5022   # Python -> task-graph viewer (live mirror).

# Timing for the assembly animation (Unity owns the actual motion/choreography).
STEP_DELAY    = 0.35   # seconds between one animation group finishing and the next starting
SLIDE_SECONDS = 0.50   # per-part slide duration

CHECKBOX_NAME = "__checkbox__"
RESET_NAME    = "__reset__"

# 7-stage per-row assembly dependency model (mirrors task_graph/gearbox_task_graph.py, not linked):
#   1 left bearing->stand   2 gears->rod+pins   3 right bearing->stand
#   4 fasten left stand to board (needs 1)
#   5 insert rod + fit right stand + screw right (needs 2, 3 & 4)
#   6 crank handle (row 1 only, needs 5)        7 verify (global, all rows done)
FINAL_STAGE = {1: 6, 2: 5, 3: 5, 4: 5}   # last per-row stage that must be complete


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
    """Map a clicked part to its (appear_stage, seat_stage) — mirrors Unity's StagesOf.
    A part is interacted with at TWO stages: when it first appears, and (later) when it seats
    onto the board. Returns (None, None) if the type isn't stage-mapped.
        appear = the stage that first reveals the part
        seat   = the stage the part is fastened down (>= appear)"""
    left = side == "Left"
    if ptype in ("Bearing", "Stand"):
        return (1, 4) if left else (3, 5)
    if ptype == "Pin" and row == 1 and side == "Right":
        return 6, 6                     # Row-1 right pin secures the crank handle at stage 6
    if ptype in ("GearRod", "Gear", "Pin"):
        return 2, 5
    if ptype == "Screw":
        return (4, 4) if left else (5, 5)
    if ptype == "CrankHandle":
        return 6, 6
    return None, None


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

    def __init__(self, send, notify=None):
        self._send = send                       # send(dict) -> publishes a command to Unity
        # notify(dict) -> publishes a semantic (row, stage) event to the task-graph viewer.
        # Optional: defaults to a no-op so headless tests can construct GearboxStateMachine(send).
        self._notify = notify or (lambda _msg: None)
        self.done = {r: {s: False for s in range(1, 7)} for r in range(1, 5)}
        self.done7 = False                      # global "verify" stage
        self.current_row = None
        self.current_stage = None

    def handle_click(self, name: str):
        if name == CHECKBOX_NAME:
            self._handle_checkbox()
        elif name == RESET_NAME:
            self.close_menu()
        else:
            self._handle_part(name)

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
            return row == 1 and d[5]
        return False

    def dependents_done(self, row: int, stage: int) -> bool:
        """True if a *completed* stage depends on (row, stage) — blocks un-checking (frontier)."""
        d = self.done[row]
        if stage == 1:
            return d[4]
        if stage in (2, 3, 4):
            return d[5]
        if stage == 5:
            return d[6] if row == 1 else self.done7
        if stage == 6:
            return self.done7
        return False

    def all_rows_done(self) -> bool:
        return all(self.done[r][FINAL_STAGE[r]] for r in range(1, 5))

    def _completed_stages(self, row: int):
        return [s for s in range(1, 7) if self.done[row][s]]

    # ── click handlers ────────────────────────────────────────────────────────
    def _handle_part(self, name: str):
        ptype, row, side = parse_part(name)
        if row is None:
            print(f"  (ignored '{name}': not a Row part)")
            return
        appear, seat = part_stages(ptype, side, row)
        if appear is None:
            print(f"  (ignored '{name}': type '{ptype}' not stage-mapped)")
            return
        # A part participates in its appear stage and (later) its seat stage. Once the seat
        # stage's prerequisites are met, clicking the part jumps FORWARD to that stage — e.g.
        # after stages 1 & 2 are done, clicking a stage-1/2 part triggers the stage-4
        # (drop-onto-board) animation instead of re-showing the earlier stage.
        stage = seat if (seat != appear and self.unlocked(row, seat)) else appear
        if stage == 6 and row != 1:
            print(f"  (ignored '{name}': stage 6 is row 1 only)")
            return

        self.current_row, self.current_stage = row, stage
        blocked = not self.unlocked(row, stage)
        self._send({"command": "stage", "row": row, "stage": stage,
                    "done_stages": self._completed_stages(row),
                    "step_delay": STEP_DELAY, "slide_seconds": SLIDE_SECONDS})
        self._send({"command": "ui", "show": True, "row": row,
                    "checked": self.done[row][stage], "blocked": blocked})
        self._notify({"event": "show", "row": row, "stage": stage})
        print(f"  row {row}: stage {stage}  "
              f"({'LOCKED' if blocked else 'ready'}, done={self.done[row][stage]})")

    def _handle_checkbox(self):
        if self.current_stage == 7:                       # global verify
            self.done7 = not self.done7
            self._send({"command": "ui", "show": True, "row": 0,
                        "checked": self.done7, "blocked": False})
            self._notify({"event": "complete" if self.done7 else "uncomplete",
                          "row": 0, "stage": 7})
            print(f"  verify done={self.done7}")
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
        else:
            if self.dependents_done(row, stage):
                print(f"  (can't un-check stage {stage}: a later stage depends on it)")
                return
            self.done[row][stage] = False

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
            self._show_stage7()

    def _show_stage7(self):
        self.current_row, self.current_stage = 0, 7
        self._send({"command": "stage", "row": 0, "stage": 7,
                    "done_stages": [1, 2, 3, 4, 5, 6]})   # everything is done -> all green
        self._send({"command": "ui", "show": True, "row": 0,
                    "checked": self.done7, "blocked": False})
        self._notify({"event": "show", "row": 0, "stage": 7})
        print("  all rows complete -> stage 7 (verify)")

    def show_verify(self):
        """Typed 'verify': re-open the stage-7 view if the whole gearbox is done."""
        if self.all_rows_done():
            self._show_stage7()
        else:
            print("  (verify unavailable: not all rows are complete)")

    def close_menu(self):
        """The X button: show the whole model again and hide the UI, KEEPING completion state
        (and the green 'done' coloring)."""
        self.current_row = self.current_stage = None
        self._send({"command": "show_all"})
        self._send({"command": "ui", "show": False})
        self._notify({"event": "close"})

    def reset(self):
        """Full restart: clear all completion, show the whole model, hide the UI.
        Reached only via the typed 'reset' command, not the X button."""
        for r in self.done:
            for s in self.done[r]:
                self.done[r][s] = False
        self.done7 = False
        self.close_menu()
        self._notify({"event": "reset"})


class GearboxController:
    """Owns the ZMQ sockets, the shared state machine, and the click-listener loop."""

    def __init__(self, ip: str, cmd_port: int, click_port: int,
                 tg_ip: str = _TASKGRAPH_IP, tg_port: int = DEFAULT_TASKGRAPH_PORT):
        self.ip, self.cmd_port, self.click_port = ip, cmd_port, click_port
        self.tg_ip, self.tg_port = tg_ip, tg_port
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

        time.sleep(0.2)  # let PUB/SUB settle (slow-joiner guard)

        self._send_lock = threading.Lock()
        self._running = True
        self.sm = GearboxStateMachine(self.send, self.send_taskgraph)

    def send(self, msg: dict):
        """Thread-safe outbound send (called from both the REPL and the click thread)."""
        with self._send_lock:
            self._pub.send_string(json.dumps(msg))

    def send_taskgraph(self, msg: dict):
        """Thread-safe send of a semantic (row, stage) event to the task-graph viewer."""
        with self._send_lock:
            self._tg_pub.send_string(json.dumps(msg))

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
    parser.add_argument("--no-repl", action="store_true",
                        help="Run only the click listener (no typed REPL).")
    args = parser.parse_args()

    ctrl = GearboxController(args.ip, args.cmd_port, args.click_port,
                             args.task_graph_ip, args.task_graph_port)

    print(BANNER)
    print(f" commands   OUT -> tcp://{args.ip}:{args.cmd_port}")
    print(f" clicks     IN  <- tcp://{args.ip}:{args.click_port}")
    print(f" task-graph OUT -> tcp://{args.task_graph_ip}:{args.task_graph_port}\n")

    click_thread = None
    try:
        if args.no_repl:
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
