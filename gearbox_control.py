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
go out on 5019 (GearboxCommandReceiver); clicks come in on 5020. ZeroMQ sockets are not
thread-safe, so every outbound send is serialized through one lock.

NOTE: main_setting.py assigns 5020 to ROBOT_CMD_PORT; don't run the robot-control server and
this click listener at the same time.
"""

import argparse
import json
import sys
import threading
import time

import zmq

try:
    import ip_setting
    _DEFAULT_IP = ip_setting.UNITY_IP
except Exception:
    _DEFAULT_IP = "127.0.0.1"

DEFAULT_CMD_PORT   = 5019   # Python -> Unity (commands), GearboxCommandReceiver.
DEFAULT_CLICK_PORT = 5020   # Unity -> Python (clicks), GearboxClickPublisher.

# Part-type -> assembly-state membership. Types are the prefix before "Row" in a part name.
STATE1_TYPES = {"GearRod", "Gear", "Pin"}
STATE2_TYPES = {"Stand", "Bearing", "Screw"}

CHECKBOX_NAME = "__checkbox__"
RESET_NAME    = "__reset__"


def parse_part(name: str):
    """'BearingRow3Left' -> ('Bearing', 3). Returns (None, None) if it isn't a Row part."""
    idx = name.find("Row")
    if idx < 0:
        return None, None
    ptype = name[:idx]
    digits = ""
    for ch in name[idx + 3:]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return None, None
    return ptype, int(digits)


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
    """Per-row assembly-state logic, driven by clicks. State lives as attributes and is mutated
    in the handlers, matching the house style of _ToolSelectionManager (main_with_robot.py)."""

    def __init__(self, send):
        self._send = send                       # send(dict) -> publishes a command to Unity
        self.done = {r: {1: False, 2: False, 3: False} for r in range(1, 5)}
        self.current_row = None
        self.current_state = None

    def handle_click(self, name: str):
        if name == CHECKBOX_NAME:
            self._handle_checkbox()
        elif name == RESET_NAME:
            self.close_menu()
        else:
            self._handle_part(name)

    def _handle_part(self, name: str):
        ptype, row = parse_part(name)
        if row is None:
            print(f"  (ignored '{name}': not a Row part)")
            return

        if ptype in STATE1_TYPES:
            state = 1
        elif ptype in STATE2_TYPES:
            state = 2
        else:
            print(f"  (ignored '{name}': type '{ptype}' not mapped to a state)")
            return

        # Both states of this row complete -> next click reveals the whole row (state 3),
        # which shows its own checkbox (marks the whole row complete) and close button.
        if self.done[row][1] and self.done[row][2]:
            self.current_row, self.current_state = row, 3
            self._send({"command": "row", "row": row})
            self._send({"command": "ui", "show": True, "row": row,
                        "checked": self.done[row][3]})
            print(f"  row {row}: state 3 (whole row)  done3={self.done[row][3]}")
            return

        # Otherwise show the clicked part's state and its checkbox.
        self.current_row, self.current_state = row, state
        types = sorted(STATE1_TYPES) if state == 1 else sorted(STATE2_TYPES)
        self._send({"command": "show_subset", "row": row, "types": types})
        self._send({"command": "ui", "show": True, "row": row,
                    "checked": self.done[row][state]})
        print(f"  row {row}: state {state}  (done={self.done[row]})")

    def _handle_checkbox(self):
        if self.current_row is None or self.current_state not in (1, 2, 3):
            print("  (checkbox ignored: no active state)")
            return
        row, state = self.current_row, self.current_state
        self.done[row][state] = not self.done[row][state]
        self._send({"command": "ui", "show": True, "row": row,
                    "checked": self.done[row][state]})
        print(f"  row {row}: state {state} done={self.done[row][state]}")

    def close_menu(self):
        """The X button: show the whole model again and hide the UI, but KEEP completion state
        (so partially/fully completed rows stay completed)."""
        self.current_row = self.current_state = None
        self._send({"command": "show_all"})
        self._send({"command": "ui", "show": False})

    def reset(self):
        """Full restart: clear all completion, show the whole model, hide the UI.
        Reached only via the typed 'reset' command, not the X button."""
        for r in self.done:
            self.done[r][1] = self.done[r][2] = self.done[r][3] = False
        self.close_menu()


class GearboxController:
    """Owns the ZMQ sockets, the shared state machine, and the click-listener loop."""

    def __init__(self, ip: str, cmd_port: int, click_port: int):
        self.ip, self.cmd_port, self.click_port = ip, cmd_port, click_port
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{cmd_port}")

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip}:{click_port}")

        time.sleep(0.2)  # let PUB/SUB settle (slow-joiner guard)

        self._send_lock = threading.Lock()
        self._running = True
        self.sm = GearboxStateMachine(self.send)

    def send(self, msg: dict):
        """Thread-safe outbound send (called from both the REPL and the click thread)."""
        with self._send_lock:
            self._pub.send_string(json.dumps(msg))

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
        # Context is shared (Context.instance()); leave it for the process to reclaim.


BANNER = """\
--------------------------------------------------------------
 Gearbox control  <->  Unity   (typed commands + clicks)
--------------------------------------------------------------
 Type:
   row 1 | row 2 | row 3 | row 4   show only that row
   show all                        show every row
   reset                           full restart (model + highlights + clear progress)
   <PartName>                      toggle a part's color (yellow)
                                   e.g. BearingRow3Right, GearRodRow4
   quit                            exit
 Click (in Unity):
   a part -> isolate its row through assembly states 1 -> 2 -> whole row
   the checkbox marks a state done; the X closes the menu (keeps progress)
--------------------------------------------------------------"""


def main():
    parser = argparse.ArgumentParser(description="Typed + click controller for the Unity gearbox scene.")
    parser.add_argument("--ip", default=_DEFAULT_IP,
                        help=f"IP of the machine running Unity (default: {_DEFAULT_IP})")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT,
                        help=f"Port GearboxCommandReceiver binds to (default: {DEFAULT_CMD_PORT})")
    parser.add_argument("--click-port", type=int, default=DEFAULT_CLICK_PORT,
                        help=f"Port Unity publishes clicks on (default: {DEFAULT_CLICK_PORT})")
    parser.add_argument("--no-repl", action="store_true",
                        help="Run only the click listener (no typed REPL).")
    args = parser.parse_args()

    ctrl = GearboxController(args.ip, args.cmd_port, args.click_port)

    print(BANNER)
    print(f" commands OUT -> tcp://{args.ip}:{args.cmd_port}")
    print(f" clicks  IN  <- tcp://{args.ip}:{args.click_port}\n")

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

        ctrl.send(cmd)
        print(f"  -> {describe(cmd)}")


if __name__ == "__main__":
    sys.exit(main())
