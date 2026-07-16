#!/usr/bin/env python3
"""
gearbox_control.py — simple REPL to drive the Unity gearbox visualization.

Type commands; the Unity scene reacts live:

    row 1 / row 2 / row 3 / row 4   show ONLY that row of gears (hide the rest)
    show all  (or: all)             make every row visible again
    reset                           clear all color highlights
    <part name>                     toggle a single part's color (yellow <-> original);
                                    e.g. BearingRow3Right, GearRow2Left, GearRodRow4
    quit / exit / q                 leave

Mirrors the repo's networking convention (see main_hand_m100_w_sim.py): Unity BINDS a
NetMQ SubscriberSocket and this script CONNECTS a ZeroMQ PUB socket, sending JSON strings.
"""

import argparse
import json
import sys
import time

import zmq

try:
    import ip_setting
    _DEFAULT_IP = ip_setting.UNITY_IP
except Exception:
    _DEFAULT_IP = "127.0.0.1"

DEFAULT_PORT = 5019


def build_command(raw: str):
    """Parse a line of user input into a command dict, or None to ignore."""
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

    # Anything else: treat the original-case text as a part name to toggle.
    return {"command": "toggle", "part": text}


def describe(cmd: dict) -> str:
    c = cmd.get("command")
    if c == "row":
        return f"show only Row{cmd['row']}"
    if c == "show_all":
        return "show all rows"
    if c == "reset":
        return "reset all highlights"
    if c == "toggle":
        return f"toggle color of '{cmd['part']}'"
    return c


BANNER = """\
--------------------------------------------------------------
 Gearbox control  ->  Unity
--------------------------------------------------------------
 Commands:
   row 1 | row 2 | row 3 | row 4   show only that row
   show all                        show every row
   reset                           clear all color highlights
   <PartName>                      toggle a part's color (yellow)
                                   e.g. BearingRow3Right, GearRodRow4
   quit                            exit
--------------------------------------------------------------"""


def main():
    parser = argparse.ArgumentParser(description="REPL controller for the Unity gearbox scene.")
    parser.add_argument("--ip", default=_DEFAULT_IP,
                        help=f"IP of the machine running Unity (default: {_DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port GearboxCommandReceiver binds to (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://{args.ip}:{args.port}")
    time.sleep(0.2)  # let the PUB/SUB connection settle (avoid ZMQ slow-joiner drop)

    print(BANNER)
    print(f" Connected PUB -> tcp://{args.ip}:{args.port}\n")

    try:
        while True:
            try:
                raw = input("gearbox> ")
            except EOFError:
                print()
                break

            cmd = build_command(raw)
            if cmd is None:
                continue
            if cmd["command"] == "quit":
                break

            pub.send_string(json.dumps(cmd))
            print(f"  -> {describe(cmd)}")
    except KeyboardInterrupt:
        print()
    finally:
        pub.close()
        ctx.term()
        print("Bye.")


if __name__ == "__main__":
    sys.exit(main())
