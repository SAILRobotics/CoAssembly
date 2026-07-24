"""test_tool_layout.py

Standalone tester: publish a tool_layout JSON into a Unity scene (e.g. VisualizationTesting)
without running the whole main_with_robot.py pipeline.

It reproduces `_ToolLayoutManager.publish()` from main_with_robot.py exactly — same pegboard→world
transform, same Open3D→Unity coordinate conversion, same message schema — and sends it on
TOOL_LAYOUT_PORT (5011), which ToolSpawner.cs binds. So dropping a ToolSpawner into
VisualizationTesting (prefabs wired by id, WorldRoot assigned) and entering Play is enough to see
the tools appear.

Networking: Unity BINDS the SUB on 0.0.0.0:5011; this script CONNECTs a PUB and sends. Because
there's no live ArUco anchor in a test scene, the pegboard pose is read from the saved scan
(scene_layout/T_world10_pegboard101.npz). With --identity (or no scan file) the tools are placed
directly in the pegboard frame instead.

The script republishes every --interval seconds (until Ctrl-C) so you can enter Play at any time and
still receive the layout; pass --once to send a single batch and exit.

It also plays the CONSUMER half of the pegboard-highlight loop (a dummy stand-in for
main_with_robot.py): it BINDs a SUB on GEARBOX_HIGHLIGHT_PORT (5024) for gearbox_control.py's
tool-id messages and colours the matching spawned tools via ToolColorReceiver (5010), owning the
HIGHLIGHT_COLOR itself. Disable with --no-highlight; it is skipped in --once mode.

Usage
-----
    python test_tool_layout.py                     # 127.0.0.1:5011, tool_layout1.json + scan pose
    python test_tool_layout.py --identity          # place tools in pegboard-frame coords
    python test_tool_layout.py --json scene_layout/tool_layout2.json
    python test_tool_layout.py --ip 192.168.1.50 --once
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Config: reuse the canonical port / IP / paths, with standalone fallbacks ───────────────────
try:
    import main_setting as cfg
    DEFAULT_IP        = getattr(cfg, "UNITY_IP", "127.0.0.1")
    DEFAULT_PORT      = getattr(cfg, "TOOL_LAYOUT_PORT", 5011)
    DEFAULT_COLOR_PORT     = getattr(cfg, "TOOL_COLOR_PORT", 5010)
    DEFAULT_HIGHLIGHT_PORT = getattr(cfg, "GEARBOX_HIGHLIGHT_PORT", 5024)
    SCENE_DIR         = getattr(cfg, "SCENE_LAYOUT_DIR", _HERE / "scene_layout")
except Exception:
    DEFAULT_IP, DEFAULT_PORT, SCENE_DIR = "127.0.0.1", 5011, _HERE / "scene_layout"
    DEFAULT_COLOR_PORT, DEFAULT_HIGHLIGHT_PORT = 5010, 5024

# Colour the pegboard tools flash when their gearbox part is the next thing to fetch. The CONSUMER
# owns this — gearbox_control.py sends only ids. Eventually main_with_robot.py picks it the same way
# it already owns SELECTED_COLOR / HOVER_COLOR in _ToolSelectionManager. RGBA, 0-1.
HIGHLIGHT_COLOR = [0.0, 1.0, 1.0, 1.0]   # cyan

# ── Open3D → Unity conversion: reuse the repo's helpers, with inline fallbacks ─────────────────
try:
    from utils.unity_conversion import (open3d_to_unity_vector,
                                         open3d_to_unity_quaternion)
except Exception:
    def open3d_to_unity_vector(pos):            # (x, y, z) → (x, z, y)
        return np.array([pos[0], pos[2], pos[1]])

    def open3d_to_unity_quaternion(q):          # [w, x, y, z] → [w, -x, -z, -y]
        w, x, y, z = q
        return [w, -x, -z, -y]

_PEGBOARD_NPZ = SCENE_DIR / "T_world10_pegboard101.npz"
_DEFAULT_JSON = SCENE_DIR / "tool_layout1.json"


def load_tools(json_path: Path) -> list:
    data = json.loads(Path(json_path).read_text())
    tools = data.get("tools", [])
    print(f"[TestToolLayout] Loaded {len(tools)} tool(s) from {Path(json_path).name}")
    return tools


def load_pegboard_T(identity: bool) -> np.ndarray:
    """Pegboard-in-world transform. Uses the saved scan pose (matching main_with_robot) unless
    --identity is set or the scan file is missing, in which case tools land in pegboard-frame."""
    if identity:
        print("[TestToolLayout] --identity: placing tools directly in the pegboard frame.")
        return np.eye(4)
    if not _PEGBOARD_NPZ.exists():
        print(f"[TestToolLayout] No pegboard scan at {_PEGBOARD_NPZ}; "
              "falling back to identity (pegboard-frame placement).")
        return np.eye(4)
    T = np.load(str(_PEGBOARD_NPZ))["T_world10_pegboard"]
    print(f"[TestToolLayout] Pegboard pose loaded from {_PEGBOARD_NPZ.name}")
    return np.asarray(T, dtype=np.float64)


def build_payload(tools: list, T: np.ndarray) -> dict:
    """Reproduce _ToolLayoutManager.publish() verbatim: pegboard→world base, add half-height to
    reach the prefab centroid, then convert to Unity coordinates + message schema."""
    out = []
    for t in tools:
        sz  = t.get("size", [0.05, 0.05, 0.05])
        rot = t.get("rotation_deg", [0.0, 0.0, 0.0])

        # peg_pos is the base in pegboard frame (new format); fall back to world_pos.
        R_world = ScipyR.from_euler('z', float(rot[2]), degrees=True).as_matrix()
        if "peg_pos" in t:
            base_w = (T @ np.append(t["peg_pos"], 1.0))[:3]
        else:
            base_w = np.array(t.get("world_pos", [0.0, 0.0, 0.0]))
        # Unity prefabs are centred at their local origin → send centroid.
        pos_w  = base_w + R_world @ np.array([0.0, 0.0, sz[2] / 2.0])
        q_xyzw = ScipyR.from_matrix(R_world).as_quat()

        pos_u  = open3d_to_unity_vector(pos_w)
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u    = open3d_to_unity_quaternion(q_wxyz)
        sz_u   = open3d_to_unity_vector(np.array(sz, dtype=float))

        out.append({
            "id":            int(t["id"]),
            "type":          t.get("type", "unknown"),
            "category":      t.get("category", "tool"),
            "position":      pos_u.tolist(),
            "rotation_xyzw": [float(q_u[1]), float(q_u[2]),
                              float(q_u[3]), float(q_u[0])],
            "size":          sz_u.tolist(),
        })
    return {"tools": out}


class ToolHighlightBridge:
    """Consumer half of the pegboard-highlight loop (dummy stand-in for main_with_robot.py).

    BINDs a SUB on GEARBOX_HIGHLIGHT_PORT (5024) — receiver binds, per the Python<->Python
    convention — and drains gearbox_control.py's `{"event":"highlight","ids":[...]}` /
    `{"event":"clear"}` messages on a background thread. It owns the colour (HIGHLIGHT_COLOR) and
    drives Unity's ToolColorReceiver over TOOL_COLOR_PORT (5010), reusing that schema exactly
    ({"tool_id":N,"color":[r,g,b,a]}, with color[0] < 0 as the restore-original sentinel).

    Replace semantics: each highlight message carries the full id set, so ids that dropped out of
    the set are restored and newly-appearing ids are coloured; a clear restores every tracked id."""

    _RESTORE = [-1.0, 0.0, 0.0, 0.0]   # ToolColorReceiver sentinel: color[0] < 0 -> original colour

    def __init__(self, ip: str, listen_port: int, color_port: int):
        self.ip, self.listen_port, self.color_port = ip, listen_port, color_port
        self._ctx = zmq.Context.instance()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.bind(f"tcp://0.0.0.0:{listen_port}")   # receiver binds

        self._color_pub = self._ctx.socket(zmq.PUB)
        self._color_pub.connect(f"tcp://{ip}:{color_port}")   # Unity ToolColorReceiver binds 5010
        time.sleep(0.2)   # slow-joiner guard so the first colour message isn't dropped

        self._highlighted: set[int] = set()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[ToolHighlight] SUB  bound tcp://0.0.0.0:{self.listen_port}  "
              f"(colours -> tcp://{self.ip}:{self.color_port})")

    def _send_color(self, tool_id: int, color):
        self._color_pub.send_string(json.dumps({"tool_id": int(tool_id), "color": color}))

    def _apply_highlight(self, ids):
        new = set(int(i) for i in ids)
        for tid in self._highlighted - new:      # dropped out of the set -> restore
            self._send_color(tid, self._RESTORE)
        for tid in new - self._highlighted:      # newly appearing -> highlight colour
            self._send_color(tid, HIGHLIGHT_COLOR)
        self._highlighted = new
        print(f"[ToolHighlight] highlight ids={sorted(new)}")

    def _apply_clear(self):
        for tid in self._highlighted:
            self._send_color(tid, self._RESTORE)
        if self._highlighted:
            print(f"[ToolHighlight] clear (restored {sorted(self._highlighted)})")
        self._highlighted = set()

    def _loop(self):
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
                except json.JSONDecodeError as e:
                    print(f"[ToolHighlight] bad message: {e}")
                    continue
                if msg.get("event") == "clear":
                    self._apply_clear()
                elif msg.get("event") == "highlight":
                    self._apply_highlight(msg.get("ids", []))

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._apply_clear()   # leave the pegboard clean
        try: self._sub.close()
        except Exception: pass
        try: self._color_pub.close()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser(description="Publish a tool_layout JSON into a Unity test scene.")
    ap.add_argument("--ip",       default=DEFAULT_IP,   help=f"Unity host (default {DEFAULT_IP})")
    ap.add_argument("--port",     type=int, default=DEFAULT_PORT,
                    help=f"TOOL_LAYOUT_PORT that ToolSpawner binds (default {DEFAULT_PORT})")
    ap.add_argument("--json",     type=Path, default=_DEFAULT_JSON,
                    help=f"tool layout JSON (default {_DEFAULT_JSON})")
    ap.add_argument("--identity", action="store_true",
                    help="place tools in the pegboard frame (ignore the saved scan pose)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between republishes (default 1.0)")
    ap.add_argument("--once",     action="store_true",
                    help="send a single batch and exit instead of republishing")
    ap.add_argument("--color-port", type=int, default=DEFAULT_COLOR_PORT,
                    help=f"TOOL_COLOR_PORT that ToolColorReceiver binds (default {DEFAULT_COLOR_PORT})")
    ap.add_argument("--highlight-port", type=int, default=DEFAULT_HIGHLIGHT_PORT,
                    help=f"port gearbox_control.py sends highlight ids on (default {DEFAULT_HIGHLIGHT_PORT})")
    ap.add_argument("--no-highlight", action="store_true",
                    help="skip the pegboard tool-highlight listener (layout publishing only)")
    args = ap.parse_args()

    tools = load_tools(args.json)
    T     = load_pegboard_T(args.identity)
    payload = build_payload(tools, T)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://{args.ip}:{args.port}")
    time.sleep(0.2)   # slow-joiner guard so the first message isn't dropped
    print(f"[TestToolLayout] PUB → tcp://{args.ip}:{args.port}  "
          f"({len(payload['tools'])} tool(s): ids {[t['id'] for t in payload['tools']]})")

    # The highlight listener is a live loop, so it only runs in republish mode (not --once).
    bridge = None
    if not args.no_highlight and not args.once:
        bridge = ToolHighlightBridge(args.ip, args.highlight_port, args.color_port)
        bridge.start()

    msg = json.dumps(payload)
    try:
        if args.once:
            pub.send_string(msg)
            print("[TestToolLayout] Sent one batch (--once). Done.")
        else:
            print(f"[TestToolLayout] Republishing every {args.interval}s — Ctrl-C to stop. "
                  "Enter Play in Unity any time.")
            while True:
                pub.send_string(msg)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[TestToolLayout] Stopped.")
    finally:
        if bridge is not None:
            bridge.stop()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
