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
    DEFAULT_IP   = getattr(cfg, "UNITY_IP", "127.0.0.1")
    DEFAULT_PORT = getattr(cfg, "TOOL_LAYOUT_PORT", 5011)
    SCENE_DIR    = getattr(cfg, "SCENE_LAYOUT_DIR", _HERE / "scene_layout")
except Exception:
    DEFAULT_IP, DEFAULT_PORT, SCENE_DIR = "127.0.0.1", 5011, _HERE / "scene_layout"

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
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
