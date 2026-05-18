"""
main_hand.py — Unity hand tracking receiver + Open3D world-frame visualizer.

Receives real (and optionally synthetic) hand joint data from Unity via ZMQ,
applies T_world_tracking broadcast by main.py, and renders both hands as
point clouds + bone line sets in an Open3D window anchored to the world frame.

Expected Unity JSON format
--------------------------
{
  "left":  { "is_valid": true, "joints": [[x,y,z], ...] },
  "right": { "is_valid": true, "joints": [[x,y,z], ...] }
}
Joint coordinates are in Unity tracking space (left-handed, Y-up, Z-forward).

Usage
-----
  # Run main.py first so the world transform is being broadcast, then:
  python main_hand.py
  python main_hand.py --unity-ip 192.168.50.201 --hand1-port 5570
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import zmq

_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

import ip_setting as cfg
from utils.unity_conversion import HAND_BONES

# ── coordinate conversion ──────────────────────────────────────────────────────

def _unity_to_o3d(pts_unity: np.ndarray) -> np.ndarray:
    """(N,3) Unity tracking frame → Open3D tracking frame  (swap Y↔Z)."""
    return pts_unity[:, [0, 2, 1]]


def _to_world(joints_unity: np.ndarray, T_world_tracking: np.ndarray) -> np.ndarray:
    """(N,3) Unity tracking-frame joints → Open3D world frame."""
    pts = _unity_to_o3d(joints_unity)
    R, t = T_world_tracking[:3, :3], T_world_tracking[:3, 3]
    return (pts @ R.T) + t


def _extract_joints(hand_block) -> np.ndarray | None:
    """
    Pull joint positions out of a Unity hand block.
    Returns (N,3) float64 ndarray, or None if the hand is absent / not tracked.
    """
    if hand_block is None:
        return None
    if not hand_block.get("is_valid", True):
        return None
    joints = hand_block.get("joints")
    if not joints:
        return None
    return np.array(joints, dtype=np.float64)


# =============================================================================
# Open3D hand visualizer
# =============================================================================

_BONES_NP  = np.array(HAND_BONES, dtype=np.int32)
_N_JOINTS  = int(_BONES_NP.max()) + 1          # minimum number of joints needed
_HIDDEN_PT = np.array([[0., -100., 0.]])        # parked far away when not tracked


class _HandVis:
    """Open3D window showing left + right hand joints in world frame."""

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Hand Tracking — World Frame", width=1000, height=680)
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.10])
        ro.point_size = 7.0
        ro.line_width = 2.0

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(world_frame)

        self._pcd_l, self._lines_l = self._make_hand([0.3, 0.6, 1.0])   # blue
        self._pcd_r, self._lines_r = self._make_hand([1.0, 0.55, 0.1])  # orange

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0., 0., 0.])
        ctr.set_front([0., -0.5, -1.])
        ctr.set_up([0., 1., 0.])
        ctr.set_zoom(0.5)

    def _make_hand(self, color: list):
        dummy = np.tile(_HIDDEN_PT, (_N_JOINTS, 1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dummy)
        pcd.paint_uniform_color(color)
        self.vis.add_geometry(pcd)

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(dummy)
        lines.lines  = o3d.utility.Vector2iVector(_BONES_NP)
        lines.paint_uniform_color(color)
        self.vis.add_geometry(lines)

        return pcd, lines

    def _set_hand(self, pcd, lines, pts: np.ndarray | None):
        if pts is None or len(pts) == 0:
            pts_use = np.tile(_HIDDEN_PT, (_N_JOINTS, 1))
        else:
            pts_use = np.zeros((_N_JOINTS, 3))
            m = min(len(pts), _N_JOINTS)
            pts_use[:m] = pts[:m]

        pcd.points   = o3d.utility.Vector3dVector(pts_use)
        lines.points = o3d.utility.Vector3dVector(pts_use)
        self.vis.update_geometry(pcd)
        self.vis.update_geometry(lines)

    def update(self, left_pts: np.ndarray | None, right_pts: np.ndarray | None):
        self._set_hand(self._pcd_l, self._lines_l, left_pts)
        self._set_hand(self._pcd_r, self._lines_r, right_pts)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# =============================================================================
# Hand receiver
# =============================================================================

class HandReceiver:
    """
    SUBs to two ZMQ streams:
      • hand data       from Unity  (real + optional synthetic)
      • T_world_tracking from main.py (broadcast on WORLD_TRANSFORM_PORT)
    """

    def __init__(
        self,
        unity_ip: str        = cfg.UNITY_IP,
        hand1_port: int      = cfg.HAND1_PORT_FROM_UNITY,
        hand2_port: int | None = None,
        world_T_host: str    = cfg.LOCALHOST,
        world_T_port: int    = cfg.WORLD_TRANSFORM_PORT,
        rx_rate_hz: float    = 30.0,
        verbose: bool        = True,
    ):
        self.verbose          = verbose
        self._rx_interval     = 1.0 / rx_rate_hz if rx_rate_hz > 0 else 0.0
        self._last_rx         = 0.0

        self.real_hand_data      = None
        self.synthetic_hand_data = None
        self.T_world_tracking    = None   # (4,4) ndarray, updated continuously

        ctx = zmq.Context.instance()

        # SUB — real hand from Unity
        self._hand1_sub = ctx.socket(zmq.SUB)
        self._hand1_sub.setsockopt(zmq.CONFLATE, 1)
        self._hand1_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._hand1_sub.connect(f"tcp://{unity_ip}:{hand1_port}")
        if verbose:
            print(f"[HandReceiver] Hand1 SUB → tcp://{unity_ip}:{hand1_port}")

        # SUB — synthetic hand from Unity (optional)
        self._hand2_sub = None
        if hand2_port is not None:
            self._hand2_sub = ctx.socket(zmq.SUB)
            self._hand2_sub.setsockopt(zmq.CONFLATE, 1)
            self._hand2_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self._hand2_sub.connect(f"tcp://{unity_ip}:{hand2_port}")
            if verbose:
                print(f"[HandReceiver] Hand2 SUB → tcp://{unity_ip}:{hand2_port}")

        # SUB — world transform from main.py
        self._world_T_sub = ctx.socket(zmq.SUB)
        self._world_T_sub.setsockopt(zmq.CONFLATE, 1)
        self._world_T_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._world_T_sub.connect(f"tcp://{world_T_host}:{world_T_port}")
        if verbose:
            print(f"[HandReceiver] WorldT  SUB → tcp://{world_T_host}:{world_T_port}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _should_rx(self) -> bool:
        if self._rx_interval <= 0.0:
            return True
        now = time.time()
        if (now - self._last_rx) < self._rx_interval:
            return False
        self._last_rx = now
        return True

    @staticmethod
    def _drain(sock: zmq.Socket) -> str | None:
        latest = None
        while True:
            try:
                latest = sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        return latest

    # ── public API ────────────────────────────────────────────────────────────

    def poll(self) -> bool:
        """Drain all sockets. Returns True if new hand data arrived."""
        # world transform — always drain, no rate limit
        raw_T = self._drain(self._world_T_sub)
        if raw_T is not None:
            try:
                self.T_world_tracking = np.array(
                    json.loads(raw_T)["T"], dtype=np.float64).reshape(4, 4)
            except Exception as e:
                if self.verbose:
                    print(f"[HandReceiver] world_T parse error: {e}")

        if not self._should_rx():
            return False

        got = False

        raw1 = self._drain(self._hand1_sub)
        if raw1 is not None:
            try:
                self.real_hand_data = json.loads(raw1)
                got = True
            except Exception as e:
                if self.verbose:
                    print(f"[HandReceiver] hand1 parse error: {e}")

        if self._hand2_sub is not None:
            raw2 = self._drain(self._hand2_sub)
            if raw2 is not None:
                try:
                    self.synthetic_hand_data = json.loads(raw2)
                    got = True
                except Exception as e:
                    if self.verbose:
                        print(f"[HandReceiver] hand2 parse error: {e}")

        return got

    def world_joints(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Convert the latest real hand data to Open3D world frame.
        Returns (left_pts, right_pts), each (N,3) ndarray or None.
        T_world_tracking must be set (i.e. ArUco anchor locked in main.py).
        """
        T   = self.T_world_tracking
        data = self.real_hand_data
        if data is None or T is None:
            return None, None

        left_pts = right_pts = None
        j = _extract_joints(data.get("left"))
        if j is not None:
            left_pts = _to_world(j, T)
        j = _extract_joints(data.get("right"))
        if j is not None:
            right_pts = _to_world(j, T)

        return left_pts, right_pts

    def close(self):
        for s in (self._hand1_sub, self._hand2_sub, self._world_T_sub):
            if s is not None:
                try:
                    s.close(0)
                except Exception:
                    pass


# =============================================================================
# Run loop
# =============================================================================

def run(unity_ip, hand1_port, hand2_port, world_T_host, world_T_port, loop_hz):
    receiver = HandReceiver(
        unity_ip      = unity_ip,
        hand1_port    = hand1_port,
        hand2_port    = hand2_port,
        world_T_host  = world_T_host,
        world_T_port  = world_T_port,
        verbose       = True,
    )
    vis = _HandVis()

    dt = 1.0 / loop_hz if loop_hz > 0 else 0.0
    print(f"\n[Running]  unity_ip={unity_ip}  hand1_port={hand1_port}")
    print("  Waiting for ArUco lock from main.py before joints appear in world frame.")
    print("  Ctrl-C to quit\n")

    try:
        while True:
            receiver.poll()
            left_pts, right_pts = receiver.world_joints()
            vis.update(left_pts, right_pts)
            if dt > 0:
                time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        vis.close()
        receiver.close()
        print("[Done]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Unity hand tracking → Open3D world-frame visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--unity-ip",     default=cfg.UNITY_IP)
    ap.add_argument("--hand1-port",   type=int, default=cfg.HAND1_PORT_FROM_UNITY)
    ap.add_argument("--hand2-port",   type=int, default=None,
                    help="Enable synthetic hand stream (default: disabled)")
    ap.add_argument("--world-T-host", default=cfg.LOCALHOST,
                    help="Host running main.py (world transform publisher)")
    ap.add_argument("--world-T-port", type=int, default=cfg.WORLD_TRANSFORM_PORT)
    ap.add_argument("--hz",           type=float, default=30.0)
    args = ap.parse_args()

    run(
        unity_ip     = args.unity_ip,
        hand1_port   = args.hand1_port,
        hand2_port   = args.hand2_port,
        world_T_host = args.world_T_host,
        world_T_port = args.world_T_port,
        loop_hz      = args.hz,
    )


if __name__ == "__main__":
    main()
