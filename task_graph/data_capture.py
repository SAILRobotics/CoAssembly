"""
data_capture.py — Standalone data-collection script for the "DataCapture" Unity scene.

Workflow
--------
  1. Hold marker 100 visible within 0.5 m — the world frame auto-locks the
     first time it's seen that close, or press ENTER while it's visible from
     any distance. Press ENTER again any time marker 100 is visible to relock
     (corrects for drift).
  2. Once running, the camera feed, hand tracking, and CenterEyeAnchor pose
     are continuously received from Unity and visualized in an Open3D window:
     camera frustum, head frustum, hand skeletons, and world/tracking origin.
  3. Press SPACE to start recording, press SPACE again to stop. Each session
     is written to captures/<timestamp>/:
       frames/######.jpg         — camera frames (BGR JPEG), one per recorded tick
       camera_intrinsics.json    — fx, fy, cx, cy, width, height (session-constant)
       poses.jsonl               — one JSON record per frame: image filename,
                                    timestamp, camera extrinsics (world frame,
                                    4x4 row-major flat), CenterEyeAnchor pose
                                    (world frame), and left/right hand joints
                                    (world frame, Nx3), all in Open3D/world
                                    coordinates (metres).

Keys (OpenCV window must be focused)
--------------------------------------
  ENTER = lock / relock world frame to marker 100
  SPACE = start / stop recording
  ESC   = quit

Usage
-----
  python data_capture.py
  python data_capture.py --quest-ip 192.168.50.201 --hand-port 5570
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import zmq

_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from utils.pose_helpers import _adapt_cx_cy
import main_setting as cfg
from main_with_robot import (
    _CamFeedReceiver, _ArucoPoseEstimator, _ArUcoWorker,
    _HandDataReceiver, _WorldAnchor,
    _SceneVis,
)


# =============================================================================
# Recorder
# =============================================================================

class _Recorder:
    """Saves camera frames + world-frame poses to captures/<timestamp>/."""

    OUT_ROOT = _FILE_DIR / "captures"

    def __init__(self):
        self._active      = False
        self._dir: "Path | None"        = None
        self._frames_dir: "Path | None" = None
        self._poses_f = None
        self.frame_idx = 0

    @property
    def active(self) -> bool:
        return self._active

    def toggle(self, camera_intrinsics: dict) -> None:
        if self._active:
            self.stop()
        else:
            self.start(camera_intrinsics)

    def start(self, camera_intrinsics: dict) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._dir        = self.OUT_ROOT / ts
        self._frames_dir = self._dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "camera_intrinsics.json").write_text(
            json.dumps(camera_intrinsics, indent=2))
        self._poses_f = open(self._dir / "poses.jsonl", "w")
        self.frame_idx = 0
        self._active = True
        print(f"[Recorder] Recording -> {self._dir}")

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._poses_f is not None:
            self._poses_f.close()
            self._poses_f = None
        print(f"[Recorder] Stopped - {self.frame_idx} frame(s) saved to {self._dir}")

    def record(self, frame: "np.ndarray | None",
               T_world_camera: "np.ndarray | None",
               T_world_center: "np.ndarray | None",
               left_pts: "np.ndarray | None",
               right_pts: "np.ndarray | None") -> None:
        if not self._active or frame is None:
            return
        fname = f"{self.frame_idx:06d}.jpg"
        cv.imwrite(str(self._frames_dir / fname), frame)
        record = {
            "frame": self.frame_idx,
            "t": time.time(),
            "image": f"frames/{fname}",
            "camera_extrinsics_world": (T_world_camera.flatten().tolist()
                                        if T_world_camera is not None else None),
            "center_eye_world": (T_world_center.flatten().tolist()
                                 if T_world_center is not None else None),
            "left_hand_world":  (left_pts.tolist()  if left_pts  is not None else None),
            "right_hand_world": (right_pts.tolist() if right_pts is not None else None),
        }
        self._poses_f.write(json.dumps(record) + "\n")
        self.frame_idx += 1

    def close(self) -> None:
        self.stop()


# =============================================================================
# DataCaptureScene
# =============================================================================

class DataCaptureScene:
    _AUTO_LOCK_MAX_DIST     = 0.5    # metres — auto-lock-on-sight only within this range
    _AUTO_LOCK_MAX_TILT_DEG = 20.0   # degrees — max tilt from vertical to auto-lock
    _PINCH_THRESHOLD        = 0.8    # index-pinch strength (0-1) counted as "pinched"

    def __init__(self, quest_ip: str, anchor_marker_id: int,
                 anchor_marker_size_m: float, hand_port: int):
        self.quest_ip         = quest_ip
        self.anchor_marker_id = anchor_marker_id
        self.hand_port        = hand_port

        self.cam    = _CamFeedReceiver(quest_ip)
        self.aruco  = _ArucoPoseEstimator(
            anchor_marker_id       = anchor_marker_id,
            pegboard_marker_id     = cfg.PEGBOARD_MARKER_ID,
            anchor_marker_size_m   = anchor_marker_size_m,
            pegboard_marker_size_m = cfg.PEGBOARD_MARKER_SIZE,
        )
        self.aruco_worker = _ArUcoWorker(self.cam, self.aruco)
        self.hands        = _HandDataReceiver(quest_ip, hand_port)
        self.anchor       = _WorldAnchor(quest_ip)
        self.recorder     = _Recorder()

        # Rising-edge pinch state: left hand toggles recording, right hand locks/relocks
        # the world frame — mirrors SPACE/ENTER but via Unity hand-tracking input instead
        # of the PC keyboard.
        self._left_pinching_prev  = False
        self._right_pinching_prev = False

        ctx = zmq.Context.instance()
        self._color_pub = ctx.socket(zmq.PUB)
        self._color_pub.connect(f"tcp://{quest_ip}:{cfg.TOOL_COLOR_PORT}")
        time.sleep(0.2)

        self.vis  = _SceneVis("Data Capture - World Frame")
        self._win = (f"Data Capture  [ENTER=lock/relock  SPACE=record  ESC=quit]"
                     f"  anchor=#{anchor_marker_id}")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)

        self._send_cube_color([1.0, 0.2, 0.2, 1.0])  # start red (not recording)

        print(f"\n[Running]  quest_ip={quest_ip}  "
              f"anchor_marker=#{anchor_marker_id}  hand_port={hand_port}")
        print(f"  Marker #{anchor_marker_id} auto-locks world on first sight within "
              f"{self._AUTO_LOCK_MAX_DIST:.1f}m (or press ENTER from any distance)"
              f" - press ENTER any time it's visible to relock")
        print("  SPACE = start/stop recording   ENTER = lock/relock   ESC = quit")
        print("  Left-hand pinch = start/stop recording   "
              "Right-hand pinch = lock/relock\n")

    def _send_cube_color(self, color: list) -> None:
        msg = {"tool_id": int(self.anchor_marker_id), "color": [float(c) for c in color]}
        try:
            self._color_pub.send_string(json.dumps(msg))
        except Exception as e:
            print(f"[DataCapture] Color send error: {e}")

    # ── Shared actions (keyboard + pinch trigger both of these) ─────────────────

    def _camera_intrinsics(self) -> dict:
        return {
            "fx": self.cam.fx, "fy": self.cam.fy,
            "cx": self.cam.cx, "cy": self.cam.cy,
            "width": self.cam.width, "height": self.cam.height,
            "sensor_width": self.cam.sensor_width,
            "sensor_height": self.cam.sensor_height,
        }

    def _lock_or_relock(self, T_cam_anchor, source: str) -> None:
        if self.cam.camera_T is None or T_cam_anchor is None:
            print(f"[{source}] Marker #{self.anchor_marker_id} not visible - cannot lock.")
            return
        self.anchor.lock(T_cam_anchor, self.cam.camera_T)
        print(f"[{source}] {'Relocked' if self.anchor.locked else 'Locked'} world to marker "
              f"#{self.anchor_marker_id}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while True:
                self.hands.poll()

                T_cam_anchor, _, _, det_vis = self.aruco_worker.get()
                anchor_ok = T_cam_anchor is not None

                _center_T = self.hands.center_eye_T()

                if self.anchor.locked:
                    self.anchor.publish()

                T_wt = self.anchor.T_world_tracking
                T_world_camleft = (self.anchor.world_T(self.cam.camera_T)
                                   if self.cam.camera_T is not None else None)
                T_world_center  = (self.anchor.world_T(_center_T)
                                   if _center_T is not None else None)
                left_pts, right_pts = self.hands.world_joints(T_wt)

                # ── Auto-lock anchor on first sight ───────────────────────────
                dist_to_anchor = (
                    float(np.linalg.norm(T_cam_anchor[:3, 3]))
                    if anchor_ok and self.cam.camera_T is not None else float('inf'))
                _cos_tilt = (T_cam_anchor[2, 3] / dist_to_anchor
                             if anchor_ok and dist_to_anchor > 1e-6 else 0.0)
                _min_cos  = np.cos(np.deg2rad(self._AUTO_LOCK_MAX_TILT_DEG))
                if (not self.anchor.locked and anchor_ok
                        and self.cam.camera_T is not None
                        and dist_to_anchor < self._AUTO_LOCK_MAX_DIST
                        and _cos_tilt > _min_cos):
                    self.anchor.lock(T_cam_anchor, self.cam.camera_T)
                    print(f"[AutoLock] Locked world to marker "
                          f"#{self.anchor_marker_id} ({dist_to_anchor:.2f} m)")

                # ── Recording ──────────────────────────────────────────────────
                if self.recorder.active:
                    self.recorder.record(self.cam.frame, T_world_camleft,
                                         T_world_center, left_pts, right_pts)

                # ── Open3D visualizer ─────────────────────────────────────────
                if self.cam.fx is not None:
                    fx, fy, cx, cy = _adapt_cx_cy(
                        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy,
                        self.cam.sensor_width, self.cam.sensor_height,
                        self.cam.width, self.cam.height)
                    self.vis.update_cam_frustum(T_world_camleft,
                                                self.cam.width, self.cam.height,
                                                fx, fy, cx, cy)
                self.vis.update_tracking(T_wt)
                self.vis.update_head(T_world_center)
                self.vis.update_hands(left_pts, right_pts)
                self.vis.tick()

                # ── OpenCV display ────────────────────────────────────────────
                disp = cv.resize(
                    det_vis if det_vis is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8),
                    (960, 540))
                locked = self.anchor.locked
                cv.putText(disp,
                           f"Marker #{self.anchor_marker_id}: "
                           f"{'DETECTED' if anchor_ok else 'searching...'}",
                           (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 255, 80) if anchor_ok else (0, 80, 255), 2)
                cv.putText(disp,
                           f"World: {'LOCKED' if locked else 'waiting for marker #' + str(self.anchor_marker_id)}",
                           (12, 68), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 255, 150) if locked else (100, 100, 100), 2)
                hand_ok = left_pts is not None or right_pts is not None
                if hand_ok:
                    hand_status = ("L+R" if (left_pts is not None and right_pts is not None)
                                   else ("L" if left_pts is not None else "R"))
                elif self.hands.receiving:
                    hand_status = f"receiving #{self.hands.message_count}"
                else:
                    hand_status = f"waiting on port {self.hand_port}"
                cv.putText(disp, f"Hands: {hand_status}",
                           (12, 102), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 255, 200) if (hand_ok or self.hands.receiving)
                           else (100, 100, 100), 2)
                if self.recorder.active:
                    cv.putText(disp, f"REC  {self.recorder.frame_idx} frames",
                               (12, 136), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(disp, "ENTER=lock/relock  SPACE=record  ESC=quit",
                           (12, disp.shape[0] - 14),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv.imshow(self._win, disp)

                # ── Key handling ──────────────────────────────────────────────
                key = cv.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == 13:  # ENTER
                    self._lock_or_relock(T_cam_anchor, source="ENTER")
                elif key == 32:  # SPACE
                    self.recorder.toggle(self._camera_intrinsics())
                    self._send_cube_color([0.2, 1.0, 0.2, 1.0] if self.recorder.active
                                          else [1.0, 0.2, 0.2, 1.0])

                # ── Pinch handling (mirrors ENTER/SPACE via Unity hand input) ───
                left_pinch  = self.hands.pinch_strength("LeftHand")
                right_pinch = self.hands.pinch_strength("RightHand")
                left_pinching  = left_pinch  is not None and left_pinch  >= self._PINCH_THRESHOLD
                right_pinching = right_pinch is not None and right_pinch >= self._PINCH_THRESHOLD

                if left_pinching and not self._left_pinching_prev:
                    self.recorder.toggle(self._camera_intrinsics())
                    self._send_cube_color([0.2, 1.0, 0.2, 1.0] if self.recorder.active
                                          else [1.0, 0.2, 0.2, 1.0])
                if right_pinching and not self._right_pinching_prev:
                    self._lock_or_relock(T_cam_anchor, source="Pinch")

                self._left_pinching_prev  = left_pinching
                self._right_pinching_prev = right_pinching

                time.sleep(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._send_cube_color([1.0, 0.2, 0.2, 1.0])  # reset to red on exit
        self.recorder.close()
        self.aruco_worker.stop()
        self.vis.close()
        cv.destroyAllWindows()
        self.anchor.close()
        self.hands.close()
        self.cam.close()
        try:
            self._color_pub.close(0)
        except Exception:
            pass
        print("[Done]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Data capture: camera + hand tracking + world-frame recording.")
    ap.add_argument("--quest-ip",           default=cfg.UNITY_IP)
    ap.add_argument("--anchor-marker",      type=int,   default=cfg.ANCHOR_MARKER_ID,
                     help="ArUco marker id used to lock the world frame.")
    ap.add_argument("--anchor-marker-size", type=float, default=cfg.ANCHOR_MARKER_SIZE,
                     help="Anchor marker size in metres.")
    ap.add_argument("--hand-port",          type=int,   default=cfg.HAND1_PORT_FROM_UNITY)
    args = ap.parse_args()

    scene = DataCaptureScene(
        quest_ip             = args.quest_ip,
        anchor_marker_id     = args.anchor_marker,
        anchor_marker_size_m = args.anchor_marker_size,
        hand_port            = args.hand_port,
    )
    scene.run()


if __name__ == "__main__":
    main()
