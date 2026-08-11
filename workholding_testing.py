#!/usr/bin/env python3
"""workholding_testing.py — standalone AR-box placement test harness.

Parks the AR "workholding" box at 10 fixed poses spanning the robot workspace, one at a
time; the Left/Right arrow keys cycle through them. It keeps the SAME ArUco-100 lock +
relock setup as main_with_robot.py (lock marker 100 first — auto on sight, or ENTER — with
the in-headset proximity relock cube), but drops the robot / pegboard / gearbox / board-
tracking machinery. Every size and count comes from main_setting.py.

Reuses main_with_robot.py's receiver classes unchanged (import only):
  _CamFeedReceiver, _ArUcoWorker, _ArucoPoseEstimator, _HandDataReceiver, _WorldAnchor,
  _GripPoseBridge, _ToolSelectionManager, _RelockCubePublisher, _OffsetTuner.

The box is shown in the barebones WorkholdingTesting Unity scene by WorkholdingBoxReceiver
(bind GRIP_STATE_PORT 5012, same message schema as GripStateReceiver — but always visible).

Usage
-----
    python workholding_testing.py                 # uses main_setting UNITY_IP etc.
Keys: Left/Right arrows (or a/d) cycle poses; ENTER lock/relock marker 100; ESC quit.
"""

import argparse
import time

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

import main_setting as cfg
from main_with_robot import (
    _CamFeedReceiver,
    _ArUcoWorker,
    _ArucoPoseEstimator,
    _HandDataReceiver,
    _WorldAnchor,
    _GripPoseBridge,
    _ToolSelectionManager,
    _RelockCubePublisher,
    _WorkspaceBoundPublisher,
    _OffsetTuner,
)

# Arrow-key codes returned by cv.waitKeyEx() on Windows; a/d are accepted as a fallback.
_KEY_LEFT  = 2424832
_KEY_RIGHT = 2555904


class WorkholdingTest:
    # Same lock/relock tuning as MainScene.
    _RELOCK_COOLDOWN        = 2.0
    _AUTO_LOCK_MAX_DIST     = 1.0    # metres — auto-lock-on-sight only within this range
    _AUTO_LOCK_MAX_TILT_DEG = 45.0   # degrees — max tilt from vertical to auto-lock

    def __init__(self, quest_ip: str, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float, hand_port: int):
        self.quest_ip           = quest_ip
        self.anchor_marker_id   = anchor_marker_id
        self.pegboard_marker_id = pegboard_marker_id
        self.hand_port          = hand_port

        # ── Receivers (same construction as MainScene; robot/pegboard/gearbox omitted) ──
        self.cam = _CamFeedReceiver(quest_ip)
        _aruco = _ArucoPoseEstimator(
            anchor_marker_id       = anchor_marker_id,
            pegboard_marker_id     = pegboard_marker_id,
            anchor_marker_size_m   = anchor_marker_size_m,
            pegboard_marker_size_m = pegboard_marker_size_m,
            board_marker_ids       = (),          # no tracked board / secondary relock markers
            board_marker_size_m    = cfg.WORLD_MARKER_SIZE)
        self.aruco_worker     = _ArUcoWorker(self.cam, _aruco)
        self.hands            = _HandDataReceiver(quest_ip, hand_port)
        self.anchor           = _WorldAnchor(quest_ip)
        self.tools            = _ToolSelectionManager(quest_ip)
        self.tuner            = _OffsetTuner()
        self.relock_cubes     = _RelockCubePublisher(quest_ip)
        self.relock_cubes.set_markers(self.anchor._T_world_marker)
        # Publish the AR box on its OWN port (not GRIP_STATE_PORT 5012) so it never clashes with a
        # GripStateReceiver — WorkholdingBoxReceiver binds this port instead.
        self.grip_pose_bridge = _GripPoseBridge(quest_ip, grip_state_port=cfg.WORKHOLDING_BOX_PORT)
        self.workspace_bound_pub = _WorkspaceBoundPublisher(quest_ip)

        # Workspace boundary box (the volume the 10 poses live in). We publish a constant
        # dist_outside above the receiver's fadeDistance (0.3 m) so the wireframe stays fully
        # visible as a reference / lock indicator, rather than fading with head proximity.
        self._ws_lo = np.array(cfg.WORKSPACE_LO, dtype=np.float64)
        self._ws_hi = np.array(cfg.WORKSPACE_HI, dtype=np.float64)
        self._BOUNDS_VIS_DIST = 0.5

        # ── The 10 world-frame box poses (numbers + count come from main_setting) ──
        self._raw   = cfg.workholding_test_poses()
        self._poses = [self._pose_to_T(pos, euler) for pos, euler in self._raw]
        self._n     = len(self._poses)
        self._idx   = 0

        # ── Lock/relock state (ported from MainScene) ──
        self._last_proximity_relock_time = 0.0
        self._anchor_highlight_until      = 0.0
        self._prev_relock_available       = False
        self._box_pub_logged              = False

        self._win = "Workholding Box Test  [<-/-> pose   ENTER=lock/relock   ESC=quit]"
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)

    @staticmethod
    def _pose_to_T(pos, euler_deg) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('xyz', euler_deg, degrees=True).as_matrix()
        T[:3, 3]  = np.asarray(pos, dtype=np.float64)
        return T

    def _lock_initial(self, T_cam_anchor: np.ndarray) -> None:
        """First-time marker-100 lock (mirrors MainScene._lock_anchor_initial, minus robot)."""
        self.anchor.lock(T_cam_anchor, self.cam.camera_T)
        self._last_proximity_relock_time = time.time()

    def _publish_box(self) -> None:
        """Park the AR box at the current preset pose P by back-computing a TCP so that
        _GripPoseBridge.publish (box centre = TCP + BOX_FORWARD_OFFSET·gripperZ) lands the
        box exactly at P — reusing the real workholding box math unchanged."""
        P = self._poses[self._idx]
        T_tcp = np.eye(4, dtype=np.float64)
        T_tcp[:3, :3] = P[:3, :3]
        T_tcp[:3, 3]  = P[:3, 3] - cfg.BOX_FORWARD_OFFSET * P[:3, 2]
        self.grip_pose_bridge.publish("grabbed", T_tcp)
        if not self._box_pub_logged:
            self._box_pub_logged = True
            print(f"[Box] Publishing AR box -> tcp://{self.quest_ip}:{cfg.WORKHOLDING_BOX_PORT} "
                  f"(pose {self._idx + 1}/{self._n}, world pos {np.round(P[:3, 3], 3).tolist()})")

    def _cycle(self, delta: int) -> None:
        self._idx = (self._idx + delta) % self._n
        pos, euler = self._raw[self._idx]
        print(f"[Pose] {self._idx + 1}/{self._n}  pos={np.round(pos, 3).tolist()}  euler_deg={euler}")

    def run(self) -> None:
        try:
            while True:
                # ── Poll streams ──────────────────────────────────────────────
                self.tools.poll(timeout_ms=0)   # marker-100 relock-cube clicks
                self.hands.poll()               # center-eye pose

                T_cam_anchor, T_cam_pegboard, _T_cam_board, det_vis = self.aruco_worker.get()
                anchor_ok = T_cam_anchor is not None

                # ── Offset tuner (nudges the whole world root, like main) ──────
                self.tuner.draw()
                pos_off, yaw_off = self.tuner.get()
                self.anchor.set_offset(pos_off, yaw_off)

                _center_T = self.hands.center_eye_T()
                _now      = time.time()

                # ── Publish everything parented under WorldRoot only when locked ──
                # (WorldRoot is only positioned once the world is locked; publishing before
                # then would place the box / bounds at the Unity scene origin, out of view.)
                if self.anchor.locked:
                    self.anchor.publish()
                    self.relock_cubes.publish()
                    self.workspace_bound_pub.publish(
                        self._ws_lo, self._ws_hi, self._BOUNDS_VIS_DIST)
                    self._publish_box()

                dist_to_anchor = (float(np.linalg.norm(T_cam_anchor[:3, 3]))
                                  if anchor_ok and self.cam.camera_T is not None
                                  else float('inf'))

                # ── Auto-lock anchor on first sight (proximity + face-on) ─────
                _cos_tilt = (T_cam_anchor[2, 3] / dist_to_anchor
                             if anchor_ok and dist_to_anchor > 1e-6 else 0.0)
                _min_cos  = np.cos(np.deg2rad(self._AUTO_LOCK_MAX_TILT_DEG))
                if (not self.anchor.locked and anchor_ok
                        and self.cam.camera_T is not None
                        and dist_to_anchor < self._AUTO_LOCK_MAX_DIST
                        and _cos_tilt > _min_cos):
                    self._lock_initial(T_cam_anchor)
                    tilt_deg = float(np.degrees(np.arccos(np.clip(_cos_tilt, -1, 1))))
                    print(f"[AutoLock] Locked world to marker #{self.anchor_marker_id} "
                          f"({dist_to_anchor:.2f} m, {tilt_deg:.1f} deg tilt)")

                # ── Anchor marker proximity relock (marker-100 cube) ──────────
                _relock_available = (self.anchor.locked and anchor_ok
                                     and self.cam.camera_T is not None
                                     and dist_to_anchor < 1.0)
                if self._anchor_highlight_until > 0.0 and _now >= self._anchor_highlight_until:
                    self._anchor_highlight_until = 0.0
                    self._prev_relock_available  = not _relock_available
                if (self._anchor_highlight_until == 0.0
                        and _relock_available != self._prev_relock_available):
                    self.tools.send_color(
                        self.anchor_marker_id,
                        _ToolSelectionManager.HOVER_COLOR if _relock_available
                        else _ToolSelectionManager.RESET_COLOR)
                    self._prev_relock_available = _relock_available
                if (self.tools.active_tool_id == self.anchor_marker_id
                        and _relock_available
                        and _now - self._last_proximity_relock_time >= self._RELOCK_COOLDOWN):
                    self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True)
                    self.tools.send_color(self.anchor_marker_id,
                                          _ToolSelectionManager.SELECTED_COLOR)
                    self._anchor_highlight_until      = _now + 1.0
                    self._prev_relock_available       = True
                    self._last_proximity_relock_time  = _now
                    print("[Relock] Relocked via proximity click")
                self.tools.deselect(self.anchor_marker_id)

                # ── OpenCV display + key input surface ────────────────────────
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
                           "Anchor: "
                           + (f"LOCKED  -> :{cfg.WORLD_ROOT_PORT}" if locked
                              else f"waiting for marker #{self.anchor_marker_id}"),
                           (12, 68), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 255, 150) if locked else (100, 100, 100), 2)
                pos, euler = self._raw[self._idx]
                cv.putText(disp,
                           f"Box {self._idx + 1}/{self._n}   "
                           f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
                           f"euler={euler}",
                           (12, 102), cv.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
                cv.putText(disp,
                           "<- / ->  cycle box pose     ENTER  lock/relock #"
                           f"{self.anchor_marker_id}     ESC  quit",
                           (12, disp.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                           (200, 200, 200), 1)
                cv.imshow(self._win, disp)

                # ── Key handling (waitKeyEx for the arrow keys) ───────────────
                key = cv.waitKeyEx(1)
                if key == -1:
                    continue
                low = key & 0xFF
                if low == 27:                                   # ESC
                    break
                elif key == _KEY_LEFT or low == ord('a'):
                    self._cycle(-1)
                elif key == _KEY_RIGHT or low == ord('d'):
                    self._cycle(+1)
                elif low == 13:                                 # ENTER — lock / relock marker 100
                    if self.cam.camera_T is None:
                        print("[ENTER] No camera pose yet (is the passthrough feed on :"
                              f"{cfg.CAM_FEED_PORT} arriving?) — skipping.")
                    elif anchor_ok:
                        if self.anchor.locked:
                            self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True)
                            self._last_proximity_relock_time = _now
                            print(f"[ENTER] Relocked world to marker #{self.anchor_marker_id}")
                        else:
                            self._lock_initial(T_cam_anchor)
                            print(f"[ENTER] Locked world to marker #{self.anchor_marker_id}")
                    else:
                        print(f"[ENTER] Marker #{self.anchor_marker_id} not visible — cannot lock.")
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        # Stop the background ArUco worker and let it finish its in-flight cam.poll()
        # BEFORE we close the cam socket it uses — otherwise libzmq can crash during
        # interpreter teardown (daemon thread recv'ing on a terminating context).
        self.aruco_worker.stop()
        time.sleep(0.05)
        cv.destroyAllWindows()
        try:
            self.tuner.close()
        except Exception:
            pass
        # Explicitly close every ZMQ receiver/publisher (mirrors MainScene.close ordering).
        for obj in (self.anchor, self.relock_cubes, self.grip_pose_bridge,
                    self.workspace_bound_pub, self.hands, self.cam, self.tools):
            try:
                obj.close()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Workholding AR-box placement test harness (10 poses, arrow-key cycling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip",             default=cfg.UNITY_IP)
    ap.add_argument("--anchor-marker",        type=int,   default=cfg.ANCHOR_MARKER_ID)
    ap.add_argument("--pegboard-marker",      type=int,   default=cfg.PEGBOARD_MARKER_ID)
    ap.add_argument("--anchor-marker-size",   type=float, default=cfg.ANCHOR_MARKER_SIZE)
    ap.add_argument("--pegboard-marker-size", type=float, default=cfg.PEGBOARD_MARKER_SIZE)
    ap.add_argument("--hand-port",            type=int,   default=cfg.HAND1_PORT_FROM_UNITY)
    args = ap.parse_args()
    if args.anchor_marker == args.pegboard_marker:
        ap.error("--anchor-marker and --pegboard-marker must be different.")

    test = WorkholdingTest(args.quest_ip, args.anchor_marker, args.pegboard_marker,
                           args.anchor_marker_size, args.pegboard_marker_size, args.hand_port)
    print(f"\n[Workholding] Show marker #{args.anchor_marker} to lock the world "
          f"(auto within {WorkholdingTest._AUTO_LOCK_MAX_DIST:.1f} m, or press ENTER).")
    print(f"[Workholding] {test._n} box poses — cycle with Left/Right arrows (or a/d). ESC to quit.\n")
    test.run()
    print("Bye.")


if __name__ == "__main__":
    main()
