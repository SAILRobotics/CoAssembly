"""world_marker_prescan.py

Pre-scanning tool for registering 5 ArUco markers relative to each other.

Pipeline
--------
  Phase – Show all 5 markers (1 reference + 4 secondary) and walk the camera
          between them to collect pose samples. When each secondary marker has
          been seen co-visible with a registered marker 30+ times, compute and
          lock its relative transform.
          Press Q to finish early; auto-finishes when all 5 are registered.

Output
------
  world_markers_T_ref_from_marker.json  – raw 4x4 transform matrices
  world_markers_summary.json            – human-readable distances + rotations
                                          from reference marker (100)
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from record3d import Record3DStream
from scipy.spatial.transform import Rotation as ScipyR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_MARKER_ID  = 100
SECONDARY_MARKER_IDS = (104, 105, 106, 107)
MARKER_SIZE          = 0.10                    # metres
N_SAMPLES            = 30
SCENE_LAYOUT_DIR     = Path("scene_layout")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_rotation_matrix(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx = s*qw*qx; wy = s*qw*qy; wz = s*qw*qz
    xx = s*qx*qx; xy = s*qx*qy; xz = s*qx*qz
    yy = s*qy*qy; yz = s*qy*qz; zz = s*qz*qz
    return np.array([
        [1-(yy+zz),  xy-wz,   xz+wy],
        [ xy+wz,  1-(xx+zz),  yz-wx],
        [ xz-wy,   yz+wx,  1-(xx+yy)],
    ], dtype=np.float64)


def _average_se3(T_list):
    """Geodesic mean of SE3 transforms via Lie-algebra log / exp."""
    if not T_list:
        return np.eye(4)
    logs = []
    for T in T_list:
        rv = ScipyR.from_matrix(T[:3, :3]).as_rotvec()
        logs.append(np.concatenate([T[:3, 3], rv]))
    avg = np.mean(logs, axis=0)
    out = np.eye(4)
    out[:3, :3] = ScipyR.from_rotvec(avg[3:]).as_matrix()
    out[:3, 3] = avg[:3]
    return out


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class WorldMarkerPrescanApp:
    DEVICE_TYPE__TRUEDEPTH = 0
    DEVICE_TYPE__LIDAR     = 1

    def __init__(self, device_idx: int = 0,
                 aruco_dict_id: int = cv2.aruco.DICT_6X6_1000):
        self.device_idx = device_idx
        self.event = Event()
        self.session = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Registration state: id -> T_ref_from_marker
        self.R = {REFERENCE_MARKER_ID: np.eye(4, dtype=np.float64)}
        self.pending = set(SECONDARY_MARKER_IDS)
        self.samples = defaultdict(list)  # id -> list of T_ref_from_marker candidates
        self._last_warning_time = defaultdict(float)  # id -> time of last warning

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print('[PrescanApp] Stream stopped')

    def connect_to_device(self, dev_idx: int = 0):
        print('Searching for devices…')
        devs = Record3DStream.get_connected_devices()
        print(f'{len(devs)} device(s) found')
        for d in devs:
            print(f'  ID: {d.product_id}  UDID: {d.udid}')
        if len(devs) <= dev_idx:
            raise RuntimeError(f'No device at index {dev_idx}')
        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)

    def _intrinsic_matrix(self) -> np.ndarray:
        c = self.session.get_intrinsic_mat()
        return np.array([[c.fx, 0, c.tx],
                         [0, c.fy, c.ty],
                         [0,    0,    1]], dtype=np.float64)

    def _detect_aruco(self, rgb_bgr: np.ndarray,
                      K: np.ndarray, dist: np.ndarray):
        """Return {marker_id: T_cam_marker (OpenCV)} + annotated image."""
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        detections = {}
        if ids is None or len(ids) == 0:
            return detections, rgb_bgr

        cv2.aruco.drawDetectedMarkers(rgb_bgr, corners, ids)

        all_marker_ids = {REFERENCE_MARKER_ID, *SECONDARY_MARKER_IDS}
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in all_marker_ids:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], MARKER_SIZE, K, dist)
            cv2.drawFrameAxes(rgb_bgr, K, dist, rvecs[0], tvecs[0], MARKER_SIZE * 0.5)
            Rm, _ = cv2.Rodrigues(rvecs[0])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rm
            T[:3, 3] = tvecs[0].reshape(3)
            detections[mid] = T

        return detections, rgb_bgr

    def _save_transforms(self):
        """Save both the raw transform matrix and the summary JSON."""
        SCENE_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

        # ── Raw transforms ──────────────────────────────────────────────
        T_file = SCENE_LAYOUT_DIR / 'world_markers_T_ref_from_marker.json'
        out_dict = {
            "reference_marker_id": REFERENCE_MARKER_ID,
            "markers": {}
        }
        for mid in sorted(self.R.keys()):
            if mid == REFERENCE_MARKER_ID:
                continue  # skip the reference (identity) in the output
            T = self.R[mid]
            out_dict["markers"][str(mid)] = T.flatten().tolist()
        with open(str(T_file), 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(f'[Prescan] Saved transforms → {T_file}')

        # ── Summary (human-readable) ─────────────────────────────────
        summary_file = SCENE_LAYOUT_DIR / 'world_markers_summary.json'
        summary_dict = {
            "reference_marker_id": REFERENCE_MARKER_ID,
            "markers": []
        }
        for mid in sorted(self.R.keys()):
            if mid == REFERENCE_MARKER_ID:
                continue
            T = self.R[mid]
            pos = T[:3, 3]
            euler_deg = ScipyR.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
            summary_dict["markers"].append({
                "id": mid,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "rot_x_deg": float(euler_deg[0]),
                "rot_y_deg": float(euler_deg[1]),
                "rot_z_deg": float(euler_deg[2])
            })
        with open(str(summary_file), 'w') as f:
            json.dump(summary_dict, f, indent=2)
        print(f'[Prescan] Saved summary → {summary_file}')

    def _status_line(self):
        """Generate a status line showing registered/pending progress."""
        reg_ids = sorted([mid for mid in self.R.keys() if mid != REFERENCE_MARKER_ID])
        pending_ids = sorted(list(self.pending))

        reg_str = '[' + ', '.join(str(mid) for mid in reg_ids) + ']' if reg_ids else '[]'
        pending_parts = []
        for mid in pending_ids:
            n = len(self.samples[mid])
            pending_parts.append(f'{mid} ({n}/{N_SAMPLES})')
        pending_str = '  '.join(pending_parts) if pending_parts else '(none)'

        return f'Registered: {reg_str}  Pending: {pending_str}'

    def run(self) -> None:
        try:
            self.connect_to_device(0)
        except Exception as e:
            print(f'[Prescan] Connection failed: {e}')
            return

        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        frame_count = 0
        start_time = time.time()

        cv2.namedWindow('World Marker Prescan', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('World Marker Prescan', 960, 540)

        print(f'[Prescan] Starting registration…')
        print(f'  Reference marker: #{REFERENCE_MARKER_ID}')
        print(f'  Secondary markers: {SECONDARY_MARKER_IDS}')
        print(f'  Samples per marker: {N_SAMPLES}')
        print(f'  Press Q to quit at any time\n')

        while True:
            self.event.wait()
            rgb = self.session.get_rgb_frame()
            K = self._intrinsic_matrix()
            self.event.clear()

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                rgb = cv2.flip(rgb, 1)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detections, rgb_bgr = self._detect_aruco(rgb_bgr, K, dist_coeffs)

            # ── Incremental registration: collect samples, lock transforms ────
            now = time.time()
            elapsed = now - start_time

            # For each pending marker, check if it's co-visible with any registered marker
            for new_id in list(self.pending):
                if new_id not in detections:
                    continue

                # Find a co-visible registered marker
                for seen_id in self.R.keys():
                    if seen_id not in detections:
                        continue

                    # Compute candidate offset: T_ref_from_new
                    T_ref_from_new = self.R[seen_id] @ np.linalg.inv(detections[seen_id]) @ detections[new_id]
                    self.samples[new_id].append(T_ref_from_new)
                    break  # one co-visible per pending marker per frame

                # Check if we have enough samples
                if len(self.samples[new_id]) >= N_SAMPLES and new_id in self.pending:
                    self.R[new_id] = _average_se3(self.samples[new_id])
                    self.pending.discard(new_id)
                    print(f'[Prescan] ✓ Marker {new_id} registered ({N_SAMPLES} samples).')

                # Warn if stalled (>30s with no samples)
                if (len(self.samples[new_id]) == 0 and
                        now - self._last_warning_time[new_id] > 30.0):
                    print(f'[Prescan] Marker {new_id} not yet seen alongside a registered marker — walk between them.')
                    self._last_warning_time[new_id] = now

            # ── Key input ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # ── Status overlay ────────────────────────────────────────
            status_line = self._status_line()
            cv2.putText(rgb_bgr, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 80), 2)
            progress_text = f'Elapsed: {elapsed:.1f}s  Frames: {frame_count}  [Press Q to quit]'
            cv2.putText(rgb_bgr, progress_text, (10, rgb_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow('World Marker Prescan', rgb_bgr)
            frame_count += 1

            # Auto-finish when all registered
            if not self.pending:
                print(f'\n[Prescan] All 5 markers registered!')
                break

        cv2.destroyAllWindows()

        # ── Save and report ────────────────────────────────────────
        if len(self.R) > 1:
            self._save_transforms()
            print(f'\n[Prescan] Registration complete:')
            print(f'  Registered markers: {sorted([mid for mid in self.R.keys() if mid != REFERENCE_MARKER_ID])}')
            missing = sorted(list(self.pending))
            if missing:
                print(f'  Missing markers: {missing}')
        else:
            print(f'\n[Prescan] No secondary markers registered.')

        self.session.disconnect()


def main():
    app = WorldMarkerPrescanApp()
    app.run()


if __name__ == '__main__':
    main()
