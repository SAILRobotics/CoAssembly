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
import open3d as o3d
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


def make_axes_lineset(T: np.ndarray, size: float = 0.10) -> o3d.geometry.LineSet:
    """RGB XYZ axes as a LineSet at the given 4×4 pose."""
    o = T[:3, 3]
    pts = np.array([o,
                    o + T[:3, 0] * size,
                    o + T[:3, 1] * size,
                    o + T[:3, 2] * size])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return ls


def make_marker_sphere(T: np.ndarray, radius: float = 0.02,
                       color=(0.8, 0.8, 0.8)) -> o3d.geometry.TriangleMesh:
    """Small sphere at marker position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(T[:3, 3])
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


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
        # samples: id -> list of T_tracking_marker poses (we compute relative transforms at the end)
        self.samples = defaultdict(list)
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

    def _T_tracking_cam(self) -> np.ndarray:
        """Camera pose in iPad's tracking frame (from Record3D ARKit tracking)."""
        p = self.session.get_camera_pose()
        R = quat_to_rotation_matrix(p.qx, p.qy, p.qz, p.qw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [p.tx, p.ty, p.tz]
        return T

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

        # ── Open3D Visualizer ──────────────────────────────────────
        vis = o3d.visualization.Visualizer()
        vis.create_window('Marker Registration (Open3D)', width=800, height=600)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.line_width = 2.0

        # Add world origin coordinate frame
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))

        # Store geometry names for updating
        self._geom_axes = {}
        self._geom_spheres = {}
        self._geom_axes[REFERENCE_MARKER_ID] = make_axes_lineset(np.eye(4), size=0.10)
        self._geom_spheres[REFERENCE_MARKER_ID] = make_marker_sphere(np.eye(4), color=(1.0, 0.5, 0.0))
        vis.add_geometry(self._geom_axes[REFERENCE_MARKER_ID])
        vis.add_geometry(self._geom_spheres[REFERENCE_MARKER_ID])

        print(f'[Prescan] Starting tracking-based registration…')
        print(f'  Method: iPad SLAM tracking (no co-visibility required)')
        print(f'  Reference marker: #{REFERENCE_MARKER_ID}')
        print(f'  Secondary markers: {SECONDARY_MARKER_IDS}')
        print(f'  Samples per marker: {N_SAMPLES}')
        print(f'  Simply walk around and point at each marker — they will auto-register.')
        print(f'  Controls: Q or ENTER to save and quit at any time\n')

        while True:
            self.event.wait()
            rgb = self.session.get_rgb_frame()
            K = self._intrinsic_matrix()
            self.event.clear()

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                rgb = cv2.flip(rgb, 1)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detections, rgb_bgr = self._detect_aruco(rgb_bgr, K, dist_coeffs)

            # ── Tracking-based registration: independent detection via iPad SLAM ────
            # Get camera pose in iPad's tracking frame
            T_tracking_cam = self._T_tracking_cam()

            now = time.time()
            elapsed = now - start_time

            # Process all detected markers (no co-visibility requirement!)
            all_marker_ids = {REFERENCE_MARKER_ID, *SECONDARY_MARKER_IDS}
            for marker_id in all_marker_ids:
                if marker_id not in detections:
                    continue

                # Compute marker pose in tracking frame
                T_cam_marker = detections[marker_id]
                T_tracking_marker = T_tracking_cam @ T_cam_marker
                self.samples[marker_id].append(T_tracking_marker)

            # Check if any pending marker has enough samples
            for new_id in list(self.pending):
                if len(self.samples[new_id]) >= N_SAMPLES:
                    # We need enough ref marker samples to anchor the registration
                    if len(self.samples[REFERENCE_MARKER_ID]) < N_SAMPLES:
                        continue

                    # Average all tracking-frame observations of both markers
                    T_tracking_ref = _average_se3(self.samples[REFERENCE_MARKER_ID])
                    T_tracking_marker = _average_se3(self.samples[new_id])

                    # Compute relative transform: T_ref_from_marker = inv(T_tracking_ref) @ T_tracking_marker
                    self.R[new_id] = np.linalg.inv(T_tracking_ref) @ T_tracking_marker
                    self.pending.discard(new_id)

                    pos = self.R[new_id][:3, 3]
                    dist = float(np.linalg.norm(pos))
                    print(f'[Prescan] ✓ Marker {new_id} registered ({N_SAMPLES} samples, {dist:.3f}m from origin).')

                    # Add to Open3D visualization
                    color = (0.2, 0.8, 1.0) if new_id in SECONDARY_MARKER_IDS else (1.0, 0.5, 0.0)
                    self._geom_axes[new_id] = make_axes_lineset(self.R[new_id], size=0.08)
                    self._geom_spheres[new_id] = make_marker_sphere(self.R[new_id], color=color)
                    vis.add_geometry(self._geom_axes[new_id], reset_bounding_box=False)
                    vis.add_geometry(self._geom_spheres[new_id], reset_bounding_box=False)

                # Warn if stalled (>30s with no samples)
                if (len(self.samples[new_id]) == 0 and
                        now - self._last_warning_time[new_id] > 30.0):
                    print(f'[Prescan] Marker {new_id} not yet detected — point the iPad at it.')
                    self._last_warning_time[new_id] = now

            # ── Key input ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 13:  # Q or ENTER to quit
                break

            # ── Status overlay ────────────────────────────────────────
            status_line = self._status_line()
            cv2.putText(rgb_bgr, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 80), 2)
            progress_text = f'Elapsed: {elapsed:.1f}s  Frames: {frame_count}  [Q or ENTER to save & quit]'
            cv2.putText(rgb_bgr, progress_text, (10, rgb_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow('World Marker Prescan', rgb_bgr)
            frame_count += 1

            # Update Open3D visualization
            vis.poll_events()
            vis.update_renderer()

        cv2.destroyAllWindows()
        vis.destroy_window()

        # ── Save and report ────────────────────────────────────────
        if len(self.R) > 1:
            self._save_transforms()
            print(f'\n[Prescan] ✓ Registration complete (tracking-based):')
            print(f'  Registered markers: {sorted([mid for mid in self.R.keys() if mid != REFERENCE_MARKER_ID])}')
            missing = sorted(list(self.pending))
            if missing:
                print(f'  Missing markers: {missing} — re-run prescan to complete.')
            # Show summary stats
            for mid in sorted(self.R.keys()):
                if mid == REFERENCE_MARKER_ID:
                    continue
                pos = self.R[mid][:3, 3]
                dist = float(np.linalg.norm(pos))
                print(f'    #{mid}: {dist:.3f}m from origin')
        else:
            print(f'\n[Prescan] No markers registered.')

        self.session.disconnect()


def main():
    app = WorldMarkerPrescanApp()
    app.run()


if __name__ == '__main__':
    main()
