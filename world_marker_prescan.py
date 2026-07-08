"""world_marker_prescan.py

Pre-scanning tool for registering 5 ArUco markers relative to each other.

Pipeline
--------
  Point the iPad at each marker and press ENTER to capture it.
  No co-visibility required — the iPad SLAM tracking frame ties everything
  together.  Capture the reference marker (100) at any point; secondary
  markers can be captured independently.

  Press Q to quit and save whatever has been registered so far.

Output
------
  world_markers_T_ref_from_marker.json  – raw 4x4 transform matrices
  world_markers_summary.json            – human-readable distances + rotations
                                          from reference marker (100)
"""

import json
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
SCENE_LAYOUT_DIR     = Path("scene_layout")

# ARKit camera frame (Record3D): X right, Y up, -Z forward
# OpenCV camera frame (aruco):   X right, Y down, +Z forward
# Multiply between them: flip Y and Z
_CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])


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


def make_axes_lineset(T: np.ndarray, size: float = 0.10) -> o3d.geometry.LineSet:
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


def make_marker_square(T: np.ndarray, size: float = 0.10,
                       color=(0.8, 0.8, 0.8)) -> o3d.geometry.LineSet:
    """Square outline of a marker in its local XY plane."""
    h = size / 2
    corners = np.array([[-h, -h, 0, 1],
                        [ h, -h, 0, 1],
                        [ h,  h, 0, 1],
                        [-h,  h, 0, 1]], dtype=np.float64)
    pts = (T @ corners.T).T[:, :3]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    ls.colors = o3d.utility.Vector3dVector([list(color)] * 4)
    return ls


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

        # One-shot captures: id -> T_tracking_marker
        self.captured: dict[int, np.ndarray] = {}

        # Registered transforms relative to reference: id -> T_ref_from_marker
        self.R: dict[int, np.ndarray] = {REFERENCE_MARKER_ID: np.eye(4, dtype=np.float64)}

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
        p = self.session.get_camera_pose()
        R = quat_to_rotation_matrix(p.qx, p.qy, p.qz, p.qw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [p.tx, p.ty, p.tz]
        return T

    def _detect_aruco(self, rgb_bgr: np.ndarray,
                      K: np.ndarray, dist: np.ndarray):
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

    def _try_register(self, vis):
        """Register any secondary marker whose capture is ready alongside the reference."""
        if REFERENCE_MARKER_ID not in self.captured:
            return
        T_tracking_ref = self.captured[REFERENCE_MARKER_ID]
        for mid in SECONDARY_MARKER_IDS:
            if mid in self.R:
                continue  # already registered
            if mid not in self.captured:
                continue
            T_tracking_marker = self.captured[mid]
            self.R[mid] = np.linalg.inv(T_tracking_ref) @ T_tracking_marker
            pos = self.R[mid][:3, 3]
            dist = float(np.linalg.norm(pos))
            print(f'[Prescan] ✓ Marker {mid} registered ({dist:.3f}m from origin).')

            color = (0.2, 0.8, 1.0)
            self._geom_axes[mid] = make_axes_lineset(self.R[mid], size=0.08)
            self._geom_squares[mid] = make_marker_square(self.R[mid], color=color)
            vis.add_geometry(self._geom_axes[mid], reset_bounding_box=False)
            vis.add_geometry(self._geom_squares[mid], reset_bounding_box=False)

    def _save_transforms(self):
        SCENE_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

        T_file = SCENE_LAYOUT_DIR / 'world_markers_T_ref_from_marker.json'
        out_dict = {
            "reference_marker_id": REFERENCE_MARKER_ID,
            "markers": {}
        }
        for mid in sorted(self.R.keys()):
            if mid == REFERENCE_MARKER_ID:
                continue
            out_dict["markers"][str(mid)] = self.R[mid].flatten().tolist()
        with open(str(T_file), 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(f'[Prescan] Saved transforms → {T_file}')

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

    def _status_line(self, visible_ids: set) -> str:
        parts = []
        all_ids = [REFERENCE_MARKER_ID, *SECONDARY_MARKER_IDS]
        for mid in all_ids:
            if mid in self.R and mid != REFERENCE_MARKER_ID:
                tag = 'REG'
            elif mid in self.captured:
                tag = 'CAP'
            elif mid in visible_ids:
                tag = 'VIS'
            else:
                tag = '---'
            parts.append(f'{mid}:{tag}')
        return '  '.join(parts)

    def run(self) -> None:
        try:
            self.connect_to_device(0)
        except Exception as e:
            print(f'[Prescan] Connection failed: {e}')
            return

        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        frame_count = 0

        cv2.namedWindow('World Marker Prescan', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('World Marker Prescan', 960, 540)

        vis = o3d.visualization.Visualizer()
        vis.create_window('Marker Registration (Open3D)', width=800, height=600)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.line_width = 2.0
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))

        self._geom_axes = {}
        self._geom_squares = {}
        self._geom_axes[REFERENCE_MARKER_ID] = make_axes_lineset(np.eye(4), size=0.10)
        self._geom_squares[REFERENCE_MARKER_ID] = make_marker_square(np.eye(4), color=(1.0, 0.5, 0.0))
        vis.add_geometry(self._geom_axes[REFERENCE_MARKER_ID])
        vis.add_geometry(self._geom_squares[REFERENCE_MARKER_ID])

        print(f'[Prescan] Ready.')
        print(f'  Reference marker: #{REFERENCE_MARKER_ID}')
        print(f'  Secondary markers: {SECONDARY_MARKER_IDS}')
        print(f'  Point at a marker and press ENTER to capture it (one-shot).')
        print(f'  No co-visibility required — capture each marker separately.')
        print(f'  Press Q to save and quit.\n')
        print(f'  Status tags:  VIS=visible now  CAP=captured  REG=registered  ---=not seen\n')

        # Current frame detections (updated every frame, read on ENTER)
        current_detections: dict[int, np.ndarray] = {}
        current_T_tracking_cam: np.ndarray = np.eye(4)

        while True:
            self.event.wait()
            rgb = self.session.get_rgb_frame()
            K = self._intrinsic_matrix()
            current_T_tracking_cam = self._T_tracking_cam()
            self.event.clear()

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                rgb = cv2.flip(rgb, 1)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            current_detections, rgb_bgr = self._detect_aruco(rgb_bgr, K, dist_coeffs)

            visible_ids = set(current_detections.keys())

            # ── Key input ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == 13:  # ENTER — one-shot capture of all currently visible markers
                newly_captured = []
                for mid, T_cam_marker in current_detections.items():
                    if mid in self.captured:
                        print(f'[Prescan] Marker {mid} already captured — skipping.')
                        continue
                    self.captured[mid] = current_T_tracking_cam @ _CV_TO_ARKIT @ T_cam_marker
                    newly_captured.append(mid)
                    print(f'[Prescan] Captured marker {mid}.')

                if not newly_captured:
                    if not current_detections:
                        print('[Prescan] No markers visible — point at a marker first.')
                    else:
                        print('[Prescan] All visible markers already captured.')

                self._try_register(vis)

            # ── Status overlay ────────────────────────────────────────
            status = self._status_line(visible_ids)
            cv2.putText(rgb_bgr, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)
            hint = 'ENTER: capture visible markers   Q: save & quit'
            cv2.putText(rgb_bgr, hint, (10, rgb_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow('World Marker Prescan', rgb_bgr)
            frame_count += 1

            vis.poll_events()
            vis.update_renderer()

        cv2.destroyAllWindows()
        vis.destroy_window()

        if len(self.R) > 1:
            self._save_transforms()
            print(f'\n[Prescan] ✓ Done:')
            for mid in sorted(self.R.keys()):
                if mid == REFERENCE_MARKER_ID:
                    continue
                pos = self.R[mid][:3, 3]
                print(f'    #{mid}: {np.linalg.norm(pos):.3f}m from origin')
            missing = [m for m in SECONDARY_MARKER_IDS if m not in self.R]
            if missing:
                print(f'  Not registered: {missing}')
        else:
            print(f'\n[Prescan] No secondary markers registered.')

        self.session.disconnect()


def visualize_from_file(json_path: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    ref_id = data.get("reference_marker_id", REFERENCE_MARKER_ID)
    markers = data.get("markers", {})

    vis = o3d.visualization.Visualizer()
    vis.create_window('Marker Layout (Open3D)', width=900, height=700)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])
    opt.line_width = 2.0

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))

    # Reference marker at origin
    vis.add_geometry(make_axes_lineset(np.eye(4), size=0.10))
    vis.add_geometry(make_marker_square(np.eye(4), color=(1.0, 0.5, 0.0)))
    print(f'  #{ref_id} (reference)  at origin')

    for mid_str, flat16 in markers.items():
        mid = int(mid_str)
        T = np.array(flat16, dtype=np.float64).reshape(4, 4)
        pos = T[:3, 3]
        dist = float(np.linalg.norm(pos))
        print(f'  #{mid}  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  dist={dist:.3f}m')
        vis.add_geometry(make_axes_lineset(T, size=0.08))
        vis.add_geometry(make_marker_square(T, color=(0.2, 0.8, 1.0)))

    print('\nClose the Open3D window to exit.')
    vis.run()
    vis.destroy_window()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='World marker prescan tool')
    parser.add_argument('--visualize', metavar='JSON',
                        nargs='?', const=str(SCENE_LAYOUT_DIR / 'world_markers_T_ref_from_marker.json'),
                        help='Visualize saved marker transforms (default: scene_layout/world_markers_T_ref_from_marker.json)')
    args = parser.parse_args()

    if args.visualize is not None:
        visualize_from_file(Path(args.visualize))
    else:
        app = WorldMarkerPrescanApp()
        app.run()


if __name__ == '__main__':
    main()
