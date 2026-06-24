"""record3d_prescanning.py

Pre-scanning session for CoAssembly.

Pipeline
--------
  Phase 1 – Show marker 10 (9 cm, world frame) live in 3-D.
             Press ENTER to lock it as the world origin.

  Phase 2 – Show marker 101 (10 cm, pegboard) live in 3-D.
             Press ENTER to lock it; saves T_world10_pegboard to scene_layout/.

  Phase 3 – TSDF fusion of every frame (ray-hit sphere shown when ray hits pegboard).
             Live mesh preview refreshes every --tsdf-interval fused frames.
             Press Q to finish; final mesh saved to scene_layout/pegboard_scan.ply.

Open3D window
    World origin: coordinate-frame axes + grey flat square (always visible).
    Locked marker-10: yellow flat square + coloured axes at identity.
    Live/locked marker-101: yellow flat square + coloured axes at its pose.
    Pegboard: LineSet rectangle (240 × 320 mm).
    Camera frustum (orange), trajectory (blue), ray (green), hit sphere (red).
    Live TSDF mesh (white) updated at each tsdf-interval milestone.

Controls (OpenCV window)
    ENTER  – lock the currently visible target marker
    Q      – quit and export mesh
"""

import argparse
import json
import threading
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from record3d import Record3DStream
from scipy.spatial.transform import Rotation as ScipyR

try:
    import dearpygui.dearpygui as dpg
    _DPG_AVAILABLE = True
except ImportError:
    _DPG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_MARKER_ID   = 10
WORLD_MARKER_SIZE = 0.09    # 9 cm

PEG_MARKER_ID   = 101
PEG_MARKER_SIZE = 0.10     # 10 cm

# Pegboard physical dimensions (metres)
_PEG_W_IN = 24      # 24 inches wide
_PEG_H_IN = 32      # 32 inches tall
_IN_TO_M   = 0.0254
PEG_W = _PEG_W_IN * _IN_TO_M   # 0.6096 m
PEG_H = _PEG_H_IN * _IN_TO_M   # 0.8128 m

# Marker-101 centre: 50 mm (half marker) + 7 mm (white border) = 57 mm from
# the right and top edges of the board.
PEG_OFFSET_X = 0.057
PEG_OFFSET_Y = 0.057

# Board corners in marker-101 local frame (OpenCV: X right, Y down).
#
# Marker 101 is at the TOP-RIGHT corner of the board.
# In practice the marker's local Y+ axis maps to world-UP (ARKit Y+), so:
#   • "above the marker" (top of board) = +Y in local frame
#   • "below the marker" (bottom of board) = -Y in local frame
#
# X (right edge near marker, left edge far away):
PEG_X_MIN = PEG_OFFSET_X - PEG_W    # left edge  (≈ -0.553 m)
PEG_X_MAX = PEG_OFFSET_X            # right edge (≈ +0.057 m, near marker)
# Y (top near marker, bottom far below):
PEG_Y_MIN = PEG_OFFSET_Y - PEG_H    # bottom edge (≈ -0.756 m)
PEG_Y_MAX = PEG_OFFSET_Y            # top edge   (≈ +0.057 m, near marker)

SCENE_LAYOUT_DIR   = Path("scene_layout")
TSDF_MESH_INTERVAL = 30   # refresh live mesh every N fused frames

# ---------------------------------------------------------------------------
# ARKit ↔ OpenCV convention
# ---------------------------------------------------------------------------
# ARKit (Record3D): X right, Y up,   Z backward.
# OpenCV (ArUco):   X right, Y down, Z forward.
# Self-inverse rotation matrix.
_ARKIT_TO_CV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
_ARKIT_TO_CV_4x4 = np.eye(4, dtype=np.float64)
_ARKIT_TO_CV_4x4[:3, :3] = _ARKIT_TO_CV

# ---------------------------------------------------------------------------
# Geometry helpers
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


def make_flat_square(T: np.ndarray, size: float,
                     color=(1.0, 0.85, 0.0)) -> o3d.geometry.TriangleMesh:
    """Flat quad mesh centred at T origin, lying in T's XY plane."""
    s = size / 2
    local = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    world = (T @ np.hstack([local, np.ones((4, 1))]).T).T[:, :3]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(world)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.paint_uniform_color(list(color))
    mesh.compute_vertex_normals()
    return mesh


def make_pegboard_lineset(T_world10_peg: np.ndarray) -> o3d.geometry.LineSet:
    """4-line rectangle outline for the 240 × 320 mm pegboard."""
    local = np.array([
        [PEG_X_MIN, PEG_Y_MAX, 0],   # top-left
        [PEG_X_MAX, PEG_Y_MAX, 0],   # top-right  (≈ marker centre)
        [PEG_X_MAX, PEG_Y_MIN, 0],   # bottom-right
        [PEG_X_MIN, PEG_Y_MIN, 0],   # bottom-left
    ])
    world = (T_world10_peg @ np.hstack([local, np.ones((4, 1))]).T).T[:, :3]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(world)
    ls.lines  = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    ls.colors = o3d.utility.Vector3dVector([[0.3, 0.5, 1.0]] * 4)
    return ls


def make_camera_frustum(T_w_cam, width, height, fx, fy, cx, cy,
                        scale=0.15) -> o3d.geometry.LineSet:
    cam_pos = T_w_cam[:3, 3]
    R_cv = T_w_cam[:3, :3] @ _ARKIT_TO_CV
    us = [0., float(width),  float(width), 0.]
    vs = [0., 0.,            float(height), float(height)]
    corners_cam = np.array([[(u-cx)/fx*scale, (v-cy)/fy*scale, scale]
                             for u, v in zip(us, vs)])
    corners_w = (R_cv @ corners_cam.T).T + cam_pos
    pts = np.vstack([cam_pos.reshape(1, 3), corners_w])
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(pts)
    ls.lines   = o3d.utility.Vector2iVector(lines)
    ls.colors  = o3d.utility.Vector3dVector([[1.0, 0.7, 0.0]] * len(lines))
    return ls


def make_ray_line(T_w_cam, length=1.5) -> o3d.geometry.LineSet:
    origin  = T_w_cam[:3, 3]
    forward = -T_w_cam[:3, 2]   # ARKit camera looks along –Z
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([origin, origin + forward * length])
    ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])
    return ls


def ray_plane_intersect(ray_o, ray_d, plane_pt, plane_n):
    denom = np.dot(ray_d, plane_n)
    if abs(denom) < 1e-8:
        return None, None
    t = np.dot(plane_pt - ray_o, plane_n) / denom
    if t < 0.01:
        return None, None
    return t, ray_o + t * ray_d


def point_in_pegboard(hit_world, T_world10_peg) -> bool:
    local = (np.linalg.inv(T_world10_peg) @ np.append(hit_world, 1.0))[:3]
    return (PEG_X_MIN <= local[0] <= PEG_X_MAX and
            PEG_Y_MIN <= local[1] <= PEG_Y_MAX)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class PreScanApp:
    DEVICE_TYPE__TRUEDEPTH = 0
    DEVICE_TYPE__LIDAR     = 1
    MAX_TRAJ_PTS = 300

    def __init__(self, device_idx: int = 0,
                 tsdf_interval: int = TSDF_MESH_INTERVAL,
                 aruco_dict_id: int = cv2.aruco.DICT_6X6_1000):
        self.device_idx    = device_idx
        self.tsdf_interval = tsdf_interval
        self.event         = Event()
        self.session       = None

        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Locked transforms (written once, never changed)
        self.T_world10_world: np.ndarray | None = None
        self.T_world10_peg:   np.ndarray | None = None

        self.fusion_count  = 0
        self._scan_dir: Path | None = None   # set when scanning starts
        self._saved_poses: list    = []       # list of (4,4) extrinsic arrays
        self._saved_K:    np.ndarray | None = None
        self._saved_hw:   tuple | None      = None

    # ------------------------------------------------------------------
    # Record3D callbacks
    # ------------------------------------------------------------------

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print('Stream stopped')

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
        self.session.on_new_frame      = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _intrinsic_matrix(self) -> np.ndarray:
        c = self.session.get_intrinsic_mat()
        return np.array([[c.fx, 0, c.tx],
                         [0, c.fy, c.ty],
                         [0,    0,    1]], dtype=np.float64)

    def _T_world_cam(self) -> np.ndarray:
        p = self.session.get_camera_pose()
        R = quat_to_rotation_matrix(p.qx, p.qy, p.qz, p.qw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = [p.tx, p.ty, p.tz]
        return T

    def _detect_aruco(self, rgb_bgr: np.ndarray,
                      K: np.ndarray, dist: np.ndarray):
        """Return {marker_id: T_cam_marker (OpenCV)} + annotated image."""
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        detections: dict[int, np.ndarray] = {}
        if ids is None or len(ids) == 0:
            return detections, rgb_bgr

        cv2.aruco.drawDetectedMarkers(rgb_bgr, corners, ids)

        sizes = {WORLD_MARKER_ID: WORLD_MARKER_SIZE,
                 PEG_MARKER_ID:   PEG_MARKER_SIZE}

        for i, mid in enumerate(ids.flatten()):
            mid  = int(mid)
            size = sizes.get(mid, WORLD_MARKER_SIZE)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], size, K, dist)
            cv2.drawFrameAxes(rgb_bgr, K, dist, rvecs[0], tvecs[0], size * 0.5)
            Rm, _ = cv2.Rodrigues(rvecs[0])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rm
            T[:3, 3]  = tvecs[0].reshape(3)
            detections[mid] = T

        return detections, rgb_bgr

    def _save_scene_layout(self):
        SCENE_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
        path = SCENE_LAYOUT_DIR / 'T_world10_pegboard101.npz'
        np.savez(str(path),
                 T_world10_pegboard=self.T_world10_peg,
                 pegboard_width_m=PEG_W,
                 pegboard_height_m=PEG_H,
                 marker_offset_right_m=PEG_OFFSET_X,
                 marker_offset_top_m=PEG_OFFSET_Y)
        print(f'[Scene] Saved → {path}')

    def _save_frame(self, rgb: np.ndarray, depth: np.ndarray,
                    K: np.ndarray, T_w10_cam: np.ndarray):
        if depth is None:
            return
        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        idx = self.fusion_count
        cv2.imwrite(str(self._scan_dir / "color" / f"{idx:06d}.jpg"), rgb[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(str(self._scan_dir / "depth" / f"{idx:06d}.npy"),
                depth.astype(np.float32))
        self._saved_poses.append(_ARKIT_TO_CV_4x4 @ np.linalg.inv(T_w10_cam))
        self._saved_K  = K
        self._saved_hw = (h, w)
        self.fusion_count += 1

    # ------------------------------------------------------------------
    # Visualiser helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove(vis, geom):
        if geom is not None:
            vis.remove_geometry(geom, reset_bounding_box=False)

    @staticmethod
    def _add(vis, geom, reset_bb=False):
        if geom is not None:
            vis.add_geometry(geom, reset_bounding_box=reset_bb)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start_processing_stream(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window('Pre-Scan Viewer  (world = marker 10)',
                          width=1100, height=750)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.10])
        opt.line_width = 2.0

        # ── Static geometry: world origin ───────────────────────────────
        # Large coordinate-frame mesh (RGB arrows) always at origin.
        vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
        # Grey flat square marks the Z=0 plane of the world origin.
        vis.add_geometry(
            make_flat_square(np.eye(4), WORLD_MARKER_SIZE * 1.5,
                             color=(0.55, 0.55, 0.55)))

        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # ── Dynamic geometry handles ────────────────────────────────────
        current_frustum    = None
        current_ray        = None
        current_hit_sphere = None
        live_axes          = None   # live preview of the not-yet-locked marker
        live_surface       = None
        traj_pts: list     = []
        traj_ls            = None
        first_pose_added   = False

        phase = 'await_world'   # → 'await_peg' → 'scanning'

        while True:
            self.event.wait()
            rgb   = self.session.get_rgb_frame()
            depth = self.session.get_depth_frame()
            K     = self._intrinsic_matrix()
            T_wc  = self._T_world_cam()
            self.event.clear()

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                rgb = cv2.flip(rgb, 1)
                if depth is not None:
                    depth = cv2.flip(depth, 1)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h_img, w_img = rgb_bgr.shape[:2]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

            detections, rgb_bgr = self._detect_aruco(rgb_bgr, K, dist_coeffs)

            # ── Key input ───────────────────────────────────────────────
            key           = cv2.waitKey(1) & 0xFF
            enter_pressed = (key == 13)
            if key == ord('q'):
                break

            # ── Phase 1: ENTER locks marker 10 as world origin ──────────
            if (phase == 'await_world' and enter_pressed
                    and WORLD_MARKER_ID in detections):
                T_arkit_wm = T_wc @ _ARKIT_TO_CV_4x4 @ detections[WORLD_MARKER_ID]
                self.T_world10_world = np.linalg.inv(T_arkit_wm)
                phase = 'await_peg'
                print(f'[Phase 1 ✓] Marker {WORLD_MARKER_ID} locked as world origin.')

                # Locked marker-10 is at the world origin (identity).
                # Bright axes + yellow surface at identity.
                self._remove(vis, live_axes)
                self._remove(vis, live_surface)
                live_axes = live_surface = None
                vis.add_geometry(make_axes_lineset(np.eye(4), WORLD_MARKER_SIZE * 0.9))
                vis.add_geometry(make_flat_square(np.eye(4), WORLD_MARKER_SIZE,
                                                  color=(1.0, 0.85, 0.0)))

            # ── Phase 2: ENTER locks marker 101 as pegboard ─────────────
            if (phase == 'await_peg' and enter_pressed
                    and PEG_MARKER_ID in detections):
                T_arkit_peg = T_wc @ _ARKIT_TO_CV_4x4 @ detections[PEG_MARKER_ID]
                self.T_world10_peg = self.T_world10_world @ T_arkit_peg
                phase = 'scanning'
                # Create output directories for this scan session
                import time as _time
                self._scan_dir = (SCENE_LAYOUT_DIR
                                  / f"rgbd_scan_{_time.strftime('%Y%m%d_%H%M%S')}")
                (self._scan_dir / "color").mkdir(parents=True)
                (self._scan_dir / "depth").mkdir(parents=True)
                print(f'[Phase 2 ✓] Marker {PEG_MARKER_ID} locked.'
                      f' Saving frames to {self._scan_dir}')

                # Replace live preview with locked geometry.
                self._remove(vis, live_axes)
                self._remove(vis, live_surface)
                live_axes = live_surface = None

                vis.add_geometry(make_axes_lineset(self.T_world10_peg,
                                                   PEG_MARKER_SIZE * 0.9))
                vis.add_geometry(make_flat_square(self.T_world10_peg,
                                                  PEG_MARKER_SIZE,
                                                  color=(1.0, 0.85, 0.0)))
                vis.add_geometry(make_pegboard_lineset(self.T_world10_peg))

                self._save_scene_layout()

            # ── Camera pose in world-10 frame ───────────────────────────
            T_w10_cam: np.ndarray | None = None
            if self.T_world10_world is not None:
                T_w10_cam = self.T_world10_world @ T_wc

            # ── Live 3-D preview of marker-101 (during await_peg) ───────
            if phase == 'await_peg' and T_w10_cam is not None:
                self._remove(vis, live_axes)
                self._remove(vis, live_surface)
                live_axes = live_surface = None

                if PEG_MARKER_ID in detections:
                    T_live = (self.T_world10_world
                              @ T_wc @ _ARKIT_TO_CV_4x4
                              @ detections[PEG_MARKER_ID])
                    live_axes    = make_axes_lineset(T_live, PEG_MARKER_SIZE * 0.9)
                    live_surface = make_flat_square(T_live, PEG_MARKER_SIZE,
                                                    color=(1.0, 1.0, 0.35))
                    vis.add_geometry(live_axes,    reset_bounding_box=False)
                    vis.add_geometry(live_surface, reset_bounding_box=False)

            # ── Status overlay ───────────────────────────────────────────
            if phase == 'await_world':
                if WORLD_MARKER_ID in detections:
                    status = (f'Marker {WORLD_MARKER_ID} visible'
                              f'  [ENTER to lock world frame]')
                else:
                    status = (f'Phase 1  Aim at marker {WORLD_MARKER_ID}'
                              f' ({WORLD_MARKER_SIZE*100:.0f} cm)')
            elif phase == 'await_peg':
                if PEG_MARKER_ID in detections:
                    status = (f'Marker {PEG_MARKER_ID} visible'
                              f'  [ENTER to lock pegboard]')
                else:
                    status = (f'Phase 2  Aim at marker {PEG_MARKER_ID}'
                              f' ({PEG_MARKER_SIZE*100:.0f} cm)')
            else:
                status = (f'Scanning  {self.fusion_count} frames fused'
                          f'  [Q to finish]')

            cv2.putText(rgb_bgr, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 80), 2)
            cv2.imshow('RGB + ArUco', rgb_bgr)

            if T_w10_cam is None:
                vis.poll_events()
                vis.update_renderer()
                continue

            reset_bb = not first_pose_added
            first_pose_added = True

            # ── Camera frustum ───────────────────────────────────────────
            self._remove(vis, current_frustum)
            current_frustum = make_camera_frustum(
                T_w10_cam, w_img, h_img, fx, fy, cx, cy)
            vis.add_geometry(current_frustum, reset_bounding_box=reset_bb)

            # ── iPad ray ─────────────────────────────────────────────────
            self._remove(vis, current_ray)
            current_ray = make_ray_line(T_w10_cam)
            vis.add_geometry(current_ray, reset_bounding_box=False)

            # ── Camera trajectory ────────────────────────────────────────
            traj_pts.append(T_w10_cam[:3, 3].copy())
            if len(traj_pts) > self.MAX_TRAJ_PTS:
                traj_pts.pop(0)
            if len(traj_pts) >= 2:
                self._remove(vis, traj_ls)
                pts   = np.stack(traj_pts)
                lines = [[i, i+1] for i in range(len(pts)-1)]
                traj_ls = o3d.geometry.LineSet()
                traj_ls.points = o3d.utility.Vector3dVector(pts)
                traj_ls.lines  = o3d.utility.Vector2iVector(lines)
                traj_ls.paint_uniform_color([0.2, 0.6, 1.0])
                vis.add_geometry(traj_ls, reset_bounding_box=False)

            # ── Ray–board intersection + TSDF ────────────────────────────
            if phase == 'scanning' and self.T_world10_peg is not None:
                ray_o    = T_w10_cam[:3, 3]
                ray_d    = -T_w10_cam[:3, 2]          # ARKit: cam looks –Z
                plane_pt = self.T_world10_peg[:3, 3]
                plane_n  = self.T_world10_peg[:3, 2]  # OpenCV Z of marker 101

                _t, hit = ray_plane_intersect(ray_o, ray_d, plane_pt, plane_n)

                self._remove(vis, current_hit_sphere)
                current_hit_sphere = None

                if hit is not None and point_in_pegboard(hit, self.T_world10_peg):
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
                    sphere.translate(hit)
                    sphere.paint_uniform_color([1.0, 0.15, 0.15])
                    sphere.compute_vertex_normals()
                    current_hit_sphere = sphere
                    vis.add_geometry(current_hit_sphere, reset_bounding_box=False)

                    self._save_frame(rgb, depth, K, T_w10_cam)

            vis.poll_events()
            vis.update_renderer()

        # ── Save metadata on quit ────────────────────────────────────────
        vis.destroy_window()
        cv2.destroyAllWindows()
        if self.fusion_count > 0 and self._scan_dir is not None:
            h, w = self._saved_hw
            K = self._saved_K
            import json as _json
            meta = {"width": w, "height": h,
                    "fx": K[0,0], "fy": K[1,1],
                    "cx": K[0,2], "cy": K[1,2]}
            (self._scan_dir / "meta.json").write_text(_json.dumps(meta, indent=2))
            np.save(str(self._scan_dir / "poses.npy"),
                    np.stack(self._saved_poses).astype(np.float64))
            print(f'[Scan] {self.fusion_count} frames saved → {self._scan_dir}')
            print(f'[Scan] Run:  python3 offline_fusion.py {self._scan_dir}')
        else:
            print('[Scan] No frames recorded.')


# ---------------------------------------------------------------------------
# Offline scene viewer
# ---------------------------------------------------------------------------

def visualize_scene():
    """
    Load saved scene_layout/ files and open a static Open3D viewer showing:
      • World origin: coordinate-frame mesh + grey flat square
      • Marker-10 (world origin): coloured axes LineSet + yellow flat square
      • Marker-101 (pegboard origin): coloured axes LineSet + yellow flat square
      • Pegboard: LineSet rectangle (240 × 320 mm)
      • Saved TSDF mesh (scene_layout/pegboard_scan.ply), if present
    """
    layout_path = SCENE_LAYOUT_DIR / 'T_world10_pegboard101.npz'
    mesh_path   = SCENE_LAYOUT_DIR / 'pegboard_scan.ply'

    if not layout_path.exists():
        print(f'[Error] No scene layout found at {layout_path}')
        print('        Run without --visualize first to scan and save the layout.')
        return

    data          = np.load(str(layout_path))
    T_world10_peg = data['T_world10_pegboard']

    geoms = []

    # World origin: large coordinate-frame mesh + grey placeholder square
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    geoms.append(make_flat_square(np.eye(4), WORLD_MARKER_SIZE * 1.5,
                                  color=(0.55, 0.55, 0.55)))

    # Marker-10 locked at world origin: coloured axes + yellow face
    geoms.append(make_axes_lineset(np.eye(4), WORLD_MARKER_SIZE * 0.9))
    geoms.append(make_flat_square(np.eye(4), WORLD_MARKER_SIZE,
                                  color=(1.0, 0.85, 0.0)))

    # Marker-101 (pegboard origin): coloured axes + yellow face
    geoms.append(make_axes_lineset(T_world10_peg, PEG_MARKER_SIZE * 0.9))
    geoms.append(make_flat_square(T_world10_peg, PEG_MARKER_SIZE,
                                  color=(1.0, 0.85, 0.0)))

    # Pegboard rectangle as LineSet
    geoms.append(make_pegboard_lineset(T_world10_peg))

    # TSDF mesh
    if mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        geoms.append(mesh)
        print(f'[Visualize] Loaded mesh from {mesh_path}')
    else:
        print(f'[Visualize] No mesh at {mesh_path} – showing layout only.')

    print(f'[Visualize] Loaded layout from {layout_path}')
    o3d.visualization.draw_geometries(
        geoms,
        window_name='Scene Layout Viewer',
        width=1200, height=800,
    )


# ---------------------------------------------------------------------------
# Annotate mode
# ---------------------------------------------------------------------------

class AnnotateApp:
    """
    GUI tool for placing and editing tool bounding-boxes on the pegboard.

    3-D view  : Open3D GUI window — shows PLY mesh + pegboard outline + boxes.
                Click on the mesh to place a new box (when pick mode is active).
    Control panel : DearPyGUI window — add / remove / edit boxes, save JSON.
    Robot     : connects to the UR10e via RTDE and moves to a viewing pose
                50 cm in front of the pegboard centre (background thread).

    Output saved to scene_layout/tool_layout2.json in the same schema as
    tool_layout.json so it can be dropped in as a replacement.
    """

    _CAL_DIR    = Path("calibration_data/results")
    _LAYOUT_DIR = SCENE_LAYOUT_DIR

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(self, robot_ip: str = "192.168.50.70"):
        self.robot_ip  = robot_ip
        self._ScipyR   = ScipyR
        self._boxes: list[dict] = []
        self._selected_idx = -1
        self._next_id      = 0
        self._pick_mode    = False

        # Scene layout
        npz_path = self._LAYOUT_DIR / "T_world10_pegboard101.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No scene layout at {npz_path}. "
                                    "Run the prescan first.")
        data = np.load(str(npz_path))
        self.T_world_peg = data["T_world10_pegboard"]
        self.peg_width   = float(data["pegboard_width_m"])
        self.peg_height  = float(data["pegboard_height_m"])
        self.offset_x    = float(data["marker_offset_right_m"])
        self.offset_y    = float(data["marker_offset_top_m"])

        # Pegboard plane parameters for ray casting
        self._plane_pt   = self.T_world_peg[:3, 3].copy()
        self._plane_n    = self.T_world_peg[:3, 2].copy()
        self._T_inv_peg  = np.linalg.inv(self.T_world_peg)

        # Robot base pose in world frame (from eye–hand calibration)
        cal_path = self._CAL_DIR / "phase1_results.npz"
        if cal_path.exists():
            cal = np.load(str(cal_path))
            self.T_world_base = cal["T_world_base"]
        else:
            self.T_world_base = np.eye(4)
            print("[Annotate] No calibration found — robot move pose may be wrong.")

        # Pre-load existing tool_layout.json if present
        tl_path = self._LAYOUT_DIR / "tool_layout.json"
        if tl_path.exists():
            raw = json.loads(tl_path.read_text())
            for t in raw.get("tools", []):
                self._boxes.append({
                    "id":           int(t["id"]),
                    "name":         str(t.get("type", "tool")),
                    "category":     str(t.get("category", "tool")),
                    "pegboard_pos": list(t.get("pegboard_pos", [0.0, 0.0])),
                    "size":         list(t.get("size",         [0.10, 0.10, 0.025])),
                    "rotation_deg": list(t.get("rotation_deg", [0.0,  0.0,  0.0])),
                })
                self._next_id = max(self._next_id, int(t["id"]) + 1)
            print(f"[Annotate] Loaded {len(self._boxes)} existing boxes "
                  f"from tool_layout.json")

        # Open3D scene references — set in run()
        self._win          = None
        self._scene_widget = None
        self._box_geom_names: list[str] = []

        # DearPyGUI tag constants
        self._TAG_LIST    = "box_listbox"
        self._TAG_NAME    = "edit_name"
        self._TAG_CAT     = "edit_cat"
        self._TAG_POSX    = "edit_posX"
        self._TAG_POSY    = "edit_posY"
        self._TAG_SIZEW   = "edit_sizeW"
        self._TAG_SIZED   = "edit_sizeD"
        self._TAG_SIZEH   = "edit_sizeH"
        self._TAG_ROTZ    = "edit_rotZ"
        self._TAG_STATUS  = "status_text"

    # ── Robot ─────────────────────────────────────────────────────────────

    def _compute_robot_view_pose(self) -> list[float]:
        """TCP pose [x,y,z,rx,ry,rz] in robot base frame facing pegboard at 50 cm."""
        # Board centre in pegboard-local frame
        cx = (PEG_X_MIN + PEG_X_MAX) / 2.0
        cy = (PEG_Y_MIN + PEG_Y_MAX) / 2.0
        centre_world = (self.T_world_peg @ np.array([cx, cy, 0.0, 1.0]))[:3]

        # Board outward normal
        normal = self._plane_n / (np.linalg.norm(self._plane_n) + 1e-9)

        # TCP is 50 cm in front of the board
        tcp_pos_world = centre_world + 0.5 * normal

        # TCP Z points INTO the board; Y follows pegboard Y axis
        z_tcp = -normal
        y_tmp = self.T_world_peg[:3, 1]
        y_tcp = y_tmp - np.dot(y_tmp, z_tcp) * z_tcp
        if np.linalg.norm(y_tcp) < 1e-6:
            y_tcp = np.array([0.0, 0.0, 1.0])
        y_tcp /= np.linalg.norm(y_tcp)
        x_tcp = np.cross(y_tcp, z_tcp)
        x_tcp /= np.linalg.norm(x_tcp)
        y_tcp  = np.cross(z_tcp, x_tcp)

        T_tcp_world = np.eye(4)
        T_tcp_world[:3, :3] = np.column_stack([x_tcp, y_tcp, z_tcp])
        T_tcp_world[:3, 3]  = tcp_pos_world

        T_base_tcp = np.linalg.inv(self.T_world_base) @ T_tcp_world
        pos     = T_base_tcp[:3, 3].tolist()
        rot_vec = self._ScipyR.from_matrix(T_base_tcp[:3, :3]).as_rotvec().tolist()
        return pos + rot_vec

    def _move_robot_to_view(self):
        """Send moveL to the viewing pose in a background thread."""
        def _worker():
            try:
                from rtde_control import RTDEControlInterface
                pose = self._compute_robot_view_pose()
                print(f"[Robot] Connecting to {self.robot_ip}…")
                ctrl = RTDEControlInterface(self.robot_ip)
                print(f"[Robot] Moving to view pose: "
                      f"{[round(v, 3) for v in pose]}")
                ctrl.moveL(pose, speed=0.08, acceleration=0.2)
                ctrl.disconnect()
                print("[Robot] Reached view pose.")
            except Exception as exc:
                print(f"[Robot] Move failed: {exc}")
        threading.Thread(target=_worker, daemon=True).start()

    # ── Geometry helpers ──────────────────────────────────────────────────

    def _box_world_corners(self, box: dict) -> np.ndarray:
        """8 world-space corners of a tool box."""
        px, py = box["pegboard_pos"]
        w, d, h = (box["size"][0]/2, box["size"][1]/2, box["size"][2])
        local = np.array([
            [-w, -d, 0], [w, -d, 0], [w, d, 0], [-w, d, 0],
            [-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h],
        ])
        R_local = self._ScipyR.from_euler(
            'xyz', box["rotation_deg"], degrees=True).as_matrix()
        peg_pts = (R_local @ local.T).T + np.array([px, py, 0.0])
        peg_h   = np.hstack([peg_pts, np.ones((8, 1))])
        return (self.T_world_peg @ peg_h.T).T[:, :3]

    def _make_box_lineset(self, box: dict, selected: bool) -> o3d.geometry.LineSet:
        corners = self._box_world_corners(box)
        edges   = [[0,1],[1,2],[2,3],[3,0],
                   [4,5],[5,6],[6,7],[7,4],
                   [0,4],[1,5],[2,6],[3,7]]
        color   = [1.0, 1.0, 0.0] if selected else [0.2, 0.9, 1.0]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines  = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector([color] * 12)
        return ls

    def _refresh_boxes(self, selected_only: bool = False):
        """Post a scene refresh onto the Open3D main-thread queue.
        Safe to call from any callback (Open3D or DearPyGUI)."""
        if self._win is None:
            return
        fn = self._do_refresh_selected_box if selected_only else self._do_refresh_boxes
        gui.Application.instance.post_to_main_thread(self._win, fn)

    def _do_refresh_boxes(self):
        """Rebuild all box wireframes in the scene (called from main loop)."""
        scene = self._scene_widget.scene
        for name in self._box_geom_names:
            scene.remove_geometry(name)
        self._box_geom_names.clear()

        mat = rendering.MaterialRecord()
        mat.shader     = "unlitLine"
        mat.line_width = 2.5

        for i, box in enumerate(self._boxes):
            ls   = self._make_box_lineset(box, selected=(i == self._selected_idx))
            name = f"__box_{i}"
            scene.add_geometry(name, ls, mat)
            self._box_geom_names.append(name)
        self._scene_widget.force_redraw()

    def _do_refresh_selected_box(self):
        """Update only the selected box wireframe (faster, for slider drags)."""
        if not (0 <= self._selected_idx < len(self._boxes)):
            return
        scene = self._scene_widget.scene
        name  = f"__box_{self._selected_idx}"
        if name in self._box_geom_names:
            scene.remove_geometry(name)
            self._box_geom_names.remove(name)

        mat = rendering.MaterialRecord()
        mat.shader     = "unlitLine"
        mat.line_width = 2.5
        ls = self._make_box_lineset(self._boxes[self._selected_idx], selected=True)
        scene.add_geometry(name, ls, mat)
        self._box_geom_names.append(name)
        self._scene_widget.force_redraw()

    # ── DearPyGUI sync helpers ────────────────────────────────────────────

    def _sync_list(self):
        items = [f"[{b['id']}]  {b['name']}  ({b['category']})"
                 for b in self._boxes]
        dpg.configure_item(self._TAG_LIST, items=items)
        if 0 <= self._selected_idx < len(items):
            dpg.set_value(self._TAG_LIST, items[self._selected_idx])

    def _sync_edit(self):
        if not (0 <= self._selected_idx < len(self._boxes)):
            return
        box = self._boxes[self._selected_idx]
        dpg.set_value(self._TAG_NAME,  box["name"])
        dpg.set_value(self._TAG_CAT,   box["category"])
        dpg.set_value(self._TAG_POSX,  float(box["pegboard_pos"][0]))
        dpg.set_value(self._TAG_POSY,  float(box["pegboard_pos"][1]))
        dpg.set_value(self._TAG_SIZEW, float(box["size"][0]))
        dpg.set_value(self._TAG_SIZED, float(box["size"][1]))
        dpg.set_value(self._TAG_SIZEH, float(box["size"][2]))
        dpg.set_value(self._TAG_ROTZ,  float(box["rotation_deg"][2]))

    def _read_edit_into_box(self):
        """Flush DearPyGUI field values into the selected box dict."""
        if not (0 <= self._selected_idx < len(self._boxes)):
            return
        box = self._boxes[self._selected_idx]
        box["name"]         = dpg.get_value(self._TAG_NAME)
        box["category"]     = dpg.get_value(self._TAG_CAT)
        box["pegboard_pos"] = [round(dpg.get_value(self._TAG_POSX), 4),
                               round(dpg.get_value(self._TAG_POSY), 4)]
        box["size"]         = [round(dpg.get_value(self._TAG_SIZEW), 4),
                               round(dpg.get_value(self._TAG_SIZED), 4),
                               round(dpg.get_value(self._TAG_SIZEH), 4)]
        box["rotation_deg"] = [0.0, 0.0,
                               round(dpg.get_value(self._TAG_ROTZ), 2)]

    # ── Save ──────────────────────────────────────────────────────────────

    def _save(self):
        payload = {"tools": [
            {
                "id":           b["id"],
                "type":         b["name"],
                "category":     b["category"],
                "pegboard_pos": b["pegboard_pos"],
                "size":         b["size"],
                "rotation_deg": b["rotation_deg"],
            }
            for b in self._boxes
        ]}
        out = self._LAYOUT_DIR / "tool_layout2.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"[Annotate] Saved {len(self._boxes)} boxes → {out}")

    # ── Main run ──────────────────────────────────────────────────────────

    def run(self):

        # ── Open3D GUI window ──────────────────────────────────────────
        app = gui.Application.instance
        app.initialize()

        o3d_win = app.create_window(
            "Pegboard Annotator — 3D View", width=1100, height=800)
        self._win = o3d_win

        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(o3d_win.renderer)
        o3d_win.add_child(scene_widget)
        self._scene_widget = scene_widget

        def _on_layout(ctx):
            scene_widget.frame = o3d_win.content_rect
        o3d_win.set_on_layout(_on_layout)

        scene = scene_widget.scene
        scene.set_background([0.06, 0.06, 0.10, 1.0])

        mat_unlit = rendering.MaterialRecord()
        mat_unlit.shader = "defaultUnlit"
        mat_line = rendering.MaterialRecord()
        mat_line.shader     = "unlitLine"
        mat_line.line_width = 2.0

        # World origin
        scene.add_geometry("world_frame",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15),
            mat_unlit)
        # Pegboard marker axes
        scene.add_geometry("peg_axes",
            make_axes_lineset(self.T_world_peg, PEG_MARKER_SIZE * 0.9),
            mat_line)
        # Pegboard outline rectangle
        scene.add_geometry("peg_outline",
            make_pegboard_lineset(self.T_world_peg),
            mat_line)
        # PLY scan mesh
        mesh_path = self._LAYOUT_DIR / "pegboard_scan.ply"
        if mesh_path.exists():
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            mesh.compute_vertex_normals()
            mat_mesh = rendering.MaterialRecord()
            mat_mesh.shader = "defaultLit"
            scene.add_geometry("ply_mesh", mesh, mat_mesh)
            print(f"[Annotate] Loaded mesh from {mesh_path}")
        else:
            print(f"[Annotate] No PLY mesh found at {mesh_path}")

        # Initial box wireframes
        self._refresh_boxes()

        # Camera: look at pegboard centre
        cx = (PEG_X_MIN + PEG_X_MAX) / 2.0
        cy = (PEG_Y_MIN + PEG_Y_MAX) / 2.0
        look_at = (self.T_world_peg @ np.array([cx, cy, 0.0, 1.0]))[:3]
        eye     = look_at + self._plane_n * 1.2
        up      = self.T_world_peg[:3, 1]
        scene_widget.look_at(look_at, eye, up)

        # Mouse callback — active only in pick mode.
        # IMPORTANT: must not call scene.add/remove_geometry here (segfault).
        # Instead mutate self._boxes and set _pending_refresh; the main loop
        # drains the flag before the next Open3D tick.
        def _on_mouse(event):
            if not self._pick_mode:
                return gui.SceneWidget.EventCallbackResult.IGNORED
            if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                    and event.is_button_down(gui.MouseButton.LEFT)):
                x = float(event.x - scene_widget.frame.x)
                y = float(event.y - scene_widget.frame.y)
                w = float(scene_widget.frame.width)
                h = float(scene_widget.frame.height)
                # Reject clicks outside the widget
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x >= w or y >= h:
                    return gui.SceneWidget.EventCallbackResult.HANDLED
                cam = scene_widget.scene.camera
                # Camera position from inverse view matrix (avoids infinite far plane)
                view_m  = np.array(cam.get_view_matrix(), dtype=float)
                cam_pos = np.linalg.inv(view_m)[:3, 3]
                near_pt = np.array(cam.unproject(x, y, 0.0, w, h), dtype=float).flatten()
                if not np.all(np.isfinite(cam_pos)) or not np.all(np.isfinite(near_pt)):
                    return gui.SceneWidget.EventCallbackResult.HANDLED
                ray_d   = near_pt - cam_pos
                rd_norm = np.linalg.norm(ray_d)
                if not np.isfinite(rd_norm) or rd_norm < 1e-9:
                    return gui.SceneWidget.EventCallbackResult.HANDLED
                ray_d  /= rd_norm

                t, hit = ray_plane_intersect(cam_pos, ray_d,
                                             self._plane_pt, self._plane_n)
                if hit is not None and not np.any(np.isnan(hit)):
                    local = (self._T_inv_peg @ np.append(hit, 1.0))[:3]
                    if np.any(np.isnan(local)):
                        return gui.SceneWidget.EventCallbackResult.HANDLED

                    # Show a small sphere at the hit point, removed on next tick
                    hit_pt = hit.copy()
                    def _place_sphere():
                        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
                        s.translate(hit_pt)
                        s.paint_uniform_color([1.0, 0.6, 0.1])
                        s.compute_vertex_normals()
                        mat_s = rendering.MaterialRecord()
                        mat_s.shader = "defaultLit"
                        scene_widget.scene.remove_geometry("__hit_sphere")
                        scene_widget.scene.add_geometry("__hit_sphere", s, mat_s)
                        scene_widget.force_redraw()
                    gui.Application.instance.post_to_main_thread(o3d_win, _place_sphere)

                    # Move selected box, or create a new one if nothing is selected
                    print(f"[pick] selected_idx={self._selected_idx}  n_boxes={len(self._boxes)}")
                    if 0 <= self._selected_idx < len(self._boxes):
                        self._boxes[self._selected_idx]["pegboard_pos"] = [
                            round(float(local[0]), 4),
                            round(float(local[1]), 4),
                        ]
                    else:
                        self._boxes.append({
                            "id":           self._next_id,
                            "name":         "new_tool",
                            "category":     "tool",
                            "pegboard_pos": [round(float(local[0]), 4),
                                             round(float(local[1]), 4)],
                            "size":         [0.10, 0.10, 0.025],
                            "rotation_deg": [0.0, 0.0, 0.0],
                        })
                        self._next_id     += 1
                        self._selected_idx = len(self._boxes) - 1
                    self._pick_mode = False
                    self._refresh_boxes()
                    self._sync_list()
                    self._sync_edit()
                    dpg.set_value(self._TAG_STATUS,
                                  f"Moved to peg ({local[0]:.3f}, {local[1]:.3f})")
                return gui.SceneWidget.EventCallbackResult.HANDLED
            return gui.SceneWidget.EventCallbackResult.IGNORED

        scene_widget.set_on_mouse(_on_mouse)

        # ── DearPyGUI panel ────────────────────────────────────────────
        dpg.create_context()
        dpg.create_viewport(title="Annotator Controls",
                            width=430, height=800, x_pos=1110, y_pos=0)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        with dpg.window(label="Tool Annotator", tag="main_win",
                        no_close=True, no_move=True, no_resize=True,
                        width=425, height=800, pos=[0, 0]):

            # ── Robot ──────────────────────────────────────────────────
            dpg.add_button(label="Move Robot to Viewing Pose (50 cm from board)",
                           callback=lambda: self._move_robot_to_view(), width=-1)
            dpg.add_separator()

            # ── Box list ───────────────────────────────────────────────
            dpg.add_text("Boxes")
            dpg.add_listbox(items=[], tag=self._TAG_LIST,
                            num_items=7, width=-1,
                            callback=lambda s, a: self._on_list_select(a))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Add",   width=70,
                               callback=lambda: self._on_add())
                dpg.add_button(label="Place on Pointcloud", width=175,
                               callback=lambda: self._on_place_mode())
                dpg.add_button(label="Remove", width=-1,
                               callback=lambda: self._on_remove())

            dpg.add_separator()

            # ── Edit panel ─────────────────────────────────────────────
            dpg.add_text("Edit Selected Box")

            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit,
                           borders_innerV=False, pad_outerX=True):
                dpg.add_table_column(init_width_or_weight=90)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)

                def _row(label, widget_fn):
                    with dpg.table_row():
                        dpg.add_text(label)
                        widget_fn()

                def _live():
                    self._read_edit_into_box()
                    self._refresh_boxes(selected_only=True)

                def _live_meta():
                    self._read_edit_into_box()
                    self._sync_list()

                _row("Name",      lambda: dpg.add_input_text(
                    tag=self._TAG_NAME, default_value="", width=-1,
                    callback=lambda: _live_meta(), on_enter=False))
                _row("Category",  lambda: dpg.add_combo(
                    ["tool", "part"], tag=self._TAG_CAT,
                    default_value="tool", width=-1,
                    callback=lambda: _live_meta()))
                _row("Pos X (m)", lambda: dpg.add_slider_float(
                    tag=self._TAG_POSX, default_value=0.0, width=-1,
                    min_value=PEG_X_MIN, max_value=PEG_X_MAX,
                    format="%.4f", callback=lambda: _live()))
                _row("Pos Y (m)", lambda: dpg.add_slider_float(
                    tag=self._TAG_POSY, default_value=0.0, width=-1,
                    min_value=PEG_Y_MIN, max_value=PEG_Y_MAX,
                    format="%.4f", callback=lambda: _live()))
                _row("Width (m)", lambda: dpg.add_slider_float(
                    tag=self._TAG_SIZEW, default_value=0.10, width=-1,
                    min_value=0.01, max_value=0.60,
                    format="%.3f", callback=lambda: _live()))
                _row("Depth (m)", lambda: dpg.add_slider_float(
                    tag=self._TAG_SIZED, default_value=0.10, width=-1,
                    min_value=0.01, max_value=0.60,
                    format="%.3f", callback=lambda: _live()))
                _row("Height(m)", lambda: dpg.add_slider_float(
                    tag=self._TAG_SIZEH, default_value=0.025, width=-1,
                    min_value=0.005, max_value=0.40,
                    format="%.3f", callback=lambda: _live()))
                _row("Rot Z (°)", lambda: dpg.add_slider_float(
                    tag=self._TAG_ROTZ, default_value=0.0, width=-1,
                    min_value=-180.0, max_value=180.0,
                    format="%.1f", callback=lambda: _live()))

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply",     width=-110,
                               callback=lambda: self._on_apply())
                dpg.add_button(label="Save JSON", width=-1,
                               callback=lambda: self._on_save())

            dpg.add_separator()
            dpg.add_text("", tag=self._TAG_STATUS, wrap=420)

        dpg.set_primary_window("main_win", True)

        # Initial population
        self._sync_list()
        if self._boxes:
            self._selected_idx = 0
            self._sync_edit()
        self._do_refresh_boxes()  # direct call is safe here (not inside a callback)

        # ── Main loop ──────────────────────────────────────────────────
        # All scene updates go through _refresh_boxes() → post_to_main_thread,
        # so they execute inside run_one_tick() at a renderer-safe point.
        while True:
            if not dpg.is_dearpygui_running():
                break
            dpg.render_dearpygui_frame()
            if not app.run_one_tick():
                break

        dpg.destroy_context()

    # ── DearPyGUI callbacks ───────────────────────────────────────────────

    def _on_list_select(self, value: str):
        items = dpg.get_item_configuration(self._TAG_LIST)["items"]
        try:
            self._selected_idx = items.index(value)
        except ValueError:
            return
        self._refresh_boxes()
        self._sync_edit()

    def _on_add(self):
        self._boxes.append({
            "id":           self._next_id,
            "name":         "new_tool",
            "category":     "tool",
            "pegboard_pos": [round((PEG_X_MIN + PEG_X_MAX) / 2, 3),
                             round((PEG_Y_MIN + PEG_Y_MAX) / 2, 3)],
            "size":         [0.10, 0.10, 0.025],
            "rotation_deg": [0.0, 0.0, 0.0],
        })
        self._next_id     += 1
        self._selected_idx = len(self._boxes) - 1
        self._refresh_boxes()
        self._sync_list()
        self._sync_edit()
        dpg.set_value(self._TAG_STATUS, "New box added at board centre.")

    def _on_place_mode(self):
        self._pick_mode = True
        dpg.set_value(self._TAG_STATUS,
                      "PICK MODE — click on the pegboard to move the selected box there.")

    def _on_remove(self):
        if not (0 <= self._selected_idx < len(self._boxes)):
            return
        removed = self._boxes.pop(self._selected_idx)
        self._selected_idx = max(0, self._selected_idx - 1) \
                             if self._boxes else -1
        self._refresh_boxes()
        self._sync_list()
        self._sync_edit()
        dpg.set_value(self._TAG_STATUS,
                      f"Removed box [{removed['id']}] '{removed['name']}'.")

    def _on_apply(self):
        self._read_edit_into_box()
        self._refresh_boxes()
        self._sync_list()
        dpg.set_value(self._TAG_STATUS, "Applied — box updated in 3D view.")

    def _on_save(self):
        self._read_edit_into_box()
        self._save()
        dpg.set_value(self._TAG_STATUS,
                      f"Saved {len(self._boxes)} boxes to tool_layout2.json")


# ---------------------------------------------------------------------------

def fuse_offline(scan_dir: Path, voxel_length: float = 0.004,
                 depth_trunc: float = 3.0):
    """Offline TSDF fusion from a directory saved by PreScanApp."""
    meta = json.loads((scan_dir / "meta.json").read_text())
    poses = np.load(str(scan_dir / "poses.npy"))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=meta["width"], height=meta["height"],
        fx=meta["fx"], fy=meta["fy"],
        cx=meta["cx"], cy=meta["cy"])

    color_files = sorted((scan_dir / "color").glob("*.jpg"))
    depth_files = sorted((scan_dir / "depth").glob("*.npy"))

    if len(color_files) != len(depth_files) or len(color_files) != len(poses):
        raise ValueError(
            f"Frame count mismatch: {len(color_files)} color, "
            f"{len(depth_files)} depth, {len(poses)} poses")

    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    n = len(color_files)
    print(f"[fuse] Integrating {n} frames  (voxel={voxel_length*100:.1f} cm)…")
    for i, (c_file, d_file, extrinsic) in enumerate(
            zip(color_files, depth_files, poses)):
        color     = o3d.io.read_image(str(c_file))
        depth_arr = np.load(str(d_file)).astype(np.float32)
        depth_img = o3d.geometry.Image(depth_arr)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_img,
            depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        tsdf.integrate(rgbd, intrinsic, extrinsic)
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  {i+1}/{n}")

    print("[fuse] Extracting mesh…")
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out = SCENE_LAYOUT_DIR / "pegboard_scan.ply"
    SCENE_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[fuse] Saved → {out}")
    o3d.visualization.draw_geometries([mesh], window_name="TSDF Reconstruction")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Record3D pre-scan: lock world + pegboard via ENTER, fuse TSDF'
    )
    parser.add_argument('--device',        type=int, default=0,
                        help='Record3D device index (default: 0)')
    parser.add_argument('--tsdf-interval', type=int, default=TSDF_MESH_INTERVAL,
                        help='Refresh live TSDF mesh every N fused frames '
                             f'(default: {TSDF_MESH_INTERVAL})')
    parser.add_argument('--visualize',     action='store_true',
                        help='Open a static viewer for the saved scene_layout/ '
                             'files (no device needed)')
    parser.add_argument('--annotate',      action='store_true',
                        help='Open the tool-box annotator GUI (no device needed)')
    parser.add_argument('--robot-ip',      default='192.168.50.70',
                        help='UR10e IP for the annotate robot-move button '
                             '(default: 192.168.50.70)')
    parser.add_argument('--fuse',          type=Path, metavar='SCAN_DIR',
                        help='Offline TSDF fusion of a saved scan directory')
    parser.add_argument('--voxel',         type=float, default=0.004,
                        help='Voxel size in metres for --fuse (default: 0.004)')
    parser.add_argument('--depth-trunc',   type=float, default=3.0,
                        help='Depth truncation in metres for --fuse (default: 3.0)')
    args = parser.parse_args()

    if args.visualize:
        visualize_scene()
        return

    if args.annotate:
        AnnotateApp(robot_ip=args.robot_ip).run()
        return

    if args.fuse:
        fuse_offline(args.fuse, voxel_length=args.voxel, depth_trunc=args.depth_trunc)
        return

    app = PreScanApp(device_idx=args.device, tsdf_interval=args.tsdf_interval)
    app.connect_to_device(dev_idx=args.device)
    app.start_processing_stream()


if __name__ == '__main__':
    main()
