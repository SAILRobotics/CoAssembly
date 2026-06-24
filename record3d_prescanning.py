"""record3d_prescanning.py

Pre-scanning session for CoAssembly.

Pipeline
--------
  Phase 1 – Show marker 10 (9 cm, world frame) live in 3-D.
             Press ENTER to lock it as the world origin.

  Phase 2 – Show marker 101 (10 cm, pegboard) live in 3-D.
             Press ENTER to lock it; saves T_world10_pegboard to scene_layout/.

  Phase 3 – TSDF fusion while the iPad ray hits the pegboard face.
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
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import open3d as o3d
from record3d import Record3DStream

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

        self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.004,
            sdf_trunc=0.02,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.fusion_count = 0

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

    def _integrate_frame(self, rgb: np.ndarray, depth: np.ndarray,
                         K: np.ndarray, T_w10_cam: np.ndarray):
        if depth is None:
            return
        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, K[0,0], K[1,1], K[0,2], K[1,2])
        # Open3D wants world-to-camera in OpenCV convention.
        extrinsic = _ARKIT_TO_CV_4x4 @ np.linalg.inv(T_w10_cam)
        self.tsdf.integrate(rgbd, intrinsic, extrinsic)
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
        tsdf_mesh_geom     = None   # live TSDF mesh shown in 3-D viewer
        last_tsdf_count    = 0
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
                print(f'[Phase 2 ✓] Marker {PEG_MARKER_ID} locked. Scanning active.')

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

                    self._integrate_frame(rgb, depth, K, T_w10_cam)

                    # ── Live TSDF mesh update ─────────────────────────────
                    if (self.fusion_count > 0
                            and self.fusion_count % self.tsdf_interval == 0
                            and self.fusion_count != last_tsdf_count):
                        last_tsdf_count = self.fusion_count
                        print(f'[TSDF] Updating mesh at {self.fusion_count} frames…')
                        self._remove(vis, tsdf_mesh_geom)
                        tsdf_mesh_geom = self.tsdf.extract_triangle_mesh()
                        tsdf_mesh_geom.compute_vertex_normals()
                        vis.add_geometry(tsdf_mesh_geom, reset_bounding_box=False)

            vis.poll_events()
            vis.update_renderer()

        # ── Export on quit ───────────────────────────────────────────────
        if self.fusion_count > 0:
            print(f'[TSDF] {self.fusion_count} frames – extracting final mesh…')
            mesh = self.tsdf.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            SCENE_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = SCENE_LAYOUT_DIR / 'pegboard_scan.ply'
            o3d.io.write_triangle_mesh(str(out_path), mesh)
            print(f'[TSDF] Mesh saved → {out_path}')
            vis.destroy_window()
            cv2.destroyAllWindows()
            o3d.visualization.draw_geometries(
                [mesh], window_name='TSDF Reconstruction')
        else:
            vis.destroy_window()
            cv2.destroyAllWindows()


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
    args = parser.parse_args()

    if args.visualize:
        visualize_scene()
        return

    app = PreScanApp(device_idx=args.device, tsdf_interval=args.tsdf_interval)
    app.connect_to_device(dev_idx=args.device)
    app.start_processing_stream()


if __name__ == '__main__':
    main()
