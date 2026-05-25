"""
main_hand.py — Quest passthrough ArUco → WorldRoot + Hand Tracking Visualizer

All functionalities of main.py, but visualizes hand joints in the Open3D window
instead of controller poses. Runs fully standalone — does not require main.py.

Transform chain (same as main.py)
----------------------------------
  T_cam_marker      = ArUco detection
  T_tracking_marker = cam.camera_T @ T_cam_marker
  T_world_tracking  = inv(T_tracking_marker)
  joints_world      = T_world_tracking @ joints_unity (with Y↔Z swap)

Keys (OpenCV window must be focused)
--------------------------------------
  ENTER  = force re-lock anchor from current detection
  ESC    = quit

Usage
-----
  python main_hand.py
  python main_hand.py --quest-ip 192.168.50.201 --hand-port 5570
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import open3d as o3d
import zmq
from scipy.spatial.transform import Rotation as ScipyR

_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from utils.unity_conversion import (
    unity_to_open3d_vector,
    unity_to_open3d_quaternion,
    open3d_to_unity_vector,
    open3d_to_unity_quaternion,
    HAND_BONES,
)
import ip_setting as cfg


# =============================================================================
# Pose helpers  (same as main.py)
# =============================================================================

def _unity_pose_to_T(pos_xyz, rot_xyzw) -> np.ndarray:
    pos_dict = {"x": float(pos_xyz[0]), "y": float(pos_xyz[1]), "z": float(pos_xyz[2])}
    p = unity_to_open3d_vector(pos_dict)
    x, y, z, w = rot_xyzw
    q_o3d = unity_to_open3d_quaternion([float(w), float(x), float(y), float(z)])
    R_cam = ScipyR.from_quat([q_o3d[1], q_o3d[2], q_o3d[3], q_o3d[0]]).as_matrix()
    R_fix = ScipyR.from_euler('x', -90.0, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam @ R_fix
    T[:3, 3]  = p
    return T


def _adapt_cx_cy(fx, fy, cx, cy, sensor_w, sensor_h, img_w, img_h):
    if sensor_w is None or sensor_h is None:
        return fx, fy, cx, cy
    crop_x = (float(sensor_w) - float(img_w)) / 2.0
    crop_y = (float(sensor_h) - float(img_h)) / 2.0
    return fx, fy, cx - crop_x, cy - crop_y


def _transform_point(T: np.ndarray, p_local) -> np.ndarray:
    """4x4 transform applied to a 3D point."""
    p = np.array([p_local[0], p_local[1], p_local[2], 1.0], dtype=np.float64)
    return (T @ p)[:3]


# =============================================================================
# Hand joint helpers
# =============================================================================

_BONES_NP  = np.array(HAND_BONES, dtype=np.int32)
_N_JOINTS  = int(_BONES_NP.max()) + 1
_HIDDEN_PT = np.array([[0., -100., 0.]])

# Order must match HAND_BONES layout: Wrist Palm Thumb Index Middle Ring Pinky
_JOINT_GROUP_ORDER = ["Wrist", "Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"]


def _unity_to_o3d(pts_unity: np.ndarray) -> np.ndarray:
    """(N,3) Unity tracking frame → Open3D frame (swap Y↔Z)."""
    return pts_unity[:, [0, 2, 1]]


def _to_world(joints_unity: np.ndarray, T_world_tracking: np.ndarray) -> np.ndarray:
    pts = _unity_to_o3d(joints_unity)
    R, t = T_world_tracking[:3, :3], T_world_tracking[:3, 3]
    return (pts @ R.T) + t


def _extract_joints(hand_block) -> np.ndarray | None:
    """
    Parse a hand block from TrackingDataManager (SerializableTrackingData format).
    Returns (N,3) float64 in HAND_BONES order, or None.
    """
    if hand_block is None:
        return None
    groups = hand_block.get("groups")
    if not groups:
        return None
    joints = []
    for group_name in _JOINT_GROUP_ORDER:
        for pose in groups.get(group_name) or []:
            if pose is None:
                joints.append([0.0, 0.0, 0.0])
            else:
                pos = pose.get("position") or {}
                joints.append([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)])
    return np.array(joints, dtype=np.float64) if joints else None


# =============================================================================
# ZMQ receivers  (same as main.py)
# =============================================================================

# class _XRStateReceiver:
#     def __init__(self, ip: str, port: int = 5559, topic: str = "xr"):
#         ctx = zmq.Context()
#         self._sub = ctx.socket(zmq.SUB)
#         self._sub.connect(f"tcp://{ip}:{port}")
#         self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
#         self.center_T: np.ndarray | None = None

#     def poll(self, timeout_ms: int = 0) -> bool:
#         poller = zmq.Poller()
#         poller.register(self._sub, zmq.POLLIN)
#         if not dict(poller.poll(timeout=timeout_ms)):
#             return False
#         latest = None
#         while True:
#             try:
#                 _      = self._sub.recv_string(flags=zmq.NOBLOCK)
#                 latest = self._sub.recv_string(flags=zmq.NOBLOCK)
#             except zmq.Again:
#                 break
#         if latest is None:
#             return False
#         msg  = json.loads(latest)
#         head = msg.get("head")
#         if head and head.get("is_valid"):
#             p, r = head.get("pos"), head.get("rot_xyzw")
#             if p and r:
#                 self.center_T = _unity_pose_to_T(p, r)
#         return True

#     def close(self):
#         try:
#             self._sub.close(0)
#         except Exception:
#             pass


class _CamFeedReceiver:
    def __init__(self, ip: str, port: int = 5560, topic: str = "cam_left"):
        ctx = zmq.Context()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{ip}:{port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.frame        = None
        self.camera_T     = None
        self.fx = self.fy = self.cx = self.cy = None
        self.sensor_width = self.sensor_height = None
        self.width        = self.height        = None

    def poll(self, timeout_ms: int = 0) -> bool:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False
        latest = None
        while True:
            try:
                parts  = self._sub.recv_multipart(flags=zmq.NOBLOCK)
                latest = parts
            except zmq.Again:
                break
        if latest is None or len(latest) != 9:
            return False
        width,  = struct.unpack("<i",    latest[2])
        height, = struct.unpack("<i",    latest[3])
        px, py, pz      = struct.unpack("<fff",  latest[4])
        qx, qy, qz, qw = struct.unpack("<ffff", latest[5])
        fx, fy, cx, cy  = struct.unpack("<ffff", latest[6])
        sw, sh           = struct.unpack("<ii",   latest[7])
        arr   = np.frombuffer(latest[8], dtype=np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            return False
        self.frame = frame
        self.width, self.height = width, height
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.sensor_width  = int(sw)
        self.sensor_height = int(sh)
        self.camera_T = _unity_pose_to_T([px, py, pz], [qx, qy, qz, qw])
        return True

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass


class _HandDataReceiver:
    """SUBs to Unity hand tracking stream (TrackingDataManager format)."""

    def __init__(self, unity_ip: str, port: int, verbose: bool = True):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{unity_ip}:{port}")
        self.data = None
        self.message_count = 0
        self.last_rx_time = None
        self.last_error = None
        if verbose:
            print(f"[HandDataReceiver] SUB → tcp://{unity_ip}:{port}")

    def poll(self, timeout_ms: int = 0):
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False

        latest = None
        while True:
            try:
                latest = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                self.last_error = str(e)
                print(f"[HandDataReceiver] recv error: {e}")
                return False

        if latest is None:
            return False

        try:
            self.data = json.loads(latest)
            self.message_count += 1
            self.last_rx_time = time.time()
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            print(f"[HandDataReceiver] parse error: {e}")
            return False

        return True

    @property
    def receiving(self) -> bool:
        return self.last_rx_time is not None and (time.time() - self.last_rx_time) < 2.0

    def center_eye_T(self) -> "np.ndarray | None":
        """Extract CenterEye pose from the latest hand-tracking frame."""
        if self.data is None:
            return None
        head = self.data.get("head") or {}
        ce = head.get("CenterEye")
        if not ce:
            return None
        pos = ce.get("position") or {}
        rot = ce.get("rotation") or {}
        pos_xyz  = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
        rot_xyzw = [rot.get("x", 0.0), rot.get("y", 0.0),
                    rot.get("z", 0.0), rot.get("w", 1.0)]
        return _unity_pose_to_T(pos_xyz, rot_xyzw)

    def world_joints(self, T_world_tracking: np.ndarray):
        """Returns (left_pts, right_pts) in world frame, each (N,3) or None.
        Prefers real hand; falls back to synth hand if real is absent."""
        if self.data is None:
            return None, None
        hands = self.data.get("hands") or {}

        def _resolve(real_key, synth_key):
            j = _extract_joints(hands.get(real_key))
            if j is None:
                j = _extract_joints(hands.get(synth_key))
            if j is None:
                return None
            if T_world_tracking is None:
                return _unity_to_o3d(j)
            return _to_world(j, T_world_tracking)

        left_pts  = _resolve("LeftHand",  "LeftHandSynth")
        right_pts = _resolve("RightHand", "RightHandSynth")
        return left_pts, right_pts

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass


# =============================================================================
# ArUco pose estimator
# =============================================================================

class _ArucoPoseEstimator:
    def __init__(self, world_marker_id: int, pegboard_marker_id: int, marker_size_m: float,
                 dictionary=cv.aruco.DICT_6X6_1000):
        self.world_marker_id    = int(world_marker_id)
        self.pegboard_marker_id = int(pegboard_marker_id)
        self.marker_size        = float(marker_size_m)
        self._dict       = cv.aruco.getPredefinedDictionary(dictionary)
        self._detector   = cv.aruco.ArucoDetector(
            self._dict, cv.aruco.DetectorParameters())
        s = self.marker_size / 2.0
        self._obj_pts = np.array([
            [-s,  s, 0.], [ s,  s, 0.],
            [ s, -s, 0.], [-s, -s, 0.],
        ], dtype=np.float64)

    def detect(self, bgr, fx, fy, cx, cy, dist=None, draw=True) -> dict:
        K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1)) if dist is None else np.array(dist).reshape(-1, 1)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        vis = bgr.copy()
        result = {"vis": vis, "T_cam_world": None, "T_cam_pegboard": None}
        if ids is None:
            return result
        if draw:
            cv.aruco.drawDetectedMarkers(vis, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid == self.world_marker_id:
                key = "T_cam_world"
            elif mid == self.pegboard_marker_id:
                key = "T_cam_pegboard"
            else:
                continue
            ok, rvec, tvec = cv.solvePnP(
                self._obj_pts, c.reshape(4, 2).astype(np.float64), K, dist,
                flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            Rm, _ = cv.Rodrigues(rvec)
            T_cam_marker = np.eye(4, dtype=np.float64)
            T_cam_marker[:3, :3] = Rm
            T_cam_marker[:3, 3]  = tvec.reshape(3)
            result[key] = T_cam_marker
            if draw:
                cv.drawFrameAxes(vis, K, dist, rvec, tvec, self.marker_size * 0.5, 2)
        return result


# =============================================================================
# Open3D scene visualizer — camera frustum + hand joints
# =============================================================================

class _SceneVis:
    FRUSTUM_SCALE = 0.2

    def __init__(self, title: str, width: int = 1000, height: int = 680):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=width, height=height)
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.10])
        ro.point_size = 7.0
        ro.line_width = 2.0

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(world_frame)

        # Camera frustum
        self._cam_frustum      = None
        self._cam_frustum_T    = self._hidden_T()

        # CenterEyeAnchor (real-time head pose) — magenta frustum
        self._head_frustum = None

        # Marker 101 (pegboard) — green sphere + frame, shown once anchor is locked
        self._pegboard_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._pegboard_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._pegboard_frame)
        self._pegboard_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.020)
        self._pegboard_sphere.paint_uniform_color([0.1, 1.0, 0.1])
        self._pegboard_sphere.compute_vertex_normals()
        self._pegboard_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._pegboard_sphere)
        self._pegboard_T = self._hidden_T()

        # Quest tracking space origin — blue sphere + frame
        self._tracking_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._tracking_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_frame)
        self._tracking_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.020)
        self._tracking_sphere.paint_uniform_color([0.2, 0.4, 1.0])
        self._tracking_sphere.compute_vertex_normals()
        self._tracking_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_sphere)
        self._tracking_T = self._hidden_T()

        # Hand joints + bones
        self._pcd_l, self._lines_l = self._make_hand([0.3, 0.6, 1.0])   # blue
        self._pcd_r, self._lines_r = self._make_hand([1.0, 0.55, 0.1])  # orange

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0., 0., 0.])
        ctr.set_front([0., -0.5, -1.])
        ctr.set_up([0., 1., 0.])
        ctr.set_zoom(0.5)

    @staticmethod
    def _hidden_T():
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0., -1.5, 0.]
        return T

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

    def update_cam_frustum(self, T: np.ndarray | None,
                           w=640, h=480, fx=400., fy=400., cx=320., cy=240.):
        T_use = T if T is not None else self._hidden_T()
        intr  = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
        new_fr = o3d.geometry.LineSet.create_camera_visualization(
            int(w), int(h), intr.intrinsic_matrix,
            np.linalg.inv(T_use), scale=self.FRUSTUM_SCALE)
        new_fr.paint_uniform_color([0.2, 1.0, 0.3])

        if self._cam_frustum is None:
            self._cam_frustum = new_fr
            self.vis.add_geometry(self._cam_frustum)
        else:
            self._cam_frustum.points = new_fr.points
            self._cam_frustum.lines  = new_fr.lines
            self._cam_frustum.colors = new_fr.colors
            self.vis.update_geometry(self._cam_frustum)

    def update_pegboard(self, T: np.ndarray | None):
        """Marker 101 in world frame (green)."""
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._pegboard_T)
        self._pegboard_frame.transform(delta)
        self._pegboard_sphere.transform(delta)
        self._pegboard_T = T_new
        self.vis.update_geometry(self._pegboard_frame)
        self.vis.update_geometry(self._pegboard_sphere)

    def update_tracking(self, T: np.ndarray | None):
        """Quest tracking space origin in world frame (blue)."""
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._tracking_T)
        self._tracking_frame.transform(delta)
        self._tracking_sphere.transform(delta)
        self._tracking_T = T_new
        self.vis.update_geometry(self._tracking_frame)
        self.vis.update_geometry(self._tracking_sphere)

    def update_head(self, T: np.ndarray | None,
                    w=640, h=480, fx=400., fy=400., cx=320., cy=240.):
        """CenterEyeAnchor — real-time head pose, magenta frustum."""
        T_use = T if T is not None else self._hidden_T()
        intr  = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
        new_fr = o3d.geometry.LineSet.create_camera_visualization(
            int(w), int(h), intr.intrinsic_matrix,
            np.linalg.inv(T_use), scale=self.FRUSTUM_SCALE)
        new_fr.paint_uniform_color([1.0, 0.1, 0.9])

        if self._head_frustum is None:
            self._head_frustum = new_fr
            self.vis.add_geometry(self._head_frustum)
        else:
            self._head_frustum.points = new_fr.points
            self._head_frustum.lines  = new_fr.lines
            self._head_frustum.colors = new_fr.colors
            self.vis.update_geometry(self._head_frustum)

    def update_hands(self, left_pts: np.ndarray | None, right_pts: np.ndarray | None):
        self._set_hand(self._pcd_l, self._lines_l, left_pts)
        self._set_hand(self._pcd_r, self._lines_r, right_pts)

    def tick(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# =============================================================================
# World anchor
# =============================================================================

class _WorldAnchor:
    _EYE_OFFSET_FILE = _FILE_DIR / "eye_offset_calibration.json"

    def __init__(self, pub_ip: str, pub_port: int = 5005, pegboard_pub_port: int = 5008):
        self._T_wt: np.ndarray | None = None
        self._T_offset = np.eye(4, dtype=np.float64)
        self._T_world_pegboard: np.ndarray | None = None
        self._T_eye_offset: np.ndarray | None = None  # inv(center_T) @ cam_T, fixed HMD geometry
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{pub_ip}:{pub_port}")
        self._pub_pegboard = ctx.socket(zmq.PUB)
        self._pub_pegboard.connect(f"tcp://{pub_ip}:{pegboard_pub_port}")
        time.sleep(0.2)
        self._load_eye_offset()

    def _load_eye_offset(self):
        if not self._EYE_OFFSET_FILE.exists():
            return
        try:
            data = json.loads(self._EYE_OFFSET_FILE.read_text())
            self._T_eye_offset = np.array(data["T_eye_offset"],
                                          dtype=np.float64).reshape(4, 4)
            print(f"[Anchor] Eye offset loaded from {self._EYE_OFFSET_FILE.name}")
        except Exception as e:
            print(f"[Anchor] Eye offset load failed: {e}")

    def _save_eye_offset(self):
        try:
            self._EYE_OFFSET_FILE.write_text(
                json.dumps({"T_eye_offset": self._T_eye_offset.flatten().tolist()},
                           indent=2))
            print(f"[Anchor] Eye offset saved to {self._EYE_OFFSET_FILE.name}")
        except Exception as e:
            print(f"[Anchor] Eye offset save failed: {e}")

    def _effective_cam_T(self, cam_T: np.ndarray,
                         center_T: np.ndarray) -> np.ndarray | None:
        """Return the best available camera pose for locking.
        Prefers center_T @ T_eye_offset (no WiFi delay) once calibrated."""
        if self._T_eye_offset is not None and center_T is not None:
            return center_T @ self._T_eye_offset
        return cam_T  # fallback to raw cam on very first lock

    def set_offset(self, pos_offset, yaw_deg: float):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = np.array(pos_offset, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('z', yaw_deg, degrees=True).as_matrix()
        self._T_offset = T

    def lock(self, T_cam_world: np.ndarray, T_cam_pegboard: np.ndarray,
             cam_T: np.ndarray, center_T: np.ndarray | None = None) -> bool:
        if T_cam_world is None or T_cam_pegboard is None:
            return False
        # Calibrate offset once if file absent and both poses available
        if self._T_eye_offset is None and cam_T is not None and center_T is not None:
            self._T_eye_offset = np.linalg.inv(center_T) @ cam_T
            self._save_eye_offset()
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        self._T_wt = np.linalg.inv(eff @ T_cam_world)
        self._T_world_pegboard = np.linalg.inv(T_cam_world) @ T_cam_pegboard
        src = "CenterEye+offset" if self._T_eye_offset is not None else "cam_T"
        t = self._T_world_pegboard[:3, 3]
        print(f"[Anchor] Locked ({src}). Pegboard: "
              f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")
        return True

    def relock(self, T_cam_world: np.ndarray,
               cam_T: np.ndarray, center_T: np.ndarray | None = None,
               T_cam_pegboard: np.ndarray | None = None) -> bool:
        """Update T_wt; also updates pegboard pose if T_cam_pegboard is provided."""
        if T_cam_world is None or not self.locked:
            return False
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        self._T_wt = np.linalg.inv(eff @ T_cam_world)
        if T_cam_pegboard is not None:
            self._T_world_pegboard = np.linalg.inv(T_cam_world) @ T_cam_pegboard
        return True

    @property
    def locked(self) -> bool:
        return self._T_wt is not None

    @property
    def T_world_tracking(self) -> np.ndarray | None:
        if self._T_wt is None:
            return None
        return self._T_offset @ self._T_wt

    @property
    def T_pegboard_in_world(self) -> np.ndarray | None:
        """Pegboard pose in the (offset-adjusted) world frame."""
        if self._T_world_pegboard is None:
            return None
        return self._T_offset @ self._T_world_pegboard

    def world_T(self, T_tracking_local: np.ndarray) -> np.ndarray | None:
        if self._T_wt is None:
            return None
        return (self._T_offset @ self._T_wt) @ T_tracking_local

    def publish(self) -> bool:
        if self._T_wt is None:
            return False
        T_tracking_world = np.linalg.inv(self._T_offset @ self._T_wt)
        R_o3d = T_tracking_world[:3, :3]
        t_o3d = T_tracking_world[:3, 3]
        q_xyzw   = ScipyR.from_matrix(R_o3d).as_quat()
        q_wxyz   = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u_wxyz = open3d_to_unity_quaternion(q_wxyz)
        t_unity  = open3d_to_unity_vector(t_o3d)
        q_unity_xyzw = [float(q_u_wxyz[1]), float(q_u_wxyz[2]),
                        float(q_u_wxyz[3]), float(q_u_wxyz[0])]
        R_unity = ScipyR.from_quat(q_unity_xyzw).as_matrix()
        T_unity = np.eye(4, dtype=np.float64)
        T_unity[:3, :3] = R_unity
        T_unity[:3, 3]  = t_unity
        msg = {
            "world_root_position":      [float(v) for v in t_unity],
            "world_root_rotation_xyzw":  q_unity_xyzw,
            "world_root_matrix":         T_unity.T.flatten().tolist(),
        }
        try:
            self._pub.send_string(json.dumps(msg))
            return True
        except Exception as e:
            print(f"[WorldRoot] Publish error: {e}")
            return False

    def publish_pegboard(self) -> bool:
        if self._T_world_pegboard is None:
            return False

        # Pegboard pose expressed in (offset-adjusted) world frame, Open3D coords
        T_o3d = self._T_offset @ self._T_world_pegboard
        R_o3d = T_o3d[:3, :3]
        t_o3d = T_o3d[:3, 3]

        # Convert to Unity coords (same dance as publish())
        q_xyzw   = ScipyR.from_matrix(R_o3d).as_quat()
        q_wxyz   = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u_wxyz = open3d_to_unity_quaternion(q_wxyz)
        t_unity  = open3d_to_unity_vector(t_o3d)
        q_unity_xyzw = [float(q_u_wxyz[1]), float(q_u_wxyz[2]),
                        float(q_u_wxyz[3]), float(q_u_wxyz[0])]
        R_unity = ScipyR.from_quat(q_unity_xyzw).as_matrix()
        T_unity = np.eye(4, dtype=np.float64)
        T_unity[:3, :3] = R_unity
        T_unity[:3, 3]  = t_unity

        msg = {
            "pegboard_root_position":      [float(v) for v in t_unity],
            "pegboard_root_rotation_xyzw":  q_unity_xyzw,
            "pegboard_root_matrix":         T_unity.T.flatten().tolist(),
        }
        try:
            self._pub_pegboard.send_string(json.dumps(msg))
            return True
        except Exception as e:
            print(f"[PegboardRoot] Publish error: {e}")
            return False

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass
        try:
            self._pub_pegboard.close(0)
        except Exception:
            pass


# =============================================================================
# Haptic publisher  (same as main.py)
# =============================================================================

class _HapticPublisher:
    def __init__(self, ip: str, port: int = 5007):
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.1)

    def vibrate(self, controller="both", amplitude=1.0, frequency=0.5, duration_ms=200):
        msg = {
            "controller":  controller,
            "amplitude":   float(np.clip(amplitude,  0., 1.)),
            "frequency":   float(np.clip(frequency,  0., 1.)),
            "duration_ms": int(max(0, duration_ms)),
        }
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception as e:
            print(f"[Haptic] Error: {e}")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


# =============================================================================
# Synthetic object publisher
# =============================================================================

class _SyntheticObject:
    def __init__(self, obj_id, centroid_o3d, width, depth, height,
                 yaw_deg=0.0, R_o3d=None, color=None):
        self.obj_id   = int(obj_id)
        self.centroid = np.array(centroid_o3d, dtype=np.float64)
        self.width    = float(np.clip(width,  0.03, 0.50))
        self.depth    = float(np.clip(depth,  0.03, 0.50))
        self.height   = float(np.clip(height, 0.01, 0.50))
        self.yaw_deg  = float(yaw_deg)
        self.R_o3d    = np.array(R_o3d, dtype=np.float64) if R_o3d is not None else None
        self.color    = list(color) if color is not None else [0.2, 0.65, 1.0]

    def to_unity_dict(self) -> dict:
        p_unity    = open3d_to_unity_vector(self.centroid)
        size_unity = open3d_to_unity_vector(
            np.array([self.width, self.depth, self.height], dtype=np.float64))

        if self.R_o3d is not None:
            # Full 3x3 rotation (e.g. pegboard orientation)
            q_xyzw = ScipyR.from_matrix(self.R_o3d).as_quat()
        else:
            # Yaw-only fallback (original behavior)
            q_xyzw = ScipyR.from_euler('z', self.yaw_deg, degrees=True).as_quat()

        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u    = open3d_to_unity_quaternion(q_wxyz)
        return {
            "id":            self.obj_id,
            "position":      [float(v) for v in p_unity],
            "rotation_xyzw": [float(q_u[1]), float(q_u[2]), float(q_u[3]), float(q_u[0])],
            "size":          [float(v) for v in size_unity],
            "color":         [float(v) for v in self.color],
        }


class _SyntheticObjectPublisher:
    def __init__(self, ip: str, port: int = 5006):
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.2)
        self._objects: list[_SyntheticObject] = []
        print(f"[SynthObjects] Connected to tcp://{ip}:{port}")

    def add(self, centroid_o3d, width, depth, height,
            color=None, yaw_deg=0.0, R_o3d=None) -> "_SyntheticObject":
        obj = _SyntheticObject(len(self._objects), centroid_o3d,
                               width, depth, height,
                               yaw_deg=yaw_deg, R_o3d=R_o3d, color=color)
        self._objects.append(obj)
        return obj

    def publish(self):
        payload = {"objects": [o.to_unity_dict() for o in self._objects]}
        try:
            self._pub.send_string(json.dumps(payload))
        except Exception as e:
            print(f"[SynthObjects] Publish error: {e}")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


# =============================================================================
# Offset tuner  (same as main.py)
# =============================================================================

class _OffsetTuner:
    WIN       = "Offset Tuner (ArUco frame)"
    SAVE_FILE = _FILE_DIR / "offset_config_passthrough.json"
    _BTN      = (10, 10, 110, 44)

    def __init__(self):
        cv.namedWindow(self.WIN, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.WIN, 420, 260)
        cv.createTrackbar("X  right  (mm×0.1)", self.WIN, 100, 200, lambda _: None)
        cv.createTrackbar("Y  away   (mm×0.1)", self.WIN, 100, 200, lambda _: None)
        cv.createTrackbar("Z  up     (mm×0.1)", self.WIN, 100, 200, lambda _: None)
        cv.createTrackbar("Yaw CCW   (0.5°)",   self.WIN,  90, 180, lambda _: None)
        self._flash_until = 0.0
        self._load()
        cv.setMouseCallback(self.WIN, self._on_mouse)

    def _raw(self):
        return {
            "x":   cv.getTrackbarPos("X  right  (mm×0.1)", self.WIN),
            "y":   cv.getTrackbarPos("Y  away   (mm×0.1)", self.WIN),
            "z":   cv.getTrackbarPos("Z  up     (mm×0.1)", self.WIN),
            "yaw": cv.getTrackbarPos("Yaw CCW   (0.5°)",   self.WIN),
        }

    def get(self):
        r = self._raw()
        return ((r["x"] - 100) * 0.001,
                (r["y"] - 100) * 0.001,
                (r["z"] - 100) * 0.001), (r["yaw"] - 90) * 0.5

    def _save(self):
        with open(self.SAVE_FILE, "w") as f:
            json.dump(self._raw(), f, indent=2)
        self._flash_until = time.time() + 1.5
        print(f"[OffsetTuner] Saved to {self.SAVE_FILE}")

    def _load(self):
        if not self.SAVE_FILE.exists():
            return
        try:
            with open(self.SAVE_FILE) as f:
                data = json.load(f)
            cv.setTrackbarPos("X  right  (mm×0.1)", self.WIN, int(data.get("x",   100)))
            cv.setTrackbarPos("Y  away   (mm×0.1)", self.WIN, int(data.get("y",   100)))
            cv.setTrackbarPos("Z  up     (mm×0.1)", self.WIN, int(data.get("z",   100)))
            cv.setTrackbarPos("Yaw CCW   (0.5°)",   self.WIN, int(data.get("yaw",  90)))
            print(f"[OffsetTuner] Loaded from {self.SAVE_FILE}")
        except Exception as e:
            print(f"[OffsetTuner] Load error: {e}")

    def _on_mouse(self, event, x, y, *_):
        if event == cv.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = self._BTN
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._save()

    def draw(self):
        img = np.zeros((60, 420, 3), dtype=np.uint8)
        x0, y0, x1, y1 = self._BTN
        flashing   = time.time() < self._flash_until
        btn_color  = (0, 200, 80)  if flashing else (50, 130, 50)
        btn_border = (0, 255, 120) if flashing else (80, 200, 80)
        label      = "  Saved!"   if flashing else "  SAVE"
        cv.rectangle(img, (x0, y0), (x1, y1), btn_color, -1)
        cv.rectangle(img, (x0, y0), (x1, y1), btn_border, 2)
        cv.putText(img, label, (x0 + 4, y0 + 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
        (px, py, pz), yaw = self.get()
        info = f"X={px*100:+.1f}cm  Y={py*100:+.1f}cm  Z={pz*100:+.1f}cm  Yaw={yaw:+.1f}°"
        cv.putText(img, info, (10, 54),
                   cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv.LINE_AA)
        cv.imshow(self.WIN, img)

    def close(self):
        try:
            cv.destroyWindow(self.WIN)
        except Exception:
            pass

class _ToolSelectionManager:
    SELECTED_COLOR = [0.0, 1.0, 0.0, 0.5]   # green
    HOVER_COLOR    = [1.0, 0.5, 0.0, 0.5]   # orange
    RESET_COLOR    = [-1.0, -1.0, -1.0, -1.0]

    def __init__(self, quest_ip: str,
                 click_port: int = 5009,
                 color_port: int = 5010):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{quest_ip}:{click_port}")
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{color_port}")
        time.sleep(0.2)
        self._active_tool_id: int | None = None
        self._hovered_tool_id: int | None = None
        print(f"[ToolSelection] SUB clicks ← {quest_ip}:{click_port}")
        print(f"[ToolSelection] PUB colors → {quest_ip}:{color_port}")

    def poll(self, timeout_ms: int = 0) -> bool:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False
        latest = None
        while True:
            try:
                latest = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return False
        try:
            msg = json.loads(latest)
            tool_id = int(msg["tool_id"])
            event_type = msg.get("event_type", "selected")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[ToolSelection] Bad message: {e}")
            return False
        self._handle_event(tool_id, event_type)
        return True

    def _handle_event(self, tool_id: int, event_type: str):
        if event_type == "selected":
            self._handle_click(tool_id)
        elif event_type == "hover_enter":
            self._handle_hover_enter(tool_id)
        elif event_type == "hover_exit":
            self._handle_hover_exit(tool_id)
        else:
            print(f"[ToolSelection] Unknown event_type: {event_type}")

    def _handle_click(self, tool_id: int):
        # When a tool is clicked, it becomes selected and (visually) un-hovered.
        # Hover state is cleared so the green isn't re-overwritten by orange next frame.
        self._hovered_tool_id = None
        updates: list[tuple[int, list[float]]] = []
        if self._active_tool_id == tool_id:
            updates.append((tool_id, self.RESET_COLOR))
            self._active_tool_id = None
            print(f"[ToolSelection] Tool {tool_id} deselected")
        elif self._active_tool_id is not None:
            updates.append((self._active_tool_id, self.RESET_COLOR))
            updates.append((tool_id,              self.SELECTED_COLOR))
            print(f"[ToolSelection] Switched: {self._active_tool_id} → off, {tool_id} → on")
            self._active_tool_id = tool_id
        else:
            updates.append((tool_id, self.SELECTED_COLOR))
            self._active_tool_id = tool_id
            print(f"[ToolSelection] Tool {tool_id} selected")
        for tid, color in updates:
            self._send_color(tid, color)

    def _handle_hover_enter(self, tool_id: int):
        # Selected tools don't show hover overlay — they stay green.
        if tool_id == self._active_tool_id:
            return
        self._hovered_tool_id = tool_id
        self._send_color(tool_id, self.HOVER_COLOR)
        print(f"[ToolSelection] Tool {tool_id} hovered")

    def _handle_hover_exit(self, tool_id: int):
        if tool_id != self._hovered_tool_id:
            return
        self._hovered_tool_id = None
        # Don't reset a selected tool — it should stay green.
        if tool_id == self._active_tool_id:
            return
        self._send_color(tool_id, self.RESET_COLOR)
        print(f"[ToolSelection] Tool {tool_id} unhovered")

    def _send_color(self, tool_id: int, color: list[float]):
        msg = {"tool_id": int(tool_id), "color": [float(c) for c in color]}
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception as e:
            print(f"[ToolSelection] Publish error: {e}")

    @property
    def active_tool_id(self) -> int | None:
        return self._active_tool_id

    def send_color(self, tool_id: int, color: list[float]):
        self._send_color(tool_id, color)

    def deselect(self, tool_id: int):
        if self._active_tool_id == tool_id:
            self._active_tool_id = None

    def close(self):
        try: self._sub.close(0)
        except Exception: pass
        try: self._pub.close(0)
        except Exception: pass# =============================================================================
# Main loop
# =============================================================================

def run(quest_ip: str, world_marker_id: int, pegboard_marker_id: int,
        marker_size_m: float, hand_port: int):

    cam   = _CamFeedReceiver(quest_ip)
    aruco = _ArucoPoseEstimator(world_marker_id, pegboard_marker_id, marker_size_m)
    hands = _HandDataReceiver(quest_ip, hand_port)

    anchor = _WorldAnchor(quest_ip, pub_port=5005)
    haptic = _HapticPublisher(quest_ip, port=5007)
    # Near the other receivers/publishers
    tools = _ToolSelectionManager(quest_ip,
                              click_port=5009,
                              color_port=5010)
    tuner  = _OffsetTuner()

    synth = _SyntheticObjectPublisher(quest_ip, port=5006)
    # Cubes around the WORLD marker
    synth.add([0.10,  0.00, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[1.0, 0.2, 0.2])
    synth.add([0.00,  0.10, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[0.2, 1.0, 0.2])
    synth.add([-0.10, 0.00, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[0.2, 0.4, 1.0])

    # Cubes around the PEGBOARD marker — offsets in pegboard-local frame.
    # Added to synth after lock, once T_world_pegboard is known.
    PEGBOARD_CUBES = [
        {"offset": [ 0.10, 0.00, 0.05], "color": [1.0, 0.6, 0.2]},  # orange
        {"offset": [ 0.00, 0.10, 0.05], "color": [0.8, 0.2, 0.8]},  # magenta
        {"offset": [-0.10, 0.00, 0.05], "color": [0.2, 1.0, 0.9]},  # cyan
    ]
    pegboard_cubes_added = False
    _pegboard_cube_start = None

    haptic_lock_sent = False
    _last_synth_pub  = 0.0
    _SYNTH_INTERVAL  = 1.0 / 30.0
    _relock_available_prev  = False
    _green_until            = 0.0
    _last_proximity_relock  = 0.0

    vis = _SceneVis(f"Hand Tracking — World Frame  (marker #{world_marker_id})")

    win = f"Quest Left Passthrough  [ENTER=re-lock  ESC=quit]  marker #{world_marker_id}"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, 960, 540)

    print(f"\n[Running]  quest_ip={quest_ip}  "
          f"world_marker={world_marker_id}  pegboard_marker={pegboard_marker_id}  "
          f"hand_port={hand_port}")
    print("  ENTER = re-lock anchor    ESC = quit\n")

    try:
        while True:
            # ── Poll streams ──────────────────────────────────────────────────
            cam.poll(timeout_ms=5)

            tools.poll(timeout_ms=0)
            hands.poll()

            # ── ArUco detection ───────────────────────────────────────────────
            T_cam_world_fresh    = None
            T_cam_pegboard_fresh = None
            det_vis = None

            if cam.frame is not None and cam.fx is not None:
                fx, fy, cx, cy = _adapt_cx_cy(
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    cam.sensor_width, cam.sensor_height,
                    cam.width, cam.height)
                det = aruco.detect(cam.frame, fx, fy, cx, cy, draw=True)
                det_vis              = det["vis"]
                T_cam_world_fresh    = det["T_cam_world"]
                T_cam_pegboard_fresh = det["T_cam_pegboard"]

            both_seen = (T_cam_world_fresh    is not None and
                         T_cam_pegboard_fresh is not None and
                         cam.camera_T        is not None)

            # ── Anchor + publishers ───────────────────────────────────────────
            tuner.draw()
            pos_off, yaw_off = tuner.get()
            anchor.set_offset(pos_off, yaw_off)

            # ── World-frame poses ─────────────────────────────────────────────
            _center_T       = hands.center_eye_T()

            if anchor.locked:
                anchor.publish()
                anchor.publish_pegboard()
                _now = time.time()
                if _now - _last_synth_pub >= _SYNTH_INTERVAL:
                    synth.publish()
                    _last_synth_pub = _now
                if not haptic_lock_sent:
                    haptic.vibrate("both", amplitude=0.8,
                                   frequency=0.3, duration_ms=300)
                    haptic_lock_sent = True

            T_wt            = anchor.T_world_tracking
            T_world_camleft = anchor.world_T(cam.camera_T) \
                              if cam.camera_T is not None else None
            T_world_center  = anchor.world_T(_center_T) if _center_T is not None else None

            left_pts, right_pts = hands.world_joints(T_wt)

            # ── World marker proximity relock ─────────────────────────────────
            _now = time.time()
            dist_to_world_marker = (float(np.linalg.norm(T_cam_world_fresh[:3, 3]))
                                    if both_seen else float('inf'))
            # Only available once already locked; initial lock stays ENTER-only.
            _relock_available = (anchor.locked and both_seen
                                 and dist_to_world_marker < 1.0)

            if _green_until > 0.0 and _now >= _green_until:
                _green_until = 0.0
                _relock_available_prev = not _relock_available  # force color refresh

            if _green_until == 0.0 and _relock_available != _relock_available_prev:
                tools.send_color(world_marker_id,
                                 _ToolSelectionManager.HOVER_COLOR if _relock_available
                                 else _ToolSelectionManager.RESET_COLOR)
                _relock_available_prev = _relock_available

            _RELOCK_COOLDOWN = 2.0
            if (tools.active_tool_id == world_marker_id and _relock_available
                    and _now - _last_proximity_relock >= _RELOCK_COOLDOWN):
                anchor.relock(T_cam_world_fresh, cam.camera_T, _center_T,
                              T_cam_pegboard_fresh)
                if pegboard_cubes_added and anchor.T_pegboard_in_world is not None:
                    T_wp = anchor.T_pegboard_in_world
                    R_wp = T_wp[:3, :3]
                    for i, cube in enumerate(PEGBOARD_CUBES):
                        obj = synth._objects[_pegboard_cube_start + i]
                        obj.centroid = _transform_point(T_wp, cube["offset"])
                        obj.R_o3d = R_wp.copy()
                tools.send_color(world_marker_id, _ToolSelectionManager.SELECTED_COLOR)
                _green_until = _now + 1.0
                _relock_available_prev = True
                _last_proximity_relock = _now
                haptic_lock_sent = False
                print("[AutoRelock] Relocked via proximity click")
            tools.deselect(world_marker_id)

            # ── Update Open3D ─────────────────────────────────────────────────
            if cam.fx is not None:
                fx, fy, cx, cy = _adapt_cx_cy(
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    cam.sensor_width, cam.sensor_height,
                    cam.width, cam.height)
                vis.update_cam_frustum(T_world_camleft,
                                       cam.width, cam.height, fx, fy, cx, cy)
            vis.update_pegboard(anchor.T_pegboard_in_world)
            vis.update_tracking(T_wt)
            vis.update_head(T_world_center)
            vis.update_hands(left_pts, right_pts)
            vis.tick()

            # ── OpenCV display ────────────────────────────────────────────────
            disp = cv.resize(
                det_vis if det_vis is not None else
                (cam.frame.copy() if cam.frame is not None
                 else np.zeros((480, 640, 3), dtype=np.uint8)),
                (960, 540))

            locked      = anchor.locked
            world_ok    = T_cam_world_fresh    is not None
            pegboard_ok = T_cam_pegboard_fresh is not None

            cv.putText(disp,
                       f"Marker #{world_marker_id}: {'DETECTED' if world_ok else 'searching...'}    "
                       f"Marker #{pegboard_marker_id}: {'DETECTED' if pegboard_ok else 'searching...'}    "
                       f"{'READY TO LOCK!' if both_seen else ''}",
                       (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 80) if both_seen else (0, 80, 255), 2)
            cv.putText(disp,
                       f"Anchor: {'LOCKED — WorldRoot :5005 + PegboardRoot :5008 + synth :5006' if locked else 'waiting for first detection'}",
                       (12, 68), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                       (0, 255, 150) if locked else (100, 100, 100), 2)
            hand_ok = left_pts is not None or right_pts is not None
            if hand_ok:
                hand_status = (
                    "L+R" if (left_pts is not None and right_pts is not None)
                    else ("L" if left_pts is not None else "R")
                )
            elif hands.receiving and not locked:
                hand_status = f"tracking-frame #{hands.message_count}"
            elif hands.receiving:
                hand_status = f"receiving #{hands.message_count}, no valid joints"
            elif hands.last_error:
                hand_status = f"parse error: {hands.last_error[:32]}"
            else:
                hand_status = f"waiting on port {hand_port}"
            cv.putText(disp,
                       f"Hands : {hand_status}",
                       (12, 102), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                       (0, 255, 200) if (hand_ok or hands.receiving) else (100, 100, 100), 2)
            cv.putText(disp,
                       "ENTER = re-lock   ESC = quit",
                       (12, disp.shape[0] - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv.imshow(win, disp)

            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 13:
                if both_seen:
                    if anchor.locked:
                        anchor.relock(T_cam_world_fresh, cam.camera_T, _center_T,
                                      T_cam_pegboard_fresh)
                        print("[ENTER] Relocked")
                        if pegboard_cubes_added and anchor.T_pegboard_in_world is not None:
                            T_wp = anchor.T_pegboard_in_world
                            R_wp = T_wp[:3, :3]
                            for i, cube in enumerate(PEGBOARD_CUBES):
                                obj = synth._objects[_pegboard_cube_start + i]
                                obj.centroid = _transform_point(T_wp, cube["offset"])
                                obj.R_o3d = R_wp.copy()
                    else:
                        anchor.lock(T_cam_world_fresh, T_cam_pegboard_fresh,
                                    cam.camera_T, center_T=_center_T)
                    haptic_lock_sent = False
                    _last_proximity_relock = _now  # block proximity relock for 2 s after any manual lock
                    if not pegboard_cubes_added and anchor.T_pegboard_in_world is not None:
                        T_wp = anchor.T_pegboard_in_world
                        R_wp = T_wp[:3, :3]
                        _pegboard_cube_start = len(synth._objects)
                        for cube in PEGBOARD_CUBES:
                            centroid_world = _transform_point(T_wp, cube["offset"])
                            synth.add(centroid_world,
                                      width=0.06, depth=0.06, height=0.10,
                                      color=cube["color"], R_o3d=R_wp)
                        pegboard_cubes_added = True
                        print(f"[Synth] Added {len(PEGBOARD_CUBES)} pegboard cubes "
                              f"based on marker #{pegboard_marker_id} pose")
                else:
                    missing = []
                    if T_cam_world_fresh is None:
                        missing.append(f"marker #{world_marker_id}")
                    if T_cam_pegboard_fresh is None:
                        missing.append(f"pegboard marker #{pegboard_marker_id}")
                    print(f"[ENTER] Need both markers — missing: {', '.join(missing)}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        vis.close()
        cv.destroyAllWindows()
        tuner.close()
        anchor.close()
        haptic.close()
        synth.close()
        hands.close()
        cam.close()
        tools.close()
        print("[Done]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Quest passthrough ArUco + hand tracking visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip",        default=cfg.UNITY_IP)
    ap.add_argument("--world-marker",    type=int, default=100,
                    help="ArUco marker ID for world anchor (defines world frame origin)")
    ap.add_argument("--pegboard-marker", type=int, default=101,
                    help="ArUco marker ID for pegboard (pose expressed relative to world)")
    ap.add_argument("--marker-size",     type=float, default=0.100,
                    help="Marker side length in metres")
    ap.add_argument("--hand-port",       type=int, default=cfg.HAND1_PORT_FROM_UNITY)
    args = ap.parse_args()
    if args.world_marker == args.pegboard_marker:
        ap.error("--world-marker and --pegboard-marker must be different.")
    run(quest_ip           = args.quest_ip,
        world_marker_id    = args.world_marker,
        pegboard_marker_id = args.pegboard_marker,
        marker_size_m      = args.marker_size,
        hand_port          = args.hand_port)


if __name__ == "__main__":
    main()
