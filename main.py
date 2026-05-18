"""
calibrate_test_passthrough_quest.py — Quest passthrough ArUco → WorldRoot

Uses the Meta Quest left passthrough camera to detect ArUco marker #113,
computes T_world_tracking, and publishes WorldRoot to Unity on port 5005.
No RealSense or pre-run calibration files required.

Transform chain
---------------
  T_cam_marker     = ArUco detection (marker in OpenCV camera frame)
  cam.camera_T     = T_tracking_camleft (Quest left cam in tracking, Open3D convention)

  T_tracking_marker = cam.camera_T @ T_cam_marker
  T_world_tracking  = inv(T_tracking_marker)         ← world = marker frame

  T_world_center    = T_world_tracking @ xr.center_T
  T_world_camleft   = T_world_tracking @ cam.camera_T
  T_world_ctrl      = T_world_tracking @ xr.left/right_T

Keys (OpenCV window must be focused)
--------------------------------------
  ENTER  = force re-lock anchor from current detection
  ESC    = quit

Usage
-----
  python calibrate_test_passthrough_quest.py --quest-ip 192.168.50.201
  python calibrate_test_passthrough_quest.py --quest-ip 192.168.50.201 \\
      --marker-id 113 --marker-size 0.100
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

# ── Path setup ─────────────────────────────────────────────────────────────────
_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from utils.unity_conversion import (
    unity_to_open3d_vector,
    unity_to_open3d_quaternion,
    open3d_to_unity_vector,
    open3d_to_unity_quaternion,
)
import ip_setting as ip_cfg


# =============================================================================
# Pose helpers
# =============================================================================

def _unity_pose_to_T(pos_xyz, rot_xyzw) -> np.ndarray:
    """Head / camera pose: Unity → Open3D with -90° X axis correction."""
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


def _unity_ctrl_pose_to_T(pos_xyz, rot_xyzw) -> np.ndarray:
    """Controller pose: Unity → Open3D (no -90° correction)."""
    pos_dict = {"x": float(pos_xyz[0]), "y": float(pos_xyz[1]), "z": float(pos_xyz[2])}
    p = unity_to_open3d_vector(pos_dict)
    x, y, z, w = rot_xyzw
    q_o3d = unity_to_open3d_quaternion([float(w), float(x), float(y), float(z)])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(q_o3d)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = p
    return T


def _adapt_cx_cy(fx, fy, cx, cy, sensor_w, sensor_h, img_w, img_h):
    if sensor_w is None or sensor_h is None:
        return fx, fy, cx, cy
    crop_x = (float(sensor_w) - float(img_w)) / 2.0
    crop_y = (float(sensor_h) - float(img_h)) / 2.0
    return fx, fy, cx - crop_x, cy - crop_y



# =============================================================================
# ZMQ receivers
# =============================================================================

class _XRStateReceiver:
    def __init__(self, ip: str, port: int = 5559, topic: str = "xr"):
        ctx = zmq.Context()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{ip}:{port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.center_T: np.ndarray | None = None
        self.left_T:   np.ndarray | None = None
        self.right_T:  np.ndarray | None = None

    def poll(self, timeout_ms: int = 0) -> bool:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False
        latest = None
        while True:
            try:
                _      = self._sub.recv_string(flags=zmq.NOBLOCK)
                latest = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return False
        msg = json.loads(latest)

        head = msg.get("head")
        if head and head.get("is_valid"):
            p, r = head.get("pos"), head.get("rot_xyzw")
            if p and r:
                self.center_T = _unity_pose_to_T(p, r)

        for attr, key in (("left_T", "left"), ("right_T", "right")):
            blk = msg.get(key)
            if blk and blk.get("is_valid"):
                p, r = blk.get("pos"), blk.get("rot_xyzw")
                if p and r:
                    setattr(self, attr, _unity_ctrl_pose_to_T(p, r))
        return True

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass


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
        px, py, pz     = struct.unpack("<fff",  latest[4])
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


# =============================================================================
# ArUco pose estimator
# =============================================================================

class _ArucoPoseEstimator:
    def __init__(self, marker_id: int, marker_size_m: float,
                 dictionary=cv.aruco.DICT_6X6_1000):
        self.marker_id   = int(marker_id)
        self.marker_size = float(marker_size_m)
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
        vis  = bgr.copy()
        null = {"found": False, "T_cam_marker": None, "vis": vis}
        if ids is None:
            return null
        if draw:
            cv.aruco.drawDetectedMarkers(vis, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            if int(mid) != self.marker_id:
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
            if draw:
                cv.drawFrameAxes(vis, K, dist, rvec, tvec, self.marker_size * 0.5, 2)
            return {"found": True, "T_cam_marker": T_cam_marker, "vis": vis}
        return null


# =============================================================================
# Open3D frustum visualizer
# =============================================================================

class _FrustumVis:
    SCALE = 0.2

    def __init__(self, title: str, width: int = 1000, height: int = 680):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=width, height=height)
        self.vis.get_render_option().background_color = np.array([0.1, 0.1, 0.12])
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(world)
        self._frames   = {}
        self._frustums = {}
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0., 0., 0.])
        ctr.set_front([0., -0.5, -1.])
        ctr.set_up([0., 1., 0.])
        ctr.set_zoom(0.4)

    @staticmethod
    def _hidden_T():
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0., -1.5, 0.]
        return T

    @staticmethod
    def _move_mesh(mesh, T_prev, T_new):
        mesh.transform(T_new @ np.linalg.inv(T_prev))
        return T_new

    def add_frame(self, name: str, size: float = 0.12):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        T0   = self._hidden_T()
        mesh.transform(T0)
        self.vis.add_geometry(mesh)
        self._frames[name] = [mesh, T0]

    def add_frustum(self, name: str, color=(0.2, 1.0, 0.3)):
        dummy = o3d.camera.PinholeCameraIntrinsic(640, 480, 400, 400, 320, 240)
        fr    = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, dummy.intrinsic_matrix,
            np.linalg.inv(self._hidden_T()), scale=self.SCALE)
        fr.paint_uniform_color(list(color))
        self.vis.add_geometry(fr)
        self._frustums[name] = [fr, list(color)]

    def update_frame(self, name: str, T: np.ndarray | None):
        mesh, T_prev = self._frames[name]
        T_new = T if T is not None else self._hidden_T()
        self._frames[name][1] = self._move_mesh(mesh, T_prev, T_new)
        self.vis.update_geometry(mesh)

    def update_frustum(self, name: str, T: np.ndarray | None,
                       w: int = 640, h: int = 480,
                       fx: float = 400., fy: float = 400.,
                       cx: float = 320., cy: float = 240.):
        fr, color = self._frustums[name]
        if T is None:
            T = self._hidden_T()
        intr   = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
        new_fr = o3d.geometry.LineSet.create_camera_visualization(
            int(w), int(h), intr.intrinsic_matrix,
            np.linalg.inv(T), scale=self.SCALE)
        new_fr.paint_uniform_color(color)
        fr.points = new_fr.points
        fr.lines  = new_fr.lines
        fr.colors = new_fr.colors
        self.vis.update_geometry(fr)

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
    """
    Locks T_world_tracking = inv(cam.camera_T @ T_cam_marker) on demand.
    Holds the locked transform until lock() is called again.
    Publishes WorldRoot (T_tracking_world) to Unity on port 5005.
    """

    def __init__(self, pub_ip: str, pub_port: int = 5005):
        self._T_wt: np.ndarray | None = None
        self._T_offset = np.eye(4, dtype=np.float64)

        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{pub_ip}:{pub_port}")
        time.sleep(0.2)

    def set_offset(self, pos_offset, yaw_deg: float):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = np.array(pos_offset, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('z', yaw_deg, degrees=True).as_matrix()
        self._T_offset = T

    def lock(self, T_cam_marker: np.ndarray, cam_T: np.ndarray) -> bool:
        """Snap anchor to current detection. Call this when ENTER is pressed."""
        if T_cam_marker is None or cam_T is None:
            return False
        self._T_wt = np.linalg.inv(cam_T @ T_cam_marker)
        print("[Anchor] Locked")
        return True

    @property
    def locked(self) -> bool:
        return self._T_wt is not None

    @property
    def T_world_tracking(self) -> np.ndarray | None:
        if self._T_wt is None:
            return None
        return self._T_offset @ self._T_wt

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

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


# =============================================================================
# Haptic publisher
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
    """
    Box-shaped object in the ArUco marker world frame (Open3D: X right, Y forward, Z up).
    Serialises to the Unity JSON format expected on port 5006.
    """
    def __init__(self, obj_id: int, centroid_o3d, width: float, depth: float,
                 height: float, yaw_deg: float = 0.0, color=None):
        self.obj_id   = int(obj_id)
        self.centroid = np.array(centroid_o3d, dtype=np.float64)
        self.width    = float(np.clip(width,  0.03, 0.50))
        self.depth    = float(np.clip(depth,  0.03, 0.50))
        self.height   = float(np.clip(height, 0.01, 0.50))
        self.yaw_deg  = float(yaw_deg)
        self.color    = list(color) if color is not None else [0.2, 0.65, 1.0]

    def to_unity_dict(self) -> dict:
        p_unity    = open3d_to_unity_vector(self.centroid)
        size_unity = open3d_to_unity_vector(
            np.array([self.width, self.depth, self.height], dtype=np.float64))
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
    """
    Publishes synthetic box objects to Unity on port 5006.
    Uses connect (not bind) — Quest binds, Linux connects.
    """
    def __init__(self, ip: str, port: int = 5006):
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.2)
        self._objects: list[_SyntheticObject] = []
        print(f"[SynthObjects] Connected to tcp://{ip}:{port}")

    def add(self, centroid_o3d, width: float, depth: float, height: float,
            color=None, yaw_deg: float = 0.0) -> "_SyntheticObject":
        obj = _SyntheticObject(len(self._objects), centroid_o3d,
                               width, depth, height, yaw_deg=yaw_deg, color=color)
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
# Offset Tuner
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


# =============================================================================
# Main loop
# =============================================================================

def run(quest_ip: str, marker_id: int, marker_size_m: float):

    cam   = _CamFeedReceiver(quest_ip)
    xr    = _XRStateReceiver(quest_ip)
    aruco = _ArucoPoseEstimator(marker_id, marker_size_m)

    anchor = _WorldAnchor(quest_ip, pub_port=5005)
    haptic = _HapticPublisher(quest_ip, port=5007)
    tuner  = _OffsetTuner()

    synth = _SyntheticObjectPublisher(quest_ip, port=5006)
    # Edit these objects to match your scene (positions in marker-113 frame, metres)
    synth.add([0.10,  0.00, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[1.0, 0.2, 0.2])   # red   — 10 cm to the right
    synth.add([0.00,  0.10, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[0.2, 1.0, 0.2])   # green — 10 cm forward
    synth.add([-0.10, 0.00, 0.05], width=0.06, depth=0.06, height=0.10,
              color=[0.2, 0.4, 1.0])   # blue  — 10 cm to the left

    haptic_lock_sent = False

    vis3d = _FrustumVis(
        f"Quest Passthrough ArUco Test — world = marker #{marker_id}")
    vis3d.add_frustum("left_cam",   color=[0.2, 1.0, 0.3])   # green
    vis3d.add_frame  ("center_eye", size=0.12)
    vis3d.add_frame  ("ctrl_left",  size=0.06)
    vis3d.add_frame  ("ctrl_right", size=0.06)

    win = f"Quest Left Passthrough  [ENTER=re-lock  ESC=quit]  marker #{marker_id}"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, 960, 540)

    # ── World-transform broadcaster (for main_hand.py) ───────────────────────
    _wt_ctx = zmq.Context()
    world_T_pub = _wt_ctx.socket(zmq.PUB)
    world_T_pub.setsockopt(zmq.SNDHWM, 1)
    world_T_pub.setsockopt(zmq.LINGER, 0)
    world_T_pub.bind(ip_cfg.bind_local(ip_cfg.WORLD_TRANSFORM_PORT))
    print(f"[WorldT] PUB bound on port {ip_cfg.WORLD_TRANSFORM_PORT}")

    print(f"\n[Running]  quest_ip={quest_ip}  marker={marker_id}")
    print(f"  ENTER = force re-lock anchor from current detection")
    print(f"  ESC   = quit\n")

    try:
        while True:
            # ── Poll streams ──────────────────────────────────────────────────
            cam.poll(timeout_ms=5)
            xr.poll(timeout_ms=0)

            # ── ArUco detection on Quest passthrough image ────────────────────
            T_cam_marker_fresh = None
            det_vis = None

            if cam.frame is not None and cam.fx is not None:
                fx, fy, cx, cy = _adapt_cx_cy(
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    cam.sensor_width, cam.sensor_height,
                    cam.width, cam.height)
                det = aruco.detect(cam.frame, fx, fy, cx, cy, draw=True)
                det_vis = det["vis"]
                if det["found"]:
                    T_cam_marker_fresh = det["T_cam_marker"]

            # ── Anchor update ─────────────────────────────────────────────────
            tuner.draw()
            pos_off, yaw_off = tuner.get()
            anchor.set_offset(pos_off, yaw_off)

            if anchor.locked:
                anchor.publish()
                synth.publish()
                T_wt = anchor.T_world_tracking
                if T_wt is not None:
                    try:
                        world_T_pub.send_string(
                            json.dumps({"T": T_wt.flatten().tolist()}),
                            zmq.NOBLOCK)
                    except Exception:
                        pass
                if not haptic_lock_sent:
                    haptic.vibrate("both", amplitude=0.8,
                                   frequency=0.3, duration_ms=300)
                    haptic_lock_sent = True
                    print("[Haptic] Lock confirmation pulse sent")

            # ── Compute world-frame poses ─────────────────────────────────────
            T_world_camleft = anchor.world_T(cam.camera_T) \
                              if cam.camera_T is not None else None
            T_world_center  = anchor.world_T(xr.center_T) \
                              if xr.center_T is not None else None
            T_world_left    = anchor.world_T(xr.left_T) \
                              if xr.left_T is not None else None
            T_world_right   = anchor.world_T(xr.right_T) \
                              if xr.right_T is not None else None

            # ── Update Open3D ─────────────────────────────────────────────────
            if cam.fx is not None:
                fx, fy, cx, cy = _adapt_cx_cy(
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    cam.sensor_width, cam.sensor_height,
                    cam.width, cam.height)
                vis3d.update_frustum("left_cam", T_world_camleft,
                                     cam.width, cam.height, fx, fy, cx, cy)
            vis3d.update_frame("center_eye", T_world_center)
            vis3d.update_frame("ctrl_left",  T_world_left)
            vis3d.update_frame("ctrl_right", T_world_right)
            vis3d.tick()

            # ── OpenCV display ────────────────────────────────────────────────
            disp = cv.resize(
                det_vis if det_vis is not None else
                (cam.frame.copy() if cam.frame is not None
                 else np.zeros((480, 640, 3), dtype=np.uint8)),
                (960, 540))

            locked = anchor.locked
            det_ok = T_cam_marker_fresh is not None

            cv.putText(disp,
                       f"Marker #{marker_id}: {'DETECTED' if det_ok else 'searching...'}",
                       (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 255, 80) if det_ok else (0, 80, 255), 2)
            cv.putText(disp,
                       f"Anchor  : {'LOCKED — WorldRoot :5005 + synth :5006' if locked else 'waiting for first detection'}",
                       (12, 68), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                       (0, 255, 150) if locked else (100, 100, 100), 2)
            cv.putText(disp,
                       "ENTER = re-lock   ESC = quit",
                       (12, disp.shape[0] - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv.imshow(win, disp)

            key = cv.waitKey(1) & 0xFF
            if key == 27:    # ESC
                break
            elif key == 13:  # ENTER — lock immediately if marker is visible
                if T_cam_marker_fresh is not None and cam.camera_T is not None:
                    anchor.lock(T_cam_marker_fresh, cam.camera_T)
                    haptic_lock_sent = False  # re-trigger confirmation pulse
                else:
                    print("[ENTER] No detection this frame — point camera at marker")

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            world_T_pub.close(0)
        except Exception:
            pass
        vis3d.close()
        cv.destroyAllWindows()
        tuner.close()
        anchor.close()
        haptic.close()
        synth.close()
        xr.close()
        cam.close()
        print("[Done]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Quest passthrough ArUco → WorldRoot publisher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip",    default="192.168.50.201")
    ap.add_argument("--marker-id",   type=int,   default=113)
    ap.add_argument("--marker-size", type=float, default=0.100,
                    help="Marker side length in metres")
    args = ap.parse_args()
    run(quest_ip      = args.quest_ip,
        marker_id     = args.marker_id,
        marker_size_m = args.marker_size)


if __name__ == "__main__":
    main()
