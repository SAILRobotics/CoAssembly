"""Study 3: AR-visualized adaptive robot-to-human handover.

This standalone study runner preserves the Quest camera, hand tracking,
marker-100 world lock/relock, pegboard registration and object visualization
from ``main_with_robot.py``.  The experimental manipulation changes only the
visibility of the intended handover pose; robot target and trajectory policy
are shared by both conditions.

Workflow
--------
  1. Hold marker 100 visible within 1.0m — the world frame locks
     automatically the first time it's seen that close (no ENTER needed),
     or press ENTER while it's visible from any distance.
     PyBullet scene is placed at the locked pose immediately.
  2. Hold marker 101 visible → press ENTER to lock the pegboard pose.
     Marker 100 does NOT need to be visible at this step.
  3. Once locked, all later re-locks require an explicit trigger — press
     ENTER again (whichever marker is visible), or click the anchor-marker
     proximity relock cube. Auto-lock-on-sight only ever fires once, before
     the first lock.
Keys (OpenCV window must be focused)
--------------------------------------
  ENTER = lock / relock (handles marker 100 and/or 101 independently)
  ESC   = quit

Usage
-----
  python study3_handover_study.py --participant-id P01 --condition ghost_color
"""

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from utils.unity_conversion import (
    open3d_to_unity_vector,
    open3d_to_unity_quaternion,
)
from utils.pose_helpers import (
    _unity_pose_to_T, _adapt_cx_cy, _unity_to_o3d, _to_world,
    _palm_quat, _extract_joints,
)
import main_setting as cfg
from scene_viewer_o3d import SceneVis as _SceneVis

try:
    from robot_client import RobotClient
    _ROBOT_CTRL_AVAILABLE = True
except ImportError as _e:
    _ROBOT_CTRL_AVAILABLE = False
    print(f"[main_with_robot] RobotClient not available: {_e}")


# =============================================================================
# ZMQ receivers
# =============================================================================

class _CamFeedReceiver:
    def __init__(self, ip: str, port: int = cfg.CAM_FEED_PORT, topic: str = "cam_left"):
        ctx = zmq.Context()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._sub.connect(f"tcp://{ip}:{port}")
        self.frame        = None
        self.camera_T     = None
        self.raw_pos      = None   # (px, py, pz) straight from Unity, no axis/CV conversion
        self.raw_rot_xyzw = None   # (qx, qy, qz, qw) straight from Unity, no axis/CV conversion
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
                parts = self._sub.recv_multipart(flags=zmq.NOBLOCK)
                latest = parts
            except zmq.Again:
                break
        if latest is None or len(latest) != 9:
            return False
        parts = latest
        width,  = struct.unpack("<i",    parts[2])
        height, = struct.unpack("<i",    parts[3])
        px, py, pz      = struct.unpack("<fff",  parts[4])
        qx, qy, qz, qw = struct.unpack("<ffff", parts[5])
        fx, fy, cx, cy  = struct.unpack("<ffff", parts[6])
        sw, sh           = struct.unpack("<ii",   parts[7])
        arr   = np.frombuffer(parts[8], dtype=np.uint8)
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
        self.raw_pos      = (px, py, pz)
        self.raw_rot_xyzw = (qx, qy, qz, qw)
        return True

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass


class _ArUcoWorker:
    """Runs camera polling and ArUco detection on a background thread so the
    main loop is not blocked by frame capture or marker detection (~15-20 ms)."""

    def __init__(self, cam: "_CamFeedReceiver", aruco):
        self._cam   = cam
        self._aruco = aruco
        self._lock  = threading.Lock()
        self._T_cam_anchor   = None
        self._T_cam_pegboard = None
        self._det_vis        = None
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self._running:
            if not self._cam.poll(timeout_ms=10):
                continue
            frame = self._cam.frame
            if frame is None or self._cam.fx is None:
                continue
            fx, fy, cx, cy = _adapt_cx_cy(
                self._cam.fx, self._cam.fy, self._cam.cx, self._cam.cy,
                self._cam.sensor_width, self._cam.sensor_height,
                self._cam.width, self._cam.height)
            det = self._aruco.detect(frame, fx, fy, cx, cy, draw=True)
            with self._lock:
                self._T_cam_anchor   = det.get("T_cam_anchor")
                self._T_cam_pegboard = det.get("T_cam_pegboard")
                self._det_vis        = det["vis"] #the camera frame with ArUco detection overlays drawn on it

    def get(self):
        """Return anchor pose, pegboard pose, and annotated frame."""
        with self._lock:
            return self._T_cam_anchor, self._T_cam_pegboard, self._det_vis

    def stop(self) -> None:
        self._running = False

# =============================================================================
# ArUco pose estimator — marker 100 (anchor) + marker 101 (pegboard)
# =============================================================================


class _ArucoPoseEstimator:
    def __init__(self, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 dictionary=cv.aruco.DICT_6X6_1000):
        self.anchor_marker_id   = int(anchor_marker_id)
        self.pegboard_marker_id = int(pegboard_marker_id)
        self.anchor_marker_size   = float(anchor_marker_size_m)
        self.pegboard_marker_size = float(pegboard_marker_size_m)
        self._dict     = cv.aruco.getPredefinedDictionary(dictionary)
        self._detector = cv.aruco.ArucoDetector(
            self._dict, cv.aruco.DetectorParameters())

        def _obj_pts(size: float) -> np.ndarray:
            s = size / 2.0
            return np.array([[-s, s, 0.], [s, s, 0.],
                              [s, -s, 0.], [-s, -s, 0.]], dtype=np.float64)

        self._anchor_obj_pts   = _obj_pts(self.anchor_marker_size)
        self._pegboard_obj_pts = _obj_pts(self.pegboard_marker_size)

    def detect(self, bgr, fx, fy, cx, cy, dist=None, draw=True) -> dict:
        K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1)) if dist is None else np.array(dist).reshape(-1, 1)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        vis = bgr.copy()
        result = {"vis": vis, "T_cam_anchor": None, "T_cam_pegboard": None}
        if ids is None:
            return result
        if draw:
            cv.aruco.drawDetectedMarkers(vis, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid == self.anchor_marker_id:
                key, obj_pts, axis_len = "T_cam_anchor", self._anchor_obj_pts, self.anchor_marker_size
            elif mid == self.pegboard_marker_id:
                key, obj_pts, axis_len = "T_cam_pegboard", self._pegboard_obj_pts, self.pegboard_marker_size
            else:
                continue
            ok, rvec, tvec = cv.solvePnP(
                obj_pts, c.reshape(4, 2).astype(np.float64), K, dist,
                flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            Rm, _ = cv.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rm
            T[:3, 3]  = tvec.reshape(3)
            result[key] = T
            if draw:
                cv.drawFrameAxes(vis, K, dist, rvec, tvec, axis_len * 0.5, 2)
        return result


class _HandDataReceiver:
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
        self._max_hand_head_dist = 2.0
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
            return False
        return True

    @property
    def receiving(self) -> bool:
        return self.last_rx_time is not None and (time.time() - self.last_rx_time) < 2.0

    def center_eye_T(self) -> "np.ndarray | None":
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
        if self.data is None:
            return None, None
        hands = self.data.get("hands") or {}

        # Head position in world frame for proximity filtering
        head_world = None
        ce = (self.data.get("head") or {}).get("CenterEye")
        if ce is not None and T_world_tracking is not None:
            p = ce.get("position") or {}
            p_unity = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]])
            head_world = _to_world(p_unity, T_world_tracking)[0]

        def _resolve(key):
            j = _extract_joints(hands.get(key))
            if j is None:
                return None
            pts = (_to_world(j, T_world_tracking) if T_world_tracking is not None
                   else _unity_to_o3d(j))
            if (head_world is not None
                    and np.linalg.norm(pts[1] - head_world) > self._max_hand_head_dist):
                return None
            return pts

        return _resolve("LeftHand"), _resolve("RightHand")

    def pinch_strength(self, hand: str) -> "float | None":
        """Index-finger pinch strength (0-1) for hand ('LeftHand'/'RightHand'),
        as reported by Unity's Hand.GetFingerPinchStrength(HandFinger.Index)."""
        if self.data is None:
            return None
        hands = self.data.get("hands") or {}
        block = hands.get(hand)
        if not block:
            return None
        return block.get("indexPinchStrength")

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass



# =============================================================================
# World anchor
# =============================================================================

class _WorldAnchor:
    def __init__(self, pub_ip: str, pub_port: int = cfg.WORLD_ROOT_PORT,
                 pegboard_pub_port: int = cfg.PEGBOARD_ROOT_PORT):
        self._T_wt: np.ndarray | None = None  #  This gives you where the marker is in tracking space, 
                                              #  and inverting that gives you the transform that converts from tracking space into world space
        self._T_world_pegboard: np.ndarray | None = None
        self._T_eye_offset: np.ndarray | None = None
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{pub_ip}:{pub_port}")
        self._pub_pegboard = ctx.socket(zmq.PUB)
        self._pub_pegboard.connect(f"tcp://{pub_ip}:{pegboard_pub_port}")
        time.sleep(0.2)
        self._load_eye_offset()

    def _load_eye_offset(self):
        if not cfg.EYE_OFFSET_FILE.exists():
            return
        try:
            data = json.loads(cfg.EYE_OFFSET_FILE.read_text())
            self._T_eye_offset = np.array(data["T_eye_offset"],
                                          dtype=np.float64).reshape(4, 4)
            print(f"[Anchor] Eye offset loaded from {cfg.EYE_OFFSET_FILE.name}")
        except Exception as e:
            print(f"[Anchor] Eye offset load failed: {e}")

    def _effective_cam_T(self, cam_T: np.ndarray,
                         center_T: np.ndarray | None) -> np.ndarray | None:
        if self._T_eye_offset is not None and center_T is not None:
            return center_T @ self._T_eye_offset
        return cam_T

    def lock(self, T_cam_anchor: np.ndarray, cam_T: np.ndarray,
             require_locked: bool = False) -> bool:
        """Lock (or relock) world frame to marker 100. Returns True on success."""
        if T_cam_anchor is None or cam_T is None:
            return False
        if require_locked and not self.locked:
            return False
        self._T_wt = np.linalg.inv(cam_T @ T_cam_anchor)
        return True

    def lock_tracking_origin(self) -> bool:
        """Lock the world frame to the Quest tracking origin — no marker, no
        passthrough. The tracking origin is fixed at app start / recenter and is
        gravity-aligned, so world == tracking frame (``_T_wt = identity``) gives
        an upright scene. Used by the --no-passthrough manual 'l' lock."""
        self._T_wt = np.eye(4, dtype=np.float64)
        return True

    def set_pegboard(self, T_world_pegboard: np.ndarray) -> None:
        """Directly set the pegboard world transform (e.g. loaded from a file)."""
        self._T_world_pegboard = np.array(T_world_pegboard, dtype=np.float64)
        t = self._T_world_pegboard[:3, 3]
        print(f"[Anchor] Pegboard set from file: "
              f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")

    def update_pegboard_from_tracking(self, cam_T: np.ndarray,
                                      T_cam_pegboard: np.ndarray) -> bool:
        """Compute pegboard pose in world frame using live Quest tracking.

        Marker 100 does NOT need to be visible — uses the locked _T_wt instead.
        """
        if not self.locked or T_cam_pegboard is None or cam_T is None:
            return False
        self._T_world_pegboard = self._T_wt @ cam_T @ T_cam_pegboard
        t = self._T_world_pegboard[:3, 3]
        print(f"[Anchor] Pegboard updated: t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")
        return True

    @property
    def locked(self) -> bool:
        return self._T_wt is not None

    @property
    def T_world_tracking(self) -> np.ndarray | None:
        if self._T_wt is None:
            return None
        return self._T_wt

    @property
    def T_pegboard_in_world(self) -> np.ndarray | None:
        if self._T_world_pegboard is None:
            return None
        return self._T_world_pegboard

    def world_T(self, T_tracking_local: np.ndarray) -> np.ndarray | None: 
        #a general-purpose helper that converts any pose from Quest tracking space into world space.
        if self._T_wt is None:
            return None
        return self._T_wt @ T_tracking_local

    @staticmethod
    def _to_unity_pose(T_o3d: np.ndarray):
        """Convert a 4×4 Open3D transform to (pos_list, rot_xyzw, matrix_flat) in Unity frame."""
        q_xyzw       = ScipyR.from_matrix(T_o3d[:3, :3]).as_quat()
        q_u_wxyz     = open3d_to_unity_quaternion([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        t_unity      = open3d_to_unity_vector(T_o3d[:3, 3])
        q_unity_xyzw = [float(q_u_wxyz[1]), float(q_u_wxyz[2]),
                        float(q_u_wxyz[3]), float(q_u_wxyz[0])]
        T_unity      = np.eye(4, dtype=np.float64)
        T_unity[:3, :3] = ScipyR.from_quat(q_unity_xyzw).as_matrix()
        T_unity[:3, 3]  = t_unity
        return [float(v) for v in t_unity], q_unity_xyzw, T_unity.T.flatten().tolist()

    def _publish_T(self, T_o3d: np.ndarray, sock, label: str,
                   pos_key: str, rot_key: str, mat_key: str) -> bool:
        pos, rot, mat = self._to_unity_pose(T_o3d)
        try:
            sock.send_string(json.dumps({pos_key: pos, rot_key: rot, mat_key: mat}))
            return True
        except Exception as e:
            print(f"[{label}] Publish error: {e}")
            return False

    def publish(self) -> bool:
        if self._T_wt is None:
            return False
        return self._publish_T(np.linalg.inv(self._T_wt),
                               self._pub, "WorldRoot",
                               "world_root_position",
                               "world_root_rotation_xyzw",
                               "world_root_matrix")

    def publish_pegboard(self) -> bool:
        if self._T_world_pegboard is None:
            return False
        return self._publish_T(self._T_world_pegboard,
                               self._pub_pegboard, "PegboardRoot",
                               "pegboard_root_position",
                               "pegboard_root_rotation_xyzw",
                               "pegboard_root_matrix")

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
            q_xyzw = ScipyR.from_matrix(self.R_o3d).as_quat()
        else:
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
    def __init__(self, ip: str, port: int = cfg.SYNTH_OBJECTS_PORT):
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.2)
        self._objects: list[_SyntheticObject] = []
        self._names:   dict[str, _SyntheticObject] = {}
        print(f"[SynthObjects] Connected to tcp://{ip}:{port}")

    def add(self, centroid_o3d, width, depth, height,
            color=None, yaw_deg=0.0, R_o3d=None,
            name: str | None = None) -> "_SyntheticObject":
        obj = _SyntheticObject(len(self._objects), centroid_o3d,
                               width, depth, height,
                               yaw_deg=yaw_deg, R_o3d=R_o3d, color=color)
        self._objects.append(obj)
        if name is not None:
            self._names[name] = obj
        return obj

    def get(self, name: str) -> "_SyntheticObject | None":
        return self._names.get(name)

    def publish(self, objects: "list[_SyntheticObject] | None" = None):
        selected = self._objects if objects is None else objects
        payload = {"objects": [o.to_unity_dict() for o in selected]}
        try:
            self._pub.send_string(json.dumps(payload))
        except Exception as e:
            print(f"[SynthObjects] Publish error: {e}")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


# Tool layout manager
# =============================================================================

class _ToolLayoutManager:
    """Loads tool_layout.json once at startup and publishes world-space tool
    definitions to Unity (port 5011). To apply changes, restart the script."""

    PORT = cfg.TOOL_LAYOUT_PORT

    def __init__(self, json_path: str, ip: str):
        self._tools: list = []
        # Confirmed releases disappear from the study's object visualization.
        self._delivered_ids: set[int] = set()
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{self.PORT}")
        time.sleep(0.2)
        try:
            data = json.loads(Path(json_path).read_text())
            self._tools = data.get("tools", [])
            print(f"[ToolLayout] Loaded {len(self._tools)} tool(s) from {Path(json_path).name}")
        except Exception as e:
            print(f"[ToolLayout] Failed to load JSON: {e}")

    # ── Publishing ───────────────────────────────────────────────────────────

    def _tool_box_world(self, t: dict, T: np.ndarray) -> tuple:
        """(centroid, R_world, size) in the Open3D world frame — exactly the geometry Unity receives
        (after the o3d→Unity conversion) and that the Open3D scene draws directly. Orientation is
        world-frame yaw only, Rz(rot[2]); the pegboard tilt T[:3,:3] is deliberately NOT composed
        in. T[:3,:3] carries a ~90° from the ArUco/camera pose chain, so composing it (as the robot
        path's _tool_world_data does) rotates the wireframe box 90° relative to the Unity tools."""
        sz  = t.get("size", [0.05, 0.05, 0.05])
        rot = t.get("rotation_deg", [0.0, 0.0, 0.0])
        # peg_pos is the base in pegboard frame (new format); fall back to world_pos.
        R_world = ScipyR.from_euler('z', float(rot[2]), degrees=True).as_matrix()
        base_w  = ((T @ np.append(t["peg_pos"], 1.0))[:3] if "peg_pos" in t
                   else np.array(t.get("world_pos", [0.0, 0.0, 0.0])))
        # Unity prefabs are centred at their local origin → return the centroid.
        centroid = base_w + R_world @ np.array([0.0, 0.0, sz[2] / 2.0])
        return centroid, R_world, sz

    def publish(self, T: np.ndarray) -> None:
        out = []
        for t in self._tools:
            if int(t["id"]) in self._delivered_ids:
                continue
            pos_w, R_world, sz = self._tool_box_world(t, T)
            q_xyzw  = ScipyR.from_matrix(R_world).as_quat()

            pos_u  = open3d_to_unity_vector(pos_w)
            q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            q_u    = open3d_to_unity_quaternion(q_wxyz)
            sz_u   = open3d_to_unity_vector(np.array(sz, dtype=float))

            out.append({
                "id":            int(t["id"]),
                "type":          t.get("type", "unknown"),
                "category":      t.get("category", "tool"),
                "position":      pos_u.tolist(),
                "rotation_xyzw": [float(q_u[1]), float(q_u[2]),
                                   float(q_u[3]), float(q_u[0])],
                "size":          sz_u.tolist(),
            })
        try:
            self._pub.send_string(json.dumps({"tools": out}))
        except Exception as e:
            print(f"[ToolLayout] Publish error: {e}")

    def mark_delivered(self, tool_id: int) -> None:
        """Exclude an object from subsequent Unity layout publications."""
        self._delivered_ids.add(int(tool_id))

    def _tool_world_data(self, t: dict, T: np.ndarray) -> tuple:
        sz      = t.get("size", [0.05, 0.05, 0.05])
        rot     = t.get("rotation_deg", [0.0, 0.0, 0.0])
        R_local = ScipyR.from_euler('xyz', rot, degrees=True).as_matrix()
        R_world = T[:3, :3] @ R_local
        base_w  = ((T @ np.append(t["peg_pos"], 1.0))[:3] if "peg_pos" in t
                   else np.array(t.get("world_pos", [0.0, 0.0, 0.0])))
        centroid = base_w + R_local @ np.array([0.0, 0.0, sz[2] / 2.0])
        return centroid, R_world, sz

    def world_boxes(self, T: np.ndarray) -> list:
        # Open3D wireframe boxes use the SAME geometry Unity gets (yaw-only orientation), so the
        # scene matches the Unity tools. NOTE: this intentionally differs from _tool_world_data /
        # get_world_data below, which compose T[:3,:3] for the robot grasp path.
        return [self._tool_box_world(t, T) for t in self._tools
                if int(t["id"]) not in self._delivered_ids]

    def get_world_data(self, tool_id: int, T: np.ndarray) -> "tuple | None":
        """Return (centroid_world, R_world, size) for tool_id, or None if not found."""
        for t in self._tools:
            if t["id"] == tool_id:
                return self._tool_world_data(t, T)
        return None

    def get_grasp_joints(self, tool_id: int) -> "list | None":
        """Return pre-recorded grasp_joints for tool_id, or None."""
        for t in self._tools:
            if t["id"] == tool_id:
                return t.get("grasp_joints")
        return None

    def get_category(self, tool_id: int) -> str:
        """Return the category string ("tool" or "part") for tool_id."""
        for t in self._tools:
            if t["id"] == tool_id:
                return t.get("category", "tool")
        return "tool"

    def get_name(self, tool_id: int) -> str:
        """Return the type/name string for tool_id (e.g. 'GEAR_ROD_ROW1')."""
        for t in self._tools:
            if t["id"] == tool_id:
                return t.get("type", f"id={tool_id}")
        return f"id={tool_id}"

    def close(self) -> None:
        try:
            self._pub.close(0)
        except Exception:
            pass

class _ObjectColorPublisher:
    """Publish only experiment-state colors; no hover/click interaction."""

    RESTING_PART = [1.0, 0.78, 0.78, 0.15]
    RESTING_TOOL = [0.80, 0.88, 1.0, 0.15]
    APPROACHING = [1.0, 0.5, 0.0, 0.30]
    GRASPED = [0.0, 1.0, 0.0, 0.30]
    GHOST_VISIBLE = [0.20, 0.90, 0.40, 0.25]
    GHOST_FIXED = [0.0, 0.0, 0.0, 0.25]  # C2 (ghost_no_color): no gradient
    GHOST_PROPOSAL = [1.00, 0.50, 0.00, 0.30]
    GHOST_INVALID = [1.00, 0.10, 0.10, 0.35]
    GHOST_HIDDEN = [0.20, 0.90, 0.40, 0.0]
    HOVER = [1.0, 0.5, 0.0, 0.15]
    SELECTED = [0.0, 1.0, 0.0, 0.15]
    RESET = [-1.0, -1.0, -1.0, -1.0]

    # Continuous proximity gradient (Study 3 conditions C3/C4): red = far /
    # outside workspace / IK-or-motion failure, ramping through orange to
    # green as the robot's live TCP nears the handover target.
    _PROXIMITY_RED    = [1.00, 0.10, 0.10]
    _PROXIMITY_ORANGE = [1.00, 0.50, 0.00]
    _PROXIMITY_GREEN  = [0.20, 0.90, 0.40]
    _PROXIMITY_ALPHA  = 0.30

    @classmethod
    def proximity_color(cls, closeness: float) -> list:
        """closeness in [0, 1]: 0 = far/invalid (red), 1 = arrived (green)."""
        closeness = float(np.clip(closeness, 0.0, 1.0))
        lo, hi, t = ((cls._PROXIMITY_RED, cls._PROXIMITY_ORANGE, closeness / 0.5)
                     if closeness <= 0.5 else
                     (cls._PROXIMITY_ORANGE, cls._PROXIMITY_GREEN,
                      (closeness - 0.5) / 0.5))
        rgb = [a + (b - a) * t for a, b in zip(lo, hi)]
        return rgb + [cls._PROXIMITY_ALPHA]

    def __init__(self, quest_ip: str, port: int = cfg.TOOL_COLOR_PORT):
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{port}")
        self._colors: dict[int, list[float]] = {}
        self._last_refresh = 0.0
        time.sleep(0.2)

    def publish(self, object_id: int, color) -> None:
        self._colors[int(object_id)] = [float(channel) for channel in color]
        self._pub.send_string(json.dumps({
            "tool_id": int(object_id),
            "color": self._colors[int(object_id)],
        }))

    def refresh(self, interval_s: float = 1.0) -> None:
        now = time.monotonic()
        if now - self._last_refresh < interval_s:
            return
        self._last_refresh = now
        for object_id, color in self._colors.items():
            self._pub.send_string(json.dumps({
                "tool_id": object_id, "color": color}))

    def close(self) -> None:
        self._pub.close(0)


class _StudyInteractionReceiver:
    """Receive robot-stop clicks and marker-100 hover feedback from Unity."""

    ROBOT_TOOL_ID = 200

    def __init__(self, quest_ip: str, port: int = cfg.TOOL_CLICK_PORT):
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{quest_ip}:{port}")
        print(f"[StudyInteraction] SUB → tcp://{quest_ip}:{port}")

    def poll(self) -> list[dict]:
        events = []
        while True:
            try:
                message = json.loads(
                    self._sub.recv_string(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except (json.JSONDecodeError, TypeError):
                continue
            try:
                events.append({
                    "tool_id": int(message.get("tool_id", -1)),
                    "event_type": message.get("event_type", "selected"),
                })
            except (TypeError, ValueError):
                continue
        return events

    def close(self) -> None:
        self._sub.close(0)


class _WorkspaceBoundPublisher:
    """Drive WorkSpaceBoundReceiver on Unity port 5015."""

    def __init__(self, quest_ip: str, port: int = cfg.WORKSPACE_BOUND_PORT):
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{port}")

    @staticmethod
    def distance_outside(position, lower: np.ndarray, upper: np.ndarray) -> float:
        if position is None:
            return 0.0
        delta = np.maximum(np.maximum(lower - position, position - upper), 0.0)
        return float(np.linalg.norm(delta))

    def publish(self, lower: np.ndarray, upper: np.ndarray,
                distance_outside: float) -> None:
        self._pub.send_string(json.dumps({
            "bounds_lo": open3d_to_unity_vector(lower).tolist(),
            "bounds_hi": open3d_to_unity_vector(upper).tolist(),
            "dist_outside": float(distance_outside),
        }))

    def close(self) -> None:
        self._pub.close(0)

class MainScene:
    """Minimal Study 3 runtime: perceive, grasp, stage, and hand over."""

    _HANDOVER_SIDE = "right"
    # Match main_with_robot.py's right-hand handover distance. This also keeps
    # the gripper collision model clear of the palm-centered CBF sphere.
    _PALM_TCP_STANDOFF_M = 0.285
    _PALM_CBF_RADIUS_M = 0.05       # 10 cm diameter
    _PALM_CBF_CLEARANCE_M = 0.01
    _PALM_CBF_PUBLISH_INTERVAL_S = 1.0 / 30.0
    _SYNTH_INTERVAL = 0.10
    # Match main_with_robot.py: first sight of marker 100 within normal Quest
    # working range locks the world automatically. Subsequent relocks remain
    # explicit (ENTER or the Unity marker interaction).
    _AUTO_LOCK_MAX_DIST = 1.0
    _RELOCK_MAX_DIST_M = 1.0
    _RELOCK_COOLDOWN_S = 2.0
    # 2x2 design: ghost gripper preview (shown/hidden) x color coding
    # (static/none vs continuous red->orange->green proximity gradient).
    # Position updates are continuous hand-tracking in all four conditions.
    C1 = "no_ghost_no_color"        # nothing shown
    C2 = "ghost_no_color"           # ghost visible, fixed color
    C3 = "no_ghost_robot_color"     # ghost hidden, real gripper proxy tinted
    C4 = "ghost_color"              # ghost visible, proximity-tinted
    ROBOT_GRIPPER_TOOL_ID = 201
    # Ghost = destination preview; not shown once the robot has actually
    # arrived (waiting_for_pull) since the real gripper is right there.
    _GHOST_ACTIVE_STATES = frozenset({
        "grasped", "retracting", "staging", "approaching_hand"})
    # Robot-gripper proxy tracks the live TCP, so it stays meaningful (and
    # green when arrived) through the pull-wait phase too.
    _ROBOT_GRIPPER_ACTIVE_STATES = frozenset({
        "grasped", "retracting", "staging", "approaching_hand",
        "waiting_for_pull"})
    _PROXIMITY_FAR_M = 0.6
    _PROXIMITY_NEAR_M = 0.03
    WORKSPACE_LO = np.asarray(cfg.WORKSPACE_LO, dtype=float)
    WORKSPACE_HI = np.asarray(cfg.WORKSPACE_HI, dtype=float)
    # Match the robot server's inset workspace exactly. A Study 3 hand target
    # must already lie inside this region; it is never projected to a boundary.
    _TARGET_WORKSPACE_LO = (
        WORKSPACE_LO + float(cfg.ROBOT_TARGET_WORKSPACE_MARGIN_M))
    _TARGET_WORKSPACE_HI = (
        WORKSPACE_HI - float(cfg.ROBOT_TARGET_WORKSPACE_MARGIN_M))
    # Four right-side gear stands followed by all six gears. Keeping a fixed
    # list makes every participant receive the same ten physical trials.
    STUDY_OBJECT_IDS = (19, 21, 23, 25, 26, 27, 28, 29, 30, 31)

    def __init__(self, quest_ip: str, anchor_marker_id: int,
                 pegboard_marker_id: int, anchor_marker_size_m: float,
                 pegboard_marker_size_m: float, hand_port: int,
                 simulation: bool, use_calibrated_robot_base: bool,
                 load_pegboard_from_file: bool, no_passthrough: bool,
                 condition: str, participant_id: str) -> None:
        self.anchor_marker_id = anchor_marker_id
        self.pegboard_marker_id = pegboard_marker_id
        self.simulation = simulation
        self.condition = condition
        self.participant_id = participant_id
        self._load_pegboard_from_file = load_pegboard_from_file
        self._no_passthrough = no_passthrough

        self.cam = _CamFeedReceiver(quest_ip)
        aruco = _ArucoPoseEstimator(
            anchor_marker_id, pegboard_marker_id,
            anchor_marker_size_m, pegboard_marker_size_m)
        self.aruco_worker = _ArUcoWorker(self.cam, aruco)
        self.hands = _HandDataReceiver(quest_ip, hand_port)
        self.anchor = _WorldAnchor(quest_ip)
        self.colors = _ObjectColorPublisher(quest_ip)
        self.study_interactions = _StudyInteractionReceiver(quest_ip)
        self.tool_layout = _ToolLayoutManager(
            cfg.SCENE_LAYOUT_DIR / "tool_layout1.json", quest_ip)
        self.synth = _SyntheticObjectPublisher(quest_ip)
        # Unity SyntheticObjectReceiver.objects[0] is
        # WorldRoot/RobotiqGripperWithAdapters on the study machine.
        self._unity_ghost = self.synth.add(
            [0.0, 0.0, 0.0], width=0.05, depth=0.05, height=0.05,
            name="handover_ghost")
        # objects[1]: a second RobotiqGripperWithAdapters instance under
        # WorldRoot, tracking the live TCP pose (not the hand target) so it
        # can stand in for "the actual robot gripper" in C3, tinted via
        # ToolColorReceiver toolId=ROBOT_GRIPPER_TOOL_ID (201).
        self._unity_robot_gripper = self.synth.add(
            [0.0, 0.0, 0.0], width=0.05, depth=0.05, height=0.05,
            name="robot_gripper_proxy")
        self.workspace = _WorkspaceBoundPublisher(quest_ip)
        self.vis = _SceneVis(
            f"Study 3 Handover — marker #{anchor_marker_id}")
        # Keep the controller's configured Cartesian workspace visible in the
        # experimenter Open3D view for the entire study.
        self.vis.update_workspace_bound(self.WORKSPACE_LO, self.WORKSPACE_HI)

        self.robot: "RobotClient | None" = None
        self.pb_scene = None
        if _ROBOT_CTRL_AVAILABLE:
            try:
                self.robot = RobotClient(
                    simulation=simulation,
                    use_calibrated_robot_base=use_calibrated_robot_base)
                self.pb_scene = self.robot.pb_scene
            except Exception as error:
                print(f"[Study3] RobotClient unavailable: {error}")

        self._win = (f"Study 3  [ENTER=lock/relock  ESC=quit]  "
                     f"anchor=#{anchor_marker_id} pegboard=#{pegboard_marker_id}")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)

        self._T_world_tcp: "np.ndarray | None" = None
        self._tcp_target_T: "np.ndarray | None" = None
        self._ghost_target_T: "np.ndarray | None" = None
        self._ghost_invalid_reason: "str | None" = None
        self._retry_handover_after = 0.0
        self._robot_state: "str | None" = None
        self._handover_tool_id: "int | None" = None
        self._pending_handover = False
        self._last_right_pts = None
        self._stable_hand_quat = {"right": None}
        self._trial = 0
        self._trial_started_at: "float | None" = None
        self._last_synth_pub_time = 0.0
        self._last_palm_cbf_pub_time = 0.0
        self._palm_cbf_active = False
        self._robot_stop_requested = False
        self._resume_hand_tracking_on_next_robot_click = False
        self._c1_arm_pull_after_stop = False
        self._c1_resume_after_pull_cancel = False
        self._marker_relock_available = False
        self._marker_relock_green_until = 0.0
        self._last_marker_relock_time = 0.0
        self._last_pegboard_T: "np.ndarray | None" = None
        self._trial_plan = []
        for object_id in self.STUDY_OBJECT_IDS:
            name = self.tool_layout.get_name(object_id)
            joints = self.tool_layout.get_grasp_joints(object_id)
            if joints is None:
                raise ValueError(
                    f"Study object {object_id} ({name}) has no recorded grasp joints")
            self._trial_plan.append((object_id, name))
        self._next_trial_index = 0
        for item in self.tool_layout._tools:
            resting = (_ObjectColorPublisher.RESTING_PART
                       if item.get("category") == "part"
                       else _ObjectColorPublisher.RESTING_TOOL)
            self.colors.publish(int(item["id"]), resting)
        self.colors.publish(200, _ObjectColorPublisher.GHOST_HIDDEN)
        self.colors.publish(
            self.ROBOT_GRIPPER_TOOL_ID, _ObjectColorPublisher.GHOST_HIDDEN)

        if simulation and load_pegboard_from_file:
            self._try_load_pegboard_from_file()
        print(f"[Study3] participant={participant_id} condition={condition}")
        print("[Study3] SPACE starts each trial; 4 right stands followed by "
              "all 6 gears")

    def _log(self, event: str, **values) -> None:
        """Compatibility hook for study events; Study 3 is not recorded."""

    def _replay_record(self, record_type: str, **payload) -> None:
        """Compatibility hook; replay recording is disabled for Study 3."""

    def _log_replay_frame(self, head_T, left_pts, right_pts,
                          robot_link_poses) -> None:
        """Replay-frame recording is disabled for the subjective study."""

    def _try_load_pegboard_from_file(self) -> bool:
        path = cfg.SCENE_LAYOUT_DIR / "T_world10_pegboard101.npz"
        if not path.exists():
            print(f"[Pegboard] Saved pose not found: {path}")
            return False
        try:
            data = np.load(path)
            self.anchor.set_pegboard(data["T_world10_pegboard"])
            self.vis.set_pegboard_outline(
                offset_x=float(data["marker_offset_right_m"]),
                offset_y=float(data["marker_offset_top_m"]),
                width=float(data["pegboard_width_m"]),
                height=float(data["pegboard_height_m"]))
            self._publish_pegboard_objects()
            return True
        except Exception as error:
            print(f"[Pegboard] Could not load saved pose: {error}")
            return False

    def _publish_pegboard_objects(self) -> None:
        T = self.anchor.T_pegboard_in_world
        if T is None:
            return
        self.tool_layout.publish(T)
        self.vis.update_tool_boxes(self.tool_layout.world_boxes(T))
        self._last_pegboard_T = T.copy()

    def _stabilize_quaternion(self, quaternion) -> np.ndarray:
        base = ScipyR.from_quat(quaternion)
        flipped = base * ScipyR.from_euler("z", 180, degrees=True)
        previous = self._stable_hand_quat["right"]
        if previous is None:
            chosen = flipped if base.apply([0., 1., 0.])[2] > 0 else base
        else:
            prior = ScipyR.from_quat(previous)
            chosen = (base if (prior.inv() * base).magnitude()
                      <= (prior.inv() * flipped).magnitude() else flipped)
        result = chosen.as_quat()
        self._stable_hand_quat["right"] = result
        return result

    def _handover_pose(self, hand_pts):
        if hand_pts is None or self._T_world_tcp is None:
            return None
        quaternion = self._stabilize_quaternion(
            _palm_quat(hand_pts, is_left=False))
        centroid = (np.asarray(hand_pts[3], float)
                    + np.asarray(hand_pts[1], float)
                    + np.asarray(hand_pts[6], float)) / 3.0
        raw_position = (
            centroid - ScipyR.from_quat(quaternion).apply([0., 0., 1.])
            * self._PALM_TCP_STANDOFF_M)
        inside_workspace = bool(
            np.all(raw_position >= self._TARGET_WORKSPACE_LO)
            and np.all(raw_position <= self._TARGET_WORKSPACE_HI))
        command_position = raw_position.copy()
        rotation = ScipyR.from_quat(quaternion).as_matrix()
        command_T = np.eye(4)
        command_T[:3, :3] = rotation
        command_T[:3, 3] = command_position
        ghost_T = np.eye(4)
        ghost_T[:3, :3] = rotation
        ghost_T[:3, 3] = raw_position
        return (command_position, quaternion, command_T, ghost_T,
                inside_workspace)

    def _set_ghost_validity(self, reason: "str | None") -> None:
        reason_changed = reason != self._ghost_invalid_reason
        self._ghost_invalid_reason = reason
        if reason and reason_changed:
            print(f"[Study3] Handover preview invalid: {reason}")
            self._log("handover_preview_invalid", success=0)

    def _proximity_closeness(self) -> float:
        """0 = far, outside workspace, or IK/motion failure (red); 1 = the
        live TCP has arrived at the handover target (green)."""
        if self._ghost_invalid_reason is not None:
            return 0.0
        if self._T_world_tcp is None or self._ghost_target_T is None:
            return 0.0
        distance = float(np.linalg.norm(
            self._T_world_tcp[:3, 3] - self._ghost_target_T[:3, 3]))
        span = self._PROXIMITY_FAR_M - self._PROXIMITY_NEAR_M
        return float(np.clip(
            (self._PROXIMITY_FAR_M - distance) / span, 0.0, 1.0))

    def _publish_condition_colors(self) -> None:
        """Drive the toolId=200 (ghost) / 201 (robot-gripper proxy) colors
        each tick per the active condition's manipulation."""
        if self.condition == self.C2:
            self.colors.publish(200, _ObjectColorPublisher.GHOST_FIXED)
        elif self.condition == self.C4:
            self.colors.publish(
                200, _ObjectColorPublisher.proximity_color(
                    self._proximity_closeness()))
        elif self.condition == self.C3:
            self.colors.publish(
                self.ROBOT_GRIPPER_TOOL_ID,
                _ObjectColorPublisher.proximity_color(
                    self._proximity_closeness()))

    def _on_grasp_complete(self, ok: bool, tool_id: int) -> None:
        self._robot_state = None
        if self._robot_stop_requested:
            self._pending_handover = False
            self._robot_stop_requested = False
            return
        if not ok or self.robot is None:
            self._log("grasp_failed", success=0)
            self.colors.publish(tool_id, _ObjectColorPublisher.RESTING_PART)
            self._handover_tool_id = None
            return
        self._handover_tool_id = tool_id
        self.colors.publish(tool_id, _ObjectColorPublisher.GRASPED)
        self._log("grasp_complete", success=1)
        # robot_control_server.execute_grasp() already finishes its retract at
        # the single canonical handover staging configuration. Starting a
        # second, different moveJ here caused an extra waypoint and could race
        # the asynchronous grasp program ("another thread is controlling the
        # robot"). Begin live hand-target tracking directly.
        self._log("handover_staging_complete", success=1)
        self._pending_handover = True

    def _on_grasp_phase(self, phase: str) -> None:
        self._robot_state = phase
        if phase != "grasped" or self.condition == self.C1:
            return
        # Preview conditions reveal the handover target at grasp success,
        # rather than waiting for retract and staging to finish.
        pose = self._handover_pose(self._last_right_pts)
        if pose is None:
            return
        _position, _quaternion, _target_T, ghost_T, inside = pose
        self._ghost_target_T = ghost_T
        self._set_ghost_validity(None if inside else "outside_workspace")
        self._log("handover_preview_shown_after_grasp")

    def _start_handover(self, hand_pts) -> bool:
        if self.robot is None:
            return False
        pose = self._handover_pose(hand_pts)
        if pose is None:
            return False
        position, quaternion, target_T, ghost_T, inside_workspace = pose
        self._ghost_target_T = ghost_T
        if not inside_workspace:
            self._set_ghost_validity("outside_workspace")
            self._tcp_target_T = None
            return False
        self._set_ghost_validity(None)
        self._trial_started_at = time.perf_counter()
        self._tcp_target_T = target_T
        self._robot_state = "approaching_hand"
        self._log("handover_motion_started")
        self.robot.move_to_pose(
            position, quaternion.tolist(), motion_profile="handover",
            on_complete=self._on_handover_arrival)
        return True

    def _on_handover_arrival(self, ok: bool) -> None:
        distance = ""
        if self._last_right_pts is not None and self._tcp_target_T is not None:
            distance = f"{np.linalg.norm(np.asarray(self._last_right_pts[1], float) - self._tcp_target_T[:3, 3]):.6f}"
        elapsed = (f"{time.perf_counter() - self._trial_started_at:.6f}"
                   if self._trial_started_at is not None else "")
        self._log("robot_arrival" if ok else "robot_arrival_failed",
                  elapsed_s=elapsed, hand_to_target_m=distance,
                  success=int(bool(ok)))
        self._robot_state = None
        if self._c1_arm_pull_after_stop:
            # With no ghost to click (C1/C3), a robot click is also an early
            # handover commit: freeze at the last safe controller target,
            # then let a physical pull open the gripper and complete the trial.
            self._c1_arm_pull_after_stop = False
            self._robot_stop_requested = False
            self._pending_handover = False
            if self.robot is not None and self._handover_tool_id is not None:
                self._robot_state = "waiting_for_pull"
                self._log("robot_stop_pull_armed", success=1)
                print("[Study3] Robot stopped — pull the object to accept it")
                self.robot.wait_for_handover_pull(on_complete=self._on_release)
            return
        if self._robot_stop_requested:
            self._pending_handover = False
            self.colors.publish(200, _ObjectColorPublisher.GHOST_HIDDEN)
            self.colors.publish(
                self.ROBOT_GRIPPER_TOOL_ID, _ObjectColorPublisher.GHOST_HIDDEN)
            self._robot_stop_requested = False
            return
        if ok and self.robot is not None:
            self._set_ghost_validity(None)
            self._robot_state = "waiting_for_pull"
            self.robot.wait_for_handover_pull(on_complete=self._on_release)
        elif self._handover_tool_id is not None:
            # A failed move includes an unreachable pose or stalled IK on the
            # robot server. Immediately retry from the latest live hand pose;
            # invalid/out-of-workspace poses remain pending and uncommanded.
            self._set_ghost_validity("ik_or_motion_failure")
            self._retry_handover_after = time.perf_counter()
            self._pending_handover = True

    def _on_release(self, ok: bool) -> None:
        elapsed = (f"{time.perf_counter() - self._trial_started_at:.6f}"
                   if self._trial_started_at is not None else "")
        self._log("transfer_complete" if ok else "transfer_failed",
                  elapsed_s=elapsed, success=int(bool(ok)))
        if ok:
            self._resume_hand_tracking_on_next_robot_click = False
        if self._c1_resume_after_pull_cancel:
            self._c1_resume_after_pull_cancel = False
            self._robot_stop_requested = False
            self._resume_hand_tracking_on_next_robot_click = False
            if not self._start_handover(self._last_right_pts):
                self._pending_handover = True
            return
        if self._robot_stop_requested:
            self._pending_handover = False
            self._robot_state = None
            self._robot_stop_requested = False
            return
        if ok and self._handover_tool_id is not None:
            self.tool_layout.mark_delivered(self._handover_tool_id)
            self._publish_pegboard_objects()
            self._handover_tool_id = None
            self._tcp_target_T = None
            self._ghost_target_T = None
            if self.robot is not None:
                self._robot_state = "returning_default"
                self._log("return_to_default_started")
                self.robot.move_to_joints(
                    cfg.ROBOT_DEFAULT_JOINT_DEG, degrees=True,
                    on_complete=self._on_default_return_complete)
                return
        elif self._handover_tool_id is not None:
            self._pending_handover = True
        self._robot_state = None

    def _on_default_return_complete(self, ok: bool) -> None:
        stopped_by_click = self._robot_stop_requested
        self._robot_stop_requested = False
        self._log("return_to_default_complete" if ok
                  else "return_to_default_failed", success=int(bool(ok)))
        self._robot_state = None
        if stopped_by_click:
            print("[Study3] Return to default was stopped by robot click")
            return
        print(f"[Study3] Return to default pose "
              f"{'complete' if ok else 'failed'}; next trial is ready")

    def _start_next_trial(self) -> None:
        if self.robot is None or self._robot_state is not None:
            print("[Study3] Robot is unavailable or still busy")
            return
        if (not self.robot.connected or self.robot.move_running
                or self.robot.tool_grasp_running):
            print("[Study3] Robot controller is disconnected or still moving")
            return
        if not self.simulation and not self.anchor.locked:
            print(f"[Study3] Lock world marker {self.anchor_marker_id} before "
                  "starting a real-robot trial")
            return
        if self.anchor.T_pegboard_in_world is None:
            print("[Study3] Lock or load the pegboard before starting a trial")
            return
        if self._next_trial_index >= len(self._trial_plan):
            print("[Study3] All 10 trials are complete")
            return
        tool_id, name = self._trial_plan[self._next_trial_index]
        joints = self.tool_layout.get_grasp_joints(tool_id)
        self._next_trial_index += 1
        self._handover_tool_id = tool_id
        self._trial = self._next_trial_index
        self._robot_state = "grasping"
        self.colors.publish(tool_id, _ObjectColorPublisher.APPROACHING)
        self._log("trial_started")
        print(f"[Study3] Trial {self._trial}/10: {name} (id={tool_id})")
        self.robot.execute_grasp(
            joints, category=self.tool_layout.get_category(tool_id),
            tool_type=self.tool_layout.get_name(tool_id),
            board_normal=self.anchor.T_pegboard_in_world[:3, 2],
            on_phase=self._on_grasp_phase,
            on_complete=lambda ok, tid=tool_id: self._on_grasp_complete(ok, tid))

    def run(self) -> None:
        try:
            while True:
                self.colors.refresh()
                anchor_clicked = False
                for interaction in self.study_interactions.poll():
                    tool_id = interaction["tool_id"]
                    event_type = interaction["event_type"]
                    self._replay_record(
                        "unity_interaction", tool_id_clicked=tool_id,
                        event_type=event_type)
                    if (tool_id == self.anchor_marker_id
                            and event_type == "selected"):
                        anchor_clicked = True
                    if (tool_id == _StudyInteractionReceiver.ROBOT_TOOL_ID
                            and event_type == "selected"):
                        if (self.condition in (self.C1, self.C3)
                                and self._robot_state == "waiting_for_pull"
                                and self._resume_hand_tracking_on_next_robot_click):
                            # A second click before the participant pulls backs
                            # out of the frozen offer and resumes tracking.
                            self._c1_resume_after_pull_cancel = True
                            self._robot_state = "resuming_hand_tracking"
                            self._log("robot_clicked_controller_resume")
                            print("[Study3] ROBOT CLICKED AGAIN — cancelling "
                                  "pull wait and resuming right-hand tracking")
                            if self.robot is not None:
                                self.robot.cancel_motion()
                            continue
                        robot_was_active = bool(
                            self._robot_state is not None
                            or (self.robot is not None
                                and (self.robot.move_running
                                     or self.robot.tool_grasp_running)))
                        if robot_was_active:
                            was_tracking_hand = (
                                self._robot_state == "approaching_hand"
                                or self._pending_handover)
                            self._robot_stop_requested = True
                            self._resume_hand_tracking_on_next_robot_click = (
                                was_tracking_hand)
                            self._c1_arm_pull_after_stop = bool(
                                self.condition in (self.C1, self.C3)
                                and was_tracking_hand
                                and self._handover_tool_id is not None)
                            self._pending_handover = False
                            self._robot_state = None
                            self._tcp_target_T = None
                            self._ghost_target_T = None
                            self.colors.publish(
                                200, _ObjectColorPublisher.GHOST_HIDDEN)
                            self.colors.publish(
                                self.ROBOT_GRIPPER_TOOL_ID,
                                _ObjectColorPublisher.GHOST_HIDDEN)
                            self._log(
                                "robot_clicked_controller_stop", success=0)
                            print("[Study3] ROBOT CLICKED — controller motion "
                                  "stopped")
                            if self.robot is not None:
                                # Stop the controller's current motion; keep
                                # Study 3 alive and do not run cancel_grasp.
                                if self._c1_arm_pull_after_stop:
                                    self.robot.cancel_move()
                                else:
                                    self.robot.cancel_motion()
                        elif self._resume_hand_tracking_on_next_robot_click:
                            if self.robot is None or not self.robot.connected:
                                print("[Study3] Cannot resume hand tracking; robot "
                                      "controller is disconnected")
                            else:
                                self._resume_hand_tracking_on_next_robot_click = False
                                print("[Study3] ROBOT CLICKED AGAIN — resuming "
                                      "right-hand tracking")
                                if not self._start_handover(
                                        self._last_right_pts):
                                    # No valid hand pose yet: resume as soon as
                                    # right-hand tracking becomes available.
                                    self._pending_handover = True
                        else:
                            print("[Study3] Robot is already idle; press SPACE "
                                  "to start the next trial")
                self.hands.poll()
                if self.robot is not None:
                    self.robot.poll()
                    self._T_world_tcp = self.robot.tcp_pose
                    link_poses = self.robot.arm_link_poses()
                else:
                    link_poses = None

                T_cam_anchor, T_cam_pegboard, frame = self.aruco_worker.get()
                center_T = self.hands.center_eye_T()
                T_wt = self.anchor.T_world_tracking
                left_pts, right_pts = self.hands.world_joints(T_wt)
                world_center = (self.anchor.world_T(center_T)
                                if center_T is not None else None)
                self._last_right_pts = right_pts
                if right_pts is None:
                    self._stable_hand_quat["right"] = None

                # The robot server expires this obstacle after 250 ms, so a
                # lost tracking stream cannot leave a stale human position in
                # the safety controller.
                now_perf = time.perf_counter()
                if self.robot is not None:
                    if (right_pts is not None
                            and now_perf - self._last_palm_cbf_pub_time
                            >= self._PALM_CBF_PUBLISH_INTERVAL_S):
                        palm_center = (
                            np.asarray(right_pts[3], float)
                            + np.asarray(right_pts[1], float)
                            + np.asarray(right_pts[6], float)) / 3.0
                        self.robot.update_palm_obstacle(
                            palm_center, radius=self._PALM_CBF_RADIUS_M,
                            clearance=self._PALM_CBF_CLEARANCE_M)
                        self._last_palm_cbf_pub_time = now_perf
                        self._palm_cbf_active = True
                    elif right_pts is None and self._palm_cbf_active:
                        self.robot.update_palm_obstacle(None)
                        self._palm_cbf_active = False

                if (not self.anchor.locked and T_cam_anchor is not None
                        and self.cam.camera_T is not None
                        and np.linalg.norm(T_cam_anchor[:3, 3]) <= self._AUTO_LOCK_MAX_DIST):
                    self.anchor.lock(T_cam_anchor, self.cam.camera_T)
                    self._last_marker_relock_time = time.time()
                    if self.robot is not None:
                        self.robot.set_scene_origin(np.eye(4))
                    if self._load_pegboard_from_file:
                        self._try_load_pegboard_from_file()
                    print("[Study3] Marker 100 locked automatically")

                # Match main_with_robot.py's marker-100 proximity relock state:
                # nearby/visible → orange; click → relock + green for one
                # second; moving away → restore the authored marker color.
                marker_distance = (
                    float(np.linalg.norm(T_cam_anchor[:3, 3]))
                    if T_cam_anchor is not None else float("inf"))
                relock_available = bool(
                    T_cam_anchor is not None
                    and self.cam.camera_T is not None
                    and marker_distance < self._RELOCK_MAX_DIST_M)
                now_wall = time.time()
                if (self._marker_relock_green_until > 0.0
                        and now_wall >= self._marker_relock_green_until):
                    self._marker_relock_green_until = 0.0
                    # Force the normal proximity color to be republished.
                    self._marker_relock_available = not relock_available
                if (self._marker_relock_green_until == 0.0
                        and relock_available != self._marker_relock_available):
                    self.colors.publish(
                        self.anchor_marker_id,
                        (_ObjectColorPublisher.HOVER if relock_available
                         else _ObjectColorPublisher.RESET))
                    self._marker_relock_available = relock_available

                if (anchor_clicked and relock_available
                        and now_wall - self._last_marker_relock_time
                        >= self._RELOCK_COOLDOWN_S):
                    was_locked = self.anchor.locked
                    if self.anchor.lock(
                            T_cam_anchor, self.cam.camera_T,
                            require_locked=was_locked):
                        if self.robot is not None:
                            self.robot.set_scene_origin(np.eye(4))
                        if self._load_pegboard_from_file:
                            self._try_load_pegboard_from_file()
                        elif T_cam_pegboard is not None:
                            self.anchor.update_pegboard_from_tracking(
                                self.cam.camera_T, T_cam_pegboard)
                            self._publish_pegboard_objects()
                        self.colors.publish(
                            self.anchor_marker_id,
                            _ObjectColorPublisher.SELECTED)
                        self._marker_relock_green_until = now_wall + 1.0
                        self._marker_relock_available = True
                        self._last_marker_relock_time = now_wall
                        print(f"[Study3] Marker 100 clicked — world "
                              f"{'relocked' if was_locked else 'locked'}")

                if self.anchor.locked:
                    self.anchor.publish()
                    self.anchor.publish_pegboard()
                    now = time.time()
                    if now - self._last_synth_pub_time >= self._SYNTH_INTERVAL:
                        ghost_visible = (
                            self.condition in (self.C2, self.C4)
                            and self._ghost_target_T is not None
                            and (self._robot_state in self._GHOST_ACTIVE_STATES
                                 or self._pending_handover))
                        robot_gripper_visible = (
                            self.condition == self.C3
                            and self._T_world_tcp is not None
                            and (self._robot_state
                                 in self._ROBOT_GRIPPER_ACTIVE_STATES
                                 or self._pending_handover))
                        visible_objects = []
                        if ghost_visible:
                            self._unity_ghost.centroid = self._ghost_target_T[:3, 3]
                            self._unity_ghost.R_o3d = self._ghost_target_T[:3, :3]
                            visible_objects.append(self._unity_ghost)
                        if robot_gripper_visible:
                            self._unity_robot_gripper.centroid = \
                                self._T_world_tcp[:3, 3]
                            self._unity_robot_gripper.R_o3d = \
                                self._T_world_tcp[:3, :3]
                            visible_objects.append(self._unity_robot_gripper)
                        self.synth.publish(visible_objects)
                        self._publish_condition_colors()
                        self._last_synth_pub_time = now

                    head_position = (world_center[:3, 3]
                                     if world_center is not None else None)
                    distances = [
                        self.workspace.distance_outside(
                            head_position, self.WORKSPACE_LO, self.WORKSPACE_HI),
                        self.workspace.distance_outside(
                            left_pts[1] if left_pts is not None else None,
                            self.WORKSPACE_LO, self.WORKSPACE_HI),
                        self.workspace.distance_outside(
                            right_pts[1] if right_pts is not None else None,
                            self.WORKSPACE_LO, self.WORKSPACE_HI),
                    ]
                    self.workspace.publish(
                        self.WORKSPACE_LO, self.WORKSPACE_HI, max(distances))

                if (self._pending_handover and right_pts is not None
                        and self.robot is not None and not self.robot.move_running
                        and time.perf_counter() >= self._retry_handover_after):
                    if self._start_handover(right_pts):
                        self._pending_handover = False

                if (self._robot_state in self._GHOST_ACTIVE_STATES
                        and right_pts is not None and self.robot is not None):
                    # Keep the ghost following the hand from the moment the
                    # robot grasps the object (through retract/staging), not
                    # just once it starts actively approaching.
                    pose = self._handover_pose(right_pts)
                    if pose is not None:
                        (position, quaternion, command_T, ghost_T,
                         inside_workspace) = pose
                        self._ghost_target_T = ghost_T
                        if inside_workspace:
                            self._set_ghost_validity(None)
                        else:
                            # Never replace the participant's target with a
                            # projected pose. Stop pursuit of the last target;
                            # the pending loop resumes as soon as the current
                            # live hand pose enters the valid inset workspace.
                            self._set_ghost_validity("outside_workspace")
                            if self._robot_state == "approaching_hand":
                                self.robot.cancel_move()
                        if (inside_workspace
                                and self._robot_state == "approaching_hand"):
                            self._tcp_target_T = command_T
                            self.robot.update_move_target(
                                position, quaternion.tolist())

                pegboard_T = self.anchor.T_pegboard_in_world
                if (pegboard_T is not None
                        and (self._last_pegboard_T is None
                             or not np.allclose(pegboard_T, self._last_pegboard_T))):
                    self._publish_pegboard_objects()
                self.vis.update_tracking(T_wt)
                self.vis.update_head(world_center)
                self.vis.update_hands(left_pts, right_pts)
                self.vis.update_palm_triangles(left_pts, right_pts)
                preview = None
                if (self.condition in (self.C2, self.C4)
                        and self._ghost_target_T is not None):
                    preview = self._ghost_target_T
                self.vis.update_left_hand_gripper(preview)
                ghost_color = self.colors._colors.get(
                    200, _ObjectColorPublisher.GHOST_VISIBLE)
                self.vis.set_left_hand_gripper_color(ghost_color[:3])
                # Draw the physical/simulated Robotiq gripper on the live TCP.
                # This is separate from the participant-preview ghost above.
                gripper_closed = self._robot_state in self._ROBOT_GRIPPER_ACTIVE_STATES
                self.vis.set_tcp_gripper_closed(gripper_closed)
                self.vis.update_tcp(self._T_world_tcp)
                self.vis.update_tcp_target(self._tcp_target_T)
                if link_poses is not None:
                    self.vis.update_robot(link_poses)
                self._log_replay_frame(
                    world_center, left_pts, right_pts, link_poses)
                self.vis.tick()

                if frame is not None:
                    cv.imshow(self._win, frame)
                key = cv.waitKey(1) & 0xFF
                if key != 255:
                    self._replay_record("keyboard", key_code=key)
                if key == 27:
                    break
                if key == 32:
                    self._start_next_trial()
                if key in (10, 13):
                    if T_cam_anchor is not None and self.cam.camera_T is not None:
                        self.anchor.lock(T_cam_anchor, self.cam.camera_T)
                        if self.robot is not None:
                            self.robot.set_scene_origin(np.eye(4))
                        print("[Study3] Marker 100 locked/relocked")
                    if self.anchor.locked and self._load_pegboard_from_file:
                        if self._try_load_pegboard_from_file():
                            print("[Study3] Pegboard loaded from saved marker-101 pose")
                    elif (T_cam_pegboard is not None
                            and self.cam.camera_T is not None
                            and self.anchor.locked):
                        self.anchor.update_pegboard_from_tracking(
                            self.cam.camera_T, T_cam_pegboard)
                        self._publish_pegboard_objects()
                        print("[Study3] Pegboard locked")
                if key == ord("l") and self._no_passthrough:
                    self.anchor.lock_tracking_origin()
                    if self._load_pegboard_from_file:
                        self._try_load_pegboard_from_file()
        finally:
            self.close()

    def close(self) -> None:
        self.aruco_worker.stop()
        self.cam.close()
        self.hands.close()
        self.colors.close()
        self.study_interactions.close()
        self.tool_layout.close()
        self.synth.close()
        self.workspace.close()
        self.anchor.close()
        if self.robot is not None:
            self.robot.close()
        self.vis.close()
        cv.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(
        description="Study 3: AR-Visualized Adaptive Handover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip",             default=cfg.UNITY_IP)
    ap.add_argument("--anchor-marker",        type=int,   default=cfg.ANCHOR_MARKER_ID,
                    help="ArUco marker ID for world frame + PyBullet scene origin")
    ap.add_argument("--pegboard-marker",      type=int,   default=cfg.PEGBOARD_MARKER_ID,
                    help="ArUco marker ID for pegboard")
    ap.add_argument("--anchor-marker-size",   type=float, default=cfg.ANCHOR_MARKER_SIZE,
                    help="Side length of the anchor marker in metres")
    ap.add_argument("--pegboard-marker-size", type=float, default=cfg.PEGBOARD_MARKER_SIZE,
                    help="Side length of the pegboard marker in metres")
    ap.add_argument("--hand-port",            type=int,   default=cfg.HAND1_PORT_FROM_UNITY)
    ap.add_argument("--simulation",      action=argparse.BooleanOptionalAction, default=cfg.SIMULATION,
                    help="Use fixed default joint angles (--simulation) or live RTDE (--no-simulation)")
    ap.add_argument("--calibrated-robot-base", action=argparse.BooleanOptionalAction,
                    default=cfg.USE_CALIBRATED_ROBOT_BASE_POSE,
                    help="Load robot base pose from calibration_data/ even in simulation mode")
    ap.add_argument("--load-pegboard-from-file", action=argparse.BooleanOptionalAction,
                    default=cfg.LOAD_PEGBOARD_FROM_FILE,
                    help="Auto-load pegboard pose from scene_layout NPZ on anchor lock "
                         "(skips needing marker 101 visible)")
    ap.add_argument("--no-passthrough", dest="no_passthrough", action="store_true",
                    help="Passthrough/ArUco unavailable: enable a manual world lock to the "
                         "Quest tracking origin via the 'l' key (marker 100 assumed to sit at "
                         "the initial/recenter origin). Everything downstream unlocks exactly "
                         "as it does on a marker lock.")
    ap.add_argument("--participant-id", required=True,
                    help="De-identified participant code shown during the session")
    ap.add_argument("--condition", dest="study3_condition",
                    choices=("no_ghost_no_color", "ghost_no_color",
                             "no_ghost_robot_color", "ghost_color"),
                    required=True,
                    help="Study condition: no_ghost_no_color (C1) / "
                         "ghost_no_color (C2) / no_ghost_robot_color (C3) / "
                         "ghost_color (C4)")
    args = ap.parse_args()
    if args.anchor_marker == args.pegboard_marker:
        ap.error("--anchor-marker and --pegboard-marker must be different.")
    scene = MainScene(
        quest_ip                   = args.quest_ip,
        anchor_marker_id           = args.anchor_marker,
        pegboard_marker_id         = args.pegboard_marker,
        anchor_marker_size_m       = args.anchor_marker_size,
        pegboard_marker_size_m     = args.pegboard_marker_size,
        hand_port                  = args.hand_port,
        simulation                 = args.simulation,
        use_calibrated_robot_base  = args.calibrated_robot_base,
        load_pegboard_from_file    = args.load_pegboard_from_file,
        no_passthrough             = args.no_passthrough,
        condition                  = args.study3_condition,
        participant_id             = args.participant_id)
    scene.run()


if __name__ == "__main__":
    main()
