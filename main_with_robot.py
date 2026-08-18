"""
main_hand_m100_w_sim.py — Same as main_hand_m10.py but uses ArUco marker 100 as the
world-frame anchor AND PyBullet scene origin.

Workflow
--------
  1. Hold marker 100 visible within 0.5m — the world frame locks
     automatically the first time it's seen that close (no ENTER needed),
     or press ENTER while it's visible from any distance.
     PyBullet scene is placed at the locked pose immediately.
  2. Hold marker 101 visible → press ENTER to lock the pegboard pose.
     Marker 100 does NOT need to be visible at this step.
  3. Once locked, all later re-locks require an explicit trigger — press
     ENTER again (whichever marker is visible), or click the anchor-marker
     proximity relock cube. Auto-lock-on-sight only ever fires once, before
     the first lock.
  4. The tracked 250x200x25mm board (markers 102/103, one on each large
     face) is tracked cont+inuously once the world frame is locked — no
     ENTER press required. Either marker being visible is enough; its pose
     is published on port 5014 (board_root_matrix) and drawn as a wireframe
     box in PyBullet.

Keys (OpenCV window must be focused)
--------------------------------------
  ENTER = lock / relock (handles marker 100 and/or 101 independently)
  ESC   = quit

Usage
-----
  python main_hand_m100_w_sim.py
  python main_hand_m100_w_sim.py --quest-ip 192.168.50.201 --hand-port 5570
"""

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path

import cv2 as cv
import dearpygui.dearpygui as dpg
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
)
from utils.pose_helpers import (
    _unity_pose_to_T, _adapt_cx_cy, _transform_point,
    _BONES_NP, _N_JOINTS, _HIDDEN_PT, _JOINT_GROUP_ORDER,
    _unity_to_o3d, _to_world, _unit, _palm_quat, _tool_grasp_quat, _extract_joints,
    BOARD_SIZE, T_BOARD_FROM_MARKER_A, T_BOARD_FROM_MARKER_B, T_UNITY_BOARD_ROOT_FROM_ORIGIN,
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
        self._T_cam_board    = {}
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
                self._T_cam_board    = det.get("T_cam_board", {})
                self._det_vis        = det["vis"] #the camera frame with ArUco detection overlays drawn on it

    def get(self):
        """Return (T_cam_anchor, T_cam_pegboard, T_cam_board, det_vis) — non-blocking."""
        with self._lock:
            return (self._T_cam_anchor, self._T_cam_pegboard,
                    self._T_cam_board, self._det_vis)

    def stop(self) -> None:
        self._running = False

# =============================================================================
# ArUco pose estimator — marker 100 (anchor) + marker 101 (pegboard)
# =============================================================================

def _load_prescan_marker_ids() -> tuple[int, ...]:
    """Secondary relock-marker IDs, read straight from the prescan file so it is
    the single source of truth: whatever was registered in
    world_markers_T_ref_from_marker.json is what the detector looks for and what
    gets a click-to-relock cube. Returns () if the file is absent/unreadable."""
    try:
        if not cfg.WORLD_MARKERS_FILE.exists():
            return ()
        data = json.loads(cfg.WORLD_MARKERS_FILE.read_text())
        return tuple(sorted(int(mid) for mid in data.get("markers", {})))
    except Exception as e:
        print(f"[Prescan] Could not read marker IDs from "
              f"{cfg.WORLD_MARKERS_FILE.name}: {e}")
        return ()


class _ArucoPoseEstimator:
    def __init__(self, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 board_marker_ids: tuple = (),
                 board_marker_size_m: float | None = None,
                 dictionary=cv.aruco.DICT_6X6_1000):
        self.anchor_marker_id   = int(anchor_marker_id)
        self.pegboard_marker_id = int(pegboard_marker_id)
        self.board_marker_ids   = tuple(int(m) for m in board_marker_ids)
        self.anchor_marker_size   = float(anchor_marker_size_m)
        self.pegboard_marker_size = float(pegboard_marker_size_m)
        self.board_marker_size    = float(board_marker_size_m
                                          if board_marker_size_m is not None
                                          else anchor_marker_size_m)
        self._dict     = cv.aruco.getPredefinedDictionary(dictionary)
        self._detector = cv.aruco.ArucoDetector(
            self._dict, cv.aruco.DetectorParameters())

        def _obj_pts(size: float) -> np.ndarray:
            s = size / 2.0
            return np.array([[-s, s, 0.], [s, s, 0.],
                              [s, -s, 0.], [-s, -s, 0.]], dtype=np.float64)

        self._anchor_obj_pts   = _obj_pts(self.anchor_marker_size)
        self._pegboard_obj_pts = _obj_pts(self.pegboard_marker_size)
        self._board_obj_pts    = _obj_pts(self.board_marker_size)

    def detect(self, bgr, fx, fy, cx, cy, dist=None, draw=True) -> dict:
        K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1)) if dist is None else np.array(dist).reshape(-1, 1)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        vis = bgr.copy()
        result = {"vis": vis, "T_cam_anchor": None, "T_cam_pegboard": None,
                  "T_cam_board": {}}
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
            elif mid in self.board_marker_ids:
                key, obj_pts, axis_len = "T_cam_board", self._board_obj_pts, self.board_marker_size
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
            if key == "T_cam_board":
                result["T_cam_board"][mid] = T
            else:
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
                 pegboard_pub_port: int = cfg.PEGBOARD_ROOT_PORT,
                 board_pub_port: int = cfg.BOARD_ROOT_PORT):
        self._T_wt: np.ndarray | None = None  #  This gives you where the marker is in tracking space, 
                                              #  and inverting that gives you the transform that converts from tracking space into world space
        self._T_offset = np.eye(4, dtype=np.float64) #From offset tuner
        self._T_world_pegboard: np.ndarray | None = None
        self._T_world_board: np.ndarray | None = None
        self._T_eye_offset: np.ndarray | None = None
        self._T_world_marker: dict[int, np.ndarray] = {}  # secondary marker relocking: id -> T_ref_from_marker
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{pub_ip}:{pub_port}")
        self._pub_pegboard = ctx.socket(zmq.PUB)
        self._pub_pegboard.connect(f"tcp://{pub_ip}:{pegboard_pub_port}")
        self._pub_board = ctx.socket(zmq.PUB)
        self._pub_board.connect(f"tcp://{pub_ip}:{board_pub_port}")
        time.sleep(0.2)
        self._load_eye_offset()
        self._load_world_markers()

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

    def _load_world_markers(self):
        """Load prescan registration transforms for secondary markers."""
        if not cfg.WORLD_MARKERS_FILE.exists():
            return
        try:
            data = json.loads(cfg.WORLD_MARKERS_FILE.read_text())
            for mid_str, T_flat in data.get("markers", {}).items():
                mid = int(mid_str)
                T = np.array(T_flat, dtype=np.float64).reshape(4, 4)
                self._T_world_marker[mid] = T
            if self._T_world_marker:
                print(f"[Anchor] World markers loaded: {sorted(self._T_world_marker.keys())}")
        except Exception as e:
            print(f"[Anchor] World marker load failed: {e}")

    def _save_eye_offset(self):
        try:
            cfg.EYE_OFFSET_FILE.write_text(
                json.dumps({"T_eye_offset": self._T_eye_offset.flatten().tolist()}, indent=2))
            print(f"[Anchor] Eye offset saved to {cfg.EYE_OFFSET_FILE.name}")
        except Exception as e:
            print(f"[Anchor] Eye offset save failed: {e}")

    def _effective_cam_T(self, cam_T: np.ndarray,
                         center_T: np.ndarray | None) -> np.ndarray | None:
        if self._T_eye_offset is not None and center_T is not None:
            return center_T @ self._T_eye_offset
        return cam_T

    def set_offset(self, pos_offset, yaw_deg: float):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = np.array(pos_offset, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('z', yaw_deg, degrees=True).as_matrix()
        self._T_offset = T

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

    def relock_from_world_marker(self, marker_id: int, T_cam_marker: np.ndarray,
                                  cam_T: np.ndarray) -> bool:
        """Relock from a secondary marker (104-107) using prescan registration.

        T_cam_marker: camera → marker (OpenCV convention)
        cam_T: camera pose in tracking frame
        Returns True if relock succeeded.
        """
        if marker_id not in self._T_world_marker or not self.locked:
            return False
        if T_cam_marker is None or cam_T is None:
            return False

        # Compute reference marker pose in camera frame
        # T_ref_from_marker[id] maps marker-local points to reference frame
        # inv(T_ref_from_marker) maps reference frame to marker-local
        # Composing with T_cam_marker gives reference marker in camera frame
        T_ref_from_marker = self._T_world_marker[marker_id]
        T_cam_ref = T_cam_marker @ np.linalg.inv(T_ref_from_marker)

        # Update tracking frame to world transform
        self._T_wt = np.linalg.inv(cam_T @ T_cam_ref)
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

    def update_board_from_tracking(self, cam_T: np.ndarray,
                                   T_cam_marker: np.ndarray,
                                   T_board_from_marker: np.ndarray) -> bool:
        """Compute tracked-board pose in world frame from whichever of
        markers 102/103 is currently visible.

        T_board_from_marker is the fixed offset (board origin expressed in
        the detected marker's local frame) for that specific marker.
        Marker 100 does NOT need to be visible — uses the locked _T_wt instead.
        """
        if not self.locked or T_cam_marker is None or cam_T is None:
            return False
        self._T_world_board = self._T_wt @ cam_T @ T_cam_marker @ T_board_from_marker
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
        if self._T_world_pegboard is None:
            return None
        return self._T_offset @ self._T_world_pegboard

    @property
    def T_board_in_world(self) -> np.ndarray | None:
        if self._T_world_board is None:
            return None
        return self._T_offset @ self._T_world_board

    def world_T(self, T_tracking_local: np.ndarray) -> np.ndarray | None: 
        #a general-purpose helper that converts any pose from Quest tracking space into world space.
        if self._T_wt is None:
            return None
        return (self._T_offset @ self._T_wt) @ T_tracking_local

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
        return self._publish_T(np.linalg.inv(self._T_offset @ self._T_wt),
                               self._pub, "WorldRoot",
                               "world_root_position",
                               "world_root_rotation_xyzw",
                               "world_root_matrix")

    def publish_pegboard(self) -> bool:
        if self._T_world_pegboard is None:
            return False
        return self._publish_T(self._T_offset @ self._T_world_pegboard,
                               self._pub_pegboard, "PegboardRoot",
                               "pegboard_root_position",
                               "pegboard_root_rotation_xyzw",
                               "pegboard_root_matrix")

    def publish_board(self) -> bool:
        if self._T_world_board is None:
            return False
        T_board = (self._T_offset @ self._T_world_board) @ T_UNITY_BOARD_ROOT_FROM_ORIGIN
        # pos, rot_xyzw, _ = self._to_unity_pose(T_board)
        # euler = ScipyR.from_quat(rot_xyzw).as_euler("xyz", degrees=True)
        # print(f"[BoardRoot] pos={pos}, rot_euler_xyz={euler.tolist()}")
        return self._publish_T(T_board,
                               self._pub_board, "BoardRoot",
                               "board_root_position",
                               "board_root_rotation_xyzw",
                               "board_root_matrix")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass
        try:
            self._pub_pegboard.close(0)
        except Exception:
            pass
        try:
            self._pub_board.close(0)
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


class _RelockCubePublisher:
    """Publishes the secondary relock cubes' world poses to Unity (PUB, port 5017).

    Positions come from the prescan registration (anchor._T_world_marker) and are
    static, but are republished periodically so a late-joining Unity receiver still
    gets them. Poses are sent in Unity coordinates; Unity's RelockCubePoseReceiver
    matches each cube to the interactable whose ToolClickPublisher.toolId == id and
    sets its localPosition (the cubes are children of WorldRoot).

    Each cube is lifted along Unity world up (+Y) by half its edge so its bottom
    face sits on top of the marker rather than the cube centre on the marker."""

    CUBE_EDGE_M = 0.10   # cube side length (metres) — keep in sync with the Unity localScale

    def __init__(self, ip: str, port: int = cfg.RELOCK_CUBE_PORT):
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.2)
        self._payload: "str | None" = None
        print(f"[RelockCubes] Connected to tcp://{ip}:{port}")

    def set_markers(self, T_world_marker: dict) -> None:
        """Build the payload from {marker_id: T_world_marker (4x4, Open3D frame)}."""
        cubes = []
        for mid, T in sorted(T_world_marker.items()):
            pos, rot_xyzw, _ = _WorldAnchor._to_unity_pose(np.asarray(T, dtype=np.float64))
            # Lift along Unity world up (+Y is up/down in Unity) so the cube's
            # bottom face sits on top of the marker instead of centred on it.
            pos = [pos[0], pos[1] + self.CUBE_EDGE_M / 2.0, pos[2]]
            cubes.append({"id": int(mid), "position": pos, "rotation_xyzw": rot_xyzw})
        self._payload = json.dumps({"cubes": cubes})
        print(f"[RelockCubes] Prepared {len(cubes)} cube pose(s): "
              f"{sorted(int(m) for m in T_world_marker)}")

    def publish(self) -> None:
        if self._payload is None:
            return
        try:
            self._pub.send_string(self._payload)
        except Exception as e:
            print(f"[RelockCubes] Publish error: {e}")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


class _HandoverSpherePublisher:
    """Publishes the chosen handover centroid to Unity (PUB, port 5018) so the
    person in the headset sees a sphere at the compromise delivery point.

    The position is sent in the WorldRoot (marker-100) frame in Unity coordinates,
    exactly like the relock cubes, so Unity's HandoverSphereReceiver can set it as
    a localPosition on a sphere GameObject parented under WorldRoot. `visible`
    toggles the renderer; the sphere is hidden once the robot arrives."""

    def __init__(self, ip: str, port: int = cfg.HANDOVER_SPHERE_PORT):
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        time.sleep(0.2)
        self._payload = json.dumps({"position": [0.0, 0.0, 0.0], "visible": False})
        print(f"[HandoverSphere] Connected to tcp://{ip}:{port}")

    def show(self, centroid_o3d) -> None:
        """centroid_o3d = (3,) world point in the Open3D/world frame."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(centroid_o3d, dtype=np.float64)
        pos, _, _ = _WorldAnchor._to_unity_pose(T)
        self._payload = json.dumps({"position": pos, "visible": True})

    def hide(self) -> None:
        self._payload = json.dumps({"position": [0.0, 0.0, 0.0], "visible": False})

    def publish(self) -> None:
        try:
            self._pub.send_string(self._payload)
        except Exception as e:
            print(f"[HandoverSphere] Publish error: {e}")

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass


# =============================================================================
# Tool layout manager
# =============================================================================

class _ToolLayoutManager:
    """Loads tool_layout.json once at startup and publishes world-space tool
    definitions to Unity (port 5011). To apply changes, restart the script."""

    PORT = cfg.TOOL_LAYOUT_PORT

    def __init__(self, json_path: str, ip: str):
        self._tools: list = []
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
        return [self._tool_box_world(t, T) for t in self._tools]

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

# =============================================================================
# Tool selection manager
# =============================================================================

class _ToolSelectionManager:
    TCP_TOOL_ID     = 200
    TCP_COLOR       = [1.0, 0.8, 0.2, 0.05]      # gold resting color; sent on port 5010
    SELECTED_COLOR = [0.0, 1.0, 0.0, 0.25]     #when cursor clicks
    HOVER_COLOR    = [1.0, 0.5, 0.0, 0.25]     #when cursor hovers
    RESET_COLOR    = [-1.0, -1.0, -1.0, -1.0]   # sentinel → restores to resting color
    TOOL_COLOR     = [0.80, 0.88, 1.0,  0.25]    # light blue for "tool" category
    PART_COLOR     = [1.0,  0.78, 0.78, 0.25]    # light red  for "part" category
    HIGHLIGHT_COLOR = [0.0, 1.0, 1.0, 0.25]       # cyan — pegboard tool needed for the current step

    def __init__(self, quest_ip: str, click_port: int = cfg.TOOL_CLICK_PORT,
                 color_port: int = cfg.TOOL_COLOR_PORT,
                 highlight_port: int = cfg.GEARBOX_HIGHLIGHT_PORT):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{quest_ip}:{click_port}")
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{color_port}")
        # Pegboard highlight ids from gearbox_control.py. Python↔Python convention:
        # the receiver BINDS (unlike the click SUB / color PUB above, which connect).
        self._hl_sub = ctx.socket(zmq.SUB)
        self._hl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._hl_sub.bind(f"tcp://0.0.0.0:{highlight_port}")
        time.sleep(0.2)
        self._active_tool_id:   int | None              = None
        self._hovered_tool_id:  int | None              = None
        self._active_hand:      str | None              = None
        self._category_colors:  dict[int, list[float]]  = {}
        self._highlighted:      set[int]                = set()
        self._assembly_events:  list[dict]              = []
        self._on_cancel = None
        self._last_color_refresh = 0.0

    def poll(self, timeout_ms: int = 0) -> bool:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False
        processed = False
        while True:
            try:
                raw = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(raw)
                tool_id    = int(msg["tool_id"])
                event_type = msg.get("event_type", "selected")
                hand       = msg.get("hand", "unknown")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[ToolSelection] Bad message: {e}")
                continue
            self._handle_event(tool_id, event_type, hand)
            processed = True
        return processed

    def _handle_event(self, tool_id: int, event_type: str, hand: str = "unknown"):
        if event_type == "selected":
            self._handle_click(tool_id, hand)
        elif event_type == "hover_enter":
            self._handle_hover_enter(tool_id)
        elif event_type == "hover_exit":
            self._handle_hover_exit(tool_id)

    def set_category_color(self, tool_id: int, color: list[float]) -> None:
        """Store the tool's base category color and paint it. Call from _apply_tool_category_colors.
        A currently-highlighted tool keeps its cyan (highlights survive a category repaint, e.g. a
        relock republish) unless it is the actively-selected tool."""
        self._category_colors[tool_id] = color
        if tool_id in self._highlighted and tool_id != self._active_tool_id:
            self.send_color(tool_id, self.HIGHLIGHT_COLOR)
        else:
            self.send_color(tool_id, color)

    def reset_to_category(self, tool_id: int) -> None:
        """Restore a tool to its category color (falls back to RESET_COLOR if not registered)."""
        self.send_color(tool_id, self._category_colors.get(tool_id, self.RESET_COLOR))

    def _restore(self, tool_id: int) -> None:
        """Return a tool to its resting appearance after a hover/deselect. If a stage menu is open
        and this tool is still highlighted, that means cyan — highlights only clear on menu
        close/reset, so cyan→orange→cyan on hover, not cyan→orange→category."""
        if tool_id in self._highlighted:
            self.send_color(tool_id, self.HIGHLIGHT_COLOR)
        else:
            self.reset_to_category(tool_id)

    def _handle_click(self, tool_id: int, hand: str = "unknown"):
        # hand was near tool A (hover) and clicked a different tool B before hover_exit(A) arrived
        if self._hovered_tool_id is not None and self._hovered_tool_id != tool_id:
            self._restore(self._hovered_tool_id)
        self._hovered_tool_id = None
        if self._active_tool_id == tool_id:
            if self._on_cancel is not None:
                # grasp in progress → cancel and retract; on_complete will deselect
                self._on_cancel()
                return
            # clicking the already-selected tool → deselect (toggle off)
            self._active_tool_id = None
            self._active_hand    = None
            self._restore(tool_id)
        elif self._active_tool_id is not None:
            # clicking a different tool while another is already selected → switch selection
            self._restore(self._active_tool_id)
            self._active_tool_id = tool_id
            self._active_hand    = hand
            self.send_color(tool_id, self.SELECTED_COLOR)
        else:
            # nothing was selected → select this tool
            self._active_tool_id = tool_id
            self._active_hand    = hand
            self.send_color(tool_id, self.SELECTED_COLOR)

    def _handle_hover_enter(self, tool_id: int):
        # hand moved from tool A to tool B without a hover_exit(A) in between → clear A first
        if self._hovered_tool_id is not None and self._hovered_tool_id != tool_id:
            self._restore(self._hovered_tool_id)
        self._hovered_tool_id = None
        # hovering over the already-selected tool — don't downgrade its color to HOVER_COLOR
        if tool_id == self._active_tool_id:
            return
        self._hovered_tool_id = tool_id
        self.send_color(tool_id, self.HOVER_COLOR)

    def _handle_hover_exit(self, tool_id: int):
        # Unity sent exit for a tool we never recorded as hovered (e.g. exit arrived after a click cleared the state)
        if tool_id != self._hovered_tool_id:
            return
        self._hovered_tool_id = None
        # tool was clicked while being hovered — it is now selected, don't strip its SELECTED_COLOR
        if tool_id == self._active_tool_id:
            return
        self._restore(tool_id)

    def send_color(self, tool_id: int, color: list[float]):
        msg = {"tool_id": int(tool_id), "color": [float(c) for c in color]}
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception as e:
            print(f"[ToolSelection] Publish error: {e}")

    def refresh_colors(self, interval_s: float = 1.0) -> None:
        """Republish effective colors so late port-5010 subscribers catch up."""
        now = time.monotonic()
        if now - self._last_color_refresh < interval_s:
            return
        self._last_color_refresh = now
        for tool_id, resting_color in self._category_colors.items():
            if tool_id == self._active_tool_id:
                color = self.SELECTED_COLOR
            elif tool_id == self._hovered_tool_id:
                color = self.HOVER_COLOR
            elif tool_id in self._highlighted:
                color = self.HIGHLIGHT_COLOR
            else:
                color = resting_color
            self.send_color(tool_id, color)

    @property
    def active_tool_id(self) -> int | None:
        return self._active_tool_id

    @property
    def highlighted(self) -> set:
        """Tool ids currently flagged by the pegboard highlight (gearbox_control.py on 5024)."""
        return self._highlighted

    @property
    def active_hand(self) -> str | None:
        return self._active_hand

    def deselect(self, tool_id: int):
        if self._active_tool_id == tool_id:
            self._active_tool_id = None
            self._active_hand    = None

    # ── Pegboard highlight (gearbox_control.py → here on GEARBOX_HIGHLIGHT_PORT) ──
    # Cyan-flag the tools needed for the current assembly step. Folded in from
    # test_tool_layout.py's ToolHighlightBridge; here the restore goes back to the
    # tool's CATEGORY colour (not the raw sentinel), and the actively-selected tool
    # is never repainted (selection wins over highlight).
    def drain_highlights(self, timeout_ms: int = 0) -> bool:
        poller = zmq.Poller()
        poller.register(self._hl_sub, zmq.POLLIN)
        if not dict(poller.poll(timeout=timeout_ms)):
            return False
        processed = False
        while True:
            try:
                raw = self._hl_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[ToolHighlight] Bad message: {e}")
                continue
            event = msg.get("event")
            if event == "clear":
                self._apply_highlight_clear()
            elif event == "highlight":
                self._apply_highlight(msg.get("ids", []))
            elif event == "assembly_state":
                self._assembly_events.append(msg)
            processed = True
        return processed

    def pop_assembly_events(self) -> list[dict]:
        events, self._assembly_events = self._assembly_events, []
        return events

    def _apply_highlight(self, ids) -> None:
        new = {int(i) for i in ids}
        for tid in self._highlighted - new:          # dropped out of the set → restore
            if tid != self._active_tool_id:
                self.reset_to_category(tid)
        for tid in new - self._highlighted:          # newly appearing → highlight colour
            if tid != self._active_tool_id:          # selection wins over highlight
                self.send_color(tid, self.HIGHLIGHT_COLOR)
        self._highlighted = new

    def _apply_highlight_clear(self) -> None:
        for tid in self._highlighted:
            if tid != self._active_tool_id and tid != self._hovered_tool_id:
                self.reset_to_category(tid)
        self._highlighted = set()

    def close(self):
        try: self._sub.close(0)
        except Exception: pass
        try: self._pub.close(0)
        except Exception: pass
        try: self._hl_sub.close(0)
        except Exception: pass

class _WorkspaceBoundPublisher:
    """Publishes the robot workspace boundary box to Unity on port 5015 (PUB).
    bounds_lo/bounds_hi are sent once (constant in world frame); dist_outside
    is sent every frame and drives the wireframe's fade-in opacity in Unity —
    0 when the user's head/hands are all inside the box, positive and growing
    the further outside any of them is.
    """

    def __init__(self, quest_ip: str, port: int = cfg.WORKSPACE_BOUND_PORT):
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{port}")

    @staticmethod
    def dist_outside(pos: "np.ndarray | None", lo: np.ndarray, hi: np.ndarray) -> float:
        """0.0 if pos is inside [lo, hi] (or untracked), else distance to the
        nearest face of the box."""
        if pos is None:
            return 0.0
        d = np.maximum(np.maximum(lo - pos, pos - hi), 0.0)
        return float(np.linalg.norm(d))

    def publish(self, lo: np.ndarray, hi: np.ndarray, dist_outside: float) -> None:
        lo_u = open3d_to_unity_vector(lo)
        hi_u = open3d_to_unity_vector(hi)
        msg = {
            "bounds_lo":    lo_u.tolist(),
            "bounds_hi":    hi_u.tolist(),
            "dist_outside": float(dist_outside),
        }
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._pub.close(0)
        except Exception:
            pass

# =============================================================================
# Grip-state / target-pose bridge (ports 5012 / 5013)
# =============================================================================

class _GripPoseBridge:
    """Bidirectional bridge for Unity's AR box manipulation workflow.

    Publishes grip state and box pose to Unity on port 5012, and receives the
    manipulated box pose back from Unity on port 5013.
    """

    def __init__(self, quest_ip: str,
                 grip_state_port: int = cfg.GRIP_STATE_PORT,
                 target_pose_port: int = cfg.TARGET_POSE_PORT):
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{grip_state_port}")
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{quest_ip}:{target_pose_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.setsockopt(zmq.RCVTIMEO, 0)

    def publish(self, grip_state: str, T_tcp_world: np.ndarray) -> None:
        """Compute box pose from TCP transform and publish."""
        # Box centre = TCP position + BOX_FORWARD_OFFSET along gripper Z
        gripper_z_world = T_tcp_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
        box_pos_w = T_tcp_world[:3, 3] + cfg.BOX_FORWARD_OFFSET * gripper_z_world

        q_xyzw  = ScipyR.from_matrix(T_tcp_world[:3, :3]).as_quat()
        pos_u   = open3d_to_unity_vector(box_pos_w)
        q_wxyz  = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u     = open3d_to_unity_quaternion(q_wxyz)
        sz_u    = open3d_to_unity_vector(np.array(cfg.BOX_SIZE, dtype=float))

        msg = {
            "grip_state":    grip_state,
            "box_pos":       pos_u.tolist(),
            "box_rot_xyzw":  [float(q_u[1]), float(q_u[2]),
                               float(q_u[3]), float(q_u[0])],
            "box_size":      sz_u.tolist(),
        }
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception:
            pass

    def poll(self) -> "np.ndarray | None":
        """Return the latest released 4×4 box pose, or None if none arrived."""
        latest = None
        while True:
            try:
                raw = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                return latest
            try:
                data = json.loads(raw)
                pos_u = data["tcp_pos"]       # Unity frame [x, y, z]
                q_u   = data["tcp_rot_xyzw"]  # Unity xyzw
                pos_w = unity_to_open3d_vector(
                    {"x": pos_u[0], "y": pos_u[1], "z": pos_u[2]})
                q_o3d = unity_to_open3d_quaternion(
                    [q_u[3], q_u[0], q_u[1], q_u[2]])
                R_w = ScipyR.from_quat(
                    [q_o3d[1], q_o3d[2], q_o3d[3], q_o3d[0]]).as_matrix()
                latest = np.eye(4, dtype=np.float64)
                latest[:3, :3] = R_w
                latest[:3, 3] = pos_w
            except Exception:
                continue

    def close(self) -> None:
        for sock in (self._pub, self._sub):
            try:
                sock.close(0)
            except Exception:
                pass



# =============================================================================
# Gearbox pose mirror (Unity -> Open3D, port 5027)
# =============================================================================

class _GearboxPoseReceiver:
    """Receives live Unity gearbox part poses for the Open3D mirror."""

    def __init__(self, quest_ip: str, port: int = cfg.GEARBOX_POSE_PORT):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{quest_ip}:{port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.RCVTIMEO, 0)
        self.states: dict | None = None
        self._first_rx_logged = False
        self._last_wait_log = 0.0

    @staticmethod
    def _unity_part_pose_to_T(pos_u, q_u) -> np.ndarray:
        pos_w = unity_to_open3d_vector({"x": pos_u[0], "y": pos_u[1], "z": pos_u[2]})
        q_o3d = unity_to_open3d_quaternion([q_u[3], q_u[0], q_u[1], q_u[2]])
        R_w = ScipyR.from_quat(
            [q_o3d[1], q_o3d[2], q_o3d[3], q_o3d[0]]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_w
        T[:3, 3] = pos_w
        return T

    @staticmethod
    def _unity_scale_to_o3d(scale_u):
        if scale_u is None:
            return None
        return unity_to_open3d_vector({"x": scale_u[0], "y": scale_u[1], "z": scale_u[2]})

    @staticmethod
    def _normalize_name(name: str) -> str:
        # Unity prefabs in this repo use underscores, but nested imported mesh
        # children can arrive as "occurrence of Part_Name" from older builds.
        prefix = "occurrence of "
        name = name.strip()
        while name.lower().startswith(prefix):
            name = name[len(prefix):].strip()
        if name.startswith("BaseBoard"):
            return "BaseBoard"
        if "_" in name:
            return name
        for prefix in ("Bearing", "Stand", "GearRod", "Gear", "Pin", "Screw"):
            if name.startswith(prefix + "Row"):
                rest = name[len(prefix):]
                for side in ("Left", "Right"):
                    if rest.endswith(side):
                        return f"{prefix}_{rest[:-len(side)]}_{side}"
                return f"{prefix}_{rest}"
        if name == "CrankHandle":
            return "CrankHandle_Row1"
        return name

    def poll(self) -> dict | None:
        latest = None
        while True:
            try:
                raw = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                if latest is not None:
                    self.states = latest
                elif self.states is None:
                    now = time.time()
                    if now - self._last_wait_log > 5.0:
                        print(f"[GearboxPose] waiting for Unity poses on :{cfg.GEARBOX_POSE_PORT}")
                        self._last_wait_log = now
                return latest
            try:
                data = json.loads(raw)
                if data.get("type") != "gearbox_pose":
                    continue
                states = {}
                for part in data.get("parts", []):
                    name = self._normalize_name(str(part.get("name", "")))
                    pos = part.get("pos")
                    rot = part.get("rot_xyzw")
                    if not name or pos is None or rot is None:
                        continue
                    states[name] = {
                        "active": bool(part.get("active", True)),
                        "T": self._unity_part_pose_to_T(pos, rot),
                        "scale": self._unity_scale_to_o3d(part.get("scale")),
                    }
                latest = states
                if not self._first_rx_logged:
                    print(f"[GearboxPose] first pose message: {len(states)} named parts")
                    self._first_rx_logged = True
            except Exception as e:
                if not self._first_rx_logged:
                    print(f"[GearboxPose] ignored malformed message: {e}")
                continue

    def close(self) -> None:
        try:
            self._sub.close(0)
        except Exception:
            pass


class _TaskGraphOpen3DReceiver:
    """Receives direct GUI selection/progress events from gearbox_task_graph.py."""

    def __init__(self, port: int = cfg.GEARBOX_STEP_SELECT_PORT):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.bind(f"tcp://0.0.0.0:{port}")

    def poll(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(json.loads(self._sub.recv_string(flags=zmq.NOBLOCK)))
            except zmq.Again:
                return events
            except (json.JSONDecodeError, TypeError):
                continue

    def close(self) -> None:
        try:
            self._sub.close(0)
        except Exception:
            pass





# =============================================================================
# Offset tuner
# =============================================================================

class _OffsetTuner:
    """DearPyGui panel to nudge the whole world (ArUco) frame live.

    Every axis is a pure DELTA applied on top of the active lock via
    ``Anchor.set_offset``.  Each axis has a coarse slider AND an editable text
    box bound to the same value: drag the slider for a quick nudge, or type an
    exact number in the box for fine tuning.
    """
    SAVE_FILE    = _FILE_DIR / "offset_config_passthrough.json"
    POS_SPAN_M   = 3.0      # total slider travel for dX/dY/dZ  → ±1.5 m
    POS_STEP_M   = 0.010    # slider snaps to 10 mm
    YAW_SPAN_DEG = 360.0    # total slider travel for dYaw       → ±180°
    YAW_STEP_DEG = 1.0

    # (key, label, is_position)
    _AXES = [
        ("dx",  "Delta X  (m)",    True),
        ("dy",  "Delta Y  (m)",    True),
        ("dz",  "Delta Z  (m)",    True),
        ("yaw", "Delta Yaw (deg)", False),
    ]

    def __init__(self):
        self.dpg = dpg
        self._alive = True
        self._flash_until = 0.0
        self._vals = {key: 0.0 for key, _, _ in self._AXES}

        dpg.create_context()
        dpg.create_viewport(title="Offset Tuner (ArUco frame)",
                            width=520, height=300)
        with dpg.window(tag="offset_window"):
            dpg.add_text("World-frame delta offset (applied on top of the lock)",
                         color=(170, 180, 195))
            dpg.add_text("Drag a slider for a quick nudge, or type an exact "
                         "value in the box.", color=(140, 150, 165))
            dpg.add_separator()
            for key, label, is_pos in self._AXES:
                half   = (self.POS_SPAN_M if is_pos else self.YAW_SPAN_DEG) / 2.0
                s_fmt  = "%.2f" if is_pos else "%.0f"    # slider: 10 mm / 1°
                i_fmt  = "%.3f" if is_pos else "%.1f"    # text box: finer
                with dpg.group(horizontal=True):
                    dpg.add_text(label, color=(200, 205, 215))
                    dpg.add_slider_float(tag=f"off_slider_{key}", width=250,
                                         default_value=0.0, min_value=-half,
                                         max_value=half, format=s_fmt,
                                         callback=self._on_slider, user_data=key)
                    dpg.add_input_float(tag=f"off_input_{key}", width=120,
                                        default_value=0.0, min_value=-half,
                                        max_value=half, min_clamped=True,
                                        max_clamped=True, step=0.0, format=i_fmt,
                                        on_enter=True, callback=self._on_input,
                                        user_data=key)
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(label="SAVE", width=90, callback=self._save)
                dpg.add_text("", tag="off_status", color=(0, 220, 120))

        dpg.set_primary_window("offset_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._load()

    def _sync(self, key, v):
        """Set the canonical value and mirror it onto both widgets.
        set_value is programmatic, so it does not re-fire the callbacks."""
        self._vals[key] = v
        self.dpg.set_value(f"off_slider_{key}", v)
        self.dpg.set_value(f"off_input_{key}", v)

    def _on_slider(self, sender, app_data, user_data):
        v = float(app_data)
        if user_data != "yaw":                       # snap position to 10 mm
            v = round(v / self.POS_STEP_M) * self.POS_STEP_M
        self._sync(user_data, v)

    def _on_input(self, sender, app_data, user_data):
        half = (self.YAW_SPAN_DEG if user_data == "yaw" else self.POS_SPAN_M) / 2.0
        self._sync(user_data, max(-half, min(half, float(app_data))))

    def get(self):
        return ((self._vals["dx"], self._vals["dy"], self._vals["dz"]),
                self._vals["yaw"])

    def _save(self, *_):
        try:
            with open(self.SAVE_FILE, "w") as f:
                json.dump({k: self._vals[k] for k in ("dx", "dy", "dz", "yaw")},
                          f, indent=2)
            self.dpg.set_value("off_status", "Saved!")
            self._flash_until = time.time() + 1.5
            print(f"[OffsetTuner] Saved to {self.SAVE_FILE}")
        except Exception as e:
            print(f"[OffsetTuner] Save error: {e}")

    def _load(self):
        if not self.SAVE_FILE.exists():
            return
        try:
            with open(self.SAVE_FILE) as f:
                data = json.load(f)
            for k in ("dx", "dy", "dz", "yaw"):
                if k in data:
                    self._sync(k, float(data[k]))
            print(f"[OffsetTuner] Loaded from {self.SAVE_FILE}")
        except Exception as e:
            print(f"[OffsetTuner] Load error: {e}")

    def draw(self):
        """Pump one DearPyGui frame; call once per main-loop iteration."""
        if not self._alive:
            return
        dpg = self.dpg
        if not dpg.is_dearpygui_running():
            self._alive = False
            return
        if self._flash_until and time.time() > self._flash_until:
            dpg.set_value("off_status", "")
            self._flash_until = 0.0
        dpg.render_dearpygui_frame()

    def close(self):
        if not self._alive:
            return
        self._alive = False
        try:
            self.dpg.destroy_context()
        except Exception:
            pass

# =============================================================================
# Jog GUI
# =============================================================================

class _JogGUI:
    """OpenCV jog panel: 6 trackbars (XYZ pos + RPY ori) and a Start/Stop button.

    Sliders are the single source of truth for the jog target.  MainScene.run()
    detects the active-state transition and issues robot commands accordingly.
    """
    WIN    = "Jog Controller"
    _POS_R = 300      # max trackbar value; centre=150 → ±1.50 m at 1 cm/step
    _ORI_R = 360      # centre=180 → ±180° at 1°/step
    _BTN   = (10, 8, 170, 50)

    def __init__(self):
        cv.namedWindow(self.WIN, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.WIN, 540, 300)
        for lbl, maxv, dflt in [
            ("X  pos  (cm)",  self._POS_R, 150),
            ("Y  pos  (cm)",  self._POS_R, 150),
            ("Z  pos  (cm)",  self._POS_R, 150),
            ("Roll   (deg)", self._ORI_R, 180),
            ("Pitch  (deg)", self._ORI_R, 180),
            ("Yaw    (deg)", self._ORI_R, 180),
        ]:
            cv.createTrackbar(lbl, self.WIN, dflt, maxv, lambda _: None)
        self._active = False
        cv.setMouseCallback(self.WIN, self._on_mouse)

    def _on_mouse(self, event, x, y, *_):
        if event == cv.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = self._BTN
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._active = not self._active

    @property
    def active(self) -> bool:
        return self._active

    def set_active(self, v: bool) -> None:
        self._active = bool(v)

    def get_pos(self) -> np.ndarray:
        x = (cv.getTrackbarPos("X  pos  (cm)", self.WIN) - 150) * 0.01
        y = (cv.getTrackbarPos("Y  pos  (cm)", self.WIN) - 150) * 0.01
        z = (cv.getTrackbarPos("Z  pos  (cm)", self.WIN) - 150) * 0.01
        return np.array([x, y, z])

    def get_quat(self) -> list:
        roll  = cv.getTrackbarPos("Roll   (deg)", self.WIN) - 180
        pitch = cv.getTrackbarPos("Pitch  (deg)", self.WIN) - 180
        yaw   = cv.getTrackbarPos("Yaw    (deg)", self.WIN) - 180
        return ScipyR.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat().tolist()

    def snap_to(self, pos: np.ndarray, quat) -> None:
        """Initialise sliders to a given pose so jog starts from the current TCP."""
        for lbl, val, ctr in [
            ("X  pos  (cm)", pos[0] * 100, 150),
            ("Y  pos  (cm)", pos[1] * 100, 150),
            ("Z  pos  (cm)", pos[2] * 100, 150),
        ]:
            cv.setTrackbarPos(lbl, self.WIN, int(np.clip(round(val) + ctr, 0, self._POS_R)))
        euler = ScipyR.from_quat(quat).as_euler('xyz', degrees=True)
        for lbl, val in [
            ("Roll   (deg)", euler[0]),
            ("Pitch  (deg)", euler[1]),
            ("Yaw    (deg)", euler[2]),
        ]:
            cv.setTrackbarPos(lbl, self.WIN, int(np.clip(round(val) + 180, 0, self._ORI_R)))

    def draw(self) -> None:
        img = np.zeros((70, 540, 3), dtype=np.uint8)
        x0, y0, x1, y1 = self._BTN
        if self._active:
            bg, border, label = (20, 30, 160), (80, 120, 255), "STOP JOG"
        else:
            bg, border, label = (30, 110, 40), (80, 200, 80), "START JOG"
        cv.rectangle(img, (x0, y0), (x1, y1), bg, -1)
        cv.rectangle(img, (x0, y0), (x1, y1), border, 2)
        cv.putText(img, label, (x0 + 6, y0 + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv.LINE_AA)
        pos = self.get_pos()
        roll  = cv.getTrackbarPos("Roll   (deg)", self.WIN) - 180
        pitch = cv.getTrackbarPos("Pitch  (deg)", self.WIN) - 180
        yaw   = cv.getTrackbarPos("Yaw    (deg)", self.WIN) - 180
        cv.putText(img,
                   f"X={pos[0]*100:+.0f}cm  Y={pos[1]*100:+.0f}cm  Z={pos[2]*100:+.0f}cm",
                   (185, 28), cv.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv.putText(img,
                   f"R={roll:+d}°  P={pitch:+d}°  Yaw={yaw:+d}°",
                   (185, 56), cv.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv.imshow(self.WIN, img)

    def close(self) -> None:
        try:
            cv.destroyWindow(self.WIN)
        except Exception:
            pass


# =============================================================================
# MainScene
# =============================================================================


class MainScene:

    _TCP_TOOL_ID     = 200    # must match ToolClickPublisher tool_id in Unity
    # 18.5 cm TCP-to-fingertip length + 10 cm clearance from the hand.
    _PALM_TCP_STANDOFF_M = 0.285
    # Hand the robot delivers grasped objects to (handover after successful grasp).
    _HANDOVER_SIDE = 'right'
    _SYNTH_INTERVAL  = 1.0 / 30.0
    _RELOCK_COOLDOWN = 2.0
    _AUTO_LOCK_MAX_DIST     = 1.0    # metres — auto-lock-on-sight only within this range
    _AUTO_LOCK_MAX_TILT_DEG = 45.0   # degrees — max tilt from vertical to auto-lock
    # _TRACK_DIST_THRESHOLD = 0.075   # metres — TCP-to-target distance considered "arrived"
    # _TRACK_HOLD_SECS      = 0.2   # seconds continuously under threshold before locking grip

    # Robot workspace boundary, relative to the anchor ArUco marker (world frame).
    WORKSPACE_BOUNDS_LO = np.array(cfg.WORKSPACE_LO)
    WORKSPACE_BOUNDS_HI = np.array(cfg.WORKSPACE_HI)

    PEGBOARD_CUBES = [
        {"offset": [ 0.10, 0.00, 0.05], "color": [1.0, 0.6, 0.2], "name": "pegboard_cube_0"},
        {"offset": [ 0.00, 0.10, 0.05], "color": [0.8, 0.2, 0.8], "name": "pegboard_cube_1"},
        {"offset": [-0.10, 0.00, 0.05], "color": [0.2, 1.0, 0.9], "name": "pegboard_cube_2"},
    ]

    def __init__(self, quest_ip: str, anchor_marker_id: int, pegboard_marker_id: int,
                 anchor_marker_size_m: float, pegboard_marker_size_m: float,
                 hand_port: int, robot_ip: str | None = None,
                 simulation: bool = True,
                 board_marker_a: int = cfg.BOARD_MARKER_A_ID,
                 board_marker_b: int = cfg.BOARD_MARKER_B_ID,
                 board_marker_size_m: float | None = None,
                 use_calibrated_robot_base: bool = cfg.USE_CALIBRATED_ROBOT_BASE_POSE,
                 load_pegboard_from_file: bool = cfg.LOAD_PEGBOARD_FROM_FILE,
                 gripper_collision: bool = True,
                 no_passthrough: bool = False,
                 flat_tcp_ori: bool = False):

        self.anchor_marker_id         = anchor_marker_id
        self.pegboard_marker_id       = pegboard_marker_id
        self.hand_port                = hand_port
        self.simulation               = simulation
        self.board_marker_a           = board_marker_a
        self.board_marker_b           = board_marker_b
        self._load_pegboard_from_file   = load_pegboard_from_file
        self._no_passthrough            = no_passthrough

        self._T_BOARD_FROM_MARKER = {
            board_marker_a: T_BOARD_FROM_MARKER_A,
            board_marker_b: T_BOARD_FROM_MARKER_B,
        }

        # ── Receivers / publishers ────────────────────────────────────────────
        self.cam          = _CamFeedReceiver(quest_ip)
        # Secondary relock markers come from the prescan file (single source of
        # truth) — the detector, the relock loop, and the Unity cubes all follow it.
        _world_marker_ids = _load_prescan_marker_ids()
        _aruco            = _ArucoPoseEstimator(
                                anchor_marker_id       = anchor_marker_id,
                                pegboard_marker_id     = pegboard_marker_id,
                                anchor_marker_size_m   = anchor_marker_size_m,
                                pegboard_marker_size_m = pegboard_marker_size_m,
                                board_marker_ids       = (*((board_marker_a, board_marker_b)), *_world_marker_ids),
                                board_marker_size_m    = cfg.WORLD_MARKER_SIZE)
        self.aruco_worker = _ArUcoWorker(self.cam, _aruco)
        self.hands        = _HandDataReceiver(quest_ip, hand_port)

        # ── Robot control client (talks to robot_control_server.py) ────────────
        # Real IK/FK for hardware control, RTDE, the gripper, and the frax CBF
        # filter all live in the dedicated robot_control_server.py process —
        # see that file's docstring. self.pb_scene here is the client's local,
        # IK-free visualization scene (robot mesh + reachability arrows only).
        self.robot: "RobotClient | None" = None
        self.pb_scene = None
        if _ROBOT_CTRL_AVAILABLE:
            try:
                self.robot = RobotClient(
                    simulation                = simulation,
                    use_calibrated_robot_base = use_calibrated_robot_base,
                )
                self.pb_scene = self.robot.pb_scene
            except Exception as e:
                print(f"[MainScene] RobotClient failed to connect: {e}")
                self.robot = None

        if self.robot is not None:
            print(f"[Robot] {'Simulation' if simulation else f'Live ({robot_ip})'} mode — "
                  f"RobotClient connected to robot_control_server.py")
        else:
            print("[Robot] No robot control — is robot_control_server.py running?")

        self.anchor      = _WorldAnchor(quest_ip)
        self.tools       = _ToolSelectionManager(quest_ip)
        # Register TCPMarker's resting state immediately, independently of the
        # pegboard/anchor lifecycle. refresh_colors() will keep advertising it
        # until Unity's port-5010 subscriber is connected.
        self.tools.set_category_color(
            _ToolSelectionManager.TCP_TOOL_ID,
            _ToolSelectionManager.TCP_COLOR)
        self.tuner       = _OffsetTuner()
        self.jog_gui     = _JogGUI()
        self.synth       = _SyntheticObjectPublisher(quest_ip)
        # Secondary relock-cube poses → Unity (positions the click cubes on their
        # physical markers using the prescan registration).
        self.relock_cubes = _RelockCubePublisher(quest_ip)
        self.relock_cubes.set_markers(self.anchor._T_world_marker)
        # self.handover_sphere = _HandoverSpherePublisher(quest_ip)
        self.tool_layout = _ToolLayoutManager(
                               cfg.SCENE_LAYOUT_DIR / "tool_layout1.json", quest_ip)
        # tool id → Open3D box index (box order == tool_layout.world_boxes() order), so the pegboard
        # highlight ids from gearbox_control.py can be mirrored onto the local Open3D tool boxes.
        self._tool_id_to_box_index = {int(t["id"]): i
                                      for i, t in enumerate(self.tool_layout._tools)}
        # TEMP DEBUG: send tool category colors immediately at startup, independent
        # of the anchor-lock flow, to isolate whether TCPMarker's gold color is a
        # "never sent" issue vs a Unity-side receiving/wiring issue.
        self._apply_tool_category_colors()

        self.grip_pose_bridge = _GripPoseBridge(quest_ip)
        self.gearbox_pose_rx = _GearboxPoseReceiver(quest_ip)
        self.taskgraph_o3d_rx = _TaskGraphOpen3DReceiver()
        self.workspace_bound_pub = _WorkspaceBoundPublisher(quest_ip)

        # Cubes around the anchor marker (world origin) — ids 0, 1, 2
        self.synth.add([ 0.10, 0.00, 0.05], width=0.06, depth=0.06, height=0.10,
                       color=[1.0, 0.2, 0.2], name="anchor_cube_x")
        self.synth.add([ 0.00, 0.10, 0.05], width=0.06, depth=0.06, height=0.10,
                       color=[0.2, 1.0, 0.2], name="anchor_cube_y")
        self.synth.add([-0.10, 0.00, 0.05], width=0.06, depth=0.06, height=0.10,
                       color=[0.2, 0.4, 1.0], name="anchor_cube_neg_x")

        # TCP marker — id 3; position updated each frame from PyBullet FK
        self._tcp_synth = (self.synth.add([0.0, 0.0, 0.0],
                                          width=0.05, depth=0.05, height=0.05,
                                          color=[1.0, 0.8, 0.2], name="tcp")
                           if self.pb_scene is not None else None)

        # ── Open3D visualizer + OpenCV window ─────────────────────────────────
        self.vis  = _SceneVis(
            f"Hand Tracking — World Frame  (marker #{anchor_marker_id})")
        self._win = (f"Quest Left Passthrough  [ENTER=lock/relock  ESC=quit]"
                     f"  anchor=#{anchor_marker_id}  pegboard=#{pegboard_marker_id}")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, 960, 540)

        # ── Per-iteration state ───────────────────────────────────────────────
        self._prev_relock_available = False  # previous relock-available flag; drives anchor-marker color change
        self._anchor_highlight_until           = 0.0   # time until anchor marker highlight reverts to normal color
        self._tcp_target_T: "np.ndarray | None" = None   # debug: most recently commanded move_tcp/step_hand_track target, drawn by vis.update_tcp_target()
        self._last_proximity_relock_time = 0.0   # timestamp of last auto-relock; enforces _RELOCK_COOLDOWN between relocks
        # Per secondary marker (104-107, …) click-to-relock state — mirrors the
        # anchor marker's relock-cube flow, keyed by marker id. Driven by the
        # prescan file: one entry per marker whose pose was loaded.
        self._world_relock_prev_available:  dict[int, bool]  = {}   # last relock-available flag per marker (drives hover-color swap)
        self._world_relock_highlight_until: dict[int, float] = {}   # time until the SELECTED flash reverts, per marker
        self._last_world_relock_time:       dict[int, float] = {}   # timestamp of last relock per marker; enforces WORLD_MARKERS_RELOCK_COOLDOWN
        for _mid in self.anchor._T_world_marker:
            self._world_relock_prev_available[_mid]  = False
            self._world_relock_highlight_until[_mid] = 0.0
            self._last_world_relock_time[_mid]       = 0.0
        self._reachability_arrows_hide_at           = 0.0   # timestamp after which reachability arrows are hidden; set to now+5s on R keypress
        self._T_world_tcp: "np.ndarray | None"          = None   # current TCP pose in world; polled each frame from robot
        self._tcp_target_T: "np.ndarray | None"         = None   # visualised TCP target pose (None = no target shown)
        self.flat_tcp_ori: bool                          = flat_tcp_ori  # True → gripper horizontal (XY-plane); False → match palm normal via _palm_quat
        self._robot_state: "str | None"                  = None   # None | 'moving_to_pose' | grasp phases
        self._motion_source: "str | None"                = None   # None | 'hand' | 'jog' | 'object'
        self._tracking_hand_side: "str | None"           = None   # 'left' | 'right' while tracking
        self._last_ar_board_T: "np.ndarray | None"    = None   # last AR box pose from _GripPoseBridge; persists between polls
        self._pending_grasp_tool_id: "int | None"           = None   # tool ID being grasped (set on click, not yet read back)
        self._grasped_objects: list                          = []     # [(id, name)] of successfully grasped objects (cancelled grasps excluded)
        self._handed_over_objects: list                      = []     # [(id, name)] released successfully to the human
        self._handover_tool_id: "int | None"                 = None   # object currently awaiting/in handover
        self._last_left_pts:  "list | None"                 = None   # most recent left  hand joints from world_joints()
        self._last_right_pts: "list | None"                 = None   # most recent right hand joints from world_joints()
        self._pending_handover: bool                         = False  # set True after successful grasp to trigger handover on next frame
        self._synth_cubes_added                  = False  # True once PEGBOARD_CUBES have been added to synth._objects
        self._synth_cube_start_idx: "int | None"     = None   # index into synth._objects where the PEGBOARD_CUBES entries begin
        self._last_synth_pub_time        = 0.0   # timestamp of last synth-object publish; throttled to _SYNTH_INTERVAL
        self._fps_ref_time         = time.perf_counter()  # reference time for FPS averaging
        self._fps_frame_count      = 0                    # frame counter since last FPS print
        self._prev_jog_active:    bool                  = False   # tracks jog_gui.active edge
        self._jog_last_pos: "np.ndarray | None"         = None    # last pos issued to move_to_pose in jog mode
        self._jog_diag_t:   float                       = 0.0     # throttle for jog diagnostic prints
        self._last_vis_pegboard_T: "np.ndarray | None" = None

        if self._load_pegboard_from_file:
            self._preview_pegboard_from_file()

        print(f"\n[Running]  quest_ip={quest_ip}  "
              f"anchor_marker=#{anchor_marker_id}  "
              f"pegboard_marker=#{pegboard_marker_id}  "
              f"hand_port={hand_port}")
        print(f"  Marker #{anchor_marker_id} auto-locks world + scene on first sight "
              f"within {self._AUTO_LOCK_MAX_DIST:.1f}m (or press ENTER from any "
              f"distance) — later re-locks need ENTER or the relock cube")
        print(f"  ENTER with marker #{pegboard_marker_id} visible (after locking)"
              f" → lock pegboard")
        _secondary = sorted(self.anchor._T_world_marker) or "none (no prescan file)"
        print(f"  Secondary relock cubes {_secondary}: click one for drift "
              f"correction when it lights up (proximity <{cfg.WORLD_MARKERS_PROXIMITY_MAX}m, "
              f"looking within {cfg.WORLD_MARKERS_TILT_MAX_DEG}° of face-on, "
              f"cooldown {cfg.WORLD_MARKERS_RELOCK_COOLDOWN}s)")
        print("  ESC = quit\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _try_load_pegboard_from_file(self) -> bool:
        """Load pegboard pose from scene_layout NPZ, inject it into the anchor,
        and publish the tool layout + PyBullet boxes immediately.
        Returns True on success. Only meaningful once the anchor is locked."""
        npz_path = cfg.SCENE_LAYOUT_DIR / "T_world10_pegboard101.npz"
        if not npz_path.exists():
            print(f"[PegboardFile] Not found: {npz_path}")
            return False
        try:
            data = np.load(npz_path)
            self.anchor.set_pegboard(data["T_world10_pegboard"])
            self.vis.set_pegboard_outline(
                offset_x=float(data["marker_offset_right_m"]),
                offset_y=float(data["marker_offset_top_m"]),
                width=float(data["pegboard_width_m"]),
                height=float(data["pegboard_height_m"]),
            )
        except Exception as e:
            print(f"[PegboardFile] Load failed: {e}")
            return False
        T_wp = self.anchor.T_pegboard_in_world
        if T_wp is not None:
            self.tool_layout.publish(T_wp)
            self._apply_tool_category_colors()
            boxes = self.tool_layout.world_boxes(T_wp)
            self.vis.update_tool_boxes(boxes)
        return True

    def _preview_pegboard_from_file(self) -> bool:
        """Draw the saved pegboard in Open3D before marker 100 is locked.

        This is local-only: Unity and the robot do not receive this pose until
        the anchor locks and _try_load_pegboard_from_file() runs normally.
        """
        npz_path = cfg.SCENE_LAYOUT_DIR / "T_world10_pegboard101.npz"
        if not npz_path.exists():
            print(f"[PegboardPreview] Not found: {npz_path}")
            return False
        try:
            data = np.load(npz_path)
            T_wp = np.asarray(data["T_world10_pegboard"], dtype=np.float64)
            self.vis.set_pegboard_outline(
                offset_x=float(data["marker_offset_right_m"]),
                offset_y=float(data["marker_offset_top_m"]),
                width=float(data["pegboard_width_m"]),
                height=float(data["pegboard_height_m"]),
            )
            self.vis.update_pegboard(T_wp)
            self.vis.update_tool_boxes(self.tool_layout.world_boxes(T_wp))
            self._last_vis_pegboard_T = T_wp.copy()
            self._sync_vis_highlight()
        except Exception as e:
            print(f"[PegboardPreview] Load failed: {e}")
            return False
        print(f"[PegboardPreview] Showing saved pegboard in Open3D before marker "
              f"#{self.anchor_marker_id} locks")
        return True

    def _sync_vis_highlight(self) -> None:
        """Mirror the pegboard tool highlight (received by self.tools on 5024 and applied to the
        Unity tools) onto the local Open3D tool boxes — cyan, exactly like gearbox_control --open-3d.
        Always on; no argument gates it."""
        handed_over = {tid for tid, _ in self._handed_over_objects}
        idxs = [self._tool_id_to_box_index[tid]
                for tid in self.tools.highlighted
                if tid in self._tool_id_to_box_index and tid not in handed_over]
        self.vis.set_tool_highlight_indices(idxs)

    def _record_grasped_object(self, tool_id: int) -> None:
        """Record a successful grasp; do not hide it until handover release."""
        name = self.tool_layout.get_name(tool_id)
        if not any(tid == tool_id for tid, _ in self._grasped_objects):
            self._grasped_objects.append((tool_id, name))
        self._handover_tool_id = tool_id

    def _record_handed_over_object(self, tool_id: int) -> None:
        """Record confirmed release and remove its Open3D pegboard box."""
        name = self.tool_layout.get_name(tool_id)
        if not any(tid == tool_id for tid, _ in self._handed_over_objects):
            self._handed_over_objects.append((tool_id, name))
        self.tool_layout.mark_delivered(tool_id)
        hidden = [self._tool_id_to_box_index[tid]
                  for tid, _ in self._handed_over_objects
                  if tid in self._tool_id_to_box_index]
        self.vis.set_tool_hidden_indices(hidden)
        T_wp = self.anchor.T_pegboard_in_world
        if T_wp is not None:
            self.vis.update_tool_boxes(self.tool_layout.world_boxes(T_wp))
            # ToolSpawner despawns IDs omitted from a refreshed full layout.
            self.tool_layout.publish(T_wp)

    def _on_object_released(self, ok: bool) -> None:
        tool_id = self._handover_tool_id
        self._robot_state = None
        self._motion_source = None
        self._tracking_hand_side = None
        self._tcp_target_T = None
        if ok and tool_id is not None:
            self._record_handed_over_object(tool_id)
            print(f"[Handover] Released '{self.tool_layout.get_name(tool_id)}' "
                  f"(id={tool_id}) to human — box removed")
            print(f"[Handed over so far] "
                  f"{[n for _, n in self._handed_over_objects]}")
            self._handover_tool_id = None
            self._pending_handover = False
        else:
            print("[Handover] Gripper release failed — object remains visible; retrying")
            self._pending_handover = True

    def _on_object_handover_target_reached(self, ok: bool) -> None:
        self._robot_state = None
        self._motion_source = None
        self._tracking_hand_side = None
        self._tcp_target_T = None
        if ok and self.robot is not None and self._handover_tool_id is not None:
            self._robot_state = 'waiting_for_handover_pull'
            print("[Handover] Hand target reached → hold object; waiting for human pull")
            self.robot.wait_for_handover_pull(on_complete=self._on_object_released)
        else:
            print("[Handover] Hand-target move failed — object remains visible; retrying")
            self._pending_handover = True

    def _apply_tool_category_colors(self) -> None:
        """Send each tool's category color via ToolColorReceiver (port 5010)
        and register it as the resting color so hover/reset cycles preserve it."""
        # TCPMarker pose comes from synthetic-object port 5006, but its
        # GripperWithAdapters appearance is owned exclusively by port 5010.
        self.tools.set_category_color(
            _ToolSelectionManager.TCP_TOOL_ID,
            _ToolSelectionManager.TCP_COLOR)
        for t in self.tool_layout._tools:
            tid  = int(t["id"])
            cat  = t.get("category", "tool")
            col  = (_ToolSelectionManager.PART_COLOR if cat == "part"
                    else _ToolSelectionManager.TOOL_COLOR)
            self.tools.set_category_color(tid, col)

    def _lock_anchor_initial(self, T_cam_anchor: np.ndarray) -> None:
        """First-time anchor lock + the same follow-up steps ENTER/relock run
        (scene origin reset, pegboard-from-file load). Used both by the
        ENTER handler and by the auto-lock-on-sight check in run()."""
        self.anchor.lock(T_cam_anchor, self.cam.camera_T)
        self._last_proximity_relock_time = time.time()
        if self.robot is not None:
            self.robot.set_scene_origin(np.eye(4))
        if self._load_pegboard_from_file:
            self._try_load_pegboard_from_file()

    def _lock_anchor_tracking_origin(self) -> None:
        """--no-passthrough manual lock: pin the world frame to the Quest
        tracking origin (marker 100 assumed to sit at the initial/recenter
        origin) and run the same follow-up steps as a marker lock (scene-origin
        reset, pegboard-from-file load) so everything downstream unlocks
        identically."""
        self.anchor.lock_tracking_origin()
        self._last_proximity_relock_time = time.time()
        if self.robot is not None:
            self.robot.set_scene_origin(np.eye(4))
        if self._load_pegboard_from_file:
            self._try_load_pegboard_from_file()

    def _on_hand_target_reached(self, ok: bool) -> None:
        self._robot_state = None
        self._motion_source = None
        self._tracking_hand_side = None
        self._tcp_target_T = None
        if ok and self.robot is not None:
            if self.simulation:
                print("[Robot] Hand target reached → workholding active (sim)")
            else:
                print("[Robot] Hand target reached → waiting for board contact")
            self.robot.start_board_interaction()
        else:
            print("[Robot] Hand-target move cancelled")

    def _on_board_move_complete(self, ok: bool) -> None:
        _tcp = self._T_world_tcp
        _tgt = self._tcp_target_T
        if _tcp is not None and _tgt is not None:
            pos_err = float(np.linalg.norm(_tcp[:3, 3] - _tgt[:3, 3]))
            R_err   = _tcp[:3, :3].T @ _tgt[:3, :3]
            ang_err = float(ScipyR.from_matrix(R_err).magnitude())
            print(f"[Robot] AR board move {'complete' if ok else 'cancelled'}"
                  f"  target={np.round(_tgt[:3, 3], 3).tolist()}"
                  f"  actual={np.round(_tcp[:3, 3], 3).tolist()}"
                  f"  err={pos_err*100:.1f} cm / {np.rad2deg(ang_err):.1f}°")
        else:
            print(f"[Robot] AR board move {'complete' if ok else 'cancelled'}")
        self._robot_state = None
        self._motion_source = None
        self._tcp_target_T = None

    def _summon_to_hand(self, target_side: str, hand_pts,
                        on_complete=None) -> bool:
        """Move the robot TCP toward a hand palm.

        ``target_side`` is 'left' or 'right'; ``hand_pts`` is the world-frame
        joint list from ``world_joints()``.  Returns True if a move was started.
        """
        if hand_pts is None or self._T_world_tcp is None or self.robot is None:
            print(f"[Handover] Cannot summon to {target_side} hand — "
                  f"hand_pts={'no' if hand_pts is None else 'yes'}, "
                  f"tcp={'no' if self._T_world_tcp is None else 'yes'}")
            return False
        is_left   = (target_side == 'left')
        palm_pos  = np.asarray(hand_pts[1], float)
        if self.flat_tcp_ori:
            tcp_pos      = self._T_world_tcp[:3, 3]
            approach_dir = palm_pos - tcp_pos
            dist_to_palm = float(np.linalg.norm(approach_dir))
            if dist_to_palm > 1e-3:
                approach_dir /= dist_to_palm
            hz = np.array([approach_dir[0], approach_dir[1], 0.0])
            hz_norm = float(np.linalg.norm(hz))
            hz = hz / hz_norm if hz_norm > 1e-3 else np.array([1.0, 0.0, 0.0])
            hy = np.array([0.0, 0.0, 1.0])
            hx = np.cross(hy, hz)
            target_quat = ScipyR.from_matrix(
                np.column_stack([hx, hy, hz])).as_quat().tolist()
            target_pos  = palm_pos - 0.185 * hz
        else:
            target_quat = _palm_quat(hand_pts, is_left=is_left)
            if ScipyR.from_quat(target_quat).apply([0., 1., 0.])[2] > 0:
                target_quat = (ScipyR.from_quat(target_quat)
                               * ScipyR.from_euler('z', 180, degrees=True)).as_quat()
            gripper_z   = ScipyR.from_quat(target_quat).apply([0., 0., 1.])
            centroid    = (np.asarray(hand_pts[3], float)
                           + np.asarray(hand_pts[1], float)
                           + np.asarray(hand_pts[6], float)) / 3.0
            target_pos  = centroid - gripper_z * self._PALM_TCP_STANDOFF_M
            target_quat = target_quat.tolist()
        target_pos = np.asarray(
            cfg.project_robot_target_position(target_pos, self._T_world_tcp[:3, 3]), float)
        print(f"[Handover] Move to {target_side} hand "
              f"(palm={np.round(palm_pos, 3).tolist()}, "
              f"target={np.round(target_pos, 3).tolist()})")
        self._robot_state        = 'moving_to_pose'
        self._motion_source      = 'hand'
        self._tracking_hand_side = target_side
        _T_tgt               = np.eye(4)
        _T_tgt[:3, 3]        = target_pos
        _T_tgt[:3, :3]       = ScipyR.from_quat(target_quat).as_matrix()
        self._tcp_target_T   = _T_tgt
        self.robot.move_to_pose(
            target_pos, target_quat,
            on_complete=on_complete or self._on_hand_target_reached,
        )
        return True

    def _board_allows_unrelated_motion(self) -> bool:
        """A simulated hold is only an AR affordance, not a physical lock."""
        if self.robot is None:
            return False
        return (self.robot.board_state == "inactive"
                or (self.simulation
                    and self.robot.board_state == "holding_board"))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while True:
                _iter_t0 = time.perf_counter()

                # ── Poll streams ──────────────────────────────────────────────
                self.tools.poll(timeout_ms=0)
                # PUB/SUB has no retained state; repeat the current colors at a
                # low rate so Unity also receives them after a late connection.
                self.tools.refresh_colors()
                if self.tools.drain_highlights(timeout_ms=0):
                    self._sync_vis_highlight()   # mirror the cyan highlight onto the Open3D boxes
                    for event in self.tools.pop_assembly_events():
                        self.vis.apply_gearbox_assembly_event(event)
                self.hands.poll()
                gearbox_states = self.gearbox_pose_rx.poll()
                for event in self.taskgraph_o3d_rx.poll():
                    if event.get("event") == "select":
                        self.tools._apply_highlight(event.get("ids", []))
                        self._sync_vis_highlight()
                        self.vis.apply_gearbox_assembly_event(
                            {**event, "event": "show"})
                    else:
                        if event.get("event") == "reset":
                            self.tools._apply_highlight_clear()
                            self._sync_vis_highlight()
                        self.vis.apply_gearbox_assembly_event(event)
                _link_poses = None
                if self.robot is not None:
                    self.robot.poll()   # drain robot_control_server.py state/events
                    self._T_world_tcp = self.robot.tcp_pose
                    _link_poses       = self.robot.arm_link_poses()
                    if self._T_world_tcp is not None and self._tcp_synth is not None:
                        # In hand-tracking mode, show Unity the same projected
                        # target represented by the magenta Open3D pose.  In all
                        # other modes, ID 3 continues to show the actual FK TCP.
                        _T_tcp_synth = (
                            self._tcp_target_T
                            if (self._robot_state == 'moving_to_pose'
                                and self._motion_source == 'hand'
                                and self._tcp_target_T is not None)
                            else self._T_world_tcp
                        )
                        self._tcp_synth.centroid = _T_tcp_synth[:3, 3]
                        self._tcp_synth.R_o3d    = _T_tcp_synth[:3, :3]

                # ── ArUco results (background thread) ─────────────────────────
                T_cam_anchor, T_cam_pegboard, T_cam_board, det_vis = \
                    self.aruco_worker.get()
                anchor_ok   = T_cam_anchor   is not None
                pegboard_ok = T_cam_pegboard is not None
                board_marker_seen = next(
                    (mid for mid in (self.board_marker_a, self.board_marker_b)
                     if mid in T_cam_board), None)
                board_ok = board_marker_seen is not None

                # ── Offset tuner ──────────────────────────────────────────────
                self.tuner.draw()
                pos_off, yaw_off = self.tuner.get()
                self.anchor.set_offset(pos_off, yaw_off)

                # ── Jog GUI — draw + button-state sync ───────────────────────
                self.jog_gui.draw()
                _jog_now = self.jog_gui.active
                if _jog_now and not self._prev_jog_active:
                    # Button just pressed → enter jog mode
                    if self.robot is not None and self._T_world_tcp is not None:
                        if self.robot.move_running:
                            self.robot.cancel_move()
                            self._robot_state        = None
                            self._motion_source      = None
                            self._tracking_hand_side = None
                        self.jog_gui.snap_to(
                            self._T_world_tcp[:3, 3],
                            ScipyR.from_matrix(self._T_world_tcp[:3, :3]).as_quat())
                        self._jog_last_pos = None   # force re-issue next frame
                        print("[Jog] Entered — sliders snapped to current TCP")
                    else:
                        self.jog_gui.set_active(False)
                        print("[Jog] Robot not ready (no TCP pose yet)")
                elif not _jog_now and self._prev_jog_active:
                    # Button just released → exit jog mode
                    if (self._robot_state == 'moving_to_pose'
                            and self._motion_source == 'jog'):
                        self._robot_state  = None
                        self._motion_source = None
                        self._tcp_target_T = None
                        if self.robot is not None:
                            self.robot.cancel_move()
                    self._jog_last_pos = None
                    print("[Jog] Exited")
                self._prev_jog_active = _jog_now

                # ── CenterEye pose ────────────────────────────────────────────
                _center_T = self.hands.center_eye_T()

                # ── Tracked board (markers A/B) — constantly updated ──────────
                if self.anchor.locked and board_ok:
                    self.anchor.update_board_from_tracking(
                        self.cam.camera_T,
                        T_cam_board[board_marker_seen],
                        self._T_BOARD_FROM_MARKER[board_marker_seen])

                # ── Publish world root + pegboard to Unity ────────────────────
                # (robot base/joint publishing to Unity is now handled by
                # robot_control_server.py itself, on its own steady cadence)
                if self.anchor.locked:
                    self.anchor.publish()
                    self.anchor.publish_pegboard()
                    self.anchor.publish_board()
                    _now = time.time()
                    if _now - self._last_synth_pub_time >= self._SYNTH_INTERVAL:
                        self.synth.publish()
                        self.relock_cubes.publish()
                        # self.handover_sphere.publish()
                        self._last_synth_pub_time = _now

                T_wt            = self.anchor.T_world_tracking
                _eff_cam_T      = self.anchor._effective_cam_T(None, _center_T)
                T_world_camleft = (self.anchor.world_T(_eff_cam_T)
                                   if _eff_cam_T is not None else None)
                T_world_center  = (self.anchor.world_T(_center_T)
                                   if _center_T is not None else None)
                left_pts, right_pts = self.hands.world_joints(T_wt)
                self._last_left_pts  = left_pts
                self._last_right_pts = right_pts

                # ── Pending handover after successful grasp ───────────────────
                if (self._pending_handover
                        and self.robot is not None
                        and not self.robot.tool_grasp_running
                        and not self.robot.move_running
                        and self.anchor.locked):
                    _hpts = (self._last_right_pts if self._HANDOVER_SIDE == 'right'
                             else self._last_left_pts)
                    if _hpts is None:
                        # Keep the request pending; begin as soon as the hand is visible.
                        pass
                    else:
                        if self._summon_to_hand(
                                self._HANDOVER_SIDE, _hpts,
                                on_complete=self._on_object_handover_target_reached):
                            self._pending_handover = False

                # ── Workspace boundary (fades in as head/hands approach/exit) ──
                if self.anchor.locked:
                    _head_pos  = T_world_center[:3, 3] if T_world_center is not None else None
                    _left_pos  = left_pts[1]  if left_pts  is not None else None
                    _right_pos = right_pts[1] if right_pts is not None else None
                    _dist_out = max(
                        _WorkspaceBoundPublisher.dist_outside(
                            _head_pos, self.WORKSPACE_BOUNDS_LO, self.WORKSPACE_BOUNDS_HI),
                        _WorkspaceBoundPublisher.dist_outside(
                            _left_pos, self.WORKSPACE_BOUNDS_LO, self.WORKSPACE_BOUNDS_HI),
                        _WorkspaceBoundPublisher.dist_outside(
                            _right_pos, self.WORKSPACE_BOUNDS_LO, self.WORKSPACE_BOUNDS_HI),
                    )
                    self.workspace_bound_pub.publish(
                        self.WORKSPACE_BOUNDS_LO, self.WORKSPACE_BOUNDS_HI, _dist_out)
                    self.vis.update_workspace_bound(
                        self.WORKSPACE_BOUNDS_LO, self.WORKSPACE_BOUNDS_HI)

                # ── Reachability arrow expiry ─────────────────────────────────
                _now = time.time()
                if self._reachability_arrows_hide_at > 0.0 and _now >= self._reachability_arrows_hide_at:
                    self.vis.hide_reachability_arrows()
                    self._reachability_arrows_hide_at = 0.0

                dist_to_anchor = (
                    float(np.linalg.norm(T_cam_anchor[:3, 3]))
                    if anchor_ok and self.cam.camera_T is not None else float('inf'))

                # ── Auto-lock anchor on first sight ───────────────────────────
                # Only fires before the very first lock — once self.anchor.locked
                # is True this is permanently skipped, and all later (re)locks
                # go back to requiring ENTER or the proximity relock cube.
                # Gated on proximity (<0.5m) AND view angle (<30° from vertical):
                # T_cam_anchor[:3, 3] is the marker position in camera frame, so
                # its Z component divided by the distance gives cos(tilt from camera
                # optical axis).  When looking nearly straight down at a flat table
                # marker the marker sits almost dead-ahead along the camera Z → cosine
                # near 1.  If the view is too oblique this stays < cos(30°) and the
                # auto-lock is suppressed until the user is more overhead.
                _cos_tilt = (T_cam_anchor[2, 3] / dist_to_anchor
                             if anchor_ok and dist_to_anchor > 1e-6 else 0.0)
                _min_cos  = np.cos(np.deg2rad(self._AUTO_LOCK_MAX_TILT_DEG))
                if (not self.anchor.locked and anchor_ok
                        and self.cam.camera_T is not None
                        and _center_T is not None
                        and dist_to_anchor < self._AUTO_LOCK_MAX_DIST
                        and _cos_tilt > _min_cos):
                    self._lock_anchor_initial(T_cam_anchor)
                    tilt_deg = float(np.degrees(np.arccos(np.clip(_cos_tilt, -1, 1))))
                    print(f"[AutoLock] Locked world to marker "
                          f"#{self.anchor_marker_id} on sight "
                          f"({dist_to_anchor:.2f} m, {tilt_deg:.1f}° tilt)")

                # ── Anchor marker proximity relock ────────────────────────────
                _relock_available = (self.anchor.locked and anchor_ok
                                     and self.cam.camera_T is not None
                                     and dist_to_anchor < 1.0)

                if self._anchor_highlight_until > 0.0 and _now >= self._anchor_highlight_until:
                    self._anchor_highlight_until = 0.0
                    self._prev_relock_available = not _relock_available

                if self._anchor_highlight_until == 0.0 and _relock_available != self._prev_relock_available:
                    self.tools.send_color(
                        self.anchor_marker_id,
                        _ToolSelectionManager.HOVER_COLOR if _relock_available
                        else _ToolSelectionManager.RESET_COLOR)
                    self._prev_relock_available = _relock_available

                if (self.tools.active_tool_id == self.anchor_marker_id
                        and _relock_available
                        and _now - self._last_proximity_relock_time >= self._RELOCK_COOLDOWN):
                    self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True)
                    if self._load_pegboard_from_file:
                        self._try_load_pegboard_from_file()
                    elif pegboard_ok:
                        self.anchor.update_pegboard_from_tracking(
                            self.cam.camera_T, T_cam_pegboard)
                    if self._synth_cubes_added and self.anchor.T_pegboard_in_world is not None:
                        T_wp = self.anchor.T_pegboard_in_world
                        R_wp = T_wp[:3, :3]
                        for i, cube in enumerate(self.PEGBOARD_CUBES):
                            obj = self.synth._objects[self._synth_cube_start_idx + i]
                            obj.centroid = _transform_point(T_wp, cube["offset"])
                            obj.R_o3d    = R_wp.copy()
                    self.tools.send_color(self.anchor_marker_id,
                                          _ToolSelectionManager.SELECTED_COLOR)
                    self._anchor_highlight_until           = _now + 1.0
                    self._prev_relock_available = True
                    self._last_proximity_relock_time = _now
                    if self.anchor.T_pegboard_in_world is not None:
                        self.tool_layout.publish(self.anchor.T_pegboard_in_world)
                        self._apply_tool_category_colors()
                        _boxes = self.tool_layout.world_boxes(self.anchor.T_pegboard_in_world)
                        self.vis.update_tool_boxes(_boxes)
                    print("[AutoRelock] Relocked via proximity click")
                self.tools.deselect(self.anchor_marker_id)

                # ── Secondary marker click-to-relock (cube logic, per marker) ──
                # Each prescanned marker (104-107, …) has an authored relock cube
                # in Unity tagged with its marker id. Same flow as the anchor cube:
                # the cube lights up (HOVER_COLOR) when you are looking nearly
                # face-on at the physical marker within range, and clicking it
                # relocks the world frame from that marker's prescan registration.
                for mid in self.anchor._T_world_marker:
                    T_cam_mid = T_cam_board.get(mid)
                    _available = False
                    _dist = _angle_deg = 0.0
                    if (self.anchor.locked and self.cam.camera_T is not None
                            and T_cam_mid is not None):
                        _dist = float(np.linalg.norm(T_cam_mid[:3, 3]))
                        # Obliquity: T_cam_mid[2, 2] is the marker normal's component
                        # along the camera optical axis — |value| ≈ 1 → face-on (0°),
                        # ≈ 0 → edge-on/grazing (90°).
                        _cos_obliq = abs(float(T_cam_mid[2, 2]))
                        _min_cos   = np.cos(np.deg2rad(cfg.WORLD_MARKERS_TILT_MAX_DEG))
                        _angle_deg = float(np.degrees(np.arccos(np.clip(_cos_obliq, -1, 1))))
                        _available = (_dist <= cfg.WORLD_MARKERS_PROXIMITY_MAX
                                      and _cos_obliq >= _min_cos)

                    # Revert the post-relock SELECTED flash once it expires
                    if (self._world_relock_highlight_until[mid] > 0.0
                            and _now >= self._world_relock_highlight_until[mid]):
                        self._world_relock_highlight_until[mid] = 0.0
                        self._world_relock_prev_available[mid] = not _available

                    # Swap the cube colour when availability changes (hover-on / reset)
                    if (self._world_relock_highlight_until[mid] == 0.0
                            and _available != self._world_relock_prev_available[mid]):
                        self.tools.send_color(
                            mid,
                            _ToolSelectionManager.HOVER_COLOR if _available
                            else _ToolSelectionManager.RESET_COLOR)
                        self._world_relock_prev_available[mid] = _available

                    # Click on the cube while available → relock world from this marker
                    if (self.tools.active_tool_id == mid
                            and _available
                            and _now - self._last_world_relock_time[mid]
                                >= cfg.WORLD_MARKERS_RELOCK_COOLDOWN):
                        if self.anchor.relock_from_world_marker(
                                mid, T_cam_mid, self.cam.camera_T):
                            self.tools.send_color(mid, _ToolSelectionManager.SELECTED_COLOR)
                            self._world_relock_highlight_until[mid] = _now + 1.0
                            self._world_relock_prev_available[mid]  = True
                            self._last_world_relock_time[mid]       = _now
                            print(f"[WorldRelock] marker #{mid}, {_dist:.3f}m, "
                                  f"{_angle_deg:.1f}° off-normal (click)")
                    self.tools.deselect(mid)

                # ── TCP click (tool_id 200) → move_to_pose ────────────────────
                if self.tools.active_tool_id == self._TCP_TOOL_ID:
                    clicking_hand = self.tools.active_hand
                    self.tools.deselect(self._TCP_TOOL_ID)
                    print(f"[TCP] Gripper clicked (hand={clicking_hand})"
                          f"  robot={self.robot is not None}"
                          f"  move_running={self.robot.move_running if self.robot else 'N/A'}"
                          f"  grasp_running={self.robot.tool_grasp_running if self.robot else 'N/A'}"
                          f"  anchor_locked={self.anchor.locked}"
                          f"  left_pts={'yes' if left_pts is not None else 'NO'}"
                          f"  right_pts={'yes' if right_pts is not None else 'NO'}"
                          f"  tcp={'yes' if self._T_world_tcp is not None else 'NO'}")
                    if self.robot is not None and self.robot.move_running:
                        print("[TCP] Gripper clicked — cancelling move_to_pose")
                        self.robot.cancel_move()
                        self._robot_state        = None
                        self._motion_source      = None
                        self._tracking_hand_side = None
                    elif (self.robot is not None
                          and not self.robot.tool_grasp_running
                          and self._board_allows_unrelated_motion()
                          and self.anchor.locked):
                        opposing = "left" if clicking_hand == "right" else "right"
                        _summon_pts = left_pts if opposing == "left" else right_pts
                        self._summon_to_hand(opposing, _summon_pts)

                # ── Tool click → grasp ────────────────────────────────────────
                _tid = self.tools.active_tool_id
                if (_tid is not None
                        and any(done_id == _tid
                                for done_id, _ in self._handed_over_objects)):
                    print(f"[User] Ignoring id={_tid}; object was already handed over")
                    self.tools.deselect(_tid)
                    _tid = None
                if (_tid is not None
                        and self._pending_grasp_tool_id is not None):
                    if _tid == self._pending_grasp_tool_id:
                        print(f"[User] Clicked active grasp id={_tid} again "
                              f"→ cancel and retract")
                        if self.robot is not None:
                            self.robot.cancel_grasp()
                    else:
                        print(f"[User] Ignoring id={_tid}; grasp "
                              f"id={self._pending_grasp_tool_id} is active")
                    self.tools.deselect(_tid)
                    _tid = None

                if (_tid is not None
                        and _tid != self._TCP_TOOL_ID
                        and self.anchor.locked
                        and self.anchor.T_pegboard_in_world is not None
                        and self.robot is not None
                        and self._board_allows_unrelated_motion()
                        and not self.robot.tool_grasp_running):
                    tool_data = self.tool_layout.get_world_data(
                        _tid, self.anchor.T_pegboard_in_world)
                    if tool_data is not None:
                        _grasp_tid = _tid
                        _centroid, _R_world, _sz = tool_data
                        self.vis.update_tool_quat_debug(_centroid, _R_world, _sz)
                        _gj   = self.tool_layout.get_grasp_joints(_tid)
                        _cat  = self.tool_layout.get_category(_tid)
                        _T_wp = self.anchor.T_pegboard_in_world
                        _hw   = 'sim' if self.simulation else 'real'
                        _name = self.tool_layout.get_name(_tid)
                        _seq  = 'approach→grasp→lift→above_approach' if _cat == 'part' else 'approach→grasp→retract'
                        if _gj is not None:
                            print(f"[User] Clicked {_cat} '{_name}' (id={_tid}) → grasp sequence  "
                                  f"[{_hw}] joint-space moveJ  |  {_seq}")
                            self._robot_state = 'approaching'
                            self._motion_source = None
                            self._pending_grasp_tool_id = _grasp_tid
                            self.robot.execute_grasp(
                                _gj,
                                category     = _cat,
                                tool_type    = _name,
                                board_normal = _T_wp[:3, 2] if _T_wp is not None else None,
                                on_phase     = lambda phase: (setattr(self, '_robot_state', phase),
                                                          print(f"[Robot] state → {phase}")),
                                on_complete  = lambda ok, tid=_grasp_tid: (
                                    self.tools.reset_to_category(tid),
                                    setattr(self, '_robot_state', None),
                                    setattr(self, '_motion_source', None),
                                    setattr(self, '_pending_grasp_tool_id', None),
                                    self._record_grasped_object(tid) if ok else None,
                                    print(f"[Robot] Grasp '{self.tool_layout.get_name(tid)}' (id={tid}) — "
                                          f"{'OK ✓ (object in gripper)' if ok else 'FAILED ✗ (empty) — not tracked'}"),
                                    print(f"[Grasped so far] {[n for _, n in self._grasped_objects]}") if ok else None,
                                    setattr(self, '_pending_handover', True) if ok else None,
                                ),
                            )
                        else:
                            print(f"[User] Clicked {_cat} '{_name}' (id={_tid}) — no grasp_joints recorded, skipping")
                    self.tools.deselect(_tid)

                # ── Hand tracking update ──────────────────────────────────────
                if (self._robot_state == 'moving_to_pose'
                        and self._motion_source == 'hand'
                        and self._tracking_hand_side is not None
                        and self.robot is not None):
                    _track_pts = (left_pts if self._tracking_hand_side == 'left'
                                  else right_pts)
                    if _track_pts is not None:
                        _palm = np.asarray(_track_pts[1], float)
                        if self.flat_tcp_ori:
                            _tcp_pos     = self._T_world_tcp[:3, 3] if self._T_world_tcp is not None else _palm
                            _app         = _palm - _tcp_pos
                            _app_norm    = float(np.linalg.norm(_app))
                            _hz          = _app / _app_norm if _app_norm > 1e-3 else np.array([1., 0., 0.])
                            _hz          = np.array([_hz[0], _hz[1], 0.])
                            _hz_n        = float(np.linalg.norm(_hz))
                            _hz          = _hz / _hz_n if _hz_n > 1e-3 else np.array([1., 0., 0.])
                            _hy          = np.array([0., 0., 1.])
                            _hx          = np.cross(_hy, _hz)
                            _tq          = ScipyR.from_matrix(np.column_stack([_hx, _hy, _hz])).as_quat().tolist()
                            _tp          = (_palm - 0.185 * _hz).tolist()
                        else:
                            _is_left = (self._tracking_hand_side == 'left')
                            _tq      = _palm_quat(_track_pts, is_left=_is_left)
                            if ScipyR.from_quat(_tq).apply([0., 1., 0.])[2] > 0:
                                _tq = (ScipyR.from_quat(_tq)
                                       * ScipyR.from_euler('z', 180, degrees=True)).as_quat()
                            _gz       = ScipyR.from_quat(_tq).apply([0., 0., 1.])
                            _centroid = (np.asarray(_track_pts[3], float)
                                         + np.asarray(_track_pts[1], float)
                                         + np.asarray(_track_pts[6], float)) / 3.0
                            _tp = (_centroid - _gz
                                   * self._PALM_TCP_STANDOFF_M).tolist()
                            _tq  = _tq.tolist()
                        _target_origin = (self._T_world_tcp[:3, 3]
                                          if self._T_world_tcp is not None else None)
                        _tp = cfg.project_robot_target_position(
                            _tp, _target_origin)
                        self.robot.update_move_target(_tp, _tq)
                        _T_tgt              = np.eye(4)
                        _T_tgt[:3, 3]       = _tp
                        _T_tgt[:3, :3]      = ScipyR.from_quat(_tq).as_matrix()
                        self._tcp_target_T  = _T_tgt

                # ── Visualizer update ─────────────────────────────────────────
                # Board AR manipulation (5012 to Unity, 5013 from Unity).
                if self.robot is not None:
                    _board_state = self.robot.board_state
                    # Drain every frame so poses sent outside the held state do
                    # not become stale commands when a board is grasped later.
                    # While a board move is active, however, a newly released
                    # AR pose replaces the current target in place.
                    _T_box_target = self.grip_pose_bridge.poll()
                    _local_board_move = (
                        self._robot_state == "moving_to_pose"
                        and self._motion_source == "object")
                    _board_move_active = (
                        _board_state == "moving_board"
                        or _local_board_move)
                    if (_T_box_target is not None
                            and (_board_state == "holding_board"
                                 or _board_move_active)):
                        self._last_ar_board_T = _T_box_target
                        _tcp_pos = (_T_box_target[:3, 3]
                                    - cfg.BOX_FORWARD_OFFSET
                                    * _T_box_target[:3, 2])
                        _target_origin = (self._T_world_tcp[:3, 3]
                                          if self._T_world_tcp is not None
                                          else None)
                        _tcp_pos = np.asarray(
                            cfg.project_robot_target_position(
                                _tcp_pos, _target_origin), float)
                        _tcp_quat = ScipyR.from_matrix(
                            _T_box_target[:3, :3]).as_quat()
                        self._robot_state = "moving_to_pose"
                        self._motion_source = "object"
                        _T_tcp_target = np.eye(4)
                        _T_tcp_target[:3, 3] = _tcp_pos
                        _T_tcp_target[:3, :3] = _T_box_target[:3, :3]
                        self._tcp_target_T = _T_tcp_target
                        _cur_tcp = (self._T_world_tcp[:3, 3]
                                    if self._T_world_tcp is not None else None)
                        _dist_str = (f"  dist={np.linalg.norm(_tcp_pos - _cur_tcp)*100:.1f} cm"
                                     if _cur_tcp is not None else "")
                        if _board_move_active:
                            print(f"[Board AR] Updated target → TCP "
                                  f"{np.round(_tcp_pos, 3).tolist()}{_dist_str}")
                            self.robot.update_move_target(
                                _tcp_pos, _tcp_quat)
                        else:
                            print(f"[Board AR] Released target → TCP "
                                  f"{np.round(_tcp_pos, 3).tolist()}{_dist_str}"
                                  f"  board={_board_state}")
                            self.robot.move_to_pose(
                                _tcp_pos, _tcp_quat,
                                board_move=True,
                                on_complete=self._on_board_move_complete)

                    if self._T_world_tcp is not None:
                        if (_board_state == "moving_board"
                                or (self._robot_state == "moving_to_pose"
                                    and self._motion_source == "object")):
                            _grip_visual_state = "moving"
                        elif _board_state == "holding_board":
                            _grip_visual_state = "grabbed"
                        else:
                            _grip_visual_state = "idle"
                        self.grip_pose_bridge.publish(
                            _grip_visual_state, self._T_world_tcp)

                if self._T_world_tcp is not None:
                    self.vis.update_tcp(self._T_world_tcp)
                self.vis.update_tcp_target(self._tcp_target_T)
                self.vis.update_gripper_tip_target(self._tcp_target_T)
                if _link_poses is not None:
                    self.vis.update_robot(_link_poses)
                if self._last_ar_board_T is not None:
                    self.vis.update_board_manip_debug(self._last_ar_board_T)

                # ── Update Open3D visualizer ───────────────────────────────────
                if self.cam.fx is not None:
                    fx, fy, cx, cy = _adapt_cx_cy(
                        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy,
                        self.cam.sensor_width, self.cam.sensor_height,
                        self.cam.width, self.cam.height)
                    self.vis.update_cam_frustum(T_world_camleft,
                                                self.cam.width, self.cam.height,
                                                fx, fy, cx, cy)
                    T_world_passthrough = (self.anchor.world_T(self.cam.camera_T)
                                           if self.cam.camera_T is not None else None)
                    self.vis.update_passthrough_cam(T_world_passthrough,
                                                    self.cam.width, self.cam.height,
                                                    fx, fy, cx, cy)
                T_vis_pegboard = self.anchor.T_pegboard_in_world
                if T_vis_pegboard is not None:
                    self.vis.update_pegboard(T_vis_pegboard)
                    if (self._last_vis_pegboard_T is None
                            or not np.allclose(T_vis_pegboard,
                                               self._last_vis_pegboard_T,
                                               atol=1e-7)):
                        self.vis.update_tool_boxes(
                            self.tool_layout.world_boxes(T_vis_pegboard))
                        self._last_vis_pegboard_T = T_vis_pegboard.copy()
                self.vis.update_board(self.anchor.T_board_in_world)
                self.vis.update_gearbox_mirror(gearbox_states)
                self.vis.update_tracking(T_wt)
                self.vis.update_head(T_world_center)
                self.vis.update_hands(left_pts, right_pts)
                self.vis.update_palm_triangles(left_pts, right_pts)
                # Palm quat debug — always visible; prefer right hand, fall back to left
                if (self._robot_state == 'moving_to_pose'
                        and self._motion_source == 'hand'):
                    _dbg_pts  = (left_pts if self._tracking_hand_side == "left"
                                 else right_pts)
                    _dbg_left = self._tracking_hand_side == "left"
                else:
                    _dbg_pts  = right_pts if right_pts is not None else left_pts
                    _dbg_left = (right_pts is None and left_pts is not None)
                self.vis.update_palm_quat_debug(_dbg_pts, is_left=_dbg_left)
                self.vis.tick()

                # ── OpenCV display ────────────────────────────────────────────
                disp = cv.resize(
                    det_vis if det_vis is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8),
                    (960, 540))
                locked = self.anchor.locked
                cv.putText(disp,
                           f"Marker #{self.anchor_marker_id}: "
                           f"{'DETECTED' if anchor_ok else 'searching...'}    "
                           f"Marker #{self.pegboard_marker_id}: "
                           f"{'DETECTED' if pegboard_ok else 'searching...'}",
                           (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 255, 80) if anchor_ok else (0, 80, 255), 2)
                cv.putText(disp,
                           f"Anchor: "
                           f"{'LOCKED' if locked else 'waiting for marker #' + str(self.anchor_marker_id)}"
                           f"{f'  :{cfg.WORLD_ROOT_PORT} + :{cfg.PEGBOARD_ROOT_PORT} + :{cfg.SYNTH_OBJECTS_PORT}' if locked else ''}",
                           (12, 68), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 255, 150) if locked else (100, 100, 100), 2)
                if self.anchor.T_pegboard_in_world is not None:
                    t = self.anchor.T_pegboard_in_world[:3, 3]
                    cv.putText(disp,
                               f"Pegboard: ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) m",
                               (12, 102), cv.FONT_HERSHEY_SIMPLEX, 0.60,
                               (80, 255, 80), 2)
                hand_ok = left_pts is not None or right_pts is not None
                if hand_ok:
                    hand_status = ("L+R" if (left_pts is not None and right_pts is not None)
                                   else ("L" if left_pts is not None else "R"))
                elif self.hands.receiving:
                    hand_status = f"receiving #{self.hands.message_count}"
                else:
                    hand_status = f"waiting on port {self.hand_port}"
                cv.putText(disp,
                           f"Hands: {hand_status}",
                           (12, 136), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 255, 200) if (hand_ok or self.hands.receiving)
                           else (100, 100, 100), 2)
                board_status = (f"marker #{board_marker_seen}" if board_ok
                                else f"searching #{self.board_marker_a}/#{self.board_marker_b}")
                if self.anchor.T_board_in_world is not None:
                    t = self.anchor.T_board_in_world[:3, 3]
                    board_status += (f"  ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) m"
                                     f"  → :{cfg.BOARD_ROOT_PORT}")
                cv.putText(disp,
                           f"Board: {board_status}",
                           (12, 170), cv.FONT_HERSHEY_SIMPLEX, 0.60,
                           (0, 255, 200) if board_ok else (100, 100, 100), 2)
                if self.jog_gui.active:
                    _jp = self.jog_gui.get_pos()
                    cv.putText(disp,
                               f"JOG  ({_jp[0]*100:+.0f}, {_jp[1]*100:+.0f}, {_jp[2]*100:+.0f}) cm"
                               f"  use sliders to move target  |  M or button = stop",
                               (12, disp.shape[0] - 14),
                               cv.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
                else:
                    cv.putText(disp,
                               f"ENTER: lock #{self.anchor_marker_id} (world+scene)  or"
                               f"  lock #{self.pegboard_marker_id} (pegboard)  "
                               f"{'  L=lock@origin' if self._no_passthrough else ''}"
                               f"  M=jog  ESC=quit",
                               (12, disp.shape[0] - 14),
                               cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv.imshow(self._win, disp)

                # ── Key handling ──────────────────────────────────────────────
                key = cv.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('m') or key == ord('M'):
                    # Toggle jog mode — the GUI sync block handles robot commands
                    self.jog_gui.set_active(not self.jog_gui.active)
                elif key == ord('r') or key == ord('R'):
                    if (self.robot is not None
                            and self.anchor.T_pegboard_in_world is not None):
                        T_wp = self.anchor.T_pegboard_in_world
                        _reach_quat = _tool_grasp_quat(T_wp[:3, :3])
                        _base_pos = self.pb_scene.T_world_base[:3, 3]
                        _peg_pos  = T_wp[:3, 3]
                        print(f"[Reachability] robot base @ "
                              f"({_base_pos[0]:+.3f}, {_base_pos[1]:+.3f}, {_base_pos[2]:+.3f})  "
                              f"pegboard @ ({_peg_pos[0]:+.3f}, {_peg_pos[1]:+.3f}, {_peg_pos[2]:+.3f})  "
                              f"dist={np.linalg.norm(_peg_pos - _base_pos):.3f} m")
                        # Blocking request/response to robot_control_server.py —
                        # rare, user-triggered, was already a blocking call.
                        _, _, _reach_pts, _reach_flags = \
                            self.robot.check_reachability(
                                T_wp, target_quat_xyzw=_reach_quat)
                        if len(_reach_pts):
                            _board_normal = T_wp[:3, 2]
                            self.vis.update_reachability_arrows(
                                _reach_pts, _reach_flags, _board_normal)
                            self._reachability_arrows_hide_at = time.time() + 5.0
                    else:
                        print("[R] Pegboard not locked yet — lock it first.")
                elif (key == ord('l') or key == ord('L')) and self._no_passthrough:
                    # --no-passthrough manual lock: pin the world to the Quest
                    # tracking origin without needing marker 100 / passthrough.
                    _was_locked = self.anchor.locked
                    self._lock_anchor_tracking_origin()
                    print(f"[L] {'Relocked' if _was_locked else 'Locked'} world to "
                          f"Quest tracking origin (no passthrough)")
                elif key == 13:  # ENTER
                    if self.cam.camera_T is None:
                        if not self.simulation:
                            print("[ENTER] No camera pose — skipping.")
                    elif _center_T is None:
                        if not self.simulation:
                            print("[ENTER] Head tracking not ready — skipping.")
                    else:
                        # ── Phase A: lock / relock world frame ────────────────
                        if anchor_ok:
                            if self.anchor.locked:
                                self.anchor.lock(T_cam_anchor, self.cam.camera_T, require_locked=True)
                                print(f"[ENTER] Relocked world to marker "
                                      f"#{self.anchor_marker_id}")
                                self._last_proximity_relock_time = _now
                                if self.robot is not None:
                                    self.robot.set_scene_origin(np.eye(4))
                                if self._load_pegboard_from_file:
                                    self._try_load_pegboard_from_file()
                            else:
                                self._lock_anchor_initial(T_cam_anchor)
                                print(f"[ENTER] Locked world to marker "
                                      f"#{self.anchor_marker_id}")
                        elif not self.anchor.locked:
                            print(f"[ENTER] Marker #{self.anchor_marker_id}"
                                  f" not visible — cannot lock.")

                        # ── Phase B: lock pegboard (skipped if loaded from file) ─
                        if not self._load_pegboard_from_file and pegboard_ok and self.anchor.locked:
                            self.anchor.update_pegboard_from_tracking(
                                self.cam.camera_T, None, T_cam_pegboard)
                            T_wp = self.anchor.T_pegboard_in_world
                            if T_wp is not None:
                                R_wp = T_wp[:3, :3]
                                if self._synth_cubes_added:
                                    for i, cube in enumerate(self.PEGBOARD_CUBES):
                                        obj = self.synth._objects[
                                            self._synth_cube_start_idx + i]
                                        obj.centroid = _transform_point(T_wp, cube["offset"])
                                        obj.R_o3d    = R_wp.copy()
                                else:
                                    self._synth_cube_start_idx = len(self.synth._objects)
                                    for cube in self.PEGBOARD_CUBES:
                                        self.synth.add(
                                            _transform_point(T_wp, cube["offset"]),
                                            width=0.06, depth=0.06, height=0.10,
                                            color=cube["color"], R_o3d=R_wp,
                                            name=cube["name"])
                                    self._synth_cubes_added = True
                                    print(f"[Synth] Added {len(self.PEGBOARD_CUBES)}"
                                          f" pegboard cubes at marker "
                                          f"#{self.pegboard_marker_id}")
                                self.tool_layout.publish(T_wp)
                                self._apply_tool_category_colors()
                                _boxes = self.tool_layout.world_boxes(T_wp)
                                self.vis.update_tool_boxes(_boxes)
                        elif pegboard_ok and not self.anchor.locked:
                            print(f"[ENTER] Marker #{self.pegboard_marker_id} visible, "
                                  f"but lock marker #{self.anchor_marker_id} first.")

                # ── Jog slider control ────────────────────────────────────────
                if (self.jog_gui.active
                        and self.robot is not None
                        and self._board_allows_unrelated_motion()):
                    _target_origin = (self._T_world_tcp[:3, 3]
                                      if self._T_world_tcp is not None else None)
                    _jp = np.asarray(cfg.project_robot_target_position(
                        self.jog_gui.get_pos(), _target_origin), float)
                    _jq = self.jog_gui.get_quat()
                    _changed = (self._jog_last_pos is None
                                or np.linalg.norm(_jp - self._jog_last_pos) > 0.004
                                or self._robot_state != 'moving_to_pose'
                                or self._motion_source != 'jog')
                    if _changed:
                        # Target moved (or move completed) — (re)issue move_to_pose
                        self._robot_state  = 'moving_to_pose'
                        self._motion_source = 'jog'
                        self._jog_last_pos = _jp.copy()
                        self.robot.move_to_pose(
                            _jp.tolist(), _jq,
                            on_complete=lambda ok: (
                                setattr(self, '_robot_state', None),
                                setattr(self, '_motion_source', None),
                                print(f"[Jog] reached target"),
                            ),
                        )
                    else:
                        # Target unchanged and move in progress — stream update
                        self.robot.update_move_target(_jp.tolist(), _jq)
                    _T_tgt = np.eye(4)
                    _T_tgt[:3, 3]  = _jp
                    _T_tgt[:3, :3] = ScipyR.from_quat(_jq).as_matrix()
                    self._tcp_target_T = _T_tgt

                    # Diagnostic: every 2 s print target vs actual TCP in world + base frames
                    _now_diag = time.perf_counter()
                    if self._T_world_tcp is not None and _now_diag - self._jog_diag_t >= 2.0:
                        self._jog_diag_t = _now_diag
                        _tcp_now  = self._T_world_tcp[:3, 3]
                        _err      = _jp - _tcp_now
                        _T_wb     = self.robot.pb_scene.T_world_base
                        _base_p   = _T_wb[:3, 3]
                        _Rz180m   = ScipyR.from_euler('z', np.pi).as_matrix()
                        _base_R   = _T_wb[:3, :3] @ _Rz180m
                        _tgt_base = _base_R.T @ (_jp - _base_p)  # frax des_pos
                        _tcp_base = _base_R.T @ (_tcp_now - _base_p)
                        print(f"[Jog diag] world  tgt={np.round(_jp,3)}  tcp={np.round(_tcp_now,3)}"
                              f"  err={np.round(_err*100,1)} cm")
                        print(f"[Jog diag] base   tgt={np.round(_tgt_base,3)}  tcp={np.round(_tcp_base,3)}"
                              f"  base_pos(world)={np.round(_base_p,3)}")

                # ── Perf stats ────────────────────────────────────────────────
                self._fps_frame_count += 1
                _iter_ms = (time.perf_counter() - _iter_t0) * 1000.0
                _elapsed  = time.perf_counter() - self._fps_ref_time
                if _elapsed >= 2.0:
                    _avg_hz = self._fps_frame_count / _elapsed
                    # print(f"[perf] loop {_avg_hz:.1f} Hz | last iter {_iter_ms:.1f} ms")
                    self._fps_ref_time    = time.perf_counter()
                    self._fps_frame_count = 0

                time.sleep(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.aruco_worker.stop()
        self.vis.close()
        if self.pb_scene is not None:
            self.pb_scene.disconnect()
        cv.destroyAllWindows()
        self.tuner.close()
        self.jog_gui.close()
        self.anchor.close()
        self.synth.close()
        self.relock_cubes.close()
        # self.handover_sphere.close()
        self.tool_layout.close()
        self.grip_pose_bridge.close()
        self.gearbox_pose_rx.close()
        self.taskgraph_o3d_rx.close()
        self.workspace_bound_pub.close()
        self.hands.close()
        self.cam.close()
        self.tools.close()
        if self.robot is not None:
            self.robot.close()
        print("[Done]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Human Robot Coassembly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quest-ip",             default=cfg.UNITY_IP)
    ap.add_argument("--anchor-marker",        type=int,   default=cfg.ANCHOR_MARKER_ID,
                    help="ArUco marker ID for world frame + PyBullet scene origin")
    ap.add_argument("--pegboard-marker",      type=int,   default=cfg.PEGBOARD_MARKER_ID,
                    help="ArUco marker ID for pegboard")
    ap.add_argument("--board-marker-a",       type=int,   default=cfg.BOARD_MARKER_A_ID,
                    help="ArUco marker ID on one face of the tracked board")
    ap.add_argument("--board-marker-b",       type=int,   default=cfg.BOARD_MARKER_B_ID,
                    help="ArUco marker ID on the opposite face of the tracked board")
    ap.add_argument("--anchor-marker-size",   type=float, default=cfg.ANCHOR_MARKER_SIZE,
                    help="Side length of the anchor marker in metres")
    ap.add_argument("--pegboard-marker-size", type=float, default=cfg.PEGBOARD_MARKER_SIZE,
                    help="Side length of the pegboard marker in metres")
    ap.add_argument("--board-marker-size",    type=float, default=cfg.BOARD_MARKER_SIZE,
                    help="Side length of board markers (A/B) in metres")
    ap.add_argument("--hand-port",            type=int,   default=cfg.HAND1_PORT_FROM_UNITY)
    ap.add_argument("--robot-ip",             default=cfg.ROBOT_IP,
                    help="UR robot controller IP for live joint angles via RTDE")
    ap.add_argument("--simulation",      action=argparse.BooleanOptionalAction, default=cfg.SIMULATION,
                    help="Use fixed default joint angles (--simulation) or live RTDE (--no-simulation)")
    ap.add_argument("--calibrated-robot-base", action=argparse.BooleanOptionalAction,
                    default=cfg.USE_CALIBRATED_ROBOT_BASE_POSE,
                    help="Load robot base pose from calibration_data/ even in simulation mode")
    ap.add_argument("--load-pegboard-from-file", action=argparse.BooleanOptionalAction,
                    default=cfg.LOAD_PEGBOARD_FROM_FILE,
                    help="Auto-load pegboard pose from scene_layout NPZ on anchor lock "
                         "(skips needing marker 101 visible)")
    ap.add_argument("--gripper-collision", action=argparse.BooleanOptionalAction, default=True,
                    help="Include gripper spheres in CBF self-collision model (--gripper-collision / --no-gripper-collision)")
    ap.add_argument("--no-passthrough", dest="no_passthrough", action="store_true",
                    help="Passthrough/ArUco unavailable: enable a manual world lock to the "
                         "Quest tracking origin via the 'l' key (marker 100 assumed to sit at "
                         "the initial/recenter origin). Everything downstream unlocks exactly "
                         "as it does on a marker lock.")
    args = ap.parse_args()
    if args.anchor_marker == args.pegboard_marker:
        ap.error("--anchor-marker and --pegboard-marker must be different.")
    if args.board_marker_a == args.board_marker_b:
        ap.error("--board-marker-a and --board-marker-b must be different.")
    if len({args.anchor_marker, args.pegboard_marker,
            args.board_marker_a, args.board_marker_b}) != 4:
        ap.error("--anchor-marker, --pegboard-marker, --board-marker-a and "
                 "--board-marker-b must all be different.")
    scene = MainScene(
        quest_ip                   = args.quest_ip,
        anchor_marker_id           = args.anchor_marker,
        pegboard_marker_id         = args.pegboard_marker,
        anchor_marker_size_m       = args.anchor_marker_size,
        pegboard_marker_size_m     = args.pegboard_marker_size,
        hand_port                  = args.hand_port,
        robot_ip                   = args.robot_ip,
        simulation                 = args.simulation,
        board_marker_a             = args.board_marker_a,
        board_marker_b             = args.board_marker_b,
        board_marker_size_m        = args.board_marker_size,
        use_calibrated_robot_base  = args.calibrated_robot_base,
        load_pegboard_from_file    = args.load_pegboard_from_file,
        gripper_collision          = args.gripper_collision,
        no_passthrough             = args.no_passthrough)
    scene.run()


if __name__ == "__main__":
    main()
