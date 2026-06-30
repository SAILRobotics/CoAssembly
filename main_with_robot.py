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
import numpy as np
import open3d as o3d
import zmq
from scipy.spatial.transform import Rotation as ScipyR

try:
    from pybullet_ik import IKScene as PyBulletScene
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False
    print("[main_with_robot] pybullet_ik import failed — IK/FK disabled.")

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
    BOARD_SIZE, T_BOARD_FROM_MARKER_A, T_BOARD_FROM_MARKER_B,
)
import main_setting as cfg

try:
    from robot_controller import RobotController as _RobotController
    _ROBOT_CTRL_AVAILABLE = True
except ImportError as _e:
    _ROBOT_CTRL_AVAILABLE = False
    print(f"[main_with_robot] RobotController not available: {_e}")


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
                self._det_vis        = det["vis"]

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

    _MAX_HAND_HEAD_DIST = 1.0   # metres — reject hands further than this from head

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

        def _resolve(real_key, synth_key):
            j = _extract_joints(hands.get(real_key))
            if j is None:
                j = _extract_joints(hands.get(synth_key))
            if j is None:
                return None
            pts = (_to_world(j, T_world_tracking) if T_world_tracking is not None
                   else _unity_to_o3d(j))
            if (head_world is not None
                    and np.linalg.norm(pts[1] - head_world) > self._MAX_HAND_HEAD_DIST):
                return None
            return pts

        return _resolve("LeftHand", "LeftHandSynth"), _resolve("RightHand", "RightHandSynth")

    def close(self):
        try:
            self._sub.close(0)
        except Exception:
            pass



# =============================================================================
# World anchor
# =============================================================================

class _WorldAnchor:
    _EYE_OFFSET_FILE = cfg.SCENE_LAYOUT_DIR / "eye_offset_calibration.json"

    def __init__(self, pub_ip: str, pub_port: int = cfg.WORLD_ROOT_PORT,
                 pegboard_pub_port: int = cfg.PEGBOARD_ROOT_PORT,
                 board_pub_port: int = cfg.BOARD_ROOT_PORT):
        self._T_wt: np.ndarray | None = None  #  This gives you where the marker is in tracking space, and inverting that gives you the transform that converts from tracking space into world space
        self._T_offset = np.eye(4, dtype=np.float64) #From offset tuner
        self._T_world_pegboard: np.ndarray | None = None
        self._T_world_board: np.ndarray | None = None
        self._T_eye_offset: np.ndarray | None = None
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{pub_ip}:{pub_port}")
        self._pub_pegboard = ctx.socket(zmq.PUB)
        self._pub_pegboard.connect(f"tcp://{pub_ip}:{pegboard_pub_port}")
        self._pub_board = ctx.socket(zmq.PUB)
        self._pub_board.connect(f"tcp://{pub_ip}:{board_pub_port}")
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
                json.dumps({"T_eye_offset": self._T_eye_offset.flatten().tolist()}, indent=2))
            print(f"[Anchor] Eye offset saved to {self._EYE_OFFSET_FILE.name}")
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

    def lock(self, T_cam_anchor: np.ndarray,
             cam_T: np.ndarray, center_T: np.ndarray | None = None) -> bool:
        """Lock world frame to marker 100 (anchor). Returns True on success."""
        if T_cam_anchor is None:
            return False
        if self._T_eye_offset is None and cam_T is not None and center_T is not None:
            self._T_eye_offset = np.linalg.inv(center_T) @ cam_T
            self._save_eye_offset()
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        self._T_wt = np.linalg.inv(eff @ T_cam_anchor)
        src = "CenterEye+offset" if self._T_eye_offset is not None else "cam_T"
        print(f"[Anchor] Locked to marker 100 ({src}).")
        return True

    def relock(self, T_cam_anchor: np.ndarray,
               cam_T: np.ndarray, center_T: np.ndarray | None = None) -> bool:
        if T_cam_anchor is None or not self.locked:
            return False
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        self._T_wt = np.linalg.inv(eff @ T_cam_anchor)
        return True

    def set_pegboard(self, T_world_pegboard: np.ndarray) -> None:
        """Directly set the pegboard world transform (e.g. loaded from a file)."""
        self._T_world_pegboard = np.array(T_world_pegboard, dtype=np.float64)
        t = self._T_world_pegboard[:3, 3]
        print(f"[Anchor] Pegboard set from file: "
              f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")

    def update_pegboard_from_tracking(self, cam_T: np.ndarray,
                                      center_T: np.ndarray | None,
                                      T_cam_pegboard: np.ndarray) -> bool:
        """Compute pegboard pose in raw world frame using live Quest tracking.

        Marker 100 does NOT need to be visible — uses the locked _T_wt instead.
        """
        if not self.locked or T_cam_pegboard is None:
            return False
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        # _T_wt = T_world_raw_tracking (raw, without _T_offset)
        # T_world_101_raw = _T_wt @ eff @ T_cam_101
        self._T_world_pegboard = self._T_wt @ eff @ T_cam_pegboard #_T_wt @ eff @ T_cam_pegboard =  tracking→world  @  camera→tracking  @  pegboard→camera =  pegboard in world space
        t = self._T_world_pegboard[:3, 3]
        print(f"[Anchor] Pegboard updated: t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")
        return True

    def update_board_from_tracking(self, cam_T: np.ndarray,
                                   center_T: np.ndarray | None,
                                   T_cam_marker: np.ndarray,
                                   T_board_from_marker: np.ndarray) -> bool:
        """Compute tracked-board pose in raw world frame from whichever of
        markers 102/103 is currently visible.

        T_board_from_marker is the fixed offset (board origin expressed in
        the detected marker's local frame) for that specific marker.
        Marker 100 does NOT need to be visible — uses the locked _T_wt instead.
        """
        if not self.locked or T_cam_marker is None:
            return False
        eff = self._effective_cam_T(cam_T, center_T)
        if eff is None:
            return False
        T_world_marker = self._T_wt @ eff @ T_cam_marker
        self._T_world_board = T_world_marker @ T_board_from_marker
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
        return self._publish_T(self._T_offset @ self._T_world_board,
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


# =============================================================================
# Tool layout manager
# =============================================================================

class _ToolLayoutManager:
    """Loads tool_layout.json once at startup and publishes world-space tool
    definitions to Unity (port 5011). To apply changes, restart the script."""

    PORT = cfg.TOOL_LAYOUT_PORT

    def __init__(self, json_path: str, ip: str):
        self._tools: list = []
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

    def publish(self, T_pegboard_in_world: np.ndarray) -> None:
        """(Re-)publish the current layout with the given pegboard transform."""
        self._publish(T_pegboard_in_world)

    def _publish(self, T: np.ndarray) -> None:
        out = []
        for t in self._tools:
            sz   = t.get("size", [0.05, 0.05, 0.05])
            rot  = t.get("rotation_deg", [0.0, 0.0, 0.0])

            # peg_pos is the base in pegboard frame (new format); fall back to world_pos.
            R_world = ScipyR.from_euler('z', float(rot[2]), degrees=True).as_matrix()
            if "peg_pos" in t:
                base_w = (T @ np.append(t["peg_pos"], 1.0))[:3]
            else:
                base_w = np.array(t.get("world_pos", [0.0, 0.0, 0.0]))
            # Unity prefabs are centred at their local origin → send centroid
            pos_w   = base_w + R_world @ np.array([0.0, 0.0, sz[2] / 2.0])
            q_xyzw  = ScipyR.from_matrix(R_world).as_quat()

            pos_u  = open3d_to_unity_vector(pos_w)
            q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            q_u    = open3d_to_unity_quaternion(q_wxyz)
            sz_u   = open3d_to_unity_vector(np.array(sz, dtype=float))

            out.append({
                "id":            int(t["id"]),
                "type":          t.get("type", "unknown"),
                "position":      pos_u.tolist(),
                "rotation_xyzw": [float(q_u[1]), float(q_u[2]),
                                   float(q_u[3]), float(q_u[0])],
                "size":          sz_u.tolist(),
            })
        try:
            self._pub.send_string(json.dumps({"tools": out}))
        except Exception as e:
            print(f"[ToolLayout] Publish error: {e}")

    # ── PyBullet data ────────────────────────────────────────────────────────

    def world_boxes(self, T: np.ndarray) -> list:
        """Return list of (centroid_world, R_world, size) for PyBullet drawing."""
        boxes = []
        for t in self._tools:
            sz      = t.get("size", [0.05, 0.05, 0.05])
            rot     = t.get("rotation_deg", [0.0, 0.0, 0.0])
            R_world = ScipyR.from_euler('z', float(rot[2]), degrees=True).as_matrix()
            if "peg_pos" in t:
                base_w = (T @ np.append(t["peg_pos"], 1.0))[:3]
            else:
                base_w = np.array(t.get("world_pos", [0.0, 0.0, 0.0]))
            centroid = base_w + R_world @ np.array([0.0, 0.0, sz[2] / 2.0])
            boxes.append((centroid, R_world, sz))
        return boxes

    def get_world_data(self, tool_id: int,
                       T: np.ndarray) -> "tuple | None":
        """Return (centroid_world, R_world, size) for tool_id, or None if not found."""
        for t in self._tools:
            if t["id"] == tool_id:
                sz      = t.get("size", [0.05, 0.05, 0.05])
                rot     = t.get("rotation_deg", [0.0, 0.0, 0.0])
                R_local = ScipyR.from_euler('xyz', rot, degrees=True).as_matrix()
                R_world = T[:3, :3] @ R_local
                if "peg_pos" in t:
                    base_w = (T @ np.append(t["peg_pos"], 1.0))[:3]
                else:
                    base_w = np.array(t.get("world_pos", [0.0, 0.0, 0.0]))
                # peg_pos/world_pos is the base (bottom face); return centroid for IK
                centroid = base_w + R_local @ np.array([0.0, 0.0, sz[2] / 2.0])
                return centroid, R_world, sz
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

    def close(self) -> None:
        try:
            self._pub.close(0)
        except Exception:
            pass

# =============================================================================
# Tool selection manager
# =============================================================================

class _ToolSelectionManager:
    SELECTED_COLOR = [0.0, 1.0, 0.0, 0.5]
    HOVER_COLOR    = [1.0, 0.5, 0.0, 0.5]
    RESET_COLOR    = [-1.0, -1.0, -1.0, -1.0]

    def __init__(self, quest_ip: str, click_port: int = cfg.TOOL_CLICK_PORT, color_port: int = cfg.TOOL_COLOR_PORT):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{quest_ip}:{click_port}")
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{color_port}")
        time.sleep(0.2)
        self._active_tool_id: int | None  = None
        self._hovered_tool_id: int | None = None
        self._active_hand: str | None     = None

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

    def _handle_click(self, tool_id: int, hand: str = "unknown"):
        self._hovered_tool_id = None # clears _hovered_tool_id at the start — if you click, the hover state is irrelevant.
        updates: list[tuple[int, list[float]]] = []
        if self._active_tool_id == tool_id:
            updates.append((tool_id, self.RESET_COLOR))
            self._active_tool_id = None
            self._active_hand    = None
        elif self._active_tool_id is not None:
            updates.append((self._active_tool_id, self.RESET_COLOR))
            updates.append((tool_id, self.SELECTED_COLOR))
            self._active_tool_id = tool_id
            self._active_hand    = hand 
        else: #nothing was selected
            updates.append((tool_id, self.SELECTED_COLOR))
            self._active_tool_id = tool_id
            self._active_hand    = hand
        for tid, color in updates:
            self.send_color(tid, color)

    def _handle_hover_enter(self, tool_id: int):
        if tool_id == self._active_tool_id:
            return
        self._hovered_tool_id = tool_id
        self.send_color(tool_id, self.HOVER_COLOR)

    def _handle_hover_exit(self, tool_id: int):
        if tool_id != self._hovered_tool_id: #it's not the tool we recorded as hovered
            return
        self._hovered_tool_id = None 
        if tool_id == self._active_tool_id: #it's currently selected (don't un-highlight a selected tool on hover exit)
            return
        self.send_color(tool_id, self.RESET_COLOR)

    def send_color(self, tool_id: int, color: list[float]):
        msg = {"tool_id": int(tool_id), "color": [float(c) for c in color]}
        try:
            self._pub.send_string(json.dumps(msg))
        except Exception as e:
            print(f"[ToolSelection] Publish error: {e}")

    @property
    def active_tool_id(self) -> int | None:
        return self._active_tool_id

    @property
    def active_hand(self) -> str | None:
        return self._active_hand

    def deselect(self, tool_id: int):
        if self._active_tool_id == tool_id:
            self._active_tool_id = None
            self._active_hand    = None

    def close(self):
        try: self._sub.close(0)
        except Exception: pass
        try: self._pub.close(0)
        except Exception: pass

# =============================================================================
# Grip state publisher / target pose receiver  (ports 5012 / 5013)
# =============================================================================

_BOX_FORWARD_OFFSET = 0.17   # metres from TCP to box centre along gripper Z
_BOX_SIZE           = [0.0254, 0.20, 0.25]   # metres (X, Y, Z in pegboard frame)


class _GripStatePublisher:
    """Publishes grip state + box pose to Unity on port 5012 (PUB).
    Unity SUB binds; Python PUB connects to Quest IP (same pattern as all other Python→Unity channels).
    """

    def __init__(self, quest_ip: str, port: int = cfg.GRIP_STATE_PORT):
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{quest_ip}:{port}")

    def publish(self, grip_state: str, T_tcp_world: np.ndarray) -> None:
        """Compute box pose from TCP transform and publish."""
        # Box centre = TCP position + BOX_FORWARD_OFFSET along gripper Z
        gripper_z_world = T_tcp_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
        box_pos_w = T_tcp_world[:3, 3] + _BOX_FORWARD_OFFSET * gripper_z_world

        q_xyzw  = ScipyR.from_matrix(T_tcp_world[:3, :3]).as_quat()
        pos_u   = open3d_to_unity_vector(box_pos_w)
        q_wxyz  = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_u     = open3d_to_unity_quaternion(q_wxyz)
        sz_u    = open3d_to_unity_vector(np.array(_BOX_SIZE, dtype=float))

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

    def close(self) -> None:
        try:
            self._pub.close(0)
        except Exception:
            pass


class _TargetPoseReceiver:
    """Receives the manipulated TCP target pose from Unity on port 5013 (SUB).
    Unity PUB binds; Python SUB connects to Quest IP (same pattern as all other Unity→Python channels).
    """

    def __init__(self, quest_ip: str, port: int = cfg.TARGET_POSE_PORT):
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{quest_ip}:{port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.setsockopt(zmq.RCVTIMEO, 0)

    def poll(self) -> "np.ndarray | None":
        """Return 4×4 TCP world transform if a new target pose has arrived, else None."""
        try:
            raw = self._sub.recv_string(flags=zmq.NOBLOCK)
            data = json.loads(raw)
            pos_u = data["tcp_pos"]       # Unity frame [x, y, z]
            q_u   = data["tcp_rot_xyzw"]  # Unity xyzw
            pos_w = unity_to_open3d_vector({"x": pos_u[0], "y": pos_u[1], "z": pos_u[2]})
            q_o3d = unity_to_open3d_quaternion([q_u[3], q_u[0], q_u[1], q_u[2]])
            R_w   = ScipyR.from_quat([q_o3d[1], q_o3d[2], q_o3d[3], q_o3d[0]]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_w
            T[:3, 3]  = pos_w
            return T
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._sub.close(0)
        except Exception:
            pass






# =============================================================================
# Offset tuner
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
# Open3D scene visualizer
# =============================================================================

class _SceneVis:
    FRUSTUM_SCALE = 0.2

    # ── Static geometry helpers ───────────────────────────────────────────────

    @staticmethod
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

    @staticmethod
    def make_box_lineset(pos: np.ndarray, R: np.ndarray,
                         size, color=(0.2, 0.9, 1.0)) -> o3d.geometry.LineSet:
        """12-edge wireframe box. pos = centre, R = rotation, size = [w, d, h]."""
        w, d, h = size[0] / 2, size[1] / 2, size[2] / 2
        corners_local = np.array([
            [-w, -d, -h], [ w, -d, -h], [ w,  d, -h], [-w,  d, -h],
            [-w, -d,  h], [ w, -d,  h], [ w,  d,  h], [-w,  d,  h],
        ])
        corners = (R @ corners_local.T).T + pos
        edges = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines  = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector([list(color)] * 12)
        return ls

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, title: str, width: int = 1000, height: int = 680):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=width, height=height)
        ro = self.vis.get_render_option()
        ro.background_color    = np.array([0.08, 0.08, 0.10])
        ro.point_size          = 7.0
        ro.line_width          = 2.0
        ro.mesh_show_back_face = True

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(world_frame)

        self._cam_frustum   = None
        self._head_frustum  = None
        self._tcp_axes      = None          # lazy — added on first update_tcp() call
        self._tool_box_linesets: list = []  # lazy — grows to match number of tool boxes
        self._pegboard_corners_local: np.ndarray | None = None

        # Pegboard (coordinate frame + sphere + rectangle outline)
        self._pegboard_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._pegboard_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._pegboard_frame)
        self._pegboard_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.020)
        self._pegboard_sphere.paint_uniform_color([0.1, 1.0, 0.1])
        self._pegboard_sphere.compute_vertex_normals()
        self._pegboard_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._pegboard_sphere)
        self._pegboard_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._pegboard_lineset)
        self._pegboard_box_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._pegboard_box_lineset)
        self._peg_box_center_local: np.ndarray | None = None
        self._peg_box_size: list | None = None
        self._pegboard_T = self._hidden_T()

        # Reachability arrows (shown for 5 s after pressing R, then hidden)
        self._reach_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._reach_lineset)

        # Tracking origin (coordinate frame + sphere)
        self._tracking_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._tracking_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_frame)
        self._tracking_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.020)
        self._tracking_sphere.paint_uniform_color([0.2, 0.4, 1.0])
        self._tracking_sphere.compute_vertex_normals()
        self._tracking_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_sphere)
        self._tracking_T = self._hidden_T()

        # Tracked board (coordinate frame + baseboard mesh)
        self._board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._board_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._board_frame)
        self._board_mesh       = None
        self._board_manip_mesh = None
        _baseboard_path = cfg.SCENE_LAYOUT_DIR.parent / "robot_assets" / "baseboard.obj"
        if _baseboard_path.exists():
            # Mesh raw: X=201mm, Y=250mm, Z=25mm(face normal), geometric centre at [0,0,-11.5mm].
            # _board_mesh      (ArUco):  Rz(+90°) — swaps X↔Y so board X=250mm; face normal stays Z.
            # _board_manip_mesh (AR box): Ry(-90°) after Rz(+90°) — rotates face-normal (Z) to X
            #   so that board X (250mm) aligns with grip_z (TCP approach direction from the edge).
            _RAW_CENTER = np.array([0.0, 0.0, -0.0115])  # mesh geometric centre before any rotation

            def _T_fix(euler_seq: str, *deg_angles) -> np.ndarray:
                R = ScipyR.from_euler(euler_seq, list(deg_angles), degrees=True).as_matrix()
                # auto-centre: translate so rotated mesh centre lands at origin
                centre_after = R @ _RAW_CENTER
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3]  = -centre_after
                return T

            def _load_baseboard(color, euler_seq, *deg_angles):
                _bm = o3d.io.read_triangle_mesh(str(_baseboard_path))
                _bm.compute_vertex_normals()
                _bm.paint_uniform_color(color)
                _bm.transform(_T_fix(euler_seq, *deg_angles))
                _bm.transform(self._hidden_T())
                self.vis.add_geometry(_bm)
                return _bm

            self._board_mesh       = _load_baseboard([0.9, 0.75, 0.5], 'z',  90.0)        # warm tan  — ArUco
            self._board_manip_mesh = _load_baseboard([0.5, 0.75, 0.9], 'zy', 90.0, -90.0) # steel blue — AR box
            print(f"[SceneVis] baseboard.obj loaded ({len(self._board_mesh.vertices)} verts)")
        else:
            print(f"[SceneVis] baseboard.obj not found at {_baseboard_path}")
        self._board_T       = self._hidden_T()
        self._board_manip_T = self._hidden_T()

        # Gripper mesh — loaded once, placed at TCP pose each frame via delta
        # transforms. OBJ tool axis is mesh-Y; Rx(+90°) is baked into vertices
        # at load time so mesh-Y aligns with TCP-Z (standard robot convention).
        self._tcp_gripper_mesh = None
        self._tcp_T = self._hidden_T()
        _gripper_path = cfg.SCENE_LAYOUT_DIR / "gripperWtihAdapters.obj"
        if _gripper_path.exists():
            _mesh = o3d.io.read_triangle_mesh(str(_gripper_path))
            _mesh.compute_vertex_normals()
            _mesh.paint_uniform_color([0.75, 0.75, 0.75])
            _T_fix_gripper = np.eye(4, dtype=np.float64)
            _T_fix_gripper[:3, :3] = ScipyR.from_euler('x', 90, degrees=True).as_matrix()
            _mesh.transform(_T_fix_gripper)
            _mesh.transform(self._hidden_T())
            self.vis.add_geometry(_mesh)
            self._tcp_gripper_mesh = _mesh

        # UR10e arm meshes — visual offsets from URDF <visual><origin> baked into
        # vertices at load time, so update_robot() only needs the PyBullet link poses.
        # Order: [base, shoulder, upper_arm, forearm, wrist1, wrist2, wrist3]
        _UR10E_VIS = [
            ("base.obj",     [0,       0,      0      ], [0,         0,       np.pi       ]),
            ("shoulder.obj", [0,       0,      0      ], [0,         0,       np.pi       ]),
            ("upperarm.obj", [0,       0,      0.1762 ], [np.pi/2,   0,      -np.pi/2    ]),
            ("forearm.obj",  [0,       0,      0.0393 ], [np.pi/2,   0,      -np.pi/2    ]),
            ("wrist1.obj",   [0,       0,     -0.135  ], [np.pi/2,   0,       0          ]),
            ("wrist2.obj",   [0,       0,     -0.12   ], [0,         0,       0          ]),
            ("wrist3.obj",   [0,       0,     -0.1168 ], [np.pi/2,   0,       0          ]),
        ]
        _mesh_dir = cfg.SCENE_LAYOUT_DIR.parent / "robot_assets" / "meshes" / "ur10e" / "visual"
        self._robot_meshes: list = []
        self._robot_mesh_Ts: list = []
        for _fname, _vis_xyz, _vis_rpy in _UR10E_VIS:
            _path = _mesh_dir / _fname
            if _path.exists():
                _m = o3d.io.read_triangle_mesh(str(_path))
                _m.compute_vertex_normals()
                _m.paint_uniform_color([0.50, 0.52, 0.58])
                _T_vis = np.eye(4, dtype=np.float64)
                _T_vis[:3, :3] = ScipyR.from_euler('xyz', _vis_rpy).as_matrix()
                _T_vis[:3, 3]  = _vis_xyz
                _m.transform(_T_vis)
                _m.transform(self._hidden_T())
                self.vis.add_geometry(_m)
                self._robot_meshes.append(_m)
            else:
                self._robot_meshes.append(None)
            self._robot_mesh_Ts.append(self._hidden_T().copy())

        self._pcd_l, self._lines_l = self._make_hand([0.3, 0.6, 1.0])
        self._pcd_r, self._lines_r = self._make_hand([1.0, 0.55, 0.1])

        # Quat debug overlays (LineSets, updated each frame during tracking / on grasp click)
        self._qd_palm_tri    = o3d.geometry.LineSet()  # palm triangle edges — gold
        self._qd_palm_normal = o3d.geometry.LineSet()  # TCP Z arrow at palm — yellow
        self._qd_palm_frame  = o3d.geometry.LineSet()  # TCP axes at palm — RGB
        self._qd_tool_face   = o3d.geometry.LineSet()  # tool front-face outline — orange
        self._qd_tool_normal = o3d.geometry.LineSet()  # board outward normal + arrowhead — yellow
        self._qd_tool_frame  = o3d.geometry.LineSet()  # TCP axes at approach standoff — RGB
        self._qd_box_grip_z  = o3d.geometry.LineSet()  # grip-Z arrow at AR box centre — yellow
        self._qd_box_tcp     = o3d.geometry.LineSet()  # TCP target frame axes — RGB
        for _ls in [self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame,
                    self._qd_tool_face, self._qd_tool_normal, self._qd_tool_frame,
                    self._qd_box_grip_z, self._qd_box_tcp]:
            self.vis.add_geometry(_ls)

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0., 0., 0.])
        ctr.set_front([0., -0.5, -1.])
        ctr.set_up([0., 1., 0.])
        ctr.set_zoom(0.5)

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    # ── Update methods ────────────────────────────────────────────────────────

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

    def update_head(self, T: np.ndarray | None,
                    w=640, h=480, fx=400., fy=400., cx=320., cy=240.):
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

    def set_pegboard_outline(self, offset_x: float, offset_y: float,
                              width: float, height: float):
        """Store pegboard corners in marker-local frame (marker = origin).
        offset_x/y: distance from marker centre to the right/top board edge.
        Call once after loading the pegboard NPZ; update_pegboard() uses it."""
        self._pegboard_corners_local = np.array([
            [ offset_x,         offset_y,          0.0],   # top-right (≈ marker)
            [ offset_x - width, offset_y,          0.0],   # top-left
            [ offset_x - width, offset_y - height, 0.0],   # bottom-left
            [ offset_x,         offset_y - height, 0.0],   # bottom-right
        ])
        self._pegboard_lineset.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0]])
        self._pegboard_lineset.colors = o3d.utility.Vector3dVector(
            [[0.1, 0.6, 1.0]] * 4)
        # Board box: thickness 2 cm behind the marker plane (−Z direction)
        _thickness = 0.02
        self._peg_box_center_local = np.array([
            offset_x - width  / 2.0,
            offset_y - height / 2.0,
            -_thickness / 2.0,
        ])
        self._peg_box_size = [width, height, _thickness]

    def update_pegboard(self, T: np.ndarray | None):
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._pegboard_T)
        self._pegboard_frame.transform(delta)
        self._pegboard_sphere.transform(delta)
        self._pegboard_T = T_new
        self.vis.update_geometry(self._pegboard_frame)
        self.vis.update_geometry(self._pegboard_sphere)
        if self._pegboard_corners_local is not None:
            corners_h = np.hstack([self._pegboard_corners_local,
                                   np.ones((4, 1))])
            pts = (T_new @ corners_h.T).T[:, :3]
            self._pegboard_lineset.points = o3d.utility.Vector3dVector(pts)
            self.vis.update_geometry(self._pegboard_lineset)
        if self._peg_box_center_local is not None and self._peg_box_size is not None:
            centre_w = (T_new @ np.append(self._peg_box_center_local, 1.0))[:3]
            R_w = T_new[:3, :3]
            new_ls = self.make_box_lineset(centre_w, R_w, self._peg_box_size,
                                           color=(0.45, 0.45, 0.45))
            self._pegboard_box_lineset.points = new_ls.points
            self._pegboard_box_lineset.lines  = new_ls.lines
            self._pegboard_box_lineset.colors = new_ls.colors
            self.vis.update_geometry(self._pegboard_box_lineset)

    def update_tracking(self, T: np.ndarray | None):
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._tracking_T)
        self._tracking_frame.transform(delta)
        self._tracking_sphere.transform(delta)
        self._tracking_T = T_new
        self.vis.update_geometry(self._tracking_frame)
        self.vis.update_geometry(self._tracking_sphere)

    def update_board(self, T: np.ndarray | None):
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._board_T)
        self._board_frame.transform(delta)
        self.vis.update_geometry(self._board_frame)
        if self._board_mesh is not None:
            self._board_mesh.transform(delta)
            self.vis.update_geometry(self._board_mesh)
        if T is not None and not np.allclose(T, self._board_T):
            p = T[:3, 3]
            print(f"[SceneVis] board → ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        self._board_T = T_new

    def update_tcp(self, T: np.ndarray | None):
        """Update the TCP axes lineset and gripper mesh to pose T."""
        T_new = T if T is not None else self._hidden_T()
        new_axes = self.make_axes_lineset(T_new, size=0.08)
        if self._tcp_axes is None:
            self._tcp_axes = new_axes
            self.vis.add_geometry(self._tcp_axes)
        else:
            self._tcp_axes.points = new_axes.points
            self._tcp_axes.lines  = new_axes.lines
            self._tcp_axes.colors = new_axes.colors
            self.vis.update_geometry(self._tcp_axes)
        if self._tcp_gripper_mesh is not None:
            delta = T_new @ np.linalg.inv(self._tcp_T)
            self._tcp_gripper_mesh.transform(delta)
            self.vis.update_geometry(self._tcp_gripper_mesh)
        self._tcp_T = T_new

    def update_tool_boxes(self, boxes):
        """Update wireframe box linesets for all tool bounding boxes.
        boxes: list of (pos_world, R_world, size) from tool_layout.world_boxes()."""
        while len(self._tool_box_linesets) < len(boxes):
            ls = o3d.geometry.LineSet()
            self.vis.add_geometry(ls)
            self._tool_box_linesets.append(ls)
        _hidden_pts = o3d.utility.Vector3dVector(np.tile([0., -1.5, 0.], (8, 1)))
        _box_edges  = o3d.utility.Vector2iVector(
            [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
        for i, ls in enumerate(self._tool_box_linesets):
            if i < len(boxes):
                pos, R, size = boxes[i]
                new_ls = self.make_box_lineset(pos, R, size)
                ls.points = new_ls.points
                ls.lines  = new_ls.lines
                ls.colors = new_ls.colors
            else:
                ls.points = _hidden_pts
                ls.lines  = _box_edges
            self.vis.update_geometry(ls)

    def update_robot(self, link_poses: list[np.ndarray]):
        """Move UR10e arm meshes to the given PyBullet link world poses.
        link_poses: 7 transforms [base, shoulder, upper_arm, forearm, wrist1, wrist2, wrist3]
        from PyBulletScene.get_arm_link_world_poses()."""
        for i, (mesh, T_new) in enumerate(zip(self._robot_meshes, link_poses)):
            if mesh is None:
                continue
            T_cur = self._robot_mesh_Ts[i]
            delta = T_new @ np.linalg.inv(T_cur)
            mesh.transform(delta)
            self.vis.update_geometry(mesh)
            self._robot_mesh_Ts[i] = T_new

    # ── Quat debug overlays ───────────────────────────────────────────────────

    @staticmethod
    def _arrow_ls_data(origin, direction, length, color):
        """Return (pts, lines, colors) for a shaft + V arrowhead LineSet."""
        d    = np.array(direction, dtype=float)
        d    = d / (np.linalg.norm(d) + 1e-9)
        tip  = origin + d * length
        perp = np.cross(d, [0., 1., 0.])
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(d, [1., 0., 0.])
        perp = perp / (np.linalg.norm(perp) + 1e-9) * length * 0.12
        base = tip - d * length * 0.22
        pts  = np.array([origin, tip, base + perp, base - perp])
        lines  = [[0, 1], [1, 2], [1, 3]]
        colors = [list(color)] * 3
        return pts, lines, colors

    def _update_arrow_ls(self, ls, origin, direction, length, color):
        pts, lines, colors = self._arrow_ls_data(origin, direction, length, color)
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(ls)

    def _clear_ls(self, *lsets):
        empty_pts = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        empty_ln  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        empty_cl  = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        for ls in lsets:
            ls.points = empty_pts; ls.lines = empty_ln; ls.colors = empty_cl
            self.vis.update_geometry(ls)

    def update_palm_quat_debug(self, pts: np.ndarray, is_left: bool = False):
        """Overlay _palm_quat geometry on the live hand.

        Shows:
          Gold  — triangle connecting ThumbMCP (pts[3]), Palm (pts[1]), IndexMCP (pts[6])
          Yellow — palm inward normal (pre TCP-offset z axis)
          RGB   — final TCP frame axes at the palm centre
        """
        if pts is None or len(pts) <= 6:
            self._clear_ls(self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame)
            return

        p_thumb = pts[3]; p_palm = pts[1]; p_index = pts[6]

        # Gold triangle (3 edges)
        self._qd_palm_tri.points = o3d.utility.Vector3dVector([p_thumb, p_palm, p_index])
        self._qd_palm_tri.lines  = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 0]])
        self._qd_palm_tri.colors = o3d.utility.Vector3dVector([[0.9, 0.75, 0.1]] * 3)
        self.vis.update_geometry(self._qd_palm_tri)

        # Final TCP frame from _palm_quat — RGB axes at palm centre
        q_tcp = _palm_quat(pts, is_left=is_left)
        R_tcp = ScipyR.from_quat(q_tcp).as_matrix()
        T_tcp = np.eye(4); T_tcp[:3, :3] = R_tcp; T_tcp[:3, 3] = p_palm
        fl = self.make_axes_lineset(T_tcp, size=0.07)
        self._qd_palm_frame.points = fl.points
        self._qd_palm_frame.lines  = fl.lines
        self._qd_palm_frame.colors = fl.colors
        self.vis.update_geometry(self._qd_palm_frame)

        # Yellow arrow = -TCP Z = direction FROM palm TOWARD robot standoff
        # (flipped from blue axis so it reads as "robot is coming from this side")
        self._update_arrow_ls(self._qd_palm_normal, p_palm, -R_tcp[:, 2], 0.10, (1., 1., 0.))

    def clear_palm_quat_debug(self):
        self._clear_ls(self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame)

    def update_tool_quat_debug(self, centroid: np.ndarray, R_world: np.ndarray,
                                size: list, approach_dist: float = 0.10):
        """Overlay _tool_grasp_quat geometry on the selected tool.

        Shows:
          Orange — outline of the tool's front face (the pegboard-facing rectangle)
          Yellow — board outward normal (R_world[:,2]) with arrowhead
          RGB   — final TCP frame axes at the approach standoff
        """
        sx, sy = size[0] / 2, size[1] / 2
        face_ctr = centroid + R_world[:, 2] * (size[2] / 2)
        face_local = np.array([[-sx, -sy, 0], [sx, -sy, 0],
                                [sx,  sy, 0], [-sx,  sy, 0]])
        face_pts = (face_local @ R_world.T) + face_ctr
        self._qd_tool_face.points = o3d.utility.Vector3dVector(face_pts)
        self._qd_tool_face.lines  = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
        self._qd_tool_face.colors = o3d.utility.Vector3dVector([[1., 0.5, 0.]] * 4)
        self.vis.update_geometry(self._qd_tool_face)

        # Yellow board outward normal arrow
        self._update_arrow_ls(self._qd_tool_normal, centroid, R_world[:, 2], 0.09, (1., 1., 0.))

        # RGB TCP frame at approach standoff
        q_tcp    = _tool_grasp_quat(R_world)
        R_tcp    = ScipyR.from_quat(q_tcp).as_matrix()
        standoff = centroid + R_world[:, 2] * approach_dist
        T_tcp    = np.eye(4); T_tcp[:3, :3] = R_tcp; T_tcp[:3, 3] = standoff
        fl = self.make_axes_lineset(T_tcp, size=0.07)
        self._qd_tool_frame.points = fl.points
        self._qd_tool_frame.lines  = fl.lines
        self._qd_tool_frame.colors = fl.colors
        self.vis.update_geometry(self._qd_tool_frame)

    def clear_tool_quat_debug(self):
        self._clear_ls(self._qd_tool_face, self._qd_tool_normal, self._qd_tool_frame)

    def update_board_manip_debug(self, T_target: np.ndarray) -> None:
        """Show AR board manipulation → robot target overlay.

        T_target : 4×4 world transform of the manipulated box (from _TargetPoseReceiver).
          Yellow arrow — box Z axis (_grip_z), the direction the gripper holds the box.
          RGB axes    — TCP target frame, placed _BOX_FORWARD_OFFSET behind box along grip_z.
        """
        grip_z  = T_target[:3, 2]                                   # box Z = gripper approach axis
        tcp_pos = T_target[:3, 3] - _BOX_FORWARD_OFFSET * grip_z   # robot TCP position

        # Move AR-controlled baseboard mesh to T_target
        if self._board_manip_mesh is not None:
            delta = T_target @ np.linalg.inv(self._board_manip_T)
            self._board_manip_mesh.transform(delta)
            self.vis.update_geometry(self._board_manip_mesh)
            self._board_manip_T = T_target

        # Yellow: grip-Z arrow from box centre (direction box Z axis points)
        self._update_arrow_ls(self._qd_box_grip_z, T_target[:3, 3], grip_z, 0.10, (1., 1., 0.))

        # RGB: TCP frame at the computed robot target position
        T_tcp = np.eye(4); T_tcp[:3, :3] = T_target[:3, :3]; T_tcp[:3, 3] = tcp_pos
        fl = self.make_axes_lineset(T_tcp, size=0.07)
        self._qd_box_tcp.points = fl.points
        self._qd_box_tcp.lines  = fl.lines
        self._qd_box_tcp.colors = fl.colors
        self.vis.update_geometry(self._qd_box_tcp)

    def clear_board_manip_debug(self):
        self._clear_ls(self._qd_box_grip_z, self._qd_box_tcp)
        if self._board_manip_mesh is not None:
            delta = self._hidden_T() @ np.linalg.inv(self._board_manip_T)
            self._board_manip_mesh.transform(delta)
            self.vis.update_geometry(self._board_manip_mesh)
            self._board_manip_T = self._hidden_T()

    def update_reachability_arrows(self, points: np.ndarray, flags: np.ndarray,
                                    board_normal: np.ndarray, arrow_len: float = 0.04):
        """Draw one arrow per grid point along board_normal: green=reachable, red=not.
        Call hide_reachability_arrows() to clear them."""
        n = len(points)
        if n == 0:
            return
        norm = board_normal / (np.linalg.norm(board_normal) + 1e-9)
        tips  = points + norm * arrow_len
        pts   = np.empty((2 * n, 3), dtype=np.float64)
        pts[0::2] = points
        pts[1::2] = tips
        lines  = [[2*i, 2*i+1] for i in range(n)]
        colors = [[0.1, 0.9, 0.1] if f else [0.9, 0.1, 0.1] for f in flags]
        self._reach_lineset.points = o3d.utility.Vector3dVector(pts)
        self._reach_lineset.lines  = o3d.utility.Vector2iVector(lines)
        self._reach_lineset.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self._reach_lineset)

    def hide_reachability_arrows(self):
        self._reach_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._reach_lineset.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        self._reach_lineset.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.vis.update_geometry(self._reach_lineset)

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
# MainScene
# =============================================================================

class MainScene:

    _SIM_Q_DEG       = [-105.97, -29.43, 87.53, 33.17, 92.40, 168.95]
    _TCP_TOOL_ID     = 200    # must match ToolClickPublisher tool_id in Unity
    _SYNTH_INTERVAL  = 1.0 / 30.0
    _RELOCK_COOLDOWN = 2.0
    _AUTO_LOCK_MAX_DIST = 0.5      # metres — auto-lock-on-sight only within this range
    _TRACK_DIST_THRESHOLD = 0.02   # metres — TCP-to-target distance considered "arrived"
    _TRACK_HOLD_FRAMES    = 15     # consecutive frames under threshold before locking grip

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
                 load_pegboard_from_file: bool = cfg.LOAD_PEGBOARD_FROM_FILE):

        self.quest_ip                 = quest_ip
        self.anchor_marker_id         = anchor_marker_id
        self.pegboard_marker_id       = pegboard_marker_id
        self.hand_port                = hand_port
        self.robot_ip                 = robot_ip
        self.simulation               = simulation
        self.board_marker_a           = board_marker_a
        self.board_marker_b           = board_marker_b
        self._use_calibrated_robot_base = use_calibrated_robot_base
        self._load_pegboard_from_file   = load_pegboard_from_file

        self._T_BOARD_FROM_MARKER = {
            board_marker_a: T_BOARD_FROM_MARKER_A,
            board_marker_b: T_BOARD_FROM_MARKER_B,
        }
        self._sim_q = np.deg2rad(self._SIM_Q_DEG)

        # ── PyBullet IK scene (headless) ──────────────────────────────────────
        _calib_dir = _FILE_DIR / "calibration_data" / "results"
        self.pb_scene: "PyBulletScene | None" = None
        if _PYBULLET_AVAILABLE:
            if simulation:
                if use_calibrated_robot_base and _calib_dir.exists():
                    try:
                        _p1 = np.load(_calib_dir / "phase1_results.npz")
                        self.pb_scene = PyBulletScene(
                            T_world_base=_p1["T_world_base"])
                        self.pb_scene.build()
                        self.pb_scene.update_robot(self._sim_q)
                        print("[PyBullet] Simulation + calibrated base pose (headless).")
                    except Exception as e:
                        print(f"[MainScene] PyBullet (calibrated) failed: {e}")
                        self.pb_scene = None
                else:
                    _T_world_base_sim = np.eye(4, dtype=float)
                    _T_world_base_sim[:3, 3] = [-0.4, -0.8, 0.4]
                    try:
                        self.pb_scene = PyBulletScene(
                            T_world_base=_T_world_base_sim)
                        self.pb_scene.build()
                        self.pb_scene.update_robot(self._sim_q)
                        print("[PyBullet] Simulation — hardcoded base pose (headless).")
                    except Exception as e:
                        print(f"[MainScene] PyBullet scene failed to build: {e}")
                        self.pb_scene = None
            else:
                if _calib_dir.exists():
                    try:
                        self.pb_scene = PyBulletScene.from_calibration(_calib_dir)
                        self.pb_scene.build()
                        self.pb_scene.update_robot(self._sim_q)
                    except Exception as e:
                        print(f"[MainScene] PyBullet scene failed to build: {e}")
                        self.pb_scene = None
                else:
                    print(f"[MainScene] Calibration dir not found: {_calib_dir}")


        # ── Receivers / publishers ────────────────────────────────────────────
        self.cam          = _CamFeedReceiver(quest_ip)
        self.aruco        = _ArucoPoseEstimator(
                                anchor_marker_id       = anchor_marker_id,
                                pegboard_marker_id     = pegboard_marker_id,
                                anchor_marker_size_m   = anchor_marker_size_m,
                                pegboard_marker_size_m = pegboard_marker_size_m,
                                board_marker_ids       = (board_marker_a, board_marker_b),
                                board_marker_size_m    = board_marker_size_m)
        self.aruco_worker = _ArUcoWorker(self.cam, self.aruco)
        self.hands        = _HandDataReceiver(quest_ip, hand_port)
        self.robot: "_RobotController | None" = None
        if _ROBOT_CTRL_AVAILABLE and self.pb_scene is not None:
            self.robot = _RobotController(
                unity_ip      = quest_ip,
                pb_scene      = self.pb_scene,
                T_world_base  = self.pb_scene.T_world_base,
                robot_ip      = robot_ip if not simulation else None,
            )

        if simulation:
            print(f"[Robot] Simulation mode — RobotController ready (no hardware)")
        elif self.robot is not None:
            print(f"[Robot] Live mode — RobotController ready ({robot_ip})")

        self.anchor      = _WorldAnchor(quest_ip)
        self.tools       = _ToolSelectionManager(quest_ip)
        self.tuner       = _OffsetTuner()
        self.synth       = _SyntheticObjectPublisher(quest_ip)
        self.tool_layout = _ToolLayoutManager(
                               cfg.SCENE_LAYOUT_DIR / "tool_layout1.json", quest_ip)
        self.grip_pub    = _GripStatePublisher(quest_ip)
        self.target_recv = _TargetPoseReceiver(quest_ip)

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
        self._last_synth_pub_time        = 0.0   # timestamp of last synth-object publish; throttled to _SYNTH_INTERVAL
        self._prev_relock_available = False  # previous relock-available flag; drives anchor-marker color change
        self._anchor_highlight_until           = 0.0   # time until anchor marker highlight reverts to normal color
        self._last_proximity_relock_time = 0.0   # timestamp of last auto-relock; enforces _RELOCK_COOLDOWN between relocks
        self._reachability_arrows_hide_at           = 0.0   # time until reachability arrows auto-hide (set on grasp complete)
        self._T_world_tcp: "np.ndarray | None"          = None   # current TCP pose in world; polled each frame from robot
        self._grip_state: "str | None"              = None   # None | 'moving_to_pose' | 'grabbed'; drives board-manip path
        self._last_ar_board_T: "np.ndarray | None"    = None   # last AR box pose from _TargetPoseReceiver; persists between polls
        self._pending_grasp_tool_id: "int | None"           = None   # tool ID being grasped (set on click, not yet read back)
        self._hand_tracking_active                         = False  # True while robot TCP is following a hand palm
        self._tracked_hand_side: "str | None"        = None   # which hand is being tracked: 'left' or 'right'
        self._track_proximity_frames                            = 0      # consecutive frames TCP has been within _TRACK_DIST_THRESHOLD
        self._track_palm_target_pos: "np.ndarray | None" = None   # world position of the palm being tracked toward
        self._last_hand_track_time: "float | None"     = None   # timestamp of last hand-track servoJ command; used to compute dt
        self._synth_cubes_added                  = False  # True once synthetic cubes have been pushed into synth._objects
        self._synth_cube_start_idx: "int | None"     = None   # index into synth._objects where the pegboard cubes begin
        self._fps_ref_time         = time.perf_counter()  # reference time for FPS averaging
        self._fps_frame_count      = 0                    # frame counter since last FPS print

        print(f"\n[Running]  quest_ip={quest_ip}  "
              f"anchor_marker=#{anchor_marker_id}  "
              f"pegboard_marker=#{pegboard_marker_id}  "
              f"hand_port={hand_port}")
        print(f"  Marker #{anchor_marker_id} auto-locks world + scene on first sight "
              f"within {self._AUTO_LOCK_MAX_DIST:.1f}m (or press ENTER from any "
              f"distance) — later re-locks need ENTER or the relock cube")
        print(f"  ENTER with marker #{pegboard_marker_id} visible (after locking)"
              f" → lock pegboard")
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
            boxes = self.tool_layout.world_boxes(T_wp)
            self.vis.update_tool_boxes(boxes)
        return True

    def _lock_anchor_initial(self, T_cam_anchor: np.ndarray,
                             center_T: "np.ndarray | None") -> None:
        """First-time anchor lock + the same follow-up steps ENTER/relock run
        (scene origin reset, pegboard-from-file load). Used both by the
        ENTER handler and by the auto-lock-on-sight check in run()."""
        self.anchor.lock(T_cam_anchor, self.cam.camera_T, center_T=center_T)
        self._last_proximity_relock_time = time.time()
        if self.pb_scene is not None:
            self.pb_scene.set_scene_origin(np.eye(4))
        if self._load_pegboard_from_file:
            self._try_load_pegboard_from_file()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while True:
                _iter_t0 = time.perf_counter()

                # ── Poll streams ──────────────────────────────────────────────
                self.tools.poll(timeout_ms=0)
                self.hands.poll()

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

                # ── CenterEye pose ────────────────────────────────────────────
                _center_T = self.hands.center_eye_T()

                # ── Tracked board (markers A/B) — constantly updated ──────────
                if self.anchor.locked and board_ok:
                    self.anchor.update_board_from_tracking(
                        self.cam.camera_T, _center_T,
                        T_cam_board[board_marker_seen],
                        self._T_BOARD_FROM_MARKER[board_marker_seen])

                # ── Publish world root + pegboard to Unity ────────────────────
                if self.anchor.locked:
                    self.anchor.publish()
                    self.anchor.publish_pegboard()
                    self.anchor.publish_board()
                    if self.pb_scene is not None:
                        if self.robot is not None:
                            self.robot.publish_base(self.pb_scene.T_world_base)
                    _now = time.time()
                    if _now - self._last_synth_pub_time >= self._SYNTH_INTERVAL:
                        self.synth.publish()
                        self._last_synth_pub_time = _now

                if self.pb_scene is not None:
                    if self.robot is not None:
                        self.robot.publish_joints(self.pb_scene.current_q)

                T_wt            = self.anchor.T_world_tracking
                T_world_camleft = (self.anchor.world_T(self.cam.camera_T)
                                   if self.cam.camera_T is not None else None)
                T_world_center  = (self.anchor.world_T(_center_T)
                                   if _center_T is not None else None)
                left_pts, right_pts = self.hands.world_joints(T_wt)

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
                # Gated on proximity (<0.5m) so it doesn't fire from across the room.
                if (not self.anchor.locked and anchor_ok
                        and self.cam.camera_T is not None
                        and dist_to_anchor < self._AUTO_LOCK_MAX_DIST):
                    self._lock_anchor_initial(T_cam_anchor, _center_T)
                    print(f"[AutoLock] Locked world to marker "
                          f"#{self.anchor_marker_id} on sight "
                          f"({dist_to_anchor:.2f} m away)")

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
                    self.anchor.relock(T_cam_anchor, self.cam.camera_T, _center_T)
                    if self._load_pegboard_from_file:
                        self._try_load_pegboard_from_file()
                    elif pegboard_ok:
                        self.anchor.update_pegboard_from_tracking(
                            self.cam.camera_T, _center_T, T_cam_pegboard)
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
                        _boxes = self.tool_layout.world_boxes(self.anchor.T_pegboard_in_world)
                        self.vis.update_tool_boxes(_boxes)
                    print("[AutoRelock] Relocked via proximity click")
                self.tools.deselect(self.anchor_marker_id)

                # ── TCP click → toggle continuous hand tracking ────────────────
                if (self.tools.active_tool_id == self._TCP_TOOL_ID
                        and self.anchor.locked
                        and self.anchor.T_pegboard_in_world is not None):
                    if self._grip_state is not None:
                        self._grip_state = None
                        self._last_ar_board_T = None
                        self.vis.clear_board_manip_debug()
                        if self.robot is not None:
                            self.robot.cancel_grasp()
                        self.grip_pub.publish(
                            'idle',
                            self._T_world_tcp if self._T_world_tcp is not None else np.eye(4))
                        print("[TCP click] Grip mode cancelled — returning to normal")
                    elif self._hand_tracking_active:
                        self._hand_tracking_active    = False
                        self._tracked_hand_side = None
                        self._track_proximity_frames       = 0
                        print("[TCP click] Hand tracking cancelled")
                    elif (not (self.robot is not None and self.robot.grasp_running)
                          and self.pb_scene is not None):
                        clicking_hand = self.tools.active_hand
                        opposite_hand = {"left": "right", "right": "left"}.get(clicking_hand)
                        if opposite_hand in ("left", "right"):
                            self._hand_tracking_active    = True
                            self._tracked_hand_side = opposite_hand
                            self._track_proximity_frames       = 0
                            print(f"[TCP click] Tracking {opposite_hand} hand palm "
                                  f"(opposite of {clicking_hand} click)")
                        else:
                            print(f"[TCP click] Unknown hand '{clicking_hand}' — ignoring click.")
                    self.tools.deselect(self._TCP_TOOL_ID)

                # ── Tool click → grasp ────────────────────────────────────────
                _tid = self.tools.active_tool_id
                if (_tid is not None
                        and _tid != self._TCP_TOOL_ID
                        and self.anchor.locked
                        and self.anchor.T_pegboard_in_world is not None
                        and self.robot is not None
                        and not self.robot.grasp_running):
                    tool_data = self.tool_layout.get_world_data(
                        _tid, self.anchor.T_pegboard_in_world)
                    if tool_data is not None:
                        _grasp_tid = _tid
                        self._pending_grasp_tool_id = _tid
                        _centroid, _R_world, _sz = tool_data
                        self.vis.update_tool_quat_debug(_centroid, _R_world, _sz)
                        self.robot.execute_grasp(
                            tool_data,
                            grasp_joints = self.tool_layout.get_grasp_joints(_tid),
                            category     = self.tool_layout.get_category(_tid),
                            on_complete  = lambda ok, tid=_grasp_tid: (
                                self.tools.send_color(tid, _ToolSelectionManager.RESET_COLOR),
                                self.vis.clear_tool_quat_debug(),
                                print(f"[ToolGrasp] id={tid} — {'OK' if ok else 'FAILED'}"),
                            ),
                        )
                        print(f"[ToolGrasp] id={_tid} — sequence started")
                    self.tools.deselect(_tid)

                # ── Update Open3D visualizer ───────────────────────────────────
                if self.cam.fx is not None:
                    fx, fy, cx, cy = _adapt_cx_cy(
                        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy,
                        self.cam.sensor_width, self.cam.sensor_height,
                        self.cam.width, self.cam.height)
                    self.vis.update_cam_frustum(T_world_camleft,
                                                self.cam.width, self.cam.height,
                                                fx, fy, cx, cy)
                self.vis.update_pegboard(self.anchor.T_pegboard_in_world)
                self.vis.update_board(self.anchor.T_board_in_world)
                self.vis.update_tracking(T_wt)
                self.vis.update_head(T_world_center)
                self.vis.update_hands(left_pts, right_pts)
                # Palm quat debug — always visible; prefer right hand, fall back to left
                if self._hand_tracking_active:
                    _dbg_pts  = left_pts  if self._tracked_hand_side == "left"  else right_pts
                    _dbg_left = self._tracked_hand_side == "left"
                else:
                    _dbg_pts  = right_pts if right_pts is not None else left_pts
                    _dbg_left = (right_pts is None and left_pts is not None)
                self.vis.update_palm_quat_debug(_dbg_pts, is_left=_dbg_left)
                self.vis.tick()

                # ── PyBullet scene update ─────────────────────────────────────
                if self.robot is not None:
                    if self._hand_tracking_active:
                        target_pts = (left_pts if self._tracked_hand_side == "left"
                                      else right_pts)
                        if target_pts is not None:
                            target_is_left = (self._tracked_hand_side == "left")
                            target_quat    = _palm_quat(target_pts, is_left=target_is_left)
                            _gripper_z     = ScipyR.from_quat(target_quat).apply([0., 0., 1.])
                            target_pos     = target_pts[1] - _gripper_z * 0.30
                            if ScipyR.from_quat(target_quat).apply([0., 1., 0.])[2] > 0:
                                target_quat = (ScipyR.from_quat(target_quat)
                                               * ScipyR.from_euler('z', 180, degrees=True)
                                               ).as_quat()
                            self._track_palm_target_pos = target_pos
                            _track_now = time.perf_counter()
                            _track_dt  = (min(_track_now - self._last_hand_track_time, 0.1)
                                          if self._last_hand_track_time is not None
                                          else 1.0 / 30.0)
                            self._last_hand_track_time = _track_now
                            self.robot.step_hand_track(target_pos.tolist(), target_quat,
                                                       _track_dt)
                        # else: hand lost this frame — hold last commanded pose
                    else:
                        if self.simulation:
                            self.robot.tick()
                        else:
                            q = self.robot.poll_q()
                            if q is not None:
                                self.pb_scene.update_robot(q)

                    # ── Visualizer update ─────────────────────────────────────
                    self._T_world_tcp = self.robot.tcp_pose
                    if self._T_world_tcp is not None:
                        self.vis.update_tcp(self._T_world_tcp)
                    _link_poses = self.robot.arm_link_poses()
                    if _link_poses is not None:
                        self.vis.update_robot(_link_poses)
                    if self._T_world_tcp is not None and self._tcp_synth is not None:
                        self._tcp_synth.centroid = self._T_world_tcp[:3, 3]
                        self._tcp_synth.R_o3d    = self._T_world_tcp[:3, :3]

                    # ── Hand-track proximity check ────────────────────────────
                    if (self._hand_tracking_active and self._T_world_tcp is not None
                            and self._track_palm_target_pos is not None):
                        dist = float(np.linalg.norm(
                            self._T_world_tcp[:3, 3] - self._track_palm_target_pos))
                        if dist < self._TRACK_DIST_THRESHOLD:
                            self._track_proximity_frames += 1
                        else:
                            self._track_proximity_frames = 0
                        if self._track_proximity_frames >= self._TRACK_HOLD_FRAMES:
                            _tracked = self._tracked_hand_side
                            self._hand_tracking_active    = False
                            self._tracked_hand_side = None
                            self._track_proximity_frames       = 0
                            self._track_palm_target_pos = None
                            self._grip_state       = 'grabbed'
                            print(f"[TCP track] Reached {_tracked} palm — "
                                  f"grip_state='grabbed'")

                    # ── Grip-state publishing + target-pose move ──────────────
                    if self._grip_state is not None and self._T_world_tcp is not None:
                        self.grip_pub.publish(self._grip_state, self._T_world_tcp)
                    if self._grip_state == 'grabbed':
                        T_target = self.target_recv.poll()
                        if T_target is not None:
                            self._last_ar_board_T = T_target
                            _grip_z  = T_target[:3, :3] @ np.array([0., 0., 1.])
                            _tcp_pos = T_target[:3, 3] - _BOX_FORWARD_OFFSET * _grip_z
                            self.robot.move_tcp(
                                _tcp_pos.tolist(),
                                ScipyR.from_matrix(T_target[:3, :3]).as_quat(),
                                on_complete=lambda ok: setattr(
                                    self, '_grip_state', 'grabbed'),
                            )
                            self._grip_state = 'moving_to_pose'
                            print("[Grip] Target pose received — moving robot")
                        if self._last_ar_board_T is not None:
                            self.vis.update_board_manip_debug(self._last_ar_board_T)

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
                cv.putText(disp,
                           f"ENTER: lock #{self.anchor_marker_id} (world+scene)  or"
                           f"  lock #{self.pegboard_marker_id} (pegboard)    ESC = quit",
                           (12, disp.shape[0] - 14),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv.imshow(self._win, disp)

                # ── Key handling ──────────────────────────────────────────────
                key = cv.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('r') or key == ord('R'):
                    if (self.pb_scene is not None
                            and self.anchor.T_pegboard_in_world is not None):
                        T_wp = self.anchor.T_pegboard_in_world
                        _reach_quat = _tool_grasp_quat(T_wp[:3, :3])
                        _, _, _reach_pts, _reach_flags = \
                            self.pb_scene.check_reachability(
                                T_wp, target_quat_xyzw=_reach_quat)
                        if len(_reach_pts):
                            _board_normal = T_wp[:3, 2]
                            self.vis.update_reachability_arrows(
                                _reach_pts, _reach_flags, _board_normal)
                            self._reachability_arrows_hide_at = time.time() + 5.0
                    else:
                        print("[R] Pegboard not locked yet — lock it first.")
                elif key == 13:  # ENTER
                    if self.simulation and self.pb_scene is not None:
                        self.pb_scene.set_scene_origin(np.eye(4))
                    if self.cam.camera_T is None:
                        if not self.simulation:
                            print("[ENTER] No camera pose — skipping.")
                    else:
                        # ── Phase A: lock / relock world frame ────────────────
                        if anchor_ok:
                            if self.anchor.locked:
                                self.anchor.relock(T_cam_anchor, self.cam.camera_T, _center_T)
                                print(f"[ENTER] Relocked world to marker "
                                      f"#{self.anchor_marker_id}")
                                self._last_proximity_relock_time = _now
                                if self.pb_scene is not None:
                                    self.pb_scene.set_scene_origin(np.eye(4))
                                if self._load_pegboard_from_file:
                                    self._try_load_pegboard_from_file()
                            else:
                                self._lock_anchor_initial(T_cam_anchor, _center_T)
                                print(f"[ENTER] Locked world to marker "
                                      f"#{self.anchor_marker_id}")
                        elif not self.anchor.locked:
                            print(f"[ENTER] Marker #{self.anchor_marker_id}"
                                  f" not visible — cannot lock.")

                        # ── Phase B: lock pegboard (skipped if loaded from file) ─
                        if not self._load_pegboard_from_file and pegboard_ok and self.anchor.locked:
                            self.anchor.update_pegboard_from_tracking(
                                self.cam.camera_T, _center_T, T_cam_pegboard)
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
                                _boxes = self.tool_layout.world_boxes(T_wp)
                                self.vis.update_tool_boxes(_boxes)
                        elif pegboard_ok and not self.anchor.locked:
                            print(f"[ENTER] Marker #{self.pegboard_marker_id} visible, "
                                  f"but lock marker #{self.anchor_marker_id} first.")

                # ── Perf stats ────────────────────────────────────────────────
                self._fps_frame_count += 1
                _iter_ms = (time.perf_counter() - _iter_t0) * 1000.0
                _elapsed  = time.perf_counter() - self._fps_ref_time
                if _elapsed >= 2.0:
                    _avg_hz = self._fps_frame_count / _elapsed
                    print(f"[perf] loop {_avg_hz:.1f} Hz | last iter {_iter_ms:.1f} ms")
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
        self.anchor.close()
        self.synth.close()
        self.tool_layout.close()
        self.grip_pub.close()
        self.target_recv.close()
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
        load_pegboard_from_file    = args.load_pegboard_from_file)
    scene.run()


if __name__ == "__main__":
    main()
