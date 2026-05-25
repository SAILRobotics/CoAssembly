"""
hand_eye_calibrator.py — Two-phase hand-eye calibration

Phase 1  Eye-in-base
  • Desk camera is FIXED.
  • ArUco marker is flat on the table  → world origin  (T_{world←aruco} = I)
  • ChArUco board is rigidly attached to the robot TCP.
  • Camera must see BOTH markers simultaneously.
  • Output: T_world_deskcam  (desk cam pose in world frame)
            T_world_base     (robot base pose in world frame)

Phase 2  Eye-in-hand
  • Hand camera is mounted on the robot TCP.
  • The SAME ArUco marker (world origin) is on the table.
  • Camera must see the ArUco marker.
  • Uses T_world_base from Phase 1.
  • Output: T_tcp_handcam  (hand cam pose in TCP frame)

Final Open3D visualisation (world frame):
  • Flat square  — ArUco marker at world origin
  • Orange sphere — Robot TCP (live FK)
  • Blue  frustum — Desk camera (static, from Phase 1)
  • Red   frustum — Hand camera (live, moves with TCP)
"""

import os
import json
import time
import threading
import numpy as np
import cv2 as cv
import open3d as o3d
import rtde_receive
import rtde_control
from pathlib import Path
from scipy.spatial.transform import Rotation as ScipyR


def _require_cv_aruco():
    if not hasattr(cv, "aruco"):
        version = getattr(cv, "__version__", "unknown")
        location = getattr(cv, "__file__", "unknown")
        raise RuntimeError(
            "OpenCV was imported, but cv2.aruco is not available.\n"
            f"  cv2 version : {version}\n"
            f"  cv2 location: {location}\n"
            "Install the contrib OpenCV wheel in this environment:\n"
            "  python -m pip install --force-reinstall opencv-contrib-python==4.10.0.84 numpy==1.26.4"
        )


# =============================================================================
# SE3 helpers
# =============================================================================

def _mat(rvec, tvec):
    """rvec (3,) + tvec (3,) → 4×4 SE3 matrix."""
    T = np.eye(4)
    T[:3, :3], _ = cv.Rodrigues(np.asarray(rvec, dtype=float).ravel())
    T[:3, 3] = np.asarray(tvec, dtype=float).ravel()
    return T


def _ur_tcp_to_mat(tcp):
    """UR TCP [x, y, z, rx, ry, rz] (metres, rotation-vector rad) → 4×4."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(tcp[:3], dtype=float)
    T[:3, :3] = ScipyR.from_rotvec(np.asarray(tcp[3:6], dtype=float)).as_matrix()
    return T


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


def _o3d_intrinsic(cam_matrix, w=1280, h=720):
    K = np.asarray(cam_matrix)
    return o3d.camera.PinholeCameraIntrinsic(
        w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def _draw_text(img, lines, origin=(10, 30), dy=32, scale=0.75, thickness=2):
    for i, (text, color) in enumerate(lines):
        cv.putText(img, text, (origin[0], origin[1] + i * dy),
                   cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)


# =============================================================================
# HandEyeCalibrator
# =============================================================================

class HandEyeCalibrator:
    """
    Two-phase hand-eye calibration.

    Parameters
    ----------
    robot_ip            UR robot IP (RTDE interface).
    save_dir            Root directory for images, poses, and results.
    world_marker_id     ID of the ArUco marker that defines the world origin.
    marker_sizes        Dict {marker_id: size_m} for ALL markers to track.
                        Must include world_marker_id.
                        e.g. {100: 0.10, 101: 0.10}
    charuco_squares_x/y ChArUco board grid dimensions.
    charuco_square_len  Physical size of one ChArUco square (metres).
    charuco_marker_len  Physical size of ChArUco embedded ArUco markers (metres).
    aruco_dict_name     e.g. "DICT_6X6_1000".
    T_tcp_to_charuco    4×4 ndarray — T_{tcp←charuco}: position and orientation
                        of the ChArUco board expressed in the TCP frame.
                        Same convention as get_T_tcp_to_charuco() in calibrator.py.
    """

    def __init__(self,
                 robot_ip,
                 save_dir,
                 world_marker_id,
                 marker_sizes,
                 charuco_squares_x,
                 charuco_squares_y,
                 charuco_square_len,
                 charuco_marker_len,
                 aruco_dict_name="DICT_6X6_1000",
                 T_tcp_to_charuco=None):

        _require_cv_aruco()

        self.robot        = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.robot_ctrl   = rtde_control.RTDEControlInterface(robot_ip)
        self.save_dir     = Path(save_dir)
        self.world_marker_id = world_marker_id
        self.marker_sizes    = dict(marker_sizes)          # {id: size_m}
        # Convenience aliases used by existing visualize / calibrate code
        self.aruco_id     = world_marker_id
        self.aruco_size_m = marker_sizes[world_marker_id]

        # ArUco single-marker detector
        aruco_dict = cv.aruco.getPredefinedDictionary(
            getattr(cv.aruco, aruco_dict_name))
        self.aruco_dict   = aruco_dict
        self.aruco_params = cv.aruco.DetectorParameters()

        # ChArUco board
        self.charuco_squares_x  = charuco_squares_x
        self.charuco_squares_y  = charuco_squares_y
        self.charuco_square_len = charuco_square_len
        self.charuco_board = cv.aruco.CharucoBoard(
            (charuco_squares_x, charuco_squares_y),
            charuco_square_len, charuco_marker_len, aruco_dict)
        ch_params  = cv.aruco.CharucoParameters()
        det_params = cv.aruco.DetectorParameters()
        det_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.charuco_detector = cv.aruco.CharucoDetector(
            self.charuco_board, ch_params, det_params)

        # T_{charuco←tcp}: how ChArUco board frame relates to TCP frame
        if T_tcp_to_charuco is None:
            T_tcp_to_charuco = np.eye(4)
        self.T_tcp_to_charuco = np.asarray(T_tcp_to_charuco, dtype=float)

        # Results (populated after calibrate_phase*)
        self.T_world_deskcam  = None   # desk cam pose in world
        self.T_world_base     = None   # robot base pose in world
        self.T_tcp_handcam    = None   # hand cam pose in TCP frame
        self.T_world_objects  = {}     # {marker_id: T_world_marker} for object markers
        self.deskcam_intrinsic = None
        self.handcam_intrinsic = None

        # Save-dir layout
        self._p1_dir      = self.save_dir / "phase1"
        self._p2_dir      = self.save_dir / "phase2"
        self._results_dir = self.save_dir / "results"

    # =========================================================================
    # Robot motion
    # =========================================================================

    def move_joints(self, q_deg, speed=0.3, accel=0.2):
        """Move robot to joint angles given in degrees. Blocks until done."""
        q_rad = np.deg2rad(q_deg).tolist()
        self.robot_ctrl.moveJ(q_rad, speed, accel)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _detect_all_markers(self, gray, cam_matrix, dist_coeffs):
        """Detect every marker listed in self.marker_sizes.
        Returns {marker_id: T_{cam←marker}} for each detected marker."""
        corners, ids, _ = cv.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return {}
        ids_flat = ids.flatten()
        result = {}
        for marker_id, size_m in self.marker_sizes.items():
            where = np.where(ids_flat == marker_id)[0]
            if len(where) == 0:
                continue
            idx = int(where[0])
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                [corners[idx]], size_m, cam_matrix, dist_coeffs)
            result[marker_id] = _mat(rvecs[0], tvecs[0])
        return result

    def _detect_world_aruco(self, gray, cam_matrix, dist_coeffs):
        """Detect the world ArUco marker. Returns T_{cam←aruco} or None."""
        return self._detect_all_markers(gray, cam_matrix, dist_coeffs).get(
            self.world_marker_id, None)

    def _detect_charuco(self, gray, cam_matrix, dist_coeffs):
        """Detect ChArUco board. Returns T_{cam←charuco} or None."""
        ch_corners, ch_ids, _, _ = self.charuco_detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 4:
            return None
        ok, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.charuco_board,
            cam_matrix, dist_coeffs, None, None)
        return _mat(rvec, tvec) if ok else None  # T_{cam←charuco}

    def _fk(self):
        """Return T_{base←tcp} from live UR RTDE FK."""
        return _ur_tcp_to_mat(self.robot.getActualTCPPose())

    def _draw_axes(self, img, T_cam_marker, cam_matrix, dist_coeffs, size=0.05):
        rvec, _ = cv.Rodrigues(T_cam_marker[:3, :3])
        cv.drawFrameAxes(img, cam_matrix, dist_coeffs,
                         rvec, T_cam_marker[:3, 3], size)

    def _make_pcd(self, color_bgr, depth_float32, cam_matrix):
        """BGR image + uint16 depth (millimetres) → Open3D PointCloud in camera frame."""
        h, w = depth_float32.shape[:2]
        intrinsic = _o3d_intrinsic(cam_matrix, w, h)
        color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_float32),
            depth_scale=1.0,      # already float32 metres from RealSense
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # =========================================================================
    # Phase 1  —  Eye-in-base  CAPTURE
    # =========================================================================

    def capture_phase1(self, get_frame_fn, cam_matrix, dist_coeffs,
                       get_depth_fn=None,
                       joint_poses_deg=None,
                       window="Phase 1 – Eye-in-base  |  ENTER=save  ESC=done"):
        """
        Interactive capture loop for Phase 1.

        get_frame_fn()    → BGR ndarray from the desk camera.
        get_depth_fn()    → uint16 ndarray (depth in mm) from the desk camera.
                            When provided a pointcloud is saved on every ENTER.
        joint_poses_deg   → list of [j1..j6] in degrees.  When provided the
                            robot moves to each pose in turn; ENTER captures
                            the current pose and moves to the next one.
                            When None, the robot is not moved automatically.

        Both markers must be visible before ENTER will save a frame.
        """
        self._p1_dir.mkdir(parents=True, exist_ok=True)
        cam_matrix  = np.asarray(cam_matrix,  dtype=float)
        dist_coeffs = np.asarray(dist_coeffs, dtype=float)
        np.savez(self._p1_dir / "intrinsics.npz",
                 camera_matrix=cam_matrix, dist_coeffs=dist_coeffs)

        poses     = list(joint_poses_deg) if joint_poses_deg is not None else []
        n_poses   = len(poses)
        pose_idx  = 0
        counter   = 0

        # Move to the first pose immediately if a list was given
        if poses:
            print(f"[Phase 1] Moving to pose 1/{n_poses} …")
            self.move_joints(poses[pose_idx])

        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, 540)
        print(f"[Phase 1 capture] {window}")

        while True:
            frame = get_frame_fn()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis  = frame.copy()

            all_markers   = self._detect_all_markers(gray, cam_matrix, dist_coeffs)
            T_cam_aruco   = all_markers.get(self.world_marker_id)
            T_cam_charuco = self._detect_charuco(gray, cam_matrix, dist_coeffs)

            for mid, T_cm in all_markers.items():
                size = 0.06 if mid == self.world_marker_id else 0.04
                self._draw_axes(vis, T_cm, cam_matrix, dist_coeffs, size)
            if T_cam_charuco is not None:
                self._draw_axes(vis, T_cam_charuco, cam_matrix, dist_coeffs, 0.03)

            ok_a = T_cam_aruco   is not None
            ok_c = T_cam_charuco is not None
            both = ok_a and ok_c

            obj_ids = [mid for mid in self.marker_sizes if mid != self.world_marker_id]
            pose_label = (f"Pose {pose_idx + 1}/{n_poses}"
                          if poses else f"Saved: {counter}")
            status_lines = [
                (f"ArUco world ({self.world_marker_id}): {'OK ✓' if ok_a else 'NOT FOUND'}",
                 (0, 220, 0) if ok_a else (0, 50, 220)),
                (f"ChArUco TCP             : {'OK ✓' if ok_c else 'NOT FOUND'}",
                 (0, 220, 0) if ok_c else (0, 50, 220)),
            ]
            for mid in obj_ids:
                seen = mid in all_markers
                status_lines.append(
                    (f"Object marker ({mid})   : {'DETECTED ✓' if seen else 'not visible'}",
                     (0, 200, 100) if seen else (120, 120, 120)))
            status_lines.append(
                (f"{pose_label}   Saved: {counter}   "
                 f"{'ENTER=save' if both else '(world+charuco needed)'}",
                 (0, 220, 220)))
            _draw_text(vis, status_lines)

            cv.imshow(window, vis)
            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == 13 and both:
                T_base_tcp = self._fk()
                save_data  = dict(T_cam_aruco=T_cam_aruco,
                                  T_cam_charuco=T_cam_charuco,
                                  T_base_tcp=T_base_tcp)
                # Save all detected object markers
                for mid, T_cm in all_markers.items():
                    if mid != self.world_marker_id:
                        save_data[f"T_cam_marker_{mid}"] = T_cm
                np.savez(self._p1_dir / f"pose_{counter:04d}.npz", **save_data)
                cv.imwrite(str(self._p1_dir / f"frame_{counter:04d}.png"), vis)
                if get_depth_fn is not None:
                    depth = get_depth_fn()
                    if depth is not None:
                        pcd = self._make_pcd(frame, depth, cam_matrix)
                        o3d.io.write_point_cloud(
                            str(self._p1_dir / f"pcd_{counter:04d}.pcd"), pcd)
                obj_seen = [mid for mid in obj_ids if mid in all_markers]
                print(f"  [Phase 1] Saved pose {counter}"
                      + (f"  (objects: {obj_seen})" if obj_seen else ""))
                counter  += 1
                pose_idx += 1

                if poses and pose_idx < n_poses:
                    print(f"[Phase 1] Moving to pose {pose_idx + 1}/{n_poses} …")
                    self.move_joints(poses[pose_idx])
                elif poses and pose_idx >= n_poses:
                    print("[Phase 1] All predefined poses captured.")
                    break

        cv.destroyWindow(window)
        print(f"[Phase 1 capture] Done — {counter} poses saved.")

    # =========================================================================
    # Phase 1  —  CALIBRATE
    # =========================================================================

    def calibrate_phase1(self):
        """
        Process Phase 1 data.

        Math (all in world = ArUco frame, T_{world←aruco} = I):

          T_{world←cam}      = inv(T_{cam←aruco})                 [camera is fixed]
          T_{world←charuco}  = T_{world←cam} @ T_{cam←charuco}
          T_{charuco←tcp}    = inv(T_tcp_to_charuco)              [T_tcp_to_charuco = T_{tcp←charuco}]
          T_{world←tcp}      = T_{world←charuco} @ T_{charuco←tcp}
          T_{world←base}     = T_{world←tcp} @ inv(T_{base←tcp})

        Matches calibrator.py: compute_base_pose_from_charuco_ur().

        Returns (T_world_deskcam, T_world_base).
        """
        files = sorted(self._p1_dir.glob("pose_*.npz"))
        if not files:
            raise RuntimeError("No Phase 1 poses found. Run capture_phase1 first.")

        intr = np.load(self._p1_dir / "intrinsics.npz")
        K    = intr["camera_matrix"]
        self.deskcam_intrinsic = _o3d_intrinsic(K)

        T_world_cam_list  = []
        T_world_base_list = []
        obj_ids = [mid for mid in self.marker_sizes if mid != self.world_marker_id]
        T_world_obj_lists = {mid: [] for mid in obj_ids}

        for f in files:
            d = np.load(f)
            T_cam_aruco   = d["T_cam_aruco"]    # T_{cam←world}
            T_cam_charuco = d["T_cam_charuco"]  # T_{cam←charuco}
            T_base_tcp    = d["T_base_tcp"]     # T_{base←tcp}  from FK

            T_world_cam      = np.linalg.inv(T_cam_aruco)               # T_{world←cam}
            T_world_charuco  = T_world_cam @ T_cam_charuco               # T_{world←charuco}
            T_charuco_to_tcp = np.linalg.inv(self.T_tcp_to_charuco)      # T_{charuco←tcp}
            T_world_tcp      = T_world_charuco @ T_charuco_to_tcp         # T_{world←tcp}
            T_world_base     = T_world_tcp @ np.linalg.inv(T_base_tcp)   # T_{world←base}

            xyz_cam     = T_world_cam[:3, 3]
            xyz_charuco = T_world_charuco[:3, 3]
            xyz_tcp     = T_world_tcp[:3, 3]
            xyz_base    = T_world_base[:3, 3]

            print(f"\n  --- Frame {f.stem} ---")
            print(f"  Camera    in world : x={xyz_cam[0]:+.4f}  y={xyz_cam[1]:+.4f}  z={xyz_cam[2]:+.4f}  m")
            print(f"  ChArUco   in world : x={xyz_charuco[0]:+.4f}  y={xyz_charuco[1]:+.4f}  z={xyz_charuco[2]:+.4f}  m")
            print(f"  TCP       in world : x={xyz_tcp[0]:+.4f}  y={xyz_tcp[1]:+.4f}  z={xyz_tcp[2]:+.4f}  m")
            print(f"  RobotBase in world : x={xyz_base[0]:+.4f}  y={xyz_base[1]:+.4f}  z={xyz_base[2]:+.4f}  m")
            print(f"  FK  TCP   in base  : x={T_base_tcp[0,3]:+.4f}  y={T_base_tcp[1,3]:+.4f}  z={T_base_tcp[2,3]:+.4f}  m")

            # Object markers
            for mid in obj_ids:
                key = f"T_cam_marker_{mid}"
                if key in d:
                    T_world_obj = T_world_cam @ d[key]
                    T_world_obj_lists[mid].append(T_world_obj)
                    p = T_world_obj[:3, 3]
                    print(f"  Marker {mid} in world : x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m")

            T_world_cam_list.append(T_world_cam)
            T_world_base_list.append(T_world_base)

        self.T_world_deskcam = _average_se3(T_world_cam_list)
        self.T_world_base    = _average_se3(T_world_base_list)

        # Average object marker poses (across frames where each was visible)
        self.T_world_objects = {}
        for mid, T_list in T_world_obj_lists.items():
            if T_list:
                self.T_world_objects[mid] = _average_se3(T_list)
                p = self.T_world_objects[mid][:3, 3]
                print(f"\n[Phase 1] Marker {mid} (averaged over {len(T_list)} frame(s)):"
                      f"  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m")
            else:
                print(f"\n[Phase 1] Marker {mid}: never detected — no pose computed.")

        self._results_dir.mkdir(exist_ok=True)
        save_kwargs = dict(T_world_deskcam=self.T_world_deskcam,
                           T_world_base=self.T_world_base)
        if self.T_world_objects:
            save_kwargs["object_marker_ids"] = np.array(
                list(self.T_world_objects.keys()), dtype=np.int32)
            for mid, T in self.T_world_objects.items():
                save_kwargs[f"T_world_marker_{mid}"] = T
        np.savez(self._results_dir / "phase1_results.npz", **save_kwargs)

        b = self.T_world_base[:3, 3]
        c = self.T_world_deskcam[:3, 3]
        print(f"\n[Phase 1] RESULT — averaged over {len(files)} pose(s):")
        print(f"  RobotBase in world  : x={b[0]:+.4f}  y={b[1]:+.4f}  z={b[2]:+.4f}  m")
        print(f"  DeskCam   in world  : x={c[0]:+.4f}  y={c[1]:+.4f}  z={c[2]:+.4f}  m")
        print()
        print("  Check: RobotBase x,y should match the physical distance from the")
        print("         ArUco marker to the robot base, measured along the marker axes.")
        print("         ArUco X = rightward along marker,  Y = upward along marker face,")
        print("         Z = out of marker face toward camera.")
        print(f"  Results saved to {self._results_dir / 'phase1_results.npz'}")
        return self.T_world_deskcam, self.T_world_base

    # =========================================================================
    # Phase 2  —  Eye-in-hand  CAPTURE
    # =========================================================================

    def capture_phase2(self, get_frame_fn, cam_matrix, dist_coeffs,
                       get_depth_fn=None,
                       joint_poses_deg=None,
                       window="Phase 2 – Eye-in-hand  |  ENTER=save  ESC=done"):
        """
        Interactive capture loop for Phase 2.

        get_frame_fn()    → BGR ndarray from the HAND camera (on the TCP).
        get_depth_fn()    → uint16 ndarray (depth in mm) from the hand camera.
                            When provided a pointcloud is saved on every ENTER.
        joint_poses_deg   → list of [j1..j6] in degrees.  When provided the
                            robot moves to each pose in turn; ENTER captures
                            the current pose and moves to the next one.
                            When None, the robot is not moved automatically.

        The ArUco world marker must be visible before ENTER saves.
        """
        self._p2_dir.mkdir(parents=True, exist_ok=True)
        cam_matrix  = np.asarray(cam_matrix,  dtype=float)
        dist_coeffs = np.asarray(dist_coeffs, dtype=float)
        np.savez(self._p2_dir / "intrinsics.npz",
                 camera_matrix=cam_matrix, dist_coeffs=dist_coeffs)

        poses    = list(joint_poses_deg) if joint_poses_deg is not None else []
        n_poses  = len(poses)
        pose_idx = 0
        counter  = 0

        if poses:
            print(f"[Phase 2] Moving to pose 1/{n_poses} …")
            self.move_joints(poses[pose_idx])

        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, 540)
        print(f"[Phase 2 capture] {window}")

        while True:
            frame = get_frame_fn()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis  = frame.copy()

            T_handcam_aruco = self._detect_world_aruco(gray, cam_matrix, dist_coeffs)
            ok = T_handcam_aruco is not None

            if ok:
                self._draw_axes(vis, T_handcam_aruco, cam_matrix, dist_coeffs, 0.06)

            pose_label = (f"Pose {pose_idx + 1}/{n_poses}"
                          if poses else f"Saved: {counter}")
            _draw_text(vis, [
                (f"ArUco world: {'OK ✓' if ok else 'NOT FOUND'}",
                 (0, 220, 0) if ok else (0, 50, 220)),
                (f"{pose_label}   Saved: {counter}   "
                 f"{'ENTER=save' if ok else '(marker needed)'}",
                 (0, 220, 220)),
            ])

            cv.imshow(window, vis)
            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == 13 and ok:
                T_base_tcp = self._fk()
                np.savez(self._p2_dir / f"pose_{counter:04d}.npz",
                         T_handcam_aruco=T_handcam_aruco,
                         T_base_tcp=T_base_tcp)
                cv.imwrite(str(self._p2_dir / f"frame_{counter:04d}.png"), vis)
                if get_depth_fn is not None:
                    depth = get_depth_fn()
                    if depth is not None:
                        pcd = self._make_pcd(frame, depth, cam_matrix)
                        o3d.io.write_point_cloud(
                            str(self._p2_dir / f"pcd_{counter:04d}.pcd"), pcd)
                print(f"  [Phase 2] Saved pose {counter}")
                counter  += 1
                pose_idx += 1

                if poses and pose_idx < n_poses:
                    print(f"[Phase 2] Moving to pose {pose_idx + 1}/{n_poses} …")
                    self.move_joints(poses[pose_idx])
                elif poses and pose_idx >= n_poses:
                    print("[Phase 2] All predefined poses captured.")
                    break

        cv.destroyWindow(window)
        print(f"[Phase 2 capture] Done — {counter} poses saved.")

    # =========================================================================
    # Phase 2  —  CALIBRATE
    # =========================================================================

    def calibrate_phase2(self, use_opencv_handeye=False):
        """
        Process Phase 2 data. Requires T_world_base from Phase 1.

        Math:
          T_{world←handcam} = inv(T_{handcam←aruco=world})   [= T_{world←aruco} @ inv(T_{handcam←aruco})]
          Also:
          T_{world←handcam} = T_{world←base} @ T_{base←tcp} @ T_{tcp←handcam}
          => T_{tcp←handcam} = inv(T_{base←tcp}) @ inv(T_{world←base}) @ inv(T_{handcam←aruco})

        use_opencv_handeye : if True, also run cv2.calibrateHandEye (TSAI) for
                             comparison. Requires >= 3 non-coplanar poses.

        Returns T_tcp_handcam (4×4).
        """
        # Load T_world_base if not already in memory
        if self.T_world_base is None:
            p = self._results_dir / "phase1_results.npz"
            if p.exists():
                self.T_world_base    = np.load(p)["T_world_base"]
                self.T_world_deskcam = np.load(p)["T_world_deskcam"]
            else:
                raise RuntimeError("T_world_base not available. Run calibrate_phase1 first.")

        files = sorted(self._p2_dir.glob("pose_*.npz"))
        if not files:
            raise RuntimeError("No Phase 2 poses found. Run capture_phase2 first.")

        intr = np.load(self._p2_dir / "intrinsics.npz")
        self.handcam_intrinsic = _o3d_intrinsic(intr["camera_matrix"])

        T_tcp_handcam_list = []
        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []  # for OpenCV hand-eye

        for f in files:
            d = np.load(f)
            T_handcam_aruco = d["T_handcam_aruco"]  # T_{handcam←world}
            T_base_tcp      = d["T_base_tcp"]        # T_{base←tcp}

            # Direct computation (one estimate per pose)
            T_tcp_handcam_i = (
                np.linalg.inv(T_base_tcp) @
                np.linalg.inv(self.T_world_base) @
                np.linalg.inv(T_handcam_aruco)
            )
            T_tcp_handcam_list.append(T_tcp_handcam_i)

            # Collect for OpenCV hand-eye (eye-in-hand convention):
            #   gripper2base = T_{base←tcp},  target2cam = T_{handcam←aruco}
            R_g2b.append(T_base_tcp[:3, :3])
            t_g2b.append(T_base_tcp[:3, 3:])
            R_t2c.append(T_handcam_aruco[:3, :3])
            t_t2c.append(T_handcam_aruco[:3, 3:])

        self.T_tcp_handcam = _average_se3(T_tcp_handcam_list)

        if use_opencv_handeye and len(files) >= 3:
            R_c2g, t_c2g = cv.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                                method=cv.CALIB_HAND_EYE_TSAI)
            T_opencv = np.eye(4)
            T_opencv[:3, :3] = R_c2g
            T_opencv[:3, 3]  = t_c2g.ravel()
            print("[Phase 2] OpenCV TSAI T_tcp_handcam:\n", T_opencv)

        self._results_dir.mkdir(exist_ok=True)
        np.savez(self._results_dir / "phase2_results.npz",
                 T_tcp_handcam=self.T_tcp_handcam)

        print("[Phase 2] T_tcp_handcam:\n", self.T_tcp_handcam)
        print(f"[Phase 2] Calibrated from {len(files)} poses.  Results saved.")
        return self.T_tcp_handcam

    # =========================================================================
    # Final Open3D visualisation  (world frame, live robot)
    # =========================================================================

    def visualize(self, window="Hand-Eye Calibration — World Frame",
                  deskcam_pcd_color=None,
                  handcam_pcd_color=None):
        """
        Live Open3D visualisation in the world frame.

        Loads results from disk automatically if not already in memory.
        Draws:
          • Coordinate frame at world origin
          • Grey flat square  — ArUco marker (world origin)
          • Orange sphere     — Robot TCP  (live FK, updates each frame)
          • Blue  frustum     — Desk camera (static, Phase 1)
          • Red   frustum     — Hand camera (live, Phase 2)

        deskcam_pcd_color  RGB list [r,g,b] in [0,1] to tint desk-cam PCDs,
                           or None to keep original colours.
        handcam_pcd_color  Same for hand-cam PCDs.

        Close the window or press Ctrl-C to exit.
        """
        self._load_results()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window, 1280, 720)
        if not vis.poll_events():
            vis.destroy_window()
            print("[Visualize] ERROR: could not create OpenGL window.")
            print("[Visualize] Try:  LIBGL_ALWAYS_SOFTWARE=1 python hand_eye_calibrator.py visualize")
            return

        # ── static geometry ────────────────────────────────────────────

        # World frame
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        vis.add_geometry(world_axes)

        # World ArUco marker as flat grey square centred on world origin (XY plane)
        s   = self.aruco_size_m
        aruco_sq = o3d.geometry.TriangleMesh.create_box(s, s, 0.002)
        aruco_sq.translate([-s / 2, -s / 2, -0.001])
        aruco_sq.paint_uniform_color([0.85, 0.85, 0.85])
        aruco_sq.compute_vertex_normals()
        vis.add_geometry(aruco_sq)

        # Object markers — coloured squares at their calibrated world poses
        _OBJ_COLORS = [
            [0.2, 0.85, 0.2],   # green
            [1.0, 0.80, 0.0],   # gold
            [0.8, 0.2,  0.8],   # magenta
            [0.2, 0.8,  0.8],   # cyan
        ]
        for i, (mid, T_world_obj) in enumerate(self.T_world_objects.items()):
            color  = _OBJ_COLORS[i % len(_OBJ_COLORS)]
            sm     = self.marker_sizes.get(mid, 0.10)
            obj_sq = o3d.geometry.TriangleMesh.create_box(sm, sm, 0.002)
            obj_sq.translate([-sm / 2, -sm / 2, -0.001])
            obj_sq.paint_uniform_color(color)
            obj_sq.compute_vertex_normals()
            obj_sq.transform(T_world_obj)
            vis.add_geometry(obj_sq)
            obj_ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            obj_ax.transform(T_world_obj)
            vis.add_geometry(obj_ax)

        # ── captured pointclouds ───────────────────────────────────────────

        # Phase 1: desk camera is fixed → all PCDs share T_world_deskcam
        if self.T_world_deskcam is not None:
            p1_count = 0
            for pcd_file in sorted(self._p1_dir.glob("pcd_*.pcd")):
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                if deskcam_pcd_color is not None:
                    pcd.paint_uniform_color(deskcam_pcd_color)
                pcd.transform(self.T_world_deskcam)
                vis.add_geometry(pcd)
                p1_count += 1
            if p1_count:
                color_note = f"color={deskcam_pcd_color}" if deskcam_pcd_color is not None else "original color"
                print(f"[Visualize] Loaded {p1_count} desk-camera PCD(s) ({color_note}).")

        # Phase 2: each PCD was shot from a different TCP pose
        if self.T_world_base is not None and self.T_tcp_handcam is not None:
            p2_count = 0
            for pcd_file in sorted(self._p2_dir.glob("pcd_*.pcd")):
                idx = pcd_file.stem[4:]           # "pcd_0000" → "0000"
                pose_file = self._p2_dir / f"pose_{idx}.npz"
                if not pose_file.exists():
                    continue
                T_base_tcp = np.load(pose_file)["T_base_tcp"]
                T_world_handcam = self.T_world_base @ T_base_tcp @ self.T_tcp_handcam
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                if handcam_pcd_color is not None:
                    pcd.paint_uniform_color(handcam_pcd_color)
                pcd.transform(T_world_handcam)
                vis.add_geometry(pcd)
                p2_count += 1
            if p2_count:
                color_note = f"color={handcam_pcd_color}" if handcam_pcd_color is not None else "original color"
                print(f"[Visualize] Loaded {p2_count} hand-camera PCD(s) ({color_note}).")

        # ── static geometry ────────────────────────────────────────────

        # Desk camera frustum (fixed in world)
        deskcam_ls = None
        if self.T_world_deskcam is not None and self.deskcam_intrinsic is not None:
            ext = np.linalg.inv(self.T_world_deskcam)          # T_{cam←world}
            deskcam_ls = o3d.geometry.LineSet.create_camera_visualization(
                self.deskcam_intrinsic, ext, scale=0.15)
            deskcam_ls.paint_uniform_color([0.0, 0.55, 1.0])   # blue
            vis.add_geometry(deskcam_ls)

        # Robot base frame (static once T_world_base is known)
        base_axes = None
        if self.T_world_base is not None:
            base_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            base_axes.transform(self.T_world_base)
            vis.add_geometry(base_axes)

        # ── dynamic geometry ───────────────────────────────────────────

        # TCP sphere (orange) — updated every loop via delta transform
        tcp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.018)
        tcp_sphere.compute_vertex_normals()
        tcp_sphere.paint_uniform_color([1.0, 0.40, 0.0])
        vis.add_geometry(tcp_sphere)
        tcp_sphere_center = np.zeros(3)  # track sphere centre for translation

        # ChArUco board (dark grey box + frame) — live, moves with TCP.
        # Only shown during eye-in-base: board is on TCP then.
        # In eye-in-hand the board is detached, so we hide it.
        show_charuco_board = (self.T_tcp_handcam is None)
        charuco_mesh = charuco_frame_vis = None
        charuco_prev_T = np.eye(4)
        if show_charuco_board:
            bw = self.charuco_squares_x * self.charuco_square_len
            bh = self.charuco_squares_y * self.charuco_square_len
            charuco_mesh = o3d.geometry.TriangleMesh.create_box(bw, bh, 0.002)
            charuco_mesh.paint_uniform_color([0.2, 0.2, 0.2])
            charuco_mesh.translate([0.0, 0.0, -0.001])
            charuco_mesh.compute_vertex_normals()
            charuco_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            vis.add_geometry(charuco_mesh)
            vis.add_geometry(charuco_frame_vis)

        # Hand camera frustum (red) — rebuilt each loop from live FK
        handcam_ls = None
        if self.T_tcp_handcam is not None and self.handcam_intrinsic is not None:
            handcam_ls = o3d.geometry.LineSet()
            handcam_ls.paint_uniform_color([1.0, 0.15, 0.15])  # red
            vis.add_geometry(handcam_ls)

        print(f"[Visualize] Running — close window to exit.")

        running = True
        while running:
            if self.T_world_base is not None:
                T_base_tcp  = self._fk()
                T_world_tcp = self.T_world_base @ T_base_tcp

                # Move TCP sphere
                new_center = T_world_tcp[:3, 3]
                tcp_sphere.translate(new_center - tcp_sphere_center)
                tcp_sphere_center = new_center.copy()
                vis.update_geometry(tcp_sphere)

                # Update ChArUco board (eye-in-base only)
                if show_charuco_board:
                    T_world_charuco = T_world_tcp @ self.T_tcp_to_charuco
                    delta = T_world_charuco @ np.linalg.inv(charuco_prev_T)
                    charuco_mesh.transform(delta)
                    charuco_frame_vis.transform(delta)
                    charuco_prev_T = T_world_charuco.copy()
                    vis.update_geometry(charuco_mesh)
                    vis.update_geometry(charuco_frame_vis)

                # Rebuild hand camera frustum at new TCP pose
                if handcam_ls is not None:
                    # T_{world←handcam} = T_{world←tcp} @ T_{tcp←handcam}
                    T_world_handcam = T_world_tcp @ self.T_tcp_handcam
                    ext_hand = np.linalg.inv(T_world_handcam)  # T_{handcam←world}
                    fresh = o3d.geometry.LineSet.create_camera_visualization(
                        self.handcam_intrinsic, ext_hand, scale=0.12)
                    handcam_ls.points = fresh.points
                    handcam_ls.lines  = fresh.lines
                    vis.update_geometry(handcam_ls)

            running = vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)

        vis.destroy_window()

    # =========================================================================
    # Base-pose refinement  —  move TCP to a known world point
    # =========================================================================

    def refine_base_pose(self, target_world_xyz, tcp_z_world=None, tcp_x_world=None,
                         speed=0.05, accel=0.05, skip_initial_move=False):
        """
        Refine T_world_base by touching a known world point.

        target_world_xyz : [x, y, z] in metres, world frame — physical fixture position.
        tcp_z_world      : expected TCP Z-axis direction in world frame at the fixture
                           (e.g. [0, 0, -1] for straight down).
        tcp_x_world      : expected TCP X-axis direction in world frame at the fixture.
                           When BOTH tcp_z_world and tcp_x_world are given, the fixture
                           fully constrains TCP orientation and T_world_base is solved
                           completely (translation + full rotation) from this single point.
                           When only tcp_z_world is given, pitch and roll are corrected
                           but yaw is left for refine_base_rotation.
                           When neither is given, only XY translation is corrected.
        skip_initial_move: if True, skip the computed moveL (robot is already near the
                           target, positioned via joint moves).
        """
        if self.T_world_base is None:
            p = self._results_dir / "phase1_results.npz"
            if p.exists():
                d = np.load(p)
                self.T_world_base    = d["T_world_base"]
                self.T_world_deskcam = d["T_world_deskcam"]
            else:
                raise RuntimeError("T_world_base not available. Run calibrate_phase1 first.")

        target_xyz   = np.asarray(target_world_xyz, dtype=float)
        T_base_world = np.linalg.inv(self.T_world_base)
        p_world_h    = np.append(target_xyz, 1.0)
        p_base       = (T_base_world @ p_world_h)[:3]

        current_tcp      = self.robot.getActualTCPPose()
        target_tcp_pose  = [p_base[0], p_base[1], p_base[2],
                            current_tcp[3], current_tcp[4], current_tcp[5]]

        rpy = ScipyR.from_matrix(self.T_world_base[:3, :3]).as_euler('xyz', degrees=True)
        print(f"\n[Refine] T_world_base rotation (rpy): r={rpy[0]:+.3f}°  p={rpy[1]:+.3f}°  y={rpy[2]:+.3f}°")
        print(f"[Refine] Target world : x={target_xyz[0]:+.4f}  y={target_xyz[1]:+.4f}  z={target_xyz[2]:+.4f}  m")
        if not skip_initial_move:
            print("[Refine] Moving robot …")
            self.robot_ctrl.moveL(target_tcp_pose, speed, accel)
            print("[Refine] Robot reached target.")
        print("[Refine] Freedrive enabled — move TCP to the target point, then press ENTER.")
        self.robot_ctrl.teachMode()

        _stop_live = threading.Event()
        def _live_print():
            while not _stop_live.is_set():
                try:
                    tcp = self.robot.getActualTCPPose()
                    p   = (self.T_world_base @ np.append(tcp[:3], 1.0))[:3]
                    print(f"\r  TCP world: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m    ", end="", flush=True)
                except Exception:
                    pass
                _stop_live.wait(0.15)
        _t = threading.Thread(target=_live_print, daemon=True)
        _t.start()
        try:
            resp = input("\n[Refine] ENTER=record  s=skip: ").strip().lower()
        finally:
            _stop_live.set()
            _t.join(timeout=1.0)
            self.robot_ctrl.endTeachMode()
            print("\n[Refine] Freedrive disabled.")

        if resp == "s":
            print("[Refine] Skipped — T_world_base unchanged.")
            return self.T_world_base, None

        T_base_tcp  = self._fk()
        p_tcp_world = (self.T_world_base @ np.append(T_base_tcp[:3, 3], 1.0))[:3]
        error       = target_xyz - p_tcp_world
        correction  = np.array([error[0], error[1], 0.0])   # Z not corrected
        print(f"\n[Refine] TCP world (computed) : x={p_tcp_world[0]:+.4f}  y={p_tcp_world[1]:+.4f}  z={p_tcp_world[2]:+.4f}  m")
        print(f"[Refine] TCP world (target)   : x={target_xyz[0]:+.4f}  y={target_xyz[1]:+.4f}  z={target_xyz[2]:+.4f}  m")
        print(f"[Refine] XY error (ignored Z) : dx={error[0]*1000:+.2f}  dy={error[1]*1000:+.2f}  dz={error[2]*1000:+.2f}  mm  (|xy|={np.linalg.norm(error[:2])*1000:.2f} mm)")
        if np.linalg.norm(correction) > 0.050:
            print(f"[Refine] WARNING: XY correction is {np.linalg.norm(correction)*1000:.1f} mm — larger than 50 mm.")
            print(f"[Refine] This may mean the TCP was not at the correct physical point, or the target coords are wrong.")
            resp2 = input("[Refine] Apply anyway?  ENTER=yes  s=skip: ").strip().lower()
            if resp2 == "s":
                print("[Refine] Skipped — T_world_base unchanged.")
                return self.T_world_base, None

        self.T_world_base = self.T_world_base.copy()
        p_A_base = T_base_tcp[:3, 3]

        if tcp_z_world is not None and tcp_x_world is not None:
            # Full rotation + translation from fixture (single-point, all 6 DOF)
            z = np.asarray(tcp_z_world, dtype=float); z /= np.linalg.norm(z)
            x = np.asarray(tcp_x_world, dtype=float); x /= np.linalg.norm(x)
            y = np.cross(z, x); y /= np.linalg.norm(y)
            x = np.cross(y, z)                         # re-orthogonalize
            R_world_tcp = np.column_stack([x, y, z])   # TCP axes expressed in world
            R_new = R_world_tcp @ np.linalg.inv(T_base_tcp[:3, :3])
            self.T_world_base[:3, :3] = R_new
            self.T_world_base[:3, 3]  = target_xyz - R_new @ p_A_base
            rpy = ScipyR.from_matrix(R_new).as_euler('xyz', degrees=True)
            print(f"[Refine] Full rotation solved  : r={rpy[0]:+.3f}°  p={rpy[1]:+.3f}°  y={rpy[2]:+.3f}°")

        elif tcp_z_world is not None:
            # Pitch + roll only (yaw left for refine_base_rotation)
            z_w = np.asarray(tcp_z_world, dtype=float); z_w /= np.linalg.norm(z_w)
            z_current = self.T_world_base[:3, :3] @ T_base_tcp[:3, 2]
            axis = np.cross(z_current, z_w)
            sin_a = np.linalg.norm(axis)
            cos_a = float(np.dot(z_current, z_w))
            if sin_a > 1e-6:
                R_pr  = ScipyR.from_rotvec(axis / sin_a * np.arctan2(sin_a, cos_a)).as_matrix()
                R_new = R_pr @ self.T_world_base[:3, :3]
                self.T_world_base[:3, :3] = R_new
                self.T_world_base[:3, 3]  = target_xyz - R_new @ p_A_base
                rpy = ScipyR.from_matrix(R_new).as_euler('xyz', degrees=True)
                print(f"[Refine] Pitch/roll corrected : r={rpy[0]:+.3f}°  p={rpy[1]:+.3f}°  y={rpy[2]:+.3f}°")
            else:
                self.T_world_base[:3, 3] += correction
                print("[Refine] TCP Z already aligned — translation-only correction applied.")

        else:
            # Translation only (legacy, no fixture orientation)
            self.T_world_base[:3, 3] += correction

        # Persist
        self._results_dir.mkdir(exist_ok=True)
        p1_path = self._results_dir / "phase1_results.npz"
        if p1_path.exists():
            saved = dict(np.load(p1_path))
        else:
            saved = dict(T_world_deskcam=self.T_world_deskcam)
        saved["T_world_base"] = self.T_world_base
        np.savez(p1_path, **saved)

        b = self.T_world_base[:3, 3]
        print(f"[Refine] Refined T_world_base: x={b[0]:+.4f}  y={b[1]:+.4f}  z={b[2]:+.4f}  m")
        print(f"[Refine] Saved to {p1_path}")
        return self.T_world_base, T_base_tcp

    # =========================================================================
    # Base-rotation refinement  —  two-point yaw correction
    # =========================================================================

    def refine_base_rotation(self, first_point_world_xyz, T_base_tcp_first, second_point_world_xyz):
        """
        Refine T_world_base yaw (Z rotation) using a second known world point.

        After refine_base_pose corrects translation at point A, freedrive the TCP
        to a second known world point B.  The A→B vector mismatch in world vs base
        frame reveals the yaw error; a world-Z rotation correction is applied and
        the translation is re-anchored at A so it stays accurate.

        first_point_world_xyz  : world XYZ of point A (refine_base_pose target).
        T_base_tcp_first       : 4×4 FK at point A (second return of refine_base_pose).
        second_point_world_xyz : world XYZ of point B (e.g. hover above marker 50).
        """
        p_A_world = np.asarray(first_point_world_xyz,  dtype=float)
        p_B_world = np.asarray(second_point_world_xyz, dtype=float)
        p_A_base  = np.asarray(T_base_tcp_first[:3, 3], dtype=float)

        print(f"\n[RotRefine] Second known point (world): "
              f"x={p_B_world[0]:+.4f}  y={p_B_world[1]:+.4f}  z={p_B_world[2]:+.4f}  m")
        print("[RotRefine] Freedrive — center TCP exactly above the second point, then press ENTER.")
        self.robot_ctrl.teachMode()

        _stop_live = threading.Event()
        def _live_print():
            while not _stop_live.is_set():
                try:
                    tcp = self.robot.getActualTCPPose()
                    p   = (self.T_world_base @ np.append(tcp[:3], 1.0))[:3]
                    print(f"\r  TCP world: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m    ",
                          end="", flush=True)
                except Exception:
                    pass
                _stop_live.wait(0.15)
        _t = threading.Thread(target=_live_print, daemon=True)
        _t.start()
        try:
            resp = input("\n[RotRefine] ENTER=record  s=skip: ").strip().lower()
        finally:
            _stop_live.set()
            _t.join(timeout=1.0)
            self.robot_ctrl.endTeachMode()
            print("\n[RotRefine] Freedrive disabled.")

        if resp == "s":
            print("[RotRefine] Skipped — T_world_base rotation unchanged.")
            return self.T_world_base

        T_base_tcp_B = self._fk()
        p_B_base = T_base_tcp_B[:3, 3]

        # A→B vector in base frame (FK ground truth) and in world frame (known)
        v_base     = p_B_base - p_A_base
        v_computed = self.T_world_base[:3, :3] @ v_base   # where current R maps it
        v_world    = p_B_world - p_A_world                 # where it should land

        dist_computed = np.linalg.norm(v_computed[:2])
        dist_world    = np.linalg.norm(v_world[:2])
        vb = v_computed[:2] / (dist_computed + 1e-12)
        vw = v_world[:2]    / (dist_world    + 1e-12)

        # Signed angle from v_computed to v_world in the XY plane
        angle = np.arctan2(vb[0]*vw[1] - vb[1]*vw[0],
                           vb[0]*vw[0] + vb[1]*vw[1])

        print(f"\n[RotRefine] |A→B| base XY  : {np.linalg.norm(v_base[:2])*1000:.1f} mm")
        print(f"[RotRefine] |A→B| world XY : {dist_world*1000:.1f} mm")
        print(f"[RotRefine] Yaw correction  : {np.degrees(angle):+.3f}°")

        if abs(np.degrees(angle)) > 10.0:
            print(f"[RotRefine] WARNING: {np.degrees(angle):+.1f}° > 10° — verify both freedrive points are correct.")
            resp2 = input("[RotRefine] Apply anyway?  ENTER=yes  s=skip: ").strip().lower()
            if resp2 == "s":
                print("[RotRefine] Skipped — T_world_base rotation unchanged.")
                return self.T_world_base

        # Apply world-Z rotation correction; re-anchor translation at A
        R_corr = ScipyR.from_euler('z', angle).as_matrix()
        R_new  = R_corr @ self.T_world_base[:3, :3]
        t_new  = p_A_world - R_new @ p_A_base

        self.T_world_base = self.T_world_base.copy()
        self.T_world_base[:3, :3] = R_new
        self.T_world_base[:3, 3]  = t_new

        self._results_dir.mkdir(exist_ok=True)
        p1_path = self._results_dir / "phase1_results.npz"
        if p1_path.exists():
            saved = dict(np.load(p1_path))
        else:
            saved = dict(T_world_deskcam=self.T_world_deskcam)
        saved["T_world_base"] = self.T_world_base
        np.savez(p1_path, **saved)

        rpy = ScipyR.from_matrix(R_new).as_euler('xyz', degrees=True)
        b   = self.T_world_base[:3, 3]
        print(f"[RotRefine] Refined T_world_base: x={b[0]:+.4f}  y={b[1]:+.4f}  z={b[2]:+.4f}  m")
        print(f"[RotRefine] New rotation  (rpy) : r={rpy[0]:+.3f}°  p={rpy[1]:+.3f}°  y={rpy[2]:+.3f}°")
        print(f"[RotRefine] Saved to {p1_path}")
        return self.T_world_base

    # =========================================================================
    # Move TCP to the centre of a detected object marker
    # =========================================================================

    def refine_marker_center_from_tcp(self, marker_id, z_offset_m=0.20):
        """
        Refine a saved marker pose after manually aligning the TCP above it.

        The current TCP is expected to be physically centered over the marker at
        the same hover height used by move_to_marker_center(). Only the marker
        XY center is corrected; marker Z and orientation stay camera-derived.
        """
        if self.T_world_base is None or not self.T_world_objects:
            self._load_results()

        if marker_id not in self.T_world_objects:
            print(f"[RefineMarker] Marker {marker_id} not found in T_world_objects. "
                  f"Available: {list(self.T_world_objects.keys())}")
            return None

        T_base_tcp = self._fk()
        p_tcp_world = (self.T_world_base @ np.append(T_base_tcp[:3, 3], 1.0))[:3]
        T_world_marker = self.T_world_objects[marker_id].copy()
        p_marker_world = T_world_marker[:3, 3].copy()
        p_expected_hover = p_marker_world + T_world_marker[:3, 2] * z_offset_m
        error = p_tcp_world - p_expected_hover

        print(f"\n[RefineMarker] TCP world now       : x={p_tcp_world[0]:+.4f}  y={p_tcp_world[1]:+.4f}  z={p_tcp_world[2]:+.4f}  m")
        print(f"[RefineMarker] Expected hover      : x={p_expected_hover[0]:+.4f}  y={p_expected_hover[1]:+.4f}  z={p_expected_hover[2]:+.4f}  m")
        print(f"[RefineMarker] Marker XY correction: dx={error[0]*1000:+.2f}  dy={error[1]*1000:+.2f}  mm")

        if np.linalg.norm(error[:2]) < 0.001:
            print("[RefineMarker] XY error is under 1 mm — marker pose unchanged.")
            return T_world_marker

        resp = input("[RefineMarker] Apply marker XY correction?  ENTER=yes  s=skip: ").strip().lower()
        if resp == "s":
            print("[RefineMarker] Skipped — marker pose unchanged.")
            return T_world_marker

        T_world_marker[:2, 3] += error[:2]
        self.T_world_objects[marker_id] = T_world_marker

        self._results_dir.mkdir(exist_ok=True)
        p1_path = self._results_dir / "phase1_results.npz"
        if p1_path.exists():
            saved = dict(np.load(p1_path))
        else:
            saved = dict(T_world_deskcam=self.T_world_deskcam,
                         T_world_base=self.T_world_base)
        saved["T_world_base"] = self.T_world_base
        saved["object_marker_ids"] = np.array(
            list(self.T_world_objects.keys()), dtype=np.int32)
        for mid, T in self.T_world_objects.items():
            saved[f"T_world_marker_{mid}"] = T
        np.savez(p1_path, **saved)

        p = T_world_marker[:3, 3]
        print(f"[RefineMarker] Refined marker {marker_id}: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m")
        print(f"[RefineMarker] Saved to {p1_path}")
        return T_world_marker

    def move_to_marker_center(self, marker_id, z_offset_m=0.10, lift_z_m=0.30, speed=0.05, accel=0.05):
        """
        Safely move the TCP to hover above the centre of a previously detected marker.

        marker_id   : ID of the marker (must be in T_world_objects from Phase 1).
        z_offset_m  : distance above the marker surface along the marker's Z axis (default 100 mm).
        lift_z_m    : how far to first lift the TCP in world Z before moving XY (default 150 mm).
        """
        if self.T_world_base is None or not self.T_world_objects:
            self._load_results()

        if marker_id not in self.T_world_objects:
            print(f"[MoveToMarker] Marker {marker_id} not found in T_world_objects. "
                  f"Available: {list(self.T_world_objects.keys())}")
            return

        T_base_world   = np.linalg.inv(self.T_world_base)
        current_tcp    = self.robot.getActualTCPPose()
        T_world_marker = self.T_world_objects[marker_id]
        marker_xy      = T_world_marker[:3, 3][:2]   # target X, Y in world

        p_now_world = (self.T_world_base @ np.append(current_tcp[:3], 1.0))[:3]
        print(f"\n[MoveToMarker] Current TCP  in world : x={p_now_world[0]:+.4f}  y={p_now_world[1]:+.4f}  z={p_now_world[2]:+.4f}  m")
        print(f"[MoveToMarker] Marker {marker_id}     in world : x={T_world_marker[0,3]:+.4f}  y={T_world_marker[1,3]:+.4f}  z={T_world_marker[2,3]:+.4f}  m")

        p_lift_world  = np.array([p_now_world[0], p_now_world[1], p_now_world[2] + lift_z_m])
        p_lift_base   = (T_base_world @ np.append(p_lift_world,  1.0))[:3]
        p_xy_world    = np.array([marker_xy[0], marker_xy[1], p_lift_world[2]])
        p_xy_base     = (T_base_world @ np.append(p_xy_world,   1.0))[:3]
        p_hover_world = T_world_marker[:3, 3] + np.array([0.0, 0.0, z_offset_m])
        p_hover_base  = (T_base_world @ np.append(p_hover_world, 1.0))[:3]

        print(f"[MoveToMarker] Planned moves (world frame):")
        print(f"  Step 1 lift  : ({p_lift_world[0]:+.4f}, {p_lift_world[1]:+.4f}, {p_lift_world[2]:+.4f})")
        print(f"  Step 2 XY    : ({p_xy_world[0]:+.4f},  {p_xy_world[1]:+.4f},  {p_xy_world[2]:+.4f})")
        print(f"  Step 3 hover : ({p_hover_world[0]:+.4f}, {p_hover_world[1]:+.4f}, {p_hover_world[2]:+.4f})")
        resp = input("[MoveToMarker] Proceed?  ENTER=yes  s=abort: ").strip().lower()
        if resp == "s":
            print("[MoveToMarker] Aborted.")
            return

        lift_pose  = [*p_lift_base,  current_tcp[3], current_tcp[4], current_tcp[5]]
        xy_pose    = [*p_xy_base,    current_tcp[3], current_tcp[4], current_tcp[5]]
        hover_pose = [*p_hover_base, current_tcp[3], current_tcp[4], current_tcp[5]]

        print(f"[MoveToMarker] Step 1 — lift …")
        self.robot_ctrl.moveL(lift_pose, speed, accel)
        print(f"[MoveToMarker] Step 2 — XY move …")
        self.robot_ctrl.moveL(xy_pose, speed, accel)
        print(f"[MoveToMarker] Step 3 — descend …")
        self.robot_ctrl.moveL(hover_pose, speed, accel)
        print("[MoveToMarker] Done.")

    # =========================================================================
    # Utility: load results from disk
    # =========================================================================

    def _load_results(self):
        p1 = self._results_dir / "phase1_results.npz"
        p2 = self._results_dir / "phase2_results.npz"

        if p1.exists():
            d = np.load(p1)
            if self.T_world_deskcam is None:
                self.T_world_deskcam = d["T_world_deskcam"]
            if self.T_world_base is None:
                self.T_world_base = d["T_world_base"]
            if not self.T_world_objects and "object_marker_ids" in d:
                for mid in d["object_marker_ids"].tolist():
                    key = f"T_world_marker_{mid}"
                    if key in d:
                        self.T_world_objects[mid] = d[key]

        if p2.exists() and self.T_tcp_handcam is None:
            self.T_tcp_handcam = np.load(p2)["T_tcp_handcam"]

        p1_intr = self._p1_dir / "intrinsics.npz"
        p2_intr = self._p2_dir / "intrinsics.npz"
        if self.deskcam_intrinsic is None and p1_intr.exists():
            self.deskcam_intrinsic = _o3d_intrinsic(
                np.load(p1_intr)["camera_matrix"])
        if self.handcam_intrinsic is None and p2_intr.exists():
            self.handcam_intrinsic = _o3d_intrinsic(
                np.load(p2_intr)["camera_matrix"])


# =============================================================================
# Predefined robot joint poses  (degrees)
# =============================================================================

# Phase 1 — Eye-in-base: desk camera must see both ArUco (world) and ChArUco (TCP)
POSES_EYE_IN_BASE = [
    [-281.11, -144.42, -107.58, -34.38, 137.03, 70.43],
    # Add more poses here for better calibration accuracy (aim for 10–15)
]

# Phase 2 — Eye-in-hand: hand camera must see the ArUco world marker
POSES_EYE_IN_HAND = [
    [-291.37, -148.59, -74.85, -229.31, -94.82, 251.42],
    # Add more poses here
]

# Refine-base — approach pose (hover) then lower pose (near reference point)
POSE_REFINE_APPROACH = [-295.69, -138.54, -102.44, -28.75, 89.03, 65.30]
POSE_REFINE_TARGET   = [-295.85, -150.18,  -98.67, -20.86, 88.99, 65.15]


# =============================================================================
# CLI entry point
# =============================================================================

def _build_calibrator():
    """Build a HandEyeCalibrator from hard-coded config (edit here)."""
    T_tcp_to_charuco = np.eye(4)
    # R_x(180°): ChArUco Z points UP (toward camera); TCP Z points DOWN (toward table).
    # Flipping Y and Z aligns the charuco frame with the physical TCP mounting.
    # T_tcp_to_charuco[:3, :3] = ScipyR.from_euler('x', 180, degrees=True).as_matrix()
    T_tcp_to_charuco[:3, 3] = [-0.0925, -0.2530, 0.0039]  # T_{tcp←charuco}

    return HandEyeCalibrator(
        robot_ip           = "192.168.50.70",
        save_dir           = "hand_eye_data",
        world_marker_id    = 10,                   # defines world origin
        marker_sizes       = {10: 0.090, 50: 0.040},  # world + object markers
        charuco_squares_x  = 4,
        charuco_squares_y  = 4,
        charuco_square_len = 0.046,
        charuco_marker_len = 0.038,
        aruco_dict_name    = "DICT_6X6_1000",
        T_tcp_to_charuco   = T_tcp_to_charuco,
    )


if __name__ == "__main__":
    import argparse
    from realsense_capture import RealSenseCapture  # same directory

    # ── Camera serial numbers ─────────────────────────────────────────────────
    # SERIAL_DESK = "250122073723"    # eye-in-base  : fixed desk camera
    SERIAL_DESK = "405622070050"    # eye-in-hand  : hand-mounted camera
    SERIAL_HAND = "323622273275"
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Hand-eye calibration — two-phase workflow",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "phase",
        choices=["record-poses", "eye-in-base", "refine-base", "eye-in-hand", "visualize"],
        help=(
            "record-poses : freedrive robot to N poses and save joint angles to file\n"
            "eye-in-base  : Phase 1 — desk camera (%(desk)s) sees ArUco (world) + ChArUco (TCP)\n"
            "refine-base  : freedrive TCP to known world point → refine T_world_base, then move to marker 50\n"
            "eye-in-hand  : Phase 2 — hand camera (%(hand)s) sees ArUco (world)\n"
            "visualize    : Open3D live view using saved results"
        ) % {"desk": SERIAL_DESK, "hand": SERIAL_HAND},
    )
    parser.add_argument(
        "--n-poses", type=int, default=10,
        help="(record-poses) number of poses to record (default 10)",
    )
    parser.add_argument(
        "--width",  type=int, default=1280, help="Camera stream width  (default 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,  help="Camera stream height (default 720)",
    )
    parser.add_argument(
        "--fps",    type=int, default=30,   help="Camera stream FPS    (default 30)",
    )
    parser.add_argument(
        "--opencv-handeye", action="store_true",
        help="(eye-in-hand) also run OpenCV TSAI solver alongside direct method",
    )
    args = parser.parse_args()

    cal = _build_calibrator()

    POSES_FILE = cal.save_dir / "eye_in_base_poses.json"

    if args.phase == "record-poses":
        if POSES_FILE.exists():
            with open(POSES_FILE) as f:
                existing = json.load(f)
            print(f"[RecordPoses] {len(existing)} poses already saved in {POSES_FILE}")
            resp = input("[RecordPoses] (r)edo from scratch  or  (k)eep existing: ").strip().lower()
            if resp != "r":
                print("[RecordPoses] Keeping existing poses.")
                exit(0)

        n = args.n_poses
        poses = []

        rs_cam = RealSenseCapture(
            serial=SERIAL_DESK, width=args.width, height=args.height, fps=args.fps)
        rs_cam.start()
        cam_matrix  = rs_cam.camera_matrix()
        dist_coeffs = rs_cam.dist_coeffs()

        win = "Record Poses — SPACE=freedrive on/off  ENTER=save (both markers needed)  ESC=done"
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 960, 540)

        freedrive_on = False
        try:
            while len(poses) < n:
                frame = rs_cam.get_bgr()
                if frame is None:
                    time.sleep(0.01)
                    continue

                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis  = frame.copy()

                all_markers   = cal._detect_all_markers(gray, cam_matrix, dist_coeffs)
                T_cam_aruco   = all_markers.get(cal.world_marker_id)
                T_cam_charuco = cal._detect_charuco(gray, cam_matrix, dist_coeffs)

                for mid, T_cm in all_markers.items():
                    size = 0.06 if mid == cal.world_marker_id else 0.04
                    cal._draw_axes(vis, T_cm, cam_matrix, dist_coeffs, size)
                if T_cam_charuco is not None:
                    cal._draw_axes(vis, T_cam_charuco, cam_matrix, dist_coeffs, 0.03)

                ok_a = T_cam_aruco   is not None
                ok_c = T_cam_charuco is not None
                both = ok_a and ok_c

                _draw_text(vis, [
                    (f"Pose {len(poses)+1}/{n}   Freedrive: {'ON — ENTER=save' if freedrive_on else 'OFF — SPACE to enable'}",
                     (0, 220, 220) if freedrive_on else (180, 180, 180)),
                    (f"ArUco world ({cal.world_marker_id}): {'OK ✓' if ok_a else 'NOT FOUND'}",
                     (0, 220, 0) if ok_a else (0, 50, 220)),
                    (f"ChArUco TCP             : {'OK ✓' if ok_c else 'NOT FOUND'}",
                     (0, 220, 0) if ok_c else (0, 50, 220)),
                    (f"Saved: {len(poses)}/{n}   {'ready to save ✓' if both else 'need both markers'}",
                     (0, 220, 0) if both else (0, 150, 220)),
                ])

                cv.imshow(win, vis)
                key = cv.waitKey(1) & 0xFF

                if key == 27:   # ESC — stop early
                    break
                elif key == 32:  # SPACE — toggle freedrive
                    if not freedrive_on:
                        cal.robot_ctrl.teachMode()
                        freedrive_on = True
                    else:
                        cal.robot_ctrl.endTeachMode()
                        freedrive_on = False
                elif key == 13 and freedrive_on and both:  # ENTER — save pose
                    q_deg = np.degrees(cal.robot.getActualQ()).tolist()
                    cal.robot_ctrl.endTeachMode()
                    freedrive_on = False
                    poses.append(q_deg)
                    print(f"\n[RecordPoses] Saved pose {len(poses)}: "
                          f"[{', '.join(f'{v:.2f}' for v in q_deg)}]")
        finally:
            if freedrive_on:
                cal.robot_ctrl.endTeachMode()
            cv.destroyWindow(win)
            rs_cam.stop()

        if poses:
            POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(POSES_FILE, "w") as f:
                json.dump(poses, f, indent=2)
            print(f"\n[RecordPoses] Saved {len(poses)} poses to {POSES_FILE}")

    elif args.phase == "visualize":
        cal.visualize()

    elif args.phase == "refine-base":
        # Step 1: move to approach (hover) pose
        print("[Refine] Moving to approach pose …")
        cal.move_joints(POSE_REFINE_APPROACH)
        input("[Refine] At approach pose. Press ENTER to descend to target pose …")
        # Step 2: descend to the reference point
        print("[Refine] Descending to target pose …")
        cal.move_joints(POSE_REFINE_TARGET)
        # Step 3: full 6-DOF correction at fixture point A
        #   tcp_z_world : fixture forces TCP Z straight down  → [0, 0, -1]
        #   tcp_x_world : fixture forces TCP X to this world direction → measure physically
        _, T_base_tcp_A = cal.refine_base_pose(
            [0.0750, 0, 0.17250],
            tcp_z_world=[0, 0, -1],
            tcp_x_world=[0, -1, 0],  # TCP X → world -Y  (measured at fixture)
            skip_initial_move=True, speed=0.3, accel=0.3,
        )
        # Step 4: move to marker 50 — verification only, no correction applied
        cal.move_to_marker_center(marker_id=50, z_offset_m=0.20, speed=0.3, accel=0.3)
        # time.sleep(1.0)
        # # Move above world origin (marker 0 = [0,0,0] in world frame): lift → XY → descend
        # T_bw = np.linalg.inv(cal.T_world_base)
        # current_tcp = cal.robot.getActualTCPPose()
        # ori = [current_tcp[3], current_tcp[4], current_tcp[5]]
        # p_now_world = (cal.T_world_base @ np.append(current_tcp[:3], 1.0))[:3]
        # # Step 1: lift
        # p_lift = np.array([p_now_world[0], p_now_world[1], p_now_world[2] + 0.15])
        # cal.robot_ctrl.moveL([*(T_bw @ np.append(p_lift, 1.0))[:3], *ori], 0.3, 0.3)
        # # Step 2: move XY to above origin at lifted height
        # p_xy = np.array([0.0, 0.0, p_lift[2]])
        # cal.robot_ctrl.moveL([*(T_bw @ np.append(p_xy, 1.0))[:3], *ori], 0.3, 0.3)
        # # Step 3: descend to hover height
        # p_hover = np.array([0.0, 0.0, 0.20])
        # p_hover_base = (T_bw @ np.append(p_hover, 1.0))[:3]
        # print(f"\n[MoveToOrigin] Descending above world origin → base ({p_hover_base[0]:+.4f}, {p_hover_base[1]:+.4f}, {p_hover_base[2]:+.4f})")
        # cal.robot_ctrl.moveL([*p_hover_base, *ori], 0.3, 0.3)
        
        

        

    else:
        serial = SERIAL_DESK if args.phase == "eye-in-base" else SERIAL_HAND

        rs_cam = RealSenseCapture(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            enable_depth=True,
        )
        rs_cam.start()

        try:
            cam_matrix  = rs_cam.camera_matrix()
            dist_coeffs = rs_cam.dist_coeffs()

            if args.phase == "eye-in-base":
                if POSES_FILE.exists():
                    with open(POSES_FILE) as f:
                        joint_poses = json.load(f)
                    print(f"[Phase 1] Loaded {len(joint_poses)} poses from {POSES_FILE}")
                else:
                    joint_poses = POSES_EYE_IN_BASE
                    print(f"[Phase 1] No poses file found — using {len(joint_poses)} hardcoded pose(s)")
                cal.capture_phase1(rs_cam.get_bgr, cam_matrix, dist_coeffs,
                                   get_depth_fn=rs_cam.get_depth_m,
                                   joint_poses_deg=joint_poses)
                cal.calibrate_phase1()
                # Test: move to marker 50 using raw phase-1 T_world_base (no refinement)
                # cal.move_to_marker_center(marker_id=50, z_offset_m=0.10)
                cal.visualize()

            elif args.phase == "eye-in-hand":
                cal.capture_phase2(rs_cam.get_bgr, cam_matrix, dist_coeffs,
                                   get_depth_fn=rs_cam.get_depth_m,
                                   joint_poses_deg=POSES_EYE_IN_HAND)
                cal.calibrate_phase2(use_opencv_handeye=args.opencv_handeye)

        finally:
            rs_cam.stop()
