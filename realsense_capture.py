"""
realsense_capture.py — Thin wrapper around pyrealsense2 for calibration use.

Usage
-----
    rs = RealSenseCapture(serial="405622070082", width=1280, height=720, fps=30)
    rs.start()

    frame_bgr   = rs.get_bgr()         # latest BGR frame (blocks until one arrives)
    K           = rs.camera_matrix()   # 3×3 numpy array
    dist        = rs.dist_coeffs()     # (5,) numpy array
    depth_m     = rs.get_depth_m()     # aligned depth image in metres (float32)

    rs.stop()
"""

import threading
import time
import numpy as np
import cv2 as cv
import pyrealsense2 as rs


class RealSenseCapture:
    """
    Starts a RealSense colour (+ optional depth) stream in a background thread
    so that get_bgr() always returns the freshest frame without blocking the
    main loop on the SDK's own wait_for_frames() latency.

    Parameters
    ----------
    serial  : Device serial number string.  None = first connected device.
    width   : Stream width  (pixels).
    height  : Stream height (pixels).
    fps     : Target framerate.
    enable_depth : Also enable the depth stream (aligned to colour).
    """

    def __init__(self,
                 serial=None,
                 width=1280,
                 height=720,
                 fps=30,
                 enable_depth=False):
        self._serial       = serial
        self._width        = width
        self._height       = height
        self._fps          = fps
        self._enable_depth = enable_depth

        self._pipeline  = rs.pipeline()
        self._config    = rs.config()
        self._profile   = None          # set after start()
        self._intrinsics = None         # rs2_intrinsics

        self._lock       = threading.Lock()
        self._bgr_frame  = None         # latest BGR image
        self._depth_frame = None        # latest depth (metres, float32)
        self._running    = False
        self._thread     = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start the RealSense pipeline and the background capture thread."""
        if self._serial:
            self._config.enable_device(self._serial)

        self._config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

        self._profile = self._pipeline.start(self._config)

        # Read intrinsics from the active stream profile
        color_profile = (
            self._profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        self._intrinsics = color_profile.get_intrinsics()

        # Align depth to colour (if depth enabled)
        if self._enable_depth:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        # Warm up: discard a few frames so auto-exposure settles
        for _ in range(20):
            self._pipeline.wait_for_frames()

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        dev_info = self._profile.get_device().get_info(rs.camera_info.serial_number)
        print(f"[RealSenseCapture] Started — serial={dev_info}  "
              f"{self._width}×{self._height}@{self._fps}fps")

    def stop(self):
        """Stop the capture thread and pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._pipeline.stop()
        print("[RealSenseCapture] Stopped.")

    # =========================================================================
    # Background capture thread
    # =========================================================================

    def _capture_loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)

                if self._align:
                    frames = self._align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                bgr = np.asanyarray(color_frame.get_data())  # already BGR (bgr8)

                depth = None
                if self._enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_scale = (
                            self._profile.get_device()
                            .first_depth_sensor()
                            .get_depth_scale()
                        )
                        depth = (
                            np.asanyarray(depth_frame.get_data()).astype(np.float32)
                            * depth_scale
                        )

                with self._lock:
                    self._bgr_frame   = bgr
                    self._depth_frame = depth

            except RuntimeError:
                # wait_for_frames timed out — pipeline may be stopping
                pass

    # =========================================================================
    # Public accessors
    # =========================================================================

    def get_bgr(self, timeout=5.0):
        """
        Return the latest BGR frame as a (H, W, 3) uint8 numpy array.
        Blocks until the first frame arrives (up to *timeout* seconds).
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._bgr_frame is not None:
                    return self._bgr_frame.copy()
            if time.time() > deadline:
                raise TimeoutError("RealSense: no colour frame received within timeout.")
            time.sleep(0.005)

    def get_depth_m(self, timeout=5.0):
        """
        Return the latest depth image as a (H, W) float32 numpy array (metres).
        Returns None if depth was not enabled.
        """
        if not self._enable_depth:
            return None
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._depth_frame is not None:
                    return self._depth_frame.copy()
            if time.time() > deadline:
                raise TimeoutError("RealSense: no depth frame received within timeout.")
            time.sleep(0.005)

    def camera_matrix(self):
        """Return the 3×3 colour camera intrinsic matrix K."""
        i = self._intrinsics
        return np.array([
            [i.fx,  0.0, i.ppx],
            [ 0.0, i.fy, i.ppy],
            [ 0.0,  0.0,  1.0],
        ], dtype=np.float64)

    def dist_coeffs(self):
        """Return the distortion coefficients as a (5,) float64 array."""
        return np.asarray(self._intrinsics.coeffs, dtype=np.float64)

    def resolution(self):
        """Return (width, height) of the colour stream."""
        return self._intrinsics.width, self._intrinsics.height


# =============================================================================
# Standalone preview
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RealSense live preview")
    parser.add_argument("--serial", default=None, help="Device serial number")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--depth",  action="store_true", help="Show depth overlay")
    args = parser.parse_args()

    cam = RealSenseCapture(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=args.depth,
    )
    cam.start()

    print("K =\n", cam.camera_matrix())
    print("dist =", cam.dist_coeffs())
    print("Press ESC to quit.")

    cv.namedWindow("RealSense Preview", cv.WINDOW_NORMAL)
    cv.resizeWindow("RealSense Preview", 960, 540)

    try:
        while True:
            bgr = cam.get_bgr()

            if args.depth:
                depth = cam.get_depth_m()
                if depth is not None:
                    # Normalise depth to 0-255 for display (clip at 4 m)
                    depth_vis = np.clip(depth / 4.0, 0, 1)
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    depth_color = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)
                    bgr = cv.addWeighted(bgr, 0.6, depth_color, 0.4, 0)

            cv.imshow("RealSense Preview", bgr)
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.stop()
        cv.destroyAllWindows()
