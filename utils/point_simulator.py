from typing import List, Tuple
import numpy as np

class WorkspaceClipper:
    def __init__(self, polygon_xy: List[Tuple[float, float]], z0=0.0, z1=0.5, xy_units="mm"):
        self.z0 = float(z0)
        self.z1 = float(z1)
        self.xy_units = str(xy_units).lower()

        poly = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
        if poly.shape[0] < 3:
            raise ValueError("polygon_xy must contain at least 3 points")

        # Convert polygon to meters
        if self.xy_units == "mm":
            poly = poly * 0.001
        elif self.xy_units == "m":
            pass
        else:
            raise ValueError('xy_units must be "mm" or "m"')

        # Ensure closed polygon
        if np.linalg.norm(poly[0] - poly[-1]) > 1e-9:
            poly = np.vstack([poly, poly[0]])

        self.poly_m = poly  # (N+1,2) closed loop in meters

    # -------------------- 2D helpers --------------------
    @staticmethod
    def _point_on_segment(px, py, ax, ay, bx, by, eps):
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > eps:
            return False
        dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
        return dot <= eps

    def _point_in_polygon_2d(self, px, py, include_boundary=True, eps=1e-9) -> bool:
        inside = False
        poly = self.poly_m
        for i in range(len(poly) - 1):
            ax, ay = poly[i]
            bx, by = poly[i + 1]

            if include_boundary and self._point_on_segment(px, py, ax, ay, bx, by, eps):
                return True

            # Ray casting
            cond = (ay > py) != (by > py)
            if cond:
                x_int = ax + (bx - ax) * (py - ay) / (by - ay + 0.0)
                if px < x_int:
                    inside = not inside
        return inside

    @staticmethod
    def _closest_point_on_segment_2d(px, py, ax, ay, bx, by, eps=1e-12):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby
        if denom < eps:
            return np.array([ax, ay], dtype=float)
        t = (apx * abx + apy * aby) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return np.array([ax + t * abx, ay + t * aby], dtype=float)

    # -------------------- main geometry routine --------------------
    def closest_point_in_prism(self, p_world_m: np.ndarray, include_boundary=True, eps=1e-9):
        """
        Returns:
          inside (bool),
          closest_m (np.ndarray shape (3,))  # always inside/on prism
          dist (float)                       # Euclidean distance in meters
        """
        p = np.asarray(p_world_m, dtype=float).reshape(3,)
        x, y, z = float(p[0]), float(p[1]), float(p[2])

        zmin, zmax = (self.z0, self.z1) if self.z0 <= self.z1 else (self.z1, self.z0)

        # inside test
        inside_xy = self._point_in_polygon_2d(x, y, include_boundary=include_boundary, eps=eps)
        if include_boundary:
            inside_z = (z >= zmin - eps) and (z <= zmax + eps)
        else:
            inside_z = (z > zmin + eps) and (z < zmax - eps)
        inside = bool(inside_xy and inside_z)

        # closest XY (either itself if inside_xy, or closest boundary point)
        if inside_xy:
            closest_xy = np.array([x, y], dtype=float)
        else:
            best = None
            best_d2 = float("inf")
            poly = self.poly_m
            for i in range(len(poly) - 1):
                ax, ay = poly[i]
                bx, by = poly[i + 1]
                c = self._closest_point_on_segment_2d(x, y, ax, ay, bx, by, eps=eps)
                d2 = float((c[0] - x) ** 2 + (c[1] - y) ** 2)
                if d2 < best_d2:
                    best_d2 = d2
                    best = c
            closest_xy = best if best is not None else np.array([x, y], dtype=float)

        # closest Z is just clamped
        z_clamped = float(np.clip(z, zmin, zmax))

        closest_m = np.array([closest_xy[0], closest_xy[1], z_clamped], dtype=float)
        dist = float(np.linalg.norm(closest_m - p))
        return inside, closest_m, dist

    # -------------------- convenience wrappers --------------------
    def closest(self, p_world_m: np.ndarray):
        return self.closest_point_in_prism(p_world_m, include_boundary=True)

    def clip(self, p_world_m: np.ndarray) -> np.ndarray:
        _, q, _ = self.closest(p_world_m)
        return q


def lissajous_point_world_m(t: float, workspace_xy_mm) -> np.ndarray:
    """
    Smooth test trajectory in WORLD frame (meters).
    Uses your workspace polygon bbox to pick a reasonable center/scale.
    """
    poly_m = np.asarray(workspace_xy_mm, dtype=float) * 0.001
    xmin, ymin = poly_m[:,0].min(), poly_m[:,1].min()
    xmax, ymax = poly_m[:,0].max(), poly_m[:,1].max()

    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    ax, ay = 0.35*(xmax-xmin), 0.35*(ymax-ymin)  # keep inside-ish but can go out

    # Lissajous params
    a = 3.0
    b = 2.0
    w = 2.0 * np.pi * 0.05   # ~0.08 Hz base
    delta = np.deg2rad(60.0)

    time_scale = 0.3
    x = cx + ax * np.sin(a * w * (time_scale*t) + delta)
    y = cy + ay * np.sin(b * w * (time_scale*t))
    z = 0.20 + 0.05 * np.sin(1.0 * w * (time_scale*t))

    return np.array([x, y, z], dtype=float)
