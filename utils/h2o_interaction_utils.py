# utils/h2o_interaction_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# Helpers (pure functions)
# ------------------------------------------------------------

def valid_center(c: Optional[np.ndarray]) -> bool:
    return c is not None and np.isfinite(c).all()

def nearest_within(
    p: np.ndarray,
    centers: Dict[str, np.ndarray],
    r: float
) -> Tuple[str, float]:
    best_k = ""
    best_d = float("inf")
    for k, c in centers.items():
        if not valid_center(c):
            continue
        d = float(np.linalg.norm(p - c))
        if d < best_d:
            best_d = d
            best_k = (k or "").strip().lower()
    if best_k and best_d <= r:
        return best_k, best_d
    return "", float("inf")

def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

# ------------------------------------------------------------
# Hand parsing
# ------------------------------------------------------------

def get_hand_joint_positions_o3d(
    hand_data: dict,
    hand_name: str,
    unity_to_open3d_vector: Callable[[Any], np.ndarray],
    max_joints: int = 24,
) -> Optional[List[np.ndarray]]:
    """
    Extracts up to `max_joints` joints in Open3D/world coordinates from Unity hand message.

    Returns:
      - List[np.ndarray] length == max_joints (NaNs padded) OR None if invalid.
    """
    if not hand_data or not isinstance(hand_data, dict):
        return None
    hands = hand_data.get("hands", {})
    h = hands.get(hand_name, None)
    if not h or "groups" not in h:
        return None

    pts: List[np.ndarray] = []
    try:
        for group in h["groups"].values():
            for joint in group:
                if joint is None:
                    pts.append(np.array([np.nan, np.nan, np.nan], dtype=float))
                else:
                    pts.append(np.asarray(unity_to_open3d_vector(joint["position"]), dtype=float))
                if len(pts) >= max_joints:
                    break
            if len(pts) >= max_joints:
                break
    except Exception:
        return None

    while len(pts) < max_joints:
        pts.append(np.array([np.nan, np.nan, np.nan], dtype=float))

    return pts

def get_right_palm_world(
    real_hand_data: dict,
    palm_idx: int,
    unity_to_open3d_vector: Callable[[Any], np.ndarray],
    hand_name: str = "RightHand",
) -> Optional[np.ndarray]:
    hand = get_hand_joint_positions_o3d(
        hand_data=real_hand_data,
        hand_name=hand_name,
        unity_to_open3d_vector=unity_to_open3d_vector,
        max_joints=max(24, palm_idx + 1),
    )
    if hand is None or len(hand) <= palm_idx:
        return None
    palm = hand[palm_idx]
    if palm is None or (not np.isfinite(palm).all()):
        return None
    return palm.astype(float)

# ------------------------------------------------------------
# Centers from Visualizer caches (still “pure” given explicit args)
# ------------------------------------------------------------

def get_top_centers_world(
    visualize: bool,
    visualizer: Any,
    objects_in_scene: Optional[List[str]],
    object_pose_estimator: Any,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not visualize or visualizer is None:
        return out

    if objects_in_scene:
        names = list(objects_in_scene)
    else:
        objs = getattr(object_pose_estimator, "objects", []) if object_pose_estimator is not None else []
        names = [o.name for o in objs]

    for name in names:
        c = visualizer.get_cached_top_face_center_world(name)
        if c is None:
            continue
        out[(name or "").strip().lower()] = np.asarray(c, dtype=float).reshape(3,)
    return out

def get_vr_top_centers_world(
    visualizer: Any,
    objects_vr_poses_o3d: Optional[Dict[str, dict]],
) -> Dict[str, np.ndarray]:
    """
    Returns: dict {key_lower: center_xyz_in_world}
    Uses Visualizer cached VR top-face centers (computed from VR box LineSet).
    Filters out inactive/missing objects if objects_vr_poses_o3d is provided.
    """
    out: Dict[str, np.ndarray] = {}
    if visualizer is None:
        return out

    def _add(name: str, c: Optional[np.ndarray]):
        if c is None:
            return
        c = np.asarray(c, dtype=float).reshape(3,)
        if not np.isfinite(c).all():
            return
        key = (name or "").strip().lower()
        if key:
            out[key] = c

    vr_info = objects_vr_poses_o3d if isinstance(objects_vr_poses_o3d, dict) else None

    # If we have per-object info, filter
    if vr_info and len(vr_info) > 0:
        for name, info in vr_info.items():
            if not isinstance(info, dict):
                continue

            active = bool(info.get("active", True))
            missing = bool(info.get("missing", False))
            T = info.get("T", None)

            visible = (T is not None) and np.any(np.asarray(T)) and active and (not missing)
            if not visible:
                continue

            c = None
            if hasattr(visualizer, "get_cached_vr_top_face_center_world"):
                c = visualizer.get_cached_vr_top_face_center_world(name)
            if c is None and hasattr(visualizer, "get_vr_top_face_center_world"):
                c = visualizer.get_vr_top_face_center_world(name)

            _add(name, c)

        return out

    # Otherwise use cache directly
    cache = getattr(visualizer, "vr_top_face_centers", {}) or {}
    for name, c in cache.items():
        _add(name, c)

    return out

# ------------------------------------------------------------
# Baseline selection logic
# ------------------------------------------------------------

@dataclass
class Baseline1State:
    selected: str = ""
    candidate: str = ""
    t0: Optional[float] = None
    release_r: Optional[float] = None

def baseline1_update(
    now: float,
    palm: np.ndarray,
    centers: Dict[str, np.ndarray],
    *,
    enter_r: float = 0.10,
    dwell_s: float = 3.0,
    use_adaptive_release: bool = True,
    keep_r_fixed: Optional[float] = None,
    release_mult: float = 1.5,
    release_min: Optional[float] = None,
    release_max: float = 0.30,
    state: Optional[Baseline1State] = None,
) -> Tuple[str, float, Baseline1State]:
    """
    Baseline1 (stateful):
      - Candidate = nearest within enter_r
      - Dwell dwell_s to select
      - Once selected, keep until leaving release radius
      - Progress: dwell progress [0..1] (candidate) or 1.0 (selected)

    Returns: (key_lower, progress01, updated_state)
    """
    st = state if state is not None else Baseline1State()

    enter_r = float(enter_r)
    dwell_s = float(dwell_s)

    keep_r = float(keep_r_fixed) if keep_r_fixed is not None else enter_r
    rmin = float(release_min) if release_min is not None else enter_r
    rmax = float(release_max)

    # 1) If selected: keep it until leaving release radius
    if st.selected:
        c = centers.get(st.selected, None)
        if not valid_center(c):
            return "", 0.0, Baseline1State()

        d = float(np.linalg.norm(palm - c))

        if use_adaptive_release:
            rr = st.release_r if st.release_r is not None else keep_r
            if d <= rr:
                return st.selected, 1.0, st
        else:
            if d <= keep_r:
                return st.selected, 1.0, st

        # left the zone -> clear selection
        return "", 0.0, Baseline1State()

    # 2) Not selected: find nearest within enter_r
    cand, cand_d = nearest_within(palm, centers, enter_r)
    if cand == "":
        st.candidate = ""
        st.t0 = None
        return "", 0.0, st

    # candidate changed -> reset dwell
    if cand != st.candidate:
        st.candidate = cand
        st.t0 = now

    elapsed = 0.0 if (st.t0 is None) else float(now - st.t0)
    prog = float(np.clip(elapsed / max(dwell_s, 1e-6), 0.0, 1.0))

    if prog >= 1.0:
        st.selected = cand
        st.candidate = cand

        if use_adaptive_release:
            rr = release_mult * float(cand_d)
            rr = float(np.clip(rr, rmin, rmax))
            st.release_r = rr
        else:
            st.release_r = None

        return cand, 1.0, st

    return "", 0, st

def baseline2_update(
    palm: np.ndarray,
    centers: Dict[str, np.ndarray],
    *,
    r: float = 0.20,
    use_distance_progress: bool = True,
) -> Tuple[str, float]:
    """
    Baseline2 (stateless):
      - Always select nearest object if within radius r
      - Progress: distance-based (1 at center, 0 at boundary) OR always 1
    """
    r = float(r)
    best_key = ""
    best_d = float("inf")

    for k, c in centers.items():
        if not valid_center(c):
            continue
        d = float(np.linalg.norm(palm - c))
        if d < best_d:
            best_d = d
            best_key = (k or "").strip().lower()

    if best_key == "" or best_d > r:
        return "", 0.0

    if not use_distance_progress:
        return best_key, 1.0

    x = 1.0 - float(np.clip(best_d / max(r, 1e-6), 0.0, 1.0))
    prog = smoothstep(x)

    prog = 1.0
    return best_key, float(np.clip(prog, 0.0, 1.0))

# ------------------------------------------------------------
# Target / completion helpers
# ------------------------------------------------------------

def find_target_containing_point(
    p_world: Optional[np.ndarray],
    target_points_xyz: Dict[str, dict],
    print_bool: bool = False
) -> Optional[str]:
    # ---- guard: missing / invalid point ----
    if p_world is None:
        return None

    try:
        p = np.asarray(p_world, dtype=float).reshape(3,)
    except Exception:
        return None

    if not np.isfinite(p).all():
        return None

    best_tid = None
    best_d = float("inf")

    for tid, info in (target_points_xyz or {}).items():
        c_raw = info.get("pos", None)
        if c_raw is None:
            continue
        try:
            c = np.asarray(c_raw, dtype=float).reshape(3,)
        except Exception:
            continue
        if not np.isfinite(c).all():
            continue

        # r = info.get("r_m", info.get("r", 0.075))
        r = 0.150
        try:
            r = float(r)
        except Exception:
            r = 0.075
        if not np.isfinite(r) or r <= 0:
            r = 0.075

        d = float(np.linalg.norm(p - c))
        if print_bool:
            print("tid:", tid, "d:", d, "r:", r)

        if d <= r and d < best_d:
            best_d = d
            best_tid = tid

    return best_tid

def get_object_world_pos_from_vr(
    objects_vr_poses_o3d: Optional[Dict[str, dict]],
    key_lower: str,
) -> Optional[np.ndarray]:
    key_lower = (key_lower or "").strip().lower()
    if not key_lower:
        return None

    vr = objects_vr_poses_o3d if isinstance(objects_vr_poses_o3d, dict) else None
    if not vr:
        return None

    for name, info in vr.items():
        if (name or "").strip().lower() != key_lower:
            continue
        if not isinstance(info, dict):
            return None

        T = info.get("T", None)
        active = bool(info.get("active", True))
        missing = bool(info.get("missing", False))

        if missing or (not active) or (T is None):
            return None

        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4) or (not np.isfinite(T).all()):
            return None

        return T[:3, 3].copy()

    return None
