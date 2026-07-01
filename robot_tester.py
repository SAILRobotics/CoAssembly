"""robot_tester.py — Standalone GUI tester for frax OSC+CBF robot control.

No Unity / Quest required.

Features
--------
  • Dear PyGui control panel  : target sliders, mode toggle, CBF readout,
                                 gains, grasp buttons, jog buttons
  • Open3D 3D view            : robot mesh, collision spheres, workspace box,
                                 EE target marker
  • Control loop thread       : runs OSC+CBF (real) or IK (sim) at ~30 Hz
  • Real robot extras         : live joint / EE display, freedrive toggle,
                                 actual-vs-commanded overlay

Usage
-----
    python robot_tester.py                        # simulation
    python robot_tester.py --real                 # real robot (reads IP from main_setting.py)
    python robot_tester.py --real --ip 192.168.50.70
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as ScipyR

import main_setting as cfg

try:
    from pybullet_ik import IKScene as _PyBulletScene
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False
    print("[tester] pybullet_ik not found — IK unavailable.")

try:
    from robot_controller import RobotController as _RobotController
    _RC_AVAILABLE = True
except ImportError:
    _RC_AVAILABLE = False
    print("[tester] robot_controller not found.")

try:
    import dearpygui.dearpygui as dpg
    _DPG_AVAILABLE = True
except ImportError:
    _DPG_AVAILABLE = False
    print("[tester] pip install dearpygui for the control panel.")

_FILE_DIR   = Path(__file__).resolve().parent
_CALIB_DIR  = _FILE_DIR / "calibration_data" / "results"
_URDF_PATH  = _FILE_DIR / "robot_assets" / "ur10e.urdf"
_LAYOUT_DIR = cfg.SCENE_LAYOUT_DIR / "tool_layout1.json"

_WS_LO  = np.array(cfg.WORKSPACE_LO, float)
_WS_HI  = np.array(cfg.WORKSPACE_HI, float)
_WS_CTR = (_WS_LO + _WS_HI) / 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Shared state (lock-protected)
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self._lock        = threading.Lock()
        # Target
        self.target_pos   = _WS_CTR.copy()
        self.target_euler = np.array([180.0, 0.0, 0.0])  # pointing down (ZYX deg)
        # Mode
        self.mode         = "hold"   # "hold" | "cbf" | "ik"
        self.prev_mode    = "cbf"    # restored on RESUME
        self.speed_scale  = 1.0
        # Gains (read-only after init; rebuilt via signal)
        self.kp_pos       = 80.0    # scale factor: 100 = full frax gains (kp=100)
        self.kp_ori       = 50.0
        self.cbf_alpha    = 10.0
        self.qdot_max     = 1.5
        # Robot state
        self.q            = None          # np.ndarray (6,) actual joint angles
        self.q_cmd        = None          # np.ndarray (6,) last commanded joints
        self.ee_pos       = None          # np.ndarray (3,) world frame
        self.ee_err_m     = None          # float position error
        self.h_values     = None          # np.ndarray of CBF barrier values
        self.h_labels     = []            # list[str]
        self.status       = "idle"
        # Actions
        self.trigger_stop      = False
        self.trigger_grasp_id  = None   # int tool id → start grasp
        self.trigger_rebuild      = False  # update qdot_max live
        self.trigger_cbf_rebuild  = False  # rebuild CBF QP (alpha changed)
        self.trigger_freedrive = False  # toggle freedrive (real robot)
        self.freedrive_on      = False
        self.grasp_tcp_T = None  # 4×4 FK at grasp_joints; shown while grasp runs

    # Thread-safe bulk update
    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in self.__dict__.items()
                    if not k.startswith("_")}


_S = _State()

# ─────────────────────────────────────────────────────────────────────────────
# Control loop thread
# ─────────────────────────────────────────────────────────────────────────────

def _control_loop(robot: _RobotController, simulation: bool):
    """Runs at ~30 Hz; drives the robot toward _S.target_pos."""
    dt = 1.0 / 30.0

    while True:
        t0 = time.perf_counter()

        snap = _S.snapshot()
        mode = snap["mode"]

        # --- Handle action triggers ---
        if snap["trigger_stop"]:
            prev = snap["mode"] if snap["mode"] != "hold" else snap["prev_mode"]
            _S.update(trigger_stop=False, mode="hold", prev_mode=prev)
            if not simulation:
                try:
                    robot.servoStop()
                except Exception:
                    pass
            continue

        if snap["trigger_rebuild"]:
            _S.update(trigger_rebuild=False)
            frax = getattr(robot, "_frax", None)
            if frax is not None:
                frax._qdot_max = float(snap["qdot_max"])

        if snap["trigger_cbf_rebuild"]:
            _S.update(trigger_cbf_rebuild=False, status="Rebuilding CBF…", mode="hold")
            frax = getattr(robot, "_frax", None)
            if frax is not None:
                try:
                    frax.rebuild_cbf(snap["cbf_alpha"])
                    _S.update(status=f"CBF rebuilt  alpha={snap['cbf_alpha']:.1f}")
                except Exception as e:
                    _S.update(status=f"CBF rebuild failed: {e}")

        if snap["trigger_freedrive"]:
            _S.update(trigger_freedrive=False)
            if not simulation:
                ctrl = getattr(robot, "_rtde_ctrl", None)
                if ctrl is None:
                    ctrl = robot._rtde_ctrl_conn()
                try:
                    if snap["freedrive_on"]:
                        ctrl.endTeachMode()
                        _S.update(freedrive_on=False, status="Freedrive OFF")
                    else:
                        ctrl.teachMode()
                        _S.update(freedrive_on=True, status="Freedrive ON")
                except Exception as e:
                    _S.update(status=f"Freedrive error: {e}")

        if snap["trigger_grasp_id"] is not None:
            tid = snap["trigger_grasp_id"]
            _resume_mode = (snap["prev_mode"] if snap["mode"] == "hold"
                            else snap["mode"])
            _S.update(trigger_grasp_id=None, mode="hold",
                      status=f"Grasping tool {tid}…")
            try:
                import json
                with open(_LAYOUT_DIR) as f:
                    layout = json.load(f)
                tool_entry = next((t for t in layout.get("tools", [])
                                   if t.get("id") == tid), None)
                if tool_entry is None:
                    _S.update(mode=_resume_mode,
                              status=f"Tool {tid} not found in layout.")
                else:
                    gj = tool_entry.get("grasp_joints")
                    if gj is None:
                        _S.update(mode=_resume_mode,
                                  status=f"Tool {tid}: no grasp_joints recorded.")
                    else:
                        # FK: show where the TCP will be at grasp_joints
                        grasp_T = None
                        if robot._pb_scene is not None:
                            saved_q = robot._pb_scene.current_q.copy()
                            robot._pb_scene.update_robot(np.array(gj, float))
                            grasp_T = robot._pb_scene.update_tcp_bodies()
                            robot._pb_scene.update_robot(saved_q)
                        _S.update(grasp_tcp_T=grasp_T)
                        robot.execute_grasp(
                            gj,
                            on_complete=lambda ok, _m=_resume_mode: _S.update(
                                mode=_m,
                                grasp_tcp_T=None,
                                status=f"Grasp {'OK' if ok else 'FAILED'}",
                            ),
                        )
            except Exception as e:
                _S.update(mode=_resume_mode, status=f"Grasp error: {e}")

        # --- Poll current q ---
        if simulation:
            q = robot._pb_scene.current_q.copy() if robot._pb_scene else None
        else:
            q = robot.poll_q()

        if q is not None:
            robot._pb_scene and robot._pb_scene.update_robot(q)

        # --- Advance sim grasp state machine (tick) ---
        if simulation and robot._sim_phase is not None:
            robot.tick()
            _S.update(status='Grasping…' if robot._sim_phase == 'grasp' else 'Grasp done')

            if robot._pb_scene is not None:
                q_tick = robot._pb_scene.current_q.copy()
                T_tick = robot._pb_scene.update_tcp_bodies()
                if T_tick is not None:
                    ee_tick = T_tick[:3, 3]
                    _S.update(q=q_tick, ee_pos=ee_tick,
                              ee_err_m=float(np.linalg.norm(ee_tick - snap["target_pos"])))
                else:
                    _S.update(q=q_tick)

            # Rate-limit: without this sleep the 60-step runner completes in
            # milliseconds and the motion is invisible.
            elapsed = time.perf_counter() - t0
            if (rem := dt - elapsed) > 0:
                time.sleep(rem)
            continue   # skip normal motion while grasp is running

        # --- Move toward target ---
        if mode != "hold" and q is not None:
            target_pos   = snap["target_pos"]
            target_euler = snap["target_euler"]
            target_rot   = ScipyR.from_euler("ZYX", target_euler, degrees=True).as_matrix()
            target_quat  = ScipyR.from_matrix(target_rot).as_quat()

            frax = getattr(robot, "_frax", None)
            if mode == "cbf" and frax is not None:
                try:
                    # Use fixed dt=0.016 (sample_code rate) so kp=100 is numerically stable.
                    # Then scale the step by kp_scale (UI slider, default 0.2 → effective kp≈20).
                    _DT_OSC = 0.016
                    kp_scale = snap["kp_pos"] / 100.0   # slider 0–150, default 20 → scale 0.2
                    q_full   = frax.servo_step(target_pos, target_rot, q, _DT_OSC)
                    q_target = q + (q_full - q) * kp_scale
                    q_target = np.clip(q_target,
                                       np.deg2rad(cfg.JOINT_MIN_DEG),
                                       np.deg2rad(cfg.JOINT_MAX_DEG))
                    if simulation:
                        robot._pb_scene.update_robot(q_target)
                    else:
                        robot.servoJ(q_target, dt)
                    _S.update(q_cmd=q_target, status="CBF tracking")
                except Exception as e:
                    _S.update(status=f"CBF error: {e}", mode="hold")
            elif mode == "ik":
                try:
                    robot.step_hand_track(target_pos.tolist(), target_quat.tolist(), dt)
                    _S.update(status="IK tracking")
                except Exception as e:
                    _S.update(status=f"IK error: {e}", mode="hold")

        # --- Compute CBF barriers for display ---
        frax = getattr(robot, "_frax", None)
        if frax is not None and q is not None and not np.any(np.isnan(q)):
            try:
                h = frax.h_barriers(q)
                _S.update(h_values=h)
            except Exception:
                pass

        # --- Update robot state for GUI ---
        ee_pos = None
        ee_err = None
        if frax is not None and q is not None:
            try:
                ee_pos = frax.ee_world_pos(q)
                ee_err = float(np.linalg.norm(ee_pos - snap["target_pos"]))
            except Exception:
                pass
        elif robot._pb_scene is not None:
            T = robot._pb_scene.update_tcp_bodies()
            if T is not None:
                ee_pos = T[:3, 3]
                ee_err = float(np.linalg.norm(ee_pos - snap["target_pos"]))

        _S.update(q=q, ee_pos=ee_pos, ee_err_m=ee_err)

        elapsed = time.perf_counter() - t0
        rem = dt - elapsed
        if rem > 0:
            time.sleep(rem)


# ─────────────────────────────────────────────────────────────────────────────
# Open3D visualiser helpers + thread
# ─────────────────────────────────────────────────────────────────────────────

def _make_workspace_box() -> o3d.geometry.LineSet:
    lo, hi = _WS_LO, _WS_HI
    pts = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
    ])
    lines = [[0,1],[2,3],[4,5],[6,7],
             [0,2],[1,3],[4,6],[5,7],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0.4, 0.8, 1.0]] * len(lines))
    return ls


def _make_sphere_wireframe(centers, radii, n=16, color=(0.3, 1.0, 0.3)):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s  = np.cos(theta), np.sin(theta)
    z     = np.zeros(n)
    all_pts, all_lines, off = [], [], 0
    for center, r in zip(centers, radii):
        for c1, c2, c3 in [(c, s, z), (c, z, s), (z, c, s)]:
            pts = np.column_stack([c1*r, c2*r, c3*r]) + center
            all_pts.append(pts)
            all_lines += [[off+i, off+(i+1)%n] for i in range(n)]
            off += n
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    ls.lines  = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(all_lines))
    return ls


_MESH_DIR = _FILE_DIR / "robot_assets" / "meshes" / "ur10e" / "visual"

# (obj_name, rpy_xyz, t_xyz) — visual origins from ur10e.urdf, link order matches
# get_arm_link_world_poses(): base, shoulder, upper_arm, forearm, wrist1/2/3
_LINK_MESH_TABLE = [
    ("base.obj",      [0, 0,  np.pi],                   [0,      0,       0     ]),
    ("shoulder.obj",  [0, 0,  np.pi],                   [0,      0,       0     ]),
    ("upperarm.obj",  [np.pi/2, 0, -np.pi/2],           [0,      0,       0.1762]),
    ("forearm.obj",   [np.pi/2, 0, -np.pi/2],           [0,      0,       0.0393]),
    ("wrist1.obj",    [np.pi/2, 0,  0],                  [0,      0,      -0.135 ]),
    ("wrist2.obj",    [0,       0,  0],                  [0,      0,      -0.12  ]),
    ("wrist3.obj",    [np.pi/2, 0,  0],                  [0,      0,      -0.1168]),
]


def _load_urdf_link_meshes() -> list:
    """Load UR10e visual OBJ meshes.

    Returns a list of (mesh, canon_verts, canon_normals, T_link_visual) tuples.
    """
    results = []
    for obj_name, rpy, xyz in _LINK_MESH_TABLE:
        path = _MESH_DIR / obj_name
        if not path.exists():
            print(f"[tester] mesh not found: {path}")
            results.append(None)
            continue

        T_vis = np.eye(4)
        T_vis[:3, :3] = ScipyR.from_euler("xyz", rpy).as_matrix()
        T_vis[:3, 3]  = xyz

        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0:
            print(f"[tester] failed to load: {path}")
            results.append(None)
            continue
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.65, 0.65, 0.72])

        results.append((mesh,
                        np.asarray(mesh.vertices).copy(),
                        np.asarray(mesh.vertex_normals).copy(),
                        T_vis))
    return results


def _apply_link_transform(item, T_world_link: np.ndarray):
    """Write new world-frame vertex positions/normals into the mesh in-place."""
    mesh, canon_v, canon_n, T_vis = item
    T = T_world_link @ T_vis
    R, t = T[:3, :3], T[:3, 3]
    mesh.vertices      = o3d.utility.Vector3dVector(canon_v @ R.T + t)
    mesh.vertex_normals = o3d.utility.Vector3dVector(canon_n @ R.T)


def _vis_thread(robot: _RobotController):
    vis = o3d.visualization.Visualizer()
    vis.create_window("Robot Tester — 3D View", width=900, height=680)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.13])
    opt.line_width = 2.0
    opt.mesh_show_back_face = True

    # ── Static geometry ───────────────────────────────────────────────────────
    ws_box = _make_workspace_box()
    vis.add_geometry(ws_box)

    floor = o3d.geometry.TriangleMesh.create_box(3.0, 3.0, 0.005)
    floor.translate([-1.5, -1.5, -0.005])
    floor.paint_uniform_color([0.22, 0.22, 0.22])
    floor.compute_vertex_normals()
    vis.add_geometry(floor)

    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(world_axes)

    # ── URDF link meshes ──────────────────────────────────────────────────────
    link_data = _load_urdf_link_meshes()   # list of (mesh, canon_v, canon_n, T_vis) | None
    for item in link_data:
        if item is not None:
            vis.add_geometry(item[0])

    # Skeleton lines (fallback when no mesh / always visible for joints)
    link_lines = o3d.geometry.LineSet()
    vis.add_geometry(link_lines)

    # ── EE target: XYZ axes frame (moves with target slider) ─────────────────
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    target_frame.translate(_WS_CTR)
    vis.add_geometry(target_frame)
    _last_target     = _WS_CTR.copy()
    _last_target_rot = np.eye(3)

    # ── Grasp TCP frame (FK from grasp_joints; visible while grasp runs) ─────
    _GRASP_FRAME_HIDDEN = np.array([0.0, 0.0, 100.0])
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    grasp_frame.translate(_GRASP_FRAME_HIDDEN)
    vis.add_geometry(grasp_frame)
    _last_grasp_T: "np.ndarray | None" = None

    # ── Collision sphere wireframe ────────────────────────────────────────────
    col_ls = o3d.geometry.LineSet()
    vis.add_geometry(col_ls)

    # ── Camera: point at workspace centre ────────────────────────────────────
    vis.poll_events()
    vis.update_renderer()
    vc = vis.get_view_control()
    vc.set_lookat(_WS_CTR.tolist())
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_front([1.2, -1.0, 0.8])
    vc.set_zoom(0.55)

    while vis.poll_events():
        snap = _S.snapshot()
        q    = snap["q"]
        frax = getattr(robot, "_frax", None)

        # ── Update grasp TCP frame ────────────────────────────────────────────
        new_grasp_T = snap["grasp_tcp_T"]
        _grasp_changed = (
            (new_grasp_T is None) != (_last_grasp_T is None)
            or (new_grasp_T is not None and _last_grasp_T is not None
                and not np.allclose(new_grasp_T, _last_grasp_T))
        )
        if _grasp_changed:
            if new_grasp_T is not None:
                gf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
                gf.transform(new_grasp_T)
            else:
                gf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
                gf.translate(_GRASP_FRAME_HIDDEN)
            grasp_frame.vertices       = gf.vertices
            grasp_frame.triangles      = gf.triangles
            grasp_frame.vertex_normals = gf.vertex_normals
            grasp_frame.vertex_colors  = gf.vertex_colors
            _last_grasp_T = new_grasp_T.copy() if new_grasp_T is not None else None
            vis.update_geometry(grasp_frame)

        # ── Move target frame ─────────────────────────────────────────────────
        new_tgt = snap["target_pos"]
        new_rot = ScipyR.from_euler("ZYX", snap["target_euler"], degrees=True).as_matrix()
        if not np.allclose(new_tgt, _last_target) or not np.allclose(new_rot, _last_target_rot):
            # Rebuild the frame mesh at the new pose
            T_tgt = np.eye(4)
            T_tgt[:3, :3] = new_rot
            T_tgt[:3, 3]  = new_tgt
            new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
            new_frame.transform(T_tgt)
            target_frame.vertices       = new_frame.vertices
            target_frame.triangles      = new_frame.triangles
            target_frame.vertex_normals = new_frame.vertex_normals
            target_frame.vertex_colors  = new_frame.vertex_colors
            _last_target     = new_tgt.copy()
            _last_target_rot = new_rot.copy()
            vis.update_geometry(target_frame)

        # ── Update URDF link meshes from PyBullet FK ──────────────────────────
        if robot._pb_scene is not None:
            try:
                poses = robot._pb_scene.get_arm_link_world_poses()  # 7 × 4×4
                for i, item in enumerate(link_data):
                    if item is not None and i < len(poses):
                        _apply_link_transform(item, poses[i])
                        vis.update_geometry(item[0])

                # Skeleton lines: base mount → link origins
                base_pt  = robot._pb_scene.T_world_base[:3, 3]
                skel_pts = np.vstack([base_pt.reshape(1, 3)]
                                     + [p[:3, 3].reshape(1, 3) for p in poses])
                skel_lns = [[i, i+1] for i in range(len(skel_pts)-1)]
                link_lines.points = o3d.utility.Vector3dVector(skel_pts)
                link_lines.lines  = o3d.utility.Vector2iVector(skel_lns)
                link_lines.colors = o3d.utility.Vector3dVector(
                    [[1.0, 0.85, 0.3]] * len(skel_lns))
                vis.update_geometry(link_lines)
            except Exception:
                pass

        # ── Collision spheres (frax) ──────────────────────────────────────────
        if frax is not None and q is not None and not np.any(np.isnan(q)):
            try:
                sp, sr = frax.link_spheres_world(q)
                if sp is not None and len(sp) and not np.any(np.isnan(sp)):
                    new_col = _make_sphere_wireframe(sp, sr)
                    col_ls.points = new_col.points
                    col_ls.lines  = new_col.lines
                    col_ls.colors = new_col.colors
                else:
                    col_ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    col_ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), int))
                vis.update_geometry(col_ls)
            except Exception:
                pass

        vis.update_renderer()
        time.sleep(0.033)

    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────────────
# Dear PyGui panel (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

_H_SECTION_LABELS = [
    "Joint low", "Joint high", "Floor", "Workspace lo", "Workspace hi",
    "Obs/self-col",
]


def _color_for_h(h: float):
    if h > 0.05:
        return (80, 220, 80, 255)
    if h > 0.0:
        return (220, 200, 50, 255)
    return (255, 60, 60, 255)


def _dpg_thread(robot: _RobotController, simulation: bool, tool_ids: list[int]):
    dpg.create_context()

    with dpg.font_registry():
        try:
            default_font = dpg.add_font(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            default_font = None

    with dpg.window(label="Robot Tester", tag="main_win",
                    width=420, height=900, no_close=True):
        if default_font:
            dpg.bind_font(default_font)

        # ── Mode ──────────────────────────────────────────────────────────────
        dpg.add_text("Control Mode", color=(200, 200, 255))
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_radio_button(
                ["hold", "cbf", "ik"],
                tag="mode_radio", default_value="hold",
                callback=lambda s, v: _S.update(mode=v))
        dpg.add_button(label="  STOP  ", tag="stop_btn",
                       callback=_on_stop_resume,
                       width=120)
        dpg.bind_item_theme("stop_btn", _red_theme())
        dpg.add_spacer(height=6)

        # ── EE Target position ────────────────────────────────────────────────
        dpg.add_text("EE Target Position (world m)", color=(200, 200, 255))
        dpg.add_separator()
        lo, hi = _WS_LO, _WS_HI
        ctr = _WS_CTR
        dpg.add_slider_float(label="X", tag="tx", default_value=float(ctr[0]),
                             min_value=float(lo[0]-0.1), max_value=float(hi[0]+0.1),
                             callback=_on_target_slider, width=280)
        dpg.add_slider_float(label="Y", tag="ty", default_value=float(ctr[1]),
                             min_value=float(lo[1]-0.1), max_value=float(hi[1]+0.1),
                             callback=_on_target_slider, width=280)
        dpg.add_slider_float(label="Z", tag="tz", default_value=float(ctr[2]),
                             min_value=float(lo[2]), max_value=float(hi[2]+0.1),
                             callback=_on_target_slider, width=280)
        dpg.add_button(label="Snap to EE", width=120,
                       callback=_snap_to_ee)
        dpg.add_spacer(height=4)

        # ── EE Target orientation ─────────────────────────────────────────────
        dpg.add_text("EE Target Orientation ZYX (deg)", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_slider_float(label="Z rot", tag="eZ", default_value=180.0,
                             min_value=-180.0, max_value=180.0,
                             callback=_on_euler_slider, width=280)
        dpg.add_slider_float(label="Y rot", tag="eY", default_value=180.0,
                             min_value=-180.0, max_value=180.0,
                             callback=_on_euler_slider, width=280)
        dpg.add_slider_float(label="X rot", tag="eX", default_value=0.0,
                             min_value=-180.0, max_value=180.0,
                             callback=_on_euler_slider, width=280)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Point down",
                           callback=lambda: _set_euler(180, 180, 0))
            dpg.add_button(label="Sideways +X",
                           callback=lambda: _set_euler(180, -90, 0))
            dpg.add_button(label="Sideways +Y",
                           callback=lambda: _set_euler(180, 0, 90))
        dpg.add_spacer(height=4)

        # ── Jog buttons ───────────────────────────────────────────────────────
        dpg.add_text("Jog (5 mm steps)", color=(200, 200, 255))
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="-X", width=45, callback=lambda: _jog(-0.005, 0, 0))
            dpg.add_button(label="+X", width=45, callback=lambda: _jog(+0.005, 0, 0))
            dpg.add_spacer(width=10)
            dpg.add_button(label="-Y", width=45, callback=lambda: _jog(0, -0.005, 0))
            dpg.add_button(label="+Y", width=45, callback=lambda: _jog(0, +0.005, 0))
            dpg.add_spacer(width=10)
            dpg.add_button(label="-Z", width=45, callback=lambda: _jog(0, 0, -0.005))
            dpg.add_button(label="+Z", width=45, callback=lambda: _jog(0, 0, +0.005))
        dpg.add_spacer(height=4)

        # ── Speed scale ───────────────────────────────────────────────────────
        dpg.add_text("Speed / Gains", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_slider_float(label="qdot_max (rad/s)", tag="qdot_max_sl",
                             default_value=1.5, min_value=0.1, max_value=5.0,
                             callback=lambda s, v: _S.update(qdot_max=v,
                                                             trigger_rebuild=True),
                             width=280)
        dpg.add_slider_float(label="kp scale (100=full)", tag="kpp_sl",
                             default_value=80.0, min_value=1.0, max_value=150.0,
                             callback=lambda s, v: _S.update(kp_pos=v),
                             width=280)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="cbf_alpha", tag="alpha_sl",
                                 default_value=10.0, min_value=0.5, max_value=50.0,
                                 callback=lambda s, v: _S.update(cbf_alpha=v),
                                 width=220)
            dpg.add_button(label="Apply", width=55,
                           callback=lambda: _S.update(trigger_cbf_rebuild=True))
        dpg.add_spacer(height=4)

        # ── Grasp ─────────────────────────────────────────────────────────────
        dpg.add_text("Grasp Tool", color=(200, 200, 255))
        dpg.add_separator()
        if tool_ids:
            dpg.add_combo([str(t) for t in tool_ids], tag="grasp_combo",
                          default_value=str(tool_ids[0]), width=120)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Execute grasp",
                               callback=lambda: _S.update(
                                   trigger_grasp_id=int(dpg.get_value("grasp_combo"))))
                if not simulation:
                    dpg.add_button(
                        label="Freedrive toggle",
                        callback=lambda: _S.update(trigger_freedrive=True))
        else:
            dpg.add_text("(no tool_layout2.json found)")
        dpg.add_spacer(height=4)

        # ── Robot state display ───────────────────────────────────────────────
        dpg.add_text("Robot State", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_text("q: —", tag="q_disp")
        dpg.add_text("EE: —", tag="ee_disp")
        dpg.add_text("Err: —", tag="err_disp")
        dpg.add_text("Status: idle", tag="status_disp")
        dpg.add_spacer(height=4)

        # ── CBF barriers ──────────────────────────────────────────────────────
        dpg.add_text("CBF Barriers  (green > 0.05 | yellow near | red < 0)",
                     color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_text("—", tag="cbf_disp")

    dpg.create_viewport(title="Robot Tester", width=440, height=940)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        _dpg_update_state()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def _on_stop_resume():
    snap = _S.snapshot()
    if snap["mode"] == "hold":
        # Resume to whatever mode was active before stopping
        _S.update(mode=snap["prev_mode"])
    else:
        # Stop: save current mode so RESUME can restore it
        _S.update(trigger_stop=True, prev_mode=snap["mode"])


def _red_theme():
    if dpg.does_item_exist("red_theme"):
        return "red_theme"
    with dpg.theme(tag="red_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 60, 60))
    return "red_theme"


def _on_target_slider(sender, value):
    with _S._lock:
        axis = {"tx": 0, "ty": 1, "tz": 2}[sender]
        _S.target_pos[axis] = value


def _on_euler_slider(sender, value):
    with _S._lock:
        axis = {"eZ": 0, "eY": 1, "eX": 2}[sender]
        _S.target_euler[axis] = value


def _jog(dx, dy, dz):
    with _S._lock:
        _S.target_pos += np.array([dx, dy, dz])
    _sync_target_sliders()


def _set_euler(z, y, x):
    _S.update(target_euler=np.array([z, y, x], float))
    if _DPG_AVAILABLE:
        dpg.set_value("eZ", z)
        dpg.set_value("eY", y)
        dpg.set_value("eX", x)


def _snap_to_ee():
    snap = _S.snapshot()
    if snap["ee_pos"] is not None:
        _S.update(target_pos=snap["ee_pos"].copy())
        _sync_target_sliders()


def _sync_target_sliders():
    if not _DPG_AVAILABLE:
        return
    with _S._lock:
        pos = _S.target_pos.copy()
    dpg.set_value("tx", float(np.clip(pos[0], _WS_LO[0]-0.1, _WS_HI[0]+0.1)))
    dpg.set_value("ty", float(np.clip(pos[1], _WS_LO[1]-0.1, _WS_HI[1]+0.1)))
    dpg.set_value("tz", float(np.clip(pos[2], _WS_LO[2], _WS_HI[2]+0.1)))


def _dpg_update_state():
    snap = _S.snapshot()

    # Mode radio — sync to _S.mode so programmatic changes are visible
    if dpg.does_item_exist("mode_radio"):
        if dpg.get_value("mode_radio") != snap["mode"]:
            dpg.set_value("mode_radio", snap["mode"])

    # Target position sliders — sync so programmatic target changes are visible
    pos = snap["target_pos"]
    if dpg.does_item_exist("tx"):
        dpg.set_value("tx", float(np.clip(pos[0], _WS_LO[0]-0.1, _WS_HI[0]+0.1)))
        dpg.set_value("ty", float(np.clip(pos[1], _WS_LO[1]-0.1, _WS_HI[1]+0.1)))
        dpg.set_value("tz", float(np.clip(pos[2], _WS_LO[2], _WS_HI[2]+0.1)))

    # Target euler sliders
    euler = snap["target_euler"]
    if dpg.does_item_exist("eZ"):
        dpg.set_value("eZ", float(euler[0]))
        dpg.set_value("eY", float(euler[1]))
        dpg.set_value("eX", float(euler[2]))

    # STOP / RESUME button label
    if dpg.does_item_exist("stop_btn"):
        label = " RESUME " if snap["mode"] == "hold" else "  STOP  "
        if dpg.get_item_label("stop_btn") != label:
            dpg.set_item_label("stop_btn", label)

    # Joint angles
    if snap["q"] is not None:
        q_deg = np.rad2deg(snap["q"])
        dpg.set_value("q_disp", "q: " + "  ".join(f"{v:+.1f}°" for v in q_deg))
    else:
        dpg.set_value("q_disp", "q: —")

    # EE position
    if snap["ee_pos"] is not None:
        p = snap["ee_pos"]
        dpg.set_value("ee_disp", f"EE: [{p[0]:+.3f}  {p[1]:+.3f}  {p[2]:+.3f}]")
    else:
        dpg.set_value("ee_disp", "EE: —")

    # Position error
    if snap["ee_err_m"] is not None:
        err = snap["ee_err_m"]
        col = (80,220,80,255) if err < 0.02 else (220,200,50,255) if err < 0.05 else (255,255,255,255)
        dpg.set_value("err_disp", f"Err: {err*1000:.1f} mm")
        dpg.configure_item("err_disp", color=col)

    # Status
    dpg.set_value("status_disp", f"Status: {snap['status']}")

    # CBF barriers
    h = snap["h_values"]
    if h is not None and len(h):
        lines = []
        n_joint = 12       # 6 low + 6 high
        n_floor = 22       # 22 link spheres
        n_ws    = 6        # 3 lo + 3 hi
        groups  = [
            ("Joint lo",   h[:6]),
            ("Joint hi",   h[6:12]),
            ("Floor",      h[12:12+n_floor] if len(h) >= 12+n_floor else h[12:]),
            ("Workspace",  h[12+n_floor:12+n_floor+n_ws] if len(h) >= 12+n_floor+n_ws else np.array([])),
            ("Self-col",   h[12+n_floor+n_ws:] if len(h) > 12+n_floor+n_ws else np.array([])),
        ]
        for label, vals in groups:
            if len(vals) == 0:
                continue
            mn = float(vals.min())
            col = _color_for_h(mn)
            lines.append(f"{label:<12}  min={mn:+.4f}")
        dpg.configure_item("cbf_disp", color=(255,255,255,255))
        dpg.set_value("cbf_disp", "\n".join(lines) if lines else "—")
    else:
        dpg.set_value("cbf_disp", "—")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot tester GUI")
    parser.add_argument("--real", action="store_true",
                        help="Connect to real robot (default: simulation)")
    parser.add_argument("--ip", default=cfg.ROBOT_IP,
                        help=f"Robot IP (default: {cfg.ROBOT_IP})")
    args = parser.parse_args()

    simulation = not args.real
    robot_ip   = None if simulation else args.ip

    print(f"[tester] Mode: {'simulation' if simulation else 'REAL ROBOT @ ' + robot_ip}")

    # ── PyBullet scene ────────────────────────────────────────────────────────
    if not _PB_AVAILABLE:
        print("[tester] PyBullet not available — cannot run.")
        return

    if _CALIB_DIR.exists():
        pb = _PyBulletScene.from_calibration(_CALIB_DIR)
    else:
        T = np.eye(4)
        T[:3, 3] = [-0.4, -0.8, 0.4]
        pb = _PyBulletScene(T_world_base=T)
        print("[tester] Calibration not found — using default base pose.")
    pb.build()
    pb.set_joint_limits(cfg.JOINT_MIN_DEG, cfg.JOINT_MAX_DEG, degrees=True)

    q0 = np.deg2rad([-122.06, -41.89, 90.84, 40.95, 89.01, -150.54])
    pb.update_robot(q0)

    # ── RobotController ───────────────────────────────────────────────────────
    if not _RC_AVAILABLE:
        print("[tester] robot_controller not available — cannot run.")
        pb.disconnect()
        return

    robot = _RobotController(
        unity_ip      = "127.0.0.1",        # dummy — no Unity
        pb_scene      = pb,
        T_world_base  = pb.T_world_base,
        robot_ip      = robot_ip,
        urdf_path     = str(_URDF_PATH) if _URDF_PATH.exists() else None,
        frax_q_min    = list(np.deg2rad(cfg.JOINT_MIN_DEG)),
        frax_q_max    = list(np.deg2rad(cfg.JOINT_MAX_DEG)),
        frax_ws_lo    = cfg.WORKSPACE_LO,
        frax_ws_hi    = cfg.WORKSPACE_HI,
    )

    frax = getattr(robot, "_frax", None)
    if frax is not None:
        print("[tester] frax OSC+CBF ready.")
    elif not simulation:
        print("[tester] WARNING: frax not available — IK only mode.")

    # ── Load tool IDs ─────────────────────────────────────────────────────────
    tool_ids = []
    if _LAYOUT_DIR.exists():
        import json
        try:
            with open(_LAYOUT_DIR) as f:
                layout = json.load(f)
            tool_ids = [t["id"] for t in layout.get("tools", []) if "id" in t]
        except Exception:
            pass

    # ── Snap initial target to current EE ────────────────────────────────────
    if frax is not None:
        try:
            _S.update(target_pos=frax.ee_world_pos(q0).copy())
        except Exception:
            pass

    # ── Start threads ─────────────────────────────────────────────────────────
    ctrl_t = threading.Thread(
        target=_control_loop, args=(robot, simulation), daemon=True)
    ctrl_t.start()
    print("[tester] Control loop started.")

    if _DPG_AVAILABLE:
        dpg_t = threading.Thread(
            target=_dpg_thread, args=(robot, simulation, tool_ids), daemon=True)
        dpg_t.start()
        print("[tester] DPG panel started.")
    else:
        print("[tester] dearpygui not installed — running without control panel.")

    # Open3D runs in main thread
    _vis_thread(robot)

    print("[tester] Exiting.")
    if not simulation:
        try:
            robot.servoStop()
        except Exception:
            pass
    pb.disconnect()


if __name__ == "__main__":
    main()
