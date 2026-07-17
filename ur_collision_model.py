"""UR10e collision model: link spheres, dense self-collision, and base self-collision.

All positions are in the corresponding URDF link frame (not link COM frame).
"""

# ------------------------- Link collision model -------------------------

link_1_pos = (
    (0, 0, -0.15),
    (0, 0, 0),
)
link_1_radii = (
    0.10,
    0.10,
)

link_2_pos = (
    (0, 0, 0.080),
    (0, 0, 0.145),
    (0, 0, 0.210),

    (-0.05, 0, 0.175),
    (-0.15, 0, 0.175),
    (-0.25, 0, 0.175),
    (-0.35, 0, 0.175),
    (-0.45, 0, 0.175),
    (-0.55, 0, 0.175),
    (-0.60, 0, 0.175),
)
link_2_radii = (
    0.10,
    0.10,
    0.10,

    0.06,
    0.06,
    0.06,
    0.06,
    0.06,
    0.06,
    0.08
)

link_3_pos = (
    (0, 0, 0.04),
    (-0.05, 0, 0.04),
    (-0.15, 0, 0.04),
    (-0.25, 0, 0.04),
    (-0.35, 0, 0.04),
    (-0.45, 0, 0.04),
    (-0.55, 0, 0.04),
)
link_3_radii = (
    0.08,
    0.06,
    0.06,
    0.06,
    0.06,
    0.06,
    0.08,
)

link_4_pos = (
    (0, 0, 0),
)
link_4_radii = (
    0.080,
)

link_5_pos = (
    (0, 0, 0),
)
link_5_radii = (
    0.080,
)

link_6_pos = (
    (0, 0, -0.025),
)
link_6_radii = (
    0.05,
)

# Gripper collision spheres (in link_6 frame, z points away from flange).
# Appended to link_6 when with_gripper=True.
# Tuned for Robotiq 85: palm body + mid-finger + fingertip region.
gripper_positions = (
    # (0, -0.080, 0.00),   # palm / top of body   r=0.055
    (0, 0, 0.05),   # palm / top of body   r=0.055
    (0, 0, 0.11),   # mid body              r=0.050
    (0, 0.05, 0.15),   # mid body              r=0.050
    (0, 0.00, 0.15),   # mid body              r=0.050
    (0, -0.05, 0.15),   # mid body              r=0.050
    (0.05, 0, 0.095),   # finger region         r=0.040
    (-0.05, 0, 0.095),   # finger region         r=0.040
)
# gripper_radii = (0.040, 0.055, 0.06, 0.040, 0.040, 0.040, 0.040, 0.040)

gripper_radii = (0.045, 0.45, 0.040, 0.040, 0.040, 0.040, 0.040)

# (Still here if you need them elsewhere)
positions = {
    "link_1": link_1_pos,
    "link_2": link_2_pos,
    "link_3": link_3_pos,
    "link_4": link_4_pos,
    "link_5": link_5_pos,
    "link_6": link_6_pos,
}
radii = {
    "link_1": link_1_radii,
    "link_2": link_2_radii,
    "link_3": link_3_radii,
    "link_4": link_4_radii,
    "link_5": link_5_radii,
    "link_6": link_6_radii,
}

positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
)
radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
)

ur_collision_data = {"positions": positions_list, "radii": radii_list}


def make_collision_data(with_gripper: bool = False) -> dict:
    """Return ur_collision_data with or without the gripper spheres on link_6."""
    if not with_gripper:
        return ur_collision_data
    positions_with_gripper = positions_list[:5] + (link_6_pos + gripper_positions,)
    radii_with_gripper     = radii_list[:5]     + (link_6_radii + gripper_radii,)
    return {"positions": positions_with_gripper, "radii": radii_with_gripper}


ur_collision_data_with_gripper = make_collision_data(with_gripper=True)

# ------------------------- Dense self-collision model -------------------------

# Reuse all link collision spheres for self-collision
positions_list_sc = positions_list
radii_list_sc = radii_list

# Compute flattened index ranges per link so pairs are always correct,
# even if you change the number of spheres later.
link_sphere_counts = [
    len(link_1_pos),
    len(link_2_pos),
    len(link_3_pos),
    len(link_4_pos),
    len(link_5_pos),
    len(link_6_pos),
]

link_start_indices = []
_start = 0
for c in link_sphere_counts:
    link_start_indices.append(_start)
    _start += c

# Ranges of global sphere indices for each link
link1_range = range(link_start_indices[0], link_start_indices[0] + link_sphere_counts[0])  # link_1
link2_range = range(link_start_indices[1], link_start_indices[1] + link_sphere_counts[1])  # link_2
link3_range = range(link_start_indices[2], link_start_indices[2] + link_sphere_counts[2])  # link_3
link4_range = range(link_start_indices[3], link_start_indices[3] + link_sphere_counts[3])  # link_4
link5_range = range(link_start_indices[4], link_start_indices[4] + link_sphere_counts[4])  # link_5
link6_range = range(link_start_indices[5], link_start_indices[5] + link_sphere_counts[5])  # link_6

# For reference (with current numbers):
# link_1: 1 sphere  -> indices [0]
# link_2: 10 spheres -> indices [1..10]
# link_3: 7 spheres  -> indices [11..17]
# link_4: 1 sphere   -> index  [18]
# link_5: 1 sphere   -> index  [19]
# link_6: 1 sphere   -> index  [20]
# Total = 21 spheres

pairs_sc_list = []


def _add_link_pair(range_a, range_b):
    """Add all pairwise combinations between two link index ranges."""
    for i in range_a:
        for j in range_b:
            pairs_sc_list.append((i, j))


# Choose link-link combinations that are likely to collide.
# - base column (link_1) vs more distal links: (3,4,5,6)
_add_link_pair(link1_range, list(link3_range)[-2:]) #
# # - shoulder link (link_2) vs forearm & flange: (4,5,6)

_add_link_pair(link2_range[:6], list(link4_range) + list(link5_range) + list(link6_range)) #
# # - mid horizontal link (link_3) vs forearm & flange


# # - elbow (link_4) vs flange (link_6)
# _add_link_pair(link4_range, link6_range)

pairs_sc = tuple(pairs_sc_list)

# Self-collision pairs when gripper is attached (gripper spheres appended to link_6).
# With gripper the flat indices are:
#   link_1: 2  → 0-1   link_2: 10 → 2-11   link_3: 7 → 12-18
#   link_4: 1  → 19    link_5:  1 → 20      link_6: 1 → 21
#   gripper: 8 → 22-29   (total 30 spheres)
_gripper_wg_start = link_start_indices[5] + link_sphere_counts[5]   # = 22
_gripper_wg_range = range(_gripper_wg_start,
                          _gripper_wg_start + len(gripper_positions))  # 22-29
pairs_sc_with_gripper = pairs_sc + tuple(
    (gi, lj)
    for gi in _gripper_wg_range
    for lj in list(link3_range) + list(link4_range)
)

ur_self_collision_data = {
    "positions": positions_list_sc,
    "radii": radii_list_sc,
    "pairs": pairs_sc,
}

# ------------------------- Base self-collision model -------------------------

# Approximate the base as a sphere just below link_1’s origin.
# You can tweak these based on your URDF base geometry.
base_position = (0.0, 0.0, -0.00)  # in base (link_0) frame
base_radius = 0.14                 # a bit larger than link_1's 0.10

# Let the base avoid collisions with all spheres on link_5 and link_6
base_sc_idxs = tuple(list(link3_range)[-2:] + list(link4_range) + list(link5_range) + list(link6_range))

ur_base_self_collision_data = {
    "position": base_position,
    "radius": base_radius,
    "indices": base_sc_idxs,
}


# ─────────────────────────────────────────────────────────────────────────────
# Interactive visualiser
# ─────────────────────────────────────────────────────────────────────────────

def visualize(q=None, T_world_base=None, with_gripper: bool = False) -> None:
    """Open an interactive Open3D + DearPyGui window.

    Shows the UR10e URDF meshes and the collision spheres defined in this file.
    Six joint-angle sliders let you explore any configuration live.

    Parameters
    ----------
    q             : initial joint angles in radians (6,). Default = home pose.
    T_world_base  : 4×4 robot base transform. Default = identity.
    with_gripper  : include Robotiq 85 gripper spheres on link_6.
    """
    import threading, time
    import numpy as np
    import open3d as o3d
    from pathlib import Path
    from scipy.spatial.transform import Rotation as ScipyR

    if q is None:
        q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    q = np.asarray(q, float).copy()

    if T_world_base is None:
        T_world_base = np.eye(4)
    T_world_base = np.asarray(T_world_base, float)

    # ── Shared state ─────────────────────────────────────────────────────────
    _lock           = threading.Lock()
    _q              = q.copy()
    _with_gripper   = [with_gripper]
    _highlight_links = [set()]          # set of link names to shade distinctly
    _dirty          = [True]            # signal vis thread to redraw

    def get_q():
        with _lock:
            return _q.copy(), _with_gripper[0], frozenset(_highlight_links[0])

    def set_q(new_q, new_grip):
        with _lock:
            _q[:] = new_q
            _with_gripper[0] = new_grip
            _dirty[0] = True

    def set_highlights(names: set):
        with _lock:
            _highlight_links[0] = names
            _dirty[0] = True

    # ── PyBullet IK scene for FK ──────────────────────────────────────────────
    from pybullet_ik import IKScene
    scene = IKScene(T_world_base=T_world_base)
    scene.build()
    scene.update_robot(q)

    # ── URDF mesh data ────────────────────────────────────────────────────────
    _HERE     = Path(__file__).resolve().parent
    _MESH_DIR = _HERE / "robot_assets" / "meshes" / "ur10e" / "visual"
    # (obj_name, rpy, xyz) — visual origins from ur10e.urdf
    # poses[0]=base, [1]=shoulder, [2]=upperarm, [3]=forearm, [4..6]=wrists
    _MESH_TABLE = [
        ("base.obj",     [0, 0,  np.pi],           [0, 0,  0      ]),
        ("shoulder.obj", [0, 0,  np.pi],           [0, 0,  0      ]),
        ("upperarm.obj", [np.pi/2, 0, -np.pi/2],  [0, 0,  0.1762 ]),
        ("forearm.obj",  [np.pi/2, 0, -np.pi/2],  [0, 0,  0.0393 ]),
        ("wrist1.obj",   [np.pi/2, 0,  0],         [0, 0, -0.135  ]),
        ("wrist2.obj",   [0,       0,  0],         [0, 0, -0.12   ]),
        ("wrist3.obj",   [np.pi/2, 0,  0],         [0, 0, -0.1168 ]),
    ]

    def _load_meshes():
        """Load OBJ files once; return (canon_verts, canon_normals, T_vis) per link."""
        items = []
        for obj_name, rpy, xyz in _MESH_TABLE:
            path = _MESH_DIR / obj_name
            T_vis = np.eye(4)
            T_vis[:3, :3] = ScipyR.from_euler("xyz", rpy).as_matrix()
            T_vis[:3, 3]  = xyz
            if not path.exists():
                items.append(None)
                continue
            m = o3d.io.read_triangle_mesh(str(path))
            if len(m.vertices) == 0:
                items.append(None)
                continue
            m.compute_vertex_normals()
            m.paint_uniform_color([0.65, 0.65, 0.72])
            items.append((m,
                          np.asarray(m.vertices).copy(),
                          np.asarray(m.vertex_normals).copy(),
                          T_vis))
        return items

    def _apply_transform(item, T_world_link):
        m, cv, cn, T_vis = item
        T = T_world_link @ T_vis
        R, t = T[:3, :3], T[:3, 3]
        m.vertices      = o3d.utility.Vector3dVector(cv @ R.T + t)
        m.vertex_normals = o3d.utility.Vector3dVector(cn @ R.T)

    _LINK_NAMES = ["link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
    _HIGHLIGHT_COLORS = {
        "link_1":  [1.0, 0.3, 0.3],
        "link_2":  [1.0, 0.6, 0.1],
        "link_3":  [1.0, 1.0, 0.1],
        "link_4":  [0.1, 1.0, 0.4],
        "link_5":  [0.1, 0.7, 1.0],
        "link_6":  [0.5, 0.3, 1.0],
        "gripper": [1.0, 0.4, 0.8],
    }
    _COL_DEFAULT = [0.2, 1.0, 0.4]
    _COL_DIM     = [0.2, 0.2, 0.2]

    def _sphere_wireframe(spheres_tagged, hl_set, n=16):
        """spheres_tagged: list of (center, radius, link_name)."""
        has_hl = bool(hl_set)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c, s  = np.cos(theta), np.sin(theta)
        z     = np.zeros(n)
        pts, lns, cols, off = [], [], [], 0
        for (ctr, r, link_name) in spheres_tagged:
            if has_hl:
                col = (_HIGHLIGHT_COLORS.get(link_name, _COL_DEFAULT)
                       if link_name in hl_set else _COL_DIM)
            else:
                col = _COL_DEFAULT
            for c1, c2, c3 in [(c, s, z), (c, z, s), (z, c, s)]:
                pts.append(np.column_stack([c1*r, c2*r, c3*r]) + ctr)
                lns  += [[off+i, off+(i+1)%n] for i in range(n)]
                cols += [col] * n
                off  += n
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.vstack(pts))
        ls.lines  = o3d.utility.Vector2iVector(lns)
        ls.colors = o3d.utility.Vector3dVector(cols)
        return ls

    def _compute_sphere_world(poses, grip):
        """Return list of (world_pos, radius, link_name) for all collision spheres."""
        result = []
        for link_i, link_name in enumerate(_LINK_NAMES):
            T = poses[link_i + 1]
            for lp, r in zip(positions_list[link_i], radii_list[link_i]):
                result.append((T[:3, :3] @ np.array(lp) + T[:3, 3], r, link_name))
        if grip:
            T = poses[6]   # wrist3 frame = link_6 pose
            for lp, r in zip(gripper_positions, gripper_radii):
                result.append((T[:3, :3] @ np.array(lp) + T[:3, 3], r, "gripper"))
        return result

    # ── DearPyGui thread ──────────────────────────────────────────────────────
    def _dpg_thread():
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            print("[ur_collision_model] pip install dearpygui for sliders.")
            return

        _ALL_HL_NAMES = ["link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "gripper"]

        def _on_joint_change():
            set_q(np.deg2rad([dpg.get_value(f"j{k}") for k in range(6)]),
                  dpg.get_value("grip_cb"))

        def _on_hl_change():
            set_highlights({n for n in _ALL_HL_NAMES if dpg.get_value(f"hl_{n}")})

        dpg.create_context()
        with dpg.window(label="Collision Model — Joints", tag="win",
                        width=340, height=480, no_close=True):
            dpg.add_text("Joint angles (degrees)", color=(200, 200, 255))
            dpg.add_separator()
            jnames = ["J1 (shoulder pan)", "J2 (shoulder lift)", "J3 (elbow)",
                      "J4 (wrist 1)", "J5 (wrist 2)", "J6 (wrist 3)"]
            for i, jname in enumerate(jnames):
                dpg.add_slider_float(
                    label=jname, tag=f"j{i}",
                    default_value=float(np.rad2deg(q[i])),
                    min_value=-360.0, max_value=360.0,
                    callback=lambda: _on_joint_change(),
                    width=280)
            dpg.add_checkbox(label="With gripper", tag="grip_cb",
                             default_value=with_gripper,
                             callback=lambda: _on_joint_change())
            dpg.add_separator()
            dpg.add_text("Highlight links", color=(255, 220, 100))
            for hl_name in _ALL_HL_NAMES:
                dpg.add_checkbox(label=hl_name, tag=f"hl_{hl_name}",
                                 default_value=False,
                                 callback=lambda: _on_hl_change())
        dpg.create_viewport(title="Joints", width=360, height=500)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    dpg_t = threading.Thread(target=_dpg_thread, daemon=True)
    dpg_t.start()

    # ── Open3D main thread ────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("UR10e Collision Model", width=1000, height=700)
    opt = vis.get_render_option()
    opt.background_color  = np.array([0.10, 0.10, 0.13])
    opt.mesh_show_back_face = True
    opt.line_width = 2.0

    floor = o3d.geometry.TriangleMesh.create_box(3, 3, 0.005)
    floor.translate([-1.5, -1.5, -0.005])
    floor.paint_uniform_color([0.20, 0.20, 0.20])
    floor.compute_vertex_normals()
    vis.add_geometry(floor)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    mesh_items = _load_meshes()
    for item in mesh_items:
        if item is not None:
            vis.add_geometry(item[0])

    # Gripper mesh — tracks tool0 TCP pose each frame
    _GRIPPER_OBJ = _HERE / "scene_layout" / "gripperWtihAdapters.obj"
    _gripper_mesh      = None
    _gripper_verts_ref = None
    _gripper_norms_ref = None
    if _GRIPPER_OBJ.exists():
        gm = o3d.io.read_triangle_mesh(str(_GRIPPER_OBJ))
        if len(gm.vertices) > 0:
            gm.compute_vertex_normals()
            gm.paint_uniform_color([0.75, 0.60, 0.40])
            _gripper_mesh      = gm
            _gripper_verts_ref = np.asarray(gm.vertices).copy()
            _gripper_norms_ref = np.asarray(gm.vertex_normals).copy()
            vis.add_geometry(_gripper_mesh)
    else:
        print(f"[ur_collision_model] gripper OBJ not found: {_GRIPPER_OBJ}")

    col_ls = o3d.geometry.LineSet()
    vis.add_geometry(col_ls)

    # Initial render pass (set camera)
    vis.poll_events(); vis.update_renderer()
    vc = vis.get_view_control()
    base_ctr = T_world_base[:3, 3]
    vc.set_lookat((base_ctr + np.array([0, 0, 0.5])).tolist())
    vc.set_up([0, 0, 1]); vc.set_front([1.2, -1.0, 0.8]); vc.set_zoom(0.5)

    while vis.poll_events():
        with _lock:
            is_dirty = _dirty[0]
            _dirty[0] = False

        if is_dirty:
            cur_q, cur_grip, cur_hl = get_q()
            scene.update_robot(cur_q)
            poses = scene.get_arm_link_world_poses()

            for i, item in enumerate(mesh_items):
                if item is not None:
                    _apply_transform(item, poses[i])
                    vis.update_geometry(item[0])

            # Update gripper mesh to follow TCP (tool0) world pose.
            # T_grip_offset aligns the OBJ frame to tool0: 90° around local X.
            if _gripper_mesh is not None:
                T_tcp = scene.update_tcp_bodies()
                if T_tcp is not None:
                    _T_grip_offset = np.eye(4)
                    _T_grip_offset[:3, :3] = ScipyR.from_euler('x', np.pi / 2).as_matrix()
                    T = T_tcp @ _T_grip_offset
                    R, t = T[:3, :3], T[:3, 3]
                    _gripper_mesh.vertices      = o3d.utility.Vector3dVector(
                        _gripper_verts_ref @ R.T + t)
                    _gripper_mesh.vertex_normals = o3d.utility.Vector3dVector(
                        _gripper_norms_ref @ R.T)
                    vis.update_geometry(_gripper_mesh)

            spheres = _compute_sphere_world(poses, cur_grip)
            if spheres:
                new_ls = _sphere_wireframe(spheres, cur_hl)
                col_ls.points = new_ls.points
                col_ls.lines  = new_ls.lines
                col_ls.colors = new_ls.colors
            else:
                col_ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                col_ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), int))
            vis.update_geometry(col_ls)

        vis.update_renderer()
        time.sleep(0.033)

    vis.destroy_window()
    scene.disconnect()


if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser(description="UR10e collision model viewer")
    parser.add_argument("--gripper", action="store_true", help="Show gripper spheres")
    parser.add_argument("--q", nargs=6, type=float, metavar="DEG",
                        default=[0, -90, 90, -90, -90, 0],
                        help="Initial joint angles in degrees (6 values)")
    args = parser.parse_args()
    visualize(q=np.deg2rad(args.q), with_gripper=args.gripper)
