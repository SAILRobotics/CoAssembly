"""scene_viewer_o3d.py — Open3D 3-D scene visualiser for CoAssembly.

Owns the Open3D Visualizer window and all geometry managed within it:
camera / head frustums, robot arm meshes, gripper mesh, pegboard outline,
hand point-clouds, tool bounding boxes, workspace boundary, handover grid,
reachability arrows, and quat-debug overlays.

Used by main_with_robot.py:
    from scene_viewer_o3d import SceneVis
"""

import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as ScipyR

import main_setting as cfg
from utils.pose_helpers import (
    _BONES_NP, _N_JOINTS, _HIDDEN_PT,
    _palm_quat, _tool_grasp_quat,
)


class SceneVis:
    FRUSTUM_SCALE = 0.2

    _WORKSPACE_COLOR = np.array([0.4, 0.7, 1.0])

    _HANDOVER_GRID_COLOR   = np.array([0.45, 0.45, 0.90])
    _HANDOVER_VALID_COLOR  = np.array([0.20, 0.90, 0.40])
    _HANDOVER_SPHERE_COLOR = (1.0, 0.85, 0.10)

    _GEARBOX_UNITY_TO_OBJ_PREFIX = {
        "BaseBoard": "BaseBoard",
        "Bearing_Row1_Left": "Row1_Bearing_Left",
        "Bearing_Row1_Right": "Row1_Bearing_Right",
        "Bearing_Row2_Left": "Row2_Bearing_Left",
        "Bearing_Row2_Right": "Row2_Bearing_Right",
        "Bearing_Row3_Left": "Row3_Bearing_Left",
        "Bearing_Row3_Right": "Row3_Bearing_Right",
        "Bearing_Row4_Left": "Row4_Bearing_Left",
        "Bearing_Row4_Right": "Row4_Bearing_Right",
        "Stand_Row1_Left": "Row1_GearStand_Left",
        "Stand_Row1_Right": "Row1_GearStand_Right",
        "Stand_Row2_Left": "Row2_GearStand_Left",
        "Stand_Row2_Right": "Row2_GearStand_Right",
        "Stand_Row3_Left": "Row3_GearStand_Left",
        "Stand_Row3_Right": "Row3_GearStand_Right",
        "Stand_Row4_Left": "Row4_GearStand_Left",
        "Stand_Row4_Right": "Row4_GearStand_Right",
        "GearRod_Row1": "Row1_GearRod",
        "GearRod_Row2": "Row2_GearRod",
        "GearRod_Row3": "Row3_GearRod",
        "GearRod_Row4": "Row4_GearRod",
        "Gear_Row1_Left": "Row1_Gear_Left",
        "Gear_Row2_Left": "Row2_Gear_Left",
        "Gear_Row2_Right": "Row2_Gear_Right",
        "Gear_Row3_Left": "Row3_Gear_Left",
        "Gear_Row3_Right": "Row3_Gear_Right",
        "Gear_Row4_Left": "Row4_Gear_Left",
        "Pin_Row1_Left": "Row1_Pin_Left",
        "Pin_Row1_Right": "Row1_Pin_Right",
        "Pin_Row2_Left": "Row2_Pin_Left",
        "Pin_Row2_Right": "Row2_Pin_Right",
        "Pin_Row3_Left": "Row3_Pin_Left",
        "Pin_Row3_Right": "Row3_Pin_Right",
        "Pin_Row4_Left": "Row4_Pin_Left",
        "Screw_Row1_Left": "Row1_Screw_Left",
        "Screw_Row1_Right": "Row1_Screw_Right",
        "Screw_Row2_Left": "Row2_Screw_Left",
        "Screw_Row2_Right": "Row2_Screw_Right",
        "Screw_Row3_Left": "Row3_Screw_Left",
        "Screw_Row3_Right": "Row3_Screw_Right",
        "Screw_Row4_Left": "Row4_Screw_Left",
        "Screw_Row4_Right": "Row4_Screw_Right",
        "CrankHandle_Row1": "Row1_Handle",
    }
    # ── Static geometry helpers ───────────────────────────────────────────────

    @staticmethod
    def make_axes_lineset(T: np.ndarray, size: float = 0.10,
                          color=None) -> o3d.geometry.LineSet:
        """RGB XYZ axes as a LineSet at the given 4×4 pose.
        Pass color=(r,g,b) to draw all three axes in a single colour."""
        o = T[:3, 3]
        pts = np.array([o,
                        o + T[:3, 0] * size,
                        o + T[:3, 1] * size,
                        o + T[:3, 2] * size])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        if color is None:
            ls.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            ls.colors = o3d.utility.Vector3dVector([color, color, color])
        return ls

    @staticmethod
    def make_sphere_wireframe(centers: np.ndarray, radii: np.ndarray,
                              n_pts: int = 16,
                              color=(0.3, 1.0, 0.3)) -> o3d.geometry.LineSet:
        """LineSet of 3 great-circles (XY/XZ/YZ) per sphere — lightweight wireframe."""
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        zeros = np.zeros(n_pts)
        all_pts, all_lines = [], []
        offset = 0
        for center, r in zip(centers, radii):
            for c1, c2, c3 in [(cos_t, sin_t, zeros),
                                (cos_t, zeros, sin_t),
                                (zeros, cos_t, sin_t)]:
                pts = np.column_stack([c1 * r, c2 * r, c3 * r]) + center
                all_pts.append(pts)
                base = offset
                for i in range(n_pts):
                    all_lines.append([base + i, base + (i + 1) % n_pts])
                offset += n_pts
        if not all_pts:
            return o3d.geometry.LineSet()
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        ls.lines  = o3d.utility.Vector2iVector(all_lines)
        ls.colors = o3d.utility.Vector3dVector([list(color)] * len(all_lines))
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

        self._cam_frustum            = None
        self._head_frustum           = None
        self._passthrough_cam_frustum = None
        self._tcp_axes        = None
        self._tcp_target_ls         = None
        self._tcp_target_axes       = None
        self._gripper_tip_target_axes = None
        self._tool_box_linesets: list = []
        # Pegboard tool-box colours: neutral grey by default; boxes whose index is in
        # _tool_highlight_indices turn cyan — mirroring gearbox_control.py --open-3d. The set is
        # remembered so an update_tool_boxes() rebuild (e.g. on relock) keeps the highlight.
        self._tool_box_neutral        = (0.55, 0.55, 0.58)
        self._tool_box_highlight      = (0.0, 1.0, 1.0)
        self._tool_highlight_indices: set = set()
        self._tool_hidden_indices: set = set()

        self.show_collision_spheres = True
        self._collision_sphere_ls = o3d.geometry.LineSet()
        self.vis.add_geometry(self._collision_sphere_ls)
        self._pegboard_corners_local: np.ndarray | None = None

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

        self._workspace_box_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._workspace_box_lineset)

        self._handover_grid_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._handover_grid_lineset)
        self._handover_sphere_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._handover_sphere_lineset)

        self._reach_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self._reach_lineset)

        self._tracking_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._tracking_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_frame)
        self._tracking_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.020)
        self._tracking_sphere.paint_uniform_color([0.2, 0.4, 1.0])
        self._tracking_sphere.compute_vertex_normals()
        self._tracking_sphere.transform(self._hidden_T())
        self.vis.add_geometry(self._tracking_sphere)
        self._tracking_T = self._hidden_T()

        self._board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._board_frame.transform(self._hidden_T())
        self.vis.add_geometry(self._board_frame)
        self._world_baseboard_mesh = None
        self._board_mesh       = None
        self._board_manip_mesh = None

        _asset_dir = cfg.SCENE_LAYOUT_DIR.parent / "robot_assets"
        _baseboard_path = _asset_dir / "baseboard.obj"
        if _baseboard_path.exists():
            _mesh = o3d.io.read_triangle_mesh(str(_baseboard_path))
            _mesh.compute_vertex_normals()
            _mesh.paint_uniform_color([0.45, 0.45, 0.45])
            self.vis.add_geometry(_mesh)
            self._world_baseboard_mesh = _mesh
            print(f"[SceneVis] baseboard.obj loaded at world origin ({len(_mesh.vertices)} verts)")
        else:
            print(f"[SceneVis] baseboard.obj not found at {_baseboard_path}")

        _tracked_board_path = _asset_dir / "NewBaseBoard.obj"
        if _tracked_board_path.exists():
            def _load_tracked_board(color):
                _bm = o3d.io.read_triangle_mesh(str(_tracked_board_path))
                _bm.compute_vertex_normals()
                _bm.paint_uniform_color(color)
                _bm.transform(self._hidden_T())
                self.vis.add_geometry(_bm)
                return _bm

            self._board_mesh = _load_tracked_board([0.9, 0.75, 0.5])
            self._board_manip_mesh = _load_tracked_board([0.5, 0.75, 0.9])
            print(f"[SceneVis] NewBaseBoard.obj loaded for tracked board ({len(self._board_mesh.vertices)} verts)")
        else:
            print(f"[SceneVis] NewBaseBoard.obj not found at {_tracked_board_path}")
        self._board_T       = self._hidden_T()
        self._board_manip_T = self._hidden_T()

        self._gearbox_parts = {}
        self._gearbox_first_update_logged = False
        self._gearbox_initial_states = None
        self._gearbox_root_initial_T = None
        self._gearbox_done = {row: set() for row in range(1, 5)}
        self._gearbox_blocked_view = None
        self._load_gearbox_mirror_meshes()


        self._tcp_gripper_mesh = None
        self._left_hand_gripper_mesh = None
        self._tcp_T = self._hidden_T()
        self._left_hand_gripper_T = self._hidden_T()
        # Full gripper/adapter model whose OBJ origin is authored at the robot
        # TCP. It follows the same live/commanded TCP transform used by the
        # handover visualization in update_tcp().
        _gripper_path = (_asset_dir / "RobotiqGripperWithAdapters.obj")
        if _gripper_path.exists():
            _mesh = o3d.io.read_triangle_mesh(str(_gripper_path))
            _mesh.compute_vertex_normals()
            _mesh.paint_uniform_color([0.75, 0.75, 0.75])
            # Keep the known-good original robot-TCP alignment separate from
            # the left-hand preview's additional OBJ-local Y flip.
            _R_fix_x = ScipyR.from_euler('x', 90, degrees=True).as_matrix()
            _R_flip_y = ScipyR.from_euler('y', 180, degrees=True).as_matrix()
            _R_flip_tcp_z = ScipyR.from_euler('z', 180, degrees=True).as_matrix()

            _left_mesh = copy.deepcopy(_mesh)
            _T_fix_left = np.eye(4, dtype=np.float64)
            # Right multiplication: rotate around the model's own Y first.
            _T_fix_left[:3, :3] = _R_fix_x @ _R_flip_y
            _left_mesh.transform(_T_fix_left)
            _left_mesh.paint_uniform_color([0.20, 0.90, 0.40])
            _left_mesh.transform(self._hidden_T())
            self.vis.add_geometry(_left_mesh)
            self._left_hand_gripper_mesh = _left_mesh

            _T_fix_robot = np.eye(4, dtype=np.float64)
            # Left multiplication rotates the already-aligned mesh around the
            # TCP frame's Z axis.
            _T_fix_robot[:3, :3] = _R_flip_tcp_z @ _R_fix_x
            _mesh.transform(_T_fix_robot)
            _mesh.transform(self._hidden_T())
            self.vis.add_geometry(_mesh)
            self._tcp_gripper_mesh = _mesh
            print(f"[SceneVis] TCP gripper loaded from {_gripper_path.name} "
                  f"({len(_mesh.vertices)} verts)")
        else:
            print(f"[SceneVis] TCP gripper not found at {_gripper_path}")

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

        self._qd_palm_tri    = o3d.geometry.LineSet()
        self._qd_palm_normal = o3d.geometry.LineSet()
        self._qd_palm_frame  = o3d.geometry.LineSet()
        self._qd_tool_face   = o3d.geometry.LineSet()
        self._qd_tool_normal = o3d.geometry.LineSet()
        self._qd_tool_frame  = o3d.geometry.LineSet()
        self._qd_box_grip_z  = o3d.geometry.LineSet()
        self._qd_box_tcp     = o3d.geometry.LineSet()
        # Always-on palm triangle + centroid + normal for both hands
        self._left_palm_tri      = o3d.geometry.LineSet()
        self._left_palm_normal   = o3d.geometry.LineSet()
        self._left_palm_centroid = o3d.geometry.PointCloud()
        self._right_palm_tri     = o3d.geometry.LineSet()
        self._right_palm_normal  = o3d.geometry.LineSet()
        self._right_palm_centroid = o3d.geometry.PointCloud()
        for _ls in [self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame,
                    self._qd_tool_face, self._qd_tool_normal, self._qd_tool_frame,
                    self._qd_box_grip_z, self._qd_box_tcp,
                    self._left_palm_tri, self._left_palm_normal,
                    self._right_palm_tri, self._right_palm_normal]:
            self.vis.add_geometry(_ls)
        for _pc in [self._left_palm_centroid, self._right_palm_centroid]:
            self.vis.add_geometry(_pc)

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

    @staticmethod
    def _gearbox_pre_rotation_T():
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('x', -90, degrees=True).as_matrix()
        return T

    @staticmethod
    def _gearbox_post_rotation_T():
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = ScipyR.from_euler('z', 180, degrees=True).as_matrix()
        return T

    @classmethod
    def _gearbox_corrected_T(cls, T_pose: np.ndarray) -> np.ndarray:
        # Apply the gearbox basis correction to orientation only. Left-multiplying
        # a full Rz(180) transform would also rotate translation, mirroring +X/-Y
        # motion into -X/+Y. Keep the received Unity/Open3D position intact.
        T = np.asarray(T_pose, dtype=np.float64).copy()
        R_pre = cls._gearbox_pre_rotation_T()[:3, :3]
        R_post = cls._gearbox_post_rotation_T()[:3, :3]
        T[:3, :3] = R_post @ T[:3, :3] @ R_pre
        return T

    def _gearbox_color_for_part(self, unity_name: str):
        """Neutral baseline for every part in the Open3D gearbox mirror."""
        return [0.55, 0.55, 0.55]

    @staticmethod
    def _gearbox_part_fields(unity_name: str):
        """Return (type, row, side) using the Unity gearbox naming convention."""
        clean = unity_name.replace("_", "")
        idx = clean.find("Row")
        if idx < 0:
            return ("BaseBoard", 0, "") if unity_name == "BaseBoard" else (None, None, None)
        ptype = clean[:idx]
        i = idx + 3
        digits = ""
        while i < len(clean) and clean[i].isdigit():
            digits += clean[i]
            i += 1
        return (ptype, int(digits), clean[i:]) if digits else (None, None, None)

    @staticmethod
    def _gearbox_part_stages(ptype: str, row: int, side: str):
        """Python copy of Unity's appear/place/seat stage definitions."""
        left = side == "Left"
        if ptype in ("Bearing", "Stand"):
            return (1, 4, 4) if left else (3, 5, 6)
        if ptype == "Pin" and row == 1 and side == "Right":
            return 7, 7, 7
        if ptype in ("GearRod", "Gear", "Pin"):
            return 2, 5, 5
        if ptype == "Screw":
            return (4, 4, 4) if left else (6, 6, 6)
        if ptype == "CrankHandle":
            return 7, 7, 7
        return None, None, None

    @staticmethod
    def _gearbox_part_in_stage(ptype: str, row: int, side: str,
                               selected_row: int, stage: int) -> bool:
        """Whether a part participates in a selected control stage.

        This mirrors Unity's GroupsForStage selectors, excluding BaseBoard so
        a blocked operation flags the actionable components rather than the
        whole assembly foundation.
        """
        if row != selected_row or ptype == "BaseBoard":
            return False
        if stage == 1:
            return ptype in ("Stand", "Bearing") and side == "Left"
        if stage == 2:
            return ptype in ("GearRod", "Gear", "Pin")
        if stage == 3:
            return ptype in ("Stand", "Bearing") and side == "Right"
        if stage == 4:
            return ((ptype in ("Stand", "Bearing") and side == "Left")
                    or (ptype == "Screw" and side == "Left"))
        if stage == 5:
            return (ptype in ("GearRod", "Gear", "Pin")
                    or (ptype in ("Stand", "Bearing") and side == "Right"))
        if stage == 6:
            return ptype == "Screw" and side == "Right"
        if stage == 7:
            return (selected_row == 1
                    and (ptype == "CrankHandle"
                         or (ptype == "Pin" and side == "Right")))
        return False

    def _refresh_gearbox_assembly_colors(self):
        seated = [0.15, 0.80, 0.20]
        built = [1.0, 0.55, 0.0]
        blocked = [0.85, 0.15, 0.15]
        for name, entry in self._gearbox_parts.items():
            ptype, row, side = self._gearbox_part_fields(name)
            color = self._gearbox_color_for_part(name)
            if row in self._gearbox_done:
                appear, _place, seat = self._gearbox_part_stages(ptype, row, side)
                done = self._gearbox_done[row]
                if seat in done:
                    color = seated
                elif appear in done:
                    color = built
                blocked_view = self._gearbox_blocked_view
                if (blocked_view is not None
                        and self._gearbox_part_in_stage(
                            ptype, row, side, blocked_view[0], blocked_view[1])):
                    color = blocked
            entry["mesh"].paint_uniform_color(color)
            self.vis.update_geometry(entry["mesh"])

    def apply_gearbox_assembly_event(self, msg: dict) -> None:
        """Update Open3D colors directly from gearbox_control semantic events."""
        event = msg.get("event")
        row = int(msg.get("row", 0))
        stage = int(msg.get("stage", 0))
        if event == "reset":
            for done in self._gearbox_done.values():
                done.clear()
            self._gearbox_blocked_view = None
        elif event == "complete" and row in self._gearbox_done:
            self._gearbox_done[row].add(stage)
            self._gearbox_blocked_view = None
        elif event == "uncomplete" and row in self._gearbox_done:
            self._gearbox_done[row].discard(stage)
            self._gearbox_blocked_view = None
        elif event == "show":
            self._gearbox_blocked_view = ((row, stage)
                                          if msg.get("blocked", False) else None)
        elif event == "close":
            self._gearbox_blocked_view = None
        else:
            return
        self._refresh_gearbox_assembly_colors()

    @staticmethod
    def _clean_obj_group_name(name: str) -> str:
        if ".001_mesh" in name:
            return name.split(".001_mesh", 1)[0]
        if "_mesh" in name:
            return name.split("_mesh", 1)[0]
        return name

    @staticmethod
    def _obj_face_index(token: str, vertex_count: int) -> int:
        idx = int(token.split("/", 1)[0])
        return idx - 1 if idx > 0 else vertex_count + idx

    def _load_obj_group_meshes(self, obj_path):
        vertices = []
        faces_by_group = {}
        current_group = None
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("g ") or line.startswith("o "):
                    current_group = self._clean_obj_group_name(line[2:].strip())
                    faces_by_group.setdefault(current_group, [])
                elif line.startswith("f ") and current_group is not None:
                    refs = [self._obj_face_index(tok, len(vertices)) for tok in line.split()[1:]]
                    if len(refs) >= 3:
                        for i in range(1, len(refs) - 1):
                            faces_by_group[current_group].append([refs[0], refs[i], refs[i + 1]])

        verts = np.asarray(vertices, dtype=np.float64)
        meshes = {}
        for group, faces in faces_by_group.items():
            if not faces:
                continue
            used = sorted({idx for face in faces for idx in face})
            remap = {old: new for new, old in enumerate(used)}
            local_faces = [[remap[idx] for idx in face] for face in faces]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts[used])
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(local_faces, dtype=np.int32))
            mesh.compute_vertex_normals()
            meshes[group] = mesh
        return meshes

    def _load_gearbox_mirror_meshes(self):
        obj_path = cfg.SCENE_LAYOUT_DIR.parent / "robot_assets" / "gearbox_parts" / "GearBoxAssembly_Uncolored.obj"
        if not obj_path.exists():
            print(f"[SceneVis] gearbox OBJ missing: {obj_path}")
            return
        obj_meshes = self._load_obj_group_meshes(obj_path)
        hidden = self._hidden_T()
        for unity_name, group_name in self._GEARBOX_UNITY_TO_OBJ_PREFIX.items():
            mesh = obj_meshes.get(group_name)
            if mesh is None:
                print(f"[SceneVis] gearbox OBJ group missing for {unity_name}: {group_name}")
                continue
            mesh.paint_uniform_color(self._gearbox_color_for_part(unity_name))
            mesh.transform(hidden)
            self.vis.add_geometry(mesh)

            self._gearbox_parts[unity_name] = {
                "mesh": mesh,
                "T": hidden.copy(),
            }
        print(f"[SceneVis] gearbox mirror loaded {len(self._gearbox_parts)} OBJ group instances from {obj_path.name}")

    @staticmethod
    def _T_from_pose_scale(T_pose: np.ndarray, scale) -> np.ndarray:
        return np.array(T_pose, dtype=np.float64, copy=True)

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

    def update_passthrough_cam(self, T: np.ndarray | None,
                               w=640, h=480, fx=400., fy=400., cx=320., cy=240.):
        T_use = T if T is not None else self._hidden_T()
        intr  = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
        new_fr = o3d.geometry.LineSet.create_camera_visualization(
            int(w), int(h), intr.intrinsic_matrix,
            np.linalg.inv(T_use), scale=self.FRUSTUM_SCALE)
        new_fr.paint_uniform_color([1.0, 1.0, 0.0])
        if self._passthrough_cam_frustum is None:
            self._passthrough_cam_frustum = new_fr
            self.vis.add_geometry(self._passthrough_cam_frustum)
        else:
            self._passthrough_cam_frustum.points = new_fr.points
            self._passthrough_cam_frustum.lines  = new_fr.lines
            self._passthrough_cam_frustum.colors = new_fr.colors
            self.vis.update_geometry(self._passthrough_cam_frustum)

    def set_pegboard_outline(self, offset_x: float, offset_y: float,
                              width: float, height: float):
        """Store pegboard corners in marker-local frame (marker = origin).
        offset_x/y: distance from marker centre to the right/top board edge.
        Call once after loading the pegboard NPZ; update_pegboard() uses it."""
        self._pegboard_corners_local = np.array([
            [ offset_x,         offset_y,          0.0],
            [ offset_x - width, offset_y,          0.0],
            [ offset_x - width, offset_y - height, 0.0],
            [ offset_x,         offset_y - height, 0.0],
        ])
        self._pegboard_lineset.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0]])
        self._pegboard_lineset.colors = o3d.utility.Vector3dVector(
            [[0.1, 0.6, 1.0]] * 4)
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

    def update_workspace_bound(self, lo: "np.ndarray | None", hi: "np.ndarray | None") -> None:
        """Axis-aligned wireframe box from lo/hi (world frame)."""
        if lo is None or hi is None:
            return
        pos  = (np.asarray(lo) + np.asarray(hi)) / 2.0
        size = np.asarray(hi) - np.asarray(lo)
        new_ls = self.make_box_lineset(pos, np.eye(3), size, color=self._WORKSPACE_COLOR)
        self._workspace_box_lineset.points = new_ls.points
        self._workspace_box_lineset.lines  = new_ls.lines
        self._workspace_box_lineset.colors = new_ls.colors
        self.vis.update_geometry(self._workspace_box_lineset)

    def update_handover(self, result):
        """Draw the handover voxel grid (valid vs invalid coloured) plus a
        wireframe sphere at the chosen centroid. `result` comes from
        MainScene._compute_handover; None hides everything."""
        if result is None:
            self.clear_handover()
            return
        cents = np.asarray(result['centroids'])
        R     = np.asarray(result['R'])
        cw, cd, ch = result['cell']
        valid = np.asarray(result['valid_mask'])
        hx, hy, hz = cw / 2.0, cd / 2.0, ch / 2.0
        local = np.array([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                          [-hx,-hy, hz],[hx,-hy, hz],[hx,hy, hz],[-hx,hy, hz]])
        rot_corners = local @ R.T
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        pts, lines, cols = [], [], []
        off = 0
        for cen, v in zip(cents, valid):
            pts.append(rot_corners + cen)
            lines.extend([[off + a, off + b] for a, b in edges])
            col = self._HANDOVER_VALID_COLOR if v else self._HANDOVER_GRID_COLOR
            cols.extend([col] * len(edges))
            off += 8
        ls = self._handover_grid_lineset
        ls.points = o3d.utility.Vector3dVector(np.vstack(pts))
        ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
        self.vis.update_geometry(ls)

        sc  = result.get('sphere_center')
        sph = self._handover_sphere_lineset
        if sc is not None:
            new_s = self.make_sphere_wireframe(np.asarray([sc], dtype=np.float64),
                                               np.array([0.04]),
                                               color=self._HANDOVER_SPHERE_COLOR)
            sph.points = new_s.points
            sph.lines  = new_s.lines
            sph.colors = new_s.colors
            self.vis.update_geometry(sph)

    def clear_handover(self):
        """Hide the handover grid + sphere (empty geometry)."""
        for ls in (self._handover_grid_lineset, self._handover_sphere_lineset):
            ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.vis.update_geometry(ls)

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
        self._board_T = T_new

    def update_gearbox_mirror(self, states: dict | None):
        if states is None:
            return
        hidden = self._hidden_T()
        if self._gearbox_initial_states is None:
            self._gearbox_initial_states = {
                name: np.array(state["T"], dtype=np.float64, copy=True)
                for name, state in states.items()
                if state is not None and "T" in state
            }
            root_state = states.get("BaseBoard")
            if root_state is not None and "T" in root_state:
                self._gearbox_root_initial_T = np.array(root_state["T"], dtype=np.float64, copy=True)
            else:
                self._gearbox_root_initial_T = np.eye(4, dtype=np.float64)

        matched = 0
        active = 0
        for name, entry in self._gearbox_parts.items():
            state = states.get(name)
            T_initial = self._gearbox_initial_states.get(name) if self._gearbox_initial_states else None
            if state is None or T_initial is None or not state.get("active", True):
                T_new = hidden
            else:
                matched += 1
                active += 1
                T_current = self._gearbox_corrected_T(self._T_from_pose_scale(state["T"], state.get("scale")))
                T_initial_corr = self._gearbox_corrected_T(T_initial)
                root_initial_corr = self._gearbox_corrected_T(self._gearbox_root_initial_T)
                # OBJ groups are already in assembled rest coordinates. Move them by
                # the live Unity rigid-body delta from the first received rest pose.
                T_new = T_current @ np.linalg.inv(T_initial_corr) @ root_initial_corr
            delta = T_new @ np.linalg.inv(entry["T"])
            entry["mesh"].transform(delta)
            entry["T"] = np.array(T_new, dtype=np.float64, copy=True)
            self.vis.update_geometry(entry["mesh"])

        if not self._gearbox_first_update_logged:
            missing = sorted(set(self._gearbox_parts) - set(states))[:8]
            root_note = "BaseBoard" if "BaseBoard" in states else "identity fallback"
            print(f"[SceneVis] gearbox mirror first update: matched={matched}/{len(self._gearbox_parts)} active={active}; root={root_note}; sample_missing={missing}")
            self._gearbox_first_update_logged = True

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

    def update_left_hand_gripper(self, T: "np.ndarray | None") -> None:
        """Keep the green handover-preview gripper at the left-hand target."""
        if self._left_hand_gripper_mesh is None:
            return
        T_new = T if T is not None else self._hidden_T()
        delta = T_new @ np.linalg.inv(self._left_hand_gripper_T)
        self._left_hand_gripper_mesh.transform(delta)
        self.vis.update_geometry(self._left_hand_gripper_mesh)
        self._left_hand_gripper_T = T_new

    def update_tcp_target(self, T: "np.ndarray | None"):
        """Draw the move_to_pose target as a magenta sphere + RGB axes frame."""
        T_new = T if T is not None else self._hidden_T()

        # Magenta wireframe sphere at target position
        new_ls = self.make_sphere_wireframe(
            np.array([T_new[:3, 3]]), np.array([0.035]),
            color=(1.0, 0.0, 1.0))
        if self._tcp_target_ls is None:
            self._tcp_target_ls = new_ls
            self.vis.add_geometry(self._tcp_target_ls)
        else:
            self._tcp_target_ls.points = new_ls.points
            self._tcp_target_ls.lines  = new_ls.lines
            self._tcp_target_ls.colors = new_ls.colors
            self.vis.update_geometry(self._tcp_target_ls)

        # RGB axes frame showing target orientation
        new_axes = self.make_axes_lineset(T_new, size=0.07)
        if self._tcp_target_axes is None:
            self._tcp_target_axes = new_axes
            self.vis.add_geometry(self._tcp_target_axes)
        else:
            self._tcp_target_axes.points = new_axes.points
            self._tcp_target_axes.lines  = new_axes.lines
            self._tcp_target_axes.colors = new_axes.colors
            self.vis.update_geometry(self._tcp_target_axes)

    _GRIPPER_TIP_OFFSET = 0.185  # metres from TCP along TCP Z axis

    def update_gripper_tip_target(self, T_tcp: "np.ndarray | None"):
        """Draw the gripper fingertip target (18.5 cm along TCP Z) as a cyan axes frame."""
        if T_tcp is not None:
            T_tip = T_tcp.copy()
            T_tip[:3, 3] += T_tcp[:3, :3] @ np.array([0.0, 0.0, self._GRIPPER_TIP_OFFSET])
        else:
            T_tip = self._hidden_T()

        new_axes = self.make_axes_lineset(T_tip, size=0.07, color=(0.0, 1.0, 1.0))
        if self._gripper_tip_target_axes is None:
            self._gripper_tip_target_axes = new_axes
            self.vis.add_geometry(self._gripper_tip_target_axes)
        else:
            self._gripper_tip_target_axes.points = new_axes.points
            self._gripper_tip_target_axes.lines  = new_axes.lines
            self._gripper_tip_target_axes.colors = new_axes.colors
            self.vis.update_geometry(self._gripper_tip_target_axes)

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
            if i < len(boxes) and i not in self._tool_hidden_indices:
                pos, R, size = boxes[i]
                col = (self._tool_box_highlight if i in self._tool_highlight_indices
                       else self._tool_box_neutral)
                new_ls = self.make_box_lineset(pos, R, size, color=col)
                ls.points = new_ls.points
                ls.lines  = new_ls.lines
                ls.colors = new_ls.colors
            else:
                ls.points = _hidden_pts
                ls.lines  = _box_edges
            self.vis.update_geometry(ls)

    def set_tool_highlight_indices(self, indices) -> None:
        """Cyan-highlight the pegboard tool boxes at these list indices (box i == tool i in
        tool_layout.world_boxes() order); every other box goes neutral. Recolours the existing
        linesets in place and is remembered so a later update_tool_boxes() keeps the highlight.
        Mirrors gearbox_control.py --open-3d's PegboardBoxViewer.set_highlight."""
        self._tool_highlight_indices = {int(i) for i in indices}
        for i, ls in enumerate(self._tool_box_linesets):
            if i in self._tool_hidden_indices:
                continue
            if len(np.asarray(ls.points)) != 8:
                continue   # skip hidden/placeholder linesets
            col = (self._tool_box_highlight if i in self._tool_highlight_indices
                   else self._tool_box_neutral)
            ls.colors = o3d.utility.Vector3dVector([list(col)] * 12)
            self.vis.update_geometry(ls)

    def set_tool_hidden_indices(self, indices) -> None:
        """Hide boxes for successfully grasped objects while preserving box order."""
        self._tool_hidden_indices = {int(i) for i in indices}
        # Rebuild from the last geometry is intentionally avoided here; callers
        # invoke update_tool_boxes immediately after changing this set.

    def update_robot(self, link_poses: list[np.ndarray]):
        """Move UR10e arm meshes to the given PyBullet link world poses.
        link_poses: 7 transforms [base, shoulder, upper_arm, forearm, wrist1, wrist2, wrist3]."""
        for i, (mesh, T_new) in enumerate(zip(self._robot_meshes, link_poses)):
            if mesh is None:
                continue
            T_cur = self._robot_mesh_Ts[i]
            delta = T_new @ np.linalg.inv(T_cur)
            mesh.transform(delta)
            self.vis.update_geometry(mesh)
            self._robot_mesh_Ts[i] = T_new

    def update_collision_spheres(self, positions: "np.ndarray | None",
                                 radii: "np.ndarray | None") -> None:
        """Draw collision-sphere wireframes. Hidden when show_collision_spheres=False."""
        empty = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        positions_ok = (positions is not None and len(positions) > 0
                        and not np.any(np.isnan(positions))
                        and not np.any(np.isinf(positions)))
        if not self.show_collision_spheres or not positions_ok:
            self._collision_sphere_ls.points = empty
            self._collision_sphere_ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), int))
            self._collision_sphere_ls.colors = empty
        else:
            new_ls = self.make_sphere_wireframe(positions, radii)
            self._collision_sphere_ls.points = new_ls.points
            self._collision_sphere_ls.lines  = new_ls.lines
            self._collision_sphere_ls.colors = new_ls.colors
        self.vis.update_geometry(self._collision_sphere_ls)

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

    def update_palm_triangles(self, left_pts, right_pts) -> None:
        """Always-on gold triangle + centroid dot + normal arrow for both hands."""
        _GOLD = (0.9, 0.75, 0.1)
        for pts, is_left, tri, norm, cen in [
            (left_pts,  True,  self._left_palm_tri,  self._left_palm_normal,  self._left_palm_centroid),
            (right_pts, False, self._right_palm_tri, self._right_palm_normal, self._right_palm_centroid),
        ]:
            if pts is None or len(pts) <= 6:
                self._clear_ls(tri, norm)
                cen.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                cen.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                self.vis.update_geometry(cen)
                continue
            p_th     = np.asarray(pts[3], float)
            p_pa     = np.asarray(pts[1], float)
            p_ix     = np.asarray(pts[6], float)
            centroid = (p_th + p_pa + p_ix) / 3.0
            # Triangle edges
            tri.points = o3d.utility.Vector3dVector([p_th, p_pa, p_ix])
            tri.lines  = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 0]])
            tri.colors = o3d.utility.Vector3dVector([_GOLD] * 3)
            self.vis.update_geometry(tri)
            # Centroid dot
            cen.points = o3d.utility.Vector3dVector([centroid])
            cen.colors = o3d.utility.Vector3dVector([_GOLD])
            self.vis.update_geometry(cen)
            # Standoff direction: -gripper_z from _palm_quat (jaw × wrist-to-palm)
            _gz = ScipyR.from_quat(_palm_quat(pts, is_left=is_left)).apply([0., 0., 1.])
            self._update_arrow_ls(norm, centroid, -_gz, 0.10, _GOLD)

    def update_palm_quat_debug(self, pts: np.ndarray, is_left: bool = False):
        if pts is None or len(pts) <= 6:
            self._clear_ls(self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame)
            return

        p_thumb = pts[3]; p_palm = pts[1]; p_index = pts[6]

        self._qd_palm_tri.points = o3d.utility.Vector3dVector([p_thumb, p_palm, p_index])
        self._qd_palm_tri.lines  = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 0]])
        self._qd_palm_tri.colors = o3d.utility.Vector3dVector([[0.9, 0.75, 0.1]] * 3)
        self.vis.update_geometry(self._qd_palm_tri)

        q_tcp = _palm_quat(pts, is_left=is_left)
        R_tcp = ScipyR.from_quat(q_tcp).as_matrix()
        T_tcp = np.eye(4); T_tcp[:3, :3] = R_tcp; T_tcp[:3, 3] = p_palm
        fl = self.make_axes_lineset(T_tcp, size=0.07)
        self._qd_palm_frame.points = fl.points
        self._qd_palm_frame.lines  = fl.lines
        self._qd_palm_frame.colors = fl.colors
        self.vis.update_geometry(self._qd_palm_frame)

        # Triangle normal: geometric cross product of palm triangle edges
        centroid  = (np.asarray(p_thumb, float) + np.asarray(p_palm, float) + np.asarray(p_index, float)) / 3.0
        _palm_n   = np.cross(np.asarray(p_palm, float) - np.asarray(p_thumb, float),
                             np.asarray(p_index, float) - np.asarray(p_thumb, float))
        _norm_len = np.linalg.norm(_palm_n)
        if _norm_len > 1e-9:
            _palm_n /= _norm_len
            if np.dot(_palm_n, -R_tcp[:, 2]) < 0:
                _palm_n = -_palm_n
        if is_left:
            self._update_arrow_ls(
                self._qd_palm_normal, centroid, _palm_n, 0.10, (1., 1., 0.))
        else:
            # The right hand already has the always-on -gripper_z standoff
            # arrow; hide this second geometric-normal debug arrow.
            self._clear_ls(self._qd_palm_normal)

    def clear_palm_quat_debug(self):
        self._clear_ls(self._qd_palm_tri, self._qd_palm_normal, self._qd_palm_frame)

    def update_tool_quat_debug(self, centroid: np.ndarray, R_world: np.ndarray,
                                size: list, approach_dist: float = 0.10):
        sx, sy = size[0] / 2, size[1] / 2
        face_ctr = centroid + R_world[:, 2] * (size[2] / 2)
        face_local = np.array([[-sx, -sy, 0], [sx, -sy, 0],
                                [sx,  sy, 0], [-sx,  sy, 0]])
        face_pts = (face_local @ R_world.T) + face_ctr
        self._qd_tool_face.points = o3d.utility.Vector3dVector(face_pts)
        self._qd_tool_face.lines  = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
        self._qd_tool_face.colors = o3d.utility.Vector3dVector([[1., 0.5, 0.]] * 4)
        self.vis.update_geometry(self._qd_tool_face)

        self._update_arrow_ls(self._qd_tool_normal, centroid, R_world[:, 2], 0.09, (1., 1., 0.))

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
        grip_z  = T_target[:3, 2]
        tcp_pos = T_target[:3, 3] - cfg.BOX_FORWARD_OFFSET * grip_z

        if self._board_manip_mesh is not None:
            delta = T_target @ np.linalg.inv(self._board_manip_T)
            self._board_manip_mesh.transform(delta)
            self.vis.update_geometry(self._board_manip_mesh)
            self._board_manip_T = T_target

        self._update_arrow_ls(self._qd_box_grip_z, T_target[:3, 3], grip_z, 0.10, (1., 1., 0.))

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
        """Draw one arrow per grid point along board_normal: green=reachable, red=not."""
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

    def tick(self, pending_vis_clears: list | None = None):
        if pending_vis_clears:
            while pending_vis_clears:
                pending_vis_clears.pop()
            self.clear_tool_quat_debug()
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass
