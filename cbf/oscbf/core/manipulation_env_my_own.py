"""Simulation environment for manipulator end-effector pose tracking"""

import time
from typing import Optional, Dict, Tuple
from functools import partial
import argparse

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from oscbf.core.manipulator_my_own import Manipulator, load_panda
from oscbf.utils.visualization import visualize_3D_box
from oscbf.utils.general_utils import stdout_redirected, find_assets_dir
from oscbf.core.controllers_my_own import PoseTaskVelocityController
from oscbf.utils.trajectory import TaskTrajectory


class ManipulationEnv:
    """Simulation environment for manipulator end-effector pose tracking

    Args:
        urdf (str): Path to the URDF file of the robot
        control_mode (str): Control mode, either "torque" or "velocity"
        xyz_min (Optional[ArrayLike]): Minimum bounds of the safe region, shape (3,). Defaults to None.
        xyz_max (Optional[ArrayLike]): Maximum bounds of the safe region, shape (3,). Defaults to None.
        target_pos (ArrayLike): Initial position of the target, shape (3,). Defaults to (0.5, 0, 0.5).
        q_init (Optional[ArrayLike]): Initial joint positions of the robot, shape (num_joints,). Defaults to None.
        traj (Optional[TaskTrajectory]): Task-space trajectory for the target to follow. Defaults to None, in
            which case the target's position and orientation can be controlled by the user in the GUI.
        collision_data (Optional[Dict[str, Tuple[ArrayLike]]]): Collision information, represented as spheres.
            The dictionary should have keys "positions" and "radii". Defaults to None.
        wb_xyz_min (Optional[ArrayLike]): Minimum bounds of the whole-body safe region, shape (3,). Defaults to None.
        wb_xyz_max (Optional[ArrayLike]): Maximum bounds of the whole-body safe region, shape (3,). Defaults to None.
        bg_color (Optional[ArrayLike]): RGB background color of the simulation. Defaults to None
            (use default background color)
        load_floor (bool): Whether to load a floor into the simulation. Defaults to True.
        qdot_max (Optional[ArrayLike]): Maximum joint velocities, shape (num_joints,). Defaults to None.
        tau_max (Optional[ArrayLike]): Maximum joint torques, shape (num_joints,). Defaults to None.
        real_time (bool): Whether to run the simulation in "real time". Defaults to False.
        timestep (float): Simulation timestep. Defaults to 1/240 (Same as PyBullet's default timestep).
        load_table (bool): Whether to load a table into the simulation. Defaults to False.
    """

    def __init__(
        self,
        urdf: str,
        control_mode: str,
        xyz_min: Optional[ArrayLike] = None,
        xyz_max: Optional[ArrayLike] = None,
        target_pos: ArrayLike = (0.5, 0, 0.5),
        q_init: Optional[ArrayLike] = None,
        traj: Optional[TaskTrajectory] = None,
        collision_data: Optional[Dict[str, Tuple[ArrayLike]]] = None,
        wb_xyz_min: Optional[ArrayLike] = None,
        wb_xyz_max: Optional[ArrayLike] = None,
        bg_color: Optional[ArrayLike] = None,
        load_floor: bool = True,
        qdot_max: Optional[ArrayLike] = None,
        tau_max: Optional[ArrayLike] = None,
        real_time: bool = False,
        timestep: float = 1 / 240,
        load_table: bool = False,
        # ✨ NEW:
        base_position: Optional[ArrayLike] = None,
        base_orientation: Optional[ArrayLike] = None,
        table_mesh_path: Optional[str] = None,

    ):
        assert isinstance(urdf, str)
        self.urdf = urdf
        assert control_mode in ["torque", "velocity"]
        self.control_mode = control_mode
        assert isinstance(traj, TaskTrajectory) or traj is None
        self.traj = traj
        with stdout_redirected():
            self.client: pybullet = BulletClient(pybullet.GUI)
        assert isinstance(timestep, float) and timestep > 0
        self.client.setTimeStep(timestep)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ✨ NEW: default base pose if not provided
        if base_position is None:
            base_position = [0.0, 0.0, 0.0]
        if base_orientation is None:
            base_orientation = [0.0, 0.0, 0.0, 1.0]

        self.robot = self.client.loadURDF(
            urdf,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
            flags=self.client.URDF_USE_INERTIA_FROM_FILE
            | self.client.URDF_MERGE_FIXED_LINKS,
        )

        # assert isinstance(load_table, bool)
        # table_z_offset = -0.35
        # if load_table:
        #     self.table = pybullet.loadURDF(
        #         "table/table.urdf",
        #         [0.5, 0, table_z_offset],
        #         globalScaling=0.7,
        #         baseOrientation=pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)),
        #     )

        assert isinstance(load_table, bool)
        self.table = None

        if load_table:
            if table_mesh_path is None:
                raise ValueError(
                    "load_table=True but table_mesh_path is None. "
                    "Provide an STL path via table_mesh_path."
                )

            # We assume the STL is already correctly scaled and oriented.
            mesh_scale = [1.0, 1.0, 1.0]

            col_shape = self.client.createCollisionShape(
                shapeType=self.client.GEOM_MESH,
                fileName=table_mesh_path,
                meshScale=mesh_scale,
            )

            vis_shape = self.client.createVisualShape(
                shapeType=self.client.GEOM_MESH,
                fileName=table_mesh_path,
                meshScale=mesh_scale,
                rgbaColor=[0.8, 0.8, 0.8, 1.0],
            )

            # Origin of the mesh is assumed to already be where you want the table.
            self.table = self.client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
            )



        if load_floor:
            if load_table:
                self.floor = self.client.loadURDF("plane.urdf", [0, 0, table_z_offset])
            else:
                self.floor = self.client.loadURDF("plane.urdf")
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        if bg_color is not None:
            assert len(bg_color) == 3
            self.client.configureDebugVisualizer(rgbBackground=bg_color)
        self.num_joints = self.client.getNumJoints(self.robot)
        # "Unlock" the joints
        self.client.setJointMotorControlArray(
            self.robot,
            list(range(self.num_joints)),
            pybullet.VELOCITY_CONTROL,
            forces=[0.1] * self.num_joints,
        )
        self.target = self.client.loadURDF(
            find_assets_dir() + "point_robot.urdf",
            basePosition=target_pos,
            # baseOrientation=self.client.getQuaternionFromEuler([np.pi, 0, 0]),
            globalScaling=0.2,
        )
        self.target_mass = 1.0
        self.client.changeVisualShape(self.target, -1, rgbaColor=[1, 0, 0, 0.6])
        self.client.changeDynamics(self.target, -1, linearDamping=10, angularDamping=30)

        # If this environment is set up for safe-set-invariance, visualize the safe box
        if xyz_min is not None and xyz_max is not None:
            self.xyz_min = np.asarray(xyz_min)
            self.xyz_max = np.asarray(xyz_max)
            assert self.xyz_min.shape == self.xyz_max.shape == (3,)
            self.box_id = visualize_3D_box(
                [self.xyz_min, self.xyz_max], rgba=(0, 1, 0, 0.3)
            )
        else:
            self.box_id = None
            self.xyz_min = None
            self.xyz_max = None

        # Slight HACK: Duplicate logic for whole-body safe set
        if wb_xyz_min is not None and wb_xyz_max is not None:
            self.wb_xyz_min = np.asarray(wb_xyz_min)
            self.wb_xyz_max = np.asarray(wb_xyz_max)
            assert self.wb_xyz_min.shape == self.wb_xyz_max.shape == (3,)
            self.wb_box_id = visualize_3D_box(
                [self.wb_xyz_min, self.wb_xyz_max], rgba=(0, 1, 0, 0.3)
            )
        else:
            self.wb_box_id = None
            self.wb_xyz_min = None
            self.wb_xyz_max = None

        self.qdot_max = qdot_max
        self.tau_max = tau_max

        self.real_time = real_time
        self.last_time = time.time()

        # Disable collisions for all links and base of the robot with the target
        for i in range(-1, self.num_joints + 1):
            self.client.setCollisionFilterPair(self.robot, self.target, i, -1, 0)
        disable_robot_floor_collisions = False
        disable_target_floor_collisions = True
        if load_floor:
            if disable_robot_floor_collisions:
                for i in range(-1, self.num_joints + 1):
                    self.client.setCollisionFilterPair(self.robot, self.floor, i, -1, 0)
            if disable_target_floor_collisions:
                self.client.setCollisionFilterPair(self.floor, self.target, -1, -1, 0)

        # Disable collisions between the target and the table
        disable_robot_table_collisions = False
        disable_target_table_collisions = True
        if load_table:
            if disable_target_table_collisions:
                self.client.setCollisionFilterPair(self.table, self.target, -1, -1, 0)
            if disable_robot_table_collisions:
                for i in range(-1, self.num_joints + 1):
                    self.client.setCollisionFilterPair(self.robot, self.table, i, -1, 0)
            # Add a stand for the robot
            self.robot_stand_id = visualize_3D_box(
                np.asarray([(-0.2, -0.1, -0.35), (0.1, 0.1, 0)]), rgba=(1, 1, 1, 1)
            )

        # Set initial joint positions if provided
        if q_init is not None:
            assert len(q_init) == self.num_joints
            for joint_index in range(len(q_init)):
                self.client.resetJointState(
                    self.robot, joint_index, q_init[joint_index]
                )

        # Handle collision info
        if collision_data is not None:
            self.collision_positions = collision_data["positions"]
            self.collision_radii = collision_data["radii"]
            self.collision_ids = []
            assert len(self.collision_positions) == len(self.collision_radii)
            for p, r in zip(self.collision_positions, self.collision_radii):
                # coll_id = self.client.createCollisionShape(
                #     self.client.GEOM_SPHERE,
                #     radius=r,
                #     collisionFramePosition=p,
                # )
                # We don't actually want to create a collision shape,
                # because then it would be hard to tell if we are avoiding it
                coll_id = -1
                vis_id = self.client.createVisualShape(
                    self.client.GEOM_SPHERE,
                    radius=r,
                    visualFramePosition=p,
                    rgbaColor=[0, 0, 1, 0.5],
                )
                body_id = self.client.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=coll_id,
                    baseVisualShapeIndex=vis_id,
                )
                self.collision_ids.append(body_id)
        else:
            self.collision_positions = None
            self.collision_radii = None
            self.collision_ids = None

        self.client.setGravity(0, 0, -9.81)
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.t = 0

        # 🔹 Draw world coordinate frame at [0, 0, 0]
        self._draw_world_axes(origin=(0.0, 0.0, 0.0), axis_length=0.3)

    def set_joint_positions(self, q):
        """Directly set robot joint positions in PyBullet (digital twin mode)."""
        q = np.asarray(q).flatten()
        assert q.shape[0] == self.num_joints
        for j in range(self.num_joints):
            self.client.resetJointState(self.robot, j, q[j])


    def get_joint_state(self) -> Array:
        joint_angles = []
        joint_velocities = []
        joint_states = self.client.getJointStates(
            self.robot, list(range(self.num_joints))
        )
        joint_angles = [joint_state[0] for joint_state in joint_states]
        joint_velocities = [joint_state[1] for joint_state in joint_states]
        return np.array([*joint_angles, *joint_velocities])

    def get_desired_ee_state(self) -> Array:
        # Follow a desired task-space trajectory if provided
        if self.traj is not None:
            pos = self.traj.position(self.t)
            rot = self.traj.rotation(self.t).ravel()
            vel = self.traj.velocity(self.t)
            omega = self.traj.omega(self.t)
            # Update the target's visuals to match the desired state
            # HACK: Assume fixed rotation (TODO get a conversion function in here)
            # quat = rotation_to_xyzw(rot.reshape(3, 3))
            quat = np.array([0, 0, 0, 1])
            self.client.resetBasePositionAndOrientation(self.target, pos, quat)
            self.client.resetBaseVelocity(self.target, vel, omega)
            return np.array([*pos, *rot, *vel, *omega])
        # Otherwise, respond to GUI inputs from the user
        pos, orn = self.client.getBasePositionAndOrientation(self.target)
        vel, omega = self.client.getBaseVelocity(self.target)

        # HACK: reset the angular vel of the target to 0
        # Sometimes, the target can start spinning out of control -- this fixes that
        self.client.resetBaseVelocity(self.target, vel, [0, 0, 0])

        # 🔒 Clamp target position into xyz_min / xyz_max if provided
        if self.xyz_min is not None and self.xyz_max is not None:
            pos_clamped = np.clip(pos, self.xyz_min, self.xyz_max)
            if not np.allclose(pos, pos_clamped):
                # snap the visual target back inside the box
                self.client.resetBasePositionAndOrientation(self.target, pos_clamped, orn)
                pos = pos_clamped

        # rot = np.ravel(self.client.getMatrixFromQuaternion(orn))
        # Rotate the target so that the Franka EE naturaly faces downwards instead of upwards
        # Also, flatten the rotation matrix to a 1D array
        rot = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ).ravel()
        # TEMP Ignore angular velocity for now
        omega = np.zeros(3)
        return np.array([*pos, *rot, *vel, *omega])

    def apply_control(self, u: Array) -> None:
        if self.control_mode == "velocity":
            if self.qdot_max is not None:
                u = np.clip(u, -self.qdot_max, self.qdot_max)
            if self.tau_max is not None:
                self.client.setJointMotorControlArray(
                    self.robot,
                    list(range(self.num_joints)),
                    self.client.VELOCITY_CONTROL,
                    targetVelocities=u,
                    forces=self.tau_max,
                )
            else:
                self.client.setJointMotorControlArray(
                    self.robot,
                    list(range(self.num_joints)),
                    self.client.VELOCITY_CONTROL,
                    targetVelocities=u,
                )
        else:  # Torque control
            if self.tau_max is not None:
                u = np.clip(u, -self.tau_max, self.tau_max)
            self.client.setJointMotorControlArray(
                self.robot,
                list(range(self.num_joints)),
                self.client.TORQUE_CONTROL,
                forces=u,
            )
        # Gravity compensation for the target robot so it doesn't fly away
        self.client.applyExternalForce(
            self.target,
            -1,
            [0, 0, 9.81 * self.target_mass],
            self.client.getBasePositionAndOrientation(self.target)[0],
            self.client.WORLD_FRAME,
        )

    def step(self):
        self.client.stepSimulation()
        self.t += self.dt
        if self.real_time:
            time.sleep(max(0, self.dt - (time.time() - self.last_time)))
            self.last_time = time.time()
    
    def step_realworld(self):
        # Keep the red target from falling, even if apply_control() is never called
        self.client.applyExternalForce(
            self.target,
            -1,
            [0, 0, 9.81 * self.target_mass],
            self.client.getBasePositionAndOrientation(self.target)[0],
            self.client.WORLD_FRAME,
        )

        self.client.stepSimulation()
        self.t += self.dt
        if self.real_time:
            time.sleep(max(0, self.dt - (time.time() - self.last_time)))
            self.last_time = time.time()
            
    def _draw_world_axes(self, origin=(0.0, 0.0, 0.0), axis_length=0.3):
        """Draw X (red), Y (green), Z (blue) axes at the world origin."""
        ox, oy, oz = origin

        # X axis: red
        self.client.addUserDebugLine(
            [ox, oy, oz],
            [ox + axis_length, oy, oz],
            [1, 0, 0],  # RGB
            lineWidth=3.0,
            lifeTime=0,  # 0 = persistent
        )

        # Y axis: green
        self.client.addUserDebugLine(
            [ox, oy, oz],
            [ox, oy + axis_length, oz],
            [0, 1, 0],
            lineWidth=3.0,
            lifeTime=0,
        )

        # Z axis: blue
        self.client.addUserDebugLine(
            [ox, oy, oz],
            [ox, oy, oz + axis_length],
            [0, 0, 1],
            lineWidth=3.0,
            lifeTime=0,
        )


class UR10eVelocityControlEnv(ManipulationEnv):
    """Simulation environment for UR10e end-effector pose tracking, with velocity control"""

    def __init__(
        self,
        xyz_min=None,
        xyz_max=None,
        target_pos=(0.5, 0, 0.5),
        q_init=None,
        traj=None,
        collision_data=None,
        wb_xyz_min=None,
        wb_xyz_max=None,
        bg_color=None,
        load_floor=True,
        real_time=False,
        timestep=1 / 240,
        load_table=False,
        # ✨ NEW:
        base_position=None,
        base_orientation=None,
        table_mesh_path=None,
    ):
        # Default joint configuration if none is provided
        # (You can replace this with your favorite "home" pose)
        if q_init is None:
            # 6-DoF UR10e
            q_init = np.zeros(6)

        # ===== Joint velocity limits (rad/s) =====
        # These are reasonable defaults; you should replace with your exact UR10e limits if you have them.
        qdot_max = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]) * 2

        # ===== Max motor forces for PyBullet VELOCITY_CONTROL =====
        # These are *not* exact UR10e torque limits, just safe-ish values for simulation.
        tau_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

        super().__init__(
            urdf="/home/skim3674/Desktop/UR10e_Development2/ur10e_bundle/ur10e.urdf",
            control_mode="velocity",
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            target_pos=target_pos,
            q_init=q_init,
            traj=traj,
            collision_data=collision_data,
            wb_xyz_min=wb_xyz_min,
            wb_xyz_max=wb_xyz_max,
            bg_color=bg_color,
            load_floor=load_floor,
            qdot_max=qdot_max,
            tau_max=tau_max,
            real_time=real_time,
            timestep=timestep,
            load_table=load_table,
             # ✨ NEW:
            base_position=base_position,
            base_orientation=base_orientation,
            table_mesh_path=table_mesh_path,
        )
        