"""Testing the performance of OSCBF in highly-constrained settings

We consider a cluttered tabletop environment with many randomized obstacles,
each represented as a sphere. We then enforce collision avoidance with 
all of the obstacles, and all of the collision bodies on the robot

There are likely "smarter" ways to filter out the collision pairs that are
least likely to cause a collision, but for now, this test just tries to see
how much we can scale up the collision avoidance while retaining real-time
performance.
"""
import sys 
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator_my_own import Manipulator
from oscbf.core.manipulation_env_my_own import UR10eVelocityControlEnv
from oscbf.core.oscbf_configs_my_own import OSCBFVelocityConfig
from oscbf.core.controllers_my_own import PoseTaskVelocityController
from oscbf.core.manipulator_my_own import load_ur10e, load_ur10e_for_world
from scipy.spatial.transform import Rotation as R

np.random.seed(0)


@jax.tree_util.register_static
class CollisionsVelocityConfig(OSCBFVelocityConfig):

    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)

        # -------- Joint limits from UR10e YAML (degrees → radians) --------
        # Order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        joint_min_deg = np.array([-370.0, -180.0, -160.0, -250.0, -190.0, -360.0])
        joint_max_deg = np.array([-210.0,  -80.0,  -60.0,    0.0,   90.0,  360.0])

        joint_min_rad = np.deg2rad(joint_min_deg)
        joint_max_rad = np.deg2rad(joint_max_deg)

        assert joint_min_rad.shape[0] == robot.num_joints
        assert joint_max_rad.shape[0] == robot.num_joints

        # Small safety margin so we don't command exactly at the hard limits
        margin = np.deg2rad(5.0)

        # q_min <= q <= q_max is the *hard* safe region
        self.q_min = jnp.array(joint_min_rad + margin)
        self.q_max = jnp.array(joint_max_rad - margin)

        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        # ---- NEW: joint angle limits ----
        # lower bound: q should stay above q_min
        h_joint_low = q - self.q_min          # > 0 => above lower limit + margin
        # upper bound: q should stay below q_max
        h_joint_high = self.q_max - q         # > 0 => below upper limit - margin

        return jnp.concatenate([h_collision, h_table, h_joint_low, h_joint_high])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    # # Set nullspace desired joint position
    # des_q = jnp.deg2rad(
    #     jnp.array([
    #         -314.27,
    #         -138.83,
    #         -115.65,
    #         -181.37,
    #         -87.52,
    #         0.0,
    #     ])
    # )
    des_q = q
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


def main(control_method="torque", num_bodies=25):
    npz_path = "/home/skim3674/Desktop/IntentPredictionProject/storage/data_capture/extrinsic_results/T_robot_in_world.npz"
    data = np.load(npz_path)
    if "T" not in data:
        raise KeyError(f"NPZ does not contain key 'T': {npz_path}")
    T_world_robot = data["T"]
    if T_world_robot.shape != (4, 4):
        raise ValueError(f"Expected T to be 4x4, got {T_world_robot.shape}")
    
    # Optionally apply the SAME yaw_deg rotation used in ROS (e.g., 180°)
    yaw_deg = 180
    yaw = np.deg2rad(yaw_deg)
    Rz = R.from_euler("z", yaw).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = Rz
    # Rotate in WORLD frame:
    T_world_robot = T_world_robot @ T_rot

    p_world_robot = T_world_robot[:3, 3]
    R_world_robot = T_world_robot[:3, :3]
    q_world_robot_xyzw = R.from_matrix(R_world_robot).as_quat()


    robot = load_ur10e_for_world("/home/skim3674/Desktop/IntentPredictionProject/ur10e_bundle/ur10e.urdf",
        base_position=p_world_robot,
        base_orientation=q_world_robot_xyzw,
    )

    #------------------------------------------------------
    joint_vel_deg = np.array([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])/5.0
    robot.joint_max_velocities = np.deg2rad(joint_vel_deg)

    #------------------------------------------------------
    z_min = 0.0
    max_num_bodies = 10
    
    # Sample a lot of collision bodies
    all_collision_pos = np.random.uniform(
        low=[-0.5, -0.4, 0.1], high=[0.5, 0.4, 0.15], size=(max_num_bodies, 3)
    )
    all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
    # Only use a subset of them based on the desired quantity
    collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
    collision_radii = all_collision_radii[:num_bodies]
    collision_data = {"positions": collision_pos, "radii": collision_radii}

    velocity_config = CollisionsVelocityConfig(
        robot, z_min, collision_pos, collision_radii
    )
    velocity_cbf = CBF.from_config(velocity_config)

    timestep = 1 / 240  #  1 / 1000
    bg_color = (1, 1, 1)
    
    q_init_ur10e = np.deg2rad(
        np.array([
            -314.27,
            -138.83,
            -115.65,
            -181.37,
            -87.52,
            0.0,
        ], dtype=float)
    )

    # env = UR10eVelocityControlEnv(
    #     real_time=True,
    #     bg_color=bg_color,
    #     load_floor=False,
    #     timestep=timestep,
    #     collision_data=collision_data,
    #     load_table=True,
    #     q_init=q_init_ur10e,
    # )

    env = UR10eVelocityControlEnv(
        real_time=True,
        bg_color=bg_color,
        load_floor=False,
        timestep=timestep,
        collision_data=collision_data,
        load_table=True,
        q_init=q_init_ur10e,
        base_position=p_world_robot,
        base_orientation=q_world_robot_xyzw,
        table_mesh_path="/home/skim3674/Desktop/gits/oscbf/Table.stl",
    )


    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.40,
        cameraYaw=104.40,
        cameraPitch=-37,
        cameraTargetPosition=(0.20, 0.07, -0.09),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    
    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )


    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )

 
    compute_control = compute_velocity_control_jit
    
    while True:
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        tau = compute_control(q_qdot, z_zdot_ee_des)
        env.apply_control(tau)
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run highly-constrained collision avoidance experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="velocity",
        help="Control method to use (default: velocity)",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=25,
        help="Number of collision bodies to simulate (default: 25)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method, num_bodies=args.num_bodies)
