#!/usr/bin/env python3
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- JAX / OSCBF imports (your stuff) ---
import jax
import jax.numpy as jnp
from cbfpy import CBF
from jax.typing import ArrayLike

from oscbf.core.manipulator_my_own import Manipulator, load_ur10e_for_world
from oscbf.core.oscbf_configs_my_own import OSCBFVelocityConfig
from oscbf.core.controllers_my_own import PoseTaskVelocityController
from oscbf.core.manipulation_env_my_own import UR10eVelocityControlEnv

# --- RTDE imports ---
import rtde_control
import rtde_receive


# ============================================================
# 1) CBF config for collisions + table + joint limits
# ============================================================

@jax.tree_util.register_static
class CollisionsVelocityConfig(OSCBFVelocityConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        joint_pos_min: ArrayLike,
        joint_pos_max: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        self.joint_pos_min = jnp.asarray(joint_pos_min)
        self.joint_pos_max = jnp.asarray(joint_pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # z = q (velocity CBF)
        q = z[: self.num_joints]

        # --- Collision avoidance with spherical obstacles ---
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]

        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)

        # distance between centers - sum of radii
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # --- Table avoidance (z >= z_min + radius) ---
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        # --- Joint angle limits ---
        # h >= 0 → inside bounds
        h_joint_low = q - self.joint_pos_min   # q >= q_min
        h_joint_high = self.joint_pos_max - q  # q <= q_max

        return jnp.concatenate([h_collision, h_table, h_joint_low, h_joint_high])

    def alpha(self, h):
        # class-K function
        return 10.0 * h

    def alpha_2(self, h_2):
        # not used in first-order CBF, but required by interface
        return 10.0 * h_2


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

    # Non-redundant case: nullspace is basically disabled by des_q = q
    des_q = q

    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


# ============================================================
# 2) Main real-robot loop
# ============================================================

def main():
    # -----------------------------
    # Config
    # -----------------------------
    robot_ip = "192.168.50.70"  # <- your robot IP

    # Control loop period (matches servoJ dt)
    CONTROL_DT = 0.016           # 125 Hz

    # servoJ tuning parameters
    SERVO_SPEED     = 1.0    # or 2–3, stay within UR specs
    SERVO_ACCEL     = 1.0    # similarly 2–3
    SERVO_LOOKAHEAD = 0.08   # slightly shorter
    SERVO_GAIN      = 200    # 150–300 is typical

    # -----------------------------
    # Load extrinsic: T_world_robot
    # -----------------------------
    npz_path = "/home/skim3674/Desktop/IntentPredictionProject/storage/data_capture/extrinsic_results/T_robot_in_world.npz"
    data = np.load(npz_path)
    if "T" not in data:
        raise KeyError(f"NPZ does not contain key 'T': {npz_path}")
    T_world_robot = data["T"]
    if T_world_robot.shape != (4, 4):
        raise ValueError(f"Expected 4x4 T, got {T_world_robot.shape}")

    # Optional yaw correction (same as in your sim code)
    yaw_deg = 180
    yaw = np.deg2rad(yaw_deg)
    Rz = R.from_euler("z", yaw).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = Rz
    T_world_robot = T_world_robot @ T_rot

    p_world_robot = T_world_robot[:3, 3]
    R_world_robot = T_world_robot[:3, :3]
    q_world_robot_xyzw = R.from_matrix(R_world_robot).as_quat()

    # -----------------------------
    # Robot model for OSCBF
    # -----------------------------
    robot = load_ur10e_for_world(
        "/home/skim3674/Desktop/IntentPredictionProject/ur10e_bundle/ur10e.urdf",
        base_position=p_world_robot,
        base_orientation=q_world_robot_xyzw,
    )

    # Joint position limits (from your YAML, in radians):
    joint_pos_min = np.deg2rad([
        -370.0,  # shoulder_pan
        -180.0,  # shoulder_lift
        -160.0,  # elbow
        -250.0,  # wrist_1
        -190.0,  # wrist_2
        -360.0,  # wrist_3
    ])
    joint_pos_max = np.deg2rad([
        -210.0,  # shoulder_pan
        -80.0,   # shoulder_lift
        -60.0,   # elbow (artificially tightened)
        0.0,     # wrist_1
        90.0,    # wrist_2
        360.0,   # wrist_3
    ])

    # -----------------------------
    # Collision spheres (same as before)
    # -----------------------------
    max_num_bodies = 5
    all_collision_pos = np.random.uniform(
        low=[-0.5, -0.4, 0.1], high=[0.5, 0.4, 0.3], size=(max_num_bodies, 3)
    )
    
    all_collision_radii = np.random.uniform(
        low=0.10, high=0.20, size=(max_num_bodies,)
    )
    collision_pos = all_collision_pos[:max_num_bodies]
    collision_radii = all_collision_radii[:max_num_bodies]
    collision_data = {"positions": collision_pos, "radii": collision_radii}
    z_min = 0.0  # table height limit

    # -----------------------------
    # Velocity CBF config
    # -----------------------------
    velocity_config = CollisionsVelocityConfig(
        robot=robot,
        z_min=z_min,
        collision_positions=collision_pos,
        collision_radii=collision_radii,
        joint_pos_min=joint_pos_min,
        joint_pos_max=joint_pos_max,
    )
    velocity_cbf = CBF.from_config(velocity_config)

    # -----------------------------
    # Digital twin env (PyBullet)
    # -----------------------------
    env = UR10eVelocityControlEnv(
        xyz_min=[-0.5, -0.50, 0.22],
        xyz_max=[0.3,  0.25, 0.50],
        real_time=False,
        bg_color=(1, 1, 1),
        load_floor=False,
        timestep=1 / 240,
        collision_data=collision_data,
        load_table=True,
        q_init=np.zeros(6),  # will be overwritten by real states anyway
        base_position=p_world_robot,
        base_orientation=q_world_robot_xyzw,
        table_mesh_path="/home/skim3674/Desktop/gits/oscbf/Table.stl",
    )

    # Camera for nicer view
    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.40,
        cameraYaw=104.40,
        cameraPitch=-37,
        cameraTargetPosition=(0.20, 0.07, -0.09),
    )

    # -----------------------------
    # OSC velocity controller
    # -----------------------------
    kp_pos = 50.0
    kp_rot = 20.0
    kp_joint = 10.0

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        qdot_min=None,  # CBFConfig already limits velocities
        qdot_max=None,
    )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )

    # -----------------------------
    # RTDE interfaces
    # -----------------------------
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    print("Connected to UR10e at", robot_ip)
    print("Starting OSCBF velocity control loop...")

    try:
            print("Connected to UR10e at", robot_ip)
            print("Starting OSCBF + servoJ velocity control loop...")

            while True:
                # 0) Let UR control timing for nice, steady dt
                t_start = rtde_c.initPeriod()

                # 1) Get real joint state
                q_real = np.array(rtde_r.getActualQ(), dtype=float)
                # qdot_real = np.array(rtde_r.getActualQd(), dtype=float)  # not needed for velocity CBF

                # 2) Update digital twin joint positions in PyBullet
                env.set_joint_positions(q_real)

                # 3) Get desired EE state from env (GUI target or trajectory)
                z_ee_des = env.get_desired_ee_state()

                # 4) Compute safe joint velocities via OSC + CBF
                z = q_real  # state for velocity CBF is just q
                qdot_cmd = np.array(compute_velocity_control_jit(z, z_ee_des))

                # Optional clamp as extra safety (should match hardware limits)
                joint_vel_limits = np.deg2rad([120, 120, 180, 180, 180, 180]) * 2
                qdot_cmd = np.clip(qdot_cmd, -joint_vel_limits, joint_vel_limits)

                # 4.1) Deadband on EE position error to avoid chattering near the goal
                M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q_real)
                pos = ee_tmat[:3, 3]
                des_pos = z_ee_des[:3]
                pos_err = des_pos - pos

                # If we're within 1 cm of the target, stop moving
                if np.linalg.norm(pos_err) < 0.01:
                    qdot_cmd[:] = 0.0

                # 4.2) Integrate velocity command to a joint position target
                #      This is the "desired q" we feed into servoJ
                q_target = q_real + qdot_cmd * CONTROL_DT

                # 5) Send servoJ command to UR10e
                rtde_c.servoJ(
                    q_target.tolist(),
                    SERVO_SPEED,
                    SERVO_ACCEL,
                    CONTROL_DT,
                    SERVO_LOOKAHEAD,
                    SERVO_GAIN,
                )

                # 6) Step PyBullet just for visual update (digital twin)
                env.step_realworld()

                # 7) Block until next control period (UR's own timing)
                rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        print("Stopping control (KeyboardInterrupt)...")
    finally:
        # Safe stop
        rtde_c.servoStop()
        rtde_c.stopScript()
        print("RTDE script stopped, exiting.")


if __name__ == "__main__":
    main()
