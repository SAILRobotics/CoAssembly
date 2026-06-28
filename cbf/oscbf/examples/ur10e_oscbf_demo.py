"""UR10e OSCBF velocity-control demo (PyBullet simulation)

Loads the calibrated world→base transform from phase1_results.npz, then runs
an operational-space CBF velocity controller with:
  - Joint limit avoidance
  - Floor (z_min) avoidance
  - Spherical obstacle collision avoidance
  - Self-collision avoidance (via ur_collision_model)

Run from the ProxyGrip root:
    python main_logic/oscbf/examples/ur10e_oscbf_demo.py
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from cbfpy import CBF

from oscbf.core.manipulator_my_own import Manipulator, load_ur10e_for_world
from oscbf.core.oscbf_configs_my_own import OSCBFVelocityConfig
from oscbf.core.controllers_my_own import PoseTaskVelocityController
from oscbf.core.manipulation_env_my_own import UR10eVelocityControlEnv

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]  # ProxyGrip/
_URDF = _ROOT / "ur10e_bundle" / "ur10e.urdf"
_PHASE1 = _ROOT / "main_logic" / "initialization" / "hand_eye_data" / "results" / "phase1_results.npz"


# ---------------------------------------------------------------------------
# CBF config
# ---------------------------------------------------------------------------
@jax.tree_util.register_static
class UR10eOSCBFConfig(OSCBFVelocityConfig):
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
        q = z[: self.num_joints]

        robot_coll = self.robot.link_collision_data(q)
        robot_pos = robot_coll[:, :3]
        robot_rad = robot_coll[:, 3]

        # Obstacle collision avoidance
        center_deltas = (
            robot_pos[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_rad[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Floor avoidance
        h_floor = robot_pos[:, 2] - robot_rad - self.z_min

        # Joint limits
        h_joint_low = q - self.joint_pos_min
        h_joint_high = self.joint_pos_max - q

        return jnp.concatenate([h_collision, h_floor, h_joint_low, h_joint_high])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load calibrated base pose ----------------------------------------
    if not _PHASE1.exists():
        print(f"[ur10e_oscbf_demo] WARNING: {_PHASE1} not found; using identity base.")
        base_pos = np.zeros(3)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        data = np.load(_PHASE1)
        T = data["T_world_base"]
        base_pos = T[:3, 3]
        base_quat = R.from_matrix(T[:3, :3]).as_quat()  # xyzw

    print(f"[ur10e_oscbf_demo] base position : {base_pos}")
    print(f"[ur10e_oscbf_demo] base quat xyzw: {base_quat}")

    # ---- Robot model -------------------------------------------------------
    robot = load_ur10e_for_world(
        urdf_filename=str(_URDF),
        base_position=base_pos,
        base_orientation=base_quat,
    )

    # ---- Joint limits (radians) -------------------------------------------
    joint_pos_min = np.deg2rad([-370.0, -180.0, -160.0, -250.0, -190.0, -360.0])
    joint_pos_max = np.deg2rad([ 370.0,    0.0,  160.0,   70.0,  190.0,  360.0])

    # ---- Obstacle spheres -------------------------------------------------
    collision_positions = np.array([
        [base_pos[0] + 0.4, base_pos[1], base_pos[2] + 0.5],
    ])
    collision_radii = np.array([0.15])
    z_min = 0.01  # world floor (slight offset)

    collision_data = {
        "positions": collision_positions,
        "radii": collision_radii,
    }

    # ---- CBF ---------------------------------------------------------------
    cbf_config = UR10eOSCBFConfig(
        robot=robot,
        z_min=z_min,
        collision_positions=collision_positions,
        collision_radii=collision_radii,
        joint_pos_min=joint_pos_min,
        joint_pos_max=joint_pos_max,
    )
    cbf = CBF.from_config(cbf_config)

    # ---- OSC velocity controller ------------------------------------------
    kp_pos, kp_rot, kp_joint = 50.0, 20.0, 10.0
    controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos] * 3 + [kp_rot] * 3),
        kp_joint=kp_joint,
        qdot_min=None,
        qdot_max=None,
    )

    # ---- JIT ---------------------------------------------------------------
    @jax.jit
    def step(q, z_ee_des):
        M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]
        u_nom = controller(
            q, pos, rot, des_pos, des_rot, des_vel, des_omega, q, J, M_inv
        )
        return cbf.safety_filter(q, u_nom)

    # Warm-up JIT
    _ = step(np.zeros(6), np.zeros(18))
    print("[ur10e_oscbf_demo] JIT compiled. Starting simulation.")

    # ---- PyBullet environment ---------------------------------------------
    q_home = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    env = UR10eVelocityControlEnv(
        xyz_min=None,
        xyz_max=None,
        target_pos=(base_pos + np.array([0.5, 0.0, 0.5])).tolist(),
        q_init=q_home,
        collision_data=collision_data,
        load_floor=True,
        real_time=True,
        timestep=1 / 240,
        base_position=base_pos.tolist(),
        base_orientation=base_quat.tolist(),
    )

    CONTROL_DT = 1 / 240
    qdot_limits = np.deg2rad([120, 120, 180, 180, 180, 180])

    try:
        while True:
            q = env.get_joint_state()[:6]
            z_ee_des = env.get_desired_ee_state()
            qdot_cmd = np.array(step(q, z_ee_des))
            qdot_cmd = np.clip(qdot_cmd, -qdot_limits, qdot_limits)

            # Deadband near target
            M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
            if np.linalg.norm(z_ee_des[:3] - ee_tmat[:3, 3]) < 0.01:
                qdot_cmd[:] = 0.0

            env.apply_control(qdot_cmd)
            env.step()
    finally:
        env.close()


if __name__ == "__main__":
    main()
