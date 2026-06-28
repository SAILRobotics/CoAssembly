"""Script to visualize the hand-designed collision model of the UR10e in Pybullet"""

import pybullet
import numpy as np

import oscbf.core.ur_collision_model as colmodel
from oscbf.core.manipulator import create_transform_numpy, load_ur10e
from oscbf.utils.visualization import visualize_3D_sphere


np.random.seed(0)

# <<< CHANGE THIS PATH TO YOUR UR10e URDF >>>
URDF = "/home/skim3674/Desktop/IntentPredictionProject/ur10e_bundle/ur10e.urdf"

UR10E_INIT_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
RANDOMIZE = True


def visualize_collision_model(
    positions,
    radii,
    pairs=None,
    base_position=None,
    base_radius=None,
    base_sc_idxs=None,
):
    """
    Visualize collision spheres:
    - positions: tuple of length num_links; positions[i] is a list/tuple of (x,y,z) in link i frame
    - radii:    same structure as positions
    - pairs:    optional list of (i, j) global sphere indices to draw red lines between
    - base_position, base_radius: optional base sphere
    - base_sc_idxs: global sphere indices to draw lines from base sphere to
    """
    pybullet.connect(pybullet.GUI)
    robot = pybullet.loadURDF(
        URDF,
        useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_MERGE_FIXED_LINKS,
    )
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    manipulator = load_ur10e(URDF)
    # Make all links semi-transparent so spheres are visible
    for link_idx in range(manipulator.num_joints):
        pybullet.changeVisualShape(robot, link_idx, rgbaColor=(0, 0, 0, 0.5))
    pybullet.changeVisualShape(robot, -1, rgbaColor=(0, 0, 0, 0.5))  # base

    # Choose joint configuration
    if RANDOMIZE:
        q = np.random.rand(manipulator.num_joints)
    else:
        q = UR10E_INIT_QPOS

    for i in range(manipulator.num_joints):
        pybullet.resetJointState(robot, i, float(q[i]))
    pybullet.stepSimulation()

    joint_transforms = manipulator.joint_to_world_transforms(q)

    sphere_ids = []
    sphere_positions = []

    # Determine the world-frame positions of the collision geometry
    for i in range(manipulator.num_joints):
        parent_to_world_tf = joint_transforms[i]
        num_collision_spheres = len(positions[i])
        for j in range(num_collision_spheres):
            # positions[i][j] is (x,y,z) in link i frame
            collision_to_parent_tf = create_transform_numpy(
                np.eye(3),
                np.array(positions[i][j])
            )
            collision_to_world_tf = parent_to_world_tf @ collision_to_parent_tf
            world_pos = collision_to_world_tf[:3, 3]
            sphere_ids.append(
                visualize_3D_sphere(world_pos, radii[i][j])
            )
            sphere_positions.append(world_pos)

    # Draw red lines between specific self-collision pairs (if provided)
    if pairs is not None:
        for pair in pairs:
            i, j = pair
            rgb = (1, 0, 0)
            pybullet.addUserDebugLine(
                sphere_positions[i].tolist(),
                sphere_positions[j].tolist(),
                rgb,
            )

    # Draw base self-collision sphere (if provided)
    if base_position is not None and base_radius is not None:
        visualize_3D_sphere(base_position, base_radius)

    # Draw lines from base sphere to dangerous arm spheres (if provided)
    if base_sc_idxs is not None and base_position is not None:
        for idx in base_sc_idxs:
            rgb = (1, 0, 0)
            pybullet.addUserDebugLine(
                list(base_position),
                sphere_positions[idx].tolist(),
                rgb,
            )

    input("Press Enter to exit")


def main(self_collision=True):
    """
    If self_collision=True:
        visualize self-collision spheres + base sphere + pair lines.
    Else:
        visualize the full link collision model (all spheres).
    """
    if self_collision:
        visualize_collision_model(
            colmodel.positions_list_sc,
            colmodel.radii_list_sc,
            colmodel.pairs_sc,
            colmodel.base_position,
            colmodel.base_radius,
            colmodel.base_sc_idxs,
        )
    else:
        visualize_collision_model(
            colmodel.positions_list,
            colmodel.radii_list,
        )


if __name__ == "__main__":
    main(self_collision=True)
