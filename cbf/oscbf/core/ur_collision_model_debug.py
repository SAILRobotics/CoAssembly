"""
ur_collision_model_debug.py — PyBullet visualizer for the UR10e collision model.

Loads the robot URDF, places all collision spheres at their correct world
positions for a given joint configuration, and draws link coordinate frames.

Usage
-----
    python3 ur_collision_model_debug.py [--urdf path/to/ur10e.urdf] [--random]
"""

import argparse
import numpy as np
import pybullet

from ur_collision_model import ur_collision_data, ur_collision_data_with_gripper
from manipulator_my_own import load_ur10e_for_world

URDF_DEFAULT = "/home/skim3674/Desktop/ProxyGrip/ur10e_bundle/ur10e.urdf"


LINK_COLORS = [
    (1.0, 0.2, 0.2, 0.5),  # link_1  red
    (1.0, 0.6, 0.1, 0.5),  # link_2  orange
    (1.0, 1.0, 0.1, 0.5),  # link_3  yellow
    (0.2, 1.0, 0.2, 0.5),  # link_4  green
    (0.1, 0.6, 1.0, 0.5),  # link_5  blue
    (0.8, 0.2, 1.0, 0.5),  # link_6  purple
]

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_sphere(center, radius, color=(0.2, 0.6, 1.0, 0.4)):
    vis = pybullet.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius, rgbaColor=color)
    pybullet.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                             basePosition=center.tolist())


def draw_frame(T, axis_len=0.12, lw=3, life=0):
    origin = T[:3, 3]
    R = T[:3, :3]
    for axis, color in zip(np.eye(3), ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        end = origin + R @ (axis * axis_len)
        pybullet.addUserDebugLine(origin.tolist(), end.tolist(), color, lw, life)


def draw_link_frames(joint_transforms, visible_links=None, axis_len=0.12):
    for i, T in enumerate(joint_transforms):
        if visible_links is not None and (i + 1) not in visible_links:
            continue
        draw_frame(T, axis_len=axis_len)
        label_pos = (T[:3, 3] + T[:3, :3] @ np.array([0, 0, axis_len * 0.2])).tolist()
        pybullet.addUserDebugText(
            f"link_{i+1}", label_pos, textColorRGB=[1, 1, 1], textSize=1.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf",   default=URDF_DEFAULT)
    parser.add_argument("--random", action="store_true",
                        help="Randomize joint angles instead of zero pose")
    args = parser.parse_args()

    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1.8, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5])

    robot = pybullet.loadURDF(
        args.urdf, useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_MERGE_FIXED_LINKS)
    for link_idx in range(pybullet.getNumJoints(robot)):
        pybullet.changeVisualShape(robot, link_idx, rgbaColor=(0.1, 0.1, 0.1, 0.5))

    manipulator = load_ur10e_for_world(args.urdf)

    np.random.seed(0)
    q = np.random.uniform(-np.pi, np.pi, manipulator.num_joints) if args.random \
        else np.zeros(manipulator.num_joints)

    for i, angle in enumerate(q):
        pybullet.resetJointState(robot, i, angle)
    pybullet.stepSimulation()

    joint_transforms = np.array(manipulator.joint_to_world_transforms(q))

    draw_link_frames(joint_transforms, visible_links=[0, 1, 2, 3, 4, 5, 6])

    ur_collision_data = ur_collision_data_with_gripper 
    for link_idx, (centers, radii) in enumerate(
            zip(ur_collision_data["positions"], ur_collision_data["radii"])):
        T_link = joint_transforms[link_idx]
        for center, radius in zip(centers, radii):
            t_local = np.array(center, dtype=float)
            world_pos = (T_link @ np.append(t_local, 1.0))[:3]
            print(f"  link_{link_idx+1} sphere at world {world_pos.round(3)}  r={radius}")
            draw_sphere(world_pos, radius, color=LINK_COLORS[min(link_idx, len(LINK_COLORS) - 1)])

    print("Collision spheres drawn. Press Enter to exit.")
    input()
    pybullet.disconnect()


if __name__ == "__main__":
    main()
