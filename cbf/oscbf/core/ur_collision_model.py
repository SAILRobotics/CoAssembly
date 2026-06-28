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
    (0, 0, 0.05),   # palm / top of body   r=0.055
    (0, 0, 0.11),   # mid body              r=0.050
    (0.05, 0, 0.095),   # finger region         r=0.040
    (-0.05, 0, 0.095),   # finger region         r=0.040
)
gripper_radii = (0.055, 0.06, 0.040, 0.040)

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
