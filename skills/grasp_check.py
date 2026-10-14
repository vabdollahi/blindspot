"""Contact-based grasp, lift, and collision checks."""

import mujoco
import numpy as np
from numpy.typing import ArrayLike

FINGER_BODY_NAMES = ("left_finger", "right_finger")
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ROBOT_ROOT_BODY_NAME = "link0"
FULLY_CLOSED_THRESHOLD = 0.002
MINIMUM_LIFT_HEIGHT = 0.08


def is_grasped(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body_name: str,
    fully_closed_threshold: float = FULLY_CLOSED_THRESHOLD,
) -> bool:
    """Return whether both finger bodies touch the target and remain open."""
    if not np.isfinite(fully_closed_threshold) or fully_closed_threshold < 0:
        raise ValueError("Fully-closed threshold must be finite and non-negative")

    target_body_id = model.body(target_body_name).id
    finger_body_ids = {model.body(name).id for name in FINGER_BODY_NAMES}
    finger_openings = [
        data.qpos[model.jnt_qposadr[model.joint(name).id]]
        for name in FINGER_JOINT_NAMES
    ]
    if any(opening <= fully_closed_threshold for opening in finger_openings):
        return False

    touching_finger_ids: set[int] = set()
    for contact in data.contact[: data.ncon]:
        first_body_id = model.geom_bodyid[contact.geom1]
        second_body_id = model.geom_bodyid[contact.geom2]
        if first_body_id == target_body_id and second_body_id in finger_body_ids:
            touching_finger_ids.add(int(second_body_id))
        elif second_body_id == target_body_id and first_body_id in finger_body_ids:
            touching_finger_ids.add(int(first_body_id))
    return touching_finger_ids == finger_body_ids


def is_lifted(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body_name: str,
    initial_position: ArrayLike,
    minimum_height: float = MINIMUM_LIFT_HEIGHT,
) -> bool:
    """Return whether the target body rose by at least the requested height."""
    initial = np.asarray(initial_position, dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial position must be a finite 3-D vector")
    if not np.isfinite(minimum_height) or minimum_height < 0:
        raise ValueError("Minimum lift height must be finite and non-negative")

    target_body_id = model.body(target_body_name).id
    return bool(data.xpos[target_body_id, 2] - initial[2] >= minimum_height)


def count_collisions(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Count robot contacts with non-robot bodies, excluding both fingers."""
    root_body_id = model.body(ROBOT_ROOT_BODY_NAME).id
    robot_body_ids = {
        body_id
        for body_id in range(1, model.nbody)
        if _is_descendant(model, body_id, root_body_id)
    }
    finger_body_ids = {
        body_id
        for finger_name in FINGER_BODY_NAMES
        for body_id in range(1, model.nbody)
        if _is_descendant(model, body_id, model.body(finger_name).id)
    }
    nonfinger_robot_ids = robot_body_ids - finger_body_ids

    collisions = 0
    for contact in data.contact[: data.ncon]:
        first_body_id = int(model.geom_bodyid[contact.geom1])
        second_body_id = int(model.geom_bodyid[contact.geom2])
        first_robot_contact = (
            first_body_id in nonfinger_robot_ids and second_body_id not in robot_body_ids
        )
        second_robot_contact = (
            second_body_id in nonfinger_robot_ids and first_body_id not in robot_body_ids
        )
        if first_robot_contact or second_robot_contact:
            collisions += 1
    return collisions


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    """Return whether a body is the named ancestor or one of its descendants."""
    current_id = body_id
    while current_id != 0:
        if current_id == ancestor_id:
            return True
        current_id = int(model.body_parentid[current_id])
    return False