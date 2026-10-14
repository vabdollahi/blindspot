import mujoco
import numpy as np

from common.seeding import named_generators
from envs.scene import build_scene
from motion.control import ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK
from skills.grasp_check import count_collisions, is_grasped, is_lifted


def _reset_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: ControlLoop,
    ik: MinkIK,
    cube_position: np.ndarray,
    cube_yaw: float,
) -> None:
    """Reset the scene with one randomized cube and the Panda at home."""
    mujoco.mj_resetData(model, data)
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]

    cube_joint_id = model.joint("cube_red_free").id
    cube_qpos_address = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_address : cube_qpos_address + 3] = cube_position
    data.qpos[cube_qpos_address + 3 : cube_qpos_address + 7] = [
        np.cos(cube_yaw / 2),
        0.0,
        0.0,
        np.sin(cube_yaw / 2),
    ]

    for name, position in (
        ("cube_blue", (0.25, 0.45, 0.02)),
        ("cube_green", (0.80, 0.45, 0.02)),
    ):
        joint_id = model.joint(f"{name}_free").id
        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address : qpos_address + 3] = position
        data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    control.reset()


def test_grasp_lift_and_hold_on_twenty_random_cube_poses() -> None:
    """Grasp, lift, and hold a cube from 20 seeded poses with under 5 mm slip."""
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    control = ControlLoop(model, data, ik)
    random = named_generators(2026, ("grasp_pose",))["grasp_pose"]
    cube_body_id = model.body("cube_red").id

    for position, yaw in zip(
        random.uniform([0.32, -0.28], [0.54, -0.06], size=(20, 2)),
        random.uniform(-np.pi / 2, np.pi / 2, size=20),
        strict=True,
    ):
        cube_position = np.array([position[0], position[1], 0.02])
        _reset_trial(model, data, control, ik, cube_position, float(yaw))
        initial_position = data.xpos[cube_body_id].copy()
        assert not is_grasped(model, data, "cube_red")

        for _ in range(60):
            control.apply([*cube_position[:2], 0.12, yaw, 1.0])
        for _ in range(20):
            control.apply([*cube_position[:2], 0.02, yaw, 1.0])
        for _ in range(30):
            control.apply([*cube_position[:2], 0.02, yaw, 0.0])

        assert is_grasped(model, data, "cube_red")
        assert count_collisions(model, data) == 0
        assert not is_lifted(model, data, "cube_red", initial_position)

        for _ in range(30):
            control.apply([*cube_position[:2], 0.12, yaw, 0.0])
        assert is_lifted(model, data, "cube_red", initial_position)
        lifted_position = data.xpos[cube_body_id].copy()

        maximum_slip = 0.0
        for _ in range(60):
            control.apply([*cube_position[:2], 0.12, yaw, 0.0])
            slip = np.linalg.norm(data.xpos[cube_body_id] - lifted_position)
            maximum_slip = max(maximum_slip, slip)

        assert maximum_slip < 0.005


def test_collision_counter_counts_nonfinger_robot_contacts() -> None:
    """Count a forced robot-link contact with a cube."""
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    control = ControlLoop(model, data, ik)
    _reset_trial(model, data, control, ik, np.array([0.0, 0.0, 0.12]), 0.0)

    assert count_collisions(model, data) > 0