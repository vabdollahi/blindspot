import mujoco
import numpy as np
import pytest

from envs.scene import build_scene
from motion.control import ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK


def _make_control_loop(*, gripper_open: bool = True) -> tuple[ControlLoop, mujoco.MjModel, mujoco.MjData]:
    """Create a scene initialized at the Panda home posture."""
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    finger_position = 0.04 if gripper_open else 0.0
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = finger_position
    mujoco.mj_forward(model, data)
    return ControlLoop(model, data, ik), model, data


def _tcp_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the current TCP position from MuJoCo's forward kinematics."""
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site("tcp").id].copy()


def test_fixed_target_settles_and_gripper_range_opens_fingers() -> None:
    """Hold a target to 2 mm and verify actuator maximum opens the gripper."""
    control, model, data = _make_control_loop(gripper_open=False)
    action = np.array([0.55, 0.0, 0.35, np.pi / 2, 1.0])

    for _ in range(50):
        control.apply(action)

    tcp = _tcp_position(model, data)
    assert np.linalg.norm(tcp - action[:3]) < 0.002
    gripper_id = model.actuator("actuator8").id
    assert data.ctrl[gripper_id] == pytest.approx(
        model.actuator_ctrlrange[gripper_id, 1]
    )
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        qpos_index = model.jnt_qposadr[joint_id]
        assert data.qpos[qpos_index] > 0.035


def test_thirty_centimeter_move_tracks_a_straight_line() -> None:
    """Move 30 cm at the waypoint rate limit and remain within 1 cm of line."""
    control, model, data = _make_control_loop()
    start_action = np.array([0.30, 0.0, 0.35, 0.0, 1.0])
    end_action = np.array([0.60, 0.0, 0.35, 0.0, 1.0])
    for _ in range(50):
        control.apply(start_action)
    start = _tcp_position(model, data)

    positions = []
    for _ in range(70):
        control.apply(end_action)
        positions.append(_tcp_position(model, data))

    path = end_action[:3] - start
    path_length_squared = float(np.dot(path, path))
    deviations = []
    for position in positions:
        fraction = np.clip(np.dot(position - start, path) / path_length_squared, 0.0, 1.0)
        projection = start + fraction * path
        deviations.append(np.linalg.norm(position - projection))

    assert np.linalg.norm(positions[-1] - end_action[:3]) < 0.002
    assert max(deviations) < 0.01