import mujoco
import numpy as np
import pytest

from envs.scene import build_scene
from motion.control import ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK
from skills import ScriptedExpert


def _make_expert_scene(
    task: str,
    *,
    target_object: str | None = None,
    instruction: str | None = None,
) -> tuple[ScriptedExpert, ControlLoop, mujoco.MjModel, mujoco.MjData]:
    """Build an L0 scene with the Panda at home and an open gripper."""
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
    mujoco.mj_forward(model, data)
    control = ControlLoop(model, data, ik)
    expert = ScriptedExpert(task, object_name=target_object)
    expert.reset(instruction)
    return expert, control, model, data


def _observation(
    task: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_pose: np.ndarray | None = None,
    target_object: str | None = None,
    instruction: str | None = None,
) -> dict[str, object]:
    """Build the state and oracle-pose fields planned for PickPlaceEnv."""
    mujoco.mj_forward(model, data)
    arm_positions = np.asarray(
        [
            data.qpos[model.jnt_qposadr[model.joint(name).id]]
            for name in ARM_JOINT_NAMES
        ],
        dtype=np.float64,
    )
    finger_width = sum(
        data.qpos[model.jnt_qposadr[model.joint(name).id]]
        for name in ("finger_joint1", "finger_joint2")
    )
    tcp_site_id = model.site("tcp").id
    tcp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3)
    tcp_yaw = float(np.arctan2(tcp_rotation[1, 0], tcp_rotation[0, 0]))
    state = np.array(
        [
            *arm_positions,
            finger_width,
            *data.site_xpos[tcp_site_id],
            tcp_yaw,
        ],
        dtype=np.float64,
    )
    object_poses = {
        name: {
            "position": data.xpos[model.body(name).id].copy(),
            "quaternion_wxyz": data.xquat[model.body(name).id].copy(),
        }
        for name in ("cube_red", "cube_blue", "cube_green", "bowl")
    }
    observation: dict[str, object] = {
        "task": task,
        "observation.state": state,
        "object_poses": object_poses,
    }
    if target_pose is not None:
        observation["target_pose"] = target_pose
    if target_object is not None:
        observation["target_object"] = target_object
    if instruction is not None:
        observation["instruction"] = instruction
    return observation


def _run_expert(
    task: str,
    *,
    target_object: str | None = None,
    target_pose: np.ndarray | None = None,
    instruction: str | None = None,
    steps: int = 300,
) -> tuple[ScriptedExpert, mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """Run a scripted expert against the L0 MuJoCo scene."""
    expert, control, model, data = _make_expert_scene(
        task,
        target_object=target_object,
        instruction=instruction,
    )
    for _ in range(steps):
        observation = _observation(
            task,
            model,
            data,
            target_pose=target_pose,
            target_object=target_object,
            instruction=instruction,
        )
        action = expert.act(observation)
        assert action.shape == (5,)
        assert np.all(np.isfinite(action))
        control.apply(action)
    return expert, model, data, data.xpos.copy()


def test_t1_expert_outputs_target_action_from_current_observation() -> None:
    """Return a valid action directly toward the oracle target point."""
    expert, _, model, data = _make_expert_scene("T1")
    target = np.array([0.50, -0.10, 0.25])
    observation = _observation("T1", model, data, target_pose=target)

    action = expert.act(observation)

    assert action.shape == (5,)
    np.testing.assert_allclose(action[:3], target)
    assert action[4] == 1.0


def test_expert_recomputes_approach_from_updated_object_pose() -> None:
    """Move the oracle cube and verify the next approach action follows it."""
    expert, _, model, data = _make_expert_scene("T2", target_object="cube_red")
    observation = _observation(
        "T2", model, data, target_pose=np.array([0.49, 0.02, 0.02])
    )

    first_action = expert.act(observation)
    observation["object_poses"]["cube_red"]["position"][0] += 0.04
    second_action = expert.act(observation)

    assert second_action[0] == pytest.approx(first_action[0] + 0.04)
    assert second_action[2] == pytest.approx(first_action[2])


def test_t2_expert_places_cube_in_l0_target_zone() -> None:
    """Use oracle state to close the loop through a complete L0 placement."""
    _, model, data, _ = _run_expert(
        "T2",
        target_object="cube_red",
        target_pose=np.array([0.49, 0.02, 0.02]),
    )

    cube_position = data.xpos[model.body("cube_red").id]
    assert np.linalg.norm(cube_position[:2] - [0.49, 0.02]) <= 0.03
    assert cube_position[2] == pytest.approx(0.02, abs=0.01)


def test_t3_expert_places_cube_in_l0_bowl() -> None:
    """Place an oracle-selected cube inside the L0 bowl."""
    _, model, data, _ = _run_expert("T3", target_object="cube_red")

    cube_position = data.xpos[model.body("cube_red").id]
    bowl_position = data.xpos[model.body("bowl").id]
    assert np.linalg.norm(cube_position[:2] - bowl_position[:2]) < 0.07
    assert cube_position[2] < bowl_position[2] + 0.056


def test_t3_l_expert_selects_named_cube_and_leaves_others_in_place() -> None:
    """Ground the named cube color and leave the other L0 cubes undisturbed."""
    initial_positions = {
        name: np.array(position)
        for name, position in (
            ("cube_red", [0.34, -0.20, 0.02]),
            ("cube_blue", [0.49, -0.20, 0.02]),
            ("cube_green", [0.64, -0.20, 0.02]),
        )
    }
    _, model, data, _ = _run_expert(
        "T3-L",
        instruction="Put the blue cube in the bowl.",
    )

    blue = data.xpos[model.body("cube_blue").id]
    bowl = data.xpos[model.body("bowl").id]
    assert np.linalg.norm(blue[:2] - bowl[:2]) < 0.07
    for name in ("cube_red", "cube_green"):
        position = data.xpos[model.body(name).id]
        assert np.linalg.norm(position[:2] - initial_positions[name][:2]) < 0.02


def test_t4_expert_stacks_red_cube_on_blue_cube() -> None:
    """Place the red cube on top of the oracle-observed blue cube."""
    _, model, data, _ = _run_expert("T4")

    red_position = data.xpos[model.body("cube_red").id]
    blue_position = data.xpos[model.body("cube_blue").id]
    assert np.linalg.norm(red_position[:2] - blue_position[:2]) <= 0.015
    assert red_position[2] == pytest.approx(blue_position[2] + 0.04, abs=0.015)
