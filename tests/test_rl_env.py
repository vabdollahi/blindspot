from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import rl_env
from envs.pick_place_env import PickPlaceEnv
from envs.rl_env import (
    CARRY_CLEARANCE,
    OBSERVATION_DIM,
    POSITION_DELTA_LIMIT,
    RL_TASKS,
    SUCCESS_BONUS,
    YAW_DELTA_LIMIT,
    DeltaCommandTracker,
    PickPlaceRLEnv,
    flatten_observation,
    transport_distance,
    wrap_to_quarter_turn,
)
from skills import ScriptedExpert

_OPEN_WIDTH = 0.08
_ZERO_ACTION = np.zeros(5, dtype=np.float32)


def _dict_observation(
    *,
    tcp: tuple[float, float, float],
    destination: tuple[float, float, float],
    object_position: tuple[float, float, float] | None = None,
    object_yaw: float = 0.0,
    tcp_yaw: float = 0.0,
    gripper_width: float = _OPEN_WIDTH,
) -> dict[str, Any]:
    """Fabricate a state-mode dict observation; no object means a T1 reach target."""
    state = np.zeros(12, dtype=np.float32)
    state[7] = gripper_width
    state[8:11] = tcp
    state[11] = tcp_yaw
    observation: dict[str, Any] = {
        "observation.state": state,
        "target_pose": np.array([*destination, 0.0], dtype=np.float32),
        "target_object": "",
        "object_poses": {},
    }
    if object_position is not None:
        observation["target_object"] = "cube_red"
        observation["object_poses"]["cube_red"] = {
            "position": np.array(object_position, dtype=np.float32),
            "quaternion_wxyz": np.array(
                [np.cos(object_yaw / 2), 0.0, 0.0, np.sin(object_yaw / 2)],
                dtype=np.float32,
            ),
        }
    return observation


def _replay_expert(
    environment: PickPlaceRLEnv, task: str
) -> tuple[bool, float, set[str], dict[str, float]]:
    """Replay the scripted expert through the delta action space for one episode."""
    expert = ScriptedExpert(task)
    expert.reset()
    dict_observation = environment.unwrapped._get_observation()
    total_reward = 0.0
    stages: set[str] = set()
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = environment.command_tracker.delta_from_absolute(
            expert.act(dict_observation)
        )
        _, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        stages.add(info["rl_stage"])
        dict_observation = environment.unwrapped._get_observation()
    return terminated, total_reward, stages, info["rl_episode"]


def test_rejects_language_conditioned_tasks() -> None:
    """Reject T3-L and T5, which need language conditioning, per roadmap 1b."""
    for task in ("T3-L", "T5"):
        with pytest.raises(ValueError, match="non-language"):
            PickPlaceRLEnv(task)


@pytest.mark.parametrize("task", RL_TASKS)
def test_rl_env_passes_gymnasium_checker(task: str) -> None:
    """Satisfy Gymnasium's reset, step, space, and determinism checks."""
    environment = PickPlaceRLEnv(task, "L0")
    check_env(environment, skip_render_check=True)
    environment.close()


@pytest.mark.parametrize("task", RL_TASKS)
def test_observation_is_a_flat_finite_float32_vector(task: str) -> None:
    """Return a fixed-size Box observation suitable for VecNormalize."""
    environment = PickPlaceRLEnv(task, "L1")
    observation, _ = environment.reset(seed=2026)
    assert observation.shape == (OBSERVATION_DIM,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    environment.close()


def test_action_space_is_symmetric_and_zero_centered() -> None:
    """Confirm the policy-facing action space is the normalized [-1, 1] delta box."""
    environment = PickPlaceRLEnv("T2", "L0")
    np.testing.assert_array_equal(environment.action_space.low, -1.0)
    np.testing.assert_array_equal(environment.action_space.high, 1.0)
    environment.close()


def test_command_tracker_moves_by_bounded_deltas_and_binarizes_the_gripper() -> None:
    """Zero holds the target, one unit moves one rate-limit step, and g is binary."""
    observation = _dict_observation(tcp=(0.45, 0.0, 0.20), destination=(0.5, 0, 0.3))
    tracker = DeltaCommandTracker()
    tracker.reset(observation)

    held = tracker.command(np.array([0.0, 0.0, 0.0, 0.0, -0.3]))
    np.testing.assert_allclose(held[:4], [0.45, 0.0, 0.20, 0.0], atol=1e-6)
    assert held[4] == 0.0

    moved = tracker.command(np.array([1.0, 0.0, 0.0, -1.0, 0.2]))
    np.testing.assert_allclose(
        moved[:4], [0.45 + POSITION_DELTA_LIMIT, 0.0, 0.20, -YAW_DELTA_LIMIT], atol=1e-6
    )
    assert moved[4] == 1.0

    start = tracker.target_position
    tracker.command(np.array([1.0, 1.0, 1.0, 0.0, 1.0]))
    assert np.linalg.norm(tracker.target_position - start) == pytest.approx(
        POSITION_DELTA_LIMIT
    )


def test_command_tracker_clips_targets_to_the_workspace() -> None:
    """Never command a target outside the §3 workspace box."""
    observation = _dict_observation(tcp=(0.70, 0.30, 0.02), destination=(0.5, 0, 0.3))
    tracker = DeltaCommandTracker()
    tracker.reset(observation)
    command = tracker.command(np.array([1.0, 1.0, -1.0, 0.0, 1.0]))
    np.testing.assert_allclose(command[:3], [0.70, 0.30, 0.02], atol=1e-9)


def test_delta_from_absolute_reproduces_an_absolute_commands_next_target() -> None:
    """Replaying an absolute command as a delta lands on the same rate-limited target."""
    observation = _dict_observation(tcp=(0.45, 0.0, 0.20), destination=(0.5, 0, 0.3))
    replayed = DeltaCommandTracker()
    replayed.reset(observation)
    for absolute in (
        np.array([0.46, 0.005, 0.195, 0.02, 1.0]),
        np.array([0.60, -0.20, 0.05, 1.2, 0.0]),
    ):
        delta = replayed.delta_from_absolute(absolute)
        expected_position, expected_yaw = rl_env.next_tcp_target(
            replayed.target_position, replayed.target_yaw, absolute
        )
        command = replayed.command(delta)
        np.testing.assert_allclose(
            replayed.target_position, expected_position, atol=1e-9
        )
        assert replayed.target_yaw == pytest.approx(expected_yaw)
        assert command[4] == absolute[4]


def test_command_tracker_mirrors_the_live_controller_target() -> None:
    """The tracker's integrated target must match ControlLoop's under random actions."""
    environment = PickPlaceRLEnv("T2", "L1")
    environment.reset(seed=3)
    control = environment.unwrapped.control
    random = np.random.default_rng(0)
    for _ in range(60):
        environment.step(random.uniform(-1.0, 1.0, size=5).astype(np.float32))
        np.testing.assert_allclose(
            environment.command_tracker.target_position,
            control.target_position,
            atol=1e-6,
        )
        assert environment.command_tracker.target_yaw == pytest.approx(
            control.target_yaw, abs=1e-6
        )
        assert environment.command_tracker.gripper_command == pytest.approx(
            control.gripper_open_fraction
        )
    environment.close()


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi / 3, np.pi / 3 - np.pi / 2), (-0.3, -0.3)],
)
def test_wrap_to_quarter_turn_known_answers(angle: float, expected: float) -> None:
    """Wrap yaw errors into the 90°-symmetric grasp range [-π/4, π/4)."""
    assert wrap_to_quarter_turn(angle) == pytest.approx(expected)


def test_observation_is_invariant_to_quarter_turns_of_the_cube() -> None:
    """A cube yawed by 90° is the same grasp problem, so it must look identical."""

    def observe(object_yaw: float) -> np.ndarray:
        observation = _dict_observation(
            tcp=(0.45, 0.0, 0.20),
            destination=(0.5, 0.05, 0.02),
            object_position=(0.40, -0.10, 0.02),
            object_yaw=object_yaw,
        )
        return flatten_observation(
            observation,
            target_position=(0.45, 0.0, 0.20),
            target_yaw=0.0,
            gripper_command=1.0,
            previous_object_position=(0.40, -0.10, 0.02),
        )

    np.testing.assert_allclose(observe(0.3), observe(0.3 + np.pi / 2), atol=1e-5)
    np.testing.assert_allclose(observe(0.3), observe(0.3 - np.pi / 2), atol=1e-5)
    assert not np.allclose(observe(0.3), observe(0.6))


def test_transport_distance_known_answers() -> None:
    """Far away the carry path is lifted by the clearance; at the goal it is zero."""
    destination = (0.50, 0.00, 0.02)
    assert transport_distance(destination, destination) == pytest.approx(0.0)
    far_on_table = (0.30, 0.00, 0.02)
    assert transport_distance(far_on_table, destination) == pytest.approx(
        np.hypot(0.20, CARRY_CLEARANCE)
    )
    far_carried = (0.30, 0.00, 0.02 + CARRY_CLEARANCE)
    assert transport_distance(far_carried, destination) == pytest.approx(0.20)


def test_reach_reward_is_nonpositive_and_peaks_at_an_open_hold_on_target() -> None:
    """T1: closer and open is better, and the best non-success step pays exactly 0."""
    environment = PickPlaceRLEnv("T1", "L0")
    environment.reset(seed=0)
    target = (0.50, 0.00, 0.30)

    def reward(tcp: tuple[float, float, float], width: float = _OPEN_WIDTH) -> float:
        observation = _dict_observation(
            tcp=tcp, destination=target, gripper_width=width
        )
        return environment._compute_reward(_ZERO_ACTION, observation, False)[0]

    assert reward(target) == pytest.approx(0.0)
    assert reward((0.50, 0.00, 0.31)) < reward((0.50, 0.00, 0.305)) < 0.0
    assert reward(target, width=0.0) < reward(target)
    observation = _dict_observation(tcp=target, destination=target)
    success_reward = environment._compute_reward(_ZERO_ACTION, observation, True)[0]
    assert success_reward == pytest.approx(SUCCESS_BONUS)
    environment.close()


def test_object_reward_stages_strictly_dominate_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every later stage beats every earlier one, so the policy is pulled forward.

    `is_grasped` reads live MuJoCo contacts, not the fabricated observation, so
    it is patched to isolate the reward arithmetic.
    """
    environment = PickPlaceRLEnv("T2", "L0")
    environment.reset(seed=0)
    destination = (0.50, 0.05, 0.02)
    cube = (0.35, -0.20, 0.02)
    grasped = {"value": False}
    monkeypatch.setattr(rl_env, "is_grasped", lambda *_: grasped["value"])

    def reward(
        tcp: tuple[float, float, float],
        object_position: tuple[float, float, float],
        *,
        holding: bool,
        width: float,
    ) -> float:
        grasped["value"] = holding
        observation = _dict_observation(
            tcp=tcp,
            destination=destination,
            object_position=object_position,
            gripper_width=width,
        )
        return environment._compute_reward(_ZERO_ACTION, observation, False)[0]

    far = reward((0.50, 0.10, 0.40), cube, holding=False, width=_OPEN_WIDTH)
    at_cube = reward(cube, cube, holding=False, width=_OPEN_WIDTH)
    grasped_on_table = reward(cube, cube, holding=True, width=0.04)
    lifted = (cube[0], cube[1], cube[2] + CARRY_CLEARANCE)
    carried = reward(lifted, lifted, holding=True, width=0.04)
    nearly_there = (destination[0] - 0.02, destination[1], destination[2] + 0.03)
    almost = reward(nearly_there, nearly_there, holding=True, width=0.04)
    placed_closed = reward(destination, destination, holding=True, width=0.04)
    placed_open = reward(destination, destination, holding=False, width=_OPEN_WIDTH)

    assert far < at_cube < grasped_on_table < carried < almost < placed_closed
    assert placed_closed < placed_open == pytest.approx(0.0)
    assert all(value <= 0.0 for value in (far, at_cube, carried, placed_closed))
    environment.close()


def test_action_penalty_discourages_motion_without_changing_stage_order() -> None:
    """A full-magnitude delta costs at most the small action-penalty weight."""
    environment = PickPlaceRLEnv("T1", "L0")
    environment.reset(seed=0)
    observation = _dict_observation(tcp=(0.5, 0.0, 0.3), destination=(0.5, 0.0, 0.3))
    still = environment._compute_reward(_ZERO_ACTION, observation, False)[0]
    moving = environment._compute_reward(np.ones(5), observation, False)[0]
    assert moving == pytest.approx(still - rl_env.ACTION_PENALTY_WEIGHT)
    environment.close()


def test_same_seed_reproduces_observation_and_reward_trajectory() -> None:
    """Reproduce the flat observation and reward bit for bit across two instances."""
    first = PickPlaceRLEnv("T2", "L1", grasped_start_probability=0.5)
    second = PickPlaceRLEnv("T2", "L1", grasped_start_probability=0.5)
    for seed in (7, 8):
        first_observation, first_info = first.reset(seed=seed)
        second_observation, second_info = second.reset(seed=seed)
        np.testing.assert_array_equal(first_observation, second_observation)
        assert first_info["start_grasped"] == second_info["start_grasped"]

        random = np.random.default_rng(seed)
        for _ in range(8):
            action = random.uniform(-1.0, 1.0, size=5).astype(np.float32)
            first_step = first.step(action)
            second_step = second.step(action)
            np.testing.assert_array_equal(first_step[0], second_step[0])
            assert first_step[1] == second_step[1]
            assert first_step[2:4] == second_step[2:4]
    first.close()
    second.close()


def test_grasped_start_curriculum_begins_holding_the_cube() -> None:
    """With probability 1, an object episode starts grasped with a closed command."""
    environment = PickPlaceRLEnv("T2", "L1", grasped_start_probability=1.0)
    held_after_hold = 0
    hold = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for seed in range(5):
        _, info = environment.reset(seed=seed)
        if info["start_grasped"]:
            assert environment.command_tracker.gripper_command == 0.0
        for _ in range(5):
            _, _, _, _, step_info = environment.step(hold)
        held_after_hold += step_info["rl_stage"] in {"carry", "placed"}
    assert held_after_hold >= 4
    environment.close()


def test_grasped_start_is_ignored_for_the_objectless_reach_task() -> None:
    """T1 has no object to hold, so the curriculum option is a no-op there."""
    environment = PickPlaceRLEnv("T1", "L1", grasped_start_probability=1.0)
    _, info = environment.reset(seed=0)
    assert info["start_grasped"] is False
    assert environment.command_tracker.gripper_command == 1.0
    environment.close()


def test_episode_statistics_are_reported_when_the_episode_ends() -> None:
    """The final step's info carries the stage funnel and reward-term sums."""
    environment = PickPlaceRLEnv("T2", "L0")
    environment.reset(seed=0)
    terminated = truncated = False
    info: dict[str, Any] = {}
    steps = 0
    total_reward = 0.0
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = environment.step(_ZERO_ACTION)
        steps += 1
        total_reward += reward
        assert ("rl_episode" in info) == (terminated or truncated)
    episode = info["rl_episode"]
    assert episode["length"] == steps
    assert episode["success"] == 0.0
    assert {"reached", "grasped", "lifted", "placed", "min_distance"} <= set(episode)
    assert episode["return"] == pytest.approx(total_reward)
    assert episode["reward/success"] == 0.0
    environment.close()


def test_scripted_expert_replay_succeeds_through_every_stage() -> None:
    """The delta action space can express the expert, and its reward clears every stage."""
    environment = PickPlaceRLEnv("T2", "L0")
    environment.reset(seed=2026)
    terminated, expert_return, stages, episode = _replay_expert(environment, "T2")
    assert terminated, (
        "the scripted expert must solve T2-L0 for this check to mean anything"
    )
    assert stages == {"reach", "carry", "placed"}
    assert episode["reached"] == episode["grasped"] == episode["lifted"] == 1.0
    assert episode["placed"] == episode["success"] == 1.0

    environment.reset(seed=2026)
    idle_return = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = environment.step(_ZERO_ACTION)
        idle_return += reward
    assert expert_return > idle_return + SUCCESS_BONUS
    environment.close()


def test_base_env_rejects_a_grasped_start_without_an_object() -> None:
    """The base env refuses the curriculum option for the objectless reach task."""
    environment = PickPlaceEnv("T1", "L1", "state")
    with pytest.raises(ValueError, match="no object"):
        environment.reset(seed=0, options={"grasped_start_probability": 1.0})
    environment.close()
