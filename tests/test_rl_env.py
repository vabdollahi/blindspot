import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.pick_place_env import PickPlaceEnv
from envs.rl_env import (
    OBSERVATION_DIM,
    RL_TASKS,
    SUCCESS_BONUS,
    TIME_STEP_PENALTY,
    PickPlaceRLEnv,
)
from skills import ScriptedExpert


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


def test_same_seed_reproduces_observation_and_reward_trajectory() -> None:
    """Reproduce the flat observation and reward bit for bit across two instances."""
    first = PickPlaceRLEnv("T2", "L1")
    second = PickPlaceRLEnv("T2", "L1")
    first_observation, _ = first.reset(seed=7)
    second_observation, _ = second.reset(seed=7)
    np.testing.assert_array_equal(first_observation, second_observation)

    action = np.array([0.50, 0.0, 0.35, 0.0, 1.0], dtype=np.float32)
    for _ in range(8):
        first_step = first.step(action)
        second_step = second.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1] == second_step[1]
        assert first_step[2:4] == second_step[2:4]
    first.close()
    second.close()


def test_shaping_reward_telescopes_regardless_of_path() -> None:
    """Confirm progress-based shaping sums to initial-minus-final distance, any path.

    T1 has no object or grasp check, so `_compute_reward` can be driven directly
    with fabricated observations, isolating the shaping arithmetic from physics.
    """
    start = np.array([0.30, 0.20, 0.30])
    target = np.array([0.50, 0.00, 0.30])
    initial_distance = float(np.linalg.norm(target - start))
    direct_path = [start, start + (target - start) * 0.5, target]
    wandering_path = [
        start,
        start + np.array([0.10, -0.10, 0.00]),
        start + np.array([-0.05, 0.15, 0.05]),
        target,
    ]

    def observation_at(tcp_position: np.ndarray) -> dict[str, object]:
        state = np.zeros(12, dtype=np.float32)
        state[8:11] = tcp_position
        return {
            "observation.state": state,
            "target_pose": np.array([*target, 0.0], dtype=np.float32),
            "target_object": None,
            "object_poses": {},
        }

    def shaping_only_total(environment: PickPlaceRLEnv, path: list[np.ndarray]) -> float:
        """Sum per-step rewards after canceling out the flat time penalty."""
        action = np.zeros(5, dtype=np.float32)
        total = 0.0
        for position in path:
            reward, _ = environment._compute_reward(
                action, observation_at(position), terminated=False
            )
            total += reward + TIME_STEP_PENALTY
        return total

    env = PickPlaceRLEnv("T1", "L0")
    env.reset(seed=11)
    direct_total = shaping_only_total(env, direct_path)
    env.reset(seed=11)
    wandering_total = shaping_only_total(env, wandering_path)
    env.close()

    assert direct_total == pytest.approx(initial_distance, abs=1e-6)
    assert wandering_total == pytest.approx(initial_distance, abs=1e-6)


def test_successful_expert_episode_earns_grasp_bonus_and_success_bonus() -> None:
    """Confirm the staged reward reaches post-grasp and pays the success bonus."""
    seed = 2026
    dict_env = PickPlaceEnv("T2", "L0", "state")
    expert = ScriptedExpert("T2")
    dict_observation, _ = dict_env.reset(seed=seed)
    expert.reset()
    actions: list[np.ndarray] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = expert.act(dict_observation)
        actions.append(np.asarray(action, dtype=np.float32))
        dict_observation, _, terminated, truncated, _ = dict_env.step(action)
    dict_env.close()
    assert terminated, "the scripted expert must solve T2-L0 for this check to mean anything"

    rl_env = PickPlaceRLEnv("T2", "L0")
    rl_env.reset(seed=seed)
    stages_seen: set[str] = set()
    final_reward = 0.0
    final_terminated = False
    for action in actions:
        _, reward, final_terminated, _, info = rl_env.step(action)
        stages_seen.add(info["rl_stage"])
        final_reward = reward
    rl_env.close()

    assert final_terminated
    assert stages_seen == {"pre_grasp", "post_grasp"}
    assert final_reward >= SUCCESS_BONUS - 1.0
