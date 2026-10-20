"""State-based RL environment: a flat observation and a dense reward.

Wraps `PickPlaceEnv` (state observation mode) instead of duplicating its
per-task destination and object bookkeeping. Scope is limited to the
non-language tasks T1, T2, T3, and T4; T3-L and T5 need language
conditioning and keep using the scripted expert (see roadmap §2.6).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.transforms import yaw_from_quat_wxyz
from envs.pick_place_env import PickPlaceEnv
from skills.grasp_check import is_grasped, is_lifted

RL_TASKS: tuple[str, ...] = ("T1", "T2", "T3", "T4")
OBSERVATION_DIM = 27

# Reward weights. Not specified numerically in the roadmap; chosen so that
# the success bonus dominates a full episode of dense shaping without being
# swamped by episode length. Logged in docs/decisions.md.
SUCCESS_BONUS = 10.0
GRASP_LIFT_BONUS = 5.0
ACTION_RATE_WEIGHT = 0.01
TIME_STEP_PENALTY = 0.01


class PickPlaceRLEnv(gym.Wrapper[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]):
    """Flatten `PickPlaceEnv`'s state observation and add a staged dense reward."""

    def __init__(
        self,
        task: str,
        level: str = "L1",
        *,
        render_mode: str | None = None,
    ) -> None:
        """Wrap a non-language `PickPlaceEnv` task with a Box observation and reward."""
        normalized_task = task.strip().upper()
        if normalized_task not in RL_TASKS:
            raise ValueError(
                f"Phase 1b RL supports only non-language tasks {RL_TASKS}, "
                f"got {task!r}"
            )
        super().__init__(
            PickPlaceEnv(normalized_task, level, "state", render_mode=render_mode)
        )
        self.task = normalized_task
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self._initial_object_position: np.ndarray | None = None
        self._grasp_achieved = False
        self._previous_action: np.ndarray | None = None
        self._previous_stage: str | None = None
        self._previous_stage_distance: float | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the wrapped task and the per-episode reward-staging state."""
        dict_observation, info = self.env.reset(seed=seed, options=options)
        self._grasp_achieved = False
        self._previous_action = None
        self._previous_stage = None
        self._previous_stage_distance = None
        target_object = dict_observation["target_object"] or None
        self._initial_object_position = (
            np.asarray(
                dict_observation["object_poses"][target_object]["position"],
                dtype=np.float64,
            ).copy()
            if target_object
            else None
        )
        return self._flatten_observation(dict_observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the wrapped task and return the flat observation and dense reward."""
        dict_observation, _, terminated, truncated, info = self.env.step(action)
        reward, stage = self._compute_reward(action, dict_observation, terminated)
        observation = self._flatten_observation(dict_observation)
        return observation, reward, terminated, truncated, {**info, "rl_stage": stage}

    def _flatten_observation(self, dict_observation: dict[str, Any]) -> np.ndarray:
        """Build the 27-D observation from state, object poses, and target pose."""
        state = np.asarray(dict_observation["observation.state"], dtype=np.float32)
        joint_positions = state[:7]
        gripper_width = state[7:8]
        tcp_position = state[8:11]
        tcp_yaw = float(state[11])
        destination_position = np.asarray(
            dict_observation["target_pose"][:3], dtype=np.float32
        )
        target_object = dict_observation["target_object"] or None
        if target_object:
            object_pose = dict_observation["object_poses"][target_object]
            object_position = np.asarray(object_pose["position"], dtype=np.float32)
            object_yaw = yaw_from_quat_wxyz(object_pose["quaternion_wxyz"])
        else:
            # T1 has no object: the reach target plays both roles (see roadmap 1b.1).
            object_position = destination_position
            object_yaw = float(dict_observation["target_pose"][3])
        object_to_tcp = object_position - tcp_position
        destination_to_object = destination_position - object_position
        return np.concatenate(
            [
                joint_positions,
                gripper_width,
                tcp_position,
                [np.sin(tcp_yaw), np.cos(tcp_yaw)],
                object_position,
                [np.sin(object_yaw), np.cos(object_yaw)],
                destination_position,
                object_to_tcp,
                destination_to_object,
            ]
        ).astype(np.float32)

    def _compute_reward(
        self,
        action: np.ndarray,
        dict_observation: dict[str, Any],
        terminated: bool,
    ) -> tuple[float, str]:
        """Return the staged dense reward and the active stage, for logging."""
        tcp_position = np.asarray(
            dict_observation["observation.state"][8:11], dtype=np.float64
        )
        destination_position = np.asarray(
            dict_observation["target_pose"][:3], dtype=np.float64
        )
        target_object = dict_observation["target_object"] or None
        if target_object:
            object_position = np.asarray(
                dict_observation["object_poses"][target_object]["position"],
                dtype=np.float64,
            )
        else:
            object_position = destination_position

        reward = 0.0
        if target_object and not self._grasp_achieved:
            model, data = self.env.model, self.env.data
            grasped = is_grasped(model, data, target_object)
            if grasped and is_lifted(
                model, data, target_object, self._initial_object_position
            ):
                self._grasp_achieved = True
                # Awarded once, not every step, so holding the object indefinitely
                # is never more rewarding than carrying it on to the destination.
                reward += GRASP_LIFT_BONUS

        if self._grasp_achieved:
            stage_distance = float(np.linalg.norm(destination_position - object_position))
            stage = "post_grasp"
        else:
            stage_distance = float(np.linalg.norm(object_position - tcp_position))
            stage = "pre_grasp"

        # Potential-based shaping: reward progress made this step (telescopes to a
        # total bounded by the stage's starting distance, regardless of path length
        # or episode duration), not the remaining distance itself (which an agent
        # pays every step it is alive, so it would grow with episode length and
        # swamp the one-time bonuses below). A stage switch resets the baseline
        # instead of comparing across two different potential functions.
        if stage == self._previous_stage and self._previous_stage_distance is not None:
            reward += self._previous_stage_distance - stage_distance
        self._previous_stage = stage
        self._previous_stage_distance = stage_distance

        if terminated:
            reward += SUCCESS_BONUS

        reward -= TIME_STEP_PENALTY

        action_array = np.asarray(action, dtype=np.float64)
        if self._previous_action is not None:
            # Penalize the change between consecutive absolute pose targets
            # (an action-rate penalty), not the raw magnitude: the action is
            # an absolute workspace pose, so penalizing its magnitude directly
            # would just bias the policy toward the coordinate origin.
            rate = action_array[:4] - self._previous_action[:4]
            reward -= ACTION_RATE_WEIGHT * float(np.sum(rate**2))
        self._previous_action = action_array

        return reward, stage


def make_rl_env(
    task: str,
    level: str = "L1",
    *,
    render_mode: str | None = None,
) -> Callable[[], PickPlaceRLEnv]:
    """Return a zero-argument factory building one env, for vectorized workers.

    Building the model inside the returned callable (rather than passing an
    already-built environment) is required for `SubprocVecEnv`: each worker
    process must construct its own MuJoCo model after forking, never before.
    """

    def _make() -> PickPlaceRLEnv:
        return PickPlaceRLEnv(task, level, render_mode=render_mode)

    return _make
