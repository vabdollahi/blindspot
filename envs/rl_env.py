"""State-based RL environment: delta actions, a flat observation, and a staged reward.

Wraps `PickPlaceEnv` (state observation mode) instead of duplicating its
per-task destination and object bookkeeping. Scope is limited to the
non-language tasks T1, T2, T3, and T4; T3-L and T5 need language
conditioning and keep using the scripted expert (see roadmap §2.6).

The policy acts in a normalized `[-1, 1]` delta space: xyz and yaw move the
controller's current TCP target by at most one step of its own rate limit, and
the gripper command is binary. `DeltaCommandTracker` integrates those deltas
into the absolute 5-D `[x, y, z, yaw, g]` command that `PickPlaceEnv` and every
`Policy` use (roadmap §3), so a frozen RL expert remains a drop-in
absolute-action policy. See `docs/decisions.md` for the rationale behind the
action, observation, and reward design.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from common.transforms import yaw_from_quat_wxyz
from envs.pick_place_env import PickPlaceEnv
from motion.control import (
    ENVIRONMENT_TIMESTEP,
    TCP_SPEED_LIMIT,
    WORKSPACE_LOWER,
    WORKSPACE_UPPER,
    YAW_RATE_LIMIT,
    next_tcp_target,
)
from skills.grasp_check import is_grasped

RL_TASKS: tuple[str, ...] = ("T1", "T2", "T3", "T4")
OBSERVATION_DIM = 36
ACTION_DIM = 5
# One normalized action unit moves the target by one step of the controller's
# own rate limit, so the controller never rescales an in-range delta.
POSITION_DELTA_LIMIT = TCP_SPEED_LIMIT * ENVIRONMENT_TIMESTEP
YAW_DELTA_LIMIT = YAW_RATE_LIMIT * ENVIRONMENT_TIMESTEP

# Staged per-step reward (docs/decisions.md). Each stage's best value exceeds
# every earlier stage's, and the per-step reward is shifted to be <= 0, so
# finishing sooner always pays and success (which terminates) is never
# worth postponing.
# Inverse length scales (1/m) averaged by `proximity`: 2 keeps a usable
# gradient across the ~0.5 m initial TCP-to-object distance, 30 resolves the
# final millimeters.
PROXIMITY_SCALES = (2.0, 8.0, 30.0)
CARRY_CLEARANCE = 0.06
CARRY_BLEND_DISTANCE = 0.05
PLACE_TOLERANCE = 0.015
PLACED_REWARD = 4.0
REACH_TASK_REWARD_MAX = 2.0
OBJECT_TASK_REWARD_MAX = PLACED_REWARD + 1.0
SUCCESS_BONUS = 10.0
ACTION_PENALTY_WEIGHT = 0.001

# Stage-funnel thresholds, used only for training diagnostics.
REACHED_DISTANCE = 0.02
LIFTED_HEIGHT = 0.02

OBJECT_REWARD_TERMS = (
    "reach",
    "grasp",
    "transport",
    "placed",
    "open",
    "action_penalty",
    "success",
)
REACH_REWARD_TERMS = ("reach", "open", "action_penalty", "success")


def wrap_to_quarter_turn(angle: float) -> float:
    """Wrap an angle to [-π/4, π/4), the yaw error of a 90°-symmetric cube grasp."""
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-π, π)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def proximity(distance: float) -> float:
    """Map a distance to a [0, 1] closeness score: steep near zero, nonzero far away."""
    return float(
        np.mean([1.0 - np.tanh(scale * distance) for scale in PROXIMITY_SCALES])
    )


def transport_distance(object_position: ArrayLike, destination: ArrayLike) -> float:
    """Return the object's distance from a carry path that clears, then descends.

    Far from the destination the reference height is `CARRY_CLEARANCE` above
    it (lift and carry, clearing a bowl rim or the support cube); within
    `CARRY_BLEND_DISTANCE` horizontally it blends down to the destination.
    """
    position = np.asarray(object_position, dtype=np.float64)
    goal = np.asarray(destination, dtype=np.float64)
    horizontal = float(np.linalg.norm(position[:2] - goal[:2]))
    reference_height = goal[2] + CARRY_CLEARANCE * min(
        1.0, horizontal / CARRY_BLEND_DISTANCE
    )
    return float(np.hypot(horizontal, position[2] - reference_height))


def _object_pose(dict_observation: dict[str, Any]) -> tuple[NDArray[np.float64], float]:
    """Return the target object's position and yaw (the reach target for T1)."""
    target_object = dict_observation["target_object"] or None
    if target_object:
        object_pose = dict_observation["object_poses"][target_object]
        return (
            np.asarray(object_pose["position"], dtype=np.float64),
            yaw_from_quat_wxyz(object_pose["quaternion_wxyz"]),
        )
    target_pose = np.asarray(dict_observation["target_pose"], dtype=np.float64)
    return target_pose[:3].copy(), float(target_pose[3])


def flatten_observation(
    dict_observation: dict[str, Any],
    *,
    target_position: ArrayLike,
    target_yaw: float,
    gripper_command: float,
    previous_object_position: ArrayLike,
) -> np.ndarray:
    """Build the 36-D observation from the dict observation and command state."""
    state = np.asarray(dict_observation["observation.state"], dtype=np.float64)
    tcp_position = state[8:11]
    tcp_yaw = float(state[11])
    destination = np.asarray(dict_observation["target_pose"][:3], dtype=np.float64)
    object_position, object_yaw = _object_pose(dict_observation)
    return np.concatenate(
        [
            state[:7],
            state[7:8],
            tcp_position,
            [np.sin(tcp_yaw), np.cos(tcp_yaw)],
            object_position,
            # 4x yaw encodes a cube's 90° symmetry uniquely and continuously.
            [np.sin(4 * object_yaw), np.cos(4 * object_yaw)],
            [wrap_to_quarter_turn(object_yaw - tcp_yaw)],
            destination,
            object_position - tcp_position,
            destination - object_position,
            np.asarray(target_position, dtype=np.float64) - tcp_position,
            [wrap_to_pi(target_yaw - tcp_yaw)],
            [gripper_command],
            object_position - np.asarray(previous_object_position, dtype=np.float64),
        ]
    ).astype(np.float32)


class DeltaCommandTracker:
    """Integrate normalized delta actions into absolute `PickPlaceEnv` commands.

    Mirrors `ControlLoop`'s target update (`next_tcp_target`) so it can run
    from observations alone: the training wrapper and the evaluation-time
    `RLPolicy` share this one implementation, keeping their action and
    observation semantics identical.
    """

    def __init__(self) -> None:
        """Start uninitialized; `reset` must see the episode's first observation."""
        self._target_position: NDArray[np.float64] | None = None
        self._target_yaw = 0.0
        self._gripper_command = 1.0
        self._previous_object_position: NDArray[np.float64] | None = None

    @property
    def initialized(self) -> bool:
        """Return whether `reset` has been called for the current episode."""
        return self._target_position is not None

    @property
    def target_position(self) -> NDArray[np.float64]:
        """Return the mirrored controller TCP position target."""
        return self._require_target().copy()

    @property
    def target_yaw(self) -> float:
        """Return the mirrored controller TCP yaw target."""
        return float(self._target_yaw)

    @property
    def gripper_command(self) -> float:
        """Return the last binary gripper command (1 open, 0 closed)."""
        return float(self._gripper_command)

    def reset(
        self,
        dict_observation: dict[str, Any],
        *,
        target_position: ArrayLike | None = None,
        target_yaw: float | None = None,
        gripper_command: float = 1.0,
    ) -> None:
        """Start an episode from the measured TCP pose or a known controller target."""
        state = np.asarray(dict_observation["observation.state"], dtype=np.float64)
        self._target_position = (
            state[8:11].copy()
            if target_position is None
            else np.asarray(target_position, dtype=np.float64).copy()
        )
        self._target_yaw = float(state[11]) if target_yaw is None else float(target_yaw)
        self._gripper_command = float(gripper_command)
        self._previous_object_position = _object_pose(dict_observation)[0]

    def observe(self, dict_observation: dict[str, Any]) -> np.ndarray:
        """Return the flat observation and remember the object position for next step."""
        if self._previous_object_position is None:
            raise RuntimeError("DeltaCommandTracker.reset must be called first")
        observation = flatten_observation(
            dict_observation,
            target_position=self._require_target(),
            target_yaw=self._target_yaw,
            gripper_command=self._gripper_command,
            previous_object_position=self._previous_object_position,
        )
        self._previous_object_position = _object_pose(dict_observation)[0]
        return observation

    def command(self, normalized_action: ArrayLike) -> NDArray[np.float64]:
        """Return the absolute command for one delta action and advance the target."""
        current = self._require_target()
        action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape ({ACTION_DIM},)")
        delta = action[:3] * POSITION_DELTA_LIMIT
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > POSITION_DELTA_LIMIT:
            delta *= POSITION_DELTA_LIMIT / delta_norm
        absolute = np.array(
            [
                *np.clip(current + delta, WORKSPACE_LOWER, WORKSPACE_UPPER),
                np.clip(
                    self._target_yaw + action[3] * YAW_DELTA_LIMIT,
                    -np.pi / 2,
                    np.pi / 2,
                ),
                1.0 if action[4] > 0.0 else 0.0,
            ],
            dtype=np.float64,
        )
        self._target_position, self._target_yaw = next_tcp_target(
            current, self._target_yaw, absolute
        )
        self._gripper_command = float(absolute[4])
        return absolute

    def delta_from_absolute(self, absolute_action: ArrayLike) -> NDArray[np.float32]:
        """Return the delta action reproducing an absolute command's next target.

        Used to replay absolute-action policies (e.g. the scripted expert)
        through the delta action space; does not advance the tracker.
        """
        current = self._require_target()
        command = np.asarray(absolute_action, dtype=np.float64)
        next_position, next_yaw = next_tcp_target(current, self._target_yaw, command)
        return np.array(
            [
                *((next_position - current) / POSITION_DELTA_LIMIT),
                (next_yaw - self._target_yaw) / YAW_DELTA_LIMIT,
                1.0 if command[4] >= 0.5 else -1.0,
            ],
            dtype=np.float32,
        ).clip(-1.0, 1.0)

    def _require_target(self) -> NDArray[np.float64]:
        """Return the target position or raise if the tracker was never reset."""
        if self._target_position is None:
            raise RuntimeError("DeltaCommandTracker.reset must be called first")
        return self._target_position


class PickPlaceRLEnv(gym.Wrapper[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]):
    """Expose delta actions, a flat observation, and a staged dense reward."""

    def __init__(
        self,
        task: str,
        level: str = "L1",
        *,
        render_mode: str | None = None,
        grasped_start_probability: float = 0.0,
    ) -> None:
        """Wrap a non-language `PickPlaceEnv` task with Box spaces and the reward.

        `grasped_start_probability` (training only) starts that fraction of
        object-task episodes with the cube already held; it is ignored for T1.
        """
        normalized_task = task.strip().upper()
        if normalized_task not in RL_TASKS:
            raise ValueError(
                f"Phase 1b RL supports only non-language tasks {RL_TASKS}, got {task!r}"
            )
        if not 0.0 <= grasped_start_probability <= 1.0:
            raise ValueError("grasped_start_probability must be within [0, 1]")
        self._task_env = PickPlaceEnv(
            normalized_task, level, "state", render_mode=render_mode
        )
        super().__init__(self._task_env)
        self.task = normalized_task
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.command_tracker = DeltaCommandTracker()
        self._grasped_start_probability = (
            0.0 if normalized_task == "T1" else float(grasped_start_probability)
        )
        self._reward_max = (
            REACH_TASK_REWARD_MAX if normalized_task == "T1" else OBJECT_TASK_REWARD_MAX
        )
        self._reward_terms = (
            REACH_REWARD_TERMS if normalized_task == "T1" else OBJECT_REWARD_TERMS
        )
        self._object_rest_height = 0.0
        self._episode: dict[str, float] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the wrapped task, the command tracker, and the episode statistics."""
        reset_options = dict(options or {})
        if self._grasped_start_probability > 0.0:
            reset_options.setdefault(
                "grasped_start_probability", self._grasped_start_probability
            )
        dict_observation, info = self.env.reset(seed=seed, options=reset_options)
        control = self._task_env.control
        self.command_tracker.reset(
            dict_observation,
            target_position=control.target_position,
            target_yaw=control.target_yaw,
            gripper_command=control.gripper_open_fraction,
        )
        target_object = dict_observation["target_object"] or None
        self._object_rest_height = (
            float(self._task_env.model.geom(f"{target_object}_geom").size[2])
            if target_object
            else 0.0
        )
        self._episode = self._new_episode_stats(bool(info.get("start_grasped", False)))
        return self.command_tracker.observe(dict_observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the wrapped task with the integrated absolute command."""
        normalized_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        command = self.command_tracker.command(normalized_action)
        dict_observation, _, terminated, truncated, info = self.env.step(command)
        reward, terms, stage = self._compute_reward(
            normalized_action, dict_observation, terminated
        )
        observation = self.command_tracker.observe(dict_observation)
        self._update_episode_stats(dict_observation, reward, terms, terminated)
        step_info = {**info, "rl_stage": stage, "rl_reward_terms": terms}
        if terminated or truncated:
            step_info["rl_episode"] = dict(self._episode)
        return observation, reward, terminated, truncated, step_info

    def _compute_reward(
        self,
        normalized_action: np.ndarray,
        dict_observation: dict[str, Any],
        terminated: bool,
    ) -> tuple[float, dict[str, float], str]:
        """Return the staged per-step reward, its named terms, and the active stage."""
        state = np.asarray(dict_observation["observation.state"], dtype=np.float64)
        tcp_position = state[8:11]
        destination = np.asarray(dict_observation["target_pose"][:3], dtype=np.float64)
        open_fraction = self._gripper_openness(dict_observation)
        terms = dict.fromkeys(self._reward_terms, 0.0)
        target_object = dict_observation["target_object"] or None

        if target_object is None:
            terms["reach"] = proximity(
                float(np.linalg.norm(destination - tcp_position))
            )
            terms["open"] = open_fraction
            stage = "reach"
        else:
            object_position, _ = _object_pose(dict_observation)
            if np.linalg.norm(object_position - destination) <= PLACE_TOLERANCE:
                # Placing is worth more than any carrying state, so releasing
                # the object here never lowers the reward; opening raises it.
                terms["placed"] = PLACED_REWARD
                terms["open"] = open_fraction
                stage = "placed"
            else:
                grasped = float(
                    is_grasped(self._task_env.model, self._task_env.data, target_object)
                )
                terms["reach"] = proximity(
                    float(np.linalg.norm(object_position - tcp_position))
                )
                terms["grasp"] = grasped
                terms["transport"] = grasped * proximity(
                    transport_distance(object_position, destination)
                )
                stage = "carry" if grasped else "reach"

        staged = sum(
            value
            for name, value in terms.items()
            if name not in ("action_penalty", "success")
        )
        shaped = (staged - self._reward_max) / self._reward_max
        terms["action_penalty"] = -ACTION_PENALTY_WEIGHT * float(
            np.mean(normalized_action[:4] ** 2)
        )
        terms["success"] = SUCCESS_BONUS if terminated else 0.0
        return shaped + terms["action_penalty"] + terms["success"], terms, stage

    def _new_episode_stats(self, start_grasped: bool) -> dict[str, float]:
        """Return zeroed per-episode success, stage-funnel, and reward-term statistics."""
        stats = {
            "start_grasped": float(start_grasped),
            "success": 0.0,
            "length": 0.0,
            "return": 0.0,
            "min_distance": float("inf"),
            "reached": 0.0,
        }
        if self.task != "T1":
            stats.update(grasped=0.0, lifted=0.0, placed=0.0)
        stats.update({f"reward/{name}": 0.0 for name in self._reward_terms})
        return stats

    def _update_episode_stats(
        self,
        dict_observation: dict[str, Any],
        reward: float,
        terms: dict[str, float],
        terminated: bool,
    ) -> None:
        """Accumulate one step into the episode's funnel and reward-term sums."""
        stats = self._episode
        state = np.asarray(dict_observation["observation.state"], dtype=np.float64)
        object_position, _ = _object_pose(dict_observation)
        distance = float(np.linalg.norm(object_position - state[8:11]))
        stats["length"] += 1.0
        stats["return"] += reward
        stats["min_distance"] = min(stats["min_distance"], distance)
        stats["reached"] = float(stats["min_distance"] <= REACHED_DISTANCE)
        stats["success"] = float(terminated)
        if self.task != "T1":
            lift = object_position[2] - self._object_rest_height
            stats["grasped"] = max(stats["grasped"], terms["grasp"])
            stats["lifted"] = max(stats["lifted"], float(lift >= LIFTED_HEIGHT))
            stats["placed"] = max(stats["placed"], float(terms["placed"] > 0.0))
        for name, value in terms.items():
            stats[f"reward/{name}"] += value

    def _gripper_openness(self, dict_observation: dict[str, Any]) -> float:
        """Return the finger aperture as a 0 (closed) to 1 (open) fraction."""
        bounds = self._task_env.observation_space["observation.state"]
        low, high = float(bounds.low[7]), float(bounds.high[7])
        raw = float(dict_observation["observation.state"][7])
        return float(np.clip((raw - low) / (high - low), 0.0, 1.0))


def make_rl_env(
    task: str,
    level: str = "L1",
    *,
    render_mode: str | None = None,
    grasped_start_probability: float = 0.0,
) -> Callable[[], PickPlaceRLEnv]:
    """Return a zero-argument factory building one env, for vectorized workers.

    Building the model inside the returned callable (rather than passing an
    already-built environment) is required for `SubprocVecEnv`: each worker
    process must construct its own MuJoCo model after forking, never before.
    """

    def _make() -> PickPlaceRLEnv:
        return PickPlaceRLEnv(
            task,
            level,
            render_mode=render_mode,
            grasped_start_probability=grasped_start_probability,
        )

    return _make
