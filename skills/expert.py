"""Closed-loop scripted policy for the pick-and-place task suite."""

import re
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from common.transforms import yaw_from_quat_wxyz
from motion.control import ENVIRONMENT_TIMESTEP, WORKSPACE_LOWER, WORKSPACE_UPPER

PREGRASP_HEIGHT = 0.10
LIFT_HEIGHT = 0.11
CARRY_HEIGHT = 0.05
RETREAT_HEIGHT = 0.10
RELEASE_CLEARANCE = 0.005
OBJECT_POSITION_TOLERANCE = 0.005
OBJECT_YAW_TOLERANCE = float(np.deg2rad(5.0))
GRIPPER_CLOSE_STEPS = 30
GRIPPER_OPEN_STEPS = 30
LIFT_SETTLE_STEPS = 30
MAX_GRASP_ATTEMPTS = 2
LONG_CARRY_SPEED_LIMIT = 0.18
SHORT_CARRY_SPEED_LIMIT = 0.25
LONG_CARRY_DISTANCE = 0.47
GRASP_OFFSET_TOLERANCE = 0.03
HELD_OFFSET_TOLERANCE = 0.02
CUBE_HALF_HEIGHT = 0.02
BOWL_FLOOR_TOP = 0.006


class ScriptedExpert:
    """Produce closed-loop 5-D actions for T1-T4 and named-cube T3-L tasks."""

    def __init__(
        self,
        task: str | None = None,
        *,
        object_name: str | None = None,
    ) -> None:
        """Configure an optional task and object override for the policy."""
        self.task = task
        self.object_name = object_name
        self.reset()

    def reset(self, instruction: str | None = None) -> None:
        """Reset the state machine for a new episode and optional instruction."""
        self._instruction = instruction
        self._phase = "approach"
        self._phase_steps = 0
        self._grasp_attempts = 0
        self._active_object: str | None = None
        self._initial_object_position: NDArray[np.float64] | None = None
        self._grasp_yaw = 0.0
        self._grasp_offset: NDArray[np.float64] | None = None
        self._lift_target: NDArray[np.float64] | None = None
        self._lift_start_z = 0.0
        self._held_offset: NDArray[np.float64] | None = None
        self._carry_speed_limit: float | None = None
        self._retreat_target: NDArray[np.float64] | None = None

    def act(self, obs: Mapping[str, object]) -> NDArray[np.float64]:
        """Return one 20 Hz absolute TCP action from the latest oracle observation."""
        if not isinstance(obs, Mapping):
            raise TypeError("Expert observations must be mappings")
        task = self._task_name(obs)
        state = self._state(obs)
        tcp_position = state[8:11]
        tcp_yaw = float(state[11])
        if task == "T1":
            target = _pose(
                obs.get("target_pose"), default_yaw=tcp_yaw, name="target_pose"
            )
            return self._action(target[:3], target[3], 1.0)

        object_poses = self._object_poses(obs)
        if self._active_object is None:
            self._active_object = self._select_object(task, obs, object_poses)
        object_position, object_yaw = (
            _pose(
                object_poses[self._active_object],
                default_yaw=tcp_yaw,
                name=f"object_poses[{self._active_object!r}]",
            )[:3],
            _pose(
                object_poses[self._active_object],
                default_yaw=tcp_yaw,
                name=f"object_poses[{self._active_object!r}]",
            )[3],
        )
        if self._initial_object_position is None:
            self._initial_object_position = object_position.copy()
            self._grasp_yaw = _nearest_symmetric_yaw(object_yaw, tcp_yaw)

        if self._phase == "approach":
            target = object_position.copy()
            target[2] += PREGRASP_HEIGHT
            if self._at_pose(state, target, self._grasp_yaw):
                self._phase = "descend"
            return self._action(target, self._grasp_yaw, 1.0)

        if self._phase == "descend":
            target = object_position.copy()
            target[2] = max(target[2], WORKSPACE_LOWER[2])
            if self._at_pose(state, target, self._grasp_yaw):
                self._phase = "close"
                self._phase_steps = 0
            return self._action(target, self._grasp_yaw, 1.0)

        if self._phase == "close":
            self._phase_steps += 1
            if self._phase_steps >= GRIPPER_CLOSE_STEPS:
                self._phase = "lift"
                self._phase_steps = 0
                self._grasp_attempts += 1
                self._grasp_offset = object_position - tcp_position
                self._lift_target = tcp_position.copy()
                self._lift_target[2] += LIFT_HEIGHT
                self._lift_start_z = float(object_position[2])
            return self._action(tcp_position, self._grasp_yaw, 0.0)

        if self._phase == "lift":
            if self._grasp_offset is None or self._lift_target is None:
                raise RuntimeError("Lift phase requires a measured grasp target")
            target = self._lift_target
            self._phase_steps += 1
            if object_position[2] >= self._lift_start_z + 0.08 and self._at_pose(
                state, target, self._grasp_yaw
            ):
                relative_offset = object_position - tcp_position
                if (
                    np.linalg.norm(relative_offset - self._grasp_offset)
                    <= GRASP_OFFSET_TOLERANCE
                ):
                    self._phase = "lift_hold"
                    self._phase_steps = 0
                    self._held_offset = relative_offset.copy()
                else:
                    return self._retry_grasp(tcp_position)
            elif (
                self._at_pose(state, target, self._grasp_yaw) and self._phase_steps >= 3
            ):
                return self._retry_grasp(tcp_position)
            return self._action(target, self._grasp_yaw, 0.0)

        if self._phase == "lift_hold":
            if (
                self._grasp_offset is None
                or self._held_offset is None
                or self._lift_target is None
            ):
                raise RuntimeError("Lift hold requires a measured grasp offset")
            target = self._lift_target
            relative_offset = object_position - tcp_position
            if (
                object_position[2] < self._lift_start_z + 0.08
                or np.linalg.norm(relative_offset - self._held_offset)
                > HELD_OFFSET_TOLERANCE
            ):
                return self._retry_grasp(tcp_position)
            elif self._at_pose(state, target, self._grasp_yaw):
                self._phase_steps += 1
                if self._phase_steps >= LIFT_SETTLE_STEPS:
                    self._phase = "carry"
                    self._phase_steps = 0
                    self._held_offset = relative_offset.copy()
            else:
                self._phase_steps = 0
            return self._action(target, self._grasp_yaw, 0.0)

        if self._phase == "carry":
            if self._held_offset is None:
                raise RuntimeError("Carry phase requires a measured grasp offset")
            relative_offset = object_position - tcp_position
            if (
                np.linalg.norm(relative_offset - self._held_offset)
                > HELD_OFFSET_TOLERANCE
            ):
                return self._retry_grasp(tcp_position)
            destination = self._destination(task, obs, object_poses, object_position)
            if self._carry_speed_limit is None:
                transfer_distance = float(
                    np.linalg.norm(destination[:2] - object_position[:2])
                )
                self._carry_speed_limit = (
                    LONG_CARRY_SPEED_LIMIT
                    if transfer_distance > LONG_CARRY_DISTANCE
                    else SHORT_CARRY_SPEED_LIMIT
                )
            carry_target = destination.copy()
            carry_target[2] = max(
                carry_target[2] + CARRY_HEIGHT,
                float(object_position[2]),
            )
            target = carry_target[:3] - self._held_offset
            if self._at_pose(state, target, self._grasp_yaw):
                self._phase = "lower"
                self._held_offset = object_position - tcp_position
                return self._action(target, self._grasp_yaw, 0.0)
            command = _step_toward(
                tcp_position,
                target,
                self._carry_speed_limit * ENVIRONMENT_TIMESTEP,
            )
            return self._action(command, self._grasp_yaw, 0.0)

        if self._phase == "lower":
            if self._held_offset is None:
                raise RuntimeError("Lower phase requires a measured grasp offset")
            destination = self._destination(task, obs, object_poses, object_position)
            destination[2] += RELEASE_CLEARANCE
            target = destination[:3] - self._held_offset
            if self._at_pose(state, target, self._grasp_yaw):
                self._phase = "release"
                self._phase_steps = 0
            return self._action(target, self._grasp_yaw, 0.0)

        if self._phase == "release":
            self._phase_steps += 1
            if self._phase_steps >= GRIPPER_OPEN_STEPS:
                self._phase = "retreat"
                self._phase_steps = 0
                self._retreat_target = tcp_position.copy()
                self._retreat_target[2] += RETREAT_HEIGHT
            return self._action(tcp_position, self._grasp_yaw, 1.0)

        if self._phase == "retreat":
            if self._retreat_target is None:
                raise RuntimeError("Retreat phase requires a target pose")
            if self._at_pose(state, self._retreat_target, self._grasp_yaw):
                self._phase = "done"
            return self._action(self._retreat_target, self._grasp_yaw, 1.0)

        return self._action(tcp_position, tcp_yaw, 1.0)

    def _task_name(self, obs: Mapping[str, object]) -> str:
        raw_task = self.task if self.task is not None else obs.get("task")
        if not isinstance(raw_task, str):
            raise TypeError("Set the expert task or provide obs['task']")
        task = raw_task.strip().upper().replace("_", "-")
        if task == "T3L":
            task = "T3-L"
        if task not in {"T1", "T2", "T3", "T3-L", "T4"}:
            raise ValueError(f"Unsupported expert task: {raw_task!r}")
        return task

    @staticmethod
    def _state(obs: Mapping[str, object]) -> NDArray[np.float64]:
        value = obs.get("observation.state", obs.get("state"))
        state = np.asarray(value, dtype=np.float64)
        if state.shape != (12,) or not np.all(np.isfinite(state)):
            raise ValueError("observation.state must be a finite 12-value vector")
        return state

    @staticmethod
    def _object_poses(obs: Mapping[str, object]) -> Mapping[str, object]:
        value = obs.get("object_poses")
        if not isinstance(value, Mapping):
            raise TypeError("Object tasks require an object_poses mapping")
        return value

    def _select_object(
        self,
        task: str,
        obs: Mapping[str, object],
        object_poses: Mapping[str, object],
    ) -> str:
        explicit = self.object_name
        if explicit is None:
            for key in ("target_object", "object_name", "target_color"):
                value = obs.get(key)
                if isinstance(value, str):
                    explicit = value
                    break
        if task == "T4":
            explicit = "cube_red"
        elif task == "T3-L" and explicit is None:
            instruction = self._instruction
            if instruction is None:
                value = obs.get("instruction")
                instruction = value if isinstance(value, str) else None
            if instruction is None:
                raise ValueError("T3-L requires a named-cube instruction")
            colors = {
                str(name)[5:] for name in object_poses if str(name).startswith("cube_")
            }
            named_colors = [
                color
                for color in colors
                if re.search(rf"\b{re.escape(color)}\b", instruction, re.IGNORECASE)
            ]
            if len(named_colors) != 1:
                raise ValueError(
                    "T3-L instruction must name exactly one available cube color"
                )
            explicit = f"cube_{named_colors[0]}"
        if explicit is not None:
            resolved = _match_object_name(explicit, object_poses)
            if resolved is None:
                raise ValueError(
                    f"Target object {explicit!r} is absent from object_poses"
                )
            return resolved

        cubes = [str(name) for name in object_poses if str(name).startswith("cube_")]
        if len(cubes) == 1:
            return cubes[0]
        if "cube_red" in object_poses:
            return "cube_red"
        raise ValueError(
            "Object tasks require target_object when multiple cubes are present"
        )

    def _destination(
        self,
        task: str,
        obs: Mapping[str, object],
        object_poses: Mapping[str, object],
        object_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if task == "T2":
            target = _pose(
                obs.get("target_pose"),
                default_yaw=self._grasp_yaw,
                name="target_pose",
            )
            target[2] = float(self._initial_object_position[2])
            target[3] = self._grasp_yaw
            return target
        if task in {"T3", "T3-L"}:
            bowl_name = _match_object_name("bowl", object_poses)
            if bowl_name is not None:
                bowl = _pose(
                    object_poses[bowl_name],
                    default_yaw=self._grasp_yaw,
                    name=f"object_poses[{bowl_name!r}]",
                )
            else:
                bowl = _pose(
                    obs.get("target_pose"),
                    default_yaw=self._grasp_yaw,
                    name="target_pose",
                )
            return np.array(
                [
                    bowl[0],
                    bowl[1],
                    bowl[2] + BOWL_FLOOR_TOP + CUBE_HALF_HEIGHT,
                    self._grasp_yaw,
                ],
                dtype=np.float64,
            )
        blue_name = _match_object_name("cube_blue", object_poses)
        if blue_name is None:
            raise ValueError("T4 requires cube_blue in object_poses")
        blue = _pose(
            object_poses[blue_name],
            default_yaw=self._grasp_yaw,
            name=f"object_poses[{blue_name!r}]",
        )
        return np.array(
            [
                blue[0],
                blue[1],
                blue[2] + 2 * CUBE_HALF_HEIGHT,
                self._grasp_yaw,
            ],
            dtype=np.float64,
        )

    def _at_pose(
        self,
        state: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_yaw: float,
    ) -> bool:
        position_error = np.linalg.norm(state[8:11] - target_position)
        yaw_error = abs(
            np.arctan2(
                np.sin(float(state[11]) - target_yaw),
                np.cos(float(state[11]) - target_yaw),
            )
        )
        return bool(
            position_error <= OBJECT_POSITION_TOLERANCE
            and yaw_error <= OBJECT_YAW_TOLERANCE
        )

    def _retry_grasp(self, tcp_position: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._grasp_attempts < MAX_GRASP_ATTEMPTS:
            self._phase = "approach"
            self._phase_steps = 0
            self._grasp_offset = None
            self._lift_target = None
            self._held_offset = None
        else:
            self._phase = "failed"
        return self._action(tcp_position, self._grasp_yaw, 1.0)

    @staticmethod
    def _action(
        position: ArrayLike,
        yaw: float,
        gripper_open: float,
    ) -> NDArray[np.float64]:
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Expert action position must be a finite 3-D vector")
        if not np.isfinite(yaw):
            raise ValueError("Expert action yaw must be finite")
        clipped_position = np.clip(target, WORKSPACE_LOWER, WORKSPACE_UPPER)
        clipped_yaw = float(np.clip(yaw, -np.pi / 2, np.pi / 2))
        return np.array(
            [*clipped_position, clipped_yaw, gripper_open],
            dtype=np.float64,
        )


def _pose(value: object, *, default_yaw: float, name: str) -> NDArray[np.float64]:
    if isinstance(value, Mapping):
        position = value.get("position", value.get("xyz"))
        yaw = value.get("yaw")
        quaternion = value.get("quaternion_wxyz", value.get("quaternion"))
        if yaw is None and quaternion is not None:
            yaw = yaw_from_quat_wxyz(quaternion)
        if position is None:
            raise ValueError(f"{name} requires a position field")
        position_values = np.asarray(position, dtype=np.float64)
        yaw_value = default_yaw if yaw is None else float(yaw)
        values = np.array([*position_values, yaw_value], dtype=np.float64)
    else:
        raw = np.asarray(value, dtype=np.float64)
        if raw.shape == (3,):
            values = np.array([*raw, default_yaw], dtype=np.float64)
        elif raw.shape == (4,):
            values = raw.copy()
        elif raw.shape == (7,):
            values = np.array([*raw[:3], yaw_from_quat_wxyz(raw[3:])])
        else:
            raise ValueError(f"{name} must contain xyz, xyz+yaw, or xyz+wxyz")
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite xyz and yaw values")
    return values


def _match_object_name(
    requested: str,
    object_poses: Mapping[str, object],
) -> str | None:
    normalized = requested.strip().lower()
    color = normalized.removeprefix("cube_")
    for name in object_poses:
        candidate = str(name).lower()
        if candidate == normalized or candidate == color:
            return str(name)
        if candidate.startswith("cube_") and candidate[5:] == color:
            return str(name)
    return None


def _nearest_symmetric_yaw(object_yaw: float, current_yaw: float) -> float:
    candidates = [
        object_yaw + quarter_turns * np.pi / 2 for quarter_turns in range(-4, 5)
    ]
    candidates = [yaw for yaw in candidates if -np.pi / 2 <= yaw <= np.pi / 2]
    return float(min(candidates, key=lambda yaw: abs(yaw - current_yaw)))


def _step_toward(
    current: NDArray[np.float64],
    target: NDArray[np.float64],
    maximum_distance: float,
) -> NDArray[np.float64]:
    """Limit one policy command's TCP displacement to the requested distance."""
    displacement = target - current
    distance = float(np.linalg.norm(displacement))
    if distance <= maximum_distance:
        return target.copy()
    return current + displacement * (maximum_distance / distance)
