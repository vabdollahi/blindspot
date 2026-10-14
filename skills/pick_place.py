"""Closed-loop pick-and-place skills built on the shared 20 Hz control loop."""

from dataclasses import dataclass
from typing import Any

import mink
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from motion.control import (
    ENVIRONMENT_TIMESTEP,
    TCP_SPEED_LIMIT,
    WORKSPACE_LOWER,
    WORKSPACE_UPPER,
    ControlLoop,
)
from motion.ik import ARM_JOINT_NAMES
from skills.grasp_check import count_collisions, is_grasped, is_lifted

DEFAULT_REACH_TIMEOUT = 10.0
DEFAULT_SKILL_TIMEOUT = 20.0
DEFAULT_SETTLE_TIMEOUT = 2.0
PREGRASP_HEIGHT = 0.10
LIFT_HEIGHT = 0.11
CARRY_HEIGHT = 0.05
RETREAT_HEIGHT = 0.10
RELEASE_CLEARANCE = 0.005
REACH_POSITION_TOLERANCE = 0.005
REACH_YAW_TOLERANCE = float(np.deg2rad(5.0))
PLACE_XY_TOLERANCE = 0.03
PLACE_HEIGHT_TOLERANCE = 0.01
PLACE_YAW_TOLERANCE = float(np.deg2rad(5.0))
SETTLED_LINEAR_SPEED = 0.01
SETTLED_ANGULAR_SPEED = 0.1
SETTLED_STEPS = 5
GRIPPER_SETTLE_STEPS = 30
SKILL_TCP_SPEED_LIMIT = 0.15


@dataclass(frozen=True)
class SkillResult:
    """Report a skill outcome and its diagnostic details."""

    success: bool
    reason: str
    info: dict[str, Any]


class PickPlaceSkills:
    """Execute reach, grasp, place, stack, and pick-and-place through one control loop."""

    def __init__(self, control: ControlLoop) -> None:
        """Bind the skill suite to an existing control loop."""
        self.control = control
        self.model = control.model
        self.data = control.data
        self._tcp_site_id = self.model.site("tcp").id
        self._arm_qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.intp,
        )

    def reach(
        self,
        pose: ArrayLike,
        *,
        timeout: float = DEFAULT_REACH_TIMEOUT,
    ) -> SkillResult:
        """Move the TCP to an [x, y, z, yaw] pose and verify 5 mm and 5 degree tolerances."""
        values = _pose_values(pose)
        deadline = self._deadline(timeout)
        result = self._move_to_pose(values, 1.0, deadline)
        return _with_context(result, skill="reach", target=values.tolist())

    def grasp(
        self,
        obj: str,
        *,
        timeout: float = DEFAULT_SKILL_TIMEOUT,
    ) -> SkillResult:
        """Top-down grasp a named free body, lift it 10 cm, and verify the lift."""
        deadline = self._deadline(timeout)
        body_id = self.model.body(obj).id
        if not self._is_free_body(body_id):
            return _failure("grasp_missed", object=obj, detail="Object is not a free body")

        initial_position = self.data.xpos[body_id].copy()
        grasp_yaw = self._nearest_symmetric_yaw(
            self._body_yaw(body_id), self._tcp_yaw()
        )
        object_position = self.data.xpos[body_id].copy()
        pregrasp_position = object_position.copy()
        pregrasp_position[2] += PREGRASP_HEIGHT

        result = self._move_to_pose(
            np.array([*pregrasp_position, grasp_yaw]), 1.0, deadline
        )
        if not result.success:
            return _with_context(result, skill="grasp", object=obj)

        object_position = self.data.xpos[body_id].copy()
        object_position[2] = max(object_position[2], WORKSPACE_LOWER[2])
        result = self._move_to_pose(
            np.array([*object_position, grasp_yaw]), 1.0, deadline
        )
        if not result.success:
            return _with_context(result, skill="grasp", object=obj)

        grasp_position = self._tcp_position()
        result = self._repeat_action(
            np.array([*grasp_position, grasp_yaw, 0.0]),
            GRIPPER_SETTLE_STEPS,
            deadline,
        )
        if result is not None:
            return _with_context(result, skill="grasp", object=obj)
        if not is_grasped(self.model, self.data, obj):
            return _failure(
                "grasp_missed",
                object=obj,
                grasp_yaw=grasp_yaw,
                object_position=self.data.xpos[body_id].tolist(),
            )

        lift_start_position = self._tcp_position()
        lift_position = lift_start_position.copy()
        lift_position[2] += LIFT_HEIGHT
        result = self._move_to_pose(
            np.array([*lift_position, grasp_yaw]), 0.0, deadline
        )
        if not result.success:
            return _with_context(result, skill="grasp", object=obj)
        result = self._repeat_action(
            np.array([*lift_position, grasp_yaw, 0.0]),
            GRIPPER_SETTLE_STEPS,
            deadline,
        )
        if result is not None:
            return _with_context(result, skill="grasp", object=obj)

        lifted_height = float(self.data.xpos[body_id, 2] - initial_position[2])
        grasp_still_holds = is_grasped(self.model, self.data, obj)
        lift_succeeded = is_lifted(self.model, self.data, obj, initial_position)
        if not grasp_still_holds or not lift_succeeded:
            return _failure(
                "slipped",
                object=obj,
                lifted_height=lifted_height,
                minimum_lift=0.08,
                grasp_still_holds=grasp_still_holds,
                lift_succeeded=lift_succeeded,
                tcp_lift_height=float(self._tcp_position()[2] - lift_start_position[2]),
            )
        return _success(
            skill="grasp",
            object=obj,
            grasp_yaw=grasp_yaw,
            initial_position=initial_position.tolist(),
            final_position=self.data.xpos[body_id].tolist(),
            lifted_height=lifted_height,
        )

    def place(
        self,
        pose: ArrayLike,
        *,
        timeout: float = DEFAULT_SKILL_TIMEOUT,
    ) -> SkillResult:
        """Place the currently held object at its requested resting-center pose."""
        values = _pose_values(pose)
        deadline = self._deadline(timeout)
        invalid_waypoint = self._placement_waypoint_failure(values)
        if invalid_waypoint is not None:
            return _with_context(invalid_waypoint, skill="place")

        body_id = self._held_body_id()
        if body_id is None:
            return _failure("grasp_missed", skill="place", detail="No object is held")
        object_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        object_half_height = self._body_half_height(body_id)
        object_to_tcp_offset = self.data.xpos[body_id] - self._tcp_position()

        carry_object_position = values[:3].copy()
        carry_object_position[2] = max(
            carry_object_position[2] + CARRY_HEIGHT,
            float(self.data.xpos[body_id, 2]),
        )
        carry_position = carry_object_position - object_to_tcp_offset
        result = self._move_to_pose(
            np.array([*carry_position, values[3]]),
            0.0,
            deadline,
            tcp_speed_limit=SKILL_TCP_SPEED_LIMIT,
        )
        if not result.success:
            return _with_context(result, skill="place", object=object_name)
        if not is_grasped(self.model, self.data, str(object_name)):
            return _failure("slipped", skill="place", object=object_name, phase="carry")
        object_to_tcp_offset = self.data.xpos[body_id] - self._tcp_position()

        lower_position = values[:3] - object_to_tcp_offset
        lower_position[2] += RELEASE_CLEARANCE
        result = self._move_to_pose(
            np.array([*lower_position, values[3]]), 0.0, deadline
        )
        if not result.success:
            return _with_context(result, skill="place", object=object_name)

        tcp_position = self._tcp_position()
        result = self._repeat_action(
            np.array([*tcp_position, values[3], 1.0]),
            GRIPPER_SETTLE_STEPS,
            deadline,
        )
        if result is not None:
            return _with_context(result, skill="place", object=object_name)
        release_position = self.data.xpos[body_id].copy()

        retreat_position = self._tcp_position()
        retreat_position[2] += RETREAT_HEIGHT
        result = self._move_to_pose(
            np.array([*retreat_position, values[3]]), 1.0, deadline
        )
        if not result.success:
            return _with_context(result, skill="place", object=object_name)

        stable = self._wait_until_settled(body_id, deadline)
        if not stable:
            return _failure("timeout", skill="place", object=object_name, phase="settle")

        actual_position = self.data.xpos[body_id].copy()
        xy_error = float(np.linalg.norm(actual_position[:2] - values[:2]))
        height_error = float(abs(actual_position[2] - values[2]))
        yaw_error = self._symmetric_yaw_error(self._body_yaw(body_id), values[3])
        if (
            xy_error > PLACE_XY_TOLERANCE
            or height_error > PLACE_HEIGHT_TOLERANCE
            or yaw_error > PLACE_YAW_TOLERANCE
        ):
            return _failure(
                "slipped",
                skill="place",
                object=object_name,
                target=values.tolist(),
                actual_position=actual_position.tolist(),
                release_position=release_position.tolist(),
                xy_error=xy_error,
                height_error=height_error,
                yaw_error=yaw_error,
            )
        return _success(
            skill="place",
            object=object_name,
            target=values.tolist(),
            actual_position=actual_position.tolist(),
            object_half_height=object_half_height,
            xy_error=xy_error,
            height_error=height_error,
            yaw_error=yaw_error,
        )

    def pick_and_place(
        self,
        obj: str,
        destination: ArrayLike,
        *,
        timeout: float = DEFAULT_SKILL_TIMEOUT,
    ) -> SkillResult:
        """Compose grasp and place for a named object and destination pose."""
        values = _pose_values(destination)
        invalid_waypoint = self._placement_waypoint_failure(values)
        if invalid_waypoint is not None:
            return _with_context(invalid_waypoint, skill="pick_and_place", object=obj)

        deadline = self._deadline(timeout)
        grasp_result = self.grasp(obj, timeout=self._remaining_timeout(deadline))
        if not grasp_result.success:
            return _with_context(
                grasp_result, skill="pick_and_place", grasp=grasp_result.info
            )
        place_result = self.place(values, timeout=self._remaining_timeout(deadline))
        return _with_context(
            place_result,
            skill="pick_and_place",
            grasp=grasp_result.info,
            place=place_result.info,
        )

    def stack(
        self,
        a: str,
        b: str,
        *,
        timeout: float = DEFAULT_SKILL_TIMEOUT,
    ) -> SkillResult:
        """Grasp object a and place it centered on top of object b."""
        if a == b:
            raise ValueError("A stack requires two different objects")
        object_a_id = self.model.body(a).id
        object_b_id = self.model.body(b).id
        if not self._is_free_body(object_a_id) or not self._is_free_body(object_b_id):
            return _failure("unreachable", skill="stack", objects=[a, b])

        deadline = self._deadline(timeout)
        grasp_result = self.grasp(a, timeout=self._remaining_timeout(deadline))
        if not grasp_result.success:
            return _with_context(
                grasp_result, skill="stack", objects=[a, b], grasp=grasp_result.info
            )

        base_position = self.data.xpos[object_b_id].copy()
        stack_position = np.array(
            [
                base_position[0],
                base_position[1],
                self._body_top_z(object_b_id) + self._body_half_height(object_a_id),
            ],
            dtype=np.float64,
        )
        stack_pose = np.array([*stack_position, self._body_yaw(object_a_id)])
        place_result = self.place(
            stack_pose, timeout=self._remaining_timeout(deadline)
        )
        return _with_context(
            place_result,
            skill="stack",
            objects=[a, b],
            grasp=grasp_result.info,
            place=place_result.info,
        )

    def _move_to_pose(
        self,
        pose: NDArray[np.float64],
        gripper_open: float,
        deadline: float,
        *,
        tcp_speed_limit: float = TCP_SPEED_LIMIT,
    ) -> SkillResult:
        position = pose[:3]
        yaw = float(pose[3])
        if not self._pose_in_workspace(position, yaw):
            return _failure("unreachable", target=pose.tolist())

        target_rotation = mink.SO3.from_z_radians(yaw) @ mink.SO3.from_x_radians(
            np.pi
        )
        target = mink.SE3.from_rotation_and_translation(target_rotation, position)
        try:
            _, reachable = self.control.ik.solve(target, self.data.qpos[self._arm_qpos_indices])
        except mink.exceptions.NoSolutionFound as error:
            return _failure("ik_failed", detail=str(error), target=pose.tolist())
        if not reachable:
            return _failure("unreachable", target=pose.tolist())

        commanded_position = self._tcp_position()
        maximum_step = tcp_speed_limit * ENVIRONMENT_TIMESTEP
        while self.data.time < deadline:
            remaining = position - commanded_position
            distance = float(np.linalg.norm(remaining))
            if distance > maximum_step:
                commanded_position = commanded_position + remaining * (
                    maximum_step / distance
                )
            else:
                commanded_position = position.copy()
            action = np.array(
                [*commanded_position, yaw, gripper_open], dtype=np.float64
            )
            failure = self._apply_once(action)
            if failure is not None:
                return _with_context(failure, target=pose.tolist())
            position_error = float(np.linalg.norm(self._tcp_position() - position))
            yaw_error = self._angle_error(self._tcp_yaw(), yaw)
            if (
                position_error <= REACH_POSITION_TOLERANCE
                and yaw_error <= REACH_YAW_TOLERANCE
            ):
                return _success(
                    target=pose.tolist(),
                    position_error=position_error,
                    yaw_error=yaw_error,
                )
        return _failure(
            "timeout",
            target=pose.tolist(),
            position_error=float(np.linalg.norm(self._tcp_position() - position)),
            yaw_error=self._angle_error(self._tcp_yaw(), yaw),
        )

    def _repeat_action(
        self,
        action: NDArray[np.float64],
        steps: int,
        deadline: float,
    ) -> SkillResult | None:
        for _ in range(steps):
            if self.data.time >= deadline:
                return _failure("timeout", phase="hold")
            failure = self._apply_once(action)
            if failure is not None:
                return failure
        return None

    def _apply_once(self, action: NDArray[np.float64]) -> SkillResult | None:
        try:
            self.control.apply(action)
        except mink.exceptions.NoSolutionFound as error:
            return _failure("ik_failed", detail=str(error))
        collision_count = count_collisions(self.model, self.data)
        if collision_count:
            return _failure("collision", collision_count=collision_count)
        return None

    def _wait_until_settled(self, body_id: int, deadline: float) -> bool:
        stable_steps = 0
        tcp_position = self._tcp_position()
        tcp_yaw = self._tcp_yaw()
        action = np.array([*tcp_position, tcp_yaw, 1.0], dtype=np.float64)
        while self.data.time < deadline:
            failure = self._apply_once(action)
            if failure is not None:
                return False
            velocity = self._body_velocity(body_id)
            if (
                np.linalg.norm(velocity[:3]) <= SETTLED_LINEAR_SPEED
                and np.linalg.norm(velocity[3:]) <= SETTLED_ANGULAR_SPEED
            ):
                stable_steps += 1
                if stable_steps >= SETTLED_STEPS:
                    return True
            else:
                stable_steps = 0
        return False

    def _placement_waypoint_failure(
        self, pose: NDArray[np.float64]
    ) -> SkillResult | None:
        carry_position = pose[:3].copy()
        carry_position[2] += CARRY_HEIGHT
        lower_position = pose[:3].copy()
        lower_position[2] += RELEASE_CLEARANCE
        if not self._pose_in_workspace(carry_position, float(pose[3])):
            return _failure("unreachable", target=pose.tolist(), phase="carry")
        if not self._pose_in_workspace(lower_position, float(pose[3])):
            return _failure("unreachable", target=pose.tolist(), phase="lower")
        return None

    def _pose_in_workspace(self, position: NDArray[np.float64], yaw: float) -> bool:
        return bool(
            np.all(position >= WORKSPACE_LOWER)
            and np.all(position <= WORKSPACE_UPPER)
            and -np.pi / 2 <= yaw <= np.pi / 2
        )

    def _deadline(self, timeout: float) -> float:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Skill timeout must be finite and positive")
        return float(self.data.time + timeout)

    def _remaining_timeout(self, deadline: float) -> float:
        remaining = deadline - float(self.data.time)
        if remaining <= 0:
            return np.finfo(np.float64).eps
        return remaining

    def _tcp_position(self) -> NDArray[np.float64]:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _tcp_yaw(self) -> float:
        rotation = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _body_yaw(self, body_id: int) -> float:
        rotation = self.data.xmat[body_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _nearest_symmetric_yaw(self, object_yaw: float, current_yaw: float) -> float:
        candidates = [
            object_yaw + quarter_turns * np.pi / 2
            for quarter_turns in range(-4, 5)
        ]
        candidates = [
            yaw for yaw in candidates if -np.pi / 2 <= yaw <= np.pi / 2
        ]
        return float(min(candidates, key=lambda yaw: abs(yaw - current_yaw)))

    def _symmetric_yaw_error(self, actual: float, target: float) -> float:
        differences = [
            self._angle_error(actual, target + quarter_turns * np.pi / 2)
            for quarter_turns in range(-2, 3)
        ]
        return min(differences)

    @staticmethod
    def _angle_error(actual: float, target: float) -> float:
        return abs(float(np.arctan2(np.sin(actual - target), np.cos(actual - target))))

    def _is_free_body(self, body_id: int) -> bool:
        if self.model.body_jntnum[body_id] != 1:
            return False
        joint_id = int(self.model.body_jntadr[body_id])
        return self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE

    def _held_body_id(self) -> int | None:
        finger_ids = {
            self.model.body("left_finger").id,
            self.model.body("right_finger").id,
        }
        touching: dict[int, set[int]] = {}
        for contact in self.data.contact[: self.data.ncon]:
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            if first in finger_ids and self._is_free_body(second):
                touching.setdefault(second, set()).add(first)
            elif second in finger_ids and self._is_free_body(first):
                touching.setdefault(first, set()).add(second)
        held = [body_id for body_id, fingers in touching.items() if fingers == finger_ids]
        return held[0] if len(held) == 1 else None

    def _body_half_height(self, body_id: int) -> float:
        geom_start = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        half_heights = []
        for geom_id in range(geom_start, geom_start + geom_count):
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            half_height = float(
                np.dot(np.abs(rotation[2]), self.model.geom_size[geom_id, :3])
            )
            half_heights.append(half_height)
        if not half_heights:
            raise ValueError("Stacking requires box-shaped objects")
        return max(half_heights)

    def _body_top_z(self, body_id: int) -> float:
        geom_start = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        tops = []
        for geom_id in range(geom_start, geom_start + geom_count):
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            half_height = float(
                np.dot(np.abs(rotation[2]), self.model.geom_size[geom_id, :3])
            )
            tops.append(float(self.data.geom_xpos[geom_id, 2] + half_height))
        if not tops:
            raise ValueError("Stacking requires box-shaped objects")
        return max(tops)

    def _body_velocity(self, body_id: int) -> NDArray[np.float64]:
        joint_id = int(self.model.body_jntadr[body_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        return self.data.qvel[dof_address : dof_address + 6].copy()


def _pose_values(pose: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("A skill pose must be a finite [x, y, z, yaw] vector")
    return values


def _success(**info: Any) -> SkillResult:
    return SkillResult(success=True, reason="success", info=info)


def _failure(reason: str, **info: Any) -> SkillResult:
    return SkillResult(success=False, reason=reason, info=info)


def _with_context(result: SkillResult, **context: Any) -> SkillResult:
    return SkillResult(
        success=result.success,
        reason=result.reason,
        info={**result.info, **context},
    )