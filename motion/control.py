"""20 Hz control loop for Panda TCP actions."""

import math

import mink
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from motion.ik import ARM_JOINT_NAMES, MinkIK

ENVIRONMENT_TIMESTEP = 0.05
IK_STEPS_PER_ENVIRONMENT_STEP = 5
PHYSICS_STEPS_PER_IK_STEP = 5
TCP_SPEED_LIMIT = 0.25
YAW_RATE_LIMIT = 1.0
WORKSPACE_LOWER = np.array([0.25, -0.30, 0.02], dtype=np.float64)
WORKSPACE_UPPER = np.array([0.70, 0.30, 0.45], dtype=np.float64)

_ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
_GRIPPER_ACTUATOR_NAME = "actuator8"
_TCP_SITE_NAME = "tcp"
_DOWN_ROTATION = mink.SO3.from_x_radians(math.pi)


def tcp_pose(position: ArrayLike, yaw: float) -> mink.SE3:
    """Return the downward-pointing TCP pose at a base-frame position and yaw."""
    rotation = mink.SO3.from_z_radians(float(yaw)) @ _DOWN_ROTATION
    return mink.SE3.from_rotation_and_translation(
        rotation, np.asarray(position, dtype=np.float64)
    )


def next_tcp_target(
    current_position: ArrayLike,
    current_yaw: float,
    command: ArrayLike,
    *,
    tcp_speed_limit: float = TCP_SPEED_LIMIT,
    yaw_rate_limit: float = YAW_RATE_LIMIT,
) -> tuple[NDArray[np.float64], float]:
    """Return the next TCP target one environment step toward a clipped command."""
    command_array = np.asarray(command, dtype=np.float64)
    position = np.asarray(current_position, dtype=np.float64)
    goal_position = np.clip(command_array[:3], WORKSPACE_LOWER, WORKSPACE_UPPER)
    goal_yaw = float(np.clip(command_array[3], -np.pi / 2, np.pi / 2))

    displacement = goal_position - position
    distance = np.linalg.norm(displacement)
    maximum_distance = tcp_speed_limit * ENVIRONMENT_TIMESTEP
    if distance > maximum_distance:
        displacement *= maximum_distance / distance
    maximum_yaw_change = yaw_rate_limit * ENVIRONMENT_TIMESTEP
    yaw_change = float(
        np.clip(goal_yaw - current_yaw, -maximum_yaw_change, maximum_yaw_change)
    )
    return position + displacement, float(current_yaw) + yaw_change


class ControlLoop:
    """Apply 5-D TCP actions through Mink and MuJoCo position servos."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik: MinkIK | None = None,
        *,
        tcp_speed_limit: float = TCP_SPEED_LIMIT,
        yaw_rate_limit: float = YAW_RATE_LIMIT,
    ) -> None:
        if not np.isclose(model.opt.timestep, 0.002):
            raise ValueError("The control loop requires a 2 ms physics timestep")
        if not np.isfinite(tcp_speed_limit) or tcp_speed_limit <= 0:
            raise ValueError("TCP speed limit must be finite and positive")
        if not np.isfinite(yaw_rate_limit) or yaw_rate_limit <= 0:
            raise ValueError("Yaw rate limit must be finite and positive")

        self.model = model
        self.data = data
        self.ik = ik if ik is not None else MinkIK()
        self.tcp_speed_limit = float(tcp_speed_limit)
        self.yaw_rate_limit = float(yaw_rate_limit)
        self._arm_qpos_indices = np.asarray(
            [
                model.jnt_qposadr[model.joint(name).id]
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._arm_actuator_ids = np.asarray(
            [model.actuator(name).id for name in _ARM_ACTUATOR_NAMES],
            dtype=np.intp,
        )
        self._arm_ctrl_limits = model.actuator_ctrlrange[self._arm_actuator_ids].copy()
        self._finger_qpos_indices = np.asarray(
            [
                model.jnt_qposadr[model.joint(name).id]
                for name in _FINGER_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._finger_joint_limits = model.jnt_range[
            [model.joint(name).id for name in _FINGER_JOINT_NAMES]
        ].copy()
        self._gripper_actuator_id = model.actuator(_GRIPPER_ACTUATOR_NAME).id
        self._gripper_ctrl_range = model.actuator_ctrlrange[
            self._gripper_actuator_id
        ].copy()
        self._tcp_site_id = model.site(_TCP_SITE_NAME).id
        self.reset()

    @property
    def target_position(self) -> NDArray[np.float64]:
        """Return the current rate-limited TCP position target."""
        return self._target_position.copy()

    @property
    def target_yaw(self) -> float:
        """Return the current rate-limited TCP yaw target."""
        return float(self._target_yaw)

    @property
    def gripper_open_fraction(self) -> float:
        """Return the most recently commanded gripper open fraction."""
        return float(self._gripper_open_fraction)

    def reset(self) -> None:
        """Synchronize command targets with the current measured simulation state."""
        mujoco.mj_forward(self.model, self.data)
        arm_q = self.data.qpos[self._arm_qpos_indices].copy()
        self.data.ctrl[self._arm_actuator_ids] = np.clip(
            arm_q,
            self._arm_ctrl_limits[:, 0],
            self._arm_ctrl_limits[:, 1],
        )
        self._target_position = self.data.site_xpos[self._tcp_site_id].copy()
        current_rotation = mink.SO3.from_matrix(
            self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        )
        yaw_rotation = current_rotation @ _DOWN_ROTATION.inverse()
        yaw_matrix = yaw_rotation.as_matrix()
        self._target_yaw = float(np.arctan2(yaw_matrix[1, 0], yaw_matrix[0, 0]))
        self._gripper_open_fraction = self._measure_gripper_open_fraction()
        self.data.ctrl[self._gripper_actuator_id] = self._gripper_control(
            self._gripper_open_fraction
        )

    def apply(self, action: ArrayLike) -> None:
        """Apply one clipped 5-D TCP action over 25 MuJoCo physics steps."""
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (5,):
            raise ValueError(f"Action must have shape (5,), got {command.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("Action values must all be finite")

        gripper_open_fraction = float(np.clip(command[4], 0.0, 1.0))
        next_position, next_yaw = next_tcp_target(
            self._target_position,
            self._target_yaw,
            command,
            tcp_speed_limit=self.tcp_speed_limit,
            yaw_rate_limit=self.yaw_rate_limit,
        )

        start_position = self._target_position.copy()
        start_yaw = self._target_yaw
        for index in range(1, IK_STEPS_PER_ENVIRONMENT_STEP + 1):
            fraction = index / IK_STEPS_PER_ENVIRONMENT_STEP
            target_position = start_position + fraction * (next_position - start_position)
            target_yaw = start_yaw + fraction * (next_yaw - start_yaw)
            target_pose = tcp_pose(target_position, target_yaw)
            arm_q = self.data.qpos[self._arm_qpos_indices].copy()
            arm_target = self.ik.step(target_pose, arm_q)
            self.data.ctrl[self._arm_actuator_ids] = np.clip(
                arm_target,
                self._arm_ctrl_limits[:, 0],
                self._arm_ctrl_limits[:, 1],
            )
            self.data.ctrl[self._gripper_actuator_id] = self._gripper_control(
                gripper_open_fraction
            )
            mujoco.mj_step(
                self.model,
                self.data,
                nstep=PHYSICS_STEPS_PER_IK_STEP,
            )

        self._target_position = next_position
        self._target_yaw = next_yaw
        self._gripper_open_fraction = gripper_open_fraction

    def _measure_gripper_open_fraction(self) -> float:
        joint_ranges = self._finger_joint_limits[:, 1] - self._finger_joint_limits[:, 0]
        if np.any(joint_ranges <= 0):
            raise ValueError("Finger joints must have non-zero position ranges")
        openings = (
            self.data.qpos[self._finger_qpos_indices] - self._finger_joint_limits[:, 0]
        ) / joint_ranges
        return float(np.clip(np.mean(openings), 0.0, 1.0))

    def _gripper_control(self, open_fraction: float) -> float:
        lower, upper = self._gripper_ctrl_range
        return float(lower + open_fraction * (upper - lower))