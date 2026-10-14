"""Mink differential inverse kinematics for the Panda robot."""

from collections.abc import Mapping
from pathlib import Path

import mink
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
IK_TIMESTEP = 0.01
POSITION_TOLERANCE = 0.001
ORIENTATION_TOLERANCE = float(np.deg2rad(0.5))
MAX_SOLVE_STEPS = 200

_PANDA_XML = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)
_TCP_POSITION = (0.0, 0.0, 0.10)


def build_ik_model() -> mujoco.MjModel:
    """Compile a robot-only Panda model with the scene's TCP site."""
    spec = mujoco.MjSpec.from_file(str(_PANDA_XML))
    spec.body("hand").add_site(
        name="tcp",
        pos=_TCP_POSITION,
        size=0.008,
        rgba=(1.0, 0.2, 0.1, 1.0),
    )
    return spec.compile()


class MinkIK:
    """Solve Panda TCP poses with Mink differential inverse kinematics."""

    def __init__(self, velocity_limits: Mapping[str, float] | None = None) -> None:
        self.model = build_ik_model()
        self._configuration = mink.Configuration(self.model)
        self._configuration.update_from_keyframe("home")
        self._rest_qpos = self._configuration.q.copy()
        self._arm_qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.intp,
        )

        self._frame_task = mink.FrameTask(
            frame_name="tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(self.model, cost=1e-3)
        self._posture_task.set_target(self._rest_qpos)
        self._limits: list[mink.Limit] = [mink.ConfigurationLimit(self.model)]
        if velocity_limits is not None:
            unknown_joints = set(velocity_limits) - set(ARM_JOINT_NAMES)
            if unknown_joints:
                raise ValueError(f"Velocity limits include unknown arm joints: {unknown_joints}")
            normalized_limits = {
                name: float(limit) for name, limit in velocity_limits.items()
            }
            if any(not np.isfinite(limit) or limit <= 0 for limit in normalized_limits.values()):
                raise ValueError("Joint velocity limits must be finite and positive")
            self._limits.append(mink.VelocityLimit(self.model, normalized_limits))

    @property
    def rest_q(self) -> NDArray[np.float64]:
        """Return the fixed home posture for the seven arm joints."""
        return self._rest_qpos[self._arm_qpos_indices].copy()

    def step(self, target_pose: mink.SE3, q: ArrayLike) -> NDArray[np.float64]:
        """Take one 10 ms differential-IK step and return seven arm targets."""
        self._update_configuration(q)
        self._frame_task.set_target(target_pose)
        velocity = mink.solve_ik(
            self._configuration,
            [self._frame_task, self._posture_task],
            dt=IK_TIMESTEP,
            solver="daqp",
            damping=1e-3,
            limits=self._limits,
        )
        self._configuration.integrate_inplace(velocity, IK_TIMESTEP)
        return self._configuration.q[self._arm_qpos_indices].copy()

    def solve(
        self,
        target_pose: mink.SE3,
        q_init: ArrayLike,
    ) -> tuple[NDArray[np.float64], bool]:
        """Iterate IK until pose tolerances are met or 200 steps are exhausted."""
        self._update_configuration(q_init)
        for _ in range(MAX_SOLVE_STEPS):
            if self._is_converged(target_pose):
                return self._arm_q(), True
            try:
                self.step(target_pose, self._configuration.q.copy())
            except mink.exceptions.NoSolutionFound:
                return self._arm_q(), False
        return self._arm_q(), self._is_converged(target_pose)

    def _update_configuration(self, q: ArrayLike) -> None:
        positions = np.asarray(q, dtype=np.float64)
        if positions.shape == (len(ARM_JOINT_NAMES),):
            qpos = self._configuration.q.copy()
            qpos[self._arm_qpos_indices] = positions
        elif positions.shape == (self.model.nq,):
            qpos = positions.copy()
        else:
            raise ValueError(
                f"Joint positions must have shape ({len(ARM_JOINT_NAMES)},) "
                f"or ({self.model.nq},), got {positions.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Joint positions must contain only finite values")
        self._configuration.update(qpos)

    def _arm_q(self) -> NDArray[np.float64]:
        return self._configuration.q[self._arm_qpos_indices].copy()

    def _is_converged(self, target_pose: mink.SE3) -> bool:
        current_pose = self._configuration.get_transform_frame_to_world("tcp", "site")
        position_error = np.linalg.norm(target_pose.translation() - current_pose.translation())
        orientation_error = np.linalg.norm(
            target_pose.rotation().rminus(current_pose.rotation())
        )
        return bool(
            position_error <= POSITION_TOLERANCE
            and orientation_error <= ORIENTATION_TOLERANCE
        )