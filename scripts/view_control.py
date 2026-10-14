"""Demonstrate the Phase 1.2 control loop in the MuJoCo viewer."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from envs.scene import build_scene
from motion.control import ENVIRONMENT_TIMESTEP, ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK


def _initialize_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: MinkIK,
) -> None:
    """Set the robot arm and gripper to the Menagerie home posture."""
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
    mujoco.mj_forward(model, data)


def _action(
    position: tuple[float, float, float],
    yaw: float,
    gripper: float,
) -> NDArray[np.float64]:
    """Build one absolute TCP action for the demo sequence."""
    return np.array([*position, yaw, gripper], dtype=np.float64)


def main() -> None:
    """Run a looping TCP and gripper demonstration in the passive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-tcp",
        action="store_true",
        help="Show the small TCP site marker between the gripper fingers.",
    )
    arguments = parser.parse_args()

    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    _initialize_home(model, data, ik)
    control = ControlLoop(model, data, ik)
    sequence = (
        ((0.30, 0.0, 0.35), 0.0, 1.0, 90),
        ((0.60, 0.0, 0.35), 0.0, 1.0, 90),
        ((0.60, 0.0, 0.35), 0.0, 0.0, 24),
        ((0.60, 0.0, 0.35), 0.0, 1.0, 24),
        ((0.30, 0.0, 0.35), np.pi / 4, 1.0, 90),
        ((0.30, 0.0, 0.35), 0.0, 1.0, 30),
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.20]
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        if not arguments.show_tcp:
            tcp_site_group = model.site_group[model.site("tcp").id]
            viewer.opt.sitegroup[tcp_site_group] = 0
        while viewer.is_running():
            for position, yaw, gripper, steps in sequence:
                for _ in range(steps):
                    if not viewer.is_running():
                        return
                    tick_start = time.perf_counter()
                    control.apply(_action(position, yaw, gripper))
                    viewer.sync()
                    remaining = ENVIRONMENT_TIMESTEP - (time.perf_counter() - tick_start)
                    if remaining > 0:
                        time.sleep(remaining)


if __name__ == "__main__":
    main()